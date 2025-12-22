use crate::adapters::InventoryClient;
use crate::error::{HpcError, SchedulerErrorExt};
use crate::models::{Job, PlacementDecision, Topology, Node, Gpu};
use crate::Result;
use std::collections::HashMap;
use uuid::Uuid;

/// Placement engine using Best-Fit Decreasing (BFD) with NUMA awareness
pub struct PlacementEngine {
    inventory_client: InventoryClient,
}

impl PlacementEngine {
    pub fn new(inventory_client: InventoryClient) -> Self {
        Self { inventory_client }
    }

    /// Find optimal placement for a job using BFD algorithm
    pub async fn find_placement(&self, job: &Job) -> Result<PlacementDecision> {
        let topology = self.inventory_client.get_topology().await?;

        // Try single-node placement first (best for performance)
        if let Some(placement) = self.try_single_node_placement(job, &topology).await {
            return Ok(placement);
        }

        // Fall back to multi-node placement
        if let Some(placement) = self.try_multi_node_placement(job, &topology).await {
            return Ok(placement);
        }

        let required_gpus = self.get_job_gpu_count(job);
        Err(HpcError::insufficient_resources(
            required_gpus,
            self.count_available_gpus(&topology),
        ))
    }

    /// Helper: Extract GPU count from job resources
    fn get_job_gpu_count(&self, job: &Job) -> usize {
        job.resources.get_gpu_spec()
            .map(|s| s.amount as usize)
            .unwrap_or(0)
    }

    /// Helper: Extract GPU type/model from job resources
    fn get_job_gpu_type(&self, job: &Job) -> Option<String> {
        job.resources.get_gpu_spec()
            .and_then(|s| s.constraints.as_ref())
            .and_then(|c| c.model.clone())
    }

    /// Try to place all GPUs on a single node (best locality)
    async fn try_single_node_placement(&self, job: &Job, topology: &Topology) -> Option<PlacementDecision> {
        let required_gpus = self.get_job_gpu_count(job);

        for node in &topology.nodes {
            if !node.available {
                continue;
            }

            // Count available GPUs on this node
            let available_gpus = self.count_node_available_gpus(node);

            if available_gpus >= required_gpus {
                // Try NUMA-aware placement within node
                if let Some(placement) = self.try_numa_aware_placement(job, node) {
                    return Some(placement);
                }
            }
        }

        None
    }

    /// Try NUMA-aware placement (prefer GPUs on same NUMA node)
    fn try_numa_aware_placement(&self, job: &Job, node: &Node) -> Option<PlacementDecision> {
        let required_gpus = self.get_job_gpu_count(job);

        // Score each NUMA node by available GPUs
        let mut numa_scores: Vec<(usize, usize)> = node
            .numa_nodes
            .iter()
            .enumerate()
            .map(|(idx, numa)| {
                let available = numa.gpus.iter().filter(|g| g.available).count();
                (idx, available)
            })
            .filter(|(_, count)| *count > 0)
            .collect();

        // Sort by descending available GPUs (Best-Fit Decreasing)
        numa_scores.sort_by(|a, b| b.1.cmp(&a.1));

        let mut selected_gpus = Vec::new();
        let mut remaining = required_gpus;

        // Allocate from NUMA nodes in order
        for (numa_idx, _) in numa_scores {
            if remaining == 0 {
                break;
            }

            let numa_node = &node.numa_nodes[numa_idx];

            for gpu in &numa_node.gpus {
                if gpu.available && self.matches_gpu_type(gpu, job) {
                    selected_gpus.push(gpu.clone());
                    remaining -= 1;

                    if remaining == 0 {
                        break;
                    }
                }
            }
        }

        if remaining == 0 {
            let gpu_ids: Vec<Uuid> = selected_gpus.iter().map(|g| g.id).collect();
            let score = self.calculate_placement_score(&selected_gpus, node);

            Some(PlacementDecision {
                job_id: job.id,
                node_ids: vec![node.id],
                gpu_ids,
                score,
            })
        } else {
            None
        }
    }

    /// Try multi-node placement using BFD
    async fn try_multi_node_placement(&self, job: &Job, topology: &Topology) -> Option<PlacementDecision> {
        let required_gpus = self.get_job_gpu_count(job);

        // Score nodes by available GPU count (BFD)
        let mut node_scores: Vec<(&Node, usize)> = topology
            .nodes
            .iter()
            .filter(|n| n.available)
            .map(|n| (n, self.count_node_available_gpus(n)))
            .filter(|(_, count)| *count > 0)
            .collect();

        node_scores.sort_by(|a, b| b.1.cmp(&a.1));

        let mut selected_gpus = Vec::new();
        let mut selected_nodes = Vec::new();
        let mut remaining = required_gpus;

        for (node, _) in node_scores {
            if remaining == 0 {
                break;
            }

            for numa_node in &node.numa_nodes {
                for gpu in &numa_node.gpus {
                    if gpu.available && self.matches_gpu_type(gpu, job) {
                        selected_gpus.push(gpu.clone());

                        if !selected_nodes.contains(&node.id) {
                            selected_nodes.push(node.id);
                        }

                        remaining -= 1;

                        if remaining == 0 {
                            break;
                        }
                    }
                }

                if remaining == 0 {
                    break;
                }
            }
        }

        if remaining == 0 {
            let gpu_ids: Vec<Uuid> = selected_gpus.iter().map(|g| g.id).collect();
            // Multi-node has lower score due to worse locality
            let score = 50.0;

            Some(PlacementDecision {
                job_id: job.id,
                node_ids: selected_nodes,
                gpu_ids,
                score,
            })
        } else {
            None
        }
    }

    /// Calculate placement quality score (higher is better)
    fn calculate_placement_score(&self, gpus: &[Gpu], node: &Node) -> f64 {
        let mut score = 100.0;

        // Bonus for single-node placement
        score += 20.0;

        // Check NUMA locality
        let numa_distribution = self.analyze_numa_distribution(gpus, node);

        // Bonus for NUMA-local placement (all GPUs on same NUMA node)
        if numa_distribution.len() == 1 {
            score += 30.0;
        } else if numa_distribution.len() == 2 {
            score += 10.0;
        }

        score
    }

    /// Analyze GPU distribution across NUMA nodes
    fn analyze_numa_distribution(&self, gpus: &[Gpu], node: &Node) -> HashMap<u32, usize> {
        let mut distribution = HashMap::new();

        for numa_node in &node.numa_nodes {
            let count = gpus.iter().filter(|g| {
                numa_node.gpus.iter().any(|ng| ng.id == g.id)
            }).count();

            if count > 0 {
                distribution.insert(numa_node.id, count);
            }
        }

        distribution
    }

    fn count_available_gpus(&self, topology: &Topology) -> usize {
        topology.nodes.iter()
            .filter(|n| n.available)
            .map(|n| self.count_node_available_gpus(n))
            .sum()
    }

    fn count_node_available_gpus(&self, node: &Node) -> usize {
        node.numa_nodes.iter()
            .flat_map(|numa| &numa.gpus)
            .filter(|g| g.available)
            .count()
    }

    fn matches_gpu_type(&self, gpu: &Gpu, job: &Job) -> bool {
        self.get_job_gpu_type(job)
            .map_or(true, |req_type| req_type == gpu.gpu_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InventoryConfig;
    use crate::models::NumaNode;

    fn create_test_gpu(id: Uuid, gpu_type: &str, available: bool) -> Gpu {
        Gpu {
            id,
            device_id: format!("gpu-{}", id),
            gpu_type: gpu_type.to_string(),
            memory_gb: 80,
            available,
        }
    }

    fn create_test_node(numa_count: usize, gpus_per_numa: usize) -> Node {
        let mut numa_nodes = Vec::new();

        for i in 0..numa_count {
            let mut gpus = Vec::new();
            for _ in 0..gpus_per_numa {
                gpus.push(create_test_gpu(Uuid::new_v4(), "H100", true));
            }

            numa_nodes.push(NumaNode {
                id: i as u32,
                gpus,
                cpu_cores: vec![0, 1, 2, 3],
                memory_gb: 256,
            });
        }

        Node {
            id: Uuid::new_v4(),
            hostname: "node1".to_string(),
            numa_nodes,
            available: true,
        }
    }

    #[test]
    fn test_count_node_available_gpus() {
        let config = InventoryConfig {
            base_url: "http://localhost:8081".to_string(),
            timeout_secs: 30,
            retry_attempts: 3,
        };

        let client = InventoryClient::new(&config).unwrap();
        let engine = PlacementEngine::new(client);

        let node = create_test_node(2, 4); // 2 NUMA nodes, 4 GPUs each
        assert_eq!(engine.count_node_available_gpus(&node), 8);
    }

    #[test]
    fn test_matches_gpu_type() {
        let config = InventoryConfig {
            base_url: "http://localhost:8081".to_string(),
            timeout_secs: 30,
            retry_attempts: 3,
        };

        let client = InventoryClient::new(&config).unwrap();
        let engine = PlacementEngine::new(client);

        let gpu_h100 = create_test_gpu(Uuid::new_v4(), "H100", true);
        let gpu_a100 = create_test_gpu(Uuid::new_v4(), "A100", true);

        let job_any = Job::builder().user_id("user1").gpu_count(1).build().unwrap();
        let job_h100 = Job::builder()
            .user_id("user1")
            .gpu_count(1)
            .gpu_type("H100")
            .build()
            .unwrap();

        assert!(engine.matches_gpu_type(&gpu_h100, &job_any));
        assert!(engine.matches_gpu_type(&gpu_a100, &job_any));
        assert!(engine.matches_gpu_type(&gpu_h100, &job_h100));
        assert!(!engine.matches_gpu_type(&gpu_a100, &job_h100));
    }
}
