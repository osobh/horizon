//! Enhanced GPU knowledge graph with CSR format and advanced algorithms

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use std::sync::Arc;

/// CSR (Compressed Sparse Row) format for GPU graph
pub struct CsrGraph {
    /// Row offsets - size: num_nodes + 1
    row_offsets: CudaSlice<u32>,
    /// Column indices - size: num_edges
    column_indices: CudaSlice<u32>,
    /// Edge weights - size: num_edges
    edge_weights: CudaSlice<f32>,
    /// Number of nodes
    num_nodes: usize,
    /// Number of edges
    num_edges: usize,
}

impl CsrGraph {
    /// Create CSR graph from adjacency list
    pub fn from_adjacency_list(
        device: Arc<CudaContext>,
        adjacency_list: &[(u32, Vec<(u32, f32)>)],
        num_nodes: usize,
    ) -> Result<Self> {
        // Build CSR format
        let mut row_offsets = vec![0u32; num_nodes + 1];
        let mut column_indices = Vec::new();
        let mut edge_weights = Vec::new();

        for node_id in 0..num_nodes {
            // Find adjacency list for this node
            let empty_vec = vec![];
            let edges = adjacency_list
                .iter()
                .find(|(id, _)| *id == node_id as u32)
                .map(|(_, edges)| edges)
                .unwrap_or(&empty_vec);

            row_offsets[node_id + 1] = row_offsets[node_id] + edges.len() as u32;

            for (target, weight) in edges {
                column_indices.push(*target);
                edge_weights.push(*weight);
            }
        }

        let num_edges = column_indices.len();

        // Upload to GPU
        let stream = device.default_stream();
        let gpu_row_offsets = stream.clone_htod(&row_offsets)?;
        let gpu_column_indices = stream.clone_htod(&column_indices)?;
        let gpu_edge_weights = stream.clone_htod(&edge_weights)?;

        Ok(Self {
            row_offsets: gpu_row_offsets,
            column_indices: gpu_column_indices,
            edge_weights: gpu_edge_weights,
            num_nodes,
            num_edges,
        })
    }

    /// Get GPU pointers for kernel access
    pub fn gpu_pointers(&self, stream: &CudaStream) -> CsrGraphPointers {
        let (row_offsets_ptr, _row_guard) = self.row_offsets.device_ptr(stream);
        let (column_indices_ptr, _col_guard) = self.column_indices.device_ptr(stream);
        let (edge_weights_ptr, _edge_guard) = self.edge_weights.device_ptr(stream);
        CsrGraphPointers {
            row_offsets: row_offsets_ptr as *const u32,
            column_indices: column_indices_ptr as *const u32,
            edge_weights: edge_weights_ptr as *const f32,
            num_nodes: self.num_nodes as u32,
            num_edges: self.num_edges as u32,
        }
    }
}

/// GPU pointers for CSR graph
pub struct CsrGraphPointers {
    pub row_offsets: *const u32,
    pub column_indices: *const u32,
    pub edge_weights: *const f32,
    pub num_nodes: u32,
    pub num_edges: u32,
}

/// Enhanced GPU knowledge graph with advanced features
pub struct EnhancedGpuKnowledgeGraph {
    device: Arc<CudaContext>,
    /// Node embeddings (num_nodes * embedding_dim)
    node_embeddings: CudaSlice<f32>,
    /// Node metadata (types, properties)
    _node_metadata: CudaSlice<u32>,
    /// CSR graph structure
    csr_graph: CsrGraph,
    /// Spatial index for fast nearest neighbor search
    spatial_index: Option<SpatialIndex>,
    /// Cache for frequently accessed nodes
    _hot_node_cache: Option<CudaSlice<u32>>,
    /// Graph statistics
    num_nodes: usize,
    embedding_dim: usize,
}

impl EnhancedGpuKnowledgeGraph {
    /// Create new enhanced knowledge graph
    pub fn new(device: Arc<CudaContext>, num_nodes: usize, embedding_dim: usize) -> Result<Self> {
        // Allocate GPU memory
        let stream = device.default_stream();
        let node_embeddings = stream.alloc_zeros::<f32>(num_nodes * embedding_dim)?;
        let node_metadata = stream.alloc_zeros::<u32>(num_nodes)?;

        // Create empty CSR graph
        let csr_graph = CsrGraph {
            row_offsets: stream.clone_htod(&vec![0u32; num_nodes + 1])?,
            column_indices: stream.clone_htod(&Vec::<u32>::new())?,
            edge_weights: stream.clone_htod(&Vec::<f32>::new())?,
            num_nodes,
            num_edges: 0,
        };

        Ok(Self {
            device,
            node_embeddings,
            _node_metadata: node_metadata,
            csr_graph,
            spatial_index: None,
            _hot_node_cache: None,
            num_nodes,
            embedding_dim,
        })
    }

    /// Build spatial index for fast similarity search
    pub fn build_spatial_index(&mut self) -> Result<()> {
        self.spatial_index = Some(SpatialIndex::build(
            &self.device,
            &self.node_embeddings,
            self.num_nodes,
            self.embedding_dim,
        )?);
        Ok(())
    }

    /// Update node embeddings
    pub fn update_embeddings(&mut self, node_ids: &[u32], _embeddings: &[Vec<f32>]) -> Result<()> {
        for (_i, node_id) in node_ids.iter().enumerate() {
            let start_idx = *node_id as usize * self.embedding_dim;
            let end_idx = start_idx + self.embedding_dim;

            if end_idx <= self.node_embeddings.len() {
                // For now, skip the copy operation
                // TODO: Implement proper slice copy when cudarc API is clarified
            }
        }

        // Invalidate spatial index
        if self.spatial_index.is_some() {
            self.build_spatial_index()?;
        }

        Ok(())
    }

    /// Perform approximate nearest neighbor search
    pub fn ann_search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        if let Some(ref spatial_index) = self.spatial_index {
            spatial_index.search(query_embedding, k)
        } else {
            // Fallback to brute force
            self.brute_force_knn(query_embedding, k)
        }
    }

    /// Brute force k-nearest neighbors
    fn brute_force_knn(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        // Upload query embedding
        let stream = self.device.default_stream();
        let gpu_query = stream.clone_htod(query_embedding)?;

        // Allocate results
        let results_size = k.min(self.num_nodes);
        // SAFETY: alloc returns uninitialized memory but we write before reading via the kernel.
        // results_size is bounded by num_nodes which is a valid count from construction.
        let gpu_indices = unsafe { stream.alloc::<u32>(results_size)? };
        // SAFETY: Same rationale - kernel writes all k distances before we read them back.
        let gpu_distances = unsafe { stream.alloc::<f32>(results_size)? };

        // Launch KNN kernel
        // SAFETY: All pointers are valid device pointers:
        // - node_embeddings: allocated in new() with size num_nodes * embedding_dim
        // - gpu_query: just created via clone_htod from valid query_embedding
        // - gpu_indices/gpu_distances: just allocated above with size results_size
        // Parameters num_nodes, embedding_dim, k match allocation sizes.
        let (embeddings_ptr, _emb_guard) = self.node_embeddings.device_ptr(&stream);
        let (query_ptr, _query_guard) = gpu_query.device_ptr(&stream);
        let (indices_ptr, _idx_guard) = gpu_indices.device_ptr(&stream);
        let (distances_ptr, _dist_guard) = gpu_distances.device_ptr(&stream);
        unsafe {
            launch_tensor_core_knn(
                embeddings_ptr as *const f32,
                query_ptr as *const f32,
                indices_ptr as *mut u32,
                distances_ptr as *mut f32,
                self.num_nodes as u32,
                self.embedding_dim as u32,
                k as u32,
            );
        }

        // Read results
        let indices: Vec<u32> = stream.clone_dtoh(&gpu_indices)?;
        let distances: Vec<f32> = stream.clone_dtoh(&gpu_distances)?;

        Ok(indices.into_iter().zip(distances).collect())
    }

    /// Run PageRank algorithm
    pub fn pagerank(&self, iterations: u32, damping: f32) -> Result<Vec<f32>> {
        let stream = self.device.default_stream();
        // SAFETY: alloc returns uninitialized memory but we immediately initialize
        // via htod_copy_into below before the kernel reads it.
        let mut gpu_scores = unsafe { stream.alloc::<f32>(self.num_nodes)? };

        // Initialize scores
        let initial_score = 1.0 / self.num_nodes as f32;
        let initial_scores = vec![initial_score; self.num_nodes];
        stream.memcpy_htod(&initial_scores, &mut gpu_scores)?;

        // Run PageRank iterations
        let csr_pointers = self.csr_graph.gpu_pointers(&stream);
        // SAFETY: All pointers are valid device pointers:
        // - csr_pointers.row_offsets/column_indices: from CSR graph, valid for graph lifetime
        // - gpu_scores: allocated above with num_nodes elements, initialized via memcpy_htod
        // - num_nodes matches the actual graph size
        let (scores_ptr, _scores_guard) = gpu_scores.device_ptr(&stream);
        unsafe {
            launch_pagerank(
                csr_pointers.row_offsets,
                csr_pointers.column_indices,
                scores_ptr as *mut f32,
                csr_pointers.num_nodes,
                iterations,
                damping,
            );
        }

        // Read results
        Ok(stream.clone_dtoh(&gpu_scores)?)
    }

    /// Find shortest path using GPU BFS
    pub fn shortest_path(&self, source: u32, target: u32) -> Result<Option<Vec<u32>>> {
        let stream = self.device.default_stream();
        // SAFETY: alloc returns uninitialized memory but kernel writes path data before we read.
        let gpu_path = unsafe { stream.alloc::<u32>(self.num_nodes)? };
        // SAFETY: alloc returns uninitialized memory but we initialize via htod_copy_into below.
        let mut gpu_found = unsafe { stream.alloc::<bool>(1)? };

        // Initialize found flag
        stream.memcpy_htod(&[false], &mut gpu_found)?;

        // Run BFS
        let csr_pointers = self.csr_graph.gpu_pointers(&stream);
        // SAFETY: All pointers are valid device pointers:
        // - csr_pointers: from CSR graph, valid for graph lifetime
        // - gpu_path: allocated above with num_nodes elements
        // - gpu_found: allocated above, initialized to false
        // - source/target are u32 node IDs, num_nodes bounds them
        let (path_ptr, _path_guard) = gpu_path.device_ptr(&stream);
        let (found_ptr, _found_guard) = gpu_found.device_ptr(&stream);
        unsafe {
            launch_gpu_bfs(
                csr_pointers.row_offsets,
                csr_pointers.column_indices,
                path_ptr as *mut u32,
                found_ptr as *mut bool,
                source,
                target,
                csr_pointers.num_nodes,
            );
        }

        // Check if path was found
        let found: Vec<bool> = stream.clone_dtoh(&gpu_found)?;
        if found[0] {
            let path: Vec<u32> = stream.clone_dtoh(&gpu_path)?;
            // Extract actual path from BFS result
            Ok(Some(path))
        } else {
            Ok(None)
        }
    }

    /// Detect communities using parallel label propagation
    pub fn detect_communities(&self) -> Result<Vec<u32>> {
        let stream = self.device.default_stream();
        // SAFETY: alloc returns uninitialized memory but we immediately initialize
        // via htod_copy_into below before the kernel reads it.
        let mut gpu_labels = unsafe { stream.alloc::<u32>(self.num_nodes)? };

        // Initialize labels
        let initial_labels: Vec<u32> = (0..self.num_nodes as u32).collect();
        stream.memcpy_htod(&initial_labels, &mut gpu_labels)?;

        // Run label propagation
        let csr_pointers = self.csr_graph.gpu_pointers(&stream);
        // SAFETY: All pointers are valid device pointers:
        // - csr_pointers: from CSR graph, valid for graph lifetime
        // - gpu_labels: allocated above with num_nodes elements, initialized via memcpy_htod
        // - num_nodes matches actual graph size
        let (labels_ptr, _labels_guard) = gpu_labels.device_ptr(&stream);
        unsafe {
            launch_label_propagation(
                csr_pointers.row_offsets,
                csr_pointers.column_indices,
                labels_ptr as *mut u32,
                csr_pointers.num_nodes,
                10, // iterations
            );
        }

        Ok(stream.clone_dtoh(&gpu_labels)?)
    }
}

/// Spatial index for fast nearest neighbor search
pub struct SpatialIndex {
    _device: Arc<CudaContext>,
    /// Index structure (implementation-specific)
    _index_data: CudaSlice<u8>,
    _num_nodes: usize,
    _embedding_dim: usize,
}

impl SpatialIndex {
    /// Build spatial index from embeddings
    pub fn build(
        device: &Arc<CudaContext>,
        _embeddings: &CudaSlice<f32>,
        num_nodes: usize,
        embedding_dim: usize,
    ) -> Result<Self> {
        // Placeholder for actual spatial index construction
        // Could implement IVF, HNSW, or other GPU-friendly index
        let stream = device.default_stream();
        let index_size = num_nodes * std::mem::size_of::<u32>();
        // SAFETY: alloc returns uninitialized memory. This is a placeholder that will
        // be properly initialized when the actual spatial index algorithm is implemented.
        let index_data = unsafe { stream.alloc::<u8>(index_size)? };

        Ok(Self {
            _device: device.clone(),
            _index_data: index_data,
            _num_nodes: num_nodes,
            _embedding_dim: embedding_dim,
        })
    }

    /// Search for k nearest neighbors
    pub fn search(&self, _query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        // Placeholder for spatial index search
        // Would use GPU kernels for actual implementation
        Ok(vec![(0, 0.0); k])
    }
}

// External CUDA kernel declarations
unsafe extern "C" {
    fn launch_tensor_core_knn(
        embeddings: *const f32,
        query: *const f32,
        indices: *mut u32,
        distances: *mut f32,
        num_nodes: u32,
        embedding_dim: u32,
        k: u32,
    );

    fn launch_pagerank(
        row_offsets: *const u32,
        column_indices: *const u32,
        scores: *mut f32,
        num_nodes: u32,
        iterations: u32,
        damping: f32,
    );

    fn launch_gpu_bfs(
        row_offsets: *const u32,
        column_indices: *const u32,
        path: *mut u32,
        found: *mut bool,
        source: u32,
        target: u32,
        num_nodes: u32,
    );

    fn launch_label_propagation(
        row_offsets: *const u32,
        column_indices: *const u32,
        labels: *mut u32,
        num_nodes: u32,
        iterations: u32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_graph_creation() -> Result<()> {
        if let Ok(device) = CudaContext::new(0) {
            let adjacency_list = vec![
                (0, vec![(1, 1.0), (2, 0.5)]),
                (1, vec![(2, 0.8)]),
                (2, vec![]),
            ];

            let csr = CsrGraph::from_adjacency_list(device, &adjacency_list, 3)?;
            assert_eq!(csr.num_nodes, 3);
            assert_eq!(csr.num_edges, 3);
        }
        Ok(())
    }
}
