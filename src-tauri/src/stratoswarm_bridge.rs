//! StratoSwarm Bridge
//!
//! Integration with StratoSwarm for agent visualization, XP tracking, and evolution.
//!
//! This bridge provides mock agent data for development and can connect to a live
//! StratoSwarm cluster when the `stratoswarm-live` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Agent tier/evolution level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentTier {
    Bronze,
    Silver,
    Gold,
    Platinum,
    Diamond,
}

impl AgentTier {
    pub fn xp_threshold(&self) -> u64 {
        match self {
            AgentTier::Bronze => 0,
            AgentTier::Silver => 1000,
            AgentTier::Gold => 5000,
            AgentTier::Platinum => 15000,
            AgentTier::Diamond => 50000,
        }
    }

    pub fn next_tier(&self) -> Option<AgentTier> {
        match self {
            AgentTier::Bronze => Some(AgentTier::Silver),
            AgentTier::Silver => Some(AgentTier::Gold),
            AgentTier::Gold => Some(AgentTier::Platinum),
            AgentTier::Platinum => Some(AgentTier::Diamond),
            AgentTier::Diamond => None,
        }
    }
}

/// Agent operational status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Idle,
    Working,
    Learning,
    Evolving,
    Offline,
    Error,
}

/// Agent specialization type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSpecialization {
    General,
    Compute,
    Storage,
    Network,
    Security,
    Analytics,
    MachineLearning,
}

/// Agent skill with proficiency level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    pub name: String,
    pub proficiency: f64, // 0.0 - 1.0
    pub xp: u64,
}

/// Individual agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmAgent {
    pub id: String,
    pub name: String,
    pub status: AgentStatus,
    pub tier: AgentTier,
    pub specialization: AgentSpecialization,
    pub xp: u64,
    pub xp_to_next_tier: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub success_rate: f64,
    pub uptime_hours: f64,
    pub skills: Vec<AgentSkill>,
    pub current_task: Option<String>,
    pub node_id: Option<String>,
    pub created_at: String,
    pub last_active: String,
}

/// Swarm-wide statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStats {
    pub total_agents: u32,
    pub active_agents: u32,
    pub idle_agents: u32,
    pub evolving_agents: u32,
    pub offline_agents: u32,
    pub total_xp: u64,
    pub tasks_completed_today: u64,
    pub tasks_in_progress: u64,
    pub average_success_rate: f64,
    pub tier_distribution: HashMap<String, u32>,
}

/// Evolution event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEvent {
    pub id: String,
    pub agent_id: String,
    pub agent_name: String,
    pub from_tier: AgentTier,
    pub to_tier: AgentTier,
    pub xp_at_evolution: u64,
    pub timestamp: String,
}

/// Task assignment for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub agent_id: String,
    pub task_type: String,
    pub description: String,
    pub status: String,
    pub progress: f64,
    pub xp_reward: u64,
    pub started_at: String,
    pub estimated_completion: Option<String>,
}

/// Connection status to StratoSwarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratoSwarmStatus {
    pub connected: bool,
    pub cluster_url: String,
    pub cluster_name: Option<String>,
    pub version: Option<String>,
    pub last_error: Option<String>,
}

/// Mock state for development
#[cfg(not(feature = "stratoswarm-live"))]
struct MockSwarmState {
    agents: Vec<SwarmAgent>,
    events: Vec<EvolutionEvent>,
    tasks: Vec<AgentTask>,
}

#[cfg(not(feature = "stratoswarm-live"))]
impl Default for MockSwarmState {
    fn default() -> Self {
        Self {
            agents: generate_mock_agents(),
            events: generate_mock_events(),
            tasks: generate_mock_tasks(),
        }
    }
}

/// StratoSwarm integration bridge
pub struct StratoSwarmBridge {
    cluster_url: Arc<RwLock<String>>,
    #[cfg(not(feature = "stratoswarm-live"))]
    mock_state: Arc<RwLock<MockSwarmState>>,
}

impl StratoSwarmBridge {
    pub fn new() -> Self {
        Self {
            cluster_url: Arc::new(RwLock::new("http://localhost:9090".to_string())),
            #[cfg(not(feature = "stratoswarm-live"))]
            mock_state: Arc::new(RwLock::new(MockSwarmState::default())),
        }
    }

    /// Initialize the bridge
    pub async fn initialize(&self) -> Result<(), String> {
        tracing::info!("StratoSwarm bridge initialized (mock mode)");
        Ok(())
    }

    /// Get connection status
    pub async fn get_status(&self) -> StratoSwarmStatus {
        let url = self.cluster_url.read().await.clone();
        StratoSwarmStatus {
            connected: true, // Mock always connected
            cluster_url: url,
            cluster_name: Some("dev-swarm".to_string()),
            version: Some("0.1.0".to_string()),
            last_error: None,
        }
    }

    /// Set cluster URL
    pub async fn set_cluster_url(&self, url: String) {
        *self.cluster_url.write().await = url;
    }

    /// Get swarm statistics
    #[cfg(not(feature = "stratoswarm-live"))]
    pub async fn get_swarm_stats(&self) -> Result<SwarmStats, String> {
        let state = self.mock_state.read().await;
        let agents = &state.agents;

        let mut tier_distribution = HashMap::new();
        for agent in agents {
            let tier_name = format!("{:?}", agent.tier).to_lowercase();
            *tier_distribution.entry(tier_name).or_insert(0) += 1;
        }

        let active = agents.iter().filter(|a| a.status == AgentStatus::Working).count() as u32;
        let idle = agents.iter().filter(|a| a.status == AgentStatus::Idle).count() as u32;
        let evolving = agents.iter().filter(|a| a.status == AgentStatus::Evolving).count() as u32;
        let offline = agents.iter().filter(|a| a.status == AgentStatus::Offline).count() as u32;

        let total_success_rate: f64 = agents.iter().map(|a| a.success_rate).sum();
        let avg_success = if agents.is_empty() {
            0.0
        } else {
            total_success_rate / agents.len() as f64
        };

        Ok(SwarmStats {
            total_agents: agents.len() as u32,
            active_agents: active,
            idle_agents: idle,
            evolving_agents: evolving,
            offline_agents: offline,
            total_xp: agents.iter().map(|a| a.xp).sum(),
            tasks_completed_today: 847,
            tasks_in_progress: state.tasks.len() as u64,
            average_success_rate: avg_success,
            tier_distribution,
        })
    }

    /// List all agents
    #[cfg(not(feature = "stratoswarm-live"))]
    pub async fn list_agents(
        &self,
        status: Option<AgentStatus>,
        tier: Option<AgentTier>,
    ) -> Result<Vec<SwarmAgent>, String> {
        let state = self.mock_state.read().await;
        let mut agents = state.agents.clone();

        if let Some(s) = status {
            agents.retain(|a| a.status == s);
        }
        if let Some(t) = tier {
            agents.retain(|a| a.tier == t);
        }

        Ok(agents)
    }

    /// Get agent by ID
    #[cfg(not(feature = "stratoswarm-live"))]
    pub async fn get_agent(&self, id: &str) -> Result<SwarmAgent, String> {
        let state = self.mock_state.read().await;
        state
            .agents
            .iter()
            .find(|a| a.id == id)
            .cloned()
            .ok_or_else(|| format!("Agent '{}' not found", id))
    }

    /// Get evolution events
    #[cfg(not(feature = "stratoswarm-live"))]
    pub async fn get_evolution_events(&self, limit: Option<usize>) -> Result<Vec<EvolutionEvent>, String> {
        let state = self.mock_state.read().await;
        let limit = limit.unwrap_or(50);
        Ok(state.events.iter().take(limit).cloned().collect())
    }

    /// Get active tasks
    #[cfg(not(feature = "stratoswarm-live"))]
    pub async fn get_active_tasks(&self) -> Result<Vec<AgentTask>, String> {
        let state = self.mock_state.read().await;
        Ok(state.tasks.clone())
    }

    /// Trigger agent evolution (if eligible)
    #[cfg(not(feature = "stratoswarm-live"))]
    pub async fn trigger_evolution(&self, agent_id: &str) -> Result<EvolutionEvent, String> {
        let mut state = self.mock_state.write().await;

        let agent = state
            .agents
            .iter_mut()
            .find(|a| a.id == agent_id)
            .ok_or_else(|| format!("Agent '{}' not found", agent_id))?;

        let next_tier = agent
            .tier
            .next_tier()
            .ok_or_else(|| "Agent is already at maximum tier".to_string())?;

        let xp_needed = next_tier.xp_threshold();
        if agent.xp < xp_needed {
            return Err(format!(
                "Agent needs {} more XP to evolve",
                xp_needed - agent.xp
            ));
        }

        let event = EvolutionEvent {
            id: format!("evo-{}", uuid::Uuid::new_v4()),
            agent_id: agent.id.clone(),
            agent_name: agent.name.clone(),
            from_tier: agent.tier,
            to_tier: next_tier,
            xp_at_evolution: agent.xp,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        agent.tier = next_tier;
        agent.xp_to_next_tier = agent.tier.next_tier().map(|t| t.xp_threshold()).unwrap_or(0);
        state.events.insert(0, event.clone());

        Ok(event)
    }

    /// Simulate activity (for demo purposes)
    #[cfg(not(feature = "stratoswarm-live"))]
    pub async fn simulate_activity(&self) -> Result<(), String> {
        let mut state = self.mock_state.write().await;

        // Add XP to random agents
        for agent in &mut state.agents {
            if agent.status == AgentStatus::Working {
                let xp_gain = rand::random::<u64>() % 50 + 10;
                agent.xp += xp_gain;
                agent.tasks_completed += 1;

                // Update skills
                for skill in &mut agent.skills {
                    skill.xp += rand::random::<u64>() % 20;
                    skill.proficiency = (skill.proficiency + 0.01).min(1.0);
                }
            }
        }

        // Randomly change some agent statuses
        let agent_count = state.agents.len();
        if agent_count > 0 {
            let idx = rand::random::<usize>() % agent_count;
            if let Some(agent) = state.agents.get_mut(idx) {
                agent.status = match rand::random::<u8>() % 4 {
                    0 => AgentStatus::Idle,
                    1 => AgentStatus::Working,
                    2 => AgentStatus::Learning,
                    _ => AgentStatus::Working,
                };
                agent.last_active = chrono::Utc::now().to_rfc3339();
            }
        }

        Ok(())
    }
}

impl Default for StratoSwarmBridge {
    fn default() -> Self {
        Self::new()
    }
}

// Mock data generators
#[cfg(not(feature = "stratoswarm-live"))]
fn generate_mock_agents() -> Vec<SwarmAgent> {
    let agent_names = [
        ("alpha", "Alpha-1", AgentTier::Diamond, AgentStatus::Working),
        ("beta", "Beta-2", AgentTier::Platinum, AgentStatus::Working),
        ("gamma", "Gamma-3", AgentTier::Gold, AgentStatus::Idle),
        ("delta", "Delta-4", AgentTier::Gold, AgentStatus::Learning),
        ("epsilon", "Epsilon-5", AgentTier::Silver, AgentStatus::Working),
        ("zeta", "Zeta-6", AgentTier::Silver, AgentStatus::Idle),
        ("eta", "Eta-7", AgentTier::Bronze, AgentStatus::Working),
        ("theta", "Theta-8", AgentTier::Bronze, AgentStatus::Evolving),
        ("iota", "Iota-9", AgentTier::Bronze, AgentStatus::Offline),
        ("kappa", "Kappa-10", AgentTier::Silver, AgentStatus::Working),
    ];

    agent_names
        .iter()
        .map(|(id, name, tier, status)| {
            let base_xp = tier.xp_threshold();
            let xp = base_xp + rand::random::<u64>() % 1000;
            let xp_to_next = tier.next_tier().map(|t| t.xp_threshold()).unwrap_or(0);

            SwarmAgent {
                id: id.to_string(),
                name: name.to_string(),
                status: *status,
                tier: *tier,
                specialization: match rand::random::<u8>() % 7 {
                    0 => AgentSpecialization::General,
                    1 => AgentSpecialization::Compute,
                    2 => AgentSpecialization::Storage,
                    3 => AgentSpecialization::Network,
                    4 => AgentSpecialization::Security,
                    5 => AgentSpecialization::Analytics,
                    _ => AgentSpecialization::MachineLearning,
                },
                xp,
                xp_to_next_tier: xp_to_next,
                tasks_completed: rand::random::<u64>() % 10000 + 100,
                tasks_failed: rand::random::<u64>() % 100,
                success_rate: 0.9 + (rand::random::<f64>() * 0.1),
                uptime_hours: rand::random::<f64>() * 720.0 + 24.0,
                skills: vec![
                    AgentSkill {
                        name: "Task Execution".to_string(),
                        proficiency: 0.5 + rand::random::<f64>() * 0.5,
                        xp: rand::random::<u64>() % 5000,
                    },
                    AgentSkill {
                        name: "Resource Management".to_string(),
                        proficiency: 0.3 + rand::random::<f64>() * 0.7,
                        xp: rand::random::<u64>() % 3000,
                    },
                    AgentSkill {
                        name: "Communication".to_string(),
                        proficiency: 0.4 + rand::random::<f64>() * 0.6,
                        xp: rand::random::<u64>() % 4000,
                    },
                ],
                current_task: if *status == AgentStatus::Working {
                    Some(format!("Task-{}", rand::random::<u32>() % 1000))
                } else {
                    None
                },
                node_id: Some(format!("node-{}", rand::random::<u8>() % 5 + 1)),
                created_at: "2024-01-15T10:00:00Z".to_string(),
                last_active: chrono::Utc::now().to_rfc3339(),
            }
        })
        .collect()
}

#[cfg(not(feature = "stratoswarm-live"))]
fn generate_mock_events() -> Vec<EvolutionEvent> {
    vec![
        EvolutionEvent {
            id: "evo-001".to_string(),
            agent_id: "alpha".to_string(),
            agent_name: "Alpha-1".to_string(),
            from_tier: AgentTier::Platinum,
            to_tier: AgentTier::Diamond,
            xp_at_evolution: 50000,
            timestamp: "2024-12-20T14:30:00Z".to_string(),
        },
        EvolutionEvent {
            id: "evo-002".to_string(),
            agent_id: "beta".to_string(),
            agent_name: "Beta-2".to_string(),
            from_tier: AgentTier::Gold,
            to_tier: AgentTier::Platinum,
            xp_at_evolution: 15000,
            timestamp: "2024-12-18T09:15:00Z".to_string(),
        },
        EvolutionEvent {
            id: "evo-003".to_string(),
            agent_id: "kappa".to_string(),
            agent_name: "Kappa-10".to_string(),
            from_tier: AgentTier::Bronze,
            to_tier: AgentTier::Silver,
            xp_at_evolution: 1000,
            timestamp: "2024-12-15T16:45:00Z".to_string(),
        },
    ]
}

#[cfg(not(feature = "stratoswarm-live"))]
fn generate_mock_tasks() -> Vec<AgentTask> {
    vec![
        AgentTask {
            id: "task-001".to_string(),
            agent_id: "alpha".to_string(),
            task_type: "compute".to_string(),
            description: "Process batch analytics job".to_string(),
            status: "running".to_string(),
            progress: 67.5,
            xp_reward: 50,
            started_at: chrono::Utc::now()
                .checked_sub_signed(chrono::Duration::minutes(15))
                .unwrap()
                .to_rfc3339(),
            estimated_completion: Some(
                chrono::Utc::now()
                    .checked_add_signed(chrono::Duration::minutes(10))
                    .unwrap()
                    .to_rfc3339(),
            ),
        },
        AgentTask {
            id: "task-002".to_string(),
            agent_id: "beta".to_string(),
            task_type: "storage".to_string(),
            description: "Replicate data across nodes".to_string(),
            status: "running".to_string(),
            progress: 45.0,
            xp_reward: 35,
            started_at: chrono::Utc::now()
                .checked_sub_signed(chrono::Duration::minutes(8))
                .unwrap()
                .to_rfc3339(),
            estimated_completion: Some(
                chrono::Utc::now()
                    .checked_add_signed(chrono::Duration::minutes(20))
                    .unwrap()
                    .to_rfc3339(),
            ),
        },
        AgentTask {
            id: "task-003".to_string(),
            agent_id: "epsilon".to_string(),
            task_type: "network".to_string(),
            description: "Monitor cluster connectivity".to_string(),
            status: "running".to_string(),
            progress: 100.0,
            xp_reward: 15,
            started_at: chrono::Utc::now()
                .checked_sub_signed(chrono::Duration::hours(2))
                .unwrap()
                .to_rfc3339(),
            estimated_completion: None,
        },
    ]
}
