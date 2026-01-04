//! Cross-agent XP sharing and collective learning mechanisms
//!
//! This module enables agents to share experiences and learn from each other,
//! creating emergent collective intelligence through XP distribution and knowledge transfer.

use crate::EvolutionError;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use stratoswarm_agent_core::agent::{Agent, AgentId};
use tokio::sync::RwLock;

/// Type of knowledge that can be shared between agents
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeType {
    /// Problem-solving patterns and strategies
    ProblemSolving,
    /// Optimization techniques discovered
    Optimization,
    /// Error recovery methods
    ErrorRecovery,
    /// Resource usage patterns
    ResourceManagement,
    /// Goal completion strategies
    GoalAchievement,
    /// Performance improvement techniques
    PerformanceBoost,
    /// Collaborative behavior patterns
    Collaboration,
    /// Adaptation strategies for new environments
    Adaptation,
    /// Custom knowledge type with identifier
    Custom(String),
}

/// Shareable knowledge package containing experience and methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePackage {
    pub id: String,
    pub knowledge_type: KnowledgeType,
    pub source_agent: AgentId,
    pub creation_time: DateTime<Utc>,
    pub experience_data: String,
    pub success_metrics: KnowledgeMetrics,
    pub usage_count: u64,
    pub effectiveness_rating: f64,
    pub prerequisites: Vec<String>, // Conditions needed to apply this knowledge
    pub context_tags: HashSet<String>,
}

/// Metrics associated with a knowledge package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeMetrics {
    pub success_rate: f64,
    pub performance_improvement: f64,
    pub resource_efficiency: f64,
    pub applicability_score: f64,
    pub learning_curve_steepness: f64,
}

/// Learning session result when an agent applies shared knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResult {
    pub learner_agent: AgentId,
    pub knowledge_package_id: String,
    pub learning_success: bool,
    pub performance_delta: f64,
    pub adaptation_time: std::time::Duration,
    pub xp_gained: u64,
    pub feedback_rating: f64,
}

/// Configuration for XP sharing behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XPSharingConfig {
    /// Base percentage of XP to share with the collective pool
    pub collective_share_percentage: f64,
    /// Percentage of XP to share directly with mentor agents
    pub mentor_share_percentage: f64,
    /// Bonus multiplier for teaching other agents
    pub teaching_bonus_multiplier: f64,
    /// Bonus multiplier for successful learning from others
    pub learning_bonus_multiplier: f64,
    /// Maximum XP that can be shared in a single transaction
    pub max_share_amount: u64,
    /// Minimum agent level required to become a mentor
    pub mentor_threshold_level: u32,
    /// Decay rate for old knowledge packages
    pub knowledge_decay_rate: f64,
}

impl Default for XPSharingConfig {
    fn default() -> Self {
        Self {
            collective_share_percentage: 0.05, // 5% to collective pool
            mentor_share_percentage: 0.10,     // 10% to mentors
            teaching_bonus_multiplier: 1.5,
            learning_bonus_multiplier: 1.2,
            max_share_amount: 100,
            mentor_threshold_level: 3,
            knowledge_decay_rate: 0.95,
        }
    }
}

/// Cross-agent learning and XP sharing coordinator
pub struct AgentXPSharingCoordinator {
    config: XPSharingConfig,
    /// Collective XP pool for community contributions
    collective_xp_pool: Arc<RwLock<u64>>,
    /// Knowledge packages available for sharing
    knowledge_repository: Arc<DashMap<String, KnowledgePackage>>,
    /// Learning relationships and mentorship networks
    learning_network: Arc<DashMap<AgentId, LearningProfile>>,
    /// Track XP sharing transactions
    sharing_history: Arc<RwLock<Vec<XPSharingTransaction>>>,
    /// Performance tracking for shared knowledge
    knowledge_performance: Arc<DashMap<String, Vec<LearningResult>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProfile {
    pub agent_id: AgentId,
    pub mentor_agents: HashSet<AgentId>,
    pub mentee_agents: HashSet<AgentId>,
    pub knowledge_contributed: Vec<String>,
    pub knowledge_learned: Vec<String>,
    pub teaching_effectiveness: f64,
    pub learning_receptivity: f64,
    pub specialization_areas: HashSet<KnowledgeType>,
    pub reputation_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XPSharingTransaction {
    pub transaction_id: String,
    pub timestamp: DateTime<Utc>,
    pub source_agent: AgentId,
    pub recipient_agent: Option<AgentId>, // None for collective pool
    pub xp_amount: u64,
    pub transaction_type: XPTransactionType,
    pub knowledge_package_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum XPTransactionType {
    CollectiveContribution,
    MentorReward,
    TeachingBonus,
    LearningBonus,
    KnowledgeRoyalty,
    CollaborationReward,
}

impl Default for AgentXPSharingCoordinator {
    fn default() -> Self {
        Self::new(XPSharingConfig::default())
    }
}

impl AgentXPSharingCoordinator {
    pub fn new(config: XPSharingConfig) -> Self {
        Self {
            config,
            collective_xp_pool: Arc::new(RwLock::new(0)),
            knowledge_repository: Arc::new(DashMap::new()),
            learning_network: Arc::new(DashMap::new()),
            sharing_history: Arc::new(RwLock::new(Vec::new())),
            knowledge_performance: Arc::new(DashMap::new()),
        }
    }

    /// Register an agent in the learning network
    pub async fn register_agent(&self, agent: &Agent) -> Result<(), EvolutionError> {
        let agent_id = agent.id();
        let stats = agent.stats().await;

        let profile = LearningProfile {
            agent_id,
            mentor_agents: HashSet::new(),
            mentee_agents: HashSet::new(),
            knowledge_contributed: Vec::new(),
            knowledge_learned: Vec::new(),
            teaching_effectiveness: 1.0,
            learning_receptivity: 1.0,
            specialization_areas: HashSet::new(),
            reputation_score: stats.level as f64,
        };

        self.learning_network.insert(agent_id, profile);
        Ok(())
    }

    /// Share XP when an agent gains experience
    pub async fn process_xp_gain(
        &self,
        agent: &Agent,
        xp_gained: u64,
        experience_type: &str,
    ) -> Result<XPSharingResult, EvolutionError> {
        let agent_id = agent.id();
        let stats = agent.stats().await;

        let mut result = XPSharingResult {
            original_xp: xp_gained,
            shared_to_collective: 0,
            shared_to_mentors: 0,
            teaching_bonus: 0,
            net_xp_retained: xp_gained,
            transactions: Vec::new(),
        };

        // Calculate sharing amounts
        let collective_share =
            ((xp_gained as f64) * self.config.collective_share_percentage) as u64;
        let collective_share = collective_share.min(self.config.max_share_amount);

        if collective_share > 0 {
            // Add to collective pool
            *self.collective_xp_pool.write().await += collective_share;
            result.shared_to_collective = collective_share;
            result.net_xp_retained -= collective_share;

            // Record transaction
            let transaction = XPSharingTransaction {
                transaction_id: uuid::Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source_agent: agent_id,
                recipient_agent: None,
                xp_amount: collective_share,
                transaction_type: XPTransactionType::CollectiveContribution,
                knowledge_package_id: None,
            };
            self.sharing_history.write().await.push(transaction.clone());
            result.transactions.push(transaction);
        }

        // Share with mentors if agent has them
        if let Some(profile) = self.learning_network.get(&agent_id) {
            let mentor_share_each = if !profile.mentor_agents.is_empty() {
                let total_mentor_share =
                    ((xp_gained as f64) * self.config.mentor_share_percentage) as u64;
                let total_mentor_share = total_mentor_share.min(self.config.max_share_amount);
                total_mentor_share / profile.mentor_agents.len() as u64
            } else {
                0
            };

            if mentor_share_each > 0 {
                for &mentor_id in &profile.mentor_agents {
                    result.shared_to_mentors += mentor_share_each;
                    result.net_xp_retained -= mentor_share_each;

                    let transaction = XPSharingTransaction {
                        transaction_id: uuid::Uuid::new_v4().to_string(),
                        timestamp: Utc::now(),
                        source_agent: agent_id,
                        recipient_agent: Some(mentor_id),
                        xp_amount: mentor_share_each,
                        transaction_type: XPTransactionType::MentorReward,
                        knowledge_package_id: None,
                    };
                    self.sharing_history.write().await.push(transaction.clone());
                    result.transactions.push(transaction);
                }
            }
        }

        // Award teaching bonus if agent is teaching others
        if stats.level >= self.config.mentor_threshold_level {
            if let Some(profile) = self.learning_network.get(&agent_id) {
                if !profile.mentee_agents.is_empty() {
                    let teaching_bonus =
                        ((xp_gained as f64) * (self.config.teaching_bonus_multiplier - 1.0)) as u64;
                    if teaching_bonus > 0 {
                        result.teaching_bonus = teaching_bonus;
                        result.net_xp_retained += teaching_bonus;

                        agent
                            .award_xp(
                                teaching_bonus,
                                format!("Teaching bonus for {} experience", experience_type),
                                "teaching_bonus".to_string(),
                            )
                            .await
                            .map_err(|e| EvolutionError::FitnessEvaluationFailed {
                                reason: format!("Failed to award teaching bonus: {}", e),
                            })?;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Create a knowledge package from an agent's successful experience
    pub async fn create_knowledge_package(
        &self,
        agent: &Agent,
        knowledge_type: KnowledgeType,
        experience_description: String,
        success_metrics: KnowledgeMetrics,
        context_tags: HashSet<String>,
    ) -> Result<String, EvolutionError> {
        let package_id = uuid::Uuid::new_v4().to_string();
        let agent_id = agent.id();

        let package = KnowledgePackage {
            id: package_id.clone(),
            knowledge_type: knowledge_type.clone(),
            source_agent: agent_id,
            creation_time: Utc::now(),
            experience_data: experience_description,
            success_metrics,
            usage_count: 0,
            effectiveness_rating: 0.0,
            prerequisites: Vec::new(),
            context_tags,
        };

        // Store the package
        self.knowledge_repository
            .insert(package_id.clone(), package);

        // Update agent's learning profile
        if let Some(mut profile) = self.learning_network.get_mut(&agent_id) {
            profile.knowledge_contributed.push(package_id.clone());
            profile.specialization_areas.insert(knowledge_type);
            profile.reputation_score += 0.1; // Small reputation boost
        }

        // Award XP for knowledge contribution
        agent
            .award_xp(
                50,
                "Knowledge package contribution".to_string(),
                "knowledge_sharing".to_string(),
            )
            .await
            .map_err(|e| EvolutionError::FitnessEvaluationFailed {
                reason: format!("Failed to award knowledge contribution XP: {}", e),
            })?;

        Ok(package_id)
    }

    /// Find relevant knowledge packages for an agent to learn
    pub async fn find_relevant_knowledge(
        &self,
        agent: &Agent,
        interests: &[KnowledgeType],
        context: &HashSet<String>,
    ) -> Vec<KnowledgePackage> {
        let mut relevant_packages = Vec::new();

        for entry in self.knowledge_repository.iter() {
            let package = entry.value();
            // Skip own packages
            if package.source_agent == agent.id() {
                continue;
            }

            // Check type relevance
            if interests.contains(&package.knowledge_type) {
                relevant_packages.push(package.clone());
                continue;
            }

            // Check context overlap
            let context_overlap = package.context_tags.intersection(context).count();
            if context_overlap > 0 {
                relevant_packages.push(package.clone());
            }
        }

        // Sort by effectiveness and relevance
        relevant_packages.sort_by(|a, b| {
            let score_a = a.effectiveness_rating + (a.usage_count as f64 * 0.01);
            let score_b = b.effectiveness_rating + (b.usage_count as f64 * 0.01);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        relevant_packages.into_iter().take(10).collect() // Limit to top 10
    }

    /// Agent learns from a knowledge package
    pub async fn apply_knowledge(
        &self,
        agent: &Agent,
        package_id: &str,
    ) -> Result<LearningResult, EvolutionError> {
        let agent_id = agent.id();
        let learning_start = std::time::Instant::now();

        // Get the knowledge package
        let package = self
            .knowledge_repository
            .get(package_id)
            .map(|r| r.clone())
            .ok_or_else(|| EvolutionError::FitnessEvaluationFailed {
                reason: "Knowledge package not found".to_string(),
            })?;

        // Simulate learning process (in real implementation, this would involve
        // applying the knowledge to the agent's behavior or capabilities)
        let _stats_before = agent.stats().await;
        let success_probability = package.success_metrics.applicability_score;
        let learning_success = rand::random::<f64>() < success_probability;

        let performance_delta = if learning_success {
            package.success_metrics.performance_improvement * 0.8 // Reduced effect for learner
        } else {
            -0.1 // Small penalty for failed learning attempt
        };

        // Calculate XP reward for learning
        let base_xp = if learning_success { 40 } else { 10 };
        let effectiveness_bonus = (package.effectiveness_rating * 30.0) as u64;
        let learning_xp = base_xp + effectiveness_bonus;

        // Award learning XP
        agent
            .award_xp(
                learning_xp,
                format!("Learning from {} knowledge package", package.source_agent),
                "knowledge_learning".to_string(),
            )
            .await
            .map_err(|e| EvolutionError::FitnessEvaluationFailed {
                reason: format!("Failed to award learning XP: {}", e),
            })?;

        // Update package usage
        if let Some(mut pkg) = self.knowledge_repository.get_mut(package_id) {
            pkg.usage_count += 1;
            // Update effectiveness based on success
            let weight = 1.0 / (pkg.usage_count as f64).max(1.0);
            pkg.effectiveness_rating = pkg.effectiveness_rating * (1.0 - weight)
                + (if learning_success { 1.0 } else { 0.0 }) * weight;
        }

        // Update learning profile
        if let Some(mut profile) = self.learning_network.get_mut(&agent_id) {
            profile.knowledge_learned.push(package_id.to_string());
            if learning_success {
                profile.learning_receptivity *= 1.01; // Small boost for successful learning
            }
        }

        // Award royalty XP to original creator
        let royalty_xp = learning_xp / 4; // 25% royalty
        if let Ok(creator_agents) = self.find_agents_by_id(&[package.source_agent]).await {
            if let Some(creator) = creator_agents.first() {
                creator
                    .award_xp(
                        royalty_xp,
                        format!("Royalty from knowledge package usage by {}", agent_id),
                        "knowledge_royalty".to_string(),
                    )
                    .await
                    .map_err(|_| EvolutionError::FitnessEvaluationFailed {
                        reason: "Failed to award royalty XP".to_string(),
                    })?;
            }
        }

        let result = LearningResult {
            learner_agent: agent_id,
            knowledge_package_id: package_id.to_string(),
            learning_success,
            performance_delta,
            adaptation_time: learning_start.elapsed(),
            xp_gained: learning_xp,
            feedback_rating: if learning_success { 4.5 } else { 2.0 },
        };

        // Record learning result
        self.knowledge_performance
            .entry(package_id.to_string())
            .or_insert_with(Vec::new)
            .push(result.clone());

        Ok(result)
    }

    /// Establish mentorship relationship between agents
    pub async fn create_mentorship(
        &self,
        mentor_agent: &Agent,
        mentee_agent: &Agent,
    ) -> Result<(), EvolutionError> {
        let mentor_id = mentor_agent.id();
        let mentee_id = mentee_agent.id();
        let mentor_stats = mentor_agent.stats().await;

        // Check if mentor is qualified
        if mentor_stats.level < self.config.mentor_threshold_level {
            return Err(EvolutionError::FitnessEvaluationFailed {
                reason: "Mentor agent level too low".to_string(),
            });
        }

        // Update learning network
        if let Some(mut mentor_profile) = self.learning_network.get_mut(&mentor_id) {
            mentor_profile.mentee_agents.insert(mentee_id);
        }

        if let Some(mut mentee_profile) = self.learning_network.get_mut(&mentee_id) {
            mentee_profile.mentor_agents.insert(mentor_id);
        }

        // Award XP for establishing mentorship
        mentor_agent
            .award_xp(
                75,
                format!("Established mentorship with agent {}", mentee_id),
                "mentorship".to_string(),
            )
            .await
            .map_err(|e| EvolutionError::FitnessEvaluationFailed {
                reason: format!("Failed to award mentorship XP: {}", e),
            })?;

        Ok(())
    }

    /// Distribute collective XP pool to deserving agents
    pub async fn distribute_collective_xp(&self) -> Result<CollectiveDistribution, EvolutionError> {
        let pool_amount = {
            let mut pool = self.collective_xp_pool.write().await;
            let amount = *pool;
            *pool = 0; // Reset pool
            amount
        };

        if pool_amount == 0 {
            return Ok(CollectiveDistribution {
                total_distributed: 0,
                recipients: Vec::new(),
            });
        }

        // Find agents eligible for collective distribution
        let mut eligible_agents: Vec<(AgentId, f64)> = self
            .learning_network
            .iter()
            .map(|entry| (entry.value().agent_id, entry.value().reputation_score))
            .collect();

        // Sort by reputation score
        eligible_agents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut distribution = CollectiveDistribution {
            total_distributed: 0,
            recipients: Vec::new(),
        };

        // Distribute based on reputation (top performers get more)
        let total_reputation: f64 = eligible_agents.iter().map(|(_, rep)| *rep).sum();
        if total_reputation > 0.0 {
            for (agent_id, reputation) in eligible_agents.iter().take(10) {
                // Top 10 agents
                let share = ((*reputation / total_reputation) * pool_amount as f64) as u64;
                if share > 0 {
                    // In a real implementation, you'd need a way to get Agent instances by ID
                    // For now, we'll just record the distribution
                    distribution.recipients.push(CollectiveDistributionEntry {
                        agent_id: *agent_id,
                        xp_amount: share,
                        reason: "Collective pool contribution based on reputation".to_string(),
                    });
                    distribution.total_distributed += share;
                }
            }
        }

        Ok(distribution)
    }

    /// Get learning statistics for an agent
    pub async fn get_learning_stats(&self, agent_id: &AgentId) -> Option<LearningStats> {
        let profile = self.learning_network.get(agent_id)?;

        let mut total_learning_attempts = 0;
        let mut successful_learning_attempts = 0;

        for package_id in &profile.knowledge_learned {
            if let Some(results) = self.knowledge_performance.get(package_id) {
                for result in results.iter() {
                    if result.learner_agent == *agent_id {
                        total_learning_attempts += 1;
                        if result.learning_success {
                            successful_learning_attempts += 1;
                        }
                    }
                }
            }
        }

        Some(LearningStats {
            agent_id: *agent_id,
            reputation_score: profile.reputation_score,
            knowledge_packages_contributed: profile.knowledge_contributed.len(),
            knowledge_packages_learned: profile.knowledge_learned.len(),
            mentor_count: profile.mentor_agents.len(),
            mentee_count: profile.mentee_agents.len(),
            teaching_effectiveness: profile.teaching_effectiveness,
            learning_success_rate: if total_learning_attempts > 0 {
                successful_learning_attempts as f64 / total_learning_attempts as f64
            } else {
                0.0
            },
            specialization_areas: profile.specialization_areas.clone(),
        })
    }

    // Helper method to find agents by ID (would need proper implementation)
    async fn find_agents_by_id(
        &self,
        _agent_ids: &[AgentId],
    ) -> Result<Vec<Agent>, EvolutionError> {
        // This is a placeholder - in a real implementation, you'd have a registry
        // or database of agents that you could query by ID
        Ok(Vec::new())
    }
}

/// Result of XP sharing process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XPSharingResult {
    pub original_xp: u64,
    pub shared_to_collective: u64,
    pub shared_to_mentors: u64,
    pub teaching_bonus: u64,
    pub net_xp_retained: u64,
    pub transactions: Vec<XPSharingTransaction>,
}

/// Result of collective XP distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveDistribution {
    pub total_distributed: u64,
    pub recipients: Vec<CollectiveDistributionEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveDistributionEntry {
    pub agent_id: AgentId,
    pub xp_amount: u64,
    pub reason: String,
}

/// Learning statistics for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    pub agent_id: AgentId,
    pub reputation_score: f64,
    pub knowledge_packages_contributed: usize,
    pub knowledge_packages_learned: usize,
    pub mentor_count: usize,
    pub mentee_count: usize,
    pub teaching_effectiveness: f64,
    pub learning_success_rate: f64,
    pub specialization_areas: HashSet<KnowledgeType>,
}

impl LearningStats {
    pub fn summary(&self) -> String {
        format!(
            "Learning Profile for {}:\n\
            Reputation: {:.2}\n\
            Knowledge: {} contributed, {} learned\n\
            Network: {} mentors, {} mentees\n\
            Success Rate: {:.1}% learning, {:.2} teaching effectiveness\n\
            Specializations: {} areas",
            self.agent_id,
            self.reputation_score,
            self.knowledge_packages_contributed,
            self.knowledge_packages_learned,
            self.mentor_count,
            self.mentee_count,
            self.learning_success_rate * 100.0,
            self.teaching_effectiveness,
            self.specialization_areas.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stratoswarm_agent_core::agent::{Agent, AgentConfig};

    async fn create_test_agent(name: &str, level: u32) -> Agent {
        let config = AgentConfig {
            name: name.to_string(),
            ..Default::default()
        };

        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();

        // Award XP to reach desired level
        if level > 1 {
            let xp_needed = match level {
                2 => 100,
                3 => 250,
                4 => 500,
                5 => 1000,
                _ => 1000 + (level - 5) as u64 * 500,
            };
            agent
                .award_xp(xp_needed, "Level setup".to_string(), "test".to_string())
                .await
                .unwrap();
        }

        agent
    }

    #[tokio::test]
    async fn test_agent_registration() {
        let coordinator = AgentXPSharingCoordinator::default();
        let agent = create_test_agent("test_agent", 1).await;

        coordinator.register_agent(&agent).await.unwrap();

        let stats = coordinator.get_learning_stats(&agent.id()).await;
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().reputation_score, 1.0);
    }

    #[tokio::test]
    async fn test_xp_sharing() {
        let coordinator = AgentXPSharingCoordinator::default();
        let agent = create_test_agent("sharing_agent", 3).await;

        coordinator.register_agent(&agent).await.unwrap();

        let result = coordinator
            .process_xp_gain(&agent, 100, "test_experience")
            .await
            .unwrap();

        assert_eq!(result.original_xp, 100);
        assert!(result.shared_to_collective > 0);
        assert!(result.net_xp_retained < 100);
        assert!(!result.transactions.is_empty());
    }

    #[tokio::test]
    async fn test_knowledge_package_creation() {
        let coordinator = AgentXPSharingCoordinator::default();
        let agent = create_test_agent("knowledge_creator", 2).await;

        coordinator.register_agent(&agent).await.unwrap();

        let metrics = KnowledgeMetrics {
            success_rate: 0.8,
            performance_improvement: 0.15,
            resource_efficiency: 0.9,
            applicability_score: 0.7,
            learning_curve_steepness: 0.6,
        };

        let mut context_tags = HashSet::new();
        context_tags.insert("optimization".to_string());
        context_tags.insert("performance".to_string());

        let package_id = coordinator
            .create_knowledge_package(
                &agent,
                KnowledgeType::Optimization,
                "Efficient resource allocation strategy".to_string(),
                metrics,
                context_tags,
            )
            .await
            .unwrap();

        assert!(!package_id.is_empty());

        // Verify package was stored
        assert!(coordinator.knowledge_repository.contains_key(&package_id));
    }

    #[tokio::test]
    async fn test_knowledge_learning() {
        let coordinator = AgentXPSharingCoordinator::default();
        let creator = create_test_agent("creator", 3).await;
        let learner = create_test_agent("learner", 2).await;

        coordinator.register_agent(&creator).await.unwrap();
        coordinator.register_agent(&learner).await.unwrap();

        // Create knowledge package
        let metrics = KnowledgeMetrics {
            success_rate: 0.9,
            performance_improvement: 0.2,
            resource_efficiency: 0.85,
            applicability_score: 0.8,
            learning_curve_steepness: 0.7,
        };

        let package_id = coordinator
            .create_knowledge_package(
                &creator,
                KnowledgeType::ProblemSolving,
                "Advanced problem-solving technique".to_string(),
                metrics,
                HashSet::new(),
            )
            .await
            .unwrap();

        // Learner applies knowledge
        let result = coordinator
            .apply_knowledge(&learner, &package_id)
            .await
            .unwrap();

        assert_eq!(result.learner_agent, learner.id());
        assert_eq!(result.knowledge_package_id, package_id);
        assert!(result.xp_gained > 0);
    }

    #[tokio::test]
    async fn test_mentorship_creation() {
        let coordinator = AgentXPSharingCoordinator::default();
        let mentor = create_test_agent("mentor", 4).await; // High level
        let mentee = create_test_agent("mentee", 1).await;

        coordinator.register_agent(&mentor).await.unwrap();
        coordinator.register_agent(&mentee).await.unwrap();

        coordinator
            .create_mentorship(&mentor, &mentee)
            .await
            .unwrap();

        // Verify mentorship was established
        let mentor_stats = coordinator.get_learning_stats(&mentor.id()).await.unwrap();
        let mentee_stats = coordinator.get_learning_stats(&mentee.id()).await.unwrap();

        assert_eq!(mentor_stats.mentee_count, 1);
        assert_eq!(mentee_stats.mentor_count, 1);
    }

    #[tokio::test]
    async fn test_collective_xp_distribution() {
        let coordinator = AgentXPSharingCoordinator::default();
        let agent = create_test_agent("contributor", 3).await;

        coordinator.register_agent(&agent).await.unwrap();

        // Add some XP to collective pool
        *coordinator.collective_xp_pool.write().await = 500;

        let distribution = coordinator.distribute_collective_xp().await.unwrap();

        assert!(distribution.total_distributed <= 500);
        // Pool should be reset
        assert_eq!(*coordinator.collective_xp_pool.read().await, 0);
    }

    #[tokio::test]
    async fn test_knowledge_relevance_finding() {
        let coordinator = AgentXPSharingCoordinator::default();
        let creator = create_test_agent("creator", 3).await;
        let seeker = create_test_agent("seeker", 2).await;

        coordinator.register_agent(&creator).await.unwrap();
        coordinator.register_agent(&seeker).await.unwrap();

        // Create relevant knowledge packages
        let metrics = KnowledgeMetrics {
            success_rate: 0.8,
            performance_improvement: 0.15,
            resource_efficiency: 0.9,
            applicability_score: 0.75,
            learning_curve_steepness: 0.6,
        };

        let mut context_tags = HashSet::new();
        context_tags.insert("performance".to_string());

        coordinator
            .create_knowledge_package(
                &creator,
                KnowledgeType::Optimization,
                "Performance optimization technique".to_string(),
                metrics,
                context_tags.clone(),
            )
            .await
            .unwrap();

        // Find relevant knowledge
        let interests = vec![KnowledgeType::Optimization];
        let packages = coordinator
            .find_relevant_knowledge(&seeker, &interests, &context_tags)
            .await;

        assert_eq!(packages.len(), 1);
        assert_eq!(packages[0].knowledge_type, KnowledgeType::Optimization);
    }

    #[tokio::test]
    async fn test_learning_stats_summary() {
        let mut specializations = HashSet::new();
        specializations.insert(KnowledgeType::Optimization);
        specializations.insert(KnowledgeType::ProblemSolving);

        let stats = LearningStats {
            agent_id: AgentId::new(),
            reputation_score: 4.5,
            knowledge_packages_contributed: 3,
            knowledge_packages_learned: 7,
            mentor_count: 2,
            mentee_count: 1,
            teaching_effectiveness: 0.85,
            learning_success_rate: 0.75,
            specialization_areas: specializations,
        };

        let summary = stats.summary();
        assert!(summary.contains("4.5"));
        assert!(summary.contains("3 contributed"));
        assert!(summary.contains("7 learned"));
        assert!(summary.contains("2 mentors"));
        assert!(summary.contains("75.0% learning"));
    }
}
