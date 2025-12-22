use exorust_business_interface::BusinessInterface;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
struct TestAgent {
    id: String,
    agent_type: AgentType,
}

#[derive(Clone, Debug, PartialEq)]
enum AgentType {
    Basic,
    LLMEnabled,
    Evolutionary,
}

struct SwarmMetrics {
    max_agents: usize,
    llm_concurrent: usize,
    generations: usize,
}

struct SwarmTester {
    agents: Arc<Mutex<Vec<TestAgent>>>,
    metrics: Arc<Mutex<SwarmMetrics>>,
}

impl SwarmTester {
    fn new() -> Self {
        Self {
            agents: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(SwarmMetrics {
                max_agents: 0,
                llm_concurrent: 0,
                generations: 0,
            })),
        }
    }

    fn spawn_agent(&self, id: String, agent_type: AgentType) -> Result<(), String> {
        let agent = TestAgent { id, agent_type };
        self.agents.lock().unwrap().push(agent);
        Ok(())
    }

    fn get_agent_count(&self) -> usize {
        self.agents.lock().unwrap().len()
    }

    fn get_agent_count_by_type(&self, agent_type: AgentType) -> usize {
        self.agents
            .lock()
            .unwrap()
            .iter()
            .filter(|a| a.agent_type == agent_type)
            .count()
    }

    fn evolve_prompt(&self, base: &str, generation: usize) -> String {
        let additions = vec![
            " using advanced algorithms",
            " with multi-dimensional analysis",
            " incorporating machine learning",
            " through distributed computation",
        ];

        format!("{}{}", base, additions.get(generation - 1).unwrap_or(&""))
    }

    fn update_metrics(&self, max_agents: usize, llm_concurrent: usize, generations: usize) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.max_agents = max_agents;
        metrics.llm_concurrent = llm_concurrent;
        metrics.generations = generations;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_spawning() {
        let tester = SwarmTester::new();

        // Test spawning basic agents
        for i in 0..10 {
            let result = tester.spawn_agent(format!("agent_{}", i), AgentType::Basic);
            assert!(result.is_ok());
        }

        assert_eq!(tester.get_agent_count(), 10);
        assert_eq!(tester.get_agent_count_by_type(AgentType::Basic), 10);
    }

    #[test]
    fn test_mixed_agent_types() {
        let tester = SwarmTester::new();

        // Spawn different types
        tester
            .spawn_agent("basic_1".to_string(), AgentType::Basic)
            .unwrap();
        tester
            .spawn_agent("llm_1".to_string(), AgentType::LLMEnabled)
            .unwrap();
        tester
            .spawn_agent("evo_1".to_string(), AgentType::Evolutionary)
            .unwrap();

        assert_eq!(tester.get_agent_count(), 3);
        assert_eq!(tester.get_agent_count_by_type(AgentType::Basic), 1);
        assert_eq!(tester.get_agent_count_by_type(AgentType::LLMEnabled), 1);
        assert_eq!(tester.get_agent_count_by_type(AgentType::Evolutionary), 1);
    }

    #[test]
    fn test_prompt_evolution() {
        let tester = SwarmTester::new();

        let base = "Analyze data and find patterns";

        let gen1 = tester.evolve_prompt(base, 1);
        assert_eq!(
            gen1,
            "Analyze data and find patterns using advanced algorithms"
        );

        let gen2 = tester.evolve_prompt(base, 2);
        assert_eq!(
            gen2,
            "Analyze data and find patterns with multi-dimensional analysis"
        );

        let gen3 = tester.evolve_prompt(base, 3);
        assert_eq!(
            gen3,
            "Analyze data and find patterns incorporating machine learning"
        );

        let gen4 = tester.evolve_prompt(base, 4);
        assert_eq!(
            gen4,
            "Analyze data and find patterns through distributed computation"
        );
    }

    #[test]
    fn test_metrics_tracking() {
        let tester = SwarmTester::new();

        // Update metrics
        tester.update_metrics(1000, 20, 4);

        let metrics = tester.metrics.lock().unwrap();
        assert_eq!(metrics.max_agents, 1000);
        assert_eq!(metrics.llm_concurrent, 20);
        assert_eq!(metrics.generations, 4);
    }

    #[test]
    fn test_large_scale_spawning() {
        let tester = SwarmTester::new();

        // Spawn many agents
        for i in 0..1000 {
            let agent_type = match i % 3 {
                0 => AgentType::Basic,
                1 => AgentType::LLMEnabled,
                _ => AgentType::Evolutionary,
            };

            tester
                .spawn_agent(format!("agent_{}", i), agent_type)
                .unwrap();
        }

        assert_eq!(tester.get_agent_count(), 1000);

        // Check distribution
        let basic_count = tester.get_agent_count_by_type(AgentType::Basic);
        let llm_count = tester.get_agent_count_by_type(AgentType::LLMEnabled);
        let evo_count = tester.get_agent_count_by_type(AgentType::Evolutionary);

        // Should be roughly equal distribution
        assert!(basic_count >= 333 && basic_count <= 334);
        assert!(llm_count >= 333 && llm_count <= 334);
        assert!(evo_count >= 333 && evo_count <= 334);
    }
}

#[tokio::test]
async fn test_concurrent_agent_operations() {
    let tester = Arc::new(SwarmTester::new());
    let mut handles = vec![];

    // Spawn agents concurrently
    for i in 0..10 {
        let tester_clone = tester.clone();
        let handle = tokio::spawn(async move {
            tester_clone.spawn_agent(format!("concurrent_{}", i), AgentType::LLMEnabled)
        });
        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    assert_eq!(tester.get_agent_count(), 10);
    assert_eq!(tester.get_agent_count_by_type(AgentType::LLMEnabled), 10);
}

#[tokio::test]
async fn test_business_interface_integration() {
    // Test that BusinessInterface can be created
    let result = BusinessInterface::new(None).await;
    assert!(
        result.is_ok(),
        "Failed to create BusinessInterface: {:?}",
        result.err()
    );

    let interface = result.unwrap();

    // Test listing goals (should be empty initially)
    let goals = interface.list_active_goals();
    assert_eq!(goals.len(), 0);
}
