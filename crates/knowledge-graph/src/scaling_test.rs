//! Isolated test for scaling module

#[cfg(test)]
mod scaling_tests {
    use crate::scaling::*;
    use crate::{Node, NodeType, Edge, EdgeType};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_scaling_module_compiles() {
        // Basic test to ensure the scaling module compiles
        let config = ScalingConfig::default();
        let graph = ScaledKnowledgeGraph::new(config).await.unwrap();
        
        // Add a simple node
        let node = Node::new(NodeType::Agent, HashMap::new());
        graph.add_node("test-node".to_string(), node).await?;
        
        // Run a simple query
        let query = DistributedQuery {
            id: "test-query".to_string(),
            target_shards: vec![0],
            pattern: QueryPattern::NodesByType(NodeType::Agent),
            limit: None,
        };
        
        let result = graph.query(query).await.unwrap();
        assert_eq!(result.nodes.len(), 1);
    }
}