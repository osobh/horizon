#[cfg(test)]
mod test_knowledge_graph_api {
    use crate::knowledge::{GraphQuery, KnowledgeEdge, KnowledgeGraph, KnowledgeNode, QueryResult};
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    #[test]
    fn test_knowledge_graph_has_required_methods() {
        let mut graph = KnowledgeGraph::new(128);

        // Test add_node method exists
        let node = KnowledgeNode {
            id: 1,
            embedding: vec![0.0; 128],
            metadata: "test".to_string(),
        };
        graph.add_node(node);

        // Test add_edge method exists
        let edge = KnowledgeEdge {
            source: 1,
            target: 2,
            edge_type: "connects".to_string(),
            weight: 1.0,
        };
        graph.add_edge(edge);

        // Test node_count method exists
        assert!(graph.node_count() > 0);

        // Test upload_to_gpu method exists
        let device = Arc::new(CudaContext::new(0).unwrap());
        let _gpu_graph = graph.upload_to_gpu(device).unwrap();
    }

    #[test]
    fn test_graph_query_fields() {
        // GraphQuery should have query_embedding field
        let query = GraphQuery {
            query_text: "test".to_string(),
            query_embedding: vec![0.0; 768],
            max_results: 10,
            threshold: 0.5,
        };

        assert_eq!(query.query_embedding.len(), 768);
    }

    #[test]
    fn test_query_result_structure() {
        // QueryResult should match expected structure
        let result = QueryResult {
            nodes: vec![],
            scores: vec![],
            execution_time_ms: 0.0,
        };

        assert_eq!(result.nodes.len(), 0);
    }
}
