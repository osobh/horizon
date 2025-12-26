# GPU Agents API Reference

## Table of Contents

1. [Core Types](#core-types)
2. [Consensus API](#consensus-api)
3. [Synthesis API](#synthesis-api)
4. [Evolution API](#evolution-api)
5. [Knowledge Graph API](#knowledge-graph-api)
6. [Streaming API](#streaming-api)
7. [Memory Management API](#memory-management-api)
8. [Performance API](#performance-api)
9. [Utilization API](#utilization-api)

## Core Types

### GpuAgent

```rust
pub struct GpuAgent {
    pub id: u32,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub fitness: f32,
    pub neural_state: [f32; 256],
    pub memory_state: [f32; 256],
    pub generation: u32,
}
```

### GpuSwarmConfig

```rust
pub struct GpuSwarmConfig {
    pub device_id: i32,
    pub max_agents: usize,
    pub block_size: u32,
    pub evolution_interval: u32,
    pub enable_llm: bool,
    pub enable_knowledge_graph: bool,
    pub enable_collective_knowledge: bool,
}
```

## Consensus API

### VotingSystem

```rust
pub struct VotingSystem {
    pub fn new(num_agents: usize, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn propose(&self, proposal: Proposal) -> Result<ProposalId>
    pub async fn vote(&self, agent_id: AgentId, proposal_id: ProposalId, vote: Vote) -> Result<()>
    pub async fn tally_votes(&self, proposal_id: ProposalId) -> Result<VoteResult>
    pub fn get_consensus_latency(&self) -> Duration
}
```

### LeaderElection

```rust
pub struct LeaderElection {
    pub fn new(config: ElectionConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn elect_leader(&self) -> Result<AgentId>
    pub async fn verify_leader(&self, leader_id: AgentId) -> Result<bool>
    pub fn get_current_leader(&self) -> Option<AgentId>
}
```

## Synthesis API

### PatternMatcher

```rust
pub struct PatternMatcher {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn add_pattern(&self, pattern: Pattern) -> Result<PatternId>
    pub async fn match_patterns(&self, input: &str) -> Result<Vec<Match>>
    pub fn compile_patterns(&self) -> Result<()>
}
```

### TemplateEngine

```rust
pub struct TemplateEngine {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn register_template(&self, name: &str, template: Template) -> Result<()>
    pub async fn instantiate(&self, name: &str, params: TemplateParams) -> Result<String>
    pub async fn batch_instantiate(&self, requests: Vec<InstantiateRequest>) -> Result<Vec<String>>
}
```

### AstProcessor

```rust
pub struct AstProcessor {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn parse(&self, code: &str) -> Result<Ast>
    pub async fn transform(&self, ast: Ast, transform: Transform) -> Result<Ast>
    pub async fn generate_code(&self, ast: Ast) -> Result<String>
}
```

## Evolution API

### GpuEvolutionEngine

```rust
pub struct GpuEvolutionEngine {
    pub fn new(config: EvolutionConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn initialize_population(&mut self, size: usize) -> Result<()>
    pub async fn evolve_generation(&mut self) -> Result<EvolutionMetrics>
    pub async fn get_best_agent(&self) -> Result<GpuAgent>
    pub fn get_generation(&self) -> u32
}
```

### Evolution Strategies

#### ADAS (Automated Design of Agentic Systems)

```rust
pub struct AdasEngine {
    pub fn new(config: AdasConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn meta_optimize(&mut self, iterations: u32) -> Result<AdasResults>
    pub async fn generate_architectures(&self, count: usize) -> Result<Vec<Architecture>>
}
```

#### DGM (Darwin Gödel Machine)

```rust
pub struct DgmEngine {
    pub fn new(config: DgmConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn self_improve(&mut self) -> Result<DgmImprovement>
    pub async fn verify_improvement(&self, improvement: &DgmImprovement) -> Result<bool>
}
```

#### Swarm (Particle Swarm Optimization)

```rust
pub struct SwarmEngine {
    pub fn new(config: SwarmConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn update_swarm(&mut self) -> Result<SwarmMetrics>
    pub async fn get_global_best(&self) -> Result<Particle>
}
```

### Hybrid Coordinator

```rust
pub struct HybridCoordinator {
    pub fn new(config: HybridConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn allocate_resources(&mut self, mode: AllocationMode) -> Result<()>
    pub async fn migrate_population(&mut self, from: Strategy, to: Strategy, count: usize) -> Result<()>
    pub async fn coordinate_strategies(&mut self) -> Result<CoordinationMetrics>
}
```

## Knowledge Graph API

### EnhancedGpuKnowledgeGraph

```rust
pub struct EnhancedGpuKnowledgeGraph {
    pub fn new(config: GraphConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn add_node(&self, node: KnowledgeNode) -> Result<NodeId>
    pub async fn add_edge(&self, edge: KnowledgeEdge) -> Result<EdgeId>
    pub async fn query(&self, query: GraphQuery) -> Result<Vec<QueryResult>>
    pub async fn run_similarity_search(&self, embedding: &[f32], k: usize) -> Result<Vec<SimilarityResult>>
}
```

### Temporal Graph

```rust
pub struct TemporalKnowledgeGraph {
    pub fn new(config: TemporalConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn add_temporal_node(&self, node: TemporalNode) -> Result<NodeId>
    pub async fn add_temporal_edge(&self, edge: TemporalEdge) -> Result<EdgeId>
    pub async fn query_time_window(&self, start: Timestamp, end: Timestamp) -> Result<TemporalSubgraph>
    pub async fn analyze_causality(&self, event_a: NodeId, event_b: NodeId) -> Result<CausalityScore>
}
```

### Reasoning Engine

```rust
pub struct ReasoningEngine {
    pub fn new(config: ReasoningConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn add_fact(&self, fact: LogicalFact) -> Result<()>
    pub async fn add_rule(&self, rule: InferenceRule) -> Result<()>
    pub async fn reason(&self, query: ReasoningQuery, max_hops: u32) -> Result<Vec<InferenceResult>>
}
```

### Graph Neural Network

```rust
pub struct GraphNeuralNetwork {
    pub fn new(config: GnnConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn forward(&self, graph: &KnowledgeGraph) -> Result<NodeEmbeddings>
    pub async fn train_step(&mut self, graph: &KnowledgeGraph, labels: &Labels) -> Result<f32>
    pub async fn predict_node_class(&self, node_id: NodeId) -> Result<NodeClass>
    pub async fn predict_link(&self, source: NodeId, target: NodeId) -> Result<f32>
}
```

## Streaming API

### GpuStreamProcessor

```rust
pub struct GpuStreamProcessor {
    pub fn new(config: GpuStreamConfig, device: Arc<CudaDevice>) -> Result<Self>
    pub async fn process_batch(&self, data: Vec<Vec<u8>>) -> Result<ProcessedBatch>
    pub async fn add_transform(&mut self, transform: TransformType) -> Result<()>
    pub fn get_metrics(&self) -> StreamMetrics
}
```

### String Operations

```rust
pub struct GpuStringProcessor {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn batch_uppercase(&self, strings: Vec<String>) -> Result<Vec<String>>
    pub async fn batch_pattern_match(&self, strings: Vec<String>, pattern: &str) -> Result<Vec<bool>>
    pub async fn batch_replace(&self, strings: Vec<String>, pattern: &str, replacement: &str) -> Result<Vec<String>>
    pub async fn batch_sort(&self, strings: Vec<String>) -> Result<Vec<String>>
}
```

### Huffman Compression

```rust
pub struct GpuHuffmanProcessor {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn compress_batch(&self, data: Vec<Vec<u8>>, level: CompressionLevel) -> Result<Vec<CompressedData>>
    pub async fn decompress_batch(&self, compressed: Vec<CompressedData>) -> Result<Vec<Vec<u8>>>
    pub fn get_compression_stats(&self) -> CompressionStats
}
```

## Memory Management API

### TierManager

```rust
pub struct TierManager {
    pub fn new(config: TierConfig) -> Result<Self>
    pub async fn allocate_page(&self, tier: TierLevel, size: usize) -> Result<PageId>
    pub async fn migrate_page(&self, page_id: PageId, target_tier: TierLevel) -> Result<()>
    pub async fn evict_pages(&self, count: usize) -> Result<Vec<PageId>>
    pub fn get_tier_stats(&self) -> TierStatistics
}
```

### UnifiedMemoryManager

```rust
pub struct UnifiedMemoryManager {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub fn allocate_unified(&self, size: usize) -> Result<UnifiedBuffer>
    pub fn prefetch_to_gpu(&self, buffer: &UnifiedBuffer) -> Result<()>
    pub fn prefetch_to_cpu(&self, buffer: &UnifiedBuffer) -> Result<()>
}
```

### PrefetchEngine

```rust
pub struct PrefetchEngine {
    pub fn new(config: PrefetchConfig) -> Result<Self>
    pub async fn record_access(&self, page_id: PageId, access_type: AccessType) -> Result<()>
    pub async fn predict_next_access(&self) -> Result<Vec<PageId>>
    pub async fn prefetch_pages(&self, pages: Vec<PageId>) -> Result<()>
    pub fn get_hit_rate(&self) -> f32
}
```

## Performance API

### PerformanceOptimizer

```rust
pub struct PerformanceOptimizer {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn analyze_workload(&self) -> Result<WorkloadAnalysis>
    pub async fn optimize_configuration(&mut self) -> Result<OptimizationResult>
    pub async fn apply_optimizations(&self, optimizations: Vec<Optimization>) -> Result<()>
    pub fn get_performance_metrics(&self) -> PerformanceMetrics
}
```

### KernelFusionEngine

```rust
pub struct KernelFusionEngine {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn analyze_kernels(&self, kernels: Vec<KernelInfo>) -> Result<Vec<FusionOpportunity>>
    pub async fn fuse_kernels(&self, opportunity: &FusionOpportunity) -> Result<FusedKernel>
    pub async fn compile_fused_kernel(&self, kernel: &FusedKernel) -> Result<CompiledKernel>
}
```

## Utilization API

### UtilizationManager

```rust
pub struct UtilizationManager {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self>
    pub async fn start_monitoring(&self) -> Result<()>
    pub async fn get_metrics(&self) -> UtilizationMetrics
    pub async fn get_optimization_strategy(&self) -> OptimizationStrategy
    pub async fn apply_optimization(&self, strategy: OptimizationStrategy) -> Result<()>
}
```

### UtilizationController

```rust
pub struct UtilizationController {
    pub async fn new(device: Arc<CudaDevice>, config: ControllerConfig) -> Result<Self>
    pub async fn start(&self) -> Result<()>
    pub async fn get_recommendations(&self) -> OptimizationRecommendations
    pub async fn generate_report(&self) -> String
}
```

### WorkloadBalancer

```rust
pub struct WorkloadBalancer {
    pub fn new(device: Arc<CudaDevice>, config: WorkloadConfig) -> Self
    pub async fn submit_work(&self, agent_count: u32, iterations: u32, priority: u8) -> Result<u64>
    pub async fn get_next_batch(&self) -> Result<Vec<WorkItem>>
    pub fn adjust_batch_size(&self, utilization: f32)
    pub async fn balance_across_streams(&self, num_streams: usize) -> Result<Vec<Vec<WorkItem>>>
}
```

## Error Handling

All GPU agent APIs use the `Result<T>` type with the following error variants:

```rust
pub enum GpuAgentError {
    CudaError(String),
    AllocationError(String),
    KernelError(String),
    ConfigError(String),
    NotInitialized,
    InvalidParameter(String),
}
```

## Usage Examples

### Basic GPU Swarm

```rust
use gpu_agents::{GpuSwarm, GpuSwarmConfig};

let config = GpuSwarmConfig::default();
let mut swarm = GpuSwarm::new(config)?;
swarm.initialize(10000)?;

// Run evolution
for _ in 0..100 {
    swarm.step()?;
}

let metrics = swarm.metrics();
println!("GPU Utilization: {:.1}%", metrics.gpu_utilization * 100.0);
```

### Knowledge Graph Query

```rust
use gpu_agents::{EnhancedGpuKnowledgeGraph, GraphQuery};

let graph = EnhancedGpuKnowledgeGraph::new(config, device)?;

// Add knowledge
graph.add_node(node).await?;
graph.add_edge(edge).await?;

// Query
let query = GraphQuery {
    query_embedding: embedding,
    max_results: 10,
    min_similarity: 0.8,
};

let results = graph.query(query).await?;
```

### Performance Optimization

```rust
use gpu_agents::{UtilizationController, ControllerConfig};

let config = ControllerConfig {
    target_utilization: 0.90,
    aggressive_mode: true,
    ..Default::default()
};

let controller = UtilizationController::new(device, config).await?;
controller.start().await?;

// Monitor and optimize
let report = controller.generate_report().await;
println!("{}", report);
```