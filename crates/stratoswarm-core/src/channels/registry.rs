//! Channel registry for managing all system channels.
//!
//! The `ChannelRegistry` is the central hub for all inter-component communication
//! in the stratoswarm system. It provides strongly-typed channels for different
//! message types and supports both point-to-point (mpsc) and broadcast patterns.

use super::messages::{
    CostMessage, EfficiencyMessage, EvolutionMessage, GovernorMessage, GpuCommand,
    KnowledgeMessage, SchedulerMessage, SystemEvent,
};
use super::patterns::Request;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex};

/// Buffer size for GPU channel (smaller for backpressure control).
const GPU_BUFFER_SIZE: usize = 100;

/// Buffer size for high-throughput channels.
const HIGH_THROUGHPUT_BUFFER_SIZE: usize = 10000;

/// Buffer size for standard channels.
const STANDARD_BUFFER_SIZE: usize = 1000;

/// Buffer size for broadcast channels.
const BROADCAST_BUFFER_SIZE: usize = 1000;

/// Central registry for all system channels.
///
/// The registry manages creation and access to all communication channels
/// in the system. It uses different channel types based on communication patterns:
///
/// - Point-to-point (mpsc): For directed messages to specific components
/// - Broadcast: For events that multiple subscribers need to observe
///
/// # Buffer Sizes
///
/// Different channels have different buffer sizes based on their characteristics:
/// - GPU: 100 (backpressure control for GPU operations)
/// - Evolution: 10000 (high-throughput evolutionary algorithms)
/// - Cost/Efficiency/Scheduler/Governor/Knowledge: 1000 (standard throughput)
/// - Events: 1000 (broadcast to all subscribers)
///
/// # Example
///
/// ```rust
/// use stratoswarm_core::channels::registry::ChannelRegistry;
/// use stratoswarm_core::channels::messages::GpuCommand;
///
/// #[tokio::main]
/// async fn main() {
///     let registry = ChannelRegistry::new();
///
///     // Get a sender for GPU commands
///     let gpu_tx = registry.gpu_sender();
///
///     // Get a receiver for GPU commands
///     let mut gpu_rx = registry.subscribe_gpu();
///
///     // Send a command
///     gpu_tx.send(GpuCommand::Synchronize { stream_id: None })
///         .await
///         .unwrap();
///
///     // Receive the command
///     let cmd = gpu_rx.recv().await.unwrap();
/// }
/// ```
#[derive(Clone, Debug)]
pub struct ChannelRegistry {
    // Point-to-point channels with broadcast for multi-consumer support
    gpu_tx: mpsc::Sender<GpuCommand>,
    gpu_rx: broadcast::Sender<GpuCommand>,

    evolution_tx: mpsc::Sender<EvolutionMessage>,
    evolution_rx: broadcast::Sender<EvolutionMessage>,

    // Request/response channels (mpsc only - single consumer pattern)
    cost_tx: mpsc::Sender<Request<CostMessage, CostMessage>>,
    cost_rx: Arc<Mutex<mpsc::Receiver<Request<CostMessage, CostMessage>>>>,

    efficiency_tx: mpsc::Sender<Request<EfficiencyMessage, EfficiencyMessage>>,
    efficiency_rx: Arc<Mutex<mpsc::Receiver<Request<EfficiencyMessage, EfficiencyMessage>>>>,

    scheduler_tx: mpsc::Sender<Request<SchedulerMessage, SchedulerMessage>>,
    scheduler_rx: Arc<Mutex<mpsc::Receiver<Request<SchedulerMessage, SchedulerMessage>>>>,

    governor_tx: mpsc::Sender<Request<GovernorMessage, GovernorMessage>>,
    governor_rx: Arc<Mutex<mpsc::Receiver<Request<GovernorMessage, GovernorMessage>>>>,

    knowledge_tx: mpsc::Sender<Request<KnowledgeMessage, KnowledgeMessage>>,
    knowledge_rx: Arc<Mutex<mpsc::Receiver<Request<KnowledgeMessage, KnowledgeMessage>>>>,

    // Broadcast channels
    events_tx: broadcast::Sender<SystemEvent>,
}

impl ChannelRegistry {
    /// Create a new channel registry with all channels initialized.
    ///
    /// This creates all channels with their configured buffer sizes and spawns
    /// bridge tasks to connect mpsc receivers to broadcast senders for multi-consumer
    /// support where appropriate.
    #[must_use]
    pub fn new() -> Self {
        // GPU channel (with broadcast support)
        let (gpu_tx, gpu_rx_mpsc) = mpsc::channel(GPU_BUFFER_SIZE);
        let (gpu_rx, _) = broadcast::channel(GPU_BUFFER_SIZE);
        Self::spawn_bridge(gpu_rx_mpsc, gpu_rx.clone());

        // Evolution channel (with broadcast support)
        let (evolution_tx, evolution_rx_mpsc) = mpsc::channel(HIGH_THROUGHPUT_BUFFER_SIZE);
        let (evolution_rx, _) = broadcast::channel(HIGH_THROUGHPUT_BUFFER_SIZE);
        Self::spawn_bridge(evolution_rx_mpsc, evolution_rx.clone());

        // Request/response channels (mpsc only - no broadcast)
        let (cost_tx, cost_rx) = mpsc::channel(STANDARD_BUFFER_SIZE);
        let (efficiency_tx, efficiency_rx) = mpsc::channel(STANDARD_BUFFER_SIZE);
        let (scheduler_tx, scheduler_rx) = mpsc::channel(STANDARD_BUFFER_SIZE);
        let (governor_tx, governor_rx) = mpsc::channel(STANDARD_BUFFER_SIZE);
        let (knowledge_tx, knowledge_rx) = mpsc::channel(STANDARD_BUFFER_SIZE);

        // Events broadcast channel (direct broadcast, no bridge needed)
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER_SIZE);

        Self {
            gpu_tx,
            gpu_rx,
            evolution_tx,
            evolution_rx,
            cost_tx,
            cost_rx: Arc::new(Mutex::new(cost_rx)),
            efficiency_tx,
            efficiency_rx: Arc::new(Mutex::new(efficiency_rx)),
            scheduler_tx,
            scheduler_rx: Arc::new(Mutex::new(scheduler_rx)),
            governor_tx,
            governor_rx: Arc::new(Mutex::new(governor_rx)),
            knowledge_tx,
            knowledge_rx: Arc::new(Mutex::new(knowledge_rx)),
            events_tx,
        }
    }

    /// Spawn a bridge task to forward messages from mpsc to broadcast.
    ///
    /// This enables multiple consumers on point-to-point channels by using
    /// a single mpsc receiver that forwards to a broadcast sender.
    fn spawn_bridge<T>(mut mpsc_rx: mpsc::Receiver<T>, broadcast_tx: broadcast::Sender<T>)
    where
        T: Clone + Send + 'static,
    {
        tokio::spawn(async move {
            while let Some(msg) = mpsc_rx.recv().await {
                // Ignore broadcast errors (no subscribers is okay)
                let _ = broadcast_tx.send(msg);
            }
        });
    }

    // GPU channel accessors

    /// Get a sender for GPU commands.
    #[must_use]
    pub fn gpu_sender(&self) -> mpsc::Sender<GpuCommand> {
        self.gpu_tx.clone()
    }

    /// Subscribe to GPU commands.
    #[must_use]
    pub fn subscribe_gpu(&self) -> broadcast::Receiver<GpuCommand> {
        self.gpu_rx.subscribe()
    }

    // Evolution channel accessors

    /// Get a sender for evolution messages.
    #[must_use]
    pub fn evolution_sender(&self) -> mpsc::Sender<EvolutionMessage> {
        self.evolution_tx.clone()
    }

    /// Subscribe to evolution messages.
    #[must_use]
    pub fn subscribe_evolution(&self) -> broadcast::Receiver<EvolutionMessage> {
        self.evolution_rx.subscribe()
    }

    // Cost channel accessors

    /// Get a sender for cost messages (request/response pattern).
    #[must_use]
    pub fn cost_sender(&self) -> mpsc::Sender<Request<CostMessage, CostMessage>> {
        self.cost_tx.clone()
    }

    /// Get shared receiver for cost messages.
    ///
    /// Note: This returns a clone of the Arc<Mutex<>> wrapped receiver.
    /// You must lock it to receive messages. This pattern ensures only
    /// one consumer receives each message.
    #[must_use]
    pub fn subscribe_cost(&self) -> Arc<Mutex<mpsc::Receiver<Request<CostMessage, CostMessage>>>> {
        Arc::clone(&self.cost_rx)
    }

    // Efficiency channel accessors

    /// Get a sender for efficiency messages (request/response pattern).
    #[must_use]
    pub fn efficiency_sender(&self) -> mpsc::Sender<Request<EfficiencyMessage, EfficiencyMessage>> {
        self.efficiency_tx.clone()
    }

    /// Get shared receiver for efficiency messages.
    #[must_use]
    pub fn subscribe_efficiency(
        &self,
    ) -> Arc<Mutex<mpsc::Receiver<Request<EfficiencyMessage, EfficiencyMessage>>>> {
        Arc::clone(&self.efficiency_rx)
    }

    // Scheduler channel accessors

    /// Get a sender for scheduler messages (request/response pattern).
    #[must_use]
    pub fn scheduler_sender(&self) -> mpsc::Sender<Request<SchedulerMessage, SchedulerMessage>> {
        self.scheduler_tx.clone()
    }

    /// Get shared receiver for scheduler messages.
    #[must_use]
    pub fn subscribe_scheduler(
        &self,
    ) -> Arc<Mutex<mpsc::Receiver<Request<SchedulerMessage, SchedulerMessage>>>> {
        Arc::clone(&self.scheduler_rx)
    }

    // Governor channel accessors

    /// Get a sender for governor messages (request/response pattern).
    #[must_use]
    pub fn governor_sender(&self) -> mpsc::Sender<Request<GovernorMessage, GovernorMessage>> {
        self.governor_tx.clone()
    }

    /// Get shared receiver for governor messages.
    #[must_use]
    pub fn subscribe_governor(
        &self,
    ) -> Arc<Mutex<mpsc::Receiver<Request<GovernorMessage, GovernorMessage>>>> {
        Arc::clone(&self.governor_rx)
    }

    // Knowledge channel accessors

    /// Get a sender for knowledge messages (request/response pattern).
    #[must_use]
    pub fn knowledge_sender(&self) -> mpsc::Sender<Request<KnowledgeMessage, KnowledgeMessage>> {
        self.knowledge_tx.clone()
    }

    /// Get shared receiver for knowledge messages.
    #[must_use]
    pub fn subscribe_knowledge(
        &self,
    ) -> Arc<Mutex<mpsc::Receiver<Request<KnowledgeMessage, KnowledgeMessage>>>> {
        Arc::clone(&self.knowledge_rx)
    }

    // Event channel accessors

    /// Get a sender for system events.
    #[must_use]
    pub fn event_sender(&self) -> broadcast::Sender<SystemEvent> {
        self.events_tx.clone()
    }

    /// Subscribe to system events.
    #[must_use]
    pub fn subscribe_events(&self) -> broadcast::Receiver<SystemEvent> {
        self.events_tx.subscribe()
    }
}

impl Default for ChannelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = ChannelRegistry::new();

        // Verify we can get all senders
        let _gpu_tx = registry.gpu_sender();
        let _evo_tx = registry.evolution_sender();
        let _cost_tx = registry.cost_sender();
        let _eff_tx = registry.efficiency_sender();
        let _sched_tx = registry.scheduler_sender();
        let _gov_tx = registry.governor_sender();
        let _know_tx = registry.knowledge_sender();
        let _event_tx = registry.event_sender();

        // Verify we can subscribe to all channels
        let _gpu_rx = registry.subscribe_gpu();
        let _evo_rx = registry.subscribe_evolution();
        let _event_rx = registry.subscribe_events();
    }

    #[tokio::test]
    async fn test_registry_clone() {
        let registry1 = ChannelRegistry::new();
        let registry2 = registry1.clone();

        // Send from one registry
        let tx = registry1.gpu_sender();
        tx.send(GpuCommand::Synchronize { stream_id: None })
            .await
            .unwrap();

        // Receive from the other
        let mut rx = registry2.subscribe_gpu();
        assert!(rx.recv().await.is_ok());
    }
}
