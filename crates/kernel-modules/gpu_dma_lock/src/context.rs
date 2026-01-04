//! GPU context management and isolation
//!
//! This module manages GPU contexts to ensure proper isolation
//! between different agents.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::{spin, KernelError, KernelResult, GPU_DMA_STATS};

/// GPU context information
#[derive(Debug)]
pub struct GpuContext {
    /// Context ID
    pub id: u64,
    /// Agent that owns this context
    pub agent_id: u64,
    /// GPU device ID
    pub device_id: u32,
    /// Context state
    pub state: ContextState,
    /// Creation timestamp
    pub created_at: u64,
    /// Last accessed timestamp
    pub last_accessed: AtomicU64,
    /// Context switch count
    pub switch_count: AtomicU32,
}

impl Clone for GpuContext {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            agent_id: self.agent_id,
            device_id: self.device_id,
            state: self.state,
            created_at: self.created_at,
            last_accessed: AtomicU64::new(self.last_accessed.load(Ordering::Relaxed)),
            switch_count: AtomicU32::new(self.switch_count.load(Ordering::Relaxed)),
        }
    }
}

/// Context state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextState {
    Active,
    Inactive,
    Suspended,
    Destroyed,
}

/// Context manager for GPU isolation
pub struct ContextManager {
    /// Next context ID
    pub next_id: AtomicU64,
    /// All contexts
    pub contexts: spin::RwLock<BTreeMap<u64, GpuContext>>,
    /// Agent to context mapping
    pub agent_contexts: spin::RwLock<BTreeMap<u64, Vec<u64>>>,
    /// Active context per GPU
    pub active_contexts: spin::RwLock<BTreeMap<u32, u64>>,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            contexts: spin::RwLock::new(BTreeMap::new()),
            agent_contexts: spin::RwLock::new(BTreeMap::new()),
            active_contexts: spin::RwLock::new(BTreeMap::new()),
        }
    }

    /// Create a new GPU context for an agent
    pub fn create_context(&self, agent_id: u64, device_id: u32) -> KernelResult<u64> {
        let context_id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let context = GpuContext {
            id: context_id,
            agent_id,
            device_id,
            state: ContextState::Inactive,
            created_at: get_current_time(),
            last_accessed: AtomicU64::new(get_current_time()),
            switch_count: AtomicU32::new(0),
        };

        // Add context
        self.contexts.write().insert(context_id, context);

        // Map to agent
        self.agent_contexts
            .write()
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(context_id);

        Ok(context_id)
    }

    /// Destroy a context
    pub fn destroy_context(&self, context_id: u64) -> KernelResult<()> {
        let mut contexts = self.contexts.write();
        let context = contexts
            .get_mut(&context_id)
            .ok_or(KernelError::InvalidArgument)?;

        context.state = ContextState::Destroyed;

        // Remove from active contexts if present
        let mut active = self.active_contexts.write();
        active.retain(|_, &mut cid| cid != context_id);

        Ok(())
    }

    /// Switch to a context on a GPU
    pub fn switch_context(&self, device_id: u32, context_id: u64) -> KernelResult<()> {
        // Verify context exists and is valid
        {
            let contexts = self.contexts.read();
            let context = contexts
                .get(&context_id)
                .ok_or(KernelError::InvalidArgument)?;

            if context.device_id != device_id {
                return Err(KernelError::InvalidDevice);
            }

            if context.state == ContextState::Destroyed {
                return Err(KernelError::ContextError);
            }
        }

        // Get current active context
        let prev_context = self.active_contexts.read().get(&device_id).copied();

        // Update active context
        self.active_contexts.write().insert(device_id, context_id);

        // Update states
        if let Some(prev_id) = prev_context {
            if let Some(prev_ctx) = self.contexts.write().get_mut(&prev_id) {
                prev_ctx.state = ContextState::Inactive;
            }
        }

        if let Some(new_ctx) = self.contexts.write().get_mut(&context_id) {
            new_ctx.state = ContextState::Active;
            new_ctx
                .last_accessed
                .store(get_current_time(), Ordering::Relaxed);
            new_ctx.switch_count.fetch_add(1, Ordering::Relaxed);
        }

        // Record context switch
        GPU_DMA_STATS.record_context_switch();

        Ok(())
    }

    /// Get active context for a GPU
    pub fn get_active_context(&self, device_id: u32) -> Option<u64> {
        self.active_contexts.read().get(&device_id).copied()
    }

    /// Get all contexts for an agent
    pub fn get_agent_contexts(&self, agent_id: u64) -> Vec<GpuContext> {
        let agent_contexts = self.agent_contexts.read();
        if let Some(context_ids) = agent_contexts.get(&agent_id) {
            let contexts = self.contexts.read();
            context_ids
                .iter()
                .filter_map(|id| contexts.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Check if two contexts are properly isolated
    pub fn check_isolation(&self, ctx1: u64, ctx2: u64) -> bool {
        if ctx1 == ctx2 {
            return false;
        }

        let contexts = self.contexts.read();
        let c1 = contexts.get(&ctx1);
        let c2 = contexts.get(&ctx2);

        match (c1, c2) {
            (Some(context1), Some(context2)) => {
                // Contexts from different agents are always isolated
                context1.agent_id != context2.agent_id
            }
            _ => true, // If either doesn't exist, consider them isolated
        }
    }

    /// Remove all contexts for an agent
    pub fn remove_agent_contexts(&self, agent_id: u64) -> KernelResult<()> {
        let context_ids: Vec<u64> = {
            let mut agent_contexts = self.agent_contexts.write();
            agent_contexts.remove(&agent_id).unwrap_or_default()
        };

        for context_id in context_ids {
            self.destroy_context(context_id)?;
        }

        Ok(())
    }
}

/// Context isolation verifier
pub struct IsolationVerifier {
    /// Isolation rules
    rules: spin::RwLock<Vec<IsolationRule>>,
}

#[derive(Debug, Clone)]
struct IsolationRule {
    name: &'static str,
    check: fn(&GpuContext, &GpuContext) -> bool,
}

impl IsolationVerifier {
    pub fn new() -> Self {
        let verifier = Self {
            rules: spin::RwLock::new(Vec::new()),
        };

        // Add default rules
        verifier.add_rule("different_agents", |c1, c2| c1.agent_id != c2.agent_id);
        verifier.add_rule("not_destroyed", |c1, c2| {
            c1.state != ContextState::Destroyed && c2.state != ContextState::Destroyed
        });

        verifier
    }

    pub fn add_rule(&self, name: &'static str, check: fn(&GpuContext, &GpuContext) -> bool) {
        self.rules.write().push(IsolationRule { name, check });
    }

    pub fn verify_isolation(&self, ctx1: &GpuContext, ctx2: &GpuContext) -> IsolationResult {
        let rules = self.rules.read();
        let mut violations = Vec::new();

        for rule in rules.iter() {
            if !(rule.check)(ctx1, ctx2) {
                violations.push(rule.name);
            }
        }

        if violations.is_empty() {
            IsolationResult::Isolated
        } else {
            IsolationResult::Violated(violations)
        }
    }
}

#[derive(Debug)]
pub enum IsolationResult {
    Isolated,
    Violated(Vec<&'static str>),
}

/// Context scheduler for fair GPU access
pub struct ContextScheduler {
    /// Time slice per context in microseconds
    time_slice_us: AtomicU64,
    /// Scheduling queue per GPU
    queues: spin::RwLock<BTreeMap<u32, Vec<u64>>>,
    /// Last scheduled time per context
    last_scheduled: spin::RwLock<BTreeMap<u64, u64>>,
}

impl ContextScheduler {
    pub fn new() -> Self {
        Self {
            time_slice_us: AtomicU64::new(10_000), // 10ms default
            queues: spin::RwLock::new(BTreeMap::new()),
            last_scheduled: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn set_time_slice(&self, us: u64) {
        self.time_slice_us.store(us, Ordering::Relaxed);
    }

    pub fn enqueue_context(&self, device_id: u32, context_id: u64) {
        let mut queues = self.queues.write();
        queues
            .entry(device_id)
            .or_insert_with(Vec::new)
            .push(context_id);
    }

    pub fn get_next_context(&self, device_id: u32) -> Option<u64> {
        let mut queues = self.queues.write();
        if let Some(queue) = queues.get_mut(&device_id) {
            if !queue.is_empty() {
                let context_id = queue.remove(0);

                // Record scheduling time
                self.last_scheduled
                    .write()
                    .insert(context_id, get_current_time());

                // Re-enqueue for round-robin
                queue.push(context_id);

                return Some(context_id);
            }
        }
        None
    }

    pub fn should_preempt(&self, context_id: u64) -> bool {
        let last_scheduled = self.last_scheduled.read();
        if let Some(&last_time) = last_scheduled.get(&context_id) {
            let elapsed = get_current_time() - last_time;
            elapsed >= self.time_slice_us.load(Ordering::Relaxed)
        } else {
            false
        }
    }
}

/// Global context manager instance
static mut CONTEXT_MANAGER: Option<ContextManager> = None;

/// Get global context manager
pub fn get_manager() -> &'static ContextManager {
    // SAFETY: This function is only called after init() has been called during
    // kernel module initialization. The static CONTEXT_MANAGER is initialized
    // once at module load time and never modified until module cleanup, at which
    // point no more calls to this function should occur. Single-threaded init
    // ensures no data races during the initialization sequence.
    unsafe {
        CONTEXT_MANAGER
            .as_ref()
            .expect("Context manager not initialized")
    }
}

/// Initialize context subsystem
pub fn init() -> KernelResult<()> {
    // SAFETY: This function is called exactly once during kernel module
    // initialization, before any other threads can access CONTEXT_MANAGER.
    // The kernel module init sequence is single-threaded, ensuring no data
    // races during this write to the static mutable.
    unsafe {
        CONTEXT_MANAGER = Some(ContextManager::new());
    }
    Ok(())
}

/// Cleanup context subsystem
pub fn cleanup() {
    // SAFETY: This function is called exactly once during kernel module
    // unload, after all other operations have completed and no threads are
    // accessing CONTEXT_MANAGER. The kernel module exit sequence ensures
    // exclusive access to module globals during cleanup.
    unsafe {
        CONTEXT_MANAGER = None;
    }
}

/// Get current time in microseconds (mock)
fn get_current_time() -> u64 {
    // In real kernel: ktime_get() / 1000
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let manager = ContextManager::new();

        let ctx1 = manager.create_context(100, 0).unwrap();
        let ctx2 = manager.create_context(101, 0).unwrap();

        assert_ne!(ctx1, ctx2);

        let agent_contexts = manager.get_agent_contexts(100);
        assert_eq!(agent_contexts.len(), 1);
        assert_eq!(agent_contexts[0].id, ctx1);
    }

    #[test]
    fn test_context_switching() {
        let manager = ContextManager::new();

        let ctx1 = manager.create_context(200, 0).unwrap();
        let ctx2 = manager.create_context(201, 0).unwrap();

        // Switch to ctx1
        manager.switch_context(0, ctx1).unwrap();
        assert_eq!(manager.get_active_context(0), Some(ctx1));

        // Switch to ctx2
        manager.switch_context(0, ctx2).unwrap();
        assert_eq!(manager.get_active_context(0), Some(ctx2));

        // Check states
        let contexts = manager.contexts.read();
        assert_eq!(contexts.get(&ctx1).unwrap().state, ContextState::Inactive);
        assert_eq!(contexts.get(&ctx2).unwrap().state, ContextState::Active);
    }

    #[test]
    fn test_isolation_verification() {
        let verifier = IsolationVerifier::new();

        let ctx1 = GpuContext {
            id: 1,
            agent_id: 300,
            device_id: 0,
            state: ContextState::Active,
            created_at: 0,
            last_accessed: AtomicU64::new(0),
            switch_count: AtomicU32::new(0),
        };

        let ctx2 = GpuContext {
            id: 2,
            agent_id: 301,
            device_id: 0,
            state: ContextState::Active,
            created_at: 0,
            last_accessed: AtomicU64::new(0),
            switch_count: AtomicU32::new(0),
        };

        match verifier.verify_isolation(&ctx1, &ctx2) {
            IsolationResult::Isolated => (),
            IsolationResult::Violated(_) => panic!("Contexts should be isolated"),
        }
    }

    #[test]
    fn test_context_scheduler() {
        let scheduler = ContextScheduler::new();
        scheduler.set_time_slice(5000); // 5ms

        scheduler.enqueue_context(0, 1);
        scheduler.enqueue_context(0, 2);
        scheduler.enqueue_context(0, 3);

        // Round-robin scheduling
        assert_eq!(scheduler.get_next_context(0), Some(1));
        assert_eq!(scheduler.get_next_context(0), Some(2));
        assert_eq!(scheduler.get_next_context(0), Some(3));
        assert_eq!(scheduler.get_next_context(0), Some(1)); // Back to first
    }
}
