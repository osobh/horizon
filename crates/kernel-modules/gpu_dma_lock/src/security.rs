//! Security and isolation enforcement for GPU DMA lock
//!
//! This module implements security policies and isolation verification
//! to ensure agents cannot access each other's GPU memory.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::{context, spin, KernelError, KernelResult};

/// Security policy configuration
pub struct SecurityPolicy {
    /// Enable strict isolation checks
    pub strict_isolation: bool,
    /// Enable memory scrubbing on deallocation
    pub memory_scrubbing: bool,
    /// Enable access logging
    pub access_logging: bool,
    /// Maximum allocation size per request
    pub max_allocation_size: usize,
    /// Minimum time between allocations (rate limiting)
    pub allocation_rate_limit_us: u64,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            strict_isolation: true,
            memory_scrubbing: true,
            access_logging: false,
            max_allocation_size: 1 << 30,   // 1GB
            allocation_rate_limit_us: 1000, // 1ms
        }
    }
}

/// Security violation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    UnauthorizedAccess,
    QuotaViolation,
    IsolationBreach,
    RateLimitExceeded,
    InvalidOperation,
    SuspiciousPattern,
}

/// Security event
#[derive(Debug, Clone)]
pub struct SecurityEvent {
    pub timestamp: u64,
    pub agent_id: u64,
    pub violation_type: ViolationType,
    pub details: [u64; 4],
}

/// Security monitor
pub struct SecurityMonitor {
    /// Current security policy
    policy: spin::RwLock<SecurityPolicy>,
    /// Security events log
    events: spin::RwLock<Vec<SecurityEvent>>,
    /// Agent trust scores
    trust_scores: spin::RwLock<BTreeMap<u64, u32>>,
    /// Last allocation time per agent
    last_allocation: spin::RwLock<BTreeMap<u64, u64>>,
    /// Quarantined agents
    quarantined: spin::RwLock<BTreeMap<u64, u64>>, // agent_id -> quarantine_until
}

impl SecurityMonitor {
    pub fn new() -> Self {
        Self {
            policy: spin::RwLock::new(SecurityPolicy::default()),
            events: spin::RwLock::new(Vec::new()),
            trust_scores: spin::RwLock::new(BTreeMap::new()),
            last_allocation: spin::RwLock::new(BTreeMap::new()),
            quarantined: spin::RwLock::new(BTreeMap::new()),
        }
    }

    /// Check if allocation is allowed
    pub fn check_allocation(&self, agent_id: u64, size: usize) -> KernelResult<()> {
        // Check if agent is quarantined
        if self.is_quarantined(agent_id) {
            self.log_violation(agent_id, ViolationType::UnauthorizedAccess, [0, 0, 0, 0]);
            return Err(KernelError::DmaAccessDenied);
        }

        let policy = self.policy.read();

        // Check allocation size
        if size > policy.max_allocation_size {
            self.log_violation(
                agent_id,
                ViolationType::QuotaViolation,
                [size as u64, policy.max_allocation_size as u64, 0, 0],
            );
            return Err(KernelError::QuotaExceeded);
        }

        // Check rate limiting
        if policy.allocation_rate_limit_us > 0 {
            let now = get_current_time();
            let mut last_alloc = self.last_allocation.write();

            if let Some(&last_time) = last_alloc.get(&agent_id) {
                if now - last_time < policy.allocation_rate_limit_us {
                    self.log_violation(
                        agent_id,
                        ViolationType::RateLimitExceeded,
                        [now, last_time, policy.allocation_rate_limit_us, 0],
                    );
                    return Err(KernelError::DmaAccessDenied);
                }
            }

            last_alloc.insert(agent_id, now);
        }

        Ok(())
    }

    /// Check DMA access permission
    pub fn check_dma_access(
        &self,
        agent_id: u64,
        addr: u64,
        mode: crate::DmaAccessMode,
    ) -> KernelResult<()> {
        // Check if agent is quarantined
        if self.is_quarantined(agent_id) {
            self.log_violation(
                agent_id,
                ViolationType::UnauthorizedAccess,
                [addr, mode as u64, 0, 0],
            );
            return Err(KernelError::DmaAccessDenied);
        }

        let policy = self.policy.read();

        // Log access if enabled
        if policy.access_logging {
            self.log_access(agent_id, addr, mode);
        }

        Ok(())
    }

    /// Verify context isolation
    pub fn verify_isolation(&self, ctx1: u64, ctx2: u64) -> KernelResult<()> {
        let ctx_manager = context::get_manager();

        if !ctx_manager.check_isolation(ctx1, ctx2) {
            // Get context details for logging
            let contexts = ctx_manager.contexts.read();
            let agent1 = contexts.get(&ctx1).map(|c| c.agent_id).unwrap_or(0);
            let agent2 = contexts.get(&ctx2).map(|c| c.agent_id).unwrap_or(0);

            self.log_violation(
                agent1,
                ViolationType::IsolationBreach,
                [ctx1, ctx2, agent1, agent2],
            );

            return Err(KernelError::ContextError);
        }

        Ok(())
    }

    /// Scrub memory on deallocation
    pub fn scrub_memory(&self, gpu_addr: u64, size: usize) {
        let policy = self.policy.read();

        if policy.memory_scrubbing {
            // In real implementation: zero out GPU memory
            // For now, just record the action
            crate::stats::log_debug(&format!(
                "Scrubbing {} bytes at GPU address 0x{:x}",
                size, gpu_addr
            ));
        }
    }

    /// Update agent trust score
    pub fn update_trust_score(&self, agent_id: u64, delta: i32) {
        let mut scores = self.trust_scores.write();
        let score = scores.entry(agent_id).or_insert(100);

        if delta > 0 {
            *score = score.saturating_add(delta as u32);
        } else {
            *score = score.saturating_sub((-delta) as u32);
        }

        // Quarantine if trust score too low
        if *score < 20 {
            self.quarantine_agent(agent_id, 3600_000_000); // 1 hour
        }
    }

    /// Quarantine an agent
    pub fn quarantine_agent(&self, agent_id: u64, duration_us: u64) {
        let until = get_current_time() + duration_us;
        self.quarantined.write().insert(agent_id, until);

        // Log the quarantine event directly without updating trust score again
        let event = SecurityEvent {
            timestamp: get_current_time(),
            agent_id,
            violation_type: ViolationType::SuspiciousPattern,
            details: [duration_us, 0, 0, 0],
        };

        self.events.write().push(event.clone());
        crate::stats::log_security_event(&event);
    }

    /// Check if agent is quarantined
    pub fn is_quarantined(&self, agent_id: u64) -> bool {
        let mut quarantined = self.quarantined.write();
        let now = get_current_time();

        // Clean up expired quarantines
        quarantined.retain(|_, &mut until| until > now);

        quarantined.contains_key(&agent_id)
    }

    /// Log security violation
    fn log_violation(&self, agent_id: u64, violation_type: ViolationType, details: [u64; 4]) {
        let event = SecurityEvent {
            timestamp: get_current_time(),
            agent_id,
            violation_type,
            details,
        };

        self.events.write().push(event.clone());

        // Update trust score
        match violation_type {
            ViolationType::UnauthorizedAccess => self.update_trust_score(agent_id, -20),
            ViolationType::QuotaViolation => self.update_trust_score(agent_id, -5),
            ViolationType::IsolationBreach => self.update_trust_score(agent_id, -30),
            ViolationType::RateLimitExceeded => self.update_trust_score(agent_id, -2),
            ViolationType::InvalidOperation => self.update_trust_score(agent_id, -10),
            ViolationType::SuspiciousPattern => self.update_trust_score(agent_id, -50),
        }

        // Log to kernel
        crate::stats::log_security_event(&event);
    }

    /// Log access for auditing
    fn log_access(&self, agent_id: u64, addr: u64, mode: crate::DmaAccessMode) {
        crate::stats::log_debug(&format!(
            "DMA access: agent={} addr=0x{:x} mode={:?}",
            agent_id, addr, mode
        ));
    }

    /// Get security events for an agent
    pub fn get_agent_events(&self, agent_id: u64) -> Vec<SecurityEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.agent_id == agent_id)
            .cloned()
            .collect()
    }

    /// Clear old security events
    pub fn cleanup_events(&self, older_than: u64) {
        let cutoff = get_current_time().saturating_sub(older_than);
        self.events.write().retain(|e| e.timestamp > cutoff);
    }
}

/// Memory isolation verifier
pub struct MemoryIsolationVerifier {
    /// Allocation boundaries by agent
    boundaries: spin::RwLock<BTreeMap<u64, Vec<MemoryBoundary>>>,
}

#[derive(Debug, Clone)]
struct MemoryBoundary {
    start: u64,
    end: u64,
    device_id: u32,
}

impl MemoryIsolationVerifier {
    pub fn new() -> Self {
        Self {
            boundaries: spin::RwLock::new(BTreeMap::new()),
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&self, agent_id: u64, start: u64, size: usize, device_id: u32) {
        let boundary = MemoryBoundary {
            start,
            end: start + size as u64,
            device_id,
        };

        self.boundaries
            .write()
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(boundary);
    }

    /// Check if memory regions overlap
    pub fn check_overlap(&self, agent_id: u64, start: u64, size: usize, device_id: u32) -> bool {
        let end = start + size as u64;
        let boundaries = self.boundaries.read();

        for (other_agent, regions) in boundaries.iter() {
            if *other_agent == agent_id {
                continue;
            }

            for region in regions {
                if region.device_id != device_id {
                    continue;
                }

                // Check for overlap
                if start < region.end && end > region.start {
                    return true;
                }
            }
        }

        false
    }

    /// Remove allocation record
    pub fn remove_allocation(&self, agent_id: u64, start: u64) {
        let mut boundaries = self.boundaries.write();
        if let Some(regions) = boundaries.get_mut(&agent_id) {
            regions.retain(|b| b.start != start);
        }
    }
}

/// Access pattern analyzer
pub struct AccessPatternAnalyzer {
    /// Access patterns by agent
    patterns: spin::RwLock<BTreeMap<u64, AccessPattern>>,
    /// Anomaly detection enabled
    anomaly_detection: AtomicBool,
}

#[derive(Debug, Clone)]
struct AccessPattern {
    allocation_count: u64,
    total_allocated: u64,
    avg_allocation_size: u64,
    access_frequency: u64,
    last_access: u64,
}

impl AccessPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: spin::RwLock::new(BTreeMap::new()),
            anomaly_detection: AtomicBool::new(true),
        }
    }

    /// Record allocation pattern
    pub fn record_allocation(&self, agent_id: u64, size: usize) {
        let mut patterns = self.patterns.write();
        let pattern = patterns.entry(agent_id).or_insert(AccessPattern {
            allocation_count: 0,
            total_allocated: 0,
            avg_allocation_size: 0,
            access_frequency: 0,
            last_access: 0,
        });

        pattern.allocation_count += 1;
        pattern.total_allocated += size as u64;
        pattern.avg_allocation_size = pattern.total_allocated / pattern.allocation_count;
        pattern.last_access = get_current_time();
    }

    /// Detect anomalous patterns
    pub fn detect_anomaly(&self, agent_id: u64, size: usize) -> bool {
        if !self.anomaly_detection.load(Ordering::Relaxed) {
            return false;
        }

        let patterns = self.patterns.read();
        if let Some(pattern) = patterns.get(&agent_id) {
            // Check for sudden large allocations
            if size as u64 > pattern.avg_allocation_size * 10 {
                return true;
            }

            // Check for rapid allocations
            let now = get_current_time();
            if pattern.last_access > 0 && now - pattern.last_access < 100 {
                return true;
            }
        }

        false
    }

    /// Enable/disable anomaly detection
    pub fn set_anomaly_detection(&self, enabled: bool) {
        self.anomaly_detection.store(enabled, Ordering::Relaxed);
    }
}

/// Global security manager
static mut SECURITY_MANAGER: Option<SecurityManager> = None;

/// Combined security management
pub struct SecurityManager {
    pub monitor: SecurityMonitor,
    pub isolation_verifier: MemoryIsolationVerifier,
    pub pattern_analyzer: AccessPatternAnalyzer,
}

impl SecurityManager {
    pub fn new() -> Self {
        Self {
            monitor: SecurityMonitor::new(),
            isolation_verifier: MemoryIsolationVerifier::new(),
            pattern_analyzer: AccessPatternAnalyzer::new(),
        }
    }
}

/// Get global security manager
pub fn get_manager() -> &'static SecurityManager {
    // SAFETY: This function is only called after init() has been called during
    // kernel module initialization. The static SECURITY_MANAGER is initialized
    // once at module load time and never modified until module cleanup, at which
    // point no more calls to this function should occur. Single-threaded init
    // ensures no data races during the initialization sequence.
    unsafe {
        SECURITY_MANAGER
            .as_ref()
            .expect("Security manager not initialized")
    }
}

/// Initialize security subsystem
pub fn init() -> KernelResult<()> {
    // SAFETY: This function is called exactly once during kernel module
    // initialization, before any other threads can access SECURITY_MANAGER.
    // The kernel module init sequence is single-threaded, ensuring no data
    // races during this write to the static mutable.
    unsafe {
        SECURITY_MANAGER = Some(SecurityManager::new());
    }
    Ok(())
}

/// Cleanup security subsystem
pub fn cleanup() {
    // SAFETY: This function is called exactly once during kernel module
    // unload, after all other operations have completed and no threads are
    // accessing SECURITY_MANAGER. The kernel module exit sequence ensures
    // exclusive access to module globals during cleanup.
    unsafe {
        SECURITY_MANAGER = None;
    }
}

/// Get current time (mock)
fn get_current_time() -> u64 {
    // In real kernel: ktime_get() / 1000
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_monitor() {
        let monitor = SecurityMonitor::new();

        // Test allocation check
        assert!(monitor.check_allocation(100, 1 << 20).is_ok()); // 1MB
        assert!(monitor.check_allocation(100, 2 << 30).is_err()); // 2GB exceeds limit

        // Test quarantine
        monitor.quarantine_agent(200, 1000);
        assert!(monitor.is_quarantined(200));
        assert!(monitor.check_allocation(200, 1 << 20).is_err());
    }

    #[test]
    fn test_memory_isolation() {
        let verifier = MemoryIsolationVerifier::new();

        // Record allocations
        verifier.record_allocation(300, 0x1000, 0x1000, 0);
        verifier.record_allocation(301, 0x3000, 0x1000, 0);

        // Check non-overlapping
        assert!(!verifier.check_overlap(302, 0x5000, 0x1000, 0));

        // Check overlapping
        assert!(verifier.check_overlap(302, 0x1500, 0x1000, 0));
    }

    #[test]
    fn test_pattern_analyzer() {
        let analyzer = AccessPatternAnalyzer::new();

        // Record normal pattern
        analyzer.record_allocation(400, 1 << 20); // 1MB
        analyzer.record_allocation(400, 2 << 20); // 2MB
        analyzer.record_allocation(400, 1 << 20); // 1MB

        // Check anomaly detection
        assert!(!analyzer.detect_anomaly(400, 3 << 20)); // 3MB is normal
        assert!(analyzer.detect_anomaly(400, 50 << 20)); // 50MB is anomalous
    }

    #[test]
    fn test_trust_scores() {
        let monitor = SecurityMonitor::new();

        // Initial trust score
        monitor.update_trust_score(500, 0);

        // Decrease trust
        monitor.update_trust_score(500, -30);
        monitor.update_trust_score(500, -30);
        monitor.update_trust_score(500, -30);

        // Should be quarantined when trust < 20
        assert!(monitor.is_quarantined(500));
    }
}
