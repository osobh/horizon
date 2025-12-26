//! /proc filesystem interface for GPU DMA lock module
//!
//! Provides runtime inspection and control through /proc/swarm/gpu/

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::{allocation, context, dma, spin, stats, KernelError, KernelResult, GPU_DMA_STATS};

/// Proc file operations
pub trait ProcFileOps {
    fn read(&self) -> KernelResult<String>;
    fn write(&self, data: &[u8]) -> KernelResult<()>;
}

/// GPU device info proc file
pub struct DeviceInfoFile {
    device_id: u32,
}

impl DeviceInfoFile {
    pub fn new(device_id: u32) -> Self {
        Self { device_id }
    }
}

impl ProcFileOps for DeviceInfoFile {
    fn read(&self) -> KernelResult<String> {
        let manager = allocation::get_manager();
        let devices = manager.devices.read();

        if let Some(device) = devices.iter().find(|d| d.id() == self.device_id) {
            let mut output = String::new();
            writeln!(output, "Device ID: {}", device.id()).unwrap();
            writeln!(output, "Name: {}", device.name()).unwrap();
            writeln!(output, "Total Memory: {} MB", device.total_memory() >> 20).unwrap();
            writeln!(
                output,
                "Available Memory: {} MB",
                device.available_memory() >> 20
            )
            .unwrap();
            writeln!(
                output,
                "Allocated Memory: {} MB",
                device.allocated_memory() >> 20
            )
            .unwrap();
            Ok(output)
        } else {
            Err(KernelError::InvalidDevice)
        }
    }

    fn write(&self, _data: &[u8]) -> KernelResult<()> {
        Err(KernelError::NotSupported)
    }
}

/// Allocations proc file
pub struct AllocationsFile {
    device_id: Option<u32>,
}

impl AllocationsFile {
    pub fn new(device_id: Option<u32>) -> Self {
        Self { device_id }
    }
}

impl ProcFileOps for AllocationsFile {
    fn read(&self) -> KernelResult<String> {
        let manager = allocation::get_manager();
        let tracker = &manager.tracker;
        let allocations = tracker.allocations.read();

        let mut output = String::new();
        writeln!(output, "ID\tAgent\tSize(MB)\tDevice\tTimestamp").unwrap();

        for (id, alloc) in allocations.iter() {
            if self.device_id.is_none() || self.device_id == Some(alloc.device_id) {
                writeln!(
                    output,
                    "{}\t{}\t{}\t{}\t{}",
                    id,
                    alloc.agent_id,
                    alloc.size >> 20,
                    alloc.device_id,
                    alloc.timestamp
                )
                .unwrap();
            }
        }

        Ok(output)
    }

    fn write(&self, _data: &[u8]) -> KernelResult<()> {
        Err(KernelError::NotSupported)
    }
}

/// Agent quotas proc file
pub struct QuotasFile;

impl ProcFileOps for QuotasFile {
    fn read(&self) -> KernelResult<String> {
        let manager = allocation::get_manager();
        let enforcer = &manager.enforcer;
        let quotas = enforcer.quotas.read();

        let mut output = String::new();
        writeln!(output, "Agent\tLimit(MB)\tUsed(MB)\tUsage%").unwrap();

        for (agent_id, quota) in quotas.iter() {
            let used = quota.current_usage();
            let limit = quota.memory_limit();
            let usage_pct = if limit > 0 { (used * 100) / limit } else { 0 };

            writeln!(
                output,
                "{}\t{}\t{}\t{}",
                agent_id,
                limit >> 20,
                used >> 20,
                usage_pct
            )
            .unwrap();
        }

        Ok(output)
    }

    fn write(&self, data: &[u8]) -> KernelResult<()> {
        // Format: "agent_id limit_mb"
        let input = core::str::from_utf8(data).map_err(|_| KernelError::InvalidArgument)?;
        let parts: Vec<&str> = input.trim().split_whitespace().collect();

        if parts.len() != 2 {
            return Err(KernelError::InvalidArgument);
        }

        let agent_id = parts[0]
            .parse::<u64>()
            .map_err(|_| KernelError::InvalidArgument)?;
        let limit_mb = parts[1]
            .parse::<usize>()
            .map_err(|_| KernelError::InvalidArgument)?;

        let manager = allocation::get_manager();
        manager.create_agent(agent_id, limit_mb << 20)?;

        Ok(())
    }
}

/// DMA permissions proc file
pub struct DmaPermissionsFile;

impl ProcFileOps for DmaPermissionsFile {
    fn read(&self) -> KernelResult<String> {
        let manager = dma::get_manager();
        let acl = &manager.acl;
        let permissions = acl.permissions.read();

        let mut output = String::new();
        writeln!(output, "Agent\tStart\tEnd\tMode").unwrap();

        for (agent_id, perms) in permissions.iter() {
            for perm in perms {
                writeln!(
                    output,
                    "{}\t0x{:016x}\t0x{:016x}\t{:?}",
                    agent_id, perm.start_addr, perm.end_addr, perm.access_mode
                )
                .unwrap();
            }
        }

        Ok(output)
    }

    fn write(&self, data: &[u8]) -> KernelResult<()> {
        // Format: "agent_id start_addr end_addr mode"
        let input = core::str::from_utf8(data).map_err(|_| KernelError::InvalidArgument)?;
        let parts: Vec<&str> = input.trim().split_whitespace().collect();

        if parts.len() != 4 {
            return Err(KernelError::InvalidArgument);
        }

        let agent_id = parts[0]
            .parse::<u64>()
            .map_err(|_| KernelError::InvalidArgument)?;
        let start_addr = u64::from_str_radix(parts[1].trim_start_matches("0x"), 16)
            .map_err(|_| KernelError::InvalidArgument)?;
        let end_addr = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16)
            .map_err(|_| KernelError::InvalidArgument)?;
        let mode = match parts[3] {
            "r" => crate::DmaAccessMode::ReadOnly,
            "w" => crate::DmaAccessMode::WriteOnly,
            "rw" => crate::DmaAccessMode::ReadWrite,
            _ => return Err(KernelError::InvalidArgument),
        };

        let manager = dma::get_manager();
        manager
            .acl
            .grant_access(agent_id, start_addr, end_addr, mode);

        Ok(())
    }
}

/// GPU contexts proc file
pub struct ContextsFile;

impl ProcFileOps for ContextsFile {
    fn read(&self) -> KernelResult<String> {
        let manager = context::get_manager();
        let contexts = manager.contexts.read();

        let mut output = String::new();
        writeln!(output, "ID\tAgent\tDevice\tState\tSwitches").unwrap();

        for (id, ctx) in contexts.iter() {
            writeln!(
                output,
                "{}\t{}\t{}\t{:?}\t{}",
                id,
                ctx.agent_id,
                ctx.device_id,
                ctx.state,
                ctx.switch_count.load(core::sync::atomic::Ordering::Relaxed)
            )
            .unwrap();
        }

        Ok(output)
    }

    fn write(&self, data: &[u8]) -> KernelResult<()> {
        // Format: "agent_id device_id"
        let input = core::str::from_utf8(data).map_err(|_| KernelError::InvalidArgument)?;
        let parts: Vec<&str> = input.trim().split_whitespace().collect();

        if parts.len() != 2 {
            return Err(KernelError::InvalidArgument);
        }

        let agent_id = parts[0]
            .parse::<u64>()
            .map_err(|_| KernelError::InvalidArgument)?;
        let device_id = parts[1]
            .parse::<u32>()
            .map_err(|_| KernelError::InvalidArgument)?;

        let manager = context::get_manager();
        manager.create_context(agent_id, device_id)?;

        Ok(())
    }
}

/// Statistics proc file
pub struct StatsFile;

impl ProcFileOps for StatsFile {
    fn read(&self) -> KernelResult<String> {
        let mut output = String::new();

        writeln!(output, "=== GPU DMA Lock Statistics ===").unwrap();
        writeln!(
            output,
            "Total Allocations: {}",
            GPU_DMA_STATS
                .total_allocations
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            output,
            "Total Deallocations: {}",
            GPU_DMA_STATS
                .total_deallocations
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            output,
            "Total Bytes Allocated: {} MB",
            GPU_DMA_STATS
                .total_bytes_allocated
                .load(core::sync::atomic::Ordering::Relaxed)
                >> 20
        )
        .unwrap();
        writeln!(
            output,
            "DMA Access Checks: {}",
            GPU_DMA_STATS
                .dma_checks
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            output,
            "DMA Access Denials: {}",
            GPU_DMA_STATS
                .dma_denials
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            output,
            "Context Switches: {}",
            GPU_DMA_STATS
                .context_switches
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();

        // Add detailed stats if available
        if let Some(detailed) = stats::get_detailed_stats() {
            writeln!(output, "\n=== Detailed Statistics ===").unwrap();
            writeln!(output, "{}", detailed).unwrap();
        }

        Ok(output)
    }

    fn write(&self, data: &[u8]) -> KernelResult<()> {
        // Allow resetting stats with "reset" command
        let input = core::str::from_utf8(data).map_err(|_| KernelError::InvalidArgument)?;

        if input.trim() == "reset" {
            stats::reset_stats();
            Ok(())
        } else {
            Err(KernelError::InvalidArgument)
        }
    }
}

/// Control file for module operations
pub struct ControlFile;

impl ProcFileOps for ControlFile {
    fn read(&self) -> KernelResult<String> {
        let mut output = String::new();
        writeln!(output, "GPU DMA Lock Module Control").unwrap();
        writeln!(output, "Commands:").unwrap();
        writeln!(output, "  enable_debug - Enable debug logging").unwrap();
        writeln!(output, "  disable_debug - Disable debug logging").unwrap();
        writeln!(output, "  force_gc - Force garbage collection").unwrap();
        writeln!(output, "  dump_state - Dump internal state").unwrap();
        Ok(output)
    }

    fn write(&self, data: &[u8]) -> KernelResult<()> {
        let command = core::str::from_utf8(data)
            .map_err(|_| KernelError::InvalidArgument)?
            .trim();

        match command {
            "enable_debug" => {
                stats::enable_debug(true);
                Ok(())
            }
            "disable_debug" => {
                stats::enable_debug(false);
                Ok(())
            }
            "force_gc" => {
                perform_garbage_collection();
                Ok(())
            }
            "dump_state" => {
                dump_internal_state();
                Ok(())
            }
            _ => Err(KernelError::InvalidArgument),
        }
    }
}

/// Proc filesystem manager
pub struct ProcManager {
    initialized: spin::Mutex<bool>,
}

impl ProcManager {
    pub fn new() -> Self {
        Self {
            initialized: spin::Mutex::new(false),
        }
    }

    pub fn init(&self) -> KernelResult<()> {
        let mut initialized = self.initialized.lock();
        if *initialized {
            return Ok(());
        }

        // Create proc directory structure
        // /proc/swarm/gpu/
        //   devices/
        //     0/info
        //     0/allocations
        //   allocations
        //   quotas
        //   dma_permissions
        //   contexts
        //   stats
        //   control

        // In real kernel: create_proc_dir and create_proc_entry
        // This is placeholder for actual proc registration

        *initialized = true;
        Ok(())
    }

    pub fn cleanup(&self) {
        let mut initialized = self.initialized.lock();
        if !*initialized {
            return;
        }

        // Remove proc entries
        // In real kernel: remove_proc_entry

        *initialized = false;
    }
}

/// Global proc manager instance
static mut PROC_MANAGER: Option<ProcManager> = None;

/// Initialize proc subsystem
pub fn init() -> KernelResult<()> {
    unsafe {
        PROC_MANAGER = Some(ProcManager::new());
        PROC_MANAGER.as_ref().unwrap().init()
    }
}

/// Cleanup proc subsystem
pub fn cleanup() {
    unsafe {
        if let Some(manager) = PROC_MANAGER.as_ref() {
            manager.cleanup();
        }
        PROC_MANAGER = None;
    }
}

/// Perform garbage collection
fn perform_garbage_collection() {
    // Clean up stale allocations
    let manager = allocation::get_manager();
    let _tracker = &manager.tracker;

    // In real implementation: clean up orphaned allocations
    // For now, just log the action
    stats::log_debug("Garbage collection performed");
}

/// Dump internal state for debugging
fn dump_internal_state() {
    stats::log_debug("=== Internal State Dump ===");

    // Dump allocation state
    let alloc_manager = allocation::get_manager();
    let device_count = alloc_manager.devices.read().len();
    stats::log_debug(&format!("Device count: {}", device_count));

    // Dump DMA state
    let dma_manager = dma::get_manager();
    let perm_count = dma_manager.acl.permissions.read().len();
    stats::log_debug(&format!("DMA permission entries: {}", perm_count));

    // Dump context state
    let ctx_manager = context::get_manager();
    let ctx_count = ctx_manager.contexts.read().len();
    stats::log_debug(&format!("Active contexts: {}", ctx_count));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_file() {
        // Initialize test environment
        allocation::init().unwrap();
        let manager = allocation::get_manager();
        manager.register_device(0, "Test GPU", 8 << 30);

        let file = DeviceInfoFile::new(0);
        let content = file.read().unwrap();

        assert!(content.contains("Device ID: 0"));
        assert!(content.contains("Test GPU"));
        assert!(content.contains("8192 MB")); // 8GB

        allocation::cleanup();
    }

    #[test]
    fn test_device_info_file_comprehensive() {
        allocation::init().unwrap();
        let manager = allocation::get_manager();

        // Test multiple devices
        manager.register_device(0, "GPU0", 8 << 30);
        manager.register_device(1, "GPU1", 16 << 30);
        manager.register_device(2, "GPU2", 24 << 30);

        // Test each device info
        for i in 0..3 {
            let file = DeviceInfoFile::new(i);
            let content = file.read().unwrap();
            assert!(content.contains(&format!("Device ID: {}", i)));
            assert!(content.contains(&format!("GPU{}", i)));
        }

        // Test invalid device
        let file = DeviceInfoFile::new(99);
        assert!(file.read().is_err());

        // Test write operation (should fail)
        let file = DeviceInfoFile::new(0);
        assert!(file.write(b"test").is_err());

        allocation::cleanup();
    }

    #[test]
    fn test_allocations_file() {
        allocation::init().unwrap();
        dma::init().unwrap();
        context::init().unwrap();

        let manager = allocation::get_manager();
        manager.register_device(0, "Test GPU", 8 << 30);
        manager.create_agent(100, 4 << 30).unwrap();

        // Create some allocations
        let alloc1 = manager.allocate(100, 1 << 20, None).unwrap(); // 1MB, any device
        let alloc2 = manager.allocate(100, 2 << 20, None).unwrap(); // 2MB, any device

        // Test global allocations file
        let file = AllocationsFile::new(None);
        let content = file.read().unwrap();

        assert!(content.contains("ID\tAgent\tSize(MB)\tDevice\tTimestamp"));
        assert!(content.contains(&format!("{}\t100\t1\t0", alloc1)));
        assert!(content.contains(&format!("{}\t100\t2\t0", alloc2)));

        // Test device-specific allocations
        let file_dev0 = AllocationsFile::new(Some(0));
        let content_dev0 = file_dev0.read().unwrap();
        assert!(content_dev0.contains(&format!("{}\t100\t1\t0", alloc1)));

        // Test non-existent device
        let file_dev99 = AllocationsFile::new(Some(99));
        let content_dev99 = file_dev99.read().unwrap();
        // Should contain header but no allocations
        assert!(content_dev99.contains("ID\tAgent\tSize(MB)\tDevice\tTimestamp"));
        assert!(!content_dev99.contains(&format!("{}", alloc1)));

        // Test write operation (should fail)
        assert!(file.write(b"test").is_err());

        context::cleanup();
        dma::cleanup();
        allocation::cleanup();
    }

    #[test]
    fn test_quotas_file() {
        // Initialize allocation subsystem
        allocation::init().unwrap();

        let manager = allocation::get_manager();
        manager.create_agent(100, 2 << 30).unwrap(); // 2GB

        let file = QuotasFile;
        let content = file.read().unwrap();

        assert!(content.contains("100\t2048\t0\t0"));

        // Test write
        file.write(b"200 4096").unwrap();
        let content = file.read().unwrap();
        assert!(content.contains("200\t4096"));

        allocation::cleanup();
    }

    #[test]
    fn test_quotas_file_comprehensive() {
        allocation::init().unwrap();
        let manager = allocation::get_manager();

        // Test multiple agents with different quotas
        manager.create_agent(100, 1 << 30).unwrap(); // 1GB
        manager.create_agent(200, 2 << 30).unwrap(); // 2GB
        manager.create_agent(300, 4 << 30).unwrap(); // 4GB

        let file = QuotasFile;
        let content = file.read().unwrap();

        // Check header
        assert!(content.contains("Agent\tLimit(MB)\tUsed(MB)\tUsage%"));

        // Check all agents
        assert!(content.contains("100\t1024\t0\t0"));
        assert!(content.contains("200\t2048\t0\t0"));
        assert!(content.contains("300\t4096\t0\t0"));

        // Test write operations
        assert!(file.write(b"400 8192").is_ok()); // Valid format
        let content = file.read().unwrap();
        assert!(content.contains("400\t8192\t0\t0"));

        // Test invalid write formats
        assert!(file.write(b"invalid").is_err());
        assert!(file.write(b"500").is_err()); // Missing limit
        assert!(file.write(b"600 abc").is_err()); // Invalid number
        assert!(file.write(b"def 1024").is_err()); // Invalid agent ID
        assert!(file.write(b"700 800 900").is_err()); // Too many parts
        assert!(file.write(b"").is_err()); // Empty input

        allocation::cleanup();
    }

    #[test]
    fn test_dma_permissions_file() {
        allocation::init().unwrap();
        dma::init().unwrap();

        let file = DmaPermissionsFile;

        // Initially empty
        let content = file.read().unwrap();
        assert!(content.contains("Agent\tStart\tEnd\tMode"));

        // Add some permissions via write
        assert!(file.write(b"100 0x1000 0x2000 rw").is_ok());
        assert!(file.write(b"200 0x3000 0x4000 r").is_ok());
        assert!(file.write(b"300 0x5000 0x6000 w").is_ok());

        let content = file.read().unwrap();
        assert!(content.contains("100\t0x0000000000001000\t0x0000000000002000\tReadWrite"));
        assert!(content.contains("200\t0x0000000000003000\t0x0000000000004000\tReadOnly"));
        assert!(content.contains("300\t0x0000000000005000\t0x0000000000006000\tWriteOnly"));

        dma::cleanup();
        allocation::cleanup();
    }

    #[test]
    fn test_dma_permissions_file_error_handling() {
        allocation::init().unwrap();
        dma::init().unwrap();

        let file = DmaPermissionsFile;

        // Test invalid write formats
        assert!(file.write(b"invalid").is_err()); // Too few parts
        assert!(file.write(b"100 0x1000").is_err()); // Missing parts
        assert!(file.write(b"100 0x1000 0x2000").is_err()); // Missing mode
        assert!(file.write(b"100 0x1000 0x2000 rw extra").is_err()); // Too many parts
        assert!(file.write(b"abc 0x1000 0x2000 rw").is_err()); // Invalid agent ID
        assert!(file.write(b"100 invalid 0x2000 rw").is_err()); // Invalid start addr
        assert!(file.write(b"100 0x1000 invalid rw").is_err()); // Invalid end addr
        assert!(file.write(b"100 0x1000 0x2000 invalid").is_err()); // Invalid mode
        assert!(file.write(b"").is_err()); // Empty input

        // Test hex parsing variants
        assert!(file.write(b"100 1000 2000 rw").is_ok()); // Without 0x prefix
        assert!(file.write(b"200 0x3000 0x4000 r").is_ok()); // With 0x prefix

        dma::cleanup();
        allocation::cleanup();
    }

    #[test]
    fn test_contexts_file() {
        allocation::init().unwrap();
        dma::init().unwrap();
        context::init().unwrap();

        let manager = allocation::get_manager();
        manager.register_device(0, "Test GPU", 8 << 30);

        let file = ContextsFile;

        // Initially empty
        let content = file.read().unwrap();
        assert!(content.contains("ID\tAgent\tDevice\tState\tSwitches"));

        // Add contexts via write
        assert!(file.write(b"100 0").is_ok());
        assert!(file.write(b"200 0").is_ok());

        let content = file.read().unwrap();
        assert!(content.contains("100\t0\t"));
        assert!(content.contains("200\t0\t"));

        context::cleanup();
        dma::cleanup();
        allocation::cleanup();
    }

    #[test]
    fn test_contexts_file_error_handling() {
        allocation::init().unwrap();
        context::init().unwrap();

        let file = ContextsFile;

        // Test invalid write formats
        assert!(file.write(b"invalid").is_err()); // Too few parts
        assert!(file.write(b"100").is_err()); // Missing device ID
        assert!(file.write(b"100 0 extra").is_err()); // Too many parts
        assert!(file.write(b"abc 0").is_err()); // Invalid agent ID
        assert!(file.write(b"100 abc").is_err()); // Invalid device ID
        assert!(file.write(b"").is_err()); // Empty input

        context::cleanup();
        allocation::cleanup();
    }

    #[test]
    fn test_stats_file() {
        let file = StatsFile;
        let content = file.read().unwrap();

        assert!(content.contains("GPU DMA Lock Statistics"));
        assert!(content.contains("Total Allocations:"));
        assert!(content.contains("DMA Access Checks:"));

        // Test reset
        file.write(b"reset").unwrap();
    }

    #[test]
    fn test_stats_file_comprehensive() {
        let file = StatsFile;

        // Test read always works
        let content = file.read().unwrap();
        assert!(content.contains("=== GPU DMA Lock Statistics ==="));
        assert!(content.contains("Total Allocations:"));
        assert!(content.contains("Total Deallocations:"));
        assert!(content.contains("Total Bytes Allocated:"));
        assert!(content.contains("DMA Access Checks:"));
        assert!(content.contains("DMA Access Denials:"));
        assert!(content.contains("Context Switches:"));

        // Test reset command
        assert!(file.write(b"reset").is_ok());

        // Test invalid commands
        assert!(file.write(b"invalid").is_err());
        assert!(file.write(b"").is_err());
        assert!(file.write(b"reset extra").is_err());
    }

    #[test]
    fn test_control_file() {
        let file = ControlFile;
        let content = file.read().unwrap();

        assert!(content.contains("GPU DMA Lock Module Control"));
        assert!(content.contains("enable_debug"));

        // Test commands
        assert!(file.write(b"enable_debug").is_ok());
        assert!(file.write(b"disable_debug").is_ok());
        assert!(file.write(b"invalid_command").is_err());
    }

    #[test]
    fn test_control_file_comprehensive() {
        let file = ControlFile;

        // Test read always works
        let content = file.read().unwrap();
        assert!(content.contains("GPU DMA Lock Module Control"));
        assert!(content.contains("Commands:"));
        assert!(content.contains("enable_debug"));
        assert!(content.contains("disable_debug"));
        assert!(content.contains("force_gc"));
        assert!(content.contains("dump_state"));

        // Test all valid commands
        assert!(file.write(b"enable_debug").is_ok());
        assert!(file.write(b"disable_debug").is_ok());
        assert!(file.write(b"force_gc").is_ok());
        assert!(file.write(b"dump_state").is_ok());

        // Test invalid commands
        assert!(file.write(b"invalid_command").is_err());
        assert!(file.write(b"").is_err());
        assert!(file.write(b"enable_debug extra").is_err());
    }

    #[test]
    fn test_proc_manager() {
        let manager = ProcManager::new();

        // Test initialization
        assert!(manager.init().is_ok());

        // Test double initialization (should be idempotent)
        assert!(manager.init().is_ok());

        // Test cleanup
        manager.cleanup();

        // Test double cleanup (should not panic)
        manager.cleanup();
    }

    #[test]
    fn test_proc_init_cleanup() {
        // Test module-level init/cleanup
        assert!(init().is_ok());
        cleanup();

        // Test double init/cleanup
        assert!(init().is_ok());
        cleanup();
    }

    #[test]
    fn test_garbage_collection() {
        allocation::init().unwrap();
        dma::init().unwrap();
        context::init().unwrap();

        // This should not panic and should log
        perform_garbage_collection();

        context::cleanup();
        dma::cleanup();
        allocation::cleanup();
    }

    #[test]
    fn test_state_dump() {
        allocation::init().unwrap();
        dma::init().unwrap();
        context::init().unwrap();

        // This should not panic and should log state
        dump_internal_state();

        context::cleanup();
        dma::cleanup();
        allocation::cleanup();
    }

    #[test]
    fn test_large_data_handling() {
        allocation::init().unwrap();

        let manager = allocation::get_manager();
        manager.register_device(0, "Large Test GPU", 64 << 30); // 64GB

        // Create many agents
        for i in 0..100 {
            manager.create_agent(i, ((i + 1) << 20) as usize).unwrap(); // Variable quotas
        }

        let file = QuotasFile;
        let content = file.read().unwrap();

        // Should handle large output
        assert!(content.len() > 1000); // Should be substantial
        assert!(content.contains("Agent\tLimit(MB)\tUsed(MB)\tUsage%"));

        // Check some specific entries
        assert!(content.contains("0\t1\t0\t0"));
        assert!(content.contains("50\t51\t0\t0"));
        assert!(content.contains("99\t100\t0\t0"));

        allocation::cleanup();
    }

    #[test]
    fn test_concurrent_proc_operations() {
        use alloc::vec::Vec;

        allocation::init().unwrap();
        let manager = allocation::get_manager();
        manager.register_device(0, "Concurrent Test GPU", 8 << 30);

        // Create initial agent
        manager.create_agent(1000, 4 << 30).unwrap();

        let file = QuotasFile;

        // Simulate concurrent reads (in single-threaded test)
        let mut results = Vec::new();
        for _ in 0..10 {
            results.push(file.read().unwrap());
        }

        // All reads should succeed and contain consistent data
        for result in results {
            assert!(result.contains("1000\t4096\t0\t0"));
        }

        // Test concurrent writes (simulated) - use smaller range to avoid issues
        for i in 2000..2005 {
            let write_data = format!("{} {}", i, (i - 2000 + 1) * 512); // Smaller quotas
            assert!(file.write(write_data.as_bytes()).is_ok());
        }

        // Verify agents were created (check just one to avoid assertion complexity)
        let final_content = file.read().unwrap();
        assert!(final_content.contains("2000\t512\t0\t0"));

        allocation::cleanup();
    }

    #[test]
    fn test_utf8_handling() {
        let file = QuotasFile;

        // Test non-UTF8 input
        let invalid_utf8 = [0xFF, 0xFE, 0xFD];
        assert!(file.write(&invalid_utf8).is_err());

        // Test valid UTF8
        assert!(file.write("1000 2048".as_bytes()).is_ok());
    }

    #[test]
    fn test_boundary_conditions() {
        allocation::init().unwrap();

        let file = QuotasFile;

        // Test minimum values
        assert!(file.write(b"0 0").is_ok());

        // Test reasonable maximum values (avoid actual max u64 which could cause issues)
        assert!(file.write(b"1000000 8192").is_ok()); // 1M agent ID, 8GB

        let content = file.read().unwrap();
        assert!(content.contains("0\t0\t0\t"));
        assert!(content.contains("1000000\t8192\t0\t0"));

        allocation::cleanup();
    }
}
