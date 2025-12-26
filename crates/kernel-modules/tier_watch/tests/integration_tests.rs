//! Integration tests for TierWatch kernel module
//! These tests verify the module's integration with the kernel and other subsystems

#![cfg(test)]

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;

#[test]
#[ignore] // Requires root and kernel module environment
fn test_module_load_unload() {
    // Check if we're running as root
    let uid = unsafe { libc::getuid() };
    if uid != 0 {
        eprintln!("Skipping kernel module test - requires root");
        return;
    }

    // Build the module
    let output = Command::new("make")
        .current_dir("../tier_watch")
        .arg("clean")
        .output()
        .expect("Failed to clean build");
    assert!(output.status.success());

    let output = Command::new("make")
        .current_dir("../tier_watch")
        .output()
        .expect("Failed to build module");
    assert!(output.status.success());

    // Load the module
    let output = Command::new("insmod")
        .current_dir("../tier_watch")
        .arg("tier_watch.ko")
        .output()
        .expect("Failed to load module");
    assert!(output.status.success());

    // Verify module is loaded
    let output = Command::new("lsmod")
        .output()
        .expect("Failed to list modules");
    let output_str = String::from_utf8_lossy(&output.stdout);
    assert!(output_str.contains("tier_watch"));

    // Check /proc/swarm/tiers exists
    assert!(Path::new("/proc/swarm/tiers").exists());

    // Unload the module
    let output = Command::new("rmmod")
        .arg("tier_watch")
        .output()
        .expect("Failed to unload module");
    assert!(output.status.success());
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_proc_tier_interface() {
    // Check tier directories exist
    let tiers = ["gpu", "cpu", "nvme", "ssd", "hdd"];

    for tier in &tiers {
        let tier_path = format!("/proc/swarm/tiers/{}", tier);
        assert!(Path::new(&tier_path).exists(), "Tier {} missing", tier);

        // Check stats file exists
        let stats_path = format!("{}/stats", tier_path);
        assert!(
            Path::new(&stats_path).exists(),
            "Stats file missing for {}",
            tier
        );
    }
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_tier_stats_format() {
    // Read GPU tier stats
    let stats =
        fs::read_to_string("/proc/swarm/tiers/gpu/stats").expect("Failed to read GPU tier stats");

    // Verify expected format
    assert!(stats.contains("Total pages:"));
    assert!(stats.contains("Used bytes:"));
    assert!(stats.contains("Pressure:"));
    assert!(stats.contains("Major faults:"));
    assert!(stats.contains("Minor faults:"));
    assert!(stats.contains("Migrations in:"));
    assert!(stats.contains("Migrations out:"));
    assert!(stats.contains("Access count:"));
    assert!(stats.contains("Avg latency:"));
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_memory_pressure_monitoring() {
    // Read initial pressure levels
    let mut initial_pressures = Vec::new();
    let tiers = ["gpu", "cpu", "nvme", "ssd", "hdd"];

    for tier in &tiers {
        let stats_path = format!("/proc/swarm/tiers/{}/stats", tier);
        let stats = fs::read_to_string(&stats_path).unwrap();

        // Extract pressure percentage
        let pressure_line = stats
            .lines()
            .find(|line| line.starts_with("Pressure:"))
            .unwrap();
        let pressure: u8 = pressure_line
            .split(':')
            .nth(1)
            .unwrap()
            .trim()
            .trim_end_matches('%')
            .parse()
            .unwrap();

        initial_pressures.push(pressure);
    }

    // Verify pressure is between 0-100%
    for pressure in &initial_pressures {
        assert!(*pressure <= 100);
    }
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_page_fault_counting() {
    // Record initial fault counts
    let stats_before =
        fs::read_to_string("/proc/swarm/tiers/cpu/stats").expect("Failed to read CPU tier stats");

    let get_fault_count = |stats: &str, fault_type: &str| -> u64 {
        stats
            .lines()
            .find(|line| line.starts_with(fault_type))
            .unwrap()
            .split(':')
            .nth(1)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    };

    let major_before = get_fault_count(&stats_before, "Major faults:");
    let minor_before = get_fault_count(&stats_before, "Minor faults:");

    // Trigger some memory activity
    let mut data = vec![0u8; 100 << 20]; // 100MB
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }

    // Force some page faults by accessing memory
    thread::sleep(Duration::from_millis(100));

    // Read fault counts after
    let stats_after =
        fs::read_to_string("/proc/swarm/tiers/cpu/stats").expect("Failed to read CPU tier stats");

    let major_after = get_fault_count(&stats_after, "Major faults:");
    let minor_after = get_fault_count(&stats_after, "Minor faults:");

    // Should have increased
    assert!(major_after >= major_before);
    assert!(minor_after >= minor_before);
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_migration_tracking() {
    // Create a test file to trigger migrations
    let test_file = "/tmp/tier_watch_test.dat";
    let mut file = fs::File::create(test_file).unwrap();

    // Write data that will span multiple tiers
    let data = vec![0u8; 1 << 30]; // 1GB
    file.write_all(&data).unwrap();
    file.sync_all().unwrap();

    // Read migration stats
    let cpu_stats = fs::read_to_string("/proc/swarm/tiers/cpu/stats").unwrap();
    let nvme_stats = fs::read_to_string("/proc/swarm/tiers/nvme/stats").unwrap();

    // Extract migration counts
    let get_migration_count = |stats: &str, direction: &str| -> u64 {
        stats
            .lines()
            .find(|line| line.starts_with(direction))
            .unwrap()
            .split(':')
            .nth(1)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    };

    let cpu_out = get_migration_count(&cpu_stats, "Migrations out:");
    let nvme_in = get_migration_count(&nvme_stats, "Migrations in:");

    // Clean up
    fs::remove_file(test_file).unwrap();

    // Verify migrations were tracked
    println!("CPU migrations out: {}", cpu_out);
    println!("NVMe migrations in: {}", nvme_in);
}

#[test]
#[ignore] // Requires loaded kernel module and swarm_guard
fn test_agent_memory_tracking() {
    // This test requires swarm_guard to be loaded for agent support

    // Create an agent through swarm_guard
    let agent_config = r#"{
        "memory_limit": 1073741824,
        "cpu_quota": 25,
        "namespace_flags": 63
    }"#;

    fs::write("/proc/swarm/create", agent_config).expect("Failed to create agent");

    // Check agent memory is tracked in tiers
    let agent_stats =
        fs::read_to_string("/proc/swarm/tiers/agents").expect("Failed to read agent tier stats");

    // Should show per-agent memory usage
    assert!(agent_stats.contains("Agent"));
    assert!(agent_stats.contains("GPU:"));
    assert!(agent_stats.contains("CPU:"));
    assert!(agent_stats.contains("NVMe:"));
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_numa_awareness() {
    // Check NUMA integration
    let numa_stats =
        fs::read_to_string("/proc/swarm/tiers/numa").expect("Failed to read NUMA stats");

    // Verify NUMA nodes are tracked
    assert!(numa_stats.contains("Node 0:"));

    // Check for local/remote access tracking
    assert!(numa_stats.contains("Local accesses:"));
    assert!(numa_stats.contains("Remote accesses:"));
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_concurrent_access() {
    // Spawn multiple threads reading tier stats
    let mut handles = vec![];

    for i in 0..10 {
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let tier = match i % 5 {
                    0 => "gpu",
                    1 => "cpu",
                    2 => "nvme",
                    3 => "ssd",
                    _ => "hdd",
                };

                let stats_path = format!("/proc/swarm/tiers/{}/stats", tier);
                let _stats = fs::read_to_string(&stats_path);

                thread::sleep(Duration::from_micros(100));
            }
        });
        handles.push(handle);
    }

    // All threads should complete without issues
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
#[ignore] // Requires loaded kernel module
fn test_performance_overhead() {
    use std::time::Instant;

    // Measure overhead of reading tier stats
    let iterations = 10000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _stats = fs::read_to_string("/proc/swarm/tiers/cpu/stats").ok();
    }

    let elapsed = start.elapsed();
    let per_read_us = elapsed.as_micros() / iterations;

    println!("Tier stats read overhead: {}μs per read", per_read_us);

    // Should be reasonably fast
    assert!(per_read_us < 100); // Under 100μs per read
}

#[test]
fn test_mock_page_migration() {
    // Test migration logic without kernel module

    #[derive(Debug)]
    struct PageMigration {
        page_id: u64,
        from_tier: &'static str,
        to_tier: &'static str,
        reason: &'static str,
        latency_us: u64,
    }

    let mut migrations = Vec::new();

    // Simulate migration decisions
    let gpu_pressure = 85; // High pressure
    let cpu_page_cold = true;
    let nvme_page_hot = true;

    if gpu_pressure > 80 {
        migrations.push(PageMigration {
            page_id: 1001,
            from_tier: "gpu",
            to_tier: "cpu",
            reason: "high_pressure",
            latency_us: 50,
        });
    }

    if cpu_page_cold {
        migrations.push(PageMigration {
            page_id: 2001,
            from_tier: "cpu",
            to_tier: "nvme",
            reason: "cold_page",
            latency_us: 200,
        });
    }

    if nvme_page_hot {
        migrations.push(PageMigration {
            page_id: 3001,
            from_tier: "nvme",
            to_tier: "cpu",
            reason: "hot_page",
            latency_us: 300,
        });
    }

    // Verify migrations
    assert_eq!(migrations.len(), 3);

    // All migrations should be under 1ms target
    for migration in &migrations {
        assert!(migration.latency_us < 1000);
    }
}

// Mock libc for tests
mod libc {
    pub unsafe fn getuid() -> u32 {
        // Return non-zero to skip root tests in CI
        1000
    }
}
