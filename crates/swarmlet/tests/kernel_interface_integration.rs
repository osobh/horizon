//! Integration tests for kernel interface configuration structures
//!
//! These tests verify the configuration builders and data structures
//! without requiring the actual kernel module to be loaded.

use swarmlet::kernel_interface::{
    mount_flags, AgentConfig, BindMountConfig, BindMountEntry, KernelInterface,
    MountIsolationConfig, OverlayFsConfig, MAX_BIND_MOUNTS, MAX_PATH_LEN,
};

/// Test AgentConfig creation and defaults
#[test]
fn test_agent_config_creation() {
    let config = AgentConfig::new("build-job-123", 4096, 100);

    // Check name is properly copied
    let name = String::from_utf8_lossy(&config.name);
    assert!(name.starts_with("build-job-123"));

    // Check memory limit is converted to bytes
    assert_eq!(config.memory_limit, 4096 * 1024 * 1024);

    // Check CPU quota
    assert_eq!(config.cpu_quota, 100);

    // Agent ID should be 0 initially (set by kernel)
    assert_eq!(config.agent_id, 0);
}

/// Test AgentConfig with long name truncation
#[test]
fn test_agent_config_long_name_truncation() {
    let long_name = "a".repeat(100);
    let config = AgentConfig::new(&long_name, 1024, 50);

    // Name should be truncated to 63 characters
    let name_end = config.name.iter().position(|&b| b == 0).unwrap_or(64);
    assert_eq!(name_end, 63);
}

/// Test AgentConfig default values
#[test]
fn test_agent_config_default() {
    let config = AgentConfig::default();

    assert_eq!(config.memory_limit, 0);
    assert_eq!(config.cpu_quota, 0);
    assert_eq!(config.gpu_memory_limit, 0);
    assert_eq!(config.namespace_flags, 0);
    assert_eq!(config.agent_id, 0);
    assert!(config.name.iter().all(|&b| b == 0));
}

/// Test BindMountConfig creation
#[test]
fn test_bind_mount_config_creation() {
    let config = BindMountConfig::new(
        42,
        "/host/path/cache",
        "/container/cache",
        mount_flags::READONLY | mount_flags::NOEXEC,
    );

    assert_eq!(config.agent_id, 42);
    assert_eq!(config.flags, mount_flags::READONLY | mount_flags::NOEXEC);

    // Check paths are properly copied
    let source = extract_path(&config.source);
    let target = extract_path(&config.target);
    assert_eq!(source, "/host/path/cache");
    assert_eq!(target, "/container/cache");
}

/// Test BindMountConfig default
#[test]
fn test_bind_mount_config_default() {
    let config = BindMountConfig::default();

    assert_eq!(config.agent_id, 0);
    assert_eq!(config.flags, 0);
    assert!(config.source.iter().all(|&b| b == 0));
    assert!(config.target.iter().all(|&b| b == 0));
}

/// Test OverlayFsConfig creation
#[test]
fn test_overlayfs_config_creation() {
    let config = OverlayFsConfig::new(99, "/lower/dir", "/upper/dir", "/work/dir", "/merged/dir");

    assert_eq!(config.agent_id, 99);

    let lower = extract_path(&config.lower_dir);
    let upper = extract_path(&config.upper_dir);
    let work = extract_path(&config.work_dir);
    let merged = extract_path(&config.merged_dir);

    assert_eq!(lower, "/lower/dir");
    assert_eq!(upper, "/upper/dir");
    assert_eq!(work, "/work/dir");
    assert_eq!(merged, "/merged/dir");
}

/// Test OverlayFsConfig default
#[test]
fn test_overlayfs_config_default() {
    let config = OverlayFsConfig::default();

    assert_eq!(config.agent_id, 0);
    assert!(config.lower_dir.iter().all(|&b| b == 0));
    assert!(config.upper_dir.iter().all(|&b| b == 0));
    assert!(config.work_dir.iter().all(|&b| b == 0));
    assert!(config.merged_dir.iter().all(|&b| b == 0));
}

/// Test MountIsolationConfig builder pattern
#[test]
fn test_mount_isolation_config_builder() {
    let config = MountIsolationConfig::new(123)
        .with_overlayfs("/lower", "/upper", "/work", "/merged")
        .with_old_root("/.oldroot");

    assert_eq!(config.agent_id, 123);
    assert_eq!(config.use_overlayfs, 1);
    assert_eq!(config.private_mounts, 1); // Default

    let lower = extract_path(&config.lower_dir);
    let merged = extract_path(&config.merged_dir);
    let old_root = extract_path(&config.old_root);

    assert_eq!(lower, "/lower");
    assert_eq!(merged, "/merged");
    assert_eq!(old_root, "/.oldroot");
}

/// Test MountIsolationConfig adding bind mounts
#[test]
fn test_mount_isolation_add_bind_mounts() {
    let mut config = MountIsolationConfig::new(456);

    // Add multiple bind mounts
    assert!(config.add_bind_mount("/host/cache", "/cache", mount_flags::READONLY));
    assert!(config.add_bind_mount(
        "/host/src",
        "/src",
        mount_flags::READONLY | mount_flags::NOEXEC
    ));
    assert!(config.add_bind_mount("/host/output", "/output", 0));

    assert_eq!(config.num_bind_mounts, 3);

    // Verify first mount
    let source0 = extract_path(&config.bind_mounts[0].source);
    let target0 = extract_path(&config.bind_mounts[0].target);
    assert_eq!(source0, "/host/cache");
    assert_eq!(target0, "/cache");
    assert_eq!(config.bind_mounts[0].flags, mount_flags::READONLY);

    // Verify third mount (no flags)
    assert_eq!(config.bind_mounts[2].flags, 0);
}

/// Test MountIsolationConfig bind mount limit
#[test]
fn test_mount_isolation_bind_mount_limit() {
    let mut config = MountIsolationConfig::new(789);

    // Fill up all bind mount slots
    for i in 0..MAX_BIND_MOUNTS {
        let source = format!("/host/{}", i);
        let target = format!("/container/{}", i);
        assert!(config.add_bind_mount(&source, &target, 0));
    }

    assert_eq!(config.num_bind_mounts as usize, MAX_BIND_MOUNTS);

    // Should fail to add more
    assert!(!config.add_bind_mount("/host/overflow", "/overflow", 0));

    // Count should remain at max
    assert_eq!(config.num_bind_mounts as usize, MAX_BIND_MOUNTS);
}

/// Test MountIsolationConfig default values
#[test]
fn test_mount_isolation_config_default() {
    let config = MountIsolationConfig::default();

    assert_eq!(config.agent_id, 0);
    assert_eq!(config.num_bind_mounts, 0);
    assert_eq!(config.use_overlayfs, 0);
    assert_eq!(config.private_mounts, 1); // Default is true
}

/// Test mount flags constants
#[test]
fn test_mount_flags() {
    assert_eq!(mount_flags::READONLY, 0x01);
    assert_eq!(mount_flags::NOSUID, 0x02);
    assert_eq!(mount_flags::NOEXEC, 0x04);
    assert_eq!(mount_flags::NODEV, 0x08);

    // Test combining flags
    let combined = mount_flags::READONLY | mount_flags::NOSUID | mount_flags::NOEXEC;
    assert_eq!(combined, 0x07);

    // All flags
    let all_flags =
        mount_flags::READONLY | mount_flags::NOSUID | mount_flags::NOEXEC | mount_flags::NODEV;
    assert_eq!(all_flags, 0x0F);
}

/// Test path length limits
#[test]
fn test_path_length_limits() {
    // Create a path at the exact limit
    let exact_path = "/".to_string() + &"a".repeat(MAX_PATH_LEN - 2); // -2 for / and null
    let config = BindMountConfig::new(1, &exact_path, "/target", 0);
    let extracted = extract_path(&config.source);
    assert!(extracted.len() < MAX_PATH_LEN);

    // Create a path exceeding the limit
    let long_path = "/".to_string() + &"b".repeat(MAX_PATH_LEN + 100);
    let config2 = BindMountConfig::new(1, &long_path, "/target", 0);
    let extracted2 = extract_path(&config2.source);
    assert!(extracted2.len() < MAX_PATH_LEN);
}

/// Test KernelInterface availability check
#[test]
fn test_kernel_interface_availability() {
    // This will typically return false unless on a system with swarm_guard loaded
    let available = KernelInterface::is_available();

    // We can't assert the value since it depends on the environment,
    // but we verify the function runs without panicking
    let _ = available;
}

/// Test BindMountEntry default
#[test]
fn test_bind_mount_entry_default() {
    let entry = BindMountEntry::default();

    assert_eq!(entry.flags, 0);
    assert!(entry.source.iter().all(|&b| b == 0));
    assert!(entry.target.iter().all(|&b| b == 0));
}

/// Test struct sizes for C ABI compatibility
#[test]
fn test_struct_sizes() {
    // These tests ensure our structs have expected sizes for FFI
    // AgentConfig should be aligned and sized for C ABI
    assert!(std::mem::size_of::<AgentConfig>() > 0);

    // BindMountConfig should contain paths and flags
    assert!(std::mem::size_of::<BindMountConfig>() >= MAX_PATH_LEN * 2 + 8 + 4);

    // MountIsolationConfig is the largest struct
    let mount_config_size = std::mem::size_of::<MountIsolationConfig>();
    assert!(mount_config_size > std::mem::size_of::<OverlayFsConfig>());
    assert!(mount_config_size > std::mem::size_of::<BindMountConfig>());
}

/// Test complex mount isolation scenario
#[test]
fn test_complex_mount_isolation_scenario() {
    let mut config = MountIsolationConfig::new(12345)
        .with_overlayfs(
            "/var/lib/swarmlet/layers/base",
            "/var/lib/swarmlet/work/12345/upper",
            "/var/lib/swarmlet/work/12345/work",
            "/var/lib/swarmlet/work/12345/merged",
        )
        .with_old_root("/.oldroot");

    // Add typical build environment mounts
    config.add_bind_mount(
        "/var/lib/swarmlet/cache/registry",
        "/root/.cargo/registry",
        mount_flags::READONLY,
    );
    config.add_bind_mount(
        "/var/lib/swarmlet/cache/git",
        "/root/.cargo/git",
        mount_flags::READONLY,
    );
    config.add_bind_mount(
        "/var/lib/swarmlet/cache/sccache",
        "/root/.cache/sccache",
        mount_flags::READONLY,
    );
    config.add_bind_mount(
        "/var/lib/swarmlet/work/12345/output",
        "/output",
        0, // Read-write for output
    );
    config.add_bind_mount(
        "/proc",
        "/proc",
        mount_flags::NOSUID | mount_flags::NOEXEC | mount_flags::NODEV,
    );
    config.add_bind_mount("/dev/null", "/dev/null", mount_flags::NOSUID);
    config.add_bind_mount("/dev/urandom", "/dev/urandom", mount_flags::NOSUID);

    assert_eq!(config.num_bind_mounts, 7);
    assert_eq!(config.use_overlayfs, 1);
    assert_eq!(config.private_mounts, 1);

    // Verify a specific mount
    let registry_mount = &config.bind_mounts[0];
    assert_eq!(registry_mount.flags, mount_flags::READONLY);
}

/// Test overlayfs configuration with relative paths
#[test]
fn test_overlayfs_relative_paths() {
    // Test that relative paths are preserved (even if not recommended)
    let config = OverlayFsConfig::new(
        1,
        "relative/lower",
        "relative/upper",
        "relative/work",
        "relative/merged",
    );

    let lower = extract_path(&config.lower_dir);
    assert_eq!(lower, "relative/lower");
}

/// Test MountIsolationConfig without overlayfs
#[test]
fn test_mount_isolation_without_overlayfs() {
    let mut config = MountIsolationConfig::new(999);

    // Just add bind mounts without overlayfs
    config.add_bind_mount("/host/src", "/src", mount_flags::READONLY);

    assert_eq!(config.use_overlayfs, 0);
    assert_eq!(config.num_bind_mounts, 1);
}

/// Test AgentConfig memory calculations
#[test]
fn test_agent_config_memory_calculations() {
    // Test various memory sizes
    let configs = [
        (512, 512u64 * 1024 * 1024),
        (1024, 1024u64 * 1024 * 1024),
        (4096, 4096u64 * 1024 * 1024),
        (8192, 8192u64 * 1024 * 1024),
        (16384, 16384u64 * 1024 * 1024),
    ];

    for (mb, expected_bytes) in configs {
        let config = AgentConfig::new("test", mb, 100);
        assert_eq!(config.memory_limit, expected_bytes, "Failed for {}MB", mb);
    }
}

/// Test CPU quota values
#[test]
fn test_agent_config_cpu_quota() {
    // Test various CPU quota percentages
    for quota in [1, 10, 25, 50, 75, 100, 200, 400] {
        let config = AgentConfig::new("test", 1024, quota);
        assert_eq!(config.cpu_quota, quota);
    }
}

// Helper function to extract a path string from a fixed-size buffer
fn extract_path(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).to_string()
}
