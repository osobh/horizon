//! Integration tests for Docker backend isolation features
//!
//! These tests verify the isolation configuration without requiring
//! an actual Docker daemon to be running.

use swarmlet::build_backend::docker::{IsolationProfile, NetworkMode, TmpfsMount, UlimitConfig};

/// Test isolation profile presets are internally consistent
#[test]
fn test_isolation_profile_default_consistency() {
    let profile = IsolationProfile::default();

    // Default should be secure but allow builds
    assert_eq!(profile.network_mode, NetworkMode::None);
    assert!(profile.drop_capabilities);
    assert!(profile.seccomp_enabled);
    assert!(!profile.read_only_rootfs); // Builds need to write
    assert!(profile.icc_disabled);
    assert!(!profile.oom_kill_disable); // OOM killer should be enabled

    // Should have tmpfs for /tmp
    assert!(!profile.tmpfs_mounts.is_empty());
    assert!(profile.tmpfs_mounts.iter().any(|m| m.path == "/tmp"));
}

/// Test minimal profile is truly minimal
#[test]
fn test_isolation_profile_minimal_is_permissive() {
    let profile = IsolationProfile::minimal();

    assert_eq!(profile.network_mode, NetworkMode::Bridge);
    assert!(!profile.drop_capabilities);
    assert!(!profile.seccomp_enabled);
    assert!(!profile.read_only_rootfs);
    assert!(!profile.icc_disabled);
    assert!(profile.tmpfs_mounts.is_empty());
    assert!(profile.run_as_user.is_none());
    assert!(profile.userns_mode.is_none());
}

/// Test high security profile is maximally restrictive
#[test]
fn test_isolation_profile_high_security_is_restrictive() {
    let profile = IsolationProfile::high_security();

    // Network should be disabled
    assert_eq!(profile.network_mode, NetworkMode::None);

    // All security features should be enabled
    assert!(profile.drop_capabilities);
    assert!(profile.seccomp_enabled);
    assert!(profile.read_only_rootfs);
    assert!(profile.icc_disabled);

    // Should run as non-root
    assert!(profile.run_as_user.is_some());

    // Should have user namespace
    assert!(profile.userns_mode.is_some());

    // Memory swap should be disabled
    assert_eq!(profile.memory_swap_bytes, Some(-1));

    // Should have multiple tmpfs mounts
    assert!(profile.tmpfs_mounts.len() >= 2);

    // Ulimits should be strict
    assert!(profile.ulimits.nofile_soft < UlimitConfig::default().nofile_soft);
    assert!(profile.ulimits.nproc_soft < UlimitConfig::default().nproc_soft);
}

/// Test network mode conversions
#[test]
fn test_network_mode_docker_strings() {
    assert_eq!(NetworkMode::None.to_docker_string(), "none");
    assert_eq!(NetworkMode::Bridge.to_docker_string(), "bridge");
    assert_eq!(NetworkMode::Host.to_docker_string(), "host");
}

/// Test ulimit configurations
#[test]
fn test_ulimit_config_values() {
    let default_ulimits = UlimitConfig::default();
    let strict_ulimits = UlimitConfig::strict();

    // Default should allow reasonable limits
    assert!(default_ulimits.nofile_soft >= 1024);
    assert!(default_ulimits.nproc_soft >= 256);

    // Strict should be more restrictive
    assert!(strict_ulimits.nofile_soft < default_ulimits.nofile_soft);
    assert!(strict_ulimits.nproc_soft < default_ulimits.nproc_soft);

    // Core dumps should be disabled in both
    assert_eq!(default_ulimits.core_soft, 0);
    assert_eq!(strict_ulimits.core_soft, 0);
}

/// Test tmpfs mount configuration
#[test]
fn test_tmpfs_mount_configuration() {
    let mount = TmpfsMount::new("/tmp", 512 * 1024 * 1024);

    assert_eq!(mount.path, "/tmp");
    assert_eq!(mount.size_bytes, 512 * 1024 * 1024);

    // Should have security options by default
    assert!(mount.options.contains(&"noexec".to_string()));
    assert!(mount.options.contains(&"nosuid".to_string()));
}

/// Test tmpfs docker options string
#[test]
fn test_tmpfs_docker_options_string() {
    let mount = TmpfsMount::new("/tmp", 1024 * 1024);
    let options = mount.to_docker_options();

    // Should contain size
    assert!(options.contains("size=1048576"));

    // Should contain security options
    assert!(options.contains("noexec"));
    assert!(options.contains("nosuid"));
}

/// Test isolation profile customization
#[test]
fn test_isolation_profile_customization() {
    let mut profile = IsolationProfile::default();

    // Can customize network mode
    profile.network_mode = NetworkMode::Bridge;
    assert_eq!(profile.network_mode, NetworkMode::Bridge);

    // Can add capabilities
    profile.add_capabilities.push("NET_ADMIN".to_string());
    assert!(profile.add_capabilities.contains(&"NET_ADMIN".to_string()));

    // Can set custom seccomp profile
    profile.seccomp_profile = Some("/etc/docker/seccomp.json".to_string());
    assert!(profile.seccomp_profile.is_some());

    // Can add tmpfs mounts
    profile
        .tmpfs_mounts
        .push(TmpfsMount::new("/var/tmp", 256 * 1024 * 1024));
    assert!(profile.tmpfs_mounts.iter().any(|m| m.path == "/var/tmp"));
}

/// Test isolation profile serialization
#[test]
fn test_isolation_profile_serialization() {
    let profile = IsolationProfile::default();

    // Should serialize to JSON without panic
    let json = serde_json::to_string(&profile).unwrap();
    assert!(!json.is_empty());

    // Should deserialize back
    let deserialized: IsolationProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.network_mode, profile.network_mode);
    assert_eq!(deserialized.drop_capabilities, profile.drop_capabilities);
}

/// Test ulimit config serialization
#[test]
fn test_ulimit_config_serialization() {
    let config = UlimitConfig::strict();

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: UlimitConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.nofile_soft, config.nofile_soft);
    assert_eq!(deserialized.nproc_soft, config.nproc_soft);
}

/// Test that profiles have valid values for Docker
#[test]
fn test_isolation_profiles_have_valid_docker_values() {
    for profile in [
        IsolationProfile::default(),
        IsolationProfile::minimal(),
        IsolationProfile::high_security(),
    ] {
        // Network mode should be valid
        let network_str = profile.network_mode.to_docker_string();
        assert!(["none", "bridge", "host"].contains(&network_str.as_str()));

        // Ulimits should be non-negative
        assert!(profile.ulimits.nofile_soft >= 0);
        assert!(profile.ulimits.nofile_hard >= profile.ulimits.nofile_soft);
        assert!(profile.ulimits.nproc_soft >= 0);
        assert!(profile.ulimits.nproc_hard >= profile.ulimits.nproc_soft);

        // Tmpfs mounts should have valid paths
        for mount in &profile.tmpfs_mounts {
            assert!(mount.path.starts_with('/'));
            assert!(mount.size_bytes > 0);
        }

        // If user is set, should be valid format
        if let Some(ref user) = profile.run_as_user {
            // Should be uid:gid format or just uid
            assert!(user.parse::<u32>().is_ok() || user.contains(':'));
        }
    }
}

/// Test memory swap configuration options
#[test]
fn test_memory_swap_configuration() {
    let mut profile = IsolationProfile::default();

    // Default: no swap limit specified
    assert!(profile.memory_swap_bytes.is_none());

    // High security: swap disabled
    let high_sec = IsolationProfile::high_security();
    assert_eq!(high_sec.memory_swap_bytes, Some(-1));

    // Can set custom swap limit
    profile.memory_swap_bytes = Some(4 * 1024 * 1024 * 1024); // 4GB
    assert_eq!(profile.memory_swap_bytes, Some(4 * 1024 * 1024 * 1024));
}

/// Test capability list consistency
#[test]
fn test_capability_configuration() {
    let profile = IsolationProfile::default();

    // Should drop capabilities by default
    assert!(profile.drop_capabilities);

    // Should not add any by default
    assert!(profile.add_capabilities.is_empty());

    // Minimal should not drop
    let minimal = IsolationProfile::minimal();
    assert!(!minimal.drop_capabilities);
}

/// Test that different profiles have appropriate security postures
#[test]
fn test_security_posture_progression() {
    let minimal = IsolationProfile::minimal();
    let default = IsolationProfile::default();
    let high = IsolationProfile::high_security();

    // Count security features enabled
    fn security_score(p: &IsolationProfile) -> u32 {
        let mut score = 0;
        if p.network_mode == NetworkMode::None {
            score += 1;
        }
        if p.drop_capabilities {
            score += 1;
        }
        if p.seccomp_enabled {
            score += 1;
        }
        if p.read_only_rootfs {
            score += 1;
        }
        if p.icc_disabled {
            score += 1;
        }
        if p.run_as_user.is_some() {
            score += 1;
        }
        if p.userns_mode.is_some() {
            score += 1;
        }
        score
    }

    // Security should increase: minimal < default < high
    assert!(security_score(&minimal) < security_score(&default));
    assert!(security_score(&default) < security_score(&high));
}
