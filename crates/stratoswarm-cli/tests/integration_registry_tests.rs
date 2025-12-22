//! Integration tests for CLI registry functionality
//!
//! These tests verify that the registry commands work correctly when
//! integrated with the CLI framework, testing the full workflow from
//! command parsing to execution.

use assert_cmd::Command;
use predicates::prelude::*;
use serial_test::serial;
use std::fs;
use tempfile::TempDir;

/// Test CLI registry build command integration
#[test]
#[serial]
fn test_cli_registry_build_rootfs() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&[
        "registry",
        "build",
        "test-ubuntu:22.04",
        "--build-type",
        "rootfs",
        "--variant",
        "base",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains(
        "Building image: test-ubuntu:22.04",
    ))
    .stdout(predicate::str::contains("✓ Image built successfully"));
}

/// Test CLI registry build command with convert type
#[test]
#[serial]
fn test_cli_registry_build_convert() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&[
        "registry",
        "build",
        "converted-app:v1.0",
        "--build-type",
        "convert",
        "--from",
        "ubuntu:20.04",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains(
        "Building image: converted-app:v1.0",
    ))
    .stdout(predicate::str::contains("✓ Image built successfully"));
}

/// Test CLI registry build command should fail without --from for convert
#[test]
#[serial]
fn test_cli_registry_build_convert_missing_from() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&[
        "registry",
        "build",
        "invalid-app:v1.0",
        "--build-type",
        "convert", // Missing --from argument
    ])
    .assert()
    .failure()
    .stderr(predicate::str::contains("--from is required"));
}

/// Test CLI registry push command integration
#[test]
#[serial]
fn test_cli_registry_push() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "push", "myapp:latest"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Pushing image: myapp:latest"))
        .stdout(predicate::str::contains("✓ Image pushed successfully"));
}

/// Test CLI registry push command with force flag
#[test]
#[serial]
fn test_cli_registry_push_force() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "push", "myapp:latest", "--force"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Pushing image: myapp:latest"))
        .stdout(predicate::str::contains("✓ Image pushed successfully"));
}

/// Test CLI registry pull command integration
#[test]
#[serial]
fn test_cli_registry_pull() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "pull", "ubuntu:22.04"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Pulling image: ubuntu:22.04"))
        .stdout(predicate::str::contains("✓ Image pulled successfully"));
}

/// Test CLI registry pull command with streaming
#[test]
#[serial]
fn test_cli_registry_pull_stream() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "pull", "ubuntu:22.04", "--stream"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Pulling image: ubuntu:22.04"))
        .stdout(predicate::str::contains("Using progressive streaming"))
        .stdout(predicate::str::contains("✓ Image pulled successfully"));
}

/// Test CLI registry list command with different output formats
#[test]
#[serial]
fn test_cli_registry_list_table() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Listing available images"));
}

#[test]
#[serial]
fn test_cli_registry_list_json() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "list", "--output", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Listing available images"))
        .stdout(predicate::str::contains(r#"{"images": []}"#));
}

#[test]
#[serial]
fn test_cli_registry_list_yaml() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "list", "--output", "yaml"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Listing available images"))
        .stdout(predicate::str::contains("images: []"));
}

/// Test CLI registry remove command
#[test]
#[serial]
fn test_cli_registry_remove() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "remove", "old-image:v0.1"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Removing image: old-image:v0.1"))
        .stdout(predicate::str::contains("✓ Image removed"));
}

/// Test CLI registry remove command with force flag
#[test]
#[serial]
fn test_cli_registry_remove_force() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "remove", "old-image:v0.1", "--force"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Removing image: old-image:v0.1"))
        .stdout(predicate::str::contains("✓ Image removed"));
}

/// Test CLI registry verify command
#[test]
#[serial]
fn test_cli_registry_verify() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "verify", "secure-app:latest"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Verifying image: secure-app:latest",
        ))
        .stdout(predicate::str::contains("✓ Image verification passed"));
}

/// Test CLI registry verify command with policy
#[test]
#[serial]
fn test_cli_registry_verify_with_policy() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&[
        "registry",
        "verify",
        "secure-app:latest",
        "--policy",
        "strict",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains(
        "Verifying image: secure-app:latest",
    ))
    .stdout(predicate::str::contains("✓ Image verification passed"));
}

/// Test invalid registry command
#[test]
#[serial]
fn test_cli_registry_invalid_command() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "invalid-command"])
        .assert()
        .failure();
}

/// Test registry command with invalid arguments
#[test]
#[serial]
fn test_cli_registry_invalid_args() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    // Missing required image argument for build
    cmd.args(&["registry", "build"]).assert().failure();
}

/// Test workflow: build -> push -> pull -> verify -> remove
#[test]
#[serial]
fn test_cli_registry_complete_workflow() {
    let image_name = "workflow-test:v1.0";

    // Build
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.args(&["registry", "build", image_name])
        .assert()
        .success()
        .stdout(predicate::str::contains("✓ Image built successfully"));

    // Push
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.args(&["registry", "push", image_name])
        .assert()
        .success()
        .stdout(predicate::str::contains("✓ Image pushed successfully"));

    // List (should show our image)
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.args(&["registry", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Listing available images"));

    // Verify
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.args(&["registry", "verify", image_name])
        .assert()
        .success()
        .stdout(predicate::str::contains("✓ Image verification passed"));

    // Remove
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.args(&["registry", "remove", image_name])
        .assert()
        .success()
        .stdout(predicate::str::contains("✓ Image removed"));
}

/// Integration test with deploy command using --image flag
#[test]
#[serial]
fn test_cli_deploy_with_image() {
    // Create a temporary swarm file
    let temp_dir = TempDir::new().unwrap();
    let swarm_file = temp_dir.path().join("test.swarm");

    fs::write(
        &swarm_file,
        r#"
        swarm test {
            agents {
                web: WebAgent {
                    replicas: 1,
                }
            }
        }
    "#,
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.args(&[
        "deploy",
        swarm_file.to_str().unwrap(),
        "--image",
        "test-image:latest",
        "--dry-run",
    ])
    .assert()
    .success()
    .stdout(predicate::str::contains(
        "Deploying from image: test-image:latest",
    ))
    .stdout(predicate::str::contains(
        "Would deploy image: test-image:latest",
    ));
}

/// Test that the CLI shows help when no registry subcommand is provided
#[test]
#[serial]
fn test_cli_registry_help() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();

    cmd.args(&["registry", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Manage container images and registry",
        ))
        .stdout(predicate::str::contains("build"))
        .stdout(predicate::str::contains("push"))
        .stdout(predicate::str::contains("pull"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("remove"))
        .stdout(predicate::str::contains("verify"));
}
