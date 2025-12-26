//! Integration tests for the stratoswarm CLI

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("StratoSwarm orchestration"))
        .stdout(predicate::str::contains("Usage"));
}

#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("stratoswarm"));
}

#[test]
fn test_deploy_command() {
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
    cmd.arg("deploy")
        .arg(swarm_file.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("Deploying"));
}

#[test]
fn test_status_command() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("status").assert().success();
}

#[test]
fn test_logs_command() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("logs")
        .arg("frontend")
        .arg("--since")
        .arg("1h")
        .assert()
        .success();
}

#[test]
fn test_scale_command() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("scale")
        .arg("backend=10")
        .assert()
        .success()
        .stdout(predicate::str::contains("Scaling"));
}

#[test]
fn test_evolve_command() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("evolve")
        .arg("backend")
        .arg("--generations")
        .arg("100")
        .arg("--dry-run")
        .assert()
        .success()
        .stdout(predicate::str::contains("evolution"));
}

#[test]
fn test_quickstart_command() {
    let temp_dir = TempDir::new().unwrap();
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("quickstart")
        .arg("--template")
        .arg("web-api")
        .arg("--gpu")
        .arg("--output")
        .arg(temp_dir.path().to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("Created"));

    // Check that files were created (quickstart creates a project directory)
    assert!(temp_dir
        .path()
        .join("web-api-app")
        .join("app.swarm")
        .exists());
}

#[test]
fn test_invalid_command() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("invalid-command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_deploy_nonexistent_file() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("deploy")
        .arg("nonexistent.swarm")
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn test_deploy_invalid_swarm_file() {
    let temp_dir = TempDir::new().unwrap();
    let swarm_file = temp_dir.path().join("invalid.swarm");

    fs::write(&swarm_file, "invalid syntax").unwrap();

    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("deploy")
        .arg(swarm_file.to_str().unwrap())
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error: Parse"));
}

#[test]
fn test_logs_with_filters() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("logs")
        .arg("frontend")
        .arg("--errors-only")
        .arg("--since")
        .arg("30m")
        .assert()
        .success();
}

#[test]
fn test_shell_help() {
    let mut cmd = Command::cargo_bin("stratoswarm").unwrap();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Enter interactive shell"));
}
