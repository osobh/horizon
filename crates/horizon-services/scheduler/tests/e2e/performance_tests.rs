// End-to-end performance tests
//
// These tests validate performance targets for multi-service operations:
// - End-to-end latency < 50ms (p99)
// - Multi-service request < 100ms (p99)
// - Concurrent throughput > 100 req/s
//
// Tests assume ALL services are running on localhost.

use anyhow::Result;
use serde_json::json;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::e2e::helpers::*;

/// Calculate percentile from sorted durations
fn calculate_percentile(durations: &mut [Duration], percentile: f64) -> Duration {
    durations.sort();
    let index = ((percentile / 100.0) * durations.len() as f64) as usize;
    durations[index.min(durations.len() - 1)]
}

/// Calculate statistics from duration samples
#[derive(Debug)]
struct Stats {
    min: Duration,
    max: Duration,
    mean: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
}

impl Stats {
    fn from_durations(durations: &mut [Duration]) -> Self {
        durations.sort();
        let total: Duration = durations.iter().sum();
        let mean = total / durations.len() as u32;

        Self {
            min: durations[0],
            max: durations[durations.len() - 1],
            mean,
            p50: calculate_percentile(durations, 50.0),
            p95: calculate_percentile(durations, 95.0),
            p99: calculate_percentile(durations, 99.0),
        }
    }

    fn print(&self, label: &str) {
        println!("\n{} Performance Statistics:", label);
        println!("  Min:  {:?}", self.min);
        println!("  Mean: {:?}", self.mean);
        println!("  P50:  {:?}", self.p50);
        println!("  P95:  {:?}", self.p95);
        println!("  P99:  {:?}", self.p99);
        println!("  Max:  {:?}", self.max);
    }
}

// ============================================================================
// End-to-End Latency Benchmark
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: api-gateway, scheduler, postgres - Performance test"]
async fn bench_e2e_job_submission_latency() -> Result<()> {
    // Measure end-to-end latency for job submission through gateway
    // Target: < 50ms p99

    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::scheduler()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();

    // Warmup: 10 requests
    for _ in 0..10 {
        let job_req = SubmitJobRequest {
            user_id: "warmup-user".to_string(),
            priority: 5,
            cpus: 2,
            memory_mb: 4096,
            gpus: None,
            gpu_type: None,
            estimated_duration_secs: Some(3600),
            metadata: None,
        };
        let _ = gateway_client.submit_job(&job_req).await?;
    }

    // Benchmark: 100 requests
    let mut durations = Vec::with_capacity(100);

    for i in 0..100 {
        let job_req = SubmitJobRequest {
            user_id: format!("bench-user-{}", i),
            priority: 5,
            cpus: 2,
            memory_mb: 4096,
            gpus: Some(1),
            gpu_type: Some("T4".to_string()),
            estimated_duration_secs: Some(3600),
            metadata: Some(json!({"benchmark": true, "iteration": i})),
        };

        let start = Instant::now();
        let response = gateway_client.submit_job(&job_req).await?;
        let elapsed = start.elapsed();

        assert!(
            response.status().is_success(),
            "Job submission failed during benchmark"
        );

        durations.push(elapsed);
    }

    // Calculate and print statistics
    let stats = Stats::from_durations(&mut durations);
    stats.print("E2E Job Submission");

    // Assert performance target
    assert!(
        stats.p99 < Duration::from_millis(50),
        "P99 latency ({:?}) exceeds target of 50ms",
        stats.p99
    );

    println!("\n✓ E2E latency target met: P99 = {:?} < 50ms", stats.p99);

    Ok(())
}

// ============================================================================
// Multi-Service Request Benchmark
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: api-gateway, governor, quota-manager, postgres - Performance test"]
async fn bench_multi_service_request_latency() -> Result<()> {
    // Measure latency for requests that touch multiple services
    // Target: < 100ms p99

    let services = vec![
        ServiceEndpoint::api_gateway(),
        ServiceEndpoint::governor(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let governor_client = GovernorClient::from_env();
    let quota_client = QuotaManagerClient::from_env();

    // Setup: Create policy
    let policy_content = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: perf-test-policy
  version: "1.0"
spec:
  principals:
    - type: user
      pattern: "*"
  resources:
    - type: job
      pattern: "*"
  rules:
    - effect: allow
      actions: [submit]
      conditions: []
"#;

    let create_policy_req = CreatePolicyRequest {
        name: "perf-test-policy".to_string(),
        content: policy_content.to_string(),
        description: Some("Performance test policy".to_string()),
    };

    governor_client.create_policy(&create_policy_req).await?;

    // Setup: Create quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "perf-test-user".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 10000.0,
        period_days: Some(30),
    };

    let quota_resp = quota_client.create_quota(&create_quota_req).await?;
    let quota: serde_json::Value = quota_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Warmup: 10 requests
    for _ in 0..10 {
        let eval_req = EvaluateRequest {
            principal: Principal {
                id: "perf-test-user".to_string(),
                principal_type: "user".to_string(),
                attributes: None,
            },
            action: "submit".to_string(),
            resource: Resource {
                id: "warmup-job".to_string(),
                resource_type: "job".to_string(),
                attributes: Some(json!({"gpus": 4})),
            },
        };
        let _ = gateway_client.evaluate_policy(&eval_req).await?;
    }

    // Benchmark: 100 multi-service requests
    let mut durations = Vec::with_capacity(100);

    for i in 0..100 {
        let start = Instant::now();

        // Step 1: Evaluate policy through gateway
        let eval_req = EvaluateRequest {
            principal: Principal {
                id: "perf-test-user".to_string(),
                principal_type: "user".to_string(),
                attributes: None,
            },
            action: "submit".to_string(),
            resource: Resource {
                id: format!("bench-job-{}", i),
                resource_type: "job".to_string(),
                attributes: Some(json!({"gpus": 4})),
            },
        };

        let eval_resp = gateway_client.evaluate_policy(&eval_req).await?;
        assert!(eval_resp.status().is_success());

        // Step 2: Check quota through gateway
        let check_req = AllocationCheckRequest {
            entity_type: "user".to_string(),
            entity_id: "perf-test-user".to_string(),
            resource_type: "gpu_hours".to_string(),
            amount: 16.0,
        };

        let check_resp = gateway_client.check_quota(&check_req).await?;
        assert!(check_resp.status().is_success());

        let elapsed = start.elapsed();
        durations.push(elapsed);
    }

    // Calculate and print statistics
    let stats = Stats::from_durations(&mut durations);
    stats.print("Multi-Service Request");

    // Assert performance target
    assert!(
        stats.p99 < Duration::from_millis(100),
        "P99 latency ({:?}) exceeds target of 100ms",
        stats.p99
    );

    println!(
        "\n✓ Multi-service latency target met: P99 = {:?} < 100ms",
        stats.p99
    );

    // Cleanup
    let _ = governor_client.delete_policy("perf-test-policy").await;
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}

// ============================================================================
// Concurrent Throughput Benchmark
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: api-gateway, scheduler, postgres - Performance test"]
async fn bench_concurrent_throughput() -> Result<()> {
    // Measure throughput under concurrent load
    // Target: > 100 req/s

    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::scheduler()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();

    // Test parameters
    let total_requests = 1000;
    let concurrency = 50; // 50 concurrent workers

    println!("\nConcurrent Throughput Benchmark:");
    println!("  Total requests: {}", total_requests);
    println!("  Concurrency:    {}", concurrency);

    let start = Instant::now();
    let mut handles = vec![];

    // Spawn concurrent workers
    for worker_id in 0..concurrency {
        let client = gateway_client.clone();
        let requests_per_worker = total_requests / concurrency;

        let handle = tokio::spawn(async move {
            let mut worker_durations = vec![];

            for i in 0..requests_per_worker {
                let job_req = SubmitJobRequest {
                    user_id: format!("throughput-user-{}-{}", worker_id, i),
                    priority: 5,
                    cpus: 2,
                    memory_mb: 4096,
                    gpus: None,
                    gpu_type: None,
                    estimated_duration_secs: Some(3600),
                    metadata: Some(json!({"worker": worker_id, "request": i})),
                };

                let req_start = Instant::now();
                let response = client.submit_job(&job_req).await;
                let req_elapsed = req_start.elapsed();

                if let Ok(resp) = response {
                    if resp.status().is_success() {
                        worker_durations.push(req_elapsed);
                    }
                }
            }

            worker_durations
        });

        handles.push(handle);
    }

    // Wait for all workers to complete
    let results = futures::future::join_all(handles).await;

    let total_elapsed = start.elapsed();

    // Collect all durations
    let mut all_durations = vec![];
    let mut successful_requests = 0;

    for durations in results.into_iter().flatten() {
        successful_requests += durations.len();
        all_durations.extend(durations);
    }

    // Calculate throughput
    let throughput = successful_requests as f64 / total_elapsed.as_secs_f64();

    println!("\nThroughput Results:");
    println!("  Successful requests: {}", successful_requests);
    println!("  Total time:          {:?}", total_elapsed);
    println!("  Throughput:          {:.2} req/s", throughput);

    // Calculate latency stats
    let stats = Stats::from_durations(&mut all_durations);
    stats.print("Per-Request Latency Under Load");

    // Assert throughput target
    assert!(
        throughput >= 100.0,
        "Throughput ({:.2} req/s) is below target of 100 req/s",
        throughput
    );

    println!(
        "\n✓ Throughput target met: {:.2} req/s >= 100 req/s",
        throughput
    );

    Ok(())
}

// ============================================================================
// Individual Service Performance Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: governor, postgres - Performance test"]
async fn bench_governor_policy_evaluation() -> Result<()> {
    // Measure policy evaluation latency
    // Target: < 5ms p99

    let services = vec![ServiceEndpoint::governor()];
    require_services(&services).await?;

    let governor_client = GovernorClient::from_env();

    // Setup: Create policy
    let policy_content = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: eval-perf-test
  version: "1.0"
spec:
  principals:
    - type: user
      pattern: "*"
  resources:
    - type: job
      pattern: "*"
  rules:
    - effect: allow
      actions: [submit]
      conditions:
        - field: resource.gpus
          operator: lte
          value: 8
"#;

    let create_policy_req = CreatePolicyRequest {
        name: "eval-perf-test".to_string(),
        content: policy_content.to_string(),
        description: Some("Evaluation performance test".to_string()),
    };

    governor_client.create_policy(&create_policy_req).await?;

    // Warmup
    for _ in 0..10 {
        let eval_req = EvaluateRequest {
            principal: Principal {
                id: "warmup-user".to_string(),
                principal_type: "user".to_string(),
                attributes: None,
            },
            action: "submit".to_string(),
            resource: Resource {
                id: "warmup-job".to_string(),
                resource_type: "job".to_string(),
                attributes: Some(json!({"gpus": 4})),
            },
        };
        let _ = governor_client.evaluate(&eval_req).await?;
    }

    // Benchmark: 100 evaluations
    let mut durations = Vec::with_capacity(100);

    for i in 0..100 {
        let eval_req = EvaluateRequest {
            principal: Principal {
                id: format!("bench-user-{}", i),
                principal_type: "user".to_string(),
                attributes: None,
            },
            action: "submit".to_string(),
            resource: Resource {
                id: format!("bench-job-{}", i),
                resource_type: "job".to_string(),
                attributes: Some(json!({"gpus": 4})),
            },
        };

        let start = Instant::now();
        let response = governor_client.evaluate(&eval_req).await?;
        let elapsed = start.elapsed();

        assert!(response.status().is_success());
        durations.push(elapsed);
    }

    // Calculate and print statistics
    let stats = Stats::from_durations(&mut durations);
    stats.print("Governor Policy Evaluation");

    // Assert performance target
    assert!(
        stats.p99 < Duration::from_millis(5),
        "P99 latency ({:?}) exceeds target of 5ms",
        stats.p99
    );

    println!(
        "\n✓ Policy evaluation target met: P99 = {:?} < 5ms",
        stats.p99
    );

    // Cleanup
    let _ = governor_client.delete_policy("eval-perf-test").await;

    Ok(())
}

#[tokio::test]
#[ignore = "Requires services: quota-manager, postgres - Performance test"]
async fn bench_quota_manager_check() -> Result<()> {
    // Measure quota check latency
    // Target: < 2ms p99

    let services = vec![ServiceEndpoint::quota_manager()];
    require_services(&services).await?;

    let quota_client = QuotaManagerClient::from_env();

    // Setup: Create quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "quota-perf-user".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 10000.0,
        period_days: Some(30),
    };

    let quota_resp = quota_client.create_quota(&create_quota_req).await?;
    let quota: serde_json::Value = quota_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Warmup
    for _ in 0..10 {
        let check_req = AllocationCheckRequest {
            entity_type: "user".to_string(),
            entity_id: "quota-perf-user".to_string(),
            resource_type: "gpu_hours".to_string(),
            amount: 10.0,
        };
        let _ = quota_client.check_allocation(&check_req).await?;
    }

    // Benchmark: 100 quota checks
    let mut durations = Vec::with_capacity(100);

    for _ in 0..100 {
        let check_req = AllocationCheckRequest {
            entity_type: "user".to_string(),
            entity_id: "quota-perf-user".to_string(),
            resource_type: "gpu_hours".to_string(),
            amount: 10.0,
        };

        let start = Instant::now();
        let response = quota_client.check_allocation(&check_req).await?;
        let elapsed = start.elapsed();

        assert!(response.status().is_success());
        durations.push(elapsed);
    }

    // Calculate and print statistics
    let stats = Stats::from_durations(&mut durations);
    stats.print("Quota Manager Check");

    // Assert performance target
    assert!(
        stats.p99 < Duration::from_millis(2),
        "P99 latency ({:?}) exceeds target of 2ms",
        stats.p99
    );

    println!("\n✓ Quota check target met: P99 = {:?} < 2ms", stats.p99);

    // Cleanup
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}
