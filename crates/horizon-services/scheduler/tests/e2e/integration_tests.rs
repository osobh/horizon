// Multi-service integration tests
//
// These tests verify that services can communicate with each other correctly.
// They test 2-service interactions without going through the full stack.
//
// Tests assume services are running on localhost. Set environment variables
// to override default ports:
// - SCHEDULER_PORT (default: 8080)
// - GOVERNOR_PORT (default: 8081)
// - QUOTA_MANAGER_PORT (default: 8082)
// - API_GATEWAY_PORT (default: 8000)

use anyhow::Result;
use serde_json::json;
use uuid::Uuid;

use crate::e2e::helpers::*;

// ============================================================================
// API Gateway ↔ Governor Integration Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires running services: api-gateway, governor"]
async fn test_gateway_to_governor_policy_evaluation() -> Result<()> {
    // Setup: Check services are available
    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::governor()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let governor_client = GovernorClient::from_env();

    // Step 1: Create a policy directly in Governor
    let policy_content = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-allow-policy
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
        name: "test-allow-policy".to_string(),
        content: policy_content.to_string(),
        description: Some("Test policy for integration testing".to_string()),
    };

    let create_resp = governor_client.create_policy(&create_policy_req).await?;
    assert!(
        create_resp.status().is_success(),
        "Failed to create policy: {:?}",
        create_resp.text().await?
    );

    // Step 2: Evaluate policy through Gateway (proxied to Governor)
    let eval_req = EvaluateRequest {
        principal: Principal {
            id: "user-123".to_string(),
            principal_type: "user".to_string(),
            attributes: Some(json!({"team": "ml-team"})),
        },
        action: "submit".to_string(),
        resource: Resource {
            id: "job-456".to_string(),
            resource_type: "job".to_string(),
            attributes: Some(json!({"gpus": 4})),
        },
    };

    let eval_resp = gateway_client.evaluate_policy(&eval_req).await?;
    assert!(
        eval_resp.status().is_success(),
        "Failed to evaluate policy through gateway: {:?}",
        eval_resp.text().await?
    );

    let eval_result: EvaluateResponse = eval_resp.json().await?;
    assert_eq!(eval_result.decision, "allow");

    // Cleanup: Delete the test policy
    let delete_resp = governor_client.delete_policy("test-allow-policy").await?;
    assert!(delete_resp.status().is_success());

    Ok(())
}

#[tokio::test]
#[ignore = "Requires running services: api-gateway, governor"]
async fn test_gateway_to_governor_policy_denial() -> Result<()> {
    // Setup: Check services are available
    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::governor()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let governor_client = GovernorClient::from_env();

    // Step 1: Create a deny policy
    let policy_content = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-deny-policy
  version: "1.0"
spec:
  principals:
    - type: user
      pattern: "*"
  resources:
    - type: job
      pattern: "*"
  rules:
    - effect: deny
      actions: [submit]
      conditions:
        - field: resource.gpus
          operator: gt
          value: 8
"#;

    let create_policy_req = CreatePolicyRequest {
        name: "test-deny-policy".to_string(),
        content: policy_content.to_string(),
        description: Some("Test deny policy".to_string()),
    };

    let create_resp = governor_client.create_policy(&create_policy_req).await?;
    assert!(create_resp.status().is_success());

    // Step 2: Evaluate policy through Gateway (should deny for > 8 GPUs)
    let eval_req = EvaluateRequest {
        principal: Principal {
            id: "user-123".to_string(),
            principal_type: "user".to_string(),
            attributes: None,
        },
        action: "submit".to_string(),
        resource: Resource {
            id: "job-456".to_string(),
            resource_type: "job".to_string(),
            attributes: Some(json!({"gpus": 16})),
        },
    };

    let eval_resp = gateway_client.evaluate_policy(&eval_req).await?;
    assert!(eval_resp.status().is_success());

    let eval_result: EvaluateResponse = eval_resp.json().await?;
    assert_eq!(eval_result.decision, "deny");

    // Cleanup
    let _ = governor_client.delete_policy("test-deny-policy").await;

    Ok(())
}

// ============================================================================
// API Gateway ↔ Quota Manager Integration Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires running services: api-gateway, quota-manager, postgres"]
async fn test_gateway_to_quota_manager_check() -> Result<()> {
    // Setup: Check services are available
    let services = vec![
        ServiceEndpoint::api_gateway(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create a quota directly in Quota Manager
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-quota-gw".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 100.0,
        period_days: Some(30),
    };

    let create_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(
        create_resp.status().is_success(),
        "Failed to create quota: {:?}",
        create_resp.text().await?
    );

    let quota: serde_json::Value = create_resp.json().await?;
    let quota_id = quota["id"].as_str().unwrap();

    // Step 2: Check allocation through Gateway (proxied to Quota Manager)
    let check_req = AllocationCheckRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-quota-gw".to_string(),
        resource_type: "gpu_hours".to_string(),
        amount: 50.0,
    };

    let check_resp = gateway_client.check_quota(&check_req).await?;
    assert!(
        check_resp.status().is_success(),
        "Failed to check quota through gateway: {:?}",
        check_resp.text().await?
    );

    let check_result: AllocationCheckResponse = check_resp.json().await?;
    assert!(check_result.allowed, "Allocation should be allowed");
    assert_eq!(check_result.requested, 50.0);

    // Cleanup
    let quota_uuid = Uuid::parse_str(quota_id)?;
    let _ = quota_client.delete_quota(quota_uuid).await;

    Ok(())
}

#[tokio::test]
#[ignore = "Requires running services: api-gateway, quota-manager, postgres"]
async fn test_gateway_to_quota_manager_exceeded() -> Result<()> {
    // Setup
    let services = vec![
        ServiceEndpoint::api_gateway(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create a small quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-quota-exceeded".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 10.0,
        period_days: Some(30),
    };

    let create_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(create_resp.status().is_success());

    let quota: serde_json::Value = create_resp.json().await?;
    let quota_id = quota["id"].as_str().unwrap();

    // Step 2: Try to allocate more than the quota through Gateway
    let check_req = AllocationCheckRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-quota-exceeded".to_string(),
        resource_type: "gpu_hours".to_string(),
        amount: 50.0, // More than limit of 10.0
    };

    let check_resp = gateway_client.check_quota(&check_req).await?;
    assert!(check_resp.status().is_success());

    let check_result: AllocationCheckResponse = check_resp.json().await?;
    assert!(!check_result.allowed, "Allocation should be denied");
    assert_eq!(check_result.available, 10.0);
    assert_eq!(check_result.requested, 50.0);

    // Cleanup
    let quota_uuid = Uuid::parse_str(quota_id)?;
    let _ = quota_client.delete_quota(quota_uuid).await;

    Ok(())
}

#[tokio::test]
#[ignore = "Requires running services: api-gateway, quota-manager, postgres"]
async fn test_gateway_to_quota_manager_allocation_flow() -> Result<()> {
    // Setup
    let services = vec![
        ServiceEndpoint::api_gateway(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create a quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-alloc-flow".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 100.0,
        period_days: Some(30),
    };

    let create_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(create_resp.status().is_success());

    let quota: serde_json::Value = create_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Step 2: Create an allocation
    let create_alloc_req = CreateAllocationRequest {
        quota_id,
        resource_id: "job-test-123".to_string(),
        resource_type: "gpu_hours".to_string(),
        amount: 30.0,
    };

    let alloc_resp = quota_client.create_allocation(&create_alloc_req).await?;
    assert!(
        alloc_resp.status().is_success(),
        "Failed to create allocation: {:?}",
        alloc_resp.text().await?
    );

    let allocation: serde_json::Value = alloc_resp.json().await?;
    let allocation_id = Uuid::parse_str(allocation["id"].as_str().unwrap())?;

    // Step 3: Verify usage stats
    let stats_resp = quota_client.get_usage_stats(quota_id).await?;
    assert!(stats_resp.status().is_success());

    let stats: serde_json::Value = stats_resp.json().await?;
    assert_eq!(stats["used"].as_f64().unwrap(), 30.0);
    assert_eq!(stats["available"].as_f64().unwrap(), 70.0);

    // Step 4: Release allocation
    let release_resp = quota_client.release_allocation(allocation_id).await?;
    assert!(release_resp.status().is_success());

    // Step 5: Verify usage after release
    let stats_resp2 = quota_client.get_usage_stats(quota_id).await?;
    assert!(stats_resp2.status().is_success());

    let stats2: serde_json::Value = stats_resp2.json().await?;
    assert_eq!(stats2["used"].as_f64().unwrap(), 0.0);
    assert_eq!(stats2["available"].as_f64().unwrap(), 100.0);

    // Cleanup
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}

// ============================================================================
// API Gateway ↔ Scheduler Integration Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires running services: api-gateway, scheduler, postgres"]
async fn test_gateway_to_scheduler_job_submission() -> Result<()> {
    // Setup
    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::scheduler()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();

    // Submit job through Gateway
    let job_req = SubmitJobRequest {
        user_id: "test-user-gw".to_string(),
        priority: 5,
        cpus: 4,
        memory_mb: 8192,
        gpus: Some(2),
        gpu_type: Some("A100".to_string()),
        estimated_duration_secs: Some(3600),
        metadata: Some(json!({"test": "gateway-integration"})),
    };

    let submit_resp = gateway_client.submit_job(&job_req).await?;
    assert!(
        submit_resp.status().is_success(),
        "Failed to submit job through gateway: {:?}",
        submit_resp.text().await?
    );

    let job: serde_json::Value = submit_resp.json().await?;
    let job_id = Uuid::parse_str(job["id"].as_str().unwrap())?;

    // Verify job can be retrieved through Gateway
    let get_resp = gateway_client.get_job(job_id).await?;
    assert!(get_resp.status().is_success());

    let retrieved_job: serde_json::Value = get_resp.json().await?;
    assert_eq!(retrieved_job["user_id"], "test-user-gw");
    assert_eq!(retrieved_job["cpus"], 4);

    Ok(())
}

#[tokio::test]
#[ignore = "Requires running services: api-gateway, scheduler, postgres"]
async fn test_gateway_to_scheduler_job_lifecycle() -> Result<()> {
    // Setup
    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::scheduler()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let scheduler_client = SchedulerClient::from_env();

    // Step 1: Submit job through Gateway
    let job_req = SubmitJobRequest {
        user_id: "test-user-lifecycle".to_string(),
        priority: 3,
        cpus: 2,
        memory_mb: 4096,
        gpus: None,
        gpu_type: None,
        estimated_duration_secs: Some(1800),
        metadata: None,
    };

    let submit_resp = gateway_client.submit_job(&job_req).await?;
    assert!(submit_resp.status().is_success());

    let job: serde_json::Value = submit_resp.json().await?;
    let job_id = Uuid::parse_str(job["id"].as_str().unwrap())?;

    // Step 2: Get job status directly from scheduler
    let get_resp = scheduler_client.get_job(job_id).await?;
    assert!(get_resp.status().is_success());

    let job_data: serde_json::Value = get_resp.json().await?;
    assert_eq!(job_data["id"], job_id.to_string());
    assert_eq!(job_data["user_id"], "test-user-lifecycle");

    // Step 3: Cancel job through scheduler
    let cancel_resp = scheduler_client.cancel_job(job_id).await?;
    assert!(cancel_resp.status().is_success());

    // Step 4: Verify cancellation through gateway
    let verify_resp = gateway_client.get_job(job_id).await?;
    assert!(verify_resp.status().is_success());

    let cancelled_job: serde_json::Value = verify_resp.json().await?;
    assert_eq!(cancelled_job["status"], "cancelled");

    Ok(())
}

// ============================================================================
// Scheduler ↔ Quota Manager Integration Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires running services: scheduler, quota-manager, postgres"]
async fn test_scheduler_to_quota_manager_check() -> Result<()> {
    // Setup
    let services = vec![
        ServiceEndpoint::scheduler(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let scheduler_client = SchedulerClient::from_env();
    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create quota for user
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-sched-quota".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 200.0,
        period_days: Some(30),
    };

    let create_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(create_resp.status().is_success());

    let quota: serde_json::Value = create_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Step 2: Check if job would be allowed
    let check_req = AllocationCheckRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-sched-quota".to_string(),
        resource_type: "gpu_hours".to_string(),
        amount: 50.0,
    };

    let check_resp = quota_client.check_allocation(&check_req).await?;
    assert!(check_resp.status().is_success());

    let check_result: AllocationCheckResponse = check_resp.json().await?;
    assert!(check_result.allowed);

    // Step 3: Submit job through scheduler
    let job_req = SubmitJobRequest {
        user_id: "test-user-sched-quota".to_string(),
        priority: 5,
        cpus: 4,
        memory_mb: 8192,
        gpus: Some(4),
        gpu_type: Some("A100".to_string()),
        estimated_duration_secs: Some(7200), // 2 hours
        metadata: None,
    };

    let submit_resp = scheduler_client.submit_job(&job_req).await?;
    assert!(submit_resp.status().is_success());

    // Cleanup
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}

#[tokio::test]
#[ignore = "Requires running services: scheduler, quota-manager, postgres"]
async fn test_scheduler_quota_allocation_on_job_start() -> Result<()> {
    // Setup
    let services = vec![
        ServiceEndpoint::scheduler(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let scheduler_client = SchedulerClient::from_env();
    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-job-start".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 100.0,
        period_days: Some(30),
    };

    let create_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(create_resp.status().is_success());

    let quota: serde_json::Value = create_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Step 2: Submit job
    let job_req = SubmitJobRequest {
        user_id: "test-user-job-start".to_string(),
        priority: 5,
        cpus: 2,
        memory_mb: 4096,
        gpus: Some(2),
        gpu_type: Some("V100".to_string()),
        estimated_duration_secs: Some(3600),
        metadata: None,
    };

    let submit_resp = scheduler_client.submit_job(&job_req).await?;
    assert!(submit_resp.status().is_success());

    let job: serde_json::Value = submit_resp.json().await?;
    let job_id = Uuid::parse_str(job["id"].as_str().unwrap())?;

    // Step 3: Simulate quota allocation when job starts
    // (In real system, scheduler would call quota manager to allocate)
    let create_alloc_req = CreateAllocationRequest {
        quota_id,
        resource_id: format!("job-{}", job_id),
        resource_type: "gpu_hours".to_string(),
        amount: 4.0, // 2 GPUs * 2 hours (estimated)
    };

    let alloc_resp = quota_client.create_allocation(&create_alloc_req).await?;
    assert!(alloc_resp.status().is_success());

    let allocation: serde_json::Value = alloc_resp.json().await?;
    let allocation_id = Uuid::parse_str(allocation["id"].as_str().unwrap())?;

    // Step 4: Verify quota usage
    let stats_resp = quota_client.get_usage_stats(quota_id).await?;
    assert!(stats_resp.status().is_success());

    let stats: serde_json::Value = stats_resp.json().await?;
    assert_eq!(stats["used"].as_f64().unwrap(), 4.0);

    // Cleanup
    let _ = quota_client.release_allocation(allocation_id).await;
    let _ = quota_client.delete_quota(quota_id).await;
    let _ = scheduler_client.cancel_job(job_id).await;

    Ok(())
}

#[tokio::test]
#[ignore = "Requires running services: scheduler, quota-manager, postgres"]
async fn test_scheduler_quota_release_on_job_completion() -> Result<()> {
    // Setup
    let services = vec![
        ServiceEndpoint::scheduler(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "test-user-job-complete".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 100.0,
        period_days: Some(30),
    };

    let create_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(create_resp.status().is_success());

    let quota: serde_json::Value = create_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Step 2: Create an allocation (simulating job start)
    let job_id = Uuid::new_v4();
    let create_alloc_req = CreateAllocationRequest {
        quota_id,
        resource_id: format!("job-{}", job_id),
        resource_type: "gpu_hours".to_string(),
        amount: 8.0,
    };

    let alloc_resp = quota_client.create_allocation(&create_alloc_req).await?;
    assert!(alloc_resp.status().is_success());

    let allocation: serde_json::Value = alloc_resp.json().await?;
    let allocation_id = Uuid::parse_str(allocation["id"].as_str().unwrap())?;

    // Step 3: Verify quota is allocated
    let stats_resp = quota_client.get_usage_stats(quota_id).await?;
    assert!(stats_resp.status().is_success());

    let stats: serde_json::Value = stats_resp.json().await?;
    assert_eq!(stats["used"].as_f64().unwrap(), 8.0);
    assert_eq!(stats["available"].as_f64().unwrap(), 92.0);

    // Step 4: Release allocation (simulating job completion)
    let release_resp = quota_client.release_allocation(allocation_id).await?;
    assert!(release_resp.status().is_success());

    // Step 5: Verify quota is released
    let stats_resp2 = quota_client.get_usage_stats(quota_id).await?;
    assert!(stats_resp2.status().is_success());

    let stats2: serde_json::Value = stats_resp2.json().await?;
    assert_eq!(stats2["used"].as_f64().unwrap(), 0.0);
    assert_eq!(stats2["available"].as_f64().unwrap(), 100.0);

    // Cleanup
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}
