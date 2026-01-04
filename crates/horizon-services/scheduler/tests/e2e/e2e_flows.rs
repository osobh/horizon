// Full end-to-end flow tests
//
// These tests verify complete request flows through the entire system:
// User → API Gateway (auth) → API Gateway (rate limit) →
// API Gateway (policy check via governor) →
// API Gateway (quota check via quota-manager) →
// Scheduler (job submission) → Response back to user
//
// Tests assume ALL services are running on localhost.

use anyhow::Result;
use serde_json::json;
use std::time::Duration;
use uuid::Uuid;

use crate::e2e::helpers::*;

// ============================================================================
// Happy Path: Complete Job Submission Flow
// ============================================================================

#[tokio::test]
#[ignore = "Requires ALL services: api-gateway, governor, quota-manager, scheduler, postgres"]
async fn test_complete_job_submission_flow() -> Result<()> {
    // This test verifies the complete happy path:
    // 1. Create policy in Governor (allow job submission)
    // 2. Create quota in Quota Manager (sufficient quota)
    // 3. Submit job through API Gateway
    // 4. Verify job is created in Scheduler
    // 5. Allocate quota
    // 6. Complete job and release quota

    // Setup: Verify all services are available
    let services = vec![
        ServiceEndpoint::api_gateway(),
        ServiceEndpoint::governor(),
        ServiceEndpoint::quota_manager(),
        ServiceEndpoint::scheduler(),
    ];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let governor_client = GovernorClient::from_env();
    let quota_client = QuotaManagerClient::from_env();
    let scheduler_client = SchedulerClient::from_env();

    // Step 1: Create allow policy
    let policy_content = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: e2e-allow-policy
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
      actions: [submit, cancel, view]
      conditions: []
"#;

    let create_policy_req = CreatePolicyRequest {
        name: "e2e-allow-policy".to_string(),
        content: policy_content.to_string(),
        description: Some("E2E test policy".to_string()),
    };

    let policy_resp = governor_client.create_policy(&create_policy_req).await?;
    assert!(
        policy_resp.status().is_success(),
        "Failed to create policy: {:?}",
        policy_resp.text().await?
    );

    // Step 2: Create quota for user
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "e2e-test-user".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 500.0,
        period_days: Some(30),
    };

    let quota_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(
        quota_resp.status().is_success(),
        "Failed to create quota: {:?}",
        quota_resp.text().await?
    );

    let quota: serde_json::Value = quota_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Step 3: Check quota availability through Gateway
    let check_req = AllocationCheckRequest {
        entity_type: "user".to_string(),
        entity_id: "e2e-test-user".to_string(),
        resource_type: "gpu_hours".to_string(),
        amount: 32.0, // 8 GPUs * 4 hours
    };

    let check_resp = gateway_client.check_quota(&check_req).await?;
    assert!(check_resp.status().is_success());

    let check_result: AllocationCheckResponse = check_resp.json().await?;
    assert!(check_result.allowed, "Quota check should pass");

    // Step 4: Evaluate policy through Gateway
    let eval_req = EvaluateRequest {
        principal: Principal {
            id: "e2e-test-user".to_string(),
            principal_type: "user".to_string(),
            attributes: Some(json!({"team": "engineering"})),
        },
        action: "submit".to_string(),
        resource: Resource {
            id: "job-e2e-test".to_string(),
            resource_type: "job".to_string(),
            attributes: Some(json!({"gpus": 8, "priority": 5})),
        },
    };

    let eval_resp = gateway_client.evaluate_policy(&eval_req).await?;
    assert!(eval_resp.status().is_success());

    let eval_result: EvaluateResponse = eval_resp.json().await?;
    assert_eq!(eval_result.decision, "allow", "Policy should allow job");

    // Step 5: Submit job through Gateway
    let job_req = SubmitJobRequest {
        user_id: "e2e-test-user".to_string(),
        priority: 5,
        cpus: 32,
        memory_mb: 65536,
        gpus: Some(8),
        gpu_type: Some("A100".to_string()),
        estimated_duration_secs: Some(14400), // 4 hours
        metadata: Some(json!({"test": "e2e-complete-flow"})),
    };

    let submit_resp = gateway_client.submit_job(&job_req).await?;
    assert!(
        submit_resp.status().is_success(),
        "Job submission should succeed: {:?}",
        submit_resp.text().await?
    );

    let job: serde_json::Value = submit_resp.json().await?;
    let job_id = Uuid::parse_str(job["id"].as_str().unwrap())?;

    // Step 6: Verify job exists in Scheduler
    let get_job_resp = scheduler_client.get_job(job_id).await?;
    assert!(get_job_resp.status().is_success());

    let job_data: serde_json::Value = get_job_resp.json().await?;
    assert_eq!(job_data["user_id"], "e2e-test-user");
    assert_eq!(job_data["cpus"], 32);
    assert_eq!(job_data["gpus"], 8);

    // Step 7: Allocate quota for the job
    let alloc_req = CreateAllocationRequest {
        quota_id,
        resource_id: format!("job-{}", job_id),
        resource_type: "gpu_hours".to_string(),
        amount: 32.0, // 8 GPUs * 4 hours
    };

    let alloc_resp = quota_client.create_allocation(&alloc_req).await?;
    assert!(alloc_resp.status().is_success());

    let allocation: serde_json::Value = alloc_resp.json().await?;
    let allocation_id = Uuid::parse_str(allocation["id"].as_str().unwrap())?;

    // Step 8: Verify quota usage
    let stats_resp = quota_client.get_usage_stats(quota_id).await?;
    assert!(stats_resp.status().is_success());

    let stats: serde_json::Value = stats_resp.json().await?;
    assert_eq!(stats["used"].as_f64().unwrap(), 32.0);
    assert_eq!(stats["available"].as_f64().unwrap(), 468.0);

    // Step 9: Complete job (cancel as proxy for completion)
    let cancel_resp = scheduler_client.cancel_job(job_id).await?;
    assert!(cancel_resp.status().is_success());

    // Step 10: Release quota
    let release_resp = quota_client.release_allocation(allocation_id).await?;
    assert!(release_resp.status().is_success());

    // Step 11: Verify quota released
    let final_stats_resp = quota_client.get_usage_stats(quota_id).await?;
    assert!(final_stats_resp.status().is_success());

    let final_stats: serde_json::Value = final_stats_resp.json().await?;
    assert_eq!(final_stats["used"].as_f64().unwrap(), 0.0);
    assert_eq!(final_stats["available"].as_f64().unwrap(), 500.0);

    // Cleanup
    let _ = governor_client.delete_policy("e2e-allow-policy").await;
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}

// ============================================================================
// Policy Denial Flow
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: api-gateway, governor, postgres"]
async fn test_policy_denial_flow() -> Result<()> {
    // This test verifies that jobs are denied when policy doesn't allow them

    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::governor()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let governor_client = GovernorClient::from_env();

    // Step 1: Create deny policy for high GPU requests
    let policy_content = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: e2e-deny-high-gpu
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
          value: 16
"#;

    let create_policy_req = CreatePolicyRequest {
        name: "e2e-deny-high-gpu".to_string(),
        content: policy_content.to_string(),
        description: Some("Deny high GPU requests".to_string()),
    };

    let policy_resp = governor_client.create_policy(&create_policy_req).await?;
    assert!(policy_resp.status().is_success());

    // Step 2: Try to submit job with too many GPUs through Gateway
    // First evaluate the policy
    let eval_req = EvaluateRequest {
        principal: Principal {
            id: "policy-test-user".to_string(),
            principal_type: "user".to_string(),
            attributes: None,
        },
        action: "submit".to_string(),
        resource: Resource {
            id: "high-gpu-job".to_string(),
            resource_type: "job".to_string(),
            attributes: Some(json!({"gpus": 32})), // More than 16
        },
    };

    let eval_resp = gateway_client.evaluate_policy(&eval_req).await?;
    assert!(eval_resp.status().is_success());

    let eval_result: EvaluateResponse = eval_resp.json().await?;
    assert_eq!(
        eval_result.decision, "deny",
        "Policy should deny high GPU request"
    );

    // Cleanup
    let _ = governor_client.delete_policy("e2e-deny-high-gpu").await;

    Ok(())
}

// ============================================================================
// Quota Exceeded Flow
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: api-gateway, quota-manager, postgres"]
async fn test_quota_exceeded_flow() -> Result<()> {
    // This test verifies that jobs are denied when quota is insufficient

    let services = vec![
        ServiceEndpoint::api_gateway(),
        ServiceEndpoint::quota_manager(),
    ];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();
    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create small quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "quota-test-user".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 10.0, // Small quota
        period_days: Some(30),
    };

    let quota_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(quota_resp.status().is_success());

    let quota: serde_json::Value = quota_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Step 2: Try to check allocation for large request
    let check_req = AllocationCheckRequest {
        entity_type: "user".to_string(),
        entity_id: "quota-test-user".to_string(),
        resource_type: "gpu_hours".to_string(),
        amount: 100.0, // Much more than limit
    };

    let check_resp = gateway_client.check_quota(&check_req).await?;
    assert!(check_resp.status().is_success());

    let check_result: AllocationCheckResponse = check_resp.json().await?;
    assert!(!check_result.allowed, "Quota check should fail");
    assert_eq!(check_result.available, 10.0);
    assert_eq!(check_result.requested, 100.0);

    // Cleanup
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}

// ============================================================================
// Concurrent Request Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: api-gateway, scheduler, postgres"]
async fn test_concurrent_job_submissions() -> Result<()> {
    // This test verifies that multiple concurrent job submissions work correctly

    let services = vec![ServiceEndpoint::api_gateway(), ServiceEndpoint::scheduler()];
    require_services(&services).await?;

    let gateway_client = ApiGatewayClient::from_env();

    // Submit 10 jobs concurrently
    let mut handles = vec![];

    for i in 0..10 {
        let client = gateway_client.clone();
        let handle = tokio::spawn(async move {
            let job_req = SubmitJobRequest {
                user_id: format!("concurrent-user-{}", i),
                priority: 5,
                cpus: 2,
                memory_mb: 4096,
                gpus: Some(1),
                gpu_type: Some("T4".to_string()),
                estimated_duration_secs: Some(3600),
                metadata: Some(json!({"concurrent_test": i})),
            };

            client.submit_job(&job_req).await
        });
        handles.push(handle);
    }

    // Wait for all submissions
    let results = futures::future::join_all(handles).await;

    // Verify all succeeded
    let mut job_ids = vec![];
    for result in results {
        let response = result.unwrap().unwrap();
        assert!(
            response.status().is_success(),
            "Concurrent job submission failed"
        );

        let job: serde_json::Value = response.json().await.unwrap();
        let job_id = Uuid::parse_str(job["id"].as_str().unwrap()).unwrap();
        job_ids.push(job_id);
    }

    // Verify we got 10 unique job IDs
    assert_eq!(job_ids.len(), 10);
    let unique_ids: std::collections::HashSet<_> = job_ids.iter().collect();
    assert_eq!(unique_ids.len(), 10, "All job IDs should be unique");

    Ok(())
}

#[tokio::test]
#[ignore = "Requires services: quota-manager, postgres"]
async fn test_concurrent_quota_allocations() -> Result<()> {
    // This test verifies quota allocation under concurrent load

    let services = vec![ServiceEndpoint::quota_manager()];
    require_services(&services).await?;

    let quota_client = QuotaManagerClient::from_env();

    // Step 1: Create quota
    let create_quota_req = CreateQuotaRequest {
        entity_type: "user".to_string(),
        entity_id: "concurrent-quota-user".to_string(),
        parent_id: None,
        resource_type: "gpu_hours".to_string(),
        limit_value: 1000.0,
        period_days: Some(30),
    };

    let quota_resp = quota_client.create_quota(&create_quota_req).await?;
    assert!(quota_resp.status().is_success());

    let quota: serde_json::Value = quota_resp.json().await?;
    let quota_id = Uuid::parse_str(quota["id"].as_str().unwrap())?;

    // Step 2: Create 10 concurrent allocations
    let mut handles = vec![];

    for i in 0..10 {
        let client = quota_client.clone();
        let qid = quota_id;
        let handle = tokio::spawn(async move {
            let alloc_req = CreateAllocationRequest {
                quota_id: qid,
                resource_id: format!("concurrent-job-{}", i),
                resource_type: "gpu_hours".to_string(),
                amount: 10.0,
            };

            client.create_allocation(&alloc_req).await
        });
        handles.push(handle);
    }

    // Wait for all allocations
    let results = futures::future::join_all(handles).await;

    // Verify all succeeded
    let mut allocation_ids = vec![];
    for result in results {
        let response = result.unwrap().unwrap();
        assert!(
            response.status().is_success(),
            "Concurrent allocation failed"
        );

        let allocation: serde_json::Value = response.json().await.unwrap();
        let alloc_id = Uuid::parse_str(allocation["id"].as_str().unwrap()).unwrap();
        allocation_ids.push(alloc_id);
    }

    // Step 3: Verify total usage (should be 10 * 10 = 100)
    let stats_resp = quota_client.get_usage_stats(quota_id).await?;
    assert!(stats_resp.status().is_success());

    let stats: serde_json::Value = stats_resp.json().await?;
    assert_eq!(stats["used"].as_f64().unwrap(), 100.0);

    // Cleanup: Release all allocations
    for alloc_id in allocation_ids {
        let _ = quota_client.release_allocation(alloc_id).await;
    }
    let _ = quota_client.delete_quota(quota_id).await;

    Ok(())
}

// ============================================================================
// Service Health and Availability Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires services: api-gateway, governor, quota-manager, scheduler"]
async fn test_all_services_health() -> Result<()> {
    // Verify all services are healthy

    let services = vec![
        ServiceEndpoint::api_gateway(),
        ServiceEndpoint::governor(),
        ServiceEndpoint::quota_manager(),
        ServiceEndpoint::scheduler(),
    ];

    // Check each service
    for service in &services {
        let is_healthy = service.is_healthy().await;
        assert!(
            is_healthy,
            "Service {} should be healthy at {}",
            service.name, service.base_url
        );
    }

    // Wait for all services with timeout
    let result = wait_for_services(&services, Duration::from_secs(10)).await;
    assert!(
        result.is_ok(),
        "All services should be available: {:?}",
        result
    );

    Ok(())
}

#[tokio::test]
async fn test_service_endpoint_configuration() {
    // Test that service endpoints can be configured via environment

    std::env::set_var("SCHEDULER_PORT", "9080");
    let scheduler = ServiceEndpoint::scheduler();
    assert_eq!(scheduler.base_url, "http://localhost:9080");
    std::env::remove_var("SCHEDULER_PORT");

    std::env::set_var("GOVERNOR_PORT", "9081");
    let governor = ServiceEndpoint::governor();
    assert_eq!(governor.base_url, "http://localhost:9081");
    std::env::remove_var("GOVERNOR_PORT");

    std::env::set_var("QUOTA_MANAGER_PORT", "9082");
    let quota_manager = ServiceEndpoint::quota_manager();
    assert_eq!(quota_manager.base_url, "http://localhost:9082");
    std::env::remove_var("QUOTA_MANAGER_PORT");

    std::env::set_var("API_GATEWAY_PORT", "9000");
    let gateway = ServiceEndpoint::api_gateway();
    assert_eq!(gateway.base_url, "http://localhost:9000");
    std::env::remove_var("API_GATEWAY_PORT");
}
