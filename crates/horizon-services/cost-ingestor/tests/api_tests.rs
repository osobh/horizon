use cost_ingestor::models::Provider;
use serde_json::json;

#[test]
fn test_ingest_request_structure() {
    let request = json!({
        "provider": "aws",
        "data": "test-data"
    });

    assert_eq!(request["provider"], "aws");
}

#[test]
fn test_query_params_structure() {
    let params = json!({
        "provider": "gcp",
        "account_id": "test-account",
        "limit": 50,
        "offset": 0
    });

    assert_eq!(params["provider"], "gcp");
    assert_eq!(params["limit"], 50);
}

#[test]
fn test_provider_enum_values() {
    let aws = Provider::Aws;
    let gcp = Provider::Gcp;
    let azure = Provider::Azure;
    let onprem = Provider::OnPrem;

    assert_eq!(aws.to_string(), "aws");
    assert_eq!(gcp.to_string(), "gcp");
    assert_eq!(azure.to_string(), "azure");
    assert_eq!(onprem.to_string(), "onprem");
}

#[test]
fn test_health_response_structure() {
    let response = json!({
        "status": "healthy",
        "service": "cost-ingestor",
        "version": "0.1.0"
    });

    assert_eq!(response["status"], "healthy");
    assert_eq!(response["service"], "cost-ingestor");
}

#[test]
fn test_ingest_response_structure() {
    let response = json!({
        "ingested_count": 5,
        "records": []
    });

    assert_eq!(response["ingested_count"], 5);
    assert!(response["records"].is_array());
}
