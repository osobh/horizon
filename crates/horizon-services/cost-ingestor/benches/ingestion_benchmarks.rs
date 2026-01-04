use cost_ingestor::{
    ingest::{AwsCurNormalizer, GcpBillingNormalizer},
    models::Provider,
    normalize::{BillingNormalizer, RawBillingData},
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const AWS_SAMPLE: &str = r#"lineItem/UsageAccountId,lineItem/ProductCode,lineItem/ResourceId,lineItem/UsageStartDate,lineItem/UsageEndDate,lineItem/UnblendedCost,lineItem/CurrencyCode,lineItem/LineItemType,product/region
123456789012,AmazonEC2,i-123,2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,1.50,USD,Usage,us-east-1
123456789012,AmazonS3,bucket-1,2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,0.25,USD,Usage,us-west-2
"#;

fn benchmark_aws_cur_parsing(c: &mut Criterion) {
    let normalizer = AwsCurNormalizer::new();
    let raw = RawBillingData {
        provider: Provider::Aws,
        data: serde_json::Value::String(AWS_SAMPLE.to_string()),
    };

    c.bench_function("aws_cur_normalize", |b| {
        b.iter(|| {
            black_box(normalizer.normalize(&raw).unwrap());
        });
    });
}

fn benchmark_gcp_billing_parsing(c: &mut Criterion) {
    let normalizer = GcpBillingNormalizer::new();
    let data = serde_json::json!([{
        "billing_account_id": "012345-6789AB-CDEF01",
        "service": {"id": "services/test", "description": "Compute Engine"},
        "sku": {"id": "0000-0000-0000", "description": "N1 Core"},
        "usage_start_time": "2024-01-01T00:00:00Z",
        "usage_end_time": "2024-01-01T01:00:00Z",
        "project": {"id": "my-project-123", "name": "My Project"},
        "cost": 1.50,
        "currency": "USD",
        "usage": {"amount": 1.0, "unit": "hour"},
        "credits": null
    }]);

    let raw = RawBillingData {
        provider: Provider::Gcp,
        data,
    };

    c.bench_function("gcp_billing_normalize", |b| {
        b.iter(|| {
            black_box(normalizer.normalize(&raw).unwrap());
        });
    });
}

criterion_group!(
    benches,
    benchmark_aws_cur_parsing,
    benchmark_gcp_billing_parsing
);
criterion_main!(benches);
