use criterion::{black_box, criterion_group, criterion_main, Criterion};
use inventory_service::{
    api::models::{CreateAssetRequest, ListAssetsQuery},
    models::{AssetType, ProviderType},
    repository::AssetRepository,
};
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::time::Duration;
use tokio::runtime::Runtime;

async fn setup_bench_pool() -> PgPool {
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/inventory_bench".to_string());

    PgPoolOptions::new()
        .max_connections(20)
        .acquire_timeout(Duration::from_secs(3))
        .connect(&database_url)
        .await
        .expect("Failed to connect to benchmark database")
}

async fn seed_assets(pool: &PgPool, count: usize) {
    let repo = AssetRepository::new(pool.clone());

    for i in 0..count {
        let req = CreateAssetRequest {
            asset_type: AssetType::Gpu,
            provider: ProviderType::Baremetal,
            provider_id: Some(format!("gpu-{}", i)),
            parent_id: None,
            hostname: Some(format!("gpu-node-{:03}", i)),
            status: None,
            location: Some("us-west-1a".to_string()),
            metadata: Some(serde_json::json!({
                "gpu_model": "H100",
                "gpu_memory_gb": 80
            })),
        };

        repo.create(req, "benchmark".to_string()).await.ok();
    }
}

fn benchmark_list_assets(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let pool = rt.block_on(async {
        let pool = setup_bench_pool().await;
        sqlx::query("TRUNCATE assets CASCADE")
            .execute(&pool)
            .await
            .ok();
        seed_assets(&pool, 100).await;
        pool
    });

    let repo = AssetRepository::new(pool.clone());

    c.bench_function("list_assets_paginated", |b| {
        b.to_async(&rt).iter(|| async {
            let query = ListAssetsQuery {
                asset_type: Some(AssetType::Gpu),
                status: None,
                provider: None,
                location: None,
                page: Some(1),
                page_size: Some(50),
                sort: None,
                order: None,
            };

            black_box(repo.list(&query).await.unwrap())
        });
    });
}

fn benchmark_create_asset(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let pool = rt.block_on(async {
        let pool = setup_bench_pool().await;
        sqlx::query("TRUNCATE assets CASCADE")
            .execute(&pool)
            .await
            .ok();
        pool
    });

    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    c.bench_function("create_asset", |b| {
        let counter_clone = counter.clone();
        let pool_clone = pool.clone();
        b.to_async(&rt).iter(|| {
            let counter = counter_clone.clone();
            let repo = AssetRepository::new(pool_clone.clone());
            async move {
                let count = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let req = CreateAssetRequest {
                    asset_type: AssetType::Gpu,
                    provider: ProviderType::Baremetal,
                    provider_id: Some(format!("benchmark-gpu-{}", count)),
                    parent_id: None,
                    hostname: Some(format!("benchmark-node-{}", count)),
                    status: None,
                    location: Some("us-west-1a".to_string()),
                    metadata: Some(serde_json::json!({"gpu_model": "H100"})),
                };

                black_box(repo.create(req, "benchmark".to_string()).await.unwrap())
            }
        });
    });
}

fn benchmark_get_asset_by_id(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let (pool, asset_id) = rt.block_on(async {
        let pool = setup_bench_pool().await;
        sqlx::query("TRUNCATE assets CASCADE")
            .execute(&pool)
            .await
            .ok();

        let repo = AssetRepository::new(pool.clone());
        let req = CreateAssetRequest {
            asset_type: AssetType::Gpu,
            provider: ProviderType::Baremetal,
            provider_id: Some("benchmark-get".to_string()),
            parent_id: None,
            hostname: Some("benchmark-get-node".to_string()),
            status: None,
            location: None,
            metadata: None,
        };

        let asset = repo.create(req, "benchmark".to_string()).await.unwrap();
        (pool, asset.id)
    });

    let repo = AssetRepository::new(pool.clone());

    c.bench_function("get_asset_by_id", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(repo.get_by_id(asset_id).await.unwrap())
        });
    });
}

criterion_group!(
    benches,
    benchmark_list_assets,
    benchmark_create_asset,
    benchmark_get_asset_by_id
);
criterion_main!(benches);
