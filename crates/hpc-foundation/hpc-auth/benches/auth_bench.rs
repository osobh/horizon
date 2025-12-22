use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hpc_auth::cert::{generate_ca_cert, generate_self_signed_cert, generate_signed_cert, ServiceIdentity};
use hpc_auth::server::{create_server_config, create_server_config_with_client_auth};
use hpc_auth::client::create_client_config;

fn bench_generate_self_signed_cert(c: &mut Criterion) {
    c.bench_function("generate_self_signed_cert", |b| {
        let identity = ServiceIdentity::new("bench-service");
        b.iter(|| {
            generate_self_signed_cert(black_box(&identity)).unwrap()
        });
    });
}

fn bench_generate_ca_cert(c: &mut Criterion) {
    c.bench_function("generate_ca_cert", |b| {
        b.iter(|| {
            generate_ca_cert(black_box("Bench CA")).unwrap()
        });
    });
}

fn bench_generate_signed_cert(c: &mut Criterion) {
    let ca = generate_ca_cert("Bench CA").unwrap();
    let identity = ServiceIdentity::new("bench-service");

    c.bench_function("generate_signed_cert", |b| {
        b.iter(|| {
            generate_signed_cert(black_box(&identity), black_box(&ca)).unwrap()
        });
    });
}

fn bench_create_server_config(c: &mut Criterion) {
    let identity = ServiceIdentity::new("bench-server");
    let cert = generate_self_signed_cert(&identity).unwrap();

    c.bench_function("create_server_config", |b| {
        b.iter(|| {
            create_server_config(black_box(&cert)).unwrap()
        });
    });
}

fn bench_create_server_config_with_client_auth(c: &mut Criterion) {
    let ca = generate_ca_cert("Bench CA").unwrap();
    let identity = ServiceIdentity::new("bench-server");
    let cert = generate_signed_cert(&identity, &ca).unwrap();

    c.bench_function("create_server_config_with_client_auth", |b| {
        b.iter(|| {
            create_server_config_with_client_auth(black_box(&cert), black_box(&ca)).unwrap()
        });
    });
}

fn bench_create_client_config(c: &mut Criterion) {
    let identity = ServiceIdentity::new("bench-client");
    let cert = generate_self_signed_cert(&identity).unwrap();

    c.bench_function("create_client_config", |b| {
        b.iter(|| {
            create_client_config(black_box(&cert)).unwrap()
        });
    });
}

fn bench_cert_validation(c: &mut Criterion) {
    let ca = generate_ca_cert("Bench CA").unwrap();
    let identity = ServiceIdentity::new("bench-service");
    let cert = generate_signed_cert(&identity, &ca).unwrap();

    c.bench_function("cert_validation", |b| {
        b.iter(|| {
            cert.verify_with_ca(black_box(&ca)).unwrap()
        });
    });
}

fn bench_hostname_validation(c: &mut Criterion) {
    let identity = ServiceIdentity::new("bench-service");
    let cert = generate_self_signed_cert(&identity).unwrap();

    c.bench_function("hostname_validation", |b| {
        b.iter(|| {
            cert.validate_hostname(black_box("bench-service")).unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_generate_self_signed_cert,
    bench_generate_ca_cert,
    bench_generate_signed_cert,
    bench_create_server_config,
    bench_create_server_config_with_client_auth,
    bench_create_client_config,
    bench_cert_validation,
    bench_hostname_validation
);

criterion_main!(benches);
