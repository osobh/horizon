//! Performance benchmarks for IP and subnet allocators

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipnet::Ipv4Net;
use std::net::Ipv4Addr;
use std::str::FromStr;
use subnet_manager::allocator::{CidrAllocator, IpAllocator, SubnetAllocator, SubnetIpAllocator};

/// Benchmark sequential IP allocation
fn bench_sequential_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ip_allocation_sequential");

    for prefix in [24u8, 20, 16].iter() {
        let cidr = Ipv4Net::from_str(&format!("10.100.0.0/{}", prefix)).unwrap();
        let capacity = 2u32.pow(32 - *prefix as u32) - 3; // Minus network, broadcast, gateway

        group.throughput(Throughput::Elements(capacity as u64));
        group.bench_with_input(BenchmarkId::new("prefix", prefix), prefix, |b, _| {
            b.iter(|| {
                let mut allocator = SubnetIpAllocator::new(cidr);
                let mut count = 0u32;
                while allocator.allocate().is_some() {
                    count += 1;
                }
                black_box(count)
            });
        });
    }

    group.finish();
}

/// Benchmark single IP allocation (first allocation)
fn bench_single_allocation(c: &mut Criterion) {
    let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();

    c.bench_function("ip_allocation_single", |b| {
        b.iter(|| {
            let mut allocator = SubnetIpAllocator::new(cidr);
            black_box(allocator.allocate())
        });
    });
}

/// Benchmark IP allocation with fragmentation
fn bench_fragmented_allocation(c: &mut Criterion) {
    let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();

    c.bench_function("ip_allocation_fragmented", |b| {
        b.iter_batched(
            || {
                // Setup: allocate half, release every other one to create fragmentation
                let mut allocator = SubnetIpAllocator::new(cidr);
                let mut allocated = Vec::new();
                for _ in 0..126 {
                    if let Some(ip) = allocator.allocate() {
                        allocated.push(ip);
                    }
                }
                // Release every other IP
                for (i, ip) in allocated.iter().enumerate() {
                    if i % 2 == 0 {
                        allocator.release(*ip);
                    }
                }
                allocator
            },
            |mut allocator| {
                // Benchmark: allocate from fragmented pool
                let mut count = 0;
                while allocator.allocate().is_some() {
                    count += 1;
                }
                black_box(count)
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark IP release
fn bench_ip_release(c: &mut Criterion) {
    let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();

    c.bench_function("ip_release", |b| {
        b.iter_batched(
            || {
                // Setup: allocate all IPs
                let mut allocator = SubnetIpAllocator::new(cidr);
                let mut allocated = Vec::new();
                while let Some(ip) = allocator.allocate() {
                    allocated.push(ip);
                }
                (allocator, allocated)
            },
            |(mut allocator, allocated)| {
                // Benchmark: release all IPs
                for ip in allocated {
                    allocator.release(ip);
                }
                black_box(allocator.available_count())
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark IP reserve (specific IP)
fn bench_ip_reserve(c: &mut Criterion) {
    let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();

    c.bench_function("ip_reserve", |b| {
        b.iter_batched(
            || SubnetIpAllocator::new(cidr),
            |mut allocator| {
                // Reserve specific IPs
                for i in 10..=100u8 {
                    let _ = allocator.reserve(Ipv4Addr::new(10, 100, 0, i));
                }
                black_box(allocator.available_count())
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark available count calculation
fn bench_available_count(c: &mut Criterion) {
    let cidr = Ipv4Net::from_str("10.100.0.0/16").unwrap();

    c.bench_function("ip_available_count", |b| {
        b.iter_batched(
            || {
                let mut allocator = SubnetIpAllocator::new(cidr);
                // Allocate ~32k IPs
                for _ in 0..32768 {
                    let _ = allocator.allocate();
                }
                allocator
            },
            |allocator| black_box(allocator.available_count()),
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark subnet CIDR allocation
fn bench_subnet_cidr_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("subnet_cidr_allocation");

    let master_cidr = Ipv4Net::from_str("10.0.0.0/8").unwrap();

    for prefix in [20u8, 24, 28].iter() {
        let prefix = *prefix;
        let subnets_count = 2u32.pow((prefix - 8) as u32) / 4; // Allocate 25% of capacity

        group.bench_with_input(BenchmarkId::new("prefix", prefix), &prefix, |b, &prefix| {
            b.iter(|| {
                let mut allocator = SubnetAllocator::new(master_cidr);
                for _ in 0..subnets_count {
                    let _ = allocator.allocate(prefix);
                }
                black_box(allocator.allocated_blocks().len())
            });
        });
    }

    group.finish();
}

/// Benchmark subnet allocation with mixed sizes
fn bench_mixed_subnet_allocation(c: &mut Criterion) {
    let master_cidr = Ipv4Net::from_str("10.0.0.0/8").unwrap();

    c.bench_function("subnet_allocation_mixed", |b| {
        b.iter(|| {
            let mut allocator = SubnetAllocator::new(master_cidr);
            // Mix of different subnet sizes
            for _ in 0..100 {
                let _ = allocator.allocate(20);
                let _ = allocator.allocate(24);
                let _ = allocator.allocate(28);
            }
            black_box(allocator.allocated_blocks().len())
        });
    });
}

/// Benchmark with_allocated restoration
fn bench_with_allocated_restoration(c: &mut Criterion) {
    let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();

    c.bench_function("with_allocated_restoration", |b| {
        // Pre-generate IPs to restore
        let allocated: Vec<Ipv4Addr> = (10..=200u8)
            .map(|i| Ipv4Addr::new(10, 100, 0, i))
            .collect();

        b.iter(|| {
            let allocator = SubnetIpAllocator::new(cidr).with_allocated(&allocated);
            black_box(allocator.available_count())
        });
    });
}

/// Benchmark is_allocated check
fn bench_is_allocated_check(c: &mut Criterion) {
    let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();

    c.bench_function("is_allocated_check", |b| {
        // Setup allocator with some IPs
        let mut allocator = SubnetIpAllocator::new(cidr);
        for _ in 0..128 {
            let _ = allocator.allocate();
        }

        // Test IPs
        let test_ips: Vec<Ipv4Addr> = (2..=254u8)
            .map(|i| Ipv4Addr::new(10, 100, 0, i))
            .collect();

        b.iter(|| {
            let mut allocated = 0;
            for ip in &test_ips {
                if allocator.is_allocated(*ip) {
                    allocated += 1;
                }
            }
            black_box(allocated)
        });
    });
}

criterion_group!(
    benches,
    bench_sequential_allocation,
    bench_single_allocation,
    bench_fragmented_allocation,
    bench_ip_release,
    bench_ip_reserve,
    bench_available_count,
    bench_subnet_cidr_allocation,
    bench_mixed_subnet_allocation,
    bench_with_allocated_restoration,
    bench_is_allocated_check,
);

criterion_main!(benches);
