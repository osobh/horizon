//! Performance benchmarks for policy evaluation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipnet::Ipv4Net;
use std::str::FromStr;
use subnet_manager::models::{
    AssignmentPolicy, MatchOperator, NodeAttribute, NodeType, PolicyRule, PolicyValue, Subnet,
    SubnetPurpose, SubnetStatus,
};
use subnet_manager::policy_engine::{NodeAttributes, PolicyEngine, PolicyEvaluator};
use uuid::Uuid;

fn create_test_subnet(id: Uuid, name: &str) -> Subnet {
    let cidr = Ipv4Net::from_str("10.100.0.0/20").unwrap();
    let mut subnet = Subnet::new(name, cidr, SubnetPurpose::NodeType, 51820);
    subnet.id = id;
    subnet.status = SubnetStatus::Active;
    subnet
}

/// Benchmark single policy evaluation
fn bench_single_policy_evaluation(c: &mut Criterion) {
    let mut engine = PolicyEngine::new();

    let subnet_id = Uuid::new_v4();
    engine.add_subnet(create_test_subnet(subnet_id, "Test Subnet"));

    let policy = AssignmentPolicy::new("Single Policy", subnet_id, 100)
        .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
    engine.add_policy(policy);

    let matching_node = NodeAttributes::new().with_node_type(NodeType::DataCenter);
    let non_matching_node = NodeAttributes::new().with_node_type(NodeType::Laptop);

    c.bench_function("policy_single_match", |b| {
        b.iter(|| black_box(engine.evaluate(&matching_node)))
    });

    c.bench_function("policy_single_no_match", |b| {
        b.iter(|| black_box(engine.evaluate(&non_matching_node)))
    });
}

/// Benchmark multi-rule policy evaluation
fn bench_multi_rule_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_multi_rule");

    for rule_count in [2u32, 5, 10].iter() {
        let mut engine = PolicyEngine::new();

        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, "Test Subnet"));

        // Build a policy with multiple rules
        let mut policy = AssignmentPolicy::new("Multi-Rule Policy", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));

        // Add additional rules
        if *rule_count >= 2 {
            policy = policy.with_rule(PolicyRule::region_equals("us-east-1"));
        }
        if *rule_count >= 3 {
            policy = policy.with_rule(PolicyRule::gpu_memory_gte(24));
        }
        if *rule_count >= 4 {
            policy = policy.with_rule(PolicyRule::new(
                NodeAttribute::CpuCores,
                MatchOperator::GreaterThanOrEqual,
                PolicyValue::Integer(32),
            ));
        }
        if *rule_count >= 5 {
            policy = policy.with_rule(PolicyRule::new(
                NodeAttribute::RamGb,
                MatchOperator::GreaterThanOrEqual,
                PolicyValue::Integer(128),
            ));
        }
        // For 10 rules, add labels
        if *rule_count >= 6 {
            for i in 0..(rule_count - 5) {
                policy = policy.with_rule(PolicyRule::has_label(format!("key{}=value{}", i, i)));
            }
        }

        engine.add_policy(policy);

        // Create a matching node with all attributes
        let mut node = NodeAttributes::new()
            .with_node_type(NodeType::DataCenter)
            .with_region("us-east-1")
            .with_gpu(4, 80, "A100")
            .with_labels(vec![
                "key0=value0".to_string(),
                "key1=value1".to_string(),
                "key2=value2".to_string(),
                "key3=value3".to_string(),
                "key4=value4".to_string(),
            ]);
        node.cpu_cores = Some(128);
        node.ram_gb = Some(1024);

        group.bench_with_input(BenchmarkId::new("rules", rule_count), rule_count, |b, _| {
            b.iter(|| black_box(engine.evaluate(&node)))
        });
    }

    group.finish();
}

/// Benchmark with multiple policies (priority ordering)
fn bench_multiple_policies(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_multiple_policies");

    for policy_count in [5u32, 20, 100].iter() {
        let mut engine = PolicyEngine::new();

        // Create subnets and policies
        for i in 0..*policy_count {
            let subnet_id = Uuid::new_v4();
            engine.add_subnet(create_test_subnet(subnet_id, &format!("Subnet {}", i)));

            // Mix of node type policies
            let node_type = match i % 4 {
                0 => NodeType::DataCenter,
                1 => NodeType::Workstation,
                2 => NodeType::Laptop,
                _ => NodeType::Edge,
            };

            let policy = AssignmentPolicy::new(&format!("Policy {}", i), subnet_id, i as i32)
                .with_rule(PolicyRule::node_type_equals(node_type));
            engine.add_policy(policy);
        }

        // Node that matches the last (highest priority) policy
        let node = NodeAttributes::new().with_node_type(match (policy_count - 1) % 4 {
            0 => NodeType::DataCenter,
            1 => NodeType::Workstation,
            2 => NodeType::Laptop,
            _ => NodeType::Edge,
        });

        group.throughput(Throughput::Elements(*policy_count as u64));
        group.bench_with_input(
            BenchmarkId::new("policies", policy_count),
            policy_count,
            |b, _| b.iter(|| black_box(engine.evaluate(&node))),
        );
    }

    group.finish();
}

/// Benchmark worst case: node matches nothing
fn bench_no_match_all_policies(c: &mut Criterion) {
    let mut engine = PolicyEngine::new();

    // Create 100 policies, none will match our test node
    for i in 0..100 {
        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, &format!("Subnet {}", i)));

        // All policies require datacenter
        let policy = AssignmentPolicy::new(&format!("Policy {}", i), subnet_id, i)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
        engine.add_policy(policy);
    }

    // Laptop won't match any policy
    let node = NodeAttributes::new().with_node_type(NodeType::Laptop);

    c.bench_function("policy_no_match_worst_case", |b| {
        b.iter(|| black_box(engine.evaluate(&node)))
    });
}

/// Benchmark dry run evaluation
fn bench_dry_run(c: &mut Criterion) {
    let mut engine = PolicyEngine::new();

    // Create 20 policies with varying rules
    for i in 0..20 {
        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, &format!("Subnet {}", i)));

        let node_type = match i % 4 {
            0 => NodeType::DataCenter,
            1 => NodeType::Workstation,
            2 => NodeType::Laptop,
            _ => NodeType::Edge,
        };

        let policy = AssignmentPolicy::new(&format!("Policy {}", i), subnet_id, i)
            .with_rule(PolicyRule::node_type_equals(node_type))
            .with_rule(PolicyRule::region_equals(&format!("region-{}", i % 5)));
        engine.add_policy(policy);
    }

    let node = NodeAttributes::new()
        .with_node_type(NodeType::DataCenter)
        .with_region("region-0");

    c.bench_function("policy_dry_run", |b| {
        b.iter(|| black_box(engine.evaluate_dry_run(&node)))
    });
}

/// Benchmark batch evaluation
fn bench_batch_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_batch_evaluation");

    let mut engine = PolicyEngine::new();

    // Create 10 policies for different node types
    for i in 0..10 {
        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, &format!("Subnet {}", i)));

        let node_type = match i % 4 {
            0 => NodeType::DataCenter,
            1 => NodeType::Workstation,
            2 => NodeType::Laptop,
            _ => NodeType::Edge,
        };

        let policy = AssignmentPolicy::new(&format!("Policy {}", i), subnet_id, i)
            .with_rule(PolicyRule::node_type_equals(node_type));
        engine.add_policy(policy);
    }

    for batch_size in [10u64, 100, 1000].iter() {
        // Create batch of nodes with varying types
        let nodes: Vec<(Option<Uuid>, NodeAttributes)> = (0..*batch_size)
            .map(|i| {
                let node_type = match i % 4 {
                    0 => NodeType::DataCenter,
                    1 => NodeType::Workstation,
                    2 => NodeType::Laptop,
                    _ => NodeType::Edge,
                };
                (
                    Some(Uuid::new_v4()),
                    NodeAttributes::new().with_node_type(node_type),
                )
            })
            .collect();

        group.throughput(Throughput::Elements(*batch_size));
        group.bench_with_input(BenchmarkId::new("nodes", batch_size), batch_size, |b, _| {
            b.iter(|| black_box(engine.evaluate_batch(&nodes)))
        });
    }

    group.finish();
}

/// Benchmark policy add/remove (engine modification)
fn bench_policy_modification(c: &mut Criterion) {
    c.bench_function("policy_add_remove", |b| {
        b.iter_batched(
            || {
                let mut engine = PolicyEngine::new();
                let subnet_id = Uuid::new_v4();
                engine.add_subnet(create_test_subnet(subnet_id, "Test"));
                (engine, subnet_id)
            },
            |(mut engine, subnet_id)| {
                // Add 50 policies
                let mut policy_ids = Vec::new();
                for i in 0..50 {
                    let policy = AssignmentPolicy::new(&format!("Policy {}", i), subnet_id, i);
                    let id = policy.id;
                    engine.add_policy(policy);
                    policy_ids.push(id);
                }
                // Remove all policies
                for id in policy_ids {
                    engine.remove_policy(id);
                }
                black_box(engine.policies().len())
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark matching_policies helper
fn bench_matching_policies(c: &mut Criterion) {
    let mut engine = PolicyEngine::new();

    // Create 50 policies with overlapping rules
    for i in 0..50 {
        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, &format!("Subnet {}", i)));

        let mut policy = AssignmentPolicy::new(&format!("Policy {}", i), subnet_id, i);

        // First 25 policies match DataCenter
        if i < 25 {
            policy = policy.with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
        }
        // Policies 10-40 also require us-east-1
        if i >= 10 && i < 40 {
            policy = policy.with_rule(PolicyRule::region_equals("us-east-1"));
        }

        engine.add_policy(policy);
    }

    let node = NodeAttributes::new()
        .with_node_type(NodeType::DataCenter)
        .with_region("us-east-1");

    c.bench_function("policy_matching_policies", |b| {
        b.iter(|| black_box(engine.matching_policies(&node)))
    });
}

/// Benchmark label matching
fn bench_label_matching(c: &mut Criterion) {
    let mut engine = PolicyEngine::new();

    let subnet_id = Uuid::new_v4();
    engine.add_subnet(create_test_subnet(subnet_id, "Label Test"));

    // Policy with multiple label rules
    let policy = AssignmentPolicy::new("Label Policy", subnet_id, 100)
        .with_rule(PolicyRule::has_label("environment=production"))
        .with_rule(PolicyRule::has_label("tier=frontend"))
        .with_rule(PolicyRule::has_label("team=platform"));
    engine.add_policy(policy);

    // Node with matching labels
    let node = NodeAttributes::new().with_labels(vec![
        "environment=production".to_string(),
        "tier=frontend".to_string(),
        "team=platform".to_string(),
        "version=1.0".to_string(),
        "region=us-east-1".to_string(),
    ]);

    c.bench_function("policy_label_matching", |b| {
        b.iter(|| black_box(engine.evaluate(&node)))
    });
}

/// Benchmark tenant ID matching
fn bench_tenant_matching(c: &mut Criterion) {
    let mut engine = PolicyEngine::new();

    // Create 50 tenant subnets
    for i in 0..50 {
        let subnet_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, &format!("Tenant {}", i)));

        let policy = AssignmentPolicy::new(&format!("Tenant Policy {}", i), subnet_id, i)
            .with_rule(PolicyRule::tenant_equals(tenant_id));
        engine.add_policy(policy);
    }

    // Add one more policy that will match
    let target_subnet_id = Uuid::new_v4();
    let target_tenant_id = Uuid::new_v4();
    engine.add_subnet(create_test_subnet(target_subnet_id, "Target Tenant"));
    let policy = AssignmentPolicy::new("Target Policy", target_subnet_id, 1000)
        .with_rule(PolicyRule::tenant_equals(target_tenant_id));
    engine.add_policy(policy);

    let node = NodeAttributes::new().with_tenant(target_tenant_id);

    c.bench_function("policy_tenant_matching", |b| {
        b.iter(|| black_box(engine.evaluate(&node)))
    });
}

criterion_group!(
    benches,
    bench_single_policy_evaluation,
    bench_multi_rule_policy,
    bench_multiple_policies,
    bench_no_match_all_policies,
    bench_dry_run,
    bench_batch_evaluation,
    bench_policy_modification,
    bench_matching_policies,
    bench_label_matching,
    bench_tenant_matching,
);

criterion_main!(benches);
