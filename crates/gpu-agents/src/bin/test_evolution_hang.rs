//! Test runner for evolution hang isolation tests
//! Run with: cargo run --bin test-evolution-hang -- <test_name>

use gpu_agents::evolution::hang_isolation_tests::run_hang_test;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    println!("🧪 Evolution Hang Isolation Test Runner");
    println!("=====================================");

    if args.len() < 2 {
        println!("Usage: {} <test_name>", args[0]);
        println!("Available tests:");
        println!("  creation      - Test evolution engine creation");
        println!("  initialization - Test random population initialization");
        println!("  fitness       - Test fitness evaluation (expected to hang)");
        println!("  generation    - Test single evolution generation (expected to hang)");
        return;
    }

    let test_name = &args[1];
    println!("Running test: {}\n", test_name);

    match run_hang_test(test_name) {
        Ok(_) => {
            println!("\n✅ Test completed successfully");
        }
        Err(e) => {
            println!("\n❌ Test failed: {}", e);
            std::process::exit(1);
        }
    }
}
