//! Test program to isolate CUDA memory allocation issues
//! This helps identify if the problem is in cudarc allocation or our usage

use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn main() -> Result<()> {
    println!("🧪 CUDA Memory Allocation Test");
    println!("==============================");

    // Initialize CUDA device
    println!("Initializing CUDA device...");
    let ctx = match CudaContext::new(0) {
        Ok(c) => {
            println!("✅ CUDA device initialized successfully");
            c
        }
        Err(e) => {
            println!("❌ Failed to initialize CUDA device: {}", e);
            return Err(e.into());
        }
    };

    // Test small allocations first
    let test_sizes = vec![1, 8, 32, 64, 128, 256, 512, 1024];

    println!("\n🔍 Testing small buffer allocations...");
    for &size in &test_sizes {
        test_allocation(&ctx, size, "small")?;
    }

    // Test the problematic size (1024 * 64 = 65536)
    println!("\n🔍 Testing problematic allocation size...");
    let population_size = 1024;
    let genome_size = 64;
    let total_size = population_size * genome_size;

    println!(
        "Population: {}, Genome: {}, Total: {}",
        population_size, genome_size, total_size
    );

    test_allocation(&ctx, total_size, "population")?;

    // Test multi-buffer allocation (like GpuPopulation does)
    println!("\n🔍 Testing multi-buffer allocation...");
    test_multi_buffer_allocation(&ctx, population_size, genome_size)?;

    println!("\n✅ All memory allocation tests completed successfully!");
    Ok(())
}

fn test_allocation(ctx: &Arc<CudaContext>, size: usize, test_name: &str) -> Result<()> {
    println!(
        "  Testing {} allocation: {} f32 elements ({} bytes)",
        test_name,
        size,
        size * std::mem::size_of::<f32>()
    );

    let stream = ctx.default_stream();

    // Test f32 allocation using stream
    let zeros = vec![0.0f32; size];
    let buffer = match stream.clone_htod(&zeros) {
        Ok(buf) => {
            println!("    ✅ f32 allocation successful");
            buf
        }
        Err(e) => {
            println!("    ❌ f32 allocation failed: {}", e);
            return Err(e.into());
        }
    };

    // Verify buffer size
    println!("    Buffer size verification: {} elements", buffer.len());
    if buffer.len() != size {
        println!(
            "    ⚠️  WARNING: Buffer size mismatch! Expected {}, got {}",
            size,
            buffer.len()
        );
    }

    println!("    ✅ Host-to-device copy successful");

    // Test reading back
    let readback: Vec<f32> = match stream.clone_dtoh(&buffer) {
        Ok(v) => {
            println!("    ✅ Device-to-host copy successful");
            v
        }
        Err(e) => {
            println!("    ❌ Device-to-host copy failed: {}", e);
            return Err(e.into());
        }
    };

    // Verify readback
    if readback.len() != size {
        println!(
            "    ⚠️  WARNING: Readback size mismatch! Expected {}, got {}",
            size,
            readback.len()
        );
    }

    Ok(())
}

fn test_multi_buffer_allocation(
    ctx: &Arc<CudaContext>,
    population_size: usize,
    genome_size: usize,
) -> Result<()> {
    println!("  Multi-buffer allocation test:");
    println!("    Population size: {}", population_size);
    println!("    Genome size: {}", genome_size);

    let stream = ctx.default_stream();
    let total_genome_size = population_size * genome_size;

    // Allocate genomes buffer (f32) using stream
    println!("    Allocating genomes: {} f32 elements", total_genome_size);
    let _genomes = match stream.alloc_zeros::<f32>(total_genome_size) {
        Ok(buf) => {
            println!("      ✅ Genomes buffer allocated");
            buf
        }
        Err(e) => {
            println!("      ❌ Genomes buffer allocation failed: {}", e);
            return Err(e.into());
        }
    };

    // Allocate fitness buffer (f32) using stream
    println!("    Allocating fitness: {} f32 elements", population_size);
    let _fitness = match stream.alloc_zeros::<f32>(population_size) {
        Ok(buf) => {
            println!("      ✅ Fitness buffer allocated");
            buf
        }
        Err(e) => {
            println!("      ❌ Fitness buffer allocation failed: {}", e);
            return Err(e.into());
        }
    };

    // Allocate fitness valid buffer (u8) using stream
    println!(
        "    Allocating fitness valid: {} u8 elements",
        population_size
    );
    let _fitness_valid = match stream.alloc_zeros::<u8>(population_size) {
        Ok(buf) => {
            println!("      ✅ Fitness valid buffer allocated");
            buf
        }
        Err(e) => {
            println!("      ❌ Fitness valid buffer allocation failed: {}", e);
            return Err(e.into());
        }
    };

    println!("    ✅ All buffers allocated successfully");
    Ok(())
}
