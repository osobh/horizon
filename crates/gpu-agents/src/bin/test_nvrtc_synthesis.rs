//! Test NVRTC (Runtime Compilation) for synthesis
//!
//! GREEN phase - tests passing with working runtime compilation

use cudarc::driver::CudaContext;
use gpu_agents::synthesis::nvrtc::{CompilationOptions, KernelTemplate, NvrtcCompiler};

fn main() -> anyhow::Result<()> {
    println!("Testing NVRTC Compilation in Synthesis (GREEN phase)");
    println!("==================================================");

    let ctx = CudaContext::new(0)?;

    // Test 1: Create NVRTC compiler
    println!("\n1. Testing NVRTC compiler creation...");
    let mut compiler = NvrtcCompiler::new(ctx.clone())?;
    println!("✅ NVRTC compiler created");

    // Test 2: Compile simple kernel
    println!("\n2. Testing simple kernel compilation...");
    let kernel_source = r#"
        extern "C" __global__ void simple_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    "#;
    let compiled = compiler.compile_kernel("simple_add", kernel_source)?;
    println!("✅ Kernel compilation succeeded");

    // Test 3: Create kernel template for synthesis
    println!("\n3. Testing kernel template creation...");
    let template = KernelTemplate::new("pattern_match_template");
    println!("✅ Kernel template created");

    // Test 4: Generate synthesis kernel from template
    println!("\n4. Testing synthesis kernel generation...");
    let synthesis_kernel = template.generate_pattern_matcher(32, 1024);
    println!("✅ Synthesis kernel generated");

    // Test 5: Compile with optimization options
    println!("\n5. Testing compilation with optimizations...");
    let options = CompilationOptions::default()
        .with_arch("sm_80")
        .with_opt_level(3)
        .with_debug_info(false);
    let optimized = compiler.compile_with_options(&synthesis_kernel, options)?;
    println!("✅ Optimized compilation succeeded");

    Ok(())
}
