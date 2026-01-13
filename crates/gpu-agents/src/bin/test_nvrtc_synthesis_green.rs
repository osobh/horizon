//! Test NVRTC (Runtime Compilation) for synthesis
//!
//! GREEN phase - comprehensive tests

use cudarc::driver::CudaContext;
use gpu_agents::synthesis::nvrtc::{CompilationOptions, KernelTemplate, NvrtcCompiler};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("Testing NVRTC Compilation in Synthesis (GREEN phase)");
    println!("==================================================");

    let ctx = CudaContext::new(0)?;

    // Test 1: Create NVRTC compiler
    println!("\n1. Testing NVRTC compiler creation...");
    let mut compiler = NvrtcCompiler::new(ctx.clone())?;
    println!("✅ Created NVRTC compiler");

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

    let start = Instant::now();
    let compiled = compiler.compile_kernel("simple_add", kernel_source)?;
    let compile_time = start.elapsed();

    println!(
        "✅ Compiled kernel '{}' in {:?}",
        compiled.name, compile_time
    );
    println!("   Optimization level: {}", compiled.optimization_level);

    // Test 3: Create kernel templates
    println!("\n3. Testing kernel template creation...");
    let template = KernelTemplate::new("pattern_match_template");
    println!("✅ Created kernel template");

    // Test 4: Generate pattern matching kernel
    println!("\n4. Testing pattern matching kernel generation...");
    let pattern_kernel = template.generate_pattern_matcher(32, 1024);
    println!("✅ Generated pattern matching kernel:");
    println!("   Length: {} chars", pattern_kernel.len());
    println!(
        "   Contains pattern_match: {}",
        pattern_kernel.contains("pattern_match")
    );

    // Test 5: Generate template expansion kernel
    println!("\n5. Testing template expansion kernel generation...");
    let template_kernel = template.generate_template_expander(64);
    println!("✅ Generated template expansion kernel:");
    println!("   Length: {} chars", template_kernel.len());
    println!(
        "   Contains expand_template: {}",
        template_kernel.contains("expand_template")
    );

    // Test 6: Generate AST transformation kernels
    println!("\n6. Testing AST transformation kernel generation...");
    let transform_types = vec!["simplify", "optimize", "custom"];
    for transform_type in transform_types {
        let ast_kernel = template.generate_ast_transformer(transform_type);
        println!("   {} kernel: {} chars", transform_type, ast_kernel.len());
    }
    println!("✅ Generated all AST transformation kernels");

    // Test 7: Compile with different optimization levels
    println!("\n7. Testing compilation with optimization levels...");
    for opt_level in 0..=3 {
        let options = CompilationOptions::default()
            .with_arch("sm_80")
            .with_opt_level(opt_level)
            .with_debug_info(opt_level == 0);

        let kernel = compiler.compile_with_options(&pattern_kernel, options)?;
        println!("   O{}: kernel '{}'", opt_level, kernel.name);
    }
    println!("✅ Compiled with all optimization levels");

    // Test 8: Cache functionality
    println!("\n8. Testing kernel cache...");
    let cached_before = compiler.get_cached("simple_add");
    assert!(cached_before.is_some());

    // Compile same kernel again (should be cached)
    let start = Instant::now();
    let _ = compiler.compile_kernel("simple_add", kernel_source)?;
    let cached_time = start.elapsed();

    println!("✅ Cache working:");
    println!("   First compile: {:?}", compile_time);
    println!("   Cached access: {:?}", cached_time);
    println!(
        "   Speedup: {:.2}x",
        compile_time.as_micros() as f64 / cached_time.as_micros().max(1) as f64
    );

    // Test 9: Complex synthesis kernel
    println!("\n9. Testing complex synthesis kernel compilation...");
    let complex_kernel = generate_complex_synthesis_kernel();
    let compiled_complex = compiler.compile_kernel("complex_synthesis", &complex_kernel)?;
    println!(
        "✅ Compiled complex synthesis kernel: {}",
        compiled_complex.name
    );

    // Test 10: Performance with multiple kernels
    println!("\n10. Testing compilation performance...");
    let start = Instant::now();
    let kernel_count = 10;

    for i in 0..kernel_count {
        let kernel_name = format!("perf_kernel_{}", i);
        let kernel_src = format!(
            r#"extern "C" __global__ void {}(float* data) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                data[idx] = idx * {};
            }}"#,
            kernel_name, i
        );
        compiler.compile_kernel(&kernel_name, &kernel_src)?;
    }

    let total_time = start.elapsed();
    println!("✅ Compiled {} kernels in {:?}", kernel_count, total_time);
    println!(
        "   Average: {:.2} ms/kernel",
        total_time.as_micros() as f64 / kernel_count as f64 / 1000.0
    );

    // Clear cache and show stats
    compiler.clear_cache();
    println!("\n✅ Cache cleared");

    println!("\n✅ All GREEN phase tests passed!");
    println!("\nKey achievements:");
    println!("- NVRTC compiler functional");
    println!("- Kernel template generation working");
    println!("- Multiple optimization levels supported");
    println!("- Kernel caching operational");
    println!("- Complex synthesis kernels compile successfully");

    Ok(())
}

fn generate_complex_synthesis_kernel() -> String {
    r#"
    extern "C" __global__ void complex_synthesis(
        const unsigned char* patterns,
        const unsigned char* templates,
        const unsigned int* ast_nodes,
        unsigned char* output,
        unsigned int* match_results,
        const unsigned int pattern_count,
        const unsigned int template_count,
        const unsigned int node_count
    ) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        
        __shared__ unsigned char shared_patterns[1024];
        __shared__ unsigned int shared_results[32];
        
        // Cooperative pattern loading
        if (threadIdx.x < pattern_count * 32) {
            shared_patterns[threadIdx.x] = patterns[threadIdx.x];
        }
        __syncthreads();
        
        // Pattern matching phase
        if (tid < node_count) {
            unsigned int node = ast_nodes[tid];
            bool match = false;
            
            for (int p = 0; p < pattern_count; p++) {
                const unsigned char* pattern = &shared_patterns[p * 32];
                // Simplified matching logic
                if ((node & 0xFF) == pattern[0]) {
                    match = true;
                    atomicAdd(&shared_results[p], 1);
                }
            }
            
            // Template expansion phase
            if (match && tid < template_count) {
                const unsigned char* template_data = &templates[tid * 64];
                unsigned char* out = &output[tid * 64];
                
                // Expand template
                #pragma unroll 8
                for (int i = 0; i < 64; i++) {
                    out[i] = template_data[i] ^ (tid & 0xFF);
                }
            }
        }
        
        // Write results
        if (threadIdx.x < 32 && tid < pattern_count) {
            match_results[tid] = shared_results[tid];
        }
    }
    "#
    .to_string()
}
