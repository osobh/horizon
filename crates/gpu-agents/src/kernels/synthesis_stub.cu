// Stub implementation for synthesis kernels to test linkage
#include <cuda_runtime.h>
#include <cstdio>

extern "C" void launch_match_patterns_stub(
    const uint8_t* patterns,
    const uint8_t* ast_nodes,
    uint32_t* matches,
    uint32_t num_patterns,
    uint32_t num_nodes
) {
    printf("[CUDA] launch_match_patterns called with %u patterns and %u nodes\n", 
           num_patterns, num_nodes);
    
    // Simple stub - mark all nodes as non-matching
    cudaMemset(matches, 0, num_nodes * 2 * sizeof(uint32_t));
}

extern "C" void launch_expand_templates_stub(
    const uint8_t* templates,
    const uint8_t* bindings,
    uint8_t* output,
    uint32_t num_templates,
    uint32_t max_output_size
) {
    printf("[CUDA] launch_expand_templates called\n");
    
    // Simple stub - write "stub" to output
    const char* stub_text = "stub_output";
    cudaMemcpy(output, stub_text, strlen(stub_text) + 1, cudaMemcpyHostToDevice);
}

extern "C" void launch_transform_ast_stub(
    const uint8_t* ast_nodes,
    const uint8_t* rules,
    uint8_t* output_nodes,
    uint32_t num_nodes,
    uint32_t num_rules
) {
    printf("[CUDA] launch_transform_ast called\n");
    
    // Simple stub - copy input to output
    cudaMemcpy(output_nodes, ast_nodes, num_nodes * 40, cudaMemcpyDeviceToDevice);
}