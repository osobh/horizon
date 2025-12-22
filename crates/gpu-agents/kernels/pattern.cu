extern "C" __global__ void pattern_kernel(
    const unsigned char* patterns,
    const unsigned char* nodes,
    unsigned int* matches,
    int pattern_size,
    int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        bool match = true;
        for (int i = 0; i < pattern_size; i++) {
            if (nodes[idx * pattern_size + i] != patterns[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            matches[idx] = 1;
        }
    }
}