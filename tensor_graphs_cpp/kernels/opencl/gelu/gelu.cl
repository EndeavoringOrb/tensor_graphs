// tensor_graphs_cpp/kernels/opencl/gelu/gelu.cl

__kernel void gelu_f32_nd(__global const float* A, __global float* Out, ulong n) {
    size_t idx = get_global_id(0);
    if (idx < n) {
        float x = A[idx];
        // exact erf mapping: 0.5 * x * (1 + erf(x / sqrt(2)))
        Out[idx] = 0.5f * x * (1.0f + erf(x * 0.7071067811865475f));
    }
}