__kernel void pow_f32_nd(__global const float* A, __global const float* B, __global float* Out, ulong n) {
    size_t idx = get_global_id(0);
    if (idx < n) {
        Out[idx] = pow(A[idx], B[idx]);
    }
}