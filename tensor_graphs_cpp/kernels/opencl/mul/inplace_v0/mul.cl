__kernel void mul_f32_nd_inplace(__global float* A, __global const float* B, ulong n) {
    size_t idx = get_global_id(0);
    if (idx < n) {
        A[idx] = A[idx] * B[idx];
    }
}