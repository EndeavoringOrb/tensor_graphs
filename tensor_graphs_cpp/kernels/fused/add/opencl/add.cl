__kernel void add_f32_nd(__global const float* A, __global const float* B, __global float* Out, ulong n) {
    ulong idx = get_global_id(0);
    if (idx < n) {
        Out[idx] = A[idx] + B[idx];
    }
}