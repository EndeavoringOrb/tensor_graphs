__kernel void dot_f32_3d(__global const float* A, __global const float* B, __global float* Out,
                         ulong B_count, ulong M, ulong K, ulong N) {
    ulong n = get_global_id(0);
    ulong m = get_global_id(1);
    ulong b = get_global_id(2);

    if (b < B_count && m < M && n < N) {
        float sum = 0.0f;
        for (ulong k = 0; k < K; ++k) {
            sum += A[b * M * K + m * K + k] * B[b * K * N + k * N + n];
        }
        Out[b * M * N + m * N + n] = sum;
    }
}