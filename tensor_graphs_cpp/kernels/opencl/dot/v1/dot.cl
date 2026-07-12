// tensor_graphs_cpp/kernels/opencl/dot/dot.cl

#define TILE_SIZE 16

__kernel void dot_f32_3d(
    __global const float* A, 
    __global const float* B, 
    __global float* Out,
    ulong B_count, ulong M, ulong K, ulong N) 
{
    __local float sharedA[TILE_SIZE][TILE_SIZE];
    __local float sharedB[TILE_SIZE][TILE_SIZE];

    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    uint bx = get_group_id(0);
    uint by = get_group_id(1);
    uint b  = get_global_id(2);

    uint row = by * TILE_SIZE + ty;
    uint col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    __global const float* A_b = A + b * M * K;
    __global const float* B_b = B + b * K * N;
    __global float* Out_b = Out + b * M * N;

    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < K) {
            sharedA[ty][tx] = A_b[row * K + t * TILE_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + ty < K) {
            sharedB[ty][tx] = B_b[(t * TILE_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (b < B_count && row < M && col < N) {
        Out_b[row * N + col] = sum;
    }
}