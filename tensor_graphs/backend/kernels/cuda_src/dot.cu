#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// --- Kernel 1: 3D/2D Projection (Flattened Row approach) ---
template <typename scalar_t>
__global__ void dot_projection_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M_flat, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_flat && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += (float)A[row * K + k] * (float)B[k * N + col];
        }
        C[row * N + col] = (scalar_t)value;
    }
}

// --- Kernel 2: 4D Batch MatMul (B, H, M, K) @ (B, H, K, N) ---
template <typename scalar_t>
__global__ void dot_batched_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int batch_size, int M, int K, int N)
{
    // Each block in z handles one (Batch * Head)
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && row < M && col < N) {
        // Compute offsets for this batch
        const scalar_t* A_ptr = A + b * (M * K);
        const scalar_t* B_ptr = B + b * (K * N);
        scalar_t* C_ptr = C + b * (M * N);

        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += (float)A_ptr[row * K + k] * (float)B_ptr[k * N + col];
        }
        C_ptr[row * N + col] = (scalar_t)value;
    }
}

void dot_cuda_dispatch(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int rank_a = A.dim();
    int rank_b = B.dim();

    dim3 threads(16, 16);

    if (rank_a == 4 && rank_b == 4) {
        // Batch MatMul: [B, H, M, K] @ [B, H, K, N]
        int B_val = A.size(0);
        int H_val = A.size(1);
        int M = A.size(2);
        int K = A.size(3);
        int N = B.size(3);
        int total_batches = B_val * H_val;

        dim3 blocks((N + threads.x - 1) / threads.x, 
                    (M + threads.y - 1) / threads.y, 
                    total_batches);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "dot_batched_cuda", ([&] {
            dot_batched_kernel<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
                total_batches, M, K, N);
        }));
    } else {
        // Projection / Standard: [..., K] @ [K, N]
        int K = B.size(0);
        int N = B.size(1);
        int M_flat = A.numel() / K;

        dim3 blocks((N + threads.x - 1) / threads.x, (M_flat + threads.y - 1) / threads.y);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "dot_projection_cuda", ([&] {
            dot_projection_kernel<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
                M_flat, K, N);
        }));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot", &dot_cuda_dispatch, "CUDA Dot Product (Batch & Projection)");
}