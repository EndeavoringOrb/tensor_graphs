#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
// -------------------------------------------------------------------------
// CUDA KERNEL
// -------------------------------------------------------------------------
template <typename scalar_t>
global void rms_norm_gemma_kernel(
const scalar_t* restrict input,
const scalar_t* restrict weight,
scalar_t* restrict output,
float epsilon,
int stride) {
code
Code
// Each block handles one row (sequence position)
int row_idx = blockIdx.x;
int tid = threadIdx.x;

// Base pointer for this row
const scalar_t* row_input = input + row_idx * stride;
scalar_t* row_output = output + row_idx * stride;

// 1. Calculate Sum of Squares
// ---------------------------------------------------------------------
float sum_sq = 0.0f;
for (int i = tid; i < stride; i += blockDim.x) {
    float x = static_cast<float>(row_input[i]);
    sum_sq += x * x;
}

// Block Reduction (Sum)
// Using shared memory for reduction within the block
extern __shared__ float shared_mem[];
shared_mem[tid] = sum_sq;
__syncthreads();

// Standard tree reduction in shared memory
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        shared_mem[tid] += shared_mem[tid + s];
    }
    __syncthreads();
}

// 2. Calculate Inverse RMS
// ---------------------------------------------------------------------
float inv_rms = 0.0f;
if (tid == 0) {
    float mean_sq = shared_mem[0] / static_cast<float>(stride);
    inv_rms = rsqrtf(mean_sq + epsilon);
}

// Broadcast inv_rms to all threads
__shared__ float shared_inv_rms;
if (tid == 0) {
    shared_inv_rms = inv_rms;
}
__syncthreads();
inv_rms = shared_inv_rms;

// 3. Apply Norm + Scaling (Gemma Style: x * inv_rms * (1 + w))
// ---------------------------------------------------------------------
for (int i = tid; i < stride; i += blockDim.x) {
    float x = static_cast<float>(row_input[i]);
    float w = static_cast<float>(weight[i]);
    
    // Note: The reference implementation in fused/rms_norm.py uses (1.0 + scale)
    // This is distinct from standard RMSNorm which is just (scale).
    float val = x * inv_rms * (1.0f + w);
    
    row_output[i] = static_cast<scalar_t>(val);
}
}
// -------------------------------------------------------------------------
// C++ BINDING
// -------------------------------------------------------------------------
void rms_norm_cuda(
torch::Tensor input,
torch::Tensor weight,
torch::Tensor output,
float epsilon) {
code
Code
const int rows = input.size(0) * input.size(1); // Flatten batch/seq dims
const int cols = input.size(2); // Hidden dim (last dim)

// Heuristic: Threads per block. 
// For hidden dims like 256-4096, 256 or 512 threads is usually good.
int threads = 256;
while (threads < cols && threads < 1024) {
    threads *= 2;
}

// Shared memory size: enough for reduction
size_t shared_mem_size = threads * sizeof(float);

// Flatten input for the grid launch
// We treat everything before the last dim as the "batch of rows"
// Since input can be (Batch, Seq, Hidden) or (TotalSeq, Hidden)
dim3 grid(input.numel() / cols);
dim3 block(threads);

AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_gemma_cuda", ([&] {
    rms_norm_gemma_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        epsilon,
        cols
    );
}));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("rms_norm", &rms_norm_cuda, "Gemma RMSNorm (CUDA)");
}