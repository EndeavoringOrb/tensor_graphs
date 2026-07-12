// tensor_graphs_cpp/kernels/opencl/rmsnorm/rmsnorm.cl

__kernel void rmsnorm_f32_3d(
    __global const float* x,
    __global const float* w,
    __global float* out,
    uint outer_size,
    uint dim_size,
    float eps) 
{
    uint r = get_global_id(0);
    if (r >= outer_size) return;

    __global const float* row_x = x + r * dim_size;
    __global float* row_out = out + r * dim_size;

    // 1. Sum of squares
    float sum_sq = 0.0f;
    for (uint d = 0; d < dim_size; ++d) {
        float val = row_x[d];
        sum_sq += val * val;
    }

    // 2. Inverse RMS
    float mean_sq = sum_sq / (float)dim_size;
    float inv_std = 1.0f / sqrt(mean_sq + eps);

    // 3. Normalize and scale
    for (uint d = 0; d < dim_size; ++d) {
        row_out[d] = row_x[d] * inv_std * w[d];
    }
}