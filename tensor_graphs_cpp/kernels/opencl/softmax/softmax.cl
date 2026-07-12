// tensor_graphs_cpp/kernels/opencl/softmax/softmax.cl

__kernel void softmax_f32_4d(
    __global const float* in,
    __global float* out,
    uint outer_size,
    uint dim_size) 
{
    uint i = get_global_id(0);
    if (i >= outer_size) return;

    __global const float* r_in = in + i * dim_size;
    __global float* r_out = out + i * dim_size;

    // 1. Find Max value for numerical stability
    float max_val = -1e30f;
    for (uint d = 0; d < dim_size; ++d) {
        if (r_in[d] > max_val) {
            max_val = r_in[d];
        }
    }

    // 2. Compute Exp and Sum
    float sum_val = 0.0f;
    for (uint d = 0; d < dim_size; ++d) {
        float e = exp(r_in[d] - max_val);
        r_out[d] = e;
        sum_val += e;
    }

    // 3. Normalize
    float inv_sum = 1.0f / sum_val;
    for (uint d = 0; d < dim_size; ++d) {
        r_out[d] *= inv_sum;
    }
}