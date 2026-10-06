typedef struct {
    uint rank;
    uint padding;
    uint shape[8];
    ulong in_strides[8];
} ContiguousParams;

__kernel void contiguous_generic(__global const uchar* src,
                                 __global uchar* dst,
                                 ulong numElements,
                                 ulong elemSize,
                                 ContiguousParams p) {
    ulong idx = get_global_id(0);
    if (idx >= numElements) {
        return;
    }

    ulong temp = idx;
    ulong src_idx = 0;

    // Unravel the output linear index into coordinates, 
    // then reconstruct the offset index using the input strides
    for (int i = 7; i >= 0; --i) {
        if (i >= (int)p.rank) {
            continue;
        }

        uint coord = temp % p.shape[i];
        temp /= p.shape[i];
        src_idx += coord * p.in_strides[i];
    }

    for (ulong b = 0; b < elemSize; ++b) {
        dst[idx * elemSize + b] = src[src_idx * elemSize + b];
    }
}