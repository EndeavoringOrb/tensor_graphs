# Kernel Performance Log

Target: Text generation < 13.83 ms / tok (faster than llama.cpp on Quadro M4000).

## Benchmarks Summary

| Kernel Added / Step | Prompt Processing (seq_len 5) | Text Generation (seq_len 6..16) | Overall Generation (seq_len 5..15) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 518.20 ms (9.65 t/s) | 63.32 ms / tok (15.79 t/s) | 104.66 ms / tok (9.55 t/s) | Baseline before optimizations |
| **Gemma_RMS_Norm_CUDA** | 608.39 ms (8.22 t/s) | 54.35 ms / tok (18.40 t/s) | 104.72 ms / tok (9.55 t/s) | Fused Gemma RMSNorm for emb_dim=640 on CUDA |
| **Gemma_K/Q_RMS_Norm_CUDA** | 899.80 ms (5.56 t/s) | 50.91 ms / tok (19.64 t/s) | 128.08 ms / tok (7.81 t/s) | Fused K (dim 256) & Q (heads 4, dim 256) RMSNorm on CUDA; decode step execution ~45.1 ms |
| **RemoveCopyChains Direct Origin Bypass** | 754.87 ms (6.62 t/s) | 37.14 ms / tok (26.92 t/s) | 102.39 ms / tok (9.77 t/s) | Multi-origin RemoveCopyChains connects fused CUDA producers directly to CuBLAS/consumers; instructions 1777 -> 1495; decode step execution ~32.0 ms |
| **Gemma_Q/K_RoPE_CUDA** | 746.40 ms (6.70 t/s) | 29.17 ms / tok (34.28 t/s) | 94.37 ms / tok (10.60 t/s) | Fused Gemma RoPE for Q (heads 4) & K (heads 1) on CUDA; instructions 1495 -> 1072; decode step execution ~26.5 ms |
| **llama.cpp Reference** | 24.18 ms (206.80 t/s) | 13.83 ms / tok (72.28 t/s) | ~14.77 ms / tok (~67.7 t/s) | Target to beat (< 13.83 ms / tok) |

