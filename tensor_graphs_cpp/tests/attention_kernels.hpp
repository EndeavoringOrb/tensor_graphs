// Masked_Attention_F32(_Threaded) and Causal_Attention_F32(_Threaded) against
// a dense scalar reference (scores + explicit additive mask), including fully
// masked rows, fewer query rows than keys, and the row-restriction identity
// SlicePushDownFused relies on: the kernel on output rows [a, b) (query and
// mask rows sliced, or diag_k + a) equals those rows of the full output.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "core/common/constants.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "core/plan/planner.hpp"
#include "kernels/fused/gemma_attention/ref.hpp"
#include "tests/common.hpp"

namespace AttentionKernelTests
{
// softmax(q k^T / sqrt(D) + mask) v; q [H, Sq, D], k v [H, Sk, D], mask
// [Sq, Sk]; output [Sq, H * D].
inline std::vector<float> denseReference(const std::vector<float> &q, const std::vector<float> &k,
                                         const std::vector<float> &v, const std::vector<float> &mask, uint32_t H,
                                         uint32_t Sq, uint32_t Sk, uint32_t D)
{
    std::vector<float> out(static_cast<size_t>(Sq) * H * D, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> scores(Sk);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t s = 0; s < Sq; ++s)
        {
            float m = -INFINITY;
            for (uint32_t j = 0; j < Sk; ++j)
            {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; ++d)
                    dot += (q[(static_cast<size_t>(h) * Sq + s) * D + d] * scale) * k[(static_cast<size_t>(h) * Sk + j) * D + d];
                scores[j] = dot + mask[static_cast<size_t>(s) * Sk + j];
                m = std::max(m, scores[j]);
            }
            float total = 0.0f;
            for (uint32_t j = 0; j < Sk; ++j)
            {
                scores[j] = std::exp(scores[j] - m);
                total += scores[j];
            }
            for (uint32_t j = 0; j < Sk; ++j)
                for (uint32_t d = 0; d < D; ++d)
                    out[static_cast<size_t>(s) * H * D + static_cast<size_t>(h) * D + d] +=
                        scores[j] / total * v[(static_cast<size_t>(h) * Sk + j) * D + d];
        }
    return out;
}

inline const KernelEntry &kernelNamed(const std::string &name)
{
    for (const auto &[uid, entry] : KernelRegistry::get().getAllKernels())
        if (entry.opName == name)
            return entry;
    Error::throw_err("[AttentionKernelTests] kernel " + name + " is not registered");
}

inline TensorView view(const std::vector<uint32_t> &shape, DType dtype)
{
    TensorNode node;
    node.dtype = dtype;
    node.setShape(shape);
    node.strides = calcContiguousStrides(shape);
    return makeView(node);
}

// Runs a 4-input attention kernel; `last` is the mask (masked) or diag_k (causal).
inline std::vector<float> runKernel(const std::string &name, const std::vector<float> &q, const std::vector<float> &k,
                                    const std::vector<float> &v, const void *last, const TensorView &last_view,
                                    uint32_t H, uint32_t Sq, uint32_t Sk, uint32_t D)
{
    const KernelEntry &kernel = kernelNamed(name);
    std::vector<TensorView> in_views{view({H, Sq, D}, DType::FLOAT32), view({H, Sk, D}, DType::FLOAT32),
                                     view({H, Sk, D}, DType::FLOAT32), last_view};
    const TensorView out_view = view({1, Sq, H * D}, DType::FLOAT32);
    std::vector<TensorNode> in_nodes;
    for (const TensorView &in : in_views)
    {
        TensorNode node;
        node.dtype = in.dtype;
        node.setShape(in.getShape());
        node.strides = calcContiguousStrides(in.getShape());
        in_nodes.push_back(node);
    }
    TensorNode out_node;
    out_node.dtype = DType::FLOAT32;
    out_node.setShape(out_view.getShape());
    out_node.strides = calcContiguousStrides(out_view.getShape());
    if (!kernel.match(in_nodes, out_node))
        Error::throw_err("[AttentionKernelTests] " + name + " does not match its test shapes");
    std::vector<float> out(static_cast<size_t>(Sq) * H * D, -12345.0f);
    KernelContext ctx({q.data(), k.data(), v.data(), last}, {out.data()}, in_views, {out_view});
    kernel.run(ctx);
    return out;
}

inline void expectClose(const std::vector<float> &actual, const std::vector<float> &expected, const std::string &what)
{
    if (actual.size() != expected.size())
        Error::throw_err("[AttentionKernelTests] " + what + ": size mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-4f * (1.0f + std::abs(expected[i])))
            Error::throw_err("[AttentionKernelTests] " + what + ": element " + std::to_string(i) + " is " +
                             std::to_string(actual[i]) + ", expected " + std::to_string(expected[i]));
}

inline std::vector<float> causalMask(uint32_t Sq, uint32_t Sk, int32_t diag_k)
{
    std::vector<float> mask(static_cast<size_t>(Sq) * Sk, 0.0f);
    for (uint32_t s = 0; s < Sq; ++s)
        for (uint32_t j = 0; j < Sk; ++j)
            if (static_cast<int64_t>(j) >= static_cast<int64_t>(s) + diag_k)
                mask[static_cast<size_t>(s) * Sk + j] = TGConstants::NEG_INF;
    return mask;
}

// Rows [a, b) of a [H, S, D] tensor (axis 1).
inline std::vector<float> headRows(const std::vector<float> &x, uint32_t H, uint32_t S, uint32_t D, uint32_t a,
                                   uint32_t b)
{
    std::vector<float> out;
    for (uint32_t h = 0; h < H; ++h)
        out.insert(out.end(), x.begin() + (static_cast<size_t>(h) * S + a) * D, x.begin() + (static_cast<size_t>(h) * S + b) * D);
    return out;
}

// The mask the Gemma model builds (compute_attention_mask(0)), then the
// model's attention (refFactoryGemmaAttentionEx). Saturating it must produce
// both the masked and the causal fused attention kernels.
inline void checkPatternsMatchModelAttention()
{
    const uint32_t H = 4, S = 16, D = 256;
    Graph g;
    const LogicalId q = g.input({H, S, D}, DType::FLOAT32);
    const LogicalId k = g.input({H, S, D}, DType::FLOAT32);
    const LogicalId v = g.input({H, S, D}, DType::FLOAT32);
    int32_t mask_shape[] = {static_cast<int32_t>(S), static_cast<int32_t>(S)};
    float one_val = 1.0f;
    LogicalId ones = g.fill(g.constant({1}, &one_val, DType::FLOAT32), g.constant({2}, mask_shape, DType::INT32));
    int32_t k_val = 1;
    LogicalId blocked = g.triu(ones, g.constant({1}, &k_val, DType::INT32));
    float neg_inf_val = -1e9f;
    int32_t neg_inf_shape[] = {1, 1};
    LogicalId neg_inf = g.reshape(g.constant({1}, &neg_inf_val, DType::FLOAT32),
                                  g.constant({2}, neg_inf_shape, DType::INT32));
    neg_inf = g.repeat(g.repeat(neg_inf, S, 0), S, 1);
    int32_t final_shape[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(S)};
    LogicalId mask = g.reshape(g.mul(blocked, neg_inf), g.constant({3}, final_shape, DType::INT32));
    const LogicalId root = refFactoryGemmaAttentionEx(q, k, v, mask, H, S, D, g);
    ShapePropagator propagator;
    propagator.inferShapeRecursive(root, g);

    CostModel cost_model(false, "");
    Settings settings;
    settings.do_saturate = true;
    Planner planner(cost_model, settings);
    Bucket full;
    for (LogicalId input : {q, k, v})
        full.inputDirtyRegions[input] = {makeRegion({{0, H}, {0, S}, {0, D}})};
    full.outputNeededRegion = {makeRegion({{0, 1}, {0, S}, {0, H * D}})};
    const EGraph result = planner.saturateBucket(root, g, full, true);
    for (const char *name : {"Masked_Attention_F32", "Causal_Attention_F32"})
    {
        bool found = false;
        for (const ENode &enode : result.getENodes())
            found = found || (enode.getOpType() == OpType::FUSED && enode.getOpName() == name);
        if (!found)
            Error::throw_err(std::string("[AttentionKernelTests] saturation did not discover ") + name +
                             " on the model's attention");
    }
}

inline void run()
{
    checkPatternsMatchModelAttention();
    const uint32_t H = 3, Sk = 9, D = 8;
    std::vector<float> q(static_cast<size_t>(H) * Sk * D), k(q.size()), v(q.size());
    for (size_t i = 0; i < q.size(); ++i)
    {
        q[i] = std::sin(0.37f * static_cast<float>(i));
        k[i] = std::cos(0.23f * static_cast<float>(i));
        v[i] = std::sin(0.11f * static_cast<float>(i) + 1.0f);
    }
    for (const char *suffix : {"", "_Threaded"})
    {
        const std::string masked = std::string("Masked_Attention_F32") + suffix;
        const std::string causal = std::string("Causal_Attention_F32") + suffix;
        // diag_k <= -(Sk) makes every row fully masked; large diag_k masks nothing.
        for (int32_t diag_k : {1, 3, 0, -2, -20, 40})
        {
            const std::vector<float> mask = causalMask(Sk, Sk, diag_k);
            const std::vector<float> expected = denseReference(q, k, v, mask, H, Sk, Sk, D);
            const std::string tag = " diag_k=" + std::to_string(diag_k);
            expectClose(runKernel(masked, q, k, v, mask.data(), view({1, Sk, Sk}, DType::FLOAT32), H, Sk, Sk, D),
                        expected, masked + tag);
            expectClose(runKernel(causal, q, k, v, &diag_k, view({1}, DType::INT32), H, Sk, Sk, D), expected,
                        causal + tag);

            // Row restriction: output rows [a, b).
            for (auto [a, b] : {std::pair<uint32_t, uint32_t>{0, 1}, {4, 5}, {2, 7}, {8, 9}})
            {
                const uint32_t rows = b - a;
                std::vector<float> expected_rows(expected.begin() + static_cast<size_t>(a) * H * D,
                                                 expected.begin() + static_cast<size_t>(b) * H * D);
                const std::vector<float> q_rows = headRows(q, H, Sk, D, a, b);
                std::vector<float> mask_rows(mask.begin() + static_cast<size_t>(a) * Sk,
                                             mask.begin() + static_cast<size_t>(b) * Sk);
                const int32_t shifted = diag_k + static_cast<int32_t>(a);
                const std::string rtag = tag + " rows=" + std::to_string(a) + ":" + std::to_string(b);
                expectClose(runKernel(masked, q_rows, k, v, mask_rows.data(), view({1, rows, Sk}, DType::FLOAT32), H,
                                      rows, Sk, D),
                            expected_rows, masked + rtag);
                expectClose(runKernel(causal, q_rows, k, v, &shifted, view({1}, DType::INT32), H, rows, Sk, D),
                            expected_rows, causal + rtag);
            }
        }
        // A non-causal additive mask (finite biases plus a masked block) on
        // the masked kernel only.
        std::vector<float> mask(static_cast<size_t>(Sk) * Sk);
        for (size_t i = 0; i < mask.size(); ++i)
            mask[i] = (i % 5 == 0) ? TGConstants::NEG_INF : 0.25f * std::sin(static_cast<float>(i));
        expectClose(runKernel(masked, q, k, v, mask.data(), view({1, Sk, Sk}, DType::FLOAT32), H, Sk, Sk, D),
                    denseReference(q, k, v, mask, H, Sk, Sk, D), masked + " arbitrary mask");
    }
    std::cout << "attention kernel tests passed" << std::endl;
}
} // namespace AttentionKernelTests
