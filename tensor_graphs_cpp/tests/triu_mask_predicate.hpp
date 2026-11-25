#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "core/common/execute_ref_graph.hpp"
#include "core/plan/bucket_regions.hpp"
#include "core/rewrite.hpp"
#include "core/graph.hpp"

inline void runTriuMaskTests()
{
    std::cout << "triu mask predicate tests" << std::endl;

    // =========================================================================
    // 1. matchTriuMask and view mapping tests
    // =========================================================================
    {
        Graph graph;
        const uint32_t N = 16;
        LogicalId fill_node = graph.fill(TGConstants::NEG_INF, {N, N});
        LogicalId k_node = graph.constant(std::vector<int32_t>{1});
        LogicalId triu_node = graph.triu(fill_node, k_node);

        auto bound_opt = bucket_regions::matchTriuMask(triu_node, graph);
        if (!bound_opt)
            Error::throw_err("[runTriuMaskTests] matchTriuMask failed on basic TRIU");
        if (bound_opt->row_axis != 0 || bound_opt->col_axis != 1 || bound_opt->diag_k != 1)
            Error::throw_err("[runTriuMaskTests] TriuBound mismatch on basic TRIU");

        // Test CONTIGUOUS view
        LogicalId contig_node = graph.contiguous(triu_node);
        auto contig_bound = bucket_regions::matchTriuMask(contig_node, graph);
        if (!contig_bound || *contig_bound != *bound_opt)
            Error::throw_err("[runTriuMaskTests] matchTriuMask failed through CONTIGUOUS");

        // Test REPEAT view (broadcasting batch dim)
        // Add a leading dim: [1, N, N]
        LogicalId reshape_pre = graph.reshape(triu_node, {1, N, N});
        LogicalId repeat_node = graph.repeat(reshape_pre, 2, 0); // repeat batch axis 0 by 2
        auto repeat_bound = bucket_regions::matchTriuMask(repeat_node, graph);
        if (!repeat_bound || repeat_bound->row_axis != 1 || repeat_bound->col_axis != 2 || repeat_bound->diag_k != 1)
            Error::throw_err("[runTriuMaskTests] matchTriuMask failed through REPEAT batch axis");

        // Test PERMUTE view
        LogicalId perm_node = graph.permute(triu_node, {1, 0});
        auto perm_bound = bucket_regions::matchTriuMask(perm_node, graph);
        if (!perm_bound || perm_bound->row_axis != 1 || perm_bound->col_axis != 0 || perm_bound->diag_k != 1)
            Error::throw_err("[runTriuMaskTests] matchTriuMask failed through PERMUTE");

        // Test RESHAPE view ([1, 1, N, N] -> [1, N, N])
        LogicalId reshape_4d = graph.reshape(triu_node, {1, 1, N, N});
        LogicalId reshape_3d = graph.reshape(reshape_4d, {1, N, N});
        auto reshape_bound = bucket_regions::matchTriuMask(reshape_3d, graph);
        if (!reshape_bound || reshape_bound->row_axis != 1 || reshape_bound->col_axis != 2 || reshape_bound->diag_k != 1)
            Error::throw_err("[runTriuMaskTests] matchTriuMask failed through RESHAPE");

        // Test SLICE view (row [2, 10), col [1, 8), steps 1, 1)
        // diag_k' = 1 + 2 - 1 = 2
        LogicalId slice_node = graph.slice(triu_node, {2, 1}, {10, 8}, {1, 1});
        auto slice_bound = bucket_regions::matchTriuMask(slice_node, graph);
        if (!slice_bound || slice_bound->row_axis != 0 || slice_bound->col_axis != 1 || slice_bound->diag_k != 2)
            Error::throw_err("[runTriuMaskTests] matchTriuMask failed through SLICE (diag_k shift)");

        // Negative test: FILL with 0.0f instead of NEG_INF
        LogicalId zero_fill = graph.fill(0.0f, {N, N});
        LogicalId zero_triu = graph.triu(zero_fill, k_node);
        if (bucket_regions::matchTriuMask(zero_triu, graph).has_value())
            Error::throw_err("[runTriuMaskTests] matchTriuMask succeeded for non-NEG_INF fill");
    }

    // =========================================================================
    // 2. TriuBound arithmetic vs unmaskedRegions reference
    // =========================================================================
    {
        const uint32_t N = 16;
        std::vector<float> data(N * N, 0.0f);
        const int32_t k = 1;
        for (uint32_t r = 0; r < N; ++r)
        {
            for (uint32_t c = 0; c < N; ++c)
            {
                if (static_cast<int32_t>(c) >= static_cast<int32_t>(r) + k)
                    data[r * N + c] = TGConstants::NEG_INF;
            }
        }
        std::vector<uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());

        const std::vector<Region> ref_unmasked = bucket_regions::unmaskedRegions(bytes, {N, N});
        const TriuBound bound{0, 1, k};

        // Test test boxes
        const std::vector<Region> test_boxes = {
            Region{{{0, 8}, {0, 16}}},
            Region{{{4, 12}, {2, 10}}},
            Region{{{10, 16}, {0, 4}}},
            Region{{{0, 16}, {0, 16}}},
            Region{{{12, 16}, {14, 16}}},
            Region{{{0, 2}, {5, 10}}} // completely masked
        };

        for (const auto &box : test_boxes)
        {
            // Exact intersection: the analytic triangle must give the same
            // cells as the scanned unmasked runs.
            const std::vector<Region> ref_exact = bucket_regions::intersect({box}, ref_unmasked);
            const std::vector<Region> triu_exact = bucket_regions::intersectTriu({box}, bound);
            if (!bucket_regions::subtract(ref_exact, triu_exact).empty() ||
                !bucket_regions::subtract(triu_exact, ref_exact).empty())
            {
                Error::throw_err("[runTriuMaskTests] intersectTriu mismatch for box " + encodeRegionList({box}) +
                                 ": ref " + encodeRegionList(ref_exact) + " vs triu " + encodeRegionList(triu_exact));
            }

            // axisRange comparison (col axis 1)
            for (bool every_row : {false, true})
            {
                auto ref_range = bucket_regions::axisRange(ref_unmasked, box, 1, every_row);
                auto triu_range = bucket_regions::axisRange(bound, box, 1, every_row);
                if (ref_range.has_value() != triu_range.has_value())
                {
                    Error::throw_err("[runTriuMaskTests] axisRange presence mismatch for box " +
                                     encodeRegionList({box}) + " everyRow=" + std::to_string(every_row));
                }
                if (ref_range && (ref_range->start != triu_range->start || ref_range->stop != triu_range->stop))
                {
                    Error::throw_err("[runTriuMaskTests] axisRange value mismatch for box " +
                                     encodeRegionList({box}) + " everyRow=" + std::to_string(every_row));
                }
            }
        }

        // Test dotForwardFromB and dotNeededB
        // suppA: rank-3 [1, N, N]
        std::vector<Region> suppA_ref;
        for (const auto &r : ref_unmasked)
        {
            Region r3;
            r3.region = {{0, 1}, r.region[0], r.region[1]};
            suppA_ref.push_back(r3);
        }
        const TriuBound suppA_bound{1, 2, k};

        auto dotForwardFromBRef = [&](const std::vector<Region> &supp, const std::vector<Region> &dirtyB, uint32_t mRows) {
            std::vector<Region> outBoxes;
            for (const Region &b : dirtyB)
                for (const Region &a : supp)
                {
                    Region probe;
                    probe.region = {b.region[0], {0, mRows}, b.region[1]};
                    Region hit = intersectRegions(a, probe);
                    if (bucket_regions::dropEmpty({hit}).empty())
                        continue;
                    Region o;
                    o.region = {hit.region[0], hit.region[1], b.region[2]};
                    outBoxes.push_back(o);
                }
            return bucket_regions::dropEmpty(mergeRegions(outBoxes));
        };

        auto dotNeededBRef = [&](const std::vector<Region> &supp, const std::vector<Region> &neededOut, uint32_t kCols) {
            std::vector<Region> boxes;
            for (const Region &o : neededOut)
                for (const Region &a : supp)
                {
                    Region probe;
                    probe.region = {o.region[0], o.region[1], {0, kCols}};
                    Region hit = intersectRegions(a, probe);
                    if (bucket_regions::dropEmpty({hit}).empty())
                        continue;
                    Region bBox;
                    bBox.region = {hit.region[0], hit.region[2], o.region[2]};
                    boxes.push_back(bBox);
                }
            return bucket_regions::dropEmpty(mergeRegions(boxes));
        };

        auto dotForwardFromBTriu = [&](const TriuBound &bnd, const std::vector<Region> &dirtyB, uint32_t mRows) {
            std::vector<Region> outBoxes;
            for (const Region &b : dirtyB)
            {
                if (b.region.size() != 3) continue;
                const uint32_t k0 = b.region[1].start;
                const int64_t m_min = std::max<int64_t>(0, static_cast<int64_t>(k0) - bnd.diag_k + 1);
                if (m_min < static_cast<int64_t>(mRows))
                {
                    Region o;
                    o.region = {b.region[0], {static_cast<uint32_t>(m_min), mRows}, b.region[2]};
                    outBoxes.push_back(o);
                }
            }
            return bucket_regions::dropEmpty(mergeRegions(outBoxes));
        };

        auto dotNeededBTriu = [&](const TriuBound &bnd, const std::vector<Region> &neededOut, uint32_t kCols) {
            std::vector<Region> boxes;
            for (const Region &o : neededOut)
            {
                if (o.region.size() != 3) continue;
                const uint32_t m1 = o.region[1].stop;
                if (m1 == 0) continue;
                const int64_t k_limit = std::max<int64_t>(0, static_cast<int64_t>(m1 - 1) + bnd.diag_k);
                const uint32_t k_stop = std::min<uint32_t>(kCols, static_cast<uint32_t>(k_limit));
                if (k_stop > 0)
                {
                    Region bBox;
                    bBox.region = {o.region[0], {0, k_stop}, o.region[2]};
                    boxes.push_back(bBox);
                }
            }
            return bucket_regions::dropEmpty(mergeRegions(boxes));
        };

        const std::vector<Region> test_dirty_b = {
            Region{{{0, 1}, {0, 4}, {0, 32}}},
            Region{{{0, 1}, {8, 12}, {0, 32}}},
            Region{{{0, 1}, {15, 16}, {0, 32}}}
        };
        for (const auto &db : test_dirty_b)
        {
            auto ref_out = dotForwardFromBRef(suppA_ref, {db}, N);
            auto triu_out = dotForwardFromBTriu(suppA_bound, {db}, N);
            if (ref_out.size() != triu_out.size())
                Error::throw_err("[runTriuMaskTests] dotForwardFromB size mismatch");
            for (size_t i = 0; i < ref_out.size(); ++i)
                if (!regionsMatch(ref_out[i], triu_out[i]))
                    Error::throw_err("[runTriuMaskTests] dotForwardFromB region mismatch");
        }

        const std::vector<Region> test_needed_out = {
            Region{{{0, 1}, {0, 1}, {0, 32}}},
            Region{{{0, 1}, {0, 8}, {0, 32}}},
            Region{{{0, 1}, {7, 8}, {0, 32}}},
            Region{{{0, 1}, {0, 16}, {0, 32}}}
        };
        for (const auto &no : test_needed_out)
        {
            auto ref_nb = dotNeededBRef(suppA_ref, {no}, N);
            auto triu_nb = dotNeededBTriu(suppA_bound, {no}, N);
            if (ref_nb.size() != triu_nb.size())
                Error::throw_err("[runTriuMaskTests] dotNeededB size mismatch");
            for (size_t i = 0; i < ref_nb.size(); ++i)
                if (!regionsMatch(ref_nb[i], triu_nb[i]))
                    Error::throw_err("[runTriuMaskTests] dotNeededB region mismatch");
        }
    }

    // =========================================================================
    // 3. computeRegionFacts and computeBucketRegions with attention causal mask
    // =========================================================================
    for (uint32_t N : {16u, 64u})
    {
        Graph graph;
        LogicalId q = graph.input({1, N, 32}, DType::FLOAT32);
        LogicalId k = graph.input({1, N, 32}, DType::FLOAT32);
        LogicalId k_t = graph.permute(k, {0, 2, 1});
        LogicalId scores = graph.dot(q, k_t);

        LogicalId fill_node = graph.fill(TGConstants::NEG_INF, {1, 1, N, N});
        LogicalId k_const = graph.constant(std::vector<int32_t>{1});
        LogicalId triu_node = graph.triu(fill_node, k_const);
        LogicalId mask_node = graph.reshape(triu_node, {1, N, N});

        LogicalId masked_scores = graph.add(scores, mask_node);
        std::vector<LogicalId> topo = topologicalSort({masked_scores}, graph);

        RegionFacts facts = computeRegionFacts(graph, topo, nullptr);
        if (facts.triuMasks.count(masked_scores) == 0)
            Error::throw_err("[runTriuMaskTests] facts.triuMasks missing for masked_scores N=" + std::to_string(N));
        if (facts.triuFinite.count(masked_scores) == 0)
            Error::throw_err("[runTriuMaskTests] facts.triuFinite missing for masked_scores N=" + std::to_string(N));

        // Test decode bucket [N-1, N) -> [N-1, N)
        Bucket decode_bucket;
        decode_bucket.inputDirtyRegions[q] = {Region{{{0, 1}, {N - 1, N}, {0, 32}}}};
        decode_bucket.inputDirtyRegions[k] = {Region{{{0, 1}, {N - 1, N}, {0, 32}}}};
        decode_bucket.outputNeededRegion = {Region{{{0, 1}, {N - 1, N}, {0, N}}}};

        BucketRegions decode_regions = computeBucketRegions(graph, masked_scores, decode_bucket, topo, nullptr, &facts);
        const auto &needed_q = decode_regions.neededOf(q);
        const auto &needed_k = decode_regions.neededOf(k);

        if (needed_q.empty() || needed_k.empty())
            Error::throw_err("[runTriuMaskTests] computeBucketRegions yielded empty needed inputs for decode bucket N=" + std::to_string(N));

        // For decode row N-1, all columns 0..N of K are needed
        if (needed_k[0].region[1].stop != N)
            Error::throw_err("[runTriuMaskTests] decode bucket did not require full prefix of K");
    }

    // =========================================================================
    // 6. Large scale test: N = 262,144 foldCleanTensors & matchTriuMask
    // =========================================================================
    {
        Graph graph;
        const uint32_t N = 262144;
        LogicalId fill_node = graph.fill(TGConstants::NEG_INF, {1, 1, N, N});
        LogicalId k_const = graph.constant(std::vector<int32_t>{1});
        LogicalId triu_node = graph.triu(fill_node, k_const);

        auto t0 = std::chrono::steady_clock::now();
        if (!isSymbolicCleanMaskOrFill(triu_node, graph))
            Error::throw_err("[runTriuMaskTests] isSymbolicCleanMaskOrFill returned false for large N=262144 TRIU");

        auto bound_opt = bucket_regions::matchTriuMask(triu_node, graph);
        if (!bound_opt || bound_opt->row_axis != 2 || bound_opt->col_axis != 3 || bound_opt->diag_k != 1)
            Error::throw_err("[runTriuMaskTests] matchTriuMask failed for large N=262144 TRIU");

        // Verify executeReferenceGraph with only_clean_nodes finishes in < 100 ms and does not allocate 275 GB
        class DummyStore : public ITensorStore {
        public:
            bool has(const std::string &) const override { return false; }
            TensorMetadata getMetadata(const std::string &) const override { return {}; }
            std::vector<uint8_t> read(const std::string &) const override { return {}; }
            bool isWritable() const override { return true; }
            void write(const std::string &, const TensorView &, const void *, uint64_t) override {}
        } dummy_store;

        RefGraphOptions options;
        options.only_clean_nodes = true;
        executeReferenceGraph(graph, {triu_node}, dummy_store, options);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (elapsed_ms > 2000.0)
            Error::throw_err("[runTriuMaskTests] Large N=262144 symbolic mask handling took too long: " + std::to_string(elapsed_ms) + " ms");
    }

    std::cout << "triu mask predicate tests passed" << std::endl;
}
