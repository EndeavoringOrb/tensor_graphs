#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "core/cost_model.hpp"
#include "core/kernels.hpp"
#include "core/plan/planner.hpp"
#include "core/rewrite.hpp"

inline void runScatterCanonicalKeysTests()
{
    std::cout << "scatter canonical keys tests" << std::endl;

    // Find Scatter_F32_Rows kernel
    KernelId scatter_kernel_id{0};
    for (const auto &[uid, entry] : KernelRegistry::get().getAllKernels())
    {
        if (entry.opType == OpType::SCATTER && entry.opName == "Scatter_F32_Rows")
        {
            scatter_kernel_id = uid;
            break;
        }
    }
    if (scatter_kernel_id.value == 0)
    {
        for (const auto &[uid, entry] : KernelRegistry::get().getAllKernels())
        {
            if (entry.opType == OpType::SCATTER)
            {
                scatter_kernel_id = uid;
                break;
            }
        }
    }
    if (scatter_kernel_id.value == 0)
        Error::throw_err("[ScatterCanonicalKeys] No SCATTER kernel registered in KernelRegistry");

    const std::vector<uint32_t> out_shape = {2, 5, 6};
    const std::vector<uint64_t> out_strides = calcContiguousStrides(out_shape);
    const DType out_dtype = DType::FLOAT32;

    const std::vector<uint32_t> upd_shape = {1, 3, 4};
    const std::vector<uint64_t> upd_strides = calcContiguousStrides(upd_shape);
    const DType upd_dtype = DType::FLOAT32;

    const std::vector<std::vector<uint32_t>> in_shapes = {
        upd_shape, {3}, {3}, {3}, {3}
    };
    const std::vector<std::vector<uint64_t>> in_strides = {
        upd_strides, {1}, {1}, {1}, {1}
    };
    const std::vector<DType> in_dtypes = {
        upd_dtype, DType::INT32, DType::INT32, DType::INT32, DType::INT32
    };

    auto makeConstants = [](const std::vector<int32_t> &starts, const std::vector<int32_t> &ends,
                             const std::vector<int32_t> &steps, const std::vector<int32_t> &shape) {
        std::vector<std::vector<uint8_t>> c(5);
        auto copyVec = [](const std::vector<int32_t> &v) {
            std::vector<uint8_t> bytes(v.size() * sizeof(int32_t));
            std::memcpy(bytes.data(), v.data(), bytes.size());
            return bytes;
        };
        c[0] = {}; // updates has no staged constant
        c[1] = copyVec(starts);
        c[2] = copyVec(ends);
        c[3] = copyVec(steps);
        c[4] = copyVec(shape);
        return c;
    };

    // 1. Verify isScatterControlInput and isScatterKernelControlInput helpers
    if (!isScatterControlInput(OpType::SCATTER, 1) || !isScatterControlInput(OpType::SCATTER, 4))
        Error::throw_err("[ScatterCanonicalKeys] isScatterControlInput returned false for scatter control index");
    if (isScatterControlInput(OpType::SCATTER, 0) || isScatterControlInput(OpType::SCATTER, 5))
        Error::throw_err("[ScatterCanonicalKeys] isScatterControlInput returned true for non-control index");
    if (isScatterControlInput(OpType::ADD, 1))
        Error::throw_err("[ScatterCanonicalKeys] isScatterControlInput returned true for ADD op");

    if (!isScatterKernelControlInput(scatter_kernel_id, 1) || !isScatterKernelControlInput(scatter_kernel_id, 4))
        Error::throw_err("[ScatterCanonicalKeys] isScatterKernelControlInput returned false for scatter control index");
    if (isScatterKernelControlInput(scatter_kernel_id, 0))
        Error::throw_err("[ScatterCanonicalKeys] isScatterKernelControlInput returned true for index 0");
    if (isScatterKernelControlInput(KernelId{0}, 1))
        Error::throw_err("[ScatterCanonicalKeys] isScatterKernelControlInput returned true for invalid kernel");

    // 2. Test CostModel::exactRecordKey
    const auto consts_bucket_1 = makeConstants({0, 1, 0}, {1, 2, 4}, {1, 1, 1}, {2, 5, 6});
    const auto consts_bucket_2 = makeConstants({0, 4, 1}, {1, 5, 5}, {1, 1, 1}, {2, 5, 6});

    const std::vector<size_t> order = CostModel::identityOrder(in_shapes.size());
    const std::string key_bucket_1 = CostModel::exactRecordKey(
        scatter_kernel_id, out_shape, out_strides, out_dtype,
        in_shapes, in_strides, in_dtypes, consts_bucket_1, order);
    const std::string key_bucket_2 = CostModel::exactRecordKey(
        scatter_kernel_id, out_shape, out_strides, out_dtype,
        in_shapes, in_strides, in_dtypes, consts_bucket_2, order);

    if (key_bucket_1 != key_bucket_2)
    {
        Error::throw_err("[ScatterCanonicalKeys] exactRecordKey mismatch for different slice controls of same shape");
    }

    // Verify key differs if update shape changes
    const std::vector<uint32_t> upd_shape_diff = {1, 2, 4};
    std::vector<std::vector<uint32_t>> in_shapes_diff = in_shapes;
    in_shapes_diff[0] = upd_shape_diff;
    std::vector<std::vector<uint64_t>> in_strides_diff = in_strides;
    in_strides_diff[0] = calcContiguousStrides(upd_shape_diff);

    const std::string key_diff_update = CostModel::exactRecordKey(
        scatter_kernel_id, out_shape, out_strides, out_dtype,
        in_shapes_diff, in_strides_diff, in_dtypes, consts_bucket_1, order);
    if (key_bucket_1 == key_diff_update)
    {
        Error::throw_err("[ScatterCanonicalKeys] exactRecordKey should differ when update shape changes");
    }

    // Verify key differs if output shape changes
    const std::vector<uint32_t> out_shape_diff = {3, 5, 6};
    const std::vector<uint64_t> out_strides_diff = calcContiguousStrides(out_shape_diff);
    const std::string key_diff_out = CostModel::exactRecordKey(
        scatter_kernel_id, out_shape_diff, out_strides_diff, out_dtype,
        in_shapes, in_strides, in_dtypes, consts_bucket_1, order);
    if (key_bucket_1 == key_diff_out)
    {
        Error::throw_err("[ScatterCanonicalKeys] exactRecordKey should differ when output shape changes");
    }

    // 3. Test CostModel::log_call deduplication
    {
        CostModel cm(false, "");
        std::vector<Record> observed_records;
        cm.callObserver = [&](const Record &r) {
            observed_records.push_back(r);
        };

        cm.log_call(scatter_kernel_id, out_shape, out_strides, out_dtype,
                    in_shapes, in_strides, in_dtypes, consts_bucket_1);
        cm.log_call(scatter_kernel_id, out_shape, out_strides, out_dtype,
                    in_shapes, in_strides, in_dtypes, consts_bucket_2);

        if (observed_records.size() != 2)
            Error::throw_err("[ScatterCanonicalKeys] Expected 2 observed calls");
        for (const auto &r : observed_records)
        {
            for (size_t i = 1; i < r.inputConstants.size(); ++i)
            {
                if (!r.inputConstants[i].empty())
                    Error::throw_err("[ScatterCanonicalKeys] log_call did not clear inputConstants[" + std::to_string(i) + "]");
            }
        }

        // Check that serialized representations of both records are identical
        const std::string s1 = serializeToString(observed_records[0]);
        const std::string s2 = serializeToString(observed_records[1]);
        if (s1 != s2)
            Error::throw_err("[ScatterCanonicalKeys] Serialized records for different buckets are not identical");
    }

    // 4. Test Planner::memoizedCost caching across buckets
    {
        CostModel cm(false, "");
        Settings settings;
        Planner planner(cm, settings);
        planner.cost_memo_ = std::make_shared<Planner::CostMemo>();

        const float cost1 = planner.memoizedCost(
            scatter_kernel_id, out_shape, out_strides, out_dtype,
            in_shapes, in_strides, in_dtypes, consts_bucket_1);
        if (planner.cost_memo_->costs.size() != 1)
            Error::throw_err("[ScatterCanonicalKeys] Expected 1 entry in cost_memo after first call");

        const float cost2 = planner.memoizedCost(
            scatter_kernel_id, out_shape, out_strides, out_dtype,
            in_shapes, in_strides, in_dtypes, consts_bucket_2);

        if (planner.cost_memo_->costs.size() != 1)
        {
            Error::throw_err("[ScatterCanonicalKeys] memoizedCost did not reuse cache entry across buckets (size = " +
                             std::to_string(planner.cost_memo_->costs.size()) + ")");
        }
        if (cost1 != cost2)
            Error::throw_err("[ScatterCanonicalKeys] memoizedCost returned different costs for identical shapes");
    }

    // 5. Test FusionRule selectionMemo signature canonicalization for SCATTER
    {
        EGraph egraph;
        const EClassId upd_class = egraph.addEClass(upd_shape, upd_strides, upd_dtype, MemSpace{1, HandleType::CPP});

        const EClassId starts_1 = egraph.addIntConst({0, 1, 0});
        const EClassId ends_1 = egraph.addIntConst({1, 2, 4});
        const EClassId steps_1 = egraph.addIntConst({1, 1, 1});
        const EClassId shape_1 = egraph.addIntConst({2, 5, 6});

        const EClassId starts_2 = egraph.addIntConst({0, 4, 1});
        const EClassId ends_2 = egraph.addIntConst({1, 5, 5});
        const EClassId steps_2 = egraph.addIntConst({1, 1, 1});
        const EClassId shape_2 = egraph.addIntConst({2, 5, 6});

        const ENode enode_1(scatter_kernel_id, OpType::SCATTER, "Scatter_F32_Rows",
                            {upd_class, starts_1, ends_1, steps_1, shape_1}, out_shape, out_strides, out_dtype,
                            MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});
        const ENode enode_2(scatter_kernel_id, OpType::SCATTER, "Scatter_F32_Rows",
                            {upd_class, starts_2, ends_2, steps_2, shape_2}, out_shape, out_strides, out_dtype,
                            MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});

        auto buildSignature = [&](const ENode &old_enode, const std::vector<EClassId> &inputs) {
            std::string sig;
            const CostModel *model = nullptr;
            FusionRule::appendSignature(sig, &model, sizeof(model));
            const std::string dummy_hash = "dummy_pattern_hash";
            FusionRule::appendSignature(sig, dummy_hash.data(), dummy_hash.size());
            const uint32_t op_type = static_cast<uint32_t>(old_enode.getOpType());
            FusionRule::appendSignature(sig, &op_type, sizeof(op_type));
            FusionRule::appendSignature(sig, old_enode.getShape().data(), old_enode.getShape().size() * sizeof(uint32_t));
            FusionRule::appendSignature(sig, old_enode.getStrides().data(), old_enode.getStrides().size() * sizeof(uint64_t));
            const uint32_t out_d = static_cast<uint32_t>(old_enode.getDType());
            FusionRule::appendSignature(sig, &out_d, sizeof(out_d));
            FusionRule::appendSignature(sig, out_strides.data(), out_strides.size() * sizeof(uint64_t));
            const MemSpace ms{1, HandleType::CPP};
            FusionRule::appendSignature(sig, &ms, sizeof(ms));
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                const EClass &child = egraph.getEClass(egraph.findConst(inputs[i]));
                FusionRule::appendSignature(sig, child.shape.data(), child.shape.size() * sizeof(uint32_t));
                FusionRule::appendSignature(sig, child.strides.data(), child.strides.size() * sizeof(uint64_t));
                const uint32_t d = static_cast<uint32_t>(child.dtype);
                FusionRule::appendSignature(sig, &d, sizeof(d));
                const MemSpace cms = child.mem_space;
                FusionRule::appendSignature(sig, &cms, sizeof(cms));
                auto staged = egraph.constantStaging.find(egraph.findConst(inputs[i]));
                if (staged != egraph.constantStaging.end() && !isScatterControlInput(old_enode.getOpType(), i))
                    FusionRule::appendSignature(sig, staged->second->data(), staged->second->size());
                else
                    FusionRule::appendSignature(sig, nullptr, 0);
            }
            return sig;
        };

        const std::string sig_1 = buildSignature(enode_1, {upd_class, starts_1, ends_1, steps_1, shape_1});
        const std::string sig_2 = buildSignature(enode_2, {upd_class, starts_2, ends_2, steps_2, shape_2});

        if (sig_1 != sig_2)
        {
            Error::throw_err("[ScatterCanonicalKeys] FusionRule selectionMemo signatures differ for different scatter controls");
        }
    }

    // 6. Test KernelBench referenceObserver SCATTER deduplication
    {
        std::map<std::string, Record> calls;
        std::mutex calls_mutex;

        auto testReferenceObserver = [&](KernelId kid, const std::vector<std::vector<uint8_t>> &constants) {
            Record record;
            record.kernelId = kid;
            record.buildContextId = BUILD_CONTEXT_ID;
            record.inputShapes = in_shapes;
            record.inputStrides = in_strides;
            record.inputDTypes = in_dtypes;
            record.outputShape = out_shape;
            record.outputStrides = out_strides;
            record.outputDType = out_dtype;
            record.inputConstants = constants;
            if (KernelRegistry::get().getKernel(record.kernelId).opType == OpType::SCATTER)
            {
                for (size_t i = 1; i < record.inputConstants.size(); ++i)
                    record.inputConstants[i].clear();
            }
            std::lock_guard<std::mutex> lock(calls_mutex);
            calls.emplace(serializeToString(record), record);
        };

        // Simulate two different buckets emitting reference scatter records
        testReferenceObserver(scatter_kernel_id, consts_bucket_1);
        testReferenceObserver(scatter_kernel_id, consts_bucket_2);

        if (calls.size() != 1)
        {
            Error::throw_err("[ScatterCanonicalKeys] referenceObserver did not deduplicate SCATTER records across buckets (size = " +
                             std::to_string(calls.size()) + ")");
        }
    }

    std::cout << "scatter canonical keys tests passed" << std::endl;
}
