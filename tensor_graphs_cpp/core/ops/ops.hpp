#pragma once
#include <array>
#include <string>

#include "core/ops/add.hpp"
#include "core/ops/and.hpp"
#include "core/ops/arange.hpp"
#include "core/ops/argmax.hpp"
#include "core/ops/cache.hpp"
#include "core/ops/cast.hpp"
#include "core/ops/concat.hpp"
#include "core/ops/contiguous.hpp"
#include "core/ops/copy_to.hpp"
#include "core/ops/cos.hpp"
#include "core/ops/div.hpp"
#include "core/ops/dot.hpp"
#include "core/ops/eq.hpp"
#include "core/ops/fill.hpp"
#include "core/ops/fused.hpp"
#include "core/ops/gather.hpp"
#include "core/ops/im2col.hpp"
#include "core/ops/input.hpp"
#include "core/ops/log.hpp"
#include "core/ops/lt.hpp"
#include "core/ops/max.hpp"
#include "core/ops/mul.hpp"
#include "core/ops/neg.hpp"
#include "core/ops/not.hpp"
#include "core/ops/op_def.hpp"
#include "core/ops/or.hpp"
#include "core/ops/permute.hpp"
#include "core/ops/pow.hpp"
#include "core/ops/repeat.hpp"
#include "core/ops/reshape.hpp"
#include "core/ops/scatter.hpp"
#include "core/ops/sin.hpp"
#include "core/ops/slice.hpp"
#include "core/ops/sum.hpp"
#include "core/ops/triu.hpp"
#include "core/ops/unpack.hpp"
#include "core/types.hpp"

class OpTable
{
  public:
    static const OpTraits &get(OpType op)
    {
        return instance().table_[static_cast<size_t>(op)];
    }

    static bool has(OpType op)
    {
        return get(op).inferShape != nullptr;
    }

  private:
    std::array<OpTraits, 64> table_{};

    static const OpTable &instance()
    {
        static const OpTable inst;
        return inst;
    }

    OpTable()
    {
        registerOp<InputOp>();
        registerOp<CacheOp>();
        registerOp<AddOp>();
        registerOp<MulOp>();
        registerOp<DivideOp>();
        registerOp<DotOp>();
        registerOp<SinOp>();
        registerOp<CosOp>();
        registerOp<NegateOp>();
        registerOp<PowerOp>();
        registerOp<SumOp>();
        registerOp<MaxOp>();
        registerOp<ReshapeOp>();
        registerOp<PermuteOp>();
        registerOp<SliceOp>();
        registerOp<ConcatOp>();
        registerOp<CastOp>();
        registerOp<UnpackOp>();
        registerOp<RepeatOp>();
        registerOp<ArangeOp>();
        registerOp<TriuOp>();
        registerOp<GatherOp>();
        registerOp<FillOp>();
        registerOp<CopyToOp>();
        registerOp<Im2ColOp>();
        registerOp<ContiguousOp>();
        registerOp<ScatterOp>();
        registerOp<LogOp>();
        registerOp<ArgmaxOp>();
        registerOp<LtOp>();
        registerOp<EqOp>();
        registerOp<AndOp>();
        registerOp<OrOp>();
        registerOp<NotOp>();
        registerOp<FusedOp>();
    }

    template <typename Op> void registerOp()
    {
        auto t = Op::traits();
        table_[static_cast<size_t>(t.op_type)] = t;
    }
};

inline const OpTraits &getOpTraits(OpType op)
{
    return OpTable::get(op);
}

inline std::string toString(OpType op)
{
    const char *name = getOpTraits(op).name;
    if (name && name[0] != '\0')
        return name;
    return "UNKNOWN_OP";
}

inline std::ostream &operator<<(std::ostream &os, OpType op)
{
    return os << toString(op);
}

inline bool isElementwise(OpType op)
{
    return getOpTraits(op).is_elementwise;
}

inline bool isConstant(OpType op, uint64_t inputIdx, uint64_t numInputs = 0)
{
    const auto &traits = getOpTraits(op);
    if (traits.isConstant)
    {
        return traits.isConstant(inputIdx, numInputs);
    }
    return false;
}

inline WorkloadMetrics computeWorkload(OpType op, const std::vector<std::vector<uint32_t>> &inShapes,
                                       const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                       DType outDType, const std::string &opName = "")
{
    const auto &traits = getOpTraits(op);
    if (traits.computeWorkload)
    {
        return traits.computeWorkload(inShapes, inDTypes, outShape, outDType, opName);
    }
    return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
}