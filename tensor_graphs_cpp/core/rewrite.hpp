#pragma once
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"
#include "core/egraph.hpp"
#include "core/repo.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <string>
#include <algorithm>
#include <cstring>

inline bool isEClassProtected(uint32_t eclassId, const std::unordered_set<uint32_t> &protectedEClasses, const EGraph &egraph)
{
    uint32_t canon = egraph.findConst(eclassId);
    if (protectedEClasses.count(canon))
        return true;
    for (uint32_t id : protectedEClasses)
    {
        if (egraph.findConst(id) == canon)
            return true;
    }
    return false;
}

struct RuleCtx
{
    EGraph &egraph;
    const std::unordered_set<uint32_t> &protectedEClasses;
    std::unordered_map<uint32_t, uint32_t> &eclassToLogical;
    Repo *repo;
};

struct Rule
{
    virtual ~Rule() = default;
    virtual std::string name() const = 0;
    virtual bool match(uint32_t eNodeIdx, RuleCtx &ctx) = 0;
    virtual void apply(uint32_t eNodeIdx, RuleCtx &ctx) = 0;
};

inline uint32_t addOpToEGraph(EGraph &egraph, OpType op, const std::vector<uint32_t> &children, const std::vector<uint32_t> &shape, const std::vector<uint64_t> &st, uint64_t viewOffset, DType dtype, Backend backend, uint32_t targetEClass = UINT32_MAX, uint32_t leafId = UINT32_MAX)
{
    uint32_t cls = (targetEClass == UINT32_MAX) ? egraph.addEClass(shape, st, viewOffset, dtype, backend) : targetEClass;

    TensorNode outNode;
    outNode.opType = op;
    outNode.dtype = dtype;
    outNode.setShape(shape);
    outNode.strides = st;
    outNode.viewOffset = viewOffset;
    outNode.backend = backend;

    std::vector<TensorNode> inNodes;
    for (uint32_t c : children)
    {
        const EClass &childCls = egraph.getEClass(egraph.find(c));
        TensorNode in;
        in.opType = OpType::INPUT;
        in.dtype = childCls.dtype;
        in.setShape(childCls.shape);
        in.strides = childCls.strides;
        in.viewOffset = childCls.viewOffset;
        in.backend = childCls.backend;
        inNodes.push_back(in);
    }

    Graph pGraph;
    std::vector<uint32_t> pInputs;
    for (auto &in : inNodes)
    {
        pInputs.push_back(pGraph.input(in.getShape(), in.dtype));
    }
    uint32_t pRoot = UINT32_MAX;
    if (op == OpType::SLICE)
        pRoot = pGraph.slice(pInputs[0], pInputs[1], pInputs[2], pInputs[3]);
    else if (op == OpType::CONTIGUOUS)
        pRoot = pGraph.contiguous(pInputs[0]);
    else if (op == OpType::ADD)
        pRoot = pGraph.add(pInputs[0], pInputs[1]);
    else if (op == OpType::MUL)
        pRoot = pGraph.mul(pInputs[0], pInputs[1]);
    else if (op == OpType::DIVIDE)
        pRoot = pGraph.div(pInputs[0], pInputs[1]);
    else if (op == OpType::POWER)
        pRoot = pGraph.pow(pInputs[0], pInputs[1]);
    else if (op == OpType::SIN)
        pRoot = pGraph.sin(pInputs[0]);
    else if (op == OpType::COS)
        pRoot = pGraph.cos(pInputs[0]);
    else if (op == OpType::NEGATE)
        pRoot = pGraph.neg(pInputs[0]);
    else if (op == OpType::CAST)
        pRoot = pGraph.cast(pInputs[0], dtype);
    else if (op == OpType::DOT)
        pRoot = pGraph.dot(pInputs[0], pInputs[1]);
    else if (op == OpType::COPY_TO)
        pRoot = pGraph.copyto(pInputs[0], backend);
    else if (op == OpType::SCATTER)
        pRoot = pGraph.scatter(pInputs[0], pInputs[1], pInputs[2], pInputs[3], pInputs[4]);
    else if (op == OpType::RESHAPE)
        pRoot = pGraph.reshape(pInputs[0], pInputs[1]);
    else if (op == OpType::PERMUTE)
        pRoot = pGraph.permute(pInputs[0], pInputs[1]);
    else if (op == OpType::CONCAT)
    {
        std::vector<uint32_t> concatIns;
        for (size_t i = 0; i < pInputs.size() - 1; ++i)
            concatIns.push_back(pInputs[i]);
        pRoot = pGraph.concat(concatIns, pInputs.back());
    }
    else if (op == OpType::REPEAT)
        pRoot = pGraph.repeat(pInputs[0], pInputs[1], pInputs[2]);
    else if (op == OpType::ARANGE)
        pRoot = pGraph.arange(pInputs[0], pInputs[1], pInputs[2]);
    else if (op == OpType::TRIU)
        pRoot = pGraph.triu(pInputs[0], pInputs[1]);
    else if (op == OpType::GATHER)
        pRoot = pGraph.gather(pInputs[0], pInputs[1]);
    else if (op == OpType::FILL)
        pRoot = pGraph.fill(pInputs[0], pInputs[1]);
    else if (op == OpType::IM2COL)
        pRoot = pGraph.im2col(pInputs[0], pInputs[1], pInputs[2], pInputs[3]);
    else if (op == OpType::SUM)
        pRoot = pGraph.sum(pInputs[0], pInputs[1]);
    else if (op == OpType::MAX)
        pRoot = pGraph.max(pInputs[0], pInputs[1]);
    // TODO: argmax
    else if (op == OpType::LT)
        pRoot = pGraph.lt(pInputs[0], pInputs[1]);
    else if (op == OpType::EQ)
        pRoot = pGraph.eq(pInputs[0], pInputs[1]);
    else if (op == OpType::AND)
        pRoot = pGraph.logical_and(pInputs[0], pInputs[1]);
    else if (op == OpType::OR)
        pRoot = pGraph.logical_or(pInputs[0], pInputs[1]);
    else if (op == OpType::NOT)
        pRoot = pGraph.logical_not(pInputs[0]);

    if (pRoot != UINT32_MAX)
    {
        auto matches = KernelRegistry::get().findMatchingKernelsByPattern(pGraph, pRoot, backend, inNodes, outNode, false);
        if (matches.empty())
        {
            std::stringstream ss;
            ss << "\n[addOpToEGraph] No matching kernel found for the given configuration!\n"
               << "  Operation:       " << toString(op) << "\n"
               << "  Target Backend:  " << toString(backend) << "\n"
               << "  Expected Output: "
               << "dtype=" << toString(dtype)
               << ", shape=" << toString(shape)
               << ", strides=" << toString(st)
               << ", viewOffset=" << viewOffset << "\n"
               << "  Inputs (" << inNodes.size() << "):\n";
            for (size_t i = 0; i < inNodes.size(); ++i)
            {
                ss << "    Input #" << i << ": "
                   << "dtype=" << toString(inNodes[i].dtype)
                   << ", shape=" << toString(inNodes[i].getShape())
                   << ", strides=" << toString(inNodes[i].strides)
                   << ", backend=" << toString(inNodes[i].backend)
                   << ", viewOffset=" << inNodes[i].viewOffset << "\n";
            }

            ss << "  Available Registered Kernels for " << toString(op) << ":\n";
            bool foundAny = false;
            for (const auto &[uid, k] : KernelRegistry::get().getAllKernels())
            {
                if (k.opType == op || (op == OpType::FUSED && k.opType == OpType::FUSED))
                {
                    foundAny = true;
                    ss << "    - " << k.getName() << " (backends: ";
                    for (size_t b = 0; b < k.backends.size(); ++b)
                    {
                        ss << toString(k.backends[b]) << (b + 1 < k.backends.size() ? "," : "");
                    }
                    ss << ")\n";
                }
            }
            if (!foundAny)
            {
                ss << "    - None registered in the system.\n";
            }

            Error::throw_err(ss.str());
        }
        for (uint64_t uid : matches)
        {
            const auto &kernel = KernelRegistry::get().getKernel(uid);
            ENode n;
            n.kernelUid = uid;
            n.opType = op;
            n.opName = kernel.opName;
            n.children = children;
            n.shape = shape;
            n.dtype = dtype;
            n.backend = backend;
            n.leafId = leafId;

            if (kernel.isView || kernel.inplace || op == OpType::COPY_TO)
            {
                n.strides = st;
                n.viewOffset = viewOffset;
            }
            else
            {
                n.strides = calcContiguousStrides(shape);
                n.viewOffset = 0;
            }

            egraph.addENode(cls, n);
        }
    }
    return cls;
}

inline uint32_t copyToBackend(EGraph &egraph, uint32_t classId, Backend targetBackend)
{
    uint32_t canon = egraph.find(classId);
    const EClass &cls = egraph.getEClass(canon);
    if (cls.backend == targetBackend)
        return canon;

    return addOpToEGraph(egraph, OpType::COPY_TO, {canon}, cls.shape, cls.strides, cls.viewOffset, cls.dtype, targetBackend);
}

inline uint32_t createCacheInputNode(EGraph &egraph, uint32_t sourceClassId, uint32_t partialPathId, std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
{
    uint32_t canonSrcClass = egraph.find(sourceClassId);
    const EClass srcClass = egraph.getEClass(canonSrcClass);

    uint32_t op_cache = egraph.addEClass(srcClass.shape, srcClass.strides, srcClass.viewOffset, srcClass.dtype, srcClass.backend);
    ENode cacheNode;
    cacheNode.kernelUid = 0;
    cacheNode.opType = OpType::CACHE;
    cacheNode.shape = srcClass.shape;
    cacheNode.strides = srcClass.strides;
    cacheNode.viewOffset = srcClass.viewOffset;
    cacheNode.dtype = srcClass.dtype;
    cacheNode.backend = srcClass.backend;
    cacheNode.leafId = partialPathId;
    op_cache = egraph.addENode(op_cache, cacheNode);

    uint32_t srcLogicalId = UINT32_MAX;
    auto it = eclassToLogical.find(canonSrcClass);
    if (it != eclassToLogical.end())
    {
        srcLogicalId = it->second;
    }
    else
    {
        for (const auto &kv : eclassToLogical)
        {
            if (egraph.find(kv.first) == canonSrcClass)
            {
                srcLogicalId = kv.second;
                break;
            }
        }
    }

    eclassToLogical[op_cache] = srcLogicalId;

    return op_cache;
}

std::vector<int32_t> getConstInt32(const EGraph &egraph, uint32_t eclassId)
{
    uint32_t canon = egraph.findConst(eclassId);
    if (egraph.constantStaging.count(canon))
    {
        const auto &data = *egraph.constantStaging.at(canon);
        const EClass &cls = egraph.getEClass(canon);
        uint64_t numElements = countElements(cls.shape);
        std::vector<int32_t> res(numElements);
        const int32_t *src = reinterpret_cast<const int32_t *>(data.data()) + cls.viewOffset;
        for (uint64_t i = 0; i < numElements; ++i)
        {
            res[i] = src[getStridedIndex(i, cls.shape, cls.strides)];
        }
        return res;
    }
    return {};
}

struct FusionRule : public Rule
{
    std::string name() const override { return "FusionRule"; }

    struct Pattern
    {
        std::string opName;
        OpType rootOpType;
        uint32_t rootId;
        std::vector<uint32_t> variables;
        std::vector<DType> dtypes;
        std::vector<std::vector<uint32_t>> dummyShapes;
        Graph graph;
    };

    struct MatchResult
    {
        const Pattern *pattern;
        std::unordered_map<uint32_t, uint32_t> binding;
        std::vector<uint32_t> variadicConcatTensorEClasses;
    };

    std::unordered_map<OpType, std::vector<Pattern>> patternsByOp;
    std::vector<MatchResult> activeMatches;

    FusionRule(bool disableFusion = false)
    {
        const auto &refGraphs = ReferenceGraphRegistry::get().getAll();
        for (const auto &pair : refGraphs)
        {
            Pattern pattern;
            pattern.opName = pair.first;
            const auto &entry = pair.second;

            for (size_t i = 0; i < entry.numInputs; ++i)
            {
                uint32_t inId = pattern.graph.input(entry.dummyShapes[i], entry.dtypes[i]);
                pattern.variables.push_back(inId);
            }
            pattern.rootId = entry.factory(pattern.variables, pattern.graph);

            if (disableFusion && pattern.graph.nodes.size() > entry.numInputs + 1)
            {
                continue;
            }

            pattern.rootOpType = pattern.graph.getNode(pattern.rootId).opType;
            pattern.dtypes = entry.dtypes;
            pattern.dummyShapes = entry.dummyShapes;

            patternsByOp[pattern.rootOpType].push_back(std::move(pattern));
        }
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        activeMatches.clear();
        const EGraph &egraph = ctx.egraph;
        const ENode &eNode = egraph.getENodes()[eNodeIdx];

        auto it = patternsByOp.find(eNode.opType);
        if (it == patternsByOp.end())
            return false;

        for (const auto &pattern : it->second)
        {
            std::unordered_map<uint32_t, uint32_t> binding;
            if (matchPatternNode(eNodeIdx, egraph, pattern.rootId, pattern, binding, ctx.protectedEClasses))
            {
                MatchResult mr;
                mr.pattern = &pattern;
                mr.binding = std::move(binding);

                if (eNode.opType == OpType::CONCAT && eNode.children.size() > 2)
                {
                    for (size_t i = 0; i < eNode.children.size() - 1; ++i)
                    {
                        mr.variadicConcatTensorEClasses.push_back(egraph.findConst(eNode.children[i]));
                    }
                }

                activeMatches.push_back(std::move(mr));
            }
        }
        return !activeMatches.empty();
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        for (const auto &match : activeMatches)
        {
            const Pattern &pattern = *match.pattern;
            const auto &binding = match.binding;

            std::vector<uint32_t> inputs;
            std::vector<TensorNode> inputNodes;

            if (!match.variadicConcatTensorEClasses.empty())
            {
                for (uint32_t tensorEClass : match.variadicConcatTensorEClasses)
                {
                    inputs.push_back(tensorEClass);
                    const EClass parent = egraph.getEClass(tensorEClass);
                    TensorNode inputNode;
                    inputNode.opType = OpType::INPUT;
                    inputNode.dtype = parent.dtype;
                    inputNode.setShape(parent.shape);
                    inputNode.strides = parent.strides;
                    inputNode.viewOffset = parent.viewOffset;
                    inputNode.backend = parent.backend;
                    inputNodes.push_back(std::move(inputNode));
                }
                uint32_t axisVar = pattern.variables.back();
                uint32_t axisEClass = binding.at(axisVar);
                inputs.push_back(axisEClass);
                const EClass axisParent = egraph.getEClass(axisEClass);
                TensorNode axisInputNode;
                axisInputNode.opType = OpType::INPUT;
                axisInputNode.dtype = axisParent.dtype;
                axisInputNode.setShape(axisParent.shape);
                axisInputNode.strides = axisParent.strides;
                axisInputNode.viewOffset = axisParent.viewOffset;
                axisInputNode.backend = axisParent.backend;
                inputNodes.push_back(std::move(axisInputNode));
            }
            else
            {
                inputs.reserve(pattern.variables.size());
                inputNodes.reserve(pattern.variables.size());

                for (uint32_t var : pattern.variables)
                {
                    uint32_t parentEClassId = binding.at(var);
                    const EClass parent = egraph.getEClass(parentEClassId);
                    inputs.push_back(parentEClassId);

                    TensorNode inputNode;
                    inputNode.opType = OpType::INPUT;
                    inputNode.dtype = parent.dtype;
                    inputNode.setShape(parent.shape);
                    inputNode.strides = parent.strides;
                    inputNode.viewOffset = parent.viewOffset;
                    inputNode.backend = parent.backend;
                    inputNodes.push_back(std::move(inputNode));
                }
            }

            const EClass matchedClass = egraph.getEClass(egraph.getENodeEClass(eNodeIdx));

            std::vector<Backend> targetBackends = {Backend::CPU};
#ifdef USE_CUDA
            targetBackends.push_back(Backend::CUDA);
#endif

            DType outDtype = matchedClass.dtype;
            std::vector<uint32_t> outShape = matchedClass.shape;
            std::vector<uint64_t> outStrides = matchedClass.strides;
            uint64_t outViewOffset = matchedClass.viewOffset;

            for (Backend targetBackend : targetBackends)
            {
                TensorNode outputNode;
                outputNode.opType = OpType::FUSED;
                outputNode.opName = pattern.opName;
                outputNode.dtype = outDtype;
                outputNode.setShape(outShape);
                outputNode.strides = outStrides;
                outputNode.viewOffset = outViewOffset;
                outputNode.backend = targetBackend;

                bool ignoreInputBackends = (pattern.rootOpType != OpType::COPY_TO);

                std::vector<uint64_t> kernelMatches = KernelRegistry::get().findMatchingKernels(
                    OpType::FUSED, pattern.opName, targetBackend, inputNodes, outputNode, false, ignoreInputBackends, true);

                for (uint64_t uid : kernelMatches)
                {
                    const KernelEntry &kernel = KernelRegistry::get().getKernel(uid);
                    addFusedNode(egraph, kernel, targetBackend, inputs, eNodeIdx);
                }
            }
        }
    }

    void addFusedNode(EGraph &egraph, const KernelEntry &kernel, Backend targetBackend, const std::vector<uint32_t> &parentIds, uint32_t eNodeIdx) const
    {
        std::vector<uint32_t> adaptedParents;
        if (!kernel.isVariadic && parentIds.size() != kernel.numInputs)
        {
            Error::throw_err("[addFusedNode] parentIds.size() != kernel.numInputs. Info:\n  Kernel: " + kernel.opName + "\n" +
                             "  Parent IDs: " + std::to_string(parentIds.size()) + "\n" +
                             "  Kernel Num Inputs: " + std::to_string(kernel.numInputs) + "\n");
        }
        if (kernel.isVariadic && parentIds.size() < 2)
        {
            Error::throw_err("[addFusedNode] variadic kernel requires at least 2 parentIds. Info:\n  Kernel: " + kernel.opName + "\n" +
                             "  Parent IDs: " + std::to_string(parentIds.size()) + "\n");
        }

        for (size_t i = 0; i < parentIds.size(); ++i)
        {
            uint32_t pid = parentIds[i];
            const EClass parent = egraph.getEClass(pid);

            size_t ruleIdx = kernel.isVariadic ? (i == parentIds.size() - 1 ? 1 : 0) : i;

            Backend expectedBackend = kernel.inputBackends[ruleIdx][0];
            bool foundBackend = false;
            for (Backend b : kernel.inputBackends[ruleIdx])
            {
                if (parent.backend == b)
                {
                    expectedBackend = parent.backend;
                    foundBackend = true;
                    break;
                }
            }

            bool needCopy = !foundBackend;
            bool needContig = kernel.requiresContiguous[ruleIdx] && !isContiguous(parent);

            if (!needCopy && !needContig)
            {
                adaptedParents.push_back(pid);
                continue;
            }

            uint32_t currentPid = pid;
            EClass currentClass = parent;

            if (needCopy)
            {
                currentPid = addOpToEGraph(egraph, OpType::COPY_TO, {currentPid}, currentClass.shape, currentClass.strides, currentClass.viewOffset, currentClass.dtype, expectedBackend);
                currentClass = egraph.getEClass(egraph.find(currentPid));
            }

            if (needContig)
            {
                currentPid = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentPid}, currentClass.shape, calcContiguousStrides(currentClass.shape), 0, currentClass.dtype, currentClass.backend);
            }
            adaptedParents.push_back(currentPid);
        }

        const ENode oldENode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        ENode enode;
        enode.kernelUid = kernel.uid;
        enode.opType = kernel.opType;
        enode.opName = kernel.opName;
        enode.children = adaptedParents;
        enode.shape = oldENode.shape;

        if (kernel.isView)
        {
            enode.strides = oldENode.strides;
            enode.viewOffset = oldENode.viewOffset;
        }
        else
        {
            enode.strides = calcContiguousStrides(oldENode.shape);
            enode.viewOffset = 0;
        }

        enode.dtype = oldENode.dtype;
        enode.backend = targetBackend;

        Backend originalBackend = egraph.getEClass(eclassId).backend;
        if (targetBackend == originalBackend)
        {
            egraph.addENode(eclassId, enode);
        }
        else
        {
            uint32_t newEClass = egraph.addEClass(enode.shape, enode.strides, enode.viewOffset, enode.dtype, targetBackend);
            newEClass = egraph.addENode(newEClass, enode);

            addOpToEGraph(egraph, OpType::COPY_TO, {newEClass}, enode.shape, enode.strides, enode.viewOffset, enode.dtype, originalBackend, eclassId);
        }
    }

    static bool isStructuralConstant(OpType op, size_t inputIdx, size_t numInputs)
    {
        if (op == OpType::REPEAT && (inputIdx == 1 || inputIdx == 2))
            return true;
        if (op == OpType::RESHAPE && inputIdx == 1)
            return true;
        if (op == OpType::PERMUTE && inputIdx == 1)
            return true;
        if (op == OpType::SLICE && (inputIdx == 1 || inputIdx == 2 || inputIdx == 3))
            return true;
        if (op == OpType::SCATTER && (inputIdx == 2 || inputIdx == 3 || inputIdx == 4))
            return true;
        if ((op == OpType::SUM || op == OpType::MAX) && inputIdx == 1)
            return true;
        if (op == OpType::CONCAT && inputIdx == numInputs - 1)
            return true;
        if (op == OpType::TRIU && inputIdx == 1)
            return true;
        if (op == OpType::FILL && inputIdx == 1)
            return true;
        if (op == OpType::IM2COL && (inputIdx == 1 || inputIdx == 2 || inputIdx == 3))
            return true;
        if (op == OpType::ARANGE && (inputIdx == 0 || inputIdx == 1 || inputIdx == 2))
            return true;
        if (op == OpType::ARGMAX && (inputIdx == 1 || inputIdx == 2))
            return true;
        return false;
    }

    static bool matchPatternClass(uint32_t eClassIdx, const EGraph &egraph,
                                  uint32_t patternId, const Pattern &pattern,
                                  std::unordered_map<uint32_t, uint32_t> &binding,
                                  const std::unordered_set<uint32_t> &protectedEClasses,
                                  bool ignoreConstantData = false)
    {
        uint32_t canonicalClassIdx = egraph.findConst(eClassIdx);

        auto itVar = std::find(pattern.variables.begin(), pattern.variables.end(), patternId);
        if (itVar != pattern.variables.end())
        {
            size_t varIdx = static_cast<size_t>(std::distance(pattern.variables.begin(), itVar));
            const EClass &eclass = egraph.getEClass(canonicalClassIdx);

            if (varIdx < pattern.dtypes.size() && eclass.dtype != pattern.dtypes[varIdx])
                return false;

            auto bIt = binding.find(patternId);
            if (bIt != binding.end())
            {
                return bIt->second == canonicalClassIdx;
            }
            binding[patternId] = canonicalClassIdx;
            return true;
        }

        if (patternId != pattern.rootId)
        {
            if (isEClassProtected(canonicalClassIdx, protectedEClasses, egraph))
                return false;
        }

        const EClass &eclass = egraph.getEClass(canonicalClassIdx);
        for (uint32_t enodeId : eclass.enodes)
        {
            std::unordered_map<uint32_t, uint32_t> localBinding = binding;
            if (matchPatternNode(enodeId, egraph, patternId, pattern, localBinding, protectedEClasses, ignoreConstantData))
            {
                binding = std::move(localBinding);
                return true;
            }
        }
        return false;
    }

    static bool matchPatternNode(uint32_t eNodeIdx, const EGraph &egraph,
                                 uint32_t patternId, const Pattern &pattern,
                                 std::unordered_map<uint32_t, uint32_t> &binding,
                                 const std::unordered_set<uint32_t> &protectedEClasses,
                                 bool ignoreConstantData = false)
    {
        const ENode &eNode = egraph.getENodes()[eNodeIdx];
        const auto &pNode = pattern.graph.getNode(patternId);

        if (eNode.opType != pNode.opType)
            return false;
        if (eNode.opType == OpType::FUSED && eNode.opName != pNode.opName)
            return false;

        if (eNode.opType == OpType::INPUT && !pNode.contentHash.empty())
        {
            if (!ignoreConstantData)
            {
                uint32_t eNodeEClass = egraph.getENodeEClass(eNodeIdx);
                uint32_t canonEClass = egraph.findConst(eNodeEClass);

                auto egraphIt = egraph.constantStaging.find(canonEClass);
                if (egraphIt == egraph.constantStaging.end())
                    return false;

                auto patternIt = pattern.graph.constantStaging.find(patternId);
                if (patternIt == pattern.graph.constantStaging.end())
                    return false;

                const auto &egraphData = *egraphIt->second;
                const auto &patternData = *patternIt->second;
                if (egraphData.size() != patternData.size())
                    return false;
                if (std::memcmp(egraphData.data(), patternData.data(), egraphData.size()) != 0)
                    return false;
            }
        }

        if (eNode.opType == OpType::CONCAT && eNode.children.size() != pNode.parentIds.size())
        {
            if (pNode.parentIds.size() < 2)
                return false;
            if (eNode.children.size() < 2)
                return false;

            if (!matchPatternClass(eNode.children.back(), egraph,
                                   pNode.parentIds.back(), pattern, binding, protectedEClasses, true))
                return false;

            bool firstTensor = true;
            for (size_t i = 0; i < eNode.children.size() - 1; ++i)
            {
                if (firstTensor)
                {
                    if (!matchPatternClass(eNode.children[i], egraph,
                                           pNode.parentIds[0], pattern, binding, protectedEClasses, false))
                        return false;
                    firstTensor = false;
                }
                else
                {
                    uint32_t canonChild = egraph.findConst(eNode.children[i]);
                    const EClass &childCls = egraph.getEClass(canonChild);
                    if (childCls.dtype != pattern.dtypes[0])
                        return false;
                }
            }
            return true;
        }

        if (eNode.children.size() != pNode.parentIds.size())
            return false;

        for (size_t i = 0; i < eNode.children.size(); ++i)
        {
            bool childIgnoreConst = isStructuralConstant(eNode.opType, i, eNode.children.size());
            if (!matchPatternClass(eNode.children[i], egraph, pNode.parentIds[i], pattern, binding, protectedEClasses, childIgnoreConst))
            {
                return false;
            }
        }
        return true;
    }
};

struct InfinityDomination : public Rule
{
    std::unordered_set<uint32_t> visited_enodes;

    std::string name() const override { return "InfinityDomination"; }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited_enodes.count(eNodeIdx))
            return false;

        const ENode &enode = ctx.egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::ADD || enode.children.size() != 2)
            return false;

        return isConstantFloat(enode.children[0], ctx) || isConstantFloat(enode.children[1], ctx);
    }

    bool isConstantFloat(uint32_t eclassId, RuleCtx &ctx) const
    {
        uint32_t canon = ctx.egraph.findConst(eclassId);
        const EClass &cls = ctx.egraph.getEClass(canon);
        if (cls.dtype != DType::FLOAT32)
            return false;
        if (ctx.egraph.constantStaging.find(canon) != ctx.egraph.constantStaging.end())
            return true;

        if (ctx.repo && ctx.repo->isValid())
        {
            uint32_t logicalId = ctx.eclassToLogical.count(canon) ? ctx.eclassToLogical.at(canon) : UINT32_MAX;
            if (logicalId != UINT32_MAX && ctx.repo->has(logicalId))
            {
                auto data = ctx.repo->read(logicalId);
                ctx.egraph.constantStaging[canon] = std::make_shared<std::vector<uint8_t>>(std::move(data));
                return true;
            }
        }
        return false;
    }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        visited_enodes.insert(eNodeIdx);

        const ENode addNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t constIdx = isConstantFloat(addNode.children[1], ctx) ? 1 : 0;
        uint32_t varIdx = 1 - constIdx;

        uint32_t constClass = egraph.find(addNode.children[constIdx]);
        uint32_t varClass = egraph.find(addNode.children[varIdx]);

        const auto &constData = *egraph.constantStaging.at(constClass);
        const float *data = reinterpret_cast<const float *>(constData.data());

        const EClass cClass = egraph.getEClass(constClass);
        uint64_t numElements = countElements(cClass.shape);

        std::vector<Region> nonInfRegions;
        bool noneInf = true;
        for (uint64_t i = 0; i < numElements; ++i)
        {
            uint64_t flat_idx = getStridedIndex(i, cClass.shape, cClass.strides) + cClass.viewOffset;
            if (data[flat_idx] > -1e8f)
            {
                auto coords = coordsFromFlatIndex(i, cClass.shape);
                Region r;
                for (uint32_t c : coords)
                {
                    r.region.push_back({c, c + 1});
                }
                nonInfRegions.push_back(r);
            }
            else
            {
                noneInf = false;
            }
        }
        if (noneInf)
        {
            return;
        }
        nonInfRegions = mergeRegions(nonInfRegions);

        const EClass vClass = egraph.getEClass(varClass);
        const EClass outClass = egraph.getEClass(eclassId);

        std::vector<uint64_t> contigStrides = calcContiguousStrides(outClass.shape);

        if (nonInfRegions.empty())
        {
            uint32_t currentTarget = constClass;
            if (cClass.backend != outClass.backend || cClass.strides != contigStrides || cClass.viewOffset != 0)
            {
                if (cClass.backend != outClass.backend)
                {
                    currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {constClass}, outClass.shape, contigStrides, 0, outClass.dtype, outClass.backend);
                }
                else
                {
                    currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {constClass}, outClass.shape, contigStrides, 0, outClass.dtype, outClass.backend);
                }
            }
            egraph.merge(eclassId, currentTarget);
            return;
        }

        if (nonInfRegions.size() == 1)
        {
            bool strictlySmaller = false;
            const Region &reg = nonInfRegions[0];
            for (size_t d = 0; d < cClass.shape.size(); ++d)
            {
                if (reg.region[d].start > 0 || reg.region[d].stop < cClass.shape[d])
                {
                    strictlySmaller = true;
                    break;
                }
            }
            if (!strictlySmaller)
                return;
        }

        uint32_t currentTarget;
        if (cClass.backend != outClass.backend)
        {
            currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {constClass}, outClass.shape, contigStrides, 0, outClass.dtype, outClass.backend);
        }
        else
        {
            currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {constClass}, outClass.shape, contigStrides, 0, outClass.dtype, outClass.backend);
        }

        for (const Region &reg : nonInfRegions)
        {
            std::vector<int32_t> starts, ends, steps;
            for (const Dim &d : reg.region)
            {
                starts.push_back(d.start);
                ends.push_back(d.stop);
                steps.push_back(1);
            }

            uint32_t startsId = addIntConst(egraph, starts);
            uint32_t endsId = addIntConst(egraph, ends);
            uint32_t stepsId = addIntConst(egraph, steps);

            std::vector<uint32_t> sliceShape;
            for (size_t d = 0; d < starts.size(); ++d)
            {
                sliceShape.push_back(ends[d] - starts[d]);
            }

            std::vector<uint64_t> sliceStridesV = vClass.strides;
            uint64_t sliceViewOffsetV = vClass.viewOffset;
            for (size_t d = 0; d < starts.size(); ++d)
            {
                sliceViewOffsetV += starts[d] * sliceStridesV[d];
                sliceStridesV[d] *= steps[d];
            }
            uint32_t sliceV = addOpToEGraph(egraph, OpType::SLICE, {varClass, startsId, endsId, stepsId}, sliceShape, sliceStridesV, sliceViewOffsetV, vClass.dtype, vClass.backend);

            std::vector<uint64_t> sliceStridesC = cClass.strides;
            uint64_t sliceViewOffsetC = cClass.viewOffset;
            for (size_t d = 0; d < starts.size(); ++d)
            {
                sliceViewOffsetC += starts[d] * sliceStridesC[d];
                sliceStridesC[d] *= steps[d];
            }
            uint32_t sliceC = addOpToEGraph(egraph, OpType::SLICE, {constClass, startsId, endsId, stepsId}, sliceShape, sliceStridesC, sliceViewOffsetC, cClass.dtype, cClass.backend);

            std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);
            uint32_t contigV = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceV}, sliceShape, sliceContigStrides, 0, vClass.dtype, vClass.backend);
            uint32_t contigC = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceC}, sliceShape, sliceContigStrides, 0, cClass.dtype, cClass.backend);

            contigV = copyToBackend(egraph, contigV, outClass.backend);
            contigC = copyToBackend(egraph, contigC, outClass.backend);

            uint32_t child0 = (constIdx == 0) ? contigC : contigV;
            uint32_t child1 = (constIdx == 1) ? contigC : contigV;
            uint32_t addId = addOpToEGraph(egraph, OpType::ADD, {child0, child1}, sliceShape, sliceContigStrides, 0, outClass.dtype, outClass.backend);

            currentTarget = addOpToEGraph(egraph, OpType::SCATTER, {currentTarget, addId, startsId, endsId, stepsId}, outClass.shape, outClass.strides, outClass.viewOffset, outClass.dtype, outClass.backend);
        }

        egraph.merge(eclassId, currentTarget);
    }
};

/*
contiguous(slice(op(x))) -> scatter(cache, contiguous(op(contiguous(slice(x)))))
op can be: add, mul, div, pow, sin, cos, neg, cast
*/
struct SlicePushDownElementwise : public Rule
{
    struct MatchKey
    {
        uint32_t contigIdx;
        uint32_t sliceIdx;
        uint32_t srcNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigIdx == o.contigIdx && sliceIdx == o.sliceIdx && srcNodeIdx == o.srcNodeIdx;
        }
    };
    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.contigIdx) ^ (std::hash<uint32_t>{}(k.sliceIdx) << 1) ^ (std::hash<uint32_t>{}(k.srcNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;
    SlicePushDownElementwise(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected) {}

    std::string name() const override { return "SlicePushDownElementwise"; }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;

        uint32_t childClass = egraph.findConst(enode.children[0]);
        for (uint32_t childNodeIdx : egraph.getEClass(childClass).enodes)
        {
            const ENode &childNode = egraph.getENodes()[childNodeIdx];
            if (childNode.opType == OpType::SLICE && childNode.children.size() == 4)
            {
                uint32_t srcClass = egraph.findConst(childNode.children[0]);

                auto starts = getConstInt32(egraph, childNode.children[1]);
                auto ends = getConstInt32(egraph, childNode.children[2]);
                auto steps = getConstInt32(egraph, childNode.children[3]);

                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                const auto &origShape = egraph.getEClass(srcClass).shape;
                bool isFull = true;
                if (starts.size() != origShape.size())
                    isFull = false;
                for (size_t d = 0; d < starts.size() && isFull; ++d)
                {
                    int32_t st = starts[d] < 0 ? starts[d] + origShape[d] : starts[d];
                    int32_t en = ends[d] < 0 ? ends[d] + origShape[d] : ends[d];
                    if (st != 0 || en != (int32_t)origShape[d] || steps[d] != 1)
                    {
                        isFull = false;
                    }
                }
                if (isFull)
                    continue;

                if (!allowPushDownOnProtected && isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                    continue;

                for (uint32_t srcNodeIdx : egraph.getEClass(srcClass).enodes)
                {
                    const ENode &opNode = egraph.getENodes()[srcNodeIdx];
                    OpType op = opNode.opType;
                    if (!(op == OpType::ADD || op == OpType::MUL || op == OpType::DIVIDE || op == OpType::POWER ||
                          op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST ||
                          op == OpType::LT || op == OpType::EQ || op == OpType::AND || op == OpType::OR || op == OpType::NOT))
                        continue;

                    bool hasBroadcastChild = false;
                    for (uint32_t cid : opNode.children)
                    {
                        const auto &cls = egraph.getEClass(egraph.findConst(cid));
                        if (cls.shape != opNode.shape)
                        {
                            hasBroadcastChild = true;
                            break;
                        }
                    }
                    if (!hasBroadcastChild)
                    {
                        MatchKey key{eNodeIdx, childNodeIdx, srcNodeIdx};
                        if (visited.find(key) == visited.end())
                            return true;
                    }
                }
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t sliceClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> sliceNodes;
        for (uint32_t childNodeIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &childNode = egraph.getENodes()[childNodeIdx];
            if (childNode.opType == OpType::SLICE && childNode.children.size() == 4)
            {
                sliceNodes.push_back(childNodeIdx);
            }
        }

        for (uint32_t sliceNodeIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceNodeIdx];

            uint32_t srcClass = egraph.find(sliceNode.children[0]);
            uint32_t startsId = sliceNode.children[1];
            uint32_t endsId = sliceNode.children[2];
            uint32_t stepsId = sliceNode.children[3];

            auto starts = getConstInt32(egraph, startsId);
            auto ends = getConstInt32(egraph, endsId);
            auto steps = getConstInt32(egraph, stepsId);
            if (starts.empty() || ends.empty() || steps.empty())
                continue;

            const std::vector<uint32_t> sliceShape = sliceNode.shape;
            std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

            std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;

            for (uint32_t srcNodeIdx : srcEnodes)
            {
                const ENode opNode = egraph.getENodes()[srcNodeIdx];
                OpType op = opNode.opType;
                if (!(op == OpType::ADD || op == OpType::MUL || op == OpType::DIVIDE || op == OpType::POWER ||
                      op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST ||
                      op == OpType::LT || op == OpType::EQ || op == OpType::AND || op == OpType::OR || op == OpType::NOT))
                {
                    continue;
                }

                MatchKey key{eNodeIdx, sliceNodeIdx, srcNodeIdx};
                if (!visited.insert(key).second)
                    continue;

                uint32_t partialPathId = srcNodeIdx | 0x80000000;

                std::vector<uint32_t> newChildren;
                for (uint32_t childId : opNode.children)
                {
                    uint32_t canonChildId = egraph.find(childId);
                    std::vector<uint64_t> childSliceStrides = egraph.getEClass(canonChildId).strides;
                    uint64_t childSliceViewOffset = egraph.getEClass(canonChildId).viewOffset;
                    std::vector<uint32_t> childShape = egraph.getEClass(canonChildId).shape;
                    DType childDtype = egraph.getEClass(canonChildId).dtype;

                    for (size_t d = 0; d < starts.size() && d < childShape.size(); ++d)
                    {
                        int32_t start = starts[d];
                        if (start < 0)
                            start += childShape[d];
                        childSliceViewOffset += start * childSliceStrides[d];
                        childSliceStrides[d] *= steps[d];
                    }

                    uint32_t childSlice = addOpToEGraph(egraph, OpType::SLICE, {canonChildId, startsId, endsId, stepsId}, sliceShape, childSliceStrides, childSliceViewOffset, childDtype, sliceNode.backend, UINT32_MAX, partialPathId);

                    uint32_t childContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {childSlice}, sliceShape, sliceContigStrides, 0, childDtype, sliceNode.backend, UINT32_MAX, partialPathId);
                    newChildren.push_back(childContig);
                }

                uint32_t opEClass = addOpToEGraph(egraph, op, newChildren, sliceShape, sliceContigStrides, 0, sliceNode.dtype, sliceNode.backend, UINT32_MAX, partialPathId);

                uint32_t contigSlicedOp = addOpToEGraph(egraph, OpType::CONTIGUOUS, {opEClass}, sliceShape, sliceContigStrides, 0, sliceNode.dtype, sliceNode.backend, UINT32_MAX, partialPathId);

                uint32_t op_cache = createCacheInputNode(egraph, srcClass, partialPathId, ctx.eclassToLogical);

                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, contigSlicedOp, startsId, endsId, stepsId}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

struct SlicePushDownDot : public Rule
{
    struct MatchKey
    {
        uint32_t eNodeIdx;
        uint32_t childNodeIdx;
        uint32_t srcNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && childNodeIdx == o.childNodeIdx && srcNodeIdx == o.srcNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^
                   (std::hash<uint32_t>{}(k.childNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.srcNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownDot(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected) {}

    std::string name() const override { return "SlicePushDownDot"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;

        uint32_t childClass = egraph.findConst(enode.children[0]);
        for (uint32_t childNodeIdx : egraph.getEClass(childClass).enodes)
        {
            const ENode &childNode = egraph.getENodes()[childNodeIdx];
            if (childNode.opType == OpType::SLICE && childNode.children.size() == 4)
            {
                uint32_t srcClass = egraph.findConst(childNode.children[0]);

                auto starts = getConstInt32(egraph, childNode.children[1]);
                auto ends = getConstInt32(egraph, childNode.children[2]);
                auto steps = getConstInt32(egraph, childNode.children[3]);

                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                const auto &origShape = egraph.getEClass(srcClass).shape;
                bool isFull = true;
                if (starts.size() != origShape.size())
                    isFull = false;
                for (size_t d = 0; d < starts.size() && isFull; ++d)
                {
                    int32_t st = starts[d] < 0 ? starts[d] + origShape[d] : starts[d];
                    int32_t en = ends[d] < 0 ? ends[d] + origShape[d] : ends[d];
                    if (st != 0 || en != (int32_t)origShape[d] || steps[d] != 1)
                    {
                        isFull = false;
                    }
                }
                if (isFull)
                    continue;

                if (!allowPushDownOnProtected && isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                    continue;

                for (uint32_t srcNodeIdx : egraph.getEClass(srcClass).enodes)
                {
                    if (egraph.getENodes()[srcNodeIdx].opType == OpType::DOT)
                    {
                        MatchKey key{eNodeIdx, childNodeIdx, srcNodeIdx};
                        if (visited.find(key) == visited.end())
                            return true;
                    }
                }
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t sliceClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> sliceNodes;
        for (uint32_t childNodeIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENodes()[childNodeIdx].opType == OpType::SLICE && egraph.getENodes()[childNodeIdx].children.size() == 4)
            {
                sliceNodes.push_back(childNodeIdx);
            }
        }

        for (uint32_t sliceNodeIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceNodeIdx];

            uint32_t srcClass = egraph.find(sliceNode.children[0]);
            std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;

            for (uint32_t srcNodeIdx : srcEnodes)
            {
                const ENode dotNode = egraph.getENodes()[srcNodeIdx];
                if (dotNode.opType != OpType::DOT)
                    continue;

                MatchKey key{eNodeIdx, sliceNodeIdx, srcNodeIdx};
                if (!visited.insert(key).second)
                    continue;

                auto starts = getConstInt32(egraph, sliceNode.children[1]);
                auto ends = getConstInt32(egraph, sliceNode.children[2]);
                auto steps = getConstInt32(egraph, sliceNode.children[3]);

                if (starts.empty() || ends.empty() || steps.empty())
                    Error::throw_err("[SlicePushDownDot.apply] can't find constants for all slice args");

                bool validSteps = true;
                for (int32_t s : steps)
                {
                    if (s != 1)
                        validSteps = false;
                }
                if (!validSteps)
                    continue;

                std::vector<uint32_t> outClassShape = egraph.getEClass(srcClass).shape;
                uint32_t rank = outClassShape.size();
                if (rank != 2 && rank != 3 && rank != 4)
                    continue;

                while (starts.size() < rank)
                    starts.push_back(0);
                while (ends.size() < rank)
                    ends.push_back(outClassShape[ends.size()]);

                for (size_t d = 0; d < rank; ++d)
                {
                    if (starts[d] < 0)
                        starts[d] += outClassShape[d];
                    if (ends[d] < 0)
                        ends[d] += outClassShape[d];
                    starts[d] = std::max(0, starts[d]);
                    ends[d] = std::min((int32_t)outClassShape[d], std::max(starts[d], ends[d]));
                }

                const std::vector<uint32_t> sliceShape = sliceNode.shape;
                std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

                uint32_t aClassId = dotNode.children[0];
                uint32_t bClassId = dotNode.children[1];

                uint32_t K = (rank == 2) ? egraph.getEClass(egraph.find(aClassId)).shape[1] : (rank == 3) ? egraph.getEClass(egraph.find(aClassId)).shape[2]
                                                                                                          : egraph.getEClass(egraph.find(aClassId)).shape[3];

                std::vector<int32_t> startsA, endsA, stepsA(rank, 1);
                std::vector<int32_t> startsB, endsB, stepsB(rank, 1);

                if (rank == 2)
                {
                    startsA = {starts[0], 0};
                    endsA = {ends[0], (int32_t)K};

                    startsB = {0, starts[1]};
                    endsB = {(int32_t)K, ends[1]};
                }
                else if (rank == 3)
                {
                    startsA = {starts[0], starts[1], 0};
                    endsA = {ends[0], ends[1], (int32_t)K};

                    startsB = {starts[0], 0, starts[2]};
                    endsB = {ends[0], (int32_t)K, ends[2]};
                }
                else if (rank == 4)
                {
                    startsA = {starts[0], starts[1], starts[2], 0};
                    endsA = {ends[0], ends[1], ends[2], (int32_t)K};
                    startsB = {starts[0], starts[1], 0, starts[3]};
                    endsB = {ends[0], ends[1], (int32_t)K, ends[3]};
                }
                else
                {
                    Error::throw_err("[SlicePushDownDot.apply] expected rank=2,3,4 got rank=" + std::to_string(rank));
                }

                uint32_t startsIdA = addIntConst(egraph, startsA);
                uint32_t endsIdA = addIntConst(egraph, endsA);
                uint32_t stepsIdA = addIntConst(egraph, stepsA);

                uint32_t startsIdB = addIntConst(egraph, startsB);
                uint32_t endsIdB = addIntConst(egraph, endsB);
                uint32_t stepsIdB = addIntConst(egraph, stepsB);

                auto createSlice = [&](uint32_t classId, const std::vector<int32_t> &st, const std::vector<int32_t> &en, uint32_t stId, uint32_t enId, uint32_t stepId)
                {
                    uint32_t canonId = egraph.find(classId);
                    std::vector<uint64_t> sStrides = egraph.getEClass(canonId).strides;
                    uint64_t sOffset = egraph.getEClass(canonId).viewOffset;
                    DType cDtype = egraph.getEClass(canonId).dtype;

                    std::vector<uint32_t> sShape;
                    for (size_t d = 0; d < st.size(); ++d)
                        sShape.push_back(en[d] - st[d]);

                    for (size_t d = 0; d < st.size(); ++d)
                    {
                        sOffset += st[d] * sStrides[d];
                    }

                    uint32_t sClass = addOpToEGraph(egraph, OpType::SLICE, {canonId, stId, enId, stepId}, sShape, sStrides, sOffset, cDtype, sliceNode.backend);
                    uint32_t sContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sClass}, sShape, calcContiguousStrides(sShape), 0, cDtype, sliceNode.backend);
                    return sContig;
                };

                uint32_t aSliced = createSlice(aClassId, startsA, endsA, startsIdA, endsIdA, stepsIdA);
                uint32_t bSliced = createSlice(bClassId, startsB, endsB, startsIdB, endsIdB, stepsIdB);

                uint32_t dotEClass = egraph.addEClass(sliceShape, sliceContigStrides, 0, sliceNode.dtype, sliceNode.backend);
                addOpToEGraph(egraph, OpType::DOT, {aSliced, bSliced}, sliceShape, sliceContigStrides, 0, sliceNode.dtype, sliceNode.backend, dotEClass);

                uint32_t contigSlicedOp = addOpToEGraph(egraph, OpType::CONTIGUOUS, {dotEClass}, sliceShape, sliceContigStrides, 0, sliceNode.dtype, sliceNode.backend, eclassId);

                uint32_t partialPathId = srcNodeIdx | 0x80000000;
                uint32_t op_cache = createCacheInputNode(egraph, srcClass, partialPathId, ctx.eclassToLogical);

                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, contigSlicedOp, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, dotNode.dtype, dotNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

// =============================================================================
// FlattenBatchDot
// =============================================================================
//
// Rewrites a 4-D batched matmul of the form
//
//     DOT( A[1, H, S, K] , B[1, H, K, S2] )  ->  OUT[1, H, S, S2]
//
// into the equivalent 3-D computation that uses the optimised 3-D DOT kernel:
//
//     rA = reshape(A, [H, S, K])
//     rB = reshape(B, [H, K, S2])
//     rY = DOT(rA, rB)            # 3-D batched matmul -> [H, S, S2]
//     OUT = reshape(rY, [1, H, S, S2])
//
// Why: the 4-D DOT kernel in this repo is the reference path
// (kernels/cpu/reference/dot/F32_4D.hpp) and is dramatically slower than the
// NEON-tuned 3-D path (kernels/cpu/general/dot/arm_neon_F32_3D.hpp,
// F32_3D_NEON.hpp, BF16_*_GEMM_NEON_*).  For the jina-embeddings-v5 vision
// attention, the 4-D QK^T and probs@V matmuls are ~95% of the runtime, so
// flattening them to 3-D turns a ~333s/image embedding into single-digit
// seconds.
//
// The reshape from (1, H, S, K) to (H, S, K) is a no-op on memory (the
// strides are already contiguous in row-major order, the leading 1 just gets
// dropped) so this rewrite is always semantically valid when:
//   - the DOT eNode has rank-4 shape with shape[0] == 1
//   - both input eclasses have the same rank-4 shape with shape[0] == 1
//   - the batched dim H (shape[1]) matches between A and B
struct FlattenBatchDot : public Rule
{
    std::unordered_set<uint32_t> visited;

    std::string name() const override { return "FlattenBatchDot"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::DOT || enode.children.size() != 2)
            return false;

        if (visited.count(eNodeIdx))
            return false;

        const std::vector<uint32_t> &outShape = enode.shape;
        // [FIX] Was: outShape[0] != 1 (only batch=1).  Now accept any batch
        // size B >= 1.  For B > 1 we merge B and H into a single dim B*H
        // so the 3-D NEON dot kernel can handle it.
        if (outShape.size() != 4)
            return false;

        uint32_t aClass = egraph.findConst(enode.children[0]);
        uint32_t bClass = egraph.findConst(enode.children[1]);
        const EClass &aCls = egraph.getEClass(aClass);
        const EClass &bCls = egraph.getEClass(bClass);

        if (aCls.shape.size() != 4)
            return false;
        if (bCls.shape.size() != 4)
            return false;
        // Both inputs must share the same batch B and head H dims.
        if (aCls.shape[0] != bCls.shape[0])
            return false;
        if (aCls.shape[1] != bCls.shape[1])
            return false;
        if (aCls.shape[3] != bCls.shape[2]) // K contraction
            return false;
        // Output dims must line up: out = [B, H, A.shape[2], B.shape[3]]
        if (outShape[0] != aCls.shape[0] ||
            outShape[1] != aCls.shape[1] ||
            outShape[2] != aCls.shape[2] ||
            outShape[3] != bCls.shape[3])
            return false;

        // Inputs must be contiguous (reshape to 3-D would otherwise need a
        // contiguous() first, which kills the perf win).  Most DOT inputs in
        // the vision attention path are already contiguous because they came
        // out of a permute + contiguous() pair.
        if (!isContiguous(aCls) || !isContiguous(bCls))
            return false;
        if (!isContiguous(egraph.getEClass(egraph.findConst(egraph.getENodeEClass(eNodeIdx)))))
            return false;

        return true;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode dotNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        if (!visited.insert(eNodeIdx).second)
            return;

        uint32_t aClass = egraph.find(dotNode.children[0]);
        uint32_t bClass = egraph.find(dotNode.children[1]);
        const EClass aCls = egraph.getEClass(aClass);
        const EClass bCls = egraph.getEClass(bClass);

        // Shapes for the 3-D intermediates.
        // [FIX] Merge batch B and heads H into a single dim B*H.
        // For B=1 this reduces to the old (H, S, K) behavior.
        // For B>1 this gives (B*H, S, K) which the 3-D NEON dot kernel
        // accepts (it treats the first dim as an independent batch).
        uint32_t BH = aCls.shape[0] * aCls.shape[1];
        std::vector<uint32_t> a3 = {BH, aCls.shape[2], aCls.shape[3]}; // (B*H, S, K)
        std::vector<uint32_t> b3 = {BH, bCls.shape[2], bCls.shape[3]}; // (B*H, K, S2)
        std::vector<uint32_t> y3 = {BH, aCls.shape[2], bCls.shape[3]}; // (B*H, S, S2)

        std::vector<int32_t> a3_int(a3.begin(), a3.end());
        std::vector<int32_t> b3_int(b3.begin(), b3.end());
        std::vector<int32_t> y3_int(y3.begin(), y3.end());

        uint32_t a3_shape_id = addIntConst(egraph, a3_int);
        uint32_t b3_shape_id = addIntConst(egraph, b3_int);
        uint32_t y3_shape_id = addIntConst(egraph, y3_int);

        // Reshape A: (B, H, S, K) -> (B*H, S, K)
        std::vector<uint64_t> a3_strides = calcContiguousStrides(a3);
        uint32_t rA = addOpToEGraph(egraph, OpType::RESHAPE, {aClass, a3_shape_id}, a3, a3_strides, 0, dotNode.dtype, dotNode.backend);

        // Reshape B: (B, H, K, S2) -> (B*H, K, S2)
        std::vector<uint64_t> b3_strides = calcContiguousStrides(b3);
        uint32_t rB = addOpToEGraph(egraph, OpType::RESHAPE, {bClass, b3_shape_id}, b3, b3_strides, 0, dotNode.dtype, dotNode.backend);

        // 3-D DOT: (B*H, S, K) x (B*H, K, S2) -> (B*H, S, S2)
        std::vector<uint64_t> y3_strides = calcContiguousStrides(y3);
        uint32_t rY = addOpToEGraph(egraph, OpType::DOT, {rA, rB}, y3, y3_strides, 0, dotNode.dtype, dotNode.backend);

        // Reshape output back: (B*H, S, S2) -> (B, H, S, S2)
        // Match the original DOT eclass's shape / strides / viewOffset.
        const EClass outCls = egraph.getEClass(egraph.findConst(eclassId));
        std::vector<int32_t> out4_int(outCls.shape.begin(), outCls.shape.end());
        uint32_t out4_shape_id = addIntConst(egraph, out4_int);
        uint32_t outReshape = addOpToEGraph(egraph, OpType::RESHAPE, {rY, out4_shape_id}, outCls.shape, outCls.strides, outCls.viewOffset, dotNode.dtype, dotNode.backend);

        // The reshape produces a semantically-equivalent tensor to the
        // original 4-D DOT - merge the two eclasses so the cost model can
        // pick whichever is cheaper (almost always the 3-D path).
        egraph.merge(eclassId, outReshape);
    }
};

// =============================================================================
// FlattenElementwise
// =============================================================================
//
// Rewrites an N-D elementwise op (POWER, MUL, NEGATE, ADD, DIVIDE, SIN, COS,
// CAST, LT, EQ, AND, OR, NOT) into a 1-D op sandwiched by reshapes:
//
//     op(X[d0, d1, ..., dn-1])               -> reshape(op(reshape(X, [N])), [d0..dn-1])
//     op(A[..], B[..])                       -> reshape(op(reshape(A, [N]), reshape(B, [N])), [d0..dn-1])
//
// where N = prod(d0..dn-1) is the total element count.
//
// Why: the kernel registry has highly-tuned 1-D kernels for several
// elementwise ops (e.g. F32_3D_1D broadcasts a 1-D bias over a 3-D tensor,
// exp_4D_1_N_N_N_NEON is hard-coded to shape [1, N, N, N]).  When the input
// happens to be 3-D or 4-D, those kernels can't match directly because the
// pattern-check rejects the rank.  Flattening to 1-D lets the planner select
// the 1-D kernel and reshape back - usually a net win because the reshape is
// a metadata-only view op.
//
// Constraints:
//   - op must be one of the supported elementwise ops (see switch in apply())
//   - all input eclasses must have the SAME shape as the output (no broadcast)
//   - all participants must be contiguous (otherwise the reshape view would
//     materialise a copy and kill the perf win)
struct FlattenElementwise : public Rule
{
    std::unordered_set<uint32_t> visited;

    std::string name() const override { return "FlattenElementwise"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    static bool isSupportedOp(OpType op)
    {
        switch (op)
        {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::DIVIDE:
        case OpType::POWER:
        case OpType::SIN:
        case OpType::COS:
        case OpType::NEGATE:
        case OpType::CAST:
        case OpType::LT:
        case OpType::EQ:
        case OpType::AND:
        case OpType::OR:
        case OpType::NOT:
            return true;
        default:
            return false;
        }
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (!isSupportedOp(enode.opType))
            return false;
        if (visited.count(eNodeIdx))
            return false;

        const std::vector<uint32_t> &outShape = enode.shape;
        if (outShape.size() < 2)
            return false; // already 1-D (or scalar) - nothing to flatten

        // All children must have the SAME shape as the output (no broadcast).
        // For CAST, the dtype differs but the shape still has to match.
        for (uint32_t childId : enode.children)
        {
            const EClass &childCls = egraph.getEClass(egraph.findConst(childId));
            if (childCls.shape != outShape)
                return false;
            if (!isContiguous(childCls))
                return false;
        }
        // Output must also be contiguous so the trailing reshape is free.
        const EClass &outCls = egraph.getEClass(egraph.findConst(egraph.getENodeEClass(eNodeIdx)));
        if (!isContiguous(outCls))
            return false;

        return true;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode opNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        if (!visited.insert(eNodeIdx).second)
            return;

        const EClass outCls = egraph.getEClass(egraph.findConst(eclassId));
        const std::vector<uint32_t> &outShape = outCls.shape;

        // Compute total element count.
        uint64_t total = 1;
        for (uint32_t d : outShape)
            total *= d;
        if (total == 0)
            return;

        // Build the 1-D shape constants.  We use int32_t because the graph's
        // reshape() expects INT32 shape tensors.
        std::vector<uint32_t> flatShape = {(uint32_t)total};
        std::vector<int32_t> flat_int = {(int32_t)total};
        std::vector<int32_t> out_int(outShape.begin(), outShape.end());
        uint32_t flat_shape_id = addIntConst(egraph, flat_int);
        uint32_t out_shape_id = addIntConst(egraph, out_int);

        std::vector<uint64_t> flatStrides = {1};

        // Reshape each input to 1-D.
        std::vector<uint32_t> flatChildren;
        for (uint32_t childId : opNode.children)
        {
            uint32_t canonChild = egraph.find(childId);
            uint32_t r = addOpToEGraph(egraph, OpType::RESHAPE, {canonChild, flat_shape_id}, flatShape, flatStrides, 0, opNode.dtype, opNode.backend);
            flatChildren.push_back(r);
        }

        // Apply the op on 1-D inputs.
        uint32_t flatOut = addOpToEGraph(egraph, opNode.opType, flatChildren, flatShape, flatStrides, 0, opNode.dtype, opNode.backend);

        // Reshape back to the original N-D shape.
        uint32_t outReshape = addOpToEGraph(egraph, OpType::RESHAPE, {flatOut, out_shape_id}, outCls.shape, outCls.strides, outCls.viewOffset, opNode.dtype, opNode.backend);

        // Merge: the cost model picks whichever path is cheaper.
        egraph.merge(eclassId, outReshape);
    }
};

// =============================================================================
// InsertContiguousRepair
// =============================================================================
//
// For every elementwise op with a non-contiguous input, create an alternative
// enode where that input is first materialized via CONTIGUOUS.  The two enodes
// are merged into the same eclass so the cost model picks the cheaper path.
//
// This unblocks NEON elementwise kernels (Div_ND_NEON_Threaded, Pow_1D_*, etc.)
// which require contiguous inputs but are ~100x faster than the scalar
// reference fallbacks.
//
// The rule is idempotent: it only fires when an input is non-contiguous, and
// the inserted CONTIGUOUS op produces a contiguous tensor so the rule won't
// fire again on the new enode.
struct InsertContiguousRepair : public Rule
{
    std::unordered_set<uint32_t> visited;

    std::string name() const override { return "InsertContiguousRepair"; }

    static bool isElementwiseOp(OpType op)
    {
        switch (op)
        {
        case OpType::ADD:
        case OpType::MUL:
        case OpType::DIVIDE:
        case OpType::POWER:
        case OpType::SIN:
        case OpType::COS:
        case OpType::NEGATE:
        case OpType::CAST:
        case OpType::LT:
        case OpType::EQ:
        case OpType::AND:
        case OpType::OR:
        case OpType::NOT:
            return true;
        default:
            return false;
        }
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (!isElementwiseOp(enode.opType))
            return false;
        if (visited.count(eNodeIdx))
            return false;

        // Fire if ANY input eclass is non-contiguous.
        for (uint32_t childId : enode.children)
        {
            const EClass &childCls = egraph.getEClass(egraph.findConst(childId));
            if (!isContiguous(childCls))
                return true;
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode opNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        if (!visited.insert(eNodeIdx).second)
            return;

        const EClass outCls = egraph.getEClass(egraph.findConst(eclassId));

        // Build a new enode with the same op, but with each non-contiguous
        // input replaced by CONTIGUOUS(input).
        std::vector<uint32_t> newChildren;
        newChildren.reserve(opNode.children.size());
        for (uint32_t childId : opNode.children)
        {
            uint32_t canonChild = egraph.find(childId);
            const EClass &childCls = egraph.getEClass(canonChild);

            if (!isContiguous(childCls))
            {
                // Insert CONTIGUOUS(child).
                // Output shape = child shape, contiguous strides, offset 0.
                std::vector<uint64_t> contigStrides = calcContiguousStrides(childCls.shape);
                uint32_t contigEnode = addOpToEGraph(
                    egraph, OpType::CONTIGUOUS, {canonChild},
                    childCls.shape, contigStrides, 0,
                    childCls.dtype, childCls.backend);
                newChildren.push_back(contigEnode);
            }
            else
            {
                newChildren.push_back(canonChild);
            }
        }

        // Create the alternative enode (same op, repaired inputs).
        uint32_t repairedEnode = addOpToEGraph(
            egraph, opNode.opType, newChildren,
            outCls.shape, calcContiguousStrides(outCls.shape), 0,
            opNode.dtype, opNode.backend);

        // Merge: cost model picks whichever is cheaper.
        // The repaired path is almost always cheaper because it can use NEON.
        egraph.merge(eclassId, repairedEnode);
    }
};