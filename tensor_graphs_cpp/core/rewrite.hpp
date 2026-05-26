// tensor_graphs_cpp/core/rewrite.hpp
#pragma once
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"
#include "core/egraph.hpp"
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

struct Rule
{
    virtual ~Rule() = default;
    virtual std::string name() const = 0;
    virtual bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) = 0;
    virtual void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) = 0;
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
            for (const auto &k : KernelRegistry::get().getAllKernels())
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

inline uint32_t createCacheInputNode(EGraph &egraph, const ENode &sourceNode, uint32_t sourceClassId, uint32_t partialPathId, std::unordered_map<uint32_t, uint32_t> &eclassToLogical)
{
    uint32_t canonSrcClass = egraph.find(sourceClassId);
    const EClass srcClass = egraph.getEClass(canonSrcClass);

    uint32_t op_cache = egraph.addEClass(srcClass.shape, srcClass.strides, srcClass.viewOffset, srcClass.dtype, srcClass.backend);
    ENode cacheNode;
    cacheNode.kernelUid = 0;
    cacheNode.opType = OpType::INPUT;
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
        // Robust fallback checking underlying merged unions.
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
        // For variadic CONCAT matches: the e-class IDs of the tensor inputs
        // (all e-node children except the last, which is the axis).
        // Empty for non-CONCAT matches.
        std::vector<uint32_t> variadicConcatTensorEClasses;
    };

    std::unordered_map<OpType, std::vector<Pattern>> patternsByOp;
    std::vector<MatchResult> activeMatches;

    FusionRule()
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
            pattern.rootOpType = pattern.graph.getNode(pattern.rootId).opType;
            pattern.dtypes = entry.dtypes;
            pattern.dummyShapes = entry.dummyShapes;

            patternsByOp[pattern.rootOpType].push_back(std::move(pattern));
        }
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        activeMatches.clear();
        const ENode &eNode = egraph.getENodes()[eNodeIdx];

        auto it = patternsByOp.find(eNode.opType);
        if (it == patternsByOp.end())
            return false;

        for (const auto &pattern : it->second)
        {
            std::unordered_map<uint32_t, uint32_t> binding;
            if (matchPatternNode(eNodeIdx, egraph, pattern.rootId, pattern, binding, protectedEClasses))
            {
                MatchResult mr;
                mr.pattern = &pattern;
                mr.binding = std::move(binding);

                // For variadic CONCAT matches, collect the tensor e-class IDs
                // (all children except the last, which is the axis).
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        for (const auto &match : activeMatches)
        {
            const Pattern &pattern = *match.pattern;
            const auto &binding = match.binding;

            std::vector<uint32_t> inputs;
            std::vector<TensorNode> inputNodes;

            if (!match.variadicConcatTensorEClasses.empty())
            {
                // Variadic CONCAT: inputs = [tensor0_eclass, tensor1_eclass, ..., axis_eclass]
                // The tensor e-classes were collected during match.
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
                // The axis e-class is the last pattern variable's binding
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
                // Standard (non-variadic) path
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

                std::vector<uint64_t> kernelMatches = KernelRegistry::get().findMatchingKernels(
                    OpType::FUSED, pattern.opName, targetBackend, inputNodes, outputNode, false, true, true);

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

            // For variadic kernels, use the same indexing rule as matches():
            //   indices [0..N-2] → rule 0 (tensor inputs)
            //   index  [N-1]    → rule 1 (axis constant)
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

        // Constant-value check: if the pattern node is a constant (INPUT with
        // a non-empty contentHash, created via graph.constant()), the e-graph
        // eclass must carry the same constant data.  This prevents, for
        // example, a SiLU pattern that uses e_val = 2.7182818f from matching
        // a subgraph that uses a different constant for the exponential base.
        if (eNode.opType == OpType::INPUT && !pNode.contentHash.empty())
        {
            if (!ignoreConstantData)
            {
                uint32_t eNodeEClass = egraph.getENodeEClass(eNodeIdx);
                uint32_t canonEClass = egraph.findConst(eNodeEClass);

                // The pattern requires a constant; the e-graph must have one too.
                auto egraphIt = egraph.constantStaging.find(canonEClass);
                if (egraphIt == egraph.constantStaging.end())
                    return false;

                // Retrieve the pattern's constant data.
                auto patternIt = pattern.graph.constantStaging.find(patternId);
                if (patternIt == pattern.graph.constantStaging.end())
                    return false;

                // Compare the raw bytes.  For small scalars (e.g. a single F32)
                // this is very cheap.
                const auto &egraphData = *egraphIt->second;
                const auto &patternData = *patternIt->second;
                if (egraphData.size() != patternData.size())
                    return false;
                if (std::memcmp(egraphData.data(), patternData.data(), egraphData.size()) != 0)
                    return false;
            }
        }

        // Variadic CONCAT: the pattern has [tensorVar, axisVar] (2 parents),
        // but the e-graph CONCAT can have N+1 children (N tensors + 1 axis).
        // We match all tensor children against the first pattern parent and
        // the last child (axis) against the last pattern parent.
        if (eNode.opType == OpType::CONCAT && eNode.children.size() != pNode.parentIds.size())
        {
            // Pattern must have at least 2 parents: [tensorVar, axisVar]
            if (pNode.parentIds.size() < 2)
                return false;
            // E-node must have at least 2 children: [tensor, axis]
            if (eNode.children.size() < 2)
                return false;

            // Match the axis: last e-node child <-> last pattern parent
            if (!matchPatternClass(eNode.children.back(), egraph,
                                   pNode.parentIds.back(), pattern, binding, protectedEClasses, true))
                return false;

            // Match all tensor children against the first pattern parent (tensorVar).
            // The first successful binding establishes the variable; subsequent
            // children only need to be valid e-classes (they'll be collected
            // separately since the binding can only hold one e-class per variable).
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
                    // For subsequent tensors, just verify the e-class exists and
                    // has a compatible dtype. We don't re-bind the pattern variable.
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

struct ConstantFolding : public Rule
{
    std::unordered_set<uint32_t> visited_enodes; // Match guard

    std::string name() const override { return "ConstantFolding"; }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        // 1. Guard against already processed nodes
        if (visited_enodes.count(eNodeIdx))
            return false;

        const ENode &eNode = egraph.getENodes()[eNodeIdx];

        if (eNode.opType == OpType::INPUT)
            return false;

        if (eNode.children.empty())
            return false;

        for (uint32_t c : eNode.children)
        {
            uint32_t childEClassId = egraph.findConst(c);
            if (egraph.constantStaging.find(childEClassId) == egraph.constantStaging.end())
                return false;
        }

        uint32_t eclassId = egraph.findConst(egraph.getENodeEClass(eNodeIdx));
        if (egraph.constantStaging.find(eclassId) != egraph.constantStaging.end())
            return false;

        const EClass &targetCls = egraph.getEClass(eclassId);

        for (uint32_t enodeId : targetCls.enodes)
        {
            const ENode &sibling = egraph.getENodes()[enodeId];
            if (sibling.opType == OpType::COPY_TO && sibling.children.size() == 1)
            {
                uint32_t srcClass = egraph.findConst(sibling.children[0]);
                if (egraph.getEClass(srcClass).backend == Backend::CPU &&
                    egraph.constantStaging.find(srcClass) != egraph.constantStaging.end())
                {
                    return false;
                }
            }
        }

        return true;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        visited_enodes.insert(eNodeIdx); // Mark as visited
        const ENode eNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.find(egraph.getENodeEClass(eNodeIdx));

        std::vector<TensorNode> inputNodes;
        std::vector<TensorView> inViews;
        std::vector<const void *> kernelInputs;

        for (uint32_t c : eNode.children)
        {
            uint32_t childEClassId = egraph.find(c);
            const EClass childCls = egraph.getEClass(childEClassId);

            TensorNode inNode;
            inNode.opType = OpType::INPUT;
            inNode.dtype = childCls.dtype;
            inNode.setShape(childCls.shape);
            inNode.strides = childCls.strides;
            inNode.viewOffset = childCls.viewOffset;
            inNode.backend = Backend::CPU;
            inputNodes.push_back(inNode);

            const auto &stagedData = egraph.constantStaging.at(childEClassId);
            size_t offsetBytes = childCls.viewOffset * getDTypeSize(childCls.dtype);

            if (offsetBytes >= stagedData->size() && stagedData->size() > 0)
            {
                return;
            }

            // CRITICAL SAFETY CHECK: Ensure the stagedData actually encompasses the maximum offset
            // required by the child class strides.
            size_t maxInOffset = childCls.viewOffset;
            for (size_t d = 0; d < childCls.shape.size(); ++d)
            {
                if (childCls.shape[d] > 0)
                {
                    maxInOffset += (childCls.shape[d] - 1) * childCls.strides[d];
                }
            }
            size_t reqInBytes = (childCls.shape.empty() ? 1 : (maxInOffset + 1)) * getDTypeSize(childCls.dtype);
            if (reqInBytes > stagedData->size() && stagedData->size() > 0)
            {
                return; // Abort folding gracefully if the buffer does not physically support this view
            }

            kernelInputs.push_back(stagedData->data() + offsetBytes);
            inViews.push_back(TensorView(inNode, 0));
        }

        const EClass targetCls = egraph.getEClass(eclassId);

        uint64_t maxOffset = targetCls.viewOffset;
        for (size_t d = 0; d < targetCls.shape.size(); ++d)
        {
            if (targetCls.shape[d] > 0)
            {
                maxOffset += (targetCls.shape[d] - 1) * targetCls.strides[d];
            }
        }
        uint64_t reqBytes = (targetCls.shape.empty() ? 1 : (maxOffset + 1)) * getDTypeSize(targetCls.dtype);

        // Prevent massive static arrays from gobbling RAM
        if (reqBytes > 16 * 1024 * 1024)
            return;

        TensorNode outNode;
        outNode.opType = eNode.opType;
        outNode.opName = eNode.opName;
        outNode.dtype = eNode.dtype;
        outNode.setShape(eNode.shape);
        outNode.strides = targetCls.strides;
        outNode.viewOffset = targetCls.viewOffset;
        outNode.backend = Backend::CPU;

        auto matches = KernelRegistry::get().findMatchingKernels(
            eNode.opType, eNode.opName, Backend::CPU, inputNodes, outNode, false);

        if (matches.empty())
            return;

        const KernelEntry *selectedKernel = nullptr;
        for (uint64_t uid : matches)
        {
            const auto &k = KernelRegistry::get().getKernel(uid);
            if (!k.inplace)
            {
                selectedKernel = &k;
                break;
            }
        }
        if (!selectedKernel)
            return;

        std::shared_ptr<std::vector<uint8_t>> outData;
        if (selectedKernel->isView)
        {
            uint32_t firstChild = egraph.find(eNode.children[0]);
            outData = egraph.constantStaging.at(firstChild);
        }
        else
        {
            if (!selectedKernel->run)
                return;

            outData = std::make_shared<std::vector<uint8_t>>(reqBytes);
            std::vector<void *> kernelOutputs = {outData->data() + targetCls.viewOffset * getDTypeSize(targetCls.dtype)};
            std::vector<TensorView> outViews = {TensorView(outNode, targetCls.viewOffset * getDTypeSize(targetCls.dtype))};
            selectedKernel->run(kernelInputs, kernelOutputs, inViews, outViews);
        }

        ENode foldedNode;
        foldedNode.kernelUid = 0;
        foldedNode.opType = OpType::INPUT;
        foldedNode.shape = targetCls.shape;
        foldedNode.strides = targetCls.strides;
        foldedNode.viewOffset = targetCls.viewOffset;
        foldedNode.dtype = targetCls.dtype;
        foldedNode.backend = Backend::CPU;
        foldedNode.leafId = eNodeIdx | 0x40000000;

        Backend originalBackend = targetCls.backend;

        if (originalBackend == Backend::CPU)
        {
            egraph.addENode(eclassId, foldedNode);
            egraph.constantStaging[eclassId] = outData;
        }
        else
        {
            uint32_t cpuEClass = egraph.addEClass(foldedNode.shape, foldedNode.strides, foldedNode.viewOffset, foldedNode.dtype, Backend::CPU);
            egraph.addENode(cpuEClass, foldedNode);
            egraph.constantStaging[cpuEClass] = outData;

            TensorNode copyInNode;
            copyInNode.opType = OpType::INPUT;
            copyInNode.dtype = foldedNode.dtype;
            copyInNode.setShape(foldedNode.shape);
            copyInNode.strides = foldedNode.strides;
            copyInNode.viewOffset = foldedNode.viewOffset;
            copyInNode.backend = Backend::CPU;

            TensorNode copyOutNode = copyInNode;
            copyOutNode.opType = OpType::COPY_TO;
            copyOutNode.backend = originalBackend;
            copyOutNode.strides = targetCls.strides;
            copyOutNode.viewOffset = targetCls.viewOffset;

            Graph copyGraph;
            uint32_t copyIn = copyGraph.input(copyInNode.getShape(), copyInNode.dtype);
            uint32_t copyRoot = copyGraph.copyto(copyIn, copyOutNode.backend);

            auto copyMatches = KernelRegistry::get().findMatchingKernelsByPattern(
                copyGraph, copyRoot, copyOutNode.backend, {copyInNode}, copyOutNode, false, false, false);

            for (uint64_t uid : copyMatches)
            {
                const auto &copyKernel = KernelRegistry::get().getKernel(uid);
                ENode copyNode;
                copyNode.kernelUid = uid;
                copyNode.opType = copyKernel.opType;
                copyNode.opName = copyKernel.opName;
                copyNode.children = {cpuEClass};
                copyNode.shape = copyOutNode.getShape();
                copyNode.strides = copyOutNode.strides;
                copyNode.viewOffset = copyOutNode.viewOffset;
                copyNode.dtype = copyOutNode.dtype;
                copyNode.backend = copyOutNode.backend;
                egraph.addENode(eclassId, copyNode);
            }
        }
    }
};

struct InfinityDomination : public Rule
{
    std::unordered_set<uint32_t> visited_enodes;

    std::string name() const override { return "InfinityDomination"; }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        if (visited_enodes.count(eNodeIdx))
            return false;

        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::ADD || enode.children.size() != 2)
            return false;

        return isConstantFloat(egraph, enode.children[0]) || isConstantFloat(egraph, enode.children[1]);
    }

    bool isConstantFloat(const EGraph &egraph, uint32_t eclassId) const
    {
        uint32_t canon = egraph.findConst(eclassId);
        const EClass &cls = egraph.getEClass(canon);
        if (cls.dtype != DType::FLOAT32)
            return false;
        return egraph.constantStaging.find(canon) != egraph.constantStaging.end();
    }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        visited_enodes.insert(eNodeIdx);

        const ENode addNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t constIdx = isConstantFloat(egraph, addNode.children[1]) ? 1 : 0;
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
            // Forces a fresh buffer initialized strictly with the mask's original state
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

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
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
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, protectedEClasses, egraph))
                    continue;

                for (uint32_t srcNodeIdx : egraph.getEClass(srcClass).enodes)
                {
                    const ENode &opNode = egraph.getENodes()[srcNodeIdx];
                    OpType op = opNode.opType;
                    if (!(op == OpType::ADD || op == OpType::MUL || op == OpType::DIVIDE || op == OpType::POWER ||
                          op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST))
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
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
                      op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST))
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

                // Now create op_cache
                uint32_t op_cache = createCacheInputNode(egraph, opNode, srcClass, partialPathId, eclassToLogical);

                // Create SCATTER and merge with srcClass
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

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
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
                // PROTECTED CHECK: do not push slice down if the dot’s eclass is protected
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, protectedEClasses, egraph))
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
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
                        validSteps = false; // Only support step=1 for pushing down through DOT
                }
                if (!validSteps)
                    continue;

                std::vector<uint32_t> outClassShape = egraph.getEClass(srcClass).shape;
                uint32_t rank = outClassShape.size();
                if (rank != 2 && rank != 3 && rank != 4)
                    continue; // DOT only supports rank 2, 3, 4

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
                    // A: [B, H, M, K], B: [B, H, K, N]
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

                // Create op_cache
                uint32_t partialPathId = srcNodeIdx | 0x80000000;
                uint32_t op_cache = createCacheInputNode(egraph, dotNode, srcClass, partialPathId, eclassToLogical);

                // Create SCATTER and merge with srcClass using the sliceNode's children for parameters
                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, contigSlicedOp, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, dotNode.dtype, dotNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};
