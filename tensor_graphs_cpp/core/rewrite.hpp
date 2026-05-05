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

    if (pRoot != UINT32_MAX)
    {
        auto matches = KernelRegistry::get().findMatchingKernelsByPattern(pGraph, pRoot, backend, inNodes, outNode, false, true, true);
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

            // AUTOMATICALLY zero the offset if the kernel allocates fresh physical memory
            if (kernel.isView || kernel.inplace)
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

    std::vector<uint64_t> contigStrides = calcContiguousStrides(cls.shape);
    return addOpToEGraph(egraph, OpType::COPY_TO, {canon}, cls.shape, contigStrides, 0, cls.dtype, targetBackend);
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
        const auto &data = egraph.constantStaging.at(canon);
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

// a*(b+c) -> (a*b)+(a*c)
struct DistributiveProperty : public Rule
{
    std::string name() const override { return "DistributiveProperty"; }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::MUL || enode.children.size() != 2)
            return false;

        return hasOp(egraph, enode.children[0], OpType::ADD) ||
               hasOp(egraph, enode.children[1], OpType::ADD);
    }

    bool hasOp(const EGraph &egraph, uint32_t eclassId, OpType op) const
    {
        for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
        {
            if (egraph.getENodes()[enodeId].opType == op)
                return true;
        }
        return false;
    }

    uint32_t findOpNode(const EGraph &egraph, uint32_t eclassId, OpType op) const
    {
        for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
        {
            if (egraph.getENodes()[enodeId].opType == op)
                return enodeId;
        }
        return UINT32_MAX;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        const ENode mulNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t addNodeIdx = findOpNode(egraph, mulNode.children[0], OpType::ADD);
        bool leftIsAdd = (addNodeIdx != UINT32_MAX);

        if (!leftIsAdd)
        {
            addNodeIdx = findOpNode(egraph, mulNode.children[1], OpType::ADD);
            if (addNodeIdx == UINT32_MAX)
                return;
        }

        const ENode addNode = egraph.getENodes()[addNodeIdx];

        uint32_t aClassId = leftIsAdd ? mulNode.children[1] : mulNode.children[0];
        uint32_t bClassId = addNode.children[0];
        uint32_t cClassId = addNode.children[1];

        // --- 1. Create (a*b) and (a*c) ---
        uint32_t abClass = egraph.addEClass(mulNode.shape, mulNode.strides, mulNode.viewOffset, mulNode.dtype, mulNode.backend);
        uint32_t acClass = egraph.addEClass(mulNode.shape, mulNode.strides, mulNode.viewOffset, mulNode.dtype, mulNode.backend);

        // Create (a*b)
        ENode abNode;
        abNode.kernelUid = mulNode.kernelUid;
        abNode.opType = OpType::MUL;
        abNode.opName = mulNode.opName;
        abNode.children = leftIsAdd ? std::vector<uint32_t>{bClassId, aClassId} : std::vector<uint32_t>{aClassId, bClassId};
        abNode.shape = mulNode.shape;
        abNode.strides = mulNode.strides;
        abNode.viewOffset = mulNode.viewOffset;
        abNode.dtype = mulNode.dtype;
        abNode.backend = mulNode.backend;
        abClass = egraph.addENode(abClass, abNode);

        // Create (a*c)
        ENode acNode;
        acNode.kernelUid = mulNode.kernelUid;
        acNode.opType = OpType::MUL;
        acNode.opName = mulNode.opName;
        acNode.children = leftIsAdd ? std::vector<uint32_t>{cClassId, aClassId} : std::vector<uint32_t>{aClassId, cClassId};
        acNode.shape = mulNode.shape;
        acNode.strides = mulNode.strides;
        acNode.viewOffset = mulNode.viewOffset;
        acNode.dtype = mulNode.dtype;
        acNode.backend = mulNode.backend;
        acClass = egraph.addENode(acClass, acNode);

        // --- 2. Create new ADD node ---
        ENode newAddNode;
        newAddNode.kernelUid = addNode.kernelUid;
        newAddNode.opType = OpType::ADD;
        newAddNode.opName = addNode.opName;
        newAddNode.children = {abClass, acClass};
        newAddNode.shape = mulNode.shape;
        newAddNode.strides = mulNode.strides;
        newAddNode.viewOffset = mulNode.viewOffset;
        newAddNode.dtype = mulNode.dtype;
        newAddNode.backend = mulNode.backend;

        egraph.addENode(eclassId, newAddNode);
    }
};

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
                activeMatches.push_back({&pattern, std::move(binding)});
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
        if (parentIds.size() != kernel.numInputs)
        {
            Error::throw_err("[addFusedNode] parentIds.size() != kernel.numInputs. Info:\n  Kernel: " + kernel.opName + "\n" +
                             "  Parent IDs: " + std::to_string(parentIds.size()) + "\n" +
                             "  Kernel Num Inputs: " + std::to_string(kernel.numInputs) + "\n");
        }

        for (size_t i = 0; i < parentIds.size(); ++i)
        {
            uint32_t pid = parentIds[i];
            const EClass parent = egraph.getEClass(pid);

            Backend expectedBackend = kernel.inputBackends[i][0];
            bool foundBackend = false;
            for (Backend b : kernel.inputBackends[i])
            {
                if (parent.backend == b)
                {
                    expectedBackend = parent.backend;
                    foundBackend = true;
                    break;
                }
            }

            bool needCopy = !foundBackend;
            bool needContig = kernel.requiresContiguous[i] && !isContiguous(parent);

            if (!needCopy && !needContig)
            {
                adaptedParents.push_back(pid);
                continue;
            }

            uint32_t currentPid = pid;
            EClass currentClass = parent;

            if (needCopy)
            {
                TensorNode inNode;
                inNode.opType = OpType::INPUT;
                inNode.dtype = currentClass.dtype;
                inNode.setShape(currentClass.shape);
                inNode.strides = currentClass.strides;
                inNode.viewOffset = currentClass.viewOffset;
                inNode.backend = currentClass.backend;

                TensorNode outNode = inNode;
                outNode.opType = OpType::COPY_TO;
                outNode.backend = expectedBackend;
                outNode.strides = calcContiguousStrides(outNode.getShape());
                outNode.viewOffset = 0;

                Graph pGraph;
                uint32_t pIn = pGraph.input(inNode.getShape(), inNode.dtype);
                uint32_t pRoot = pGraph.copyto(pIn, outNode.backend);

                auto matches = KernelRegistry::get().findMatchingKernelsByPattern(pGraph, pRoot, outNode.backend, {inNode}, outNode, false, false, false);
                if (matches.empty())
                    return;

                uint32_t newEClass = egraph.addEClass(outNode.getShape(), outNode.strides, outNode.viewOffset, outNode.dtype, outNode.backend);
                for (uint64_t uid : matches)
                {
                    const auto &copyKernel = KernelRegistry::get().getKernel(uid);
                    ENode copyNode;
                    copyNode.kernelUid = uid;
                    copyNode.opType = copyKernel.opType;
                    copyNode.opName = copyKernel.opName;
                    copyNode.children = {currentPid};
                    copyNode.shape = outNode.getShape();
                    copyNode.strides = outNode.strides;
                    copyNode.viewOffset = outNode.viewOffset;
                    copyNode.dtype = outNode.dtype;
                    copyNode.backend = outNode.backend;
                    egraph.addENode(newEClass, copyNode);
                }
                currentPid = newEClass;
                currentClass = egraph.getEClass(newEClass);
            }

            if (needContig)
            {
                TensorNode inNode;
                inNode.opType = OpType::INPUT;
                inNode.dtype = currentClass.dtype;
                inNode.setShape(currentClass.shape);
                inNode.strides = currentClass.strides;
                inNode.viewOffset = currentClass.viewOffset;
                inNode.backend = currentClass.backend;

                TensorNode outNode = inNode;
                outNode.opType = OpType::CONTIGUOUS;
                outNode.strides = calcContiguousStrides(outNode.getShape());
                outNode.viewOffset = 0;

                Graph pGraph;
                uint32_t pIn = pGraph.input(inNode.getShape(), inNode.dtype);
                uint32_t pRoot = pGraph.contiguous(pIn);

                auto matches = KernelRegistry::get().findMatchingKernelsByPattern(pGraph, pRoot, outNode.backend, {inNode}, outNode, false, false, false);
                if (matches.empty())
                    return;

                uint32_t newEClass = egraph.addEClass(outNode.getShape(), outNode.strides, outNode.viewOffset, outNode.dtype, outNode.backend);
                for (uint64_t uid : matches)
                {
                    const auto &contigKernel = KernelRegistry::get().getKernel(uid);
                    ENode contigNode;
                    contigNode.kernelUid = uid;
                    contigNode.opType = contigKernel.opType;
                    contigNode.opName = contigKernel.opName;
                    contigNode.children = {currentPid};
                    contigNode.shape = outNode.getShape();
                    contigNode.strides = outNode.strides;
                    contigNode.viewOffset = outNode.viewOffset;
                    contigNode.dtype = outNode.dtype;
                    contigNode.backend = outNode.backend;
                    egraph.addENode(newEClass, contigNode);
                }
                currentPid = newEClass;
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

            TensorNode dummyIn;
            dummyIn.opType = OpType::INPUT;
            dummyIn.setShape(enode.shape);
            dummyIn.strides = enode.strides;
            dummyIn.viewOffset = enode.viewOffset;
            dummyIn.dtype = enode.dtype;
            dummyIn.backend = targetBackend;

            TensorNode dummyOut = dummyIn;
            dummyOut.opType = OpType::COPY_TO;
            dummyOut.backend = originalBackend;
            dummyOut.strides = calcContiguousStrides(dummyOut.getShape());
            dummyOut.viewOffset = 0;

            Graph pGraph;
            uint32_t pIn = pGraph.input(dummyIn.getShape(), dummyIn.dtype);
            uint32_t pRoot = pGraph.copyto(pIn, dummyOut.backend);

            auto matches = KernelRegistry::get().findMatchingKernelsByPattern(pGraph, pRoot, dummyOut.backend, {dummyIn}, dummyOut, false, false, false);
            for (uint64_t uid : matches)
            {
                const auto &copyKernel = KernelRegistry::get().getKernel(uid);
                ENode copyNode;
                copyNode.kernelUid = uid;
                copyNode.opType = copyKernel.opType;
                copyNode.opName = copyKernel.opName;
                copyNode.children = {newEClass};
                copyNode.shape = dummyOut.getShape();
                copyNode.strides = dummyOut.strides;
                copyNode.viewOffset = dummyOut.viewOffset;
                copyNode.dtype = dummyOut.dtype;
                copyNode.backend = dummyOut.backend;
                egraph.addENode(eclassId, copyNode);
            }
        }
    }

    static bool matchPatternClass(uint32_t eClassIdx, const EGraph &egraph,
                                  uint32_t patternId, const Pattern &pattern,
                                  std::unordered_map<uint32_t, uint32_t> &binding,
                                  const std::unordered_set<uint32_t> &protectedEClasses)
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
            if (matchPatternNode(enodeId, egraph, patternId, pattern, localBinding, protectedEClasses))
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
                                 const std::unordered_set<uint32_t> &protectedEClasses)
    {
        const ENode &eNode = egraph.getENodes()[eNodeIdx];
        const auto &pNode = pattern.graph.getNode(patternId);

        if (eNode.opType != pNode.opType)
            return false;
        if (eNode.opType == OpType::FUSED && eNode.opName != pNode.opName)
            return false;
        if (eNode.children.size() != pNode.parentIds.size())
            return false;

        for (size_t i = 0; i < eNode.children.size(); ++i)
        {
            if (!matchPatternClass(eNode.children[i], egraph, pNode.parentIds[i], pattern, binding, protectedEClasses))
            {
                return false;
            }
        }
        return true;
    }
};

struct CopyToOfContiguous : public Rule
{
    std::string name() const override { return "CopyToOfContiguous"; }

    bool hasOp(const EGraph &egraph, uint32_t eclassId, OpType op) const
    {
        for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
        {
            if (egraph.getENodes()[enodeId].opType == op)
                return true;
        }
        return false;
    }

    uint32_t findOpNode(const EGraph &egraph, uint32_t eclassId, OpType op) const
    {
        for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
        {
            if (egraph.getENodes()[enodeId].opType == op)
                return enodeId;
        }
        return UINT32_MAX;
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::COPY_TO || enode.children.size() != 1)
            return false;
        return hasOp(egraph, enode.children[0], OpType::CONTIGUOUS);
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        const ENode copyToNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t contigNodeIdx = findOpNode(egraph, copyToNode.children[0], OpType::CONTIGUOUS);
        if (contigNodeIdx == UINT32_MAX)
            return;

        const ENode contigNode = egraph.getENodes()[contigNodeIdx];
        uint32_t xClassId = contigNode.children[0];
        const EClass xClass = egraph.getEClass(xClassId);

        TensorNode copyInNode;
        copyInNode.opType = OpType::INPUT;
        copyInNode.dtype = xClass.dtype;
        copyInNode.setShape(xClass.shape);
        copyInNode.strides = xClass.strides;
        copyInNode.viewOffset = xClass.viewOffset;
        copyInNode.backend = xClass.backend;

        TensorNode copyOutNode = copyInNode;
        copyOutNode.opType = OpType::COPY_TO;
        copyOutNode.backend = copyToNode.backend;
        copyOutNode.strides = calcContiguousStrides(copyOutNode.getShape());
        copyOutNode.viewOffset = 0;

        Graph copyGraph;
        uint32_t copyIn = copyGraph.input(copyInNode.getShape(), copyInNode.dtype);
        uint32_t copyRoot = copyGraph.copyto(copyIn, copyOutNode.backend);
        auto copyMatches = KernelRegistry::get().findMatchingKernelsByPattern(
            copyGraph, copyRoot, copyOutNode.backend, {copyInNode}, copyOutNode, false, false, false);
        if (copyMatches.empty())
            return;

        TensorNode contigInNode;
        contigInNode.opType = OpType::INPUT;
        contigInNode.dtype = copyOutNode.dtype;
        contigInNode.setShape(copyOutNode.getShape());
        contigInNode.strides = copyOutNode.strides;
        contigInNode.viewOffset = copyOutNode.viewOffset;
        contigInNode.backend = copyOutNode.backend;

        TensorNode contigOutNode = contigInNode;
        contigOutNode.opType = OpType::CONTIGUOUS;
        contigOutNode.strides = calcContiguousStrides(contigOutNode.getShape());
        contigOutNode.viewOffset = 0;

        Graph contigGraph;
        uint32_t contigIn = contigGraph.input(contigInNode.getShape(), contigInNode.dtype);
        uint32_t contigRoot = contigGraph.contiguous(contigIn);
        auto contigMatches = KernelRegistry::get().findMatchingKernelsByPattern(
            contigGraph, contigRoot, contigOutNode.backend, {contigInNode}, contigOutNode, false, false, false);
        if (contigMatches.empty())
            return;

        uint32_t copyEClass = egraph.addEClass(
            copyOutNode.getShape(), copyOutNode.strides, copyOutNode.viewOffset,
            copyOutNode.dtype, copyOutNode.backend);
        for (uint64_t uid : copyMatches)
        {
            const auto &kernel = KernelRegistry::get().getKernel(uid);
            ENode copyENode;
            copyENode.kernelUid = uid;
            copyENode.opType = kernel.opType;
            copyENode.opName = kernel.opName;
            copyENode.children = {xClassId};
            copyENode.shape = copyOutNode.getShape();
            copyENode.strides = copyOutNode.strides;
            copyENode.viewOffset = copyOutNode.viewOffset;
            copyENode.dtype = copyOutNode.dtype;
            copyENode.backend = copyOutNode.backend;
            egraph.addENode(copyEClass, copyENode);
        }

        for (uint64_t uid : contigMatches)
        {
            const auto &kernel = KernelRegistry::get().getKernel(uid);
            ENode contigENode;
            contigENode.kernelUid = uid;
            contigENode.opType = kernel.opType;
            contigENode.opName = kernel.opName;
            contigENode.children = {copyEClass};
            contigENode.shape = contigOutNode.getShape();
            contigENode.strides = contigOutNode.strides;
            contigENode.viewOffset = contigOutNode.viewOffset;
            contigENode.dtype = contigOutNode.dtype;
            contigENode.backend = contigOutNode.backend;
            egraph.addENode(eclassId, contigENode);
        }
    }
};

struct ContiguousOfCopyTo : public Rule
{
    std::string name() const override { return "ContiguousOfCopyTo"; }

    bool hasOp(const EGraph &egraph, uint32_t eclassId, OpType op) const
    {
        for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
        {
            if (egraph.getENodes()[enodeId].opType == op)
                return true;
        }
        return false;
    }

    uint32_t findOpNode(const EGraph &egraph, uint32_t eclassId, OpType op) const
    {
        for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
        {
            if (egraph.getENodes()[enodeId].opType == op)
                return enodeId;
        }
        return UINT32_MAX;
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.size() != 1)
            return false;
        return hasOp(egraph, enode.children[0], OpType::COPY_TO);
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        const ENode contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t copyToNodeIdx = findOpNode(egraph, contigNode.children[0], OpType::COPY_TO);
        if (copyToNodeIdx == UINT32_MAX)
            return;

        const ENode copyToNode = egraph.getENodes()[copyToNodeIdx];
        uint32_t xClassId = copyToNode.children[0];
        const EClass xClass = egraph.getEClass(xClassId);

        TensorNode contigInNode;
        contigInNode.opType = OpType::INPUT;
        contigInNode.dtype = xClass.dtype;
        contigInNode.setShape(xClass.shape);
        contigInNode.strides = xClass.strides;
        contigInNode.viewOffset = xClass.viewOffset;
        contigInNode.backend = xClass.backend;

        TensorNode contigOutNode = contigInNode;
        contigOutNode.opType = OpType::CONTIGUOUS;
        contigOutNode.strides = calcContiguousStrides(contigOutNode.getShape());
        contigOutNode.viewOffset = 0;

        Graph contigGraph;
        uint32_t contigIn = contigGraph.input(contigInNode.getShape(), contigInNode.dtype);
        uint32_t contigRoot = contigGraph.contiguous(contigIn);
        auto contigMatches = KernelRegistry::get().findMatchingKernelsByPattern(
            contigGraph, contigRoot, contigOutNode.backend, {contigInNode}, contigOutNode, false, false, false);
        if (contigMatches.empty())
            return;

        TensorNode copyInNode;
        copyInNode.opType = OpType::INPUT;
        copyInNode.dtype = contigOutNode.dtype;
        copyInNode.setShape(contigOutNode.getShape());
        copyInNode.strides = contigOutNode.strides;
        copyInNode.viewOffset = contigOutNode.viewOffset;
        copyInNode.backend = contigOutNode.backend;

        TensorNode copyOutNode = copyInNode;
        copyOutNode.opType = OpType::COPY_TO;
        copyOutNode.backend = contigNode.backend;
        copyOutNode.strides = calcContiguousStrides(copyOutNode.getShape());
        copyOutNode.viewOffset = 0;

        Graph copyGraph;
        uint32_t copyIn = copyGraph.input(copyInNode.getShape(), copyInNode.dtype);
        uint32_t copyRoot = copyGraph.copyto(copyIn, copyOutNode.backend);
        auto copyMatches = KernelRegistry::get().findMatchingKernelsByPattern(
            copyGraph, copyRoot, copyOutNode.backend, {copyInNode}, copyOutNode, false, false, false);
        if (copyMatches.empty())
            return;

        uint32_t contigEClass = egraph.addEClass(
            contigOutNode.getShape(), contigOutNode.strides, contigOutNode.viewOffset,
            contigOutNode.dtype, contigOutNode.backend);
        for (uint64_t uid : contigMatches)
        {
            const auto &kernel = KernelRegistry::get().getKernel(uid);
            ENode contigENode;
            contigENode.kernelUid = uid;
            contigENode.opType = kernel.opType;
            contigENode.opName = kernel.opName;
            contigENode.children = {xClassId};
            contigENode.shape = contigOutNode.getShape();
            contigENode.strides = contigOutNode.strides;
            contigENode.viewOffset = contigOutNode.viewOffset;
            contigENode.dtype = contigOutNode.dtype;
            contigENode.backend = contigOutNode.backend;
            egraph.addENode(contigEClass, contigENode);
        }

        for (uint64_t uid : copyMatches)
        {
            const auto &kernel = KernelRegistry::get().getKernel(uid);
            ENode copyENode;
            copyENode.kernelUid = uid;
            copyENode.opType = kernel.opType;
            copyENode.opName = kernel.opName;
            copyENode.children = {contigEClass};
            copyENode.shape = copyOutNode.getShape();
            copyENode.strides = copyOutNode.strides;
            copyENode.viewOffset = copyOutNode.viewOffset;
            copyENode.dtype = copyOutNode.dtype;
            copyENode.backend = copyOutNode.backend;
            egraph.addENode(eclassId, copyENode);
        }
    }
};

struct ContiguousElimination : public Rule
{
    std::unordered_set<uint64_t> visited_pairs;
    uint32_t matched_i;
    uint32_t matched_childENodeIdx;

    std::string name() const override { return "ContiguousElimination"; }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];

        if (enode.opType == OpType::INPUT || enode.children.empty() || enode.opType == OpType::CONTIGUOUS)
            return false;

        for (size_t i = 0; i < enode.children.size(); ++i)
        {
            const EClass &childCls = egraph.getEClass(egraph.findConst(enode.children[i]));
            for (uint32_t childENodeIdx : childCls.enodes)
            {
                uint64_t pair_id = (static_cast<uint64_t>(eNodeIdx) << 32) | childENodeIdx;
                if (visited_pairs.count(pair_id))
                    continue;

                const ENode &childENode = egraph.getENodes()[childENodeIdx];
                if (childENode.opType == OpType::CONTIGUOUS && !childENode.children.empty())
                {
                    matched_i = i;
                    matched_childENodeIdx = childENodeIdx;
                    return true;
                }
            }
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        uint64_t pair_id = (static_cast<uint64_t>(eNodeIdx) << 32) | matched_childENodeIdx;
        visited_pairs.insert(pair_id);

        const ENode consumerNode = egraph.getENodes()[eNodeIdx];
        uint32_t consumerEClassId = egraph.getENodeEClass(eNodeIdx);

        ENode childENode = egraph.getENodes()[matched_childENodeIdx];
        uint32_t sourceEClassId = childENode.children[0];

        std::vector<uint32_t> optimizedChildren = consumerNode.children;
        optimizedChildren[matched_i] = sourceEClassId;

        if (isContiguous(egraph.getEClass(egraph.findConst(sourceEClassId))))
        {
            ENode n = consumerNode;
            n.children = optimizedChildren;
            egraph.addENode(consumerEClassId, n);
            return;
        }

        std::vector<TensorNode> inputNodes;
        Graph pGraph;
        std::vector<uint32_t> pInputs;

        for (uint32_t cId : optimizedChildren)
        {
            const EClass cClass = egraph.getEClass(egraph.findConst(cId));
            TensorNode n;
            n.opType = OpType::INPUT;
            n.dtype = cClass.dtype;
            n.setShape(cClass.shape);
            n.strides = cClass.strides;
            n.backend = cClass.backend;
            inputNodes.push_back(n);

            pInputs.push_back(pGraph.input(n.getShape(), n.dtype));
        }

        TensorNode outNode;
        outNode.opType = consumerNode.opType;
        outNode.opName = consumerNode.opName;
        outNode.dtype = consumerNode.dtype;
        outNode.setShape(consumerNode.shape);
        outNode.strides = consumerNode.strides;
        outNode.backend = consumerNode.backend;

        uint32_t pRoot = UINT32_MAX;
        if (consumerNode.opType == OpType::FUSED)
        {
            const auto *refEntry = ReferenceGraphRegistry::get().getFactory(consumerNode.opName);
            if (refEntry)
            {
                pRoot = refEntry->factory(pInputs, pGraph);
            }
        }
        else
        {
            TensorNode &n = pGraph.allocateNode(consumerNode.opType, "", consumerNode.dtype, pInputs);
            pRoot = n.id;
        }

        if (pRoot != UINT32_MAX)
        {
            auto matches = KernelRegistry::get().findMatchingKernelsByPattern(
                pGraph, pRoot, consumerNode.backend, inputNodes, outNode, false, false, false);

            for (uint64_t uid : matches)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                ENode n = consumerNode;
                n.children = optimizedChildren;
                n.kernelUid = uid;
                n.opType = kernel.opType;
                n.opName = kernel.opName;
                egraph.addENode(consumerEClassId, n);
            }
        }
    }
};

struct ConstantFolding : public Rule
{
    std::string name() const override { return "ConstantFolding"; }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
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

            if (offsetBytes >= stagedData.size() && stagedData.size() > 0)
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
            if (reqInBytes > stagedData.size() && stagedData.size() > 0)
            {
                return; // Abort folding gracefully if the buffer does not physically support this view
            }

            kernelInputs.push_back(stagedData.data() + offsetBytes);
            inViews.push_back(TensorView(inNode, 0));
        }

        const EClass targetCls = egraph.getEClass(eclassId);
        TensorNode outNode;
        outNode.opType = eNode.opType;
        outNode.opName = eNode.opName;
        outNode.dtype = eNode.dtype;
        outNode.setShape(eNode.shape);
        outNode.strides = targetCls.strides;
        outNode.viewOffset = targetCls.viewOffset;
        outNode.backend = Backend::CPU;

        auto matches = KernelRegistry::get().findMatchingKernels(
            eNode.opType, eNode.opName, Backend::CPU, inputNodes, outNode, false, true, true);

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

        std::vector<uint8_t> outData;
        if (selectedKernel->isView)
        {
            uint32_t firstChild = egraph.find(eNode.children[0]);
            outData = egraph.constantStaging.at(firstChild);
        }
        else
        {
            if (!selectedKernel->run)
                return;

            uint64_t maxOffset = targetCls.viewOffset;
            for (size_t d = 0; d < targetCls.shape.size(); ++d)
            {
                if (targetCls.shape[d] > 0)
                {
                    maxOffset += (targetCls.shape[d] - 1) * targetCls.strides[d];
                }
            }
            uint64_t reqBytes = (targetCls.shape.empty() ? 1 : (maxOffset + 1)) * getDTypeSize(targetCls.dtype);
            outData.resize(reqBytes);

            std::vector<void *> kernelOutputs = {outData.data() + targetCls.viewOffset * getDTypeSize(targetCls.dtype)};
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
            egraph.constantStaging[eclassId] = std::move(outData);
        }
        else
        {
            uint32_t cpuEClass = egraph.addEClass(foldedNode.shape, foldedNode.strides, foldedNode.viewOffset, foldedNode.dtype, Backend::CPU);
            egraph.addENode(cpuEClass, foldedNode);
            egraph.constantStaging[cpuEClass] = std::move(outData);

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

        const auto &constData = egraph.constantStaging.at(constClass);
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

        uint32_t currentTarget = createCacheInputNode(egraph, addNode, eclassId, eNodeIdx | 0x80000000, eclassToLogical);

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
                // PROTECTED CHECK
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
                        return true;
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
                if (rank != 2 && rank != 3)
                    continue; // DOT only supports rank 2 and 3

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

                uint32_t K = (rank == 2) ? egraph.getEClass(egraph.find(aClassId)).shape[1] : egraph.getEClass(egraph.find(aClassId)).shape[2];

                std::vector<int32_t> startsA, endsA, stepsA(rank, 1);
                std::vector<int32_t> startsB, endsB, stepsB(rank, 1);

                if (rank == 2)
                {
                    startsA = {starts[0], 0};
                    endsA = {ends[0], (int32_t)K};

                    startsB = {0, starts[1]};
                    endsB = {(int32_t)K, ends[1]};
                }
                else
                {
                    startsA = {starts[0], starts[1], 0};
                    endsA = {ends[0], ends[1], (int32_t)K};

                    startsB = {starts[0], 0, starts[2]};
                    endsB = {ends[0], (int32_t)K, ends[2]};
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

struct SlicePullUpDot : public Rule
{
    struct MatchKey
    {
        uint32_t eNodeIdx;
        uint32_t aIdx;
        uint32_t bIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && aIdx == o.aIdx && bIdx == o.bIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^
                   (std::hash<uint32_t>{}(k.aIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.bIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePullUpDot(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected) {}

    std::string name() const override { return "SlicePullUpDot"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    void getSliceEnodes(const EGraph &egraph, uint32_t eclassId, std::vector<uint32_t> &outSlices) const
    {
        uint32_t canon = egraph.findConst(eclassId);
        for (uint32_t enodeIdx : egraph.getEClass(canon).enodes)
        {
            const ENode &enode = egraph.getENodes()[enodeIdx];
            if (enode.opType == OpType::CONTIGUOUS && !enode.children.empty())
            {
                uint32_t childClass = egraph.findConst(enode.children[0]);
                for (uint32_t sliceIdx : egraph.getEClass(childClass).enodes)
                {
                    if (egraph.getENodes()[sliceIdx].opType == OpType::SLICE && egraph.getENodes()[sliceIdx].children.size() == 4)
                    {
                        outSlices.push_back(sliceIdx);
                    }
                }
            }
        }
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;

        uint32_t dotClass = egraph.findConst(enode.children[0]);
        // PROTECTED CHECK
        if (!allowPushDownOnProtected && isEClassProtected(dotClass, protectedEClasses, egraph))
            return false;

        for (uint32_t dotNodeIdx : egraph.getEClass(dotClass).enodes)
        {
            const ENode &dotNode = egraph.getENodes()[dotNodeIdx];
            if (dotNode.opType == OpType::DOT && dotNode.children.size() == 2)
            {
                std::vector<uint32_t> aSlices, bSlices;
                getSliceEnodes(egraph, dotNode.children[0], aSlices);
                getSliceEnodes(egraph, dotNode.children[1], bSlices);

                for (uint32_t aIdx : aSlices)
                {
                    for (uint32_t bIdx : bSlices)
                    {
                        MatchKey key{eNodeIdx, aIdx, bIdx};
                        if (visited.find(key) == visited.end())
                            return true;
                    }
                }
            }
        }
        return false;
    }

    struct SliceInfo
    {
        uint32_t sliceEnodeIdx;
        uint32_t baseClass;
        std::vector<int32_t> starts;
        std::vector<int32_t> ends;
        std::vector<int32_t> steps;
        std::vector<uint32_t> fullShape;
    };

    bool resolveSliceInfo(const EGraph &egraph, uint32_t sliceIdx, SliceInfo &info) const
    {
        const ENode &sliceNode = egraph.getENodes()[sliceIdx];
        info.sliceEnodeIdx = sliceIdx;
        info.baseClass = egraph.findConst(sliceNode.children[0]);
        info.starts = getConstInt32(egraph, sliceNode.children[1]);
        info.ends = getConstInt32(egraph, sliceNode.children[2]);
        info.steps = getConstInt32(egraph, sliceNode.children[3]);
        info.fullShape = egraph.getEClass(info.baseClass).shape;

        if (info.starts.empty() || info.ends.empty() || info.steps.empty())
            return false;

        // Normalize and Pad
        while (info.starts.size() < info.fullShape.size())
            info.starts.push_back(0);
        while (info.ends.size() < info.fullShape.size())
            info.ends.push_back(info.fullShape[info.ends.size()]);
        while (info.steps.size() < info.fullShape.size())
            info.steps.push_back(1);

        for (size_t d = 0; d < info.fullShape.size(); ++d)
        {
            if (info.starts[d] < 0)
                info.starts[d] += info.fullShape[d];
            if (info.ends[d] < 0)
                info.ends[d] += info.fullShape[d];
        }
        return true;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        const ENode contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);
        uint32_t dotClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> dotNodes;
        for (uint32_t dotNodeIdx : egraph.getEClass(dotClass).enodes)
        {
            if (egraph.getENodes()[dotNodeIdx].opType == OpType::DOT && egraph.getENodes()[dotNodeIdx].children.size() == 2)
            {
                dotNodes.push_back(dotNodeIdx);
            }
        }

        for (uint32_t dotNodeIdx : dotNodes)
        {
            const ENode dotNode = egraph.getENodes()[dotNodeIdx];
            uint32_t aClassId = dotNode.children[0];
            uint32_t bClassId = dotNode.children[1];

            std::vector<uint32_t> aSlices, bSlices;
            getSliceEnodes(egraph, aClassId, aSlices);
            getSliceEnodes(egraph, bClassId, bSlices);

            for (uint32_t aIdx : aSlices)
            {
                for (uint32_t bIdx : bSlices)
                {
                    MatchKey key{eNodeIdx, aIdx, bIdx};
                    if (!visited.insert(key).second)
                    {
                        continue; // Already processed this pair
                    }

                    SliceInfo aInfo, bInfo;
                    if (!resolveSliceInfo(egraph, aIdx, aInfo) || !resolveSliceInfo(egraph, bIdx, bInfo))
                        continue;

                    // Check rank equality and compatibility
                    if (aInfo.fullShape.size() != bInfo.fullShape.size())
                        continue;
                    size_t rank = aInfo.fullShape.size();
                    if (rank != 2 && rank != 3)
                        continue;

                    // Check steps == 1
                    bool validSteps = true;
                    for (int32_t s : aInfo.steps)
                        if (s != 1)
                            validSteps = false;
                    for (int32_t s : bInfo.steps)
                        if (s != 1)
                            validSteps = false;
                    if (!validSteps)
                        continue;

                    // Verify reduction dimensions match exact matrix multiplication boundaries (K)
                    if (rank == 2)
                    {
                        if (aInfo.fullShape[1] != bInfo.fullShape[0])
                            continue;
                        uint32_t K = aInfo.fullShape[1];
                        if (aInfo.starts[1] != 0 || aInfo.ends[1] != (int32_t)K)
                            continue;
                        if (bInfo.starts[0] != 0 || bInfo.ends[0] != (int32_t)K)
                            continue;
                    }
                    else
                    {
                        if (aInfo.fullShape[0] != bInfo.fullShape[0])
                            continue;
                        if (aInfo.fullShape[2] != bInfo.fullShape[1])
                            continue;

                        // Batches must slice equivalently
                        if (aInfo.starts[0] != bInfo.starts[0] || aInfo.ends[0] != bInfo.ends[0])
                            continue;

                        uint32_t K = aInfo.fullShape[2];
                        if (aInfo.starts[2] != 0 || aInfo.ends[2] != (int32_t)K)
                            continue;
                        if (bInfo.starts[1] != 0 || bInfo.ends[1] != (int32_t)K)
                            continue;
                    }

                    // Generate the full un-sliced DOT
                    std::vector<uint32_t> fullDotShape;
                    if (rank == 2)
                    {
                        fullDotShape = {aInfo.fullShape[0], bInfo.fullShape[1]};
                    }
                    else
                    {
                        fullDotShape = {aInfo.fullShape[0], aInfo.fullShape[1], bInfo.fullShape[2]};
                    }

                    std::vector<uint64_t> fullDotStrides = calcContiguousStrides(fullDotShape);

                    uint32_t fullDotEClass = addOpToEGraph(egraph, OpType::DOT, {aInfo.baseClass, bInfo.baseClass}, fullDotShape, fullDotStrides, 0, dotNode.dtype, dotNode.backend);

                    // Re-apply SLICE to the new generic DOT Output
                    std::vector<int32_t> dotStarts, dotEnds, dotSteps(rank, 1);
                    if (rank == 2)
                    {
                        dotStarts = {aInfo.starts[0], bInfo.starts[1]};
                        dotEnds = {aInfo.ends[0], bInfo.ends[1]};
                    }
                    else
                    {
                        dotStarts = {aInfo.starts[0], aInfo.starts[1], bInfo.starts[2]};
                        dotEnds = {aInfo.ends[0], aInfo.ends[1], bInfo.ends[2]};
                    }

                    uint32_t startsId = addIntConst(egraph, dotStarts);
                    uint32_t endsId = addIntConst(egraph, dotEnds);
                    uint32_t stepsId = addIntConst(egraph, dotSteps);

                    std::vector<uint32_t> sliceShape = contigNode.shape;
                    std::vector<uint64_t> sliceStrides = fullDotStrides;
                    uint64_t sliceViewOffset = 0;
                    for (size_t d = 0; d < dotStarts.size(); ++d)
                    {
                        sliceViewOffset += dotStarts[d] * sliceStrides[d];
                    }

                    uint32_t sliceEClass = addOpToEGraph(egraph, OpType::SLICE, {fullDotEClass, startsId, endsId, stepsId}, sliceShape, sliceStrides, sliceViewOffset, dotNode.dtype, dotNode.backend);

                    // Generate the required target CONTIGUOUS and wrap it around the output
                    addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceEClass}, sliceShape, calcContiguousStrides(sliceShape), 0, dotNode.dtype, dotNode.backend, eclassId);
                }
            }
        }
    }
};

struct ScatterSliceCancellation : public Rule
{
    struct MatchKey
    {
        uint32_t contigIdx;
        uint32_t sliceIdx;
        uint32_t scatterIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigIdx == o.contigIdx && sliceIdx == o.sliceIdx && scatterIdx == o.scatterIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.contigIdx) ^
                   (std::hash<uint32_t>{}(k.sliceIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.scatterIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    ScatterSliceCancellation(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected) {}

    std::string name() const override { return "ScatterSliceCancellation"; }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &contigNode = egraph.getENodes()[eNodeIdx];
        if (contigNode.opType != OpType::CONTIGUOUS || contigNode.children.empty())
            return false;

        uint32_t sliceClassId = egraph.findConst(contigNode.children[0]);
        for (uint32_t sliceEnodeIdx : egraph.getEClass(sliceClassId).enodes)
        {
            const ENode &sliceNode = egraph.getENodes()[sliceEnodeIdx];
            if (sliceNode.opType == OpType::SLICE && sliceNode.children.size() == 4)
            {
                uint32_t scatterClassId = egraph.findConst(sliceNode.children[0]);
                if (!allowPushDownOnProtected && isEClassProtected(scatterClassId, protectedEClasses, egraph))
                    continue; // *only if not in protected set*

                for (uint32_t scatterEnodeIdx : egraph.getEClass(scatterClassId).enodes)
                {
                    const ENode &scatterNode = egraph.getENodes()[scatterEnodeIdx];
                    if (scatterNode.opType == OpType::SCATTER && scatterNode.children.size() == 5)
                    {
                        MatchKey key{eNodeIdx, sliceEnodeIdx, scatterEnodeIdx};
                        if (visited.find(key) != visited.end())
                            continue;

                        auto st1 = getConstInt32(egraph, scatterNode.children[2]);
                        auto en1 = getConstInt32(egraph, scatterNode.children[3]);
                        auto step1 = getConstInt32(egraph, scatterNode.children[4]);

                        auto st2 = getConstInt32(egraph, sliceNode.children[1]);
                        auto en2 = getConstInt32(egraph, sliceNode.children[2]);
                        auto step2 = getConstInt32(egraph, sliceNode.children[3]);

                        if (!st1.empty() && st1 == st2 && en1 == en2 && step1 == step2)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses, std::unordered_map<uint32_t, uint32_t> &eclassToLogical) override
    {
        const ENode &contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t contigClassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t sliceClassId = egraph.find(contigNode.children[0]);
        std::vector<uint32_t> updates;

        for (uint32_t sliceEnodeIdx : egraph.getEClass(sliceClassId).enodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceEnodeIdx];
            if (sliceNode.opType == OpType::SLICE && sliceNode.children.size() == 4)
            {
                uint32_t scatterClassId = egraph.find(sliceNode.children[0]);
                if (!allowPushDownOnProtected && isEClassProtected(scatterClassId, protectedEClasses, egraph))
                    continue;

                for (uint32_t scatterEnodeIdx : egraph.getEClass(scatterClassId).enodes)
                {
                    const ENode scatterNode = egraph.getENodes()[scatterEnodeIdx];
                    if (scatterNode.opType == OpType::SCATTER && scatterNode.children.size() == 5)
                    {
                        MatchKey key{eNodeIdx, sliceEnodeIdx, scatterEnodeIdx};
                        if (!visited.insert(key).second)
                            continue;

                        auto st1 = getConstInt32(egraph, scatterNode.children[2]);
                        auto en1 = getConstInt32(egraph, scatterNode.children[3]);
                        auto step1 = getConstInt32(egraph, scatterNode.children[4]);

                        auto st2 = getConstInt32(egraph, sliceNode.children[1]);
                        auto en2 = getConstInt32(egraph, sliceNode.children[2]);
                        auto step2 = getConstInt32(egraph, sliceNode.children[3]);

                        if (!st1.empty() && st1 == st2 && en1 == en2 && step1 == step2)
                        {
                            updates.push_back(scatterNode.children[1]); // The contiguous updated chunk
                        }
                    }
                }
            }
        }

        for (uint32_t updateClass : updates)
        {
            if (egraph.getEClass(updateClass).strides == egraph.getEClass(contigClassId).strides)
            {
                egraph.merge(contigClassId, updateClass);
            }
            else if (egraph.getEClass(updateClass).strides == egraph.getEClass(sliceClassId).strides)
            {
                egraph.merge(sliceClassId, updateClass);
            }
            // TODO: handle else case?
        }
    }
};

struct SlicePushDownContiguous : public Rule
{
    struct MatchKey
    {
        uint32_t eNodeIdx;
        uint32_t sliceNodeIdx;
        uint32_t opNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && sliceNodeIdx == o.sliceNodeIdx && opNodeIdx == o.opNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^
                   (std::hash<uint32_t>{}(k.sliceNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.opNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownContiguous(bool allow = false) : allowPushDownOnProtected(allow) {}

    std::string name() const override { return "SlicePushDownContiguous"; }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;
        uint32_t sliceClass = egraph.findConst(enode.children[0]);

        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &sliceNode = egraph.getENodes()[sliceIdx];
            if (sliceNode.opType == OpType::SLICE && sliceNode.children.size() == 4)
            {
                uint32_t srcClass = egraph.findConst(sliceNode.children[0]);
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, protectedEClasses, egraph))
                    continue;

                const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
                for (uint32_t srcIdx : srcEnodes)
                {
                    if (egraph.getENodes()[srcIdx].opType == OpType::CONTIGUOUS)
                    {
                        MatchKey key{eNodeIdx, sliceIdx, srcIdx};
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
        const ENode &contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);
        uint32_t sliceClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> sliceNodes;
        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENodes()[sliceIdx].opType == OpType::SLICE && egraph.getENodes()[sliceIdx].children.size() == 4)
            {
                sliceNodes.push_back(sliceIdx);
            }
        }

        for (uint32_t sliceIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceIdx];
            uint32_t srcClass = egraph.find(sliceNode.children[0]);

            const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
            for (uint32_t srcIdx : srcEnodes)
            {
                const ENode opNode = egraph.getENodes()[srcIdx];
                if (opNode.opType != OpType::CONTIGUOUS)
                    continue;

                MatchKey key{eNodeIdx, sliceIdx, srcIdx};
                if (!visited.insert(key).second)
                    continue;

                uint32_t aClassId = opNode.children[0];
                uint32_t partialPathId = srcIdx | 0x80000000;

                auto starts = getConstInt32(egraph, sliceNode.children[1]);
                auto ends = getConstInt32(egraph, sliceNode.children[2]);
                auto steps = getConstInt32(egraph, sliceNode.children[3]);
                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                const EClass aClass = egraph.getEClass(egraph.find(aClassId));
                std::vector<uint64_t> sliceStrides = aClass.strides;
                uint64_t sliceOffset = aClass.viewOffset;
                for (size_t d = 0; d < starts.size() && d < sliceStrides.size(); ++d)
                {
                    int32_t start = starts[d];
                    if (start < 0)
                        start += aClass.shape[d];
                    sliceOffset += start * sliceStrides[d];
                    sliceStrides[d] *= steps[d];
                }

                uint32_t newSlice = addOpToEGraph(egraph, OpType::SLICE, {aClassId, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, sliceNode.shape, sliceStrides, sliceOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                uint32_t newContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {newSlice}, sliceNode.shape, calcContiguousStrides(sliceNode.shape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t op_cache = createCacheInputNode(egraph, opNode, srcClass, partialPathId, eclassToLogical);
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, newContig, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

struct SlicePushDownPermute : public Rule
{
    struct MatchKey
    {
        uint32_t eNodeIdx;
        uint32_t sliceNodeIdx;
        uint32_t opNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && sliceNodeIdx == o.sliceNodeIdx && opNodeIdx == o.opNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^
                   (std::hash<uint32_t>{}(k.sliceNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.opNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownPermute(bool allow = false) : allowPushDownOnProtected(allow) {}

    std::string name() const override { return "SlicePushDownPermute"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;
        uint32_t sliceClass = egraph.findConst(enode.children[0]);

        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &sliceNode = egraph.getENodes()[sliceIdx];
            if (sliceNode.opType == OpType::SLICE && sliceNode.children.size() == 4)
            {
                uint32_t srcClass = egraph.findConst(sliceNode.children[0]);
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, protectedEClasses, egraph))
                    continue;

                const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
                for (uint32_t srcIdx : srcEnodes)
                {
                    if (egraph.getENodes()[srcIdx].opType == OpType::PERMUTE)
                    {
                        MatchKey key{eNodeIdx, sliceIdx, srcIdx};
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
        const ENode &contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);
        uint32_t sliceClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> sliceNodes;
        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENodes()[sliceIdx].opType == OpType::SLICE && egraph.getENodes()[sliceIdx].children.size() == 4)
            {
                sliceNodes.push_back(sliceIdx);
            }
        }

        for (uint32_t sliceIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceIdx];
            uint32_t srcClass = egraph.find(sliceNode.children[0]);

            const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
            for (uint32_t srcIdx : srcEnodes)
            {
                const ENode opNode = egraph.getENodes()[srcIdx];
                if (opNode.opType != OpType::PERMUTE)
                    continue;

                MatchKey key{eNodeIdx, sliceIdx, srcIdx};
                if (!visited.insert(key).second)
                    continue;

                uint32_t aClassId = opNode.children[0];
                auto dims = getConstInt32(egraph, opNode.children[1]);
                if (dims.empty())
                    continue;

                auto starts = getConstInt32(egraph, sliceNode.children[1]);
                auto ends = getConstInt32(egraph, sliceNode.children[2]);
                auto steps = getConstInt32(egraph, sliceNode.children[3]);
                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                size_t rank = dims.size();
                const EClass aClass = egraph.getEClass(egraph.find(aClassId));

                if (rank != aClass.shape.size())
                    continue;

                bool dimsOk = true;
                for (int32_t dim : dims)
                {
                    if (dim < 0 || dim >= (int32_t)rank)
                    {
                        dimsOk = false;
                        break;
                    }
                }
                if (!dimsOk)
                    continue;
                while (starts.size() < rank)
                    starts.push_back(0);
                while (ends.size() < rank)
                    ends.push_back(opNode.shape[ends.size()]);
                while (steps.size() < rank)
                    steps.push_back(1);

                std::vector<int32_t> new_starts(rank, 0), new_ends(rank, 0), new_steps(rank, 1);
                for (size_t d = 0; d < rank; ++d)
                {
                    new_starts[dims[d]] = starts[d];
                    new_ends[dims[d]] = ends[d];
                    new_steps[dims[d]] = steps[d];
                }

                uint32_t partialPathId = srcIdx | 0x80000000;
                uint32_t nStartsId = addIntConst(egraph, new_starts);
                uint32_t nEndsId = addIntConst(egraph, new_ends);
                uint32_t nStepsId = addIntConst(egraph, new_steps);

                std::vector<uint64_t> sliceStrides = aClass.strides;
                uint64_t sliceOffset = aClass.viewOffset;
                std::vector<uint32_t> sliceShape(rank);

                for (size_t d = 0; d < rank; ++d)
                {
                    int32_t start = new_starts[d];
                    if (start < 0)
                        start += aClass.shape[d];
                    sliceOffset += start * sliceStrides[d];
                    sliceStrides[d] *= new_steps[d];
                    sliceShape[d] = (new_ends[d] - new_starts[d] + new_steps[d] - 1) / new_steps[d];
                }

                uint32_t newSlice = addOpToEGraph(egraph, OpType::SLICE, {aClassId, nStartsId, nEndsId, nStepsId}, sliceShape, sliceStrides, sliceOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                uint32_t newContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {newSlice}, sliceShape, calcContiguousStrides(sliceShape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                std::vector<uint64_t> childStrides = calcContiguousStrides(sliceShape);
                std::vector<uint64_t> pStrides(rank);
                for (size_t d = 0; d < rank; ++d)
                    pStrides[d] = childStrides[dims[d]];

                uint32_t newPermute = addOpToEGraph(egraph, OpType::PERMUTE, {newContig, opNode.children[1]}, sliceNode.shape, pStrides, 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                uint32_t newPermuteContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {newPermute}, sliceNode.shape, calcContiguousStrides(sliceNode.shape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t op_cache = createCacheInputNode(egraph, opNode, srcClass, partialPathId, eclassToLogical);
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, newPermuteContig, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

struct SlicePushDownReshape : public Rule
{
    struct MatchKey
    {
        uint32_t eNodeIdx;
        uint32_t sliceNodeIdx;
        uint32_t opNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && sliceNodeIdx == o.sliceNodeIdx && opNodeIdx == o.opNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^
                   (std::hash<uint32_t>{}(k.sliceNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.opNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownReshape(bool allow = false) : allowPushDownOnProtected(allow) {}

    std::string name() const override { return "SlicePushDownReshape"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;
        uint32_t sliceClass = egraph.findConst(enode.children[0]);

        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &sliceNode = egraph.getENodes()[sliceIdx];
            if (sliceNode.opType == OpType::SLICE && sliceNode.children.size() == 4)
            {
                uint32_t srcClass = egraph.findConst(sliceNode.children[0]);
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, protectedEClasses, egraph))
                    continue;

                const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
                for (uint32_t srcIdx : srcEnodes)
                {
                    if (egraph.getENodes()[srcIdx].opType == OpType::RESHAPE)
                    {
                        MatchKey key{eNodeIdx, sliceIdx, srcIdx};
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
        const ENode &contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);
        uint32_t sliceClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> sliceNodes;
        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENodes()[sliceIdx].opType == OpType::SLICE && egraph.getENodes()[sliceIdx].children.size() == 4)
            {
                sliceNodes.push_back(sliceIdx);
            }
        }

        for (uint32_t sliceIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceIdx];
            uint32_t srcClass = egraph.find(sliceNode.children[0]);

            const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
            for (uint32_t srcIdx : srcEnodes)
            {
                const ENode opNode = egraph.getENodes()[srcIdx];
                if (opNode.opType != OpType::RESHAPE)
                    continue;

                MatchKey key{eNodeIdx, sliceIdx, srcIdx};
                if (!visited.insert(key).second)
                    continue;

                uint32_t aClassId = opNode.children[0];
                const EClass aClass = egraph.getEClass(egraph.find(aClassId));
                std::vector<uint32_t> aShape = aClass.shape;
                uint32_t aRank = aShape.size();

                auto starts = getConstInt32(egraph, sliceNode.children[1]);
                auto ends = getConstInt32(egraph, sliceNode.children[2]);
                auto steps = getConstInt32(egraph, sliceNode.children[3]);
                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                size_t rank = opNode.shape.size();
                while (starts.size() < rank)
                    starts.push_back(0);
                while (ends.size() < rank)
                    ends.push_back(opNode.shape[ends.size()]);
                while (steps.size() < rank)
                    steps.push_back(1);

                bool stepsOk = true;
                for (auto s : steps)
                    if (s != 1)
                        stepsOk = false;
                if (!stepsOk)
                    continue;

                for (size_t d = 0; d < rank; ++d)
                {
                    if (starts[d] < 0)
                        starts[d] += opNode.shape[d];
                    if (ends[d] < 0)
                        ends[d] += opNode.shape[d];
                }

                uint64_t out_vol = 1;
                for (size_t i = 0; i < rank; ++i)
                    out_vol *= (ends[i] - starts[i]);

                std::vector<uint64_t> strides(rank, 1);
                for (int i = rank - 2; i >= 0; --i)
                    strides[i] = strides[i + 1] * opNode.shape[i + 1];

                std::vector<uint32_t> min_coords(aRank, UINT32_MAX);
                std::vector<uint32_t> max_coords(aRank, 0);

                uint64_t num_corners = 1ULL << rank;
                for (uint64_t i = 0; i < num_corners; ++i)
                {
                    uint64_t flat_idx = 0;
                    for (uint32_t d = 0; d < rank; ++d)
                    {
                        uint32_t coord = ((i >> d) & 1) ? (ends[d] - 1) : starts[d];
                        flat_idx += coord * strides[d];
                    }

                    uint64_t temp = flat_idx;
                    for (int d = aRank - 1; d >= 0; --d)
                    {
                        uint32_t c = temp % aShape[d];
                        temp /= aShape[d];
                        if (c < min_coords[d])
                            min_coords[d] = c;
                        if (c > max_coords[d])
                            max_coords[d] = c;
                    }
                }

                uint64_t new_vol = 1;
                std::vector<int32_t> new_starts, new_ends, new_steps(aRank, 1);
                std::vector<uint32_t> sliceShape;
                for (size_t d = 0; d < aRank; ++d)
                {
                    new_starts.push_back(min_coords[d]);
                    new_ends.push_back(max_coords[d] + 1);
                    sliceShape.push_back(max_coords[d] + 1 - min_coords[d]);
                    new_vol *= sliceShape.back();
                }

                if (new_vol != out_vol || out_vol == 0)
                    continue;

                uint32_t partialPathId = srcIdx | 0x80000000;
                uint32_t nStartsId = addIntConst(egraph, new_starts);
                uint32_t nEndsId = addIntConst(egraph, new_ends);
                uint32_t nStepsId = addIntConst(egraph, new_steps);

                std::vector<uint64_t> sliceStrides = aClass.strides;
                uint64_t sliceOffset = aClass.viewOffset;
                for (size_t d = 0; d < aRank; ++d)
                {
                    sliceOffset += new_starts[d] * sliceStrides[d];
                }

                uint32_t newSlice = addOpToEGraph(egraph, OpType::SLICE, {aClassId, nStartsId, nEndsId, nStepsId}, sliceShape, sliceStrides, sliceOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                uint32_t newContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {newSlice}, sliceShape, calcContiguousStrides(sliceShape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                std::vector<int32_t> new_target_shape;
                for (size_t d = 0; d < rank; ++d)
                    new_target_shape.push_back(ends[d] - starts[d]);
                uint32_t nShapeId = addIntConst(egraph, new_target_shape);

                uint32_t newReshape = addOpToEGraph(egraph, OpType::RESHAPE, {newContig, nShapeId}, sliceNode.shape, calcContiguousStrides(sliceNode.shape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                uint32_t newReshapeContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {newReshape}, sliceNode.shape, calcContiguousStrides(sliceNode.shape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t op_cache = createCacheInputNode(egraph, opNode, srcClass, partialPathId, eclassToLogical);
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, newReshapeContig, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

struct SlicePushDownConcat : public Rule
{
    struct MatchKey
    {
        uint32_t eNodeIdx;
        uint32_t sliceNodeIdx;
        uint32_t opNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && sliceNodeIdx == o.sliceNodeIdx && opNodeIdx == o.opNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^
                   (std::hash<uint32_t>{}(k.sliceNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.opNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownConcat(bool allow = false) : allowPushDownOnProtected(allow) {}

    std::string name() const override { return "SlicePushDownConcat"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;
        uint32_t sliceClass = egraph.findConst(enode.children[0]);

        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &sliceNode = egraph.getENodes()[sliceIdx];
            if (sliceNode.opType == OpType::SLICE && sliceNode.children.size() == 4)
            {
                uint32_t srcClass = egraph.findConst(sliceNode.children[0]);
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, protectedEClasses, egraph))
                    continue;

                const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
                for (uint32_t srcIdx : srcEnodes)
                {
                    if (egraph.getENodes()[srcIdx].opType == OpType::CONCAT)
                    {
                        MatchKey key{eNodeIdx, sliceIdx, srcIdx};
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
        const ENode &contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);
        uint32_t sliceClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> sliceNodes;
        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENodes()[sliceIdx].opType == OpType::SLICE && egraph.getENodes()[sliceIdx].children.size() == 4)
            {
                sliceNodes.push_back(sliceIdx);
            }
        }

        for (uint32_t sliceIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceIdx];
            uint32_t srcClass = egraph.find(sliceNode.children[0]);

            const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
            for (uint32_t srcIdx : srcEnodes)
            {
                const ENode opNode = egraph.getENodes()[srcIdx];
                if (opNode.opType != OpType::CONCAT)
                    continue;

                MatchKey key{eNodeIdx, sliceIdx, srcIdx};
                if (!visited.insert(key).second)
                    continue;

                auto starts = getConstInt32(egraph, sliceNode.children[1]);
                auto ends = getConstInt32(egraph, sliceNode.children[2]);
                auto steps = getConstInt32(egraph, sliceNode.children[3]);
                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                size_t rank = opNode.shape.size();
                while (starts.size() < rank)
                    starts.push_back(0);
                while (ends.size() < rank)
                    ends.push_back(opNode.shape[ends.size()]);
                while (steps.size() < rank)
                    steps.push_back(1);

                for (size_t d = 0; d < rank; ++d)
                {
                    if (starts[d] < 0)
                        starts[d] += opNode.shape[d];
                    if (ends[d] < 0)
                        ends[d] += opNode.shape[d];
                }

                auto axisVec = getConstInt32(egraph, opNode.children.back());
                if (axisVec.empty())
                    continue;
                int32_t axis = axisVec[0];
                if (axis < 0)
                    axis += rank;

                if (axis < 0 || axis >= (int32_t)rank || steps[axis] != 1)
                    continue;

                std::vector<uint32_t> inputs;
                std::vector<uint32_t> offsets;
                uint32_t current_offset = 0;
                for (size_t i = 0; i < opNode.children.size() - 1; ++i)
                {
                    inputs.push_back(opNode.children[i]);
                    offsets.push_back(current_offset);
                    current_offset += egraph.getEClass(egraph.find(opNode.children[i])).shape[axis];
                }

                int32_t s_start = starts[axis];
                int32_t s_end = ends[axis];

                std::vector<uint32_t> valid_sliced_inputs;
                uint32_t partialPathId = srcIdx | 0x80000000;

                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    uint32_t inClass = egraph.find(inputs[i]);
                    uint32_t L = egraph.getEClass(inClass).shape[axis];
                    int32_t in_start = std::max(0, s_start - (int32_t)offsets[i]);
                    int32_t in_end = std::min((int32_t)L, s_end - (int32_t)offsets[i]);

                    if (in_start >= in_end)
                        continue;

                    std::vector<int32_t> cur_starts = starts;
                    std::vector<int32_t> cur_ends = ends;
                    cur_starts[axis] = in_start;
                    cur_ends[axis] = in_end;

                    uint32_t curStartsId = addIntConst(egraph, cur_starts);
                    uint32_t curEndsId = addIntConst(egraph, cur_ends);
                    uint32_t curStepsId = addIntConst(egraph, steps);

                    std::vector<uint32_t> sliceShape = opNode.shape;
                    for (size_t d = 0; d < rank; ++d)
                    {
                        sliceShape[d] = (cur_ends[d] - cur_starts[d] + steps[d] - 1) / steps[d];
                    }

                    std::vector<uint64_t> sliceStrides = egraph.getEClass(inClass).strides;
                    uint64_t sliceOffset = egraph.getEClass(inClass).viewOffset;
                    for (size_t d = 0; d < rank; ++d)
                    {
                        sliceOffset += cur_starts[d] * sliceStrides[d];
                        sliceStrides[d] *= steps[d];
                    }

                    uint32_t newSlice = addOpToEGraph(egraph, OpType::SLICE, {inClass, curStartsId, curEndsId, curStepsId}, sliceShape, sliceStrides, sliceOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                    uint32_t newContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {newSlice}, sliceShape, calcContiguousStrides(sliceShape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                    valid_sliced_inputs.push_back(newContig);
                }

                if (valid_sliced_inputs.empty())
                    continue;

                uint32_t resultClass;
                if (valid_sliced_inputs.size() == 1)
                {
                    resultClass = valid_sliced_inputs[0];
                }
                else
                {
                    valid_sliced_inputs.push_back(opNode.children.back());
                    resultClass = addOpToEGraph(egraph, OpType::CONCAT, valid_sliced_inputs, sliceNode.shape, calcContiguousStrides(sliceNode.shape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                }

                uint32_t resultContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {resultClass}, sliceNode.shape, calcContiguousStrides(sliceNode.shape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t op_cache = createCacheInputNode(egraph, opNode, srcClass, partialPathId, eclassToLogical);
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, resultContig, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

struct SlicePushDownRepeat : public Rule
{
    struct MatchKey
    {
        uint32_t eNodeIdx;
        uint32_t sliceNodeIdx;
        uint32_t opNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && sliceNodeIdx == o.sliceNodeIdx && opNodeIdx == o.opNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::size_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^
                   (std::hash<uint32_t>{}(k.sliceNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.opNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownRepeat(bool allow = false) : allowPushDownOnProtected(allow) {}

    std::string name() const override { return "SlicePushDownRepeat"; }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::CONTIGUOUS || enode.children.empty())
            return false;
        uint32_t sliceClass = egraph.findConst(enode.children[0]);

        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &sliceNode = egraph.getENodes()[sliceIdx];
            if (sliceNode.opType == OpType::SLICE && sliceNode.children.size() == 4)
            {
                uint32_t srcClass = egraph.findConst(sliceNode.children[0]);
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, protectedEClasses, egraph))
                    continue;

                const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
                for (uint32_t srcIdx : srcEnodes)
                {
                    if (egraph.getENodes()[srcIdx].opType == OpType::REPEAT)
                    {
                        MatchKey key{eNodeIdx, sliceIdx, srcIdx};
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
        const ENode &contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);
        uint32_t sliceClass = egraph.find(contigNode.children[0]);

        std::vector<uint32_t> sliceNodes;
        for (uint32_t sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENodes()[sliceIdx].opType == OpType::SLICE && egraph.getENodes()[sliceIdx].children.size() == 4)
            {
                sliceNodes.push_back(sliceIdx);
            }
        }

        for (uint32_t sliceIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENodes()[sliceIdx];
            uint32_t srcClass = egraph.find(sliceNode.children[0]);
            const std::vector<uint32_t> srcEnodes = egraph.getEClass(srcClass).enodes;
            for (uint32_t srcIdx : srcEnodes)
            {
                const ENode opNode = egraph.getENodes()[srcIdx];
                if (opNode.opType != OpType::REPEAT)
                    continue;

                MatchKey key{eNodeIdx, sliceIdx, srcIdx};
                if (!visited.insert(key).second)
                    continue;

                auto repeatsVec = getConstInt32(egraph, opNode.children[1]);
                auto axisVec = getConstInt32(egraph, opNode.children[2]);
                if (repeatsVec.empty() || axisVec.empty())
                    continue;
                int32_t repeats = repeatsVec[0];
                int32_t axis = axisVec[0];

                auto starts = getConstInt32(egraph, sliceNode.children[1]);
                auto ends = getConstInt32(egraph, sliceNode.children[2]);
                auto steps = getConstInt32(egraph, sliceNode.children[3]);
                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                size_t rank = opNode.shape.size();
                while (starts.size() < rank)
                    starts.push_back(0);
                while (ends.size() < rank)
                    ends.push_back(opNode.shape[ends.size()]);
                while (steps.size() < rank)
                    steps.push_back(1);

                if (axis < 0)
                    axis += rank;
                if (axis < 0 || axis >= (int32_t)rank || steps[axis] != 1)
                    continue;

                for (size_t d = 0; d < rank; ++d)
                {
                    if (starts[d] < 0)
                        starts[d] += opNode.shape[d];
                    if (ends[d] < 0)
                        ends[d] += opNode.shape[d];
                }

                int32_t s_start = starts[axis];
                int32_t s_end = ends[axis];

                int32_t in_start = s_start / repeats;
                int32_t in_end = (s_end + repeats - 1) / repeats;

                std::vector<int32_t> new_starts = starts;
                std::vector<int32_t> new_ends = ends;
                new_starts[axis] = in_start;
                new_ends[axis] = in_end;

                uint32_t partialPathId = srcIdx | 0x80000000;
                uint32_t nStartsId = addIntConst(egraph, new_starts);
                uint32_t nEndsId = addIntConst(egraph, new_ends);
                uint32_t nStepsId = addIntConst(egraph, steps);

                uint32_t aClassId = egraph.find(opNode.children[0]);
                const EClass aClass = egraph.getEClass(aClassId);

                std::vector<uint32_t> sliceInShape = aClass.shape;
                std::vector<uint64_t> sliceInStrides = aClass.strides;
                uint64_t sliceInOffset = aClass.viewOffset;

                for (size_t d = 0; d < rank; ++d)
                {
                    sliceInShape[d] = (new_ends[d] - new_starts[d] + steps[d] - 1) / steps[d];
                    sliceInOffset += new_starts[d] * sliceInStrides[d];
                    sliceInStrides[d] *= steps[d];
                }

                uint32_t sliceIn = addOpToEGraph(egraph, OpType::SLICE, {aClassId, nStartsId, nEndsId, nStepsId}, sliceInShape, sliceInStrides, sliceInOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                uint32_t contigIn = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceIn}, sliceInShape, calcContiguousStrides(sliceInShape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                std::vector<uint32_t> repNewShape = sliceInShape;
                repNewShape[axis] *= repeats;
                std::vector<uint64_t> repNewStrides = calcContiguousStrides(sliceInShape);
                for (size_t d = 0; d < rank; ++d)
                {
                    if (sliceInShape[d] != repNewShape[d])
                    {
                        repNewStrides[d] = 0;
                    }
                }

                uint32_t repNew = addOpToEGraph(egraph, OpType::REPEAT, {contigIn, opNode.children[1], opNode.children[2]}, repNewShape, repNewStrides, 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                int32_t fine_start = s_start % repeats;
                int32_t fine_end = fine_start + (s_end - s_start);

                std::vector<int32_t> fine_starts(rank, 0);
                std::vector<int32_t> fine_ends;
                for (auto x : repNewShape)
                    fine_ends.push_back((int32_t)x);
                fine_starts[axis] = fine_start;
                fine_ends[axis] = fine_end;

                std::vector<int32_t> fine_steps(rank, 1);

                uint32_t fStartsId = addIntConst(egraph, fine_starts);
                uint32_t fEndsId = addIntConst(egraph, fine_ends);
                uint32_t fStepsId = addIntConst(egraph, fine_steps);

                std::vector<uint64_t> sliceOutStrides = repNewStrides;
                uint64_t sliceOutOffset = 0;
                for (size_t d = 0; d < rank; ++d)
                {
                    sliceOutOffset += fine_starts[d] * sliceOutStrides[d];
                }

                uint32_t sliceOut = addOpToEGraph(egraph, OpType::SLICE, {repNew, fStartsId, fEndsId, fStepsId}, sliceNode.shape, sliceOutStrides, sliceOutOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);
                uint32_t contigOut = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceOut}, sliceNode.shape, calcContiguousStrides(sliceNode.shape), 0, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                const EClass srcEClass = egraph.getEClass(egraph.find(srcClass));
                uint32_t op_cache = createCacheInputNode(egraph, opNode, srcClass, partialPathId, eclassToLogical);
                uint32_t scatterClass = addOpToEGraph(egraph, OpType::SCATTER, {op_cache, contigOut, sliceNode.children[1], sliceNode.children[2], sliceNode.children[3]}, srcEClass.shape, srcEClass.strides, srcEClass.viewOffset, opNode.dtype, opNode.backend, UINT32_MAX, partialPathId);

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};