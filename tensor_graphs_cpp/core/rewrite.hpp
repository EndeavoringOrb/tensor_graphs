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

struct Rule
{
    virtual ~Rule() = default;
    virtual bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) = 0;
    virtual void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) = 0;
};

inline uint32_t addOpToEGraph(EGraph &egraph, OpType op, const std::vector<uint32_t> &children, const std::vector<uint32_t> &shape, const std::vector<uint64_t> &st, uint64_t viewOffset, DType dtype, Backend backend, uint32_t targetEClass = UINT32_MAX)
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
            n.strides = st;
            n.viewOffset = viewOffset;
            n.dtype = dtype;
            n.backend = backend;
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

// a*(b+c) -> (a*b)+(a*c)
struct DistributiveProperty : public Rule
{
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
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

        // Optimization: Canonicalize protected classes once per match call
        std::unordered_set<uint32_t> canonicalProtected;
        for (uint32_t id : protectedEClasses)
        {
            canonicalProtected.insert(egraph.findConst(id));
        }

        for (const auto &pattern : it->second)
        {
            std::unordered_map<uint32_t, uint32_t> binding;
            if (matchPatternNode(eNodeIdx, egraph, pattern.rootId, pattern, binding, canonicalProtected))
            {
                activeMatches.push_back({&pattern, std::move(binding)});
            }
        }
        return !activeMatches.empty();
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
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
                const EClass &parent = egraph.getEClass(parentEClassId);
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

            const EClass &matchedClass = egraph.getEClass(egraph.getENodeEClass(eNodeIdx));

            // Setup the backends we want the EGraph to explore
            std::vector<Backend> targetBackends = {Backend::CPU};
#ifdef USE_CUDA
            targetBackends.push_back(Backend::CUDA);
#endif

            // Explore Fused kernels on ALL available backends
            for (Backend targetBackend : targetBackends)
            {
                TensorNode outputNode;
                outputNode.opType = OpType::FUSED;
                outputNode.opName = pattern.opName;
                outputNode.dtype = matchedClass.dtype;
                outputNode.setShape(matchedClass.shape);
                outputNode.strides = matchedClass.strides;
                outputNode.viewOffset = matchedClass.viewOffset;
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
            const EClass &parent = egraph.getEClass(pid);

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

        const ENode &oldENode = egraph.getENodes()[eNodeIdx];
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
                                  const std::unordered_set<uint32_t> &canonicalProtected)
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

        // Optimization: O(1) check for protected eclasses
        if (patternId != pattern.rootId)
        {
            if (canonicalProtected.count(canonicalClassIdx))
                return false;
        }

        const EClass &eclass = egraph.getEClass(canonicalClassIdx);
        for (uint32_t enodeId : eclass.enodes)
        {
            std::unordered_map<uint32_t, uint32_t> localBinding = binding;
            if (matchPatternNode(enodeId, egraph, patternId, pattern, localBinding, canonicalProtected))
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
                                 const std::unordered_set<uint32_t> &canonicalProtected)
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
            if (!matchPatternClass(eNode.children[i], egraph, pNode.parentIds[i], pattern, binding, canonicalProtected))
            {
                return false;
            }
        }
        return true;
    }
};

// copyto(contiguous(x)) -> contiguous(copyto(x))
struct CopyToOfContiguous : public Rule
{
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode copyToNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t contigNodeIdx = findOpNode(egraph, copyToNode.children[0], OpType::CONTIGUOUS);
        if (contigNodeIdx == UINT32_MAX)
            return;

        const ENode contigNode = egraph.getENodes()[contigNodeIdx];
        uint32_t xClassId = contigNode.children[0];
        const EClass &xClass = egraph.getEClass(xClassId);

        // --- Look up CopyTo(x) kernel ---
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

        // --- Look up Contiguous kernel on target backend ---
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

        // --- Create CopyTo(x) eclass ---
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

        // --- Create Contiguous(CopyTo(x)) and add to original eclass ---
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

// contiguous(copyto(x)) -> copyto(contiguous(x))
struct ContiguousOfCopyTo : public Rule
{
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode contigNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t copyToNodeIdx = findOpNode(egraph, contigNode.children[0], OpType::COPY_TO);
        if (copyToNodeIdx == UINT32_MAX)
            return;

        const ENode copyToNode = egraph.getENodes()[copyToNodeIdx];
        uint32_t xClassId = copyToNode.children[0];
        const EClass &xClass = egraph.getEClass(xClassId);

        // --- Look up Contiguous(x) kernel on x's original backend ---
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

        // --- Look up CopyTo kernel from x's backend to target backend ---
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

        // --- Create Contiguous(x) eclass ---
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

        // --- Create CopyTo(Contiguous(x)) and add to original eclass ---
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
    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];

        if (enode.opType == OpType::INPUT || enode.children.empty() || enode.opType == OpType::CONTIGUOUS)
            return false;

        for (uint32_t childEClassId : enode.children)
        {
            const EClass &childCls = egraph.getEClass(childEClassId);
            for (uint32_t childENodeIdx : childCls.enodes)
            {
                const ENode &childENode = egraph.getENodes()[childENodeIdx];
                if (childENode.opType == OpType::CONTIGUOUS && !childENode.children.empty())
                {
                    return true;
                }
            }
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode consumerNode = egraph.getENodes()[eNodeIdx];
        uint32_t consumerEClassId = egraph.getENodeEClass(eNodeIdx);

        for (size_t i = 0; i < consumerNode.children.size(); ++i)
        {
            uint32_t childEClassId = consumerNode.children[i];
            // Copy to avoid iterator invalidation during addENode
            std::vector<uint32_t> childENodeIndices = egraph.getEClass(childEClassId).enodes;

            for (uint32_t childENodeIdx : childENodeIndices)
            {
                // Copy by value because getENodes() might reallocate during addENode
                ENode childENode = egraph.getENodes()[childENodeIdx];

                if (childENode.opType == OpType::CONTIGUOUS && !childENode.children.empty())
                {
                    uint32_t sourceEClassId = childENode.children[0];

                    std::vector<uint32_t> optimizedChildren = consumerNode.children;
                    optimizedChildren[i] = sourceEClassId;

                    // Fast path: source is natively contiguous, old kernel still applies perfectly
                    if (isContiguous(egraph.getEClass(sourceEClassId)))
                    {
                        ENode n = consumerNode;
                        n.children = optimizedChildren;
                        egraph.addENode(consumerEClassId, n);
                        break;
                    }

                    // Strict validation: Does a kernel exist that natively supports this strided tensor?
                    std::vector<TensorNode> inputNodes;
                    Graph pGraph;
                    std::vector<uint32_t> pInputs;

                    for (uint32_t cId : optimizedChildren)
                    {
                        const EClass &cClass = egraph.getEClass(cId);
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
                    break;
                }
            }
        }
    }
};

struct ConstantFolding : public Rule
{
    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &eNode = egraph.getENodes()[eNodeIdx];

        // If the operation is already an input/constant, nothing to fold.
        if (eNode.opType == OpType::INPUT)
            return false;

        if (eNode.children.empty())
            return false;

        // Ensure all children are fully evaluated constants
        for (uint32_t c : eNode.children)
        {
            uint32_t childEClassId = egraph.findConst(c);
            if (egraph.constantStaging.find(childEClassId) == egraph.constantStaging.end())
                return false;
        }

        uint32_t eclassId = egraph.findConst(egraph.getENodeEClass(eNodeIdx));
        if (egraph.constantStaging.find(eclassId) != egraph.constantStaging.end())
            return false; // Already folded

        // If there's already a COPY_TO node connecting this to an evaluated CPU constant, skip folding to prevent infinite recursion
        for (uint32_t enodeId : egraph.getEClass(eclassId).enodes)
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode eNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.find(egraph.getENodeEClass(eNodeIdx));

        std::vector<TensorNode> inputNodes;
        std::vector<TensorView> inViews;
        std::vector<const void *> kernelInputs;

        for (uint32_t c : eNode.children)
        {
            uint32_t childEClassId = egraph.find(c);
            const EClass &childCls = egraph.getEClass(childEClassId);

            TensorNode inNode;
            inNode.opType = OpType::INPUT;
            inNode.dtype = childCls.dtype;
            inNode.setShape(childCls.shape);
            inNode.strides = childCls.strides;
            inNode.viewOffset = childCls.viewOffset;
            inNode.backend = Backend::CPU; // Constant staged data evaluates on CPU
            inputNodes.push_back(inNode);

            const auto &stagedData = egraph.constantStaging.at(childEClassId);
            size_t offsetBytes = childCls.viewOffset * getDTypeSize(childCls.dtype);

            if (offsetBytes >= stagedData.size() && stagedData.size() > 0)
            {
                return; // Safeguard bounds
            }

            kernelInputs.push_back(stagedData.data() + offsetBytes);
            inViews.push_back(TensorView(inNode, 0));
        }

        TensorNode outNode;
        outNode.opType = eNode.opType;
        outNode.opName = eNode.opName;
        outNode.dtype = eNode.dtype;
        outNode.setShape(eNode.shape);
        outNode.strides = calcContiguousStrides(eNode.shape);
        outNode.viewOffset = 0;
        outNode.backend = Backend::CPU;

        auto matches = KernelRegistry::get().findMatchingKernels(
            eNode.opType, eNode.opName, Backend::CPU, inputNodes, outNode, false, true, true);

        if (matches.empty())
            return;

        const KernelEntry *selectedKernel = nullptr;
        for (uint64_t uid : matches)
        {
            const auto &k = KernelRegistry::get().getKernel(uid);
            if (!k.inplace) // Avoid modifying existing staged constant data
            {
                selectedKernel = &k;
                break;
            }
        }
        if (!selectedKernel)
            return; // Need a non-inplace suitable kernel implementation to fold

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
            outData.resize(getSizeBytes(outNode.getShape(), outNode.dtype));
            std::vector<void *> kernelOutputs = {outData.data()};
            std::vector<TensorView> outViews = {TensorView(outNode, 0)};
            selectedKernel->run(kernelInputs, kernelOutputs, inViews, outViews);
        }

        ENode foldedNode;
        foldedNode.kernelUid = 0;
        foldedNode.opType = OpType::INPUT;
        foldedNode.shape = eNode.shape;
        if (selectedKernel->isView)
        {
            foldedNode.strides = eNode.strides;
            foldedNode.viewOffset = eNode.viewOffset;
        }
        else
        {
            foldedNode.strides = outNode.strides;
            foldedNode.viewOffset = 0;
        }
        foldedNode.dtype = eNode.dtype;
        foldedNode.backend = Backend::CPU;

        Backend originalBackend = egraph.getEClass(eclassId).backend;

        // If identical backends, inject folded node directly.
        if (originalBackend == Backend::CPU)
        {
            egraph.addENode(eclassId, foldedNode);
            egraph.constantStaging[eclassId] = std::move(outData);
        }
        else
        {
            // Non-CPU original eclass. Create CPU eclass bridging to the original via COPY_TO.
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
            copyOutNode.strides = calcContiguousStrides(copyOutNode.getShape());
            copyOutNode.viewOffset = 0;

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
    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
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

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &addNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        uint32_t constIdx = isConstantFloat(egraph, addNode.children[1]) ? 1 : 0;
        uint32_t varIdx = 1 - constIdx;

        uint32_t constClass = egraph.find(addNode.children[constIdx]);
        uint32_t varClass = egraph.find(addNode.children[varIdx]);

        const auto &constData = egraph.constantStaging.at(constClass);
        const float *data = reinterpret_cast<const float *>(constData.data());

        const EClass cClass = egraph.getEClass(constClass);
        uint64_t numElements = countElements(cClass.shape);

        std::vector<uint32_t> minBounds(cClass.shape.size(), UINT32_MAX);
        std::vector<uint32_t> maxBounds(cClass.shape.size(), 0);

        std::vector<Region> nonInfRegions;
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
        }
        nonInfRegions = mergeRegions(nonInfRegions);

        const EClass vClass = egraph.getEClass(varClass);
        const EClass outClass = egraph.getEClass(eclassId);

        std::vector<uint64_t> contigStrides = calcContiguousStrides(outClass.shape);

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

        if (nonInfRegions.empty())
        {
            // add(a, -inf) -> -inf
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

            currentTarget = addOpToEGraph(egraph, OpType::SCATTER, {currentTarget, addId, startsId, endsId, stepsId}, outClass.shape, contigStrides, 0, outClass.dtype, outClass.backend);
        }

        egraph.merge(eclassId, currentTarget);
    }
};

struct SlicePushDownElementwise : public Rule
{
    std::vector<int32_t> getConstInt32(const EGraph &egraph, uint32_t eclassId) const
    {
        uint32_t canon = egraph.findConst(eclassId);
        if (egraph.constantStaging.count(canon))
        {
            const auto &data = egraph.constantStaging.at(canon);
            std::vector<int32_t> res(data.size() / sizeof(int32_t));
            std::memcpy(res.data(), data.data(), data.size());
            return res;
        }
        return {};
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::SLICE || enode.children.size() != 4)
            return false;

        uint32_t srcClass = egraph.findConst(enode.children[0]);
        for (uint32_t srcNodeIdx : egraph.getEClass(srcClass).enodes)
        {
            OpType op = egraph.getENodes()[srcNodeIdx].opType;
            if (op == OpType::ADD || op == OpType::MUL || op == OpType::DIVIDE || op == OpType::POWER ||
                op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST)
            {
                return true;
            }
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &sliceNode = egraph.getENodes()[eNodeIdx];
        uint32_t sliceClass = egraph.getENodeEClass(eNodeIdx);

        uint32_t srcClass = egraph.find(sliceNode.children[0]);
        uint32_t startsId = sliceNode.children[1];
        uint32_t endsId = sliceNode.children[2];
        uint32_t stepsId = sliceNode.children[3];

        auto starts = getConstInt32(egraph, startsId);
        auto ends = getConstInt32(egraph, endsId);
        auto steps = getConstInt32(egraph, stepsId);
        if (starts.empty() || ends.empty() || steps.empty())
            return;

        const std::vector<uint32_t> &sliceShape = sliceNode.shape;
        std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

        for (uint32_t srcNodeIdx : egraph.getEClass(srcClass).enodes)
        {
            const ENode &opNode = egraph.getENodes()[srcNodeIdx];
            OpType op = opNode.opType;
            if (!(op == OpType::ADD || op == OpType::MUL || op == OpType::DIVIDE || op == OpType::POWER ||
                  op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST))
            {
                continue;
            }

            std::vector<uint32_t> newChildren;
            for (uint32_t childId : opNode.children)
            {
                const EClass &cClass = egraph.getEClass(egraph.find(childId));

                std::vector<uint64_t> sliceStrides = cClass.strides;
                uint64_t sliceViewOffset = cClass.viewOffset;
                for (size_t d = 0; d < starts.size() && d < cClass.shape.size(); ++d)
                {
                    int32_t start = starts[d];
                    if (start < 0)
                        start += cClass.shape[d];
                    sliceViewOffset += start * sliceStrides[d];
                    sliceStrides[d] *= steps[d];
                }

                uint32_t childSlice = addOpToEGraph(egraph, OpType::SLICE, {childId, startsId, endsId, stepsId}, sliceShape, sliceStrides, sliceViewOffset, cClass.dtype, sliceNode.backend);

                uint32_t childContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {childSlice}, sliceShape, sliceContigStrides, 0, cClass.dtype, sliceNode.backend);
                newChildren.push_back(childContig);
            }

            addOpToEGraph(egraph, op, newChildren, sliceShape, sliceContigStrides, 0, sliceNode.dtype, sliceNode.backend, sliceClass);
        }
    }
};

struct SlicePushDownDot : public Rule
{
    std::vector<int32_t> getConstInt32(const EGraph &egraph, uint32_t eclassId) const
    {
        uint32_t canon = egraph.findConst(eclassId);
        if (egraph.constantStaging.count(canon))
        {
            const auto &data = egraph.constantStaging.at(canon);
            std::vector<int32_t> res(data.size() / sizeof(int32_t));
            std::memcpy(res.data(), data.data(), data.size());
            return res;
        }
        return {};
    }

    uint32_t addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, Backend::CPU, vals);
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::SLICE || enode.children.size() != 4)
            return false;

        uint32_t srcClass = egraph.findConst(enode.children[0]);
        for (uint32_t srcNodeIdx : egraph.getEClass(srcClass).enodes)
        {
            if (egraph.getENodes()[srcNodeIdx].opType == OpType::DOT)
            {
                return true;
            }
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &protectedEClasses) override
    {
        const ENode &sliceNode = egraph.getENodes()[eNodeIdx];
        uint32_t sliceClass = egraph.getENodeEClass(eNodeIdx);

        uint32_t srcClass = egraph.find(sliceNode.children[0]);
        auto starts = getConstInt32(egraph, sliceNode.children[1]);
        auto ends = getConstInt32(egraph, sliceNode.children[2]);
        auto steps = getConstInt32(egraph, sliceNode.children[3]);

        if (starts.empty() || ends.empty() || steps.empty())
            return;

        for (int32_t s : steps)
        {
            if (s != 1)
                return; // Only support step=1 for pushing down through DOT
        }

        const EClass &outClass = egraph.getEClass(srcClass);
        uint32_t rank = outClass.shape.size();
        if (rank != 2 && rank != 3)
            return; // DOT only supports rank 2 and 3

        while (starts.size() < rank)
            starts.push_back(0);
        while (ends.size() < rank)
            ends.push_back(outClass.shape[ends.size()]);

        for (size_t d = 0; d < rank; ++d)
        {
            if (starts[d] < 0)
                starts[d] += outClass.shape[d];
            if (ends[d] < 0)
                ends[d] += outClass.shape[d];
            starts[d] = std::max(0, starts[d]);
            ends[d] = std::min((int32_t)outClass.shape[d], std::max(starts[d], ends[d]));
        }

        const std::vector<uint32_t> &sliceShape = sliceNode.shape;
        std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

        for (uint32_t srcNodeIdx : egraph.getEClass(srcClass).enodes)
        {
            const ENode &dotNode = egraph.getENodes()[srcNodeIdx];
            if (dotNode.opType != OpType::DOT)
                continue;

            uint32_t aClassId = dotNode.children[0];
            uint32_t bClassId = dotNode.children[1];
            const EClass &aClass = egraph.getEClass(egraph.find(aClassId));
            const EClass &bClass = egraph.getEClass(egraph.find(bClassId));

            uint32_t K = (rank == 2) ? aClass.shape[1] : aClass.shape[2];

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

            auto createSlice = [&](uint32_t classId, const EClass &cls, const std::vector<int32_t> &st, const std::vector<int32_t> &en, uint32_t stId, uint32_t enId, uint32_t stepId)
            {
                std::vector<uint32_t> sShape;
                for (size_t d = 0; d < st.size(); ++d)
                    sShape.push_back(en[d] - st[d]);

                std::vector<uint64_t> sStrides = cls.strides;
                uint64_t sOffset = cls.viewOffset;
                for (size_t d = 0; d < st.size(); ++d)
                {
                    sOffset += st[d] * sStrides[d];
                }

                uint32_t sClass = addOpToEGraph(egraph, OpType::SLICE, {classId, stId, enId, stepId}, sShape, sStrides, sOffset, cls.dtype, sliceNode.backend);
                uint32_t sContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sClass}, sShape, calcContiguousStrides(sShape), 0, cls.dtype, sliceNode.backend);
                return sContig;
            };

            uint32_t aSliced = createSlice(aClassId, aClass, startsA, endsA, startsIdA, endsIdA, stepsIdA);
            uint32_t bSliced = createSlice(bClassId, bClass, startsB, endsB, startsIdB, endsIdB, stepsIdB);

            addOpToEGraph(egraph, OpType::DOT, {aSliced, bSliced}, sliceShape, sliceContigStrides, 0, sliceNode.dtype, sliceNode.backend, sliceClass);
        }
    }
};