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
            return false; // TODO: maybe remove this, if optype and opname are equal, this should never be hit

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
                        uint32_t canonCId = egraph.find(cId);
                        const EClass &cClass = egraph.getEClass(canonCId);
                        TensorNode n;
                        n.opType = OpType::INPUT;
                        n.dtype = cClass.dtype;
                        n.setShape(cClass.shape);
                        n.strides = cClass.strides;
                        n.backend = cClass.backend;
                        inputNodes.push_back(n);

                        uint32_t pId = pGraph.input(n.getShape(), n.dtype);
                        pInputs.push_back(pId);
                        if (egraph.constantStaging.count(canonCId))
                        {
                            pGraph.constantStaging[pId] = egraph.constantStaging.at(canonCId);
                        }
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

// Helper for rules
static uint32_t addConstInt32Array(EGraph &egraph, const std::vector<int32_t> &vals)
{
    uint32_t cls = egraph.addEClass({(uint32_t)vals.size()}, {1}, 0, DType::INT32, Backend::CPU);
    ENode n;
    n.opType = OpType::INPUT;
    n.dtype = DType::INT32;
    n.shape = {(uint32_t)vals.size()};
    n.strides = {1};
    n.backend = Backend::CPU;
    egraph.addENode(cls, n);
    std::vector<uint8_t> bytes(vals.size() * sizeof(int32_t));
    std::memcpy(bytes.data(), vals.data(), bytes.size());
    egraph.constantStaging[cls] = bytes;
    return cls;
}

struct ConstantFolding : public Rule
{
    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType == OpType::INPUT || enode.opType == OpType::FUSED)
            return false;
        if (enode.children.empty())
            return false;
        if (!isContiguous(enode.strides, enode.shape))
            return false;
        for (uint32_t child : enode.children)
        {
            if (egraph.constantStaging.find(egraph.findConst(child)) == egraph.constantStaging.end())
            {
                return false;
            }
        }
        return true;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode enode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        if (egraph.constantStaging.count(egraph.find(eclassId)))
            return;

        Graph tempGraph;
        std::vector<uint32_t> pInputs;
        std::vector<TensorNode> inputNodes;
        std::vector<const void *> inputPtrs;
        std::vector<TensorView> inputViews;

        for (uint32_t child : enode.children)
        {
            uint32_t cId = egraph.find(child);
            const EClass &cls = egraph.getEClass(cId);
            uint32_t pId = tempGraph.constant(cls.shape, egraph.constantStaging.at(cId).data(), cls.dtype);
            pInputs.push_back(pId);
            inputNodes.push_back(tempGraph.getNode(pId));
            inputPtrs.push_back(tempGraph.constantStaging.at(pId).data());

            TensorView v;
            v.setShape(cls.shape);
            v.strides = cls.strides;
            v.dtype = cls.dtype;
            inputViews.push_back(v);
        }

        TensorNode &outNode = tempGraph.allocateNode(enode.opType, enode.opName, enode.dtype, pInputs, enode.shape, enode.strides, enode.backend);
        auto matches = KernelRegistry::get().findMatchingKernels(enode.opType, enode.opName, enode.backend, inputNodes, outNode, true);
        if (matches.empty())
            return;

        const auto &kernel = KernelRegistry::get().getKernel(matches[0]);
        std::vector<uint8_t> outBytes(countElements(enode.shape) * getDTypeSize(enode.dtype));
        std::vector<void *> outputPtrs = {outBytes.data()};
        TensorView outView;
        outView.setShape(enode.shape);
        outView.strides = enode.strides;
        outView.dtype = enode.dtype;
        std::vector<TensorView> outViews = {outView};

        if (kernel.run)
        {
            kernel.run(inputPtrs, outputPtrs, inputViews, outViews);
        }

        ENode constNode;
        constNode.opType = OpType::INPUT;
        constNode.dtype = enode.dtype;
        constNode.shape = enode.shape;
        constNode.strides = calcContiguousStrides(enode.shape);
        constNode.backend = Backend::CPU;

        uint32_t newClass = egraph.addEClass(constNode.shape, constNode.strides, 0, constNode.dtype, constNode.backend);
        egraph.addENode(newClass, constNode);
        egraph.constantStaging[newClass] = std::move(outBytes);
        egraph.merge(eclassId, newClass);
    }
};

struct InfinityDomination : public Rule
{
    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::ADD)
            return false;
        for (uint32_t child : enode.children)
        {
            if (egraph.constantStaging.count(egraph.findConst(child)))
                return true;
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode enode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        int constChildIdx = -1;
        for (size_t i = 0; i < enode.children.size(); ++i)
        {
            if (egraph.constantStaging.count(egraph.find(enode.children[i])))
            {
                constChildIdx = i;
                break;
            }
        }
        if (constChildIdx == -1)
            return;

        uint32_t constClass = egraph.find(enode.children[constChildIdx]);
        const auto &bytes = egraph.constantStaging.at(constClass);
        const float *fptr = reinterpret_cast<const float *>(bytes.data());
        uint64_t elems = bytes.size() / sizeof(float);

        std::vector<uint32_t> minCoords(enode.shape.size(), UINT32_MAX);
        std::vector<uint32_t> maxCoords(enode.shape.size(), 0);
        bool foundAny = false;

        const EClass &cCls = egraph.getEClass(constClass);

        for (uint64_t i = 0; i < elems; ++i)
        {
            float val = fptr[getStridedIndex(i, cCls.shape, cCls.strides)];
            if (val > -1e8f)
            {
                foundAny = true;
                auto coords = coordsFromFlatIndex(i, cCls.shape);
                for (size_t d = 0; d < coords.size(); ++d)
                {
                    minCoords[d] = std::min(minCoords[d], coords[d]);
                    maxCoords[d] = std::max(maxCoords[d], coords[d]);
                }
            }
        }

        if (!foundAny)
            return;

        bool isSmaller = false;
        for (size_t d = 0; d < enode.shape.size(); ++d)
        {
            if (minCoords[d] > 0 || maxCoords[d] + 1 < enode.shape[d])
            {
                isSmaller = true;
                break;
            }
        }
        if (!isSmaller)
            return;

        std::vector<int32_t> starts, ends, steps;
        std::vector<uint32_t> sliceShape;
        for (size_t d = 0; d < enode.shape.size(); ++d)
        {
            starts.push_back(minCoords[d]);
            ends.push_back(maxCoords[d] + 1);
            steps.push_back(1);
            sliceShape.push_back(maxCoords[d] + 1 - minCoords[d]);
        }
        uint32_t startsId = addConstInt32Array(egraph, starts);
        uint32_t endsId = addConstInt32Array(egraph, ends);
        uint32_t stepsId = addConstInt32Array(egraph, steps);

        auto makeSlice = [&](uint32_t inputClass)
        {
            const EClass &iCls = egraph.getEClass(egraph.find(inputClass));
            std::vector<uint64_t> sliceStrides(iCls.strides.size());
            uint64_t sliceViewOffset = iCls.viewOffset;
            for (size_t d = 0; d < iCls.strides.size(); ++d)
            {
                sliceViewOffset += starts[d] * iCls.strides[d];
                sliceStrides[d] = iCls.strides[d];
            }
            uint32_t sliceCls = egraph.addEClass(sliceShape, sliceStrides, sliceViewOffset, iCls.dtype, iCls.backend);
            ENode sNode;
            sNode.opType = OpType::SLICE;
            sNode.children = {inputClass, startsId, endsId, stepsId};
            sNode.shape = sliceShape;
            sNode.strides = sliceStrides;
            sNode.viewOffset = sliceViewOffset;
            sNode.dtype = iCls.dtype;
            sNode.backend = iCls.backend;
            egraph.addENode(sliceCls, sNode);

            uint32_t contigCls = egraph.addEClass(sliceShape, calcContiguousStrides(sliceShape), 0, iCls.dtype, iCls.backend);
            ENode cNode;
            cNode.opType = OpType::CONTIGUOUS;
            cNode.children = {sliceCls};
            cNode.shape = sliceShape;
            cNode.strides = calcContiguousStrides(sliceShape);
            cNode.dtype = iCls.dtype;
            cNode.backend = iCls.backend;
            egraph.addENode(contigCls, cNode);
            return contigCls;
        };

        uint32_t sliceA = makeSlice(enode.children[0]);
        uint32_t sliceB = makeSlice(enode.children[1]);

        uint32_t addCls = egraph.addEClass(sliceShape, calcContiguousStrides(sliceShape), 0, enode.dtype, enode.backend);
        ENode addNode;
        addNode.opType = OpType::ADD;
        addNode.children = {sliceA, sliceB};
        addNode.shape = sliceShape;
        addNode.strides = calcContiguousStrides(sliceShape);
        addNode.dtype = enode.dtype;
        addNode.backend = enode.backend;
        egraph.addENode(addCls, addNode);

        // -1e9 base tensor to scatter into
        std::vector<float> infData(countElements(enode.shape), -1e9f);
        uint32_t infCls = egraph.addEClass(enode.shape, calcContiguousStrides(enode.shape), 0, enode.dtype, Backend::CPU);
        ENode infNode;
        infNode.opType = OpType::INPUT;
        infNode.shape = enode.shape;
        infNode.strides = calcContiguousStrides(enode.shape);
        infNode.dtype = enode.dtype;
        infNode.backend = Backend::CPU;
        egraph.addENode(infCls, infNode);
        std::vector<uint8_t> infBytes(infData.size() * sizeof(float));
        std::memcpy(infBytes.data(), infData.data(), infBytes.size());
        egraph.constantStaging[infCls] = infBytes;

        ENode scatterNode;
        scatterNode.opType = OpType::SCATTER;
        scatterNode.children = {infCls, addCls, startsId, endsId, stepsId};
        scatterNode.shape = enode.shape;
        scatterNode.strides = calcContiguousStrides(enode.shape);
        scatterNode.dtype = enode.dtype;
        scatterNode.backend = enode.backend;

        egraph.addENode(eclassId, scatterNode);
    }
};

struct SlicePushBackward : public Rule
{
    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType != OpType::SLICE)
            return false;
        const EClass &childCls = egraph.getEClass(egraph.findConst(enode.children[0]));
        for (uint32_t cId : childCls.enodes)
        {
            if (egraph.getENodes()[cId].opType != OpType::INPUT)
                return true;
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode sliceNode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        const EClass &cCls = egraph.getEClass(egraph.find(sliceNode.children[0]));
        auto startsBytes = egraph.constantStaging.at(egraph.find(sliceNode.children[1]));
        auto endsBytes = egraph.constantStaging.at(egraph.find(sliceNode.children[2]));
        auto stepsBytes = egraph.constantStaging.at(egraph.find(sliceNode.children[3]));

        std::vector<int32_t> starts(startsBytes.size() / 4), ends(endsBytes.size() / 4), steps(stepsBytes.size() / 4);
        std::memcpy(starts.data(), startsBytes.data(), startsBytes.size());
        std::memcpy(ends.data(), endsBytes.data(), endsBytes.size());
        std::memcpy(steps.data(), stepsBytes.data(), stepsBytes.size());

        Region r;
        for (size_t d = 0; d < cCls.shape.size(); ++d)
        {
            uint32_t start = d < starts.size() ? starts[d] : 0;
            uint32_t end = d < ends.size() ? ends[d] : cCls.shape[d];
            r.region.push_back({start, end});
        }

        ShapePropagator prop;
        for (uint32_t childNodeIdx : cCls.enodes)
        {
            const ENode opNode = egraph.getENodes()[childNodeIdx];
            if (opNode.opType == OpType::INPUT || opNode.opType == OpType::FUSED)
                continue;

            Graph mockGraph;
            std::vector<uint32_t> mockParents;
            for (uint32_t gpId : opNode.children)
            {
                uint32_t canonGpId = egraph.find(gpId);
                const EClass &gpCls = egraph.getEClass(canonGpId);
                uint32_t pId = mockGraph.input(gpCls.shape, gpCls.dtype);
                mockParents.push_back(pId);
                if (egraph.constantStaging.count(canonGpId))
                {
                    mockGraph.constantStaging[pId] = egraph.constantStaging.at(canonGpId);
                }
            }
            TensorNode &tNode = mockGraph.allocateNode(opNode.opType, opNode.opName, opNode.dtype, mockParents, opNode.shape, opNode.strides, opNode.backend);

            auto backwardRegions = prop.backward(tNode, mockGraph, {r});
            std::vector<uint32_t> newChildren;

            for (size_t i = 0; i < opNode.children.size(); ++i)
            {
                const auto &bRegs = backwardRegions[i];
                if (bRegs.empty())
                    continue; // Should not happen in well-formed backward pass

                const Region &pReg = bRegs[0];
                std::vector<int32_t> pSt, pEn, pSp;
                std::vector<uint32_t> sShape;
                for (const auto &dim : pReg.region)
                {
                    pSt.push_back(dim.start);
                    pEn.push_back(dim.stop);
                    pSp.push_back(1);
                    sShape.push_back(dim.stop - dim.start);
                }

                const EClass gpCls = egraph.getEClass(egraph.find(opNode.children[i]));
                uint32_t sId = addConstInt32Array(egraph, pSt);
                uint32_t eId = addConstInt32Array(egraph, pEn);
                uint32_t spId = addConstInt32Array(egraph, pSp);

                uint32_t sCls = egraph.addEClass(sShape, calcContiguousStrides(sShape), 0, gpCls.dtype, gpCls.backend);
                ENode sNode;
                sNode.opType = OpType::SLICE;
                sNode.children = {egraph.find(opNode.children[i]), sId, eId, spId};
                sNode.shape = sShape;
                sNode.strides = calcContiguousStrides(sShape);
                sNode.dtype = gpCls.dtype;
                sNode.backend = gpCls.backend;
                egraph.addENode(sCls, sNode);

                uint32_t cCls = egraph.addEClass(sShape, calcContiguousStrides(sShape), 0, gpCls.dtype, gpCls.backend);
                ENode cNode;
                cNode.opType = OpType::CONTIGUOUS;
                cNode.children = {sCls};
                cNode.shape = sShape;
                cNode.strides = calcContiguousStrides(sShape);
                cNode.dtype = gpCls.dtype;
                cNode.backend = gpCls.backend;
                egraph.addENode(cCls, cNode);

                newChildren.push_back(cCls);
            }

            if (newChildren.size() == opNode.children.size())
            {
                ENode newOp = opNode;
                newOp.children = newChildren;
                newOp.shape = sliceNode.shape;
                newOp.strides = calcContiguousStrides(sliceNode.shape);
                egraph.addENode(eclassId, newOp);
            }
        }
    }
};

struct ScatterPushForward : public Rule
{
    bool match(const EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode &enode = egraph.getENodes()[eNodeIdx];
        if (enode.opType == OpType::INPUT || enode.opType == OpType::SCATTER || enode.opType == OpType::FUSED)
            return false;

        for (uint32_t child : enode.children)
        {
            const EClass &cls = egraph.getEClass(egraph.findConst(child));
            for (uint32_t cId : cls.enodes)
            {
                if (egraph.getENodes()[cId].opType == OpType::SCATTER)
                    return true;
            }
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx, const std::unordered_set<uint32_t> &) override
    {
        const ENode enode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        ShapePropagator prop;
        Graph mockGraph;
        std::vector<uint32_t> mockParents;
        for (uint32_t gpId : enode.children)
        {
            uint32_t canonGpId = egraph.find(gpId);
            const EClass &gpCls = egraph.getEClass(canonGpId);
            uint32_t pId = mockGraph.input(gpCls.shape, gpCls.dtype);
            mockParents.push_back(pId);
            if (egraph.constantStaging.count(canonGpId))
            {
                mockGraph.constantStaging[pId] = egraph.constantStaging.at(canonGpId);
            }
        }
        TensorNode &tNode = mockGraph.allocateNode(enode.opType, enode.opName, enode.dtype, mockParents, enode.shape, enode.strides, enode.backend);

        std::vector<std::vector<Region>> parentRegions(enode.children.size());

        bool foundScatter = false;
        std::vector<uint32_t> baseChildren = enode.children;
        std::vector<uint32_t> updateChildren = enode.children;
        std::vector<int32_t> starts, ends, steps;

        for (size_t i = 0; i < enode.children.size(); ++i)
        {
            const EClass &cls = egraph.getEClass(egraph.find(enode.children[i]));
            bool hasScatter = false;
            for (uint32_t cId : cls.enodes)
            {
                const ENode &cNode = egraph.getENodes()[cId];
                if (cNode.opType == OpType::SCATTER)
                {
                    baseChildren[i] = cNode.children[0];   // Target Cache
                    updateChildren[i] = cNode.children[1]; // Updates

                    if (!foundScatter)
                    { // Pick bounding box from first scatter
                        auto stB = egraph.constantStaging.at(egraph.find(cNode.children[2]));
                        auto enB = egraph.constantStaging.at(egraph.find(cNode.children[3]));
                        auto spB = egraph.constantStaging.at(egraph.find(cNode.children[4]));
                        starts.resize(stB.size() / 4);
                        ends.resize(enB.size() / 4);
                        steps.resize(spB.size() / 4);
                        std::memcpy(starts.data(), stB.data(), stB.size());
                        std::memcpy(ends.data(), enB.data(), enB.size());
                        std::memcpy(steps.data(), spB.data(), spB.size());
                        foundScatter = true;
                    }

                    Region r;
                    for (size_t d = 0; d < cls.shape.size(); ++d)
                    {
                        r.region.push_back({(uint32_t)starts[d], (uint32_t)ends[d]});
                    }
                    parentRegions[i].push_back(r);
                    hasScatter = true;
                    break;
                }
            }
            if (!hasScatter)
            {
                Region full;
                for (uint32_t s : cls.shape)
                    full.region.push_back({0, s});
                parentRegions[i].push_back(full);
            }
        }

        if (!foundScatter)
            return;

        auto forwardRegions = prop.forward(tNode, mockGraph, parentRegions);
        if (forwardRegions.empty())
            return;
        const Region &fReg = forwardRegions[0];

        std::vector<int32_t> fSt, fEn, fSp;
        std::vector<uint32_t> uShape;
        for (const auto &dim : fReg.region)
        {
            fSt.push_back(dim.start);
            fEn.push_back(dim.stop);
            fSp.push_back(1);
            uShape.push_back(dim.stop - dim.start);
        }

        // Create Base Op
        ENode baseOp = enode;
        baseOp.children = baseChildren;
        uint32_t baseCls = egraph.addEClass(baseOp.shape, baseOp.strides, 0, baseOp.dtype, baseOp.backend);
        egraph.addENode(baseCls, baseOp);

        // Create Update Op
        ENode updateOp = enode;
        updateOp.children = updateChildren;
        updateOp.shape = uShape;
        updateOp.strides = calcContiguousStrides(uShape);
        uint32_t updateCls = egraph.addEClass(uShape, updateOp.strides, 0, updateOp.dtype, updateOp.backend);
        egraph.addENode(updateCls, updateOp);

        // Create Scatter
        ENode scNode;
        scNode.opType = OpType::SCATTER;
        scNode.children = {baseCls, updateCls, addConstInt32Array(egraph, fSt), addConstInt32Array(egraph, fEn), addConstInt32Array(egraph, fSp)};
        scNode.shape = enode.shape;
        scNode.strides = calcContiguousStrides(enode.shape);
        scNode.dtype = enode.dtype;
        scNode.backend = enode.backend;

        egraph.addENode(eclassId, scNode);
    }
};