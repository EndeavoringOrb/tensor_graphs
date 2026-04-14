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

            TensorNode outputNode;
            outputNode.opType = OpType::FUSED;
            outputNode.opName = pattern.opName;
            outputNode.dtype = matchedClass.dtype;
            outputNode.setShape(matchedClass.shape);
            outputNode.strides = matchedClass.strides;
            outputNode.viewOffset = matchedClass.viewOffset;
            outputNode.backend = matchedClass.backend;

            std::vector<uint64_t> kernelMatches = KernelRegistry::get().findMatchingKernels(
                OpType::FUSED, pattern.opName, outputNode.backend, inputNodes, outputNode, {}, false);

            for (uint64_t uid : kernelMatches)
            {
                const KernelEntry &kernel = KernelRegistry::get().getKernel(uid);
                addFusedNode(egraph, kernel, outputNode.backend, inputs, eNodeIdx);
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

                auto matches = KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", outNode.backend, {inNode}, outNode, {}, true);
                if (matches.empty())
                    return;

                uint32_t newEClass = egraph.addEClass(outNode.getShape(), outNode.strides, outNode.viewOffset, outNode.dtype, outNode.backend);
                for (uint64_t uid : matches)
                {
                    ENode copyNode;
                    copyNode.kernelUid = uid;
                    copyNode.opType = OpType::COPY_TO;
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

                auto matches = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", outNode.backend, {inNode}, outNode, {}, true);
                if (matches.empty())
                    return;

                uint32_t newEClass = egraph.addEClass(outNode.getShape(), outNode.strides, outNode.viewOffset, outNode.dtype, outNode.backend);
                for (uint64_t uid : matches)
                {
                    ENode contigNode;
                    contigNode.kernelUid = uid;
                    contigNode.opType = OpType::CONTIGUOUS;
                    contigNode.children = {currentPid};
                    contigNode.shape = outNode.getShape();
                    contigNode.strides = outNode.strides;
                    contigNode.viewOffset = outNode.viewOffset;
                    contigNode.dtype = outNode.dtype;
                    contigNode.backend = outNode.backend;
                    egraph.addENode(newEClass, contigNode);
                }
                currentPid = newEClass;
                currentClass = egraph.getEClass(newEClass);
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

            auto matches = KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", dummyOut.backend, {dummyIn}, dummyOut, {}, true);
            for (uint64_t uid : matches)
            {
                ENode copyNode;
                copyNode.kernelUid = uid;
                copyNode.opType = OpType::COPY_TO;
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