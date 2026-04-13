#pragma once
#include "core/graph.hpp"
#include "core/hashing.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"
#include "core/egraph.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <string>

struct Rule
{
    virtual ~Rule() = default;
    virtual bool match(const EGraph &egraph, uint32_t eNodeIdx) const = 0;
    virtual void apply(EGraph &egraph, uint32_t eNodeIdx) const = 0;
};

// a*(b+c) -> (a*b)+(a*c)
struct DistributiveProperty : public Rule
{
    bool match(const EGraph &egraph, uint32_t eNodeIdx) const override
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

    void apply(EGraph &egraph, uint32_t eNodeIdx) const override
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
        uint32_t rootId;
        std::vector<uint32_t> variables;
        std::vector<DType> dtypes;
        std::vector<std::vector<uint32_t>> dummyShapes;
        Graph graph;
    };

    std::vector<Pattern> patterns;

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
            pattern.dtypes = entry.dtypes;
            pattern.dummyShapes = entry.dummyShapes;
            patterns.push_back(std::move(pattern));
        }
    }

    bool match(const EGraph &egraph, uint32_t eNodeIdx) const override
    {
        for (const auto &pattern : patterns)
        {
            std::unordered_map<uint32_t, uint32_t> binding;
            if (matchPattern(eNodeIdx, egraph, pattern.rootId, pattern.graph, pattern.variables, binding, pattern.dtypes))
            {
                return true;
            }
        }
        return false;
    }

    void apply(EGraph &egraph, uint32_t eNodeIdx) const override
    {
        for (const auto &pattern : patterns)
        {
            std::unordered_map<uint32_t, uint32_t> binding;
            if (matchPattern(eNodeIdx, egraph, pattern.rootId, pattern.graph, pattern.variables, binding, pattern.dtypes))
            {
                std::vector<uint32_t> inputs;
                inputs.reserve(pattern.variables.size());
                for (uint32_t var : pattern.variables)
                    inputs.push_back(binding[var]);

                for (const auto &kernel : KernelRegistry::get().getAllKernels())
                {
                    if (kernel.opType == OpType::FUSED && kernel.opName == pattern.opName)
                    {
                        for (Backend targetBackend : kernel.backends)
                        {
                            addFusedNode(egraph, kernel, targetBackend, inputs, eNodeIdx);
                        }
                    }
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
                // The planner's extraction phase evaluates inplace safety incrementally
                // using local_ref_counts during path traversal. We do not need to modify
                // static ENode refCounts here to ensure correct inplace checks.
                adaptedParents.push_back(pid);
                continue;
            }
            else
            {
                uint32_t currentPid = pid;
                EClass currentClass = parent;

                if (needCopy)
                {
                    TensorNode inNode;
                    inNode.opType = OpType::INPUT; // dummy
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
                        return; // cannot copy

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
                    inNode.opType = OpType::INPUT; // dummy
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
                        return; // cannot make contiguous

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
        }

        const ENode &oldENode = egraph.getENodes()[eNodeIdx];
        uint32_t eclassId = egraph.getENodeEClass(eNodeIdx);

        ENode enode;
        enode.kernelUid = kernel.uid;
        enode.opType = kernel.opType;
        enode.opName = kernel.opName;
        for (uint32_t pid : adaptedParents)
            enode.children.push_back(pid);
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

    static bool matchPattern(uint32_t eNodeIdx, const EGraph &egraph,
                             uint32_t patternId, const Graph &patternGraph,
                             const std::vector<uint32_t> &patternVariables,
                             std::unordered_map<uint32_t, uint32_t> &binding,
                             const std::vector<DType> &patternDtypes)
    {
        auto itVar = std::find(patternVariables.begin(), patternVariables.end(), patternId);
        if (itVar != patternVariables.end())
        {
            size_t varIdx = static_cast<size_t>(std::distance(patternVariables.begin(), itVar));
            const ENode &eNode = egraph.getENodes()[eNodeIdx];
            if (varIdx < patternDtypes.size() && eNode.dtype != patternDtypes[varIdx])
                return false;

            if (binding.count(patternId))
            {
                return binding[patternId] == eNodeIdx;
            }
            binding[patternId] = eNodeIdx;
            return true;
        }

        const ENode &eNode = egraph.getENodes()[eNodeIdx];
        const auto &pNode = patternGraph.getNode(patternId);

        if (eNode.opType != pNode.opType)
            return false;
        if (eNode.opType == OpType::FUSED && eNode.opName != pNode.opName)
            return false;
        if (eNode.children.size() != pNode.parentIds.size())
            return false;

        for (size_t i = 0; i < eNode.children.size(); ++i)
        {
            if (!matchPattern(eNode.children[i], egraph, pNode.parentIds[i], patternGraph, patternVariables, binding, patternDtypes))
            {
                return false;
            }
        }

        return true;
    }
};