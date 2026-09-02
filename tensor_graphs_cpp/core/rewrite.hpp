#pragma once
#include <algorithm>
#include <cstring>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/ops/ops.hpp"
#include "core/repo.hpp"
#include "core/shape_propagator.hpp"

inline std::vector<std::vector<MemSpace>> findMemSpacePaths(MemSpace src, MemSpace dst, const TensorNode &node,
                                                            const std::vector<Engine> &engines)
{
    if (src == dst)
        return {{src}};

    const auto &all_spaces = System::get().getAvailableMemSpaces();
    std::unordered_map<MemSpace, std::vector<MemSpace>> adj;

    for (const auto &s1 : all_spaces)
    {
        for (const auto &s2 : all_spaces)
        {
            if (s1 == s2)
                continue;
            TensorNode dummyIn = node;
            TensorNode dummyOut = node;
            auto refs = KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", {dummyIn}, dummyOut, false, s2,
                                                                  {s1}, engines, false, false, true, true);
            if (!refs.empty())
            {
                adj[s1].push_back(s2);
            }
        }
    }

    std::vector<std::vector<MemSpace>> all_paths;
    std::vector<MemSpace> current_path = {src};
    std::unordered_set<MemSpace> visited = {src};

    std::function<void(MemSpace)> dfs = [&](MemSpace curr) {
        if (curr == dst)
        {
            all_paths.push_back(current_path);
            return;
        }

        auto it = adj.find(curr);
        if (it == adj.end())
            return;

        for (MemSpace next : it->second)
        {
            if (visited.find(next) == visited.end())
            {
                visited.insert(next);
                current_path.push_back(next);
                dfs(next);
                current_path.pop_back();
                visited.erase(next);
            }
        }
    };

    dfs(src);
    return all_paths;
}

inline bool isEClassProtected(EClassId e_class_id, const std::unordered_set<EClassId> &protectedEClasses,
                              const EGraph &egraph)
{
    EClassId canon = egraph.findConst(e_class_id);
    if (protectedEClasses.count(canon))
        return true;
    for (EClassId id : protectedEClasses)
    {
        if (egraph.findConst(id) == canon)
            return true;
    }
    return false;
}

struct RuleCtx
{
    EGraph &egraph;
    const std::unordered_set<EClassId> &protectedEClasses;
    std::unordered_map<EClassId, LogicalId> &eclassToLogical;
    Repo *repo;
    CostModel *costModel = nullptr;
};

struct Rule
{
    virtual ~Rule() = default;
    virtual std::string name() const = 0;
    virtual bool match(uint32_t eNodeIdx, RuleCtx &ctx) = 0;
    virtual void apply(uint32_t eNodeIdx, RuleCtx &ctx) = 0;
};

inline EClassId addOpToEGraph(EGraph &egraph, OpType op, const std::vector<EClassId> &children,
                              const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides, DType dtype,
                              MemSpace mem_space, EClassId targetEClass = EClassId(),
                              SourceLocation loc = SourceLocation::current(), const std::string &debugOrigin = "")
{
    EClassId cls = targetEClass;
    if (cls == EClassId())
    {
        cls = egraph.addEClass(shape, strides, dtype, mem_space);
    }
    std::string origin = debugOrigin.empty() ? toString(loc) : debugOrigin;

    TensorNode outNode;
    outNode.opType = op;
    outNode.dtype = dtype;
    outNode.setShape(shape);
    outNode.strides = strides;

    std::vector<TensorNode> inNodes;
    std::vector<MemSpace> input_mem_spaces;
    for (EClassId c : children)
    {
        const EClass &childCls = egraph.getEClass(egraph.findConst(c));
        TensorNode in;
        in.opType = OpType::INPUT;
        in.dtype = childCls.dtype;
        in.setShape(childCls.shape);
        in.strides = childCls.strides;
        inNodes.push_back(in);
        input_mem_spaces.push_back(childCls.mem_space);
    }

    Graph pGraph;
    std::vector<LogicalId> pInputs;
    for (auto &in : inNodes)
    {
        pInputs.push_back(pGraph.input(in.getShape(), in.dtype));
    }

    const auto &traits = getOpTraits(op);
    LogicalId pRoot = traits.buildPattern ? traits.buildPattern(pGraph, pInputs, dtype) : LogicalId();

    if (pRoot != LogicalId())
    {
        auto matches = KernelRegistry::get().findMatchingKernelsByPattern(
            pGraph, pRoot, inNodes, outNode, false, mem_space, input_mem_spaces, {}, false, false, true);
        if (matches.empty())
        {
            std::stringstream ss;
            ss << "\n[addOpToEGraph] No matching kernel found for the given configuration at " << toString(loc) << "\n"
               << "  Operation:       " << toString(op) << "\n"
               << "  Target MemSpace: " << toString(mem_space) << "\n"
               << "  Expected Output: "
               << "dtype=" << toString(dtype) << ", shape=" << toString(shape) << ", strides=" << toString(strides)
               << "\n"
               << "  Inputs (" << inNodes.size() << "):\n";
            for (uint64_t i = 0; i < inNodes.size(); ++i)
            {
                ss << "    Input #" << i << ": "
                   << "dtype=" << toString(inNodes[i].dtype) << ", shape=" << toString(inNodes[i].getShape())
                   << ", strides=" << toString(inNodes[i].strides) << ", mem_space=" << toString(input_mem_spaces[i])
                   << "\n";
            }

            Error::throw_err(ss.str());
        }
        for (KernelId uid : matches)
        {
            const auto &kernel = KernelRegistry::get().getKernel(uid);
            std::vector<Engine> actual_engines;
            kernel.matches(inNodes, outNode, mem_space, input_mem_spaces, {}, false, false, true, true,
                           &actual_engines);

            ENode n(uid, op, kernel.opName, children, shape, kernel.is_view ? strides : calcContiguousStrides(shape),
                    dtype, mem_space, actual_engines, "", 0, origin);

            cls = egraph.addENode(cls, n);
        }
    }
    return cls;
}

inline EClassId copyTo(EGraph &egraph, EClassId class_id, MemSpace target_mem_space,
                       const std::string &debugOrigin = "")
{
    EClassId canon = egraph.find(class_id);
    const EClass cls = egraph.getEClass(canon);
    if (cls.mem_space == target_mem_space)
        return canon;

    return addOpToEGraph(egraph, OpType::COPY_TO, {canon}, cls.shape, cls.strides, cls.dtype, target_mem_space,
                         EClassId(), SourceLocation::current(), debugOrigin);
}

inline EClassId createCacheInputNode(EGraph &egraph, EClassId sourceClassId,
                                     std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                     const std::string &debugOrigin = "")
{
    EClassId canonSrcClass = egraph.findConst(sourceClassId);
    const EClass srcClass = egraph.getEClass(canonSrcClass);

    EClassId op_cache = egraph.addEClass(srcClass.shape, srcClass.strides, srcClass.dtype, srcClass.mem_space);

    LogicalId srcLogicalId;
    auto it = eclassToLogical.find(canonSrcClass);
    if (it != eclassToLogical.end())
    {
        srcLogicalId = it->second;
    }
    else
    {
        for (const auto &kv : eclassToLogical)
        {
            if (egraph.findConst(kv.first) == canonSrcClass)
            {
                srcLogicalId = kv.second;
                break;
            }
        }
    }

    ENode cacheNode(KernelId{0}, OpType::CACHE, "", {}, srcClass.shape, srcClass.strides, srcClass.dtype,
                    srcClass.mem_space, {}, toString(srcLogicalId), 0, debugOrigin);
    op_cache = egraph.addENode(op_cache, cacheNode);

    eclassToLogical[op_cache] = srcLogicalId;

    return op_cache;
}

struct FusionRule : public Rule
{
    std::string name() const override
    {
        return "FusionRule";
    }

    struct Pattern
    {
        std::string opName;
        OpType rootOpType;
        LogicalId rootId;
        std::vector<LogicalId> variables;
        std::vector<DType> dtypes;
        std::vector<std::vector<uint32_t>> dummyShapes;
        Graph graph;
    };

    struct MatchResult
    {
        const Pattern *pattern;
        std::unordered_map<LogicalId, EClassId> binding;
        std::vector<EClassId> variadicConcatTensorEClasses;
    };

    std::unordered_map<OpType, std::vector<Pattern>> patternsByOp;
    std::vector<MatchResult> activeMatches;

    FusionRule(bool disableFusion = false)
    {
        const auto &refGraphs = ReferenceGraphRegistry::get().getAll();
        for (const auto &pair : refGraphs)
        {
            const auto &entry = pair.second;
            Pattern pattern;
            pattern.opName = pair.first;

            for (uint64_t i = 0; i < entry.min_num_inputs; ++i)
            {
                LogicalId inId = pattern.graph.input(entry.dummyShapes[i], entry.dtypes[i]);
                pattern.variables.push_back(inId);
            }
            pattern.rootId = entry.factory(pattern.variables, pattern.graph);

            if (disableFusion && pattern.graph.nodes.size() > entry.min_num_inputs + 1)
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
        const ENode &eNode = egraph.getENode(ENodeId{eNodeIdx});

        auto it = patternsByOp.find(eNode.getOpType());
        if (it == patternsByOp.end())
            return false;

        for (const auto &pattern : it->second)
        {
            std::unordered_map<LogicalId, EClassId> binding;
            if (matchPatternNode(ENodeId{eNodeIdx}, egraph, pattern.rootId, pattern, binding, ctx.protectedEClasses))
            {
                MatchResult mr;
                mr.pattern = &pattern;
                mr.binding = std::move(binding);

                if (eNode.getOpType() == OpType::CONCAT && eNode.getChildren().size() > 2)
                {
                    for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
                    {
                        mr.variadicConcatTensorEClasses.push_back(egraph.findConst(eNode.getChildren()[i]));
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

            std::vector<EClassId> inputs;
            std::vector<TensorNode> inputNodes;

            if (!match.variadicConcatTensorEClasses.empty())
            {
                for (EClassId tensorEClass : match.variadicConcatTensorEClasses)
                {
                    inputs.push_back(tensorEClass);
                    const EClass parent = egraph.getEClass(tensorEClass);
                    TensorNode inputNode;
                    inputNode.opType = OpType::INPUT;
                    inputNode.dtype = parent.dtype;
                    inputNode.setShape(parent.shape);
                    inputNode.strides = parent.strides;
                    inputNodes.push_back(std::move(inputNode));
                }
                LogicalId axisVar = pattern.variables.back();
                EClassId axisEClass = binding.at(axisVar);
                inputs.push_back(axisEClass);
                const EClass &axisParent = egraph.getEClass(axisEClass);
                TensorNode axisInputNode;
                axisInputNode.opType = OpType::INPUT;
                axisInputNode.dtype = axisParent.dtype;
                axisInputNode.setShape(axisParent.shape);
                axisInputNode.strides = axisParent.strides;
                inputNodes.push_back(std::move(axisInputNode));
            }
            else
            {
                inputs.reserve(pattern.variables.size());
                inputNodes.reserve(pattern.variables.size());

                for (LogicalId var : pattern.variables)
                {
                    EClassId parentEClassId = binding.at(var);
                    const EClass &parent = egraph.getEClass(parentEClassId);
                    inputs.push_back(parentEClassId);

                    TensorNode inputNode;
                    inputNode.opType = OpType::INPUT;
                    inputNode.dtype = parent.dtype;
                    inputNode.setShape(parent.shape);
                    inputNode.strides = parent.strides;
                    inputNodes.push_back(std::move(inputNode));
                }
            }

            const EClass matchedClass = egraph.getEClass(egraph.getENodeEClass(ENodeId{eNodeIdx}));

            DType outDtype = matchedClass.dtype;
            std::vector<uint32_t> outShape = matchedClass.shape;
            std::vector<uint64_t> outStrides = matchedClass.strides;

            TensorNode outputNode;
            outputNode.opType = OpType::FUSED;
            outputNode.opName = pattern.opName;
            outputNode.dtype = outDtype;
            outputNode.setShape(outShape);
            outputNode.strides = outStrides;

            bool ignoreInputMemSpaces = (pattern.rootOpType != OpType::COPY_TO);

            std::vector<MemSpace> candidateSpaces = {matchedClass.mem_space};
            for (const auto &avail_ms : System::get().getAvailableMemSpaces())
            {
                if (avail_ms.type != HandleType::STORAGE && !(avail_ms == matchedClass.mem_space))
                {
                    candidateSpaces.push_back(avail_ms);
                }
            }

            for (const auto &target_ms : candidateSpaces)
            {
                std::vector<KernelId> kernelMatches = KernelRegistry::get().findMatchingKernelsByPattern(
                    pattern.graph, pattern.rootId, inputNodes, outputNode, false, target_ms, {}, {}, true,
                    ignoreInputMemSpaces, true, true);

                std::vector<std::pair<const KernelEntry *, std::pair<std::vector<Engine>, std::vector<MemSpace>>>>
                    matched_kernels;
                for (KernelId uid : kernelMatches)
                {
                    const KernelEntry &kernel = KernelRegistry::get().getKernel(uid);
                    std::vector<Engine> mapped_engines;
                    std::vector<MemSpace> mapped_input_spaces;
                    if (kernel.matches(inputNodes, outputNode, target_ms, {}, {}, false, true, true, true,
                                       &mapped_engines, &mapped_input_spaces))
                    {
                        matched_kernels.push_back({&kernel, {mapped_engines, mapped_input_spaces}});
                    }
                }

                addFusedCandidates(ctx, matched_kernels, target_ms, inputs, ENodeId{eNodeIdx});
            }
        }
    }

    struct FusedCandidate
    {
        const KernelEntry *kernel;
        MemSpace target_mem_space;
        std::vector<Engine> mapped_engines;
        std::vector<MemSpace> mapped_input_spaces;
        std::vector<std::vector<std::vector<MemSpace>>> child_mem_paths;
        std::vector<bool> child_need_contig;
        std::vector<bool> child_need_copy;
        float cost = std::numeric_limits<float>::infinity();
    };

    void addFusedCandidates(
        RuleCtx &ctx,
        const std::vector<std::pair<const KernelEntry *, std::pair<std::vector<Engine>, std::vector<MemSpace>>>>
            &matched_kernels,
        MemSpace target_mem_space, const std::vector<EClassId> &child_ids, ENodeId eNodeIdx)
    {
        EGraph &egraph = ctx.egraph;
        const ENode oldENode = egraph.getENode(eNodeIdx);
        EClassId e_class_id = egraph.getENodeEClass(eNodeIdx);

        const std::vector<uint32_t> &outShape = oldENode.getShape();
        DType outDtype = oldENode.getDType();

        std::vector<FusedCandidate> candidates;
        candidates.reserve(matched_kernels.size());

        // -------------------------------------------------------------------------
        // Stage 1: Resolve adaptation requirements & compute true execution cost
        // -------------------------------------------------------------------------
        for (const auto &item : matched_kernels)
        {
            const KernelEntry *kernel = item.first;
            const auto &mapped_engines = item.second.first;
            const auto &mapped_input_spaces = item.second.second;

            FusedCandidate cand;
            cand.kernel = kernel;
            cand.target_mem_space = target_mem_space;
            cand.mapped_engines = mapped_engines;
            cand.mapped_input_spaces = mapped_input_spaces;
            cand.child_mem_paths.resize(child_ids.size());
            cand.child_need_contig.resize(child_ids.size(), false);
            cand.child_need_copy.resize(child_ids.size(), false);

            bool valid_paths = true;
            std::vector<std::vector<uint32_t>> actualInShapes;
            std::vector<std::vector<uint64_t>> actualInStrides;
            std::vector<DType> actualInDTypes;
            std::vector<std::vector<uint8_t>> actualInConstants;

            for (uint64_t i = 0; i < child_ids.size(); ++i)
            {
                EClassId pid = child_ids[i];
                const EClass parent = egraph.getEClass(egraph.findConst(pid));

                MemSpace expectedMemSpace =
                    (i < mapped_input_spaces.size()) ? mapped_input_spaces[i] : target_mem_space;
                bool foundMemSpace = (parent.mem_space == expectedMemSpace);
                bool needCopy = !foundMemSpace;
                bool needContig = false;

                uint64_t ruleIdx =
                    std::min(i, static_cast<uint64_t>(
                                    kernel->requiresContiguous.empty() ? 0 : kernel->requiresContiguous.size() - 1));
                if (ruleIdx < kernel->requiresContiguous.size())
                {
                    needContig = (kernel->requiresContiguous[ruleIdx] || needCopy) && !isContiguous(parent);
                }
                else
                {
                    needContig = needCopy && !isContiguous(parent);
                }

                cand.child_need_contig[i] = needContig;
                cand.child_need_copy[i] = needCopy;

                if (needCopy)
                {
                    TensorNode dummyNode;
                    dummyNode.opType = OpType::INPUT;
                    dummyNode.dtype = parent.dtype;
                    dummyNode.setShape(parent.shape);
                    dummyNode.strides = parent.strides;

                    cand.child_mem_paths[i] =
                        findMemSpacePaths(parent.mem_space, expectedMemSpace, dummyNode, mapped_engines);
                    if (cand.child_mem_paths[i].empty())
                    {
                        valid_paths = false;
                        break;
                    }
                }

                // Determine the ACTUAL strides the kernel will receive at execution
                actualInShapes.push_back(parent.shape);
                actualInDTypes.push_back(parent.dtype);
                if (needContig || (needCopy && !isContiguous(parent)))
                {
                    actualInStrides.push_back(calcContiguousStrides(parent.shape));
                }
                else
                {
                    actualInStrides.push_back(parent.strides);
                }

                EClassId canonChild = egraph.findConst(pid);
                if (egraph.constantStaging.count(canonChild))
                    actualInConstants.push_back(*egraph.constantStaging.at(canonChild));
                else
                    actualInConstants.push_back({});
            }

            if (!valid_paths)
                continue;

            std::vector<uint64_t> actualOutStrides =
                kernel->is_view ? oldENode.getStrides() : calcContiguousStrides(outShape);

            if (ctx.costModel)
            {
                cand.cost =
                    ctx.costModel->estimateCost(kernel->uid, outShape, actualOutStrides, outDtype, actualInShapes,
                                                actualInStrides, actualInDTypes, actualInConstants,
                                                /*exactRecordOnly=*/true);
            }

            candidates.push_back(std::move(cand));
        }

        // -------------------------------------------------------------------------
        // Stage 2: Filter equivalent candidates using exact benchmark costs
        // -------------------------------------------------------------------------
        std::vector<bool> keep(candidates.size(), true);
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (candidates[i].cost == std::numeric_limits<float>::infinity())
                continue; // Keep unbenchmarked candidates

            for (size_t j = 0; j < candidates.size(); ++j)
            {
                if (i == j || candidates[j].cost == std::numeric_limits<float>::infinity())
                    continue;

                const auto &c1 = candidates[i];
                const auto &c2 = candidates[j];

                if (c1.mapped_engines != c2.mapped_engines)
                    continue;
                if (c1.mapped_input_spaces != c2.mapped_input_spaces)
                    continue;
                if (c1.kernel->requiresContiguous != c2.kernel->requiresContiguous)
                    continue;
                if (c1.kernel->safe_inplace_idxs != c2.kernel->safe_inplace_idxs)
                    continue;
                if (c1.kernel->is_view != c2.kernel->is_view)
                    continue;

                if (c2.cost < c1.cost - 1e-9f)
                {
                    keep[i] = false;
                    break;
                }
                if (std::abs(c1.cost - c2.cost) <= 1e-9f && c2.kernel->uid < c1.kernel->uid)
                {
                    keep[i] = false;
                    break;
                }
            }
        }

        // -------------------------------------------------------------------------
        // Stage 3: Materialize only the winning candidates into the EGraph
        // -------------------------------------------------------------------------
        for (size_t k = 0; k < candidates.size(); ++k)
        {
            if (!keep[k])
                continue;

            const auto &cand = candidates[k];
            std::vector<EClassId> adapted_children;
            adapted_children.reserve(child_ids.size());

            for (uint64_t i = 0; i < child_ids.size(); ++i)
            {
                EClassId pid = child_ids[i];
                const EClass parent = egraph.getEClass(egraph.findConst(pid));

                bool needCopy = cand.child_need_copy[i];
                bool needContig = cand.child_need_contig[i];

                if (!needCopy && !needContig)
                {
                    adapted_children.push_back(pid);
                    continue;
                }

                EClassId currentPid = pid;
                EClass currentClass = parent;

                if (needContig)
                {
                    currentPid = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentPid}, currentClass.shape,
                                               calcContiguousStrides(currentClass.shape), currentClass.dtype,
                                               currentClass.mem_space);
                    currentClass = egraph.getEClass(egraph.findConst(currentPid));
                }

                if (needCopy)
                {
                    EClassId finalTargetClass = EClassId();
                    for (const auto &path : cand.child_mem_paths[i])
                    {
                        EClassId pathPid = currentPid;
                        EClass pathClass = currentClass;
                        for (uint64_t p_idx = 1; p_idx < path.size(); ++p_idx)
                        {
                            MemSpace next_ms = path[p_idx];
                            pathPid = addOpToEGraph(egraph, OpType::COPY_TO, {pathPid}, pathClass.shape,
                                                    pathClass.strides, pathClass.dtype, next_ms,
                                                    (p_idx == path.size() - 1) ? finalTargetClass : EClassId());
                            if (p_idx == path.size() - 1)
                            {
                                finalTargetClass = pathPid;
                            }
                            pathClass = egraph.getEClass(egraph.findConst(pathPid));
                        }
                    }
                    currentPid = finalTargetClass;
                }

                adapted_children.push_back(currentPid);
            }

            std::vector<uint64_t> strides =
                cand.kernel->is_view ? oldENode.getStrides() : calcContiguousStrides(oldENode.getShape());

            ENode enode(cand.kernel->uid, cand.kernel->opType, cand.kernel->opName, adapted_children,
                        oldENode.getShape(), strides, oldENode.getDType(), cand.target_mem_space, cand.mapped_engines,
                        "", 0, oldENode.getDebugOrigin());

            MemSpace originalMemSpace = egraph.getEClass(egraph.findConst(e_class_id)).mem_space;
            if (cand.target_mem_space == originalMemSpace)
            {
                egraph.addENode(e_class_id, enode);
            }
            else
            {
                EClassId newEClass =
                    egraph.addEClass(enode.getShape(), enode.getStrides(), enode.getDType(), cand.target_mem_space);
                newEClass = egraph.addENode(newEClass, enode);

                EClassId srcForCopy = newEClass;
                const EClass srcCls = egraph.getEClass(egraph.findConst(srcForCopy));

                if (!isContiguous(srcCls))
                {
                    srcForCopy = addOpToEGraph(egraph, OpType::CONTIGUOUS, {srcForCopy}, srcCls.shape,
                                               calcContiguousStrides(srcCls.shape), srcCls.dtype, srcCls.mem_space);
                }

                const EClass copySrcCls = egraph.getEClass(egraph.findConst(srcForCopy));
                addOpToEGraph(egraph, OpType::COPY_TO, {srcForCopy}, copySrcCls.shape, copySrcCls.strides,
                              copySrcCls.dtype, originalMemSpace, e_class_id);
            }
        }
    }

    static bool matchPatternClass(EClassId eClassIdx, const EGraph &egraph, LogicalId patternId, const Pattern &pattern,
                                  std::unordered_map<LogicalId, EClassId> &binding,
                                  const std::unordered_set<EClassId> &protectedEClasses,
                                  bool ignoreConstantData = false)
    {
        eClassIdx = egraph.findConst(eClassIdx);

        auto itVar = std::find(pattern.variables.begin(), pattern.variables.end(), patternId);
        if (itVar != pattern.variables.end())
        {
            uint64_t varIdx = static_cast<uint64_t>(std::distance(pattern.variables.begin(), itVar));
            const EClass &eclass = egraph.getEClass(eClassIdx);

            if (varIdx < pattern.dtypes.size() && eclass.dtype != pattern.dtypes[varIdx])
                return false;

            auto bIt = binding.find(patternId);
            if (bIt != binding.end())
            {
                return bIt->second == eClassIdx;
            }
            binding[patternId] = eClassIdx;
            return true;
        }

        if (patternId != pattern.rootId)
        {
            if (isEClassProtected(eClassIdx, protectedEClasses, egraph))
                return false;
        }

        const EClass &eclass = egraph.getEClass(eClassIdx);
        for (ENodeId enodeId : eclass.enodes)
        {
            std::unordered_map<LogicalId, EClassId> localBinding = binding;
            if (matchPatternNode(enodeId, egraph, patternId, pattern, localBinding, protectedEClasses,
                                 ignoreConstantData))
            {
                binding = std::move(localBinding);
                return true;
            }
        }
        return false;
    }

    static bool matchPatternNode(ENodeId eNodeId, const EGraph &egraph, LogicalId patternId, const Pattern &pattern,
                                 std::unordered_map<LogicalId, EClassId> &binding,
                                 const std::unordered_set<EClassId> &protectedEClasses, bool ignoreConstantData = false)
    {
        const ENode &eNode = egraph.getENode(eNodeId);
        const auto &pNode = pattern.graph.getNode(patternId);

        if (eNode.getOpType() != pNode.opType)
            return false;
        if (eNode.getOpType() == OpType::FUSED && eNode.getOpName() != pNode.opName)
            return false;

        if (eNode.getOpType() == OpType::INPUT && !pNode.contentHash.empty())
        {
            if (!ignoreConstantData)
            {
                EClassId eNodeEClass = egraph.getENodeEClass(eNodeId);
                eNodeEClass = egraph.findConst(eNodeEClass);

                auto egraphIt = egraph.constantStaging.find(eNodeEClass);
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

        if (eNode.getOpType() == OpType::CONCAT && eNode.getChildren().size() != pNode.child_ids.size())
        {
            if (pNode.child_ids.size() < 2)
                return false;
            if (eNode.getChildren().size() < 2)
                return false;

            // Match axis (first element)
            if (!matchPatternClass(eNode.getChildren()[0], egraph, pNode.child_ids[0], pattern, binding,
                                   protectedEClasses, false))
                return false;

            // The rest are tensors. The pattern has one tensor at index 1.
            bool firstTensor = true;
            for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
            {
                if (firstTensor)
                {
                    if (!matchPatternClass(eNode.getChildren()[i], egraph, pNode.child_ids[1], pattern, binding,
                                           protectedEClasses, true))
                        return false;
                    firstTensor = false;
                }
                else
                {
                    EClassId canonChild = egraph.findConst(eNode.getChildren()[i]);
                    const EClass &childCls = egraph.getEClass(canonChild);
                    if (childCls.dtype != pattern.dtypes[1])
                        return false;
                }
            }
            return true;
        }

        if (eNode.getChildren().size() != pNode.child_ids.size())
            return false;

        for (uint64_t i = 0; i < eNode.getChildren().size(); ++i)
        {
            bool childIgnoreConst = isConstant(eNode.getOpType(), i, eNode.getChildren().size());
            if (!matchPatternClass(eNode.getChildren()[i], egraph, pNode.child_ids[i], pattern, binding,
                                   protectedEClasses, childIgnoreConst))
            {
                return false;
            }
        }
        return true;
    }
};

struct InfinityDomination : public Rule
{
    std::unordered_set<ENodeId> visited_enodes;

    std::string name() const override
    {
        return "InfinityDomination";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited_enodes.count(ENodeId{eNodeIdx}))
            return false;

        const ENode &enode = ctx.egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::ADD || enode.getChildren().size() != 2)
            return false;

        return isConstantFloat(enode.getChildren()[0], ctx) || isConstantFloat(enode.getChildren()[1], ctx);
    }

    bool isConstantFloat(EClassId e_class_id, RuleCtx &ctx) const
    {
        e_class_id = ctx.egraph.findConst(e_class_id);
        const EClass &cls = ctx.egraph.getEClass(e_class_id);
        if (cls.dtype != DType::FLOAT32)
            return false;
        if (ctx.egraph.constantStaging.find(e_class_id) != ctx.egraph.constantStaging.end())
            return true;

        if (ctx.repo && ctx.repo->isValid())
        {
            LogicalId logical_id;
            if (ctx.eclassToLogical.count(e_class_id))
                logical_id = ctx.eclassToLogical.at(e_class_id);

            if (logical_id != LogicalId() && ctx.repo->has(logical_id))
            {
                auto data = ctx.repo->read(logical_id);
                ctx.egraph.constantStaging[e_class_id] = std::make_shared<std::vector<uint8_t>>(std::move(data));
                return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        visited_enodes.insert(ENodeId{eNodeIdx});

        // COPY instead of reference to prevent UAF
        const ENode addNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        uint32_t constIdx = isConstantFloat(addNode.getChildren()[1], ctx) ? 1 : 0;
        uint32_t varIdx = 1 - constIdx;

        EClassId constClass = egraph.findConst(addNode.getChildren()[constIdx]);
        EClassId varClass = egraph.findConst(addNode.getChildren()[varIdx]);

        const auto &constData = *egraph.constantStaging.at(constClass);
        const float *data = reinterpret_cast<const float *>(constData.data());

        const EClass cClass = egraph.getEClass(constClass);
        uint64_t numElements = countElements(cClass.shape);

        std::vector<Region> nonInfRegions;
        bool noneInf = true;
        for (uint64_t i = 0; i < numElements; ++i)
        {
            uint64_t flat_idx = getStridedIndex(i, cClass.shape, cClass.strides);
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
        const EClass outClass = egraph.getEClass(e_class_id);

        std::vector<uint64_t> contigStrides = calcContiguousStrides(outClass.shape);

        if (nonInfRegions.empty())
        {
            EClassId currentTarget = constClass;
            if (cClass.strides != contigStrides)
            {
                currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentTarget}, outClass.shape,
                                              contigStrides, outClass.dtype, outClass.mem_space);
            }
            if (cClass.mem_space != outClass.mem_space)
            {
                currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {currentTarget}, outClass.shape, contigStrides,
                                              outClass.dtype, outClass.mem_space);
            }
            egraph.merge(e_class_id, currentTarget);
            return;
        }

        if (nonInfRegions.size() == 1)
        {
            bool strictlySmaller = false;
            const Region &reg = nonInfRegions[0];
            for (uint64_t d = 0; d < cClass.shape.size(); ++d)
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

        EClassId currentTarget = constClass;
        if (cClass.strides != contigStrides)
        {
            currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentTarget}, outClass.shape, contigStrides,
                                          outClass.dtype, outClass.mem_space);
        }
        if (cClass.mem_space != outClass.mem_space)
        {
            currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {currentTarget}, outClass.shape, contigStrides,
                                          outClass.dtype, outClass.mem_space);
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

            EClassId startsId = egraph.addIntConst(starts);
            EClassId endsId = egraph.addIntConst(ends);
            EClassId stepsId = egraph.addIntConst(steps);

            std::vector<uint32_t> sliceShape;
            for (uint64_t d = 0; d < starts.size(); ++d)
            {
                sliceShape.push_back(ends[d] - starts[d]);
            }

            std::vector<uint64_t> sliceStridesV = vClass.strides;
            for (uint64_t d = 0; d < starts.size(); ++d)
            {
                sliceStridesV[d] *= steps[d];
            }
            EClassId sliceV = addOpToEGraph(egraph, OpType::SLICE, {varClass, startsId, endsId, stepsId}, sliceShape,
                                            sliceStridesV, vClass.dtype, vClass.mem_space);

            std::vector<uint64_t> sliceStridesC = cClass.strides;
            for (uint64_t d = 0; d < starts.size(); ++d)
            {
                sliceStridesC[d] *= steps[d];
            }
            EClassId sliceC = addOpToEGraph(egraph, OpType::SLICE, {currentTarget, startsId, endsId, stepsId},
                                            sliceShape, sliceStridesC, cClass.dtype, cClass.mem_space);

            std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);
            EClassId contigV = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceV}, sliceShape, sliceContigStrides,
                                             vClass.dtype, vClass.mem_space);
            EClassId contigC = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceC}, sliceShape, sliceContigStrides,
                                             cClass.dtype, cClass.mem_space);

            contigV = copyTo(egraph, contigV, outClass.mem_space);
            contigC = copyTo(egraph, contigC, outClass.mem_space);

            EClassId child0 = (constIdx == 0) ? contigC : contigV;
            EClassId child1 = (constIdx == 1) ? contigC : contigV;
            EClassId addId = addOpToEGraph(egraph, OpType::ADD, {child0, child1}, sliceShape, sliceContigStrides,
                                           outClass.dtype, outClass.mem_space);

            currentTarget = addOpToEGraph(egraph, OpType::SCATTER, {currentTarget, addId, startsId, endsId, stepsId},
                                          outClass.shape, outClass.strides, outClass.dtype, outClass.mem_space);
        }

        egraph.merge(e_class_id, currentTarget);
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
        std::uint64_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.contigIdx) ^ (std::hash<uint32_t>{}(k.sliceIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.srcNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;
    SlicePushDownElementwise(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected)
    {
    }

    std::string name() const override
    {
        return "SlicePushDownElementwise";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;

        EClassId childClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId childNodeIdx : egraph.getEClass(childClass).enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() == OpType::SLICE && childNode.getChildren().size() == 4)
            {
                EClassId srcClass = egraph.findConst(childNode.getChildren()[0]);

                if (egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[1])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[2])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[3])) == 0)
                    continue;

                auto starts = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[1]));
                auto ends = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[2]));
                auto steps = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[3]));

                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                const auto &origShape = egraph.getEClass(srcClass).shape;
                bool isFull = true;
                if (starts.size() != origShape.size())
                    isFull = false;
                for (uint64_t d = 0; d < starts.size() && isFull; ++d)
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

                for (ENodeId srcNodeIdx : egraph.getEClass(srcClass).enodes)
                {
                    const ENode opNode = egraph.getENode(srcNodeIdx);
                    OpType op = opNode.getOpType();
                    if (!isElementwise(op) || op == OpType::COPY_TO || op == OpType::CONTIGUOUS)
                    {
                        continue;
                    }

                    bool hasBroadcastChild = false;
                    for (EClassId cid : opNode.getChildren())
                    {
                        const auto &cls = egraph.getEClass(egraph.findConst(cid));
                        if (cls.shape != opNode.getShape())
                        {
                            hasBroadcastChild = true;
                            break;
                        }
                    }
                    if (!hasBroadcastChild)
                    {
                        MatchKey key{eNodeIdx, childNodeIdx.value, srcNodeIdx.value};
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
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);

        std::vector<ENodeId> sliceNodes;
        for (ENodeId childNodeIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() == OpType::SLICE && childNode.getChildren().size() == 4)
            {
                sliceNodes.push_back(childNodeIdx);
            }
        }

        for (ENodeId sliceNodeIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENode(sliceNodeIdx);

            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            EClassId startsId = sliceNode.getChildren()[1];
            EClassId endsId = sliceNode.getChildren()[2];
            EClassId stepsId = sliceNode.getChildren()[3];

            if (egraph.constantStaging.count(egraph.findConst(startsId)) == 0 ||
                egraph.constantStaging.count(egraph.findConst(endsId)) == 0 ||
                egraph.constantStaging.count(egraph.findConst(stepsId)) == 0)
                continue;

            auto starts = egraph.getConstantInt32(egraph.findConst(startsId));
            auto ends = egraph.getConstantInt32(egraph.findConst(endsId));
            auto steps = egraph.getConstantInt32(egraph.findConst(stepsId));
            if (starts.empty() || ends.empty() || steps.empty())
                continue;

            const std::vector<uint32_t> sliceShape = sliceNode.getShape();
            std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

            std::vector<ENodeId> srcEnodes = egraph.getEClass(srcClass).enodes;

            for (ENodeId srcNodeIdx : srcEnodes)
            {
                const ENode &opNode = egraph.getENode(srcNodeIdx);
                OpType op = opNode.getOpType();
                if (!isElementwise(op) || op == OpType::COPY_TO || op == OpType::CONTIGUOUS)
                {
                    continue;
                }

                bool hasBroadcastChild = false;
                for (EClassId cid : opNode.getChildren())
                {
                    const auto &cls = egraph.getEClass(egraph.findConst(cid));
                    if (cls.shape != opNode.getShape())
                    {
                        hasBroadcastChild = true;
                        break;
                    }
                }
                if (hasBroadcastChild)
                    continue;

                MatchKey key{eNodeIdx, sliceNodeIdx.value, srcNodeIdx.value};
                if (!visited.insert(key).second)
                    continue;

                std::vector<EClassId> newChildren;
                for (uint64_t childIdx = 0; childIdx < opNode.getChildren().size(); ++childIdx)
                {
                    EClassId childId = opNode.getChildren()[childIdx];
                    EClassId canonChildId = egraph.findConst(childId);
                    const EClass childCls = egraph.getEClass(canonChildId);
                    std::vector<uint64_t> childSliceStrides = childCls.strides;

                    for (uint64_t d = 0; d < starts.size() && d < childCls.shape.size(); ++d)
                    {
                        int32_t start = starts[d];
                        if (start < 0)
                            start += childCls.shape[d];
                        childSliceStrides[d] *= steps[d];
                    }

                    EClassId childSlice =
                        addOpToEGraph(egraph, OpType::SLICE, {canonChildId, startsId, endsId, stepsId}, sliceShape,
                                      childSliceStrides, childCls.dtype, childCls.mem_space);

                    // Determine if the op kernel requires a contiguous tensor for this input
                    bool reqContig = true;
                    if (opNode.getKernelId().value != 0 && KernelRegistry::get().hasKernel(opNode.getKernelId()))
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(opNode.getKernelId());
                        if (!kernel.requiresContiguous.empty())
                        {
                            uint64_t ruleIdx =
                                std::min(childIdx, static_cast<uint64_t>(kernel.requiresContiguous.size() - 1));
                            reqContig = kernel.requiresContiguous[ruleIdx];
                        }
                    }

                    if (reqContig)
                    {
                        childSlice = addOpToEGraph(egraph, OpType::CONTIGUOUS, {childSlice}, sliceShape,
                                                   sliceContigStrides, childCls.dtype, childCls.mem_space);
                    }
                    newChildren.push_back(childSlice);
                }

                EClassId opEClass = addOpToEGraph(egraph, op, newChildren, sliceShape, sliceContigStrides,
                                                  sliceNode.getDType(), sliceNode.getMemSpace());

                EClassId op_cache = createCacheInputNode(egraph, srcClass, ctx.eclassToLogical);

                const EClass srcEClass = egraph.getEClass(srcClass);
                EClassId scatterClass =
                    addOpToEGraph(egraph, OpType::SCATTER, {op_cache, opEClass, startsId, endsId, stepsId},
                                  srcEClass.shape, srcEClass.strides, opNode.getDType(), opNode.getMemSpace());

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
        std::uint64_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^ (std::hash<uint32_t>{}(k.childNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.srcNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownDot(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected)
    {
    }

    std::string name() const override
    {
        return "SlicePushDownDot";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;

        EClassId childClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId childNodeIdx : egraph.getEClass(childClass).enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() == OpType::SLICE && childNode.getChildren().size() == 4)
            {
                EClassId srcClass = egraph.findConst(childNode.getChildren()[0]);

                if (egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[1])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[2])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[3])) == 0)
                    continue;

                auto starts = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[1]));
                auto ends = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[2]));
                auto steps = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[3]));

                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                const auto &origShape = egraph.getEClass(srcClass).shape;
                bool isFull = true;
                if (starts.size() != origShape.size())
                    isFull = false;
                for (uint64_t d = 0; d < starts.size() && isFull; ++d)
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

                for (ENodeId srcNodeIdx : egraph.getEClass(srcClass).enodes)
                {
                    const ENode &opNode = egraph.getENode(srcNodeIdx);
                    if (opNode.getOpType() == OpType::DOT)
                    {
                        MatchKey key{eNodeIdx, childNodeIdx.value, srcNodeIdx.value};
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
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);

        std::vector<ENodeId> sliceNodes;
        for (ENodeId childNodeIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENode(childNodeIdx).getOpType() == OpType::SLICE &&
                egraph.getENode(childNodeIdx).getChildren().size() == 4)
            {
                sliceNodes.push_back(childNodeIdx);
            }
        }

        for (ENodeId sliceNodeIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENode(sliceNodeIdx);

            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            std::vector<ENodeId> srcEnodes = egraph.getEClass(srcClass).enodes;

            for (ENodeId srcNodeIdx : srcEnodes)
            {
                const ENode dotNode = egraph.getENode(srcNodeIdx);
                if (dotNode.getOpType() != OpType::DOT)
                    continue;

                MatchKey key{eNodeIdx, sliceNodeIdx.value, srcNodeIdx.value};
                if (!visited.insert(key).second)
                    continue;

                EClassId startsId = sliceNode.getChildren()[1];
                EClassId endsId = sliceNode.getChildren()[2];
                EClassId stepsId = sliceNode.getChildren()[3];

                if (egraph.constantStaging.count(egraph.findConst(startsId)) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(endsId)) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(stepsId)) == 0)
                    continue;

                auto starts = egraph.getConstantInt32(egraph.findConst(startsId));
                auto ends = egraph.getConstantInt32(egraph.findConst(endsId));
                auto steps = egraph.getConstantInt32(egraph.findConst(stepsId));

                if (starts.empty() || ends.empty() || steps.empty())
                    Error::throw_err("[SlicePushDownDot.apply] can't find constants for "
                                     "all slice args");

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

                for (uint64_t d = 0; d < rank; ++d)
                {
                    if (starts[d] < 0)
                        starts[d] += outClassShape[d];
                    if (ends[d] < 0)
                        ends[d] += outClassShape[d];
                    starts[d] = std::max(0, starts[d]);
                    ends[d] = std::min((int32_t)outClassShape[d], std::max(starts[d], ends[d]));
                }

                const std::vector<uint32_t> sliceShape = sliceNode.getShape();
                std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

                EClassId aClassId = dotNode.getChildren()[0];
                EClassId bClassId = dotNode.getChildren()[1];

                uint32_t K = (rank == 2)   ? egraph.getEClass(egraph.findConst(aClassId)).shape[1]
                             : (rank == 3) ? egraph.getEClass(egraph.findConst(aClassId)).shape[2]
                                           : egraph.getEClass(egraph.findConst(aClassId)).shape[3];

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

                EClassId startsIdA = egraph.addIntConst(startsA);
                EClassId endsIdA = egraph.addIntConst(endsA);
                EClassId stepsIdA = egraph.addIntConst(stepsA);

                EClassId startsIdB = egraph.addIntConst(startsB);
                EClassId endsIdB = egraph.addIntConst(endsB);
                EClassId stepsIdB = egraph.addIntConst(stepsB);

                auto createSlice = [&](EClassId classId, const std::vector<int32_t> &st, const std::vector<int32_t> &en,
                                       EClassId stId, EClassId enId, EClassId stepId) {
                    EClassId canonId = egraph.findConst(classId);
                    const EClass cls = egraph.getEClass(canonId);
                    std::vector<uint64_t> sStrides = cls.strides;
                    DType cDtype = cls.dtype;

                    std::vector<uint32_t> sShape;
                    for (uint64_t d = 0; d < st.size(); ++d)
                        sShape.push_back(en[d] - st[d]);

                    for (uint64_t d = 0; d < st.size(); ++d)
                    {
                        sStrides[d] *= steps[d];
                    }

                    EClassId sClass = addOpToEGraph(egraph, OpType::SLICE, {canonId, stId, enId, stepId}, sShape,
                                                    sStrides, cDtype, sliceNode.getMemSpace());
                    EClassId sContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sClass}, sShape,
                                                     calcContiguousStrides(sShape), cDtype, sliceNode.getMemSpace());
                    return sContig;
                };

                EClassId aSliced = createSlice(aClassId, startsA, endsA, startsIdA, endsIdA, stepsIdA);
                EClassId bSliced = createSlice(bClassId, startsB, endsB, startsIdB, endsIdB, stepsIdB);

                EClassId dotEClass = addOpToEGraph(egraph, OpType::DOT, {aSliced, bSliced}, sliceShape,
                                                   sliceContigStrides, sliceNode.getDType(), sliceNode.getMemSpace());

                EClassId op_cache = createCacheInputNode(egraph, srcClass, ctx.eclassToLogical);

                const EClass srcEClass = egraph.getEClass(srcClass);
                EClassId scatterClass =
                    addOpToEGraph(egraph, OpType::SCATTER, {op_cache, dotEClass, startsId, endsId, stepsId},
                                  srcEClass.shape, srcEClass.strides, dotNode.getDType(), dotNode.getMemSpace());

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

// Splits up dot into 2 smaller dots. Can reduce peak memory usage, and combined with FusionRule naturally discovers
// tensor parallism across engines.
struct DotSplitRule : public Rule
{
    uint32_t splitThreshold = 32768; // Only split dimensions >= splitThreshold
    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "DotSplitRule";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::DOT)
            return false;

        EClassId aClass = egraph.findConst(enode.getChildren()[0]);
        EClassId bClass = egraph.findConst(enode.getChildren()[1]);
        const EClass &aCls = egraph.getEClass(aClass);
        const EClass &bCls = egraph.getEClass(bClass);

        // Rank 3: A [B, M, K], B [B, K, N]
        if (aCls.shape.size() == 3 && bCls.shape.size() == 3)
        {
            uint32_t K = aCls.shape[2];
            uint32_t N = bCls.shape[2];
            // Match if Column dimension N or Reduction dimension K is large
            if ((N >= splitThreshold && N % 2 == 0) || (K >= splitThreshold && K % 2 == 0))
                return true;
        }

        if (aCls.mem_space != enode.getMemSpace() || bCls.mem_space != enode.getMemSpace())
            return false;

        return false;
    }

    // -----------------------------------------------------------------
    // Strategy A: Column-Parallel Split (Split N -> N1, N2)
    // A [B, M, K] * B1 [B, K, N1] -> C1 [B, M, N1]
    // A [B, M, K] * B2 [B, K, N2] -> C2 [B, M, N2]
    // C = CONCAT([C1, C2], axis=2)
    // -----------------------------------------------------------------
    void applyStrategyA(EGraph &egraph, EClassId e_class_id, const ENode dotNode, EClassId aClass, EClassId bClass,
                        const EClass aCls, const EClass bCls)
    {
        uint32_t B = aCls.shape[0];
        uint32_t M = aCls.shape[1];
        uint32_t K = aCls.shape[2];
        uint32_t N = bCls.shape[2];

        uint32_t N1 = N / 2;
        uint32_t N2 = N - N1;

        // B1 Slice: [0..B, 0..K, 0..N1]
        EClassId st1 = egraph.addIntConst({0, 0, 0});
        EClassId en1 = egraph.addIntConst({(int32_t)B, (int32_t)K, (int32_t)N1});
        EClassId step = egraph.addIntConst({1, 1, 1});

        EClassId b1 = addOpToEGraph(egraph, OpType::SLICE, {bClass, st1, en1, step}, {B, K, N1},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);

        // B2 Slice: [0..B, 0..K, N1..N]
        EClassId st2 = egraph.addIntConst({0, 0, (int32_t)N1});
        EClassId en2 = egraph.addIntConst({(int32_t)B, (int32_t)K, (int32_t)N});

        EClassId b2 = addOpToEGraph(egraph, OpType::SLICE, {bClass, st2, en2, step}, {B, K, N2},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);

        EClassId b1_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b1}, {B, K, N1},
                                           calcContiguousStrides({B, K, N1}), bCls.dtype, bCls.mem_space);
        EClassId b2_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b2}, {B, K, N2},
                                           calcContiguousStrides({B, K, N2}), bCls.dtype, bCls.mem_space);

        // C1 = A * B1, C2 = A * B2
        EClassId c1 = addOpToEGraph(egraph, OpType::DOT, {aClass, b1_contig}, {B, M, N1},
                                    calcContiguousStrides({B, M, N1}), dotNode.getDType(), dotNode.getMemSpace());
        EClassId c2 = addOpToEGraph(egraph, OpType::DOT, {aClass, b2_contig}, {B, M, N2},
                                    calcContiguousStrides({B, M, N2}), dotNode.getDType(), dotNode.getMemSpace());

        // C = CONCAT([C1, C2], axis=2)
        EClassId axis2 = egraph.addIntConst({2});
        EClassId c_concat = addOpToEGraph(egraph, OpType::CONCAT, {axis2, c1, c2}, {B, M, N},
                                          calcContiguousStrides({B, M, N}), dotNode.getDType(), dotNode.getMemSpace());

        egraph.merge(e_class_id, c_concat);
    }

    // -----------------------------------------------------------------
    // Strategy B: Row-Parallel Split (Split K -> K1, K2)
    // A1 [B, M, K1] * B1 [B, K1, N] -> C1 [B, M, N]
    // A2 [B, M, K2] * B2 [B, K2, N] -> C2 [B, M, N]
    // C = ADD(C1, C2)
    // -----------------------------------------------------------------
    void applyStrategyB(EGraph &egraph, EClassId e_class_id, const ENode dotNode, EClassId aClass, EClassId bClass,
                        const EClass aCls, const EClass bCls)
    {
        uint32_t B = aCls.shape[0];
        uint32_t M = aCls.shape[1];
        uint32_t K = aCls.shape[2];
        uint32_t N = bCls.shape[2];

        uint32_t K1 = K / 2;
        uint32_t K2 = K - K1;

        EClassId step = egraph.addIntConst({1, 1, 1});

        // A1 Slice [0..B, 0..M, 0..K1], A2 Slice [0..B, 0..M, K1..K]
        EClassId a_st1 = egraph.addIntConst({0, 0, 0});
        EClassId a_en1 = egraph.addIntConst({(int32_t)B, (int32_t)M, (int32_t)K1});
        EClassId a_st2 = egraph.addIntConst({0, 0, (int32_t)K1});
        EClassId a_en2 = egraph.addIntConst({(int32_t)B, (int32_t)M, (int32_t)K});

        EClassId a1 = addOpToEGraph(egraph, OpType::SLICE, {aClass, a_st1, a_en1, step}, {B, M, K1},
                                    {aCls.strides[0], aCls.strides[1], aCls.strides[2]}, aCls.dtype, aCls.mem_space);
        EClassId a2 = addOpToEGraph(egraph, OpType::SLICE, {aClass, a_st2, a_en2, step}, {B, M, K2},
                                    {aCls.strides[0], aCls.strides[1], aCls.strides[2]}, aCls.dtype, aCls.mem_space);

        // B1 Slice [0..B, 0..K1, 0..N], B2 Slice [0..B, K1..K, 0..N]
        EClassId b_st1 = egraph.addIntConst({0, 0, 0});
        EClassId b_en1 = egraph.addIntConst({(int32_t)B, (int32_t)K1, (int32_t)N});
        EClassId b_st2 = egraph.addIntConst({0, (int32_t)K1, 0});
        EClassId b_en2 = egraph.addIntConst({(int32_t)B, (int32_t)K, (int32_t)N});

        EClassId b1 = addOpToEGraph(egraph, OpType::SLICE, {bClass, b_st1, b_en1, step}, {B, K1, N},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);
        EClassId b2 = addOpToEGraph(egraph, OpType::SLICE, {bClass, b_st2, b_en2, step}, {B, K2, N},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);

        EClassId a1_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {a1}, {B, M, K1},
                                           calcContiguousStrides({B, M, K1}), aCls.dtype, aCls.mem_space);
        EClassId a2_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {a2}, {B, M, K2},
                                           calcContiguousStrides({B, M, K2}), aCls.dtype, aCls.mem_space);
        EClassId b1_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b1}, {B, K1, N},
                                           calcContiguousStrides({B, K1, N}), bCls.dtype, bCls.mem_space);
        EClassId b2_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b2}, {B, K2, N},
                                           calcContiguousStrides({B, K2, N}), bCls.dtype, bCls.mem_space);

        // C1 = A1 * B1, C2 = A2 * B2
        EClassId c1 = addOpToEGraph(egraph, OpType::DOT, {a1_contig, b1_contig}, {B, M, N},
                                    calcContiguousStrides({B, M, N}), dotNode.getDType(), dotNode.getMemSpace());
        EClassId c2 = addOpToEGraph(egraph, OpType::DOT, {a2_contig, b2_contig}, {B, M, N},
                                    calcContiguousStrides({B, M, N}), dotNode.getDType(), dotNode.getMemSpace());

        // C = ADD(C1, C2)
        EClassId c_add = addOpToEGraph(egraph, OpType::ADD, {c1, c2}, {B, M, N}, calcContiguousStrides({B, M, N}),
                                       dotNode.getDType(), dotNode.getMemSpace());

        egraph.merge(e_class_id, c_add);
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        visited.insert(eNodeIdx);

        const ENode dotNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId aClass = egraph.findConst(dotNode.getChildren()[0]);
        EClassId bClass = egraph.findConst(dotNode.getChildren()[1]);
        const EClass aCls = egraph.getEClass(aClass);
        const EClass bCls = egraph.getEClass(bClass);

        if (aCls.shape.size() != 3 || bCls.shape.size() != 3)
            return;

        if (!isContiguous(aCls))
        {
            aClass = addOpToEGraph(egraph, OpType::CONTIGUOUS, {aClass}, aCls.shape, calcContiguousStrides(aCls.shape),
                                   aCls.dtype, aCls.mem_space);
        }
        if (!isContiguous(bCls))
        {
            bClass = addOpToEGraph(egraph, OpType::CONTIGUOUS, {bClass}, bCls.shape, calcContiguousStrides(bCls.shape),
                                   bCls.dtype, bCls.mem_space);
        }

        uint32_t K = aCls.shape[2];
        uint32_t N = bCls.shape[2];

        if (N >= splitThreshold * 2)
        {
            applyStrategyA(egraph, e_class_id, dotNode, aClass, bClass, aCls, bCls);
        }

        if (K >= splitThreshold * 2)
        {
            applyStrategyB(egraph, e_class_id, dotNode, aClass, bClass, aCls, bCls);
        }
    }
};

// Remove unneeded contiguous ops. op(contiguous(x)) -> op(x)
struct RemoveContiguous : public Rule
{
    std::string name() const override
    {
        return "RemoveContiguous";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        if (enode.getOpType() == OpType::CONTIGUOUS || enode.getOpType() == OpType::INPUT ||
            enode.getOpType() == OpType::CACHE)
            return false;

        // Check if the kernel explicitly requires contiguous inputs for any child
        if (!KernelRegistry::get().hasKernel(enode.getKernelId()))
            return false;
        const KernelEntry kernel = KernelRegistry::get().getKernel(enode.getKernelId());

        if (kernel.is_view)
            return false; // view kernels output strides depend on input strides so will break things when merge eclass

        const auto &children = enode.getChildren();
        for (uint64_t i = 0; i < children.size(); ++i)
        {
            // If the kernel requires contiguous input at this index, DO NOT remove CONTIGUOUS
            if (!kernel.requiresContiguous.empty())
            {
                uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(kernel.requiresContiguous.size() - 1));
                if (kernel.requiresContiguous[ruleIdx])
                    continue; // Skip unwrapping
            }

            EClassId childClsId = egraph.findConst(children[i]);
            const EClass &childCls = egraph.getEClass(childClsId);
            for (ENodeId cEnodeId : childCls.enodes)
            {
                const ENode &cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::CONTIGUOUS && !cEnode.getChildren().empty())
                {
                    EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                    if (unwrappedChildId != childClsId)
                    {
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

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});
        const auto children = enode.getChildren();
        const KernelEntry &kernel = KernelRegistry::get().getKernel(enode.getKernelId());

        std::vector<std::vector<EClassId>> candidateChildrenPerPos(children.size());

        for (uint64_t i = 0; i < children.size(); ++i)
        {
            EClassId childClsId = egraph.findConst(children[i]);
            candidateChildrenPerPos[i].push_back(childClsId);

            // If the kernel requires contiguous input at this index, DO NOT remove CONTIGUOUS
            if (!kernel.requiresContiguous.empty())
            {
                uint64_t ruleIdx = std::min(i, static_cast<uint64_t>(kernel.requiresContiguous.size() - 1));
                if (kernel.requiresContiguous[ruleIdx])
                    continue; // Skip unwrapping
            }

            const EClass &childCls = egraph.getEClass(childClsId);
            for (ENodeId cEnodeId : childCls.enodes)
            {
                const ENode &cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::CONTIGUOUS && !cEnode.getChildren().empty())
                {
                    EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                    if (unwrappedChildId != childClsId)
                    {
                        candidateChildrenPerPos[i].push_back(unwrappedChildId);
                    }
                }
            }
        }

        std::vector<std::vector<EClassId>> childCombinations;
        std::vector<EClassId> currentCombination(children.size());

        std::function<void(uint64_t, bool)> generateCombos = [&](uint64_t pos, bool hasUnwrapped) {
            if (pos == children.size())
            {
                if (hasUnwrapped)
                {
                    childCombinations.push_back(currentCombination);
                }
                return;
            }
            for (uint64_t cIdx = 0; cIdx < candidateChildrenPerPos[pos].size(); ++cIdx)
            {
                currentCombination[pos] = candidateChildrenPerPos[pos][cIdx];
                generateCombos(pos + 1, hasUnwrapped || (cIdx > 0));
            }
        };

        generateCombos(0, false);

        const EClass outCls = egraph.getEClass(egraph.findConst(e_class_id));
        TensorNode outNode;
        outNode.opType = enode.getOpType();
        outNode.opName = enode.getOpName();
        outNode.dtype = outCls.dtype;
        outNode.setShape(outCls.shape);
        outNode.strides = outCls.strides;

        for (const auto &newChildren : childCombinations)
        {
            std::vector<TensorNode> inNodes;
            std::vector<MemSpace> inMemSpaces;
            for (EClassId cid : newChildren)
            {
                const EClass &cCls = egraph.getEClass(egraph.findConst(cid));
                TensorNode inNode;
                inNode.opType = OpType::INPUT;
                inNode.dtype = cCls.dtype;
                inNode.setShape(cCls.shape);
                inNode.strides = cCls.strides;
                inNodes.push_back(inNode);
                inMemSpaces.push_back(cCls.mem_space);
            }

            if (kernel.matches(inNodes, outNode, outCls.mem_space, inMemSpaces, enode.getEngines()))
            {
                ENode newENode(kernel.uid, enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                               enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(), "", 0,
                               enode.getDebugOrigin());
                egraph.addENode(e_class_id, newENode);
            }
        }
    }
};

// Remove chains of COPY_TO operations where the start and end MemSpaces match.
// Matches the consumer node and rewrites its children to point directly to the
// earlier EClass in the same MemSpace, bypassing intermediate COPY_TO nodes.
struct RemoveCopyChains : public Rule
{
    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "RemoveCopyChains";
    }

    static EClassId findCopyChainOrigin(EClassId startClassId, const EGraph &egraph)
    {
        EClassId startCanon = egraph.findConst(startClassId);
        MemSpace targetMemSpace = egraph.getEClass(startCanon).mem_space;

        EClassId currClass = startCanon;
        EClassId bestOrigin = startCanon;
        std::unordered_set<EClassId> visitedClasses = {currClass};

        while (true)
        {
            const EClass &cls = egraph.getEClass(currClass);
            EClassId nextClass = EClassId{UINT32_MAX};

            for (ENodeId enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENode(enodeId);
                if (enode.getOpType() == OpType::COPY_TO && !enode.getChildren().empty())
                {
                    nextClass = egraph.findConst(enode.getChildren()[0]);
                    break;
                }
            }

            if (nextClass == EClassId{UINT32_MAX} || visitedClasses.count(nextClass))
            {
                break;
            }

            visitedClasses.insert(nextClass);
            if (egraph.getEClass(nextClass).mem_space == targetMemSpace)
            {
                bestOrigin = nextClass;
            }
            currClass = nextClass;
        }

        return bestOrigin;
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            return false;

        const auto &children = enode.getChildren();
        for (uint64_t i = 0; i < children.size(); ++i)
        {
            EClassId childCanon = egraph.findConst(children[i]);
            EClassId origin = findCopyChainOrigin(childCanon, egraph);
            if (origin != childCanon)
            {
                return true;
            }
        }

        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        visited.insert(eNodeIdx);

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId eClassId = egraph.getENodeEClass(ENodeId{eNodeIdx});

        std::vector<EClassId> newChildren;
        newChildren.reserve(enode.getChildren().size());
        bool changed = false;

        for (uint64_t i = 0; i < enode.getChildren().size(); ++i)
        {
            EClassId childCanon = egraph.findConst(enode.getChildren()[i]);
            EClassId origin = findCopyChainOrigin(childCanon, egraph);
            if (origin != childCanon)
            {
                newChildren.push_back(origin);
                changed = true;
            }
            else
            {
                newChildren.push_back(childCanon);
            }
        }

        if (changed)
        {
            ENode newENode(enode.getKernelId(), enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                           enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                           enode.getContentHash(), 0, enode.getDebugOrigin());
            egraph.addENode(eClassId, newENode);
        }
    }
};

struct ConsumerWeightReuseRule : public Rule
{
    std::unordered_set<uint32_t> visited_enodes;

    std::string name() const override
    {
        return "ConsumerWeightReuseRule";
    }

    // Helper: Checks if an E-class represents a loaded storage weight and returns its tensor name
    bool getStorageWeightName(const EGraph &egraph, EClassId classId, std::string &out_name, EClassId &out_canon) const
    {
        out_canon = egraph.findConst(classId);
        const EClass &cls = egraph.getEClass(out_canon);
        for (ENodeId eid : cls.enodes)
        {
            const ENode &enode = egraph.getENode(eid);
            if (enode.getOpType() == OpType::COPY_TO && !enode.getChildren().empty())
            {
                EClassId storageClsId = egraph.findConst(enode.getChildren()[0]);
                const EClass &storageCls = egraph.getEClass(storageClsId);
                if (storageCls.mem_space.type == HandleType::STORAGE)
                {
                    for (ENodeId sEid : storageCls.enodes)
                    {
                        const ENode &sNode = egraph.getENode(sEid);
                        if (sNode.getOpType() == OpType::INPUT && !sNode.getOpName().empty())
                        {
                            out_name = sNode.getOpName();
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited_enodes.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE ||
            enode.getOpType() == OpType::COPY_TO)
            return false;

        // Check if any child is a loaded storage weight that has an earlier duplicate in the EGraph
        for (EClassId child : enode.getChildren())
        {
            std::string tensor_name;
            EClassId childCanon;
            if (getStorageWeightName(egraph, child, tensor_name, childCanon))
            {
                // Check if an earlier E-Class exists with the same tensor name
                for (uint32_t i = 0; i < childCanon.value; ++i)
                {
                    EClassId otherCanon = egraph.findConst(EClassId{i});
                    if (otherCanon == childCanon || otherCanon.value >= childCanon.value)
                        continue;

                    std::string other_name;
                    EClassId dummy;
                    if (getStorageWeightName(egraph, otherCanon, other_name, dummy) && other_name == tensor_name)
                    {
                        const EClass &c1 = egraph.getEClass(childCanon);
                        const EClass &c2 = egraph.getEClass(otherCanon);
                        if (c1.mem_space == c2.mem_space && c1.dtype == c2.dtype && c1.shape == c2.shape)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        visited_enodes.insert(eNodeIdx);

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId eclass_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        std::vector<EClassId> newChildren = enode.getChildren();
        bool changed = false;

        for (size_t i = 0; i < newChildren.size(); ++i)
        {
            std::string tensor_name;
            EClassId childCanon;
            if (getStorageWeightName(egraph, newChildren[i], tensor_name, childCanon))
            {
                // Find the earliest matching E-Class to avoid cycles and preserve a canonical root
                for (uint32_t c = 0; c < childCanon.value; ++c)
                {
                    EClassId otherCanon = egraph.findConst(EClassId{c});
                    if (otherCanon == childCanon || otherCanon.value >= childCanon.value)
                        continue;

                    std::string other_name;
                    EClassId dummy;
                    if (getStorageWeightName(egraph, otherCanon, other_name, dummy) && other_name == tensor_name)
                    {
                        const EClass &c1 = egraph.getEClass(childCanon);
                        const EClass &c2 = egraph.getEClass(otherCanon);
                        if (c1.mem_space == c2.mem_space && c1.dtype == c2.dtype && c1.shape == c2.shape)
                        {
                            newChildren[i] = otherCanon;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }

        if (changed)
        {
            ENode altENode(enode.getKernelId(), enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                           enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                           enode.getContentHash());
            egraph.addENode(eclass_id, altENode);
        }
    }
};

// Remove redundant and identity RESHAPE operations.
// 1. Identity Reshape: RESHAPE(x, S) where x.shape == S -> merge(RESHAPE, x)
// 2. Chained Reshapes: RESHAPE(RESHAPE(x, S1), S2) -> RESHAPE(x, S2)
// 3. Consumer Unwrapping: op(..., RESHAPE(x), ...) -> op(..., x, ...) when x.shape matches
struct RemoveRedundantReshape : public Rule
{
    std::string name() const override
    {
        return "RemoveRedundantReshape";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        // Pattern 1 & 2: Direct RESHAPE node
        if (enode.getOpType() == OpType::RESHAPE && !enode.getChildren().empty())
        {
            EClassId srcClassId = egraph.findConst(enode.getChildren()[0]);
            const EClass &srcCls = egraph.getEClass(srcClassId);

            // Case 1: Identity Reshape (input shape == output shape)
            if (srcCls.shape == enode.getShape() && srcCls.dtype == enode.getDType())
            {
                EClassId e_class_id = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));
                if (e_class_id != srcClassId)
                {
                    return true;
                }
            }

            // Case 2: Chained Reshapes
            for (ENodeId cEnodeId : srcCls.enodes)
            {
                const ENode &cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                {
                    return true;
                }
            }
        }

        // Pattern 3: Consumer node taking an identity RESHAPE child
        if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE)
        {
            if (KernelRegistry::get().hasKernel(enode.getKernelId()))
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                if (kernel.is_view)
                    return false; // Skip view kernels where stride calculation depends on input view
            }

            for (EClassId child : enode.getChildren())
            {
                EClassId childClsId = egraph.findConst(child);
                const EClass &childCls = egraph.getEClass(childClsId);

                for (ENodeId cEnodeId : childCls.enodes)
                {
                    const ENode &cEnode = egraph.getENode(cEnodeId);
                    if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                    {
                        EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                        const EClass &unwrappedCls = egraph.getEClass(unwrappedChildId);
                        if (unwrappedCls.shape == childCls.shape && unwrappedChildId != childClsId)
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));

        // 1. Direct Identity Reshape -> Merge EClasses
        if (enode.getOpType() == OpType::RESHAPE && !enode.getChildren().empty())
        {
            EClassId srcClassId = egraph.findConst(enode.getChildren()[0]);
            const EClass srcCls = egraph.getEClass(srcClassId);

            if (srcCls.shape == enode.getShape() && srcCls.dtype == enode.getDType() &&
                srcCls.mem_space == enode.getMemSpace())
            {
                egraph.merge(e_class_id, srcClassId);
                return;
            }

            // 2. Chained Reshape: RESHAPE(RESHAPE(x, S1), S2) -> add RESHAPE(x, S2)
            for (ENodeId cEnodeId : srcCls.enodes)
            {
                const ENode cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                {
                    EClassId grandChildId = egraph.findConst(cEnode.getChildren()[0]);
                    std::vector<EClassId> newChildren = enode.getChildren();
                    newChildren[0] = grandChildId;

                    ENode collapsed(enode.getKernelId(), OpType::RESHAPE, "", newChildren, enode.getShape(),
                                    enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                                    enode.getContentHash());
                    egraph.addENode(e_class_id, collapsed);
                }
            }
        }

        // 3. Consumer child unwrapping
        if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE &&
            enode.getOpType() != OpType::RESHAPE)
        {
            std::vector<std::vector<EClassId>> candidateChildrenPerPos(enode.getChildren().size());

            for (uint64_t i = 0; i < enode.getChildren().size(); ++i)
            {
                EClassId childClsId = egraph.findConst(enode.getChildren()[i]);
                candidateChildrenPerPos[i].push_back(childClsId);

                const EClass &childCls = egraph.getEClass(childClsId);
                for (ENodeId cEnodeId : childCls.enodes)
                {
                    const ENode &cEnode = egraph.getENode(cEnodeId);
                    if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                    {
                        EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                        const EClass &unwrappedCls = egraph.getEClass(unwrappedChildId);
                        if (unwrappedCls.shape == childCls.shape && unwrappedChildId != childClsId)
                        {
                            candidateChildrenPerPos[i].push_back(unwrappedChildId);
                        }
                    }
                }
            }

            std::vector<std::vector<EClassId>> childCombinations;
            std::vector<EClassId> currentCombination(enode.getChildren().size());

            std::function<void(uint64_t, bool)> generateCombos = [&](uint64_t pos, bool hasUnwrapped) {
                if (pos == enode.getChildren().size())
                {
                    if (hasUnwrapped)
                        childCombinations.push_back(currentCombination);
                    return;
                }
                for (uint64_t cIdx = 0; cIdx < candidateChildrenPerPos[pos].size(); ++cIdx)
                {
                    currentCombination[pos] = candidateChildrenPerPos[pos][cIdx];
                    generateCombos(pos + 1, hasUnwrapped || (cIdx > 0));
                }
            };

            generateCombos(0, false);

            for (const auto &newChildren : childCombinations)
            {
                ENode newENode(enode.getKernelId(), enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                               enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                               enode.getContentHash());
                egraph.addENode(e_class_id, newENode);
            }
        }
    }
};