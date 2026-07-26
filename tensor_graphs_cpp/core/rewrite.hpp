// tensor_graphs_cpp/core/rewrite.hpp
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
#include "core/repo.hpp"
#include "core/shapes.hpp"

inline std::vector<std::vector<MemSpace>> findMemSpacePaths(MemSpace src, MemSpace dst, const TensorNode &node,
                                                            const std::vector<Engine> &engines)
{
    if (src == dst)
        return {{src}};

    std::unordered_map<MemSpace, std::vector<MemSpace>> adj;
    for (const auto &[uid, k] : KernelRegistry::get().getAllKernels())
    {
        if (k.opType == OpType::COPY_TO && k.input_mem_spaces.size() == 1)
        {
            adj[k.input_mem_spaces[0]].push_back(k.output_mem_space);
        }
    }

    std::vector<std::vector<MemSpace>> all_paths;
    std::vector<MemSpace> current_path = {src};
    std::unordered_set<MemSpace> visited = {src};

    std::function<void(MemSpace)> dfs = [&](MemSpace curr)
    {
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
                TensorNode dummyIn = node;
                TensorNode dummyOut = node;
                auto refs = KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", {dummyIn}, dummyOut, false,
                                                                      next, {curr}, engines, false, false, true, true);

                if (!refs.empty())
                {
                    visited.insert(next);
                    current_path.push_back(next);
                    dfs(next);
                    current_path.pop_back();
                    visited.erase(next);
                }
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
                              MemSpace mem_space, EClassId targetEClass = EClassId(), std::source_location loc = std::source_location::current())
{
    EClassId cls = targetEClass;
    if (cls == EClassId())
    {
        cls = egraph.addEClass(shape, strides, dtype, mem_space);
    }

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
    LogicalId pRoot;
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
        pRoot = pGraph._copyto(pInputs[0]);
    else if (op == OpType::SCATTER)
        pRoot = pGraph.scatter(pInputs[0], pInputs[1], pInputs[2], pInputs[3], pInputs[4]);
    else if (op == OpType::RESHAPE)
        pRoot = pGraph.reshape(pInputs[0], pInputs[1]);
    else if (op == OpType::PERMUTE)
        pRoot = pGraph.permute(pInputs[0], pInputs[1]);
    else if (op == OpType::CONCAT)
    {
        std::vector<LogicalId> concatIns;
        for (uint64_t i = 1; i < pInputs.size(); ++i)
            concatIns.push_back(pInputs[i]);
        pRoot = pGraph.concat(concatIns, pInputs[0]);
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

    if (pRoot != LogicalId())
    {
        auto matches = KernelRegistry::get().findMatchingKernelsByPattern(
            pGraph, pRoot, inNodes, outNode, false, mem_space, input_mem_spaces, {}, false, false, true);
        if (matches.empty())
        {
            std::stringstream ss;
            ss << "\n[addOpToEGraph] No matching kernel found for the given "
                  "configuration at "
               << toString(loc) << "\n"
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
            ENode n(uid, op, kernel.opName, children, shape, kernel.is_view ? strides : calcContiguousStrides(shape),
                    dtype, mem_space, kernel.engines);

            egraph.addENode(cls, n);
        }
    }
    return cls;
}

inline EClassId copyTo(EGraph &egraph, EClassId class_id, MemSpace target_mem_space)
{
    EClassId canon = egraph.find(class_id);
    const EClass cls = egraph.getEClass(canon);
    if (cls.mem_space == target_mem_space)
        return canon;

    return addOpToEGraph(egraph, OpType::COPY_TO, {canon}, cls.shape, cls.strides, cls.dtype, target_mem_space);
}

inline EClassId createCacheInputNode(EGraph &egraph, EClassId sourceClassId,
                                     std::unordered_map<EClassId, LogicalId> &eclassToLogical)
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
                    srcClass.mem_space, {}, toString(srcLogicalId));
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
                    for (uint64_t i = 0; i < eNode.getChildren().size() - 1; ++i)
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

            std::vector<KernelId> kernelMatches = KernelRegistry::get().findMatchingKernelsByPattern(
                pattern.graph, pattern.rootId, inputNodes, outputNode, false, matchedClass.mem_space, {}, {}, true,
                ignoreInputMemSpaces, true, true);

            for (KernelId uid : kernelMatches)
            {
                const KernelEntry &kernel = KernelRegistry::get().getKernel(uid);
                addFusedNode(ctx, kernel, kernel.output_mem_space, inputs, ENodeId{eNodeIdx});
            }
        }
    }

    void addFusedNode(RuleCtx &ctx, const KernelEntry &kernel, MemSpace target_mem_space,
                      const std::vector<EClassId> &child_ids, ENodeId eNodeIdx) const
    {
        EGraph &egraph = ctx.egraph;
        std::vector<EClassId> adapted_children;
        if (child_ids.size() < kernel.min_num_inputs || child_ids.size() > kernel.max_num_inputs)
        {
            Error::throw_err("[addFusedNode] child_ids.size() < kernel.min_num_inputs || "
                             "child_ids.size() > kernel.max_num_inputs");
        }

        // Pre-validate that all children can be routed to the required memory
        // spaces
        std::vector<std::vector<std::vector<MemSpace>>> child_mem_paths(child_ids.size());
        std::vector<bool> child_need_contig(child_ids.size(), false);
        std::vector<bool> child_need_copy(child_ids.size(), false);

        for (uint64_t i = 0; i < child_ids.size(); ++i)
        {
            EClassId pid = child_ids[i];
            const EClass parent = egraph.getEClass(egraph.findConst(pid));

            uint64_t ruleIdx = i;
            if (kernel.min_num_inputs != kernel.max_num_inputs)
            {
                ruleIdx = (i == child_ids.size() - 1)
                              ? (kernel.input_mem_spaces.empty() ? 0 : kernel.input_mem_spaces.size() - 1)
                              : 0;
            }

            MemSpace expectedMemSpace = {1, HandleType::CPP};
            if (!kernel.input_mem_spaces.empty() && ruleIdx < kernel.input_mem_spaces.size())
            {
                expectedMemSpace = kernel.input_mem_spaces[ruleIdx];
            }

            bool foundMemSpace = (parent.mem_space == expectedMemSpace);

            bool needCopy = !foundMemSpace;
            bool needContig = false;
            if (ruleIdx < kernel.requiresContiguous.size())
            {
                needContig = (kernel.requiresContiguous[ruleIdx] || needCopy) && !isContiguous(parent);
            }
            else
            {
                needContig = needCopy && !isContiguous(parent);
            }

            child_need_contig[i] = needContig;
            child_need_copy[i] = needCopy;

            if (needCopy)
            {
                TensorNode dummyNode;
                dummyNode.opType = OpType::INPUT;
                dummyNode.dtype = parent.dtype;
                dummyNode.setShape(parent.shape);
                dummyNode.strides = parent.strides;

                child_mem_paths[i] = findMemSpacePaths(parent.mem_space, expectedMemSpace, dummyNode, kernel.engines);
                if (child_mem_paths[i].empty())
                {
                    return; // Cannot satisfy memory constraints, abort adding this fused
                            // node
                }
            }
        }

        // We make a COPY to prevent dangling references since addOpToEGraph pushes
        // to egraph.enodes
        const ENode oldENode = egraph.getENode(eNodeIdx);
        EClassId e_class_id = egraph.getENodeEClass(eNodeIdx);

        for (uint64_t i = 0; i < child_ids.size(); ++i)
        {
            EClassId pid = child_ids[i];
            const EClass parent = egraph.getEClass(egraph.findConst(pid));

            bool needCopy = child_need_copy[i];
            bool needContig = child_need_contig[i];

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
                for (const auto &path : child_mem_paths[i])
                {
                    EClassId pathPid = currentPid;
                    EClass pathClass = currentClass;
                    for (uint64_t p_idx = 1; p_idx < path.size(); ++p_idx)
                    {
                        MemSpace next_ms = path[p_idx];
                        pathPid = addOpToEGraph(egraph, OpType::COPY_TO, {pathPid}, pathClass.shape, pathClass.strides,
                                                pathClass.dtype, next_ms,
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

        std::vector<uint64_t> strides;
        if (kernel.is_view)
        {
            strides = oldENode.getStrides();
        }
        else
        {
            strides = calcContiguousStrides(oldENode.getShape());
        }

        ENode enode(kernel.uid, kernel.opType, kernel.opName, adapted_children, oldENode.getShape(), strides,
                    oldENode.getDType(), target_mem_space, kernel.engines);

        MemSpace originalMemSpace = egraph.getEClass(egraph.findConst(e_class_id)).mem_space;
        if (target_mem_space == originalMemSpace)
        {
            egraph.addENode(e_class_id, enode);
        }
        else
        {
            EClassId newEClass =
                egraph.addEClass(enode.getShape(), enode.getStrides(), enode.getDType(), target_mem_space);
            newEClass = egraph.addENode(newEClass, enode);

            addOpToEGraph(egraph, OpType::COPY_TO, {newEClass}, enode.getShape(), enode.getStrides(), enode.getDType(),
                          originalMemSpace, e_class_id);

            auto it = ctx.eclassToLogical.find(egraph.findConst(e_class_id));
            if (it != ctx.eclassToLogical.end())
            {
                ctx.eclassToLogical[newEClass] = it->second;
            }
        }
    }

    static bool isStructuralConstant(OpType op, uint64_t inputIdx, uint64_t numInputs)
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
        if (op == OpType::CONCAT && inputIdx == 0)
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

            if (!matchPatternClass(eNode.getChildren().back(), egraph, pNode.child_ids.back(), pattern, binding,
                                   protectedEClasses, true))
                return false;

            bool firstTensor = true;
            for (uint64_t i = 0; i < eNode.getChildren().size() - 1; ++i)
            {
                if (firstTensor)
                {
                    if (!matchPatternClass(eNode.getChildren()[i], egraph, pNode.child_ids[0], pattern, binding,
                                           protectedEClasses, false))
                        return false;
                    firstTensor = false;
                }
                else
                {
                    EClassId canonChild = egraph.findConst(eNode.getChildren()[i]);
                    const EClass &childCls = egraph.getEClass(canonChild);
                    if (childCls.dtype != pattern.dtypes[0])
                        return false;
                }
            }
            return true;
        }

        if (eNode.getChildren().size() != pNode.child_ids.size())
            return false;

        for (uint64_t i = 0; i < eNode.getChildren().size(); ++i)
        {
            bool childIgnoreConst = isStructuralConstant(eNode.getOpType(), i, eNode.getChildren().size());
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

    EClassId addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
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
            if (cClass.mem_space != outClass.mem_space || cClass.strides != contigStrides)
            {
                if (cClass.mem_space != outClass.mem_space)
                {
                    currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {constClass}, outClass.shape, contigStrides,
                                                  outClass.dtype, outClass.mem_space);
                }
                else
                {
                    currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {constClass}, outClass.shape,
                                                  contigStrides, outClass.dtype, outClass.mem_space);
                }
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

        EClassId currentTarget;
        if (cClass.mem_space != outClass.mem_space)
        {
            currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {constClass}, outClass.shape, contigStrides,
                                          outClass.dtype, outClass.mem_space);
        }
        else
        {
            currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {constClass}, outClass.shape, contigStrides,
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

            EClassId startsId = addIntConst(egraph, starts);
            EClassId endsId = addIntConst(egraph, ends);
            EClassId stepsId = addIntConst(egraph, steps);

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
            EClassId sliceC = addOpToEGraph(egraph, OpType::SLICE, {constClass, startsId, endsId, stepsId}, sliceShape,
                                            sliceStridesC, cClass.dtype, cClass.mem_space);

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
                    const ENode &opNode = egraph.getENode(srcNodeIdx);
                    OpType op = opNode.getOpType();
                    if (!(op == OpType::ADD || op == OpType::MUL || op == OpType::DIVIDE || op == OpType::POWER ||
                          op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST ||
                          op == OpType::LT || op == OpType::EQ || op == OpType::AND || op == OpType::OR ||
                          op == OpType::NOT))
                        continue;

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
                const ENode opNode = egraph.getENode(srcNodeIdx);
                OpType op = opNode.getOpType();
                if (!(op == OpType::ADD || op == OpType::MUL || op == OpType::DIVIDE || op == OpType::POWER ||
                      op == OpType::SIN || op == OpType::COS || op == OpType::NEGATE || op == OpType::CAST ||
                      op == OpType::LT || op == OpType::EQ || op == OpType::AND || op == OpType::OR ||
                      op == OpType::NOT))
                {
                    continue;
                }

                MatchKey key{eNodeIdx, sliceNodeIdx.value, srcNodeIdx.value};
                if (!visited.insert(key).second)
                    continue;

                std::vector<EClassId> newChildren;
                for (EClassId childId : opNode.getChildren())
                {
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

                    if (!isContiguous(childCls))
                    {
                        canonChildId = addOpToEGraph(egraph, OpType::CONTIGUOUS, {canonChildId}, childCls.shape,
                                                     calcContiguousStrides(childCls.shape), childCls.dtype, childCls.mem_space);
                    }

                    EClassId childSlice =
                        addOpToEGraph(egraph, OpType::SLICE, {canonChildId, startsId, endsId, stepsId}, sliceShape,
                                      childSliceStrides, childCls.dtype, childCls.mem_space);

                    EClassId sliceContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {childSlice}, sliceShape,
                                                         sliceContigStrides, childCls.dtype, childCls.mem_space);
                    newChildren.push_back(sliceContig);
                }

                EClassId opEClass = addOpToEGraph(egraph, op, newChildren, sliceShape, sliceContigStrides,
                                                  sliceNode.getDType(), sliceNode.getMemSpace());

                EClassId contigSlicedOp =
                    addOpToEGraph(egraph, OpType::CONTIGUOUS, {opEClass}, sliceShape, sliceContigStrides,
                                  sliceNode.getDType(), sliceNode.getMemSpace());

                EClassId op_cache = createCacheInputNode(egraph, srcClass, ctx.eclassToLogical);

                const EClass srcEClass = egraph.getEClass(srcClass);
                EClassId scatterClass =
                    addOpToEGraph(egraph, OpType::SCATTER, {op_cache, contigSlicedOp, startsId, endsId, stepsId},
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

    EClassId addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
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
                    if (egraph.getENode(srcNodeIdx).getOpType() == OpType::DOT)
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

                EClassId startsIdA = addIntConst(egraph, startsA);
                EClassId endsIdA = addIntConst(egraph, endsA);
                EClassId stepsIdA = addIntConst(egraph, stepsA);

                EClassId startsIdB = addIntConst(egraph, startsB);
                EClassId endsIdB = addIntConst(egraph, endsB);
                EClassId stepsIdB = addIntConst(egraph, stepsB);

                auto createSlice = [&](EClassId classId, const std::vector<int32_t> &st, const std::vector<int32_t> &en,
                                       EClassId stId, EClassId enId, EClassId stepId)
                {
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

                EClassId contigSlicedOp =
                    addOpToEGraph(egraph, OpType::CONTIGUOUS, {dotEClass}, sliceShape, sliceContigStrides,
                                  sliceNode.getDType(), sliceNode.getMemSpace());

                EClassId op_cache = createCacheInputNode(egraph, srcClass, ctx.eclassToLogical);

                const EClass srcEClass = egraph.getEClass(srcClass);
                EClassId scatterClass =
                    addOpToEGraph(egraph, OpType::SCATTER, {op_cache, contigSlicedOp, startsId, endsId, stepsId},
                                  srcEClass.shape, srcEClass.strides, dotNode.getDType(), dotNode.getMemSpace());

                egraph.merge(srcClass, scatterClass);
            }
        }
    }
};

struct FlattenBatchDot : public Rule
{
    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "FlattenBatchDot";
    }

    EClassId addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::DOT || enode.getChildren().size() != 2)
            return false;

        if (visited.count(eNodeIdx))
            return false;

        const std::vector<uint32_t> &outShape = enode.getShape();
        if (outShape.size() != 4 || outShape[0] != 1)
            return false;

        EClassId aClass = egraph.findConst(enode.getChildren()[0]);
        EClassId bClass = egraph.findConst(enode.getChildren()[1]);
        const EClass aCls = egraph.getEClass(aClass);
        const EClass bCls = egraph.getEClass(bClass);

        if (aCls.shape.size() != 4 || aCls.shape[0] != 1)
            return false;
        if (bCls.shape.size() != 4 || bCls.shape[0] != 1)
            return false;
        if (aCls.shape[1] != bCls.shape[1])
            return false;
        if (aCls.shape[3] != bCls.shape[2])
            return false;

        if (outShape[1] != aCls.shape[1] || outShape[2] != aCls.shape[2] || outShape[3] != bCls.shape[3])
            return false;

        if (!isContiguous(aCls) || !isContiguous(bCls))
            return false;
        if (!isContiguous(egraph.getEClass(egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})))))
            return false;

        return true;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode dotNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        if (!visited.insert(eNodeIdx).second)
            return;

        EClassId aClass = egraph.findConst(dotNode.getChildren()[0]);
        EClassId bClass = egraph.findConst(dotNode.getChildren()[1]);
        const EClass aCls = egraph.getEClass(aClass);
        const EClass bCls = egraph.getEClass(bClass);

        std::vector<uint32_t> a3 = {aCls.shape[1], aCls.shape[2], aCls.shape[3]};
        std::vector<uint32_t> b3 = {bCls.shape[1], bCls.shape[2], bCls.shape[3]};
        std::vector<uint32_t> y3 = {aCls.shape[1], aCls.shape[2], bCls.shape[3]};

        std::vector<int32_t> a3_int(a3.begin(), a3.end());
        std::vector<int32_t> b3_int(b3.begin(), b3.end());
        std::vector<int32_t> y3_int(y3.begin(), y3.end());

        EClassId a3_shape_id = addIntConst(egraph, a3_int);
        EClassId b3_shape_id = addIntConst(egraph, b3_int);
        EClassId y3_shape_id = addIntConst(egraph, y3_int);

        std::vector<uint64_t> a3_strides = calcContiguousStrides(a3);
        EClassId rA = addOpToEGraph(egraph, OpType::RESHAPE, {aClass, a3_shape_id}, a3, a3_strides, dotNode.getDType(),
                                    dotNode.getMemSpace());

        std::vector<uint64_t> b3_strides = calcContiguousStrides(b3);
        EClassId rB = addOpToEGraph(egraph, OpType::RESHAPE, {bClass, b3_shape_id}, b3, b3_strides, dotNode.getDType(),
                                    dotNode.getMemSpace());

        std::vector<uint64_t> y3_strides = calcContiguousStrides(y3);
        EClassId rY =
            addOpToEGraph(egraph, OpType::DOT, {rA, rB}, y3, y3_strides, dotNode.getDType(), dotNode.getMemSpace());

        const EClass outCls = egraph.getEClass(egraph.findConst(e_class_id));
        std::vector<int32_t> out4_int(outCls.shape.begin(), outCls.shape.end());
        EClassId out4_shape_id = addIntConst(egraph, out4_int);
        EClassId outReshape = addOpToEGraph(egraph, OpType::RESHAPE, {rY, out4_shape_id}, outCls.shape, outCls.strides,
                                            dotNode.getDType(), dotNode.getMemSpace());

        egraph.merge(e_class_id, outReshape);
    }
};

struct FlattenElementwise : public Rule
{
    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "FlattenElementwise";
    }

    EClassId addIntConst(EGraph &egraph, const std::vector<int32_t> &vals) const
    {
        return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
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
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (!isSupportedOp(enode.getOpType()))
            return false;
        if (visited.count(eNodeIdx))
            return false;

        const std::vector<uint32_t> &outShape = enode.getShape();
        if (outShape.size() < 2)
            return false;

        for (EClassId childId : enode.getChildren())
        {
            const EClass &childCls = egraph.getEClass(egraph.findConst(childId));
            if (childCls.shape != outShape)
                return false;
            if (!isContiguous(childCls))
                return false;
        }

        const EClass &outCls = egraph.getEClass(egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})));
        if (!isContiguous(outCls))
            return false;

        return true;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode opNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        if (!visited.insert(eNodeIdx).second)
            return;

        const EClass outCls = egraph.getEClass(egraph.findConst(e_class_id));
        const std::vector<uint32_t> &outShape = outCls.shape;

        uint64_t total = 1;
        for (uint32_t d : outShape)
            total *= d;
        if (total == 0)
            return;

        std::vector<uint32_t> flatShape = {(uint32_t)total};
        std::vector<int32_t> flat_int = {(int32_t)total};
        std::vector<int32_t> out_int(outShape.begin(), outShape.end());
        EClassId flat_shape_id = addIntConst(egraph, flat_int);
        EClassId out_shape_id = addIntConst(egraph, out_int);

        std::vector<uint64_t> flatStrides = {1};

        std::vector<EClassId> flatChildren;
        for (EClassId childId : opNode.getChildren())
        {
            EClassId canonChild = egraph.findConst(childId);
            const EClass childCls = egraph.getEClass(canonChild);
            EClassId r = addOpToEGraph(egraph, OpType::RESHAPE, {canonChild, flat_shape_id}, flatShape, flatStrides,
                                       childCls.dtype, opNode.getMemSpace());
            flatChildren.push_back(r);
        }

        EClassId flatOut = addOpToEGraph(egraph, opNode.getOpType(), flatChildren, flatShape, flatStrides,
                                         opNode.getDType(), opNode.getMemSpace());

        EClassId outReshape = addOpToEGraph(egraph, OpType::RESHAPE, {flatOut, out_shape_id}, outCls.shape,
                                            outCls.strides, opNode.getDType(), opNode.getMemSpace());

        egraph.merge(e_class_id, outReshape);
    }
};