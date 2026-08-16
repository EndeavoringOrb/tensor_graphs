#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/logging.hpp"
#include "core/misc.hpp"
#include "core/plan/extractor.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/plan/validators/cycle.hpp"
#include "core/plan/validators/mem.hpp"
#include "core/rewrite.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

struct ExtractionResult
{
    std::unordered_map<EClassId, uint32_t> selection_map;
    std::vector<EClassId> order;
    std::vector<ParallelBuffer> buffers;
    std::unordered_map<EClassId, BufferId> eclass_to_buf;
    float cost;
    std::unordered_map<EClassId, float> eclass_to_cost;
};

struct CacheIterator
{
    const Graph &graph;
    std::vector<LogicalId> candidate_nodes;
    std::vector<MemSpace> avail_mem_spaces;
    std::shared_ptr<SearchDelegate> delegate;

    std::vector<uint32_t> num_users;
    std::vector<std::vector<int>> valid_choices;

    int k = 0;
    bool is_done = false;
    bool first_yield = true;
    std::vector<int> state;
    std::vector<std::vector<uint32_t>> choice_orders;
    std::unordered_map<LogicalId, MemSpace> current_cache_selection;

    CacheIterator(const Graph &_graph, const std::vector<LogicalId> &_candidates,
                  const std::vector<MemSpace> &_avail_mem_spaces, std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : graph(_graph), candidate_nodes(_candidates), avail_mem_spaces(_avail_mem_spaces), delegate(_delegate)
    {
        init();
    }

    void init()
    {
        uint32_t N = static_cast<uint32_t>(candidate_nodes.size());
        state.assign(N, 0);
        choice_orders.resize(N);
        valid_choices.resize(N);
        num_users.assign(N, 0);

        std::unordered_map<LogicalId, uint32_t> user_counts;
        for (const auto &pair : graph.nodes)
        {
            for (LogicalId child_id : pair.second.child_ids)
            {
                user_counts[child_id]++;
            }
        }

        for (uint32_t i = 0; i < N; ++i)
        {
            LogicalId id = candidate_nodes[i];
            num_users[i] = user_counts[id];

            // Choice 0: Not cached
            valid_choices[i].push_back(0);

            // Choice 1..M: Cached in avail_mem_spaces[choice - 1]
            for (size_t m = 0; m < avail_mem_spaces.size(); ++m)
            {
                valid_choices[i].push_back(static_cast<int>(m + 1));
            }
        }

        if (delegate && N > 0)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;
            std::unordered_map<LogicalId, uint32_t> id_to_idx;

            for (uint32_t i = 0; i < N; ++i)
            {
                LogicalId id = candidate_nodes[i];
                id_to_idx[id] = i;
                const TensorNode &node = graph.getNode(id);

                node_features.push_back(static_cast<float>(node.getSizeBytes()));
                node_features.push_back(static_cast<float>(node.opType));
                node_features.push_back(static_cast<float>(node.dtype));
                bool is_storage =
                    (graph.input_data_types.count(id) && graph.input_data_types.at(id) == InputDataType::STORAGE);
                node_features.push_back(is_storage ? 1.0f : 0.0f);
                node_features.push_back(static_cast<float>(num_users[i]));
            }

            for (uint32_t i = 0; i < N; ++i)
            {
                LogicalId id = candidate_nodes[i];
                const TensorNode &node = graph.getNode(id);
                for (LogicalId pid : node.child_ids)
                {
                    auto p_it = id_to_idx.find(pid);
                    if (p_it != id_to_idx.end())
                    {
                        edge_src.push_back(p_it->second);
                        edge_dst.push_back(i);
                    }
                }
            }

            delegate->init_cache_graph(node_features, edge_src, edge_dst);
        }
    }

    bool ascend()
    {
        k--;
        while (k >= 0)
        {
            if (valid_choices[k].empty())
            {
                k--;
                continue;
            }
            if (delegate && valid_choices[k].size() > 1)
            {
                delegate->pop_state();
            }

            LogicalId id = candidate_nodes[k];
            current_cache_selection.erase(id);

            if (state[k] < valid_choices[k].size())
            {
                return true;
            }
            state[k] = 0;
            k--;
        }
        return false;
    }

    bool getNextCacheSelection(std::unordered_map<LogicalId, MemSpace> &out_cached_nodes)
    {
        if (is_done)
            return false;

        uint32_t N = static_cast<uint32_t>(candidate_nodes.size());
        if (N == 0)
        {
            if (first_yield)
            {
                first_yield = false;
                out_cached_nodes.clear();
                return true;
            }
            is_done = true;
            return false;
        }

        if (!first_yield)
        {
            if (!ascend())
            {
                is_done = true;
                return false;
            }
        }
        first_yield = false;

        while (k >= 0)
        {
            if (k == static_cast<int>(N))
            {
                out_cached_nodes = current_cache_selection;
                return true;
            }

            if (valid_choices[k].empty())
            {
                k++;
                continue;
            }

            LogicalId id = candidate_nodes[k];
            const TensorNode &node = graph.getNode(id);

            if (state[k] == 0)
            {
                if (delegate && valid_choices[k].size() > 1)
                {
                    delegate->push_state();

                    std::vector<ActionFeatureCache> features;
                    features.reserve(valid_choices[k].size());

                    uint64_t node_size = node.getSizeBytes();

                    for (int choice : valid_choices[k])
                    {
                        ActionFeatureCache f;
                        f.size = node_size;
                        f.op_type = static_cast<uint32_t>(node.opType);
                        f.num_users = static_cast<float>(num_users[k]);
                        f.logical_id = id.value;

                        if (choice == 0)
                        {
                            f.is_cached = 0.0f;
                            f.mem_space = MemSpace{0, HandleType::STORAGE};
                        }
                        else
                        {
                            f.is_cached = 1.0f;
                            f.mem_space = avail_mem_spaces[choice - 1];
                        }
                        features.push_back(f);
                    }

                    choice_orders[k] = delegate->order_cache(features);
                }
                else
                {
                    choice_orders[k].resize(valid_choices[k].size());
                    std::iota(choice_orders[k].begin(), choice_orders[k].end(), 0u);
                }
            }

            if (state[k] < valid_choices[k].size())
            {
                uint32_t choice_idx = choice_orders[k][state[k]];
                int choice = valid_choices[k][choice_idx];
                state[k]++;

                if (choice > 0)
                {
                    current_cache_selection[id] = avail_mem_spaces[choice - 1];
                }
                else
                {
                    current_cache_selection.erase(id);
                }

                k++;
            }
            else
            {
                state[k] = 0;
                if (!ascend())
                {
                    is_done = true;
                    return false;
                }
            }
        }

        is_done = true;
        return false;
    }
};

struct Planner
{
    CostModel &costModel;
    std::unordered_map<MemSpace, uint64_t> mem_caps;

    void preallocateLogicalBuffers(const Graph &graph, const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                   std::unordered_map<LogicalId, ParallelBuffer> &out) const
    {
        out.clear();

        struct PreAllocEntry
        {
            LogicalId logicalId;
            MemSpace memSpace;
            std::vector<uint32_t> shape;
            DType dtype;
        };
        std::vector<PreAllocEntry> entries;

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};

        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType != OpType::INPUT)
                continue;

            auto idtIt = graph.input_data_types.find(node.id);
            if (idtIt != graph.input_data_types.end() && idtIt->second == InputDataType::STORAGE)
                continue;

            entries.push_back({node.id, ram, node.getShape(), node.dtype});
        }

        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
            MemSpace ms = kv.second;
            if (!graph.hasNode(logicalId))
                continue;
            const TensorNode &node = graph.getNode(logicalId);
            bool alreadyAdded = false;
            for (const auto &e : entries)
            {
                if (e.logicalId == logicalId)
                {
                    alreadyAdded = true;
                    break;
                }
            }
            if (alreadyAdded)
                continue;
            entries.push_back({logicalId, ms, node.getShape(), node.dtype});
        }

        std::sort(entries.begin(), entries.end(),
                  [](const PreAllocEntry &a, const PreAllocEntry &b) { return a.logicalId < b.logicalId; });

        std::unordered_map<MemSpace, uint64_t> cursor;
        BufferId nextId{0};
        for (const auto &e : entries)
        {
            if (e.memSpace == storage)
                continue;

            uint64_t size_bytes = getSizeBytes(e.shape, e.dtype);
            if (size_bytes == 0)
                continue;
            size_bytes = (size_bytes + 4095) & ~4095ULL;

            uint64_t offset = cursor[e.memSpace];
            cursor[e.memSpace] = offset + size_bytes;

            ParallelBuffer buf;
            buf.id = nextId++;
            buf.mem_space = e.memSpace;
            buf.size = size_bytes;
            buf.start = 0;
            buf.end = std::numeric_limits<uint32_t>::max();
            buf.offset = static_cast<int64_t>(offset);
            out[e.logicalId] = std::move(buf);
        }
    }

    std::unordered_map<LogicalId, MemSpace> searchBestCacheNodes(LogicalId rootId, const Graph &graph,
                                                                 const std::vector<Bucket> &buckets,
                                                                 std::shared_ptr<SearchDelegate> delegate = nullptr,
                                                                 float minCompileSeconds = 0.0f)
    {
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);

        std::unordered_map<LogicalId, bool> logicalDirty;
        for (const auto &bucket : buckets)
        {
            for (LogicalId nodeId : topo)
            {
                if (bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty())
                {
                    logicalDirty[nodeId] = true;
                }
                else
                {
                    bool isDirty = false;
                    for (LogicalId pid : graph.getNode(nodeId).child_ids)
                    {
                        if (logicalDirty[pid])
                        {
                            isDirty = true;
                            break;
                        }
                    }
                    if (isDirty)
                        logicalDirty[nodeId] = true;
                }
            }
        }

        std::vector<LogicalId> candidates;
        for (LogicalId nodeId : topo)
        {
            if (!logicalDirty[nodeId] && graph.getNode(nodeId).getSizeBytes() > 0)
            {
                candidates.push_back(nodeId);
            }
        }

        std::vector<MemSpace> avail_mem_spaces;
        for (const auto &kv : mem_caps)
        {
            if (kv.first.type != HandleType::STORAGE)
            {
                avail_mem_spaces.push_back(kv.first);
            }
        }
        std::sort(avail_mem_spaces.begin(), avail_mem_spaces.end(), [](const MemSpace &a, const MemSpace &b) {
            if (a.type != b.type)
                return a.type < b.type;
            return a.idx < b.idx;
        });

        CacheIterator cache_iter(graph, candidates, avail_mem_spaces, delegate);
        std::unordered_map<LogicalId, MemSpace> current_cache;
        std::unordered_map<LogicalId, MemSpace> best_cache;
        float best_cost = TGConstants::INF;

        uint32_t rep_bucket_idx = buckets.size() > 1 ? 1 : 0;
        const Bucket &rep_bucket = buckets[rep_bucket_idx];

        auto start_time = std::chrono::high_resolution_clock::now();
        uint32_t max_cache_evals = 100;
        uint32_t eval_count = 0;

        while (cache_iter.getNextCacheSelection(current_cache))
        {
            eval_count++;
            try
            {
                std::unordered_map<LogicalId, ParallelBuffer> preallocated;
                preallocateLogicalBuffers(graph, current_cache, preallocated);

                CompiledGraph plan_res =
                    plan(rootId, graph, rep_bucket, current_cache, true, true, nullptr, preallocated, 0.0f, delegate);
                float cost = plan_res.cost();
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_cache = current_cache;
                }
            }
            catch (...)
            {
            }

            if (best_cost < TGConstants::INF && minCompileSeconds == 0.0f && delegate == nullptr)
            {
                break;
            }

            if (minCompileSeconds > 0.0f)
            {
                auto now = std::chrono::high_resolution_clock::now();
                float elapsed = std::chrono::duration<float>(now - start_time).count();
                if (elapsed >= minCompileSeconds)
                    break;
            }
            if (eval_count >= max_cache_evals)
                break;
        }

        return best_cache;
    }

    void inferShapes(const std::vector<LogicalId> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (LogicalId nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
        }
    }

    void saturate(EGraph &egraph, const std::unordered_set<EClassId> &protectedEClasses,
                  std::unordered_map<EClassId, LogicalId> &eclassToLogical, bool injected,
                  bool allowPushDownOnProtected = false, Repo *repo = nullptr)
    {
        RuleCtx ctx{egraph, protectedEClasses, eclassToLogical, repo};
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(std::make_unique<FusionRule>());
        rules.emplace_back(std::make_unique<FlattenBatchDot>());
        rules.emplace_back(std::make_unique<FlattenElementwise>());
        rules.emplace_back(std::make_unique<DotSplitRule>());
        rules.emplace_back(std::make_unique<RemoveContiguous>());
        rules.emplace_back(std::make_unique<RemoveCopyChains>());
        if (injected)
        {
            rules.emplace_back(std::make_unique<InfinityDomination>());
            rules.emplace_back(std::make_unique<SlicePushDownElementwise>(allowPushDownOnProtected));
            rules.emplace_back(std::make_unique<SlicePushDownDot>(allowPushDownOnProtected));
        }

        std::map<std::string, uint32_t> ruleMatchCounts;
        uint64_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;
        ProgressTimer timer(0, "saturating");
        while (changed)
        {
            if (InterruptManager::isInterrupted())
            {
                std::cerr << "\n[Planner.saturate] Interrupt detected, aborting execution..." << std::endl;
                InterruptManager::cleanup();
                std::exit(SIGINT);
            }
            iterations++;
            uint32_t preUniqueNodes = egraph.getNumUniqueENodes();
            for (uint32_t eNodeIdx = 0; eNodeIdx < egraph.getENodes().size(); eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    if (!rule->match(eNodeIdx, ctx))
                        continue;

                    rule->apply(eNodeIdx, ctx);
                    changed = true;
                    ruleMatchCounts[rule->name()]++;
                    nMatches++;
                }
            }
            egraph.rebuild();
            uint32_t postUniqueNodes = egraph.getNumUniqueENodes();
            changed = preUniqueNodes != postUniqueNodes;
            std::stringstream ss;
            ss << "\n--- Saturation Summary (" << iterations << " iterations) ---" << std::endl;
            for (auto const &[name, count] : ruleMatchCounts)
            {
                ss << "  " << name << ": " << count << " matches\n";
            }
            ss << "Total Matches: " << nMatches;
            LOG(L_DEBUG) << ss.str();
            if (!changed)
            {
                LOG(L_INFO) << ss.str();
            }
            timer.tick();
        }
    }

    // Propagate invalidity from empty EClasses upward to parent ENodes
    uint32_t deathCascade(EGraph &egraph)
    {
        uint32_t numClasses = egraph.getClasses().size();
        std::vector<bool> enode_valid(egraph.getENodes().size(), false);
        std::vector<uint32_t> valid_enode_count(numClasses, 0);
        std::vector<std::vector<ENodeId>> parents_map(numClasses);

        // construct parents_map
        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            const EClass &cls = egraph.getEClass(e_class_id);
            valid_enode_count[e_class_id.value] = static_cast<uint32_t>(cls.enodes.size());

            for (ENodeId enodeId : cls.enodes)
            {
                enode_valid[enodeId.value] = true;
                const ENode &enode = egraph.getENode(enodeId);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = egraph.findConst(child);
                    parents_map[canon_child.value].push_back(enodeId);
                }
            }
        }

        std::vector<EClassId> dead_worklist;
        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            if (valid_enode_count[e_class_id.value] == 0)
            {
                dead_worklist.push_back(e_class_id);
            }
        }

        uint32_t cascadePruned = 0;
        while (!dead_worklist.empty())
        {
            EClassId dead_cls = dead_worklist.back();
            dead_worklist.pop_back();

            for (ENodeId parent_enode_id : parents_map[dead_cls.value])
            {
                if (enode_valid[parent_enode_id.value])
                {
                    enode_valid[parent_enode_id.value] = false;
                    cascadePruned++;

                    EClassId parent_cls = egraph.findConst(egraph.getENodeEClass(parent_enode_id));
                    if (valid_enode_count[parent_cls.value] > 0)
                    {
                        valid_enode_count[parent_cls.value]--;
                        if (valid_enode_count[parent_cls.value] == 0)
                        {
                            dead_worklist.push_back(parent_cls);
                        }
                    }
                }
            }
        }

        if (cascadePruned == 0)
            return 0;

        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            std::vector<ENodeId> filteredEnodes;
            filteredEnodes.reserve(cls.enodes.size());
            for (ENodeId enodeId : cls.enodes)
            {
                if (enode_valid[enodeId.value])
                {
                    filteredEnodes.push_back(enodeId);
                }
            }
            cls.enodes = std::move(filteredEnodes);
        }

        return cascadePruned;
    }

    std::vector<ENodeInfo> computeENodeInfos(const EGraph &egraph,
                                             const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                             const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                             bool strictCache)
    {
        std::vector<ENodeInfo> enodeInfos(egraph.getENodes().size());

        auto isConstantNeeded = [](OpType op, uint64_t inputIdx, uint64_t numInputs) -> bool {
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
        };

        ProgressTimer timer3(egraph.getENodes().size(), "calculating enode info");
        for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            const ENode &enode = egraph.getENodes()[i];
            ENodeInfo info;
            info.is_view = false;

            if (enode.getKernelId() != KernelId{0})
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                info.is_view = kernel.is_view;
            }

            if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            {
                info.cost = 0.0f;
                if (strictCache && enode.getOpType() == OpType::CACHE)
                {
                    EClassId e_class_id = egraph.getENodeEClass(ENodeId{i});
                    EClassId canonId = egraph.findConst(e_class_id);
                    LogicalId logicalId =
                        eclassToLogical.count(canonId) ? eclassToLogical.at(canonId) : LogicalId{UINT32_MAX};
                    if (logicalId == LogicalId{UINT32_MAX} || cachedNodes.find(logicalId) == cachedNodes.end())
                    {
                        info.cost = TGConstants::INF;
                    }
                    else if (enode.getMemSpace() != cachedNodes.at(logicalId))
                    {
                        info.cost = TGConstants::INF;
                    }
                }
            }
            else if (enode.getKernelId() != KernelId{0})
            {
                std::vector<std::vector<uint32_t>> inShapes;
                std::vector<std::vector<uint64_t>> inStrides;
                std::vector<DType> inDTypes;
                std::vector<std::vector<uint8_t>> inConstants;

                inShapes.reserve(enode.getChildren().size());
                inStrides.reserve(enode.getChildren().size());
                inDTypes.reserve(enode.getChildren().size());
                inConstants.reserve(enode.getChildren().size());

                const ReferenceGraphEntry *refEntry = nullptr;
                std::unique_ptr<Graph> pGraph;
                std::vector<LogicalId> pInputs;

                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                if (enode.getOpType() == OpType::FUSED)
                {
                    refEntry = ReferenceGraphRegistry::get().getFactory(kernel.opName);
                    if (refEntry)
                    {
                        pGraph = std::make_unique<Graph>();
                        for (uint64_t k = 0; k < kernel.min_num_inputs; ++k)
                        {
                            pInputs.push_back(pGraph->input(kernel.dummyShapes[k], kernel.dtypes[k]));
                        }
                        refEntry->factory(pInputs, *pGraph);
                    }
                }

                for (uint64_t j = 0; j < enode.getChildren().size(); j++)
                {
                    EClassId childEClassId = enode.getChildren()[j];
                    const EClass &childCls = egraph.getEClass(egraph.findConst(childEClassId));
                    inShapes.push_back(childCls.shape);

                    std::vector<uint64_t> strides_cast;
                    strides_cast.reserve(childCls.strides.size());
                    for (uint64_t s : childCls.strides)
                        strides_cast.push_back(s);
                    inStrides.push_back(std::move(strides_cast));

                    inDTypes.push_back(childCls.dtype);

                    EClassId canonChild = egraph.findConst(childEClassId);
                    bool needed = false;

                    if (enode.getOpType() == OpType::FUSED)
                    {
                        if (refEntry && pGraph)
                        {
                            auto traceToInputIdx = [&](LogicalId pid) -> int {
                                LogicalId curr = pid;
                                while (pGraph->hasNode(curr) && (pGraph->getNode(curr).opType == OpType::CONTIGUOUS ||
                                                                 pGraph->getNode(curr).opType == OpType::CAST ||
                                                                 pGraph->getNode(curr).opType == OpType::COPY_TO ||
                                                                 pGraph->getNode(curr).opType == OpType::RESHAPE ||
                                                                 pGraph->getNode(curr).opType == OpType::PERMUTE))
                                {
                                    if (pGraph->getNode(curr).child_ids.empty())
                                        break;
                                    curr = pGraph->getNode(curr).child_ids[0];
                                }
                                for (uint64_t k = 0; k < pInputs.size(); ++k)
                                {
                                    if (pInputs[k] == curr)
                                        return (int)k;
                                }
                                return -1;
                            };

                            for (const auto &pair : pGraph->nodes)
                            {
                                const TensorNode &n = pair.second;
                                for (uint64_t p_idx = 0; p_idx < n.child_ids.size(); ++p_idx)
                                {
                                    if (isConstantNeeded(n.opType, p_idx, n.child_ids.size()))
                                    {
                                        int inputIdx = traceToInputIdx(n.child_ids[p_idx]);
                                        if (kernel.min_num_inputs != kernel.max_num_inputs)
                                        {
                                            if (inputIdx == 0 && j == 0)
                                            {
                                                needed = true;
                                                break;
                                            }
                                            else if (inputIdx >= 1 && j >= 1)
                                            {
                                                needed = true;
                                                break;
                                            }
                                        }
                                        else if (inputIdx == (int)j)
                                        {
                                            needed = true;
                                            break;
                                        }
                                    }
                                }
                                if (needed)
                                    break;
                            }
                        }
                    }
                    else
                    {
                        needed = isConstantNeeded(enode.getOpType(), j, enode.getChildren().size());
                    }

                    if (needed && egraph.constantStaging.count(canonChild))
                    {
                        inConstants.push_back(*egraph.constantStaging.at(canonChild));
                    }
                    else
                    {
                        inConstants.push_back({});
                    }
                }

                info.cost = costModel.estimateCost(enode.getKernelId(), enode.getShape(), enode.getStrides(),
                                                   enode.getDType(), inShapes, inStrides, inDTypes, inConstants);
            }
            else
            {
                info.cost = TGConstants::INF;
            }

            enodeInfos[i] = std::move(info);
            timer3.tick();
        }
        return enodeInfos;
    }

    void pruneEGraph(EGraph &egraph, const std::vector<ENodeInfo> &enodeInfos)
    {
        bool droppedInf = false;
        uint32_t totalPruned = 0;
        for (uint32_t i = 0; i < egraph.getClasses().size(); ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            std::vector<ENodeId> validEnodes;
            validEnodes.reserve(cls.enodes.size());

            // 1. Remove infinite-cost nodes
            for (ENodeId enodeId : cls.enodes)
            {
                if (enodeInfos[enodeId.value].cost == TGConstants::INF)
                {
                    droppedInf = true;
                }
                else
                {
                    validEnodes.push_back(enodeId);
                }
            }

            // 2. Prune duplicated nodes to minimize search space bloat
            std::vector<ENodeId> deduped;
            deduped.reserve(validEnodes.size());
            for (uint64_t idxA = 0; idxA < validEnodes.size(); ++idxA)
            {
                ENodeId idA = validEnodes[idxA];
                const ENode &a = egraph.getENode(idA);
                const ENodeInfo &ia = enodeInfos[idA.value];

                bool dominated = false;
                for (uint64_t idxB = 0; idxB < validEnodes.size(); ++idxB)
                {
                    if (idxA == idxB)
                        continue;
                    ENodeId idB = validEnodes[idxB];
                    const ENode &b = egraph.getENode(idB);
                    const ENodeInfo &ib = enodeInfos[idB.value];

                    if (a != b)
                        continue;

                    if (ib.cost < ia.cost - 1e-9f)
                    {
                        dominated = true;
                        break;
                    }
                    if (std::abs(ib.cost - ia.cost) <= 1e-9f && idB < idA)
                    {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated)
                    deduped.push_back(idA);
            }
            totalPruned += (validEnodes.size() - deduped.size());
            cls.enodes = std::move(deduped);
        }

        totalPruned += deathCascade(egraph);

        if (totalPruned > 0)
        {
            std::cout << "[Planner.pruneEGraph] Pruned " << totalPruned << " dominated enodes from the search space."
                      << std::endl;
        }
    }

    ExtractionResult extractBest(const LogicalId rootId, const Graph &graph, const EGraph &egraph,
                                 const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                 const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                 const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                 const std::unordered_map<LogicalId, ParallelBuffer> &preallocatedBuffers,
                                 bool stopOnFirstValid = true, bool strictCache = false, float minCompileSeconds = 0.0f,
                                 std::shared_ptr<SearchDelegate> delegate = nullptr,
                                 const std::vector<ENodeInfo> &enodeInfos = {})
    {
        auto rootIt = nodeToEClass.find(rootId);
        if (rootIt == nodeToEClass.end())
        {
            Error::throw_err("[Planner.extractBest] Root node missing from nodeToEClass.");
        }
        EClassId rootEClassId = egraph.findConst(rootIt->second);
        if (egraph.getEClass(rootEClassId).enodes.empty())
        {
            Error::throw_err("[Planner.extractBest] Root EClass has no valid ENodes remaining after pruning.");
        }

        const uint64_t numClasses = egraph.getClasses().size();
        LOG(L_INFO) << "numClasses=" << numClasses;

        if (delegate)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;

            uint32_t num_classes = egraph.getClasses().size();
            uint32_t num_enodes = egraph.getENodes().size();

            for (uint32_t i = 0; i < num_classes; ++i)
            {
                const EClass &cls = egraph.getClasses()[i];
                node_features.push_back(1.0f); // is_eclass
                node_features.push_back(0.0f); // is_enode
                node_features.push_back((float)countElements(cls.shape) * getDTypeSize(cls.dtype));
                node_features.push_back((float)cls.dtype);

                for (ENodeId enode_id : cls.enodes)
                {
                    edge_src.push_back(i);
                    edge_dst.push_back(num_classes + enode_id.value);
                }
            }
            for (uint32_t i = 0; i < num_enodes; ++i)
            {
                const ENode &enode = egraph.getENodes()[i];
                node_features.push_back(0.0f); // is_eclass
                node_features.push_back(1.0f); // is_enode
                node_features.push_back(enodeInfos[i].cost);
                node_features.push_back((float)enode.getOpType());

                for (EClassId child : enode.getChildren())
                {
                    edge_src.push_back(num_classes + i);
                    edge_dst.push_back(egraph.findConst(child).value);
                }
            }
            delegate->init_egraph(node_features, edge_src, edge_dst);
        }

        Extractor extractor = Extractor(egraph, rootEClassId, enodeInfos, delegate);
        // extractor.registerValidator(std::make_unique<CycleStepValidator>(egraph));
        extractor.registerValidator(std::make_unique<CycleValidator>(egraph));
        extractor.registerValidator(std::make_unique<MemValidator>(egraph, enodeInfos, mem_caps, eclassToLogical,
                                                                   preallocatedBuffers, delegate));

        float best_cost = TGConstants::INF;
        std::unordered_map<EClassId, uint32_t> best_selection_map;
        std::vector<EClassId> best_order;
        std::vector<ParallelBuffer> best_buffers;
        std::unordered_map<EClassId, BufferId> best_eclass_to_buf;

        int max_iters = 10'000'000;
        int remaining_iters = max_iters;
        ProgressTimer timer(max_iters, "extracting graphs");
        ProgressTimer loopTimer(0, "", true);
        auto start_time = std::chrono::high_resolution_clock::now();
        LOG(L_INFO) << "entering loop";

        while (remaining_iters-- > 0)
        {
#ifdef DEBUG
            std::cout << "loop " << std::to_string(loopTimer.getElapsed() * 1000) << "ms";
            if (max_iters - (remaining_iters + 1) > 0)
            {
                std::cout << ", avg loop "
                          << std::to_string(timer.getElapsed() * 1000 / (max_iters - (remaining_iters + 1))) << "ms";
            }
            std::cout << std::endl;
            loopTimer.reset();
#endif

            if (!extractor.getNextSelection())
            {
                extractor.ascend();
                timer.tick();
                continue;
            }
            LOG(L_DEBUG) << "got selection";

            const std::unordered_map<EClassId, uint32_t> &selection_map = extractor.selection_map;

            bool valid = true;
            std::vector<EClassId> order;
            std::vector<ParallelBuffer> buffers;
            std::unordered_map<EClassId, BufferId> eclass_to_buf;
            float cost = TGConstants::INF;
            std::vector<EClassId> conflict_nodes;

            // 2. Compute dispatch order
            Engine first_engine = Engine{UINT32_MAX, EngineType::CPU};
            bool single_engine = true;
            for (const auto &kv : selection_map)
            {
                uint32_t sel = kv.second;
                ENodeId enode_id = egraph.getEClass(kv.first).enodes[sel];
                const ENode &enode = egraph.getENode(enode_id);
                for (const auto &eng : enode.getEngines())
                {
                    if (first_engine == Engine{UINT32_MAX, EngineType::CPU})
                    {
                        first_engine = eng;
                    }
                    else if (first_engine != eng)
                    {
                        single_engine = false;
                        break;
                    }
                }
                if (!single_engine)
                    break;
            }

            DispatchIterator dispatch_iterator(egraph, selection_map, enodeInfos);
            while (dispatch_iterator.getNextDispatchOrder(selection_map, order))
            {
                valid = extractor.validate(selection_map, order, buffers, eclass_to_buf, cost, conflict_nodes);
                if (!valid)
                    break;
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_selection_map = selection_map;
                    best_order = order;
                    best_buffers = buffers;
                    best_eclass_to_buf = eclass_to_buf;
                    LOG(L_INFO) << "new best cost " << best_cost;
                }
                if (stopOnFirstValid || single_engine)
                {
                    break;
                }
            }

            if (extractor.active_options == 0)
            {
                std::cout << "finished extracting: no more graphs" << std::endl;
                break;
            }

            if (valid && stopOnFirstValid)
            {
                break;
            }

            if (valid && minCompileSeconds > 0.0f)
            {
                auto current_time = std::chrono::high_resolution_clock::now();
                float elapsed = std::chrono::duration<float>(current_time - start_time).count();
                if (elapsed >= minCompileSeconds)
                {
                    std::cout << "finished extracting: hit minimum required compile time" << std::endl;
                    break;
                }
            }

            if (!valid)
            {
                int max_conflict_path_pos = -1;
                int best_conflict_pos = -1;

                for (EClassId c_node : conflict_nodes)
                {
                    int pos = extractor.path_pos[c_node.value];
                    if (pos != -1)
                    {
                        if (pos > max_conflict_path_pos)
                        {
                            max_conflict_path_pos = pos;
                        }

                        auto sel_it = selection_map.find(c_node);
                        if (sel_it != selection_map.end())
                        {
                            uint32_t sel_idx = sel_it->second;
                            if (sel_idx + 1 < egraph.getEClass(c_node).enodes.size())
                            {
                                if (pos > best_conflict_pos)
                                {
                                    best_conflict_pos = pos;
                                }
                            }
                        }
                    }
                }

                if (best_conflict_pos != -1)
                {
                    extractor.target_backtrack_eclass = extractor.path[best_conflict_pos];
                    LOG(L_DEBUG) << "[Planner.extractBest] [iter " << std::to_string(max_iters - remaining_iters)
                                 << "] backjumping to eclass " << toString(extractor.target_backtrack_eclass)
                                 << " (path index " << best_conflict_pos << ") to resolve OOM.";
                }
                else if (max_conflict_path_pos != -1)
                {
                    extractor.target_backtrack_eclass = extractor.path[max_conflict_path_pos];
                    LOG(L_DEBUG) << "[Planner.extractBest] [iter " << std::to_string(max_iters - remaining_iters)
                                 << "] backtracking to eclass " << toString(extractor.target_backtrack_eclass)
                                 << " (path index " << max_conflict_path_pos << ")";
                }
            }

            extractor.ascend();
            timer.tick();
        }

        if (best_cost == TGConstants::INF)
        {
            Error::throw_err("[Planner.extractBest] no valid extraction found under "
                             "given constraints. try running bench");
        }

        std::unordered_map<EClassId, float> best_eclass_to_cost;
        for (const auto &pair : best_selection_map)
        {
            best_eclass_to_cost[pair.first] =
                enodeInfos[egraph.getEClass(pair.first).enodes[best_selection_map.at(pair.first)].value].cost;
        }
        ExtractionResult result = {best_selection_map, best_order, best_buffers,
                                   best_eclass_to_buf, best_cost,  best_eclass_to_cost};
        std::cout << "best_cost=" << std::to_string(best_cost) << std::endl;

        return result;
    }

    CompiledGraph buildCompiledGraph(LogicalId rootId, const Graph &graph, const EGraph &egraph,
                                     const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                     const ExtractionResult &extraction,
                                     const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                     const std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        CompiledGraph compiled;

        for (EClassId eclass_id : extraction.order)
        {
            const ENode &enode =
                egraph.getENode(egraph.getEClass(eclass_id).enodes[extraction.selection_map.at(eclass_id)]);
            LogicalId logical_id = eclassToLogical.count(eclass_id) ? eclassToLogical.at(eclass_id) : LogicalId();

            OpInstruction inst;
            inst.eclass_id = eclass_id;
            inst.logical_id = logical_id;
            inst.kernel_id = enode.getKernelId();
            for (EClassId child : enode.getChildren())
            {
                inst.children.push_back(egraph.findConst(child));
            }
            inst.inBuffers.resize(inst.children.size());
            for (uint32_t i = 0; i < extraction.buffers.size(); i++)
            {
                if (extraction.buffers[i].id == extraction.eclass_to_buf.at(eclass_id))
                {
                    inst.outBuffer = extraction.buffers[i];
                }
                for (uint32_t j = 0; j < inst.children.size(); j++)
                {
                    if (extraction.buffers[i].id == extraction.eclass_to_buf.at(inst.children[j]))
                    {
                        inst.inBuffers[j] = extraction.buffers[i];
                    }
                }
            }

            if (graph.hasNode(logical_id))
            {
                inst.debugOrigin = graph.getNode(logical_id).debugOrigin;
            }

            if (egraph.constantStaging.count(eclass_id))
            {
                compiled.constantStaging[eclass_id] = egraph.constantStaging.at(eclass_id);
            }

            bool is_view = false;
            const KernelEntry *kernel_ptr = nullptr;
            if (enode.getKernelId().value != 0)
            {
                kernel_ptr = &KernelRegistry::get().getKernel(enode.getKernelId());
                is_view = kernel_ptr->is_view;
            }

            uint64_t final_offset_bytes = inst.outBuffer.offset;
            std::vector<uint64_t> final_strides = enode.getStrides();

            if (is_view && kernel_ptr && kernel_ptr->inferView)
            {
                Graph tempGraph;
                std::vector<TensorNode> dummyInputNodes;

                for (uint32_t i = 0; i < inst.children.size(); i++)
                {
                    EClassId child_id = inst.children[i];
                    const TensorView &childView = compiled.nodeViews.at(child_id);

                    LogicalId fakeId = tempGraph.input(childView.getShape(), childView.dtype, childView.strides);

                    if (egraph.constantStaging.count(child_id))
                    {
                        tempGraph.constantStaging[fakeId] = egraph.constantStaging.at(child_id);
                    }

                    dummyInputNodes.push_back(tempGraph.getNode(fakeId));
                }

                final_offset_bytes = compiled.nodeViews.at(inst.children[0]).offset;

                TensorView dummyOutView(enode.getShape(), final_offset_bytes, enode.getStrides(), enode.getDType());

                kernel_ptr->inferView(dummyInputNodes, dummyOutView, tempGraph);

                final_offset_bytes = dummyOutView.offset;
                final_strides = dummyOutView.strides;
            }

            compiled.nodeViews[eclass_id] =
                TensorView(enode.getShape(), final_offset_bytes, final_strides, enode.getDType());

            if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE)
            {
                compiled.instructions.push_back(inst);
            }
        }

        compiled.nodeCosts = extraction.eclass_to_cost;
        compiled.eclass_to_logical = eclassToLogical;

        return compiled;
    }

    struct BaseEGraphState
    {
        EGraph egraph;
        std::unordered_map<LogicalId, EClassId> nodeToEClass;
        std::unordered_map<EClassId, LogicalId> eclassToLogical;
    };

    BaseEGraphState baseState;
    bool baseStateInitialized = false;

    // Initialize baseState.egraph from graph
    void initBaseEGraph(LogicalId rootId, Graph &graph, const std::vector<LogicalId> &topo, Repo *repo = nullptr)
    {
        if (KernelRegistry::get().nKernels() == 0)
        {
            Error::throw_err("KernelRegistry has 0 registered kernels! "
                             "Did you forget to `#include \"generated/kernels_all.gen.hpp\"` "
                             "in your entry point (e.g. bindings.cpp or main.cpp)?");
        }
        if (baseStateInitialized)
            return;

        inferShapes(topo, graph);

        baseState.nodeToEClass.reserve(graph.nodes.size());

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};
        Engine cpu = Engine{0, EngineType::CPU};

        for (LogicalId nodeId : topo)
        {
            TensorNode &node = graph.getNode(nodeId);
            MemSpace mem_space = ram;
            if (node.opType == OpType::INPUT && graph.getInputDataType(nodeId) == InputDataType::STORAGE)
            {
                mem_space = storage;
            }
            EClassId e_class_id = baseState.egraph.addEClass(node.getShape(), node.strides, node.dtype, mem_space);
            baseState.nodeToEClass[nodeId] = e_class_id;
            if (graph.constantStaging.count(nodeId))
            {
                baseState.egraph.constantStaging[e_class_id] = graph.constantStaging.at(nodeId);
            }
        }

        for (LogicalId nodeId : topo)
        {
            const TensorNode &node = graph.getNode(nodeId);
            EClassId e_class_id = baseState.nodeToEClass[nodeId];

            if (node.opType == OpType::INPUT)
            {
                std::vector<EClassId> children;
                for (LogicalId pid : node.child_ids)
                    children.push_back(baseState.egraph.findConst(baseState.nodeToEClass[pid]));
                ENode enode = ENode(
                    KernelId{0}, node.opType, node.opName, children, node.getShape(), node.strides, node.dtype,
                    graph.getInputDataType(nodeId) == InputDataType::STORAGE ? storage : ram, {cpu}, node.contentHash);
                baseState.egraph.addENode(e_class_id, enode);
                continue;
            }

            std::vector<TensorNode> inputs;
            std::vector<MemSpace> input_mem_spaces;
            for (LogicalId pid : node.child_ids)
            {
                inputs.push_back(graph.getNode(pid));
                EClassId pid_eclass = baseState.egraph.findConst(baseState.nodeToEClass[pid]);
                input_mem_spaces.push_back(baseState.egraph.getEClass(pid_eclass).mem_space);
            }

            bool ignore_in_ms = (node.opType != OpType::COPY_TO);
            std::vector<KernelId> refs =
                KernelRegistry::get().findMatchingKernels(node.opType, node.opName, inputs, node, true, ram,
                                                          input_mem_spaces, {cpu}, false, ignore_in_ms, false, true);

            if (refs.size() == 0)
            {
                Error::throw_err("[Planner.initBaseEGraph] couldn't find any kernels "
                                 "to init EClass " +
                                 toString(e_class_id) + " " + toString(baseState.egraph.getEClass(e_class_id)) +
                                 "\nNode " + toString(node, graph));
            }

            bool any_success = false;
            for (KernelId uid : refs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);

                bool path_exists = true;
                std::vector<EClassId> children;

                for (uint64_t i = 0; i < node.child_ids.size(); ++i)
                {
                    LogicalId pid = node.child_ids[i];
                    EClassId p_eclass = baseState.egraph.findConst(baseState.nodeToEClass[pid]);
                    MemSpace src_ms = input_mem_spaces[i];

                    uint64_t ruleIdx = i;
                    if (kernel.min_num_inputs != kernel.max_num_inputs)
                    {
                        ruleIdx = std::min(
                            i, static_cast<uint64_t>(kernel.min_num_inputs > 0 ? kernel.min_num_inputs - 1 : 0));
                    }
                    MemSpace dst_ms = ram;
                    if (!kernel.input_mem_spaces.empty() && ruleIdx < kernel.input_mem_spaces.size())
                    {
                        dst_ms = kernel.input_mem_spaces[ruleIdx];
                    }

                    bool requires_contig = false;
                    if (ruleIdx < kernel.requiresContiguous.size())
                    {
                        requires_contig = kernel.requiresContiguous[ruleIdx];
                    }

                    if (src_ms == dst_ms)
                    {
                        EClassId curr_eclass = p_eclass;
                        EClass curr_cls = baseState.egraph.getEClass(curr_eclass);
                        if (requires_contig && !isContiguous(curr_cls))
                        {
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::CONTIGUOUS, {curr_eclass},
                                                        curr_cls.shape, calcContiguousStrides(curr_cls.shape),
                                                        curr_cls.dtype, curr_cls.mem_space);
                        }
                        children.push_back(curr_eclass);
                    }
                    else
                    {
                        std::vector<std::vector<MemSpace>> paths = findMemSpacePaths(src_ms, dst_ms, inputs[i], {cpu});
                        if (paths.empty())
                        {
                            path_exists = false;
                            break;
                        }
                        const auto &path = paths[0];

                        EClassId curr_eclass = p_eclass;
                        EClass curr_cls = baseState.egraph.getEClass(curr_eclass);

                        if (!isContiguous(curr_cls))
                        {
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::CONTIGUOUS, {curr_eclass},
                                                        curr_cls.shape, calcContiguousStrides(curr_cls.shape),
                                                        curr_cls.dtype, curr_cls.mem_space);
                            curr_cls = baseState.egraph.getEClass(baseState.egraph.findConst(curr_eclass));
                        }

                        for (uint64_t p_idx = 1; p_idx < path.size(); ++p_idx)
                        {
                            MemSpace next_ms = path[p_idx];
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::COPY_TO, {curr_eclass},
                                                        curr_cls.shape, curr_cls.strides, curr_cls.dtype, next_ms);
                        }
                        children.push_back(curr_eclass);
                    }
                }

                if (!path_exists)
                    continue;
                any_success = true;

                std::vector<uint64_t> strides;
                if (kernel.is_view)
                {
                    strides = node.strides;
                }
                else
                {
                    strides = calcContiguousStrides(node.getShape());
                }
                ENode enode =
                    ENode(uid, node.opType, node.opName, children, node.getShape(), strides, node.dtype, ram, {cpu});
                baseState.egraph.addENode(e_class_id, enode);
            }

            if (!any_success)
            {
                Error::throw_err("[Planner.initBaseEGraph] found kernels, but could not route "
                                 "memory spaces to satisfy input constraints for node " +
                                 toString(nodeId) + "\n" + toString(node, graph));
            }
        }

        for (const auto &kv : baseState.nodeToEClass)
        {
            baseState.eclassToLogical[baseState.egraph.findConst(kv.second)] = kv.first;
        }

        baseStateInitialized = true;
    }

    bool injectPartialPath(EGraph &egraph, const Graph &graph, LogicalId logicalId, const std::vector<Region> &regions,
                           const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                           const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                           std::unordered_map<EClassId, LogicalId> &eclassToLogical, bool strictCache = false)
    {
        bool injected = false;
        EClassId E_L = egraph.find(nodeToEClass.at(logicalId));
        const TensorNode &sourceNode = graph.getNode(logicalId);

        bool isFullRegion = false;
        if (regions.size() == 1)
        {
            const Region &reg = regions[0];
            const auto &shape = sourceNode.getShape();
            if (reg.region.size() == shape.size())
            {
                isFullRegion = true;
                for (uint64_t d = 0; d < shape.size(); ++d)
                {
                    if (reg.region[d].start != 0 || reg.region[d].stop != shape[d])
                    {
                        isFullRegion = false;
                        break;
                    }
                }
            }
        }

        if (isFullRegion)
        {
            return false;
        }

        MemSpace ram = MemSpace{1, HandleType::CPP};
        Engine cpu = Engine{0, EngineType::CPU};

        MemSpace target_mem_space = ram;
        auto it = cachedNodes.find(logicalId);
        if (it != cachedNodes.end())
        {
            target_mem_space = it->second;
        }
        else if (strictCache)
        {
            return false;
        }

        const EClass lClass = egraph.getEClass(E_L);

        EClassId E_Cache = egraph.addEClass(lClass.shape, lClass.strides, lClass.dtype, target_mem_space);
        ENode cacheNode(KernelId{0}, OpType::CACHE, "", {}, lClass.shape, lClass.strides, lClass.dtype,
                        target_mem_space, {cpu}, toString(logicalId));
        egraph.addENode(E_Cache, cacheNode);

        eclassToLogical[E_Cache] = logicalId;
        EClassId current_E = E_Cache;

        auto addConst = [&](const std::vector<int32_t> &vals) {
            return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
        };

        for (uint64_t r = 0; r < regions.size(); ++r)
        {
            const Region &recomputeRegion = regions[r];

            std::vector<uint32_t> partialShape;
            for (const Dim &d : recomputeRegion.region)
                partialShape.push_back(d.stop - d.start);

            ShapePropagator prop;
            std::vector<std::vector<Region>> dirtyInputRegions = prop.backward(sourceNode, graph, {recomputeRegion});

            std::vector<int32_t> starts, ends, steps;
            for (const Dim &d : recomputeRegion.region)
            {
                starts.push_back(d.start);
                ends.push_back(d.stop);
                steps.push_back(1);
            }

            EClassId startsId = addConst(starts);
            EClassId endsId = addConst(ends);
            EClassId stepsId = addConst(steps);

            EClassId slicedEClass;

            if (sourceNode.opType == OpType::INPUT)
            {
                std::vector<uint64_t> sliceStrides = lClass.strides;

                for (uint64_t d = 0; d < starts.size(); ++d)
                {
                    int32_t start = starts[d];
                    if (start < 0)
                        start += lClass.shape[d];
                    sliceStrides[d] *= steps[d];
                }

                slicedEClass = egraph.addEClass(partialShape, sliceStrides, lClass.dtype, lClass.mem_space);

                TensorNode dOut;
                dOut.setShape(partialShape);
                dOut.dtype = lClass.dtype;
                std::vector<TensorNode> dIns(4);
                dIns[0].setShape(lClass.shape);
                dIns[0].dtype = lClass.dtype;
                dIns[1].setShape({(uint32_t)starts.size()});
                dIns[1].dtype = DType::INT32;
                dIns[2].setShape({(uint32_t)ends.size()});
                dIns[2].dtype = DType::INT32;
                dIns[3].setShape({(uint32_t)steps.size()});
                dIns[3].dtype = DType::INT32;

                std::vector<MemSpace> input_mem_spaces = {lClass.mem_space, ram, ram, ram};

                auto sliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", dIns, dOut, true,
                                                                           lClass.mem_space, input_mem_spaces, {cpu});
                for (KernelId kid : sliceRefs)
                {
                    ENode sliceNode(kid, OpType::SLICE, "", {E_L, startsId, endsId, stepsId}, partialShape,
                                    sliceStrides, lClass.dtype, lClass.mem_space, {cpu});
                    egraph.addENode(slicedEClass, sliceNode);
                }
            }
            else
            {
                std::vector<EClassId> slicedInputs;
                std::vector<TensorNode> dummyInputNodes;
                std::vector<MemSpace> dummyInputMemSpaces;

                for (uint64_t p_idx = 0; p_idx < sourceNode.child_ids.size(); ++p_idx)
                {
                    LogicalId parentLogicalId = sourceNode.child_ids[p_idx];
                    EClassId E_parent = egraph.find(nodeToEClass.at(parentLogicalId));
                    const EClass pClass = egraph.getEClass(E_parent);

                    std::vector<Region> inputSliceRegions = dirtyInputRegions[p_idx];
                    if (inputSliceRegions.size() != 1)
                    {
                        Error::throw_err("[Planner.injectPartialPath] expected exactly 1 "
                                         "input slice region for parent " +
                                         std::to_string(p_idx) + " but got " +
                                         std::to_string(inputSliceRegions.size()));
                    }
                    Region inputSliceRegion = inputSliceRegions[0];

                    std::vector<uint32_t> pPartialShape;
                    for (const Dim &d : inputSliceRegion.region)
                        pPartialShape.push_back(d.stop - d.start);

                    std::vector<int32_t> pStarts, pEnds, pSteps;
                    for (const Dim &d : inputSliceRegion.region)
                    {
                        pStarts.push_back(d.start);
                        pEnds.push_back(d.stop);
                        pSteps.push_back(1);
                    }

                    EClassId pStartsId = addConst(pStarts);
                    EClassId pEndsId = addConst(pEnds);
                    EClassId pStepsId = addConst(pSteps);

                    std::vector<uint64_t> pSliceStrides = pClass.strides;
                    for (uint64_t d = 0; d < pStarts.size(); ++d)
                    {
                        int32_t start = pStarts[d];
                        if (start < 0)
                            start += pClass.shape[d];
                        pSliceStrides[d] *= pSteps[d];
                    }

                    EClassId pSliceEClass =
                        egraph.addEClass(pPartialShape, pSliceStrides, pClass.dtype, pClass.mem_space);

                    TensorNode pOut;
                    pOut.setShape(pPartialShape);
                    pOut.dtype = pClass.dtype;

                    std::vector<TensorNode> pIns(4);
                    pIns[0].setShape(pClass.shape);
                    pIns[0].dtype = pClass.dtype;
                    pIns[1].setShape({(uint32_t)pStarts.size()});
                    pIns[1].dtype = DType::INT32;
                    pIns[2].setShape({(uint32_t)pEnds.size()});
                    pIns[2].dtype = DType::INT32;
                    pIns[3].setShape({(uint32_t)pSteps.size()});
                    pIns[3].dtype = DType::INT32;

                    std::vector<MemSpace> pSliceInputMemSpaces = {pClass.mem_space, ram, ram, ram};
                    auto pSliceRefs = KernelRegistry::get().findMatchingKernels(
                        OpType::SLICE, "", pIns, pOut, true, pClass.mem_space, pSliceInputMemSpaces, {cpu});

                    for (KernelId uid : pSliceRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        std::vector<uint64_t> strides =
                            kernel.is_view ? pSliceStrides : calcContiguousStrides(pPartialShape);
                        ENode sn(uid, OpType::SLICE, "", {E_parent, pStartsId, pEndsId, pStepsId}, pPartialShape,
                                 strides, pClass.dtype, pClass.mem_space, {cpu});
                        egraph.addENode(pSliceEClass, sn);
                    }

                    EClassId pContigEClass = egraph.addEClass(pPartialShape, calcContiguousStrides(pPartialShape),
                                                              pClass.dtype, pClass.mem_space);

                    TensorNode cOut;
                    cOut.setShape(pPartialShape);
                    cOut.dtype = pClass.dtype;
                    cOut.strides = calcContiguousStrides(pPartialShape);

                    TensorNode cIn;
                    cIn.setShape(pPartialShape);
                    cIn.dtype = pClass.dtype;
                    cIn.strides = pSliceStrides;

                    auto contigRefs = KernelRegistry::get().findMatchingKernels(
                        OpType::CONTIGUOUS, "", {cIn}, cOut, true, pClass.mem_space, {pClass.mem_space}, {cpu});
                    for (KernelId uid : contigRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        std::vector<uint64_t> strides =
                            kernel.is_view ? pSliceStrides : calcContiguousStrides(pPartialShape);
                        ENode cn(uid, OpType::CONTIGUOUS, "", {pSliceEClass}, pPartialShape, strides, pClass.dtype,
                                 pClass.mem_space, {cpu});
                        egraph.addENode(pContigEClass, cn);
                    }

                    slicedInputs.push_back(pContigEClass);

                    TensorNode dummyIn;
                    dummyIn.opType = OpType::INPUT;
                    dummyIn.setShape(pPartialShape);
                    dummyIn.dtype = pClass.dtype;
                    dummyIn.strides = calcContiguousStrides(pPartialShape);
                    dummyInputNodes.push_back(dummyIn);
                    dummyInputMemSpaces.push_back(pClass.mem_space);
                }

                TensorNode dummyOut;
                dummyOut.opType = sourceNode.opType;
                dummyOut.opName = sourceNode.opName;
                dummyOut.setShape(partialShape);
                dummyOut.dtype = sourceNode.dtype;
                dummyOut.strides = calcContiguousStrides(partialShape);

                auto opRefs = KernelRegistry::get().findMatchingKernels(sourceNode.opType, sourceNode.opName,
                                                                        dummyInputNodes, dummyOut, true,
                                                                        target_mem_space, dummyInputMemSpaces, {cpu});
                if (opRefs.size() == 0)
                {
                    Error::throw_err("[Planner.injectPartialPath] couldn't find any "
                                     "slice kernels for op " +
                                     toString(sourceNode.opType));
                }

                slicedEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), sourceNode.dtype,
                                                target_mem_space);
                for (KernelId uid : opRefs)
                {
                    ENode sn(uid, sourceNode.opType, sourceNode.opName, slicedInputs, partialShape,
                             calcContiguousStrides(partialShape), sourceNode.dtype, target_mem_space, {cpu});
                    egraph.addENode(slicedEClass, sn);
                }
            }

            EClassId contigEClass =
                egraph.addEClass(partialShape, calcContiguousStrides(partialShape), sourceNode.dtype, target_mem_space);

            TensorNode cOut;
            cOut.setShape(partialShape);
            cOut.dtype = sourceNode.dtype;
            cOut.strides = calcContiguousStrides(partialShape);

            TensorNode cIn;
            cIn.setShape(partialShape);
            cIn.dtype = sourceNode.dtype;
            cIn.strides = calcContiguousStrides(partialShape);

            auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", {cIn}, cOut, true,
                                                                        target_mem_space, {target_mem_space}, {cpu});
            for (KernelId uid : contigRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<uint64_t> strides = kernel.is_view ? cIn.strides : calcContiguousStrides(partialShape);
                ENode cn(uid, OpType::CONTIGUOUS, "", {slicedEClass}, partialShape, strides, sourceNode.dtype,
                         target_mem_space, {cpu});
                egraph.addENode(contigEClass, cn);
            }

            EClassId scatterEClass = egraph.addEClass(lClass.shape, lClass.strides, lClass.dtype, target_mem_space);

            TensorNode sOut;
            sOut.setShape(lClass.shape);
            sOut.dtype = lClass.dtype;

            std::vector<TensorNode> sIns(5);
            sIns[0].setShape(egraph.getEClass(current_E).shape);
            sIns[0].dtype = lClass.dtype;
            sIns[1].setShape(partialShape);
            sIns[1].dtype = lClass.dtype;
            sIns[2].setShape({(uint32_t)starts.size()});
            sIns[2].dtype = DType::INT32;
            sIns[3].setShape({(uint32_t)ends.size()});
            sIns[3].dtype = DType::INT32;
            sIns[4].setShape({(uint32_t)steps.size()});
            sIns[4].dtype = DType::INT32;

            std::vector<MemSpace> scatterInputSpaces = {target_mem_space, target_mem_space, ram, ram, ram};

            auto scatterRefs = KernelRegistry::get().findMatchingKernels(OpType::SCATTER, "", sIns, sOut, true,
                                                                         target_mem_space, scatterInputSpaces, {cpu});
            for (KernelId uid : scatterRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<uint64_t> strides =
                    (kernel.is_view) ? lClass.strides : calcContiguousStrides(lClass.shape); // TODO check inplace based
                                                                                             // on bufferization stuff
                ENode sn(uid, OpType::SCATTER, "", {current_E, contigEClass, startsId, endsId, stepsId}, lClass.shape,
                         strides, lClass.dtype, target_mem_space, {cpu});
                egraph.addENode(scatterEClass, sn);
            }

            current_E = scatterEClass;
        }

        egraph.merge(E_L, current_E);
        eclassToLogical[egraph.find(E_L)] = logicalId;
        injected = true;
        return injected;
    }

    bool injectInputPartialPaths(EGraph &egraph, const Graph &graph,
                                 const std::unordered_map<LogicalId, std::vector<Region>> &dirtyOutputRegions,
                                 const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                 const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                 std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        bool injected = false;
        for (const auto &kv : dirtyOutputRegions)
        {
            LogicalId nodeId = kv.first;
            if (!graph.hasNode(nodeId))
                continue;
            if (nodeToEClass.find(nodeId) == nodeToEClass.end())
                continue;

            const TensorNode &node = graph.getNode(nodeId);
            if (node.opType == OpType::INPUT && graph.constantStaging.count(nodeId) == 0)
            {
                if (!kv.second.empty())
                {
                    injected = injected || injectPartialPath(egraph, graph, nodeId, kv.second, cachedNodes,
                                                             nodeToEClass, eclassToLogical);
                }
            }
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

    bool injectOutputPartialPaths(EGraph &egraph, const Graph &graph, LogicalId rootId,
                                  const std::vector<Region> &outputNeeded,
                                  const std::unordered_map<LogicalId, MemSpace> &cachedNodes,
                                  const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                  std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        bool injected = false;
        if (!outputNeeded.empty() && nodeToEClass.find(rootId) != nodeToEClass.end())
        {
            injected =
                injectPartialPath(egraph, graph, rootId, outputNeeded, cachedNodes, nodeToEClass, eclassToLogical);
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

    Planner(CostModel &costModel, const std::unordered_map<MemSpace, uint64_t> &mem_caps)
        : costModel(costModel), mem_caps(mem_caps)
    {
    }

    CompiledGraph plan(LogicalId rootId, const Graph &graph, const Bucket &bucket,
                       const std::unordered_map<LogicalId, MemSpace> &cachedNodes, bool doSaturate = true,
                       bool strictCache = false, Repo *repo = nullptr,
                       const std::unordered_map<LogicalId, ParallelBuffer> &preallocatedBuffers = {},
                       float minCompileSeconds = 0.0f, std::shared_ptr<SearchDelegate> delegate = nullptr)
    {
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph tempGraph = graph;
        initBaseEGraph(rootId, tempGraph, topo, repo);

        EGraph egraph = baseState.egraph;
        auto eclassToLogical = baseState.eclassToLogical;

        std::unordered_map<LogicalId, bool> logicalDirty;
        for (LogicalId nodeId : topo)
        {
            if (bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty())
            {
                logicalDirty[nodeId] = true;
            }
            else
            {
                bool isDirty = false;
                for (LogicalId pid : graph.getNode(nodeId).child_ids)
                {
                    if (logicalDirty[pid])
                    {
                        isDirty = true;
                        break;
                    }
                }
                logicalDirty[nodeId] = isDirty;
            }
        }

        // Add cache enodes
        Engine cpu = Engine{0, EngineType::CPU};
        for (const auto &cls : egraph.getClasses())
        {
            EClassId canonId = egraph.find(cls.id);
            if (canonId != cls.id)
                continue;
            if (strictCache)
            {
                if (eclassToLogical.count(canonId) == 0)
                    continue;
                if (cachedNodes.count(eclassToLogical.at(canonId)) == 0)
                    continue;
            }
            for (int i = 0; i < cls.enodes.size(); i++)
            {
                if (egraph.getENode(cls.enodes[i]).getOpType() == OpType::CACHE)
                {
                    continue;
                }
            }

            LogicalId logicalId;
            auto it = eclassToLogical.find(canonId);
            if (it != eclassToLogical.end())
            {
                logicalId = it->second;
            }

            if (logicalId != LogicalId{UINT32_MAX} && !logicalDirty[logicalId])
            {
                ENode cacheNode = ENode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides, cls.dtype,
                                        cls.mem_space, {cpu}, toString(logicalId));
                egraph.addENode(canonId, cacheNode);
            }
        }

        std::unordered_set<EClassId> protectedEClasses;
        for (const auto &kv : cachedNodes)
        {
            LogicalId logicalId = kv.first;
            if (baseState.nodeToEClass.count(logicalId))
            {
                protectedEClasses.insert(egraph.find(baseState.nodeToEClass.at(logicalId)));
            }
        }

        bool dirtyInjected = injectInputPartialPaths(egraph, graph, bucket.inputDirtyRegions, cachedNodes,
                                                     baseState.nodeToEClass, eclassToLogical);

        bool neededInjected = injectOutputPartialPaths(egraph, graph, rootId, bucket.outputNeededRegion, cachedNodes,
                                                       baseState.nodeToEClass, eclassToLogical);

        if (doSaturate)
        {
            saturate(egraph, protectedEClasses, eclassToLogical, true, false, repo);
        }

        bool injected = dirtyInjected || neededInjected;

        std::unordered_map<EClassId, LogicalId> updatedEClassToLogical;
        for (const auto &kv : eclassToLogical)
        {
            updatedEClassToLogical[egraph.find(kv.first)] = kv.second;
        }
        eclassToLogical = std::move(updatedEClassToLogical);

        const std::vector<ENodeInfo> enodeInfos = computeENodeInfos(egraph, eclassToLogical, cachedNodes, strictCache);
        pruneEGraph(egraph, enodeInfos);

        auto extraction = extractBest(rootId, graph, egraph, baseState.nodeToEClass, cachedNodes, eclassToLogical,
                                      preallocatedBuffers, minCompileSeconds == 0.0f, strictCache, minCompileSeconds,
                                      delegate, enodeInfos);
        return buildCompiledGraph(rootId, graph, egraph, baseState.nodeToEClass, extraction, cachedNodes,
                                  eclassToLogical);
    }
};