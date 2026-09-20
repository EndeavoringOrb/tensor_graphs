#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/plan/extractor.hpp"
#include "core/plan/mem.hpp"

enum class DecisionPhase : uint8_t
{
    EXTRACT = 0,
    DISPATCH = 1,
    BUFFERIZE = 2,
    MALLOC = 3
};

struct Decision
{
    DecisionPhase phase = DecisionPhase::EXTRACT;
    uint32_t target_id = 0;
    int64_t choice = 0;

    bool operator==(const Decision &other) const
    {
        return phase == other.phase && target_id == other.target_id && choice == other.choice;
    }

    bool operator!=(const Decision &other) const
    {
        return !(*this == other);
    }
};

using DecisionPath = std::vector<Decision>;

struct DecisionPathNode;
using DecisionPathRef = std::shared_ptr<const DecisionPathNode>;

struct DecisionPathNode
{
    Decision decision;
    DecisionPathRef parent;
    size_t depth = 0;
};

// Mutable, phase-independent search state.  The planner keeps one instance and
// changes branches by undoing decision deltas, then replaying the target path.
// The heavyweight graph and problem inputs remain borrowed references.
class SearchState
{
  public:
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<EClassId, LogicalId> &eclassToLogical;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;
    const std::unordered_set<EClassId> *cachedEClasses;
    const std::unordered_set<EClassId> *cleanEClasses;
    EClassId rootEClassId;
    const float *best_cost_ptr;
    float selected_operation_lower_bound = 0.0f;

    prune::PruningRuleSet<InfiniteCostSkipRule, CachedENodeValidityRule, MissingCachedEClassRule,
                          ExtractorCycleStepRule, ExtractorJacksonCarlierRule>
        extract_rules;
    prune::PruningRuleSet<InputDispatchDominationRule, MemoryPressureDispatchRule, DispatchCostPruningRule,
                          DispatchCycleRule>
        dispatch_rules;
    prune::PruningRuleSet<MemSpaceMismatchInplaceRule, CommutativeInplaceSymmetryRule, PeakMemoryPruningRule,
                          BufferizeCostPruningRule>
        bufferize_rules;
    prune::PruningRuleSet<CapRespectRule, OffsetMonotoneRule, IdMaxSymmetryRule, HMinBoundRule> malloc_rules;

    DecisionPath active_path;
    DecisionPathRef active_path_ref;

    std::unordered_map<EClassId, uint32_t> selection_map;
    std::vector<EClassId> extract_path;
    std::vector<EClassId> to_process;
    bool extraction_leaf_validated = false;
    bool extraction_leaf_rejected = false;

    std::vector<EClassId> ordered;
    std::vector<EClassId> current_ready;
    std::vector<int32_t> current_in_degree;
    std::vector<std::vector<EClassId>> dependents;
    std::vector<std::vector<EClassId>> dispatch_added_ready_trail;

    uint32_t k_buf = 0;
    std::vector<int> bufferize_choices;
    std::unordered_map<EClassId, EClassId> inplace_alias;
    std::unordered_map<EClassId, uint32_t> birth_times;
    std::unordered_map<EClassId, uint32_t> death_times;
    std::vector<std::vector<int>> valid_inplace_choices;

    std::vector<ParallelBuffer> unallocated_buffers;
    std::unordered_map<EClassId, BufferId> eclass_to_buf;
    std::unordered_map<MemSpace, std::vector<ParallelBuffer>> memspace_buffers;
    bool buffers_materialized = false;
    uint32_t k_malloc = 0;
    std::vector<ParallelBuffer> allocated_buffers;

    struct MallocTrailEntry
    {
        int buf_idx = 0;
        int64_t old_offset = 0;
    };
    std::vector<std::vector<MallocTrailEntry>> malloc_undo_trail;

  private:
    struct UndoFrame
    {
        Decision decision;
        float old_selected_operation_lower_bound = 0.0f;
        DecisionPathRef old_path_ref;
        size_t ready_index = 0;
        std::vector<EClassId> added_ready;
        std::vector<std::pair<EClassId, int32_t>> degree_changes;
        EClassId aliased_eclass{UINT32_MAX};
        EClassId old_alias{UINT32_MAX};
        bool had_alias = false;
        int64_t old_offset = 0;
    };
    std::vector<UndoFrame> undo_stack;

    void initExtractRules()
    {
        ExtractContext ctx{egraph, enodeInfos, selection_map, extract_path, EClassId{UINT32_MAX}, 0,
                           &to_process, best_cost_ptr, &mem_caps, cachedEClasses, cleanEClasses};
        extract_rules.init(ctx);
    }

    void initDispatch()
    {
        const uint32_t class_count = static_cast<uint32_t>(egraph.getClasses().size());
        current_in_degree.assign(class_count, 0);
        dependents.assign(class_count, {});
        current_ready.clear();

        std::vector<uint8_t> in_selection(class_count, 0);
        for (const auto &entry : selection_map)
        {
            EClassId node = egraph.findConst(entry.first);
            if (node.value < class_count)
                in_selection[node.value] = 1;
        }
        for (const auto &entry : selection_map)
        {
            EClassId node = egraph.findConst(entry.first);
            if (node.value >= class_count)
                continue;
            const ENode &enode = egraph.getENode(egraph.getEClass(node).enodes[entry.second]);
            std::vector<EClassId> children;
            for (EClassId child : enode.getChildren())
            {
                EClassId canonical = egraph.findConst(child);
                if (canonical != node && canonical.value < class_count && in_selection[canonical.value] &&
                    std::find(children.begin(), children.end(), canonical) == children.end())
                    children.push_back(canonical);
            }
            current_in_degree[node.value] = static_cast<int32_t>(children.size());
            if (children.empty())
                current_ready.push_back(node);
            for (EClassId child : children)
                dependents[child.value].push_back(node);
        }
        DispatchContext ctx{egraph, selection_map, enodeInfos, ordered, current_ready, 0, mem_caps, best_cost_ptr};
        dispatch_rules.init(ctx);
    }

    void clearBufferizationState()
    {
        k_buf = 0;
        bufferize_choices.clear();
        inplace_alias.clear();
        birth_times.clear();
        death_times.clear();
        valid_inplace_choices.clear();
        unallocated_buffers.clear();
        eclass_to_buf.clear();
        memspace_buffers.clear();
        buffers_materialized = false;
        k_malloc = 0;
        allocated_buffers.clear();
        malloc_undo_trail.clear();
    }

    void normalizeExtractProgress()
    {
        to_process.clear();
        if (selection_map.empty())
        {
            to_process.push_back(rootEClassId);
            return;
        }

        for (EClassId eclass : extract_path)
        {
            auto selection = selection_map.find(eclass);
            if (selection == selection_map.end())
                continue;
            const ENode &node = egraph.getENode(egraph.getEClass(eclass).enodes[selection->second]);
            for (auto child = node.getChildren().rbegin(); child != node.getChildren().rend(); ++child)
            {
                EClassId canonical = egraph.findConst(*child);
                if (selection_map.count(canonical) ||
                    std::find(to_process.begin(), to_process.end(), canonical) != to_process.end())
                    continue;
                to_process.push_back(canonical);
            }
        }
    }

    void initBufferize()
    {
        birth_times.clear();
        death_times.clear();
        valid_inplace_choices.assign(ordered.size(), {});
        bufferize_choices.assign(ordered.size(), -1);
        inplace_alias.clear();

        for (uint32_t i = 0; i < ordered.size(); ++i)
        {
            EClassId eclass = ordered[i];
            birth_times[eclass] = i;
            death_times[eclass] = i + 1;
            auto selected = selection_map.find(eclass);
            if (selected == selection_map.end())
                continue;
            const ENode &node = egraph.getENode(egraph.getEClass(eclass).enodes[selected->second]);
            if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
                birth_times[eclass] = 0;
            for (EClassId child : node.getChildren())
            {
                EClassId base = resolve_view_alias(child, egraph, selection_map, enodeInfos);
                death_times[base] = std::max(death_times[base], i);
            }

            // Views do not own storage and therefore do not participate in
            // bufferization choices. Their e-class is mapped to the backing
            // buffer when buffers are materialized below.
            ENodeId node_id = egraph.getEClass(eclass).enodes[selected->second];
            if (enodeInfos[node_id.value].is_view)
                continue;

            valid_inplace_choices[i].push_back(-1);
            if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
                continue;

            if (selected->second < egraph.getEClass(eclass).enodes.size() &&
                node.getKernelId().value != 0 && KernelRegistry::get().hasKernel(node.getKernelId()))
            {
                const KernelEntry &kernel = KernelRegistry::get().getKernel(node.getKernelId());
                for (uint32_t child_index : kernel.safe_inplace_idxs)
                {
                    if (child_index >= node.getChildren().size())
                        continue;
                    EClassId child = egraph.findConst(node.getChildren()[child_index]);
                    EClassId child_base = resolve_view_alias(child, egraph, selection_map, enodeInfos);
                    if (death_times[child_base] != i || !selection_map.count(child_base))
                        continue;
                    const ENode &child_node = egraph.getENode(
                        egraph.getEClass(child_base).enodes[selection_map.at(child_base)]);
                    if (child_node.getOpType() == OpType::INPUT || child_node.getOpType() == OpType::CACHE)
                        continue;
                    if (getSizeBytes(node.getShape(), node.getDType()) <=
                        getSizeBytes(child_node.getShape(), child_node.getDType()))
                        valid_inplace_choices[i].push_back(static_cast<int>(child_index));
                }
            }
        }
        BufferizeContext ctx{ordered, egraph, selection_map, enodeInfos, birth_times, death_times,
                             inplace_alias, {}, 0, mem_caps, best_cost_ptr};
        bufferize_rules.init(ctx);
        normalizeBufferizeProgress();
    }

    EClassId getInplaceAlias(EClassId id) const
    {
        EClassId current = id;
        std::unordered_set<EClassId> visited;
        while (true)
        {
            auto it = inplace_alias.find(current);
            if (it == inplace_alias.end() || visited.count(current))
                return current;
            visited.insert(current);
            current = it->second;
        }
    }

    void normalizeBufferizeProgress()
    {
        while (k_buf < valid_inplace_choices.size() && valid_inplace_choices[k_buf].empty())
            ++k_buf;
        if (k_buf == ordered.size() && !buffers_materialized)
            buildBuffers();
    }

    void buildBuffers()
    {
        if (buffers_materialized)
            return;

        unallocated_buffers.clear();
        eclass_to_buf.clear();

        std::unordered_map<EClassId, uint32_t> active_birth_times = birth_times;
        std::unordered_map<EClassId, uint32_t> active_death_times = death_times;
        for (uint32_t i = 0; i < ordered.size(); ++i)
        {
            EClassId eclass = ordered[i];
            auto selection = selection_map.find(eclass);
            if (selection == selection_map.end())
                continue;
            ENodeId node_id = egraph.getEClass(eclass).enodes[selection->second];
            const ENodeInfo &info = enodeInfos[node_id.value];
            EClassId base = info.is_view ? resolve_view_alias(eclass, egraph, selection_map, enodeInfos) : eclass;
            EClassId target = getInplaceAlias(base);
            if (target != eclass)
            {
                active_birth_times[target] = std::min(active_birth_times[target], active_birth_times[eclass]);
                active_death_times[target] = std::max(active_death_times[target], active_death_times[eclass]);
            }
        }

        std::unordered_map<EClassId, BufferId> base_to_buffer;
        for (EClassId eclass : ordered)
        {
            auto selection = selection_map.find(eclass);
            if (selection == selection_map.end())
                continue;
            ENodeId node_id = egraph.getEClass(eclass).enodes[selection->second];
            EClassId base = enodeInfos[node_id.value].is_view
                                ? resolve_view_alias(eclass, egraph, selection_map, enodeInfos)
                                : eclass;
            EClassId target = getInplaceAlias(base);
            if (!selection_map.count(target))
                target = base;
            if (!selection_map.count(target))
                target = eclass;

            if (!base_to_buffer.count(target))
            {
                auto target_selection = selection_map.find(target);
                if (target_selection == selection_map.end())
                    continue;
                BufferId buffer_id{static_cast<uint32_t>(unallocated_buffers.size())};
                base_to_buffer[target] = buffer_id;
                ENodeId target_node_id = egraph.getEClass(target).enodes[target_selection->second];
                const ENode &target_node = egraph.getENode(target_node_id);
                uint64_t size_bytes = getSizeBytes(target_node.getShape(), target_node.getDType());
                if (size_bytes == 0)
                    Error::throw_err("empty node");
                size_bytes = (size_bytes + 4095) & ~4095ULL;
                uint32_t start = active_birth_times.count(target) ? active_birth_times[target] : 0;
                uint32_t end = active_death_times.count(target) ? active_death_times[target] : 1;
                unallocated_buffers.push_back(
                    {buffer_id, target_node.getMemSpace(), size_bytes, start, end, -1});
            }
            eclass_to_buf[eclass] = base_to_buffer[target];
        }
        buffers_materialized = true;
    }

  public:
    SearchState(const EGraph &graph, EClassId root, const std::vector<ENodeInfo> &infos,
                const std::unordered_map<EClassId, LogicalId> &logical_map,
                const std::unordered_map<MemSpace, uint64_t> &caps,
                const Settings &settings, const float *best_cost = nullptr,
                const std::unordered_set<EClassId> *cached = nullptr,
                const std::unordered_set<EClassId> *clean = nullptr)
        : egraph(graph), enodeInfos(infos), eclassToLogical(logical_map), mem_caps(caps), cachedEClasses(cached),
          cleanEClasses(clean), rootEClassId(root), best_cost_ptr(best_cost),
          extract_rules(prune::instantiate_rules<AllExtractRuleTypes>("extract", settings)),
          dispatch_rules(prune::instantiate_rules<AllDispatchRuleTypes>("dispatch", settings)),
          bufferize_rules(prune::instantiate_rules<AllBufferizeRuleTypes>("bufferize", settings)),
          malloc_rules(prune::PruningRuleSet<CapRespectRule, OffsetMonotoneRule, IdMaxSymmetryRule, HMinBoundRule>(
              CapRespectRule(settings.is_rule_enabled("malloc", "CapRespectRule")),
              OffsetMonotoneRule(settings.is_rule_enabled("malloc", "OffsetMonotoneRule")),
              IdMaxSymmetryRule(settings.is_rule_enabled("malloc", "IdMaxSymmetryRule")),
              HMinBoundRule(settings.is_rule_enabled("malloc", "HMinBoundRule"))))
    {
        to_process.push_back(rootEClassId);
        initExtractRules();
    }

    bool isLeaf()
    {
        prepareNextPhase();
        return to_process.empty() && ordered.size() == selection_map.size() && buffers_materialized &&
               k_buf == ordered.size();
    }

    void prepareNextPhase()
    {
        normalizeExtractProgress();
        if (to_process.empty() && ordered.empty() && !selection_map.empty())
        {
            if (!extraction_leaf_validated)
            {
                ExtractContext ctx{egraph, enodeInfos, selection_map, extract_path, EClassId{UINT32_MAX}, 0,
                                   &to_process, best_cost_ptr, &mem_caps, cachedEClasses, cleanEClasses};
                extraction_leaf_rejected = !extract_rules.validate_leaf(ctx);
                extraction_leaf_validated = true;
            }
            if (!extraction_leaf_rejected && current_in_degree.empty())
                initDispatch();
        }
        if (to_process.empty() && ordered.size() == selection_map.size() && valid_inplace_choices.empty())
            initBufferize();
        if (ordered.size() == selection_map.size() && !valid_inplace_choices.empty())
            normalizeBufferizeProgress();
    }

    void push_decision(const Decision &decision)
    {
        prepareNextPhase();
        UndoFrame frame;
        frame.decision = decision;
        frame.old_path_ref = active_path_ref;
        if (decision.phase == DecisionPhase::EXTRACT)
        {
            EClassId current{decision.target_id};
            uint32_t selected = static_cast<uint32_t>(decision.choice);
            frame.old_selected_operation_lower_bound = selected_operation_lower_bound;
            auto it = std::find(to_process.begin(), to_process.end(), current);
            if (it != to_process.end())
                to_process.erase(it);
            extract_path.push_back(current);
            selection_map[current] = selected;
            float selected_cost = enodeInfos[egraph.getEClass(current).enodes[selected].value].cost;
            if (selected_cost < TGConstants::INF && std::isfinite(selected_cost))
                selected_operation_lower_bound = std::max(selected_operation_lower_bound, selected_cost);
            extraction_leaf_validated = false;
            extraction_leaf_rejected = false;
            ExtractContext ctx{egraph, enodeInfos, selection_map, extract_path, current, selected, &to_process,
                               best_cost_ptr, &mem_caps, cachedEClasses, cleanEClasses};
            extract_rules.on_push(egraph.getEClass(current).enodes[selected], ctx);
            const ENode &node = egraph.getENode(egraph.getEClass(current).enodes[selected]);
            for (auto child = node.getChildren().rbegin(); child != node.getChildren().rend(); ++child)
            {
                EClassId canonical = egraph.findConst(*child);
                if (!selection_map.count(canonical))
                {
                    to_process.push_back(canonical);
                }
            }
            normalizeExtractProgress();
        }
        else if (decision.phase == DecisionPhase::DISPATCH)
        {
            prepareNextPhase();
            EClassId node{static_cast<uint32_t>(decision.choice)};
            auto ready = std::find(current_ready.begin(), current_ready.end(), node);
            if (ready == current_ready.end())
                throw std::runtime_error("invalid dispatch decision");
            frame.ready_index = static_cast<size_t>(ready - current_ready.begin());
            current_ready.erase(ready);
            ordered.push_back(node);
            for (EClassId dependent : dependents[node.value])
            {
                int32_t old_degree = current_in_degree[dependent.value];
                current_in_degree[dependent.value]--;
                frame.degree_changes.push_back({dependent, old_degree});
                if (current_in_degree[dependent.value] == 0)
                {
                    current_ready.push_back(dependent);
                    frame.added_ready.push_back(dependent);
                }
            }
            DispatchContext ctx{egraph, selection_map, enodeInfos, ordered, current_ready,
                                static_cast<uint32_t>(ordered.size() - 1), mem_caps, best_cost_ptr};
            dispatch_rules.on_push(node, ctx);
        }
        else if (decision.phase == DecisionPhase::BUFFERIZE)
        {
            prepareNextPhase();
            uint32_t pos = decision.target_id;
            if (pos >= bufferize_choices.size())
                throw std::runtime_error("invalid bufferization decision");
            int choice = static_cast<int>(decision.choice);
            bufferize_choices[pos] = choice;
            EClassId eclass = ordered[pos];
            BufferizeContext ctx{ordered, egraph, selection_map, enodeInfos, birth_times, death_times,
                                 inplace_alias, valid_inplace_choices[pos], pos, mem_caps, best_cost_ptr};
            bufferize_rules.on_push(choice, ctx);
            if (choice >= 0)
            {
                const ENode &node = egraph.getENode(egraph.getEClass(eclass).enodes[selection_map.at(eclass)]);
                EClassId child = resolve_view_alias(node.getChildren()[choice], egraph, selection_map, enodeInfos);
                frame.aliased_eclass = eclass;
                auto old = inplace_alias.find(eclass);
                if (old != inplace_alias.end())
                {
                    frame.had_alias = true;
                    frame.old_alias = old->second;
                }
                inplace_alias[eclass] = child;
            }
            k_buf = std::max(k_buf, pos + 1);
            normalizeBufferizeProgress();
        }
        else
        {
            uint32_t index = decision.target_id;
            if (index >= unallocated_buffers.size())
                throw std::runtime_error("invalid malloc decision");
            frame.old_offset = unallocated_buffers[index].offset;
            unallocated_buffers[index].offset = decision.choice;
            allocated_buffers.push_back(unallocated_buffers[index]);
            k_malloc = std::max(k_malloc, index + 1);
        }
        active_path.push_back(decision);
        active_path_ref = std::make_shared<DecisionPathNode>(DecisionPathNode{decision, active_path_ref,
                                                                                active_path.size()});
        undo_stack.push_back(std::move(frame));
    }

    void pop_decision()
    {
        if (undo_stack.empty() || active_path.empty())
            return;
        UndoFrame frame = std::move(undo_stack.back());
        undo_stack.pop_back();
        active_path.pop_back();
        active_path_ref = frame.old_path_ref;
        const Decision &decision = frame.decision;

        if (decision.phase == DecisionPhase::EXTRACT)
        {
            EClassId current{decision.target_id};
            const ENodeId node_id = egraph.getEClass(current).enodes[static_cast<uint32_t>(decision.choice)];
            // The original extractor invokes rule pop hooks after removing
            // the node from its active path. Preserve that invariant when
            // replaying queued branches.
            if (!extract_path.empty())
                extract_path.pop_back();
            ExtractContext ctx{egraph, enodeInfos, selection_map, extract_path, current,
                               static_cast<uint32_t>(decision.choice), &to_process, best_cost_ptr, &mem_caps,
                               cachedEClasses, cleanEClasses};
            extract_rules.on_pop(node_id, ctx);
            selection_map.erase(current);
            selected_operation_lower_bound = frame.old_selected_operation_lower_bound;
            normalizeExtractProgress();
            extraction_leaf_validated = false;
            extraction_leaf_rejected = false;
            ordered.clear();
            current_ready.clear();
            current_in_degree.clear();
            dependents.clear();
            dispatch_added_ready_trail.clear();
            clearBufferizationState();
        }
        else if (decision.phase == DecisionPhase::DISPATCH)
        {
            EClassId node{static_cast<uint32_t>(decision.choice)};
            DispatchContext ctx{egraph, selection_map, enodeInfos, ordered, current_ready,
                                static_cast<uint32_t>(ordered.size() - 1), mem_caps, best_cost_ptr};
            dispatch_rules.on_pop(node, ctx);
            for (auto it = frame.added_ready.rbegin(); it != frame.added_ready.rend(); ++it)
            {
                auto ready = std::find(current_ready.begin(), current_ready.end(), *it);
                if (ready != current_ready.end())
                    current_ready.erase(ready);
            }
            for (const auto &[dependent, old_degree] : frame.degree_changes)
                current_in_degree[dependent.value] = old_degree;
            if (!ordered.empty())
                ordered.pop_back();
            current_ready.insert(current_ready.begin() + std::min(frame.ready_index, current_ready.size()), node);
            if (ordered.size() < selection_map.size())
                clearBufferizationState();
        }
        else if (decision.phase == DecisionPhase::BUFFERIZE)
        {
            uint32_t pos = decision.target_id;
            int choice = static_cast<int>(decision.choice);
            BufferizeContext ctx{ordered, egraph, selection_map, enodeInfos, birth_times, death_times,
                                 inplace_alias, valid_inplace_choices[pos], pos, mem_caps, best_cost_ptr};
            bufferize_rules.on_pop(choice, ctx);
            bufferize_choices[pos] = -1;
            if (frame.aliased_eclass.value != UINT32_MAX)
            {
                if (frame.had_alias)
                    inplace_alias[frame.aliased_eclass] = frame.old_alias;
                else
                    inplace_alias.erase(frame.aliased_eclass);
            }
            k_buf = pos;
            unallocated_buffers.clear();
            eclass_to_buf.clear();
            buffers_materialized = false;
            k_malloc = 0;
            allocated_buffers.clear();
            normalizeBufferizeProgress();
        }
        else
        {
            uint32_t index = decision.target_id;
            if (index < unallocated_buffers.size())
                unallocated_buffers[index].offset = frame.old_offset;
            if (!allocated_buffers.empty())
                allocated_buffers.pop_back();
            k_malloc = index;
        }
    }

    void transition_to(const DecisionPath &target_path)
    {
        size_t lca_depth = 0;
        const size_t common = std::min(active_path.size(), target_path.size());
        while (lca_depth < common && active_path[lca_depth] == target_path[lca_depth])
            ++lca_depth;
        while (active_path.size() > lca_depth)
            pop_decision();
        for (size_t i = lca_depth; i < target_path.size(); ++i)
            push_decision(target_path[i]);
    }

    void transition_to(const DecisionPathRef &target_path)
    {
        DecisionPath path;
        for (DecisionPathRef cursor = target_path; cursor; cursor = cursor->parent)
            path.push_back(cursor->decision);
        std::reverse(path.begin(), path.end());
        transition_to(path);
    }
};
