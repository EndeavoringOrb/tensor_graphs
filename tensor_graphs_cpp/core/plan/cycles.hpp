#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "core/egraph.hpp"
#include "core/logging.hpp"
#include "core/ops/ops.hpp"
#include "core/types.hpp"

struct CycleEdge
{
    uint32_t from;
    uint32_t to;
    ENodeId enode;
    uint32_t childIndex;
};

// Flat (CSR) per-class edge lists; `adjacency[c]` is the range of c's edges in
// insertion order.
struct CycleAdjacency
{
    struct Range
    {
        const CycleEdge *first = nullptr;
        const CycleEdge *last = nullptr;
        const CycleEdge *begin() const { return first; }
        const CycleEdge *end() const { return last; }
        bool empty() const { return first == last; }
    };
    std::vector<uint32_t> offsets;
    std::vector<CycleEdge> edges;
    Range operator[](uint32_t class_id) const
    {
        return Range{edges.data() + offsets[class_id], edges.data() + offsets[class_id + 1]};
    }
};

inline std::vector<std::vector<uint32_t>> findCyclicComponents(
    const EGraph &egraph,
    CycleAdjacency &outgoing,
    CycleAdjacency &incoming,
    std::vector<uint8_t> &canonical)
{
    const uint32_t class_count = static_cast<uint32_t>(egraph.getNumClasses());
    canonical.assign(class_count, 0);

    for (uint32_t class_id = 0; class_id < class_count; ++class_id)
        canonical[class_id] = egraph.findConst(EClassId{class_id}).value == class_id;

    // Edges in class order, then enode order, then child order; `incoming`
    // keeps the same relative order per child.
    outgoing.offsets.assign(class_count + 1, 0);
    outgoing.edges.clear();
    std::vector<uint32_t> in_count(class_count + 1, 0);
    for (uint32_t class_id = 0; class_id < class_count; ++class_id)
    {
        outgoing.offsets[class_id] = static_cast<uint32_t>(outgoing.edges.size());
        if (!canonical[class_id])
            continue;
        const EClass &eclass = egraph.getEClass(EClassId{class_id});
        for (ENodeId enode_id : eclass.enodes)
        {
            const ENode &enode = egraph.getENode(enode_id);
            const auto &children = enode.getChildren();
            for (uint32_t child_index = 0; child_index < children.size(); ++child_index)
            {
                const uint32_t child_id = egraph.findConst(children[child_index]).value;
                if (child_id >= class_count || !canonical[child_id])
                    continue;
                outgoing.edges.push_back(CycleEdge{class_id, child_id, enode_id, child_index});
                ++in_count[child_id + 1];
            }
        }
    }
    outgoing.offsets[class_count] = static_cast<uint32_t>(outgoing.edges.size());
    for (uint32_t c = 0; c < class_count; ++c)
        in_count[c + 1] += in_count[c];
    incoming.offsets = in_count;
    incoming.edges.resize(outgoing.edges.size());
    {
        std::vector<uint32_t> cursor(in_count.begin(), in_count.end() - 1);
        for (const CycleEdge &edge : outgoing.edges)
            incoming.edges[cursor[edge.to]++] = edge;
    }

    std::vector<int32_t> index(class_count, -1);
    std::vector<int32_t> low_link(class_count, -1);
    std::vector<uint8_t> on_stack(class_count, 0);
    std::vector<uint32_t> stack;
    std::vector<std::vector<uint32_t>> components;
    int32_t next_index = 0;

    // Iterative Tarjan; visits children in the same order as the recursive form.
    std::vector<std::pair<uint32_t, uint32_t>> frames; // (node, next outgoing edge offset)
    auto open = [&](uint32_t node) {
        index[node] = next_index;
        low_link[node] = next_index++;
        stack.push_back(node);
        on_stack[node] = 1;
        frames.push_back({node, outgoing.offsets[node]});
    };
    for (uint32_t root = 0; root < class_count; ++root)
    {
        if (!canonical[root] || index[root] != -1)
            continue;
        open(root);
        while (!frames.empty())
        {
            auto &[node, cursor] = frames.back();
            if (cursor < outgoing.offsets[node + 1])
            {
                const uint32_t child = outgoing.edges[cursor++].to;
                if (index[child] == -1)
                    open(child);
                else if (on_stack[child])
                    low_link[node] = std::min(low_link[node], index[child]);
                continue;
            }
            const uint32_t done = node;
            frames.pop_back();
            if (low_link[done] == index[done])
            {
                std::vector<uint32_t> component;
                while (true)
                {
                    const uint32_t member = stack.back();
                    stack.pop_back();
                    on_stack[member] = 0;
                    component.push_back(member);
                    if (member == done)
                        break;
                }
                components.push_back(std::move(component));
            }
            if (!frames.empty())
            {
                const uint32_t parent = frames.back().first;
                low_link[parent] = std::min(low_link[parent], low_link[done]);
            }
        }
    }

    std::vector<std::vector<uint32_t>> cyclic_components;
    for (auto &component : components)
    {
        bool cyclic = component.size() > 1;
        if (!cyclic)
        {
            const uint32_t node = component.front();
            for (const CycleEdge &edge : outgoing[node])
                cyclic = cyclic || edge.to == node;
        }
        if (!cyclic)
            continue;
        std::sort(component.begin(), component.end());
        cyclic_components.push_back(std::move(component));
    }

    std::sort(cyclic_components.begin(), cyclic_components.end(),
              [](const auto &left, const auto &right) { return left.front() < right.front(); });

    return cyclic_components;
}

inline uint32_t countCyclicComponents(const EGraph &egraph)
{
    CycleAdjacency outgoing, incoming;
    std::vector<uint8_t> canonical;
    return static_cast<uint32_t>(findCyclicComponents(egraph, outgoing, incoming, canonical).size());
}

inline void printEGraphCycles(const EGraph &egraph, std::ostream &output)
{
    CycleAdjacency outgoing, incoming;
    std::vector<uint8_t> canonical;
    auto cyclic_components = findCyclicComponents(egraph, outgoing, incoming, canonical);

    output << "[EGraph cycles] Tarjan found " << cyclic_components.size() << " cyclic component(s) across "
           << std::count(canonical.begin(), canonical.end(), static_cast<uint8_t>(1))
           << " canonical e-class(es).\n";

    auto print_edge = [&](const CycleEdge &edge)
    {
        const ENode &enode = egraph.getENode(edge.enode);
        output << "      EClass " << edge.from << " --ENode " << edge.enode.value << " ["
               << toString(enode.getOpType());
        if (!enode.getOpName().empty())
            output << " (" << enode.getOpName() << ")";
        output << ", kernel=" << toString(enode.getKernelId())
               << ", child[" << edge.childIndex << "]--> EClass " << edge.to
               << " | shape=" << toString(enode.getShape())
               << ", dtype=" << toString(enode.getDType());
        if (!enode.getDebugOrigin().empty())
            output << ", debugOrigin=" << enode.getDebugOrigin();
        output << "\n";
    };

    for (uint32_t component_index = 0; component_index < cyclic_components.size(); ++component_index)
    {
        const auto &component = cyclic_components[component_index];
        output << "[EGraph cycles] Component " << component_index << " (" << component.size()
               << " e-classes): [";
        for (uint32_t i = 0; i < component.size(); ++i)
            output << component[i] << (i + 1 == component.size() ? "]\n" : ", ");

        for (uint32_t class_id : component)
        {
            const EClass &eclass = egraph.getEClass(EClassId{class_id});
            output << "  EClass " << class_id << " (base=" << eclass.base_eclass_id.value
                   << ", shape=" << toString(eclass.shape) << ", strides=" << toString(eclass.strides)
                   << ", dtype=" << toString(eclass.dtype) << ", mem_space=" << toString(eclass.mem_space)
                   << ")\n";
            output << "    Incoming edges (all inputs):\n";
            if (incoming[class_id].empty())
                output << "      <none>\n";
            for (const CycleEdge &edge : incoming[class_id])
                print_edge(edge);
            output << "    Outgoing edges (all outputs):\n";
            if (outgoing[class_id].empty())
                output << "      <none>\n";
            for (const CycleEdge &edge : outgoing[class_id])
                print_edge(edge);
        }
    }
}

// A plan selects one enode per class. If class A has an enode with a child B in
// A's cyclic component and B cannot be derived at all once A is forbidden, every
// derivation of B reaches A again, so that enode is only ever selected inside a
// cycle. Removing it keeps every acyclic plan. Classes outside the component
// count as derivable, which only removes less. Subsumes direct self-references.
inline uint32_t removeDominatedCycleENodes(EGraph &egraph)
{
    uint32_t total_removed = 0;
    std::vector<int32_t> local_of;
    while (true)
    {
        CycleAdjacency outgoing, incoming;
        std::vector<uint8_t> canonical;
        auto cyclic_components = findCyclicComponents(egraph, outgoing, incoming, canonical);
        local_of.assign(egraph.getNumClasses(), -1);

        uint32_t iter_removed = 0;
        for (const auto &component : cyclic_components)
        {
            const uint32_t k = static_cast<uint32_t>(component.size());
            for (uint32_t i = 0; i < k; ++i)
                local_of[component[i]] = static_cast<int32_t>(i);

            // Flattened enodes of the component with their distinct internal children.
            struct LocalENode
            {
                uint32_t owner;
                std::vector<uint32_t> internal_children;
            };
            std::vector<LocalENode> enodes;
            std::vector<std::vector<uint32_t>> enodes_of(k), readers_of(k);
            for (uint32_t i = 0; i < k; ++i)
                for (ENodeId enode_id : egraph.getEClass(EClassId{component[i]}).enodes)
                {
                    LocalENode local{i, {}};
                    for (EClassId child : egraph.getENode(enode_id).getChildren())
                    {
                        const int32_t c = local_of[egraph.findConst(child).value];
                        if (c >= 0 && std::find(local.internal_children.begin(), local.internal_children.end(),
                                                static_cast<uint32_t>(c)) == local.internal_children.end())
                            local.internal_children.push_back(static_cast<uint32_t>(c));
                    }
                    const uint32_t e = static_cast<uint32_t>(enodes.size());
                    enodes_of[i].push_back(e);
                    for (uint32_t c : local.internal_children)
                        readers_of[c].push_back(e);
                    enodes.push_back(std::move(local));
                }

            std::vector<std::vector<uint8_t>> removed(k);
            std::vector<uint8_t> derivable(k);
            std::vector<uint32_t> missing(enodes.size());
            std::vector<uint32_t> worklist;
            for (uint32_t forbidden = 0; forbidden < k; ++forbidden)
            {
                // Least fixpoint: a class is derivable without `forbidden` once
                // one of its enodes has every internal child derivable.
                std::fill(derivable.begin(), derivable.end(), 0);
                worklist.clear();
                for (uint32_t e = 0; e < enodes.size(); ++e)
                {
                    missing[e] = static_cast<uint32_t>(enodes[e].internal_children.size());
                    const uint32_t owner = enodes[e].owner;
                    if (missing[e] == 0 && owner != forbidden && !derivable[owner])
                    {
                        derivable[owner] = 1;
                        worklist.push_back(owner);
                    }
                }
                while (!worklist.empty())
                {
                    const uint32_t c = worklist.back();
                    worklist.pop_back();
                    for (uint32_t e : readers_of[c])
                    {
                        const uint32_t owner = enodes[e].owner;
                        if (--missing[e] == 0 && owner != forbidden && !derivable[owner])
                        {
                            derivable[owner] = 1;
                            worklist.push_back(owner);
                        }
                    }
                }
                removed[forbidden].assign(enodes_of[forbidden].size(), 0);
                for (uint32_t j = 0; j < enodes_of[forbidden].size(); ++j)
                    for (uint32_t c : enodes[enodes_of[forbidden][j]].internal_children)
                        if (c == forbidden || !derivable[c])
                            removed[forbidden][j] = 1;
            }

            // Every removal is justified against the same snapshot, so applying
            // them together is exact.
            for (uint32_t i = 0; i < k; ++i)
            {
                EClass &eclass = egraph.getEClass(EClassId{component[i]});
                if (eclass.enodes.size() != removed[i].size())
                    Error::throw_err("removeDominatedCycleENodes: enode count changed while pruning");
                std::vector<ENodeId> kept;
                kept.reserve(eclass.enodes.size());
                for (uint32_t j = 0; j < eclass.enodes.size(); ++j)
                {
                    if (removed[i][j])
                        ++iter_removed;
                    else
                        kept.push_back(eclass.enodes[j]);
                }
                eclass.enodes = std::move(kept);
            }
            for (uint32_t member : component)
                local_of[member] = -1;
        }

        if (iter_removed == 0)
            break;
        total_removed += iter_removed;
    }
    return total_removed;
}

struct CycleStripPass
{
    const char *name;
    const char *description;
    uint32_t (*run)(EGraph &);
};

// Every pass registered here must keep every acyclic plan;
// tests/test_planner_oracle checks each one alone and the whole pipeline.
// Removed after tests/planner_oracle_cases/cycle_strip_* showed they drop
// optimal plans: single-external connection (an internal class with a leaf
// producer, or the root inside the component), single-external consumer and
// single-external compute producer. Direct self-reference and single-external
// input were exact but are subsumed by dominated_child.
inline const std::vector<CycleStripPass> &cycleStripPasses()
{
    static const std::vector<CycleStripPass> passes{
        {"dominated_child", "enode(s) whose child needs the enode's own class", &removeDominatedCycleENodes},
    };
    return passes;
}
