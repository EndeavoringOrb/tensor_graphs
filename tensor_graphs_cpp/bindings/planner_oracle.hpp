#pragma once

#include <json.hpp>
#include "core/plan/cycles.hpp"
#include "core/plan/search_engine.hpp"

// Adapter only: the independent specification and enumeration live in Python.
// This builds small synthetic egraphs and invokes the production search stack.
namespace planner_oracle
{
using json = nlohmann::json;
using namespace plan;

inline Domain makeDomain(const json &values)
{
    const auto numbers = values.get<std::vector<int32_t>>();
    if (numbers.empty())
        Error::throw_err("Oracle adapter received an empty domain");
    const int32_t lo = numbers.front();
    const int32_t hi = numbers.back();
    if (static_cast<int64_t>(hi) - lo + 1 == static_cast<int64_t>(numbers.size()))
        return Domain::makeRange(lo, hi);
    if (static_cast<int64_t>(hi) - lo >= 32)
        Error::throw_err("Oracle sparse domains must fit within 32 values");
    uint32_t mask = 0;
    for (int32_t value : numbers)
        mask |= uint32_t{1} << (value - lo);
    return Domain::makeMask(mask, lo);
}

inline json domainValues(const Domain &domain)
{
    json values = json::array();
    if (!domain.isEmpty())
        for (int64_t value = domain.getMin(); value <= domain.getMax(); ++value)
            if (domain.contains(static_cast<int32_t>(value)))
                values.push_back(value);
    return values;
}

inline SearchState makeState(const json &problem)
{
    SearchState state;
    const MemSpace space{1, HandleType::CPP};
    state.page_alignments[space] = 4;
    state.mem_caps[space] = problem.at("capacity_pages").get<uint64_t>() * 4;
    for (const auto &cache : problem.at("caches"))
    {
        VarInfo info;
        info.type = VarType::CACHED;
        info.base_eclass_id = BaseEClassId{cache.at("base_id").get<uint32_t>()};
        info.mem_space = space;
        info.size_bytes = cache.at("size_pages").get<uint64_t>() * 4;
        info.name = "cached_" + std::to_string(info.base_eclass_id.value);
        state.cached_vars[info.base_eclass_id] = state.addVar(info, makeDomain(cache.at("domain")));
        CacheCandidate candidate;
        candidate.base_eclass_id = info.base_eclass_id;
        candidate.size_bytes = info.size_bytes;
        candidate.mem_space = space;
        state.candidates.push_back(candidate);
    }
    for (uint32_t b = 0; b < problem.at("buckets").size(); ++b)
    {
        const auto &bucket = problem.at("buckets").at(b);
        state.buckets.emplace_back();
        state.bucket_weights.push_back(bucket.at("weight").get<float>());
        state.bucket_egraphs.emplace_back();
        state.bucket_enode_infos.emplace_back();
        state.selected_vars.emplace_back();
        state.start_vars.emplace_back();
        state.offset_vars.emplace_back();
        state.reachable_cids.emplace_back();
        state.bucket_root_ids.push_back(EClassId{bucket.at("root").get<uint32_t>()});
        auto &graph = state.bucket_egraphs.back();
        const auto &nodes = bucket.at("nodes");
        for (const auto &node : nodes)
        {
            const uint32_t pages = node.at("size_pages").get<uint32_t>();
            const EClassId cid = graph.addEClass({pages}, {1}, DType::FLOAT32, space);
            graph.getEClass(cid).base_eclass_id = BaseEClassId{node.at("base_id").get<uint32_t>()};
            state.reachable_cids.back().push_back(cid);
        }
        for (uint32_t c = 0; c < nodes.size(); ++c)
        {
            const auto &node = nodes.at(c);
            const EClassId cid{c};
            for (uint32_t a = 0; a < node.at("alternatives").size(); ++a)
            {
                const auto &alternative = node.at("alternatives").at(a);
                const std::string op = alternative.at("op").get<std::string>();
                const OpType op_type = op == "input" ? OpType::INPUT : op == "cache" ? OpType::CACHE :
                                       op == "scatter" ? OpType::SCATTER : OpType::ADD;
                std::vector<EClassId> children;
                for (uint32_t child : alternative.at("children").get<std::vector<uint32_t>>())
                    children.push_back(EClassId{child});
                std::vector<Engine> engines;
                for (uint32_t engine : alternative.at("engines").get<std::vector<uint32_t>>())
                    engines.emplace_back(engine, EngineType::CPU);
                const uint32_t pages = node.at("size_pages").get<uint32_t>();
                graph.addENode(cid, ENode(KernelId{0}, op_type,
                    "oracle_" + std::to_string(c) + "_" + std::to_string(a), children,
                    {pages}, {1}, DType::FLOAT32, space, engines));
                const ENodeId en_id = graph.getEClass(cid).enodes.back();
                state.bucket_enode_infos[b].resize(en_id.value + 1);
                auto &info = state.bucket_enode_infos[b][en_id.value];
                info.cost = alternative.at("cost").get<float>();
                info.is_view = alternative.at("view").get<bool>();
            }
            VarInfo info;
            info.bucket_idx = b;
            info.eclass_id = cid;
            info.base_eclass_id = graph.getEClass(cid).base_eclass_id;
            info.mem_space = space;
            info.size_pages = node.at("size_pages").get<uint32_t>();
            info.size_bytes = info.size_pages * 4;
            info.type = VarType::SELECTED;
            info.name = "selected_" + std::to_string(b) + "_" + std::to_string(c);
            state.selected_vars[b][cid] = state.addVar(info, makeDomain(node.at("selected")));
            info.selection_var = state.selected_vars[b][cid];
            info.type = VarType::START;
            info.name = "start_" + std::to_string(b) + "_" + std::to_string(c);
            state.start_vars[b][cid] = state.addVar(info, makeDomain(node.at("start")));
            info.type = VarType::OFFSET;
            info.name = "offset_" + std::to_string(b) + "_" + std::to_string(c);
            state.offset_vars[b][cid] = state.addVar(info, makeDomain(node.at("offset")));
        }
    }
    return state;
}

inline json exportDomains(const SearchState &state, const json &problem)
{
    json out = problem;
    for (size_t c = 0; c < out.at("caches").size(); ++c)
    {
        const BaseEClassId base{out["caches"][c]["base_id"].get<uint32_t>()};
        out["caches"][c]["domain"] = domainValues(state.domains[state.cached_vars.at(base)]);
    }
    for (uint32_t b = 0; b < out.at("buckets").size(); ++b)
        for (uint32_t c = 0; c < out["buckets"][b]["nodes"].size(); ++c)
        {
            const EClassId cid{c};
            auto &node = out["buckets"][b]["nodes"][c];
            node["selected"] = domainValues(state.domains[state.selected_vars[b].at(cid)]);
            node["start"] = domainValues(state.domains[state.start_vars[b].at(cid)]);
            node["offset"] = domainValues(state.domains[state.offset_vars[b].at(cid)]);
        }
    return out;
}


inline std::vector<std::string> cycleStripPassNames()
{
    std::vector<std::string> names;
    for (const CycleStripPass &pass : cycleStripPasses())
        names.push_back(pass.name);
    return names;
}

// Rebuild each bucket's alternatives (those still in their selection domain)
// as an egraph, run one cycle strip pass (or "pipeline": all of them, in
// planner order) and narrow selection domains to the surviving enodes.
inline json stripCycles(const json &problem, const std::string &pass_name)
{
    std::vector<const CycleStripPass *> passes;
    for (const CycleStripPass &pass : cycleStripPasses())
        if (pass_name == "pipeline" || pass_name == pass.name)
            passes.push_back(&pass);
    if (passes.empty())
        Error::throw_err("Unknown cycle strip pass: " + pass_name);
    json out = problem;
    json removed_counts = json::array();
    for (uint32_t b = 0; b < out.at("buckets").size(); ++b)
    {
        auto &nodes = out["buckets"][b]["nodes"];
        const MemSpace space{1, HandleType::CPP};
        EGraph graph;
        for (const auto &node : nodes)
        {
            const uint32_t pages = node.at("size_pages").get<uint32_t>();
            const EClassId cid = graph.addEClass({pages}, {1}, DType::FLOAT32, space);
            graph.getEClass(cid).base_eclass_id = BaseEClassId{node.at("base_id").get<uint32_t>()};
        }
        std::unordered_map<uint32_t, int32_t> value_of_enode;
        for (uint32_t c = 0; c < nodes.size(); ++c)
        {
            const auto &node = nodes.at(c);
            const auto selected = node.at("selected").get<std::vector<int32_t>>();
            for (uint32_t a = 0; a < node.at("alternatives").size(); ++a)
            {
                const int32_t value = static_cast<int32_t>(a + 1);
                if (std::find(selected.begin(), selected.end(), value) == selected.end())
                    continue;
                const auto &alternative = node.at("alternatives").at(a);
                const std::string op = alternative.at("op").get<std::string>();
                const OpType op_type = op == "input" ? OpType::INPUT : op == "cache" ? OpType::CACHE :
                                       op == "scatter" ? OpType::SCATTER : OpType::ADD;
                std::vector<EClassId> children;
                for (uint32_t child : alternative.at("children").get<std::vector<uint32_t>>())
                    children.push_back(EClassId{child});
                const uint32_t pages = node.at("size_pages").get<uint32_t>();
                const size_t before = graph.getEClass(EClassId{c}).enodes.size();
                graph.addENode(EClassId{c}, ENode(KernelId{0}, op_type,
                    "oracle_" + std::to_string(c) + "_" + std::to_string(a), children,
                    {pages}, {1}, DType::FLOAT32, space, {}));
                if (graph.getEClass(EClassId{c}).enodes.size() != before + 1)
                    Error::throw_err("Oracle strip adapter: alternative was deduplicated");
                value_of_enode[graph.getEClass(EClassId{c}).enodes.back().value] = value;
            }
        }
        uint32_t removed = 0;
        for (const CycleStripPass *pass : passes)
            removed += pass->run(graph);
        removed_counts.push_back(removed);
        for (uint32_t c = 0; c < nodes.size(); ++c)
        {
            json selected = json::array();
            for (int32_t value : nodes[c].at("selected").get<std::vector<int32_t>>())
                if (value == 0)
                    selected.push_back(0);
            std::vector<int32_t> kept;
            for (ENodeId enode_id : graph.getEClass(EClassId{c}).enodes)
                kept.push_back(value_of_enode.at(enode_id.value));
            std::sort(kept.begin(), kept.end());
            for (int32_t value : kept)
                selected.push_back(value);
            nodes[c]["selected"] = selected;
        }
    }
    return json{{"problem", out}, {"removed", removed_counts}};
}

inline std::string runCase(const std::string &problem_text, const std::string &mode, float cutoff)
{
    const json problem = json::parse(problem_text);
    if (mode.rfind("strip:", 0) == 0)
        return stripCycles(problem, mode.substr(6)).dump();
    SearchEngine engine(makeState(problem));
    engine.measure_search_volume = false;
    addAllPropagators(engine, false, problem.at("is_dag").get<bool>());
    json out;
    if (mode == "propagate")
    {
        engine.state.best_cost = cutoff;
        std::string conflict;
        out["consistent"] = engine.runPropagators(kInvalidVarId, &conflict);
        out["conflict"] = conflict;
        out["lower_bound"] = engine.state.lower_bound;
        out["problem"] = exportDomains(engine.state, problem);
    }
    else if (mode == "solve")
    {
        // Sample the certified bound at every iteration so its validity is
        // checked against the independent optimum.
        engine.measure_search_gap = true;
        engine.gap_sample_interval_seconds = 0.0;
        out["solved"] = engine.solve(-1.0f);
        out["root_lower_bound"] = engine.root_lower_bound;
        out["max_certified_bound"] = engine.max_certified_bound > -TGConstants::INF
                                         ? json(engine.max_certified_bound) : json(nullptr);
        out["exhausted"] = engine.selector->empty();
        out["cost"] = engine.incumbent_best_cost < TGConstants::INF ? json(engine.incumbent_best_cost) : json(nullptr);
        if (!engine.incumbent_extractions.empty())
        {
            json witness = json::array();
            for (uint32_t b = 0; b < problem.at("buckets").size(); ++b)
            {
                const auto &extraction = engine.incumbent_extractions[b];
                json selected = json::array(), starts = json::array(), offsets = json::array();
                for (uint32_t c = 0; c < problem["buckets"][b]["nodes"].size(); ++c)
                {
                    const EClassId cid{c};
                    const auto selected_it = extraction.selection_map.find(cid);
                    if (selected_it == extraction.selection_map.end())
                    {
                        selected.push_back(0); starts.push_back(0); offsets.push_back(0);
                        continue;
                    }
                    selected.push_back(selected_it->second + 1);
                    const auto rank = std::find(extraction.order.begin(), extraction.order.end(), cid);
                    // Extracted order has discarded absolute rank gaps. Return
                    // an order; Python finds allowed ranks realizing that order.
                    starts.push_back(static_cast<uint32_t>(rank - extraction.order.begin()));
                    const BufferId buffer_id = extraction.eclass_to_buf.at(cid);
                    const auto buffer_it = std::find_if(extraction.buffers.begin(), extraction.buffers.end(),
                        [&](const ParallelBuffer &buffer) { return buffer.id == buffer_id; });
                    if (buffer_it == extraction.buffers.end())
                        Error::throw_err("Oracle witness references a missing buffer");
                    offsets.push_back(buffer_it->offset / 4);
                }
                witness.push_back({{"selected", selected}, {"order", starts}, {"offset", offsets}});
            }
            json cached = json::array();
            for (const auto &cache : problem.at("caches"))
                cached.push_back(engine.incumbent_cached_nodes.count(BaseEClassId{cache.at("base_id").get<uint32_t>()}) ? 1 : 0);
            out["witness"] = {{"cached", cached}, {"buckets", witness}};
        }
    }
    else
        Error::throw_err("Unknown planner oracle adapter mode: " + mode);
    return out.dump();
}
} // namespace planner_oracle
