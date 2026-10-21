// tensor_graphs_cpp/core/plan/relaxation_bound.hpp
#pragma once

// Selection relaxation of one bucket: which alternatives can still be chosen,
// their simulated durations and engines. Used by RelaxationBoundPropagator
// (a root lower bound on each bucket's makespan) and by the root primal
// heuristic. See docs/core/relaxation_bound.md.

#include <algorithm>
#include <atomic>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/logging.hpp"
#include "core/plan/search_state.hpp"
#include "core/serialization.hpp"

namespace plan
{

struct RelaxationAlternative
{
    int32_t value = 0;       // selection value (1-based e-node index)
    double duration = 0.0;   // simulated duration (0 for views, INPUT and CACHE)
    uint32_t engine_mask = 0; // bucket-local engines occupied for a positive `duration`
    uint32_t sched_mask = 0;  // engines the simulator waits for (any non-view, non-source op)
    bool is_copy = false;    // COPY_TO: same logical tensor in another memory space
    bool is_storage_copy = false; // COPY_TO from STORAGE: produces tensor in memory
    bool is_view = false;
    bool is_source = false;  // INPUT or CACHE: finishes at time 0
    bool requires_cache = false; // filled by markCacheAlternatives
    bool is_alias = false;
    std::vector<uint32_t> children; // local class indices, deduplicated
    std::vector<uint32_t> orig_children; // original local class indices (< orig_class_count)
};

inline const std::vector<uint32_t> &childrenOf(const RelaxationAlternative &alt)
{
    return alt.orig_children.empty() ? alt.children : alt.orig_children;
}

struct RelaxationClass
{
    EClassId cid;
    VarId selected_var = kInvalidVarId;
    BaseEClassId base;
    std::vector<RelaxationAlternative> alternatives; // feasible alternatives only
};

struct RelaxationBucket
{
    uint32_t bucket_idx = 0;
    std::vector<RelaxationClass> classes;
    uint32_t orig_class_count = 0;
    std::vector<uint32_t> orig_class_of;
    std::vector<uint32_t> topo; // children before parents
    std::vector<uint32_t> topo_pos;
    uint32_t root = UINT32_MAX;
    std::vector<Engine> engines;
    // Classes linked by COPY_TO alternatives hold the same logical tensor.
    std::vector<uint32_t> group;
    std::vector<std::vector<uint32_t>> group_members;
    std::unordered_map<uint32_t, uint32_t> storage_rep;
    bool acyclic = true;
    bool root_feasible = false;
};

inline uint32_t relaxationEngineIndex(RelaxationBucket &rb, const Engine &engine)
{
    for (uint32_t i = 0; i < rb.engines.size(); ++i)
        if (rb.engines[i] == engine)
            return i;
    if (rb.engines.size() >= 32)
        Error::throw_err("Relaxation bound supports at most 32 engines per bucket");
    rb.engines.push_back(engine);
    return static_cast<uint32_t>(rb.engines.size() - 1);
}

inline void tg_serialize(BinaryWriter &bw, const RelaxationAlternative &alt)
{
    bw.write(alt.value);
    bw.write(alt.duration);
    bw.write(alt.engine_mask);
    bw.write(alt.sched_mask);
    bw.write(alt.is_copy);
    bw.write(alt.is_storage_copy);
    bw.write(alt.is_view);
    bw.write(alt.is_source);
    bw.write(alt.requires_cache);
    bw.write(alt.is_alias);
    bw.write(alt.children);
    bw.write(alt.orig_children);
}

inline void tg_deserialize(BinaryReader &br, RelaxationAlternative &alt)
{
    br.read(alt.value);
    br.read(alt.duration);
    br.read(alt.engine_mask);
    br.read(alt.sched_mask);
    br.read(alt.is_copy);
    br.read(alt.is_storage_copy);
    br.read(alt.is_view);
    br.read(alt.is_source);
    br.read(alt.requires_cache);
    br.read(alt.is_alias);
    br.read(alt.children);
    br.read(alt.orig_children);
}

inline void tg_serialize(BinaryWriter &bw, const RelaxationClass &rc)
{
    bw.write(rc.cid);
    bw.write(rc.selected_var);
    bw.write(rc.base);
    bw.write(rc.alternatives);
}

inline void tg_deserialize(BinaryReader &br, RelaxationClass &rc)
{
    br.read(rc.cid);
    br.read(rc.selected_var);
    br.read(rc.base);
    br.read(rc.alternatives);
}

inline void tg_serialize(BinaryWriter &bw, const RelaxationBucket &rb)
{
    const uint32_t magic = 0x54475242; // 'TGRB'
    const uint32_t version = 1;
    bw.write(magic);
    bw.write(version);
    bw.write(rb.bucket_idx);
    bw.write(rb.orig_class_count);
    bw.write(rb.root);
    bw.write(rb.acyclic);
    bw.write(rb.root_feasible);
    bw.write(rb.engines);
    bw.write(rb.classes);
    bw.write(rb.orig_class_of);
    bw.write(rb.topo);
    bw.write(rb.topo_pos);
    bw.write(rb.group);
    bw.write(rb.group_members);
    bw.write(rb.storage_rep);
}

inline void tg_deserialize(BinaryReader &br, RelaxationBucket &rb)
{
    uint32_t magic = 0;
    uint32_t version = 0;
    br.read(magic);
    if (magic != 0x54475242)
        throw std::runtime_error("Invalid magic number in RelaxationBucket checkpoint");
    br.read(version);
    if (version != 1)
        throw std::runtime_error("Unsupported version in RelaxationBucket checkpoint");
    br.read(rb.bucket_idx);
    br.read(rb.orig_class_count);
    br.read(rb.root);
    br.read(rb.acyclic);
    br.read(rb.root_feasible);
    br.read(rb.engines);
    br.read(rb.classes);
    br.read(rb.orig_class_of);
    br.read(rb.topo);
    br.read(rb.topo_pos);
    br.read(rb.group);
    br.read(rb.group_members);
    br.read(rb.storage_rep);
}

inline bool saveRelaxationBucket(const RelaxationBucket &rb, const std::string &path)
{
    try
    {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open())
            return false;
        BinaryWriter bw(out);
        bw.write(rb);
        return out.good();
    }
    catch (const std::exception &e)
    {
        LOG(WARNING) << "Failed to save RelaxationBucket to " << path << ": " << e.what();
        return false;
    }
}

inline bool loadRelaxationBucket(const std::string &path, RelaxationBucket &rb)
{
    try
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open())
            return false;
        BinaryReader br(in);
        br.read(rb);
        return true;
    }
    catch (const std::exception &e)
    {
        LOG(WARNING) << "Failed to load RelaxationBucket from " << path << ": " << e.what();
        return false;
    }
}

inline uint64_t hashBucketEGraph(const EGraph &egraph, const SearchState &state, uint32_t b)
{
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ull;
    };
    mix(state.buckets.size());
    mix(b);
    mix(egraph.classes.size());
    mix(egraph.enodes.size());
    mix(egraph.root_class_id.value);
    for (const auto &enode : egraph.enodes)
    {
        mix(static_cast<uint64_t>(enode.getOpType()));
        mix(enode.getKernelId().value);
        for (const auto &ch : enode.getChildren())
            mix(ch.value);
    }
    return h;
}

// Durations and engines mirror SearchEngine::evaluateMakespan and
// MakespanSimulator: INPUT/CACHE finish at 0 without an engine, views take no
// engine time, an infinite cost simulates as 1, and an empty engine list runs
// on CPU engine 0.
inline RelaxationBucket buildRelaxationBucket(const SearchState &state, uint32_t b, bool allow_checkpoints = true)
{
    const bool ignore_checkpoints = !allow_checkpoints ||
                                    (std::getenv("TG_IGNORE_CHECKPOINTS") != nullptr &&
                                     std::string(std::getenv("TG_IGNORE_CHECKPOINTS")) != "0");
    std::string cp_dir = "checkpoints";
    if (const char *dir_env = std::getenv("TG_CHECKPOINT_DIR"))
        cp_dir = dir_env;

    const EGraph &egraph = state.bucket_egraphs.at(b);
    const uint64_t eg_hash = hashBucketEGraph(egraph, state, b);
    char hash_buf[32];
    std::snprintf(hash_buf, sizeof(hash_buf), "%016llx", static_cast<unsigned long long>(eg_hash));
    const std::string cp_file = cp_dir + "/relaxation_bucket_nb" + std::to_string(state.buckets.size()) +
                                "_b" + std::to_string(b) + "_" + hash_buf + ".bin";
    const std::string cp_latest = cp_dir + "/relaxation_bucket_latest.bin";

    if (!ignore_checkpoints && std::filesystem::exists(cp_file))
    {
        RelaxationBucket cached_rb;
        if (loadRelaxationBucket(cp_file, cached_rb))
        {
            LOG(INFO) << "[RelaxationBucket] Loaded checkpoint from " << cp_file
                      << " (classes=" << cached_rb.classes.size() << ")";
            return cached_rb;
        }
    }

    RelaxationBucket rb;
    rb.bucket_idx = b;
    const auto &infos = state.bucket_enode_infos.at(b);
    std::unordered_map<EClassId, uint32_t> local;
    for (EClassId cid : state.reachable_cids.at(b))
    {
        auto it = state.selected_vars.at(b).find(cid);
        if (it == state.selected_vars[b].end())
            continue;
        local.emplace(cid, static_cast<uint32_t>(rb.classes.size()));
        RelaxationClass rc;
        rc.cid = cid;
        rc.selected_var = it->second;
        rc.base = egraph.getEClass(cid).base_eclass_id;
        rb.classes.push_back(std::move(rc));
        if (std::getenv("TG_DEBUG_BOUND") && ((cid.value >= 6870 && cid.value <= 6890) || (cid.value >= 3655 && cid.value <= 3662) || cid.value == 1))
        {
            LOG(INFO) << "[debug cid map] cid=" << cid.value << " -> u=" << (rb.classes.size() - 1);
        }
    }
    const auto root_it = local.find(egraph.findConst(state.bucket_root_ids.at(b)));
    if (root_it == local.end())
        Error::throw_err("Relaxation bound: bucket " + std::to_string(b) + " root has no selection variable");
    rb.root = root_it->second;

    if (std::getenv("TG_DEBUG_BOUND"))
    {
        EClassId root_cid = rb.classes[rb.root].cid;
        const EClass &rcls = egraph.getEClass(root_cid);
        LOG(INFO) << "[debug root enodes] root_cid=" << root_cid.value << " enodes=" << rcls.enodes.size();
        for (uint32_t ei = 0; ei < rcls.enodes.size(); ++ei)
        {
            const ENode &en = egraph.getENode(rcls.enodes[ei]);
            float c = rcls.enodes[ei].value < infos.size() ? infos[rcls.enodes[ei].value].cost : 0.0f;
            std::string kids_str;
            for (auto ch : en.getChildren()) kids_str += std::to_string(egraph.findConst(ch).value) + " ";
            LOG(INFO) << "  root enode " << ei << " op=" << static_cast<int>(en.getOpType()) << ":" << en.getOpName()
                      << " cost=" << c << " kids=[" << kids_str << "]";
        }
    }

    for (size_t u_idx = 0; u_idx < rb.classes.size(); ++u_idx)
    {
        RelaxationClass &rc = rb.classes[u_idx];
        const Domain &domain = state.domains[rc.selected_var];
        const EClass &cls = egraph.getEClass(rc.cid);
        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            const int32_t value = static_cast<int32_t>(en_idx + 1);
            if (!domain.contains(value))
                continue;
            const ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = egraph.getENode(en_id);
            RelaxationAlternative alt;
            alt.value = value;
            bool is_storage_copy = false;
            if (enode.getOpType() == OpType::COPY_TO && !enode.getChildren().empty())
            {
                const EClassId src_cid = egraph.findConst(enode.getChildren()[0]);
                if (egraph.getEClass(src_cid).mem_space.type == HandleType::STORAGE)
                    is_storage_copy = true;
                if (std::getenv("TG_DEBUG_BOUND") && rc.cid.value == 8432)
                {
                    LOG(INFO) << "[debug cid 8432] en_id=" << en_id.value << " op=" << static_cast<int>(enode.getOpType())
                              << " ch0=" << enode.getChildren()[0].value << " src_cid=" << src_cid.value
                              << " src_mem=" << static_cast<int>(egraph.getEClass(src_cid).mem_space.type);
                }
            }
            alt.is_storage_copy = is_storage_copy;
            alt.is_copy = enode.getOpType() == OpType::COPY_TO && !is_storage_copy;
            const bool is_view = en_id.value < infos.size() && infos[en_id.value].is_view;
            const bool is_source = enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE;
            alt.is_view = is_view;
            alt.is_source = is_source;
            if (!is_view && !is_source)
            {
                if (enode.getEngines().empty())
                    alt.sched_mask |= 1u << relaxationEngineIndex(rb, Engine{0, EngineType::CPU});
                for (const Engine &engine : enode.getEngines())
                    alt.sched_mask |= 1u << relaxationEngineIndex(rb, engine);
                float cost = en_id.value < infos.size() ? infos[en_id.value].cost : 0.0f;
                if (cost == TGConstants::INF || std::isinf(cost))
                    cost = 1.0f;
                if (std::isnan(cost) || cost < 0.0f)
                    Error::throw_err("Relaxation bound: invalid cost for e-node " + std::to_string(en_id.value));
                alt.duration = cost;
                if (std::getenv("TG_DEBUG_BOUND") && (rc.cid.value == 41 || rc.cid.value == 4 || rc.cid.value == 3659 || rc.cid.value == 6882 || rc.cid.value == 6884))
                {
                    std::string kids_str;
                    for (auto ch : enode.getChildren()) kids_str += std::to_string(egraph.findConst(ch).value) + " ";
                    LOG(INFO) << "[debug cid " << rc.cid.value << "] val=" << value << " en_id=" << en_id.value
                              << " op=" << static_cast<int>(enode.getOpType()) << " cost=" << cost
                              << " kname=" << enode.getOpName() << " kids=[" << kids_str << "]";
                }
                if (std::getenv("TG_DEBUG_BOUND") && cost > 5.0)
                {
                    std::string kname = enode.getOpName();
                    std::string kids_str;
                    for (auto ch : enode.getChildren()) kids_str += std::to_string(egraph.findConst(ch).value) + " ";
                    LOG(INFO) << "[debug heavy enode] cid=" << rc.cid.value << " u=" << u_idx << " en_id=" << en_id.value
                              << " op=" << static_cast<int>(enode.getOpType()) << " cost=" << cost
                              << " kname=" << kname << " kids=[" << kids_str << "]"
                              << " is_storage_copy=" << is_storage_copy
                              << " engines=" << enode.getEngines().size()
                              << " shape=" << (cls.shape.empty() ? 0 : cls.shape[0]);
                }
                if (cost > 0.0f)
                {
                    if (enode.getEngines().empty())
                        alt.engine_mask |= 1u << relaxationEngineIndex(rb, Engine{0, EngineType::CPU});
                    for (const Engine &engine : enode.getEngines())
                        alt.engine_mask |= 1u << relaxationEngineIndex(rb, engine);
                }
            }
            for (EClassId child : enode.getChildren())
            {
                const auto child_it = local.find(egraph.findConst(child));
                if (child_it == local.end())
                    Error::throw_err("Relaxation bound: child class " + std::to_string(child.value) +
                                     " of a reachable class has no selection variable");
                if (std::find(alt.children.begin(), alt.children.end(), child_it->second) == alt.children.end())
                    alt.children.push_back(child_it->second);
            }
            rc.alternatives.push_back(std::move(alt));
        }
    }



    // Original class count before cycle unrolling.
    const uint32_t orig_n = static_cast<uint32_t>(rb.classes.size());
    rb.orig_class_count = orig_n;
    rb.orig_class_of.resize(orig_n);
    for (uint32_t u = 0; u < orig_n; ++u)
        rb.orig_class_of[u] = u;

    // Eliminate direct self-referential alternatives (u in alt.children).
    for (uint32_t u = 0; u < orig_n; ++u)
    {
        auto &alts = rb.classes[u].alternatives;
        alts.erase(std::remove_if(alts.begin(), alts.end(), [&](const RelaxationAlternative &alt) {
            return std::find(alt.children.begin(), alt.children.end(), u) != alt.children.end();
        }), alts.end());
    }

    // Build adjacency list among original classes.
    std::vector<std::vector<uint32_t>> adj(orig_n);
    for (uint32_t u = 0; u < orig_n; ++u)
    {
        for (const auto &alt : rb.classes[u].alternatives)
        {
            for (uint32_t ch : alt.children)
            {
                if (ch != u && std::find(adj[u].begin(), adj[u].end(), ch) == adj[u].end())
                    adj[u].push_back(ch);
            }
        }
    }

    // Find SCCs with Tarjan's algorithm.
    std::vector<int32_t> dfn(orig_n, -1);
    std::vector<int32_t> low(orig_n, -1);
    std::vector<uint8_t> in_stack(orig_n, 0);
    std::vector<uint32_t> tarjan_stack;
    int32_t timer = 0;
    std::vector<std::vector<uint32_t>> sccs;

    struct DFSFrame
    {
        uint32_t u;
        size_t edge_idx;
    };
    std::vector<DFSFrame> call_stack;

    for (uint32_t i = 0; i < orig_n; ++i)
    {
        if (dfn[i] != -1)
            continue;
        call_stack.push_back({i, 0});
        dfn[i] = low[i] = timer++;
        tarjan_stack.push_back(i);
        in_stack[i] = 1;

        while (!call_stack.empty())
        {
            auto &frame = call_stack.back();
            uint32_t u = frame.u;
            if (frame.edge_idx < adj[u].size())
            {
                uint32_t v = adj[u][frame.edge_idx++];
                if (dfn[v] == -1)
                {
                    dfn[v] = low[v] = timer++;
                    tarjan_stack.push_back(v);
                    in_stack[v] = 1;
                    call_stack.push_back({v, 0});
                }
                else if (in_stack[v])
                {
                    low[u] = std::min(low[u], dfn[v]);
                }
            }
            else
            {
                call_stack.pop_back();
                if (!call_stack.empty())
                {
                    uint32_t p = call_stack.back().u;
                    low[p] = std::min(low[p], low[u]);
                }
                if (dfn[u] == low[u])
                {
                    std::vector<uint32_t> comp;
                    while (true)
                    {
                        uint32_t w = tarjan_stack.back();
                        tarjan_stack.pop_back();
                        in_stack[w] = 0;
                        comp.push_back(w);
                        if (w == u)
                            break;
                    }
                    if (comp.size() > 1)
                        sccs.push_back(std::move(comp));
                }
            }
        }
    }

    // Unroll cyclic components: a component of k classes becomes k levels
    // (k^2 classes; a cycle-free selection inside it is a path of at most k
    // classes), so components above kMaxUnrolledComponent are not unrolled.
    constexpr size_t kMaxUnrolledComponent = 16;
    for (const auto &comp : sccs)
    {
        if (comp.size() > kMaxUnrolledComponent)
        {
            rb.acyclic = false;
            return rb;
        }
    }

    for (const auto &comp : sccs)
    {
        const uint32_t k = static_cast<uint32_t>(comp.size());
        std::vector<uint8_t> in_comp(orig_n, 0);
        for (uint32_t c : comp)
            in_comp[c] = 1;

        std::unordered_map<uint32_t, std::vector<uint32_t>> helper_classes;
        for (uint32_t l = 0; l < k - 1; ++l)
        {
            for (uint32_t c : comp)
            {
                const uint32_t h_idx = static_cast<uint32_t>(rb.classes.size());
                helper_classes[c].push_back(h_idx);
                RelaxationClass rc;
                rc.cid = rb.classes[c].cid;
                rc.selected_var = kInvalidVarId;
                rc.base = rb.classes[c].base;
                rb.classes.push_back(std::move(rc));
                rb.orig_class_of.push_back(c);
            }
        }

        auto classAtLevel = [&](uint32_t c, uint32_t level) -> uint32_t {
            if (level == k - 1)
                return c;
            return helper_classes[c][level];
        };

        for (uint32_t c : comp)
        {
            std::vector<RelaxationAlternative> external_alts, internal_alts;
            for (auto &alt : rb.classes[c].alternatives)
            {
                bool is_internal = false;
                for (uint32_t ch : alt.children)
                {
                    if (in_comp[ch])
                    {
                        is_internal = true;
                        break;
                    }
                }
                if (is_internal)
                    internal_alts.push_back(std::move(alt));
                else
                    external_alts.push_back(std::move(alt));
            }

            // Level 0: external alternatives only
            {
                const uint32_t h0 = classAtLevel(c, 0);
                for (const auto &orig_alt : external_alts)
                {
                    RelaxationAlternative alt = orig_alt;
                    alt.orig_children = orig_alt.children;
                    rb.classes[h0].alternatives.push_back(std::move(alt));
                }
            }

            // Level l in [1, k-2]: alias to l-1, external alternatives, internal alts pointing to l-1
            for (uint32_t l = 1; l < k - 1; ++l)
            {
                const uint32_t hl = classAtLevel(c, l);
                RelaxationAlternative alias;
                alias.value = -1;
                alias.duration = 0.0;
                alias.engine_mask = 0;
                alias.sched_mask = 0;
                alias.is_copy = true;
                alias.is_view = true;
                alias.is_source = false;
                alias.requires_cache = false;
                alias.is_alias = true;
                alias.children = { classAtLevel(c, l - 1) };
                alias.orig_children = { c };
                rb.classes[hl].alternatives.push_back(std::move(alias));

                for (const auto &orig_alt : external_alts)
                {
                    RelaxationAlternative alt = orig_alt;
                    alt.orig_children = orig_alt.children;
                    rb.classes[hl].alternatives.push_back(std::move(alt));
                }

                for (const auto &orig_alt : internal_alts)
                {
                    RelaxationAlternative alt = orig_alt;
                    alt.orig_children = orig_alt.children;
                    alt.children.clear();
                    for (uint32_t ch : orig_alt.children)
                    {
                        if (in_comp[ch])
                            alt.children.push_back(classAtLevel(ch, l - 1));
                        else
                            alt.children.push_back(ch);
                    }
                    rb.classes[hl].alternatives.push_back(std::move(alt));
                }
            }

            // Level k - 1 (the original class c): alias to k-2, external alternatives, internal alts pointing to k-2
            {
                rb.classes[c].alternatives.clear();
                RelaxationAlternative alias;
                alias.value = -1;
                alias.duration = 0.0;
                alias.engine_mask = 0;
                alias.sched_mask = 0;
                alias.is_copy = true;
                alias.is_view = true;
                alias.is_source = false;
                alias.requires_cache = false;
                alias.is_alias = true;
                alias.children = { classAtLevel(c, k - 2) };
                alias.orig_children = { c };
                rb.classes[c].alternatives.push_back(std::move(alias));

                for (const auto &orig_alt : external_alts)
                {
                    RelaxationAlternative alt = orig_alt;
                    alt.orig_children = orig_alt.children;
                    rb.classes[c].alternatives.push_back(std::move(alt));
                }

                for (const auto &orig_alt : internal_alts)
                {
                    RelaxationAlternative alt = orig_alt;
                    alt.orig_children = orig_alt.children;
                    alt.children.clear();
                    for (uint32_t ch : orig_alt.children)
                    {
                        if (in_comp[ch])
                            alt.children.push_back(classAtLevel(ch, k - 2));
                        else
                            alt.children.push_back(ch);
                    }
                    rb.classes[c].alternatives.push_back(std::move(alt));
                }
            }
        }
    }

    // Children-first order; a cycle disables the relaxation for this bucket.
    const uint32_t n = static_cast<uint32_t>(rb.classes.size());
    std::vector<uint8_t> color(n, 0);
    std::vector<std::pair<uint32_t, uint32_t>> stack;
    rb.topo.reserve(n);
    for (uint32_t start = 0; start < n && rb.acyclic; ++start)
    {
        if (color[start] != 0)
            continue;
        color[start] = 1;
        stack.push_back({start, 0});
        while (!stack.empty() && rb.acyclic)
        {
            auto &[u, next] = stack.back();
            // Flatten (alternative, child) pairs into one cursor.
            uint32_t cursor = next++;
            const auto &alts = rb.classes[u].alternatives;
            bool pushed = false;
            for (const auto &alt : alts)
            {
                if (cursor < alt.children.size())
                {
                    const uint32_t v = alt.children[cursor];
                    if (color[v] == 1)
                        rb.acyclic = false;
                    else if (color[v] == 0)
                    {
                        color[v] = 1;
                        stack.push_back({v, 0});
                    }
                    pushed = true;
                    break;
                }
                cursor -= static_cast<uint32_t>(alt.children.size());
            }
            if (!pushed)
            {
                color[u] = 2;
                rb.topo.push_back(u);
                stack.pop_back();
            }
        }
    }
    if (!rb.acyclic)
        return rb;
    rb.topo_pos.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i)
        rb.topo_pos[rb.topo[i]] = i;

    // Alternatives needing a class that can never be active are infeasible.
    std::vector<uint8_t> feasible(n, 0);
    for (uint32_t u : rb.topo)
    {
        auto &alts = rb.classes[u].alternatives;
        alts.erase(std::remove_if(alts.begin(), alts.end(), [&](const RelaxationAlternative &alt) {
                       for (uint32_t child : alt.children)
                           if (!feasible[child])
                               return true;
                       return false;
                   }),
                   alts.end());
        feasible[u] = !alts.empty();
    }
    rb.root_feasible = feasible[rb.root] != 0;

    std::vector<uint32_t> parent(orig_n);
    std::iota(parent.begin(), parent.end(), 0u);
    auto find = [&](uint32_t x) {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    std::unordered_map<uint32_t, uint32_t> storage_rep;
    for (uint32_t u = 0; u < orig_n; ++u)
    {
        for (const auto &alt : rb.classes[u].alternatives)
        {
            if (alt.is_copy)
            {
                for (uint32_t child : childrenOf(alt))
                    if (child < orig_n)
                        parent[find(child)] = find(u);
            }
            else if (alt.is_storage_copy)
            {
                for (uint32_t child : childrenOf(alt))
                {
                    if (child < orig_n)
                    {
                        auto it = storage_rep.find(child);
                        if (it != storage_rep.end())
                            parent[find(u)] = find(it->second);
                        else
                            storage_rep[child] = u;
                    }
                }
            }
        }
    }
    // The classes of a cyclic component can each be derived from the others
    // (copies, views, a slice of a larger tensor and its scatter back), so
    // every plan that needs one of them holds one of them: they form one
    // group, like copies. A cycle-free plan still produces some member
    // through an alternative from outside the component.
    for (const auto &comp : sccs)
        for (uint32_t c : comp)
            parent[find(c)] = find(comp.front());
    rb.storage_rep = storage_rep;
    std::vector<uint32_t> group_of_rep(orig_n, UINT32_MAX);
    rb.group.assign(n, 0);
    for (uint32_t u = 0; u < orig_n; ++u)
    {
        const uint32_t rep = find(u);
        if (group_of_rep[rep] == UINT32_MAX)
        {
            group_of_rep[rep] = static_cast<uint32_t>(rb.group_members.size());
            rb.group_members.emplace_back();
        }
        rb.group[u] = group_of_rep[rep];
        rb.group_members[rb.group[u]].push_back(u);
    }
    for (uint32_t h = orig_n; h < n; ++h)
    {
        rb.group[h] = rb.group[rb.orig_class_of[h]];
    }
    // A group's members are the classes that can be active: those an
    // alternative below the root reaches, level copies of unrolled cycles
    // included (when a cycle is entered through another member, only lower
    // levels of a class may be reachable). Containment of groups and
    // charging look at members.
    {
        std::vector<uint8_t> reached(n, 0);
        std::vector<uint32_t> stack{rb.root};
        reached[rb.root] = 1;
        while (!stack.empty())
        {
            const uint32_t u = stack.back();
            stack.pop_back();
            for (const auto &alt : rb.classes[u].alternatives)
                for (uint32_t ch : alt.children)
                    if (!reached[ch])
                    {
                        reached[ch] = 1;
                        stack.push_back(ch);
                    }
        }
        for (auto &members : rb.group_members)
            members.clear();
        for (uint32_t u = 0; u < n; ++u)
            if (reached[u])
                rb.group_members[rb.group[u]].push_back(u);
    }
    uint32_t n_storage_copies = 0;
    double sum_storage_copy_dur = 0.0;
    for (const auto &rc : rb.classes)
        for (const auto &alt : rc.alternatives)
            if (alt.is_storage_copy)
            {
                n_storage_copies++;
                sum_storage_copy_dur += alt.duration;
            }
    if (std::getenv("TG_DEBUG_BOUND"))
    {
        LOG(INFO) << "[debug storage copies] count=" << n_storage_copies << " total_dur=" << sum_storage_copy_dur;
        LOG(INFO) << "[debug step 1] buildRelaxationBucket done n=" << n << " groups=" << rb.group_members.size();
    }
    if (!ignore_checkpoints && rb.classes.size() >= 10)
    {
        std::error_code ec;
        std::filesystem::create_directories(cp_dir, ec);
        saveRelaxationBucket(rb, cp_file);
        saveRelaxationBucket(rb, cp_latest);
    }
    return rb;
}

// Everything the bounds depend on, serialized: two buckets with equal
// signatures have identical relaxations and therefore identical bounds.
inline std::vector<uint64_t> relaxationSignature(const RelaxationBucket &rb)
{
    std::vector<uint64_t> sig;
    auto bits = [](double d) {
        uint64_t u = 0;
        std::memcpy(&u, &d, sizeof(u));
        return u;
    };
    sig.push_back(rb.classes.size());
    sig.push_back(rb.orig_class_count);
    sig.push_back(rb.root);
    sig.push_back(rb.acyclic ? 1 : 0);
    sig.push_back(rb.engines.size());
    for (const Engine &e : rb.engines)
        sig.push_back((static_cast<uint64_t>(e.type) << 32) | static_cast<uint64_t>(e.idx));
    for (uint32_t u : rb.topo)
        sig.push_back(u);
    for (uint32_t g : rb.group)
        sig.push_back(g);
    for (const RelaxationClass &rc : rb.classes)
    {
        sig.push_back(rc.alternatives.size());
        for (const RelaxationAlternative &alt : rc.alternatives)
        {
            sig.push_back(static_cast<uint64_t>(static_cast<uint32_t>(alt.value)));
            sig.push_back(bits(alt.duration));
            sig.push_back((static_cast<uint64_t>(alt.engine_mask) << 32) | alt.sched_mask);
            sig.push_back((alt.is_copy ? 1u : 0u) | (alt.is_view ? 2u : 0u) | (alt.is_source ? 4u : 0u) |
                          (alt.requires_cache ? 8u : 0u) | (alt.is_alias ? 16u : 0u));
            sig.push_back(alt.children.size());
            for (uint32_t ch : alt.children)
                sig.push_back(ch);
            sig.push_back(alt.orig_children.size());
            for (uint32_t ch : alt.orig_children)
                sig.push_back(ch);
        }
    }
    return sig;
}

namespace relaxation_detail
{
inline size_t words(size_t bits) { return (bits + 63) / 64; }
inline void setBit(uint64_t *w, size_t i) { w[i >> 6] |= 1ull << (i & 63); }
inline bool testBit(const uint64_t *w, size_t i) { return (w[i >> 6] >> (i & 63)) & 1ull; }
} // namespace relaxation_detail

// Lower bound on the minimum over plans of sum_e lambda_e * load_e(plan) for
// one bucket, where load_e is the simulated work on engine e. Because each
// engine runs its operations serially, makespan >= max_e load_e >= this value
// whenever lambda is a probability vector.
//
// Recursion (one region at a time, see docs/core/relaxation_bound.md):
//   L(c) = min over alternatives a of [ w(a) + sum of L(anchor) over children
//          with pairwise disjoint supports + sum of mp(G) over necessary
//          groups G of a's children disjoint from those supports ]
// with the invariant L(c) <= weighted cost of the plan's classes in S(c), the
// support of c. Regions follow a containment chain of necessary groups of the
// root, so the shared history below a link is counted only once.
class RelaxationBoundComputer
{
  public:
    explicit RelaxationBoundComputer(const RelaxationBucket &rb) : rb_(rb)
    {
        if (!rb.acyclic || !rb.root_feasible)
            return;
        prepare();
        for (const Region &region : regions_)
            degenerate_ = degenerate_ || region.nodes.size() > kMaxRegionClasses;
        ready_ = true;
    }

    bool ready() const { return ready_; }
    // The region decomposition found (almost) no chain of necessary groups
    // (a region above kMaxRegionClasses classes, e.g. when routes through
    // other tensors avoid every link): the load recursion would take minutes,
    // so the bound functions prove nothing (return 0) and evaluate() only
    // fills cheap estimates for the primal heuristic.
    bool degenerate() const { return degenerate_; }
    static constexpr size_t kMaxRegionClasses = 32768;
    size_t numRegions() const { return regions_.size(); }
    size_t numIntervals() const { return intervals_.size(); }
    // Per-window values of the last interval bound, in chain order; the last
    // window ends at the root. With stride s, window l (but the last) spans
    // chain links l*s .. (l+1)*s - 1, so the windows of stride 2s are unions
    // of two consecutive windows of stride s.
    const std::vector<double> &windowValues() const { return window_values_; }
    // Chain index of the closing link of each window but the last (the last
    // window closes at the root); window l spans links after
    // windowLastLinks()[l-1] up to windowLastLinks()[l].
    const std::vector<size_t> &windowLastLinks() const { return window_last_link_; }

    // Time-interval bound (docs/core/relaxation_bound.md, "Interval bound").
    // Along a chain G_1, ..., G_k of the root's necessary groups in which every
    // member of G_i necessarily depends on G_{i-1}, let t_i be the earliest
    // finish of an active member of G_i (t_0 = 0, the final link is the root).
    // A class that necessarily depends on G_{i-1} starts at or after t_{i-1};
    // one in the selected sub-DAG of the earliest member of G_i finishes by t_i.
    // So such classes run inside [t_{i-1}, t_i], and that window is at least
    // their longest dependency chain, at least their lambda-weighted load and
    // at least mu * (any chain) + (1 - mu) * (weighted load), with lambda and
    // mu chosen per window. The windows are disjoint and ordered, so their
    // values add up to a lower bound on the makespan.
    size_t intervalChainLength() const { return ichain_.size(); }

    // Search-node budget of the exact window enumeration (tests set 0 to
    // exercise the cut DP).
    void setExactWindowNodes(uint64_t nodes) { exact_window_nodes_ = nodes; }
    // Tests: the exact window enumeration without its pruning (chain and
    // load bounds, left-out classes, duplicate alternatives) must give the
    // same values.
    void setExactWindowPruning(bool enabled) { exact_window_pruning_ = enabled; }
    void setExactWindowStage(bool enabled) { exact_window_stage_ = enabled; }

    // Interval bound over flexible windows (docs/core/relaxation_bound.md,
    // "Flexible windows"): every window between chain links a < b with
    // b - a <= max_len (and from every link to the root) is valued like a
    // stride window, and a DP keeps the best partition of the chain into
    // consecutive windows. Windows that follow a model's blocks also count
    // the parallel branches (work after link a and below link b that no
    // finer window holds), which uniform strides rarely align with.
    static constexpr size_t kFlexibleMaxClasses = 512;
    double flexibleIntervalBound(size_t max_len, std::chrono::steady_clock::time_point deadline, size_t long_len = 0)
    {
        if (!ready_)
            Error::throw_err("Interval bound evaluated on an unprepared bucket");
        if (degenerate_)
        {
            last_partition_.clear();
            last_partition_values_.clear();
            return 0.0;
        }
        const size_t k = ichain_.size();
        last_partition_.clear();
        if (k == 0)
            return 0.0;
        // value[a + 1][b - a - 1]: window after link a (a = -1: from the
        // start) up to link b (b = k: the root).
        std::vector<std::vector<double>> value(k + 1);
        // Exact window values in parallel (read-only); windows whose
        // enumeration does not finish are valued afterwards.
        std::vector<std::vector<Region>> pending(k + 1);
        std::vector<std::vector<size_t>> pending_slot(k + 1);
        std::atomic<bool> late{false};
        ThreadPool::get().parallel_for(static_cast<uint32_t>(k + 1), [&](uint32_t a1) {
            if (std::chrono::steady_clock::now() >= deadline)
            {
                late = true;
                return;
            }
            WindowScratch ws;
            std::vector<Region> scratch;
            prepareWindowStart(a1, ws);
            for (size_t b = a1; b <= std::min(k, a1 + max_len - 1); ++b)
            {
                double v = 0.0;
                if (buildWindow(a1, b, ws, scratch) && !scratch[0].zero_value)
                {
                    v = exactWindow(scratch[0], exact_window_nodes_);
                    if (!(v > 0.0))
                    {
                        pending_slot[a1].push_back(value[a1].size());
                        pending[a1].push_back(std::move(scratch[0]));
                    }
                }
                value[a1].push_back(v);
            }
        });
        if (late)
            return 0.0; // nothing proven: callers keep their bound
        // The rest in parallel too, each thread with its own weight table.
        ThreadPool::get().parallel_for(static_cast<uint32_t>(k + 1), [&](uint32_t a1) {
            std::vector<double> local;
            thread_weights_ = &local;
            for (size_t p = 0; p < pending[a1].size(); ++p)
            {
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    late = true;
                    break;
                }
                value[a1][pending_slot[a1][p]] = windowValue(pending[a1][p]);
            }
            thread_weights_ = nullptr;
        });
        if (late)
            return 0.0;
        // best[a1]: largest sum of windows covering the chain up to link
        // a1 - 1; from[a1] the window that reaches it.
        const double none_value = -std::numeric_limits<double>::infinity();
        auto bestPartition = [&]() {
            std::vector<double> best(k + 1, none_value);
            std::vector<size_t> from(k + 1, SIZE_MAX);
            best[0] = 0.0;
            double total = 0.0;
            size_t total_from = SIZE_MAX;
            for (size_t a1 = 0; a1 <= k; ++a1)
            {
                if (best[a1] == none_value)
                    continue;
                for (size_t j = 0; j < value[a1].size(); ++j)
                {
                    const size_t b = a1 + j;
                    const double v = best[a1] + value[a1][j];
                    if (b == k)
                    {
                        if (v > total || total_from == SIZE_MAX)
                        {
                            total = std::max(total, v);
                            total_from = a1;
                        }
                    }
                    else if (v > best[b + 1])
                    {
                        best[b + 1] = v;
                        from[b + 1] = a1;
                    }
                }
            }
            last_partition_.clear();
            last_partition_values_.clear();
            for (size_t a1 = total_from, b = k; a1 != SIZE_MAX;)
            {
                last_partition_.push_back({a1, b});
                last_partition_values_.push_back(value[a1][b - a1]);
                if (a1 == 0)
                    break;
                b = a1 - 1;
                a1 = from[a1];
            }
            std::reverse(last_partition_.begin(), last_partition_.end());
            std::reverse(last_partition_values_.begin(), last_partition_values_.end());
            return total;
        };
        double total = bestPartition();
        // Long windows (more than max_len links, up to long_len) ending
        // where the partition has a boundary: they hold parallel branches
        // that fork before a short window and join at its end. Valued by the
        // cut DP alone (their exact enumeration rarely finishes).
        if (long_len > max_len)
        {
            std::vector<std::pair<size_t, size_t>> candidates;
            for (const auto &[pa1, pb] : last_partition_)
                for (size_t len = max_len + 1; len <= long_len && len <= pb + 1; ++len)
                    candidates.push_back({pb + 1 - len, pb});
            std::vector<double> long_value(candidates.size(), 0.0);
            ThreadPool::get().parallel_for(static_cast<uint32_t>(candidates.size()), [&](uint32_t c) {
                if (late || std::chrono::steady_clock::now() >= deadline)
                {
                    late = true;
                    return;
                }
                WindowScratch ws;
                std::vector<Region> scratch;
                const auto [a1, b] = candidates[c];
                prepareWindowStart(a1, ws);
                if (!buildWindow(a1, b, ws, scratch) || scratch[0].zero_value ||
                    scratch[0].nodes.size() > kCutDpMaxClasses)
                    return;
                long_value[c] = cutDpWindow(scratch[0], criticalPathRegion(scratch[0]), kCutDpStates, kCutDpPoints);
            });
            if (late)
                return total; // the short windows' partition stands
            for (size_t c = 0; c < candidates.size(); ++c)
            {
                const auto [a1, b] = candidates[c];
                if (value[a1].size() <= b - a1)
                    value[a1].resize(b - a1 + 1, 0.0);
                value[a1][b - a1] = std::max(value[a1][b - a1], long_value[c]);
            }
            total = bestPartition();
        }
        return total;
    }

    // Windows (a + 1, b) of the partition chosen by the last
    // flexibleIntervalBound, in chain order.
    const std::vector<std::pair<size_t, size_t>> &lastPartition() const { return last_partition_; }
    // Their values, and the members of chain link i (local classes).
    const std::vector<double> &lastPartitionValues() const { return last_partition_values_; }
    const std::vector<uint32_t> &chainLinkMembers(size_t i) const { return rb_.group_members[ichain_[i]]; }

    // The `count` best selections, by the window relaxation's value, of
    // window (a1, b) that a search of `budget` nodes finds (best first; not
    // proven optimal when the search does not finish), then those of them a
    // local search on the window value improves; empty when the window is
    // too large. Each entry is a class and its alternative indices: the
    // chosen one first, then those equal to it inside the window (they
    // differ only in children that add nothing there). Classes that add
    // nothing to the window are left out. Used by the primal heuristic.
    struct WindowChoice
    {
        uint32_t cls;
        std::vector<uint32_t> alts;
    };
    std::vector<std::vector<WindowChoice>> windowSelections(size_t a1, size_t b, size_t count, uint64_t budget,
                                                            std::chrono::steady_clock::time_point stop)
    {
        std::vector<std::vector<WindowChoice>> out;
        WindowScratch ws;
        std::vector<Region> scratch;
        prepareWindowStart(a1, ws);
        if (!buildWindow(a1, b, ws, scratch))
            return out;
        // Two searches of half the budget: shortest chains first, least
        // work first; the better selections are kept.
        std::vector<std::pair<double, std::vector<int32_t>>> top, by_work;
        std::vector<std::vector<std::vector<uint32_t>>> equivalents;
        const auto half = std::chrono::steady_clock::now() + (stop - std::chrono::steady_clock::now()) / 2;
        exactWindow(scratch[0], budget / 2, nullptr, &top, count, &equivalents, false, half);
        exactWindow(scratch[0], budget / 2, nullptr, &by_work, count, nullptr, true, stop);
        top.insert(top.end(), by_work.begin(), by_work.end());
        std::stable_sort(top.begin(), top.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
        if (top.size() > count)
            top.resize(count);
        // Each also polished by local search on the window value (listed
        // after the found ones when it changes them).
        for (size_t t = 0, found = top.size(); t < found; ++t)
        {
            std::vector<int32_t> polished = top[t].second;
            const double value = improveWindowChoice(scratch[0], polished, top[t].first);
            if (value < top[t].first - 1e-12)
                top.push_back({value, std::move(polished)});
        }
        const Region &region = scratch[0];
        for (const auto &[value, choice] : top)
        {
            out.emplace_back();
            for (uint32_t i = 0; i < region.nodes.size(); ++i)
            {
                if (choice[i] < 0)
                    continue;
                WindowChoice entry{region.nodes[i], {region.alts[i][static_cast<size_t>(choice[i])].alt_index}};
                for (const auto &group : equivalents[i])
                    if (std::find(group.begin(), group.end(), static_cast<uint32_t>(choice[i])) != group.end())
                        for (uint32_t a : group)
                            if (a != static_cast<uint32_t>(choice[i]))
                                entry.alts.push_back(region.alts[i][a].alt_index);
                out.back().push_back(std::move(entry));
            }
        }
        return out;
    }

    // Interval bound with windows at every `stride`-th chain link.
    // Past `deadline` the remaining windows count 0 and the bound is 0
    // (nothing proven by this call; the window values stay valid).
    double intervalBound(size_t stride, size_t phase = 0,
                         std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max())
    {
        if (!ready_)
            Error::throw_err("Interval bound evaluated on an unprepared bucket");
        if (degenerate_)
            return 0.0;
        buildIntervals(stride, phase);
        return evaluateIntervals(deadline);
    }

    double evaluateIntervals(std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max())
    {
        if (!ready_)
            Error::throw_err("Interval bound evaluated on an unprepared bucket");
        const size_t ne = rb_.engines.size();
        if (ne == 0)
            return 0.0;
        alt_weight_.resize(alt_offset_.back());
        double total = 0.0;
        std::vector<double> lambda(ne);
        window_values_.clear();
        bool late = false;
        for (Region &region : intervals_)
        {
            window_values_.push_back(0.0);
            late = late || std::chrono::steady_clock::now() >= deadline;
            if (region.zero_value || late)
                continue;
            std::vector<double> cp_nodes, path;
            const double cp = criticalPathRegion(region, cp_nodes);
            static const double kMu[] = {0.0, 0.2, 0.4, 0.6, 0.8, 0.9};
            auto at = [&](double l0) {
                if (ne == 1)
                    lambda[0] = 1.0;
                else
                {
                    lambda[0] = l0;
                    for (size_t e = 1; e < ne; ++e)
                        lambda[e] = (1.0 - l0) / static_cast<double>(ne - 1);
                }
                fillRegionWeights(region, lambda);
                double v = evaluateRegion(region);
                for (double mu : kMu)
                    if (mu > 0.0)
                        v = std::max(v, evaluateRegionPath(region, mu, cp_nodes, path));
                return v;
            };
            double best = 0.0;
            if (ne == 1)
                best = at(1.0);
            else
            {
                // Golden-section search on lambda_0; every evaluated point is valid.
                const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
                double a = 0.0, b = 1.0;
                double c = b - phi * (b - a), d = a + phi * (b - a);
                double fc = at(c), fd = at(d);
                best = std::max({at(0.0), at(1.0), fc, fd});
                for (int it = 0; it < 16; ++it)
                {
                    if (fc < fd)
                    {
                        a = c;
                        c = d;
                        fc = fd;
                        d = a + phi * (b - a);
                        fd = at(d);
                        best = std::max(best, fd);
                    }
                    else
                    {
                        b = d;
                        d = c;
                        fd = fc;
                        c = b - phi * (b - a);
                        fc = at(c);
                        best = std::max(best, fc);
                    }
                }
            }
            window_values_.back() = std::max(cp, best);
        }
        // Exact enumeration and cut DP read only the window and the bucket,
        // so windows are processed in parallel.
        ThreadPool::get().parallel_for(static_cast<uint32_t>(intervals_.size()), [&](uint32_t w) {
            const Region &region = intervals_[w];
            if (region.zero_value || late || std::chrono::steady_clock::now() >= deadline)
                return;
            const double exact = exactWindow(region, exact_window_nodes_);
            const double proven = std::max(window_values_[w], exact);
            // A finished enumeration is already the window relaxation's optimum.
            window_values_[w] = exact > 0.0 || region.nodes.size() > kCutDpMaxClasses
                                    ? proven
                                    : cutDpWindow(region, proven, kCutDpStates, kCutDpPoints);
        });
        for (double v : window_values_)
            total += v;
        return late || std::chrono::steady_clock::now() >= deadline ? 0.0 : total;
    }

    // After evaluate(): each class's bound within its own region, i.e. its
    // cost excluding the shared history below the region (0 outside regions).
    void regionEstimates(std::vector<double> &estimates) const
    {
        if (degenerate_)
        {
            estimates = cheap_estimates_;
            return;
        }
        estimates.assign(rb_.classes.size(), 0.0);
        for (const Region &region : regions_)
            if (!region.zero_value)
                for (size_t i = 0; i < region.nodes.size(); ++i)
                    estimates[region.nodes[i]] = region.L[i];
    }

    // Least weighted cost (weights at `lambda`) to produce each class from
    // scratch, shared classes counted per use: an estimate, not a bound.
    // The primal heuristic starts from these too (they price work below
    // other regions, which region values treat as already paid).
    void cheapEstimates(const std::vector<double> &lambda, std::vector<double> &estimates)
    {
        if (!ready_)
            Error::throw_err("Cheap estimates on an unprepared bucket");
        if (lambda.size() != rb_.engines.size())
            Error::throw_err("Cheap estimates: lambda has the wrong number of engines");
        alt_weight_.resize(alt_offset_.back());
        for (uint32_t u = 0; u < rb_.classes.size(); ++u)
        {
            const auto &alts = rb_.classes[u].alternatives;
            for (uint32_t a = 0; a < alts.size(); ++a)
            {
                double scale = 0.0;
                for (uint32_t e = 0; e < lambda.size(); ++e)
                    if (alts[a].engine_mask & (1u << e))
                        scale += lambda[e];
                alt_weight_[alt_offset_[u] + a] = alts[a].duration * scale;
            }
        }
        fillCheapEstimates(estimates);
    }

    double evaluate(const std::vector<double> &lambda)
    {
        if (!ready_)
            Error::throw_err("Relaxation bound evaluated on an unprepared bucket");
        if (lambda.size() != rb_.engines.size())
            Error::throw_err("Relaxation bound: lambda has the wrong number of engines");
        alt_weight_.resize(alt_offset_.back());
        for (uint32_t u = 0; u < rb_.classes.size(); ++u)
        {
            const auto &alts = rb_.classes[u].alternatives;
            for (uint32_t a = 0; a < alts.size(); ++a)
            {
                double scale = 0.0;
                for (uint32_t e = 0; e < lambda.size(); ++e)
                    if (alts[a].engine_mask & (1u << e))
                        scale += lambda[e];
                alt_weight_[alt_offset_[u] + a] = alts[a].duration * scale;
            }
        }
        if (degenerate_)
        {
            fillCheapEstimates(cheap_estimates_);
            return 0.0;
        }
        double total = 0.0;
        for (size_t ri = 0; ri < regions_.size(); ++ri)
        {
            double rv = evaluateRegion(regions_[ri]);
            total += rv;
            if (std::getenv("TG_DEBUG_BOUND") && rv > 0.5 && lambda.size() >= 2 && lambda[0] > 0.98)
            {
                LOG(INFO) << "[debug heavy region] ri=" << ri << " val=" << rv
                          << " nodes=" << regions_[ri].nodes.size()
                          << " groups=" << regions_[ri].group_members.size()
                          << " target0=" << (regions_[ri].targets.empty() ? 999999 : regions_[ri].nodes[regions_[ri].targets[0]]);
            }
        }
        return total;
    }

    // Global case split (docs/core/relaxation_bound.md, "Global split") at
    // `lambda` over the `candidates` classes whose bound most exceeds their
    // region's value. A plan either selects x, and then x's region costs at
    // least L(x) (x is active and its claims lie in the region) with every
    // other region unchanged, or avoids x, and then is a plan of every
    // region with x forbidden (infeasibility carried to later regions
    // through the children below them). The bound is the best over x of the
    // smaller branch; the plain load bound if no candidate helps. Region
    // values proven by threshold splits at the same lambda (evaluateSplit)
    // hold for every plan, so both branches keep them as floors.
    // Candidates stop at `deadline` (each finished one is valid by itself).
    double evaluateGlobalSplit(const std::vector<double> &lambda, size_t candidates,
                               std::chrono::steady_clock::time_point deadline =
                                   std::chrono::steady_clock::time_point::max())
    {
        if (degenerate_)
            return 0.0;
        evaluate(lambda);
        const bool floors = lambda == split_lambda_;
        const uint32_t n = static_cast<uint32_t>(rb_.classes.size());
        const uint32_t orig_n = rb_.orig_class_count ? rb_.orig_class_count : n;
        auto origOf = [&](uint32_t u) { return u < orig_n ? u : rb_.orig_class_of[u]; };
        // Every copy of a class (cycle unrolling makes level copies) with
        // its region and position, and each region's value.
        std::vector<double> value(regions_.size(), 0.0);
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> copies(orig_n); // (region, node)
        for (uint32_t ri = 0; ri < regions_.size(); ++ri)
        {
            const Region &region = regions_[ri];
            if (region.zero_value)
                continue;
            value[ri] = std::max(regionValue(region), floors ? split_value_[ri] : 0.0);
            for (uint32_t i = 0; i < region.nodes.size(); ++i)
                copies[origOf(region.nodes[i])].push_back({ri, i});
        }
        double plain = 0.0;
        for (double v : value)
            plain += v;
        // Selecting class x (any copy) raises the copy's region to at least
        // its L: the gain is the smallest raise over the copies (0 if a copy
        // lies outside every evaluated region).
        std::vector<std::pair<double, uint32_t>> ranked;
        std::vector<uint8_t> in_region(n, 0);
        for (const auto &list : copies)
            for (const auto &[ri, i] : list)
                in_region[regions_[ri].nodes[i]] = 1;
        for (uint32_t x = 0; x < orig_n; ++x)
        {
            if (copies[x].empty())
                continue;
            double gain = std::numeric_limits<double>::infinity();
            for (const auto &[ri, i] : copies[x])
                gain = std::min(gain, std::max(0.0, regions_[ri].L[i] - value[ri]));
            for (uint32_t u = orig_n; u < n; ++u)
                if (origOf(u) == x && !in_region[u])
                    gain = 0.0;
            if (!in_region[x])
                gain = 0.0;
            if (gain > 1e-9 * std::max(1.0, plain))
                ranked.push_back({gain, x});
        }
        std::sort(ranked.rbegin(), ranked.rend());
        if (ranked.size() > candidates)
            ranked.resize(candidates);
        double best = plain;
        std::vector<double> L;
        std::vector<Claims> support;
        for (const auto &[gain, x] : ranked)
        {
            if (std::chrono::steady_clock::now() >= deadline)
                break;
            const double used = plain + gain;
            if (!(used > best))
                continue; // min(avoided, used) cannot beat best
            // Avoided: forbid every copy of x and carry infeasibility region
            // by region (children below a region come from earlier ones).
            std::vector<uint8_t> infeasible(n, 0);
            double avoided = 0.0;
            for (uint32_t ri = 0; ri < regions_.size() && std::isfinite(avoided); ++ri)
            {
                const Region &region = regions_[ri];
                if (region.zero_value)
                    continue;
                std::vector<uint8_t> forbid(region.nodes.size(), 0);
                for (uint32_t i = 0; i < region.nodes.size(); ++i)
                    forbid[i] = origOf(region.nodes[i]) == x ? 1 : 0;
                const std::vector<uint64_t> zero(region.support_words, 0);
                avoided += std::max(value[ri], evaluateCore(region, &forbid, &zero, L, support, nullptr, &infeasible));
                for (uint32_t i = 0; i < region.nodes.size(); ++i)
                {
                    bool feasible = false;
                    if (!forbid[i])
                        for (const AltInfo &info : region.alts[i])
                        {
                            bool usable = true;
                            for (uint32_t k : info.all_kids)
                                usable = usable && !infeasible[region.nodes[k]];
                            for (uint32_t c : info.below)
                                usable = usable && !infeasible[c];
                            feasible = feasible || usable;
                        }
                    if (!feasible)
                        infeasible[region.nodes[i]] = 1;
                }
            }
            if (std::getenv("TG_DEBUG_PROBE"))
                LOG(INFO) << "[global split] x=" << x << " copies=" << copies[x].size() << " used=" << used
                          << " avoided=" << avoided;
            best = std::max(best, std::min(avoided, used));
        }
        return best;
    }

    // Load bound at `lambda` where every region of at least `min_nodes`
    // classes also takes its best nested case split (probeRegion) over its
    // `candidates` largest shared classes, then its best threshold split.
    double evaluateProbed(const std::vector<double> &lambda, int depth, size_t candidates, size_t min_nodes,
                          size_t with_candidates)
    {
        if (degenerate_)
            return 0.0;
        evaluate(lambda); // fills the weights and every region's L
        double total = 0.0;
        for (const Region &region : regions_)
        {
            if (region.zero_value)
                continue;
            double value = regionValue(region);
            if (region.nodes.size() >= min_nodes)
            {
                std::vector<uint8_t> forbid(region.nodes.size(), 0);
                std::vector<uint64_t> zero(region.support_words, 0);
                const double probed = probeRegion(region, forbid, zero, depth, probeCandidates(region, candidates));
                if (std::isfinite(probed))
                    value = std::max(value, probed);
                value = std::max(value, bestThresholdRegion(region, value, 40, with_candidates, 0, kNoDeadline));
            }
            total += value;
        }
        return total;
    }

    // Load bound at `lambda` with every region raised by its best threshold
    // split, nested `depth` deep with `with_candidates` sharing candidates
    // per round (docs/core/relaxation_bound.md, "Case splits"). Region values
    // proven by earlier calls at the same lambda are kept and only higher
    // thresholds are tested. Past `deadline` regions keep what is proven.
    double evaluateSplit(const std::vector<double> &lambda, int depth, size_t with_candidates,
                         std::chrono::steady_clock::time_point deadline)
    {
        if (degenerate_)
            return 0.0;
        evaluate(lambda); // fills the weights and every region's L
        if (lambda != split_lambda_)
        {
            split_lambda_ = lambda;
            split_value_.assign(regions_.size(), 0.0);
        }
        double total = 0.0;
        for (size_t ri = 0; ri < regions_.size(); ++ri)
        {
            const Region &region = regions_[ri];
            if (region.zero_value)
                continue;
            double value = std::max(regionValue(region), split_value_[ri]);
            if (std::chrono::steady_clock::now() < deadline)
                value = std::max(value, bestThresholdRegion(region, value, 40, with_candidates, depth, deadline));
            split_value_[ri] = value;
            if (std::getenv("TG_DEBUG_PROBE") && region.nodes.size() >= 64)
                LOG(INFO) << "[RelaxationBound split] depth=" << depth << " nodes=" << region.nodes.size()
                          << " plain=" << regionValue(region) << " split=" << value;
            total += value;
        }
        return total;
    }

  private:
    // Claims of a class: (region node, fraction of its own weight), sorted by
    // node. See evaluateCore.
    using Claims = std::vector<std::pair<uint32_t, double>>;

    struct AltInfo
    {
        uint32_t alt_index = 0;
        std::vector<uint32_t> kids;   // region-local indices of positive children
        std::vector<uint32_t> extras; // region-local groups with positive production cost
        std::vector<uint32_t> all_kids; // region-local indices of every child in the region
        std::vector<uint32_t> below;    // bucket class indices of the children below the region
        bool weighted = true;           // interval regions: the alternative runs inside the window
    };
    struct Region
    {
        std::vector<uint32_t> nodes; // children first
        std::vector<std::vector<AltInfo>> alts;
        std::vector<uint8_t> pos;
        size_t support_words = 0;
        std::vector<uint64_t> support; // nodes.size() * support_words
        std::vector<uint32_t> targets; // region-local indices whose min is the region value
        bool zero_value = false;       // a target lies below the region
        std::vector<std::vector<uint32_t>> group_members; // region-local
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> group_production; // (global class, alt)
        std::vector<double> L;
        std::vector<Claims> tsupport; // claims of the last evaluate()
        std::vector<double> mp;
    };

    static constexpr std::chrono::steady_clock::time_point kNoDeadline = std::chrono::steady_clock::time_point::max();

    const RelaxationBucket &rb_;
    bool ready_ = false;
    std::vector<double> split_lambda_;
    std::vector<double> split_value_; // per region, proven at split_lambda_
    // From alt_weight_ (children first).
    void fillCheapEstimates(std::vector<double> &estimates) const
    {
        estimates.assign(rb_.classes.size(), 0.0);
        for (uint32_t u : rb_.topo)
        {
            double best = std::numeric_limits<double>::infinity();
            const auto &alts = rb_.classes[u].alternatives;
            for (uint32_t a = 0; a < alts.size(); ++a)
            {
                double cost = alt_weight_[alt_offset_[u] + a];
                for (uint32_t ch : alts[a].children)
                    cost += estimates[ch];
                best = std::min(best, cost);
            }
            estimates[u] = std::isfinite(best) ? best : 0.0;
        }
    }

    std::vector<Region> regions_;
    bool degenerate_ = false;
    std::vector<double> cheap_estimates_;
    std::vector<Region> intervals_;
    std::vector<double> window_values_;
    std::vector<size_t> window_last_link_;
    std::vector<uint32_t> ichain_;                   // interval chain (groups), smallest first
    std::vector<std::vector<uint64_t>> ichain_desc_; // descendant bits of each chain group
    std::vector<uint64_t> desc_, gnd_;
    size_t wc_ = 0, wg_ = 0;
    std::vector<uint32_t> alt_offset_;
    std::vector<double> alt_weight_;

    // Per-alternative weights. A thread valuing windows in parallel sets
    // its own table (thread_weights_); everything else uses alt_weight_.
    static inline thread_local std::vector<double> *thread_weights_ = nullptr;
    std::vector<double> &weights() { return thread_weights_ ? *thread_weights_ : alt_weight_; }
    double weight(uint32_t u, uint32_t a) const
    {
        return (thread_weights_ ? *thread_weights_ : alt_weight_)[alt_offset_[u] + a];
    }

    void prepare()
    {
        using namespace relaxation_detail;
        const uint32_t n = static_cast<uint32_t>(rb_.classes.size());
        const uint32_t ng = static_cast<uint32_t>(rb_.group_members.size());
        alt_offset_.assign(n + 1, 0);
        for (uint32_t u = 0; u < n; ++u)
            alt_offset_[u + 1] = alt_offset_[u] + static_cast<uint32_t>(rb_.classes[u].alternatives.size());

        // Descendants (any alternative) and necessary groups, children first.
        const size_t wc = words(n), wg = words(ng);
        std::vector<uint64_t> desc(static_cast<size_t>(n) * wc, 0);
        std::vector<uint64_t> gnd(static_cast<size_t>(n) * wg, 0);
        std::vector<uint64_t> acc(wg), uni(wg);
        for (uint32_t u : rb_.topo)
        {
            uint64_t *du = &desc[static_cast<size_t>(u) * wc];
            setBit(du, u);
            bool first = true;
            std::fill(acc.begin(), acc.end(), 0);
            for (const auto &alt : rb_.classes[u].alternatives)
            {
                std::fill(uni.begin(), uni.end(), 0);
                for (uint32_t ch : alt.children)
                {
                    const uint64_t *dc = &desc[static_cast<size_t>(ch) * wc];
                    for (size_t w = 0; w < wc; ++w)
                        du[w] |= dc[w];
                    const uint64_t *gc = &gnd[static_cast<size_t>(ch) * wg];
                    for (size_t w = 0; w < wg; ++w)
                        uni[w] |= gc[w];
                    setBit(uni.data(), rb_.group[ch]);

                    const uint32_t orig_ch = ch < rb_.orig_class_count ? ch : rb_.orig_class_of[ch];
                    auto s_it = rb_.storage_rep.find(orig_ch);
                    if (s_it != rb_.storage_rep.end())
                    {
                        const uint32_t u_copy = s_it->second;
                        const uint64_t *dc_copy = &desc[static_cast<size_t>(u_copy) * wc];
                        for (size_t w = 0; w < wc; ++w)
                            du[w] |= dc_copy[w];
                        const uint64_t *gc_copy = &gnd[static_cast<size_t>(u_copy) * wg];
                        for (size_t w = 0; w < wg; ++w)
                            uni[w] |= gc_copy[w];
                        setBit(uni.data(), rb_.group[u_copy]);
                        setBit(du, u_copy);
                    }
                }
                if (first)
                    acc = uni;
                else
                    for (size_t w = 0; w < wg; ++w)
                        acc[w] &= uni[w];
                first = false;
            }
            uint64_t *gu = &gnd[static_cast<size_t>(u) * wg];
            std::copy(acc.begin(), acc.end(), gu);
            gu[rb_.group[u] >> 6] &= ~(1ull << (rb_.group[u] & 63));
        }

        if (std::getenv("TG_DEBUG_BOUND"))
        {
            LOG(INFO) << "[debug root alts] root_u=" << rb_.root << " alts=" << rb_.classes[rb_.root].alternatives.size();
            for (size_t a = 0; a < rb_.classes[rb_.root].alternatives.size(); ++a)
            {
                const auto &alt = rb_.classes[rb_.root].alternatives[a];
                bool alt_has_g1 = false;
                for (uint32_t ch : alt.children)
                {
                    if (rb_.group[ch] == 1 || testBit(&gnd[static_cast<size_t>(ch) * wg], 1))
                        alt_has_g1 = true;
                }
                LOG(INFO) << "  root alt " << a << " val=" << alt.value << " dur=" << alt.duration
                          << " ch_count=" << alt.children.size() << " has_g1=" << alt_has_g1;
                for (uint32_t ch : alt.children)
                {
                    LOG(INFO) << "    child u=" << ch << " grp=" << rb_.group[ch]
                              << " has_g1=" << (rb_.group[ch] == 1 || testBit(&gnd[static_cast<size_t>(ch) * wg], 1));
                }
            }
        }

        // Identify which groups are weight / storage copy groups:
        std::vector<uint8_t> is_storage_group(ng, 0);
        for (uint32_t g = 0; g < ng; ++g)
        {
            for (uint32_t m : rb_.group_members[g])
            {
                for (const auto &alt : rb_.classes[m].alternatives)
                {
                    if (alt.is_storage_copy)
                    {
                        is_storage_group[g] = 1;
                        break;
                    }
                }
                if (is_storage_group[g])
                    break;
            }
        }

        // Containment chain of the root's necessary groups.
        auto groupDesc = [&](uint32_t g) {
            std::vector<uint64_t> bits(wc, 0);
            for (uint32_t m : rb_.group_members[g])
            {
                const uint64_t *dm = &desc[static_cast<size_t>(m) * wc];
                for (size_t w = 0; w < wc; ++w)
                    bits[w] |= dm[w];
            }
            // If an external storage/weight group is NOT necessary for g,
            // it must not be included in g's region or swallowed into below!
            std::vector<uint64_t> unneeded_storage_bits(wg, 0);
            for (uint32_t s = 0; s < ng; ++s)
            {
                if (s != g && is_storage_group[s])
                {
                    bool necessary_for_g = true;
                    for (uint32_t m : rb_.group_members[g])
                    {
                        if (rb_.classes[m].alternatives.empty())
                            continue;
                        if (!testBit(&gnd[static_cast<size_t>(m) * wg], s))
                        {
                            necessary_for_g = false;
                            break;
                        }
                    }
                    if (!necessary_for_g)
                        setBit(unneeded_storage_bits.data(), s);
                }
            }
            for (uint32_t u = 0; u < n; ++u)
            {
                if (!testBit(bits.data(), u))
                    continue;
                const uint32_t gu = rb_.group[u];
                if (testBit(unneeded_storage_bits.data(), gu))
                {
                    bits[u >> 6] &= ~(1ull << (u & 63));
                    continue;
                }
                const uint64_t *gu_bits = &gnd[static_cast<size_t>(u) * wg];
                bool depends_on_unneeded = false;
                for (size_t w = 0; w < wg; ++w)
                {
                    if (gu_bits[w] & unneeded_storage_bits[w])
                    {
                        depends_on_unneeded = true;
                        break;
                    }
                }
                if (depends_on_unneeded)
                {
                    bits[u >> 6] &= ~(1ull << (u & 63));
                }
            }
            return bits;
        };
        std::vector<std::pair<size_t, uint32_t>> necessary;
        std::vector<std::vector<uint64_t>> necessary_desc(ng);
        const uint64_t *groot = &gnd[static_cast<size_t>(rb_.root) * wg];
        for (uint32_t g = 0; g < ng; ++g)
        {
            if (!testBit(groot, g))
                continue;
            necessary_desc[g] = groupDesc(g);
            size_t count = 0;
            for (uint64_t w : necessary_desc[g])
                count += static_cast<size_t>(__builtin_popcountll(w));
            necessary.push_back({count, g});
        }
        std::sort(necessary.begin(), necessary.end());
        // Longest containment chain (each group's members lie below the next
        // one, with a different descendant set), by DP over the groups in
        // order of size: one group that nothing larger contains (a greedy
        // pass would stop there) only ends the chains through it.
        const size_t nn = necessary.size();
        std::vector<uint32_t> longest(nn, 1), prev(nn, UINT32_MAX);
        for (size_t j = 0; j < nn; ++j)
        {
            const uint32_t g = necessary[j].second;
            for (size_t i = 0; i < j; ++i)
            {
                if (longest[i] + 1 <= longest[j])
                    continue;
                const uint32_t h = necessary[i].second;
                bool contains = necessary_desc[g] != necessary_desc[h];
                for (uint32_t m : rb_.group_members[h])
                    contains = contains && testBit(necessary_desc[g].data(), m);
                if (contains)
                {
                    longest[j] = longest[i] + 1;
                    prev[j] = static_cast<uint32_t>(i);
                }
            }
        }
        std::vector<uint32_t> chain;
        if (nn > 0)
        {
            size_t end = 0;
            for (size_t j = 1; j < nn; ++j)
                if (longest[j] > longest[end])
                    end = j;
            for (uint32_t j = static_cast<uint32_t>(end); j != UINT32_MAX; j = prev[j])
                chain.push_back(necessary[j].second);
            std::reverse(chain.begin(), chain.end());
        }
        if (std::getenv("TG_DEBUG_BOUND"))
        {
            LOG(INFO) << "[debug prepare] ng=" << ng
                      << " necessary_root=" << necessary.size()
                      << " chain_size=" << chain.size();
        }

        std::vector<uint64_t> below(wc, 0);
        std::vector<uint64_t> region_bits(wc, 0);
        for (size_t link = 0; link < chain.size(); ++link)
        {
            const std::vector<uint64_t> &top = necessary_desc[chain[link]];
            for (size_t w = 0; w < wc; ++w)
                region_bits[w] = top[w] & ~below[w];
            std::vector<uint32_t> targets_global = rb_.group_members[chain[link]];
            buildRegion(regions_, region_bits, below, targets_global, nullptr);
            for (size_t w = 0; w < wc; ++w)
                below[w] |= top[w];
        }

        // Before the final region: any necessary group of the root whose non-source
        // dependencies are already in `below` can be safely evaluated in its own region.
        // Because each such group is necessary for the root, every feasible plan contains
        // an active member of the group, and since its nodes are disjoint from `below`,
        // its charge is disjoint and sound.
        std::vector<uint8_t> group_done(ng, 0);
        for (uint32_t u : rb_.topo)
        {
            const uint32_t g = rb_.group[u];
            if (group_done[g] || !testBit(groot, g))
                continue;
            group_done[g] = 1;
            bool in_below = false;
            for (uint32_t m : rb_.group_members[g])
                if (testBit(below.data(), m))
                {
                    in_below = true;
                    break;
                }
            if (in_below)
                continue;

            bool ready = true;
            for (uint32_t m : rb_.group_members[g])
            {
                const uint64_t *dm = &desc[static_cast<size_t>(m) * wc];
                for (uint32_t v = 0; v < n; ++v)
                {
                    if (testBit(dm, v) && !testBit(below.data(), v) && rb_.group[v] != g)
                    {
                        bool is_src = false;
                        for (const auto &alt : rb_.classes[v].alternatives)
                            if (alt.is_source) { is_src = true; break; }
                        if (!is_src && !rb_.classes[v].alternatives.empty())
                        {
                            ready = false;
                            break;
                        }
                    }
                }
                if (!ready) break;
            }
            if (!ready)
                continue;

            std::vector<uint64_t> top(wc, 0);
            for (uint32_t m : rb_.group_members[g])
            {
                const uint64_t *dm = &desc[static_cast<size_t>(m) * wc];
                for (size_t w = 0; w < wc; ++w)
                    top[w] |= dm[w];
            }
            for (size_t w = 0; w < wc; ++w)
                region_bits[w] = top[w] & ~below[w];
            std::vector<uint32_t> targets_global = rb_.group_members[g];
            buildRegion(regions_, region_bits, below, targets_global, nullptr);
            for (size_t w = 0; w < wc; ++w)
                below[w] |= top[w];
        }

        // Final region for the root.
        {
            const std::vector<uint64_t> top(desc.begin() + static_cast<std::ptrdiff_t>(rb_.root * wc),
                                            desc.begin() + static_cast<std::ptrdiff_t>((rb_.root + 1) * wc));
            for (size_t w = 0; w < wc; ++w)
                region_bits[w] = top[w] & ~below[w];
            std::vector<uint32_t> targets_global = {rb_.root};
            buildRegion(regions_, region_bits, below, targets_global, nullptr);
            for (size_t w = 0; w < wc; ++w)
                below[w] |= top[w];
        }

        if (std::getenv("TG_DEBUG_BOUND"))
        {
            std::vector<int> group_region_count(ng, 0);
            for (const auto &reg : regions_)
            {
                if (reg.zero_value) continue;
                for (size_t lg = 0; lg < reg.group_members.size(); ++lg)
                {
                    if (reg.group_members[lg].empty()) continue;
                    uint32_t global_m = reg.nodes[reg.group_members[lg][0]];
                    uint32_t g = rb_.group[global_m];
                    group_region_count[g]++;
                }
            }
            uint32_t n_weight_groups = 0, n_weight_in_root = 0, n_weight_in_regions = 0;
            double sum_weight_min_prod = 0.0;
            for (uint32_t g = 0; g < ng; ++g)
            {
                double min_prod = 1e9;
                bool is_weight = false;
                for (uint32_t m : rb_.group_members[g])
                {
                    for (const auto &alt : rb_.classes[m].alternatives)
                    {
                        if (alt.is_storage_copy) is_weight = true;
                        if (!alt.is_copy)
                            min_prod = std::min(min_prod, alt.duration);
                    }
                }
                if (is_weight)
                {
                    n_weight_groups++;
                    if (min_prod < 1e8) sum_weight_min_prod += min_prod;
                    const bool in_root = testBit(groot, g);
                    if (in_root) n_weight_in_root++;
                    if (group_region_count[g] > 0) n_weight_in_regions++;
                    if (n_weight_groups <= 5 || min_prod == 0.0 || !in_root || group_region_count[g] == 0)
                    {
                        LOG(INFO) << "[debug weight group " << n_weight_groups << "] g=" << g
                                  << " min_prod=" << min_prod << " in_root=" << in_root
                                  << " in_regions=" << group_region_count[g]
                                  << " members=" << rb_.group_members[g].size();
                    }
                }
            }
            LOG(INFO) << "[debug weight summary] total=" << n_weight_groups
                      << " in_root=" << n_weight_in_root
                      << " in_regions=" << n_weight_in_regions
                      << " sum_min_prod=" << sum_weight_min_prod;

            for (size_t ri = 0; ri < regions_.size(); ++ri)
            {
                for (size_t lg = 0; lg < regions_[ri].group_members.size(); ++lg)
                {
                    if (regions_[ri].group_members[lg].empty()) continue;
                    uint32_t gm0 = regions_[ri].nodes[regions_[ri].group_members[lg][0]];
                    uint32_t g = rb_.group[gm0];
                    if (g == 1 || g == 34)
                    {
                        LOG(INFO) << "[debug g region] g=" << g << " in region ri=" << ri
                                  << " targets=" << regions_[ri].targets.size()
                                  << " target0=" << (regions_[ri].targets.empty() ? 999999 : regions_[ri].nodes[regions_[ri].targets[0]]);
                    }
                }
            }
        }

        // Interval chain: every member of the next link necessarily depends on
        // the previous link, so the earliest finishes t_i are nondecreasing.
        // Any subsequence is again such a chain (necessary groups are
        // transitive), which intervalBound uses to form coarser windows.
        ichain_.clear();
        for (const auto &[count, g] : necessary)
        {
            if (!ichain_.empty())
            {
                const uint32_t last = ichain_.back();
                if (necessary_desc[g] == necessary_desc[last])
                    continue;
                bool after = true;
                for (uint32_t m : rb_.group_members[g])
                    if (!testBit(&gnd[static_cast<size_t>(m) * wg], last))
                        after = false;
                if (!after)
                    continue;
            }
            ichain_.push_back(g);
        }
        ichain_desc_.clear();
        for (uint32_t g : ichain_)
            ichain_desc_.push_back(std::move(necessary_desc[g]));
        desc_ = std::move(desc);
        gnd_ = std::move(gnd);
        wc_ = wc;
        wg_ = wg;
    }

    // Windows for every `stride`-th link of the interval chain (and the root).
    // A class runs inside window i when it necessarily depends on link i-1;
    // only classes with a member of link i-1 below them are kept.
    void buildIntervals(size_t stride, size_t phase = 0)
    {
        using namespace relaxation_detail;
        if (stride == 0)
            Error::throw_err("Interval bound stride must be positive");
        intervals_.clear();
        const uint32_t n = static_cast<uint32_t>(rb_.classes.size());
        std::vector<size_t> links;
        for (size_t i = 0; i < ichain_.size(); ++i)
            if ((i + 1) % stride == phase % stride || i + 1 == ichain_.size())
                links.push_back(i);
        window_last_link_.assign(links.begin(), links.end());
        const std::vector<uint64_t> none(wc_, 0);
        std::vector<uint64_t> region_bits(wc_, 0);
        std::vector<uint8_t> weighted(alt_offset_.back(), 1);
        for (size_t l = 0; l <= links.size(); ++l)
        {
            const bool final_region = l == links.size();
            const uint64_t *top = final_region ? &desc_[static_cast<size_t>(rb_.root) * wc_]
                                               : ichain_desc_[links[l]].data();
            std::fill(region_bits.begin(), region_bits.end(), 0);
            if (l == 0)
            {
                std::fill(weighted.begin(), weighted.end(), 1);
                for (size_t w = 0; w < wc_; ++w)
                    region_bits[w] = top[w];
            }
            else
            {
                const uint32_t prev = ichain_[links[l - 1]];
                for (uint32_t u = 0; u < n; ++u)
                {
                    const uint8_t after = testBit(&gnd_[static_cast<size_t>(u) * wg_], prev) ? 1 : 0;
                    for (uint32_t a = 0; a < rb_.classes[u].alternatives.size(); ++a)
                        weighted[alt_offset_[u] + a] = after;
                    if (!testBit(top, u))
                        continue;
                    const uint64_t *du = &desc_[static_cast<size_t>(u) * wc_];
                    for (uint32_t m : rb_.group_members[prev])
                        if (testBit(du, m))
                        {
                            setBit(region_bits.data(), u);
                            break;
                        }
                }
            }
            std::vector<uint32_t> targets_global =
                final_region ? std::vector<uint32_t>{rb_.root} : rb_.group_members[ichain_[links[l]]];
            buildRegion(intervals_, region_bits, none, targets_global, &weighted);
        }
    }

    void fillRegionWeights(const Region &region, const std::vector<double> &lambda)
    {
        for (size_t i = 0; i < region.nodes.size(); ++i)
        {
            const uint32_t u = region.nodes[i];
            const auto &alts = rb_.classes[u].alternatives;
            for (uint32_t a = 0; a < alts.size(); ++a)
            {
                double scale = 0.0;
                if (region.alts[i][a].weighted)
                    for (uint32_t e = 0; e < lambda.size(); ++e)
                        if (alts[a].engine_mask & (1u << e))
                            scale += lambda[e];
                weights()[alt_offset_[u] + a] = alts[a].duration * scale;
            }
        }
    }

    // Longest chain of window classes: for every plan in which the class is
    // active and the value is positive, the class finishes at least this long
    // after the window opens (children finish before their consumer starts).
    double criticalPathRegion(const Region &region) const
    {
        std::vector<double> cp;
        return criticalPathRegion(region, cp);
    }
    double criticalPathRegion(const Region &region, std::vector<double> &cp) const
    {
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        cp.assign(rn, 0.0);
        for (uint32_t i = 0; i < rn; ++i)
        {
            const auto &alts = rb_.classes[region.nodes[i]].alternatives;
            double best = std::numeric_limits<double>::infinity();
            for (const AltInfo &info : region.alts[i])
            {
                double below = 0.0;
                for (uint32_t k : info.all_kids)
                    below = std::max(below, cp[k]);
                const double own = info.weighted && alts[info.alt_index].engine_mask
                                       ? alts[info.alt_index].duration
                                       : 0.0;
                best = std::min(best, own + below);
            }
            cp[i] = std::isfinite(best) ? best : 0.0;
        }
        double value = std::numeric_limits<double>::infinity();
        for (uint32_t t : region.targets)
            value = std::min(value, cp[t]);
        return std::isfinite(value) ? value : 0.0;
    }

    void buildRegion(std::vector<Region> &out, const std::vector<uint64_t> &region_bits,
                     const std::vector<uint64_t> &below, const std::vector<uint32_t> &targets_global,
                     const std::vector<uint8_t> *weighted)
    {
        using namespace relaxation_detail;
        Region region;
        for (uint32_t u : rb_.topo)
            if (testBit(region_bits.data(), u))
            {
                region.nodes.push_back(u);
                if (std::getenv("TG_DEBUG_BOUND") && (u == 1 || u == 7825))
                {
                    LOG(INFO) << "[debug node in region] u=" << u << " region=" << out.size()
                              << " targets_count=" << targets_global.size()
                              << " target0=" << (targets_global.empty() ? 999999 : targets_global[0]);
                }
            }
        for (uint32_t t : targets_global)
            if (testBit(below.data(), t))
                region.zero_value = true;
        if (region.nodes.empty() || region.zero_value)
        {
            region.zero_value = true;
            out.push_back(std::move(region));
            return;
        }
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        std::unordered_map<uint32_t, uint32_t> local;
        local.reserve(rn * 2);
        for (uint32_t i = 0; i < rn; ++i)
            local.emplace(region.nodes[i], i);
        for (uint32_t t : targets_global)
        {
            auto it = local.find(t);
            if (it == local.end())
                Error::throw_err("Relaxation bound: chain target outside its region");
            region.targets.push_back(it->second);
        }

        // Groups whose members all lie in the region (and, for an interval,
        // run inside its window); only they may be charged here, so different
        // regions never count the same class.
        std::unordered_map<uint32_t, uint32_t> local_group;
        for (uint32_t i = 0; i < rn; ++i)
        {
            const uint32_t g = rb_.group[region.nodes[i]];
            if (local_group.count(g))
                continue;
            bool inside = true;
            for (uint32_t m : rb_.group_members[g])
            {
                if (!local.count(m))
                    inside = false;
                else if (weighted)
                    for (uint32_t a = 0; a < rb_.classes[m].alternatives.size(); ++a)
                        if (!rb_.classes[m].alternatives[a].is_copy && !(*weighted)[alt_offset_[m] + a])
                            inside = false;
            }
            if (!inside)
                continue;
            const uint32_t lg = static_cast<uint32_t>(region.group_members.size());
            local_group.emplace(g, lg);
            region.group_members.emplace_back();
            region.group_production.emplace_back();
            for (uint32_t m : rb_.group_members[g])
            {
                region.group_members[lg].push_back(local.at(m));
                const auto &alts = rb_.classes[m].alternatives;
                for (uint32_t a = 0; a < alts.size(); ++a)
                    if (!alts[a].is_copy)
                        region.group_production[lg].push_back({m, a});
            }
        }
        const uint32_t lgn = static_cast<uint32_t>(region.group_members.size());
        // A group's cheapest production is positive for every interior lambda
        // exactly when all its production alternatives have positive duration.
        std::vector<uint8_t> group_pos(lgn, 1);
        for (uint32_t lg = 0; lg < lgn; ++lg)
        {
            if (region.group_production[lg].empty())
                group_pos[lg] = 1; // no production: never active, any charge is vacuous
            for (const auto &[m, a] : region.group_production[lg])
                if (!(rb_.classes[m].alternatives[a].duration > 0.0 && rb_.classes[m].alternatives[a].engine_mask))
                    group_pos[lg] = 0;
        }

        const size_t wl = words(lgn);
        std::vector<uint64_t> gloc(static_cast<size_t>(rn) * wl, 0);
        std::vector<uint64_t> acc(wl), uni(wl);
        region.alts.resize(rn);
        region.pos.assign(rn, 0);
        for (uint32_t i = 0; i < rn; ++i)
        {
            const uint32_t u = region.nodes[i];
            const auto &alts = rb_.classes[u].alternatives;
            const auto own_it = local_group.find(rb_.group[u]);
            bool first = true;
            bool pos = true;
            for (uint32_t a = 0; a < alts.size(); ++a)
            {
                std::fill(uni.begin(), uni.end(), 0);
                AltInfo info;
                info.alt_index = a;
                info.weighted = weighted ? (*weighted)[alt_offset_[u] + a] != 0 : true;
                for (uint32_t ch : alts[a].children)
                {
                    auto it = local.find(ch);
                    if (std::getenv("TG_DEBUG_BOUND") && (u == 3659 || u == 6810 || u == 8980 || u == 9030 || u == 9050 || u == 7825 || u == 7841 || u == 8979))
                    {
                        LOG(INFO) << "[debug child of " << u << "] ch=" << ch
                                  << " in_local=" << (it != local.end())
                                  << " grp=" << rb_.group[ch]
                                  << " in_local_grp=" << (local_group.count(rb_.group[ch]) ? (int)local_group.at(rb_.group[ch]) : -1);
                    }
                    if (it == local.end())
                    {
                        // Below the region: counted by an earlier region.
                        if (std::find(info.below.begin(), info.below.end(), ch) == info.below.end())
                            info.below.push_back(ch);
                        continue;
                    }
                    const uint32_t li = it->second;
                    if (std::find(info.all_kids.begin(), info.all_kids.end(), li) == info.all_kids.end())
                        info.all_kids.push_back(li);
                    const uint64_t *gc = &gloc[static_cast<size_t>(li) * wl];
                    for (size_t w = 0; w < wl; ++w)
                        uni[w] |= gc[w];
                    auto g_it = local_group.find(rb_.group[ch]);
                    if (g_it != local_group.end())
                        setBit(uni.data(), g_it->second);
                    if (region.pos[li])
                        info.kids.push_back(li);
                }
                if (own_it != local_group.end())
                    uni[own_it->second >> 6] &= ~(1ull << (own_it->second & 63));
                for (uint32_t lg = 0; lg < lgn; ++lg)
                    if (testBit(uni.data(), lg) && group_pos[lg])
                        info.extras.push_back(lg);
                const bool alt_pos = (info.weighted && alts[a].duration > 0.0 && alts[a].engine_mask) ||
                                     !info.kids.empty() || !info.extras.empty();
                if (std::getenv("TG_DEBUG_BOUND") && (u == 7825 || u == 8742 || u == 3659 || u == 6810 || u == 6812 || u == 3660 || u == 7082 || u == 7083 || u == 8313 || u == 8854 || u == 7394 || u == 7395 || u == 8980 || u == 9030 || u == 9050 || u == 7841 || u == 8979))
                {
                    LOG(INFO) << "[debug alt detail] u=" << u << " alt=" << a
                              << " dur=" << alts[a].duration << " mask=" << alts[a].engine_mask
                              << " is_copy=" << alts[a].is_copy << " is_alias=" << alts[a].is_alias
                              << " is_view=" << alts[a].is_view << " is_source=" << alts[a].is_source
                              << " ch=" << alts[a].children.size() << " kids=" << info.kids.size()
                              << " extras=" << info.extras.size() << " weighted=" << info.weighted
                              << " alt_pos=" << alt_pos << " uni_has_0=" << (int)testBit(uni.data(), 0);
                }
                if (!alt_pos)
                {
                    pos = false;
                    if (std::getenv("TG_DEBUG_BOUND") && alts[a].duration > 0.05)
                    {
                        LOG(INFO) << "[debug alt_pos false] u=" << u << " alt=" << a
                                  << " dur=" << alts[a].duration << " weighted=" << info.weighted
                                  << " mask=" << alts[a].engine_mask;
                    }
                }
                if (first)
                    acc = uni;
                else
                    for (size_t w = 0; w < wl; ++w)
                        acc[w] &= uni[w];
                first = false;
                region.alts[i].push_back(std::move(info));
            }
            if (alts.empty())
                pos = false;
            std::copy(acc.begin(), acc.end(), &gloc[static_cast<size_t>(i) * wl]);
            region.pos[i] = pos ? 1 : 0;
            if (std::getenv("TG_DEBUG_BOUND") && (u == 3659 || u == 6810 || u == 6812 || u == 3660 || u == 7082 || u == 7083 || u == 8313 || u == 8854 || u == 7394 || u == 7395))
            {
                LOG(INFO) << "[debug gloc assigned] u=" << u << " i=" << i << " pos=" << (int)region.pos[i]
                          << " acc_has_0=" << (int)testBit(&gloc[static_cast<size_t>(i) * wl], 0);
            }
            if (std::getenv("TG_DEBUG_BOUND") && !region.pos[i])
            {
                for (const auto &alt : alts)
                {
                    if (alt.duration > 0.1)
                    {
                        LOG(INFO) << "[debug pos 0] class u=" << u << " in region has heavy alt dur=" << alt.duration
                                  << " total_alts=" << alts.size() << " but pos=0!";
                    }
                }
            }
        }

        // Supports: a positive class, every positive child's support and every
        // charged group's members, over all alternatives.
        region.support_words = words(rn);
        const size_t ws = region.support_words;
        region.support.assign(static_cast<size_t>(rn) * ws, 0);
        for (uint32_t i = 0; i < rn; ++i)
        {
            if (!region.pos[i])
                continue;
            uint64_t *si = &region.support[static_cast<size_t>(i) * ws];
            setBit(si, i);
            for (const AltInfo &info : region.alts[i])
            {
                for (uint32_t k : info.kids)
                {
                    const uint64_t *sk = &region.support[static_cast<size_t>(k) * ws];
                    for (size_t w = 0; w < ws; ++w)
                        si[w] |= sk[w];
                }
                for (uint32_t lg : info.extras)
                    for (uint32_t m : region.group_members[lg])
                        setBit(si, m);
            }
        }
        region.L.assign(rn, 0.0);
        region.mp.assign(lgn, 0.0);
        out.push_back(std::move(region));
    }

    std::vector<std::pair<size_t, size_t>> last_partition_;
    std::vector<double> last_partition_values_;

    // Per window start: alternatives running after the opening link
    // (weighted) and the classes with a member of the opening link below
    // them.
    struct WindowScratch
    {
        std::vector<uint8_t> weighted;
        std::vector<uint64_t> below_prev;
        bool from_start = true;
    };

    void prepareWindowStart(size_t a1, WindowScratch &ws) const
    {
        using namespace relaxation_detail;
        const uint32_t n = static_cast<uint32_t>(rb_.classes.size());
        ws.weighted.assign(alt_offset_.back(), 1);
        ws.below_prev.assign(wc_, 0);
        ws.from_start = a1 == 0;
        if (ws.from_start)
            return;
        const uint32_t prev = ichain_[a1 - 1];
        for (uint32_t u = 0; u < n; ++u)
        {
            const uint8_t after = testBit(&gnd_[static_cast<size_t>(u) * wg_], prev) ? 1 : 0;
            for (uint32_t a = 0; a < rb_.classes[u].alternatives.size(); ++a)
                ws.weighted[alt_offset_[u] + a] = after;
            const uint64_t *du = &desc_[static_cast<size_t>(u) * wc_];
            for (uint32_t m : rb_.group_members[prev])
                if (testBit(du, m))
                {
                    setBit(ws.below_prev.data(), u);
                    break;
                }
        }
    }

    // Builds window (a1, b) into out[0]; false when it exceeds
    // kFlexibleMaxClasses (it is then worth 0).
    bool buildWindow(size_t a1, size_t b, const WindowScratch &ws, std::vector<Region> &out)
    {
        const size_t k = ichain_.size();
        const bool final_region = b == k;
        const uint64_t *top = final_region ? &desc_[static_cast<size_t>(rb_.root) * wc_] : ichain_desc_[b].data();
        std::vector<uint64_t> region_bits(wc_);
        size_t size = 0;
        for (size_t w = 0; w < wc_; ++w)
        {
            region_bits[w] = ws.from_start ? top[w] : (top[w] & ws.below_prev[w]);
            size += static_cast<size_t>(__builtin_popcountll(region_bits[w]));
        }
        if (size > kFlexibleMaxClasses)
            return false;
        out.clear();
        const std::vector<uint64_t> none(wc_, 0);
        buildRegion(out, region_bits, none, final_region ? std::vector<uint32_t>{rb_.root} : rb_.group_members[ichain_[b]],
                    &ws.weighted);
        return true;
    }

    // Local search on a window selection (primal use; the window value is
    // the exact enumeration's: max(chain, engine loads, one-machine values)
    // of the classes below the target). Single-class switches; classes a
    // switch newly needs take their least-work alternative (recursively),
    // classes no longer needed drop; a switch is kept when the value drops.
    // Returns the final value.
    double improveWindowChoice(const Region &region, std::vector<int32_t> &choice, double value) const
    {
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        const size_t ne = rb_.engines.size();
        uint32_t target = UINT32_MAX;
        for (uint32_t t : region.targets)
            if (choice[t] >= 0 && (target == UINT32_MAX || t > target))
                target = t;
        if (target == UINT32_MAX)
            return value;
        auto own = [&](uint32_t i, int32_t a) {
            const AltInfo &info = region.alts[i][static_cast<size_t>(a)];
            const auto &alt = rb_.classes[region.nodes[i]].alternatives[info.alt_index];
            return info.weighted && alt.engine_mask ? alt.duration : 0.0;
        };
        auto mask = [&](uint32_t i, int32_t a) {
            return rb_.classes[region.nodes[i]].alternatives[region.alts[i][static_cast<size_t>(a)].alt_index].engine_mask;
        };
        // Classes that add nothing for any selection stay unchosen (the
        // plan keeps its own choice there), as in the exact enumeration.
        std::vector<uint8_t> nil(rn, 0);
        for (uint32_t i = 0; i < rn; ++i) // children first
        {
            bool all_nil = !region.alts[i].empty() &&
                           std::find(region.targets.begin(), region.targets.end(), i) == region.targets.end();
            for (size_t a = 0; a < region.alts[i].size() && all_nil; ++a)
            {
                all_nil = own(i, static_cast<int32_t>(a)) == 0.0;
                for (uint32_t k : region.alts[i][a].all_kids)
                    all_nil = all_nil && nil[k];
            }
            nil[i] = all_nil;
        }
        std::vector<double> least_work(rn, 0.0);
        std::vector<int32_t> least_alt(rn, -1);
        for (uint32_t i = 0; i < rn; ++i) // children first
        {
            double v = std::numeric_limits<double>::infinity();
            for (size_t a = 0; a < region.alts[i].size(); ++a)
            {
                double w = own(i, static_cast<int32_t>(a));
                for (uint32_t k : region.alts[i][a].all_kids)
                    w += least_work[k];
                if (w < v)
                {
                    v = w;
                    least_alt[i] = static_cast<int32_t>(a);
                }
            }
            least_work[i] = std::isfinite(v) ? v : 0.0;
        }
        std::vector<double> finish(rn), tail(rn), loads(ne);
        std::vector<uint8_t> reached(rn);
        // Completes `c` from the target (least-work alternatives for classes
        // without one), drops what is not reached; returns its value.
        auto settle = [&](std::vector<int32_t> &c) {
            std::fill(reached.begin(), reached.end(), 0);
            reached[target] = 1;
            for (uint32_t i = target + 1; i-- > 0;) // parents first
            {
                if (!reached[i] || nil[i])
                {
                    c[i] = -1;
                    continue;
                }
                if (c[i] < 0)
                    c[i] = least_alt[i];
                if (c[i] < 0)
                    return std::numeric_limits<double>::infinity();
                for (uint32_t k : region.alts[i][static_cast<size_t>(c[i])].all_kids)
                    reached[k] = 1;
            }
            for (uint32_t i = target + 1; i < rn; ++i)
                c[i] = -1;
            std::fill(loads.begin(), loads.end(), 0.0);
            double chain = 0.0;
            for (uint32_t i = 0; i <= target; ++i) // children first
            {
                if (c[i] < 0)
                    continue;
                double below = 0.0;
                for (uint32_t k : region.alts[i][static_cast<size_t>(c[i])].all_kids)
                    below = std::max(below, finish[k]);
                const double d = own(i, c[i]);
                finish[i] = below + d;
                for (size_t e = 0; e < ne; ++e)
                    if (d > 0.0 && (mask(i, c[i]) & (1u << e)))
                        loads[e] += d;
                chain = std::max(chain, finish[i]);
            }
            std::fill(tail.begin(), tail.end(), -1.0);
            tail[target] = 0.0;
            for (uint32_t i = target + 1; i-- > 0;)
            {
                if (c[i] < 0 || tail[i] < 0.0)
                    continue;
                const double up = tail[i] + own(i, c[i]);
                for (uint32_t k : region.alts[i][static_cast<size_t>(c[i])].all_kids)
                    tail[k] = std::max(tail[k], up);
            }
            double v = chain;
            for (size_t e = 0; e < ne; ++e)
            {
                double head = std::numeric_limits<double>::infinity(), last = head;
                for (uint32_t i = 0; i <= target; ++i)
                {
                    if (c[i] < 0 || tail[i] < 0.0)
                        continue;
                    const double d = own(i, c[i]);
                    if (!(d > 0.0) || !(mask(i, c[i]) & (1u << e)))
                        continue;
                    head = std::min(head, finish[i] - d);
                    last = std::min(last, tail[i]);
                }
                v = std::max(v, loads[e] > 0.0 && std::isfinite(head) ? head + loads[e] + last : loads[e]);
            }
            return v;
        };
        std::vector<int32_t> trial;
        for (bool improved = true; improved;)
        {
            improved = false;
            for (uint32_t i = 0; i <= target && !improved; ++i)
            {
                if (choice[i] < 0)
                    continue;
                for (size_t a = 0; a < region.alts[i].size(); ++a)
                {
                    if (static_cast<int32_t>(a) == choice[i])
                        continue;
                    trial = choice;
                    trial[i] = static_cast<int32_t>(a);
                    // Classes only the old alternative needed choose afresh.
                    for (uint32_t k = 0; k < i; ++k)
                        trial[k] = choice[k];
                    const double v = settle(trial);
                    if (v < value - 1e-12)
                    {
                        choice.swap(trial);
                        value = v;
                        improved = true;
                        break;
                    }
                }
            }
        }
        return value;
    }

    // Value of one interval window: the exact window value when the
    // enumeration finishes, otherwise the best of its critical path, the
    // lambda / mu path-weighted values and the cut DP.
    double windowValue(Region &region)
    {
        if (region.zero_value)
            return 0.0;
        const double exact = exactWindow(region, exact_window_nodes_);
        if (exact > 0.0)
            return exact;
        const size_t ne = rb_.engines.size();
        if (ne == 0)
            return 0.0;
        weights().resize(alt_offset_.back());
        std::vector<double> lambda(ne), cp_nodes, path;
        const double cp = criticalPathRegion(region, cp_nodes);
        static const double kMu[] = {0.0, 0.2, 0.4, 0.6, 0.8, 0.9};
        double best = cp;
        for (double l0 : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0})
        {
            if (ne == 1)
                lambda[0] = 1.0;
            else
            {
                lambda[0] = l0;
                for (size_t e = 1; e < ne; ++e)
                    lambda[e] = (1.0 - l0) / static_cast<double>(ne - 1);
            }
            fillRegionWeights(region, lambda);
            best = std::max(best, evaluateRegion(region));
            for (double mu : kMu)
                if (mu > 0.0)
                    best = std::max(best, evaluateRegionPath(region, mu, cp_nodes, path));
        }
        return region.nodes.size() > kCutDpMaxClasses ? best : cutDpWindow(region, best, kCutDpStates, kCutDpPoints);
    }

    // Value of a region after evaluate(): its cheapest target.
    static double regionValue(const Region &region)
    {
        double value = std::numeric_limits<double>::infinity();
        for (uint32_t t : region.targets)
            value = std::min(value, region.L[t]);
        return std::isfinite(value) ? value : 0.0;
    }

    double evaluateRegion(Region &region)
    {
        if (region.zero_value)
            return 0.0;
        // The interval bound's path recursion reads the groups' cheapest
        // production.
        const uint32_t lgn = static_cast<uint32_t>(region.group_members.size());
        region.mp.assign(lgn, 0.0);
        for (uint32_t lg = 0; lg < lgn; ++lg)
        {
            double best = std::numeric_limits<double>::infinity();
            for (const auto &[m, a] : region.group_production[lg])
                best = std::min(best, weight(m, a));
            region.mp[lg] = std::isfinite(best) ? best : 0.0;
        }
        const double value = evaluateCore(region, nullptr, nullptr, region.L, region.tsupport);
        return std::isfinite(value) ? value : 0.0;
    }

    // The region recursion restricted to a branch of case splits
    // (docs/core/relaxation_bound.md, "Probing"): plans that select no class
    // of `forbid` (alternatives with a forbidden or infeasible child are
    // skipped), counting only classes outside `zero` (they take L = 0 and
    // are removed from every support). Fills L and the trimmed supports per
    // region node and returns the region value, +inf when no target is
    // feasible.
    double evaluateRegionMasked(const Region &region, const std::vector<uint8_t> &forbid,
                                const std::vector<uint64_t> &zero, std::vector<double> &L,
                                std::vector<Claims> &support, std::vector<double> *overlap_loss = nullptr) const
    {
        return evaluateCore(region, &forbid, &zero, L, support, overlap_loss);
    }

    // The region recursion (class docs above) with fractional supports.
    // Every counted class c gets L(c) and claims phi_c (a fraction in [0, 1]
    // of each region class's own weight) with the invariant
    //   L(c) <= sum_x phi_c(x) * w_plan(x)  for every plan of the branch with
    //   c active (w_plan(x) = duration of x's selected alternative, 0 when x
    //   is inactive).
    // For an alternative a, children are packed greedily (every child tried
    // first): an anchor k is taken at the largest alpha <= 1 that keeps the
    // cover (the sum of the claims taken so far) at most 1 on every class,
    // adding alpha * L(k); the unclaimed fraction g of a child's own weight
    // adds g times its cheapest weight; a charged group G (an active member
    // produces it) adds gamma * mp(G) for the fraction gamma free on all its
    // members. L(c) = min over usable alternatives of w(a) + the best
    // packing. For a plan selecting a, the packing's claims are at most 1 per
    // class, so w(a) plus the packing is at most sum_x phi(x) w_plan(x).
    // phi_c is 1 on c plus the componentwise maximum over alternatives of
    // their packings scaled to what they must justify (L(c) - w(a), nothing
    // when w(a) >= L(c)). Set supports are the special case phi in {0, 1};
    // fractions let a class shared by several children (a slab or a GPU copy
    // each child needs only a little of) count for all of them.
    // With `overlap_loss`, the class that limited an anchor's alpha
    // accumulates the value that anchor lost.
    double evaluateCore(const Region &region, const std::vector<uint8_t> *forbid, const std::vector<uint64_t> *zero,
                        std::vector<double> &L, std::vector<Claims> &support,
                        std::vector<double> *overlap_loss = nullptr,
                        const std::vector<uint8_t> *infeasible_below = nullptr) const
    {
        using namespace relaxation_detail;
        constexpr double kEps = 1e-12;
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        const uint32_t lgn = static_cast<uint32_t>(region.group_members.size());
        auto zeroed = [&](uint32_t i) { return zero && testBit(zero->data(), i); };
        std::vector<double> mp(lgn, 0.0);
        for (uint32_t lg = 0; lg < lgn; ++lg)
        {
            double best = std::numeric_limits<double>::infinity();
            for (const auto &[m, a] : region.group_production[lg])
                best = std::min(best, weight(m, a));
            mp[lg] = std::isfinite(best) ? best : 0.0;
        }
        std::vector<double> min_weight(rn, 0.0);
        for (uint32_t i = 0; i < rn; ++i)
        {
            double mw = std::numeric_limits<double>::infinity();
            for (const AltInfo &info : region.alts[i])
                mw = std::min(mw, weight(region.nodes[i], info.alt_index));
            min_weight[i] = std::isfinite(mw) ? mw : 0.0;
        }
        std::vector<uint8_t> feasible(rn, 0);
        // Dense scratch: the cover of the current packing, the claims of the
        // best packing of the current alternative, and the class's claims.
        std::vector<double> cover(rn, 0.0), packed(rn, 0.0), best_packed(rn, 0.0), merged(rn, 0.0);
        std::vector<uint32_t> cover_touched, packed_touched, best_touched, merged_touched;
        std::vector<uint32_t> order;
        L.assign(rn, 0.0);
        support.assign(rn, Claims{});
        if (overlap_loss)
            overlap_loss->assign(rn, 0.0);
        auto touch = [](std::vector<double> &dense, std::vector<uint32_t> &touched, uint32_t x, double add) {
            if (dense[x] == 0.0)
                touched.push_back(x);
            dense[x] += add;
        };
        auto clear = [](std::vector<double> &dense, std::vector<uint32_t> &touched) {
            for (uint32_t x : touched)
                dense[x] = 0.0;
            touched.clear();
        };
        for (uint32_t i = 0; i < rn; ++i)
        {
            if (forbid && (*forbid)[i])
                continue;
            const uint32_t u = region.nodes[i];
            // Every class not zeroed is counted; one with a free usable
            // alternative and no positive children gets L = 0 by itself.
            const bool counted = !zeroed(i);
            double best = std::numeric_limits<double>::infinity();
            clear(merged, merged_touched);
            // Usable alternatives' weights and packings, to scale afterwards.
            struct AltPacking
            {
                double w = 0.0;
                double value = 0.0;
                Claims claims;
            };
            std::vector<AltPacking> packings;
            for (const AltInfo &info : region.alts[i])
            {
                bool usable = true;
                for (uint32_t k : info.all_kids)
                    usable = usable && feasible[k];
                if (infeasible_below)
                    for (uint32_t c : info.below)
                        usable = usable && !(*infeasible_below)[c];
                if (!usable)
                    continue;
                feasible[i] = 1;
                if (!counted)
                    continue;
                double cand = 0.0;
                clear(best_packed, best_touched);
                order.clear();
                for (uint32_t k : info.all_kids)
                    if (L[k] > 0.0 || min_weight[k] > 0.0)
                        order.push_back(k);
                if (!order.empty() || !info.extras.empty())
                {
                    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return L[x] > L[y]; });
                    const size_t tries = std::max<size_t>(1, order.size());
                    cand = -1.0;
                    for (size_t first = 0; first < tries; ++first)
                    {
                        clear(cover, cover_touched);
                        double sum = 0.0;
                        for (size_t step = 0; step < order.size(); ++step)
                        {
                            const uint32_t k = order[step == 0 ? first : (step <= first ? step - 1 : step)];
                            if (zeroed(k))
                                continue;
                            double alpha = 1.0;
                            uint32_t limiter = UINT32_MAX;
                            for (const auto &[x, f] : support[k])
                            {
                                const double room = std::max(0.0, 1.0 - cover[x]) / f;
                                if (room < alpha)
                                {
                                    alpha = room;
                                    limiter = x;
                                }
                            }
                            if (alpha > kEps)
                            {
                                sum += alpha * L[k];
                                for (const auto &[x, f] : support[k])
                                    touch(cover, cover_touched, x, alpha * f);
                            }
                            else
                                alpha = 0.0;
                            if (overlap_loss && first == 0 && limiter != UINT32_MAX && alpha < 1.0)
                                (*overlap_loss)[limiter] += (1.0 - alpha) * L[k];
                            // The fraction of k's own weight still free.
                            const double g = std::max(0.0, 1.0 - cover[k]);
                            if (g > kEps && min_weight[k] > 0.0)
                            {
                                sum += g * min_weight[k];
                                touch(cover, cover_touched, k, g);
                            }
                        }
                        for (uint32_t lg : info.extras)
                        {
                            double gamma = 1.0;
                            for (uint32_t m : region.group_members[lg])
                                gamma = zeroed(m) ? 0.0 : std::min(gamma, std::max(0.0, 1.0 - cover[m]));
                            if (gamma > kEps && mp[lg] > 0.0)
                            {
                                sum += gamma * mp[lg];
                                for (uint32_t m : region.group_members[lg])
                                    touch(cover, cover_touched, m, gamma);
                            }
                        }
                        if (sum > cand)
                        {
                            cand = sum;
                            clear(best_packed, best_touched);
                            for (uint32_t x : cover_touched)
                                touch(best_packed, best_touched, x, cover[x]);
                        }
                    }
                }
                const double w = weight(u, info.alt_index);
                best = std::min(best, w + cand);
                AltPacking packing;
                packing.w = w;
                packing.value = std::max(cand, 0.0);
                for (uint32_t x : best_touched)
                    packing.claims.push_back({x, best_packed[x]});
                packings.push_back(std::move(packing));
            }
            if (!counted || !feasible[i])
                continue;
            L[i] = best;
            if (!(best > 0.0))
                continue; // L = 0 needs no claims
            touch(merged, merged_touched, i, 1.0);
            for (const AltPacking &packing : packings)
            {
                const double need = best - packing.w;
                if (!(need > 0.0) || !(packing.value > 0.0))
                    continue;
                // Scale the packing to what this alternative must justify;
                // keep all of it when float error makes it barely enough.
                const double scale = need >= packing.value * (1.0 - 1e-9) ? 1.0 : need / packing.value;
                for (const auto &[x, f] : packing.claims)
                {
                    const double claim = std::min(1.0, f * scale);
                    if (claim > merged[x])
                    {
                        if (merged[x] == 0.0)
                            merged_touched.push_back(x);
                        merged[x] = claim;
                    }
                }
            }
            std::sort(merged_touched.begin(), merged_touched.end());
            Claims &claims = support[i];
            for (uint32_t x : merged_touched)
                if (merged[x] > 0.0 && !zeroed(x))
                    claims.push_back({x, merged[x]});
        }
        double value = std::numeric_limits<double>::infinity();
        for (uint32_t t : region.targets)
            if (feasible[t])
                value = std::min(value, L[t]);
        return value;
    }

    // Marks every class claimed by `claims` in `zero`.
    static void zeroClaims(const Claims &claims, std::vector<uint64_t> &zero)
    {
        for (const auto &[x, f] : claims)
            relaxation_detail::setBit(zero.data(), x);
    }

    // Best case split over `candidates`, nested `depth` deep: every plan of
    // the branch (forbid, zero) either selects x, and then counts at least
    // L(x) on S(x) plus the branch value with S(x) zeroed (disjoint classes),
    // or avoids x, and then is a plan of the branch with x forbidden.
    double probeRegion(const Region &region, std::vector<uint8_t> &forbid, std::vector<uint64_t> &zero, int depth,
                       const std::vector<uint32_t> &candidates) const
    {
        std::vector<double> L;
        std::vector<Claims> support;
        const double base = evaluateRegionMasked(region, forbid, zero, L, support);
        if (depth <= 0 || !std::isfinite(base))
            return base;
        const size_t ws = region.support_words;
        double best = base;
        for (uint32_t x : candidates)
        {
            if (forbid[x] || relaxation_detail::testBit(zero.data(), x) || !(L[x] > 0.0))
                continue;
            forbid[x] = 1;
            const double without = probeRegion(region, forbid, zero, depth - 1, candidates);
            forbid[x] = 0;
            if (!(without > best))
                continue; // min(without, with) cannot beat best
            const std::vector<uint64_t> saved = zero;
            zeroClaims(support[x], zero);
            const double with = L[x] + probeRegion(region, forbid, zero, depth - 1, candidates);
            zero = saved;
            best = std::max(best, std::min(without, with));
        }
        return best;
    }

    // Threshold split (docs/core/relaxation_bound.md, "Threshold split"): a
    // plan of the branch that avoids F_j and selects a class x with
    // L_{F_j}(x) >= theta counts at least theta on S(x), inside the region.
    // So with F_0 = {} and F_{j+1} = F_j plus every class whose L under F_j
    // reaches theta, every plan either counts theta or avoids the final F,
    // and the region value is at least min(theta, value under the final F).
    double thresholdRegion(const Region &region, double theta, size_t max_with, int depth,
                           std::chrono::steady_clock::time_point deadline) const
    {
        std::vector<uint8_t> forbid(region.nodes.size(), 0);
        const std::vector<uint64_t> zero(region.support_words, 0);
        return thresholdBranch(region, theta, max_with, depth, forbid, zero, deadline);
    }

    // The threshold split inside the branch (forbid, zero); `forbid` is
    // extended in place. A class's "with" test L(x) + (rest with S(x)
    // zeroed) uses a nested split `depth` levels deep: any valid bound of
    // that branch may stand for the rest. Past `deadline` it returns -inf
    // (nothing proven).
    double thresholdBranch(const Region &region, double theta, size_t max_with, int depth,
                           std::vector<uint8_t> &forbid, const std::vector<uint64_t> &zero,
                           std::chrono::steady_clock::time_point deadline) const
    {
        std::vector<double> L;
        std::vector<Claims> support;
        std::vector<double> Lz;
        std::vector<Claims> support_z;
        const size_t ws = region.support_words;
        std::vector<double> loss;
        while (true)
        {
            if (deadline != kNoDeadline && std::chrono::steady_clock::now() >= deadline)
                return -std::numeric_limits<double>::infinity();
            const double value = evaluateRegionMasked(region, forbid, zero, L, support, &loss);
            if (!(value < theta))
                return theta;
            bool added = false;
            std::vector<uint32_t> lighter;
            for (uint32_t i = 0; i < region.nodes.size(); ++i)
            {
                if (forbid[i] || relaxation_detail::testBit(zero.data(), i))
                    continue;
                if (L[i] >= theta)
                {
                    forbid[i] = 1;
                    added = true;
                }
                else if (loss[i] > 0.0 && L[i] > 0.0)
                    lighter.push_back(i);
            }
            // A selected x also counts the rest of the region with S(x)
            // zeroed (disjoint classes, same branch). Try the classes whose
            // sharing lost the most value first.
            std::sort(lighter.begin(), lighter.end(), [&](uint32_t a, uint32_t b) { return loss[a] > loss[b]; });
            if (lighter.size() > max_with)
                lighter.resize(max_with);
            for (uint32_t x : lighter)
            {
                std::vector<uint64_t> with_zero(zero);
                zeroClaims(support[x], with_zero);
                double rest;
                if (depth > 0)
                {
                    std::vector<uint8_t> inner(forbid);
                    rest = thresholdBranch(region, theta - L[x], max_with, depth - 1, inner, with_zero, deadline);
                }
                else
                    rest = evaluateRegionMasked(region, forbid, with_zero, Lz, support_z);
                if (L[x] + rest >= theta)
                {
                    forbid[x] = 1;
                    added = true;
                }
            }
            if (!added)
                return value;
        }
    }

    // Largest threshold split value found by doubling then bisection from
    // `base` (every tested theta gives a valid bound; the best is kept).
    double bestThresholdRegion(const Region &region, double base, int steps, size_t with_candidates, int depth,
                               std::chrono::steady_clock::time_point deadline) const
    {
        if (!(base > 0.0))
            return base;
        double best = base;
        double lo = base, hi = 0.0;
        for (int s = 0; s < steps; ++s)
        {
            const double theta = hi > 0.0 ? 0.5 * (lo + hi) : 2.0 * lo;
            const double v = thresholdRegion(region, theta, with_candidates, depth, deadline);
            if (v == -std::numeric_limits<double>::infinity())
                break; // deadline
            best = std::max(best, v);
            if (v >= theta)
                lo = theta;
            else
                hi = theta;
            if (hi > 0.0 && hi - lo <= 1e-3 * hi)
                break;
        }
        return best;
    }

    // Probe candidates: counted classes with at least two parents in the
    // region (sharing is what makes supports overlap), largest L first.
    std::vector<uint32_t> probeCandidates(const Region &region, size_t count) const
    {
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        std::vector<uint32_t> parents(rn, 0), last_parent(rn, UINT32_MAX);
        for (uint32_t i = 0; i < rn; ++i)
            for (const AltInfo &info : region.alts[i])
                for (uint32_t k : info.all_kids)
                    if (last_parent[k] != i)
                    {
                        last_parent[k] = i;
                        ++parents[k];
                    }
        std::vector<uint32_t> out;
        for (uint32_t i = 0; i < rn; ++i)
            if (region.pos[i] && parents[i] >= 2 && region.L[i] > 0.0)
                out.push_back(i);
        std::sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) { return region.L[a] > region.L[b]; });
        if (out.size() > count)
            out.resize(count);
        return out;
    }

    // Cut-DP window bound (docs/core/relaxation_bound.md, "Cut DP").
    //
    // For a threshold T, an alternative whose shortest chain below (head)
    // plus the shortest chain from its class up to a target (tail) exceeds T
    // cannot appear in a selection with chain <= T; such alternatives are
    // removed and heads and tails recomputed until nothing changes. Over the
    // remaining alternatives, a dynamic program over the window's classes,
    // parents first, computes exactly the minimum over selections of the
    // largest engine load: a state is the set of required, not yet processed
    // classes (each class is processed once, so shared classes count once),
    // and it keeps the Pareto set of engine-load vectors that reach it. If
    // that minimum exceeds T, no selection has max(chain, loads) <= T, so the
    // window is longer than T. A bisection keeps the largest proven T.
    //
    // Limits keep it valid: past `state_cap` states the DP gives up (no
    // proof); a state's Pareto set past `point_cap` points has its two
    // closest points merged into their componentwise minimum, which only
    // lowers the computed minimum.
    double cutDpWindow(const Region &region, double proven, size_t state_cap, size_t point_cap) const
    {
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        const size_t ne = rb_.engines.size();
        if (region.zero_value || rn == 0 || ne == 0)
            return proven;
        const double inf = std::numeric_limits<double>::infinity();
        auto ownDuration = [&](uint32_t i, size_t a) {
            const AltInfo &info = region.alts[i][a];
            const auto &alt = rb_.classes[region.nodes[i]].alternatives[info.alt_index];
            return info.weighted && alt.engine_mask ? alt.duration : 0.0;
        };
        auto ownMask = [&](uint32_t i, size_t a) {
            return rb_.classes[region.nodes[i]].alternatives[region.alts[i][a].alt_index].engine_mask;
        };
        std::vector<std::vector<uint8_t>> allowed(rn);
        std::vector<double> head(rn), tail(rn);
        auto altHead = [&](uint32_t i, size_t a) {
            double below = 0.0;
            for (uint32_t k : region.alts[i][a].all_kids)
                below = std::max(below, head[k]);
            return ownDuration(i, a) + below;
        };
        auto prune = [&](double T) {
            for (uint32_t i = 0; i < rn; ++i)
                allowed[i].assign(region.alts[i].size(), 1);
            for (bool changed = true; changed;)
            {
                changed = false;
                for (uint32_t i = 0; i < rn; ++i) // children first
                {
                    head[i] = inf;
                    for (size_t a = 0; a < region.alts[i].size(); ++a)
                        if (allowed[i][a])
                            head[i] = std::min(head[i], altHead(i, a));
                }
                std::fill(tail.begin(), tail.end(), inf);
                for (uint32_t t : region.targets)
                    tail[t] = 0.0;
                for (uint32_t i = rn; i-- > 0;) // parents first
                {
                    if (!std::isfinite(tail[i]))
                        continue;
                    for (size_t a = 0; a < region.alts[i].size(); ++a)
                    {
                        if (!allowed[i][a] || !std::isfinite(altHead(i, a)))
                            continue;
                        const double through = tail[i] + ownDuration(i, a);
                        for (uint32_t k : region.alts[i][a].all_kids)
                            tail[k] = std::min(tail[k], through);
                    }
                }
                for (uint32_t i = 0; i < rn; ++i)
                {
                    const double up = std::isfinite(tail[i]) ? tail[i] : 0.0;
                    for (size_t a = 0; a < region.alts[i].size(); ++a)
                        if (allowed[i][a] && altHead(i, a) + up > T)
                        {
                            allowed[i][a] = 0;
                            changed = true;
                        }
                }
            }
        };
        using Point = std::vector<double>;
        // A point: engine loads, then per engine the shortest head and the
        // shortest tail of its operations so far (infinite before the
        // first; one-machine value head + load + tail, see exactWindow),
        // then the tail into each required class.
        const size_t so = 3 * ne;
        const bool stage_ = exact_window_stage_;
        auto addPoint = [&](std::vector<Point> &set, Point p) {
            for (const Point &q : set)
            {
                bool dominated = true;
                for (size_t e = 0; e < p.size() && dominated; ++e)
                    dominated = q[e] <= p[e];
                if (dominated)
                    return;
            }
            set.erase(std::remove_if(set.begin(), set.end(),
                                     [&](const Point &q) {
                                         for (size_t e = 0; e < p.size(); ++e)
                                             if (p[e] > q[e])
                                                 return false;
                                         return true;
                                     }),
                      set.end());
            set.push_back(std::move(p));
            while (set.size() > point_cap)
            {
                // Merge the closest pair into its componentwise minimum.
                size_t bx = 0, by = 1;
                double bd = inf;
                for (size_t x = 0; x < set.size(); ++x)
                    for (size_t y = x + 1; y < set.size(); ++y)
                    {
                        double d = 0.0;
                        for (size_t e = 0; e < set[x].size(); ++e)
                            d += std::abs(set[x][e] - set[y][e]);
                        if (d < bd)
                        {
                            bd = d;
                            bx = x;
                            by = y;
                        }
                    }
                for (size_t e = 0; e < set[bx].size(); ++e)
                    set[bx][e] = std::min(set[bx][e], set[by][e]);
                set.erase(set.begin() + static_cast<std::ptrdiff_t>(by));
            }
        };
        struct KeyHash
        {
            size_t operator()(const std::vector<uint32_t> &v) const
            {
                uint64_t h = 1469598103934665603ull;
                for (uint32_t x : v)
                    h = (h ^ x) * 1099511628211ull;
                return static_cast<size_t>(h);
            }
        };
        // Exact minimum over selections of the largest engine load, or -1
        // when the state cap is exceeded.
        // Exact minimum over selections with chain <= T of the largest engine
        // load, or -1 when the state cap is exceeded. A point holds the
        // engine loads and, for each required class (in key order), the
        // longest chain from the target down to that class in the partial
        // selection; a class may take an alternative only if that chain plus
        // the alternative's duration and its children's shortest heads fits.
        auto minMaxLoad = [&](double T) -> double {
            std::unordered_map<std::vector<uint32_t>, std::vector<Point>, KeyHash> states, next;
            for (uint32_t t : region.targets)
                if (std::isfinite(head[t]))
                {
                    Point start(so + 1, 0.0);
                    for (size_t e = ne; e < so; ++e)
                        start[e] = inf; // no operation on the engine yet
                    addPoint(states[std::vector<uint32_t>{t}], std::move(start));
                }
            std::vector<double> kid_tail;
            for (uint32_t i = rn; i-- > 0;)
            {
                next.clear();
                for (auto &[req, points] : states)
                {
                    if (req.empty() || req.back() != i) // sorted ascending: the largest index is processed first
                    {
                        auto &dst = next[req];
                        for (const Point &p : points)
                            addPoint(dst, p);
                        continue;
                    }
                    std::vector<uint32_t> base(req.begin(), req.end() - 1);
                    for (size_t a = 0; a < region.alts[i].size(); ++a)
                    {
                        if (!allowed[i][a])
                            continue;
                        const double d = ownDuration(i, a);
                        const uint32_t mask = ownMask(i, a);
                        const double alt_head = altHead(i, a);
                        const auto &kids = region.alts[i][a].all_kids;
                        std::vector<uint32_t> key = base;
                        for (uint32_t k : kids)
                            key.push_back(k);
                        std::sort(key.begin(), key.end());
                        key.erase(std::unique(key.begin(), key.end()), key.end());
                        auto &dst = next[key];
                        for (const Point &p : points)
                        {
                            const double tail_i = p[so + base.size()]; // i is the last key entry
                            if (tail_i + alt_head > T)
                                continue;
                            Point q(so + key.size());
                            for (size_t e = 0; e < ne; ++e)
                            {
                                const bool on = d > 0.0 && (mask & (1u << e));
                                q[e] = p[e] + (on ? d : 0.0);
                                q[ne + e] = on && stage_ ? std::min(p[ne + e], alt_head - d) : p[ne + e];
                                q[2 * ne + e] = on && stage_ ? std::min(p[2 * ne + e], tail_i) : p[2 * ne + e];
                            }
                            size_t bi = 0;
                            for (size_t x = 0; x < key.size(); ++x)
                            {
                                double tail = 0.0;
                                while (bi < base.size() && base[bi] < key[x])
                                    ++bi;
                                if (bi < base.size() && base[bi] == key[x])
                                    tail = p[so + bi];
                                if (std::find(kids.begin(), kids.end(), key[x]) != kids.end())
                                    tail = std::max(tail, tail_i + d);
                                q[so + x] = tail;
                            }
                            addPoint(dst, std::move(q));
                        }
                    }
                }
                for (auto it = next.begin(); it != next.end();)
                    it = it->second.empty() ? next.erase(it) : std::next(it);
                states.swap(next);
                if (states.size() > state_cap)
                    return -1.0;
            }
            double best = inf;
            auto it = states.find(std::vector<uint32_t>{});
            if (it != states.end())
                for (const Point &p : it->second)
                {
                    double v = 0.0;
                    for (size_t e = 0; e < ne; ++e)
                        v = std::max(v, p[e] > 0.0 && std::isfinite(p[ne + e]) && std::isfinite(p[2 * ne + e])
                                            ? p[ne + e] + p[e] + p[2 * ne + e]
                                            : p[e]);
                    best = std::min(best, v);
                }
            return best;
        };
        bool gave_up = false;
        auto longer = [&](double T) {
            if (gave_up)
                return false;
            prune(T);
            bool any_target = false;
            for (uint32_t t : region.targets)
                any_target |= std::isfinite(head[t]);
            if (!any_target)
                return true;
            const double v = minMaxLoad(T);
            gave_up = v < 0.0; // past the state cap: stop this window
            return v > T;
        };
        double lo = proven, hi = std::max(proven, 1e-9) * 1.25;
        for (int grow = 0; grow < 4 && longer(hi); ++grow)
        {
            lo = hi;
            hi *= 1.25;
        }
        for (int it = 0; it < 14; ++it)
        {
            const double mid = 0.5 * (lo + hi);
            if (longer(mid))
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }

    // Enumeration budget (search nodes) for the exact window value.
    uint64_t exact_window_nodes_ = 20000;
    bool exact_window_pruning_ = true;
    bool exact_window_stage_ = true;
    // Cut DP limits: states per class step and Pareto points per state.
    static constexpr size_t kCutDpStates = 5000;
    static constexpr size_t kCutDpPoints = 16;
    static constexpr size_t kCutDpMaxClasses = 512; // larger windows: DP not attempted

    // Exact window value when the window's selections can be enumerated
    // within `budget` search nodes: the minimum over the selections below a
    // member of the closing link of max(critical path, load on each engine),
    // each class counted once. Every plan's window is at least this long (the
    // chain and each engine's load fit inside it). Returns 0 when the budget
    // runs out, which is always valid.
    // With `top`, also keeps the `top_count` best complete selections
    // (value, choice), cheapest first.
    // With `equivalents`, also gives for every class the alternatives
    // merged into each one tried (the tried one first).
    // `by_work` tries alternatives with the least total work below first
    // instead of the shortest chain (finds load-bound selections sooner;
    // the value does not depend on it).
    // `stop` ends the enumeration like the budget (checked every 4096
    // nodes).
    double exactWindow(const Region &region, uint64_t budget, std::vector<int32_t> *best_choice = nullptr,
                       std::vector<std::pair<double, std::vector<int32_t>>> *top = nullptr, size_t top_count = 0,
                       std::vector<std::vector<std::vector<uint32_t>>> *equivalents = nullptr,
                       bool by_work = false,
                       std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::time_point::max()) const
    {
        if (region.zero_value)
            return 0.0;
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        const size_t ne = rb_.engines.size();
        std::vector<int32_t> choice(rn, -1);
        std::vector<double> loads(ne, 0.0), finish(rn, 0.0);
        double best = std::numeric_limits<double>::infinity();
        uint64_t visited = 0;
        bool exhausted = false;
        auto ownDuration = [&](uint32_t i, int32_t a) {
            const AltInfo &info = region.alts[i][static_cast<size_t>(a)];
            const auto &alt = rb_.classes[region.nodes[i]].alternatives[info.alt_index];
            return info.weighted && alt.engine_mask ? alt.duration : 0.0;
        };
        // Classes whose every alternative (and child) adds nothing can be
        // left out: dropped from children lists, never chosen. Alternatives
        // then identical (own duration, engines, effective children) are
        // tried once.
        std::vector<uint8_t> nil(rn, 0);
        std::vector<std::vector<std::vector<uint32_t>>> eff_kids(rn);
        if (equivalents)
            equivalents->assign(rn, {});
        std::vector<std::vector<uint32_t>> alive(rn);
        for (uint32_t i = 0; i < rn; ++i) // children first
        {
            bool all_nil = true;
            eff_kids[i].resize(region.alts[i].size());
            std::vector<std::tuple<double, uint32_t, std::vector<uint32_t>>> keys;
            for (size_t a = 0; a < region.alts[i].size(); ++a)
            {
                for (uint32_t k : region.alts[i][a].all_kids)
                    if (!nil[k] || !exact_window_pruning_)
                        eff_kids[i][a].push_back(k);
                std::sort(eff_kids[i][a].begin(), eff_kids[i][a].end());
                const double d = ownDuration(i, static_cast<int32_t>(a));
                all_nil = all_nil && d == 0.0 && eff_kids[i][a].empty();
                const uint32_t mask =
                    d > 0.0 ? rb_.classes[region.nodes[i]].alternatives[region.alts[i][a].alt_index].engine_mask : 0;
                std::tuple<double, uint32_t, std::vector<uint32_t>> key{d, mask, eff_kids[i][a]};
                const auto same = std::find(keys.begin(), keys.end(), key);
                if (exact_window_pruning_ && same != keys.end())
                {
                    if (equivalents)
                        (*equivalents)[i][static_cast<size_t>(same - keys.begin())].push_back(static_cast<uint32_t>(a));
                    continue;
                }
                keys.push_back(std::move(key));
                alive[i].push_back(static_cast<uint32_t>(a));
                if (equivalents)
                    (*equivalents)[i].push_back({static_cast<uint32_t>(a)});
            }
            nil[i] = all_nil && !region.alts[i].empty() &&
                     std::find(region.targets.begin(), region.targets.end(), i) == region.targets.end();
        }
        auto chainOf = [&](uint32_t target) {
            for (uint32_t i = 0; i < rn; ++i) // children first
            {
                if (choice[i] < 0)
                    continue;
                double below = 0.0;
                for (uint32_t k : eff_kids[i][static_cast<size_t>(choice[i])])
                    below = std::max(below, finish[k]);
                finish[i] = below + ownDuration(i, choice[i]);
            }
            return finish[target];
        };
        // One-machine value of a complete selection: an engine's operations
        // run one at a time inside the window, the first no earlier than the
        // shortest chain below any of them (head) and the last followed by
        // the shortest chain from any of them up to the target (tail), so
        // the window is at least min head + load + min tail on each engine.
        // Uses finish times from chainOf.
        std::vector<double> stage_tail(rn, 0.0);
        auto stageValue = [&](uint32_t target) {
            if (!exact_window_stage_)
                return 0.0;
            std::fill(stage_tail.begin(), stage_tail.end(), -1.0);
            stage_tail[target] = 0.0;
            for (uint32_t i = target + 1; i-- > 0;) // parents first
            {
                if (choice[i] < 0 || stage_tail[i] < 0.0)
                    continue;
                const double up = stage_tail[i] + ownDuration(i, choice[i]);
                for (uint32_t k : eff_kids[i][static_cast<size_t>(choice[i])])
                    stage_tail[k] = std::max(stage_tail[k], up);
            }
            double value = 0.0;
            for (size_t e = 0; e < ne; ++e)
            {
                double load = 0.0, head = std::numeric_limits<double>::infinity(),
                       tail = std::numeric_limits<double>::infinity();
                for (uint32_t i = 0; i <= target; ++i)
                {
                    if (choice[i] < 0 || stage_tail[i] < 0.0)
                        continue;
                    const double d = ownDuration(i, choice[i]);
                    const uint32_t mask =
                        rb_.classes[region.nodes[i]].alternatives[region.alts[i][static_cast<size_t>(choice[i])].alt_index]
                            .engine_mask;
                    if (!(d > 0.0) || !(mask & (1u << e)))
                        continue;
                    load += d;
                    head = std::min(head, finish[i] - d);
                    tail = std::min(tail, stage_tail[i]);
                }
                if (load > 0.0)
                    value = std::max(value, head + load + tail);
            }
            return value;
        };
        // Lower bound on the target's chain under any completion of the
        // partial selection: chosen classes use their alternative, the
        // others the best alternative over the same bounds of its children.
        std::vector<double> chain_lb(rn, 0.0);
        auto chainBound = [&](uint32_t target) {
            for (uint32_t i = 0; i <= target; ++i) // children first
            {
                auto through = [&](size_t a) {
                    double below = 0.0;
                    for (uint32_t k : eff_kids[i][a])
                        below = std::max(below, chain_lb[k]);
                    return below + ownDuration(i, static_cast<int32_t>(a));
                };
                if (choice[i] >= 0)
                    chain_lb[i] = through(static_cast<size_t>(choice[i]));
                else
                {
                    double v = std::numeric_limits<double>::infinity();
                    for (uint32_t a : alive[i])
                        v = std::min(v, through(a));
                    chain_lb[i] = std::isfinite(v) ? v : 0.0;
                }
            }
            return chain_lb[target];
        };
        // Alternatives tried cheapest first (own duration plus the unchosen
        // bound of their children).
        chainBound(rn - 1);
        // Least total work below each class (shared classes counted per
        // use: an ordering key only).
        std::vector<double> least_work(rn, 0.0);
        if (by_work)
            for (uint32_t i = 0; i < rn; ++i) // children first
            {
                double v = std::numeric_limits<double>::infinity();
                for (uint32_t a : alive[i])
                {
                    double w = ownDuration(i, static_cast<int32_t>(a));
                    for (uint32_t k : eff_kids[i][a])
                        w += least_work[k];
                    v = std::min(v, w);
                }
                least_work[i] = std::isfinite(v) ? v : 0.0;
            }
        std::vector<std::vector<uint32_t>> alt_order(rn);
        for (uint32_t i = 0; i < rn; ++i)
        {
            std::vector<std::pair<double, uint32_t>> keyed;
            for (uint32_t a : alive[i])
            {
                double below = 0.0;
                for (uint32_t k : eff_kids[i][a])
                    below = by_work ? below + least_work[k] : std::max(below, chain_lb[k]);
                keyed.push_back({below + ownDuration(i, static_cast<int32_t>(a)), static_cast<uint32_t>(a)});
            }
            std::stable_sort(keyed.begin(), keyed.end());
            for (const auto &[key, a] : keyed)
                alt_order[i].push_back(a);
        }
        std::vector<double> extra(ne), least(ne), path_load(static_cast<size_t>(rn) * ne, 0.0);
        std::vector<uint64_t> seen(rn, 0);
        uint64_t stamp = 0;
        // Depth-first over pending classes; loads only grow, so a partial
        // selection whose maximum load or chain bound reaches the best value
        // is pruned.
        auto dfs = [&](auto &&self, uint32_t target, std::vector<uint32_t> pending) -> void {
            if (exhausted)
                return;
            if (++visited > budget ||
                ((visited & 4095) == 0 && std::chrono::steady_clock::now() >= stop))
            {
                exhausted = true;
                return;
            }
            double max_load = 0.0;
            for (double l : loads)
                max_load = std::max(max_load, l);
            const double cutoff = top && top->size() < top_count ? std::numeric_limits<double>::infinity()
                                  : top ? top->back().first : best;
            if (max_load >= cutoff ||
                (exact_window_pruning_ && std::isfinite(cutoff) && chainBound(target) >= cutoff))
                return;
            while (!pending.empty() && choice[pending.back()] >= 0)
                pending.pop_back();
            // Every pending class takes an alternative that keeps each of
            // its engines below the cutoff: on each engine it then adds at
            // least the minimum, over those alternatives, of its duration
            // there (0 for alternatives elsewhere).
            if (exact_window_pruning_ && std::isfinite(cutoff) && !pending.empty())
            {
                std::fill(extra.begin(), extra.end(), 0.0);
                ++stamp;
                for (uint32_t i : pending)
                {
                    if (choice[i] >= 0 || seen[i] == stamp)
                        continue;
                    seen[i] = stamp;
                    std::fill(least.begin(), least.end(), std::numeric_limits<double>::infinity());
                    bool feasible = false;
                    for (uint32_t a : alive[i])
                    {
                        const double d = ownDuration(i, static_cast<int32_t>(a));
                        const uint32_t mask =
                            rb_.classes[region.nodes[i]].alternatives[region.alts[i][a].alt_index].engine_mask;
                        bool fits = true;
                        for (size_t e = 0; e < ne && fits; ++e)
                            fits = !(mask & (1u << e)) || loads[e] + d < cutoff;
                        if (!fits)
                            continue;
                        feasible = true;
                        for (size_t e = 0; e < ne; ++e)
                            least[e] = std::min(least[e], (mask & (1u << e)) ? d : 0.0);
                    }
                    if (!feasible)
                        return;
                    for (size_t e = 0; e < ne; ++e)
                        extra[e] += least[e];
                }
                for (size_t e = 0; e < ne; ++e)
                    if (loads[e] + extra[e] >= cutoff)
                        return;
                // Along any one route below a pending class the loads add
                // up: path_load[i * ne + e] bounds what class i still adds
                // on engine e through its heaviest child (classes chosen
                // already count no own duration).
                for (uint32_t i = 0; i <= target; ++i) // children first
                {
                    double *pl = &path_load[static_cast<size_t>(i) * ne];
                    if (choice[i] >= 0)
                    {
                        std::fill(pl, pl + ne, 0.0);
                        for (uint32_t k : eff_kids[i][static_cast<size_t>(choice[i])])
                            for (size_t e = 0; e < ne; ++e)
                                pl[e] = std::max(pl[e], path_load[static_cast<size_t>(k) * ne + e]);
                        continue;
                    }
                    std::fill(pl, pl + ne, std::numeric_limits<double>::infinity());
                    bool feasible = false;
                    for (uint32_t a : alive[i])
                    {
                        const double d = ownDuration(i, static_cast<int32_t>(a));
                        const uint32_t mask =
                            rb_.classes[region.nodes[i]].alternatives[region.alts[i][a].alt_index].engine_mask;
                        bool fits = true;
                        for (size_t e = 0; e < ne && fits; ++e)
                            fits = !(mask & (1u << e)) || loads[e] + d < cutoff;
                        if (!fits)
                            continue;
                        feasible = true;
                        for (size_t e = 0; e < ne; ++e)
                        {
                            double below = 0.0;
                            for (uint32_t k : eff_kids[i][a])
                                below = std::max(below, path_load[static_cast<size_t>(k) * ne + e]);
                            pl[e] = std::min(pl[e], below + ((mask & (1u << e)) ? d : 0.0));
                        }
                    }
                    if (!feasible) // infinite: no completion through i fits
                        std::fill(pl, pl + ne, std::numeric_limits<double>::infinity());
                }
                for (uint32_t i : pending)
                    for (size_t e = 0; e < ne; ++e)
                        if (choice[i] < 0 && loads[e] + path_load[static_cast<size_t>(i) * ne + e] >= cutoff)
                            return;
            }
            if (pending.empty())
            {
                const double v = std::max({max_load, chainOf(target), stageValue(target)});
                if (v < best)
                {
                    best = v;
                    if (best_choice)
                        *best_choice = choice;
                }
                if (top && (top->size() < top_count || v < top->back().first))
                {
                    top->insert(std::upper_bound(top->begin(), top->end(), v,
                                                 [](double x, const auto &e) { return x < e.first; }),
                                {v, choice});
                    if (top->size() > top_count)
                        top->pop_back();
                }
                return;
            }
            const uint32_t i = pending.back();
            pending.pop_back();
            for (const uint32_t a : alt_order[i])
            {
                if (exhausted)
                    break;
                const AltInfo &info = region.alts[i][a];
                const auto &alt = rb_.classes[region.nodes[i]].alternatives[info.alt_index];
                const double d = ownDuration(i, static_cast<int32_t>(a));
                choice[i] = static_cast<int32_t>(a);
                for (size_t e = 0; e < ne; ++e)
                    if (alt.engine_mask & (1u << e))
                        loads[e] += d;
                std::vector<uint32_t> next = pending;
                for (uint32_t k : eff_kids[i][a])
                    if (choice[k] < 0)
                        next.push_back(k);
                self(self, target, std::move(next));
                for (size_t e = 0; e < ne; ++e)
                    if (alt.engine_mask & (1u << e))
                        loads[e] -= d;
                choice[i] = -1;
            }
        };
        for (uint32_t t : region.targets)
        {
            dfs(dfs, t, std::vector<uint32_t>{t});
            if (exhausted)
                return 0.0;
        }
        return std::isfinite(best) ? best : 0.0;
    }


    // Path-weighted window value (after evaluateRegion filled region.L for
    // the same weights). Any chain of window classes runs sequentially inside
    // the window, and the window's load is at most its length on every
    // engine, so for mu in [0, 1]:
    //   window >= mu * (chain duration) + (1 - mu) * (lambda-weighted load).
    // P(c) bounds mu * (some chain below c) + (1 - mu) * (weighted cost of the
    // plan's classes in S(c) ∩ T(c)): the load part follows L (anchors with
    // disjoint supports, charged groups); the chain continues either through
    // one anchor (its P replaces its (1 - mu) L) or through any child (mu
    // times its critical path, no load).
    double evaluateRegionPath(const Region &region, double mu, const std::vector<double> &cp,
                              std::vector<double> &P) const
    {
        using namespace relaxation_detail;
        if (region.zero_value)
            return 0.0;
        const size_t ws = region.support_words;
        std::vector<uint64_t> cover(ws);
        std::vector<uint32_t> order, chosen;
        const uint32_t rn = static_cast<uint32_t>(region.nodes.size());
        P.assign(rn, 0.0);
        const double nu = 1.0 - mu;
        for (uint32_t i = 0; i < rn; ++i)
        {
            const uint32_t u = region.nodes[i];
            const auto &alts = rb_.classes[u].alternatives;
            double best = std::numeric_limits<double>::infinity();
            for (const AltInfo &info : region.alts[i])
            {
                double path_any = 0.0;
                for (uint32_t k : info.all_kids)
                    path_any = std::max(path_any, mu * cp[k]);
                double cand = path_any;
                if (!info.kids.empty() || !info.extras.empty())
                {
                    order = info.kids;
                    std::sort(order.begin(), order.end(),
                              [&](uint32_t x, uint32_t y) { return region.L[x] > region.L[y]; });
                    const size_t tries = std::max<size_t>(1, order.size());
                    cand = -1.0;
                    for (size_t first = 0; first < tries; ++first)
                    {
                        std::fill(cover.begin(), cover.end(), 0);
                        double sum = 0.0, bonus = path_any;
                        for (size_t step = 0; step < order.size(); ++step)
                        {
                            const uint32_t k = order[step == 0 ? first : (step <= first ? step - 1 : step)];
                            const uint64_t *sk = &region.support[static_cast<size_t>(k) * ws];
                            bool overlap = false;
                            for (size_t w = 0; w < ws && !overlap; ++w)
                                overlap = (cover[w] & sk[w]) != 0;
                            if (overlap)
                                continue;
                            sum += region.L[k];
                            bonus = std::max(bonus, P[k] - nu * region.L[k]);
                            for (size_t w = 0; w < ws; ++w)
                                cover[w] |= sk[w];
                        }
                        for (uint32_t lg : info.extras)
                        {
                            bool overlap = false;
                            for (uint32_t m : region.group_members[lg])
                                if (testBit(cover.data(), m))
                                {
                                    overlap = true;
                                    break;
                                }
                            if (!overlap)
                                sum += region.mp[lg];
                        }
                        cand = std::max(cand, nu * sum + bonus);
                    }
                }
                const double own_d =
                    info.weighted && alts[info.alt_index].engine_mask ? alts[info.alt_index].duration : 0.0;
                best = std::min(best, nu * weight(u, info.alt_index) + mu * own_d + cand);
            }
            P[i] = std::isfinite(best) ? best : 0.0;
        }
        double value = std::numeric_limits<double>::infinity();
        for (uint32_t t : region.targets)
            value = std::min(value, P[t]);
        return std::isfinite(value) ? value : 0.0;
    }
};

// One evaluation of the interval bound: the closing chain link of every
// window but the last, and the value of every window (the last one ends at
// the root and is the same for every run).
struct IntervalRun
{
    std::vector<size_t> last_links;
    std::vector<double> values;
};

// Best partition of the interval chain into windows taken from any of the
// runs. A window spans the chain links after one closing link up to the next,
// and its value bounds the time between those links, so any sequence of
// consecutive windows covering the chain (followed by the final window to the
// root) is a chain of disjoint, ordered time windows: its sum is a valid
// lower bound. best[j] is the largest sum over windows covering links 0..j-1.
inline double bestWindowPartition(size_t chain, const std::vector<IntervalRun> &runs)
{
    if (runs.empty() || runs[0].values.empty())
        return 0.0;
    struct Window
    {
        size_t first, last;
        double value;
    };
    std::vector<Window> windows;
    for (const IntervalRun &run : runs)
    {
        if (run.values.size() != run.last_links.size() + 1)
            Error::throw_err("Relaxation bound: interval run has mismatched windows");
        size_t first = 0;
        for (size_t l = 0; l < run.last_links.size(); ++l)
        {
            if (run.last_links[l] < first || run.last_links[l] >= chain)
                Error::throw_err("Relaxation bound: interval window outside the chain");
            windows.push_back({first, run.last_links[l], run.values[l]});
            first = run.last_links[l] + 1;
        }
        if (first != chain)
            Error::throw_err("Relaxation bound: interval run does not cover the chain");
    }
    std::sort(windows.begin(), windows.end(), [](const Window &a, const Window &b) { return a.last < b.last; });
    const double none = -std::numeric_limits<double>::infinity();
    std::vector<double> best(chain + 1, none);
    best[0] = 0.0;
    for (const Window &w : windows)
        if (best[w.first] != none)
            best[w.last + 1] = std::max(best[w.last + 1], best[w.first] + w.value);
    if (best[chain] == none)
        Error::throw_err("Relaxation bound: no window partition covers the chain");
    return best[chain] + runs[0].values.back();
}

// Best lower bound over a small set of engine weightings. Any probability
// vector gives a valid bound; the search over lambda only tightens it.
inline double maximizeRelaxationBound(RelaxationBoundComputer &computer, size_t num_engines,
                                      std::vector<double> *best_lambda = nullptr)
{
    if (num_engines == 0)
        return 0.0;
    double best = -1.0;
    std::vector<double> best_l;
    auto consider = [&](const std::vector<double> &lambda) {
        const double v = computer.evaluate(lambda);
        if (v > best)
        {
            best = v;
            best_l = lambda;
        }
        return v;
    };
    if (num_engines == 1)
        consider({1.0});
    else if (num_engines == 2)
    {
        // Golden-section search on lambda_0 in [lo, hi], keeping both weights
        // positive so zero-duration structure stays lambda-independent.
        const double lo = 0.01, hi = 0.99, phi = (std::sqrt(5.0) - 1.0) / 2.0;
        double a = lo, b = hi;
        double c = b - phi * (b - a), d = a + phi * (b - a);
        double fc = consider({c, 1.0 - c}), fd = consider({d, 1.0 - d});
        for (int it = 0; it < 18; ++it)
        {
            if (fc < fd)
            {
                a = c;
                c = d;
                fc = fd;
                d = a + phi * (b - a);
                fd = consider({d, 1.0 - d});
            }
            else
            {
                b = d;
                d = c;
                fd = fc;
                c = b - phi * (b - a);
                fc = consider({c, 1.0 - c});
            }
        }
        consider({0.5, 0.5});
    }
    else
    {
        std::vector<double> uniform(num_engines, 1.0 / static_cast<double>(num_engines));
        consider(uniform);
        for (size_t e = 0; e < num_engines; ++e)
        {
            std::vector<double> lambda(num_engines, 0.5 / static_cast<double>(num_engines - 1));
            lambda[e] = 0.5;
            consider(lambda);
            std::vector<double> heavy(num_engines, 0.1 / static_cast<double>(num_engines - 1));
            heavy[e] = 0.9;
            consider(heavy);
        }
    }
    if (best_lambda)
        *best_lambda = best_l;
    return std::max(best, 0.0);
}

} // namespace plan
