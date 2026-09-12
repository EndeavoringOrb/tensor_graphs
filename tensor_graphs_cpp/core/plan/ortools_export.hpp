#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <json.hpp>

#include "core/common/constants.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/logging.hpp"
#include "core/plan/planner.hpp"
#include "core/settings.hpp"
#include "core/types.hpp"

namespace ortools_export {

using json = nlohmann::json;

inline json memSpaceToJson(const MemSpace &ms)
{
    return json{{"idx", ms.idx}, {"type", static_cast<int>(ms.type)}};
}

inline MemSpace memSpaceFromJson(const json &j)
{
    return MemSpace{j.value("idx", 0u), static_cast<HandleType>(j.value("type", 1))};
}

inline json serializeProblem(
    const std::vector<Bucket> &buckets,
    const std::vector<EGraph> &bucket_egraphs,
    const std::vector<EClassId> &bucket_root_eclass_ids,
    const std::vector<std::unordered_map<EClassId, LogicalId>> &bucket_eclass_to_logicals,
    const std::vector<std::vector<ENodeInfo>> &bucket_enode_infos,
    const std::vector<LogicalId> &candidates,
    const std::vector<std::vector<uint32_t>> &candidate_clean_buckets,
    const std::vector<std::unordered_set<EClassId>> &bucket_clean_eclasses,
    const Graph &graph,
    const std::unordered_map<LogicalId, ParallelBuffer> &preallocated_buffers,
    const Settings &settings)
{
    json root_json;
    root_json["min_compile_seconds"] = settings.min_compile_seconds;
    root_json["use_ortools_full"] = settings.use_ortools_full;

    // Mem caps
    json mem_caps_json = json::object();
    for (const auto &kv : settings.mem_caps)
    {
        std::string key = settings.use_ortools_full
                              ? std::to_string(static_cast<int>(kv.first.type)) + ":" + std::to_string(kv.first.idx)
                              : toString(kv.first.type) + std::to_string(kv.first.idx);
        mem_caps_json[key] = kv.second;
    }
    root_json["mem_caps"] = mem_caps_json;

    // Candidates
    json candidates_json = json::array();
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        LogicalId cid = candidates[i];
        if (!graph.hasNode(cid))
            continue;
        const TensorNode &node = graph.getNode(cid);
        uint64_t size_bytes = (node.getSizeBytes() + 4095) & ~4095ULL;
        json c_obj;
        c_obj["logical_id"] = cid.value;
        c_obj["size_bytes"] = size_bytes;
        c_obj["raw_size_bytes"] = node.getSizeBytes();
        c_obj["mem_space"] = memSpaceToJson(MemSpace{1, HandleType::CPP}); // Default CPU
        if (settings.use_ortools_full)
        {
            // A cache can live in any space with a full-sized representation.
            std::unordered_set<MemSpace> spaces;
            for (size_t b = 0; b < bucket_egraphs.size(); ++b)
                for (const auto &entry : bucket_eclass_to_logicals[b])
                    if (entry.second == cid)
                    {
                        const auto &cls = bucket_egraphs[b].getEClass(bucket_egraphs[b].findConst(entry.first));
                        if (cls.mem_space.type != HandleType::STORAGE &&
                            ((getSizeBytes(cls.shape, cls.dtype) + 4095) & ~4095ULL) == size_bytes)
                            spaces.insert(cls.mem_space);
                    }
            std::vector<MemSpace> sorted_spaces(spaces.begin(), spaces.end());
            std::sort(sorted_spaces.begin(), sorted_spaces.end());
            c_obj["mem_spaces"] = json::array();
            for (const auto &ms : sorted_spaces)
                c_obj["mem_spaces"].push_back(memSpaceToJson(ms));
        }
        c_obj["clean_buckets"] = candidate_clean_buckets[i];
        candidates_json.push_back(c_obj);
    }
    root_json["candidates"] = candidates_json;

    // Preallocated buffers
    json preallocated_json = json::array();
    for (const auto &kv : preallocated_buffers)
    {
        json p_obj;
        p_obj["logical_id"] = kv.first.value;
        p_obj["buffer_id"] = kv.second.id.value;
        p_obj["offset"] = kv.second.offset;
        p_obj["size"] = kv.second.size;
        p_obj["raw_size_bytes"] = graph.getNode(kv.first).getSizeBytes();
        p_obj["mem_space"] = memSpaceToJson(kv.second.mem_space);
        preallocated_json.push_back(p_obj);
    }
    root_json["preallocated_buffers"] = preallocated_json;

    // Buckets
    json buckets_json = json::array();
    for (size_t b = 0; b < buckets.size(); ++b)
    {
        json b_obj;
        b_obj["bucket_idx"] = b;
        b_obj["weight"] = buckets[b].weight;
        b_obj["root_eclass_id"] = bucket_root_eclass_ids[b].value;
        json clean_eclasses_json = json::array();
        for (EClassId clean_id : bucket_clean_eclasses[b])
            clean_eclasses_json.push_back(clean_id.value);
        b_obj["clean_eclasses"] = std::move(clean_eclasses_json);

        const EGraph &egraph = bucket_egraphs[b];
        const auto &eclass_to_logical = bucket_eclass_to_logicals[b];
        const auto &enode_infos = bucket_enode_infos[b];

        json eclass_to_logical_json = json::object();
        for (const auto &kv : eclass_to_logical)
        {
            eclass_to_logical_json[std::to_string(kv.first.value)] = kv.second.value;
        }
        b_obj["eclass_to_logical"] = eclass_to_logical_json;

        json classes_json = json::array();
        for (const auto &cls : egraph.getClasses())
        {
            EClassId canon_id = egraph.findConst(cls.id);
            if (canon_id != cls.id)
                continue;

            json cls_json;
            cls_json["id"] = cls.id.value;
            cls_json["shape"] = cls.shape;
            cls_json["dtype"] = static_cast<int>(cls.dtype);
            cls_json["mem_space"] = memSpaceToJson(cls.mem_space);
            uint64_t size_bytes = (getSizeBytes(cls.shape, cls.dtype) + 4095) & ~4095ULL;
            cls_json["size_bytes"] = size_bytes;
            cls_json["raw_size_bytes"] = getSizeBytes(cls.shape, cls.dtype);

            json enodes_json = json::array();
            for (size_t e_idx = 0; e_idx < cls.enodes.size(); ++e_idx)
            {
                ENodeId eid = cls.enodes[e_idx];
                const ENode &enode = egraph.getENode(eid);
                const ENodeInfo &info = enode_infos[eid.value];

                json enode_json;
                enode_json["enode_idx"] = e_idx;
                enode_json["enode_id"] = eid.value;
                enode_json["op_type"] = static_cast<int>(enode.getOpType());
                enode_json["kernel_id"] = enode.getKernelId().value;
                enode_json["is_view"] = info.is_view;
                enode_json["cost"] = info.cost;
                enode_json["mem_space"] = memSpaceToJson(enode.getMemSpace());
                enode_json["engines"] = json::array();
                for (const auto &engine : enode.getEngines())
                    enode_json["engines"].push_back(json{{"idx", engine.idx}, {"type", static_cast<int>(engine.type)}});
                enode_json["safe_inplace_idxs"] = json::array();
                if (enode.getKernelId().value != 0 && KernelRegistry::get().hasKernel(enode.getKernelId()))
                    enode_json["safe_inplace_idxs"] = KernelRegistry::get().getKernel(enode.getKernelId()).safe_inplace_idxs;

                std::vector<uint32_t> canon_children;
                for (EClassId child : enode.getChildren())
                {
                    canon_children.push_back(egraph.findConst(child).value);
                }
                enode_json["children"] = canon_children;

                bool is_cache = (enode.getOpType() == OpType::CACHE);
                bool is_input = (enode.getOpType() == OpType::INPUT);
                bool is_scatter = (enode.getOpType() == OpType::SCATTER);
                enode_json["is_cache"] = is_cache;
                enode_json["is_input"] = is_input;
                enode_json["is_scatter"] = is_scatter;

                int64_t lid = -1;
                if (is_cache || is_input)
                {
                    if (eclass_to_logical.count(cls.id))
                        lid = static_cast<int64_t>(eclass_to_logical.at(cls.id).value);
                }
                enode_json["logical_id"] = lid;

                enodes_json.push_back(enode_json);
            }
            cls_json["enodes"] = enodes_json;
            classes_json.push_back(cls_json);
        }
        b_obj["classes"] = classes_json;
        buckets_json.push_back(b_obj);
    }
    root_json["buckets"] = buckets_json;

    return root_json;
}

inline bool deserializeSolution(
    const json &sol_json,
    std::unordered_map<LogicalId, MemSpace> &out_cached_nodes,
    std::vector<ExtractionResult> &out_extractions)
{
    out_cached_nodes.clear();
    out_extractions.clear();

    if (!sol_json.contains("extractions") || !sol_json["extractions"].is_array())
    {
        LOG(ERROR) << "[deserializeSolution] Missing 'extractions' array in solution JSON.";
        return false;
    }

    if (sol_json.contains("cached_nodes") && sol_json["cached_nodes"].is_array())
    {
        for (const auto &item : sol_json["cached_nodes"])
        {
            LogicalId lid{item.value("logical_id", 0u)};
            MemSpace ms = memSpaceFromJson(item.value("mem_space", json::object()));
            out_cached_nodes[lid] = ms;
        }
    }

    for (const auto &ext_json : sol_json["extractions"])
    {
        ExtractionResult res;
        res.cost = ext_json.value("cost", 0.0f);

        if (ext_json.contains("selection_map") && ext_json["selection_map"].is_object())
        {
            for (auto it = ext_json["selection_map"].begin(); it != ext_json["selection_map"].end(); ++it)
            {
                EClassId cid{static_cast<uint32_t>(std::stoul(it.key()))};
                res.selection_map[cid] = it.value().get<uint32_t>();
            }
        }

        if (ext_json.contains("order") && ext_json["order"].is_array())
        {
            for (const auto &val : ext_json["order"])
            {
                res.order.push_back(EClassId{val.get<uint32_t>()});
            }
        }

        if (ext_json.contains("eclass_to_buf") && ext_json["eclass_to_buf"].is_object())
        {
            for (auto it = ext_json["eclass_to_buf"].begin(); it != ext_json["eclass_to_buf"].end(); ++it)
            {
                EClassId cid{static_cast<uint32_t>(std::stoul(it.key()))};
                res.eclass_to_buf[cid] = BufferId{it.value().get<uint32_t>()};
            }
        }

        if (ext_json.contains("eclass_to_cost") && ext_json["eclass_to_cost"].is_object())
        {
            for (auto it = ext_json["eclass_to_cost"].begin(); it != ext_json["eclass_to_cost"].end(); ++it)
            {
                EClassId cid{static_cast<uint32_t>(std::stoul(it.key()))};
                res.eclass_to_cost[cid] = it.value().get<float>();
            }
        }

        if (ext_json.contains("buffers") && ext_json["buffers"].is_array())
        {
            for (const auto &b_item : ext_json["buffers"])
            {
                ParallelBuffer buf;
                buf.id = BufferId{b_item.value("id", 0u)};
                buf.mem_space = memSpaceFromJson(b_item.value("mem_space", json::object()));
                buf.size = b_item.value("size", 0ULL);
                buf.start = b_item.value("start", 0u);
                buf.end = b_item.value("end", 0u);
                buf.offset = b_item.value("offset", 0LL);
                res.buffers.push_back(buf);
            }
        }

        out_extractions.push_back(std::move(res));
    }

    return true;
}

inline std::string findPythonExecutable()
{
    if (const char *ortools_python = std::getenv("TENSOR_GRAPHS_ORTOOLS_PYTHON");
        ortools_python != nullptr && *ortools_python != '\0')
    {
        return std::filesystem::path(ortools_python).make_preferred().string();
    }

    const std::vector<std::string> candidates = {
        ".venvx64\\Scripts\\python.exe",
        ".venvx64/Scripts/python.exe",
        ".venvx64/bin/python",
        ".venv\\Scripts\\python.exe",
        ".venv/Scripts/python.exe",
        ".venv/bin/python",
        "python.exe",
        "python",
        "python3"
    };
    for (const auto &c : candidates)
    {
        if (std::filesystem::exists(c))
        {
            return std::filesystem::path(c).make_preferred().string();
        }
    }
#if defined(_WIN32)
    return ".venv\\Scripts\\python.exe";
#else
    return ".venv/bin/python";
#endif
}

inline bool runOrtoolsSolverProcess(const std::string &problem_json_path, const std::string &solution_json_path)
{
    std::string py = findPythonExecutable();
    std::string prob_p = std::filesystem::path(problem_json_path).make_preferred().string();
    std::string sol_p = std::filesystem::path(solution_json_path).make_preferred().string();
    std::ostringstream cmd;
    if (std::filesystem::exists("ortools_solver.py"))
    {
        cmd << "\"" << py << "\" ortools_solver.py \"" << prob_p << "\" \"" << sol_p << "\"";
    }
    else
    {
        cmd << "\"" << py << "\" -m tensor_graphs.ortools_solver \"" << prob_p << "\" \"" << sol_p << "\"";
    }
    std::cout << "[Ortools] Invoking solver command: " << cmd.str() << std::endl;
#if defined(_WIN32)
    std::string wrapped_cmd = "\"" + cmd.str() + "\"";
    int ret = std::system(wrapped_cmd.c_str());
#else
    int ret = std::system(cmd.str().c_str());
#endif
    if (ret != 0)
    {
        LOG(ERROR) << "[Ortools] Solver process exited with code " << ret;
        return false;
    }
    return std::filesystem::exists(solution_json_path);
}

} // namespace ortools_export
