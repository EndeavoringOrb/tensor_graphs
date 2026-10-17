#pragma once
#include "core/egraph.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// Opt-in e-graph diagnostics (all off unless their environment variable is set).
//   TG_CLASS_SIZE_HISTOGRAM=1        class-size histogram per bucket and stage
//   TG_DUMP_CLASS=<bucket>:<class>   derivation dump of a class (3 levels), after
//                                    pruning, using the ids SearchState reports;
//                                    comma separated, <class> may be b<base id>
namespace egraph_diagnostics
{

inline bool classSizeHistogramEnabled()
{
    static const bool enabled = std::getenv("TG_CLASS_SIZE_HISTOGRAM") != nullptr;
    return enabled;
}

struct ClassDumpRequest
{
    uint32_t bucket;
    bool by_base_id;
    uint32_t id;
};

inline const std::vector<ClassDumpRequest> &classDumpRequests()
{
    static const std::vector<ClassDumpRequest> requests = [] {
        std::vector<ClassDumpRequest> out;
        const char *env = std::getenv("TG_DUMP_CLASS");
        if (!env)
            return out;
        std::stringstream ss(env);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            const size_t colon = item.find(':');
            if (colon == std::string::npos || colon + 1 >= item.size())
                Error::throw_err("[TG_DUMP_CLASS] expected <bucket>:<class> or <bucket>:b<base id>, got '" + item + "'");
            ClassDumpRequest request;
            request.bucket = static_cast<uint32_t>(std::stoul(item.substr(0, colon)));
            std::string id = item.substr(colon + 1);
            request.by_base_id = id[0] == 'b';
            request.id = static_cast<uint32_t>(std::stoul(request.by_base_id ? id.substr(1) : id));
            out.push_back(request);
        }
        return out;
    }();
    return requests;
}

inline void describeClassLine(const EGraph &egraph, EClassId cid, std::ostream &out)
{
    const EClass &cls = egraph.getEClass(cid);
    out << "class " << cid.value << " base=" << cls.base_eclass_id.value << " logical=" << cls.logical_id.value
        << " shape=" << toString(cls.shape) << " strides=" << toString(cls.strides) << " mem=" << cls.mem_space.idx
        << " clean=" << (egraph.isClean(cid) ? 1 : 0) << " enodes=" << cls.enodes.size();
}

inline void dumpClassDerivation(const EGraph &egraph, EClassId cid, uint32_t depth, const std::string &indent,
                                std::unordered_set<uint32_t> &expanded, std::ostream &out)
{
    cid = egraph.findConst(cid);
    out << indent;
    describeClassLine(egraph, cid, out);
    if (!expanded.insert(cid.value).second)
    {
        out << " (expanded above)\n";
        return;
    }
    out << "\n";
    for (ENodeId enode_id : egraph.getEClass(cid).enodes)
    {
        const ENode &enode = egraph.getENode(enode_id);
        out << indent << "  enode " << enode_id.value << " " << toString(enode.getOpType()) << ":" << enode.getOpName()
            << " origin=" << (enode.getDebugOrigin().empty() ? "N/A" : enode.getDebugOrigin()) << " children=[";
        for (size_t i = 0; i < enode.getChildren().size(); ++i)
            out << (i ? "," : "") << egraph.findConst(enode.getChildren()[i]).value;
        out << "]\n";
        if (depth == 0)
            continue;
        for (EClassId child : enode.getChildren())
            dumpClassDerivation(egraph, child, depth - 1, indent + "    ", expanded, out);
    }
}

inline void dumpRequestedClasses(const EGraph &egraph, uint32_t bucket)
{
    for (const ClassDumpRequest &request : classDumpRequests())
    {
        if (request.bucket != bucket)
            continue;
        EClassId cid = request.by_base_id ? egraph.findEClassByBaseId(BaseEClassId{request.id}) : EClassId{request.id};
        if (cid == EClassId{} || cid.value >= egraph.getNumClasses())
            Error::throw_err("[TG_DUMP_CLASS] class " + std::to_string(request.id) + " not found in bucket " +
                             std::to_string(bucket));
        std::unordered_set<uint32_t> expanded;
        std::stringstream out;
        out << "[class dump] bucket=" << bucket << "\n";
        dumpClassDerivation(egraph, cid, 3, "  ", expanded, out);
        std::cout << out.str() << std::flush;
    }
}

inline void logClassSizeHistogram(const EGraph &egraph, const std::string &label)
{
    if (!classSizeHistogramEnabled())
        return;
    uint32_t over16 = 0, over32 = 0, over63 = 0, classes = 0;
    std::vector<std::pair<uint32_t, EClassId>> sizes;
    for (uint32_t i = 0; i < egraph.getNumClasses(); ++i)
    {
        const EClassId cid{i};
        if (egraph.findConst(cid) != cid)
            continue;
        const uint32_t n = static_cast<uint32_t>(egraph.getEClass(cid).enodes.size());
        ++classes;
        over16 += n > 16;
        over32 += n > 32;
        over63 += n > 63;
        sizes.push_back({n, cid});
    }
    const size_t top = std::min<size_t>(20, sizes.size());
    std::partial_sort(sizes.begin(), sizes.begin() + top, sizes.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    std::stringstream out;
    out << "[class size histogram] " << label << " classes=" << classes << " >16=" << over16 << " >32=" << over32
        << " >63=" << over63 << "\n";
    for (size_t i = 0; i < top; ++i)
    {
        const EClassId cid = sizes[i].second;
        out << "  ";
        describeClassLine(egraph, cid, out);
        std::map<std::string, uint32_t> breakdown;
        for (ENodeId enode_id : egraph.getEClass(cid).enodes)
        {
            const ENode &enode = egraph.getENode(enode_id);
            breakdown[toString(enode.getOpType()) + "@" +
                      (enode.getDebugOrigin().empty() ? std::string("N/A") : enode.getDebugOrigin())]++;
        }
        for (const auto &[key, count] : breakdown)
            out << " " << key << "x" << count;
        out << "\n";
    }
    std::cout << out.str() << std::flush;
}

} // namespace egraph_diagnostics
