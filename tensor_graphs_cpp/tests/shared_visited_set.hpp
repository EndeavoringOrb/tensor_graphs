#pragma once

#include <exception>
#include <thread>
#include <vector>

#include "core/shared_visited_set.hpp"
#include "core/egraph.hpp"
#include "core/types.hpp"

namespace SharedVisitedSetRegression
{
inline void run()
{
    SharedVisitedSet<uint32_t> original;
    for (uint32_t key = 0; key < 10000; ++key)
        if (!original.insert(key))
            Error::throw_err("[SharedVisitedSet] initial insertion failed");

    constexpr uint32_t bucket_count = 12;
    std::vector<SharedVisitedSet<uint32_t>> buckets(bucket_count, original);
    std::vector<std::exception_ptr> errors(bucket_count);
    std::vector<std::thread> threads;
    for (uint32_t bucket = 0; bucket < bucket_count; ++bucket)
        threads.emplace_back([&, bucket] {
            try
            {
                for (uint32_t key = 0; key < 10000; ++key)
                    if (!buckets[bucket].contains(key) || buckets[bucket].insert(key))
                        Error::throw_err("[SharedVisitedSet] inherited visit was lost or inserted twice");
                if (!buckets[bucket].insert(10000 + bucket) || buckets[bucket].size() != 10001)
                    Error::throw_err("[SharedVisitedSet] bucket insertion or size is incorrect");
            }
            catch (...)
            {
                errors[bucket] = std::current_exception();
            }
        });
    for (auto &thread : threads)
        thread.join();
    for (const auto &error : errors)
        if (error)
            std::rethrow_exception(error);

    if (original.size() != 10000 || !original.insert(20000))
        Error::throw_err("[SharedVisitedSet] writing clones changed their source");
    for (uint32_t bucket = 0; bucket < bucket_count; ++bucket)
    {
        if (buckets[bucket].contains(20000))
            Error::throw_err("[SharedVisitedSet] source write leaked into a clone");
        for (uint32_t other = 0; other < bucket_count; ++other)
            if (buckets[bucket].contains(10000 + other) != (bucket == other))
                Error::throw_err("[SharedVisitedSet] writes leaked between buckets");
        auto nested = buckets[bucket];
        if (!nested.insert(30000) || buckets[bucket].contains(30000) || nested.size() != 10002)
            Error::throw_err("[SharedVisitedSet] nested clone is not independent");
    }

    SharedMemo<uint32_t, uint32_t> memo;
    memo.insert(7, 11);
    auto first = memo;
    auto second = memo;
    first.insert(8, 12);
    second.insert(8, 13);
    second.insert(7, 99);
    if (memo.find(8) || !first.find(8) || *first.find(8) != 12 ||
        !second.find(8) || *second.find(8) != 13 || *second.find(7) != 11)
        Error::throw_err("[SharedMemo] cloned entries or updates are not independent");
    first.clear();
    if (first.find(7) || !second.find(7) || *memo.find(7) != 11)
        Error::throw_err("[SharedMemo] clearing a clone changed another memo");

    first = memo;
    first.insertOrAssign(7, 99);
    auto nested_memo = first;
    nested_memo.insertOrAssign(7, 100);
    uint32_t seen = 0;
    nested_memo.forEach([&](uint32_t key, uint32_t value) {
        if (key != 7 || value != 100)
            Error::throw_err("[SharedMemo] iteration exposed a shadowed value");
        ++seen;
    });
    if (*memo.find(7) != 11 || *first.find(7) != 99 || seen != 1)
        Error::throw_err("[SharedMemo] replacing entries changed another layer");

    EGraph graph;
    graph.addEClass({1}, {1}, DType::INT32, MemSpace{1, HandleType::CPP});
    const EClassId constant = graph.addIntConst({42});
    graph.op_materializations.insert("test", {constant, {constant}});
    auto graph_copy = graph;
    graph.merge(EClassId{0}, constant);
    graph.rebuild(true);
    if (graph.op_materializations.find("test") || !graph_copy.op_materializations.find("test"))
        Error::throw_err("[SharedMemo] graph compaction did not invalidate only its own memo");

    EGraph baseline, optimized;
    for (uint32_t iteration = 0; iteration < 64; ++iteration)
    {
        ENode node(KernelId{0}, OpType::INPUT, "same", {}, {1}, {1}, DType::FLOAT32,
                   MemSpace{1, HandleType::CPP}, {});
        const EClassId allocated = baseline.addEClass({1}, {1}, DType::FLOAT32, MemSpace{1, HandleType::CPP});
        const EClassId expected = baseline.addENode(allocated, node);
        const EClassId actual = optimized.addENodeInNewClass(node);
        if (baseline.ufSize[expected.value] != optimized.ufSize[actual.value] ||
            baseline.enodes != optimized.enodes ||
            baseline.getEClass(expected).enodes != optimized.getEClass(actual).enodes)
            Error::throw_err("[EGraph] optimized alias allocation changed union decisions or expressions");
        if (baseline.findConst(baseline.nodeToEClass[0]) != expected ||
            optimized.findConst(optimized.nodeToEClass[0]) != actual)
            Error::throw_err("[EGraph] optimized alias allocation changed class membership");
    }
    if (optimized.classes.size() != 2)
        Error::throw_err("[EGraph] immediately losing aliases were allocated");
    const EClassId old_root = optimized.findConst(EClassId{0});
    const uint32_t old_weight = optimized.ufSize[old_root.value];
    const auto old_enodes = optimized.getEClass(old_root).enodes;
    const auto root_map = optimized.compactClasses();
    if (optimized.classes.size() != 1 || optimized.ufSize[0] != old_weight ||
        optimized.classes[0].enodes != old_enodes || root_map[old_root.value] != EClassId{0})
        Error::throw_err("[EGraph] class-only compaction changed expression order or union weights");
}
}
