#pragma once
#include <functional>
#include <random>

#include "core/plan/relaxation_bound.hpp"
#include "tests/incremental_propagators.hpp"

// Randomized check of the selection-relaxation bound against brute-force
// enumeration. For every evaluated engine weighting lambda the bound must not
// exceed min over plans of sum_e lambda_e * load_e, and the maximized bound
// must not exceed min over plans of max_e load_e. Graphs mix shared children,
// optional alternatives, COPY_TO groups (absent from the oracle schema),
// views, sources, infinite costs, empty and multi-engine lists, and layered
// chains that create many bound regions.
class RelaxationBoundRegression
{
    using GraphSpec = incremental_propagator_test::GraphSpec;
    using AlternativeSpec = incremental_propagator_test::AlternativeSpec;

    static Engine cpu() { return Engine{0, EngineType::CPU}; }
    static Engine gpu() { return Engine{0, EngineType::QUALCOMM_IGPU}; }
    static Engine cpu1() { return Engine{1, EngineType::CPU}; }

    static void require(bool condition, const std::string &message)
    {
        if (!condition)
            Error::throw_err("RelaxationBoundRegression: " + message);
    }

    struct Brute
    {
        const GraphSpec &spec;
        const std::vector<Engine> &engines;
        std::vector<int> choice;
        std::vector<std::vector<double>> plan_loads;
        double min_makespan = std::numeric_limits<double>::infinity();
        // Cyclic specs: any child id; selections with a cycle are no plans.
        bool allow_cycles = false;

        double duration(const AlternativeSpec &alt) const
        {
            if (alt.is_view || alt.op_type == OpType::INPUT || alt.op_type == OpType::CACHE)
                return 0.0;
            return std::isinf(alt.cost) ? 1.0 : alt.cost;
        }
        size_t engineIndex(const Engine &e) const
        {
            for (size_t i = 0; i < engines.size(); ++i)
                if (engines[i] == e)
                    return i;
            return engines.size();
        }
        // Every dispatch order of the active classes (children first),
        // simulated like MakespanSimulator; keeps the minimum makespan.
        void orders(std::vector<uint32_t> &active, std::vector<uint8_t> &done, std::vector<double> &finish,
                    std::vector<double> &engine_free, size_t placed, double makespan)
        {
            if (makespan >= min_makespan)
                return;
            if (placed == active.size())
            {
                min_makespan = makespan;
                return;
            }
            for (uint32_t u : active)
            {
                if (done[u])
                    continue;
                const AlternativeSpec &alt = spec[u][static_cast<size_t>(choice[u])];
                bool ready = true;
                double start = 0.0;
                for (uint32_t ch : alt.children)
                {
                    if (!done[ch])
                        ready = false;
                    else
                        start = std::max(start, finish[ch]);
                }
                if (!ready)
                    continue;
                const bool source = alt.op_type == OpType::INPUT || alt.op_type == OpType::CACHE;
                std::vector<double> saved = engine_free;
                std::vector<size_t> used;
                if (source)
                    start = 0.0;
                else if (!alt.is_view)
                {
                    if (alt.engines.empty())
                        used.push_back(engineIndex(cpu()));
                    for (const Engine &e : alt.engines)
                        used.push_back(engineIndex(e));
                    for (size_t idx : used)
                    {
                        require(idx < engines.size(), "plan uses an engine unknown to the relaxation");
                        start = std::max(start, engine_free[idx]);
                    }
                }
                const double end = start + duration(alt);
                for (size_t idx : used)
                    engine_free[idx] = end;
                done[u] = 1;
                finish[u] = end;
                orders(active, done, finish, engine_free, placed + 1, std::max(makespan, end));
                done[u] = 0;
                engine_free = saved;
            }
        }
        bool selectionCyclic() const
        {
            std::vector<uint8_t> state(spec.size(), 0);
            std::function<bool(uint32_t)> visit = [&](uint32_t u) {
                if (state[u] == 1)
                    return true;
                if (state[u] == 2)
                    return false;
                state[u] = 1;
                for (uint32_t ch : spec[u][static_cast<size_t>(choice[u])].children)
                    if (visit(ch))
                        return true;
                state[u] = 2;
                return false;
            };
            for (uint32_t u = 0; u < spec.size(); ++u)
                if (choice[u] >= 0 && visit(u))
                    return true;
            return false;
        }
        void record()
        {
            if (allow_cycles && selectionCyclic())
                return;
            std::vector<uint32_t> active;
            for (uint32_t u = 0; u < spec.size(); ++u)
                if (choice[u] >= 0)
                    active.push_back(u);
            std::vector<uint8_t> done(spec.size(), 0);
            std::vector<double> finish(spec.size(), 0.0), engine_free(engines.size(), 0.0);
            orders(active, done, finish, engine_free, 0, 0.0);
            std::vector<double> loads(engines.size(), 0.0);
            for (uint32_t u = 0; u < spec.size(); ++u)
            {
                if (choice[u] < 0)
                    continue;
                const AlternativeSpec &alt = spec[u][static_cast<size_t>(choice[u])];
                const double d = duration(alt);
                if (d <= 0.0)
                    continue;
                std::vector<Engine> used = alt.engines.empty() ? std::vector<Engine>{cpu()} : alt.engines;
                for (const Engine &e : used)
                {
                    size_t idx = engines.size();
                    for (size_t i = 0; i < engines.size(); ++i)
                        if (engines[i] == e)
                            idx = i;
                    require(idx < engines.size(), "plan uses an engine unknown to the relaxation");
                    loads[idx] += d;
                }
            }
            plan_loads.push_back(loads);
        }
        // Assign alternatives to every class needed from the root.
        void enumerate(std::vector<uint32_t> pending)
        {
            while (!pending.empty() && choice[pending.back()] >= 0)
                pending.pop_back();
            if (pending.empty())
            {
                record();
                return;
            }
            const uint32_t u = pending.back();
            pending.pop_back();
            for (size_t a = 0; a < spec[u].size(); ++a)
            {
                choice[u] = static_cast<int>(a);
                std::vector<uint32_t> next = pending;
                bool ok = true;
                for (uint32_t ch : spec[u][a].children)
                {
                    if (ch <= u && !allow_cycles)
                        ok = false; // acyclic generators only create edges to larger ids
                    if (choice[ch] < 0)
                        next.push_back(ch);
                }
                if (ok)
                    enumerate(next);
                choice[u] = -1;
            }
        }
    };

    static GraphSpec randomGraph(std::mt19937 &rng)
    {
        std::uniform_int_distribution<int> nodes_dist(3, 8), alts_dist(1, 3), pick(0, 99);
        const uint32_t n = static_cast<uint32_t>(nodes_dist(rng));
        GraphSpec spec(n);
        for (uint32_t u = 0; u < n; ++u)
        {
            const uint32_t remaining = n - 1 - u;
            const int alts = remaining == 0 ? 1 : alts_dist(rng);
            for (int a = 0; a < alts; ++a)
            {
                AlternativeSpec alt;
                const int kind = pick(rng);
                if (remaining == 0 || kind < 10)
                {
                    alt.op_type = OpType::INPUT;
                    alt.cost = 0.0f;
                    spec[u].push_back(alt);
                    continue;
                }
                std::uniform_int_distribution<uint32_t> child_dist(u + 1, n - 1);
                const int num_children = 1 + pick(rng) % std::min<uint32_t>(3, remaining);
                for (int c = 0; c < num_children; ++c)
                    alt.children.push_back(child_dist(rng));
                if (kind < 25)
                {
                    alt.op_type = OpType::COPY_TO;
                    alt.children.resize(1);
                }
                alt.is_view = kind >= 25 && kind < 30;
                alt.cost = static_cast<float>(pick(rng) % 33) / 4.0f;
                if (pick(rng) < 3)
                    alt.cost = TGConstants::INF;
                const int engine_kind = pick(rng);
                if (engine_kind < 40)
                    alt.engines = {cpu()};
                else if (engine_kind < 80)
                    alt.engines = {gpu()};
                else if (engine_kind < 88)
                    alt.engines = {cpu(), gpu()};
                else if (engine_kind < 94)
                    alt.engines = {cpu1()};
                spec[u].push_back(alt);
            }
        }
        return spec;
    }

    // Residual-style chain: h_l on CPU and its GPU copy, alternative kernels
    // per layer, a split head with two halves and an optional fused bypass.
    static GraphSpec layeredGraph(std::mt19937 &rng)
    {
        std::uniform_int_distribution<int> pick(0, 99);
        auto cost = [&]() { return static_cast<float>(1 + pick(rng) % 16) / 4.0f; };
        const uint32_t layers = 1 + static_cast<uint32_t>(pick(rng) % 3);
        // ids: 0 root, 1 half_a, 2 half_b, then per layer l (from top): cpu, gpu, then input last.
        const uint32_t n = 3 + 2 * layers + 1;
        GraphSpec spec(n);
        const uint32_t input = n - 1;
        auto cpu_id = [&](uint32_t l) { return 3 + 2 * (layers - l); };     // l = layers..1
        auto gpu_id = [&](uint32_t l) { return 3 + 2 * (layers - l) + 1; };
        auto below_cpu = [&](uint32_t l) { return l == 1 ? input : cpu_id(l - 1); };
        auto below_gpu = [&](uint32_t l) { return l == 1 ? input : gpu_id(l - 1); };
        spec[input].push_back(AlternativeSpec({}, {}, 0.0f, false, OpType::INPUT));
        for (uint32_t l = layers; l >= 1; --l)
        {
            spec[cpu_id(l)].push_back(AlternativeSpec({below_cpu(l)}, {cpu()}, cost()));
            spec[cpu_id(l)].push_back(AlternativeSpec({gpu_id(l)}, {cpu()}, cost() / 4.0f, false, OpType::COPY_TO));
            spec[gpu_id(l)].push_back(AlternativeSpec({below_gpu(l)}, {gpu()}, cost()));
            if (pick(rng) < 50)
                spec[gpu_id(l)].push_back(AlternativeSpec({below_cpu(l)}, {gpu()}, cost()));
        }
        const uint32_t top_cpu = cpu_id(layers), top_gpu = gpu_id(layers);
        spec[1].push_back(AlternativeSpec({top_cpu}, {cpu()}, cost()));
        spec[1].push_back(AlternativeSpec({top_gpu}, {gpu()}, cost()));
        spec[2].push_back(AlternativeSpec({top_cpu}, {cpu()}, cost()));
        spec[2].push_back(AlternativeSpec({top_gpu}, {gpu()}, cost()));
        spec[0].push_back(AlternativeSpec({1, 2}, {cpu()}, cost() / 4.0f));
        spec[0].push_back(AlternativeSpec({top_cpu}, {cpu()}, cost() * 3.0f));
        if (pick(rng) < 30)
            spec[0].push_back(AlternativeSpec({input}, {gpu()}, cost() * 4.0f));
        return spec;
    }

    // Logits-style sharing: a concat of chunks, each computed on the CPU from
    // its own transpose (or a cheaper one reading a shared contiguous Q) or
    // copied from a GPU chunk that reads the shared, expensive GPU copy of Q.
    static GraphSpec chunkedGraph(std::mt19937 &rng)
    {
        std::uniform_int_distribution<int> pick(0, 99);
        auto cost = [&]() { return static_cast<float>(1 + pick(rng) % 16) / 4.0f; };
        const uint32_t m = 2 + static_cast<uint32_t>(pick(rng) % 2);
        // ids: 0 root, 1 Q, 2 G (GPU copy of Q), then per chunk C, T, D, input last.
        const uint32_t n = 3 + 3 * m + 1;
        const uint32_t input = n - 1;
        GraphSpec spec(n);
        spec[input].push_back(AlternativeSpec({}, {}, 0.0f, false, OpType::INPUT));
        spec[1].push_back(AlternativeSpec({input}, {cpu()}, 2.0f * cost()));
        spec[2].push_back(AlternativeSpec({1}, {cpu()}, 2.0f * cost(), false, OpType::COPY_TO));
        std::vector<uint32_t> chunks;
        for (uint32_t i = 0; i < m; ++i)
        {
            const uint32_t c = 3 + 3 * i, t = c + 1, d = c + 2;
            chunks.push_back(c);
            spec[c].push_back(AlternativeSpec({t}, {cpu()}, cost()));
            spec[c].push_back(AlternativeSpec({d}, {cpu()}, cost() / 8.0f, false, OpType::COPY_TO));
            if (pick(rng) < 40)
                spec[c].push_back(AlternativeSpec({input}, {cpu()}, 4.0f * cost()));
            spec[t].push_back(AlternativeSpec({input}, {cpu()}, cost()));
            spec[t].push_back(AlternativeSpec({1}, {cpu()}, cost() / 2.0f));
            if (pick(rng) < 30)
                spec[t].push_back(AlternativeSpec({1}, {}, 0.0f, true, OpType::SLICE));
            spec[d].push_back(AlternativeSpec({2}, {gpu()}, cost() / 4.0f));
        }
        spec[0].push_back(AlternativeSpec(chunks, {cpu()}, cost() / 8.0f));
        if (pick(rng) < 50)
            spec[0].push_back(AlternativeSpec({input}, {cpu()}, 6.0f * cost()));
        return spec;
    }

    // Two-form tensors per layer, as in incremental decode: a full tensor F
    // (from the previous full one, or scattered from the part), a part P (from
    // the previous part, or a slice of F) and a GPU copy G of F (computed from
    // the previous G, or copied). F, P and G form a cycle per layer.
    static GraphSpec cyclicGraph(std::mt19937 &rng)
    {
        std::uniform_int_distribution<int> pick(0, 99);
        auto cost = [&]() { return static_cast<float>(1 + pick(rng) % 16) / 4.0f; };
        const uint32_t layers = 1 + static_cast<uint32_t>(pick(rng) % 2);
        // ids: 0 root, per layer l (1-based): F, P, G at 1 + 3 (l - 1); input last.
        const uint32_t n = 1 + 3 * layers + 1;
        const uint32_t input = n - 1;
        GraphSpec spec(n);
        auto f = [&](uint32_t l) { return l == 0 ? input : 1 + 3 * (l - 1); };
        auto p = [&](uint32_t l) { return l == 0 ? input : 2 + 3 * (l - 1); };
        auto g = [&](uint32_t l) { return l == 0 ? input : 3 + 3 * (l - 1); };
        spec[input].push_back(AlternativeSpec({}, {}, 0.0f, false, OpType::INPUT));
        for (uint32_t l = 1; l <= layers; ++l)
        {
            spec[f(l)].push_back(AlternativeSpec({f(l - 1)}, {cpu()}, 2.0f * cost()));
            spec[f(l)].push_back(AlternativeSpec({p(l)}, {cpu()}, cost() / 4.0f));
            spec[f(l)].push_back(AlternativeSpec({g(l)}, {cpu()}, cost() / 4.0f, false, OpType::COPY_TO));
            spec[p(l)].push_back(AlternativeSpec({p(l - 1)}, {pick(rng) < 50 ? cpu() : gpu()}, cost()));
            spec[p(l)].push_back(AlternativeSpec({f(l)}, {}, 0.0f, true, OpType::SLICE));
            spec[g(l)].push_back(AlternativeSpec({g(l - 1)}, {gpu()}, 2.0f * cost()));
            spec[g(l)].push_back(AlternativeSpec({f(l)}, {gpu()}, cost() / 4.0f, false, OpType::COPY_TO));
        }
        spec[0].push_back(AlternativeSpec({p(layers)}, {cpu()}, cost()));
        spec[0].push_back(AlternativeSpec({g(layers)}, {gpu()}, cost()));
        if (pick(rng) < 30)
            spec[0].push_back(AlternativeSpec({f(layers), p(layers)}, {cpu()}, cost() / 2.0f));
        return spec;
    }

    // Small graphs whose alternatives may read any non-root class, so cyclic
    // components of every shape appear (entered through any member, shared
    // with acyclic parts of the graph, with several components per graph).
    static GraphSpec randomCyclicGraph(std::mt19937 &rng)
    {
        std::uniform_int_distribution<int> nodes_dist(4, 7), alts_dist(1, 3), pick(0, 99);
        const uint32_t n = static_cast<uint32_t>(nodes_dist(rng));
        const uint32_t input = n - 1;
        GraphSpec spec(n);
        spec[input].push_back(AlternativeSpec({}, {}, 0.0f, false, OpType::INPUT));
        std::uniform_int_distribution<uint32_t> child_dist(1, n - 1);
        for (uint32_t u = 0; u + 1 < n; ++u)
        {
            const int alts = alts_dist(rng);
            for (int a = 0; a < alts; ++a)
            {
                AlternativeSpec alt;
                const int kind = pick(rng);
                const int num_children = kind < 45 ? 1 : 1 + pick(rng) % 3;
                for (int c = 0; c < num_children; ++c)
                {
                    uint32_t ch = child_dist(rng);
                    if (ch == u)
                        ch = input;
                    alt.children.push_back(ch);
                }
                if (kind < 20)
                    alt.op_type = OpType::COPY_TO;
                else if (kind < 35)
                {
                    alt.is_view = true;
                    alt.op_type = OpType::SLICE;
                }
                // Free ops (not views) derive members of a component from
                // each other at no cost, as in a slice and its scatter back.
                alt.cost = alt.is_view || (kind >= 35 && kind < 45) ? 0.0f : static_cast<float>(pick(rng) % 33) / 4.0f;
                if (!alt.is_view)
                    alt.engines = {pick(rng) < 50 ? cpu() : gpu()};
                spec[u].push_back(alt);
            }
        }
        return spec;
    }

    static void checkGraph(const GraphSpec &spec, uint32_t trial, size_t &bound_checks, size_t &positive)
    {
        using namespace incremental_propagator_test;
        SearchState state = makeState(spec);
        RelaxationBucket rb = buildRelaxationBucket(state, 0, /*allow_checkpoints=*/false);
        require(rb.acyclic, "generated graph must be acyclic (cycles unrolled)");
        bool cyclic = false;
        for (uint32_t u = 0; u < spec.size(); ++u)
            for (const auto &alt : spec[u])
                for (uint32_t ch : alt.children)
                    cyclic = cyclic || ch <= u;
        Brute brute{spec, rb.engines, std::vector<int>(spec.size(), -1), {}};
        brute.allow_cycles = cyclic;
        brute.enumerate({0});
        if (!rb.root_feasible)
        {
            require(brute.plan_loads.empty(), "relaxation reports an infeasible root with a feasible plan");
            return;
        }
        require(!brute.plan_loads.empty(), "relaxation feasible root without any plan");
        RelaxationBoundComputer computer(rb);
        require(computer.ready(), "computer not ready");
        const size_t ne = rb.engines.size();
        auto best_weighted = [&](const std::vector<double> &lambda) {
            double best = std::numeric_limits<double>::infinity();
            for (const auto &loads : brute.plan_loads)
            {
                double v = 0.0;
                for (size_t e = 0; e < ne; ++e)
                    v += lambda[e] * loads[e];
                best = std::min(best, v);
            }
            return best;
        };
        std::vector<std::vector<double>> lambdas;
        if (ne == 0)
            return;
        if (ne == 1)
            lambdas.push_back({1.0});
        else
            for (int k = 1; k < 10; ++k)
            {
                std::vector<double> lambda(ne, 0.0);
                const double t = k / 10.0;
                lambda[0] = t;
                for (size_t e = 1; e < ne; ++e)
                    lambda[e] = (1.0 - t) / static_cast<double>(ne - 1);
                lambdas.push_back(lambda);
            }
        for (const auto &lambda : lambdas)
        {
            std::vector<double> prepared_estimates, cheap_estimates;
            computer.cheapEstimates(lambda, prepared_estimates);
            cheapRelaxationEstimates(rb, lambda, cheap_estimates);
            require(cheap_estimates == prepared_estimates, "cheap estimates differ from the prepared computation");
            const double bound = computer.evaluate(lambda);
            const double truth = best_weighted(lambda);
            ++bound_checks;
            positive += bound > 1e-9;
            require(bound <= truth + 1e-9, "trial " + std::to_string(trial) + ": bound " + std::to_string(bound) +
                                               " exceeds weighted optimum " + std::to_string(truth));
            // Case splits (probes and threshold splits) on every region.
            const double probed = computer.evaluateProbed(lambda, 2, 6, 1, 8);
            ++probe_checks;
            probe_stronger += probed > bound + 1e-9;
            require(probed <= truth + 1e-9, "trial " + std::to_string(trial) + ": probed bound " +
                                                std::to_string(probed) + " exceeds weighted optimum " +
                                                std::to_string(truth));
            // Global case splits on the classes hidden by the region decomposition.
            const double global = computer.evaluateGlobalSplit(lambda, 8);
            ++global_checks;
            global_stronger += global > bound + 1e-9;
            require(global <= truth + 1e-9, "trial " + std::to_string(trial) + ": global split bound " +
                                                std::to_string(global) + " exceeds weighted optimum " +
                                                std::to_string(truth));
            // Nested threshold splits, refined level by level like the search does.
            for (int depth = 0; depth <= 2; ++depth)
            {
                const double split = computer.evaluateSplit(lambda, depth, 4, std::chrono::steady_clock::time_point::max());
                ++split_checks;
                split_stronger += split > bound + 1e-9;
                require(split <= truth + 1e-9, "trial " + std::to_string(trial) + ": depth-" + std::to_string(depth) +
                                                   " split bound " + std::to_string(split) +
                                                   " exceeds weighted optimum " + std::to_string(truth));
            }
        }
        double minmax = std::numeric_limits<double>::infinity();
        for (const auto &loads : brute.plan_loads)
            minmax = std::min(minmax, *std::max_element(loads.begin(), loads.end()));
        const double best = maximizeRelaxationBound(computer, ne);
        require(best <= minmax + 1e-9, "trial " + std::to_string(trial) + ": maximized bound " +
                                           std::to_string(best) + " exceeds min-max load " + std::to_string(minmax));
        // The best dyadic partition over strides 1, 2, 4, 8 must be valid,
        // both with exact window values and with the cut DP.
        for (uint64_t exact_nodes : {uint64_t(20000), uint64_t(0)})
        {
            computer.setExactWindowNodes(exact_nodes);
            std::vector<IntervalRun> runs;
            for (size_t stride : {size_t(1), size_t(2), size_t(3), size_t(4)})
                for (size_t phase = 0; phase < stride; ++phase)
                {
                    computer.intervalBound(stride, phase);
                    runs.push_back({computer.windowLastLinks(), computer.windowValues()});
                }
            const double combined = bestWindowPartition(computer.intervalChainLength(), runs);
            ++dyadic_checks;
            require(combined <= brute.min_makespan + 1e-9,
                    "trial " + std::to_string(trial) + ": best window partition " + std::to_string(combined) +
                        " exceeds the minimum makespan " + std::to_string(brute.min_makespan));
        }
        // The exact window enumeration's pruning keeps its values: with
        // budgets both finish, flexible values with and without it agree.
        computer.setExactWindowNodes(5000000);
        for (size_t max_len : {size_t(2), size_t(8)})
        {
            const double pruned = computer.flexibleIntervalBound(max_len, std::chrono::steady_clock::time_point::max());
            computer.setExactWindowPruning(false);
            const double plain = computer.flexibleIntervalBound(max_len, std::chrono::steady_clock::time_point::max());
            computer.setExactWindowPruning(true);
            ++pruning_checks;
            require(std::abs(pruned - plain) <= 1e-9,
                    "trial " + std::to_string(trial) + ": exact windows with pruning give " + std::to_string(pruned) +
                        ", without " + std::to_string(plain) + " (max_len " + std::to_string(max_len) + ")");
        }
        computer.setExactWindowNodes(20000);
        // Long windows at the partition's boundaries (cut DP values) too.
        for (const auto &[max_len, long_len] : {std::pair<size_t, size_t>{1, 4}, std::pair<size_t, size_t>{2, 8}})
        {
            const double flexible =
                computer.flexibleIntervalBound(max_len, std::chrono::steady_clock::time_point::max(), long_len);
            ++flexible_checks;
            require(flexible <= brute.min_makespan + 1e-9,
                    "trial " + std::to_string(trial) + ": flexible interval bound " + std::to_string(flexible) +
                        " with long windows (" + std::to_string(max_len) + ", " + std::to_string(long_len) +
                        ") exceeds the minimum makespan " + std::to_string(brute.min_makespan));
        }
        // Flexible windows (every short window, best partition) must be valid.
        for (size_t max_len : {size_t(1), size_t(3), size_t(8)})
        {
            const double flexible = computer.flexibleIntervalBound(max_len, std::chrono::steady_clock::time_point::max());
            ++flexible_checks;
            require(flexible <= brute.min_makespan + 1e-9,
                    "trial " + std::to_string(trial) + ": flexible interval bound " + std::to_string(flexible) +
                        " (max_len " + std::to_string(max_len) + ") exceeds the minimum makespan " +
                        std::to_string(brute.min_makespan));
        }
        // Every chain stride gives its own windows; each must be valid.
        for (size_t stride : {size_t(1), size_t(2), size_t(3)})
        {
            const double interval = computer.intervalBound(stride);
            ++interval_checks;
            interval_positive += interval > 1e-9;
            interval_stronger += interval > best + 1e-9;
            require(interval <= brute.min_makespan + 1e-9,
                    "trial " + std::to_string(trial) + ": interval bound " + std::to_string(interval) +
                        " (stride " + std::to_string(stride) + ") exceeds the minimum makespan " +
                        std::to_string(brute.min_makespan));
            // Without exact enumeration the cut DP takes over: still valid,
            // and never above the exact window values.
            computer.setExactWindowNodes(0);
            const double cut = computer.intervalBound(stride);
            computer.setExactWindowNodes(20000);
            ++cut_checks;
            cut_tight += cut >= interval - 1e-6;
            require(cut <= brute.min_makespan + 1e-9,
                    "trial " + std::to_string(trial) + ": cut-DP interval bound " + std::to_string(cut) + " (stride " +
                        std::to_string(stride) + ") exceeds the minimum makespan " +
                        std::to_string(brute.min_makespan));
            require(cut <= interval + 1e-9, "trial " + std::to_string(trial) + ": cut-DP interval bound " +
                                                std::to_string(cut) + " exceeds the exact window value " +
                                                std::to_string(interval));
        }
    }
    static inline size_t interval_checks = 0, interval_positive = 0, interval_stronger = 0;
    static inline size_t cut_checks = 0, cut_tight = 0, dyadic_checks = 0;
    static inline size_t probe_checks = 0, probe_stronger = 0, split_checks = 0, split_stronger = 0;
    static inline size_t global_checks = 0, global_stronger = 0, flexible_checks = 0, pruning_checks = 0;

  public:
    static void run()
    {
        // An expired optimization budget is a zero lower bound, not a
        // partially evaluated weighting or a stale best-lambda output.
        {
            using namespace incremental_propagator_test;
            std::mt19937 deadline_rng(20261022);
            SearchState state = makeState(randomGraph(deadline_rng));
            RelaxationBucket rb = buildRelaxationBucket(state, 0, /*allow_checkpoints=*/false);
            RelaxationBoundComputer expired(rb, std::chrono::steady_clock::now());
            require(!expired.ready(), "expired preparation must not expose a partially prepared bound");
            RelaxationBoundComputer computer(rb);
            std::vector<double> lambda{1.0};
            const double bound = maximizeRelaxationBound(
                computer, rb.engines.size(), &lambda, std::chrono::steady_clock::now());
            require(bound == 0.0 && lambda.empty(), "expired bound optimization must return zero with no weighting");
        }
        std::mt19937 rng(20261012);
        size_t checks = 0, positive = 0;
        // Own generator, so the other families keep their graphs.
        std::mt19937 cyclic_rng(20261021);
        for (uint32_t trial = 0; trial < 2000; ++trial)
            checkGraph(randomCyclicGraph(cyclic_rng), 400000 + trial, checks, positive);
        for (uint32_t trial = 0; trial < 4000; ++trial)
            checkGraph(randomGraph(rng), trial, checks, positive);
        for (uint32_t trial = 0; trial < 2000; ++trial)
            checkGraph(layeredGraph(rng), 100000 + trial, checks, positive);
        for (uint32_t trial = 0; trial < 2000; ++trial)
            checkGraph(cyclicGraph(rng), 300000 + trial, checks, positive);
        const size_t probe_stronger_before = probe_stronger;
        for (uint32_t trial = 0; trial < 2000; ++trial)
            checkGraph(chunkedGraph(rng), 200000 + trial, checks, positive);
        require(probe_stronger > probe_stronger_before, "case splits never tighten chunked graphs; the test is not "
                                                        "exercising them");
        require(positive > checks / 4, "bound is almost always zero; the test is not exercising it");
        require(interval_stronger > interval_checks / 20, "interval bound rarely beats the load bound; the test is "
                                                          "not exercising it");
        std::cout << "Relaxation bound regression passed (" << checks << " weighted checks, " << positive
                  << " positive; " << interval_checks << " interval checks, " << interval_positive << " positive, "
                  << interval_stronger << " above the load bound; " << cut_checks << " cut-DP checks, " << cut_tight
                  << " reaching the exact window values; " << dyadic_checks << " window partitions; " << probe_checks
                  << " probed checks, " << probe_stronger << " above the load bound; " << split_checks
                  << " nested split checks, " << split_stronger << " above the load bound; " << global_checks
                  << " global split checks, " << global_stronger << " above the load bound; " << flexible_checks
                  << " flexible window checks; " << pruning_checks << " exact pruning checks)" << std::endl;
    }
};
