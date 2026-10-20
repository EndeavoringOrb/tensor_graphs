#pragma once
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
        void record()
        {
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
                    if (ch <= u)
                        ok = false; // generators only create edges to larger ids
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

    static void checkGraph(const GraphSpec &spec, uint32_t trial, size_t &bound_checks, size_t &positive)
    {
        using namespace incremental_propagator_test;
        SearchState state = makeState(spec);
        RelaxationBucket rb = buildRelaxationBucket(state, 0, /*allow_checkpoints=*/false);
        require(rb.acyclic, "generated graph must be acyclic");
        Brute brute{spec, rb.engines, std::vector<int>(spec.size(), -1), {}};
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
            const double bound = computer.evaluate(lambda);
            const double truth = best_weighted(lambda);
            ++bound_checks;
            positive += bound > 1e-9;
            require(bound <= truth + 1e-9, "trial " + std::to_string(trial) + ": bound " + std::to_string(bound) +
                                               " exceeds weighted optimum " + std::to_string(truth));
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
        computer.setExactWindowNodes(20000);
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

  public:
    static void run()
    {
        std::mt19937 rng(20261012);
        size_t checks = 0, positive = 0;
        for (uint32_t trial = 0; trial < 4000; ++trial)
            checkGraph(randomGraph(rng), trial, checks, positive);
        for (uint32_t trial = 0; trial < 2000; ++trial)
            checkGraph(layeredGraph(rng), 100000 + trial, checks, positive);
        require(positive > checks / 4, "bound is almost always zero; the test is not exercising it");
        require(interval_stronger > interval_checks / 20, "interval bound rarely beats the load bound; the test is "
                                                          "not exercising it");
        std::cout << "Relaxation bound regression passed (" << checks << " weighted checks, " << positive
                  << " positive; " << interval_checks << " interval checks, " << interval_positive << " positive, "
                  << interval_stronger << " above the load bound; " << cut_checks << " cut-DP checks, " << cut_tight
                  << " reaching the exact window values; " << dyadic_checks << " window partitions)" << std::endl;
    }
};
