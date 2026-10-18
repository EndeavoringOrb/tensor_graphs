// tensor_graphs_cpp/core/plan/propagators/relaxation_bound.hpp
#pragma once

#include <chrono>
#include <sstream>
#include <thread>

#include "core/common/thread_pool.hpp"
#include "core/plan/propagators/base.hpp"
#include "core/plan/relaxation_bound.hpp"

namespace plan
{

// Root lower bound on every bucket's makespan from the selection relaxation
// (docs/core/relaxation_bound.md). It is computed once, at the quiescence of
// the first (root) propagation, over the root domains: a GLOBAL batch runs
// only after the worklist and every domain-changing batch are exhausted.
// Every search node's domains are a subset of the root's, so the bound stays
// valid for the whole tree.
class RelaxationBoundPropagator : public Propagator
{
    bool computed_ = false;
    std::string conflict_;

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }
    PropagationBatch propagationBatch(const SearchState &, VarId) const override
    {
        return computed_ ? PropagationBatch::SKIP : PropagationBatch::GLOBAL;
    }
    bool changesDomains() const override { return false; }
    std::string name() const override { return "RelaxationBoundPropagator"; }
    std::string conflictReason() const override { return conflict_; }

    bool propagate(SearchState &state, VarId, std::vector<VarId> &) override
    {
        if (computed_)
            return true;
        computed_ = true;
        conflict_.clear();
        const auto start = std::chrono::steady_clock::now();
        const uint32_t nb = static_cast<uint32_t>(state.buckets.size());
        std::vector<double> bounds(nb, 0.0);
        std::vector<std::string> notes(nb);
        std::vector<char> infeasible(nb, 0);
        auto &pool = ThreadPool::get();

        // 1. Relaxations of the weighted buckets.
        std::vector<RelaxationBucket> rbs(nb);
        std::vector<char> active(nb, 0);
        pool.parallel_for(nb, [&](uint32_t b) {
            const float weight = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0f;
            if (!(weight > 0.0f))
            {
                notes[b] = "skipped (zero weight)";
                return;
            }
            try {
                rbs[b] = buildRelaxationBucket(state, b);
            } catch (const std::exception &e) {
                LOG(ERROR) << "[buildRelaxationBucket exception] bucket=" << b << ": " << e.what();
                throw;
            }
            if (!rbs[b].acyclic)
            {
                notes[b] = "skipped (cyclic bucket graph)";
                return;
            }
            if (!rbs[b].root_feasible)
            {
                infeasible[b] = 1;
                return;
            }
            if (std::getenv("TG_DEBUG_BOUND"))
            {
                double sum_min_cpu = 0.0;
                double sum_min_all = 0.0;
                uint32_t n_with_opencl = 0;
                for (const auto &rc : rbs[b].classes)
                {
                    double min_cpu = 1e9;
                    double min_all = 1e9;
                    bool has_opencl = false;
                    for (const auto &alt : rc.alternatives)
                    {
                        if (alt.is_alias || alt.is_copy || alt.is_view || alt.is_source) continue;
                        if (alt.duration > 0.0)
                        {
                            min_all = std::min(min_all, alt.duration);
                            if (alt.engine_mask & 1)
                                min_cpu = std::min(min_cpu, alt.duration);
                            if (alt.engine_mask & 2)
                                has_opencl = true;
                        }
                    }
                    if (min_cpu < 1e8) sum_min_cpu += min_cpu;
                    if (min_all < 1e8) sum_min_all += min_all;
                    if (has_opencl) n_with_opencl++;
                }
                LOG(INFO) << "[debug bound stats] bucket=" << b
                          << " sum_min_cpu=" << sum_min_cpu
                          << " sum_min_all=" << sum_min_all
                          << " n_with_opencl=" << n_with_opencl
                          << " total_classes=" << rbs[b].classes.size();
            }
            active[b] = 1;
        });

        // 2. Buckets with identical relaxations (decode buckets often are)
        // have identical bounds; compute each distinct relaxation once.
        std::vector<std::vector<uint64_t>> signature(nb);
        pool.parallel_for(nb, [&](uint32_t b) {
            if (active[b])
                signature[b] = relaxationSignature(rbs[b]);
        });
        std::vector<uint32_t> rep(nb, UINT32_MAX), reps;
        for (uint32_t b = 0; b < nb; ++b)
        {
            if (!active[b])
                continue;
            for (uint32_t r : reps)
                if (signature[r] == signature[b])
                {
                    rep[b] = r;
                    break;
                }
            if (rep[b] == UINT32_MAX)
            {
                rep[b] = b;
                reps.push_back(b);
            }
        }

        // 3. Per distinct relaxation (in parallel): load bound, and interval
        // bound over a geometric family of window strides, from every link
        // upwards, doubling, until a stride falls clearly below the best
        // value (coarser windows then only lose) or the windows cover a
        // quarter of the chain. Every stride gives a valid bound.
        std::vector<size_t> chain(nb, 0), best_stride(nb, 0);
        std::vector<double> load_value(nb, 0.0), interval_value(nb, 0.0);
        std::vector<std::vector<double>> load_lambda(nb);
        pool.parallel_for(static_cast<uint32_t>(reps.size()), [&](uint32_t k) {
            try {
                const uint32_t r = reps[k];
                RelaxationBoundComputer computer(rbs[r]);
            load_value[r] = maximizeRelaxationBound(computer, rbs[r].engines.size(), &load_lambda[r]);
            chain[r] = computer.intervalChainLength();
            if (std::getenv("TG_DEBUG_BOUND"))
            {
                LOG(INFO) << "[debug bound] bucket=" << r
                          << " load_val=" << load_value[r]
                          << " lambda=[" << (load_lambda[r].size() > 0 ? load_lambda[r][0] : 0)
                          << ", " << (load_lambda[r].size() > 1 ? load_lambda[r][1] : 0) << "]"
                          << " chain_len=" << chain[r];
            }
            // Strides 1, 2, 4, ... (stopping once a stride falls clearly below
            // the best), each also with windows shifted by half a stride; the
            // best partition into windows from any of these runs is kept.
            std::vector<IntervalRun> runs;
            for (size_t stride = 1; stride <= std::max<size_t>(1, chain[r]); stride *= 2)
            {
                const double v = computer.intervalBound(stride);
                runs.push_back({computer.windowLastLinks(), computer.windowValues()});
                if (std::getenv("TG_DEBUG_BOUND"))
                {
                    LOG(INFO) << "[debug bound] stride=" << stride << " v=" << v << " n_windows=" << computer.windowValues().size();
                    for (size_t wi = 0; wi < std::min<size_t>(10, computer.windowValues().size()); ++wi)
                        LOG(INFO) << "  w[" << wi << "]=" << computer.windowValues()[wi];
                }
                if (stride >= 2)
                {
                    computer.intervalBound(stride, stride / 2);
                    runs.push_back({computer.windowLastLinks(), computer.windowValues()});
                }
                if (v > interval_value[r])
                {
                    interval_value[r] = v;
                    best_stride[r] = stride;
                }
                else if (v < 0.99 * interval_value[r])
                    break;
            }
            const double combined = bestWindowPartition(chain[r], runs);
            if (combined > interval_value[r])
            {
                interval_value[r] = combined;
                best_stride[r] = 0; // mixed strides
            }
            if (std::getenv("TG_DEBUG_BOUND"))
            {
                LOG(INFO) << "[debug bound] final interval_val=" << interval_value[r] << " combined=" << combined;
            }
            } catch (const std::exception &e) {
                LOG(ERROR) << "[computer exception]: " << e.what();
                throw;
            }
        });

        // 4. Every bucket takes its representative's bound.
        for (uint32_t b = 0; b < nb; ++b)
        {
            if (!active[b])
                continue;
            const uint32_t r = rep[b];
            const double value = std::max(load_value[r], interval_value[r]);
            // Float simulation of n operations can undershoot the exact
            // engine load by a relative n * 2^-24; keep the bound below it.
            const double n = static_cast<double>(rbs[r].classes.size()) + 64.0;
            bounds[b] = std::max(0.0, value * (1.0 - n * 1.2e-7) - 1e-7);
            std::ostringstream os;
            os << "load=" << load_value[r] << " interval=" << interval_value[r] << " stride=" << best_stride[r]
               << " chain=" << chain[r] << " same-as=" << r << " lambda=[";
            for (size_t e = 0; e < load_lambda[r].size(); ++e)
                os << (e ? "," : "") << load_lambda[r][e];
            os << "]";
            notes[b] = os.str();
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        state.bucket_relaxation_lower_bounds.assign(nb, 0.0f);
        for (uint32_t b = 0; b < nb; ++b)
        {
            if (infeasible[b])
            {
                conflict_ = "bucket " + std::to_string(b) + " root has no feasible alternative";
                return false;
            }
            // Round toward zero so the stored float never exceeds the bound.
            float stored = static_cast<float>(bounds[b]);
            if (static_cast<double>(stored) > bounds[b])
                stored = std::nextafter(stored, 0.0f);
            state.bucket_relaxation_lower_bounds[b] = stored;
            state.updateLowerBoundBucket(b);
            LOG(INFO) << "[RelaxationBound] bucket=" << b << " bound=" << stored << " " << notes[b];
        }
        LOG(INFO) << "[RelaxationBound] total lower bound=" << state.lower_bound << " elapsed=" << elapsed << "s";
        return true;
    }
};

} // namespace plan
