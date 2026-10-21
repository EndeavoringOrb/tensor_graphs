// tensor_graphs_cpp/core/plan/propagators/relaxation_bound.hpp
#pragma once

#include <chrono>
#include <memory>
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
    // Kept after the root computation so refine() can tighten the bounds.
    std::vector<RelaxationBucket> rbs_;
    std::vector<std::unique_ptr<RelaxationBoundComputer>> computers_; // per representative bucket
    std::vector<uint32_t> rep_, reps_;
    std::vector<char> active_;
    std::vector<double> load_value_, interval_value_;
    std::vector<std::vector<double>> load_lambda_;

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
        std::vector<std::string> notes(nb);
        std::vector<char> infeasible(nb, 0);
        auto &pool = ThreadPool::get();

        // 1. Relaxations of the weighted buckets.
        rbs_.assign(nb, RelaxationBucket{});
        std::vector<RelaxationBucket> &rbs = rbs_;
        active_.assign(nb, 0);
        std::vector<char> &active = active_;
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
        rep_.assign(nb, UINT32_MAX);
        reps_.clear();
        std::vector<uint32_t> &rep = rep_, &reps = reps_;
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

        // 3. Per distinct relaxation (in parallel): the load bound. The
        // interval bound (computeIntervals) comes after the root heuristic.
        std::vector<size_t> chain(nb, 0), best_stride(nb, 0);
        load_value_.assign(nb, 0.0);
        interval_value_.assign(nb, 0.0);
        load_lambda_.assign(nb, {});
        computers_.clear();
        computers_.resize(nb);
        std::vector<double> &load_value = load_value_, &interval_value = interval_value_;
        std::vector<std::vector<double>> &load_lambda = load_lambda_;
        pool.parallel_for(static_cast<uint32_t>(reps.size()), [&](uint32_t k) {
            try {
                const uint32_t r = reps[k];
                const auto stage_start = std::chrono::steady_clock::now();
                auto stage = [&]() {
                    return std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count();
                };
                computers_[r] = std::make_unique<RelaxationBoundComputer>(rbs[r]);
                RelaxationBoundComputer &computer = *computers_[r];
                const double prepared_at = stage();
                if (computer.degenerate())
                {
                    LOG(INFO) << "[RelaxationBound] bucket=" << r << " degenerate region decomposition (chain "
                              << computer.intervalChainLength() << "): no bound";
                    return;
                }
            load_value[r] = maximizeRelaxationBound(computer, rbs[r].engines.size(), &load_lambda[r]);
            // Threshold splits (not nested) at the best weighting; refine()
            // nests them later. When the load evaluation is slow (large
            // regions), they wait for refine() so the heuristic starts
            // sooner.
            if (!load_lambda[r].empty() && stage() - prepared_at <= kRootSplitSeconds)
            {
                load_value[r] = std::max(load_value[r], computer.evaluateSplit(load_lambda[r], 0, kSplitCandidates,
                                                                                std::chrono::steady_clock::time_point::max()));
                // Then a case split on the class the region decomposition
                // hides most (docs/core/relaxation_bound.md, "Global split").
                load_value[r] = std::max(load_value[r], computer.evaluateGlobalSplit(load_lambda[r], kGlobalCandidates));
            }
            chain[r] = computer.intervalChainLength();
            LOG(INFO) << "[RelaxationBound timing] bucket=" << r << " prepare=" << prepared_at << "s load+splits="
                      << stage() - prepared_at << "s";
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
            std::ostringstream os;
            os << "load=" << load_value[r] << " chain=" << chain[r] << " same-as=" << r << " lambda=[";
            for (size_t e = 0; e < load_lambda[r].size(); ++e)
                os << (e ? "," : "") << load_lambda[r][e];
            os << "]";
            notes[b] = os.str();
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (uint32_t b = 0; b < nb; ++b)
            if (infeasible[b])
            {
                conflict_ = "bucket " + std::to_string(b) + " root has no feasible alternative";
                return false;
            }
        state.bucket_relaxation_lower_bounds.assign(nb, 0.0f);
        storeBounds(state);
        for (uint32_t b = 0; b < nb; ++b)
            state.updateLowerBoundBucket(b);
        for (uint32_t b = 0; b < nb; ++b)
            LOG(INFO) << "[RelaxationBound] bucket=" << b << " bound=" << state.bucket_relaxation_lower_bounds[b] << " "
                      << notes[b];
        LOG(INFO) << "[RelaxationBound] total lower bound=" << state.lower_bound << " elapsed=" << elapsed << "s";
        return true;
    }

    // Interval bound (docs/core/relaxation_bound.md, "Interval bound") of
    // every representative, over a geometric family of window strides from
    // every link upwards, doubling until a stride does not improve the best
    // value (coarser windows cost about twice as much and rarely gain), and
    // the best partition into windows from any of these runs (each stride
    // also with windows shifted by half a stride). Every stride gives a
    // valid bound. Run after the root heuristic: it can take seconds; no
    // further stride starts after `deadline`.
    // Returns whether any bucket bound rose.
    bool computeIntervals(SearchState &state, std::chrono::steady_clock::time_point deadline)
    {
        if (!computed_ || !conflict_.empty() || intervals_done_)
            return false;
        intervals_done_ = true;
        const auto start = std::chrono::steady_clock::now();
        const std::vector<float> before = state.bucket_relaxation_lower_bounds;
        ThreadPool::get().parallel_for(static_cast<uint32_t>(reps_.size()), [&](uint32_t k) {
            const uint32_t r = reps_[k];
            if (!computers_[r] || !computers_[r]->ready())
                return;
            RelaxationBoundComputer &computer = *computers_[r];
            const size_t chain = computer.intervalChainLength();
            std::vector<IntervalRun> runs;
            double best = 0.0;
            for (size_t stride = 1; stride <= std::max<size_t>(1, chain); stride *= 2)
            {
                if (std::chrono::steady_clock::now() >= deadline)
                    break;
                const double v = computer.intervalBound(stride, 0, deadline);
                runs.push_back({computer.windowLastLinks(), computer.windowValues()});
                if (stride >= 2)
                {
                    computer.intervalBound(stride, stride / 2, deadline);
                    runs.push_back({computer.windowLastLinks(), computer.windowValues()});
                }
                if (v > best)
                    best = v;
                else
                    break;
            }
            interval_value_[r] = std::max(best, bestWindowPartition(chain, runs));
        });
        storeBounds(state);
        for (uint32_t r : reps_)
            LOG(INFO) << "[RelaxationBound] intervals bucket=" << r << " interval=" << interval_value_[r]
                      << " load=" << load_value_[r];
        LOG(INFO) << "[RelaxationBound] intervals elapsed="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "s";
        return state.bucket_relaxation_lower_bounds != before;
    }

    // Flexible interval windows (every window spanning at most
    // kFlexibleLength chain links, best partition) of every representative,
    // (before computeIntervals, which it usually dominates and which is
    // slower). Returns whether any bucket bound rose.
    bool computeFlexibleIntervals(SearchState &state, std::chrono::steady_clock::time_point deadline,
                                  size_t long_len = 0)
    {
        if (!computed_ || !conflict_.empty())
            return false;
        const auto start = std::chrono::steady_clock::now();
        const std::vector<float> before = state.bucket_relaxation_lower_bounds;
        ThreadPool::get().parallel_for(static_cast<uint32_t>(reps_.size()), [&](uint32_t k) {
            const uint32_t r = reps_[k];
            if (!computers_[r] || !computers_[r]->ready())
                return;
            interval_value_[r] = std::max(interval_value_[r],
                                          computers_[r]->flexibleIntervalBound(kFlexibleLength, deadline, long_len));
        });
        storeBounds(state);
        LOG(INFO) << "[RelaxationBound] flexible windows (long " << long_len << ") elapsed="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "s";
        return state.bucket_relaxation_lower_bounds != before;
    }

    // Nests every representative's threshold splits `depth` levels deep
    // (until `deadline`) and raises the stored relaxation bounds. They hold
    // for every plan of the root domains, so the whole tree; the caller
    // folds them into node bounds (the state's own bucket bounds may be
    // stale until the next propagation). Returns whether any bound rose.
    bool refine(SearchState &state, int depth, std::chrono::steady_clock::time_point deadline)
    {
        if (!computed_ || !conflict_.empty())
            return false;
        const std::vector<float> before = state.bucket_relaxation_lower_bounds;
        ThreadPool::get().parallel_for(static_cast<uint32_t>(reps_.size()), [&](uint32_t k) {
            const uint32_t r = reps_[k];
            if (!computers_[r] || !computers_[r]->ready() || load_lambda_[r].empty())
                return;
            load_value_[r] = std::max(load_value_[r], computers_[r]->evaluateSplit(load_lambda_[r], depth,
                                                                                   kSplitCandidates, deadline));
            load_value_[r] = std::max(load_value_[r], computers_[r]->evaluateGlobalSplit(load_lambda_[r],
                                                                                         kGlobalCandidates, deadline));
        });
        storeBounds(state);
        return state.bucket_relaxation_lower_bounds != before;
    }

    // The computer of bucket b's representative relaxation (its flexible
    // partition is set once computeFlexibleIntervals ran), or null.
    RelaxationBoundComputer *computerFor(uint32_t b) const
    {
        if (b >= rep_.size() || !active_[b] || rep_[b] == UINT32_MAX || !computers_[rep_[b]] ||
            !computers_[rep_[b]]->ready())
            return nullptr;
        return computers_[rep_[b]].get();
    }
    const RelaxationBucket &relaxationOf(uint32_t b) const { return rbs_[rep_[b]]; }

  private:
    bool intervals_done_ = false;
    static constexpr size_t kSplitCandidates = 4;
    static constexpr double kRootSplitSeconds = 1.0;
    static constexpr size_t kFlexibleLength = 8;
    static constexpr size_t kGlobalCandidates = 8;

    // Original classes with an alternative of positive duration.
    static size_t positiveClasses(const RelaxationBucket &rb)
    {
        const size_t n = rb.orig_class_count ? rb.orig_class_count : rb.classes.size();
        size_t count = 0;
        for (size_t u = 0; u < n; ++u)
        {
            bool positive = false;
            for (const auto &alt : rb.classes[u].alternatives)
                positive = positive || alt.duration > 0.0;
            count += positive;
        }
        return count;
    }

    // Stores max(load, interval) of each bucket's representative, below the
    // float margin and never below the bound already stored.
    void storeBounds(SearchState &state) const
    {
        const uint32_t nb = static_cast<uint32_t>(state.buckets.size());
        state.bucket_relaxation_lower_bounds.resize(nb, 0.0f);
        for (uint32_t b = 0; b < nb; ++b)
        {
            if (!active_[b])
                continue;
            const uint32_t r = rep_[b];
            const double value = std::max(load_value_[r], interval_value_[r]);
            // Float simulation can undershoot the exact makespan by a
            // relative n * 2^-24, n the operations of positive duration on
            // the chain that ends it, at most the original classes with such
            // an alternative (a plan selects each once); keep below it.
            const double n = static_cast<double>(positiveClasses(rbs_[r])) + 64.0;
            const double bound = std::max(0.0, value * (1.0 - n * 1.2e-7) - 1e-7);
            // Round toward zero so the stored float never exceeds the bound.
            float stored = static_cast<float>(bound);
            if (static_cast<double>(stored) > bound)
                stored = std::nextafter(stored, 0.0f);
            state.bucket_relaxation_lower_bounds[b] = std::max(state.bucket_relaxation_lower_bounds[b], stored);
        }
    }
};

} // namespace plan
