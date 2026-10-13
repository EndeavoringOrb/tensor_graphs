// tensor_graphs_cpp/core/plan/propagators/relaxation_bound.hpp
#pragma once

#include <chrono>
#include <sstream>
#include <thread>

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
        auto work = [&](uint32_t b) {
            const float weight = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0f;
            if (!(weight > 0.0f))
            {
                notes[b] = "skipped (zero weight)";
                return;
            }
            RelaxationBucket rb = buildRelaxationBucket(state, b);
            if (!rb.acyclic)
            {
                notes[b] = "skipped (cyclic bucket graph)";
                return;
            }
            if (!rb.root_feasible)
            {
                infeasible[b] = 1;
                return;
            }
            RelaxationBoundComputer computer(rb);
            std::vector<double> lambda;
            const double value = maximizeRelaxationBound(computer, rb.engines.size(), &lambda);
            // Float simulation of n operations can undershoot the exact
            // engine load by a relative n * 2^-24; keep the bound below it.
            const double n = static_cast<double>(rb.classes.size()) + 64.0;
            bounds[b] = std::max(0.0, value * (1.0 - n * 1.2e-7) - 1e-7);
            std::ostringstream os;
            os << "regions=" << computer.numRegions() << " lambda=[";
            for (size_t e = 0; e < lambda.size(); ++e)
                os << (e ? "," : "") << lambda[e];
            os << "]";
            notes[b] = os.str();
        };
        const unsigned hw = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
        if (hw > 1 && nb > 1)
        {
            std::vector<std::thread> threads;
            std::vector<std::exception_ptr> errors(hw);
            for (unsigned t = 0; t < hw; ++t)
                threads.emplace_back([&, t]() {
                    try
                    {
                        for (uint32_t b = t; b < nb; b += hw)
                            work(b);
                    }
                    catch (...)
                    {
                        errors[t] = std::current_exception();
                    }
                });
            for (auto &thread : threads)
                thread.join();
            for (auto &error : errors)
                if (error)
                    std::rethrow_exception(error);
        }
        else
        {
            for (uint32_t b = 0; b < nb; ++b)
                work(b);
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
