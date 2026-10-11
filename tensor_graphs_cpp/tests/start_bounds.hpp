#pragma once
#include "tests/incremental_propagators.hpp"

inline void runStartBoundsTests()
{
    using namespace incremental_propagator_test;
    std::mt19937 random(17391);
    for (uint32_t trial = 0; trial < 1000; ++trial)
    {
        const uint32_t count = 1 + random() % 6;
        SearchEngine engine(makeState(GraphSpec(count, {AlternativeSpec{}})));
        engine.addPropagator(std::make_unique<StartBoundsPropagator>());
        selectAll(engine.state);
        std::vector<int32_t> lower(count), upper(count), values(count);
        std::vector<int32_t> supported_min(count, INT32_MAX), supported_max(count, INT32_MIN);
        for (uint32_t i = 0; i < count; ++i)
        {
            lower[i] = random() % 7;
            upper[i] = lower[i] + random() % (7 - lower[i]);
            engine.state.setDomain(start(engine.state, i), Domain::makeRange(lower[i], upper[i]));
        }
        bool feasible = false;
        std::function<void(uint32_t, uint32_t)> enumerate = [&](uint32_t node, uint32_t used) {
            if (node == count)
            {
                feasible = true;
                for (uint32_t i = 0; i < count; ++i)
                {
                    supported_min[i] = std::min(supported_min[i], values[i]);
                    supported_max[i] = std::max(supported_max[i], values[i]);
                }
                return;
            }
            for (int32_t value = lower[node]; value <= upper[node]; ++value)
                if (!(used & (1u << value))) { values[node] = value; enumerate(node + 1, used | (1u << value)); }
        };
        enumerate(0, 0);
        std::string reason;
        const size_t marker = engine.state.getTrailMarker();
        const auto before = engine.state.domains;
        const bool ok = engine.runPropagators(kInvalidVarId, &reason);
        require(ok == feasible, "Hall feasibility mismatch at trial " + std::to_string(trial) + ": " + reason);
        if (ok)
            for (uint32_t i = 0; i < count; ++i)
            {
                const Domain &domain = engine.state.domains[start(engine.state, i)];
                require(domain.getMin() == supported_min[i] && domain.getMax() == supported_max[i],
                        "Hall bounds mismatch at trial " + std::to_string(trial) + " node=" + std::to_string(i));
            }
        engine.state.backtrackTo(marker);
        require(engine.state.domains == before, "Hall propagation did not restore domains");
        require(engine.runPropagators(kInvalidVarId) == feasible, "Hall result changed after restore");
    }
    {
        SearchEngine engine(makeState(GraphSpec(3, {AlternativeSpec{}})));
        engine.addPropagator(std::make_unique<StartBoundsPropagator>());
        for (uint32_t i = 0; i < 2; ++i)
            engine.state.setDomain(selection(engine.state, i), Domain::makeFixed(1, true));
        for (uint32_t i = 0; i < 3; ++i)
            engine.state.setDomain(start(engine.state, i), Domain::makeRange(4, 5));
        require(engine.runPropagators(kInvalidVarId), "Optional operation counted in Hall interval");
        engine.state.setDomain(selection(engine.state, 2), Domain::makeFixed(1, true));
        std::string reason;
        require(!engine.runPropagators(selection(engine.state, 2), &reason), "Selection did not activate Hall overload");
        require(reason.find("capacity=2 mandatory=3") != std::string::npos, "Missing Hall overload witness");
    }
    {
        const GraphSpec spec = {{AlternativeSpec({1})}, {AlternativeSpec({2})}, {AlternativeSpec{}}};
        SearchEngine engine(makeState(spec));
        engine.addPropagator(std::make_unique<ConsumerStartPrecedencePropagator>(true));
        engine.addPropagator(std::make_unique<InputPruneStartPrecedencePropagator>());
        engine.addPropagator(std::make_unique<StartPrecedencePropagator>());
        selectAll(engine.state);
        for (uint32_t i = 0; i < 3; ++i)
            engine.state.setDomain(start(engine.state, i), Domain::makeRange(0, 20));
        require(engine.runPropagators(kInvalidVarId), "Initial precedence closure failed");
        engine.state.setDomain(start(engine.state, 2), Domain::makeRange(8, 18));
        require(engine.runPropagators(start(engine.state, 2)), "Nonfixed lower propagation failed");
        require(engine.state.domains[start(engine.state, 0)].getMin() == 10, "Forward closure waited for fixed starts");
        engine.state.setDomain(start(engine.state, 0), Domain::makeRange(10, 14));
        require(engine.runPropagators(start(engine.state, 0)), "Nonfixed upper propagation failed");
        require(engine.state.domains[start(engine.state, 2)].getMax() == 12, "Backward closure waited for fixed starts");
    }
    {
        const GraphSpec spec = {{AlternativeSpec({1}), AlternativeSpec({2})}, {AlternativeSpec{}}, {AlternativeSpec{}}};
        SearchEngine engine(makeState(spec));
        engine.addPropagator(std::make_unique<ConsumerStartPrecedencePropagator>());
        engine.addPropagator(std::make_unique<InputPruneStartPrecedencePropagator>());
        engine.addPropagator(std::make_unique<StartPrecedencePropagator>());
        for (uint32_t i = 0; i < 3; ++i)
            engine.state.setDomain(selection(engine.state, i), i == 0 ? Domain::makeMask(6) : i == 1 ? Domain::makeMask(3) : Domain::makeFixed(1, true));
        engine.state.setDomain(start(engine.state, 0), Domain::makeRange(0, 10));
        engine.state.setDomain(start(engine.state, 1), Domain::makeRange(0, 20));
        engine.state.setDomain(start(engine.state, 2), Domain::makeRange(0, 20));
        require(engine.runPropagators(kInvalidVarId), "Alternative precedence initial failed");
        require(engine.state.domains[start(engine.state, 1)].getMax() == 20, "Conditional edge treated as mandatory");
        engine.state.setDomain(start(engine.state, 1), Domain::makeRange(11, 20));
        require(engine.runPropagators(start(engine.state, 1)), "Alternative pruning failed");
        require(engine.state.domains[selection(engine.state, 0)].fixedValue() == 2, "Unsupported alternative survived");
        require(engine.state.domains[start(engine.state, 2)].getMax() == 9, "Selection did not activate backward edge");
    }
    {
        // A possible view dependency is not an unconditional alias path.
        const GraphSpec spec = {
            {AlternativeSpec({2})}, {AlternativeSpec{}},
            {AlternativeSpec({1}, {}, 0.0f, true), AlternativeSpec({3}, {}, 0.0f, true)},
            {AlternativeSpec{}}
        };
        SearchState state = makeState(spec);
        state.setDomain(selection(state, 0), Domain::makeFixed(1, true));
        state.setDomain(selection(state, 1), Domain::makeFixed(1, true));
        state.setDomain(start(state, 0), Domain::makeRange(1, 5));
        state.setDomain(start(state, 1), Domain::makeRange(8, 10));
        state.setDomain(start(state, 2), Domain::makeRange(0, 20));
        ConsumerStartPrecedencePropagator propagator;
        std::vector<VarId> worklist;
        require(propagator.propagate(state, start(state, 1), worklist),
                "Possible view alias caused an unconditional precedence conflict");
        require(state.domains[start(state, 0)].getMax() == 5, "Bypassed source constrained the view's consumer");
    }
    // Mask domains are propagated by their hull: preserve every real solution,
    // without claiming domain consistency for holes.
    for (uint32_t trial = 0; trial < 150; ++trial)
    {
        SearchEngine engine(makeState(GraphSpec(4, {AlternativeSpec{}}), 2));
        engine.addPropagator(std::make_unique<StartBoundsPropagator>());
        selectAll(engine.state);
        std::vector<uint32_t> masks(4);
        std::vector<int32_t> minimum(4, INT32_MAX), maximum(4, INT32_MIN), values(4);
        for (uint32_t i = 0; i < 4; ++i)
        {
            masks[i] = 1 + random() % 63;
            for (uint32_t bucket = 0; bucket < 2; ++bucket)
                engine.state.setDomain(start(engine.state, i, bucket), Domain::makeMask(masks[i]));
        }
        bool feasible = false;
        std::function<void(uint32_t, uint32_t)> enumerate = [&](uint32_t node, uint32_t used) {
            if (node == 4)
            {
                feasible = true;
                for (uint32_t i = 0; i < 4; ++i) { minimum[i] = std::min(minimum[i], values[i]); maximum[i] = std::max(maximum[i], values[i]); }
                return;
            }
            for (int32_t value = 0; value < 6; ++value)
                if ((masks[node] & (1u << value)) && !(used & (1u << value)))
                { values[node] = value; enumerate(node + 1, used | (1u << value)); }
        };
        enumerate(0, 0);
        const bool ok = engine.runPropagators(kInvalidVarId);
        if (feasible)
        {
            require(ok, "Hall hull rejected a feasible masked assignment");
            for (uint32_t bucket = 0; bucket < 2; ++bucket)
                for (uint32_t i = 0; i < 4; ++i)
                {
                    const Domain &domain = engine.state.domains[start(engine.state, i, bucket)];
                    require(domain.getMin() <= minimum[i] && domain.getMax() >= maximum[i], "Hall mask removed supported bounds");
                }
        }
    }
    {
        SearchEngine engine(makeState(GraphSpec(3, {AlternativeSpec{}})));
        engine.addPropagator(std::make_unique<StartBoundsPropagator>());
        selectAll(engine.state);
        engine.state.setDomain(start(engine.state, 0), Domain::makeRange(INT32_MAX - 2, INT32_MAX - 1));
        engine.state.setDomain(start(engine.state, 1), Domain::makeRange(INT32_MAX - 2, INT32_MAX - 1));
        engine.state.setDomain(start(engine.state, 2), Domain::makeRange(INT32_MAX - 2, INT32_MAX));
        require(engine.runPropagators(kInvalidVarId), "Large-rank Hall propagation failed");
        require(engine.state.domains[start(engine.state, 2)].fixedValue() == INT32_MAX, "Large-rank reservation failed");
    }
    {
        constexpr uint32_t count = 20000;
        SearchEngine engine(makeState(GraphSpec(count, {AlternativeSpec{}})));
        engine.addPropagator(std::make_unique<StartBoundsPropagator>());
        selectAll(engine.state);
        for (uint32_t i = 0; i < count; ++i)
            engine.state.setDomain(start(engine.state, i), i < count - 10 ? Domain::makeFixed(i)
                : Domain::makeRange(0, count - 1));
        const auto begin = std::chrono::steady_clock::now();
        require(engine.runPropagators(kInvalidVarId), "Large Hall suffix failed");
        for (uint32_t i = count - 10; i < count; ++i)
            require(engine.state.domains[start(engine.state, i)].getMin() == count - 10, "Large Hall prefix not reserved");
        std::cout << "20,000-start Hall propagation: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() << "s" << std::endl;
    }
    std::cout << "Start bounds and precedence tests passed" << std::endl;
}
