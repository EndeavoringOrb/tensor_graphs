#pragma once
#include "tests/incremental_propagators.hpp"

// Group-based rank windows: x_cpp (1) and x_ocl (2) hold the same tensor and
// are linked by COPY_TO. The root reaches either copy, so no single class is
// necessary below it, but the {1, 2} group is.
class StartRankWindowRegression
{
  public:
    static void run()
    {
        using namespace incremental_propagator_test;
        SearchState state = makeState({
            {AlternativeSpec({1}), AlternativeSpec({2})},
            {AlternativeSpec({3})},
            {AlternativeSpec({3}), AlternativeSpec({1}, {}, 1.0f, false, OpType::COPY_TO)},
            {AlternativeSpec({}, {}, 0.0f, false, OpType::INPUT)},
        });
        state.recordStartDomainShapeAtEntry();
        StartRankWindowPropagator windows;
        std::vector<VarId> worklist;
        require(windows.propagate(state, kInvalidVarId, worklist), "Rank windows reported a conflict");
        require(state.domains[start(state, 3)].isFixed() && state.domains[start(state, 3)].fixedValue() == 0,
                "Source must take prefix rank 0");
        require(state.domains[start(state, 0)].getMin() == 2,
                "Root must follow the source and one member of the COPY_TO group, got " +
                    state.domains[start(state, 0)].toString());
        // Canonical witnesses of every plan: (node, rank) of the active classes.
        const std::vector<std::vector<std::pair<uint32_t, int32_t>>> witnesses = {
            {{3, 0}, {1, 1}, {0, 2}},
            {{3, 0}, {2, 1}, {0, 2}},
            {{3, 0}, {1, 1}, {2, 2}, {0, 3}},
        };
        for (const auto &witness : witnesses)
            for (const auto &[node, rank] : witness)
                require(state.domains[start(state, node)].contains(rank),
                        "Rank window excludes a canonical witness: node " + std::to_string(node) + " rank " +
                            std::to_string(rank) + " domain " + state.domains[start(state, node)].toString());
        std::cout << "Start rank window group regression passed" << std::endl;
    }
};
