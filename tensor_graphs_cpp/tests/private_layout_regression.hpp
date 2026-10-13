#pragma once
#include "tests/incremental_propagators.hpp"

class PrivateLayoutRegression
{
    static plan::SearchState makePaddedState()
    {
        using namespace incremental_propagator_test;
        SearchState state = makeState({{{{}}}, {{{0}}}});
        state.bucket_root_ids[0] = EClassId{1};
        const MemSpace space{1, HandleType::CPP};
        state.mem_caps[space] = 20; // five four-byte pages
        auto &graph = state.bucket_egraphs[0];
        graph.addENode(EClassId{0}, ENode(KernelId{0}, OpType::ADD, "padded_footprint", {},
            {2}, {2}, DType::FLOAT32, space, {}));
        const auto en_id = graph.getEClass(EClassId{0}).enodes.back();
        state.bucket_enode_infos[0].resize(en_id.value + 1);
        state.bucket_enode_infos[0][en_id.value].cost = 1.0f;
        state.setDomain(selection(state, 0), Domain::makeFixed(2, true));
        state.setDomain(selection(state, 1), Domain::makeFixed(1, true));
        state.setDomain(start(state, 0), Domain::makeFixed(0));
        state.setDomain(start(state, 1), Domain::makeFixed(1));
        for (const auto &[base, cached_var] : state.cached_vars)
            state.setDomain(cached_var, Domain::makeFixed(0, true));
        for (uint32_t node = 0; node < 2; ++node)
            state.setDomain(offset(state, node), Domain::makeRange(0, 3));
        return state;
    }

  public:
    static void run()
    {
        using namespace incremental_propagator_test;
        const SearchState initial_state = makePaddedState();
        SearchState witness = initial_state;
        witness.setDomain(offset(witness, 0), Domain::makeFixed(0));
        witness.setDomain(offset(witness, 1), Domain::makeFixed(3));
        FixedOffsetStartPropagator witness_memory;
        std::vector<VarId> worklist;
        require(witness_memory.propagate(witness, kInvalidVarId, worklist),
                "Padded alternative has no valid original witness");
        SearchState canonical = initial_state;
        StaticPrivateLayoutPropagator layout;
        worklist.clear();
        require(layout.propagate(canonical, kInvalidVarId, worklist), "Private layout propagation failed");
        FixedOffsetStartPropagator canonical_memory;
        worklist.clear();
        require(canonical_memory.propagate(canonical, kInvalidVarId, worklist),
                "Private layout destroyed feasible padded-alternative witness");
        std::cout << "Private layout padded-alternative regression passed" << std::endl;
    }
};
