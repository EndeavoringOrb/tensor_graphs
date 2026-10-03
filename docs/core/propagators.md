
propagator rules
CORRECTNESS
1. if VarType::SELECTED, update reachable, fix unreachable to {0}
2. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() > 0, for each eclass that is definitely selected and fixed to enode e, all children of enode e cannot be 0.
3. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() == 0, fix start to {0} and offset to {min_p} because they don't matter.
4. if VarType::CACHED && dom.isFixed() && dom.fixedValue() == 0, remove CACHE/SCATTER (and FUSED with CACHE/SCATTER in refFactory graph) from corresponding selected domain in all buckets
5. if VarType::START, and corresponding selected (dom.isFixed() && dom.fixedValue() > 0), all consumers (including through views) starts setMin(changed start + 1)
6. if (VarType::START and dom.isFixed()) and corresponding selected (dom.isFixed() && dom.fixedValue() > 0), remove start from domain of all other start domains in the same bucket where corresponding selected is not fixed to 0.
7. WRITE AFTER READ if (VarType::OFFSET and dom.isFixed() and corresponding start is fixed || VarType::START and dom.isFixed() and corresponding offset is fixed) and corresponding selected is not fixed to 0, and overlaps with another fixed start+offset+size+selected_not_{0} in the same bucket and mem_space. Sort by start to get A, B. let R(A) be all readers of a direct or through view(s), use max(reader_start+1) as end.
-if both A and B are views, return true. ignore
-if B is view of A and (B.offset >= A.offset && B.offset+B.size <= A.offset+A.size), or A is a view of B and A is within B, return true. ignore
-if A is INPUT/CACHE/ROOT, or B is INPUT/CACHE/ROOT, return false.
-if B is a reader (direct or through view(s)) of A, make sure A is in safe_inplace_idxs. if not, return false.
-if B is a reader, but not (B.offset >= A.offset && B.offset+B.size <= A.offset+A.size) return false.
-for every reader C = R(A)/B, B.setMin(reader.start.getMin + 1), C.setMax(start_B - 1)
-TODO: separate into a few rules
8. if (VarType::OFFSET and dom.isFixed() and corresponding start is fixed and corresponding selected is not fixed to 0), any reader views should have offset fixed to equal this offset (or plus a bit like with a slice).
9. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() > 0, pearce-kelly cycle detection given other fixed nonzero selections.
10. contrapositive of 2. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() == 0, any parent enode must be removed from eclass selection domain
11. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() > 0 && optype CACHE/SCATTER (or FUSED with root SCATTER in refFactory), fix corresponding cached var to {1}

EXTRA
12. if VarType::SELECTED, update dynamic programming bottom up critical path cp = cost + max(children cp), lower_bound = max(critical_path, lower_bound), if lower_bound > incumbent return false
13. if VarType::CACHED && dom.isFixed() && dom.fixedValue() == 1, state.cache_sums[mem_space] += size if state.cache_sum > mem_cap
14. TODO: make plan more descriptive. restrictOffset/restrictStart. basically better versions of 7, but they restrict domain of multiple others instead of only working on pairs
15. if VarType::SELECTED, update per engine workload, lower_bound = max(lower_bound, engine_workload) for engine_workload in workloads
16. if VarType::CACHED && dom.isFixed() && dom.fixedValue() == 1, full bucket must have something selected, and cannot have CACHE/SCATTER (or FUSED with root SCATTER in refFactory).
17. if VarType::CACHED && dom.isFixed() && dom.fixedValue() == 1, any unselected cache nodes that alone push cache size over mem cap must be fixed to {0}
18. if VarType::CACHED && dom.isFixed() && dom.fixedValue() == 1, fix offset to max(offset+size)+1 for all existing fixed cached enodes
19. if VarType::OFFSET, any selected views of this must have offset domain updated as well.