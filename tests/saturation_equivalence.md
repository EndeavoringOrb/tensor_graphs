# Comparing saturated egraphs

`tensor_graphs_cpp/tests/saturation_equivalence.cpp` runs masked attention through
full saturation and two seeded partial buckets. It dumps expressions, equivalence
classes, layouts, kernels, engines, constants, logical anchors, clean flags, and
persistent buffer identities, and union weights. Duplicate copies of an identical enode do not
change the represented expressions and are collapsed in the dump.

Build the fixture against baseline and candidate headers separately. Use the
same compiler flags, generated kernel registry, CUDA object, hardware, and cost
records for both runs. This fixture disables benchmark records explicitly.
For example, from the candidate checkout, with the baseline header tree in
`/tmp/tg-original`:

```sh
clang++ -O0 -std=c++17 -march=native -DTG_USE_CUDA \
  -I/tmp/tg-original -Itensor_graphs_cpp \
  tensor_graphs_cpp/tests/saturation_equivalence.cpp \
  tensor_graphs_cpp/generated/cuda_kernels.o \
  -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lcudart -lcublas -o /tmp/egraph-baseline
clang++ -O0 -std=c++17 -march=native -DTG_USE_CUDA -DTG_TEST_COMPACT \
  -Itensor_graphs_cpp tensor_graphs_cpp/tests/saturation_equivalence.cpp \
  tensor_graphs_cpp/generated/cuda_kernels.o \
  -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lcudart -lcublas -o /tmp/egraph-candidate
/tmp/egraph-baseline /tmp/baseline seeded
/tmp/egraph-candidate /tmp/candidate seeded
.venv/bin/python tests/compare_saturation_egraphs.py /tmp/baseline /tmp/candidate
```

The baseline overlay must contain its original `core/egraph.hpp`,
`core/rewrite.hpp`, and `core/plan/planner.hpp`. Remove the CUDA flags and libraries
for a CPU-only comparison. The CUDA and CPU graphs differ, so each needs its own
baseline run.

The comparator joins all buckets into one directed graph. Ordered inputs and
class membership are labeled edges; buffer identities are shared vertices across
buckets, including identities embedded in scatter tags. This allows IDs to change
while preserving the relationship to the full graph's persistent buffers.
An exact check first tries an explicit mapping by class creation order. If that
mapping does not verify every expression and edge, color refinement prunes a
general exact graph-isomorphism search. Different creation orders can still pass.

Matching candidate graphs plus matching extraction costs and constraints preserve
the feasible plan choices and optimum objective. They do not require solving the
optimization problem. Equal-cost plans may still be selected in a different tie
order. These small fixtures are regression evidence; they do not establish
preservation for every possible model or bucket.
