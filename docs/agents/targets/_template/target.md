# <Target title>

<One paragraph: what to improve, on which workload, and why.> The rules in `docs/agents/targets/README.md` apply.

## Objective

- **Metric:** <what is measured, its source line, and its direction>.
- **Goal:** <a threshold to reach, or "no fixed goal: keep improving">.

## Fixed evaluation contract

```powershell
.venv/Scripts/python.exe build.py --targets bindings test --log-level INFO
<evaluation command>
.venv/Scripts/python.exe -m tests.test_planner_oracle
```

- <Hardware, build options, budgets, what counts inside a budget.>
- <What must stay fixed: evaluator, reference data, tolerances, cost model, ...>
- <What is agent-editable.>

## Result lines

| Line | Meaning |
|---|---|
| `<prefix> ...` | <meaning> |

A run that <crashes / exceeds the budget / prints no result line / fails correctness> is a **failed** candidate.

## Acceptance gate

1. **Correctness:** <oracle, unit and native tests, output checks>.
2. **Primary improvement:** <how three runs are compared with the accepted implementation>.
3. **No regression:** <secondary metrics that must not get worse>.

## Target-specific rules

- <Anything beyond the shared rules.>

## Starting points

- <Known bottlenecks, earlier findings, pointers. Verify before relying on them.>
