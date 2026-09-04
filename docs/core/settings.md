`Settings` struct holds all settings for running/benchmarking/testing.

Priority (lowest -> highest):

- mem_caps: System -> settings.json -> cmd line/runtime args
- pruning rules: test file -> settings.json -> cmd line/runtime args

`bucket_weights` is an optional JSON array in bucket insertion order. Values must
be finite and non-negative with a positive sum. The planner normalizes them and
minimizes the weighted cost across all buckets for each shared cache selection.
When omitted, every bucket has equal weight. The equivalent command-line option
is `--bucket-weights 9,1`.
