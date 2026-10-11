`Settings` struct holds all settings for running/benchmarking/testing.

Priority (lowest -> highest):

- mem_caps: System -> settings.json -> cmd line/runtime args
- pruning rules: test file -> settings.json -> cmd line/runtime args

`bucket_weights` is an optional JSON array in bucket insertion order. Values must
be finite and non-negative with a positive sum for the weighted buckets. The
automatically added full bucket always has weight zero because native extraction
provides its valid plan. The planner normalizes the remaining weights and
minimizes their weighted cost for each shared cache selection. When omitted,
every weighted bucket has equal weight. The equivalent command-line option is
`--bucket-weights 9,1,0` for two user buckets followed by the automatic full
bucket.
