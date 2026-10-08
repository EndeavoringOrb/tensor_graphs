# Search performance log

Measurements use `.venv/bin/python tests/embeddinggemma-2/run.py text --config full`
on the same workspace and machine. Wall time includes model setup, planning,
search, and execution.

| Stage | Wall time | Change |
| --- | ---: | --- |
| Baseline | 297.21 s | Original search implementation. |
| Optimization 1 | 278.45 s | Cache a completed selection-variable scan until the selection-domain revision changes. This avoids rescanning fixed selection variables while starts and offsets are assigned. |
| Optimization 2 | 283.87 s | Tried a single forward pass over sorted offset obstacles. The same output was produced, but the measured run was 5.42 s slower than optimization 1, so this change was removed. |
| Optimization 3 | 278.99 s | Tried incremental maintenance of the fixed-offset allocation index. The later profile run with this change took 270.12 s end to end (254.18 s search), but this does not isolate a clear gain from run-to-run variation. |
| Optimization 4 | 272.78 s | Added no-op guards for irrelevant memory-propagator inputs. Profile time stayed at 218.35 s, so this did not help materially. |
| Optimization 5 | 256.84 s | Check offset-domain bounds before alias/read analysis when neither allocation can be forced to move. Saved 15.94 s (5.8%) versus optimization 4; write-after-read time fell from 221.89 s to 201.14 s. |
| Optimization 6 | 264.37 s | Tried another bound check for unfixed offsets. It was 7.53 s slower than optimization 5, and write-after-read rose to 208.58 s, so the extra check was removed. |
| Optimization 7 | 94.25 s | Query ordered spatial indexes for fixed offset conflicts instead of scanning every temporal neighbor. The same numerical result was produced; search time was 78.46 s and write-after-read time fell to 40.81 s. |
| Optimization 8 | 90.56 s | Removed forward pruning against unfixed offsets; later assignments still check fixed pairs. Same numerical result, 3.69 s faster than optimization 7. |
| Optimization 9 | 74.93 s | Register the variable types each propagator handles, then skip unrelated calls. Search time fell to 59.06 s; output remained unchanged. |
| Optimization 10 | 62.88 s | Query the fixed-allocation index for possible spatial conflicts even when the changed offset is still a range. Write-after-read time fell from 36.81 s to 25.15 s; output remained unchanged. |
| Optimization 11 | >90 s (stopped) | Removed write-after-read pruning for unfixed offsets. Search generated more than 109,000 nodes without a solution (versus 25,243 nodes in optimization 10), so the change was reverted. |
| Optimization 12 | 57.70 s | Use the fixed-allocation index records directly instead of repeating selection, offset, and allocation lookups for each spatial candidate. Search fell from 51.26 s to 41.65 s; ranged-offset write-after-read fell from 28.79 s to 19.25 s. Output remained unchanged. |
| Optimization 13 | 49.32 s | Cache each class’s start bounds and use them in temporal overlap checks. Ranged-offset write-after-read fell to 11.23 s; output remained unchanged. |
| Optimization 14 | 49.45 s | Pass the changed start variable directly into input pruning and hoist its repeated bound lookup. No measurable gain. |
| Optimization 15 | 39.44 s | Coalesce duplicate pending variables in the propagation worklist. Search fell to 23.50 s with the same 25,243 nodes and output. |
| Optimization 16 | 39.00 s | Avoid copying input-prune selection domains unless a candidate is removed. No meaningful gain. |
| Optimization 17 | 39.55 s | Add scheduler guards for start propagators that already skip optional selections. No meaningful gain. |
| Optimization 18 | 34.23 s | Replace per-call input occurrence hash maps and sets with reusable counters and fixed small storage. Input-prune time fell from 7.45 s to 3.25 s; output remained unchanged. |
| Optimization 19 | 34.35 s | Skip consumer-precedence calls with no consumers. This data set rarely reached that case, so there was no measurable gain. |
| Optimization 20 | 34.02 s | Dispatch directly through propagator lists indexed by variable type. Propagation overhead fell from 3.36 s to 2.81 s. |
| Optimization 21 | 32.87 s | Cache canonical child classes and variable IDs for input-prune alternatives. Search fell to 16.97 s; output remained unchanged. |
| Optimization 22 | 32.67 s | Avoid repeating propagation-state initialization checks on changed starts. No meaningful gain. |
| Optimization 23 | 32.75 s | Lazily copy consumer selection domains only when removing candidates. No meaningful gain. |
| Optimization 24 | 29.10 s | Sample per-propagator timing on one call in sixteen and scale the duration estimate. Call counts remain exact; the profile times are estimates. Output remained unchanged. |

The final run finished search in 13.19 s and the full command in 29.10 s. It
produced `text: cosine=0.9999768` and `max_abs=0.0016230`. Profiled propagator
call counts are exact; per-propagator time totals are estimated from the one in
sixteen samples.
