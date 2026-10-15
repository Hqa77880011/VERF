# VERF

Implementation of **VERF: Versioned Evolving Fuzzy Rules with Delayed-Label Rollback across Manufacturing Tools**.

VERF shares fuzzy antecedent regions across tools while maintaining local class consequents. Updates enter a canary state. Returned labels compare the issued prediction with its cached parent alternative on exactly the same sample, then support commitment, local rollback or a compensating candidate.

## Install

Use Python 3.11 or later. Inside an activated environment:

```bash
python -m pip install -e .
```

## Prepare a stream

For Tennessee Eastman, use the released operating-mode/fault records and assign each physical tool-like unit a stable tool ID. For SECOM, use production order and actual tool metadata when available; do not treat a random row permutation as chronology. Electricity can be exported as a recurring-drift comparison. Respect the original providers' data terms.

Export one row per observation:

```csv
time,tool,label,delay,sensor_1,sensor_2,drift
0,tool_a,0,100,1.2,0.7,0
```

`time` is a unique increasing integer clock, `delay` is a nonnegative integer in the same units, and labels are consecutive IDs starting at zero. `drift=1` marks known change points for recovery evaluation only. Use finite features; any imputation or feature selection must be fitted on the warmup portion. At least two tools and every evaluated class must occur in warmup.

```bash
verf prepare --csv data/stream.csv --features sensor_1 sensor_2 --output data/stream.npz
verf run --data data/stream.npz --output outputs/full --warmup 1000 --variant full --seed 13
```

The first `--warmup` observations are labeled initialization data. Their mean and standard deviation normalize the complete stream. Later labels are released through an arrival-time queue after the current tick's prediction, including zero-delay labels; they never update an already issued prediction. The queue is not artificially drained after the stream ends.

## Controlled delayed-drift streams

```bash
verf generate --output data/synthetic.npz --samples 20000 --tools 4 --features 8 --drift-interval 5000 --delay 100 --delay-kind fixed --seed 13
verf run --data data/synthetic.npz --output outputs/synthetic --warmup 1000 --inject-every 2000
```

The generator creates four phases of a rotating linear decision boundary, including recurrence to the initial boundary, tool offsets and a configurable rare-class threshold. It is an explicit controlled stream generator. `lognormal` and `fault_dependent` delays test variation and label-dependent return times. `--inject-every` cyclically permutes a compatible rule's class counts when an eligible branch exists; it does not assume every such intervention is harmful.

Separate held-out audit samples with phase-specific labels evaluate each proposed version's effect inside and outside its affected region. Audit labels and harm decisions are used only by the reporting layer. They are never given to VERF's learner or evidence gate. For real streams, supply `--version-truth data/version_truth.json` with entries such as `{"12": true, "18": false}` derived from an independent controlled-update audit. Without this information, rollback precision and false rollback rate are null rather than inferred from the model's own rollback decisions.

## Method and choices

Warmup K-means creates shared antecedents only for regions observed on at least two tools. Gaussian lower/upper membership widths differ by a factor of 1.25; their midpoint provides interval support. Local class counts determine consequents, and support confidence attenuates low-count rules. A tool's exponentially smoothed coverage mismatch adjusts provisional exposure. Novel regions require compatible labeled observations from two tools before adding a common antecedent; antecedents remain fixed after creation, while each tool's consequent evolves through versions.

A candidate stores the changed tool/rule, parent node and before/after counts. The parent remains available in shadow mode. Canary probability is `canary_fraction * regional_membership / (1 + recent_mismatch)`. Cached parent probabilities retain the actual historical context instead of reevaluating an old model on a changed rule base.

Attribution uses Brier-loss differences weighted by regional membership and `exp(-label_delay/recency)`. The effective support is squared weight sum divided by squared-weight sum. A weighted normal interval provides the explicit evidence convention; its nominal level is a diagnostic approximation under dependent feedback, not an anytime-valid statistical guarantee. Below the support threshold the version remains unresolved. Upper bounds below the commitment tolerance commit a version; lower bounds above the harm tolerance trigger recovery.

Independent descendants change different tool/rule keys and can be reattached. A dependent descendant uses the same key, so recovery creates a new candidate from the parent counts and delayed regional label counts. Cooldown delays a new proposal after reversal and increases the support requirement. Committed single-child chains older than the audit horizon are compacted only when no unresolved cached prediction references them. This retains unresolved branches instead of pretending compaction bounds an unlimited label-delay horizon.

| Option | Default | Meaning |
|---|---:|---|
| `--rules`, `--minimum-width` | 8, 0.5 | Initial granules and lower width bound |
| `--update-support`, `--granule-support` | 10, 20 | Labels for consequent and new-region proposals |
| `--minimum-feedback` | 30 | Minimum effective returned-label support |
| `--canary-fraction` | 0.2 | Maximum provisional exposure before regional adjustment |
| `--commit-tolerance`, `--harm-tolerance` | 0.001, 0.01 | Paired Brier-loss decision thresholds |
| `--alpha` | 0.05 | Two-sided interval level |
| `--recency`, `--audit-horizon` | 2000, 5000 | Feedback weights and retained audit interval |
| `--forgetting`, `--cooldown` | 0.95, 100 | Consequent count retention and post-reversal wait |
| `--batch-size` | 100 | Reporting/recovery batch length, not a model minibatch |

## Comparisons

```bash
verf run --data data/synthetic.npz --output outputs/efs --variant no_versions --seed 13
verf run --data data/synthetic.npz --output outputs/no_rollback --variant no_rollback --seed 13
verf run --data data/synthetic.npz --output outputs/global --variant global_rollback --seed 13
verf run --data data/synthetic.npz --output outputs/no_canary --variant no_canary --seed 13
verf run --data data/synthetic.npz --output outputs/no_compensation --variant no_compensation --seed 13
```

`no_versions` is a direct count-updating evolving-fuzzy baseline with no version graph. `no_rollback` retains version tracking and canary commitment but cannot reverse harmful updates. Other variants replace local recovery with global rollback, expose every candidate immediately, or leave dependent harmful branches unresolved instead of creating compensation. The immediate-label reference is a separate direct-update learner initialized from the same warmup; it is used only to calculate delayed regret.

## Results

`metrics.json` reports prequential Macro-F1, rare-fault recall, cumulative delayed Brier regret, recovery batches, rollback precision and false rollback rate when externally auditable, rule count, active version count, mean inference latency, retained-state size and unresolved labels. Regret is signed and can be negative because the fixed immediate-label comparator is a reference algorithm, not a hindsight optimum. Recovery requires three consecutive reporting batches to reach the five-batch pre-drift accuracy minus `--recovery-tolerance`.

`history.json` tracks performance and state size over time. Memory is the serialized learner-plus-pending-queue size, measured at reporting ticks; it is distinct from process RSS. `versions.json` records proposals, commits, rollback and compensation lineage. `predictions.json` preserves issued predictions and contributing version IDs. `probabilities.npz` retains class distributions, reference predictions, labels and per-sample regret. No already-issued prediction is replaced after feedback.

```bash
verf plot --runs outputs/synthetic outputs/efs outputs/no_rollback --output outputs/figures
```

Repeat stream orderings and delay draws using separate seeds. Keep the feature set, warmup, ordering and delay realization identical within each method comparison.
