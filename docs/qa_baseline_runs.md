# Seed-0 QA arm comparison

This run reuses the completed original-QA baseline checkpoint and trains exactly
three new arms. The baseline trainer and metric source are restored byte-for-byte
from its archived source snapshot into a separate remote runtime. The QA exporter
does not replace the trainer or its prompt.

| Arm | Training rows | Original rows | Added signals |
|---|---:|---:|---:|
| Original QA (existing baseline) | 82,829 | 82,829 | 0 |
| Original + paraphrase | 165,658 | 82,829 | 0 |
| Original + grounded | 152,118 | 82,829 | 0 |
| Original + grounded + acquisition | 209,722 | 82,829 | 4,820 |

Every export preserves the baseline signal prefix and copies selection,
validation and test parquet files byte-for-byte. Grounded adds 69,289 compatible
questions; acquisition adds 57,604. The final arm excludes 3,132 two-ECG questions
and 11,264 short-crop questions because the unchanged trainer requires one
12-lead, 1,000-sample ECG. Numeric questions retain their empty options line.
Question-only arms retain the baseline signals and records checksums; their
training table and manifest fingerprints change.

The three manifests are:

- Paraphrase: `6757fff9e668a2d2e501286c9b5171e38ece3745382664b826c4833faad62445`
- Grounded: `0bbf9e36a278b037d18403954df30b88293a6b7a18cd98ce49a24b0d77c0d4e8`
- Grounded + acquisition: `6e48bb03e80bc89e7df70dc7b3524fc6e53bc1d7fedbbd941cb2cef882387f43`

## Execution

Export with `python -m syntide.generate.qa export --corpus ...
--baseline-prepared ... --output ...`. Baseline original training rows are
included automatically. The output is immutable and checksummed; the exporter
checks source identity and training-patient lineage. See `qa_generation.md` for
the complete command examples.

The runtime uses the baseline's exact container image, package pins and OpenTSLM
commit. `scripts/runpod/qa_preflight.py` checks source hashes, package versions,
initial weights and parameter counts, then tokenizes every distinct training row
through the original collator. Any truncated rationale rejects the preflight.
It opens training and selection only.

`scripts/runpod/run_qa_arms.py` runs the new arms serially with seed 0 and derives
all training arguments from the original baseline config. In particular,
`--selection-method option-nll --eval-batch-size 256` must be explicit; the
trainer's defaults would change the experiment. The remaining settings include
15,531 updates, batch 8, accumulation 2, learning rate 0.0002, maximum length
1,536, generation cap 384, and selection every 1,000 updates.

All four results must pass configuration, completion, first-selection-argmax,
checkpoint-checksum and zero-truncation audits before global checkpoint freeze.
If any best step is at or beyond 80% of its budget, the status is
`needs_budget_review` and test stays closed. No automatic budget expansion or
new seeds are permitted. Interrupted training is preserved for explicit recovery.

After freeze, the unchanged evaluator generates full validation and test outputs
and their fixed-text zero-waveform controls, once per question. Complete published
prediction files can be rescored without another generation pass.
The baseline's original checkpoint is evaluated alongside the new
ones; it is not retrained. An interrupted test claim without complete published
predictions is preserved and requires review; the runner refuses to regenerate
that split. Majority answers are fitted once on original training
rows and scored on the fixed selection subset. Each arm gets `summary.json`,
`seed0/config.json`, `seed0/train_result.json`, and a field-level configuration
comparison. Across-seed standard deviation is null because there is one seed.

## RunPod and collection

The initial account check found no pods and zero hourly spend. One secure A100
80 GB was provisioned at $1.79/GPU-hour, with $1.809/hour observed total including
storage. A separate training/selection/generation smoke measures its economics
before production. No smoke checkpoint counts as an arm.

The measured projection is 24.93 hours and $45.10 for production, approximately
$45.76 including observed setup spending. It includes three trainings with 16
selection passes each and 432,840 generated responses across the four models,
validation/test and real/zero ECG conditions. These are estimates from a short
smoke run; trained-model generation and the larger arms may differ. Detailed
assumptions and timing sensitivity are in `outputs/qa-runs/cost-plan.json`.

`scripts/runpod/qa_pod.py` reads the credential directly from macOS Keychain and
provides only the existing SSH public key to the pod. Creation records its
identity, price ceiling and requested API `stopAfter` timestamp, initially 24 hours.
Enforcement of this creation field is unverified: the current API does not expose
the saved scheduled-stop state, and `PodEditJobInput` rejects `stopAfter` updates.
The confirmed deadline guard is the running local supervisor. RunPod's
[management documentation](https://docs.runpod.io/pods/manage-pods) describes
scheduled stopping through an external timer and stop command.

`scripts/runpod/watch_qa_study.py --execute` watches only the created study pod.
It copies terminal results, compares every local file to two stable remote
SHA-256 inventories, and terminates the pod only after verification. If its
deadline or spending guard is reached before safe collection, it stops the pod
and preserves the volume. Bootstrap logs and the environment freeze are included
in collection. Exact account cost and termination status are recorded separately
from estimated per-arm compute.

Without an override, the supervisor uses the original receipt's `stopAfter` as
its local deadline. `--deadline-utc` accepts only an ISO timestamp with an explicit
UTC zone (`Z` or `+00:00`). It changes the local deadline while preserving the
creation receipt and making no API scheduling change. State records
`effective_deadline_utc`, `deadline_source`, the original requested API timestamp,
and `max_spend_usd`. Restart the monitor to apply an override. For the reviewed
28-hour deadline measured from this pod's API creation timestamp:

```sh
python3 -u scripts/runpod/watch_qa_study.py --execute \
  --deadline-utc 2026-10-24T01:42:38.340Z --max-spend-usd 50
```

The $50 cap still applies if that deadline is extended. The monitor requires the
local machine and its network connection to remain available; the original API
request is not a verified independent fallback.
The override also does not cancel any possible API stop at the originally
requested `2026-10-23T21:42:37Z`. The follow-up monitor must compare observed
evaluation throughput with that earlier timestamp and preserve test-once
integrity if an interruption appears likely.

Local receipts, verification records, logs and downloaded results are under
`outputs/qa-runs/`; prepared arm folders are under `runs/ecgqa-data-*`. No credentials
are stored in these artifacts. Arm 5 and extra seeds are outside this run.

## Separate DiffuSETS extension authorized on 2026-10-23

The user subsequently authorized a separate parallel `grounded_acq_gen` arm,
seed 0, with exactly 5,000 DiffuSETS candidates and no replacement of rejected
candidates. It does not alter the original four-arm protocol or checkpoint
freeze. Its receipts and frozen inputs are in
`outputs/qa-runs/diffusets-extension/`.

The frozen allocation uses only TRAIN records and the original selection
results: 1,000 slow-rate, 3,000 middle-rate and 1,000 fast-rate slots, with
80% regular/20% irregular parents and balanced parent noise strata within
each rate group. Parent noise is provenance, not a DiffuSETS control. The
3,874 parents retain their original reports, age and sex; requested HR is
measured from the parent. Published author embeddings are matched to those
exact records, with the unknown original API input recorded explicitly.
No new embedding API calls are made. This is a selection-tuned exploratory
extension, not an additional arm of the original confirmatory freeze.

Every candidate is measured independently. Acceptance requires usable HR,
beat-count and RR-regularity evidence at confidence >=0.8 and amplitude
within 0.2–10 mV. Accepted ECGs provide three or four rhythm questions;
near-boundary rate-band questions can be omitted. Parent diagnoses never
become generated answers. The prepared mix appends these rows to the
original + grounded + acquisition rows and retains byte-identical evaluation
files. All baseline training settings and recorded packages remain fixed.

The A40 pod is `0il9463gvr3mmv`; the original A100 pod remains separate.
`run_diffusets_pipeline.py` runs generation, preparation, complete token
preflight, seed-0 training for 15,531 updates, and guarded evaluation.
`run_diffusets_arm.py` requires matching initialization and source hashes,
zero truncation, and the same convergence rule before opening test.
An interrupted claimed test is never regenerated automatically.

Generation evaluation uses batch 256 on the A40, versus 512 on the original
A100. Training batch 8, accumulation 2 and option-NLL selection batch 256
are unchanged, as are greedy decoding and the 384-token output limit.
A selection-only benchmark generated 256 questions in 32.42 seconds and
used a peak observed 27.70 GiB, with no held-out data access.

The extension has a $40 spending cap and a pod-local stop guard at
2026-10-25T11:12:47Z, fifteen minutes before its absolute 48-hour deadline.
The guard uses the platform-injected credential scoped to its own pod;
the local account key is never copied. The separate `diffusets_watch.py`
supervisor verifies two stable remote inventories and every collected file
before termination. Unsafe collection stops the pod and preserves its
volume. Original account-balance monitoring remains at $50 and includes
overlapping spend; extension attribution uses its own bounded hourly rate.
The heartbeat follows both jobs until collection and resource release.

## Deadline extension and laptop sleep protection, 2026-10-23

The user authorized a longer DiffuSETS deadline while retaining the $40 cap and
all baseline training settings. The effective absolute deadline is now
`2026-10-25T23:27:47Z` (Monday 26 October, 00:27 Zurich). Its pod-side stop and
work deadline is fifteen minutes earlier, `2026-10-25T23:12:47Z`. Sixty hours
at the conservative $0.62/hour bounds running cost at $37.20; a separate $2.50
storage reserve keeps the planned bound at $39.70. Stored results remain
billable after a preserving stop, so the reserve is finite and collection must
not be postponed indefinitely.

The creation receipt, original scientific pipeline configuration/protocol,
generator source, candidate allocation, training settings and test-once rules
remain unchanged. A hash-pinned operational override and reviewed orchestration
copies live under
`outputs/qa-runs/diffusets-extension/operations/deadline-extension-20261023/`.
The override SHA256 is
`26c51bd82114d04402c03f8d32d0abe396a580e2bf02aca778debf179b97dc64`.
Read `adoption-install.json` and `deployment-verification.json` before recovery.

The old pipeline parent was replaced without interrupting its separately
sessioned generator. Generator PID 3768 and its start time, frozen identity and
inherited execution locks were verified unchanged. The detached adopter waits
for the exact 5,000 completed bundles, verifies all published hashes, and records
the orchestration transition before preparation, preflight and training. It
does not launch another generation process. The original immutable files remain
available to audit the operational change.

The extension watcher now needs `--operational-override` pointing to the local
override and `--operational-override-sha256` with the pin above, in addition to
`--execute`. The current command and PID are in `supervisor-pid.json`; never
launch a duplicate. The pod-side guard uses its injected credential scoped to
that pod; no account credential was transferred.

Both studies now have pod-side stop-preserving guards that work while the Mac
sleeps. The original comparison's guard keeps its existing local deadline and
$50 account-delta bound, conservatively including concurrent spending; its
receipt is `outputs/qa-runs/pod-backstop-install.json`. Neither original trainer
nor original protocol was modified. Its local watcher additionally uses
`--collection-lease-path /workspace/qa-backstop/collection-lease.json`.

On terminal completion/failure, each pod-side guard requires released execution
locks and durable terminal evidence, then gives the collector fifteen minutes.
A collector may obtain a bounded thirty-minute lease. Leases never override a
deadline or budget stop. If the Mac is asleep, the pod stops GPU billing while
preserving its volume; local collection and verified termination follow when
access returns. No guard terminates uncollected results. The operational change
passed 167 tests and a Linux handoff smoke on the same persistent filesystem.

## Original-volume recovery pending, 2026-10-23 22:02 UTC

The original pod `y9z0q4fgr4zvn6` is `EXITED` with its volume retained. The local
supervisor exited after recording this status at 21:50 UTC. Its last remote
workload status predates a connectivity gap and cannot establish either the
stop reason or whether all evaluations finished. These must be read from the
retained pod-side receipts and study files when access returns.

Recovery evidence is in `outputs/qa-runs/recovery-20261023T2200/status.json`.
RunPod refused a GPU resume for insufficient GPU capacity, a zero-GPU request
for the same reason, and an explicit `gpuCount: 0, computeType: CPU` resume for
insufficient host memory. All three requests left the pod stopped. No new pod,
training process or evaluation pass was started. Observed total account delta
was $44.4043 at 22:02 UTC, leaving $5.5957 under the original $50 guard. The
account was billing $0.637/hour for the running extension and retained storage.

On the next heartbeat, inspect this recovery status and current API state,
then retry CPU-only access once if the collection budget still fits. If access
returns, preserve the previous backstop state, inspect terminal study evidence,
arm a collection-only stop guard bounded to 45 minutes and the existing
absolute deadline/$50 cap, and restore one local collector. Collect and verify
all files before termination. Do not restart training or any claimed test. If
capacity remains unavailable as the deadline or budget margin approaches,
report the concrete remaining decision; never delete the uncollected volume.

DiffuSETS remains healthy and independent: 1,721 completed candidates, 1,530
accepted at 22:02 UTC, with its original generator identity and execution locks
verified. The measured remaining-generation estimate was 18.7 hours, leaving
30.5 hours before the extended work deadline for preparation, training and
evaluation. Keep monitoring both studies until their artifacts are collected
and resources released.

## Original comparison collected and released, 2026-10-23 22:43 UTC

All 16 original evaluation passes completed at 18:23:48 UTC, before the
pod-local completion guard stopped the pod at 18:39 UTC. The user restored
CPU-only access. The collector downloaded 143 files (2,956,003,352 bytes),
verified every SHA256 against two identical remote inventories, and terminated
only original pod `y9z0q4fgr4zvn6`. A separate API check confirms it is absent.
No training or evaluation was restarted. The original watcher has finished;
do not restart it or recreate the original pod.

The local completion audit verifies all four configurations, zero truncation,
convergence guards, frozen checkpoint identities, all 16 prediction/claim
hashes and row counts, matching evaluation fingerprints and complete summaries.
The report is `outputs/qa-runs/comparison-report.md`; evidence is in
`completion-audit.json`, `collected-remote-manifest.json`,
`completion-account-check.json` and `supervisor.json`. Full artifacts are in
`outputs/qa-runs/study/`. The earlier recovery status is archived under
`recovery-20261023T2200/`.

Acquisition improves option-NLL selection accuracy but reduces generated-answer
test accuracy. Noisy test accuracy is 39.53% baseline versus 34.51% acquisition;
noisy template macro-F1 is 23.77% versus 26.21%. Single-seed SD remains null.
The selected grounded checkpoint is step 4,000; end-of-training exposure counts
must not be mistaken for exposure at the selected checkpoint.

Observed account decrease since original launch is $44.8263 at release,
including overlapping DiffuSETS, retained storage and CPU recovery. This stays
under the original $50 guard; it is not a per-arm or per-pod bill. The released
original pod no longer bills. The account now bills $0.609/hour for the separate
running DiffuSETS pod. Its $40 cap and extended deadline are unchanged.
The heartbeat now monitors only that remaining extension. Do not use these
held-out results to alter its frozen candidate allocation or acceptance rules.
