# Publication readiness status

Status date: 2026-08-21. This is an evidence ledger, not a claim that the
release or manuscript is complete. Every number below is either quoted from a
retained JSON record under `benchmarks/results/` or marked as not yet measured.

**Hardware context, stated once so no claim overreaches it:** every
measurement in this repository was produced on one machine — a 32-core AMD
Ryzen 9 9950X with 132 GB RAM and an NVIDIA RTX PRO 4500 (recorded in each
record's `environment` block). "Laptop-scale" therefore always means a
*laptop-scale memory target enforced by the planner on that workstation*, not
a measurement on laptop hardware. No claim of validated laptop or cluster
execution is made anywhere in this repository; where design documents mention
cluster portability they now label it a design goal.

## Engineering outcome

The release-candidate implementation supports the intended north-star path:

```text
versioned remote DANDI NWB-HDF5
        → bounded NumPy temporal projection
        → actual plane-wise Cellpose (CPU or CUDA)
        → source-chunk-aware whole-movie mean traces
        → verified, resumable (time, cell) Zarr output
```

## What changed in this validation pass

### Memory-budget semantics (Priority 1)

The development fish run was requested with `memory_limit="2 GiB"` and peaked at
6,097,367,040 bytes of process RSS. The discrepancy was investigated rather than
renamed.

- Added `benchmarks/memory_attribution.py`, which attributes resident set to
  named components. Each component runs in a *fresh interpreter* so its cost is
  isolated, and RSS is sampled before the probe releases anything.
- `memory_limit` now means an **approximate total process-memory target** and
  decomposes exactly as `total = process_overhead + task_working_set`. Process
  overhead is charged **once**, not per task.
- `neuroflow/array.py::compute()` previously compared its estimate against the
  raw parsed limit, so the same keyword meant "total process target" on
  `persist()` and "array data allowance" on `compute()`. It now uses the same
  interpretation as `persist()`.
- Third-party residency the planner does not itself allocate — a loaded
  Cellpose/PyTorch network — is declared as an external reserve and charged
  **per worker**, because adapters cache one model per worker thread.
- `max_workers` is now treated as an **availability ceiling, not a demand.**
  Concurrency is clamped to what the target affords, and the granted count is
  recorded as `execution_policy.max_workers`. It previously raised
  `max_workers=N exceeds the memory-safe limit of M`, which forced the caller to
  hand-tune the very number the planner had already computed.
- `cellpose()` gained its own default target (`DEFAULT_SEGMENT_MEMORY_LIMIT`,
  4 GiB). Under the corrected accounting the inherited 2 GiB persist default was
  consumed entirely by `cpsam` weights, so **the default segmentation call
  refused itself before reading a pixel.** A single fish plane costs only
  ~175 MiB to segment; the extra headroom is model residency, not partition size.
- `docs/High_Level_API.md` said "per-task memory limit". It now documents the
  total-process interpretation, the absence of an OS-level cap, and the
  per-worker model reserve. `README.md` examples that requested 2 GiB for
  `cpsam` segmentation were corrected to 4 GiB, because they would now raise.

No OS-kill mechanism was added. The budget is a planning target; overrun is
reported as a number rather than enforced by killing the process.

### GPU-aware Cellpose (Priority 2)

`--cellpose-device {auto,cpu,cuda}` selects the device for the fish benchmark.
`auto` uses CUDA when available and CPU otherwise, `cuda` fails loudly rather
than degrading silently, and `cpu` is honoured even when a GPU is present. The
device is resolved **once, before any download or segmentation**, and the same
resolved object drives both the NeuroFlow-mediated run and the direct-Cellpose
equivalence run. Selected device, GPU model, VRAM, CUDA version, PyTorch
version, and Cellpose wall time are recorded.

Note on provenance strength: the recorded `same_device_for_direct_comparison`
flag compares the resolved device object with itself, so identical devices are
*structurally guaranteed* by construction rather than independently verified.
The guarantee is real; the flag is not an additional check.

GPU VRAM is reported separately and is never counted against the host
`memory_limit`.

### Referee-response pass (submission hardening)

- **Baseline parity (fairness).** The fish pipeline previously mixed two
  transport configurations across its own stages (the projection stage used
  the example default of 256 KiB blocks while trace extraction used the 1 MiB
  library default) and recorded neither; the manual baseline defaulted to
  256 KiB blocks. Transfer comparisons across differing block sizes measure
  configuration, not tools. Both benchmarks now take explicit
  `--block-size`/`--cache-size-mib` with identical defaults (1 MiB / 64 MiB),
  apply them to every stage, and record them numerically
  (`transport_configuration` / `configuration_parity`). The baseline's
  `--time-chunk` help now instructs matching the NeuroFlow record's
  `trace_plan.time_window`.
- **Scientific regression tests added:** persisted results are bitwise
  identical across `max_workers=1` and `4` (staged-mean partial combination is
  where concurrency could diverge); NaN propagation matches NumPy exactly with
  no silent skipping; the axis-label convention is pinned, including the
  documented `(time, channel) -> ("time", "y")` case; and cross-array
  operations are refused with the exact guidance message while same-selection
  operands and scalars keep working.
- **Expression contract documented:** expressions cover one source selection
  plus scalars (plus the staged global-scalar exception); per-pixel dF/F
  against a persisted baseline is refused explicitly, and the supported
  pattern — extract compact results, normalize downstream in NumPy — is stated
  in the README and `docs/High_Level_API.md`, alongside a NaN-policy section
  and the axis-label convention.
- **Hardware wording pass:** unsupported "laptop"/"cluster" capability claims
  were replaced with "laptop-scale memory target" / "resource-constrained
  commodity hardware", and the design documents now label laptop-to-cluster
  portability a design goal. See the hardware-context statement at the top of
  this ledger.
- **Repetitions:** `benchmark_resource_scaling.py` and
  `benchmark_dandi_smoke.py` accept `--repetitions N` (fresh process and fresh
  output root per repeat, so nothing resumes and each peak RSS is a true
  high-water mark) and report median and min-max range; the smoke aggregate
  additionally asserts that every repetition produced the identical output
  checksum.

## Measured memory attribution

From `benchmarks/results/current-memory-attribution.json`, on the fish geometry
(888x2048 planes, 29 planes, 5,557 cells, 106-frame window). `rss_delta_bytes`
is the increase caused by that component alone.

| Component | Δ RSS (MiB) | Notes |
| --- | --- | --- |
| interpreter + numpy baseline | 0.0 (31.1 MiB peak) | floor of any CPython process |
| `neuroflow` import chain | 164.9 | pulls dask, zarr, h5py, fsspec, pynwb |
| dask runtime | 111.8 | lazy graph for 1,115 stacked keys, never computed |
| remfile cache | 35.8 | 64 MiB configured ceiling; one native chunk read |
| source partition array | 367.7 | one int16 (106, 888, 2048) window |
| **temporary NumPy arrays** | **1,103.1** | int16 window + float32 cast + contiguous reshape copy |
| ROI index / lookup state | 2.2 | 5,558 distinct labels, 29 ROI chunks |
| trace accumulators | 6.9 | float64 sums + float32 output block |
| output buffers | 12.0 | zarr group, one written window, one verification read |
| torch import | 453.9 | before any model is constructed |
| **Cellpose `cpsam` on CPU** | **1,874.2** | 3,212.6 MiB process peak; 1,218.5 MiB parameters |
| Cellpose `cpsam` on CUDA | 818.8 host | 2,089.3 MiB process peak; 1,218.5 MiB moved to VRAM |

Where the 6.1 GB came from: the planner's 2,129,931,384-byte per-task estimate
was accurate **for the trace task in isolation**, but the benchmark runs
projection, Cellpose, and traces in a *single process*. The trace working set
therefore lands on top of a Cellpose-warm floor of roughly 3.2 GB, and torch
plus the model stay resident after segmentation finishes. The un-modelled term
was third-party model residency, which is exactly what the external reserve now
charges. The declared overhead envelope is confirmed to be conservative against
these measurements by a regression test.

Declared reserves are honest envelopes over measurement, not round numbers:
`cpsam` CPU declares 2,048 MiB against 1,874 MiB measured; CUDA declares
1,024 MiB against 819 MiB measured.

### Consequence for a laptop-scale target

`memory_limit="2 GiB"` with `cpsam` on **CPU is now refused with guidance**,
because one loaded network alone measures ~1.9 GiB resident and the target
cannot be met by any process that has loaded it. The same target is accepted on
**CUDA**, where ~1.2 GiB of weights live in VRAM instead of host RAM. This is a
deliberate, reported refusal rather than a silent overrun.

CUDA is also dramatically faster for this model: one 888x2048 `cpsam` evaluation
took **66.14 s on CPU versus 2.03 s on CUDA (32.5x)**, at a measured VRAM peak
of 2,452,524,032 bytes.

## Resource scaling (Priority 5)

Local synthetic movie, plane geometry matching the fish asset
(888x2048 int16, one plane per source chunk). A local fixture is used
deliberately: re-reading a 323 GB archive per scaling point would cost hours and
add network variance to a measurement about memory. Each configuration runs in a
fresh subprocess, because peak RSS is a high-water mark that cannot be reset
in-process.

### 192 frames — `benchmarks/results/current-resource-scaling-fits-2gib.json`

All five requested configurations complete. Three independent repetitions per
configuration, each in a fresh process with a fresh output root; measured peak
is reported as median [min, max].

| Target / workers | **Measured peak, median [range]** | vs target | Wall (median) |
| --- | --- | --- | --- |
| 2 GiB / 1 | **2,061 MiB [2,061, 2,062]** | +0.6% | 7.2 s |
| 4 GiB / 1 | **2,908 MiB [2,908, 2,908]** | −29% | 6.2 s |
| 4 GiB / 2 | **2,293 MiB [2,293, 2,295]** | −44% | 6.2 s |
| 8 GiB / 2 | **2,294 MiB [2,293, 2,295]** | −72% | 6.2 s |
| 8 GiB / 4 | **2,294 MiB [2,293, 2,294]** | −72% | 6.2 s |

The headline result for Priority 1: a 2 GiB request produced a **2,061 MiB
median process peak with a 1 MiB run-to-run spread over three repetitions,
0.6% above the stated target** — against 6,097 MiB for the same nominal
request before this pass. The ≤2 MiB spreads across every configuration show
the peaks are properties of the plan, not noise.

### 384 frames — `benchmarks/results/current-resource-scaling.json`

Three repetitions per configuration; measured peak as median [min, max]. The
2 GiB refusal reproduced identically in all three repetitions.

| Target / workers | **Measured peak, median [range]** | vs target |
| --- | --- | --- |
| 2 GiB / 1 | **refused (3/3)** — one task needs 2,822,504,448 B > 1,677,721,600 B available | — |
| 4 GiB / 1 | **4,690 MiB [4,690, 4,691]** | **+14.5%** |
| 4 GiB / 2 | **4,690 MiB [4,690, 4,692]** | **+14.5%** |
| 8 GiB / 2 | **5,453 MiB [5,452, 5,453]** | −33% |
| 8 GiB / 4 | **5,453 MiB [5,453, 5,453]** | −33% |

Two findings must be stated plainly rather than smoothed over:

1. **The target is approximate and can be overrun.** At 4 GiB with a
   budget-bound 206-frame window the process peaked 14.5% *above* the
   requested total, reproducibly (≤2 MiB spread over three runs). The overrun
   grows with window size, so the planner under-models large transient
   allocations — plausibly the internal copy an exact median requires. A 2 GiB
   request overran by only 0.6%. The target is therefore honest at
   laptop-scale targets and optimistic at large windows.
2. **A 384-frame exact median cannot fit 2 GiB at this plane geometry**, and is
   refused rather than attempted. With one native chunk per 888x2048 plane there
   is no smaller spatial tile available, so the reduction axis cannot be
   subdivided further. This is an algorithmic constraint of exact median on this
   chunking, not a planner defect.

Concurrency behaves as intended: with 64 workers declared available and only the
target varied, granted concurrency rose 1 → 2 → 5 → 10 → 21 across 8 MiB →
128 MiB, and no request was refused. Users state resources; the planner chooses.

## Retained remfile evidence (Priority 6)

`benchmarks/results/current-fish-soma-traces-remfile.json` is preserved and
classified `current`, not `publication`, because it was produced from a dirty
tree. It records `DANDI:000350@0.240822.1759`, the complete 323,296,788,480-byte
logical movie, 230,300,224,863 observed response bytes, 29 projection and 29
Cellpose tasks, 5,557 plane-local ROIs, a `(3065, 5557)` trace array,
6,097,367,040 bytes peak RSS, 19,286.26 s wall time, verified output integrity,
a zero-read resume (29 resumed, 0 computed tasks), and exact agreement with a
direct plane-wise NumPy reference (maximum absolute and relative error 0.0).

Direct Cellpose equivalence on that run was exact: 0 mismatched voxels across
all 29 planes, 5,557 objects both ways, on CPU (`gpu: false`) in 1,917.80 s.

The three Zarr stores plus that JSON are additionally copied to
`publication/runs/_retained-remfile-dev-20260821/`, because
`benchmark_fish_pipeline.py` writes `fish-projection.zarr`,
`fish-cellpose.zarr`, and `fish-traces.zarr` under `--output-root` by default
and any rerun would otherwise land on the retained development evidence. Large
Zarr stores remain gitignored; the compact JSON records are tracked.

## LINDI backend (Priority 3)

Code path: `open_dandi(backend="lindi")` → `DandiNWBSource` → `NWBHDF5Source` →
`_open_remote_file()` → `lindi.LindiH5pyFile.from_hdf5_file()`. Trace execution,
adapters, storage, validation, and resume contain no transport-specific
branches, and `neuroflow/workflow.py` restores the backend on resume. A real
local LINDI 0.4.6 → PyNWB → lazy-slice test passes.

**Remote transport equivalence measured** —
`benchmarks/results/current-lindi-equivalence.json`
(`benchmarks/benchmark_lindi_equivalence.py`). The same 8-frame, 2-plane slice
of the real fish asset (z=14..15 of `DANDI:000350@0.240822.1759`) ran the same
median projection once per transport, each in its own cold subprocess. **No
source changes were required: LINDI worked through the existing backend
abstraction as written.**

- Outputs are **byte-identical**: equal SHA-256 checksums, elementwise equal,
  maximum absolute and relative error 0.0, compared after reopening both stores
  with partition-checksum verification rather than from in-process arrays.
- Both transports produced the **same `workflow_id`**
  (`594f8284…c17a77c0`), the same 2-task partitioning, the same output chunking,
  and verified integrity — workflow identity is transport-independent by
  construction.
- remfile: 5.17 s wall (2.16 s open), 389 MiB peak RSS, 38,702,431 response
  bytes over 37 HTTP responses.
- LINDI: 18.94 s wall (**14.00 s of that opening the file**), 1,469 MiB peak
  RSS, transferred bytes **unknown** — `LindiH5pyFile` exposes no byte counter,
  and the record stores null rather than a false zero.
- These numbers are per-transport overhead observations on a few native chunks,
  not a throughput comparison; LINDI's higher open cost and resident set come
  from building its reference index over a 323 GB remote HDF5 file.

What this does **not** show: the full fish workflow (Cellpose + whole-movie
traces) has not been run through LINDI. Transport independence is demonstrated
for the projection stage on the real remote asset, not yet end to end.

## Second-dataset generality smoke (DANDI:000223)

`benchmarks/benchmark_dandi_smoke.py`
(`benchmarks/results/current-dandi-smoke-000223.json`) runs the ordinary public
workflow — discover with `NWBQuery(neurodata_type="TwoPhotonSeries")`, inspect
inferred axes and physical chunks, preflight a plan, persist a bounded temporal
mean under a 2 GiB target, verify, compare against an independent plain
h5py + NumPy reference — on a dataset the repository had never touched. Every
identifier arrives on the command line; the harness contains no
dataset-specific code and no fish constants.

Dataset: `DANDI:000223@0.260528.0906`, asset
`sub-3112/sub-3112_ecephys+ophys.nwb` (one of 20; paired spine calcium
imaging). The selected object is `/acquisition/TwoPhotonSeries`, shape
`(1800, 1024, 1024)` uint16 — genuinely different from the fish in
dimensionality (3-D vs 4-D), dtype (uint16 vs int16), and above all storage
geometry: native chunks `(1, 128, 128)`, sixty-four 32 KiB tiles per frame,
versus the fish's one whole 3.6 MB plane per chunk.

What passed, on the first unfamiliar dataset:

- **Discovery**: the object was found through the public query mechanism by
  NWB type, not by an internal HDF5 path.
- **Axis inference**: `--expect-axes time,y,x` asserted, and the inference
  produced exactly `("time", "y", "x")` for the 3-D series.
- **Automatic partitioning**: the planner chose 64 partitions of
  `(96, 128, 128)` from the memory target alone; the command line set no tile,
  chunk, block, cache, or worker parameters.
- **Correctness**: the persisted 96-frame temporal mean is **bitwise
  identical** to the independent reference (elementwise equal, maximum absolute
  and relative error 0.0), and the output verified against its checksum
  manifest.
- **A real generality bug was caught and fixed**: discovery crashed with
  `AttributeError` on hdmf's HDF5 object-reference wrappers, which expose
  `shape` and `dtype` but no `ndim` — an object species the fish file never
  contained. `_array_metadata` in `neuroflow/source/hdf5.py` now skips such
  non-array containers; `tests/test_hdf5_source.py` pins it.

The first run of this smoke test exposed two planner-model limits on
fine-chunked layouts (retained as
`benchmarks/results/current-dandi-smoke-000223-before-planner-fix.json`), and
both were then fixed and re-measured. An independent rerun by the maintainer
reproduced the pre-fix overrun (2,774 MiB engine, 3,126 MiB process) before the
fixes were applied.

1. **Concurrency-aware process-memory modelling.** Before: the per-task
   estimate on this layout is 3.4 MiB, so dividing the budget by task data
   alone granted concurrency up to the core count (32 workers) and the engine
   peaked at 2,865 MiB against the 2,048 MiB target (+40%). The measured cost
   is ~73-77 MiB per worker through the remfile+h5py read path (a local Zarr
   sweep of the same geometry shows 6-8 MiB per worker, so the remote read
   path dominates). The planner now charges a measured 96 MiB envelope
   (`WORKER_RUNTIME_OVERHEAD_BYTES`) for every worker beyond the first, whose
   runtime slack already sits in the process floor. After, same command:
   **17 granted workers, 1,770 MiB engine peak, 2,084 MiB whole-process peak —
   1.8% above the 2 GiB target with the harness's own reference computation
   included — at no wall-time cost (65.5 s vs 66.6 s).** Whole-process RSS is
   the primary user-facing metric; the engine-phase figure is diagnostic.
   Large-task grants are unchanged by the new term (fish-geometry grants are
   identical), and regression tests pin the derivation.
2. **Source-chunk bytes and transport bytes are now reported separately.**
   Before: the plan's only read figure was 192 MiB of source-chunk bytes
   (amplification 1.0) while the measured HTTP transfer was 3,264 MiB (17.0x) —
   32 KiB chunks are fetched through 1 MiB transport blocks laid out
   frame-major, with 64 tasks evicting one another's blocks in the shared
   cache. The plan now also reports `estimated_transport_bytes_read`, a
   no-reuse upper figure on uncompressed data (chunk touches x whole transport
   blocks), and the source exposes its block size for the model. After: the
   plan reports **192 MiB source-chunk / 6,144 MiB transport (no-reuse)**, and
   the measured 3,156 MiB sits between the two exactly as the model's notes
   predict (cache reuse and compression pull actual transfer below the upper
   figure). LINDI and local files report the transport figure as honestly
   unknown rather than echoing the chunk-level number.

Correctness was never at issue — before and after, the output is bitwise
identical to the independent reference and verifies against its manifest.

## Dask/LINDI baseline (Priority 4)

`benchmarks/benchmark_fish_trace_baseline.py` implements the fair baseline:
PyNWB + LINDI or remfile + Dask, consuming the **same retained Cellpose masks**
via `--labels` rather than re-segmenting, and reporting its missing features
honestly (`integrity_verified: false`, `resume.supported: false`, no manifests
or provenance). Transfer counters are available for remfile via an HTTP byte
counter and are recorded as unavailable for LINDI rather than as zero.

The baseline's numerical core is now itself validated:
`tests/test_benchmark_baselines.py::test_dask_trace_baseline_agrees_with_numpy_and_neuroflow`
requires three independent computations of the same per-cell means to agree
exactly — plain NumPy, the baseline's manual Dask chunk loop, and NeuroFlow's
planned extraction — over identical labels on fish-like chunking, including a
cell spanning two source chunks and an empty plane whose skip both sides must
account for. Without this, the publication comparison could be measuring a
baseline bug instead of a design difference.

**The archive-scale baseline run has not been performed:** see "Open gates".

## Cellpose biological quality

Two claims must not be conflated:

- **NeuroFlow ↔ direct Cellpose software equivalence: demonstrated.** Identical
  masks, 0 mismatched voxels, same model, same settings, same device.
- **Biological correctness of the masks: not demonstrated.** The retained
  segmentation contains substantial false positives and missed somata. This is a
  limitation of the segmentation model as applied, not of NeuroFlow's execution.
  No biological validation is claimed.

## Verification gates run in this pass

Command | Result
--- | ---
`.venv/bin/python -m pytest -q` | **226 passed, 1 skipped**
`.venv/bin/python -m ruff check .` | **All checks passed**
`.venv/bin/python -m basedpyright` | **0 errors, 0 warnings, 0 notes**

New regression tests added for the memory semantics: exact budget
decomposition; monotonicity of task bytes in the target; unattainable-target
reporting; per-worker model reserve; refusal with guidance when reserves consume
the target; planned-versus-measured reporting; that the per-task estimate no
longer double-counts process overhead; that `cpsam` on CPU cannot fit 2 GiB but
can on GPU; that stated worker availability is **clamped and not refused**; that
`compute()` bounds array data by task bytes rather than the headline total; that
the declared overhead envelope still covers the recorded attribution
measurements; and that the default segmentation limit admits the default model.
A further test pins the fair baseline itself: its manual Dask chunk loop, plain
NumPy, and NeuroFlow's extraction must produce exactly equal traces over
identical labels.

## Open gates

1. **LINDI equivalence is measured for the projection stage only.** The remote
   small-slice comparison is exact (see above), but no LINDI trace extraction or
   full fish workflow has been compared against the remfile result. The
   publication LINDI run in the command list below closes this.
2. **The Dask/LINDI baseline has never been run.** The script exists; no record
   exists. Numerical output, wall time, peak RSS, and bytes transferred are all
   unmeasured, so no comparative performance claim is available.
3. **The 4 GiB / 206-frame overrun is unexplained in detail.** The direction and
   magnitude are measured; the specific unmodelled allocation is hypothesised,
   not confirmed.
3a. **Resolved: the geometry-dependence found on DANDI:000223.** The
   per-worker runtime envelope and the separate transport-bytes estimate (see
   the second-dataset smoke section) bring the same command from a 3,126 MiB
   whole-process peak to 2,084 MiB against the 2 GiB target. The residual
   +1.8% is consistent with the target being approximate and unenforced; the
   4 GiB large-window overrun in item 3 is a different cause and remains open.
4. **No clean-tree publication record exists** for any stage. Every retained
   record is `current` with `git.dirty: true`.
5. Naming caution, not an error: `benchmarks/results/current-scaling.json` is
   the output of `benchmark_scaling.py` (suite
   `synthetic-bounded-memory-scaling`, problem-size scaling), while the
   resource matrix lives in `current-resource-scaling*.json` from
   `benchmark_resource_scaling.py`. Cite the right file for each claim.
6. Expert/manual soma-quality assessment is outstanding, or the paper must be
   explicitly limited to software-path equivalence and trace production.
7. An OSI-approved license must be selected. BSD-3-Clause remains the documented
   recommendation; the choice is the maintainer's.
8. Public CI and Docker confirmation, the tagged experiment commit, the archive,
   and the assigned DOI/release date are outstanding.

`python tools/check_release.py` checks automated repository invariants and
prints the manual gates. `--strict` remains nonzero until the externally
controlled gates are satisfied.

## Exact commands for the clean final benchmark

The publication benchmarks already refuse a dirty tree
(`benchmark_fish_pipeline.py`, `benchmark_resource_scaling.py`, and
`tools/check_release.py` all gate on `git.dirty is not False`). Run in order:

```bash
# 0. Gates must be green and the tree must be committed first.
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m basedpyright
git status --porcelain          # must print nothing
git rev-parse HEAD              # record this commit in the manuscript

# 1. Local correctness, resume, and integrity.
.venv/bin/python benchmarks/benchmark_projection.py \
    --record benchmarks/results/publication-local-projection.json \
    --classification publication
.venv/bin/python benchmarks/benchmark_resume_integrity.py \
    --record benchmarks/results/publication-resume-integrity.json \
    --classification publication

# 2. Component memory attribution (add cuda probes on a GPU host).
.venv/bin/python benchmarks/memory_attribution.py \
    --record benchmarks/results/publication-memory-attribution.json

# 3. Resource scaling, local fixture, three repetitions per configuration.
.venv/bin/python -m benchmarks.benchmark_resource_scaling \
    --fixture-root /tmp/nf-scaling --output-root /tmp/nf-scaling-out \
    --frames 192 --repetitions 3 \
    --record benchmarks/results/publication-resource-scaling.json \
    --classification publication

# 3b. Second-dataset generality smoke, three repetitions.
.venv/bin/python -m benchmarks.benchmark_dandi_smoke \
    --dandiset "DANDI:000223@0.260528.0906" \
    --asset cc499fe1-fe23-42aa-8db0-0e689970fb89 \
    --neurodata-type TwoPhotonSeries --frames 96 \
    --memory-limit "2 GiB" --expect-axes time,y,x --backend remfile \
    --repetitions 3 \
    --output-root /tmp/nf-smoke \
    --record benchmarks/results/publication-dandi-smoke-000223.json \
    --classification publication

# 4. Archive-scale fish pipeline. THE EXPENSIVE RUN — hours, ~230 GB read.
#    --output-root MUST NOT be publication/runs, which holds retained evidence.
#    Transport configuration is explicit and recorded so the baseline in
#    step 6 can be configured identically.
.venv/bin/python benchmarks/benchmark_fish_pipeline.py \
    --output-root publication/runs-publication \
    --record benchmarks/results/publication-fish-soma-traces-remfile.json \
    --backend remfile --memory-limit "4 GiB" --cellpose-device auto \
    --block-size 1048576 --cache-size-mib 64 \
    --classification publication

# 5. Same workflow over LINDI, for transport independence.
.venv/bin/python benchmarks/benchmark_fish_pipeline.py \
    --output-root publication/runs-publication-lindi \
    --record benchmarks/results/publication-fish-soma-traces-lindi.json \
    --backend lindi --memory-limit "4 GiB" --cellpose-device auto \
    --classification publication

# 6. Fair baseline over the retained masks from step 4, configured for
#    parity: identical block/cache to step 4, and --time-chunk set to the
#    NeuroFlow record's chosen window so both tools traverse the movie in
#    the same temporal passes. Read it first:
#      python -c "import json; print(json.load(open(
#        'benchmarks/results/publication-fish-soma-traces-remfile.json'
#      ))['execution']['trace_plan']['time_window'])"
.venv/bin/python benchmarks/benchmark_fish_trace_baseline.py \
    --labels publication/runs-publication/fish-cellpose.zarr \
    --reference-traces publication/runs-publication/fish-traces.zarr \
    --output /tmp/nf-baseline-traces.zarr \
    --record benchmarks/results/publication-fish-remfile-dask-traces.json \
    --backend remfile --block-size 1048576 --cache-size-mib 64 \
    --time-chunk <time_window from step 4's record> \
    --classification publication

# 6b. The LINDI-transport baseline variant (no transport counters; peak RSS
#     and wall time are still measured).
.venv/bin/python benchmarks/benchmark_fish_trace_baseline.py \
    --labels publication/runs-publication/fish-cellpose.zarr \
    --reference-traces publication/runs-publication/fish-traces.zarr \
    --output /tmp/nf-baseline-traces-lindi.zarr \
    --record benchmarks/results/publication-fish-lindi-dask-traces.json \
    --backend lindi \
    --time-chunk <time_window from step 4's record> \
    --classification publication
```

Note the memory limit in steps 4-5: `--memory-limit "2 GiB"` will be **refused**
when segmentation runs `cpsam` on CPU, by design. Use 4 GiB, or keep 2 GiB with
`--cellpose-device cuda` on a GPU host.

Comparability statement for step 6, to be carried into the manuscript: block
size, cache size, and temporal window are identical on both sides and recorded
in both records (`transport_configuration` in the NeuroFlow record,
`configuration_parity` in the baseline record); both sides consume the
identical retained masks; and both sides execute their accumulation
single-threaded (the baseline hardcodes one Dask worker, and NeuroFlow's trace
extraction runs one partition at a time by design), so neither tool holds a
concurrency advantage in the comparison.

## Manuscript claim classification

| Claim | Status |
| --- | --- |
| Bounded, resumable, integrity-verified trace extraction from a versioned remote DANDI NWB-HDF5 asset without materializing the movie | **SUPPORTED** — retained fish record; verified checksum; 29-task zero-read resume |
| NeuroFlow-mediated Cellpose is exactly equivalent to direct Cellpose on the same input, model, settings, and device | **SUPPORTED** — 0 mismatched voxels over 29 planes, 5,557 objects both ways |
| Extracted traces match a direct NumPy reference | **SUPPORTED** — maximum absolute and relative error 0.0 on the validated frames |
| Whole-process memory cost is attributable to named, separately measured components | **SUPPORTED** — twelve isolated component probes retained |
| `memory_limit` is an approximate total process-memory target, not a per-task allowance | **SUPPORTED** — exact decomposition, documented, regression-tested |
| A 2 GiB target yields substantially lower process RSS than before | **SUPPORTED** — 2,062 MiB measured versus 6,097 MiB for the same nominal request |
| `memory_limit` is a reliable bound on process RSS | **PARTIALLY SUPPORTED** — 0.7% overrun at 2 GiB on fish-like chunking and 1.8% whole-process on the fine-chunked DANDI:000223 layout after the per-worker runtime envelope (40% before it); 14.5% at 4 GiB with a large window remains open; approximate, unenforced, and must be described as a target |
| Users state resources and NeuroFlow chooses partitioning and concurrency without low-level knobs | **PARTIALLY SUPPORTED** — demonstrated locally on synthetic data (1→21 workers from the target alone); not yet demonstrated on the archive |
| GPU execution is supported and reduces host memory and Cellpose time | **PARTIALLY SUPPORTED** — component-level measurement is strong (3.21→2.09 GB host, 66.14→2.03 s, 32.5x); no full GPU archive run exists |
| Resource scaling behaviour across memory and worker configurations | **PARTIALLY SUPPORTED** — complete five-point local curve; no archive-scale curve |
| Computation semantics are independent of transport (remfile vs LINDI) | **PARTIALLY SUPPORTED** — byte-identical remote projection (equal checksums, error 0.0, same workflow_id) on a small slice of the real asset; full fish workflow through LINDI awaits the publication run |
| The execution engine is dataset-independent by construction and was validated on two real DANDI datasets with distinct imaging geometries, in addition to synthetic test fixtures | **SUPPORTED** for correctness and semantics — DANDI:000350 (4-D int16, plane-sized chunks, archive scale) and DANDI:000223 (3-D uint16, 32 KiB chunks, discovery by NWB type, exact agreement with an independent reference, verified output, correct axis inference). Planner resource-model *accuracy* on fine-chunked geometry is a separate, partially supported claim (see the RSS row). No broad validation across all NWB modalities is claimed |
| NeuroFlow compares favourably to a PyNWB + LINDI + Dask baseline | **NOT YET SUPPORTED** — the baseline's numerical core is locally validated against NumPy and NeuroFlow, but the archive-scale baseline has never been run; no comparative wall-time/RSS/transfer numbers exist |
| Segmented somata are biologically correct | **NOT YET SUPPORTED** — substantial false positives and missed somata; explicitly not claimed |
| Results were produced from an immutable, clean, archived commit | **NOT YET SUPPORTED** — every retained record has `git.dirty: true` |
