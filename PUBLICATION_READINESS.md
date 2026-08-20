# Publication readiness status

Status date: 2026-08-20. This is an evidence ledger, not a claim that the
release or manuscript is complete.

## Engineering outcome

The release-candidate implementation now supports the intended north-star path:

```text
versioned remote DANDI NWB-HDF5
        → bounded NumPy temporal projection
        → actual plane-wise Cellpose
        → source-chunk-aware whole-movie mean traces
        → verified, resumable (time, cell) Zarr output
```

The same analysis semantics can run above remfile or the optional LINDI/PyNWB
bridge. Trace planning reads only the compact labels, indexes masks by physical
movie chunks, skips empty chunks, chooses a time window from a 2 GiB default
policy, and reports estimated versus unknown quantities explicitly. Each time
partition has a checksum manifest; execution attempts preserve their own wall
time, transfer, RSS, computed, and resumed task counts.

The repository also contains:

- an opt-in actual-Cellpose test comparing exact direct and NeuroFlow-mediated
  labels on the same deterministic reference projection;
- a flagship DANDI runner that repeats that direct comparison on every fish
  projection plane, validates leading traces with direct NumPy, and exercises a
  zero-recomputation completed-result resume;
- a manual PyNWB + remfile/LINDI + Dask source-chunk trace baseline over the
  exact same masks, with its missing resume/integrity/provenance features
  represented honestly;
- manual-only publication CI so large archive reads never run on pull requests.

## Retained evidence

- The current-engine projection development run used
  `DANDI:000350@0.240822.1759`, selected 50 frames (5,274,009,600 logical
  bytes) from a 323,296,788,480-byte logical movie, transferred 2,594,344,287
  observed response bytes, reached 1,648,193,536 bytes peak RSS, completed in
  218.416 seconds, wrote 54,755,424 bytes, and passed integrity verification.
- That record is deliberately classified `current`, not `publication`, because
  its captured Git state is dirty and it lacks an independent numerical
  reference.
- Deterministic local trace tests compare exact results with direct NumPy,
  including labels spanning chunks, empty-chunk skipping, resume, corruption
  detection, and repair.
- Fresh dirty-tree `current` records retain exact local projection agreement
  with direct NumPy/Dask and a three-partition interruption/repair run in which
  one completed partition survived interruption and one deliberately corrupted
  partition alone was recomputed.
- A real local LINDI 0.4.6 → PyNWB → lazy dataset slice test passes in the uv
  environment. LINDI transfer counters remain unknown through this bridge.

## Gates still open

- Run the real Cellpose test with downloaded model weights and retain the clean
  release-candidate job evidence.
- Run the complete fish pipeline from a clean immutable commit. No full-movie
  trace timing, transfer, peak RSS, object count, or biological result is
  claimed until its JSON exists.
- Run the LINDI/Dask baseline, and the remfile baseline if included in the paper.
- Complete expert/manual soma-quality assessment or explicitly limit the paper
  to software-path equivalence and fluorescence-trace production.
- Select an OSI-approved license. BSD-3-Clause is documented as the current
  recommendation, but this legal choice remains with the maintainer.
- Confirm public CI and Docker, tag the exact experiment commit, archive it, and
  add the assigned DOI/release date to metadata.

`python tools/check_release.py` checks automated repository invariants and
prints these manual gates. `python tools/check_release.py --strict` remains
nonzero until all externally controlled release gates are satisfied.
