# Fish projection and segmentation validation protocol

This protocol separates engine correctness from biological interpretation.
Passing automated software checks is not evidence that detected objects are
neurons or that extracted traces are biologically meaningful.

## Software validation

1. Pin the Dandiset version, asset ID, NWB object, selection, and NeuroFlow Git
   commit.
2. Run the bounded temporal projection and retain its workflow specification,
   provenance, manifests, checksums, benchmark JSON, and preview.
3. Independently calculate at least three predetermined spatial crops with
   direct PyNWB/NumPy and compare shape, dtype, absolute error, relative error,
   and checksums using declared tolerances.
4. Interrupt a fresh run, resume it, and confirm completed partitions were not
   recomputed. Corrupt one owned partition and confirm verification fails and
   repair recomputes only that partition.
5. If segmentation or trace extraction is included, compare extracted traces
   with a direct NumPy reference for the identical labels and time interval.

## Scientific validation requiring an expert

1. Freeze the segmentation model, weights, thresholds, image normalization,
   plane selection, and random seeds.
2. Have a domain expert annotate a preregistered sample of planes/crops without
   seeing which method produced each candidate.
3. Record inclusion/exclusion rules and ambiguous objects. Report object-level
   precision, recall, F1, and overlap statistics with confidence intervals.
4. Review a preregistered sample of traces for motion artifacts, neuropil
   contamination, saturation, and correspondence with accepted calcium-event
   morphology.
5. Retain annotation files, reviewer checklist, adjudication log, figures,
   metric JSON, and the exact workflow specification.

No scientific-validation box should be checked until those retained artifacts
exist and an identified qualified reviewer has signed the checklist.
