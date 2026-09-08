# Benchmark summary

| benchmark | classification | backend | shape | selected_bytes | tasks | peak_rss_bytes | wall_time_seconds | numerically_valid | integrity_verified | resumed_partitions | git_sha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dandi-fish-manual-lindi-dask-traces | publication | pynwb-lindi-dask | 3065×888×2048×29 | 323296788480 | 1015 | 4041764864 | 16235.174540011998 | True | False | unknown | 5afa77b0eac63e49f3811b5380cf7c21278198eb |
| dandi-fish-manual-remfile-dask-traces | publication | pynwb-remfile-dask | 3065×888×2048×29 | 323296788480 | 1015 | 2948861952 | 18445.333676120004 | True | False | unknown | 5afa77b0eac63e49f3811b5380cf7c21278198eb |
| dandi-fish-cellpose-soma-traces | publication | dandi-nwb-hdf5-lindi | 3065×888×2048×29 | 323296788480 | 93 | 5511471104 | 15670.866997247998 | True | True | unknown | 5afa77b0eac63e49f3811b5380cf7c21278198eb |
| dandi-fish-cellpose-soma-traces | publication | dandi-nwb-hdf5-remfile | 3065×888×2048×29 | 323296788480 | 93 | 4240654336 | 15548.405751163984 | True | True | unknown | 5afa77b0eac63e49f3811b5380cf7c21278198eb |
| local-projection-correctness | publication | nwb-zarr | 40×256×256 | 10485760 | 1 | 303562752 | 0.06453156500356272 | True | True | 0 | 5afa77b0eac63e49f3811b5380cf7c21278198eb |
| resume-integrity | publication | nwb-zarr | 12×8×8 | 3072 | 3 | 214810624 | 0.15078698098659515 | True | True | 1 | 5afa77b0eac63e49f3811b5380cf7c21278198eb |
