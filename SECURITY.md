# Security policy

NeuroFlow 0.1 is the currently supported release line. Please report suspected
security vulnerabilities privately to peterhogg2006@gmail.com rather than in a
public issue. Include affected versions, reproduction steps, and impact when
possible. Receipt will be acknowledged within seven days.

NeuroFlow reads untrusted scientific files through PyNWB, HDF5, Zarr, fsspec,
and optional adapters. Users should process untrusted data in an isolated
environment and review external model weights before loading them.
