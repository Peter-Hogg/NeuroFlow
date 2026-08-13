# Reference test data

The network-free test suite generates a deterministic NWB movie containing
`numpy.arange(120, dtype=float32).reshape(10, 3, 4)`. Its expected first-five-frame
temporal median is stored in `projection_reference.json`. Both NWB-Zarr and
NWB-HDF5 fixtures are compared with this reference and with direct NumPy.

The fixture is deliberately generated during testing rather than committed as
a binary archive so reviewers can inspect every input value and construction
step in `tests/conftest.py`.
