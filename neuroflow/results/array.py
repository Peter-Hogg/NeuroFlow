"""Lazy Zarr-backed result arrays."""

from dataclasses import dataclass

import dask.array as da


@dataclass(frozen=True)
class ArrayResult:
    uri: str
    component: str

    def as_dask_array(self) -> da.Array:
        return da.from_zarr(self.uri, component=self.component)

    def __getitem__(self, key: object) -> da.Array:
        return self.as_dask_array()[key]
