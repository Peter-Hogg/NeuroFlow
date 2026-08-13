"""Lazy Parquet-backed result tables."""

from dataclasses import dataclass

import dask.dataframe as dd

from neuroflow.storage.base import join_uri


@dataclass(frozen=True)
class TableResult:
    uri: str
    name: str

    def as_dask_dataframe(self) -> dd.DataFrame:
        return dd.read_parquet(join_uri(self.uri, "tables", self.name))

    def query(self, expression: str) -> dd.DataFrame:
        return self.as_dask_dataframe().query(expression)
