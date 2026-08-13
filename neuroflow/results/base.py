"""Backend-neutral result handle contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from neuroflow.results.verification import VerificationReport


@dataclass(frozen=True)
class ResultStatus:
    state: Literal["planned", "running", "complete", "partial", "failed"]
    completed_partitions: tuple[str, ...] = ()
    failed_partitions: tuple[str, ...] = ()


class Result(Protocol):
    @property
    def arrays(self) -> Mapping[str, object]: ...

    @property
    def tables(self) -> Mapping[str, object]: ...

    @property
    def status(self) -> ResultStatus: ...

    @property
    def provenance(self) -> Mapping[str, object] | None: ...

    @property
    def failed_partitions(self) -> tuple[str, ...]: ...

    def execute(self) -> "Result": ...

    def resume(self) -> "Result": ...

    def verify(self, *, checksums: bool = True) -> VerificationReport: ...
