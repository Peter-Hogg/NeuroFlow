"""Partition contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from neuroflow.selection.query import Selection


@dataclass(frozen=True)
class Partition:
    key: str
    read_slices: tuple[slice, ...]
    output_slices: tuple[slice, ...]
    trim_slices: tuple[slice, ...]
    coordinates: tuple[int | str, ...]

    def to_dict(self) -> dict[str, object]:
        def encode(values: tuple[slice, ...]) -> list[list[int | None]]:
            return [[item.start, item.stop, item.step] for item in values]

        return {
            "key": self.key,
            "read_slices": encode(self.read_slices),
            "output_slices": encode(self.output_slices),
            "trim_slices": encode(self.trim_slices),
            "coordinates": list(self.coordinates),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Partition":
        def decode(name: str) -> tuple[slice, ...]:
            raw = value.get(name)
            if not isinstance(raw, list):
                raise ValueError(f"partition {name} must be a list")
            decoded: list[slice] = []
            for item in raw:
                if not isinstance(item, list) or len(item) != 3:
                    raise ValueError(f"invalid slice in partition {name}")
                if any(part is not None and not isinstance(part, int) for part in item):
                    raise ValueError(f"invalid slice bound in partition {name}")
                start, stop, step = item
                if step not in (None, 1):
                    raise ValueError(f"partition {name} only supports unit slices")
                if start is None or stop is None or start < 0 or stop < start:
                    raise ValueError(f"invalid slice bounds in partition {name}")
                decoded.append(slice(item[0], item[1], item[2]))
            return tuple(decoded)

        coordinates = value.get("coordinates")
        if not isinstance(coordinates, list) or any(
            not isinstance(item, (int, str)) for item in coordinates
        ):
            raise ValueError("partition coordinates must contain integers or strings")
        return cls(
            key=str(value["key"]),
            read_slices=decode("read_slices"),
            output_slices=decode("output_slices"),
            trim_slices=decode("trim_slices"),
            coordinates=tuple(coordinates),
        )


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class PartitionPlan(Protocol):
    def build(self, selection: "Selection") -> Sequence[Partition]: ...

    def validate(self, selection: "Selection") -> ValidationReport: ...
