"""Pure metadata calculations used during planning."""

import math


def slice_shape(slices: tuple[slice, ...]) -> tuple[int, ...]:
    return tuple((item.stop or 0) - (item.start or 0) for item in slices)


def element_count(shape: tuple[int, ...]) -> int:
    return math.prod(shape)
