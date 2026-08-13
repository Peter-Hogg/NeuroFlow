"""Conservative metadata for commonly array-backed NWB types.

This registry is descriptive only; semantic selection still uses the actual PyNWB
class hierarchy stored with each object.
"""

ARRAY_NEURODATA_TYPES: frozenset[str] = frozenset(
    {
        "ElectricalSeries",
        "ImageSeries",
        "OnePhotonSeries",
        "RoiResponseSeries",
        "SpatialSeries",
        "TimeSeries",
        "TwoPhotonSeries",
    }
)
