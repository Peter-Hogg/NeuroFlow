# pyright: reportCallIssue=false

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from hdmf_zarr import NWBZarrIO, ZarrDataIO
from pynwb import NWBFile, TimeSeries


@pytest.fixture()
def nwb_zarr(tmp_path: Path) -> tuple[Path, np.ndarray]:
    path = tmp_path / "session.nwb.zarr"
    data = np.arange(120, dtype=np.float32).reshape(10, 3, 4)
    nwb = NWBFile(
        session_description="test session",
        identifier="session-1",
        session_start_time=datetime.now(timezone.utc),
        session_id="session-1",
    )
    nwb.add_acquisition(
        TimeSeries(
            name="movie",
            data=ZarrDataIO(data, chunks=(2, 3, 4)),
            unit="a.u.",
            rate=2.0,
        )
    )
    nwb.add_acquisition(
        TimeSeries(
            name="irregular",
            data=ZarrDataIO(data[:4], chunks=(2, 3, 4)),
            unit="a.u.",
            timestamps=ZarrDataIO(
                np.array([0.0, 0.4, 1.1, 2.0], dtype=np.float64), chunks=(2,)
            ),
        )
    )
    nwb.add_acquisition(
        TimeSeries(
            name="other",
            data=ZarrDataIO(data[:2], chunks=(1, 3, 4)),
            unit="a.u.",
            rate=1.0,
        )
    )
    with NWBZarrIO(path, mode="w") as io:
        io.write(nwb)
    return path, data
