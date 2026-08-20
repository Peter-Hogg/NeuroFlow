from pathlib import Path

from tools.check_metadata import metadata_errors


def test_release_metadata_is_internally_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert metadata_errors(root) == []
