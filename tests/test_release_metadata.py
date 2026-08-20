from pathlib import Path

from tools.check_metadata import metadata_errors
from tools.check_release import audit_release


def test_release_metadata_is_internally_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert metadata_errors(root) == []


def test_automated_release_audit_has_no_repository_errors() -> None:
    root = Path(__file__).resolve().parents[1]
    assert audit_release(root).errors == ()
