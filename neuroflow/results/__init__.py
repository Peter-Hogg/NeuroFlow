"""Lazy handles for persisted outputs."""

from neuroflow.results.base import Result, ResultStatus
from neuroflow.results.verification import VerificationReport
from neuroflow.results.workflow import PersistedResult, WorkflowResult

__all__ = [
    "PersistedResult",
    "Result",
    "ResultStatus",
    "VerificationReport",
    "WorkflowResult",
]
