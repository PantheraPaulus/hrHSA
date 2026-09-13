"""Validation strategies and result containers for SSFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LeaveOneIndividualOut:
    """Exact new-individual validation for SSFs."""

    heldout: Any = "all"
    n_train_per_id: int | None = None
    n_test_per_id: int | None = None
    seed: int = 42

    def run(self, analysis, **kwargs):
        from hsa.ssf.validation import validate_loio
        return validate_loio(analysis, self, **kwargs)


@dataclass(frozen=True)
class TemporalBlockCV:
    """Exact refit with a contiguous temporal holdout and optional embargo."""

    n_blocks: int = 5
    holdout_block: int = 2
    embargo: str = "24h"
    n_train_per_id: int | None = None
    n_test_per_id: int | None = None
    seed: int = 42

    def __post_init__(self):
        if self.n_blocks < 2:
            raise ValueError("n_blocks must be at least 2.")
        if not 0 <= self.holdout_block < self.n_blocks:
            raise ValueError("holdout_block must index one of n_blocks.")

    def run(self, analysis, **kwargs):
        from hsa.ssf.validation import validate_temporal_block
        return validate_temporal_block(analysis, self, **kwargs)


@dataclass
class SSFValidationResult:
    """Common wrapper for exact SSF validation refits."""

    summary: pd.DataFrame
    diagnostics: dict[Any, Any]
    scheme: Any
    analysis: Any = None

    def __len__(self):
        return len(self.summary)


__all__ = ["LeaveOneIndividualOut", "TemporalBlockCV", "SSFValidationResult"]
