from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SolverResult:
    """Container for the output of any heuristic solver."""
    top_k_bitstrings: list[tuple[str, float]] = field(default_factory=list)
    heuristic_used: str = ""
    circuit_analysis: dict = field(default_factory=dict)
    accuracy_estimate: float = 0.0
    compute_time_ms: float = 0.0
    truncation_error: float = 0.0
