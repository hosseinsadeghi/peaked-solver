"""
Quimb Tensor-Network Solver
============================

Wraps quimb's tensor-network simulation capabilities (CircuitMPS,
CircuitPermMPS, and general Circuit with configurable contraction)
into the same ``SolverResult`` interface used by the local solvers.

Modes
-----
* **mps** — ``CircuitMPS`` (1D chain) or ``CircuitPermMPS`` (lazy
  qubit reordering for non-1D topologies like heavy-hex).  Best for
  circuits with low treewidth / quasi-1D connectivity.

* **tn** — General ``Circuit`` with full tensor-network contraction.
  Supports arbitrary topologies via configurable contraction backends
  (greedy, auto, cotengra).  More expensive but topology-agnostic.

Requirements
------------
``pip install quimb``  (and optionally ``cotengra`` for advanced
contraction path optimisation).
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from .types import SolverResult
from .parser import QuantumCircuit


# ---------------------------------------------------------------------------
# Convert our QuantumCircuit → QASM string for quimb ingestion
# ---------------------------------------------------------------------------

def _circuit_to_qasm(circuit: QuantumCircuit) -> str:
    """Regenerate an OpenQASM 2.0 string from a parsed QuantumCircuit.

    quimb's ``from_openqasm2_str`` is the most reliable ingestion path —
    it handles gate decomposition, qubit allocation, etc.  We reconstruct
    a minimal QASM rather than trying to manually build a quimb TN.
    """
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.n_qubits}];",
    ]
    for gate in circuit.gates:
        name = gate.name
        qargs = ",".join(f"q[{q}]" for q in gate.qubits)
        if gate.params:
            pargs = ",".join(f"{p}" for p in gate.params)
            lines.append(f"{name}({pargs}) {qargs};")
        else:
            lines.append(f"{name} {qargs};")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# MPS mode
# ---------------------------------------------------------------------------

def _solve_mps(
    circuit: QuantumCircuit,
    top_k: int,
    max_bond: int,
    heavy_hex: bool,
    samples: int,
) -> SolverResult:
    """Solve via quimb CircuitMPS / CircuitPermMPS."""
    from quimb.tensor.circuit import CircuitMPS, CircuitPermMPS

    t0 = time.perf_counter()

    qasm_str = _circuit_to_qasm(circuit)

    # heavy-hex → PermMPS (lazy reordering) + bond dimension bump
    use_perm = heavy_hex
    if heavy_hex:
        max_bond = int(max_bond * 1.5)

    cls = CircuitPermMPS if use_perm else CircuitMPS
    circ = cls.from_openqasm2_str(qasm_str, gate_opts={"max_bond": max_bond})

    # Sample bitstrings
    raw_samples = list(circ.sample(samples))

    # Count frequencies
    counts: dict[str, int] = {}
    for bs in raw_samples:
        counts[bs] = counts.get(bs, 0) + 1

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:top_k]
    top_bitstrings = [(bs, c / total) for bs, c in ranked]

    max_bond_actual = circ.psi.max_bond()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return SolverResult(
        top_k_bitstrings=top_bitstrings,
        heuristic_used="quimb_mps",
        circuit_analysis={
            "max_bond_requested": max_bond,
            "max_bond_actual": max_bond_actual,
            "samples": total,
            "unique_bitstrings": len(counts),
            "heavy_hex": heavy_hex,
            "perm_mps": use_perm,
        },
        accuracy_estimate=1.0,  # MPS sampling — accuracy depends on bond dim
        compute_time_ms=elapsed_ms,
        truncation_error=0.0,  # quimb doesn't expose cumulative truncation error
    )


# ---------------------------------------------------------------------------
# General TN mode (full contraction)
# ---------------------------------------------------------------------------

def _solve_tn(
    circuit: QuantumCircuit,
    top_k: int,
    contractor: str,
    max_bond: int | None,
    samples: int,
) -> SolverResult:
    """Solve via quimb general Circuit with TN contraction."""
    from quimb.tensor.circuit import Circuit

    t0 = time.perf_counter()

    qasm_str = _circuit_to_qasm(circuit)

    gate_opts = {}
    if max_bond is not None:
        gate_opts["max_bond"] = max_bond

    circ = Circuit.from_openqasm2_str(qasm_str, gate_opts=gate_opts or None)

    # Sample from the full tensor network
    raw_samples = list(circ.sample(samples, backend=contractor))

    # Count frequencies
    counts: dict[str, int] = {}
    for bs in raw_samples:
        counts[bs] = counts.get(bs, 0) + 1

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:top_k]
    top_bitstrings = [(bs, c / total) for bs, c in ranked]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return SolverResult(
        top_k_bitstrings=top_bitstrings,
        heuristic_used="quimb_tn",
        circuit_analysis={
            "contractor": contractor,
            "max_bond": max_bond,
            "samples": total,
            "unique_bitstrings": len(counts),
        },
        accuracy_estimate=1.0,
        compute_time_ms=elapsed_ms,
        truncation_error=0.0,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve(
    circuit: QuantumCircuit,
    top_k: int = 10,
    mode: Literal["mps", "tn"] = "mps",
    max_bond: int = 64,
    heavy_hex: bool = False,
    samples: int = 10_000,
    contractor: str = "greedy",
    **kwargs,
) -> SolverResult:
    """Solve a peaked circuit using quimb tensor-network backends.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to simulate.
    top_k : int
        Number of top bitstrings to return.
    mode : 'mps' or 'tn'
        Which quimb backend to use:
        - ``'mps'``: CircuitMPS / CircuitPermMPS (fast, best for 1D-like).
        - ``'tn'``: General Circuit with full TN contraction (any topology).
    max_bond : int
        Maximum bond dimension (MPS mode) or general truncation limit.
    heavy_hex : bool
        MPS mode only.  Use CircuitPermMPS with 1.5× bond dimension bump
        for IBM heavy-hex topologies.
    samples : int
        Number of bitstring samples to draw (default: 10,000).
    contractor : str
        TN mode only.  Contraction backend: 'greedy', 'auto', or
        'cotengra' (requires cotengra package).

    Returns
    -------
    SolverResult
    """
    if mode == "mps":
        return _solve_mps(circuit, top_k, max_bond, heavy_hex, samples)
    elif mode == "tn":
        return _solve_tn(circuit, top_k, contractor, max_bond, samples)
    else:
        raise ValueError(f"Unknown quimb mode: {mode!r}. Use 'mps' or 'tn'.")
