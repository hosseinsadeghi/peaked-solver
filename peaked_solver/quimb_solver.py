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

* **cotengra** — Precomputes an optimised contraction tree using
  cotengra's hyper-optimizer, then evaluates exact amplitudes for
  many candidate bitstrings reusing the same tree.  Best for circuits
  with moderate treewidth (≤ ~28).

Requirements
------------
``pip install quimb``  (and optionally ``cotengra``, ``kahypar``,
``optuna`` for advanced contraction path optimisation).
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
# Cotengra mode (precomputed contraction tree + amplitude sweep)
# ---------------------------------------------------------------------------

def _solve_cotengra(
    circuit: QuantumCircuit,
    top_k: int,
    opt_time: float,
    n_candidates: int,
    seed_bitstrings: list[str] | None,
) -> SolverResult:
    """Solve via cotengra-optimised contraction tree + amplitude evaluation.

    1. Build the tensor network for an arbitrary output bitstring.
    2. Use cotengra HyperOptimizer to find the best contraction tree.
    3. Generate candidate bitstrings (seeded from MPS sampling + random).
    4. For each candidate, compute the exact amplitude using the tree.
    5. Return the top-k by probability.
    """
    import cotengra as ctg
    from quimb.tensor.circuit import Circuit

    t0 = time.perf_counter()

    qasm_str = _circuit_to_qasm(circuit)
    n = circuit.n_qubits

    circ = Circuit.from_openqasm2_str(qasm_str)

    # --- Step 1: Find optimal contraction tree ---
    print(f"  [cotengra] Finding contraction tree (opt_time={opt_time}s)...")
    t_opt = time.perf_counter()

    # Build the TN for a reference bitstring to get the structure
    ref_bs = "0" * n
    tn_ref = circ.amplitude_tn(ref_bs)

    opt = ctg.HyperOptimizer(
        methods=["greedy", "kahypar"],
        max_time=opt_time,
        progbar=True,
    )
    tree = tn_ref.contraction_tree(optimize=opt)

    tw = tree.contraction_width()
    cost = tree.contraction_cost()
    print(f"  [cotengra] Tree found in {time.perf_counter() - t_opt:.1f}s")
    print(f"  [cotengra] Contraction width: {tw:.1f}, log10(cost): {np.log10(float(cost)):.1f}")

    # --- Step 2: Generate candidate bitstrings ---
    candidates = set()

    # Add seeds if provided
    if seed_bitstrings:
        candidates.update(seed_bitstrings)
        print(f"  [cotengra] Added {len(seed_bitstrings)} seed bitstrings")

    # Try low-bond MPS sampling for more seeds
    try:
        from quimb.tensor.circuit import CircuitMPS
        mps_circ = CircuitMPS.from_openqasm2_str(qasm_str, gate_opts={"max_bond": 32})
        mps_samples = list(mps_circ.sample(min(1000, n_candidates)))
        candidates.update(mps_samples)
        print(f"  [cotengra] Added {len(mps_samples)} MPS seed samples (bond=32)")
    except Exception as e:
        print(f"  [cotengra] MPS seeding failed: {e}")

    # Random candidates to fill up
    rng = np.random.default_rng(42)
    while len(candidates) < n_candidates:
        bs = "".join(str(b) for b in rng.integers(0, 2, size=n))
        candidates.add(bs)

    candidates = list(candidates)
    print(f"  [cotengra] Evaluating {len(candidates)} candidates...")

    # --- Step 3: Compute amplitudes ---
    results: list[tuple[str, float]] = []
    t_eval = time.perf_counter()

    for i, bs in enumerate(candidates):
        tn_bs = circ.amplitude_tn(bs)
        amp = tn_bs.contract(optimize=tree)
        prob = abs(complex(amp)) ** 2
        results.append((bs, prob))

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_eval
            rate = (i + 1) / elapsed
            best_so_far = max(results, key=lambda x: x[1])
            print(f"  [cotengra] {i+1}/{len(candidates)} ({rate:.0f}/s) "
                  f"best prob so far: {best_so_far[1]:.6e}")

    # Sort and return top-k
    results.sort(key=lambda x: -x[1])
    top_bitstrings = results[:top_k]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    eval_ms = (time.perf_counter() - t_eval) * 1000.0

    return SolverResult(
        top_k_bitstrings=top_bitstrings,
        heuristic_used="quimb_cotengra",
        circuit_analysis={
            "contraction_width": float(tw),
            "log10_cost": float(np.log10(float(cost))),
            "opt_time_s": round(time.perf_counter() - t_opt, 1),
            "candidates_evaluated": len(candidates),
            "eval_time_ms": round(eval_ms, 0),
            "ms_per_amplitude": round(eval_ms / max(len(candidates), 1), 1),
        },
        accuracy_estimate=1.0,  # exact amplitudes
        compute_time_ms=elapsed_ms,
        truncation_error=0.0,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve(
    circuit: QuantumCircuit,
    top_k: int = 10,
    mode: Literal["mps", "tn", "cotengra"] = "mps",
    max_bond: int = 64,
    heavy_hex: bool = False,
    samples: int = 10_000,
    contractor: str = "greedy",
    opt_time: float = 60.0,
    n_candidates: int = 2000,
    seed_bitstrings: list[str] | None = None,
    **kwargs,
) -> SolverResult:
    """Solve a peaked circuit using quimb tensor-network backends.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to simulate.
    top_k : int
        Number of top bitstrings to return.
    mode : 'mps', 'tn', or 'cotengra'
        Which quimb backend to use:
        - ``'mps'``: CircuitMPS / CircuitPermMPS (fast, best for 1D-like).
        - ``'tn'``: General Circuit with full TN contraction (any topology).
        - ``'cotengra'``: Precompute optimal contraction tree, then sweep
          candidate bitstrings computing exact amplitudes.  Best for
          moderate treewidth (≤ ~28).  Requires ``cotengra`` package.
    max_bond : int
        Maximum bond dimension (MPS mode) or general truncation limit.
    heavy_hex : bool
        MPS mode only.  Use CircuitPermMPS with 1.5× bond dimension bump
        for IBM heavy-hex topologies.
    samples : int
        Number of bitstring samples to draw (mps/tn modes, default: 10,000).
    contractor : str
        TN mode only.  Contraction backend: 'greedy', 'auto', or
        'cotengra' (requires cotengra package).
    opt_time : float
        Cotengra mode only.  Time budget in seconds for the hyper-optimizer
        to search for the best contraction tree (default: 60).
    n_candidates : int
        Cotengra mode only.  Number of candidate bitstrings to evaluate
        (default: 2000).  Includes MPS-seeded and random candidates.
    seed_bitstrings : list[str] or None
        Cotengra mode only.  Known candidate bitstrings to evaluate
        (e.g. from a prior low-bond MPS run).

    Returns
    -------
    SolverResult
    """
    if mode == "mps":
        return _solve_mps(circuit, top_k, max_bond, heavy_hex, samples)
    elif mode == "tn":
        return _solve_tn(circuit, top_k, contractor, max_bond, samples)
    elif mode == "cotengra":
        return _solve_cotengra(circuit, top_k, opt_time, n_candidates, seed_bitstrings)
    else:
        raise ValueError(f"Unknown quimb mode: {mode!r}. Use 'mps', 'tn', or 'cotengra'.")
