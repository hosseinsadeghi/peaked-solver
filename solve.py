#!/usr/bin/env python3
"""Solve a peaked circuit QASM file and display the bitstring distribution.

Usage:
    python solve.py circuits/P1_little_peak.qasm
    python solve.py circuits/P3_sharp_peak.qasm --solver mps --top-k 20 --chi-max 128
    python solve.py circuits/P7_heavy_hex_1275.qasm --solver mps --heavy-hex
    python solve.py circuits/P2_swift_rise.qasm --solver 2d --timeout 120
    python solve.py circuits/P1_little_peak.qasm --solver quimb --quimb-mode mps --max-bond 64
    python solve.py circuits/P1_little_peak.qasm --solver quimb --quimb-mode tn --contractor greedy
"""

import argparse
import sys
import time
from pathlib import Path

from peaked_solver import QuantumCircuit, mps_solve, tn_solve


def main():
    parser = argparse.ArgumentParser(
        description="Solve a peaked circuit and show bitstring distribution",
    )
    parser.add_argument("qasm", type=Path, help="Path to .qasm file")
    parser.add_argument("--solver", choices=["mps", "2d", "quimb"], default="mps",
                        help="Solver: mps (local MPS), 2d (local TN contraction), "
                             "quimb (quimb tensor network). Default: mps")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top bitstrings to show (default: 10)")
    parser.add_argument("--chi-max", type=int, default=None,
                        help="MPS bond dimension (mps solver only)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout in seconds (2d solver only, default: 60)")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Number of candidate bitstrings (2d solver only)")
    parser.add_argument("--heavy-hex", action="store_true",
                        help="Heavy-hex topology mode (mps and quimb solvers)")

    # quimb-specific options
    quimb_group = parser.add_argument_group("quimb solver options")
    quimb_group.add_argument("--quimb-mode", choices=["mps", "tn", "cotengra"], default="mps",
                             help="quimb backend: mps (CircuitMPS), tn (general TN), cotengra (optimised contraction tree). Default: mps")
    quimb_group.add_argument("--max-bond", type=int, default=64,
                             help="Max bond dimension for quimb (default: 64)")
    quimb_group.add_argument("--samples", type=int, default=10000,
                             help="Number of bitstring samples for quimb mps/tn (default: 10000)")
    quimb_group.add_argument("--contractor", type=str, default="greedy",
                             help="TN contraction backend for quimb tn mode: greedy, auto, cotengra (default: greedy)")
    quimb_group.add_argument("--opt-time", type=float, default=60.0,
                             help="Cotengra hyper-optimizer time budget in seconds (default: 60)")
    quimb_group.add_argument("--n-candidates", type=int, default=2000,
                             help="Number of candidate bitstrings for cotengra mode (default: 2000)")
    quimb_group.add_argument("--seeds", type=str, default=None,
                             help="Comma-separated seed bitstrings for cotengra mode")

    args = parser.parse_args()

    # Resolve file path
    qasm_path = args.qasm
    if not qasm_path.exists():
        # Try in circuits/ directory
        alt = Path("circuits") / qasm_path.name
        if alt.exists():
            qasm_path = alt
        else:
            print(f"Error: {args.qasm} not found")
            sys.exit(1)

    # Parse circuit
    qasm_str = qasm_path.read_text()
    circuit = QuantumCircuit.from_openqasm(qasm_str)
    print(f"Circuit: {qasm_path.name}")
    print(f"Qubits:  {circuit.n_qubits}")
    print(f"Gates:   {len(circuit.gates)}")
    print(f"Solver:  {args.solver}" + (f" ({args.quimb_mode})" if args.solver == "quimb" else ""))
    print()

    # Solve
    t0 = time.time()
    if args.solver == "mps":
        kwargs = {"top_k": args.top_k, "heavy_hex": args.heavy_hex}
        if args.chi_max is not None:
            kwargs["chi_max"] = args.chi_max
        result = mps_solve(circuit, **kwargs)
    elif args.solver == "2d":
        kwargs = {"top_k": args.top_k, "timeout": args.timeout}
        if args.candidates is not None:
            kwargs["n_candidates"] = args.candidates
        result = tn_solve(circuit, **kwargs)
    elif args.solver == "quimb":
        from peaked_solver import quimb_solve
        if quimb_solve is None:
            print("Error: quimb is not installed. Run: pip install quimb", file=sys.stderr)
            sys.exit(1)
        seed_bitstrings = None
        if args.seeds:
            seed_bitstrings = [s.strip() for s in args.seeds.split(",")]
        result = quimb_solve(
            circuit,
            top_k=args.top_k,
            mode=args.quimb_mode,
            max_bond=args.max_bond,
            heavy_hex=args.heavy_hex,
            samples=args.samples,
            contractor=args.contractor,
            opt_time=args.opt_time,
            n_candidates=args.n_candidates,
            seed_bitstrings=seed_bitstrings,
        )
    wall_time = time.time() - t0

    # Display results
    bitstrings = result.top_k_bitstrings
    if not bitstrings:
        print("No bitstrings found.")
        sys.exit(1)

    total_prob = sum(p for _, p in bitstrings)
    max_prob = bitstrings[0][1] if bitstrings else 0

    print(f"{'Rank':<6} {'Bitstring':<{circuit.n_qubits + 4}} {'Probability':>12}  Bar")
    print("-" * (30 + circuit.n_qubits))

    bar_width = 40
    for i, (bs, prob) in enumerate(bitstrings, 1):
        bar_len = int(prob / max_prob * bar_width) if max_prob > 0 else 0
        bar = chr(0x2588) * bar_len
        print(f"{i:<6} {bs:<{circuit.n_qubits + 4}} {prob:>12.6f}  {bar}")

    print("-" * (30 + circuit.n_qubits))
    print(f"Top-{len(bitstrings)} total probability: {total_prob:.6f}")
    print(f"Accuracy estimate:          {result.accuracy_estimate:.4e}")
    print(f"Compute time:               {result.compute_time_ms:.0f} ms (wall: {wall_time:.1f}s)")
    print(f"Truncation error:           {result.truncation_error:.2e}")
    print(f"Analysis:                   {result.circuit_analysis}")


if __name__ == "__main__":
    main()
