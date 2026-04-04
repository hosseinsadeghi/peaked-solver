#!/usr/bin/env python3
"""Solve a peaked circuit QASM file and display the bitstring distribution.

Usage:
    python solve.py circuits/P1_little_peak.qasm
    python solve.py circuits/P3_sharp_peak.qasm --solver mps --top-k 20 --chi-max 128
    python solve.py circuits/P7_heavy_hex_1275.qasm --solver mps --heavy-hex
    python solve.py circuits/P2_swift_rise.qasm --solver 2d --timeout 120
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
    parser.add_argument("--solver", choices=["mps", "2d"], default="mps",
                        help="Solver: mps (Matrix Product State) or 2d (general tensor network). Default: mps")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top bitstrings to show (default: 10)")
    parser.add_argument("--chi-max", type=int, default=None,
                        help="MPS bond dimension (mps solver only)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout in seconds (2d solver only, default: 60)")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Number of candidate bitstrings (2d solver only)")
    parser.add_argument("--heavy-hex", action="store_true",
                        help="Heavy-hex topology mode: Cuthill-McKee qubit reordering + 1.5x chi bump (mps solver only)")
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
    print(f"Solver:  {args.solver}")
    print()

    # Solve
    t0 = time.time()
    if args.solver == "mps":
        kwargs = {"top_k": args.top_k, "heavy_hex": args.heavy_hex}
        if args.chi_max is not None:
            kwargs["chi_max"] = args.chi_max
        result = mps_solve(circuit, **kwargs)
    else:
        kwargs = {"top_k": args.top_k, "timeout": args.timeout}
        if args.candidates is not None:
            kwargs["n_candidates"] = args.candidates
        result = tn_solve(circuit, **kwargs)
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
