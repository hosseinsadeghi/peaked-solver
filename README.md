# Peaked Circuit Solver

Lightweight MPS and tensor-network solvers for peaked quantum circuits. Only dependency: **numpy**.

## Quick start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hosseinsadeghi/peaked-solver/blob/main/peaked_solver_colab.ipynb)

Click the badge above, or:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File > Open notebook > GitHub tab
3. Paste: `hosseinsadeghi/peaked-solver`
4. Select `peaked_solver_colab.ipynb`
5. Runtime > Change runtime type > **High-RAM** (if available)

Colab gives you 12 GB RAM (free) or 50+ GB (Pro) — enough for the large circuits that OOM locally.

## Quick start (local)

```bash
git clone https://github.com/hosseinsadeghi/peaked-solver.git
cd peaked-solver
python solve.py circuits/P1_little_peak.qasm
python solve.py circuits/P3_sharp_peak.qasm --solver mps --top-k 20 --chi-max 128
python solve.py circuits/P7_heavy_hex_1275.qasm --solver mps --heavy-hex
python solve.py circuits/P2_swift_rise.qasm --solver 2d --timeout 120
```

## Solvers

| Solver | Flag | Best for |
|--------|------|----------|
| MPS | `--solver mps` (default) | Linear/sparse/heavy-hex topologies, low treewidth |
| Tensor Network | `--solver 2d` | Arbitrary topologies, 2D grids |

### MPS options

- `--chi-max N` — Max bond dimension (default: auto-selected from circuit structure)
- `--heavy-hex` — Enable IBM heavy-hex topology adjustments (Cuthill-McKee qubit reordering + 1.5x chi bump)

### Tensor network options

- `--timeout S` — Max seconds (default: 60)
- `--candidates N` — Number of candidate bitstrings to evaluate

## Circuits included

10 peaked circuits (P1-P10) from the [Peakbit Peaked Circuit Challenge](https://peakbit.io), ranging from 4-qubit toy circuits to 4020-qubit heavy-hex topologies.

## Python API

```python
from peaked_solver import QuantumCircuit, mps_solve, tn_solve

circuit = QuantumCircuit.from_openqasm(open('circuits/P3_sharp_peak.qasm').read())
result = mps_solve(circuit, top_k=10, chi_max=128)

for bitstring, prob in result.top_k_bitstrings:
    print(f'{bitstring}  p={prob:.6f}')
```
