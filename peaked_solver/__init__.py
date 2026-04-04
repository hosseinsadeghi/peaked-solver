"""Peaked Circuit Solver — lightweight MPS and tensor-network solvers for peaked quantum circuits."""

from .types import SolverResult
from .parser import QuantumCircuit
from .mps import solve as mps_solve
from .tensor_network import solve as tn_solve
from .backend import use_jax, is_jax

__all__ = ["SolverResult", "QuantumCircuit", "mps_solve", "tn_solve", "use_jax", "is_jax"]
