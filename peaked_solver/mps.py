"""
Tensor Network (MPS) Heuristic
===============================

Simulates a quantum circuit using a **Matrix Product State** (MPS)
representation -- the work-horse of one-dimensional tensor-network
methods.

Background
----------
Any n-qubit state |psi> can be written exactly as an MPS:

    |psi> = sum_{s1,...,sn} A1^{s1} A2^{s2} ... An^{sn} |s1 s2 ... sn>

where each A_i^{s_i} is a matrix of size (chi_{i-1}, chi_i) and s_i
in {0,1}.  The numbers chi_i are called **bond dimensions**.

For a product state chi = 1 everywhere.  Entanglement grows chi; in
the worst case chi ~ 2^{n/2}.  *Peaked* circuits generate states with
limited entanglement across most cuts, so chi stays manageable.

Key operations
--------------
* **Single-qubit gate on qubit i:** Contract the 2x2 gate matrix with
  the physical index of tensor A_i.  Cost: O(chi^2).

* **Two-qubit gate on adjacent qubits (i, i+1):**
  1. Contract A_i and A_{i+1} into a single tensor Theta of shape
     (chi_{i-1}, 2, 2, chi_{i+1}).
  2. Reshape gate to (4, 4) and apply to the (2, 2) physical indices.
  3. SVD: Theta -> U @ diag(S) @ V^dagger, truncate to chi_max.
  4. Absorb singular values: new A_i = U @ diag(S), new A_{i+1} = V^dagger.
  The **truncation error** from discarding small singular values is
  bounded by the sum of their squares (Eckart-Young theorem).

* **Non-adjacent two-qubit gate (i, j):**  SWAP qubits to make them
  adjacent, apply the gate, then SWAP back.

Peaked-circuit insight
~~~~~~~~~~~~~~~~~~~~~~
For peaked circuits (concentrated output distribution), the entanglement
entropy across most bipartitions is O(log n) rather than O(n), meaning
chi stays polynomial -- exactly the regime where MPS excels.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .types import SolverResult
from .parser import QuantumCircuit, GATE_LIBRARY
from .backend import xp, svd, to_numpy, from_numpy, is_jax

# ---------------------------------------------------------------------------
# Qubit reordering for near-1D topologies (heavy-hex, etc.)
# ---------------------------------------------------------------------------


def _cuthill_mckee_order(circuit: QuantumCircuit) -> list[int] | None:
    """Compute a Cuthill-McKee linear ordering that minimises bandwidth.

    Returns a permutation list ``perm`` where ``perm[new_pos] = old_qubit``,
    or *None* if reordering would not help (already linear / too small).
    """
    n = circuit.n_qubits
    if n < 4:
        return None

    # Build interaction adjacency
    adj: dict[int, set[int]] = {q: set() for q in range(n)}
    for g in circuit.gates:
        if len(g.qubits) >= 2:
            for qi in g.qubits:
                for qj in g.qubits:
                    if qi != qj:
                        adj[qi].add(qj)

    # Check if already linear (max distance between connected qubits is 1)
    max_dist = max(
        (abs(qi - qj) for q in range(n) for qj in adj[q] for qi in [q]),
        default=0,
    )
    if max_dist <= 1:
        return None

    # Cuthill-McKee: BFS from a peripheral node (lowest degree), sorted by
    # degree at each level.  Produces a good bandwidth-reducing permutation.
    from collections import deque

    start = min(range(n), key=lambda q: len(adj[q]))
    visited = set()
    order: list[int] = []
    queue = deque([start])
    visited.add(start)

    while queue:
        v = queue.popleft()
        order.append(v)
        neighbours = sorted(adj[v] - visited, key=lambda q: len(adj[q]))
        for nb in neighbours:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    # Add any disconnected qubits at the end
    for q in range(n):
        if q not in visited:
            order.append(q)

    # Check that reordering actually improves max hop distance
    inv = [0] * n
    for new_pos, old_q in enumerate(order):
        inv[old_q] = new_pos
    new_max_dist = max(
        (abs(inv[qi] - inv[qj]) for q in range(n) for qj in adj[q] for qi in [q]),
        default=0,
    )
    if new_max_dist >= max_dist:
        return None  # no improvement

    return order


def _reorder_circuit(
    circuit: QuantumCircuit, perm: list[int]
) -> QuantumCircuit:
    """Return a new circuit with qubit indices remapped by *perm*.

    ``perm[new_pos] = old_qubit``, so we need the inverse map.
    """
    n = circuit.n_qubits
    inv = [0] * n
    for new_pos, old_q in enumerate(perm):
        inv[old_q] = new_pos

    reordered = QuantumCircuit(n)
    for gate in circuit.gates:
        new_qubits = [inv[q] for q in gate.qubits]
        reordered.add_gate(gate.name, new_qubits, gate.params)
    return reordered


# ---------------------------------------------------------------------------
# SWAP gate matrix  (used internally to move non-adjacent qubits)
# ---------------------------------------------------------------------------
_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


# ---------------------------------------------------------------------------
# MPS core
# ---------------------------------------------------------------------------

class MPS:
    """Matrix Product State for n qubits.

    Each tensor ``self.tensors[i]`` has shape ``(chi_left, 2, chi_right)``
    where the middle index is the physical (computational basis) index.

    The state is initialized to |0...0>.
    """

    def __init__(self, n_qubits: int, chi_max: int = 64) -> None:
        self.n = n_qubits
        self.chi_max = chi_max
        self.total_truncation_error: float = 0.0

        # |0...0> : each tensor is shape (1, 2, 1)
        self.tensors = []
        for _ in range(n_qubits):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0
            self.tensors.append(from_numpy(t))

    # ---- diagonal gate optimisation (Papers 71, 75) -------------------------

    def apply_diagonal_two_adjacent(
        self, diag_values, q0: int, q1: int
    ) -> None:
        """Apply a diagonal two-qubit gate to adjacent qubits."""
        assert q1 == q0 + 1
        A = self.tensors[q0]
        B = self.tensors[q1]

        theta = xp.tensordot(A, B, axes=([2], [0]))  # (chi_l, 2, 2, chi_r)

        # Apply diagonal phases — JAX-compatible (no in-place mutation)
        diag_2d = from_numpy(np.array(diag_values, dtype=complex).reshape(2, 2))
        theta = theta * diag_2d[xp.newaxis, :, :, xp.newaxis]

        chi_l, _, _, chi_r = theta.shape
        chi_m = A.shape[2]
        mat = theta.reshape(chi_l * 2, 2 * chi_r)
        U, S, Vh = svd(mat, full_matrices=False)
        chi_new = min(self.chi_max, len(S), chi_m + 1)
        if len(S) > chi_new:
            self.total_truncation_error += float(to_numpy(xp.sum(S[chi_new:] ** 2)))
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]
        self.tensors[q0] = (U * S[xp.newaxis, :]).reshape(chi_l, 2, chi_new)
        self.tensors[q1] = Vh.reshape(chi_new, 2, chi_r)

    @staticmethod
    def is_diagonal_gate(gate_matrix, tol: float = 1e-10) -> bool:
        """Check if a gate matrix is diagonal."""
        m = to_numpy(gate_matrix)
        return np.allclose(m - np.diag(np.diag(m)), 0, atol=tol)

    # ---- single-qubit gate ------------------------------------------------

    def apply_single(self, gate_matrix, qubit: int) -> None:
        """Apply a 2x2 unitary to qubit *qubit*."""
        g = from_numpy(np.asarray(gate_matrix, dtype=complex))
        self.tensors[qubit] = xp.einsum(
            "ij, ajb -> aib", g, self.tensors[qubit]
        )

    # ---- two-qubit gate on adjacent qubits --------------------------------

    def apply_two_adjacent(
        self, gate_matrix, q0: int, q1: int
    ) -> None:
        """Apply a 4x4 gate to adjacent qubits q0, q1 (q1 = q0+1)."""
        assert q1 == q0 + 1, "Qubits must be adjacent for this method"

        A = self.tensors[q0]
        B = self.tensors[q1]
        g = from_numpy(np.asarray(gate_matrix, dtype=complex))

        theta = xp.tensordot(A, B, axes=([2], [0]))

        gate_4d = g.reshape(2, 2, 2, 2)
        theta = xp.einsum("ijkl, aklb -> aijb", gate_4d, theta)

        chi_l = theta.shape[0]
        chi_r = theta.shape[3]
        theta_mat = theta.reshape(chi_l * 2, 2 * chi_r)

        U, S, Vh = svd(theta_mat, full_matrices=False)

        chi_new = min(self.chi_max, len(S))
        if len(S) > chi_new:
            self.total_truncation_error += float(to_numpy(xp.sum(S[chi_new:] ** 2)))
        S = S[:chi_new]
        U = U[:, :chi_new]
        Vh = Vh[:chi_new, :]

        US = U * S[xp.newaxis, :]
        self.tensors[q0] = US.reshape(chi_l, 2, chi_new)
        self.tensors[q1] = Vh.reshape(chi_new, 2, chi_r)

    # ---- two-qubit gate on non-adjacent qubits ----------------------------

    def apply_two(self, gate_matrix, q0: int, q1: int) -> None:
        """Apply a 4x4 gate to potentially non-adjacent qubits.

        If the qubits are not adjacent, we SWAP qubit q0 towards q1
        until they are neighbours, apply the gate, then SWAP back.

        Uses diagonal fast-path when the gate is diagonal (CZ, etc.)
        to avoid unnecessary bond dimension growth.

        The SWAP gates are exact (no truncation beyond chi_max), so
        the logical state is preserved.
        """
        if abs(q0 - q1) == 1:
            lo, hi = min(q0, q1), max(q0, q1)
            gm = to_numpy(gate_matrix)
            if q0 < q1:
                if self.is_diagonal_gate(gm):
                    self.apply_diagonal_two_adjacent(np.diag(gm), lo, hi)
                else:
                    self.apply_two_adjacent(gate_matrix, lo, hi)
            else:
                g4 = gm.reshape(2, 2, 2, 2)
                g4_swapped = g4.transpose(1, 0, 3, 2).reshape(4, 4)
                self.apply_two_adjacent(g4_swapped, lo, hi)
            return

        # Non-adjacent: move q0 next to q1 via SWAPs
        direction = 1 if q1 > q0 else -1
        current_pos = q0
        target = q1 - direction  # one step before q1

        # Move q0 towards q1
        swap_positions = []
        while current_pos != target:
            next_pos = current_pos + direction
            lo, hi = min(current_pos, next_pos), max(current_pos, next_pos)
            self.apply_two_adjacent(_SWAP, lo, hi)
            swap_positions.append((lo, hi))
            current_pos = next_pos

        # Now current_pos and q1 are adjacent -- apply the actual gate
        lo, hi = min(current_pos, q1), max(current_pos, q1)
        if current_pos < q1:
            self.apply_two_adjacent(gate_matrix, lo, hi)
        else:
            g4 = gate_matrix.reshape(2, 2, 2, 2)
            g4_swapped = g4.transpose(1, 0, 3, 2).reshape(4, 4)
            self.apply_two_adjacent(g4_swapped, lo, hi)

        # SWAP back (reverse order)
        for lo, hi in reversed(swap_positions):
            self.apply_two_adjacent(_SWAP, lo, hi)

    # ---- amplitude extraction ---------------------------------------------

    def get_amplitude(self, bitstring: str) -> complex:
        """Compute <bitstring|psi> by contracting the MPS chain."""
        result = self.tensors[0][:, int(bitstring[0]), :]
        for i in range(1, self.n):
            bit = int(bitstring[i])
            mat = self.tensors[i][:, bit, :]
            result = result @ mat
        return complex(to_numpy(result)[0, 0])

    def get_probabilities(self, bitstrings: list[str]) -> list[float]:
        """Compute |<bitstring|psi>|^2 for a list of candidate bitstrings."""
        return [abs(self.get_amplitude(bs)) ** 2 for bs in bitstrings]

    def get_top_k_bitstrings(self, k: int) -> list[tuple[str, float]]:
        """Find the k most probable bitstrings via greedy MPS sampling."""
        beam = [("", from_numpy(np.ones((1,), dtype=complex)))]

        for i in range(self.n):
            candidates = []
            for prefix, vec in beam:
                for bit in [0, 1]:
                    mat = self.tensors[i][:, bit, :]
                    new_vec = vec @ mat

                    prob_est = float(to_numpy(xp.sum(xp.abs(new_vec) ** 2)))
                    candidates.append((prefix + str(bit), new_vec, prob_est))

            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = [(bs, vec) for bs, vec, _ in candidates[: max(2 * k, 20)]]

        results = []
        for bs, vec in beam:
            v = to_numpy(vec)
            amp = complex(v[0]) if v.shape[0] == 1 else complex(np.sum(v))
            prob = abs(amp) ** 2
            results.append((bs, prob))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# ---------------------------------------------------------------------------
# Public solve() function
# ---------------------------------------------------------------------------

def _adaptive_chi_max(circuit: QuantumCircuit, user_chi: int | None) -> int:
    """Choose chi_max based on circuit structure.

    For shallow / low-entanglement circuits, a smaller chi suffices.
    For deep circuits with many two-qubit gates, we use more.
    """
    if user_chi is not None:
        return user_chi
    n = circuit.n_qubits
    depth = circuit.depth()
    two_q = sum(1 for g in circuit.gates if len(g.qubits) == 2)

    # Heuristic: entanglement grows with depth * two-qubit density
    density = two_q / max(n * depth, 1)

    if depth <= 6 or density < 0.1:
        return 32
    elif depth <= 20 or density < 0.3:
        return 64
    elif n <= 30:
        return 128
    else:
        return 64  # memory-constrained for large circuits


def solve(
    circuit: QuantumCircuit,
    top_k: int = 10,
    chi_max: int | None = None,
    heavy_hex: bool = False,
    **kwargs,
) -> SolverResult:
    """Solve a peaked circuit using Matrix Product State simulation.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to simulate.
    top_k : int
        Number of top bitstrings to return.
    chi_max : int or None
        Maximum bond dimension.  If None, automatically chosen based on
        circuit structure.  Larger = more accurate but slower.
    heavy_hex : bool
        When True, apply adjustments for heavy-hex (IBM Eagle/Heron)
        topology:
        - Reorder qubits via Cuthill-McKee to minimise SWAP overhead
          (degree-3 bridge qubits are placed to minimise long-range hops)
        - Bump chi_max by 1.5x to absorb the extra entanglement from
          non-linear connectivity

    Returns
    -------
    SolverResult
    """
    t0 = time.perf_counter()

    # --- heavy-hex adjustments ---
    qubit_perm = None
    if heavy_hex:
        qubit_perm = _cuthill_mckee_order(circuit)
        if qubit_perm is not None:
            circuit = _reorder_circuit(circuit, qubit_perm)

    effective_chi = _adaptive_chi_max(circuit, chi_max)
    if heavy_hex and chi_max is None:
        # Degree-3 bridge nodes create modestly more entanglement than
        # a pure 1D chain.  A 1.5x bump keeps truncation error low.
        effective_chi = int(effective_chi * 1.5)

    mps = MPS(circuit.n_qubits, chi_max=effective_chi)

    n_diagonal_gates = 0
    for gate in circuit.gates:
        if len(gate.qubits) == 1:
            mps.apply_single(gate.matrix, gate.qubits[0])
        elif len(gate.qubits) == 2:
            if MPS.is_diagonal_gate(gate.matrix):
                n_diagonal_gates += 1
            mps.apply_two(gate.matrix, gate.qubits[0], gate.qubits[1])
        elif len(gate.qubits) == 3:
            _apply_three_qubit_gate(mps, gate)
        else:
            raise NotImplementedError(
                f"Gates on {len(gate.qubits)} qubits not supported in MPS"
            )

    top_bitstrings = mps.get_top_k_bitstrings(top_k)

    # --- un-reorder bitstrings back to original qubit labelling ---
    if qubit_perm is not None:
        inv = [0] * circuit.n_qubits
        for new_pos, old_q in enumerate(qubit_perm):
            inv[old_q] = new_pos
        remapped = []
        for bs, prob in top_bitstrings:
            original_bs = ["0"] * circuit.n_qubits
            for new_pos, old_q in enumerate(qubit_perm):
                original_bs[old_q] = bs[new_pos]
            remapped.append(("".join(original_bs), prob))
        top_bitstrings = remapped

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    trunc = mps.total_truncation_error
    accuracy = max(0.0, 1.0 - trunc)

    # Max bond dimension actually used
    max_bond = max(
        (t.shape[0] for t in mps.tensors), default=1
    )

    return SolverResult(
        top_k_bitstrings=top_bitstrings,
        heuristic_used="tensor_network",
        circuit_analysis={
            "chi_max": effective_chi,
            "max_bond_used": max_bond,
            "n_diagonal_gates": n_diagonal_gates,
            "heavy_hex": heavy_hex,
            "qubit_reordered": qubit_perm is not None,
        },
        accuracy_estimate=accuracy,
        compute_time_ms=elapsed_ms,
        truncation_error=trunc,
    )


def _apply_three_qubit_gate(mps: MPS, gate) -> None:
    """Apply a 3-qubit gate to an MPS by contracting three sites.

    For the Toffoli gate on qubits (q0, q1, q2), we:
    1. SWAP q0, q1 to be adjacent to q2
    2. Contract three tensors, apply 8x8 matrix, SVD split back

    For simplicity, we use a brute-force approach: bring all three qubits
    adjacent, contract into one big tensor, apply gate, split via two SVDs.
    """
    q0, q1, q2 = gate.qubits
    qs = sorted([q0, q1, q2])

    # Bring qubits to adjacent positions starting at qs[0]
    # First, let's use a simpler strategy: move qubits next to each other
    # via SWAPs, apply the gate as two 2-qubit operations, swap back.

    # Actually, for Toffoli specifically, we can use its known decomposition
    # into CNOT + T + H gates. But for generality, let's do the tensor contraction.

    # Strategy: move all three qubits to consecutive positions,
    # contract 3 tensors, apply 8x8 gate, split via 2 SVDs.

    # For simplicity, we move q1 and q2 to be right after q0.
    # This is done by the SWAP-based approach.

    n = mps.n
    # Map original positions to target positions
    targets = [qs[0], qs[0] + 1, qs[0] + 2]

    if qs[0] + 2 >= n:
        # Shift left if we'd go out of bounds
        targets = [n - 3, n - 2, n - 1]

    # Build permutation: which original qubit is at each target position
    # We just move the three qubits to be adjacent, respecting their order

    # Simple approach: SWAP the qubits into consecutive positions
    _SWAP_mat = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex)

    # Move q1 next to q0 (if not already)
    swap_log_1 = []
    pos1 = qs[1]
    target1 = qs[0] + 1
    while pos1 > target1:
        mps.apply_two_adjacent(_SWAP_mat, pos1 - 1, pos1)
        swap_log_1.append(pos1 - 1)
        pos1 -= 1
    while pos1 < target1:
        mps.apply_two_adjacent(_SWAP_mat, pos1, pos1 + 1)
        swap_log_1.append(pos1)
        pos1 += 1

    # Move q2 next to q1's new position
    swap_log_2 = []
    pos2 = qs[2] if qs[2] > qs[1] else qs[2]
    # After moving q1, positions may have shifted
    # q2's position: if q2 > q1 and we moved q1 left, q2 stays
    # if q2 > q1 and we moved q1 right past q2... etc.
    # Simpler: track actual position of q2 after the swaps
    actual_pos2 = qs[2]
    # The swaps in swap_log_1 swap positions, potentially moving q2
    for swap_pos in swap_log_1:
        if actual_pos2 == swap_pos:
            actual_pos2 = swap_pos + 1
        elif actual_pos2 == swap_pos + 1:
            actual_pos2 = swap_pos

    target2 = qs[0] + 2
    if target2 >= n:
        target2 = n - 1

    while actual_pos2 > target2:
        mps.apply_two_adjacent(_SWAP_mat, actual_pos2 - 1, actual_pos2)
        swap_log_2.append(actual_pos2 - 1)
        actual_pos2 -= 1
    while actual_pos2 < target2:
        mps.apply_two_adjacent(_SWAP_mat, actual_pos2, actual_pos2 + 1)
        swap_log_2.append(actual_pos2)
        actual_pos2 += 1

    # Now the three qubits are at consecutive positions starting at qs[0]
    base = qs[0]
    if base + 2 >= n:
        base = n - 3

    # Contract three tensors into one
    A = mps.tensors[base]      # (chi_l, 2, chi1)
    B = mps.tensors[base + 1]  # (chi1, 2, chi2)
    C = mps.tensors[base + 2]  # (chi2, 2, chi_r)

    AB = xp.tensordot(A, B, axes=([2], [0]))
    theta = xp.tensordot(AB, C, axes=([3], [0]))

    sorted_to_gate = [gate.qubits.index(sq) for sq in qs]

    gate_8x8 = from_numpy(np.asarray(gate.matrix, dtype=complex))
    gate_tensor = gate_8x8.reshape(2, 2, 2, 2, 2, 2)
    perm_in = [3 + sorted_to_gate[j] for j in range(3)]
    perm_out = [sorted_to_gate[j] for j in range(3)]
    gate_tensor = gate_tensor.transpose(perm_out + perm_in)

    chi_l = theta.shape[0]
    chi_r = theta.shape[4]
    theta_flat = theta.reshape(chi_l, 8, chi_r)
    gate_flat = gate_tensor.reshape(8, 8)
    theta_new = xp.einsum("ij, ajb -> aib", gate_flat, theta_flat)
    theta = theta_new.reshape(chi_l, 2, 2, 2, chi_r)

    mat1 = theta.reshape(chi_l * 2, 4 * chi_r)
    U1, S1, Vh1 = svd(mat1, full_matrices=False)
    chi1_new = min(mps.chi_max, len(S1))
    if len(S1) > chi1_new:
        mps.total_truncation_error += float(to_numpy(xp.sum(S1[chi1_new:] ** 2)))
    U1 = U1[:, :chi1_new]
    S1 = S1[:chi1_new]
    Vh1 = Vh1[:chi1_new, :]

    mps.tensors[base] = (U1 * S1[xp.newaxis, :]).reshape(chi_l, 2, chi1_new)

    remainder = Vh1.reshape(chi1_new, 2, 2, chi_r)
    mat2 = remainder.reshape(chi1_new * 2, 2 * chi_r)
    U2, S2, Vh2 = svd(mat2, full_matrices=False)
    chi2_new = min(mps.chi_max, len(S2))
    if len(S2) > chi2_new:
        mps.total_truncation_error += float(to_numpy(xp.sum(S2[chi2_new:] ** 2)))
    U2 = U2[:, :chi2_new]
    S2 = S2[:chi2_new]
    Vh2 = Vh2[:chi2_new, :]

    mps.tensors[base + 1] = (U2 * S2[xp.newaxis, :]).reshape(chi1_new, 2, chi2_new)
    mps.tensors[base + 2] = Vh2.reshape(chi2_new, 2, chi_r)

    # Undo swaps in reverse order
    for swap_pos in reversed(swap_log_2):
        lo = swap_pos
        mps.apply_two_adjacent(_SWAP_mat, lo, lo + 1)

    for swap_pos in reversed(swap_log_1):
        lo = swap_pos
        mps.apply_two_adjacent(_SWAP_mat, lo, lo + 1)
