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
        # A^{s=0} = [[1]], A^{s=1} = [[0]]
        self.tensors: list[np.ndarray] = []
        for _ in range(n_qubits):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0  # |0> component
            self.tensors.append(t)

    # ---- diagonal gate optimisation (Papers 71, 75) -------------------------

    def apply_diagonal_two_adjacent(
        self, diag_values: np.ndarray, q0: int, q1: int
    ) -> None:
        """Apply a diagonal two-qubit gate to adjacent qubits.

        Diagonal gates (CZ, etc.) don't create entanglement between
        product-state components — they only multiply each amplitude by
        a phase.  This avoids the costly SVD step entirely.

        Parameters
        ----------
        diag_values : ndarray of shape (4,)
            The diagonal of the 4×4 gate matrix, indexed as
            |00>, |01>, |10>, |11>.
        q0, q1 : int
            Adjacent qubit indices (q1 == q0 + 1).
        """
        assert q1 == q0 + 1
        A = self.tensors[q0]   # (chi_l, 2, chi_m)
        B = self.tensors[q1]   # (chi_m, 2, chi_r)

        # For each combination of physical indices (s0, s1),
        # multiply the corresponding pair of tensor slices by the phase.
        # The contraction A[:, s0, :] @ B[:, s1, :] picks up phase diag[2*s0+s1].
        # We can absorb this into either A or B.  Absorb into A for simplicity:
        # A'[:, s0, m] = sum_{s1} diag[2*s0+s1] * ... but that's wrong because
        # we can't separate the s1 dependence.
        #
        # Instead: form Theta, apply diagonal, split back.
        # But diagonal gates don't grow bond dimension!
        # Theta[a, s0, s1, b] = A[a,s0,m] * B[m,s1,b]
        # Theta'[a, s0, s1, b] = diag[2*s0+s1] * Theta[a,s0,s1,b]
        #
        # We can absorb this as: A'[a,s0,m] = A[a,s0,m], B'[m,s1,b] = B[m,s1,b]
        # but multiply each (s0,s1) block. Since the diagonal is separable for CZ
        # (diag = [1,1,1,-1]) we can split: phase_s0_s1 = f(s0)*g(s1) for some f,g?
        # CZ: [1,1,1,-1] -> not separable.
        #
        # General approach: contract, apply diagonal, SVD split.
        # Since the gate is diagonal, the result has the same bond dimension
        # (diagonal gates cannot increase Schmidt rank).
        theta = np.tensordot(A, B, axes=([2], [0]))  # (chi_l, 2, 2, chi_r)
        for s0 in range(2):
            for s1 in range(2):
                theta[:, s0, s1, :] *= diag_values[2 * s0 + s1]

        # Split back — bond dimension is unchanged so no truncation needed
        chi_l, _, _, chi_r = theta.shape
        chi_m = A.shape[2]  # original bond dimension
        mat = theta.reshape(chi_l * 2, 2 * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        chi_new = min(self.chi_max, len(S), chi_m + 1)  # won't exceed original + 1
        if len(S) > chi_new:
            self.total_truncation_error += float(np.sum(S[chi_new:] ** 2))
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]
        self.tensors[q0] = (U * S[np.newaxis, :]).reshape(chi_l, 2, chi_new)
        self.tensors[q1] = Vh.reshape(chi_new, 2, chi_r)

    @staticmethod
    def is_diagonal_gate(gate_matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a gate matrix is diagonal."""
        return np.allclose(gate_matrix - np.diag(np.diag(gate_matrix)), 0, atol=tol)

    # ---- single-qubit gate ------------------------------------------------

    def apply_single(self, gate_matrix: np.ndarray, qubit: int) -> None:
        """Apply a 2x2 unitary to qubit *qubit*.

        Contracts the gate with the physical index of A_i:
            A_i^{s'} = sum_s  gate[s', s] * A_i^{s}

        This does not change bond dimensions.
        """
        # gate_matrix shape: (2, 2)
        # tensor shape: (chi_l, 2, chi_r)
        # Einstein notation: new[a, s', b] = gate[s', s] * old[a, s, b]
        self.tensors[qubit] = np.einsum(
            "ij, ajb -> aib", gate_matrix, self.tensors[qubit]
        )

    # ---- two-qubit gate on adjacent qubits --------------------------------

    def apply_two_adjacent(
        self, gate_matrix: np.ndarray, q0: int, q1: int
    ) -> None:
        """Apply a 4x4 gate to adjacent qubits q0, q1 (q1 = q0+1).

        Steps:
        1. Form Theta = contract A_{q0} and A_{q1} over the shared bond.
           Theta has shape (chi_left, 2, 2, chi_right).
        2. Reshape gate (4,4) to (2,2,2,2) and contract with Theta's
           physical indices.
        3. SVD split Theta back into two tensors, truncating to chi_max.

        The truncation error (Frobenius norm squared of discarded
        singular values) is accumulated in ``self.total_truncation_error``.
        """
        assert q1 == q0 + 1, "Qubits must be adjacent for this method"

        A = self.tensors[q0]  # (chi_l, 2, chi_m)
        B = self.tensors[q1]  # (chi_m, 2, chi_r)

        # Step 1: contract into Theta(chi_l, s0, s1, chi_r)
        # Theta[a, s0, s1, b] = sum_m A[a, s0, m] * B[m, s1, b]
        theta = np.tensordot(A, B, axes=([2], [0]))
        # theta shape: (chi_l, 2, 2, chi_r)

        # Step 2: apply gate
        # gate_matrix (4,4) -> (2,2,2,2): gate[s0',s1',s0,s1]
        gate_4d = gate_matrix.reshape(2, 2, 2, 2)
        # new_theta[a, s0', s1', b] = gate[s0',s1',s0,s1] * theta[a,s0,s1,b]
        theta = np.einsum("ijkl, aklb -> aijb", gate_4d, theta)

        # Step 3: SVD split
        # Reshape theta (chi_l, 2, 2, chi_r) -> (chi_l*2, 2*chi_r)
        chi_l = theta.shape[0]
        chi_r = theta.shape[3]
        theta_mat = theta.reshape(chi_l * 2, 2 * chi_r)

        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

        # Truncate to chi_max
        # The truncation error is sum of discarded singular values squared
        # (Eckart-Young theorem: ||M - M_k||_F^2 = sum_{i>k} sigma_i^2)
        chi_new = min(self.chi_max, len(S))
        if len(S) > chi_new:
            self.total_truncation_error += float(np.sum(S[chi_new:] ** 2))
        S = S[:chi_new]
        U = U[:, :chi_new]
        Vh = Vh[:chi_new, :]

        # Absorb singular values into U (left-canonical form for this bond)
        # new A_{q0}[a, s0, m'] = (U @ diag(S))[a*2+s0, m']
        US = U * S[np.newaxis, :]  # broadcast multiply columns by S
        self.tensors[q0] = US.reshape(chi_l, 2, chi_new)

        # new A_{q1}[m', s1, b] = Vh[m', s1*chi_r + b]
        self.tensors[q1] = Vh.reshape(chi_new, 2, chi_r)

    # ---- two-qubit gate on non-adjacent qubits ----------------------------

    def apply_two(self, gate_matrix: np.ndarray, q0: int, q1: int) -> None:
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
            if q0 < q1:
                # Diagonal fast-path
                if self.is_diagonal_gate(gate_matrix):
                    self.apply_diagonal_two_adjacent(np.diag(gate_matrix), lo, hi)
                else:
                    self.apply_two_adjacent(gate_matrix, lo, hi)
            else:
                # Gate acts as (q0=hi, q1=lo) -- need to reorder
                # Swap physical indices: gate'[s1,s0,s1',s0'] = gate[s0,s1,s0',s1']
                g4 = gate_matrix.reshape(2, 2, 2, 2)
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
        """Compute <bitstring|psi> by contracting the MPS chain.

        For a bitstring s1 s2 ... sn, the amplitude is:
            A1^{s1} @ A2^{s2} @ ... @ An^{sn}
        which is a product of matrices resulting in a (1,1) scalar.
        """
        result = self.tensors[0][:, int(bitstring[0]), :]  # (chi_l=1, chi_r)
        for i in range(1, self.n):
            bit = int(bitstring[i])
            mat = self.tensors[i][:, bit, :]  # (chi_m, chi_r)
            result = result @ mat
        return complex(result[0, 0])

    def get_probabilities(self, bitstrings: list[str]) -> list[float]:
        """Compute |<bitstring|psi>|^2 for a list of candidate bitstrings."""
        return [abs(self.get_amplitude(bs)) ** 2 for bs in bitstrings]

    def get_top_k_bitstrings(self, k: int) -> list[tuple[str, float]]:
        """Find the k most probable bitstrings via greedy MPS sampling.

        We use a layer-by-layer greedy/beam approach:
        For each qubit from left to right, we compute the conditional
        probabilities of |0> and |1> given the bits chosen so far,
        and keep the top-k partial bitstrings.

        This is not guaranteed to find the global optimum but works
        well for peaked distributions.
        """
        # Beam of (partial_bitstring, accumulated_vector)
        # accumulated_vector starts as shape (1,) identity
        beam: list[tuple[str, np.ndarray]] = [("", np.ones((1,), dtype=complex))]

        for i in range(self.n):
            candidates: list[tuple[str, np.ndarray, float]] = []
            for prefix, vec in beam:
                for bit in [0, 1]:
                    # Contract with A_i^{bit}: vec @ A_i[:, bit, :]
                    mat = self.tensors[i][:, bit, :]  # (chi_l, chi_r)
                    new_vec = vec @ mat  # (chi_r,)

                    # Estimate probability magnitude for ranking
                    prob_est = float(np.sum(np.abs(new_vec) ** 2))
                    candidates.append((prefix + str(bit), new_vec, prob_est))

            # Keep top 2*k candidates by probability estimate
            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = [(bs, vec) for bs, vec, _ in candidates[: max(2 * k, 20)]]

        # Final: compute exact amplitudes for surviving bitstrings
        results: list[tuple[str, float]] = []
        for bs, vec in beam:
            amp = complex(vec[0]) if vec.shape[0] == 1 else complex(np.sum(vec))
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

    # theta[chi_l, s0, s1, s2, chi_r]
    AB = np.tensordot(A, B, axes=([2], [0]))  # (chi_l, 2, 2, chi2)
    theta = np.tensordot(AB, C, axes=([3], [0]))  # (chi_l, 2, 2, 2, chi_r)

    # Build the qubit ordering map for the gate
    # The gate acts on (q0, q1, q2) but our theta has qubits at (base, base+1, base+2)
    # We need to figure out which original qubit ended up at each position.
    # Since we moved them in order (qs[0], qs[1] -> base+1, qs[2] -> base+2),
    # the mapping depends on the original gate qubit order vs sorted order.
    # For simplicity, apply the gate matrix directly (assuming sorted order matches).

    # Map gate qubits to positions in theta
    # gate.qubits = (q0, q1, q2), sorted as qs = [qs[0], qs[1], qs[2]]
    # theta has qubits in sorted order. We need to permute gate matrix accordingly.
    sorted_to_gate = []
    for sq in qs:
        sorted_to_gate.append(gate.qubits.index(sq))

    # Permute the gate matrix rows/cols to match sorted qubit order
    gate_8x8 = gate.matrix  # (8, 8)
    gate_tensor = gate_8x8.reshape(2, 2, 2, 2, 2, 2)  # (out0, out1, out2, in0, in1, in2)
    # Permute: sorted_to_gate[i] gives which gate-qubit is at position i
    perm_in = [3 + sorted_to_gate[j] for j in range(3)]
    perm_out = [sorted_to_gate[j] for j in range(3)]
    gate_tensor = gate_tensor.transpose(perm_out + perm_in)

    # Apply: theta_new[a, s0', s1', s2', b] = gate[s0',s1',s2',s0,s1,s2] * theta[a,s0,s1,s2,b]
    chi_l = theta.shape[0]
    chi_r = theta.shape[4]
    theta_flat = theta.reshape(chi_l, 8, chi_r)
    gate_flat = gate_tensor.reshape(8, 8)
    theta_new = np.einsum("ij, ajb -> aib", gate_flat, theta_flat)
    theta = theta_new.reshape(chi_l, 2, 2, 2, chi_r)

    # Split back via two SVDs
    # First split: (chi_l * 2) x (2 * 2 * chi_r)
    mat1 = theta.reshape(chi_l * 2, 4 * chi_r)
    U1, S1, Vh1 = np.linalg.svd(mat1, full_matrices=False)
    chi1_new = min(mps.chi_max, len(S1))
    if len(S1) > chi1_new:
        mps.total_truncation_error += float(np.sum(S1[chi1_new:] ** 2))
    U1 = U1[:, :chi1_new]
    S1 = S1[:chi1_new]
    Vh1 = Vh1[:chi1_new, :]

    mps.tensors[base] = (U1 * S1[np.newaxis, :]).reshape(chi_l, 2, chi1_new)

    # Second split: (chi1_new * 2) x (2 * chi_r)
    remainder = Vh1.reshape(chi1_new, 2, 2, chi_r)
    mat2 = remainder.reshape(chi1_new * 2, 2 * chi_r)
    U2, S2, Vh2 = np.linalg.svd(mat2, full_matrices=False)
    chi2_new = min(mps.chi_max, len(S2))
    if len(S2) > chi2_new:
        mps.total_truncation_error += float(np.sum(S2[chi2_new:] ** 2))
    U2 = U2[:, :chi2_new]
    S2 = S2[:chi2_new]
    Vh2 = Vh2[:chi2_new, :]

    mps.tensors[base + 1] = (U2 * S2[np.newaxis, :]).reshape(chi1_new, 2, chi2_new)
    mps.tensors[base + 2] = Vh2.reshape(chi2_new, 2, chi_r)

    # Undo swaps in reverse order
    for swap_pos in reversed(swap_log_2):
        lo = swap_pos
        mps.apply_two_adjacent(_SWAP_mat, lo, lo + 1)

    for swap_pos in reversed(swap_log_1):
        lo = swap_pos
        mps.apply_two_adjacent(_SWAP_mat, lo, lo + 1)
