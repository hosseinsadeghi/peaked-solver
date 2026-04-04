"""
Quantum Circuit Representation
===============================

Provides :class:`QuantumCircuit` (an ordered list of :class:`Gate` objects)
together with a standard gate library and a brute-force statevector
simulator for validation on small instances (<= 25 qubits).

Gate matrices follow the standard textbook conventions (column-major
ordering of computational-basis states).

OpenQASM 2.0 Parsing
---------------------
A lightweight ``from_openqasm`` class-method handles the subset of
OpenQASM 2.0 used by our test circuits:  ``qreg``, ``creg`` (ignored),
``h``, ``x``, ``y``, ``z``, ``s``, ``t``, ``cx``, ``cz``, ``ccx``,
``rx``, ``ry``, ``rz``, ``measure`` (ignored), and ``barrier`` (ignored).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Standard gate matrices
# ---------------------------------------------------------------------------

# Pauli gates
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gate  S = diag(1, i)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)

# T gate  T = diag(1, e^{i pi/4})
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Two-qubit gates -- 4x4 in computational basis {|00>, |01>, |10>, |11>}
_CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

_CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1],
], dtype=complex)

# Toffoli (CCX) -- 8x8 matrix
_TOFFOLI = np.eye(8, dtype=complex)
_TOFFOLI[6, 6] = 0
_TOFFOLI[7, 7] = 0
_TOFFOLI[6, 7] = 1
_TOFFOLI[7, 6] = 1


_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2

_ISWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """U3 gate: the most general single-qubit rotation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
    ], dtype=complex)


def _rzz(theta: float) -> np.ndarray:
    """RZZ gate: exp(-i * theta/2 * ZZ)."""
    e_neg = np.exp(-1j * theta / 2)
    e_pos = np.exp(1j * theta / 2)
    return np.diag([e_neg, e_pos, e_pos, e_neg]).astype(complex)


def _rx(theta: float) -> np.ndarray:
    """Rotation about X axis: Rx(theta) = exp(-i theta X / 2).

    .. math::
        R_x(\\theta) = \\begin{pmatrix}
            \\cos(\\theta/2) & -i\\sin(\\theta/2) \\\\
            -i\\sin(\\theta/2) & \\cos(\\theta/2)
        \\end{pmatrix}
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    """Rotation about Y axis: Ry(theta) = exp(-i theta Y / 2).

    .. math::
        R_y(\\theta) = \\begin{pmatrix}
            \\cos(\\theta/2) & -\\sin(\\theta/2) \\\\
            \\sin(\\theta/2) & \\cos(\\theta/2)
        \\end{pmatrix}
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    """Rotation about Z axis: Rz(theta) = exp(-i theta Z / 2).

    .. math::
        R_z(\\theta) = \\begin{pmatrix}
            e^{-i\\theta/2} & 0 \\\\
            0 & e^{i\\theta/2}
        \\end{pmatrix}
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex)


# Registry: gate_name -> (n_qubits, matrix_or_callable)
# For parameterised gates the value is a callable(theta) -> matrix.
GATE_LIBRARY: dict[str, tuple[int, object]] = {
    "i":       (1, _I),
    "id":      (1, _I),
    "x":       (1, _X),
    "y":       (1, _Y),
    "z":       (1, _Z),
    "h":       (1, _H),
    "s":       (1, _S),
    "t":       (1, _T),
    "sdg":     (1, _S.conj().T),
    "tdg":     (1, _T.conj().T),
    "cx":      (2, _CNOT),
    "cnot":    (2, _CNOT),
    "cz":      (2, _CZ),
    "ccx":     (3, _TOFFOLI),
    "toffoli": (3, _TOFFOLI),
    "rx":      (1, _rx),
    "ry":      (1, _ry),
    "rz":      (1, _rz),
    "sx":      (1, _SX),
    "u3":      (1, _u3),
    "u":       (1, _u3),
    "iswap":   (2, _ISWAP),
    "rzz":     (2, _rzz),
}


# ---------------------------------------------------------------------------
# Gate dataclass
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    """A single quantum gate applied to specific qubits.

    Attributes
    ----------
    name : str
        Lower-case canonical name, e.g. ``'h'``, ``'cx'``, ``'rz'``.
    qubits : tuple[int, ...]
        Qubit indices this gate acts on (control qubits first for
        multi-qubit gates).
    matrix : np.ndarray
        The unitary matrix (2x2, 4x4, or 8x8).
    params : tuple[float, ...]
        Rotation angles or other continuous parameters (empty for
        non-parameterised gates).
    """

    name: str
    qubits: tuple[int, ...]
    matrix: np.ndarray
    params: tuple[float, ...] = ()

    def n_qubits(self) -> int:
        """Number of qubits this gate acts on."""
        return len(self.qubits)

    def __repr__(self) -> str:
        pstr = f", params={self.params}" if self.params else ""
        return f"Gate({self.name}, qubits={self.qubits}{pstr})"


# ---------------------------------------------------------------------------
# QuantumCircuit
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """Ordered sequence of quantum gates on a fixed-size qubit register.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the register.

    Examples
    --------
    >>> qc = QuantumCircuit(2)
    >>> qc.add_gate('h', (0,))
    >>> qc.add_gate('cx', (0, 1))
    >>> print(qc.gate_count())
    2
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.gates: list[Gate] = []

    # ----- gate construction ------------------------------------------------

    def add_gate(
        self,
        name: str,
        qubits: Sequence[int] | int,
        params: Optional[Sequence[float]] = None,
    ) -> "QuantumCircuit":
        """Append a gate to the circuit.

        Parameters
        ----------
        name : str
            Gate name (looked up in GATE_LIBRARY).
        qubits : int or sequence of int
            Target qubit indices.
        params : sequence of float, optional
            Rotation parameters for parameterised gates (rx, ry, rz).

        Returns
        -------
        self
            For method chaining.
        """
        name_lower = name.lower()
        if name_lower not in GATE_LIBRARY:
            raise ValueError(f"Unknown gate '{name}'. Available: {list(GATE_LIBRARY)}")

        if isinstance(qubits, int):
            qubits = (qubits,)
        else:
            qubits = tuple(qubits)

        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise IndexError(
                    f"Qubit index {q} out of range for {self.n_qubits}-qubit circuit"
                )

        expected_nq, mat_or_fn = GATE_LIBRARY[name_lower]
        if len(qubits) != expected_nq:
            raise ValueError(
                f"Gate '{name}' expects {expected_nq} qubit(s), got {len(qubits)}"
            )

        if callable(mat_or_fn):
            # Parameterised gate -- must supply params
            if params is None or len(params) == 0:
                raise ValueError(f"Gate '{name}' requires parameters (e.g. angle)")
            matrix = mat_or_fn(*params)
            gate_params = tuple(params)
        else:
            matrix = mat_or_fn
            gate_params = ()

        self.gates.append(Gate(
            name=name_lower,
            qubits=qubits,
            matrix=np.array(matrix, dtype=complex),
            params=gate_params,
        ))
        return self

    # ----- export ------------------------------------------------------------

    def to_qasm(self) -> str:
        """Export the circuit as an OpenQASM 2.0 string."""
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.n_qubits}];",
        ]
        for gate in self.gates:
            qubits_str = ", ".join(f"q[{q}]" for q in gate.qubits)
            if gate.params:
                params_str = ", ".join(str(p) for p in gate.params)
                lines.append(f"{gate.name}({params_str}) {qubits_str};")
            else:
                lines.append(f"{gate.name} {qubits_str};")
        return "\n".join(lines)

    # ----- circuit analysis helpers -----------------------------------------

    def depth(self) -> int:
        """Circuit depth: length of the critical path.

        We assign each gate to the earliest time-step where all its
        qubits are free, then return the maximum time-step + 1.
        """
        if not self.gates:
            return 0
        # next_free[q] = earliest time step qubit q is available
        next_free = [0] * self.n_qubits
        max_depth = 0
        for gate in self.gates:
            t = max(next_free[q] for q in gate.qubits)
            for q in gate.qubits:
                next_free[q] = t + 1
            max_depth = max(max_depth, t + 1)
        return max_depth

    def gate_count(self) -> int:
        """Total number of gates in the circuit."""
        return len(self.gates)

    def gate_count_by_name(self) -> dict[str, int]:
        """Gate count grouped by name."""
        counts: dict[str, int] = {}
        for g in self.gates:
            counts[g.name] = counts.get(g.name, 0) + 1
        return counts

    def layers(self) -> list[list[Gate]]:
        """Partition gates into layers of non-overlapping gates.

        A layer is a maximal set of gates whose qubit sets are disjoint,
        processed in circuit order.  This is the natural structure for
        time-step-based simulation.
        """
        layers: list[list[Gate]] = []
        # next_free[q] = index of the layer where qubit q is next available
        next_free = [0] * self.n_qubits

        for gate in self.gates:
            t = max(next_free[q] for q in gate.qubits)
            # Ensure we have enough layers
            while len(layers) <= t:
                layers.append([])
            layers[t].append(gate)
            for q in gate.qubits:
                next_free[q] = t + 1

        return layers

    # ----- full unitary (brute force, small circuits only) ------------------

    def to_unitary(self) -> np.ndarray:
        """Compute the full 2^n x 2^n unitary by dense matrix multiplication.

        .. warning:: Exponential in *n_qubits*.  Only practical for <= 25 qubits.

        The unitary is built by embedding each gate into the full
        Hilbert space via Kronecker products and multiplying left-to-right.
        """
        n = self.n_qubits
        dim = 1 << n
        U = np.eye(dim, dtype=complex)

        for gate in self.gates:
            U_gate = _embed_gate(gate.matrix, gate.qubits, n)
            # U_total = U_gate @ U_total  (apply gate after everything so far)
            U = U_gate @ U

        return U

    # ----- OpenQASM 2.0 parser ---------------------------------------------

    @classmethod
    def from_openqasm(cls, qasm_str: str) -> "QuantumCircuit":
        """Parse a minimal subset of OpenQASM 2.0 into a QuantumCircuit.

        Supported constructs:

        * ``OPENQASM 2.0;`` header (checked but not required)
        * ``include "qelib1.inc";`` (ignored)
        * ``qreg name[size];``
        * ``creg name[size];`` (ignored)
        * ``barrier ...;`` (ignored)
        * ``measure ...;`` (ignored)
        * Standard gates: h, x, y, z, s, t, sdg, tdg, cx, cz, ccx,
          rx(angle), ry(angle), rz(angle)

        Parameters
        ----------
        qasm_str : str
            OpenQASM 2.0 source code.

        Returns
        -------
        QuantumCircuit
        """
        lines = qasm_str.strip().split(";")
        qreg_map: dict[str, tuple[int, int]] = {}  # name -> (start_index, size)
        total_qubits = 0

        # First pass: collect qubit registers
        for raw_line in lines:
            line = raw_line.strip()
            m = re.match(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", line)
            if m:
                name, size = m.group(1), int(m.group(2))
                qreg_map[name] = (total_qubits, size)
                total_qubits += size

        if total_qubits == 0:
            raise ValueError("No qreg declarations found in QASM string")

        circuit = cls(total_qubits)

        def _resolve_qubit(token: str) -> int:
            """Resolve 'q[3]' or 'q' (if single qubit reg) to an integer index."""
            m2 = re.match(r"(\w+)\s*\[\s*(\d+)\s*\]", token.strip())
            if m2:
                reg_name, idx = m2.group(1), int(m2.group(2))
                if reg_name not in qreg_map:
                    raise ValueError(f"Unknown qreg '{reg_name}'")
                start, size = qreg_map[reg_name]
                if idx >= size:
                    raise IndexError(f"Index {idx} out of range for qreg '{reg_name}' (size {size})")
                return start + idx
            # Bare register name -- must be size 1
            token = token.strip()
            if token in qreg_map:
                start, size = qreg_map[token]
                if size != 1:
                    raise ValueError(f"Bare register '{token}' has size {size} > 1; index required")
                return start
            raise ValueError(f"Cannot resolve qubit token '{token}'")

        # Gate pattern: optional params in parens, then comma-separated qubits
        gate_re = re.compile(
            r"(\w+)"                          # gate name
            r"(?:\s*\(\s*([^)]*)\s*\))?"      # optional (params)
            r"\s+(.+)"                         # qubit arguments
        )

        skip = {"openqasm", "include", "creg", "qreg", "barrier", "measure", "gate", "//", ""}

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            # Skip custom gate definitions: "gate name params { body }"
            # These may span across semicolons; strip any leading "}" from
            # lines that follow a gate body.
            if line.startswith("}"):
                line = line[1:].strip()
                if not line:
                    continue
            first_word = line.split()[0].lower() if line.split() else ""
            if first_word in skip or line.startswith("//"):
                continue

            # Try to match a gate line
            m = gate_re.match(line)
            if not m:
                continue

            gate_name = m.group(1).lower()
            param_str = m.group(2)  # may be None
            qubit_tokens = [t.strip() for t in m.group(3).split(",")]

            # Skip non-gate keywords that slipped through
            if gate_name in skip:
                continue

            params: list[float] = []
            if param_str is not None:
                for p in param_str.split(","):
                    p = p.strip()
                    # Handle simple expressions: pi, pi/4, 2*pi, -pi/8, etc.
                    params.append(_eval_param(p))

            qubits = [_resolve_qubit(tok) for tok in qubit_tokens]
            circuit.add_gate(gate_name, qubits, params if params else None)

        return circuit


def _eval_param(expr: str) -> float:
    """Evaluate a simple OpenQASM parameter expression.

    Handles: numeric literals, ``pi``, ``+``, ``-``, ``*``, ``/``.
    """
    expr = expr.strip().replace("pi", str(np.pi))
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # safe-ish: only numbers & ops
    except Exception:
        raise ValueError(f"Cannot evaluate parameter expression: '{expr}'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_gate(matrix: np.ndarray, qubits: tuple[int, ...], n_qubits: int) -> np.ndarray:
    """Embed a k-qubit gate matrix into the full 2^n Hilbert space.

    For a gate acting on qubits ``(q0, q1, ...)``, we construct the
    2^n x 2^n matrix that acts as *matrix* on those qubits and as the
    identity on all others.

    The approach:  for every pair (row, col) of computational-basis
    indices, extract the sub-indices on the gate qubits, look up the
    matrix element, and require agreement on all non-gate qubits.
    """
    n = n_qubits
    dim = 1 << n
    k = len(qubits)

    # For efficiency we build the matrix via index manipulation rather
    # than repeated Kronecker products (which would need qubit reordering).
    full = np.zeros((dim, dim), dtype=complex)

    for row in range(dim):
        for col in range(dim):
            # Check that non-gate qubits match
            match = True
            for q in range(n):
                if q not in qubits:
                    # bit q of row must equal bit q of col
                    if ((row >> (n - 1 - q)) & 1) != ((col >> (n - 1 - q)) & 1):
                        match = False
                        break
            if not match:
                continue

            # Extract the gate-qubit sub-indices
            sub_row = 0
            sub_col = 0
            for ki, q in enumerate(qubits):
                sub_row |= ((row >> (n - 1 - q)) & 1) << (k - 1 - ki)
                sub_col |= ((col >> (n - 1 - q)) & 1) << (k - 1 - ki)

            full[row, col] = matrix[sub_row, sub_col]

    return full


# ---------------------------------------------------------------------------
# Brute-force statevector simulator
# ---------------------------------------------------------------------------

def brute_force_statevector(circuit: QuantumCircuit) -> np.ndarray:
    """Compute the exact output statevector by applying gates to |0...0>.

    This is the gold-standard reference for testing heuristics.

    The statevector is a length-2^n complex vector in computational-basis
    order (big-endian: qubit 0 is the most significant bit).

    .. warning:: Exponential memory.  Use only for n_qubits <= 25.

    Parameters
    ----------
    circuit : QuantumCircuit

    Returns
    -------
    np.ndarray
        Complex statevector of length 2^n_qubits.
    """
    n = circuit.n_qubits
    if n > 25:
        raise ValueError(
            f"brute_force_statevector is limited to 25 qubits, got {n}"
        )
    dim = 1 << n

    # Start in |0...0>
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    # Apply each gate via its full-space embedding
    for gate in circuit.gates:
        U_gate = _embed_gate(gate.matrix, gate.qubits, n)
        state = U_gate @ state

    return state
