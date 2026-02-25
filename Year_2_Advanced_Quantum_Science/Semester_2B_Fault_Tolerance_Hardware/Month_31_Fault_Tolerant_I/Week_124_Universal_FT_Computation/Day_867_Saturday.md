# Day 867: Computational Lab - Universal FT Computation

## Week 124: Universal Fault-Tolerant Computation | Month 31: Fault-Tolerant QC I

---

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Solovay-Kitaev implementation |
| Afternoon | 2.5 hrs | T-count analysis tools |
| Evening | 2.0 hrs | Integration and visualization |

---

### Learning Objectives

By the end of today, you will be able to:

1. **Implement a complete Solovay-Kitaev algorithm** for SU(2)
2. **Build T-count analysis tools** for quantum circuits
3. **Visualize gate coverage** and synthesis quality
4. **Compare synthesis algorithms** quantitatively
5. **Profile resource requirements** for sample algorithms
6. **Create reusable tools** for fault-tolerant compilation analysis

---

### Lab Overview

Today's lab synthesizes all concepts from Week 124 into working code. We will build:

1. **Complete Solovay-Kitaev Implementation** - Full recursive decomposition
2. **T-Count Analyzer** - Analyze circuits for T-gate requirements
3. **Resource Profiler** - Estimate physical resources from logical circuits
4. **Visualization Suite** - Interactive exploration of synthesis quality
5. **Algorithm Case Studies** - QFT, Grover, and simulation examples

---

### Lab 1: Complete Solovay-Kitaev Implementation

```python
"""
Lab 1: Complete Solovay-Kitaev Algorithm Implementation
Full implementation with optimizations and analysis
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

# Pauli and Clifford gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Conjugate transposes
S_dag = np.conj(S).T
T_dag = np.conj(T).T

@dataclass
class GateSequence:
    """Represents a sequence of gates"""
    gates: List[str]
    matrix: np.ndarray
    t_count: int

    def __len__(self):
        return len(self.gates)

    def __str__(self):
        if not self.gates:
            return "I"
        return " ".join(self.gates)

class CliffordTGroup:
    """Manages the Clifford+T gate set"""

    GATE_MATRICES = {
        'I': I, 'X': X, 'Y': Y, 'Z': Z,
        'H': H, 'S': S, 'S†': S_dag,
        'T': T, 'T†': T_dag,
    }

    GATE_T_COUNTS = {
        'I': 0, 'X': 0, 'Y': 0, 'Z': 0,
        'H': 0, 'S': 0, 'S†': 0,
        'T': 1, 'T†': 1,
    }

    @classmethod
    def get_matrix(cls, gate_name: str) -> np.ndarray:
        return cls.GATE_MATRICES[gate_name]

    @classmethod
    def get_t_count(cls, gate_name: str) -> int:
        return cls.GATE_T_COUNTS[gate_name]

    @classmethod
    def compose(cls, gates: List[str]) -> np.ndarray:
        """Compose a list of gates into a single matrix"""
        result = I.copy()
        for gate in gates:
            result = cls.get_matrix(gate) @ result
        return result


def matrix_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Compute operator norm distance ||A - B||"""
    return np.linalg.norm(A - B, ord=2)


def normalize_su2(U: np.ndarray) -> np.ndarray:
    """Project matrix to SU(2)"""
    det = np.linalg.det(U)
    if np.abs(det) < 1e-10:
        return I
    return U / np.sqrt(det)


def matrix_to_rotation(U: np.ndarray) -> Tuple[float, np.ndarray]:
    """Convert SU(2) matrix to rotation angle and axis"""
    U = normalize_su2(U)
    trace = np.trace(U)
    cos_half = np.clip(np.real(trace) / 2, -1, 1)
    theta = 2 * np.arccos(cos_half)

    if np.abs(np.sin(theta/2)) < 1e-10:
        return 0, np.array([0, 0, 1])

    sin_half = np.sin(theta/2)
    nx = np.imag(U[0, 1] + U[1, 0]) / (2 * sin_half)
    ny = np.real(U[0, 1] - U[1, 0]) / (2 * sin_half)
    nz = np.imag(U[0, 0] - U[1, 1]) / (2 * sin_half)

    axis = np.array([nx, ny, nz])
    norm = np.linalg.norm(axis)
    if norm > 1e-10:
        axis = axis / norm

    return theta, axis


def rotation_to_matrix(theta: float, axis: np.ndarray) -> np.ndarray:
    """Convert rotation angle and axis to SU(2) matrix"""
    if np.linalg.norm(axis) < 1e-10:
        return I

    axis = axis / np.linalg.norm(axis)
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    nx, ny, nz = axis

    return np.array([
        [c - 1j*s*nz, -s*ny - 1j*s*nx],
        [s*ny - 1j*s*nx, c + 1j*s*nz]
    ], dtype=complex)


class SolovayKitaev:
    """
    Complete Solovay-Kitaev algorithm implementation.
    """

    def __init__(self, base_depth: int = 4):
        """
        Initialize SK algorithm.

        Args:
            base_depth: Depth for building base approximation table
        """
        self.base_depth = base_depth
        self.epsilon_net: Dict[tuple, GateSequence] = {}
        self.stats = {'cache_hits': 0, 'decompositions': 0}
        self._build_epsilon_net()

    def _matrix_key(self, M: np.ndarray, precision: int = 5) -> tuple:
        """Convert matrix to hashable key with phase normalization"""
        # Normalize global phase
        if np.abs(M[0, 0]) > 0.01:
            phase = np.exp(-1j * np.angle(M[0, 0]))
        else:
            phase = 1
        M_norm = M * phase

        # Create key from real and imaginary parts
        flat = M_norm.flatten()
        key = tuple(np.round(np.concatenate([np.real(flat), np.imag(flat)]), precision))
        return key

    def _build_epsilon_net(self):
        """Build lookup table of base approximations"""
        print(f"Building epsilon-net with depth {self.base_depth}...")

        # Base gates
        base_gates = ['H', 'S', 'S†', 'T', 'T†']

        # Start with identity
        self.epsilon_net[self._matrix_key(I)] = GateSequence([], I.copy(), 0)

        # Generate products up to base_depth
        current_sequences = [GateSequence([], I.copy(), 0)]

        for depth in range(1, self.base_depth + 1):
            new_sequences = []

            for seq in current_sequences:
                for gate in base_gates:
                    new_gates = [gate] + seq.gates
                    new_matrix = CliffordTGroup.get_matrix(gate) @ seq.matrix
                    new_t_count = seq.t_count + CliffordTGroup.get_t_count(gate)

                    key = self._matrix_key(new_matrix)
                    if key not in self.epsilon_net:
                        new_seq = GateSequence(new_gates, new_matrix, new_t_count)
                        self.epsilon_net[key] = new_seq
                        new_sequences.append(new_seq)

            current_sequences = new_sequences
            print(f"  Depth {depth}: {len(self.epsilon_net)} sequences")

        print(f"Epsilon-net complete: {len(self.epsilon_net)} unique unitaries")

    def _find_closest(self, U: np.ndarray) -> GateSequence:
        """Find closest sequence in epsilon-net"""
        best_seq = None
        best_dist = float('inf')

        for key, seq in self.epsilon_net.items():
            dist = matrix_distance(U, seq.matrix)
            if dist < best_dist:
                best_dist = dist
                best_seq = seq

        return best_seq

    def _gc_decompose(self, Delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Group commutator decomposition.
        Find V, W such that [V, W] ≈ Delta.
        """
        theta, axis = matrix_to_rotation(Delta)

        if np.abs(theta) < 1e-10:
            return I, I

        # For small rotations: [V, W] ≈ rotation by O(phi^2)
        # where V, W are rotations by phi
        phi = np.sqrt(np.abs(theta) / 2)

        # Choose perpendicular axes
        if np.abs(axis[2]) < 0.9:
            v_axis = np.cross(axis, [0, 0, 1])
        else:
            v_axis = np.cross(axis, [1, 0, 0])
        v_axis = v_axis / np.linalg.norm(v_axis)

        w_axis = np.cross(v_axis, axis)
        w_axis = w_axis / np.linalg.norm(w_axis)

        V = rotation_to_matrix(phi, v_axis)
        W = rotation_to_matrix(phi, w_axis)

        return V, W

    def _invert_sequence(self, seq: GateSequence) -> GateSequence:
        """Invert a gate sequence"""
        inverse_map = {
            'H': 'H', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
            'S': 'S†', 'S†': 'S', 'T': 'T†', 'T†': 'T'
        }

        inverted_gates = [inverse_map[g] for g in reversed(seq.gates)]
        inverted_matrix = np.conj(seq.matrix).T
        return GateSequence(inverted_gates, inverted_matrix, seq.t_count)

    def decompose(self, U: np.ndarray, depth: int = 3) -> GateSequence:
        """
        Solovay-Kitaev decomposition.

        Args:
            U: Target SU(2) matrix
            depth: Recursion depth

        Returns:
            GateSequence approximating U
        """
        self.stats['decompositions'] += 1
        U = normalize_su2(U)

        # Base case: lookup in epsilon-net
        if depth == 0:
            return self._find_closest(U)

        # Recursive case
        # Step 1: Get coarse approximation
        U_approx = self.decompose(U, depth - 1)

        # Step 2: Compute error
        Delta = U @ np.conj(U_approx.matrix).T

        # Check if already good enough
        if matrix_distance(Delta, I) < 1e-12:
            return U_approx

        # Step 3: GC decomposition
        V, W = self._gc_decompose(Delta)

        # Step 4: Recursively approximate V and W
        V_approx = self.decompose(V, depth - 1)
        W_approx = self.decompose(W, depth - 1)

        # Step 5: Compute inverses
        V_inv = self._invert_sequence(V_approx)
        W_inv = self._invert_sequence(W_approx)

        # Step 6: Combine: [V, W] * U_approx
        combined_gates = (V_approx.gates + W_approx.gates +
                          V_inv.gates + W_inv.gates + U_approx.gates)

        # Compute actual matrix
        combined_matrix = (V_approx.matrix @ W_approx.matrix @
                           V_inv.matrix @ W_inv.matrix @ U_approx.matrix)

        combined_t_count = (V_approx.t_count + W_approx.t_count +
                            V_inv.t_count + W_inv.t_count + U_approx.t_count)

        return GateSequence(combined_gates, combined_matrix, combined_t_count)

    def analyze_approximation(self, U: np.ndarray, max_depth: int = 5) -> Dict:
        """Analyze approximation quality at different depths"""
        results = []

        for depth in range(max_depth + 1):
            start = time.time()
            seq = self.decompose(U, depth)
            elapsed = time.time() - start

            error = matrix_distance(U, seq.matrix)
            results.append({
                'depth': depth,
                'gate_count': len(seq),
                't_count': seq.t_count,
                'error': error,
                'time_s': elapsed
            })

        return results


# Demonstration
print("="*70)
print("Solovay-Kitaev Algorithm - Complete Implementation")
print("="*70)

sk = SolovayKitaev(base_depth=4)

# Test targets
test_cases = [
    ("Rz(pi/5)", rotation_to_matrix(np.pi/5, [0, 0, 1])),
    ("Rx(0.123)", rotation_to_matrix(0.123, [1, 0, 0])),
    ("Ry(1.0)", rotation_to_matrix(1.0, [0, 1, 0])),
    ("R(0.5, [1,1,1])", rotation_to_matrix(0.5, [1, 1, 1])),
]

print("\n--- Approximation Analysis ---")
for name, target in test_cases:
    print(f"\nTarget: {name}")
    results = sk.analyze_approximation(target, max_depth=4)

    print(f"{'Depth':<8} {'Gates':<10} {'T-count':<10} {'Error':<15} {'Time':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['depth']:<8} {r['gate_count']:<10} {r['t_count']:<10} "
              f"{r['error']:<15.2e} {r['time_s']:<10.4f}s")

# Verify theoretical error scaling
print("\n--- Error Scaling Verification ---")
print("Theory: ε_n ≈ ε_0^(1.5^n)")

target = rotation_to_matrix(0.7, [1, 2, 3])
results = sk.analyze_approximation(target, max_depth=5)

errors = [r['error'] for r in results]
eps_0 = errors[0] if errors[0] > 0 else 0.1

print(f"\n{'Depth':<8} {'Actual Error':<15} {'Theoretical':<15} {'Ratio':<10}")
print("-" * 50)
for n, r in enumerate(results):
    theoretical = eps_0 ** (1.5 ** n) if n < 6 else 0
    ratio = r['error'] / theoretical if theoretical > 1e-15 else 0
    print(f"{n:<8} {r['error']:<15.2e} {theoretical:<15.2e} {ratio:<10.2f}")

print("\n" + "="*70)
```

---

### Lab 2: T-Count Analysis Tools

```python
"""
Lab 2: T-Count Analysis Tools
Analyze quantum circuits for T-gate requirements
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

@dataclass
class Gate:
    """Represents a quantum gate"""
    name: str
    qubits: Tuple[int, ...]
    params: Dict = field(default_factory=dict)

    @property
    def t_count(self) -> int:
        """Return T-count for this gate"""
        t_counts = {
            'I': 0, 'X': 0, 'Y': 0, 'Z': 0,
            'H': 0, 'S': 0, 'Sdg': 0,
            'T': 1, 'Tdg': 1,
            'CNOT': 0, 'CX': 0, 'CZ': 0,
            'SWAP': 0,
            'Toffoli': 7, 'CCX': 7,
            'CCZ': 7,
        }

        if self.name in t_counts:
            return t_counts[self.name]

        # Rotation gates: estimate based on precision
        if self.name.startswith('R'):
            precision = self.params.get('precision', 1e-8)
            return int(3 * np.log2(1/precision))

        return 0

    def __str__(self):
        return f"{self.name}({','.join(map(str, self.qubits))})"


class QuantumCircuit:
    """Simple quantum circuit representation for analysis"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[Gate] = []

    def add(self, gate: Gate):
        self.gates.append(gate)

    def h(self, q: int):
        self.add(Gate('H', (q,)))

    def s(self, q: int):
        self.add(Gate('S', (q,)))

    def t(self, q: int):
        self.add(Gate('T', (q,)))

    def cx(self, control: int, target: int):
        self.add(Gate('CX', (control, target)))

    def ccx(self, c1: int, c2: int, target: int):
        self.add(Gate('CCX', (c1, c2, target)))

    def rz(self, q: int, angle: float, precision: float = 1e-8):
        self.add(Gate('Rz', (q,), {'angle': angle, 'precision': precision}))


class TCountAnalyzer:
    """Analyze T-count and T-depth of circuits"""

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit

    def total_t_count(self) -> int:
        """Calculate total T-count"""
        return sum(g.t_count for g in self.circuit.gates)

    def t_depth(self) -> Tuple[int, List[List[int]]]:
        """
        Calculate T-depth and return T-layers.

        Returns:
            (t_depth, list of gate indices per layer)
        """
        # Track when each qubit becomes available
        qubit_time = {q: 0 for q in range(self.circuit.n_qubits)}
        t_layers = defaultdict(list)

        for i, gate in enumerate(self.circuit.gates):
            # Earliest time this gate can start
            start = max(qubit_time[q] for q in gate.qubits)

            # If T-gate, assign to a layer
            if gate.t_count > 0:
                t_layers[start].append(i)
                end = start + 1
            else:
                end = start  # Clifford gates are "instant"

            # Update qubit availability
            for q in gate.qubits:
                qubit_time[q] = end

        depth = len(t_layers)
        layers = [t_layers[t] for t in sorted(t_layers.keys())]

        return depth, layers

    def gate_breakdown(self) -> Dict[str, int]:
        """Count gates by type"""
        counts = defaultdict(int)
        for gate in self.circuit.gates:
            counts[gate.name] += 1
        return dict(counts)

    def t_parallelism(self) -> float:
        """Calculate average T-gates per T-layer"""
        depth, layers = self.t_depth()
        if depth == 0:
            return 0
        total_t = sum(self.circuit.gates[i].t_count for layer in layers for i in layer)
        return total_t / depth

    def analyze(self) -> Dict:
        """Complete analysis"""
        depth, layers = self.t_depth()

        return {
            'n_qubits': self.circuit.n_qubits,
            'total_gates': len(self.circuit.gates),
            't_count': self.total_t_count(),
            't_depth': depth,
            'gate_breakdown': self.gate_breakdown(),
            'max_parallel_t': max(len(l) for l in layers) if layers else 0,
            'avg_parallel_t': self.t_parallelism(),
            't_layers': layers,
        }


def build_qft_circuit(n: int, precision: float = 1e-8) -> QuantumCircuit:
    """Build Quantum Fourier Transform circuit"""
    circuit = QuantumCircuit(n)

    for i in range(n):
        circuit.h(i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            circuit.rz(j, angle, precision)
            circuit.cx(i, j)

    # Swap for bit reversal
    for i in range(n // 2):
        circuit.cx(i, n - 1 - i)
        circuit.cx(n - 1 - i, i)
        circuit.cx(i, n - 1 - i)

    return circuit


def build_grover_iteration(n: int, oracle_toffolis: int) -> QuantumCircuit:
    """Build one Grover iteration"""
    circuit = QuantumCircuit(n + 1)  # n qubits + 1 ancilla

    # Oracle (represented by Toffolis)
    for i in range(oracle_toffolis):
        circuit.ccx(i % n, (i + 1) % n, n)

    # Diffusion
    for i in range(n):
        circuit.h(i)
        circuit.s(i)

    # Multi-controlled Z (as Toffolis)
    for i in range(n - 2):
        circuit.ccx(i, i + 1, n)

    for i in range(n):
        circuit.s(i)
        circuit.h(i)

    return circuit


# Demonstrations
print("="*70)
print("T-Count Analysis Tools")
print("="*70)

# Analyze QFT circuits
print("\n--- QFT Circuit Analysis ---")
print(f"{'Qubits':<10} {'Gates':<10} {'T-count':<12} {'T-depth':<10} {'Parallelism':<12}")
print("-" * 55)

for n in [4, 6, 8, 10, 12]:
    qft = build_qft_circuit(n, precision=1e-8)
    analyzer = TCountAnalyzer(qft)
    result = analyzer.analyze()

    print(f"{n:<10} {result['total_gates']:<10} {result['t_count']:<12} "
          f"{result['t_depth']:<10} {result['avg_parallel_t']:<12.1f}")

# Analyze Grover circuits
print("\n--- Grover Iteration Analysis ---")
print(f"{'Qubits':<10} {'Oracle T':<10} {'T-count':<12} {'T-depth':<10}")
print("-" * 45)

for n in [4, 6, 8, 10]:
    for oracle_t in [5, 10, 20]:
        grover = build_grover_iteration(n, oracle_t)
        analyzer = TCountAnalyzer(grover)
        result = analyzer.analyze()

        print(f"{n:<10} {oracle_t:<10} {result['t_count']:<12} {result['t_depth']:<10}")

# Gate breakdown visualization
print("\n--- Gate Breakdown Example (8-qubit QFT) ---")
qft8 = build_qft_circuit(8)
analyzer = TCountAnalyzer(qft8)
result = analyzer.analyze()

for gate, count in sorted(result['gate_breakdown'].items()):
    print(f"  {gate:<10}: {count}")

print(f"\n  Total T-count: {result['t_count']}")
print(f"  T-depth: {result['t_depth']}")
print(f"  T-parallelism: {result['avg_parallel_t']:.2f}")
```

---

### Lab 3: Visualization Suite

```python
"""
Lab 3: Visualization Suite for FT Computation Analysis
Interactive exploration of synthesis and resources
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_sk_coverage(sk, max_depth=3):
    """Visualize SU(2) coverage by Solovay-Kitaev at different depths"""
    fig = plt.figure(figsize=(16, 5))

    # Sample random rotations
    n_samples = 500
    np.random.seed(42)

    for idx, depth in enumerate([0, 2, 4]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        errors = []
        points_x, points_y, points_z = [], [], []

        for _ in range(n_samples):
            # Random rotation
            theta = np.random.uniform(0, 2*np.pi)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)

            target = rotation_to_matrix(theta, axis)

            # Approximate
            if depth <= 2:  # Only compute for reasonable depths
                seq = sk.decompose(target, depth=depth)
                error = matrix_distance(target, seq.matrix)
            else:
                error = 0.1 ** (1.5 ** depth)  # Theoretical estimate

            # Record point (rotation axis scaled by angle)
            points_x.append(axis[0] * theta)
            points_y.append(axis[1] * theta)
            points_z.append(axis[2] * theta)
            errors.append(error)

        # Color by error (log scale)
        log_errors = np.log10(np.array(errors) + 1e-15)

        scatter = ax.scatter(points_x, points_y, points_z,
                            c=log_errors, cmap='viridis_r',
                            s=10, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'SK Depth {depth}\nMean error: {np.mean(errors):.2e}')

        # Draw unit sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs = np.cos(u)*np.sin(v) * np.pi
        ys = np.sin(u)*np.sin(v) * np.pi
        zs = np.cos(v) * np.pi
        ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.1)

    plt.tight_layout()
    plt.savefig('sk_coverage.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_t_count_scaling():
    """Visualize T-count scaling for different operations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Rotation synthesis T-count vs precision
    ax1 = axes[0, 0]
    precisions = np.logspace(-2, -12, 50)
    t_counts_theory = 3 * np.log2(1/precisions)

    ax1.semilogx(precisions, t_counts_theory, 'b-', linewidth=2, label='~3 log₂(1/ε)')
    ax1.set_xlabel('Precision (ε)')
    ax1.set_ylabel('T-count')
    ax1.set_title('T-count for Rotation Synthesis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Plot 2: Algorithm T-count scaling
    ax2 = axes[0, 1]
    n_bits = np.arange(100, 4100, 100)

    # Shor: O(n^3)
    shor_t = 2 * n_bits**3

    # Grover (n-bit search, 100 Toffoli oracle): sqrt(2^n) * 7 * 100
    grover_n = np.array([20, 30, 40, 50])
    grover_t = np.sqrt(2**grover_n) * 700

    ax2.semilogy(n_bits, shor_t, 'b-', linewidth=2, label='Shor (n-bit)')
    ax2.set_xlabel('Problem Size (bits)')
    ax2.set_ylabel('T-count')
    ax2.set_title('Algorithm T-count Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Physical qubits vs code distance
    ax3 = axes[1, 0]
    distances = np.arange(3, 51, 2)
    n_logicals = [10, 50, 100, 500]

    for n_l in n_logicals:
        phys = n_l * 2 * distances**2
        ax3.semilogy(distances, phys, linewidth=2, label=f'{n_l} logical')

    ax3.set_xlabel('Code Distance d')
    ax3.set_ylabel('Physical Qubits')
    ax3.set_title('Physical Qubit Scaling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Runtime breakdown
    ax4 = axes[1, 1]
    t_counts = np.logspace(6, 12, 7)
    factory_sizes = [10, 50, 100, 500]
    cycle_time = 1e-6  # 1 microsecond

    for n_f in factory_sizes:
        t_rate = n_f * 1e4  # 10k T-states/s per factory
        runtime_s = t_counts / t_rate
        runtime_hours = runtime_s / 3600
        ax4.loglog(t_counts, runtime_hours, linewidth=2, label=f'{n_f} factories')

    ax4.set_xlabel('T-count')
    ax4.set_ylabel('Runtime (hours)')
    ax4.set_title('Runtime vs Factory Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ft_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_resource_comparison():
    """Compare resources for different algorithms"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Data for comparison
    algorithms = ['RSA-1024', 'RSA-2048', 'RSA-4096', 'Grover-40', 'FeMoco']
    t_counts = [2.1e9, 1.7e10, 1.4e11, 2e9, 1e12]
    phys_qubits = [4e6, 20e6, 100e6, 6e5, 4e6]
    runtimes_hrs = [0.5, 8, 100, 6, 500]

    x = np.arange(len(algorithms))
    width = 0.35

    # Plot 1: T-count and qubits
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    bars1 = ax1.bar(x - width/2, t_counts, width, label='T-count', color='steelblue')
    bars2 = ax1_twin.bar(x + width/2, phys_qubits, width, label='Physical Qubits', color='coral')

    ax1.set_yscale('log')
    ax1_twin.set_yscale('log')
    ax1.set_ylabel('T-count', color='steelblue')
    ax1_twin.set_ylabel('Physical Qubits', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_title('T-count and Physical Qubits')

    # Plot 2: Runtime
    ax2 = axes[1]
    bars3 = ax2.bar(x, runtimes_hrs, color='forestgreen')
    ax2.set_yscale('log')
    ax2.set_ylabel('Runtime (hours)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_title('Estimated Runtime')

    # Add value labels
    for bar, val in zip(bars3, runtimes_hrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.0f}h', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# Run visualizations
print("="*70)
print("Generating Visualizations...")
print("="*70)

# Need SK instance for coverage visualization
sk = SolovayKitaev(base_depth=3)

print("\n1. SK Coverage Visualization")
visualize_sk_coverage(sk, max_depth=4)

print("\n2. T-Count Scaling Visualization")
visualize_t_count_scaling()

print("\n3. Resource Comparison Visualization")
visualize_resource_comparison()

print("\n" + "="*70)
print("Visualization Suite Complete")
print("="*70)
```

---

### Lab 4: Complete Integration Example

```python
"""
Lab 4: Complete Integration - End-to-End Example
From algorithm to resource estimate
"""

import numpy as np
from typing import Dict

class CompleteFTAnalyzer:
    """
    Complete fault-tolerant analysis pipeline.
    From algorithm specification to physical resource estimate.
    """

    def __init__(self, physical_error_rate: float = 1e-3):
        self.p_phys = physical_error_rate
        self.p_threshold = 1e-2
        self.cycle_time_us = 1.0

    def analyze_algorithm(self, name: str, n_qubits: int,
                          t_count: int, t_depth: int,
                          target_error: float = 1e-6) -> Dict:
        """
        Complete analysis of an algorithm.

        Returns comprehensive resource estimate.
        """
        # Step 1: Determine code distance
        d = self._required_distance(n_qubits, t_depth, target_error)

        # Step 2: Initial runtime estimate
        runtime_us = t_depth * d * self.cycle_time_us * 100  # rough

        # Step 3: Size factories
        n_factories = self._size_factories(t_count, runtime_us, d)

        # Step 4: Refined runtime
        t_rate = n_factories * 1e4  # T/s per factory
        runtime_us = max(
            t_depth * d * self.cycle_time_us,
            t_count / t_rate * 1e6
        )

        # Step 5: Recalculate distance with new runtime
        effective_depth = runtime_us / (d * self.cycle_time_us)
        d = self._required_distance(n_qubits, effective_depth, target_error)

        # Step 6: Final calculations
        n_factories = self._size_factories(t_count, runtime_us, d)
        phys_qubits = self._physical_qubits(n_qubits, d, n_factories)

        return {
            'algorithm': name,
            'logical_qubits': n_qubits,
            't_count': t_count,
            't_depth': t_depth,
            'target_error': target_error,
            'code_distance': d,
            'n_factories': n_factories,
            'physical_qubits': phys_qubits,
            'runtime_us': runtime_us,
            'runtime_hours': runtime_us / (1e6 * 3600),
        }

    def _required_distance(self, n_qubits: int, depth: float,
                            target_error: float) -> int:
        """Calculate minimum code distance"""
        error_per_cycle = target_error / (n_qubits * max(depth, 1))
        ratio = self.p_phys / self.p_threshold

        if ratio >= 1 or error_per_cycle <= 0:
            return 99

        d_min = 2 * np.log(error_per_cycle / 0.1) / np.log(ratio) - 1
        d = max(3, int(np.ceil(d_min)))
        if d % 2 == 0:
            d += 1
        return min(d, 51)

    def _size_factories(self, t_count: int, runtime_us: float, d: int) -> int:
        """Calculate number of factories needed"""
        if runtime_us <= 0:
            return 1

        t_rate_needed = t_count / runtime_us * 1e6  # T/s needed
        distill_time_us = 10 * d * self.cycle_time_us
        rate_per_factory = 1 / distill_time_us * 1e6  # T/s per factory

        n = int(np.ceil(t_rate_needed / rate_per_factory))
        return max(1, min(n, 1000))

    def _physical_qubits(self, n_logical: int, d: int, n_factories: int) -> Dict:
        """Calculate physical qubit breakdown"""
        data = n_logical * 2 * d**2
        factory = n_factories * 16 * d**2
        routing = int(0.5 * data)

        return {
            'data': data,
            'factory': factory,
            'routing': routing,
            'total': data + factory + routing
        }

    def compare_algorithms(self, algorithms: list) -> None:
        """Compare multiple algorithms"""
        print(f"\n{'Algorithm':<20} {'Logical Q':<12} {'T-count':<12} "
              f"{'Distance':<10} {'Phys Q':<15} {'Runtime':<12}")
        print("-" * 85)

        for algo in algorithms:
            result = self.analyze_algorithm(**algo)
            print(f"{result['algorithm']:<20} {result['logical_qubits']:<12} "
                  f"{result['t_count']:<12.2e} {result['code_distance']:<10} "
                  f"{result['physical_qubits']['total']:<15,.0f} "
                  f"{result['runtime_hours']:<12.2f} hrs")


# Demonstration
print("="*70)
print("Complete FT Analysis Pipeline")
print("="*70)

analyzer = CompleteFTAnalyzer()

# Define algorithms to analyze
algorithms = [
    {
        'name': 'Shor-2048',
        'n_qubits': 4100,
        't_count': int(1.7e10),
        't_depth': int(2048**2),
    },
    {
        'name': 'Grover-40',
        'n_qubits': 150,
        't_count': int(2e9),
        't_depth': int(1e6),
    },
    {
        'name': 'QFT-1000',
        'n_qubits': 1000,
        't_count': int(3e9),
        't_depth': int(1e6),
    },
    {
        'name': 'Chemistry-100',
        'n_qubits': 200,
        't_count': int(1e11),
        't_depth': int(1e7),
        'target_error': 1e-8,
    },
]

analyzer.compare_algorithms(algorithms)

# Detailed analysis of one algorithm
print("\n" + "="*70)
print("Detailed Analysis: Shor-2048")
print("="*70)

result = analyzer.analyze_algorithm(
    name='Shor-2048',
    n_qubits=4100,
    t_count=int(1.7e10),
    t_depth=int(2048**2)
)

print(f"\nAlgorithm: {result['algorithm']}")
print(f"\n[Logical Resources]")
print(f"  Logical qubits: {result['logical_qubits']:,}")
print(f"  T-count: {result['t_count']:,.0f}")
print(f"  T-depth: {result['t_depth']:,}")

print(f"\n[Physical Resources]")
print(f"  Code distance: {result['code_distance']}")
print(f"  Data qubits: {result['physical_qubits']['data']:,}")
print(f"  Factory qubits: {result['physical_qubits']['factory']:,}")
print(f"  Routing qubits: {result['physical_qubits']['routing']:,}")
print(f"  TOTAL: {result['physical_qubits']['total']:,}")

print(f"\n[Performance]")
print(f"  Number of factories: {result['n_factories']}")
print(f"  Runtime: {result['runtime_hours']:.2f} hours")

print("\n" + "="*70)
print("Complete Integration Lab Finished")
print("="*70)
```

---

### Summary

Today's computational lab covered:

1. **Complete Solovay-Kitaev Implementation**
   - Full recursive decomposition algorithm
   - Error scaling verification
   - Performance analysis

2. **T-Count Analysis Tools**
   - Circuit analysis for T-gates
   - T-depth calculation
   - Parallelism estimation

3. **Visualization Suite**
   - SU(2) coverage plots
   - Scaling analysis graphs
   - Algorithm comparisons

4. **Integration Pipeline**
   - End-to-end resource estimation
   - Multi-algorithm comparison
   - Detailed breakdowns

---

### Key Outputs

All code produces:
- `sk_coverage.png` - Solovay-Kitaev approximation quality
- `ft_scaling.png` - Resource scaling analysis
- `algorithm_comparison.png` - Algorithm resource comparison
- Console output with detailed analysis

---

### Daily Checklist

- [ ] Implemented complete Solovay-Kitaev algorithm
- [ ] Built T-count analyzer for quantum circuits
- [ ] Created resource estimation pipeline
- [ ] Generated visualization plots
- [ ] Analyzed multiple algorithm case studies
- [ ] Saved all figures for future reference

---

### Preview: Day 868

Tomorrow is the **Month 31 Capstone**, where we synthesize all material from this month. We will build a complete end-to-end fault-tolerant quantum computing pipeline, from algorithm specification through physical implementation. This serves as both a comprehensive review and a preview of Month 32's advanced topics.

---

*Day 867 provides hands-on implementation experience---tomorrow we bring everything together.*
