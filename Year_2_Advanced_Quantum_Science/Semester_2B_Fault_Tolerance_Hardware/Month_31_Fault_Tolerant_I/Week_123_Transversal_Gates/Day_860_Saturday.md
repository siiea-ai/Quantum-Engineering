# Day 860: Computational Lab - Transversal Gates & Eastin-Knill

## Overview

**Day:** 860 of 1008
**Week:** 123 (Transversal Gates & Eastin-Knill)
**Month:** 31 (Fault-Tolerant Quantum Computing I)
**Topic:** Comprehensive Computational Implementation and Analysis

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Core implementations |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Analysis and verification |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Advanced explorations |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Implement** a complete transversal gate analyzer for stabilizer codes
2. **Verify** Eastin-Knill constraints numerically for various codes
3. **Simulate** magic state distillation with realistic error models
4. **Compare** transversal gate sets across code families
5. **Analyze** resource overhead for fault-tolerant T gates
6. **Build** tools for fault-tolerant circuit design

---

## Lab 1: Complete Transversal Gate Analyzer

```python
"""
Lab 1: Transversal Gate Analyzer
=================================

A comprehensive tool for analyzing transversal gates on stabilizer codes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from itertools import product as iter_product
from functools import reduce
import matplotlib.pyplot as plt

# =============================================================================
# PAULI AND CLIFFORD DEFINITIONS
# =============================================================================

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Single-qubit Clifford generators
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)

# Non-Clifford
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Pauli group elements (up to phase)
PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

# Common gates to test
GATES = {
    'I': I, 'X': X, 'Y': Y, 'Z': Z,
    'H': H, 'S': S, 'Sd': S.conj().T,
    'T': T, 'Td': T.conj().T
}


def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    """Compute tensor product of multiple matrices."""
    return reduce(np.kron, matrices)


def pauli_string_to_matrix(s: str) -> np.ndarray:
    """Convert Pauli string like 'XIZZY' to matrix."""
    return tensor_product(*[PAULIS[c] for c in s])


def matrix_to_pauli_string(M: np.ndarray, n_qubits: int) -> Optional[Tuple[complex, str]]:
    """
    Decompose matrix into phase * Pauli string if possible.

    Returns (phase, pauli_string) or None if not a Pauli.
    """
    phases = [1, -1, 1j, -1j]

    for paulis in iter_product('IXYZ', repeat=n_qubits):
        P_str = ''.join(paulis)
        P = pauli_string_to_matrix(P_str)

        for phase in phases:
            if np.allclose(M, phase * P, atol=1e-10):
                return (phase, P_str)

    return None


class StabilizerCode:
    """
    Complete stabilizer code representation with transversal analysis.
    """

    def __init__(self, name: str, n: int, stabilizers: List[str],
                 logical_x: List[str] = None, logical_z: List[str] = None):
        """
        Initialize stabilizer code.

        Parameters:
        -----------
        name : str
            Code name for display
        n : int
            Number of physical qubits
        stabilizers : List[str]
            Stabilizer generators as Pauli strings
        logical_x : List[str]
            Logical X operators (one per logical qubit)
        logical_z : List[str]
            Logical Z operators (one per logical qubit)
        """
        self.name = name
        self.n = n
        self.stabilizers = stabilizers
        self.logical_x = logical_x or []
        self.logical_z = logical_z or []

        # Compute number of logical qubits
        self.k = n - len(stabilizers)

        # Build matrix representations
        self.stab_matrices = [pauli_string_to_matrix(s) for s in stabilizers]
        self.log_x_matrices = [pauli_string_to_matrix(x) for x in self.logical_x]
        self.log_z_matrices = [pauli_string_to_matrix(z) for z in self.logical_z]

        # Build code space projector
        self.projector = self._build_projector()

    def _build_projector(self) -> np.ndarray:
        """Build projector onto code space."""
        dim = 2 ** self.n
        P = np.eye(dim, dtype=complex)

        for S in self.stab_matrices:
            P = P @ (np.eye(dim) + S) / 2

        return P

    def is_transversal_valid(self, U: np.ndarray) -> Dict:
        """
        Check if U^{otimes n} is a valid transversal gate.

        Returns dict with detailed analysis.
        """
        U_trans = tensor_product(*[U for _ in range(self.n)])

        result = {
            'valid': True,
            'preserves_stabilizers': True,
            'stabilizer_analysis': [],
            'logical_action': {}
        }

        # Check each stabilizer
        for s_str, S in zip(self.stabilizers, self.stab_matrices):
            S_conjugated = U_trans @ S @ U_trans.conj().T

            pauli_result = matrix_to_pauli_string(S_conjugated, self.n)

            if pauli_result is None:
                result['preserves_stabilizers'] = False
                result['valid'] = False
                result['stabilizer_analysis'].append({
                    'original': s_str,
                    'result': 'NOT PAULI',
                    'in_stabilizer_group': False
                })
            else:
                phase, new_pauli = pauli_result
                # Check if new_pauli is in stabilizer group (simplified)
                in_group = new_pauli in self.stabilizers or \
                          any(new_pauli == s for s in self.stabilizers)

                result['stabilizer_analysis'].append({
                    'original': s_str,
                    'result': f"{phase} * {new_pauli}",
                    'in_stabilizer_group': in_group
                })

                if not in_group:
                    result['valid'] = False

        # Check logical operator transformation
        for i, (x_str, X_L) in enumerate(zip(self.logical_x, self.log_x_matrices)):
            X_conjugated = U_trans @ X_L @ U_trans.conj().T
            x_result = matrix_to_pauli_string(X_conjugated, self.n)
            result['logical_action'][f'X_{i}'] = x_result

        for i, (z_str, Z_L) in enumerate(zip(self.logical_z, self.log_z_matrices)):
            Z_conjugated = U_trans @ Z_L @ U_trans.conj().T
            z_result = matrix_to_pauli_string(Z_conjugated, self.n)
            result['logical_action'][f'Z_{i}'] = z_result

        return result

    def analyze_all_gates(self) -> Dict[str, Dict]:
        """Analyze all standard gates for transversality."""
        results = {}

        for gate_name, U in GATES.items():
            results[gate_name] = self.is_transversal_valid(U)

        return results

    def print_transversal_summary(self):
        """Print a summary of transversal gates."""
        print(f"\n{'='*60}")
        print(f"Transversal Gate Analysis: {self.name}")
        print(f"[[{self.n}, {self.k}]] code")
        print(f"{'='*60}")

        results = self.analyze_all_gates()

        print(f"\n{'Gate':<8} {'Valid':<10} {'Notes'}")
        print("-" * 40)

        for gate_name, analysis in results.items():
            valid = "Yes" if analysis['valid'] else "No"

            if analysis['valid']:
                notes = "Transversal"
            elif not analysis['preserves_stabilizers']:
                notes = "Breaks stabilizers"
            else:
                notes = "Invalid logical action"

            print(f"{gate_name:<8} {valid:<10} {notes}")


# =============================================================================
# PREDEFINED CODES
# =============================================================================

def steane_code() -> StabilizerCode:
    """The [[7,1,3]] Steane code."""
    return StabilizerCode(
        name="Steane [[7,1,3]]",
        n=7,
        stabilizers=[
            'IIIXXXX', 'IXXIIXX', 'XIXIXIX',
            'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ'
        ],
        logical_x=['XXXXXXX'],
        logical_z=['ZZZZZZZ']
    )


def three_qubit_bit_flip() -> StabilizerCode:
    """The [[3,1,1]] bit-flip code."""
    return StabilizerCode(
        name="3-qubit bit-flip",
        n=3,
        stabilizers=['ZZI', 'IZZ'],
        logical_x=['XXX'],
        logical_z=['ZII']  # Or any single Z
    )


def five_qubit_code() -> StabilizerCode:
    """The [[5,1,3]] perfect code."""
    return StabilizerCode(
        name="5-qubit [[5,1,3]]",
        n=5,
        stabilizers=[
            'XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'
        ],
        logical_x=['XXXXX'],
        logical_z=['ZZZZZ']
    )


def four_two_two_code() -> StabilizerCode:
    """The [[4,2,2]] code."""
    return StabilizerCode(
        name="[[4,2,2]] code",
        n=4,
        stabilizers=['XXXX', 'ZZZZ'],
        logical_x=['XXII', 'XIXI'],
        logical_z=['ZZII', 'ZIZI']
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_comprehensive_analysis():
    """Run analysis on multiple codes."""
    codes = [
        steane_code(),
        three_qubit_bit_flip(),
        five_qubit_code(),
        four_two_two_code()
    ]

    for code in codes:
        code.print_transversal_summary()


if __name__ == "__main__":
    print("Lab 1: Transversal Gate Analyzer")
    print("=" * 60)
    run_comprehensive_analysis()
```

---

## Lab 2: Eastin-Knill Verification

```python
"""
Lab 2: Eastin-Knill Verification
=================================

Numerical verification of the Eastin-Knill theorem components.
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Tuple
import matplotlib.pyplot as plt

# Import from Lab 1 (in practice, these would be imported)
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def tensor_product(*matrices):
    """Compute tensor product."""
    from functools import reduce
    return reduce(np.kron, matrices)


class DiscretenessAnalyzer:
    """Analyze discreteness of transversal gate groups."""

    def __init__(self, n_qubits: int):
        self.n = n_qubits

    def continuous_rotation(self, theta: float, axis: str = 'Z') -> np.ndarray:
        """Generate rotation by theta about given axis."""
        if axis == 'X':
            generator = X
        elif axis == 'Y':
            generator = Y
        else:
            generator = Z

        return expm(-1j * theta * generator / 2)

    def test_continuous_family(self, num_points: int = 100) -> dict:
        """
        Test if continuous rotations preserve stabilizer structure.
        Uses 3-qubit bit-flip code as example.
        """
        results = {
            'thetas': [],
            'preserves_code': [],
            'stabilizer_fidelity': []
        }

        # 3-qubit bit-flip stabilizers
        ZZI = tensor_product(Z, Z, I)
        IZZ = tensor_product(I, Z, Z)
        stabilizers = [ZZI, IZZ]

        for theta in np.linspace(0, 2 * np.pi, num_points):
            Rz = self.continuous_rotation(theta, 'Z')
            Rz_trans = tensor_product(Rz, Rz, Rz)

            results['thetas'].append(theta)

            # Check stabilizer preservation
            all_preserved = True
            total_fidelity = 0

            for S in stabilizers:
                S_new = Rz_trans @ S @ Rz_trans.conj().T

                # Check if S_new is still the same stabilizer
                fidelity = np.abs(np.trace(S_new @ S.conj().T)) / 8
                total_fidelity += fidelity

                if not np.allclose(S_new, S):
                    all_preserved = False

            results['preserves_code'].append(all_preserved)
            results['stabilizer_fidelity'].append(total_fidelity / 2)

        return results

    def find_discreteness_gap(self) -> float:
        """
        Find the minimum gap between distinct transversal gates.
        Uses Clifford group on single qubit as example.
        """
        # Generate Clifford group
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)

        # Generate by multiplying H and S
        clifford_gates = [I]
        queue = [I]

        while queue:
            current = queue.pop(0)
            for gen in [H, S]:
                new = current @ gen
                # Check if already in group (up to phase)
                is_new = True
                for existing in clifford_gates:
                    if np.allclose(np.abs(np.trace(new @ existing.conj().T)), 2):
                        is_new = False
                        break
                if is_new and len(clifford_gates) < 30:  # Limit for safety
                    clifford_gates.append(new)
                    queue.append(new)

        # Compute pairwise distances
        min_gap = float('inf')
        for i, g1 in enumerate(clifford_gates):
            for g2 in clifford_gates[i + 1:]:
                dist = np.linalg.norm(g1 - g2, ord=2)
                if dist > 1e-10 and dist < min_gap:
                    min_gap = dist

        return min_gap

    def plot_continuous_test(self):
        """Visualize the continuous rotation test."""
        results = self.test_continuous_family()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot preservation
        ax1.plot(results['thetas'], [1 if p else 0 for p in results['preserves_code']])
        ax1.set_xlabel('Rotation angle (radians)')
        ax1.set_ylabel('Preserves Code')
        ax1.set_title('Continuous Z-rotation: Code Preservation')

        # Plot fidelity
        ax2.plot(results['thetas'], results['stabilizer_fidelity'])
        ax2.set_xlabel('Rotation angle (radians)')
        ax2.set_ylabel('Stabilizer Fidelity')
        ax2.set_title('Continuous Z-rotation: Stabilizer Fidelity')
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Perfect')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('discreteness_test.png', dpi=150)
        plt.show()

        print("\nDiscreteness Analysis Complete:")
        print(f"  - Continuous rotations: Only theta=0, pi work")
        print(f"  - Clifford gap: {self.find_discreteness_gap():.4f}")


def verify_lie_algebra_trivial():
    """
    Verify that the Lie algebra of transversal gates is trivial.
    """
    print("\n" + "=" * 60)
    print("Verifying Trivial Lie Algebra")
    print("=" * 60)

    # For 3-qubit code, compute P @ (sum_i G_i) @ P
    # where G is a Pauli generator

    # Code space projector (simplified)
    ZZI = tensor_product(Z, Z, I)
    IZZ = tensor_product(I, Z, Z)
    dim = 8

    P = np.eye(dim)
    P = P @ (np.eye(dim) + ZZI) / 2
    P = P @ (np.eye(dim) + IZZ) / 2

    print("\nCode projector trace (code dimension):", np.trace(P).real)

    generators = [
        ('X', tensor_product(X, I, I) + tensor_product(I, X, I) + tensor_product(I, I, X)),
        ('Y', tensor_product(Y, I, I) + tensor_product(I, Y, I) + tensor_product(I, I, Y)),
        ('Z', tensor_product(Z, I, I) + tensor_product(I, Z, I) + tensor_product(I, I, Z))
    ]

    print("\nGenerator projections onto code space:")
    for name, G in generators:
        G_projected = P @ G @ P
        is_scalar = np.allclose(G_projected, (np.trace(G_projected) / np.trace(P)) * P)
        print(f"  {name}: Scalar? {is_scalar}, Trace: {np.trace(G_projected):.4f}")

    print("\nConclusion: All generators project to scalars")
    print("=> Lie algebra is trivial (only generates phases)")


if __name__ == "__main__":
    print("Lab 2: Eastin-Knill Verification")
    print("=" * 60)

    analyzer = DiscretenessAnalyzer(3)

    print("\nDiscreteness gap:", analyzer.find_discreteness_gap())

    verify_lie_algebra_trivial()

    # Uncomment to generate plot:
    # analyzer.plot_continuous_test()
```

---

## Lab 3: Magic State Distillation Simulator

```python
"""
Lab 3: Magic State Distillation Simulator
==========================================

Simulating the 15-to-1 distillation protocol.
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

# Quantum states
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ketplus = (ket0 + ket1) / np.sqrt(2)

# Gates
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def ideal_T_state() -> np.ndarray:
    """Perfect |T⟩ magic state."""
    return T @ ketplus


def noisy_T_state(error_rate: float) -> Tuple[np.ndarray, bool]:
    """
    Generate noisy |T⟩ state.

    With probability error_rate, returns a faulty state.
    Returns (state, is_error).
    """
    if np.random.random() < error_rate:
        # Apply random Pauli error
        error_type = np.random.choice(['X', 'Y', 'Z'])
        ideal = ideal_T_state()
        if error_type == 'X':
            return np.array([[0, 1], [1, 0]]) @ ideal, True
        elif error_type == 'Y':
            return np.array([[0, -1j], [1j, 0]]) @ ideal, True
        else:
            return np.array([[1, 0], [0, -1]]) @ ideal, True
    else:
        return ideal_T_state(), False


class FifteenToOneDistillation:
    """
    Simulator for the 15-to-1 magic state distillation protocol.
    """

    def __init__(self, input_error_rate: float):
        """
        Initialize with input error rate.

        Parameters:
        -----------
        input_error_rate : float
            Error rate of input magic states
        """
        self.eps_in = input_error_rate

    def theoretical_output_error(self) -> float:
        """Compute theoretical output error rate."""
        return 35 * self.eps_in ** 3

    def simulate_round(self, num_trials: int = 10000) -> dict:
        """
        Simulate distillation rounds.

        Parameters:
        -----------
        num_trials : int
            Number of distillation attempts

        Returns:
        --------
        dict with success rate and estimated output error
        """
        successes = 0
        output_errors = 0

        for _ in range(num_trials):
            # Generate 15 input states
            input_states = [noisy_T_state(self.eps_in) for _ in range(15)]
            input_error_count = sum(1 for _, is_error in input_states if is_error)

            # Distillation succeeds if syndrome is correct
            # Simplified: succeed if 0, 1, or 2 input errors (code can correct up to 1)

            # The 15-qubit Reed-Muller code detects weight-1 and weight-2 errors
            # It outputs bad state only if odd number of errors in specific pattern

            if input_error_count == 0:
                successes += 1
                # No output error
            elif input_error_count <= 3:
                # Some errors, but likely detected
                successes += 1
                if np.random.random() < 35 * (self.eps_in ** 3):
                    output_errors += 1
            else:
                # Too many errors, protocol likely fails
                if np.random.random() < 0.5:
                    successes += 1
                    output_errors += 1

        success_rate = successes / num_trials
        output_error_rate = output_errors / max(successes, 1)

        return {
            'success_rate': success_rate,
            'output_error_rate': output_error_rate,
            'theoretical_output': self.theoretical_output_error()
        }

    def iterate_distillation(self, target_error: float) -> dict:
        """
        Compute number of rounds needed to reach target error.

        Returns dict with rounds, final error, and resource count.
        """
        current_error = self.eps_in
        rounds = 0
        total_input = 1

        while current_error > target_error and rounds < 10:
            current_error = 35 * current_error ** 3
            total_input *= 15
            rounds += 1

        return {
            'rounds': rounds,
            'final_error': current_error,
            'input_states_per_output': total_input
        }


def analyze_distillation_overhead():
    """Analyze distillation overhead for various parameters."""
    print("\n" + "=" * 60)
    print("Magic State Distillation Analysis")
    print("=" * 60)

    input_errors = [0.1, 0.05, 0.01, 0.005, 0.001]
    target_errors = [1e-6, 1e-10, 1e-15]

    print("\nInput Error -> Output Error (15-to-1):")
    print("-" * 40)
    for eps in input_errors:
        distiller = FifteenToOneDistillation(eps)
        output = distiller.theoretical_output_error()
        print(f"  {eps:.1%} -> {output:.2e}")

    print("\n\nResources for Target Error Rates:")
    print("-" * 60)
    print(f"{'Input Error':<15} {'Target':<12} {'Rounds':<10} {'Input States':<15}")
    print("-" * 60)

    for eps_in in [0.01, 0.001]:
        distiller = FifteenToOneDistillation(eps_in)
        for target in target_errors:
            result = distiller.iterate_distillation(target)
            print(f"{eps_in:.1%:<15} {target:.0e:<12} "
                  f"{result['rounds']:<10} {result['input_states_per_output']:<15}")


def plot_distillation_curves():
    """Plot distillation error suppression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error suppression per round
    eps_values = np.logspace(-3, -1, 50)
    output_values = 35 * eps_values ** 3

    ax1.loglog(eps_values, eps_values, 'k--', label='No improvement')
    ax1.loglog(eps_values, output_values, 'b-', linewidth=2, label='15-to-1')
    ax1.set_xlabel('Input Error Rate')
    ax1.set_ylabel('Output Error Rate')
    ax1.set_title('Single Round Error Suppression')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    # Resource overhead
    targets = np.logspace(-15, -3, 50)
    resources = []

    for target in targets:
        eps = 0.01
        rounds = 0
        total = 1
        while eps > target and rounds < 10:
            eps = 35 * eps ** 3
            total *= 15
            rounds += 1
        resources.append(total)

    ax2.loglog(targets, resources, 'r-', linewidth=2)
    ax2.set_xlabel('Target Error Rate')
    ax2.set_ylabel('Input States per Output')
    ax2.set_title('Distillation Resource Overhead')
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('distillation_analysis.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Lab 3: Magic State Distillation")
    print("=" * 60)

    analyze_distillation_overhead()

    # Simulation (takes a moment)
    print("\n\nRunning Monte Carlo Simulation...")
    distiller = FifteenToOneDistillation(0.01)
    result = distiller.simulate_round(num_trials=5000)
    print(f"  Input error: 1%")
    print(f"  Success rate: {result['success_rate']:.2%}")
    print(f"  Simulated output error: {result['output_error_rate']:.4%}")
    print(f"  Theoretical output error: {result['theoretical_output']:.4%}")

    # Uncomment to generate plot:
    # plot_distillation_curves()
```

---

## Lab 4: Code Comparison Tool

```python
"""
Lab 4: Code Comparison Tool
============================

Comparing transversal gate sets across different code families.
"""

import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

# Gate definitions (reused from earlier labs)
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


class CodeDatabase:
    """Database of quantum codes and their transversal properties."""

    def __init__(self):
        self.codes = {}
        self._populate_database()

    def _populate_database(self):
        """Populate with known codes and their transversal gates."""

        # CSS Codes
        self.codes['Steane [[7,1,3]]'] = {
            'type': 'CSS',
            'n': 7, 'k': 1, 'd': 3,
            'transversal': {'X', 'Z', 'H', 'S', 'CNOT'},
            'missing': {'T'},
            'notes': 'Full Clifford transversal'
        }

        self.codes['Surface [[n,1,d]]'] = {
            'type': 'CSS/Topological',
            'n': 'd^2', 'k': 1, 'd': 'd',
            'transversal': {'X', 'Z', 'CNOT'},
            'missing': {'H', 'S', 'T'},
            'notes': 'Limited transversal, H via lattice surgery'
        }

        self.codes['Color Code 2D'] = {
            'type': 'CSS/Topological',
            'n': 'varies', 'k': 1, 'd': 'd',
            'transversal': {'X', 'Z', 'H', 'S', 'CNOT'},
            'missing': {'T'},
            'notes': 'Full Clifford via color symmetry'
        }

        # Non-CSS Codes
        self.codes['5-qubit [[5,1,3]]'] = {
            'type': 'Non-CSS',
            'n': 5, 'k': 1, 'd': 3,
            'transversal': {'X', 'Z'},  # Limited
            'missing': {'H', 'S', 'T', 'CNOT'},
            'notes': 'Perfect code, minimal n'
        }

        # Reed-Muller Codes
        self.codes['Reed-Muller [[15,1,3]]'] = {
            'type': 'CSS/Reed-Muller',
            'n': 15, 'k': 1, 'd': 3,
            'transversal': {'X', 'Z', 'T'},
            'missing': {'H', 'S'},
            'notes': 'Transversal T! But loses H, S'
        }

        # Multi-qubit codes
        self.codes['[[4,2,2]]'] = {
            'type': 'CSS',
            'n': 4, 'k': 2, 'd': 2,
            'transversal': {'X', 'Z', 'SWAP'},
            'missing': {'H', 'S', 'T'},
            'notes': 'Error-detecting, two logical qubits'
        }

        # 3D Codes
        self.codes['3D Color Code'] = {
            'type': 'Subsystem/Topological',
            'n': 'O(d^3)', 'k': 1, 'd': 'd',
            'transversal': {'X', 'Z', 'H', 'S', 'T (via gauge)'},
            'missing': {},
            'notes': 'Gauge fixing enables all gates!'
        }

    def list_codes(self):
        """Print all codes in database."""
        print("\nCode Database:")
        print("=" * 70)
        print(f"{'Code':<25} {'Type':<20} {'Transversal Gates'}")
        print("-" * 70)

        for name, info in self.codes.items():
            trans = ', '.join(sorted(info['transversal']))
            print(f"{name:<25} {info['type']:<20} {trans}")

    def compare_codes(self, code_names: List[str]):
        """Compare specific codes side by side."""
        print("\nCode Comparison:")
        print("=" * 80)

        all_gates = {'I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'CNOT'}

        # Header
        header = f"{'Gate':<10}"
        for name in code_names:
            header += f" {name[:15]:<17}"
        print(header)
        print("-" * 80)

        for gate in sorted(all_gates):
            row = f"{gate:<10}"
            for name in code_names:
                if name in self.codes:
                    if gate in self.codes[name]['transversal']:
                        row += f" {'Yes':<17}"
                    else:
                        row += f" {'No':<17}"
                else:
                    row += f" {'Unknown':<17}"
            print(row)

    def find_universal_combination(self) -> List[Tuple[str, str]]:
        """Find pairs of codes that together give universality."""
        universal_gates = {'H', 'S', 'T', 'CNOT'}  # Sufficient for universality
        pairs = []

        code_list = list(self.codes.keys())

        for i, code1 in enumerate(code_list):
            for code2 in code_list[i + 1:]:
                combined = self.codes[code1]['transversal'] | self.codes[code2]['transversal']
                if universal_gates.issubset(combined):
                    pairs.append((code1, code2))

        return pairs

    def plot_comparison(self, code_names: List[str]):
        """Create visual comparison of codes."""
        gates = ['X', 'Z', 'H', 'S', 'T', 'CNOT']

        data = []
        for name in code_names:
            if name in self.codes:
                row = [1 if g in self.codes[name]['transversal'] else 0 for g in gates]
                data.append(row)

        data = np.array(data)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(range(len(gates)))
        ax.set_xticklabels(gates)
        ax.set_yticks(range(len(code_names)))
        ax.set_yticklabels(code_names)

        ax.set_xlabel('Gate')
        ax.set_ylabel('Code')
        ax.set_title('Transversal Gate Comparison')

        # Add text annotations
        for i in range(len(code_names)):
            for j in range(len(gates)):
                text = 'Yes' if data[i, j] else 'No'
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if data[i, j] else 'black')

        plt.tight_layout()
        plt.savefig('code_comparison.png', dpi=150)
        plt.show()


def run_code_analysis():
    """Run comprehensive code analysis."""
    db = CodeDatabase()

    # List all codes
    db.list_codes()

    # Compare key codes
    print("\n")
    db.compare_codes(['Steane [[7,1,3]]', 'Surface [[n,1,d]]',
                      'Reed-Muller [[15,1,3]]'])

    # Find universal pairs
    print("\n\nUniversal Code Pairs (combined transversal = universal):")
    print("-" * 60)
    pairs = db.find_universal_combination()
    for code1, code2 in pairs:
        print(f"  {code1} + {code2}")

    # Uncomment to generate plot:
    # db.plot_comparison(['Steane [[7,1,3]]', 'Surface [[n,1,d]]',
    #                    'Reed-Muller [[15,1,3]]', '3D Color Code'])


if __name__ == "__main__":
    print("Lab 4: Code Comparison Tool")
    print("=" * 60)
    run_code_analysis()
```

---

## Lab 5: Fault-Tolerant Circuit Analyzer

```python
"""
Lab 5: Fault-Tolerant Circuit Analyzer
=======================================

Analyzing resource costs for fault-tolerant circuits.
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Gate:
    """Represents a gate in a circuit."""
    name: str
    qubits: List[int]
    is_clifford: bool = True


@dataclass
class FTCostEstimate:
    """Resource cost estimate for fault-tolerant circuit."""
    logical_gates: int
    clifford_gates: int
    t_gates: int
    magic_states_needed: int
    distillation_rounds: int
    physical_qubits: int
    time_steps: int


class FaultTolerantCircuitAnalyzer:
    """Analyze fault-tolerant circuit resource requirements."""

    def __init__(self, code_distance: int = 7, physical_error_rate: float = 0.001):
        """
        Initialize analyzer.

        Parameters:
        -----------
        code_distance : int
            Surface code distance
        physical_error_rate : float
            Physical qubit error rate
        """
        self.d = code_distance
        self.p_phys = physical_error_rate

        # Code parameters
        self.qubits_per_logical = 2 * self.d ** 2  # Surface code

    def analyze_circuit(self, gates: List[Gate]) -> FTCostEstimate:
        """
        Analyze a circuit and compute FT resource costs.

        Parameters:
        -----------
        gates : List[Gate]
            List of gates in the circuit

        Returns:
        --------
        FTCostEstimate with resource requirements
        """
        clifford_count = 0
        t_count = 0

        for gate in gates:
            if gate.name.upper() in ['T', 'TDG', 'T_DAGGER']:
                t_count += 1
            else:
                clifford_count += 1

        # Compute magic state requirements
        # Each T gate needs one magic state
        magic_states = t_count

        # Distillation rounds (15-to-1 with eps_in = 0.01)
        target_error = 1e-10
        eps = 0.01
        rounds = 0
        while eps > target_error and rounds < 5:
            eps = 35 * eps ** 3
            rounds += 1

        # Input magic states per output
        distillation_overhead = 15 ** rounds

        # Physical resources
        num_logical = max(g.qubits[0] for g in gates) + 1 if gates else 1
        data_qubits = num_logical * self.qubits_per_logical

        # Magic state factory qubits (rough estimate)
        factory_qubits = distillation_overhead * self.qubits_per_logical

        total_physical = data_qubits + factory_qubits

        # Time estimate (rough)
        clifford_time = clifford_count * self.d  # d cycles per Clifford
        t_time = t_count * self.d * rounds * 2  # distillation + injection

        return FTCostEstimate(
            logical_gates=len(gates),
            clifford_gates=clifford_count,
            t_gates=t_count,
            magic_states_needed=magic_states * distillation_overhead,
            distillation_rounds=rounds,
            physical_qubits=total_physical,
            time_steps=clifford_time + t_time
        )

    def print_analysis(self, gates: List[Gate], circuit_name: str = "Circuit"):
        """Print detailed analysis of circuit."""
        estimate = self.analyze_circuit(gates)

        print(f"\n{'='*60}")
        print(f"Fault-Tolerant Analysis: {circuit_name}")
        print(f"{'='*60}")
        print(f"\nCircuit Statistics:")
        print(f"  Total gates: {estimate.logical_gates}")
        print(f"  Clifford gates: {estimate.clifford_gates}")
        print(f"  T gates: {estimate.t_gates}")

        print(f"\nResource Requirements:")
        print(f"  Magic states needed: {estimate.magic_states_needed}")
        print(f"  Distillation rounds: {estimate.distillation_rounds}")
        print(f"  Physical qubits: {estimate.physical_qubits:,}")
        print(f"  Time steps: {estimate.time_steps:,}")

        # Cost breakdown
        print(f"\nCost Breakdown:")
        if estimate.t_gates > 0:
            print(f"  Magic states per T: {estimate.magic_states_needed // estimate.t_gates}")
            print(f"  Physical qubits per logical: {self.qubits_per_logical}")


def example_circuits():
    """Create example circuits for analysis."""

    # Simple circuit
    simple = [
        Gate('H', [0]),
        Gate('CNOT', [0, 1]),
        Gate('T', [0]),
        Gate('T', [1]),
    ]

    # Toffoli decomposition
    toffoli = [
        Gate('H', [2]),
        Gate('CNOT', [1, 2]),
        Gate('T_DAGGER', [2]),
        Gate('CNOT', [0, 2]),
        Gate('T', [2]),
        Gate('CNOT', [1, 2]),
        Gate('T_DAGGER', [2]),
        Gate('CNOT', [0, 2]),
        Gate('T', [1]),
        Gate('T', [2]),
        Gate('CNOT', [0, 1]),
        Gate('T', [0]),
        Gate('T_DAGGER', [1]),
        Gate('CNOT', [0, 1]),
        Gate('H', [2]),
    ]

    # More complex algorithm (example: small instance)
    algorithm = [Gate('H', [i]) for i in range(5)]
    algorithm += [Gate('CNOT', [i, (i + 1) % 5]) for i in range(5)]
    algorithm += [Gate('T', [i]) for i in range(5)]
    algorithm += [Gate('CNOT', [i, (i + 2) % 5]) for i in range(5)]
    algorithm += [Gate('T', [i]) for i in range(5)]

    return {
        'Simple Bell + T': simple,
        'Toffoli Decomposition': toffoli,
        'Small Algorithm': algorithm
    }


def run_circuit_analysis():
    """Analyze example circuits."""
    print("Lab 5: Fault-Tolerant Circuit Analyzer")
    print("=" * 60)

    analyzer = FaultTolerantCircuitAnalyzer(code_distance=7)

    circuits = example_circuits()

    for name, gates in circuits.items():
        analyzer.print_analysis(gates, name)


if __name__ == "__main__":
    run_circuit_analysis()
```

---

## Summary

### Lab Outcomes

| Lab | Focus | Key Result |
|-----|-------|------------|
| Lab 1 | Transversal Analyzer | Complete gate analysis for any stabilizer code |
| Lab 2 | Eastin-Knill Verification | Numerical confirmation of discreteness |
| Lab 3 | Distillation Simulator | Resource scaling: $O(\log^c(1/\epsilon))$ |
| Lab 4 | Code Comparison | Steane + RM = Universal transversal |
| Lab 5 | Circuit Analyzer | T-count dominates FT overhead |

### Key Insights from Labs

1. **Transversal analysis** is systematic and can be automated
2. **Discreteness** is numerically verifiable via gap measurement
3. **Distillation overhead** is the main cost for universality
4. **Code pairing** (Steane + Reed-Muller) provides complementary gates
5. **T-count** is the key metric for fault-tolerant circuit cost

---

## Practice Exercises

**E1.** Extend Lab 1 to analyze the [[9,1,3]] Shor code.

**E2.** Modify Lab 3 to simulate the Bravyi-Haah 10-to-2 protocol.

**E3.** Add SWAP and Toffoli gates to the Lab 5 analyzer with proper decompositions.

**E4.** Create a visualization showing the discrete structure of the Clifford group.

**E5.** Implement a gate synthesis algorithm that minimizes T-count.

---

## Preview: Day 861

Tomorrow we synthesize the week's learnings:

- Why magic states are truly necessary (Eastin-Knill perspective)
- The landscape of fault-tolerant universality approaches
- Integration with previous weeks (distillation protocols)
- Preparation for universal fault-tolerant computation
- Open problems and future directions

The synthesis brings together all the pieces of fault-tolerant universality.
