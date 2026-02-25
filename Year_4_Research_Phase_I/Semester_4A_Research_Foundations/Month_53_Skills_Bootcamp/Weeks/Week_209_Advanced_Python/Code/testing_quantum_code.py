"""
Testing Quantum Computing Code with pytest
==========================================

This module demonstrates best practices for testing quantum computing
code including numerical tolerances, parameterized tests, and fixtures.

Author: Quantum Engineering PhD Program
Week 209: Advanced Python for Reproducible Research

Run tests with: pytest testing_quantum_code.py -v
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from typing import Tuple


# =============================================================================
# Quantum Utility Functions to Test
# =============================================================================

def normalize_state(state: np.ndarray) -> np.ndarray:
    """
    Normalize a quantum state vector.

    Parameters
    ----------
    state : np.ndarray
        Unnormalized state vector.

    Returns
    -------
    np.ndarray
        Normalized state vector with unit norm.
    """
    norm = np.linalg.norm(state)
    if norm < 1e-15:
        raise ValueError("Cannot normalize zero state")
    return state / norm


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Compute the fidelity between two pure quantum states.

    F = |<psi|phi>|^2

    Parameters
    ----------
    state1, state2 : np.ndarray
        Normalized quantum state vectors.

    Returns
    -------
    float
        Fidelity between 0 and 1.
    """
    return np.abs(np.vdot(state1, state2)) ** 2


def apply_unitary(state: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """
    Apply a unitary operator to a quantum state.

    Parameters
    ----------
    state : np.ndarray
        Input state vector.
    unitary : np.ndarray
        Unitary matrix.

    Returns
    -------
    np.ndarray
        Evolved state vector.
    """
    return unitary @ state


def create_pauli_matrices() -> dict:
    """
    Create the Pauli matrices.

    Returns
    -------
    dict
        Dictionary with keys 'I', 'X', 'Y', 'Z' mapping to 2x2 matrices.
    """
    return {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }


def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    """
    Compute the tensor (Kronecker) product of matrices.

    Parameters
    ----------
    *matrices : np.ndarray
        Variable number of matrices to tensor.

    Returns
    -------
    np.ndarray
        Tensor product of all input matrices.
    """
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def partial_trace(rho: np.ndarray, dims: Tuple[int, ...],
                  trace_out: int) -> np.ndarray:
    """
    Compute the partial trace of a density matrix.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of the composite system.
    dims : tuple of int
        Dimensions of each subsystem.
    trace_out : int
        Index of the subsystem to trace out (0-indexed).

    Returns
    -------
    np.ndarray
        Reduced density matrix.
    """
    n_systems = len(dims)
    rho_reshaped = rho.reshape(dims * 2)

    # Build the axes to trace over
    axes_to_trace = [trace_out, trace_out + n_systems]

    # Trace over the specified subsystem
    result = np.trace(rho_reshaped, axis1=axes_to_trace[0],
                      axis2=axes_to_trace[1])

    # Compute new dimensions
    new_dims = list(dims)
    del new_dims[trace_out]

    if len(new_dims) == 0:
        return result

    total_dim = np.prod(new_dims)
    return result.reshape(total_dim, total_dim)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def pauli():
    """Provide Pauli matrices."""
    return create_pauli_matrices()


@pytest.fixture
def computational_basis():
    """Provide computational basis states."""
    return {
        '0': np.array([1, 0], dtype=complex),
        '1': np.array([0, 1], dtype=complex),
        '+': np.array([1, 1], dtype=complex) / np.sqrt(2),
        '-': np.array([1, -1], dtype=complex) / np.sqrt(2),
    }


@pytest.fixture
def bell_states():
    """Provide the four Bell states."""
    return {
        'phi+': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        'phi-': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        'psi+': np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        'psi-': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }


@pytest.fixture
def random_state():
    """Provide a reproducible random quantum state."""
    rng = np.random.default_rng(42)
    state = rng.random(4) + 1j * rng.random(4)
    return normalize_state(state)


@pytest.fixture
def random_unitary():
    """Provide a reproducible random unitary matrix."""
    rng = np.random.default_rng(42)
    # Generate random matrix and compute QR decomposition
    A = rng.random((4, 4)) + 1j * rng.random((4, 4))
    Q, R = np.linalg.qr(A)
    # Ensure determinant is 1
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


# =============================================================================
# Test Classes
# =============================================================================

class TestNormalization:
    """Tests for state normalization."""

    def test_already_normalized(self, computational_basis):
        """Normalized state should be unchanged."""
        state = computational_basis['0']
        result = normalize_state(state)
        assert_allclose(result, state)

    def test_unnormalized(self):
        """Unnormalized state should be normalized."""
        state = np.array([3, 4], dtype=complex)
        result = normalize_state(state)
        assert_allclose(np.linalg.norm(result), 1.0)
        assert_allclose(result, np.array([0.6, 0.8], dtype=complex))

    def test_complex_state(self):
        """Complex state should be properly normalized."""
        state = np.array([1 + 1j, 1 - 1j], dtype=complex)
        result = normalize_state(state)
        assert_allclose(np.linalg.norm(result), 1.0)

    def test_zero_state_raises(self):
        """Zero state should raise ValueError."""
        state = np.array([0, 0], dtype=complex)
        with pytest.raises(ValueError, match="Cannot normalize zero state"):
            normalize_state(state)


class TestFidelity:
    """Tests for fidelity computation."""

    def test_identical_states(self, computational_basis):
        """Fidelity of identical states should be 1."""
        state = computational_basis['0']
        assert_allclose(compute_fidelity(state, state), 1.0)

    def test_orthogonal_states(self, computational_basis):
        """Fidelity of orthogonal states should be 0."""
        state0 = computational_basis['0']
        state1 = computational_basis['1']
        assert_allclose(compute_fidelity(state0, state1), 0.0, atol=1e-14)

    def test_symmetry(self, random_state, computational_basis):
        """Fidelity should be symmetric."""
        state1 = random_state[:2] / np.linalg.norm(random_state[:2])
        state2 = computational_basis['+']

        f12 = compute_fidelity(state1, state2)
        f21 = compute_fidelity(state2, state1)

        assert_allclose(f12, f21)

    def test_bell_states_orthogonal(self, bell_states):
        """Bell states should be mutually orthogonal."""
        names = list(bell_states.keys())
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                fidelity = compute_fidelity(bell_states[name1], bell_states[name2])
                expected = 1.0 if i == j else 0.0
                assert_allclose(fidelity, expected, atol=1e-14,
                              err_msg=f"Fidelity between {name1} and {name2}")

    @pytest.mark.parametrize("theta", [0, np.pi/4, np.pi/2, np.pi])
    def test_fidelity_rotation(self, theta):
        """Test fidelity for rotated states."""
        state1 = np.array([1, 0], dtype=complex)
        state2 = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)

        expected_fidelity = np.cos(theta/2) ** 2
        actual_fidelity = compute_fidelity(state1, state2)

        assert_allclose(actual_fidelity, expected_fidelity, atol=1e-14)


class TestUnitaryApplication:
    """Tests for unitary operations."""

    def test_identity_unchanged(self, computational_basis):
        """Identity operation should not change state."""
        state = computational_basis['+']
        identity = np.eye(2, dtype=complex)
        result = apply_unitary(state, identity)
        assert_allclose(result, state)

    def test_pauli_x_flip(self, computational_basis, pauli):
        """Pauli X should flip |0> to |1>."""
        result = apply_unitary(computational_basis['0'], pauli['X'])
        assert_allclose(result, computational_basis['1'])

    def test_pauli_z_phase(self, computational_basis, pauli):
        """Pauli Z should add phase to |1>."""
        result = apply_unitary(computational_basis['1'], pauli['Z'])
        assert_allclose(result, -computational_basis['1'])

    def test_hadamard_superposition(self, computational_basis):
        """Hadamard should create superposition."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        result = apply_unitary(computational_basis['0'], H)
        assert_allclose(result, computational_basis['+'])

    def test_preserves_norm(self, random_state, random_unitary):
        """Unitary should preserve state norm."""
        # Use 2-qubit portion
        state = random_state
        result = apply_unitary(state, random_unitary)
        assert_allclose(np.linalg.norm(result), 1.0, rtol=1e-10)


class TestPauliMatrices:
    """Tests for Pauli matrix properties."""

    def test_hermitian(self, pauli):
        """Pauli matrices should be Hermitian."""
        for name, mat in pauli.items():
            assert_allclose(mat, mat.conj().T,
                          err_msg=f"Pauli {name} not Hermitian")

    def test_unitary(self, pauli):
        """Pauli matrices should be unitary."""
        for name, mat in pauli.items():
            product = mat @ mat.conj().T
            assert_allclose(product, np.eye(2),
                          err_msg=f"Pauli {name} not unitary")

    def test_square_to_identity(self, pauli):
        """Pauli matrices should square to identity."""
        for name, mat in pauli.items():
            assert_allclose(mat @ mat, pauli['I'],
                          err_msg=f"Pauli {name}^2 != I")

    def test_anticommutation(self, pauli):
        """Different Pauli matrices should anticommute."""
        X, Y, Z = pauli['X'], pauli['Y'], pauli['Z']

        # {X, Y} = XY + YX = 0
        assert_allclose(X @ Y + Y @ X, np.zeros((2, 2)), atol=1e-14)
        assert_allclose(Y @ Z + Z @ Y, np.zeros((2, 2)), atol=1e-14)
        assert_allclose(Z @ X + X @ Z, np.zeros((2, 2)), atol=1e-14)

    def test_commutation(self, pauli):
        """Pauli matrices should satisfy [X,Y] = 2iZ, etc."""
        X, Y, Z = pauli['X'], pauli['Y'], pauli['Z']

        assert_allclose(X @ Y - Y @ X, 2j * Z, atol=1e-14)
        assert_allclose(Y @ Z - Z @ Y, 2j * X, atol=1e-14)
        assert_allclose(Z @ X - X @ Z, 2j * Y, atol=1e-14)


class TestTensorProduct:
    """Tests for tensor product operations."""

    def test_two_qubits(self, pauli):
        """Test tensor product of two Pauli matrices."""
        XZ = tensor_product(pauli['X'], pauli['Z'])
        assert XZ.shape == (4, 4)

        # X tensor Z should have specific structure
        expected = np.kron(pauli['X'], pauli['Z'])
        assert_allclose(XZ, expected)

    def test_identity_property(self, pauli):
        """I tensor A = A tensor I in terms of eigenvalues."""
        X = pauli['X']
        I = pauli['I']

        IX = tensor_product(I, X)
        XI = tensor_product(X, I)

        # Both should have same eigenvalues
        eig_IX = np.sort(np.linalg.eigvalsh(IX))
        eig_XI = np.sort(np.linalg.eigvalsh(XI))
        assert_allclose(eig_IX, eig_XI)

    def test_three_qubits(self, pauli):
        """Test tensor product of three matrices."""
        XYZ = tensor_product(pauli['X'], pauli['Y'], pauli['Z'])
        assert XYZ.shape == (8, 8)


class TestPartialTrace:
    """Tests for partial trace operations."""

    def test_pure_product_state(self, computational_basis):
        """Partial trace of product state gives pure state."""
        # |00><00| = |0><0| tensor |0><0|
        state0 = computational_basis['0']
        rho00 = np.outer(np.kron(state0, state0), np.kron(state0, state0).conj())

        # Trace out second qubit
        rho_A = partial_trace(rho00, (2, 2), trace_out=1)

        # Should get |0><0|
        expected = np.outer(state0, state0.conj())
        assert_allclose(rho_A, expected, atol=1e-14)

    def test_bell_state_maximally_mixed(self, bell_states):
        """Partial trace of Bell state should be maximally mixed."""
        phi_plus = bell_states['phi+']
        rho = np.outer(phi_plus, phi_plus.conj())

        # Trace out either qubit
        rho_A = partial_trace(rho, (2, 2), trace_out=1)

        # Should be maximally mixed: I/2
        expected = np.eye(2) / 2
        assert_allclose(rho_A, expected, atol=1e-14)

    def test_trace_preservation(self):
        """Partial trace should preserve total trace."""
        # Create a random density matrix
        rng = np.random.default_rng(42)
        psi = rng.random(4) + 1j * rng.random(4)
        psi = psi / np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

        # Trace out second qubit
        rho_A = partial_trace(rho, (2, 2), trace_out=1)

        # Trace of reduced density matrix should be 1
        assert_allclose(np.trace(rho_A), 1.0, atol=1e-14)


# =============================================================================
# Parameterized Tests
# =============================================================================

@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_identity_dimension(n_qubits):
    """Test identity matrix for various qubit numbers."""
    dim = 2 ** n_qubits
    I = np.eye(dim, dtype=complex)
    assert I.shape == (dim, dim)
    assert_allclose(I @ I, I)


@pytest.mark.parametrize("angle,expected_phase", [
    (0, 1),
    (np.pi/2, 1j),
    (np.pi, -1),
    (3*np.pi/2, -1j),
])
def test_phase_gate(angle, expected_phase):
    """Test phase gate for various angles."""
    P = np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)
    state1 = np.array([0, 1], dtype=complex)
    result = P @ state1
    expected = expected_phase * state1
    assert_allclose(result, expected, atol=1e-14)


# =============================================================================
# Demo Function
# =============================================================================

def demo_tests():
    """Run a demonstration of the tests."""
    print("=" * 60)
    print("Quantum Code Testing Demonstration")
    print("=" * 60)

    print("\nTo run these tests, use:")
    print("  pytest testing_quantum_code.py -v")
    print("\nTo run with coverage:")
    print("  pytest testing_quantum_code.py --cov --cov-report=term-missing")
    print("\nTo run specific test class:")
    print("  pytest testing_quantum_code.py::TestFidelity -v")
    print("\nTo run parameterized tests:")
    print("  pytest testing_quantum_code.py -k 'parametrize' -v")

    # Run a quick sanity check
    print("\n" + "-" * 60)
    print("Quick Sanity Check:")
    print("-" * 60)

    pauli = create_pauli_matrices()
    print(f"Pauli X:\n{pauli['X']}")
    print(f"\nPauli X squared:\n{pauli['X'] @ pauli['X']}")
    print(f"\nIs identity? {np.allclose(pauli['X'] @ pauli['X'], pauli['I'])}")

    state0 = np.array([1, 0], dtype=complex)
    state1 = np.array([0, 1], dtype=complex)
    print(f"\nFidelity(|0>, |0>) = {compute_fidelity(state0, state0)}")
    print(f"Fidelity(|0>, |1>) = {compute_fidelity(state0, state1)}")


if __name__ == "__main__":
    demo_tests()
