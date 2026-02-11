"""
Qiskit Examples for Quantum Computing Research
===============================================

This module provides comprehensive examples of using Qiskit for
quantum algorithm development and simulation.

Author: Quantum Engineering PhD Program
Week 210: Quantum Computing Frameworks

Requirements:
    pip install qiskit qiskit-aer
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def check_qiskit_available():
    """Check if Qiskit is available and return version info."""
    try:
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        return True, qiskit.__version__
    except ImportError as e:
        return False, str(e)


# =============================================================================
# Basic Circuit Construction
# =============================================================================

def create_bell_state():
    """
    Create a Bell state (maximally entangled state).

    Returns
    -------
    QuantumCircuit
        Circuit that creates |Phi+> = (|00> + |11>)/sqrt(2)

    Examples
    --------
    >>> circuit = create_bell_state()
    >>> print(circuit)
         ┌───┐
    q_0: ┤ H ├──■──
         └───┘┌─┴─┐
    q_1: ─────┤ X ├
              └───┘
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, name='Bell')
    qc.h(0)
    qc.cx(0, 1)
    return qc


def create_ghz_state(n_qubits: int = 3):
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 3)

    Returns
    -------
    QuantumCircuit
        Circuit creating (|00...0> + |11...1>)/sqrt(2)
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name=f'GHZ_{n_qubits}')
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def create_parameterized_ansatz(n_qubits: int, n_layers: int = 2):
    """
    Create a hardware-efficient parameterized ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    n_layers : int
        Number of variational layers

    Returns
    -------
    tuple
        (QuantumCircuit, list of Parameters)
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n_params = n_qubits * n_layers * 2
    params = ParameterVector('theta', n_params)

    qc = QuantumCircuit(n_qubits, name='Ansatz')

    p_idx = 0
    for layer in range(n_layers):
        # Rotation layer
        for q in range(n_qubits):
            qc.ry(params[p_idx], q)
            qc.rz(params[p_idx + 1], q)
            p_idx += 2

        # Entangling layer (linear connectivity)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        qc.barrier()

    return qc, list(params)


# =============================================================================
# Simulation
# =============================================================================

def simulate_statevector(circuit):
    """
    Simulate circuit and return statevector.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to simulate

    Returns
    -------
    np.ndarray
        Final statevector
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector

    # Method 1: Using Statevector class (simpler)
    sv = Statevector.from_instruction(circuit)
    return np.array(sv.data)


def simulate_shots(circuit, shots: int = 1024) -> Dict[str, int]:
    """
    Simulate circuit with measurement shots.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to simulate (should include measurements)
    shots : int
        Number of measurement shots

    Returns
    -------
    dict
        Measurement counts
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    # Add measurements if not present
    if circuit.num_clbits == 0:
        circuit = circuit.copy()
        circuit.measure_all()

    simulator = AerSimulator(method='automatic')
    result = simulator.run(circuit, shots=shots).result()
    return result.get_counts()


def simulate_with_noise(circuit, p_depol: float = 0.01, shots: int = 1024):
    """
    Simulate circuit with depolarizing noise.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to simulate
    p_depol : float
        Depolarizing error probability
    shots : int
        Number of measurement shots

    Returns
    -------
    dict
        Noisy measurement counts
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    # Build noise model
    noise_model = NoiseModel()

    # Single-qubit depolarizing
    error_1q = depolarizing_error(p_depol, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h'])

    # Two-qubit depolarizing (typically higher)
    error_2q = depolarizing_error(p_depol * 10, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    # Add measurements if not present
    if circuit.num_clbits == 0:
        circuit = circuit.copy()
        circuit.measure_all()

    # Run noisy simulation
    noisy_sim = AerSimulator(noise_model=noise_model)
    result = noisy_sim.run(circuit, shots=shots).result()
    return result.get_counts()


# =============================================================================
# Quantum Algorithms
# =============================================================================

def vqe_h2_molecule():
    """
    Run VQE for H2 molecule ground state energy.

    Returns
    -------
    dict
        Results including optimal energy and parameters
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from scipy.optimize import minimize

    # H2 Hamiltonian in STO-3G basis (at equilibrium bond length)
    H_coeffs = [
        ("II", -1.0523732457728587),
        ("IZ", 0.39793742484317896),
        ("ZI", -0.39793742484317896),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423142),
    ]
    H = SparsePauliOp.from_list(H_coeffs)
    H_matrix = H.to_matrix()

    # Create ansatz
    n_qubits = 2
    params = ParameterVector('theta', 4)

    ansatz = QuantumCircuit(n_qubits)
    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)
    ansatz.cx(0, 1)
    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)

    # Cost function
    def cost_fn(param_values):
        bound_circuit = ansatz.assign_parameters(
            {p: v for p, v in zip(params, param_values)}
        )
        sv = Statevector.from_instruction(bound_circuit)
        state = np.array(sv.data)
        return np.real(state.conj() @ H_matrix @ state)

    # Optimization
    initial_params = np.random.randn(4) * 0.1

    history = []
    def callback(xk):
        history.append(cost_fn(xk))

    result = minimize(
        cost_fn,
        initial_params,
        method='COBYLA',
        callback=callback,
        options={'maxiter': 200}
    )

    # Exact ground state for comparison
    exact_energy = np.linalg.eigvalsh(H_matrix)[0]

    return {
        'optimal_energy': result.fun,
        'exact_energy': exact_energy,
        'error': abs(result.fun - exact_energy),
        'optimal_params': result.x,
        'n_iterations': len(history),
        'convergence_history': history,
        'success': result.success
    }


def qaoa_maxcut(edges: List[Tuple[int, int]], p: int = 1):
    """
    QAOA for MaxCut problem.

    Parameters
    ----------
    edges : list of tuples
        Graph edges as (node1, node2) pairs
    p : int
        Number of QAOA layers

    Returns
    -------
    dict
        Results including best cut and optimal parameters
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector

    # Find number of nodes
    n_nodes = max(max(e) for e in edges) + 1

    # Create QAOA circuit
    gammas = [Parameter(f'gamma_{i}') for i in range(p)]
    betas = [Parameter(f'beta_{i}') for i in range(p)]

    qc = QuantumCircuit(n_nodes)

    # Initial state: uniform superposition
    qc.h(range(n_nodes))

    for layer in range(p):
        # Cost layer (ZZ interactions)
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2 * gammas[layer], j)
            qc.cx(i, j)

        # Mixer layer
        for i in range(n_nodes):
            qc.rx(2 * betas[layer], i)

    # Cost function
    def cost_fn(params):
        param_dict = {}
        for i in range(p):
            param_dict[gammas[i]] = params[i]
            param_dict[betas[i]] = params[p + i]

        bound_circuit = qc.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound_circuit)
        probs = np.abs(sv.data) ** 2

        # Expected cut value
        expected_cut = 0
        for bitstring_int, prob in enumerate(probs):
            bitstring = format(bitstring_int, f'0{n_nodes}b')
            cut_value = sum(
                1 for i, j in edges
                if bitstring[n_nodes - 1 - i] != bitstring[n_nodes - 1 - j]
            )
            expected_cut += prob * cut_value

        return -expected_cut  # Minimize negative cut

    # Optimize
    from scipy.optimize import minimize
    initial_params = np.random.uniform(0, np.pi, 2 * p)

    result = minimize(
        cost_fn,
        initial_params,
        method='COBYLA',
        options={'maxiter': 200}
    )

    # Get most likely bitstring
    param_dict = {}
    for i in range(p):
        param_dict[gammas[i]] = result.x[i]
        param_dict[betas[i]] = result.x[p + i]

    final_circuit = qc.assign_parameters(param_dict)
    sv = Statevector.from_instruction(final_circuit)
    probs = np.abs(sv.data) ** 2
    best_bitstring = format(np.argmax(probs), f'0{n_nodes}b')

    # Count cut
    best_cut = sum(
        1 for i, j in edges
        if best_bitstring[n_nodes - 1 - i] != best_bitstring[n_nodes - 1 - j]
    )

    return {
        'best_bitstring': best_bitstring,
        'best_cut': best_cut,
        'expected_cut': -result.fun,
        'optimal_params': result.x,
        'success': result.success
    }


def grover_search(oracle_indices: List[int], n_qubits: int):
    """
    Grover's search algorithm.

    Parameters
    ----------
    oracle_indices : list
        Indices to mark (0 to 2^n_qubits - 1)
    n_qubits : int
        Number of qubits

    Returns
    -------
    dict
        Results including measurement probabilities
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    N = 2 ** n_qubits
    M = len(oracle_indices)

    # Optimal number of iterations
    n_iterations = int(np.round(np.pi / 4 * np.sqrt(N / M)))
    n_iterations = max(1, n_iterations)

    # Create circuit
    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    qc.h(range(n_qubits))

    for _ in range(n_iterations):
        # Oracle (mark target states)
        for idx in oracle_indices:
            # Convert index to binary and apply Z to flip phase
            binary = format(idx, f'0{n_qubits}b')
            # Apply X gates to flip 0s to 1s
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(n_qubits - 1 - i)
            # Multi-controlled Z (as H-MCX-H on last qubit)
            if n_qubits > 1:
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)
            else:
                qc.z(0)
            # Undo X gates
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(n_qubits - 1 - i)

        # Diffusion operator
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    # Get probabilities
    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2

    # Probability of finding marked states
    success_prob = sum(probs[idx] for idx in oracle_indices)

    return {
        'probabilities': {format(i, f'0{n_qubits}b'): p for i, p in enumerate(probs) if p > 0.01},
        'success_probability': success_prob,
        'n_iterations': n_iterations,
        'marked_states': [format(idx, f'0{n_qubits}b') for idx in oracle_indices]
    }


# =============================================================================
# Demonstration
# =============================================================================

def demo():
    """Run demonstration of Qiskit examples."""
    print("=" * 60)
    print("Qiskit Examples Demonstration")
    print("=" * 60)

    # Check installation
    available, version = check_qiskit_available()
    if not available:
        print(f"Qiskit not available: {version}")
        print("Install with: pip install qiskit qiskit-aer")
        return

    print(f"\nQiskit version: {version}")

    # Demo 1: Bell state
    print("\n" + "-" * 40)
    print("1. Bell State")
    print("-" * 40)

    bell = create_bell_state()
    print(bell.draw())

    sv = simulate_statevector(bell)
    print(f"Statevector: {sv}")
    print(f"Probabilities: |00>={np.abs(sv[0])**2:.3f}, |11>={np.abs(sv[3])**2:.3f}")

    # Demo 2: Shot-based simulation
    print("\n" + "-" * 40)
    print("2. Shot-based Simulation")
    print("-" * 40)

    counts = simulate_shots(bell, shots=1000)
    print(f"Measurement counts: {counts}")

    # Demo 3: Noisy simulation
    print("\n" + "-" * 40)
    print("3. Noisy Simulation")
    print("-" * 40)

    noisy_counts = simulate_with_noise(bell, p_depol=0.02, shots=1000)
    print(f"Noisy counts: {noisy_counts}")

    # Demo 4: VQE for H2
    print("\n" + "-" * 40)
    print("4. VQE for H2 Molecule")
    print("-" * 40)

    vqe_result = vqe_h2_molecule()
    print(f"Optimal energy: {vqe_result['optimal_energy']:.6f} Ha")
    print(f"Exact energy: {vqe_result['exact_energy']:.6f} Ha")
    print(f"Error: {vqe_result['error']:.6f} Ha")
    print(f"Iterations: {vqe_result['n_iterations']}")

    # Demo 5: QAOA MaxCut
    print("\n" + "-" * 40)
    print("5. QAOA for MaxCut")
    print("-" * 40)

    # Simple triangle graph
    edges = [(0, 1), (1, 2), (0, 2)]
    qaoa_result = qaoa_maxcut(edges, p=2)
    print(f"Graph: Triangle (3 nodes)")
    print(f"Best bitstring: {qaoa_result['best_bitstring']}")
    print(f"Best cut: {qaoa_result['best_cut']}")
    print(f"Expected cut: {qaoa_result['expected_cut']:.3f}")

    # Demo 6: Grover's search
    print("\n" + "-" * 40)
    print("6. Grover's Search")
    print("-" * 40)

    grover_result = grover_search(oracle_indices=[5], n_qubits=3)
    print(f"Searching for state: {grover_result['marked_states']}")
    print(f"Success probability: {grover_result['success_probability']:.3f}")
    print(f"Iterations: {grover_result['n_iterations']}")
    print(f"Top probabilities: {grover_result['probabilities']}")

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
