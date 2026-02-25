"""
PennyLane Examples for Quantum Machine Learning
================================================

This module provides comprehensive examples of using PennyLane for
quantum machine learning and variational quantum algorithms.

Author: Quantum Engineering PhD Program
Week 210: Quantum Computing Frameworks

Requirements:
    pip install pennylane pennylane-lightning
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


def check_pennylane_available():
    """Check if PennyLane is available and return version info."""
    try:
        import pennylane as qml
        return True, qml.__version__
    except ImportError as e:
        return False, str(e)


# =============================================================================
# Basic Circuits
# =============================================================================

def create_bell_state_pennylane():
    """
    Create and execute a Bell state circuit in PennyLane.

    Returns
    -------
    np.ndarray
        Final statevector
    """
    import pennylane as qml

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def bell_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    return bell_circuit()


def create_ghz_state_pennylane(n_qubits: int = 3):
    """
    Create a GHZ state in PennyLane.

    Parameters
    ----------
    n_qubits : int
        Number of qubits

    Returns
    -------
    np.ndarray
        GHZ state vector
    """
    import pennylane as qml

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def ghz_circuit():
        qml.Hadamard(wires=0)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    return ghz_circuit()


# =============================================================================
# Automatic Differentiation
# =============================================================================

def demonstrate_gradients():
    """
    Demonstrate automatic differentiation in PennyLane.

    Returns
    -------
    dict
        Results including gradients and Hessians
    """
    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, diff_method='backprop')
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    params = np.array([0.1, 0.2, 0.3], requires_grad=True)

    # Forward pass
    value = circuit(params)

    # Gradient
    grad_fn = qml.grad(circuit)
    gradient = grad_fn(params)

    # Hessian (second derivatives)
    hess_fn = qml.jacobian(qml.grad(circuit))
    hessian = hess_fn(params)

    return {
        'params': params,
        'value': float(value),
        'gradient': np.array(gradient),
        'hessian': np.array(hessian)
    }


def compare_differentiation_methods():
    """
    Compare different differentiation methods in PennyLane.

    Returns
    -------
    dict
        Gradients computed with different methods
    """
    import pennylane as qml
    from pennylane import numpy as np

    results = {}

    # Create device
    dev = qml.device('default.qubit', wires=2)

    def circuit_template(params):
        qml.RY(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[1], wires=1)
        return qml.expval(qml.PauliZ(0))

    params = np.array([0.5, 0.3], requires_grad=True)

    # Method 1: Backpropagation (fastest for simulation)
    @qml.qnode(dev, diff_method='backprop')
    def circuit_backprop(params):
        return circuit_template(params)

    results['backprop'] = qml.grad(circuit_backprop)(params)

    # Method 2: Parameter-shift (hardware compatible)
    @qml.qnode(dev, diff_method='parameter-shift')
    def circuit_psr(params):
        return circuit_template(params)

    results['parameter-shift'] = qml.grad(circuit_psr)(params)

    # Method 3: Adjoint (memory efficient)
    @qml.qnode(dev, diff_method='adjoint')
    def circuit_adjoint(params):
        return circuit_template(params)

    results['adjoint'] = qml.grad(circuit_adjoint)(params)

    # Method 4: Finite differences (numerical)
    @qml.qnode(dev, diff_method='finite-diff')
    def circuit_fd(params):
        return circuit_template(params)

    results['finite-diff'] = qml.grad(circuit_fd)(params)

    return {method: np.array(grad) for method, grad in results.items()}


# =============================================================================
# VQE Implementation
# =============================================================================

def vqe_h2_molecule_pennylane():
    """
    VQE for H2 molecule using PennyLane.

    Returns
    -------
    dict
        Optimization results
    """
    import pennylane as qml
    from pennylane import numpy as np

    # Define device
    dev = qml.device('default.qubit', wires=2)

    # H2 Hamiltonian
    coeffs = np.array([-1.0523, 0.3979, -0.3979, -0.0112, 0.1809])
    obs = [
        qml.Identity(0) @ qml.Identity(1),
        qml.Identity(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.Identity(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1),
    ]
    H = qml.Hamiltonian(coeffs, obs)

    # Ansatz
    @qml.qnode(dev, diff_method='backprop')
    def cost_fn(params):
        # Layer 1
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        # Layer 2
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        return qml.expval(H)

    # Initialize
    params = np.random.randn(4, requires_grad=True) * 0.1

    # Optimize
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    history = []

    for step in range(100):
        params, energy = opt.step_and_cost(cost_fn, params)
        history.append(float(energy))

    # Exact ground state
    H_matrix = qml.matrix(H)
    exact_energy = np.linalg.eigvalsh(H_matrix)[0]

    return {
        'optimal_energy': history[-1],
        'exact_energy': float(exact_energy),
        'error': abs(history[-1] - exact_energy),
        'optimal_params': np.array(params),
        'convergence_history': history
    }


def vqe_with_different_optimizers():
    """
    Compare different optimizers for VQE.

    Returns
    -------
    dict
        Results for each optimizer
    """
    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device('default.qubit', wires=2)

    # Simple Hamiltonian
    H = qml.Hamiltonian(
        [1.0, 0.5],
        [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliX(1)]
    )

    @qml.qnode(dev, diff_method='backprop')
    def cost(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(H)

    results = {}

    optimizers = {
        'GradientDescent': qml.GradientDescentOptimizer(stepsize=0.1),
        'Adam': qml.AdamOptimizer(stepsize=0.1),
        'Adagrad': qml.AdagradOptimizer(stepsize=0.5),
        'Momentum': qml.MomentumOptimizer(stepsize=0.1, momentum=0.9),
    }

    for name, opt in optimizers.items():
        np.random.seed(42)
        params = np.random.randn(2, requires_grad=True) * 0.1
        history = []

        for step in range(50):
            params, energy = opt.step_and_cost(cost, params)
            history.append(float(energy))

        results[name] = {
            'final_energy': history[-1],
            'history': history
        }

    return results


# =============================================================================
# Quantum Machine Learning
# =============================================================================

def quantum_classifier():
    """
    Train a quantum classifier on synthetic data.

    Returns
    -------
    dict
        Training results including accuracy
    """
    import pennylane as qml
    from pennylane import numpy as np

    n_qubits = 4
    dev = qml.device('default.qubit', wires=n_qubits)

    def layer(params, wires):
        """Single variational layer."""
        for i, w in enumerate(wires):
            qml.RY(params[i], wires=w)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    @qml.qnode(dev, diff_method='backprop')
    def circuit(params, x):
        # Encode data
        qml.AngleEmbedding(x, wires=range(n_qubits))

        # Variational layers
        n_layers = len(params) // n_qubits
        for l in range(n_layers):
            layer(params[l * n_qubits:(l + 1) * n_qubits], range(n_qubits))

        return qml.expval(qml.PauliZ(0))

    def cost(params, X, Y):
        """Mean squared error loss."""
        predictions = np.array([circuit(params, x) for x in X])
        return np.mean((predictions - Y) ** 2)

    def accuracy(params, X, Y):
        """Classification accuracy."""
        predictions = np.array([circuit(params, x) for x in X])
        pred_labels = (predictions > 0).astype(float)
        true_labels = (Y > 0).astype(float)
        return np.mean(pred_labels == true_labels)

    # Generate data
    np.random.seed(42)
    n_samples = 50
    X_train = np.random.randn(n_samples, n_qubits)
    # Labels based on sum of features
    Y_train = np.sign(np.sum(X_train, axis=1))

    # Initialize
    n_layers = 3
    params = np.random.randn(n_layers * n_qubits, requires_grad=True) * 0.1

    # Train
    opt = qml.AdamOptimizer(stepsize=0.1)
    history = {'cost': [], 'accuracy': []}

    for step in range(30):
        params, c = opt.step_and_cost(lambda p: cost(p, X_train, Y_train), params)
        acc = accuracy(params, X_train, Y_train)
        history['cost'].append(float(c))
        history['accuracy'].append(float(acc))

    return {
        'final_cost': history['cost'][-1],
        'final_accuracy': history['accuracy'][-1],
        'history': history,
        'optimal_params': np.array(params)
    }


def quantum_kernel():
    """
    Demonstrate quantum kernel for machine learning.

    Returns
    -------
    dict
        Kernel matrix and classification results
    """
    import pennylane as qml
    from pennylane import numpy as np

    n_qubits = 2
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def feature_map(x):
        """Quantum feature map."""
        # First layer of rotations
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i % len(x)], wires=i)

        # Entangling
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Second layer
        for i in range(n_qubits):
            qml.RY(x[(i + 1) % len(x)], wires=i)

        return qml.state()

    def kernel(x1, x2):
        """Quantum kernel: k(x1, x2) = |<phi(x1)|phi(x2)>|^2."""
        state1 = feature_map(x1)
        state2 = feature_map(x2)
        return np.abs(np.vdot(state1, state2)) ** 2

    def compute_kernel_matrix(X):
        """Compute kernel matrix for dataset."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = kernel(X[i], X[j])
                K[j, i] = K[i, j]
        return K

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10
    X = np.random.randn(n_samples, 2)
    Y = np.sign(X[:, 0] * X[:, 1])  # XOR-like pattern

    # Compute kernel matrix
    K = compute_kernel_matrix(X)

    # Simple kernel SVM-like classification
    # Using kernel ridge regression for simplicity
    alpha = 0.1
    weights = np.linalg.solve(K + alpha * np.eye(len(K)), Y)

    def predict(x_new):
        k_vec = np.array([kernel(x_new, x) for x in X])
        return np.sign(np.dot(k_vec, weights))

    # Test accuracy
    predictions = np.array([predict(x) for x in X])
    accuracy_val = np.mean(predictions == Y)

    return {
        'kernel_matrix': K,
        'accuracy': float(accuracy_val),
        'n_samples': n_samples,
        'weights': weights
    }


# =============================================================================
# Advanced Features
# =============================================================================

def noise_model_example():
    """
    Demonstrate noisy simulation in PennyLane.

    Returns
    -------
    dict
        Comparison of ideal vs noisy results
    """
    import pennylane as qml
    from pennylane import numpy as np

    # Ideal simulation
    dev_ideal = qml.device('default.qubit', wires=2)

    @qml.qnode(dev_ideal)
    def ideal_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    ideal_result = float(ideal_circuit())

    # Noisy simulation with mixed state
    dev_noisy = qml.device('default.mixed', wires=2)

    @qml.qnode(dev_noisy)
    def noisy_circuit():
        qml.Hadamard(wires=0)
        qml.DepolarizingChannel(0.1, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.DepolarizingChannel(0.1, wires=0)
        qml.DepolarizingChannel(0.1, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    noisy_result = float(noisy_circuit())

    return {
        'ideal_expectation': ideal_result,
        'noisy_expectation': noisy_result,
        'difference': abs(ideal_result - noisy_result)
    }


def circuit_drawing():
    """
    Demonstrate circuit visualization in PennyLane.

    Returns
    -------
    str
        Circuit diagram as string
    """
    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device('default.qubit', wires=3)

    @qml.qnode(dev)
    def sample_circuit(theta, phi):
        qml.RY(theta, wires=0)
        qml.RZ(phi, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=2)
        qml.CNOT(wires=[1, 2])
        qml.RX(theta, wires=2)
        return qml.expval(qml.PauliZ(0))

    # Draw the circuit
    theta, phi = 0.5, 0.3
    circuit_str = qml.draw(sample_circuit)(theta, phi)

    return circuit_str


# =============================================================================
# Demonstration
# =============================================================================

def demo():
    """Run demonstration of PennyLane examples."""
    print("=" * 60)
    print("PennyLane Examples Demonstration")
    print("=" * 60)

    # Check installation
    available, version = check_pennylane_available()
    if not available:
        print(f"PennyLane not available: {version}")
        print("Install with: pip install pennylane")
        return

    print(f"\nPennyLane version: {version}")

    # Demo 1: Bell state
    print("\n" + "-" * 40)
    print("1. Bell State")
    print("-" * 40)

    bell = create_bell_state_pennylane()
    print(f"Bell state: {bell}")
    print(f"|00>: {np.abs(bell[0])**2:.3f}, |11>: {np.abs(bell[3])**2:.3f}")

    # Demo 2: Automatic differentiation
    print("\n" + "-" * 40)
    print("2. Automatic Differentiation")
    print("-" * 40)

    grad_results = demonstrate_gradients()
    print(f"Parameters: {grad_results['params']}")
    print(f"Value: {grad_results['value']:.6f}")
    print(f"Gradient: {grad_results['gradient']}")
    print(f"Hessian shape: {grad_results['hessian'].shape}")

    # Demo 3: Compare differentiation methods
    print("\n" + "-" * 40)
    print("3. Differentiation Method Comparison")
    print("-" * 40)

    diff_results = compare_differentiation_methods()
    for method, grad in diff_results.items():
        print(f"{method:20s}: {grad}")

    # Demo 4: VQE for H2
    print("\n" + "-" * 40)
    print("4. VQE for H2 Molecule")
    print("-" * 40)

    vqe_result = vqe_h2_molecule_pennylane()
    print(f"Optimal energy: {vqe_result['optimal_energy']:.6f} Ha")
    print(f"Exact energy: {vqe_result['exact_energy']:.6f} Ha")
    print(f"Error: {vqe_result['error']:.6f} Ha")

    # Demo 5: Quantum classifier
    print("\n" + "-" * 40)
    print("5. Quantum Classifier")
    print("-" * 40)

    classifier_result = quantum_classifier()
    print(f"Final cost: {classifier_result['final_cost']:.4f}")
    print(f"Final accuracy: {classifier_result['final_accuracy']:.2%}")

    # Demo 6: Quantum kernel
    print("\n" + "-" * 40)
    print("6. Quantum Kernel")
    print("-" * 40)

    kernel_result = quantum_kernel()
    print(f"Kernel matrix shape: {kernel_result['kernel_matrix'].shape}")
    print(f"Classification accuracy: {kernel_result['accuracy']:.2%}")

    # Demo 7: Noise model
    print("\n" + "-" * 40)
    print("7. Noisy Simulation")
    print("-" * 40)

    noise_result = noise_model_example()
    print(f"Ideal expectation: {noise_result['ideal_expectation']:.4f}")
    print(f"Noisy expectation: {noise_result['noisy_expectation']:.4f}")
    print(f"Difference: {noise_result['difference']:.4f}")

    # Demo 8: Circuit drawing
    print("\n" + "-" * 40)
    print("8. Circuit Visualization")
    print("-" * 40)

    circuit_diagram = circuit_drawing()
    print(circuit_diagram)

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
