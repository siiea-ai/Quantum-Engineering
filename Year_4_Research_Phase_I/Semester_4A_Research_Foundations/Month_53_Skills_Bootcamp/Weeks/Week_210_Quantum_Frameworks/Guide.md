# Quantum Computing Frameworks Guide

## Mastering Qiskit, Cirq, and PennyLane

---

## Table of Contents

1. [Qiskit](#1-qiskit)
2. [Cirq](#2-cirq)
3. [PennyLane](#3-pennylane)
4. [Framework Interoperability](#4-framework-interoperability)
5. [Best Practices](#5-best-practices)

---

## 1. Qiskit

### 1.1 Installation and Setup

```bash
# Core installation
pip install qiskit

# With Aer simulator
pip install qiskit-aer

# For IBM hardware access
pip install qiskit-ibm-runtime

# For quantum chemistry
pip install qiskit-nature

# For optimization
pip install qiskit-optimization
```

### 1.2 Circuit Construction

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import numpy as np

# Basic circuit
qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
qc.h(0)          # Hadamard on qubit 0
qc.cx(0, 1)      # CNOT with control=0, target=1
qc.measure([0, 1], [0, 1])

# With explicit registers
qr = QuantumRegister(3, 'q')
cr = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qr, cr)

# Parameterized circuit
theta = Parameter('θ')
phi = Parameter('φ')

qc = QuantumCircuit(2)
qc.ry(theta, 0)
qc.rz(phi, 0)
qc.cx(0, 1)
qc.ry(theta, 1)

# Bind parameters
bound_circuit = qc.assign_parameters({theta: np.pi/4, phi: np.pi/2})

# Visualization
print(qc.draw())  # Text drawing
qc.draw('mpl')    # Matplotlib figure
```

### 1.3 Gates Reference

```python
from qiskit import QuantumCircuit
import numpy as np

qc = QuantumCircuit(3)

# Single-qubit gates
qc.x(0)           # Pauli X (NOT)
qc.y(0)           # Pauli Y
qc.z(0)           # Pauli Z
qc.h(0)           # Hadamard
qc.s(0)           # S gate (sqrt(Z))
qc.t(0)           # T gate (sqrt(S))
qc.sdg(0)         # S-dagger
qc.tdg(0)         # T-dagger

# Rotation gates
qc.rx(np.pi/4, 0)  # Rotation around X
qc.ry(np.pi/4, 0)  # Rotation around Y
qc.rz(np.pi/4, 0)  # Rotation around Z
qc.p(np.pi/4, 0)   # Phase gate
qc.u(np.pi/4, np.pi/4, np.pi/4, 0)  # General U3 gate

# Two-qubit gates
qc.cx(0, 1)       # CNOT
qc.cy(0, 1)       # Controlled-Y
qc.cz(0, 1)       # Controlled-Z
qc.swap(0, 1)     # SWAP
qc.iswap(0, 1)    # iSWAP
qc.cp(np.pi/4, 0, 1)  # Controlled-Phase

# Three-qubit gates
qc.ccx(0, 1, 2)   # Toffoli (CCX)
qc.cswap(0, 1, 2) # Fredkin (CSWAP)

# Custom unitary
U = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
qc.unitary(U, [0], label='custom')
```

### 1.4 Simulation with Aer

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# Create circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Statevector simulation
sv_sim = AerSimulator(method='statevector')
qc_sv = qc.copy()
qc_sv.save_statevector()
result = sv_sim.run(qc_sv).result()
statevector = result.get_statevector()
print(f"Statevector: {statevector}")

# Using Statevector class directly
sv = Statevector.from_instruction(qc)
print(f"Probabilities: {sv.probabilities_dict()}")

# Shot-based simulation
qc_meas = qc.copy()
qc_meas.measure_all()
qasm_sim = AerSimulator(method='automatic')
result = qasm_sim.run(qc_meas, shots=1024).result()
counts = result.get_counts()
print(f"Counts: {counts}")

# Density matrix simulation
dm_sim = AerSimulator(method='density_matrix')
qc_dm = qc.copy()
qc_dm.save_density_matrix()
result = dm_sim.run(qc_dm).result()
dm = result.data()['density_matrix']
```

### 1.5 Noise Simulation

```python
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

# Build noise model
noise_model = NoiseModel()

# Single-qubit depolarizing error
p_1q = 0.001  # Error probability
error_1q = depolarizing_error(p_1q, 1)
noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])

# Two-qubit depolarizing error
p_2q = 0.01
error_2q = depolarizing_error(p_2q, 2)
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# Thermal relaxation
T1 = 50e3  # T1 in ns
T2 = 70e3  # T2 in ns
gate_time = 50  # Gate time in ns
error_thermal = thermal_relaxation_error(T1, T2, gate_time)
noise_model.add_all_qubit_quantum_error(error_thermal, ['u1', 'u2', 'u3'])

# Run noisy simulation
noisy_sim = AerSimulator(noise_model=noise_model)
result = noisy_sim.run(qc_meas, shots=1024).result()
noisy_counts = result.get_counts()
```

### 1.6 IBM Quantum Hardware

```python
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator

# Save account (once)
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

# Load service
service = QiskitRuntimeService()

# List available backends
backends = service.backends()
print([b.name for b in backends])

# Select backend
backend = service.least_busy(simulator=False, min_num_qubits=5)
print(f"Using backend: {backend.name}")

# Run with Sampler primitive
sampler = Sampler(backend)
job = sampler.run([qc_meas])
result = job.result()

# Run with Estimator primitive (for expectation values)
from qiskit.quantum_info import SparsePauliOp
observable = SparsePauliOp.from_list([("ZZ", 1.0)])

estimator = Estimator(backend)
job = estimator.run([(qc, observable)])
result = job.result()
expectation = result[0].data.evs
```

### 1.7 VQE Implementation

```python
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator

# Define Hamiltonian (e.g., H2 molecule)
H = SparsePauliOp.from_list([
    ("II", -1.0523),
    ("IZ", 0.3979),
    ("ZI", -0.3979),
    ("ZZ", -0.0112),
    ("XX", 0.1809)
])

# Create hardware-efficient ansatz
n_qubits = 2
n_layers = 2
params = ParameterVector('θ', n_qubits * n_layers * 2)

ansatz = QuantumCircuit(n_qubits)
p_idx = 0
for layer in range(n_layers):
    for q in range(n_qubits):
        ansatz.ry(params[p_idx], q)
        ansatz.rz(params[p_idx + 1], q)
        p_idx += 2
    for q in range(n_qubits - 1):
        ansatz.cx(q, q + 1)

# Run VQE
estimator = Estimator()
optimizer = COBYLA(maxiter=500)

vqe = VQE(estimator, ansatz, optimizer)
result = vqe.compute_minimum_eigenvalue(H)

print(f"Ground state energy: {result.optimal_value:.6f}")
print(f"Optimal parameters: {result.optimal_parameters}")
```

---

## 2. Cirq

### 2.1 Installation

```bash
pip install cirq
pip install tensorflow-quantum  # Optional: for TFQ
```

### 2.2 Circuit Construction

```python
import cirq
import numpy as np

# Define qubits
q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)

# Or use named qubits
qa = cirq.NamedQubit('a')
qb = cirq.NamedQubit('b')

# Grid qubits (for hardware topology)
grid_qubits = cirq.GridQubit.rect(3, 3)  # 3x3 grid

# Build circuit
circuit = cirq.Circuit()
circuit.append(cirq.H(q0))
circuit.append(cirq.CNOT(q0, q1))
circuit.append(cirq.measure(q0, q1, key='result'))

# One-liner with moments
circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
])

# Using moments explicitly for parallel operations
circuit = cirq.Circuit([
    cirq.Moment([cirq.H(q0), cirq.H(q1)]),  # Parallel Hadamards
    cirq.Moment([cirq.CNOT(q0, q1)]),
    cirq.Moment([cirq.measure(q0, q1, key='result')])
])

print(circuit)
```

### 2.3 Parameterized Circuits

```python
import cirq
import sympy
import numpy as np

q0, q1 = cirq.LineQubit.range(2)

# Symbolic parameters
theta = sympy.Symbol('theta')
phi = sympy.Symbol('phi')

circuit = cirq.Circuit([
    cirq.ry(theta).on(q0),
    cirq.rz(phi).on(q0),
    cirq.CNOT(q0, q1),
    cirq.ry(theta).on(q1),
])

# Resolve parameters
resolver = cirq.ParamResolver({'theta': np.pi/4, 'phi': np.pi/2})
resolved_circuit = cirq.resolve_parameters(circuit, resolver)

# Parameter sweep
sweep = cirq.Linspace('theta', 0, np.pi, 10)
```

### 2.4 Gates Reference

```python
import cirq
import numpy as np

q0, q1, q2 = cirq.LineQubit.range(3)

# Single-qubit gates
cirq.X(q0)
cirq.Y(q0)
cirq.Z(q0)
cirq.H(q0)
cirq.S(q0)
cirq.T(q0)

# Rotation gates
cirq.rx(np.pi/4).on(q0)
cirq.ry(np.pi/4).on(q0)
cirq.rz(np.pi/4).on(q0)
cirq.ZPowGate(exponent=0.5).on(q0)  # sqrt(Z)

# Two-qubit gates
cirq.CNOT(q0, q1)
cirq.CZ(q0, q1)
cirq.SWAP(q0, q1)
cirq.ISWAP(q0, q1)
cirq.ZZPowGate(exponent=0.5).on(q0, q1)
cirq.FSim(theta=np.pi/4, phi=np.pi/6).on(q0, q1)

# Three-qubit gates
cirq.TOFFOLI(q0, q1, q2)
cirq.FREDKIN(q0, q1, q2)

# Custom gate
custom_unitary = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
cirq.MatrixGate(custom_unitary).on(q0)
```

### 2.5 Simulation

```python
import cirq
import numpy as np

q0, q1 = cirq.LineQubit.range(2)

circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1)
])

# Statevector simulation
simulator = cirq.Simulator()
result = simulator.simulate(circuit)
print(f"Statevector: {result.final_state_vector}")

# Density matrix simulation
dm_simulator = cirq.DensityMatrixSimulator()
result = dm_simulator.simulate(circuit)
print(f"Density matrix:\n{result.final_density_matrix}")

# Shot-based simulation
circuit_with_meas = circuit + cirq.measure(q0, q1, key='result')
result = simulator.run(circuit_with_meas, repetitions=1000)
print(f"Histogram: {result.histogram(key='result')}")

# Parameter sweep simulation
theta = sympy.Symbol('theta')
param_circuit = cirq.Circuit([cirq.ry(theta).on(q0)])

sweep = cirq.Linspace('theta', 0, np.pi, 5)
results = simulator.simulate_sweep(param_circuit, sweep)
for i, result in enumerate(results):
    print(f"theta={i}: state={result.final_state_vector}")
```

### 2.6 Noise Simulation

```python
import cirq

q0, q1 = cirq.LineQubit.range(2)

# Create noisy circuit
circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.depolarize(p=0.01).on(q0),  # Insert noise
    cirq.CNOT(q0, q1),
    cirq.depolarize(p=0.02).on_each(q0, q1),
    cirq.measure(q0, q1, key='result')
])

# Or use a noise model
noise_model = cirq.ConstantQubitNoiseModel(
    qubit_noise_gate=cirq.depolarize(p=0.01)
)

noisy_simulator = cirq.DensityMatrixSimulator(noise=noise_model)
result = noisy_simulator.run(circuit, repetitions=1000)

# Available noise channels
cirq.depolarize(p=0.01)              # Depolarizing
cirq.amplitude_damp(gamma=0.01)      # Amplitude damping
cirq.phase_damp(gamma=0.01)          # Phase damping
cirq.bit_flip(p=0.01)                # Bit flip
cirq.phase_flip(p=0.01)              # Phase flip
cirq.asymmetric_depolarize(p_x=0.01, p_y=0.01, p_z=0.01)
```

### 2.7 VQE in Cirq

```python
import cirq
import sympy
import numpy as np
from scipy.optimize import minimize

# Define qubits
qubits = cirq.LineQubit.range(2)

# Define parameterized ansatz
def create_ansatz(qubits, n_layers=2):
    params = []
    circuit = cirq.Circuit()

    for layer in range(n_layers):
        layer_params = []
        for q in qubits:
            theta = sympy.Symbol(f'theta_{layer}_{q}')
            phi = sympy.Symbol(f'phi_{layer}_{q}')
            layer_params.extend([theta, phi])
            circuit.append([cirq.ry(theta).on(q), cirq.rz(phi).on(q)])

        # Entangling layer
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

        params.extend(layer_params)

    return circuit, params

# Define Hamiltonian as Pauli string coefficients
hamiltonian = {
    (): -1.0523,
    ((0, 'Z'),): 0.3979,
    ((1, 'Z'),): -0.3979,
    ((0, 'Z'), (1, 'Z')): -0.0112,
    ((0, 'X'), (1, 'X')): 0.1809,
}

def pauli_string_to_circuit(qubits, pauli_string):
    """Convert Pauli string to measurement basis rotation."""
    ops = []
    for qubit_idx, pauli in pauli_string:
        if pauli == 'X':
            ops.append(cirq.H(qubits[qubit_idx]))
        elif pauli == 'Y':
            ops.append(cirq.rx(np.pi/2).on(qubits[qubit_idx]))
    return ops

def compute_expectation(circuit, params, param_values, hamiltonian, qubits, n_shots=1000):
    """Compute expectation value of Hamiltonian."""
    simulator = cirq.Simulator()
    resolver = cirq.ParamResolver({str(p): v for p, v in zip(params, param_values)})

    expectation = 0.0

    for pauli_string, coeff in hamiltonian.items():
        if not pauli_string:
            expectation += coeff
            continue

        # Add basis rotation and measurement
        meas_circuit = circuit.copy()
        meas_ops = pauli_string_to_circuit(qubits, pauli_string)
        meas_circuit.append(meas_ops)
        measured_qubits = [qubits[q] for q, _ in pauli_string]
        meas_circuit.append(cirq.measure(*measured_qubits, key='m'))

        # Run and compute expectation
        result = simulator.run(meas_circuit, resolver, repetitions=n_shots)
        bits = result.measurements['m']

        # Compute parity
        parities = np.prod(1 - 2*bits, axis=1)
        expectation += coeff * np.mean(parities)

    return expectation

# Create ansatz and optimize
ansatz, params = create_ansatz(qubits, n_layers=2)
initial_params = np.random.randn(len(params)) * 0.1

def cost_function(param_values):
    return compute_expectation(ansatz, params, param_values, hamiltonian, qubits)

result = minimize(cost_function, initial_params, method='COBYLA',
                 options={'maxiter': 500})

print(f"Ground state energy: {result.fun:.6f}")
```

---

## 3. PennyLane

### 3.1 Installation

```bash
# Core installation
pip install pennylane

# Device plugins
pip install pennylane-qiskit
pip install pennylane-cirq
pip install pennylane-lightning  # Fast C++ simulator
```

### 3.2 Basic Concepts

```python
import pennylane as qml
import numpy as np

# Create a device
dev = qml.device('default.qubit', wires=2)

# Define a quantum function (QNode)
@qml.qnode(dev)
def circuit(theta, phi):
    qml.RY(theta, wires=0)
    qml.RZ(phi, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Execute
result = circuit(np.pi/4, np.pi/2)
print(f"Expectation value: {result}")

# Automatic differentiation
grad_fn = qml.grad(circuit)
gradients = grad_fn(np.pi/4, np.pi/2)
print(f"Gradients: {gradients}")
```

### 3.3 Gates Reference

```python
import pennylane as qml

# Single-qubit gates
qml.PauliX(wires=0)
qml.PauliY(wires=0)
qml.PauliZ(wires=0)
qml.Hadamard(wires=0)
qml.S(wires=0)
qml.T(wires=0)

# Rotation gates
qml.RX(theta, wires=0)
qml.RY(theta, wires=0)
qml.RZ(theta, wires=0)
qml.Rot(phi, theta, omega, wires=0)  # General rotation
qml.PhaseShift(phi, wires=0)

# Two-qubit gates
qml.CNOT(wires=[0, 1])
qml.CY(wires=[0, 1])
qml.CZ(wires=[0, 1])
qml.SWAP(wires=[0, 1])
qml.IsingXX(phi, wires=[0, 1])
qml.IsingYY(phi, wires=[0, 1])
qml.IsingZZ(phi, wires=[0, 1])
qml.CRot(phi, theta, omega, wires=[0, 1])

# Three-qubit gates
qml.Toffoli(wires=[0, 1, 2])
qml.CSWAP(wires=[0, 1, 2])

# Custom unitary
U = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
qml.QubitUnitary(U, wires=0)
```

### 3.4 Measurements

```python
import pennylane as qml

dev = qml.device('default.qubit', wires=2, shots=1000)

@qml.qnode(dev)
def circuit_measurements():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    # Different measurement types
    return [
        qml.expval(qml.PauliZ(0)),              # Expectation value
        qml.var(qml.PauliZ(0)),                  # Variance
        qml.probs(wires=[0, 1]),                 # Probabilities
        qml.sample(qml.PauliZ(0)),               # Individual samples
        qml.counts(wires=[0, 1]),                # Count dictionary
    ]

# Single measurement
@qml.qnode(dev)
def get_expval():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# State and density matrix (analytical simulation)
dev_exact = qml.device('default.qubit', wires=2)

@qml.qnode(dev_exact)
def get_state():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

@qml.qnode(dev_exact)
def get_density_matrix():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.density_matrix(wires=[0, 1])
```

### 3.5 Automatic Differentiation

```python
import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, diff_method='backprop')  # Fast for simulation
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

params = np.array([0.1, 0.2, 0.3])

# Gradient
grad_fn = qml.grad(circuit)
gradients = grad_fn(params)
print(f"Gradients: {gradients}")

# Jacobian for multiple outputs
@qml.qnode(dev)
def multi_output(params):
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=1)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

jacobian_fn = qml.jacobian(multi_output)
jacobian = jacobian_fn(params[:2])

# Different differentiation methods
@qml.qnode(dev, diff_method='parameter-shift')   # Hardware compatible
def circuit_ps(params):
    pass

@qml.qnode(dev, diff_method='adjoint')           # Memory efficient
def circuit_adj(params):
    pass

@qml.qnode(dev, diff_method='finite-diff')       # Numerical
def circuit_fd(params):
    pass
```

### 3.6 Templates and Layers

```python
import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def circuit_with_templates(params):
    # Data encoding
    qml.AngleEmbedding(params[:4], wires=range(4))

    # Or amplitude embedding
    # qml.AmplitudeEmbedding(params, wires=range(4), normalize=True)

    # Variational layers
    qml.StronglyEntanglingLayers(params[4:].reshape(2, 4, 3), wires=range(4))

    # Or basic entangler
    # qml.BasicEntanglerLayers(params, wires=range(4))

    return qml.expval(qml.PauliZ(0))

# Using templates directly
n_layers = 2
n_qubits = 4

# Shape for StronglyEntanglingLayers: (n_layers, n_qubits, 3)
weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
weights = np.random.randn(*weight_shape)

@qml.qnode(dev)
def variational_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(4))
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return qml.expval(qml.PauliZ(0))
```

### 3.7 VQE in PennyLane

```python
import pennylane as qml
from pennylane import numpy as np

# Define device
dev = qml.device('default.qubit', wires=2)

# Define Hamiltonian
coeffs = [-1.0523, 0.3979, -0.3979, -0.0112, 0.1809]
obs = [
    qml.Identity(0) @ qml.Identity(1),
    qml.Identity(0) @ qml.PauliZ(1),
    qml.PauliZ(0) @ qml.Identity(1),
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1),
]
H = qml.Hamiltonian(coeffs, obs)

# Define ansatz
def ansatz(params, wires):
    n_layers = len(params) // (len(wires) * 2)
    p_idx = 0
    for layer in range(n_layers):
        for w in wires:
            qml.RY(params[p_idx], wires=w)
            qml.RZ(params[p_idx + 1], wires=w)
            p_idx += 2
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])

# Define cost function
@qml.qnode(dev, diff_method='backprop')
def cost_fn(params):
    ansatz(params, wires=[0, 1])
    return qml.expval(H)

# Initialize parameters
n_layers = 2
n_qubits = 2
n_params = n_layers * n_qubits * 2
params = np.random.randn(n_params, requires_grad=True) * 0.1

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.4)

for step in range(100):
    params, energy = opt.step_and_cost(cost_fn, params)
    if step % 20 == 0:
        print(f"Step {step}: Energy = {energy:.6f}")

print(f"\nFinal energy: {cost_fn(params):.6f}")
print(f"Exact ground state: {np.linalg.eigvalsh(qml.matrix(H))[0]:.6f}")
```

### 3.8 Quantum Machine Learning

```python
import pennylane as qml
from pennylane import numpy as np

# Quantum classifier
n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

def layer(params, wires):
    """Single variational layer."""
    for i, w in enumerate(wires):
        qml.RY(params[i], wires=w)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

@qml.qnode(dev, diff_method='backprop')
def quantum_classifier(params, x):
    """Quantum classifier circuit."""
    # Encode input
    qml.AngleEmbedding(x, wires=range(n_qubits))

    # Variational layers
    n_layers = len(params) // n_qubits
    for l in range(n_layers):
        layer(params[l*n_qubits:(l+1)*n_qubits], range(n_qubits))

    return qml.expval(qml.PauliZ(0))

def cost(params, X, Y):
    """Binary cross-entropy cost."""
    predictions = np.array([quantum_classifier(params, x) for x in X])
    # Map from [-1, 1] to [0, 1]
    predictions = (predictions + 1) / 2
    # Clip for numerical stability
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    return -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))

def accuracy(params, X, Y):
    """Compute classification accuracy."""
    predictions = np.array([quantum_classifier(params, x) for x in X])
    predicted_labels = (predictions > 0).astype(int)
    return np.mean(predicted_labels == Y)

# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(50, n_qubits)
Y_train = (np.sum(X_train, axis=1) > 0).astype(int)

# Initialize and train
n_layers = 3
params = np.random.randn(n_layers * n_qubits, requires_grad=True) * 0.1
opt = qml.AdamOptimizer(stepsize=0.1)

for step in range(50):
    params, c = opt.step_and_cost(lambda p: cost(p, X_train, Y_train), params)
    if step % 10 == 0:
        acc = accuracy(params, X_train, Y_train)
        print(f"Step {step}: Cost = {c:.4f}, Accuracy = {acc:.2%}")
```

---

## 4. Framework Interoperability

### 4.1 Qiskit to Cirq

```python
from qiskit import QuantumCircuit
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

# Create Qiskit circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Export to QASM
qasm_str = qc.qasm()

# Import to Cirq
cirq_circuit = circuit_from_qasm(qasm_str)
print(cirq_circuit)
```

### 4.2 Cirq to Qiskit

```python
import cirq
from qiskit import QuantumCircuit
from qiskit.qasm2 import loads

# Create Cirq circuit
q0, q1 = cirq.LineQubit.range(2)
cirq_circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1)
])

# Export to QASM
qasm_str = cirq.qasm(cirq_circuit)

# Import to Qiskit
qiskit_circuit = loads(qasm_str)
print(qiskit_circuit.draw())
```

### 4.3 PennyLane Device Plugins

```python
import pennylane as qml

# Use Qiskit backend
dev_qiskit = qml.device('qiskit.aer', wires=2)

# Use Cirq backend
dev_cirq = qml.device('cirq.simulator', wires=2)

# Use Lightning (fast C++ simulator)
dev_lightning = qml.device('lightning.qubit', wires=2)

# Same circuit works on all devices
@qml.qnode(dev_qiskit)
def circuit(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
```

---

## 5. Best Practices

### 5.1 Choosing the Right Framework

| Use Case | Recommended Framework |
|----------|----------------------|
| IBM Quantum hardware | Qiskit |
| Google hardware | Cirq |
| Gradient-based optimization | PennyLane |
| Quantum chemistry | Qiskit Nature or PennyLane |
| Hybrid quantum-classical ML | PennyLane or TFQ |
| Educational purposes | Qiskit (best docs) |
| Production deployment | Depends on hardware |

### 5.2 Performance Tips

```python
# 1. Use appropriate simulator
# - default.qubit: Good for small circuits, autodiff
# - lightning.qubit: Fast for medium circuits
# - Aer: Good noise modeling

# 2. Batch parameter evaluation
import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

# Vectorized execution
params_batch = np.random.randn(100, 1)
results = [circuit(p) for p in params_batch]  # Slow

# Better: use qml.batch_params
# Or design circuit to handle batches

# 3. Choose appropriate diff_method
# - 'backprop': Fast for simulators
# - 'adjoint': Memory efficient for deep circuits
# - 'parameter-shift': Required for hardware
```

### 5.3 Debugging Circuits

```python
# Qiskit - visualize
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print(qc.draw())
qc.draw('mpl')  # Matplotlib

# Cirq - visualize
import cirq
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])
print(circuit)
# SVG in notebooks: cirq.contrib.svg.SVGCircuit(circuit)

# PennyLane - visualize
import pennylane as qml
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

# Draw circuit
print(qml.draw(circuit)())
# Or: qml.draw_mpl(circuit)()
```

### 5.4 Testing Quantum Code

```python
import numpy as np

def test_bell_state():
    """Test that circuit creates Bell state."""
    # Expected Bell state
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)

    # Get state from circuit
    state = get_statevector_from_circuit()

    # Compare with tolerance
    np.testing.assert_allclose(np.abs(state), np.abs(expected), atol=1e-10)

def test_unitary():
    """Test that circuit is unitary."""
    matrix = get_circuit_matrix()
    identity = matrix @ matrix.conj().T
    np.testing.assert_allclose(identity, np.eye(len(matrix)), atol=1e-10)
```

---

## Quick Reference

### Framework Cheat Sheet

```python
# Bell state in each framework

# Qiskit
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Cirq
import cirq
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

# PennyLane
import pennylane as qml
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def bell_state():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()
```

---

*This guide covers the essentials of Qiskit, Cirq, and PennyLane for quantum computing research. Each framework has its strengths - choose based on your specific research needs.*
