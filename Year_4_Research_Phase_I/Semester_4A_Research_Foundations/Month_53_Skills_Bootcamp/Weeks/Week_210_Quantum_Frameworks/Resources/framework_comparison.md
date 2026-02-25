# Quantum Framework Comparison Guide

## Qiskit vs Cirq vs PennyLane

---

## Quick Reference Table

| Feature | Qiskit | Cirq | PennyLane |
|---------|--------|------|-----------|
| **Developer** | IBM | Google | Xanadu |
| **First Release** | 2017 | 2018 | 2018 |
| **Primary Language** | Python | Python | Python |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Focus** | General QC | NISQ algorithms | QML/Autodiff |

---

## Installation

```bash
# Qiskit (full stack)
pip install qiskit qiskit-aer qiskit-ibm-runtime

# Cirq
pip install cirq

# PennyLane
pip install pennylane pennylane-lightning
```

---

## Circuit Construction Comparison

### Bell State Creation

**Qiskit:**
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
```

**Cirq:**
```python
import cirq

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1)
])
```

**PennyLane:**
```python
import pennylane as qml

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()
```

---

## Parameterized Circuits

### Variational Ansatz

**Qiskit:**
```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta = Parameter('theta')
qc = QuantumCircuit(2)
qc.ry(theta, 0)
qc.cx(0, 1)

# Bind parameters
bound = qc.assign_parameters({theta: 0.5})
```

**Cirq:**
```python
import cirq
import sympy

theta = sympy.Symbol('theta')
q0, q1 = cirq.LineQubit.range(2)

circuit = cirq.Circuit([
    cirq.ry(theta).on(q0),
    cirq.CNOT(q0, q1)
])

# Resolve parameters
resolver = cirq.ParamResolver({'theta': 0.5})
resolved = cirq.resolve_parameters(circuit, resolver)
```

**PennyLane:**
```python
import pennylane as qml

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Parameters are function arguments
result = circuit(0.5)
```

---

## Simulation

### Statevector Simulation

**Qiskit:**
```python
from qiskit.quantum_info import Statevector

sv = Statevector.from_instruction(qc)
print(sv.data)
```

**Cirq:**
```python
simulator = cirq.Simulator()
result = simulator.simulate(circuit)
print(result.final_state_vector)
```

**PennyLane:**
```python
@qml.qnode(dev)
def get_state():
    # ... circuit operations ...
    return qml.state()

state = get_state()
```

### Shot-based Simulation

**Qiskit:**
```python
from qiskit_aer import AerSimulator

qc.measure_all()
sim = AerSimulator()
result = sim.run(qc, shots=1000).result()
counts = result.get_counts()
```

**Cirq:**
```python
circuit.append(cirq.measure(q0, q1, key='result'))
result = simulator.run(circuit, repetitions=1000)
counts = result.histogram(key='result')
```

**PennyLane:**
```python
dev = qml.device('default.qubit', wires=2, shots=1000)

@qml.qnode(dev)
def circuit():
    # ... circuit operations ...
    return qml.counts()

counts = circuit()
```

---

## Gradient Computation

### Automatic Differentiation

**Qiskit:** Limited native support, typically uses finite differences or external optimizers

**Cirq:** Via TensorFlow Quantum
```python
import tensorflow_quantum as tfq
import tensorflow as tf

# Wrap circuit in TFQ layer
circuit_tensor = tfq.convert_to_tensor([circuit])
# Use tf.GradientTape for autodiff
```

**PennyLane:** Native autodiff (strongest feature)
```python
import pennylane as qml
from pennylane import numpy as np

@qml.qnode(dev, diff_method='backprop')
def circuit(params):
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

# Gradient
grad_fn = qml.grad(circuit)
gradient = grad_fn(np.array([0.5]))

# Hessian
hess_fn = qml.jacobian(qml.grad(circuit))
hessian = hess_fn(np.array([0.5]))
```

---

## Noise Simulation

**Qiskit:**
```python
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

noise_model = NoiseModel()
error = depolarizing_error(0.01, 1)
noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

noisy_sim = AerSimulator(noise_model=noise_model)
```

**Cirq:**
```python
circuit.append(cirq.depolarize(p=0.01).on(q0))
# Or use noise model
noise_model = cirq.ConstantQubitNoiseModel(
    qubit_noise_gate=cirq.depolarize(p=0.01)
)
```

**PennyLane:**
```python
dev = qml.device('default.mixed', wires=2)

@qml.qnode(dev)
def noisy_circuit():
    qml.Hadamard(wires=0)
    qml.DepolarizingChannel(0.01, wires=0)
    return qml.expval(qml.PauliZ(0))
```

---

## Hardware Access

| Framework | Hardware Provider | Access Method |
|-----------|-------------------|---------------|
| **Qiskit** | IBM Quantum | `qiskit-ibm-runtime` |
| **Cirq** | Google Quantum AI | Google Cloud (limited) |
| **PennyLane** | Multiple | Device plugins |

### Qiskit Hardware
```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.least_busy(simulator=False)
```

### PennyLane Hardware (via plugins)
```python
# IBM via Qiskit plugin
dev = qml.device('qiskit.ibmq', wires=5, backend='ibmq_manila')

# IonQ
dev = qml.device('ionq.qpu', wires=11)

# Rigetti
dev = qml.device('rigetti.qpu', wires=30)
```

---

## Ecosystem and Extensions

### Qiskit Ecosystem
- **Qiskit Aer**: High-performance simulators
- **Qiskit Nature**: Quantum chemistry
- **Qiskit Optimization**: Combinatorial optimization
- **Qiskit Machine Learning**: QML primitives
- **Qiskit Finance**: Financial applications

### Cirq Ecosystem
- **TensorFlow Quantum**: Hybrid quantum-classical ML
- **Cirq-FT**: Fault-tolerant quantum computing
- **OpenFermion**: Quantum chemistry integration
- **ReCirq**: Research algorithms

### PennyLane Ecosystem
- **pennylane-qchem**: Quantum chemistry
- **pennylane-lightning**: Fast C++ simulator
- **Device plugins**: Qiskit, Cirq, Braket, etc.
- **Built-in QML**: Native machine learning support

---

## When to Use Each Framework

### Use Qiskit When:
- Accessing IBM Quantum hardware
- Need comprehensive quantum chemistry (qiskit-nature)
- Want extensive documentation and tutorials
- Building production applications for IBM systems
- Need noise modeling from real hardware

### Use Cirq When:
- Working with Google quantum hardware
- Building NISQ algorithms for superconducting qubits
- Need fine-grained control over circuit structure
- Using TensorFlow for hybrid models
- Implementing hardware-specific optimizations

### Use PennyLane When:
- Doing quantum machine learning research
- Need automatic differentiation
- Want hardware-agnostic code
- Using PyTorch or JAX for classical ML
- Exploring variational quantum algorithms
- Need to switch between backends easily

---

## Performance Comparison

### Simulation Speed (approximate, varies by circuit)

| Simulator | 20 qubits | Notes |
|-----------|-----------|-------|
| Qiskit Aer | ~1s | Well-optimized C++ |
| Cirq | ~2s | Pure Python, extensible |
| PennyLane default | ~3s | Focus on flexibility |
| PennyLane Lightning | ~1s | C++ backend |

### Memory Usage
- All frameworks: ~16GB for 30 qubits (full statevector)
- Tensor network methods can reduce memory for certain circuits

---

## Converting Between Frameworks

### QASM as Interchange Format

```python
# Qiskit to QASM
qasm_str = qc.qasm()

# QASM to Cirq
from cirq.contrib.qasm_import import circuit_from_qasm
cirq_circuit = circuit_from_qasm(qasm_str)

# QASM to Qiskit
from qiskit.qasm2 import loads
qiskit_circuit = loads(qasm_str)
```

### PennyLane Device Plugins

```python
import pennylane as qml

# Use Qiskit as backend
dev_qiskit = qml.device('qiskit.aer', wires=5)

# Use Cirq as backend
dev_cirq = qml.device('cirq.simulator', wires=5)

# Same QNode works on both
@qml.qnode(dev_qiskit)  # or dev_cirq
def my_circuit(params):
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))
```

---

## Learning Resources

### Qiskit
- [Qiskit Textbook](https://qiskit.org/textbook)
- [IBM Quantum Learning](https://learning.quantum-computing.ibm.com)
- [Qiskit YouTube](https://youtube.com/qiskit)

### Cirq
- [Cirq Documentation](https://quantumai.google/cirq)
- [TensorFlow Quantum Tutorials](https://www.tensorflow.org/quantum/tutorials)

### PennyLane
- [PennyLane Demos](https://pennylane.ai/qml)
- [PennyLane Documentation](https://pennylane.ai/documentation)
- [Xanadu Codebook](https://codebook.xanadu.ai)

---

## Summary Recommendations

| Research Area | Primary Framework | Secondary |
|--------------|-------------------|-----------|
| Quantum Chemistry | Qiskit Nature | PennyLane |
| Quantum ML | PennyLane | Cirq + TFQ |
| NISQ Algorithms | Qiskit or Cirq | PennyLane |
| Hardware Experiments | Framework for your hardware | - |
| Education | Qiskit | PennyLane |
| Variational Algorithms | PennyLane | Qiskit |

---

*Week 210: Quantum Computing Frameworks - Framework Comparison Guide*
