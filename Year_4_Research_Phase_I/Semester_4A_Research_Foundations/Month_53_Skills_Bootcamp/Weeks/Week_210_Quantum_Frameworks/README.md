# Week 210: Quantum Computing Frameworks

## Days 1464-1470 | Year 4, Semester 4A

---

## Overview

This week provides comprehensive training in the three leading quantum computing frameworks: Qiskit, Cirq, and PennyLane. You will gain proficiency in building, simulating, and optimizing quantum circuits across multiple platforms, enabling flexibility in your research workflows.

**Prerequisites:** Basic quantum computing concepts, Python proficiency (Week 209)

**Learning Outcomes:**
- Build and execute quantum circuits in Qiskit, Cirq, and PennyLane
- Understand framework-specific strengths and use cases
- Implement variational quantum algorithms across platforms
- Convert circuits between frameworks
- Interface with real quantum hardware

---

## Daily Schedule

### Day 1464 (Monday): Qiskit Fundamentals

**Morning (3 hours): Qiskit Architecture**
- Qiskit modules: Terra, Aer, IBMQ
- QuantumCircuit construction and manipulation
- Built-in gates and custom unitaries
- Circuit visualization and export

**Afternoon (3 hours): Qiskit Simulation**
- Aer simulators: statevector, qasm, unitary
- Noise models and error simulation
- Transpilation and optimization levels
- Executing on IBM Quantum hardware

**Evening (1 hour): Lab**
- Build a parameterized VQE circuit in Qiskit

---

### Day 1465 (Tuesday): Advanced Qiskit

**Morning (3 hours): Qiskit Algorithms**
- VQE and QAOA implementations
- Quantum Phase Estimation
- Grover's search algorithm
- Custom ansatz design

**Afternoon (3 hours): Qiskit Optimization & Chemistry**
- Qiskit Optimization module
- QUBO and Ising formulations
- Molecular Hamiltonians with qiskit-nature
- Hardware-efficient ansatze

**Evening (1 hour): Lab**
- Simulate H2 molecule ground state energy

---

### Day 1466 (Wednesday): Cirq Fundamentals

**Morning (3 hours): Cirq Architecture**
- Qubits, Gates, and Moments
- Circuit construction patterns
- Native gate sets and decomposition
- GridQubits and device topology

**Afternoon (3 hours): Cirq Simulation & Hardware**
- Cirq simulators: StateVectorSimulator, DensityMatrixSimulator
- Noise channels and error models
- Google quantum hardware integration
- Circuit scheduling and optimization

**Evening (1 hour): Lab**
- Implement quantum random walk in Cirq

---

### Day 1467 (Thursday): Advanced Cirq

**Morning (3 hours): Cirq for NISQ Algorithms**
- Variational circuits in Cirq
- Parameter resolution and sweeps
- Quantum approximate optimization
- Cross-entropy benchmarking (XEB)

**Afternoon (3 hours): TensorFlow Quantum Integration**
- Hybrid classical-quantum models
- Quantum layers in neural networks
- Training quantum circuits with TensorFlow
- Quantum datasets and data encoding

**Evening (1 hour): Lab**
- Build a quantum classifier with TFQ

---

### Day 1468 (Friday): PennyLane Fundamentals

**Morning (3 hours): PennyLane Architecture**
- QNodes and device abstraction
- Automatic differentiation for quantum circuits
- Built-in templates and layers
- Device plugins (qiskit, cirq, lightning)

**Afternoon (3 hours): Quantum Machine Learning**
- Variational quantum classifiers
- Quantum neural networks
- Data encoding strategies
- Cost function design

**Evening (1 hour): Lab**
- Train a quantum classifier in PennyLane

---

### Day 1469 (Saturday): Advanced PennyLane

**Morning (3 hours): Advanced Differentiation**
- Parameter-shift rules
- Adjoint differentiation
- Backpropagation on simulators
- Gradient-free optimization

**Afternoon (3 hours): Research Applications**
- Quantum chemistry with PennyLane
- Quantum kernels for machine learning
- Error mitigation techniques
- Pulse-level programming

**Evening (1 hour): Lab**
- Implement ADAPT-VQE in PennyLane

---

### Day 1470 (Sunday): Framework Comparison & Integration

**Morning (3 hours): Cross-Framework Development**
- Converting circuits between frameworks
- Choosing the right framework for your problem
- Benchmarking performance
- Reproducibility across platforms

**Afternoon (3 hours): Portfolio Development**
- Create a multi-framework quantum library
- Document framework-specific patterns
- Build abstraction layers

**Evening (1 hour): Week Review**
- Framework mastery assessment
- Research application planning

---

## Framework Comparison

| Feature | Qiskit | Cirq | PennyLane |
|---------|--------|------|-----------|
| **Company** | IBM | Google | Xanadu |
| **Primary Focus** | General QC | Superconducting | QML/Autodiff |
| **Hardware** | IBM Quantum | Google Sycamore | Multiple |
| **Autodiff** | Limited | Via TFQ | Native |
| **ML Integration** | Manual | TensorFlow Quantum | PyTorch/JAX |
| **Documentation** | Excellent | Good | Excellent |
| **Community** | Large | Medium | Growing |

---

## Key Resources

| Resource | Description | Link |
|----------|-------------|------|
| Qiskit Textbook | Comprehensive learning | [qiskit.org/textbook](https://qiskit.org/textbook) |
| Cirq Documentation | Official docs | [quantumai.google/cirq](https://quantumai.google/cirq) |
| PennyLane Demos | Interactive tutorials | [pennylane.ai/qml](https://pennylane.ai/qml) |
| TensorFlow Quantum | TFQ tutorials | [tensorflow.org/quantum](https://www.tensorflow.org/quantum) |
| Qiskit Nature | Chemistry module | [qiskit-community.github.io/qiskit-nature](https://qiskit-community.github.io/qiskit-nature) |

---

## Assessment Criteria

By the end of this week, you should be able to:

- [ ] Build and execute quantum circuits in all three frameworks
- [ ] Implement VQE in each framework
- [ ] Use Qiskit for hardware execution on IBM Quantum
- [ ] Build hybrid quantum-classical models with TFQ
- [ ] Use PennyLane's autodiff for gradient-based optimization
- [ ] Convert circuits between frameworks
- [ ] Choose appropriate framework for given research tasks

---

## Hardware Access

### IBM Quantum
- Free account at [quantum-computing.ibm.com](https://quantum-computing.ibm.com)
- Access to real quantum processors
- Use `qiskit-ibm-runtime` for job submission

### Google Quantum AI
- Limited access program for researchers
- Cirq simulation locally available
- Contact Google for hardware access

### Xanadu Cloud
- PennyLane Cloud access
- Photonic quantum hardware
- Free tier available

---

## Connection to Research

These frameworks enable your research by:
1. **Hardware access** - Run experiments on real quantum computers
2. **Algorithm development** - Rapid prototyping of quantum algorithms
3. **Benchmarking** - Compare implementations across platforms
4. **Publication** - Reproducible code for papers
5. **Collaboration** - Work with diverse research groups

---

*Previous Week: Week 209 - Advanced Python*
*Next Week: Week 211 - Simulation & Analysis*
