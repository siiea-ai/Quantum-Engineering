# Week 81: Single-Qubit Gates

## Overview

This week introduces the fundamental building blocks of quantum computation: single-qubit gates. These unitary operations on a single qubit form the foundation of all quantum algorithms. We explore the mathematical structure of these gates, their geometric interpretation on the Bloch sphere, and their decomposition properties.

## Learning Goals

By the end of this week, you will be able to:

1. **Apply Pauli gates** (X, Y, Z) and understand their algebraic properties
2. **Use the Hadamard gate** to create and manipulate superposition states
3. **Implement phase gates** (S, T) for Z-axis rotations and phase control
4. **Construct rotation gates** Rx(θ), Ry(θ), Rz(θ) from exponential forms
5. **Visualize any single-qubit gate** as a rotation on the Bloch sphere
6. **Decompose arbitrary single-qubit gates** using Euler angle decomposition

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 561 (Sun) | Pauli Gates | X, Y, Z matrices, eigenvalues, anti-commutation relations |
| 562 (Mon) | Hadamard Gate | Superposition creation, H² = I, basis transformation |
| 563 (Tue) | Phase Gates | S = √Z, T = √S, phase kickback mechanism |
| 564 (Wed) | Rotation Gates | Rx(θ), Ry(θ), Rz(θ), exponential form exp(-iθσ/2) |
| 565 (Thu) | Bloch Sphere Representation | U = exp(iα)Rn̂(θ), axis-angle parameterization |
| 566 (Fri) | Gate Decomposition | Euler angles, ZYZ decomposition, universality |
| 567 (Sat) | Week Review | Comprehensive problems and synthesis |

## Prerequisites

- Density matrices and mixed states (Month 19)
- Entanglement theory fundamentals (Month 20)
- Linear algebra: eigenvalues, matrix exponentials
- Complex numbers and Euler's formula

## Key Mathematical Tools

### Pauli Matrices

$$\sigma_x = X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Rotation Gates

$$R_j(\theta) = e^{-i\theta\sigma_j/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}\sigma_j$$

### General Single-Qubit Gate

$$U = e^{i\alpha} R_z(\beta) R_y(\gamma) R_z(\delta)$$

## Connections to Quantum Computing

Single-qubit gates, combined with entangling two-qubit gates (Week 82), form a universal gate set capable of implementing any quantum algorithm. The efficient decomposition of arbitrary unitaries into native gate sets is crucial for practical quantum computing on real hardware.

## Resources

- Nielsen & Chuang, "Quantum Computation and Quantum Information," Chapter 4
- Preskill, "Quantum Computing," Lecture Notes Chapter 2
- Qiskit Textbook: Single Qubit Gates
- IBM Quantum Experience: Gate Reference

## Assessment Focus

- Matrix multiplication and verification of gate properties
- Bloch sphere visualization and geometric reasoning
- Gate decomposition calculations
- Python simulation of gate operations
