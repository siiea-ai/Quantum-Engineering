# Week 82: Two-Qubit Gates

## Overview

This week explores two-qubit quantum gates, which are essential for creating entanglement and achieving universal quantum computation. Building on Week 81's single-qubit gates, we now study how qubits can interact through controlled operations, SWAP gates, and other two-qubit unitaries.

## Learning Goals

By the end of this week, you will be able to:

1. **Implement the CNOT gate** and understand its entangling properties
2. **Construct controlled-U gates** for arbitrary single-qubit U
3. **Apply SWAP and √SWAP gates** for qubit exchange
4. **Quantify entangling power** of two-qubit operations
5. **Derive gate identities** for circuit optimization
6. **Compute tensor products** and understand circuit-matrix correspondence

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 568 (Sun) | CNOT Gate | Controlled-NOT, matrix form, entangling property |
| 569 (Mon) | Controlled Gates | CZ, controlled-U, control qubit formulation |
| 570 (Tue) | SWAP and √SWAP | Qubit exchange, partial swap, iSWAP |
| 571 (Wed) | Entangling Power | Bell state creation, entangling capacity |
| 572 (Thu) | Gate Identities | CNOT relations, HXH=Z, circuit equivalences |
| 573 (Fri) | Tensor Products | Circuit-matrix correspondence, Kronecker products |
| 574 (Sat) | Week Review | Comprehensive problems and synthesis |

## Prerequisites

- Week 81: Single-Qubit Gates (complete)
- Tensor products and Kronecker products
- Basic understanding of entanglement
- Linear algebra: 4×4 matrices

## Key Mathematical Framework

### Two-Qubit Computational Basis

$$|00\rangle, |01\rangle, |10\rangle, |11\rangle$$

As column vectors:
$$|00\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, |01\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, |10\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, |11\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

### CNOT Gate

$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

### Bell States

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = \text{CNOT}(H \otimes I)|00\rangle$$

### Universal Gate Set

{Single-qubit gates, CNOT} forms a universal gate set for quantum computation.

## Connections to Quantum Computing

Two-qubit gates enable:
- **Entanglement generation** for quantum advantage
- **Universal computation** when combined with single-qubit gates
- **Quantum error correction** through syndrome extraction
- **Quantum algorithms** (Shor's, Grover's, etc.)

## Resources

- Nielsen & Chuang, "Quantum Computation and Quantum Information," Chapter 4
- Preskill, "Quantum Computing," Lecture Notes Chapter 4
- Qiskit Textbook: Multiple Qubits and Entanglement
- Cirq Documentation: Two-Qubit Gates

## Assessment Focus

- Matrix representation of two-qubit gates
- Circuit diagram interpretation
- Entanglement detection and Bell state preparation
- Gate identity verification
- Python simulation of multi-qubit circuits
