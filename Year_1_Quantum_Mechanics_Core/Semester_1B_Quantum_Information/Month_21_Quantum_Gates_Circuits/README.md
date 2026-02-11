# Month 21: Quantum Gates & Circuits

## Overview
**Days 561-588** | Weeks 81-84 | Year 1, Semester 1B

This month covers the circuit model of quantum computation—the standard framework for describing quantum algorithms using quantum gates as elementary operations.

---

## Learning Objectives

By the end of this month, you will be able to:

1. **Implement** all standard single-qubit gates (Pauli, Hadamard, phase, T)
2. **Apply** two-qubit gates (CNOT, CZ, SWAP, controlled-U)
3. **Decompose** arbitrary unitaries into universal gate sets
4. **Analyze** quantum circuits using tensor network notation
5. **Prove** universality of various gate sets
6. **Design** circuits for specific quantum operations

---

## Month Structure

### Week 81: Single-Qubit Gates (Days 561-567)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 561 | Pauli Gates | X, Y, Z matrices, Bloch sphere rotations |
| 562 | Hadamard Gate | H creates superposition, H² = I |
| 563 | Phase Gates | S, T gates, Z-axis rotations |
| 564 | Rotation Gates | Rx(θ), Ry(θ), Rz(θ), exponential form |
| 565 | Bloch Sphere Visualization | Any U = e^{iα}Rn(θ) |
| 566 | Gate Decomposition | Euler angles, ZYZ decomposition |
| 567 | Week Review | Single-qubit universality |

### Week 82: Two-Qubit Gates (Days 568-574)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 568 | CNOT Gate | Controlled-NOT, entangling gate |
| 569 | Controlled Gates | CZ, controlled-U, control swap |
| 570 | SWAP and √SWAP | Particle exchange, partial swap |
| 571 | Entangling Power | Creating Bell states, CNOT circuits |
| 572 | Gate Identities | CNOT relations, circuit equivalences |
| 573 | Tensor Products | Circuit-matrix correspondence |
| 574 | Week Review | Two-qubit operations |

### Week 83: Circuit Model (Days 575-581)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 575 | Circuit Diagrams | Wire conventions, time ordering |
| 576 | Circuit Composition | Sequential and parallel gates |
| 577 | Measurement in Circuits | Mid-circuit measurement, deferred |
| 578 | Classical Control | If-then gates, feedforward |
| 579 | Circuit Depth and Width | Complexity measures |
| 580 | Circuit Optimization | Gate cancellation, commutation |
| 581 | Week Review | Circuit analysis |

### Week 84: Universality (Days 582-588)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 582 | Universal Gate Sets | {H, T, CNOT}, {Rx, Ry, CNOT} |
| 583 | Solovay-Kitaev Theorem | Efficient approximation |
| 584 | Clifford Gates | Stabilizer formalism preview |
| 585 | Clifford+T | Universal but classically hard |
| 586 | Native Gate Sets | Hardware constraints |
| 587 | Compiling to Hardware | Transpilation basics |
| 588 | Month Review | Comprehensive assessment |

---

## Key Gates Reference

### Single-Qubit Gates
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad
S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \quad
T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

### Two-Qubit Gates
$$CNOT = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&0&1 \\ 0&0&1&0 \end{pmatrix}, \quad
CZ = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&1&0 \\ 0&0&0&-1 \end{pmatrix}$$

### Rotation Gates
$$R_x(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}X$$

---

## Prerequisites

From Months 19-20:
- Density matrices and quantum states
- Tensor products and composite systems
- Entanglement fundamentals

---

## Primary References

- Nielsen & Chuang, Ch. 4 (Quantum Circuits)
- Preskill Lecture Notes, Ch. 5
- Mermin, "Quantum Computer Science", Ch. 1-3

---

## Progress Tracking

| Week | Status | Days Complete |
|------|--------|---------------|
| Week 81: Single-Qubit Gates | 🔄 In Progress | 0/7 |
| Week 82: Two-Qubit Gates | ⬜ Not Started | 0/7 |
| Week 83: Circuit Model | ⬜ Not Started | 0/7 |
| Week 84: Universality | ⬜ Not Started | 0/7 |

**Month Progress:** 0/28 days (0%)

---

*Next: Month 22 — Quantum Algorithms I*
