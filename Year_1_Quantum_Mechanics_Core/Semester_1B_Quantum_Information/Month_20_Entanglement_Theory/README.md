# Month 20: Entanglement Theory

## Overview
**Days 533-560** | Weeks 77-80 | Year 1, Semester 1B

This month provides comprehensive coverage of quantum entanglement—the defining feature of quantum mechanics that enables quantum computing, quantum communication, and quantum cryptography.

---

## Learning Objectives

By the end of this month, you will be able to:

1. **Characterize entanglement** using mathematical criteria (separability, witnesses)
2. **Derive and test Bell inequalities** (CHSH violation)
3. **Quantify entanglement** using entropy and operational measures
4. **Implement quantum protocols** (teleportation, superdense coding)
5. **Analyze entanglement** in multi-qubit systems
6. **Connect theory to experiments** (loophole-free Bell tests)

---

## Month Structure

### Week 77: Entanglement Basics (Days 533-539)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 533 | Separable vs Entangled | Product states, tensor product structure |
| 534 | Bell States | Maximally entangled basis, creation |
| 535 | Entanglement Detection | Witnesses, Schmidt criterion |
| 536 | PPT Criterion Deep Dive | Partial transpose, Peres-Horodecki |
| 537 | Bound Entanglement | NPT vs PPT entanglement |
| 538 | GHZ and W States | Multipartite entanglement |
| 539 | Week Review | Problem solving, synthesis |

### Week 78: Bell Inequalities (Days 540-546)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 540 | EPR Paradox | Einstein's argument, local realism |
| 541 | Bell's Theorem | Local hidden variables, Bell's inequality |
| 542 | CHSH Inequality | Practical formulation, Tsirelson bound |
| 543 | Quantum Violation | Why QM violates classical bounds |
| 544 | Experimental Tests | Aspect, Zeilinger, loopholes |
| 545 | Device-Independent QKD | Bell tests for cryptography |
| 546 | Week Review | Historical context, synthesis |

### Week 79: Entanglement Measures (Days 547-553)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 547 | Von Neumann Entropy | S(ρ) = -Tr(ρ log ρ), properties |
| 548 | Entropy of Entanglement | Pure state measure, Schmidt coefficients |
| 549 | Concurrence | Wootters formula, two-qubit systems |
| 550 | Negativity | Computable measure, PPT connection |
| 551 | Entanglement of Formation | Convex roof extension |
| 552 | Operational Measures | Distillable entanglement, cost |
| 553 | Week Review | Comparing measures, synthesis |

### Week 80: Entanglement Applications (Days 554-560)
| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 554 | Quantum Teleportation | Protocol, fidelity, no-cloning |
| 555 | Superdense Coding | Two classical bits from one qubit |
| 556 | Entanglement Swapping | Creating entanglement at distance |
| 557 | Quantum Repeaters | Long-distance quantum communication |
| 558 | Entanglement Distillation | Purifying noisy entanglement |
| 559 | LOCC Operations | Local operations, classical communication |
| 560 | Month Review | Comprehensive assessment |

---

## Key Formulas

### Separability and Detection
$$\rho_{sep} = \sum_i p_i \rho_i^A \otimes \rho_i^B$$
$$\text{Tr}(W\rho) < 0 \Rightarrow \rho \text{ entangled (witness)}$$

### Bell Inequalities
$$|S| = |\langle AB \rangle - \langle AB' \rangle + \langle A'B \rangle + \langle A'B' \rangle| \leq 2 \text{ (classical)}$$
$$S_{QM} = 2\sqrt{2} \text{ (Tsirelson bound)}$$

### Entanglement Measures
$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$
$$C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4) \text{ (concurrence)}$$
$$\mathcal{N}(\rho) = \frac{\|\rho^{T_B}\|_1 - 1}{2} \text{ (negativity)}$$

### Quantum Protocols
$$|\psi\rangle_{teleport}: \alpha|0\rangle + \beta|1\rangle \xrightarrow{\text{Bell + LOCC}} \text{transmitted}$$
$$\text{Superdense: } 1 \text{ qubit} + 1 \text{ ebit} \rightarrow 2 \text{ classical bits}$$

---

## Primary References

### Textbooks
- Nielsen & Chuang, Ch. 2.6, 12 (canonical treatment)
- Preskill Lecture Notes, Ch. 4 (entanglement theory)
- Wilde, "Quantum Information Theory" Ch. 5-6

### Key Papers
- Bell (1964): "On the Einstein Podolsky Rosen Paradox"
- Aspect et al. (1982): First convincing Bell test
- Bennett et al. (1993): Quantum teleportation
- Wootters (1998): Concurrence formula
- Hensen et al. (2015): Loophole-free Bell test

---

## Computational Tools

```python
# Core imports for Month 20
import numpy as np
from scipy.linalg import sqrtm, logm
import matplotlib.pyplot as plt

# Bell states
def bell_states():
    """Return the four Bell states"""
    phi_plus = np.array([1,0,0,1]) / np.sqrt(2)
    phi_minus = np.array([1,0,0,-1]) / np.sqrt(2)
    psi_plus = np.array([0,1,1,0]) / np.sqrt(2)
    psi_minus = np.array([0,1,-1,0]) / np.sqrt(2)
    return phi_plus, phi_minus, psi_plus, psi_minus

# Von Neumann entropy
def von_neumann_entropy(rho):
    """S(ρ) = -Tr(ρ log₂ ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

# Concurrence for two-qubit states
def concurrence(rho):
    """Wootters concurrence for 2-qubit density matrix"""
    Y = np.array([[0,-1j],[1j,0]])
    YY = np.kron(Y, Y)
    rho_tilde = YY @ rho.conj() @ YY
    R = sqrtm(sqrtm(rho) @ rho_tilde @ sqrtm(rho))
    eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    return max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
```

---

## Progress Tracking

| Week | Status | Days Complete |
|------|--------|---------------|
| Week 77: Entanglement Basics | 🔄 In Progress | 0/7 |
| Week 78: Bell Inequalities | ⬜ Not Started | 0/7 |
| Week 79: Entanglement Measures | ⬜ Not Started | 0/7 |
| Week 80: Entanglement Applications | ⬜ Not Started | 0/7 |

**Month Progress:** 0/28 days (0%)

---

## Prerequisites

From Month 19:
- Density matrix formalism
- Partial trace operations
- Schmidt decomposition
- Quantum channels basics

---

*Next: Month 21 — Quantum Gates & Circuits*
