# Week 107: CSS Codes & Related Constructions

## Overview

**Days:** 743-749 (7 days)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Calderbank-Shor-Steane Codes and Extensions

---

## Status: ✅ COMPLETE

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 743 | Monday | Introduction to CSS Codes | ✅ Complete |
| 744 | Tuesday | Dual Containment Condition | ✅ Complete |
| 745 | Wednesday | CSS Code Examples | ✅ Complete |
| 746 | Thursday | Surface Codes as CSS | ✅ Complete |
| 747 | Friday | Hypergraph Product Codes | ✅ Complete |
| 748 | Saturday | Transversal Gates in CSS | ✅ Complete |
| 749 | Sunday | Week Synthesis | ✅ Complete |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Construct** CSS codes from pairs of classical linear codes
2. **Verify** the dual containment condition C₂⊥ ⊆ C₁
3. **Compute** CSS code parameters [[n, k, d]] from classical codes
4. **Recognize** surface codes as CSS codes on lattices
5. **Understand** hypergraph product constructions
6. **Identify** transversal gates in CSS codes
7. **Design** CSS codes for specific applications
8. **Connect** CSS structure to fault-tolerant operations

---

## Core Concepts

### CSS Code Definition

**Calderbank-Shor-Steane (CSS) codes** are constructed from two classical linear codes C₁ and C₂ satisfying:

$$C_2^\perp \subseteq C_1$$

The resulting quantum code has:
- X-type stabilizers from C₂⊥
- Z-type stabilizers from C₁⊥

### Stabilizer Structure

For CSS code CSS(C₁, C₂):

$$S_X = \{X^{\mathbf{c}} : \mathbf{c} \in C_2^\perp\}$$
$$S_Z = \{Z^{\mathbf{c}} : \mathbf{c} \in C_1^\perp\}$$

The condition C₂⊥ ⊆ C₁ ensures these commute.

### Code Parameters

If C₁ is [n, k₁, d₁] and C₂ is [n, k₂, d₂]:

$$[[n, k_1 + k_2 - n, \min(d_1, d_2^\perp)]]$$

where d₂⊥ is the distance of C₂⊥.

### Parity Check Matrix

CSS codes have a structured parity check matrix:

$$H = \begin{pmatrix} H_X & 0 \\ 0 & H_Z \end{pmatrix}$$

where H_X and H_Z are the parity check matrices of C₂ and C₁ respectively.

### Surface Codes

Surface codes are CSS codes defined on a 2D lattice:
- Vertices → Z stabilizers
- Faces → X stabilizers
- Edges → Qubits

$$[[n, k, d]] \text{ where } n = O(d^2)$$

### Hypergraph Product

For classical codes C₁ and C₂, the hypergraph product yields:

$$HP(C_1, C_2) = [[n_1 n_2 + m_1 m_2, k_1 k_2, \min(d_1, d_2)]]$$

---

## Weekly Breakdown

### Day 743: Introduction to CSS Codes

- Historical motivation: Calderbank-Shor and Steane
- Construction from classical codes
- Dual containment condition
- Simple examples

### Day 744: Dual Containment Condition

- Classical dual codes C⊥
- Verifying C₂⊥ ⊆ C₁
- Self-dual and self-orthogonal codes
- CSS from single code: C⊥ ⊆ C

### Day 745: CSS Code Examples

- Steane [[7,1,3]] code
- Repetition-based CSS codes
- Reed-Muller CSS codes
- Parameter optimization

### Day 746: Surface Codes as CSS

- Toric code structure
- Planar code variants
- Stabilizer from lattice
- Logical operators as cycles

### Day 747: Hypergraph Product Codes

- Hypergraph product construction
- Parameter analysis
- qLDPC codes from HP
- Beyond surface codes

### Day 748: Transversal Gates in CSS

- Transversal X and Z gates
- CNOT between CSS codes
- Limitations and Eastin-Knill theorem
- Magic state injection

### Day 749: Week Synthesis

- Comprehensive review
- Integration problems
- Preparation for Week 108

---

## Key Equations

**CSS Construction:**
$$\boxed{CSS(C_1, C_2): \quad C_2^\perp \subseteq C_1}$$

**CSS Parameters:**
$$\boxed{[[n, k_1 + k_2 - n, \min(d(C_1), d(C_2^\perp))]]}$$

**CSS Stabilizers:**
$$\boxed{S_X = \langle X^{\mathbf{h}} : \mathbf{h} \in C_2^\perp \rangle, \quad S_Z = \langle Z^{\mathbf{h}} : \mathbf{h} \in C_1^\perp \rangle}$$

**Self-Orthogonal CSS:**
$$\boxed{C^\perp \subseteq C \Rightarrow CSS(C, C) \text{ valid}}$$

**Hypergraph Product:**
$$\boxed{HP(C_1, C_2) = [[n_1 n_2 + m_1 m_2, k_1 k_2, \min(d_1, d_2)]]}$$

---

## Computational Skills

```python
import numpy as np
from typing import Tuple, List

class ClassicalCode:
    """Classical linear code over F_2."""

    def __init__(self, parity_check: np.ndarray):
        """
        Initialize with parity check matrix H.

        Parameters:
        -----------
        parity_check : np.ndarray
            m × n binary parity check matrix
        """
        self.H = parity_check % 2
        self.m, self.n = self.H.shape
        self.k = self.n - np.linalg.matrix_rank(self.H)

    def dual(self) -> 'ClassicalCode':
        """Return the dual code C^⊥."""
        # Generator matrix of C is parity check of C^⊥
        G = self._generator_matrix()
        return ClassicalCode(G)

    def _generator_matrix(self) -> np.ndarray:
        """Compute generator matrix from parity check."""
        # Row reduce H to find kernel basis
        H = self.H.copy()
        n = self.n

        # Use null space computation
        from scipy.linalg import null_space
        kernel = null_space(H.astype(float))
        G = np.round(kernel.T) % 2
        return G.astype(int)

    def contains(self, other: 'ClassicalCode') -> bool:
        """Check if self contains other (other ⊆ self)."""
        # other ⊆ self iff every codeword of other is in self
        # Equivalently: H_self · G_other^T = 0
        G_other = other._generator_matrix()
        product = (self.H @ G_other.T) % 2
        return np.all(product == 0)


def css_code(C1: ClassicalCode, C2: ClassicalCode) -> dict:
    """
    Construct CSS code from C1, C2 with C2^⊥ ⊆ C1.

    Returns:
    --------
    dict with keys: 'valid', 'n', 'k', 'H_X', 'H_Z'
    """
    # Check dual containment
    C2_dual = C2.dual()

    if not C1.contains(C2_dual):
        return {'valid': False, 'error': 'Dual containment violated'}

    n = C1.n
    k = C1.k + C2.k - n

    # CSS parity check matrices
    H_X = C2.H  # X stabilizers from C2
    H_Z = C1.H  # Z stabilizers from C1

    return {
        'valid': True,
        'n': n,
        'k': k,
        'H_X': H_X,
        'H_Z': H_Z
    }
```

---

## References

### Primary Sources

- Calderbank & Shor, "Good quantum error-correcting codes exist" (1996)
- Steane, "Error Correcting Codes in Quantum Theory" (1996)
- Kitaev, "Fault-tolerant quantum computation" (2003)

### Key Papers

- Tillich & Zémor, "Quantum LDPC Codes with Positive Rate" (2009)
- Bravyi & Hastings, "Homological Product Codes" (2014)
- Breuckmann & Eberhardt, "Quantum Low-Density Parity-Check Codes" (2021)

### Online Resources

- [Error Correction Zoo - CSS Codes](https://errorcorrectionzoo.org/)
- [Surface Code Tutorial](https://quantum-computing.ibm.com/)

---

## Connections

### Prerequisites (Week 106)

- Graph states and their stabilizers
- Binary symplectic representation
- Parity check matrix formalism

### Leads to (Week 108)

- Color codes
- Reed-Muller quantum codes
- Advanced code families

---

## Summary

CSS codes provide a systematic method to construct quantum error-correcting codes from classical codes. Their special structure—with separate X and Z stabilizers—enables transversal implementation of certain gates and simplifies decoding. Surface codes, the leading candidates for fault-tolerant quantum computing, are CSS codes. Understanding CSS construction is essential for quantum error correction.
