# Week 70: Second Quantization

## Overview

**Week 70** | Days 484-490 | The Occupation Number Formalism

This week introduces second quantization—an elegant reformulation of quantum mechanics that treats particle number as a dynamical variable. This framework is essential for many-body physics and quantum field theory.

---

## Daily Schedule

| Day | Topic | Focus |
|-----|-------|-------|
| **484** | Occupation Number Representation | Fock states, number operator |
| **485** | Bosonic Operators | Creation/annihilation, commutation |
| **486** | Fermionic Operators | Anticommutation relations |
| **487** | Field Operators | Position-space formulation |
| **488** | Many-Body Hamiltonians | One-body and two-body terms |
| **489** | Applications | Tight-binding, Hubbard preview |
| **490** | Week Review | Integration and problems |

---

## Key Concepts

### The Idea of Second Quantization

**First quantization:** Wave functions ψ(r₁, r₂, ..., rₙ) for fixed N particles

**Second quantization:** States |n₁, n₂, ...⟩ specifying occupation of each mode

### Why "Second" Quantization?

1. First: Classical → Quantum (operators, commutators)
2. Second: Wave function → Operator (field quantization)

Actually a misnomer—it's an equivalent reformulation, not a further quantization.

### Advantages

- Particle number can vary (grand canonical)
- Symmetrization automatic
- Compact notation for many-body systems
- Natural language for QFT

---

## Key Formulas

### Bosons
$$[a_i, a_j^\dagger] = \delta_{ij}, \quad [a_i, a_j] = 0$$
$$a^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle, \quad a|n\rangle = \sqrt{n}|n-1\rangle$$

### Fermions
$$\{c_i, c_j^\dagger\} = \delta_{ij}, \quad \{c_i, c_j\} = 0$$
$$c^\dagger|0\rangle = |1\rangle, \quad c|1\rangle = |0\rangle$$

### Number Operator
$$\hat{n}_i = a_i^\dagger a_i \text{ (bosons)}, \quad \hat{n}_i = c_i^\dagger c_i \text{ (fermions)}$$

---

## References

- Shankar §10
- Fetter & Walecka Ch. 1-2
- Altland & Simons Ch. 2
