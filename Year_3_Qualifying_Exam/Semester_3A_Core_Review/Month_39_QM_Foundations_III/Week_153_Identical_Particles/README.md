# Week 153: Identical Particles

## Overview

**Days:** 1065-1071
**Theme:** Permutation symmetry, bosons, fermions, spin-statistics, and second quantization
**Prerequisites:** Angular momentum, spin, multi-particle quantum mechanics

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Construct** properly symmetrized wavefunctions for bosons and fermions
2. **Apply** the Pauli exclusion principle to determine allowed quantum states
3. **Explain** the spin-statistics theorem and its physical basis
4. **Build** Slater determinants for multi-electron systems
5. **Use** second quantization formalism to represent many-body Hamiltonians
6. **Calculate** the ground state energy of helium using exchange symmetry
7. **Distinguish** direct and exchange integrals in two-electron problems

---

## Daily Schedule

| Day | Focus | Key Concepts |
|-----|-------|--------------|
| **1065** | Permutation Symmetry | Exchange operator, symmetric/antisymmetric states |
| **1066** | Bosons & Fermions | Spin-statistics theorem, Pauli exclusion |
| **1067** | Slater Determinants | Multi-electron wavefunctions, antisymmetrization |
| **1068** | Second Quantization I | Creation/annihilation operators, commutation |
| **1069** | Second Quantization II | Fock space, field operators, Hamiltonians |
| **1070** | Helium Atom | Ground state, exchange energy, excited states |
| **1071** | Review & Oral Practice | Problem synthesis, oral exam preparation |

---

## Core Concepts

### 1. Permutation Symmetry

In quantum mechanics, identical particles are truly indistinguishable - there is no physical operation that can label them. This has profound consequences.

**Exchange Operator:**
$$\hat{P}_{12}\Psi(\mathbf{r}_1, s_1; \mathbf{r}_2, s_2) = \Psi(\mathbf{r}_2, s_2; \mathbf{r}_1, s_1)$$

Properties:
- $\hat{P}_{12}^2 = \mathbf{1}$ (applying twice returns original)
- Eigenvalues: $+1$ (symmetric) or $-1$ (antisymmetric)
- For N particles: $N!$ permutation operators form a group

### 2. Bosons and Fermions

All particles in nature fall into exactly two categories:

| Property | Bosons | Fermions |
|----------|--------|----------|
| Spin | Integer (0, 1, 2, ...) | Half-integer (1/2, 3/2, ...) |
| Symmetry | Symmetric under exchange | Antisymmetric under exchange |
| Examples | Photons, gluons, $^4$He | Electrons, protons, $^3$He |
| Statistics | Bose-Einstein | Fermi-Dirac |
| Occupation | Any number in same state | At most one per state |

### 3. Spin-Statistics Theorem

**Statement:** Particles with integer spin are bosons; particles with half-integer spin are fermions.

This is not an axiom but a theorem provable from:
- Lorentz invariance
- Locality (causality)
- Positive energy (stability)

### 4. Slater Determinants

For N fermions in single-particle states $\phi_1, \phi_2, \ldots, \phi_N$:

$$\Psi_F(1,2,\ldots,N) = \frac{1}{\sqrt{N!}}\begin{vmatrix}
\phi_1(1) & \phi_1(2) & \cdots & \phi_1(N) \\
\phi_2(1) & \phi_2(2) & \cdots & \phi_2(N) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_N(1) & \phi_N(2) & \cdots & \phi_N(N)
\end{vmatrix}$$

Key properties:
- Automatically antisymmetric (property of determinants)
- Vanishes if any two single-particle states are identical (Pauli exclusion)
- $1/\sqrt{N!}$ ensures normalization

### 5. Second Quantization

**Bosonic Operators:**
$$[a_i, a_j^\dagger] = \delta_{ij}, \quad [a_i, a_j] = 0, \quad [a_i^\dagger, a_j^\dagger] = 0$$

**Fermionic Operators:**
$$\{c_i, c_j^\dagger\} = \delta_{ij}, \quad \{c_i, c_j\} = 0, \quad \{c_i^\dagger, c_j^\dagger\} = 0$$

**Number States (Fock Space):**
$$|n_1, n_2, \ldots\rangle = \frac{(a_1^\dagger)^{n_1}}{\sqrt{n_1!}}\frac{(a_2^\dagger)^{n_2}}{\sqrt{n_2!}}\cdots|0\rangle$$

For fermions: $n_i \in \{0, 1\}$ only.

---

## Key Equations

### Two-Particle Wavefunctions

**Symmetric (Bosons):**
$$\Psi_S(1,2) = \frac{1}{\sqrt{2}}[\phi_a(1)\phi_b(2) + \phi_a(2)\phi_b(1)]$$

**Antisymmetric (Fermions):**
$$\Psi_A(1,2) = \frac{1}{\sqrt{2}}[\phi_a(1)\phi_b(2) - \phi_a(2)\phi_b(1)]$$

### Second Quantization Operators

**One-body operator:** $\hat{F} = \sum_i f(\mathbf{r}_i)$
$$\hat{F} = \sum_{i,j}\langle i|f|j\rangle a_i^\dagger a_j$$

**Two-body operator:** $\hat{G} = \frac{1}{2}\sum_{i\neq j} g(\mathbf{r}_i, \mathbf{r}_j)$
$$\hat{G} = \frac{1}{2}\sum_{i,j,k,l}\langle ij|g|kl\rangle a_i^\dagger a_j^\dagger a_l a_k$$

### Helium Atom

**Hamiltonian:**
$$H = -\frac{\hbar^2}{2m}(\nabla_1^2 + \nabla_2^2) - \frac{Ze^2}{r_1} - \frac{Ze^2}{r_2} + \frac{e^2}{r_{12}}$$

**Ground State Energy (first-order perturbation):**
$$E_0 \approx 2E_1(\text{H}) \cdot Z^2 + \frac{5}{4}\frac{Z e^2}{a_0}$$

where $E_1(\text{H}) = -13.6$ eV.

---

## Study Resources

### Primary Texts
- Shankar, Chapter 10 (Identical Particles)
- Sakurai, Chapter 7 (Identical Particles)
- Griffiths, Chapter 5 (Identical Particles)

### Supplementary
- [Physics LibreTexts - Second Quantization](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Quantum_Mechanics_III_(Chong)/04:_Identical_Particles/4.03:_Second_Quantization)
- [Cambridge Second Quantization Notes](https://www.tcm.phy.cam.ac.uk/~bds10/tp3/secqu.pdf)
- [ETH Zurich Second Quantization](https://ethz.ch/content/dam/ethz/special-interest/phys/theoretical-physics/cmtm-dam/documents/qg/Chapter_05-06.pdf)

---

## Qualifying Exam Relevance

### Typical Problem Types
1. Construct ground state wavefunction for N electrons in a potential
2. Calculate exchange energy and explain its physical origin
3. Convert first-quantized Hamiltonian to second-quantized form
4. Apply Pauli exclusion to determine allowed states

### Common Mistakes to Avoid
- Forgetting normalization factors in Slater determinants
- Sign errors in fermionic anticommutation
- Confusing direct and exchange integrals
- Applying bosonic commutators to fermions

---

## Week Checklist

- [ ] Master exchange operator properties
- [ ] Understand spin-statistics theorem
- [ ] Practice building Slater determinants
- [ ] Derive second quantization commutation relations
- [ ] Solve helium atom ground state
- [ ] Complete all problem sets
- [ ] Practice oral explanations

---

**Created:** February 2026
**Status:** NOT STARTED
