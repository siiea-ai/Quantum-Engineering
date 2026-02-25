# Day 419: Tensor Operators

## Overview
**Day 419** | Year 1, Month 15, Week 60 | Irreducible Tensor Operators

Tensor operators transform in a definite way under rotations, enabling systematic application of the Wigner-Eckart theorem.

---

## Core Content

### Spherical Tensor Operators

A spherical tensor T^{(k)} of rank k has 2k+1 components T^{(k)}_q (q = -k, ..., +k).

**Transformation under rotation:**
$$\hat{D}^\dagger(\alpha,\beta,\gamma)\hat{T}^{(k)}_q\hat{D}(\alpha,\beta,\gamma) = \sum_{q'} D^{k}_{q'q}(\alpha,\beta,\gamma)\hat{T}^{(k)}_{q'}$$

### Examples of Tensor Operators

**Rank 0 (Scalar):**
- Ĥ, Ĵ², any rotationally invariant operator

**Rank 1 (Vector):**
- Position: r̂^{(1)}_q
- Momentum: p̂^{(1)}_q
- Angular momentum: Ĵ^{(1)}_q

**Rank 2 (Quadrupole):**
- Q̂^{(2)}_q = Σᵢ(3zᵢ² - rᵢ²) for nuclear quadrupole

### Commutation Relations

The defining property:
$$[\hat{J}_z, \hat{T}^{(k)}_q] = \hbar q\, \hat{T}^{(k)}_q$$

$$[\hat{J}_\pm, \hat{T}^{(k)}_q] = \hbar\sqrt{k(k+1) - q(q\pm 1)}\, \hat{T}^{(k)}_{q\pm 1}$$

These are exactly the ladder relations for angular momentum!

### Product of Tensor Operators

The product of two tensors can be decomposed:
$$T^{(k_1)} \otimes T^{(k_2)} = \sum_{k=|k_1-k_2|}^{k_1+k_2} T^{(k)}$$

Using CG coefficients:
$$\hat{T}^{(k)}_q = \sum_{q_1,q_2} \langle k_1, q_1; k_2, q_2 | k, q\rangle \hat{T}^{(k_1)}_{q_1}\hat{T}^{(k_2)}_{q_2}$$

### 6j and 9j Symbols (Preview)

For recoupling three or more angular momenta, higher-order symbols appear:
- **6j symbol:** Recoupling of three angular momenta
- **9j symbol:** Recoupling of four angular momenta

---

## Quantum Computing Connection

Understanding tensor operators helps with:
- Multi-qubit gate decomposition
- Symmetry-adapted quantum algorithms
- Quantum simulation of spin systems

---

## Practice Problems
1. Verify [Ĵᵤ, Ĵ₊] = ℏĴ₊ matches the tensor operator commutation.
2. What rank tensor is Ĵ²?
3. Decompose the product of two rank-1 tensors.

---

**Next:** [Day_420_Sunday.md](Day_420_Sunday.md) — Month 15 Capstone
