# Day 460: Linear Variational Method

## Overview
**Day 460** | Year 1, Month 17, Week 66 | Systematic Basis Expansion

Today we develop the linear variational method, where trial functions are linear combinations of basis functions.

---

## Core Content

### Linear Expansion

$$|\psi\rangle = \sum_{n=1}^{N} c_n |\phi_n\rangle$$

where {|φ_n⟩} is a chosen basis (not necessarily orthonormal).

### The Generalized Eigenvalue Problem

Minimize E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ with respect to c_n:

$$\boxed{\mathbf{H}\mathbf{c} = E\mathbf{S}\mathbf{c}}$$

where:
- H_mn = ⟨φ_m|H|φ_n⟩ (Hamiltonian matrix)
- S_mn = ⟨φ_m|φ_n⟩ (overlap matrix)

### Secular Equation

$$\det(\mathbf{H} - E\mathbf{S}) = 0$$

Gives N eigenvalues E_1 ≤ E_2 ≤ ... ≤ E_N

### Variational Theorem for Excited States

E_k (k-th eigenvalue) ≥ true E_k of the full Hamiltonian!

### Orthonormal Basis (S = I)

Simplifies to standard eigenvalue problem:
$$\mathbf{H}\mathbf{c} = E\mathbf{c}$$

---

## Practice Problems

1. Set up the 2×2 secular equation for H₂⁺.
2. Show that the variational bound holds for each eigenvalue.
3. How does increasing N improve the bounds?

---

**Next:** [Day_461_Saturday.md](Day_461_Saturday.md) — Ritz Method
