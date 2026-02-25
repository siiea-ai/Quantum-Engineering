# Day 456: The Variational Principle

## Overview
**Day 456** | Year 1, Month 17, Week 66 | A Rigorous Inequality

Today we prove and explore the variational principle — the foundation of the variational method and the basis for VQE in quantum computing.

---

## Learning Objectives

By the end of today, you will be able to:
1. State and prove the variational principle
2. Understand why E[ψ] ≥ E₀
3. Identify when equality holds
4. Set up variational calculations
5. Connect to quantum optimization algorithms
6. Recognize the power of rigorous bounds

---

## Core Content

### The Variational Principle

**Theorem:** For any normalized state |ψ⟩:
$$\boxed{E[\psi] = \langle\psi|\hat{H}|\psi\rangle \geq E_0}$$

where E₀ is the true ground state energy.

### Proof

Expand |ψ⟩ in energy eigenbasis:
$$|\psi\rangle = \sum_n c_n |n\rangle, \quad \hat{H}|n\rangle = E_n|n\rangle$$

Then:
$$E[\psi] = \sum_n |c_n|^2 E_n \geq E_0 \sum_n |c_n|^2 = E_0$$

Equality holds iff |ψ⟩ = |0⟩ (ground state).

### The Variational Method

1. Choose a trial wavefunction ψ(α) with parameter(s) α
2. Compute E(α) = ⟨ψ(α)|H|ψ(α)⟩
3. Minimize: ∂E/∂α = 0
4. Result: E_min ≥ E₀ (upper bound)

### Why It Works

- Always gives an upper bound (rigorous!)
- Better trial functions → tighter bounds
- No requirement for H' to be small
- Works for any system

### For Unnormalized Wavefunctions

$$E[\psi] = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}$$

---

## Quantum Computing Connection

### VQE (Variational Quantum Eigensolver)

The variational principle is the foundation of VQE:
1. Prepare parameterized quantum state |ψ(θ)⟩
2. Measure ⟨H⟩ = Σ_i h_i ⟨P_i⟩ (Pauli decomposition)
3. Classical optimizer minimizes E(θ)
4. Result: approximation to ground state

---

## Worked Example

**Problem:** Use ψ(α) = e^{-αx²} as trial for harmonic oscillator.

**Solution:**
$$E(\alpha) = \frac{\langle\psi|(-\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + \frac{1}{2}m\omega^2 x^2)|\psi\rangle}{\langle\psi|\psi\rangle}$$

After calculation:
$$E(\alpha) = \frac{\hbar^2 \alpha}{2m} + \frac{m\omega^2}{8\alpha}$$

Minimize: ∂E/∂α = 0 gives α = mω/(2ℏ)

$$E_{min} = \frac{1}{2}\hbar\omega = E_0 \text{ (exact!)}$$

---

## Practice Problems

1. Prove the variational principle for excited states with orthogonality constraints.
2. Try ψ = A(a² - x²) for particle in box [-a, a].
3. Why can't variational method give a lower bound?

---

## Summary

| Property | Statement |
|----------|-----------|
| Principle | E[ψ] ≥ E₀ always |
| Equality | Only if ψ = ground state |
| Method | Minimize over parameters |
| Bound | Rigorous upper bound |

---

**Next:** [Day_457_Tuesday.md](Day_457_Tuesday.md) — Trial Wavefunctions
