# Day 471: First-Order Transitions

## Overview
**Day 471** | Year 1, Month 17, Week 68 | Transition Amplitudes

---

## Core Content

### Expansion in Unperturbed Basis

$$|\psi_I(t)\rangle = \sum_n c_n(t)|n\rangle$$

where H₀|n⟩ = E_n|n⟩.

### First-Order Approximation

Starting in state |i⟩ at t = 0:
$$c_f^{(1)}(t) = -\frac{i}{\hbar}\int_0^t \langle f|V_I(t')|i\rangle\,dt'$$

$$= -\frac{i}{\hbar}\int_0^t V_{fi}(t')e^{i\omega_{fi}t'}\,dt'$$

where ω_{fi} = (E_f - E_i)/ℏ.

### Transition Probability

$$\boxed{P_{i \to f}(t) = |c_f^{(1)}(t)|^2 = \frac{1}{\hbar^2}\left|\int_0^t V_{fi}(t')e^{i\omega_{fi}t'}\,dt'\right|^2}$$

### Constant Perturbation (Turned on at t = 0)

If V(t) = V for t > 0:
$$c_f^{(1)}(t) = -\frac{V_{fi}}{\hbar\omega_{fi}}(e^{i\omega_{fi}t} - 1)$$

$$P_{i \to f}(t) = \frac{4|V_{fi}|^2}{\hbar^2\omega_{fi}^2}\sin^2\left(\frac{\omega_{fi}t}{2}\right)$$

### Energy Conservation Peak

As t → ∞, transition probability peaks sharply at E_f = E_i.

---

**Next:** [Day_472_Wednesday.md](Day_472_Wednesday.md) — Fermi's Golden Rule
