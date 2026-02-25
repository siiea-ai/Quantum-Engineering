# Day 470: The Interaction Picture

## Overview
**Day 470** | Year 1, Month 17, Week 68 | A New Representation

Today we introduce the interaction picture, which separates the unperturbed evolution from the perturbation's effects.

---

## Core Content

### Three Pictures of QM

| Picture | States | Operators |
|---------|--------|-----------|
| Schrödinger | Time-dependent | Time-independent |
| Heisenberg | Time-independent | Time-dependent |
| **Interaction** | Partial evolution | Partial evolution |

### Setup

$$H = H_0 + V(t)$$

where H₀ is time-independent and exactly solvable.

### Interaction Picture States

$$|\psi_I(t)\rangle = e^{iH_0 t/\hbar}|\psi_S(t)\rangle$$

The H₀ evolution is "factored out."

### Interaction Picture Operators

$$\hat{V}_I(t) = e^{iH_0 t/\hbar}\hat{V}(t)e^{-iH_0 t/\hbar}$$

### Equation of Motion

$$i\hbar\frac{d}{dt}|\psi_I(t)\rangle = \hat{V}_I(t)|\psi_I(t)\rangle$$

Only the perturbation drives the evolution!

### Time Evolution Operator

$$|\psi_I(t)\rangle = \hat{U}_I(t,0)|\psi_I(0)\rangle$$

$$\hat{U}_I(t,0) = \mathcal{T}\exp\left(-\frac{i}{\hbar}\int_0^t V_I(t')\,dt'\right)$$

where 𝒯 is time-ordering.

---

## Quantum Computing Connection

The interaction picture is essential for:
- **Rotating wave approximation** in qubit control
- **Pulse sequence design**
- **Magnus expansion** for gate analysis

---

**Next:** [Day_471_Tuesday.md](Day_471_Tuesday.md) — First-Order Transitions
