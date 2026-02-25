# Day 526: Unitary Evolution

## Overview
**Day 526** | Week 76, Day 1 | Year 1, Month 19 | Closed System Dynamics

Today we study how density matrices evolve under unitary (closed system) dynamics.

---

## Learning Objectives
1. Derive the von Neumann equation
2. Solve for time evolution of density matrices
3. Compare Schrödinger and Heisenberg pictures
4. Understand evolution of mixed states
5. Apply to simple quantum systems

---

## Core Content

### von Neumann Equation

For a closed system with Hamiltonian H:

$$\boxed{\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho]}$$

**Solution:**
$$\rho(t) = U(t)\rho(0)U^\dagger(t), \quad U(t) = e^{-iHt/\hbar}$$

### Properties of Unitary Evolution

1. **Trace preserving:** Tr(ρ(t)) = Tr(ρ(0)) = 1
2. **Purity preserving:** Tr(ρ²(t)) = Tr(ρ²(0))
3. **Reversible:** ρ(0) = U†(t)ρ(t)U(t)

### Heisenberg Picture

For observable A:
$$\langle A \rangle(t) = \text{Tr}(\rho(0) A(t)), \quad A(t) = U^\dagger(t) A U(t)$$

### Evolution of Pure vs Mixed States

**Pure state:** |ψ(t)⟩ = U(t)|ψ(0)⟩ → ρ(t) = |ψ(t)⟩⟨ψ(t)|

**Mixed state:** ρ(t) = U(t)(Σᵢpᵢ|ψᵢ⟩⟨ψᵢ|)U†(t) = Σᵢpᵢ|ψᵢ(t)⟩⟨ψᵢ(t)|

---

## Computational Lab

```python
"""Day 526: Unitary Evolution"""
import numpy as np
import matplotlib.pyplot as plt

def evolve_density_matrix(rho0, H, t):
    """Evolve ρ(0) → ρ(t) = U ρ U†"""
    U = np.linalg.matrix_power(np.eye(len(H)) - 1j*H*0.01, int(t/0.01))
    return U @ rho0 @ U.conj().T

# Rabi oscillations
H = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)  # H = ωX/2
rho0 = np.array([[1, 0], [0, 0]], dtype=complex)  # Start in |0⟩

times = np.linspace(0, 4*np.pi, 100)
probs = []
for t in times:
    U = np.array([[np.cos(t/2), -1j*np.sin(t/2)],
                  [-1j*np.sin(t/2), np.cos(t/2)]], dtype=complex)
    rho_t = U @ rho0 @ U.conj().T
    probs.append(rho_t[1,1].real)

plt.plot(times, probs)
plt.xlabel('Time')
plt.ylabel('P(|1⟩)')
plt.title('Rabi Oscillations')
plt.show()
```

---

## Summary
- von Neumann equation governs closed system evolution
- Unitary evolution preserves purity and trace
- Foundation for understanding open system dynamics

---
*Next: Day 527 — Quantum Operations*
