# Day 436: Spin-Orbit Coupling

## Overview
**Day 436** | Year 1, Month 16, Week 63 | The L·S Interaction

Today we derive spin-orbit coupling — the interaction between the electron's spin magnetic moment and the magnetic field from its orbital motion.

---

## Learning Objectives

1. Understand the physical origin of spin-orbit coupling
2. Derive the L·S Hamiltonian
3. Include the Thomas precession factor
4. Calculate the first-order energy correction
5. Recognize the j quantum number importance

---

## Core Content

### Physical Origin

In the electron's rest frame:
- The nucleus appears to orbit the electron
- Creates a magnetic field at the electron
- Electron spin interacts with this field

### The Spin-Orbit Hamiltonian

$$\boxed{\hat{H}'_{SO} = \frac{1}{2m_e^2 c^2} \frac{1}{r}\frac{dV}{dr} \hat{\mathbf{L}} \cdot \hat{\mathbf{S}}}$$

For Coulomb: dV/dr = e²/r²

$$\hat{H}'_{SO} = \frac{e^2}{2m_e^2 c^2 r^3} \hat{\mathbf{L}} \cdot \hat{\mathbf{S}}$$

The factor of 1/2 is the **Thomas precession** correction!

### Evaluating L·S

Using J = L + S:
$$J^2 = L^2 + 2\mathbf{L}\cdot\mathbf{S} + S^2$$

$$\boxed{\mathbf{L}\cdot\mathbf{S} = \frac{1}{2}(J^2 - L^2 - S^2)}$$

Eigenvalue: ℏ²/2 [j(j+1) - l(l+1) - 3/4]

### First-Order Correction

$$E^{(1)}_{SO} = \frac{e^2}{2m_e^2 c^2}\langle\frac{1}{r^3}\rangle_{nl} \cdot \frac{\hbar^2}{2}[j(j+1) - l(l+1) - 3/4]$$

For l > 0:
$$\boxed{E^{(1)}_{SO} = \frac{E_n^2}{m_e c^2} \frac{j(j+1) - l(l+1) - 3/4}{l(l+1/2)(l+1)}}$$

### j Values

For given l:
- j = l + 1/2 (spin parallel to L)
- j = l - 1/2 (spin antiparallel to L) for l > 0

---

## Quantum Computing Connection

Spin-orbit coupling is crucial for:
- **Spin qubits**: Sets energy splittings
- **Spintronic devices**: Spin manipulation
- **Topological insulators**: Edge states

---

## Practice Problems

1. What is ⟨L·S⟩ for the state j = l + 1/2?
2. Calculate E^{(1)}_SO for the 2p_{1/2} and 2p_{3/2} states.
3. Why is E^{(1)}_SO = 0 for s-orbitals?

---

## Summary

| Quantity | Formula |
|----------|---------|
| L·S | ℏ²/2 [j(j+1) - l(l+1) - 3/4] |
| j values | l ± 1/2 |
| SO energy | ∝ ⟨1/r³⟩ × [j(j+1) - l(l+1) - 3/4] |

---

**Next:** [Day_437_Wednesday.md](Day_437_Wednesday.md) — Darwin Term
