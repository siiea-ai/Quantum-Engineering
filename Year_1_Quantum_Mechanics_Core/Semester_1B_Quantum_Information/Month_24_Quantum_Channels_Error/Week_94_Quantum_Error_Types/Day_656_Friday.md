# Day 656: Amplitude Damping

## Week 94: Quantum Error Types | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Derive** amplitude damping from physical principles (spontaneous emission)
2. **Analyze** the non-unital nature of amplitude damping
3. **Understand** T1 relaxation in quantum systems
4. **Compare** amplitude damping to Pauli errors

---

## Core Content

### 1. Physical Motivation: Spontaneous Emission

A two-level atom in excited state $|1\rangle$ can spontaneously emit a photon and decay to $|0\rangle$:

$$|1\rangle \xrightarrow{\gamma} |0\rangle + \text{photon}$$

The ground state $|0\rangle$ is stable. This asymmetry makes amplitude damping fundamentally different from Pauli channels.

### 2. Kraus Representation

$$\boxed{K_0 = \begin{pmatrix}1 & 0\\0 & \sqrt{1-\gamma}\end{pmatrix}, \quad K_1 = \begin{pmatrix}0 & \sqrt{\gamma}\\0 & 0\end{pmatrix}}$$

where $\gamma \in [0,1]$ is the decay probability.

**Verification:** $K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix}1 & 0\\0 & 1-\gamma\end{pmatrix} + \begin{pmatrix}0 & 0\\0 & \gamma\end{pmatrix} = I$ ✓

### 3. Effect on States

**On $|0\rangle$ (ground state):**
$$\mathcal{E}_{AD}(|0\rangle\langle 0|) = |0\rangle\langle 0|$$
Ground state is stable.

**On $|1\rangle$ (excited state):**
$$\mathcal{E}_{AD}(|1\rangle\langle 1|) = (1-\gamma)|1\rangle\langle 1| + \gamma|0\rangle\langle 0|$$
Excited state decays with probability $\gamma$.

**On superposition $|+\rangle$:**
$$\mathcal{E}_{AD}(|+\rangle\langle +|) = \begin{pmatrix}\frac{1+\gamma}{2} & \frac{\sqrt{1-\gamma}}{2}\\\frac{\sqrt{1-\gamma}}{2} & \frac{1-\gamma}{2}\end{pmatrix}$$

Both populations AND coherences are affected!

### 4. Non-Unital Property

Amplitude damping is **NOT unital**: $\mathcal{E}_{AD}(I) \neq I$

$$\mathcal{E}_{AD}(I/2) = \begin{pmatrix}\frac{1+\gamma}{2} & 0\\0 & \frac{1-\gamma}{2}\end{pmatrix} \neq I/2$$

The maximally mixed state is NOT a fixed point—only $|0\rangle$ is.

### 5. Bloch Sphere Effect

Amplitude damping:
- **Contracts** the Bloch sphere by $\sqrt{1-\gamma}$ in $x$ and $y$
- **Contracts** by $(1-\gamma)$ in $z$
- **Shifts** toward $|0\rangle$ (north pole)

$$\vec{r} = (r_x, r_y, r_z) \mapsto (\sqrt{1-\gamma}r_x, \sqrt{1-\gamma}r_y, \gamma + (1-\gamma)r_z)$$

### 6. T1 Relaxation Time

In experiments, continuous amplitude damping is characterized by **T1**:

$$\gamma(t) = 1 - e^{-t/T_1}$$

$$P_1(t) = P_1(0) \cdot e^{-t/T_1}$$

T1 is the characteristic time for the excited state population to decay by factor $e$.

### 7. Stinespring Dilation

The environment (electromagnetic field) records whether a photon was emitted:

$$U|0\rangle_S|0\rangle_E = |0\rangle_S|0\rangle_E$$
$$U|1\rangle_S|0\rangle_E = \sqrt{1-\gamma}|1\rangle_S|0\rangle_E + \sqrt{\gamma}|0\rangle_S|1\rangle_E$$

The environment state $|1\rangle_E$ indicates photon emission.

### 8. Generalized Amplitude Damping

At finite temperature, both decay AND excitation occur:

$$K_0 = \sqrt{p}\begin{pmatrix}1 & 0\\0 & \sqrt{1-\gamma}\end{pmatrix}, \quad K_1 = \sqrt{p}\begin{pmatrix}0 & \sqrt{\gamma}\\0 & 0\end{pmatrix}$$
$$K_2 = \sqrt{1-p}\begin{pmatrix}\sqrt{1-\gamma} & 0\\0 & 1\end{pmatrix}, \quad K_3 = \sqrt{1-p}\begin{pmatrix}0 & 0\\\sqrt{\gamma} & 0\end{pmatrix}$$

where $p$ relates to temperature via the Boltzmann factor.

---

## Worked Example

**Problem:** A qubit starts in $|1\rangle$. After amplitude damping with $\gamma = 0.3$, what are the populations and coherences?

**Solution:**
$$\rho_{\text{out}} = \gamma|0\rangle\langle 0| + (1-\gamma)|1\rangle\langle 1| = \begin{pmatrix}0.3 & 0\\0 & 0.7\end{pmatrix}$$

- $P(|0\rangle) = 0.3$ (30% decayed)
- $P(|1\rangle) = 0.7$ (70% survived)
- Coherences = 0 (started with none)

---

## Practice Problems

1. Show that $|0\rangle$ is the unique fixed point of amplitude damping.
2. Derive the continuous master equation for T1 decay.
3. For what $\gamma$ does amplitude damping equal the completely dephasing channel?
4. Find the complementary channel (what the environment sees).

---

## Computational Lab

```python
"""Day 656: Amplitude Damping"""

import numpy as np
import matplotlib.pyplot as plt

def amplitude_damping(rho, gamma):
    """Apply amplitude damping with decay probability gamma."""
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

# Visualize decay of excited state
gamma_values = np.linspace(0, 1, 50)
rho_1 = np.array([[0, 0], [0, 1]], dtype=complex)

p0_values = [amplitude_damping(rho_1, g)[0, 0].real for g in gamma_values]
p1_values = [amplitude_damping(rho_1, g)[1, 1].real for g in gamma_values]

plt.figure(figsize=(10, 6))
plt.plot(gamma_values, p0_values, 'b-', linewidth=2, label='P(|0⟩)')
plt.plot(gamma_values, p1_values, 'r-', linewidth=2, label='P(|1⟩)')
plt.xlabel('Decay probability γ')
plt.ylabel('Population')
plt.title('Amplitude Damping: Excited State Decay')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('amplitude_damping_decay.png', dpi=150)
plt.show()

# T1 decay simulation
T1 = 50  # arbitrary units
times = np.linspace(0, 200, 100)
gammas = 1 - np.exp(-times/T1)

rho = rho_1.copy()
p1_vs_time = []
for t, g in zip(times, gammas):
    rho_out = amplitude_damping(rho_1, g)
    p1_vs_time.append(rho_out[1, 1].real)

plt.figure(figsize=(10, 6))
plt.plot(times, p1_vs_time, 'g-', linewidth=2)
plt.axhline(y=1/np.e, color='r', linestyle='--', label=f'1/e level (T₁={T1})')
plt.axvline(x=T1, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('P(|1⟩)')
plt.title(f'T₁ Relaxation (T₁ = {T1})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('t1_relaxation.png', dpi=150)
plt.show()

print("Amplitude damping: irreversible energy loss to environment")
```

---

## Summary

- **Amplitude damping** models spontaneous emission/energy relaxation
- **Non-unital**: ground state is the unique fixed point
- **Asymmetric**: affects $|1\rangle$ but not $|0\rangle$
- **T1 time**: characteristic decay timescale
- **Breaks time-reversal symmetry**: irreversible process
- Different from Pauli channels: not a simple error probability

---

## Preview: Day 657

Tomorrow: **Error Channels in Practice** - connecting theory to real device parameters.
