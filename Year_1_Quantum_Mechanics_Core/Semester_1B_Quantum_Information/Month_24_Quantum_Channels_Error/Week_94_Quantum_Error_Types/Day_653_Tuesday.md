# Day 653: Phase-Flip Errors (Z)

## Week 94: Quantum Error Types | Month 24: Quantum Channels & Error Introduction

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Phase-flip channel theory, uniquely quantum aspects |
| **Afternoon** | 2.5 hours | Problem solving and dephasing analysis |
| **Evening** | 1.5 hours | Computational lab: simulating phase errors |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define** the phase-flip channel and compare it to bit-flip
2. **Explain** why phase-flip errors are "uniquely quantum"
3. **Analyze** the effect of phase errors on quantum superpositions
4. **Connect** phase-flip to physical dephasing processes
5. **Identify** which states are protected from phase errors
6. **Understand** why phase errors are critical for quantum computing

---

## Core Content

### 1. The Phase-Flip Channel

The **phase-flip channel** applies the Pauli $Z$ operator with probability $p$:

$$\boxed{\mathcal{E}_Z(\rho) = (1-p)\rho + pZ\rho Z}$$

**Kraus operators:**
$$K_0 = \sqrt{1-p} \cdot I, \quad K_1 = \sqrt{p} \cdot Z$$

### 2. Why Phase-Flip is Uniquely Quantum

**Key insight:** Phase-flip has no classical analog!

- Classical bits have no phase
- Phase is a purely quantum concept
- Phase errors destroy quantum superposition without changing populations

**Classical view:** A bit in state 0 or 1 is unaffected by "phase flip"
**Quantum view:** A qubit in $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ becomes $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$

### 3. Effect on Computational Basis States

For $|0\rangle$:
$$Z|0\rangle = |0\rangle$$
$$\mathcal{E}_Z(|0\rangle\langle 0|) = |0\rangle\langle 0|$$

For $|1\rangle$:
$$Z|1\rangle = -|1\rangle$$
$$\mathcal{E}_Z(|1\rangle\langle 1|) = |1\rangle\langle 1|$$

**Computational basis states are unaffected!** This is why phase errors are "invisible" to computational basis measurements.

### 4. Effect on Superposition States

For $|+\rangle$:
$$Z|+\rangle = |-\rangle$$
$$\mathcal{E}_Z(|+\rangle\langle +|) = (1-p)|+\rangle\langle +| + p|-\rangle\langle -|$$

**The superposition is destroyed!** The state becomes a classical mixture of $|+\rangle$ and $|-\rangle$.

### 5. Effect on General States

For $\rho = \begin{pmatrix}a & b\\b^* & 1-a\end{pmatrix}$:

$$Z\rho Z = \begin{pmatrix}a & -b\\-b^* & 1-a\end{pmatrix}$$

$$\mathcal{E}_Z(\rho) = \begin{pmatrix}a & (1-2p)b\\(1-2p)b^* & 1-a\end{pmatrix}$$

**Key observation:**
- **Diagonal elements unchanged** (populations preserved)
- **Off-diagonal elements decay** by factor $(1-2p)$

This is **pure dephasing**—coherence is lost without energy exchange.

### 6. Bloch Sphere Representation

Bloch vector transformation:
$$\vec{r} = (r_x, r_y, r_z) \mapsto ((1-2p)r_x, (1-2p)r_y, r_z)$$

**Effect:** Bloch sphere compressed toward the $z$-axis.

- States on $z$-axis ($|0\rangle$, $|1\rangle$) are fixed
- States in $xy$-plane experience maximum decoherence

### 7. Fixed Points

Fixed points satisfy $\mathcal{E}_Z(\rho_*) = \rho_*$.

**Solution:** Any diagonal density matrix:
$$\rho_* = \begin{pmatrix}a & 0\\0 & 1-a\end{pmatrix} = a|0\rangle\langle 0| + (1-a)|1\rangle\langle 1|$$

**Fixed points form the $z$-axis** of the Bloch sphere.

### 8. Comparison: Bit-Flip vs Phase-Flip

| Property | Bit-Flip ($X$) | Phase-Flip ($Z$) |
|----------|---------------|-----------------|
| Channel | $(1-p)\rho + pX\rho X$ | $(1-p)\rho + pZ\rho Z$ |
| Affects populations | Yes | No |
| Affects coherences | Yes | Yes |
| Fixed points | $\|+\rangle$, $\|-\rangle$, mixtures | $\|0\rangle$, $\|1\rangle$, mixtures |
| Classical analog | Yes (BSC) | No |
| Bloch contraction | Toward $x$-axis | Toward $z$-axis |

### 9. Physical Origins of Dephasing

**What causes phase-flip errors?**

1. **Longitudinal field fluctuations:** Random energy shifts
2. **1/f noise:** Low-frequency noise common in solid-state systems
3. **Photon scattering:** In atomic/optical systems
4. **Charge noise:** In superconducting and semiconductor qubits

**T2 time:** Characteristic dephasing time in experiments
$$\rho_{01}(t) = \rho_{01}(0) \cdot e^{-t/T_2}$$

### 10. The Dephasing Basis

Phase errors are defined relative to a basis. The standard phase-flip uses $Z$, but we could define:
- $X$-basis dephasing: $\mathcal{E}(\rho) = (1-p)\rho + pX\rho X$ (this is bit-flip!)
- $Y$-basis dephasing: $\mathcal{E}(\rho) = (1-p)\rho + pY\rho Y$

**Insight:** Bit-flip IS dephasing in the $X$-basis!

---

## Quantum Computing Connection

### Why Phase Errors Matter

1. **Quantum algorithms depend on phase:** Interference requires coherent superposition
2. **Phase errors are sneaky:** Can't detect by measuring in computational basis
3. **Phase errors are common:** Dephasing often faster than energy relaxation

### Quantum Phase Estimation Vulnerability

QPE crucially depends on phase information. Phase errors directly corrupt:
- Phase kickback mechanism
- Interference patterns
- Final measurement outcomes

### Detecting Phase Errors

Must measure in $X$ or $Y$ basis to detect phase errors:
$$\langle X \rangle = 2\text{Re}(\rho_{01}) \xrightarrow{\text{phase flip}} (1-2p) \cdot 2\text{Re}(\rho_{01})$$

---

## Worked Examples

### Example 1: Phase-Flip on $|+\rangle$ State

**Problem:** Apply phase-flip channel with $p = 0.3$ to $|+\rangle$.

**Solution:**

Initial state:
$$\rho = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix}1 & 1\\1 & 1\end{pmatrix}$$

Apply channel:
$$\mathcal{E}_Z(\rho) = \frac{1}{2}\begin{pmatrix}1 & 1-2(0.3)\\1-2(0.3) & 1\end{pmatrix} = \frac{1}{2}\begin{pmatrix}1 & 0.4\\0.4 & 1\end{pmatrix}$$

**Interpretation:**
- Populations unchanged (both 0.5)
- Coherence reduced: $0.5 \to 0.2$
- Purity: $\text{Tr}(\rho^2) = \frac{1}{4}(1 + 0.16 + 0.16 + 1) = 0.58$

---

### Example 2: Undetectable in Z-basis

**Problem:** Show that measuring $|+\rangle$ and $\mathcal{E}_Z(|+\rangle\langle +|)$ in the computational basis gives the same statistics.

**Solution:**

For $|+\rangle$:
- $P(0) = |\langle 0|+\rangle|^2 = 1/2$
- $P(1) = |\langle 1|+\rangle|^2 = 1/2$

For $\mathcal{E}_Z(|+\rangle\langle +|)$ with any $p$:
- $P(0) = \rho_{00} = 1/2$
- $P(1) = \rho_{11} = 1/2$

**Same statistics!** Phase errors are invisible to computational basis measurements.

---

### Example 3: Detecting Phase Errors

**Problem:** Design a measurement to detect if phase-flip occurred on $|+\rangle$.

**Solution:**

Measure in $X$-basis (Hadamard then measure):

For undamaged $|+\rangle$:
- $H|+\rangle = |0\rangle$
- Always measure 0

For $|-\rangle$ (after phase flip):
- $H|-\rangle = |1\rangle$
- Always measure 1

For mixed state $\mathcal{E}_Z(|+\rangle\langle +|)$:
- $P(\text{measure } 0) = 1 - p$
- $P(\text{measure } 1) = p$

**X-basis measurement reveals phase errors!**

---

## Practice Problems

### Direct Application

1. **Problem 1:** Apply the phase-flip channel with $p = 0.2$ to the state $|1\rangle\langle 1|$.

2. **Problem 2:** Calculate the purity of $|+\rangle$ after passing through phase-flip with $p = 0.25$.

3. **Problem 3:** Show that the phase-flip channel is self-conjugate: $\mathcal{E}_Z = \mathcal{E}_Z^\dagger$.

### Intermediate

4. **Problem 4:** Prove that any state on the $z$-axis of the Bloch sphere is a fixed point.

5. **Problem 5:** Two phase-flip channels with $p_1$ and $p_2$ are composed. Find the effective probability.

6. **Problem 6:** Express the phase-flip channel in terms of the Hadamard-transformed bit-flip channel.

### Challenging

7. **Problem 7:** Design a quantum circuit that converts phase errors to bit errors (and vice versa).

8. **Problem 8:** For continuous dephasing $\frac{d\rho}{dt} = \gamma(Z\rho Z - \rho)$, solve for $\rho(t)$.

9. **Problem 9:** Show that a phase error on one qubit of a Bell pair can be detected by the other qubit.

---

## Computational Lab

```python
"""
Day 653 Computational Lab: Phase-Flip Errors
============================================
Topics: Phase-flip channel, dephasing, detection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Standard operators
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def phase_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply phase-flip channel with probability p."""
    return (1 - p) * rho + p * Z @ rho @ Z


def bit_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply bit-flip channel for comparison."""
    return (1 - p) * rho + p * X @ rho @ X


def purity(rho: np.ndarray) -> float:
    """Compute purity."""
    return np.real(np.trace(rho @ rho))


def coherence(rho: np.ndarray) -> float:
    """Compute magnitude of off-diagonal element."""
    return np.abs(rho[0, 1])


# ===== DEMONSTRATIONS =====

print("=" * 70)
print("PART 1: Phase-Flip vs Bit-Flip Comparison")
print("=" * 70)

p = 0.2
rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)

print(f"\nError probability p = {p}")

print("\nEffect on |+⟩:")
rho_plus_pf = phase_flip_channel(rho_plus, p)
rho_plus_bf = bit_flip_channel(rho_plus, p)

print(f"  Original coherence: {coherence(rho_plus):.4f}")
print(f"  After phase-flip: {coherence(rho_plus_pf):.4f}")
print(f"  After bit-flip: {coherence(rho_plus_bf):.4f}")

print("\nEffect on |0⟩:")
rho_0_pf = phase_flip_channel(rho_0, p)
rho_0_bf = bit_flip_channel(rho_0, p)

print(f"  Original P(0): {rho_0[0,0].real:.4f}")
print(f"  After phase-flip P(0): {rho_0_pf[0,0].real:.4f}")
print(f"  After bit-flip P(0): {rho_0_bf[0,0].real:.4f}")


print("\n" + "=" * 70)
print("PART 2: Bloch Sphere Transformation")
print("=" * 70)

def visualize_phase_flip_transform(p: float, n_points: int = 20):
    """Visualize phase-flip transformation of Bloch sphere."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 6))

    # Generate Bloch sphere points
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x_in = np.sin(theta_grid) * np.cos(phi_grid)
    y_in = np.sin(theta_grid) * np.sin(phi_grid)
    z_in = np.cos(theta_grid)

    # Phase-flip transforms: x,y contract, z unchanged
    x_out = (1 - 2*p) * x_in
    y_out = (1 - 2*p) * y_in
    z_out = z_in

    # Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x_in, y_in, z_in, alpha=0.3, color='blue')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Input: Bloch Sphere')
    ax1.set_xlim([-1.1, 1.1]); ax1.set_ylim([-1.1, 1.1]); ax1.set_zlim([-1.1, 1.1])

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x_out, y_out, z_out, alpha=0.3, color='red')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title(f'After Phase-Flip (p={p})')
    ax2.set_xlim([-1.1, 1.1]); ax2.set_ylim([-1.1, 1.1]); ax2.set_zlim([-1.1, 1.1])

    plt.tight_layout()
    plt.savefig('phase_flip_bloch.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: phase_flip_bloch.png")

visualize_phase_flip_transform(0.3)


print("\n" + "=" * 70)
print("PART 3: Invisibility to Z-Basis Measurement")
print("=" * 70)

def measure_z_statistics(rho: np.ndarray, n_samples: int = 10000) -> Tuple[float, float]:
    """Simulate Z-basis measurement statistics."""
    p0 = np.real(rho[0, 0])
    p1 = np.real(rho[1, 1])
    outcomes = np.random.choice([0, 1], size=n_samples, p=[p0, p1])
    return np.mean(outcomes == 0), np.mean(outcomes == 1)

print("\nZ-basis measurement statistics:")
print("State          | P(0)  | P(1)")
print("-" * 35)

states = {
    '|+⟩': rho_plus,
    'E_Z(|+⟩), p=0.1': phase_flip_channel(rho_plus, 0.1),
    'E_Z(|+⟩), p=0.3': phase_flip_channel(rho_plus, 0.3),
    'E_Z(|+⟩), p=0.5': phase_flip_channel(rho_plus, 0.5)
}

for name, rho in states.items():
    p0, p1 = measure_z_statistics(rho)
    print(f"{name:15s} | {p0:.3f} | {p1:.3f}")

print("\nAll states give same Z-basis statistics! Phase errors are invisible.")


print("\n" + "=" * 70)
print("PART 4: Detection via X-Basis Measurement")
print("=" * 70)

def measure_x_statistics(rho: np.ndarray) -> Tuple[float, float]:
    """X-basis measurement (apply H then measure Z)."""
    rho_h = H @ rho @ H
    return np.real(rho_h[0, 0]), np.real(rho_h[1, 1])

print("\nX-basis measurement statistics:")
print("State          | P(+)  | P(-)")
print("-" * 35)

for name, rho in states.items():
    p_plus, p_minus = measure_x_statistics(rho)
    print(f"{name:15s} | {p_plus:.3f} | {p_minus:.3f}")

print("\nX-basis reveals phase errors! P(-) increases with error probability.")


print("\n" + "=" * 70)
print("PART 5: Coherence Decay")
print("=" * 70)

p_values = [0.05, 0.1, 0.2, 0.3, 0.4]
n_applications = 30

plt.figure(figsize=(10, 6))

for p in p_values:
    coherences = [(1 - 2*p)**n * 0.5 for n in range(n_applications + 1)]
    plt.plot(range(n_applications + 1), coherences, linewidth=2, label=f'p={p}')

plt.xlabel('Number of phase-flip applications')
plt.ylabel('Coherence |ρ₀₁|')
plt.title('Coherence Decay Under Phase-Flip Channel')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phase_flip_coherence.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: phase_flip_coherence.png")


print("\n" + "=" * 70)
print("PART 6: Relationship to Hadamard-Conjugated Bit-Flip")
print("=" * 70)

# Phase-flip = H @ Bit-flip @ H
p_test = 0.2
rho_test = np.array([[0.6, 0.3+0.1j], [0.3-0.1j, 0.4]], dtype=complex)

# Direct phase-flip
rho_pf = phase_flip_channel(rho_test, p_test)

# Hadamard-conjugated bit-flip
rho_hbfh = H @ bit_flip_channel(H @ rho_test @ H, p_test) @ H

print(f"Phase-flip channel output:")
print(np.array2string(rho_pf, precision=4))

print(f"\nH @ BitFlip @ H output:")
print(np.array2string(rho_hbfh, precision=4))

print(f"\nDifference: {np.max(np.abs(rho_pf - rho_hbfh)):.2e}")
print("\nPhase-flip IS bit-flip in the Hadamard basis!")


print("\n" + "=" * 70)
print("PART 7: Physical Dephasing Model (T2 Decay)")
print("=" * 70)

def continuous_dephasing(rho_init: np.ndarray, gamma: float, t: float) -> np.ndarray:
    """
    Continuous dephasing model: ρ₀₁(t) = ρ₀₁(0) * exp(-γt)
    Equivalent to phase-flip with p = (1 - exp(-γt))/2
    """
    decay = np.exp(-gamma * t)
    rho = rho_init.copy()
    rho[0, 1] *= decay
    rho[1, 0] *= decay
    return rho

# Simulate T2 decay
T2 = 50  # microseconds
gamma = 1 / T2
times = np.linspace(0, 200, 100)

coherences_t2 = [coherence(continuous_dephasing(rho_plus, gamma, t)) for t in times]

plt.figure(figsize=(10, 6))
plt.plot(times, coherences_t2, 'b-', linewidth=2)
plt.axhline(y=0.5/np.e, color='r', linestyle='--', label=f'1/e level (T₂={T2}μs)')
plt.axvline(x=T2, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Time (μs)')
plt.ylabel('Coherence |ρ₀₁|')
plt.title(f'T₂ Dephasing (T₂ = {T2} μs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('t2_decay.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: t2_decay.png")


print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Phase-flip channel | $\mathcal{E}_Z(\rho) = (1-p)\rho + pZ\rho Z$ |
| Effect on coherence | $\rho_{01} \mapsto (1-2p)\rho_{01}$ |
| Bloch transform | $(r_x, r_y, r_z) \mapsto ((1-2p)r_x, (1-2p)r_y, r_z)$ |
| T2 decay | $\rho_{01}(t) = \rho_{01}(0)e^{-t/T_2}$ |

### Main Takeaways

1. **Phase-flip is uniquely quantum**—no classical analog
2. **Populations unchanged**, only coherences affected
3. **Computational basis states are fixed points**
4. **Invisible to Z-basis measurement**—must measure in X or Y basis
5. **Models physical dephasing** (T2 processes)
6. **Phase-flip = Bit-flip in Hadamard basis**

---

## Daily Checklist

- [ ] I understand why phase-flip has no classical analog
- [ ] I can compute the effect on arbitrary states
- [ ] I know which states are protected from phase errors
- [ ] I understand how to detect phase errors
- [ ] I can relate discrete phase-flip to continuous T2 decay
- [ ] I completed the computational lab

---

## Preview: Day 654

Tomorrow we study **general Pauli errors**, combining bit-flip, phase-flip, and their combination (Y errors) into a unified framework—the Pauli channel.

---

*"Phase errors are the quantum world's way of saying that coherence is fragile—the very thing that makes quantum computing powerful is also its greatest vulnerability."*
