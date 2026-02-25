# Day 401: Spin-1/2 Formalism

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Two-dimensional Hilbert space |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 401, you will be able to:

1. Describe the two-dimensional Hilbert space for spin-1/2
2. Write general spinor states in the computational basis
3. Calculate S² and Sᵤ eigenvalues for spin-1/2
4. Understand the relationship between spin states and qubits
5. Perform basic calculations with spinors

---

## Core Content

### 1. Two-Dimensional Hilbert Space

Spin-1/2 particles live in a **two-dimensional complex Hilbert space**. The basis states are:

$$|{+}\rangle = |\uparrow\rangle = |0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

$$|{-}\rangle = |\downarrow\rangle = |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**Notation conventions:**
- Physics: |↑⟩, |↓⟩ or |+⟩, |-⟩
- Quantum computing: |0⟩, |1⟩ (computational basis)

### 2. General Spinor State

Any spin-1/2 state can be written:

$$\boxed{|\chi\rangle = \alpha|\uparrow\rangle + \beta|\downarrow\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}}$$

where α, β ∈ ℂ and |α|² + |β|² = 1 (normalization).

**This is exactly a qubit state!**

### 3. Spin Operators

For spin-1/2 (s = 1/2):

$$\hat{S}_i = \frac{\hbar}{2}\sigma_i$$

where σᵢ are the **Pauli matrices** (Day 402).

The eigenvalues of Ŝᵤ are:
$$\hat{S}_z|\uparrow\rangle = +\frac{\hbar}{2}|\uparrow\rangle$$
$$\hat{S}_z|\downarrow\rangle = -\frac{\hbar}{2}|\downarrow\rangle$$

The eigenvalue of Ŝ²:
$$\hat{S}^2|\chi\rangle = \hbar^2 s(s+1)|\chi\rangle = \hbar^2 \cdot \frac{1}{2}\left(\frac{1}{2}+1\right)|\chi\rangle = \frac{3\hbar^2}{4}|\chi\rangle$$

### 4. Inner Products and Probabilities

For state |χ⟩ = α|↑⟩ + β|↓⟩:

$$P(S_z = +\hbar/2) = |\langle\uparrow|\chi\rangle|^2 = |\alpha|^2$$
$$P(S_z = -\hbar/2) = |\langle\downarrow|\chi\rangle|^2 = |\beta|^2$$

**Inner product in matrix form:**
$$\langle\phi|\chi\rangle = \begin{pmatrix} \phi_1^* & \phi_2^* \end{pmatrix}\begin{pmatrix} \chi_1 \\ \chi_2 \end{pmatrix} = \phi_1^*\chi_1 + \phi_2^*\chi_2$$

### 5. Normalization and Global Phase

**Normalization:** |α|² + |β|² = 1

**Global phase irrelevance:** |χ⟩ and e^{iθ}|χ⟩ represent the same physical state.

Using these, we can parameterize any spin-1/2 state with just two real parameters:
$$|\chi\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle$$

This is the **Bloch sphere parameterization** (Day 403).

---

## Quantum Computing Connection

| Spin-1/2 | Qubit |
|----------|-------|
| \|↑⟩, \|↓⟩ | \|0⟩, \|1⟩ |
| α\|↑⟩ + β\|↓⟩ | α\|0⟩ + β\|1⟩ |
| Sᵤ measurement | Z-basis measurement |
| ℏ/2, -ℏ/2 outcomes | 0, 1 classical bits |

**The qubit IS a spin-1/2 system.** All single-qubit operations are rotations in spin space.

---

## Worked Examples

### Example 1: Normalization

**Problem:** Normalize the state |χ⟩ = 3|↑⟩ + 4i|↓⟩.

**Solution:**
$$\langle\chi|\chi\rangle = |3|^2 + |4i|^2 = 9 + 16 = 25$$

Normalized state:
$$|\chi\rangle_{norm} = \frac{1}{5}(3|\uparrow\rangle + 4i|\downarrow\rangle) = \begin{pmatrix} 3/5 \\ 4i/5 \end{pmatrix}$$

### Example 2: Measurement Probability

**Problem:** For |χ⟩ = (|↑⟩ + |↓⟩)/√2, find P(Sᵤ = +ℏ/2).

**Solution:**
$$P(S_z = +\hbar/2) = |\langle\uparrow|\chi\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

### Example 3: Expectation Value

**Problem:** Calculate ⟨Ŝᵤ⟩ for |χ⟩ = (|↑⟩ + |↓⟩)/√2.

**Solution:**
$$\langle\hat{S}_z\rangle = P(+\hbar/2)\cdot(+\hbar/2) + P(-\hbar/2)\cdot(-\hbar/2)$$
$$= \frac{1}{2}\cdot\frac{\hbar}{2} + \frac{1}{2}\cdot\left(-\frac{\hbar}{2}\right) = 0$$

---

## Practice Problems

### Direct Application

1. Normalize |χ⟩ = 2|↑⟩ - i|↓⟩.

2. For |χ⟩ = (|↑⟩ - i|↓⟩)/√2, find P(Sᵤ = ±ℏ/2).

3. Calculate ⟨χ|χ⟩ for |χ⟩ = cos(π/3)|↑⟩ + sin(π/3)|↓⟩.

### Intermediate

4. Show that |+⟩ = (|↑⟩ + |↓⟩)/√2 and |-⟩ = (|↑⟩ - |↓⟩)/√2 are orthonormal.

5. Express |↑⟩ and |↓⟩ in terms of |+⟩ and |-⟩.

6. For |χ⟩ = α|↑⟩ + β|↓⟩, find ⟨Ŝᵤ⟩ in terms of α and β.

### Challenging

7. Prove that any two orthonormal states in the spin-1/2 space can serve as a basis.

8. Show that the space of physical spin states (up to global phase) is 2-dimensional over ℝ.

---

## Computational Lab

```python
"""
Day 401 Computational Lab: Spin-1/2 Formalism
"""

import numpy as np
import matplotlib.pyplot as plt

# Define basis states
ket_up = np.array([[1], [0]], dtype=complex)
ket_down = np.array([[0], [1]], dtype=complex)

def normalize(state):
    """Normalize a state vector."""
    norm = np.sqrt(np.vdot(state, state))
    return state / norm

def inner_product(phi, chi):
    """Calculate <phi|chi>."""
    return np.vdot(phi, chi)

def probability(state, basis_state):
    """Calculate probability of measuring basis_state."""
    amp = inner_product(basis_state, state)
    return np.abs(amp)**2

def expectation_Sz(state, hbar=1):
    """Calculate <Sz> for a spin-1/2 state."""
    Sz = (hbar/2) * np.array([[1, 0], [0, -1]], dtype=complex)
    return np.real(np.vdot(state, Sz @ state))

class Spinor:
    """Class representing a spin-1/2 state."""

    def __init__(self, alpha, beta):
        self.state = normalize(np.array([[alpha], [beta]], dtype=complex))
        self.alpha = self.state[0, 0]
        self.beta = self.state[1, 0]

    def prob_up(self):
        return np.abs(self.alpha)**2

    def prob_down(self):
        return np.abs(self.beta)**2

    def expectation_Sz(self, hbar=1):
        return (hbar/2) * (self.prob_up() - self.prob_down())

    def __repr__(self):
        return f"({self.alpha:.4f})|↑⟩ + ({self.beta:.4f})|↓⟩"

def demonstrate_spinor_states():
    """Demonstrate various spinor calculations."""
    print("Spin-1/2 State Examples")
    print("=" * 50)

    # Example 1: |↑⟩
    s1 = Spinor(1, 0)
    print(f"\nState: |↑⟩ = {s1}")
    print(f"  P(↑) = {s1.prob_up():.4f}")
    print(f"  P(↓) = {s1.prob_down():.4f}")
    print(f"  ⟨Sz⟩/ℏ = {s1.expectation_Sz():.4f}")

    # Example 2: |+⟩ = (|↑⟩ + |↓⟩)/√2
    s2 = Spinor(1, 1)
    print(f"\nState: |+⟩ = {s2}")
    print(f"  P(↑) = {s2.prob_up():.4f}")
    print(f"  P(↓) = {s2.prob_down():.4f}")
    print(f"  ⟨Sz⟩/ℏ = {s2.expectation_Sz():.4f}")

    # Example 3: Arbitrary state
    s3 = Spinor(3, 4j)
    print(f"\nState: 3|↑⟩ + 4i|↓⟩ (normalized) = {s3}")
    print(f"  P(↑) = {s3.prob_up():.4f}")
    print(f"  P(↓) = {s3.prob_down():.4f}")
    print(f"  ⟨Sz⟩/ℏ = {s3.expectation_Sz():.4f}")

def visualize_probability_distribution():
    """Visualize measurement probabilities for various states."""
    fig, ax = plt.subplots(figsize=(10, 6))

    states = [
        (Spinor(1, 0), "|↑⟩"),
        (Spinor(0, 1), "|↓⟩"),
        (Spinor(1, 1), "|+⟩"),
        (Spinor(1, -1), "|-⟩"),
        (Spinor(1, 1j), "|+i⟩"),
        (Spinor(3, 4), "3|↑⟩+4|↓⟩"),
    ]

    x = np.arange(len(states))
    width = 0.35

    prob_ups = [s.prob_up() for s, _ in states]
    prob_downs = [s.prob_down() for s, _ in states]

    bars1 = ax.bar(x - width/2, prob_ups, width, label='P(↑)', color='steelblue')
    bars2 = ax.bar(x + width/2, prob_downs, width, label='P(↓)', color='coral')

    ax.set_ylabel('Probability')
    ax.set_title('Measurement Probabilities in Z-basis')
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in states])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('spin_probabilities.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Day 401: Spin-1/2 Formalism")
    print("=" * 50)

    demonstrate_spinor_states()

    print("\nVisualizing probability distributions...")
    visualize_probability_distribution()

    print("\nLab complete!")
```

---

## Summary

| Concept | Formula |
|---------|---------|
| Basis states | \|↑⟩ = (1,0)ᵀ, \|↓⟩ = (0,1)ᵀ |
| General state | \|χ⟩ = α\|↑⟩ + β\|↓⟩ |
| Normalization | \|α\|² + \|β\|² = 1 |
| Ŝ² eigenvalue | ℏ²(3/4) for all spin-1/2 states |
| Ŝᵤ eigenvalues | ±ℏ/2 |
| Probability | P(↑) = \|α\|², P(↓) = \|β\|² |

---

## Daily Checklist

- [ ] I understand the 2D Hilbert space for spin-1/2
- [ ] I can normalize spinor states
- [ ] I can calculate measurement probabilities
- [ ] I see the connection to qubits
- [ ] I completed the computational lab

---

## Preview: Day 402

Tomorrow we introduce the **Pauli matrices**—the fundamental building blocks for describing spin-1/2 dynamics and single-qubit gates.

---

**Next:** [Day_402_Wednesday.md](Day_402_Wednesday.md) — Pauli Matrices
