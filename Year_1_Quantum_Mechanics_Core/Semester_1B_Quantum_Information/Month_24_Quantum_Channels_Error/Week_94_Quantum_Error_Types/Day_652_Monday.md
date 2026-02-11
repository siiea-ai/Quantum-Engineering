# Day 652: Bit-Flip Errors (X)

## Week 94: Quantum Error Types | Month 24: Quantum Channels & Error Introduction

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Bit-flip channel theory and analysis |
| **Afternoon** | 2.5 hours | Problem solving and applications |
| **Evening** | 1.5 hours | Computational lab: simulating bit-flip errors |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define** the bit-flip channel and its Kraus representation
2. **Analyze** the effect of bit-flip errors on quantum states
3. **Compute** the output of the bit-flip channel for arbitrary inputs
4. **Identify** fixed points and invariant subspaces
5. **Connect** bit-flip errors to classical error models
6. **Understand** physical mechanisms that cause bit-flip errors

---

## Core Content

### 1. The Bit-Flip Channel

The **bit-flip channel** applies the Pauli $X$ operator with probability $p$:

$$\boxed{\mathcal{E}_X(\rho) = (1-p)\rho + pX\rho X}$$

**Interpretation:**
- With probability $1-p$: state unchanged
- With probability $p$: bit is flipped ($|0\rangle \leftrightarrow |1\rangle$)

### 2. Kraus Representation

**Kraus operators:**
$$K_0 = \sqrt{1-p} \cdot I = \sqrt{1-p}\begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}$$

$$K_1 = \sqrt{p} \cdot X = \sqrt{p}\begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}$$

**Verification:**
$$K_0^\dagger K_0 + K_1^\dagger K_1 = (1-p)I + pX^\dagger X = (1-p)I + pI = I \checkmark$$

### 3. Classical Analog: Binary Symmetric Channel

The bit-flip channel is the quantum analog of the **binary symmetric channel (BSC)** in classical information theory:

```
Input    Output
  0  →(1-p)→  0
     ↘(p)↗
     ↗(p)↘
  1  →(1-p)→  1
```

**Key difference:** The quantum channel also affects superpositions!

### 4. Effect on Computational Basis States

For $|0\rangle$:
$$\mathcal{E}_X(|0\rangle\langle 0|) = (1-p)|0\rangle\langle 0| + p|1\rangle\langle 1|$$

For $|1\rangle$:
$$\mathcal{E}_X(|1\rangle\langle 1|) = (1-p)|1\rangle\langle 1| + p|0\rangle\langle 0|$$

These are classical mixtures—the populations are exchanged with probability $p$.

### 5. Effect on Superposition States

For $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

$$\rho_+ = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix}1 & 1\\1 & 1\end{pmatrix}$$

Since $X|+\rangle = |+\rangle$ (eigenstate!):
$$\mathcal{E}_X(\rho_+) = (1-p)\rho_+ + p\rho_+ = \rho_+$$

**The $|+\rangle$ state is unchanged by bit-flip errors!**

Similarly, $|-\rangle$ is also preserved since $X|-\rangle = -|-\rangle$ and $|-\rangle\langle -|$ is invariant.

### 6. Effect on General States

For a general state $\rho = \begin{pmatrix}a & b\\b^* & 1-a\end{pmatrix}$:

$$X\rho X = \begin{pmatrix}1-a & b^*\\b & a\end{pmatrix}$$

$$\mathcal{E}_X(\rho) = \begin{pmatrix}(1-p)a + p(1-a) & (1-2p)b\\(1-2p)b^* & (1-p)(1-a) + pa\end{pmatrix}$$

$$= \begin{pmatrix}a + p(1-2a) & (1-2p)b\\(1-2p)b^* & (1-a) - p(1-2a)\end{pmatrix}$$

**Key observations:**
- Diagonal elements (populations) mix toward equal populations
- Off-diagonal elements (coherences) contract by factor $(1-2p)$

### 7. Bloch Sphere Representation

Using Bloch vector $\vec{r} = (r_x, r_y, r_z)$:

$$\rho = \frac{1}{2}(I + r_x X + r_y Y + r_z Z)$$

After bit-flip channel:
$$\mathcal{E}_X(\rho) = \frac{1}{2}(I + r_x X + (1-2p)r_y Y + (1-2p)r_z Z)$$

**Effect on Bloch vector:**
$$\vec{r} \mapsto (r_x, (1-2p)r_y, (1-2p)r_z)$$

The Bloch sphere is **compressed toward the $x$-axis** by factor $(1-2p)$.

### 8. Fixed Points

**Fixed point equation:** $\mathcal{E}_X(\rho_*) = \rho_*$

**Solution:** Any state with $r_y = r_z = 0$:
$$\rho_* = \frac{1}{2}(I + r_x X) = \frac{1}{2}\begin{pmatrix}1 & r_x\\r_x & 1\end{pmatrix}$$

**Fixed points form a line** from $|+\rangle\langle +|$ through $I/2$ to $|-\rangle\langle -|$.

### 9. Repeated Application

Applying the bit-flip channel $n$ times:
$$\mathcal{E}_X^{(n)}(\rho) = (1-p_n)\rho + p_n X\rho X$$

where the effective probability is:
$$p_n = \frac{1 - (1-2p)^n}{2}$$

**Limit:** As $n \to \infty$, $p_n \to 1/2$ (maximally mixing in $y$-$z$ plane).

### 10. Physical Origins

**What causes bit-flip errors in real systems?**

1. **Transverse field fluctuations:** Random fields perpendicular to the quantization axis
2. **Resonant noise:** Noise at the qubit transition frequency
3. **Control errors:** Imperfect gate calibration
4. **Crosstalk:** Unintended interactions between qubits

**Example:** In superconducting qubits, flux noise can cause rotations around the $x$-axis.

---

## Quantum Computing Connection

### Error Probability in Gates

A single-qubit gate with bit-flip error probability $p$:
$$\mathcal{E}_{\text{noisy}}(\rho) = (1-p)U\rho U^\dagger + pXU\rho U^\dagger X$$

If $U = I$ (idle qubit), this reduces to the bit-flip channel.

### Error Accumulation

For a circuit with $n$ gates, each with error probability $p$:
- Total error probability $\approx np$ (for small $p$)
- Circuit fidelity $\approx (1-p)^n \approx e^{-np}$

### Why Bit-Flip is "Easy"

Bit-flip errors are considered "classical" because:
- They have a direct classical analog
- They can be detected by measuring in the computational basis
- Classical repetition codes can correct them

---

## Worked Examples

### Example 1: Bit-Flip on a Mixed State

**Problem:** Apply the bit-flip channel with $p = 0.1$ to the state $\rho = 0.8|0\rangle\langle 0| + 0.2|1\rangle\langle 1|$.

**Solution:**

Initial state:
$$\rho = \begin{pmatrix}0.8 & 0\\0 & 0.2\end{pmatrix}$$

Apply channel:
$$\mathcal{E}_X(\rho) = 0.9\begin{pmatrix}0.8 & 0\\0 & 0.2\end{pmatrix} + 0.1\begin{pmatrix}0.2 & 0\\0 & 0.8\end{pmatrix}$$

$$= \begin{pmatrix}0.72 + 0.02 & 0\\0 & 0.18 + 0.08\end{pmatrix} = \begin{pmatrix}0.74 & 0\\0 & 0.26\end{pmatrix}$$

**Result:** Populations have mixed toward 50-50.

---

### Example 2: Effect on Coherence

**Problem:** How does the bit-flip channel with $p = 0.2$ affect the coherence of a state starting with $\rho_{01} = 0.4$?

**Solution:**

From our analysis, off-diagonal elements transform as:
$$\rho_{01} \mapsto (1-2p)\rho_{01} = (1-0.4) \times 0.4 = 0.24$$

**Coherence reduced by factor $1-2p = 0.6$.**

---

### Example 3: Finding the Error Probability

**Problem:** A state $|+\rangle$ passes through a bit-flip channel and emerges with purity 0.95. What is the error probability?

**Solution:**

Since $|+\rangle$ is invariant under bit-flip, its purity remains 1.

Wait, let me reconsider. If the state is $|0\rangle$ instead:

Initial: $\rho = |0\rangle\langle 0|$, purity = 1

After bit-flip:
$$\rho' = (1-p)|0\rangle\langle 0| + p|1\rangle\langle 1| = \begin{pmatrix}1-p & 0\\0 & p\end{pmatrix}$$

Purity:
$$\text{Tr}(\rho'^2) = (1-p)^2 + p^2 = 1 - 2p + 2p^2 = 0.95$$

Solving: $2p^2 - 2p + 0.05 = 0$
$$p = \frac{2 \pm \sqrt{4 - 0.4}}{4} = \frac{2 \pm 1.897}{4}$$

$p = 0.026$ or $p = 0.974$ (physically, $p = 0.026$)

**Error probability $p \approx 0.026$.**

---

## Practice Problems

### Direct Application

1. **Problem 1:** Apply the bit-flip channel with $p = 0.15$ to the state $|1\rangle$.

2. **Problem 2:** Show that the maximally mixed state $I/2$ is a fixed point of the bit-flip channel.

3. **Problem 3:** Compute the Choi matrix for the bit-flip channel with $p = 0.1$.

### Intermediate

4. **Problem 4:** Find all eigenstates of the bit-flip channel (states $\rho$ such that $\mathcal{E}_X(\rho) = \lambda\rho$).

5. **Problem 5:** Two bit-flip channels with $p_1$ and $p_2$ are composed. Show the result is another bit-flip channel and find the effective probability.

6. **Problem 6:** Prove that the bit-flip channel is its own inverse when $p = 0.5$.

### Challenging

7. **Problem 7:** Design a protocol to distinguish a bit-flip channel with $p = 0.1$ from one with $p = 0.2$ using a single probe state.

8. **Problem 8:** Show that bit-flip errors can be corrected by the classical repetition code $|0\rangle \to |000\rangle$, $|1\rangle \to |111\rangle$.

9. **Problem 9:** Derive the master equation for continuous bit-flip noise: $\frac{d\rho}{dt} = \gamma(X\rho X - \rho)$.

---

## Computational Lab

```python
"""
Day 652 Computational Lab: Bit-Flip Errors
==========================================
Topics: Bit-flip channel implementation, analysis, visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

# Standard operators
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bit_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply bit-flip channel with probability p."""
    return (1 - p) * rho + p * X @ rho @ X


def density_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]:
    """Convert density matrix to Bloch vector."""
    r_x = 2 * np.real(rho[0, 1])
    r_y = 2 * np.imag(rho[1, 0])
    r_z = np.real(rho[0, 0] - rho[1, 1])
    return r_x, r_y, r_z


def bloch_to_density(r_x: float, r_y: float, r_z: float) -> np.ndarray:
    """Convert Bloch vector to density matrix."""
    return 0.5 * (I + r_x * X + r_y * Y + r_z * Z)


def purity(rho: np.ndarray) -> float:
    """Compute purity of density matrix."""
    return np.real(np.trace(rho @ rho))


# ===== DEMONSTRATIONS =====

print("=" * 70)
print("PART 1: Bit-Flip Channel on Basis States")
print("=" * 70)

p = 0.1
print(f"\nBit-flip probability: p = {p}")

# Computational basis states
rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)
rho_1 = np.array([[0, 0], [0, 1]], dtype=complex)

# Superposition states
rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
rho_minus = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex)

states = {
    '|0⟩': rho_0,
    '|1⟩': rho_1,
    '|+⟩': rho_plus,
    '|-⟩': rho_minus
}

print("\nEffect of bit-flip channel:")
for name, rho in states.items():
    rho_out = bit_flip_channel(rho, p)
    purity_in = purity(rho)
    purity_out = purity(rho_out)
    print(f"\n{name}:")
    print(f"  Input purity:  {purity_in:.4f}")
    print(f"  Output purity: {purity_out:.4f}")
    print(f"  Output state:\n{np.array2string(rho_out, precision=4)}")


print("\n" + "=" * 70)
print("PART 2: Bloch Sphere Transformation")
print("=" * 70)

def visualize_bit_flip_transform(p: float, n_points: int = 20):
    """Visualize how bit-flip channel transforms the Bloch sphere."""
    fig = plt.figure(figsize=(14, 6))

    # Generate points on Bloch sphere surface
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Input points (on sphere surface)
    x_in = np.sin(theta_grid) * np.cos(phi_grid)
    y_in = np.sin(theta_grid) * np.sin(phi_grid)
    z_in = np.cos(theta_grid)

    # Apply bit-flip channel to each point
    x_out = x_in.copy()  # x unchanged
    y_out = (1 - 2*p) * y_in  # y contracted
    z_out = (1 - 2*p) * z_in  # z contracted

    # Plot input (Bloch sphere)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x_in, y_in, z_in, alpha=0.3, color='blue')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Input: Bloch Sphere')
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-1.1, 1.1])

    # Plot output (compressed ellipsoid)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x_out, y_out, z_out, alpha=0.3, color='red')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Output: After Bit-Flip (p={p})')
    ax2.set_xlim([-1.1, 1.1])
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_zlim([-1.1, 1.1])

    plt.tight_layout()
    plt.savefig('bit_flip_bloch_transform.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: bit_flip_bloch_transform.png")

print("\nVisualizing Bloch sphere transformation:")
visualize_bit_flip_transform(0.2)


print("\n" + "=" * 70)
print("PART 3: Repeated Application")
print("=" * 70)

def effective_probability(p: float, n: int) -> float:
    """Effective bit-flip probability after n applications."""
    return 0.5 * (1 - (1 - 2*p)**n)

p = 0.1
n_applications = range(1, 51)

# Track effective probability and purity
eff_probs = [effective_probability(p, n) for n in n_applications]

# Track actual state evolution
rho = rho_0.copy()
purities = [1.0]
for n in n_applications:
    rho = bit_flip_channel(rho, p)
    purities.append(purity(rho))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(n_applications, eff_probs, 'b-', linewidth=2)
ax1.axhline(y=0.5, color='r', linestyle='--', label='Limit (p=0.5)')
ax1.set_xlabel('Number of applications')
ax1.set_ylabel('Effective error probability')
ax1.set_title(f'Effective Bit-Flip Probability (p={p})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(len(purities)), purities, 'g-', linewidth=2)
ax2.axhline(y=0.5, color='r', linestyle='--', label='Limit (maximally mixed)')
ax2.set_xlabel('Number of applications')
ax2.set_ylabel('Purity')
ax2.set_title('Purity Decay of |0⟩ State')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bit_flip_repeated.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bit_flip_repeated.png")


print("\n" + "=" * 70)
print("PART 4: Comparing Different Error Probabilities")
print("=" * 70)

error_probs = [0.01, 0.05, 0.1, 0.2, 0.3]

print("\nOff-diagonal contraction factor (1-2p):")
for p in error_probs:
    print(f"  p = {p:.2f}: contraction = {1-2*p:.2f}")

# Visualize coherence decay for different p values
fig, ax = plt.subplots(figsize=(10, 6))

n_steps = 50
for p in error_probs:
    coherences = [(1 - 2*p)**n for n in range(n_steps + 1)]
    ax.plot(range(n_steps + 1), coherences, linewidth=2, label=f'p={p}')

ax.set_xlabel('Number of bit-flip applications')
ax.set_ylabel('Coherence factor')
ax.set_title('Coherence Decay Under Repeated Bit-Flip')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('bit_flip_coherence_decay.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bit_flip_coherence_decay.png")


print("\n" + "=" * 70)
print("PART 5: Fixed Points Analysis")
print("=" * 70)

def is_fixed_point(rho: np.ndarray, p: float, tol: float = 1e-10) -> bool:
    """Check if rho is a fixed point of bit-flip channel."""
    rho_out = bit_flip_channel(rho, p)
    return np.max(np.abs(rho_out - rho)) < tol

print("\nTesting fixed points:")
test_states = {
    '|0⟩': rho_0,
    '|1⟩': rho_1,
    '|+⟩': rho_plus,
    '|-⟩': rho_minus,
    'I/2': I / 2,
    '(|+⟩+I/2)/2': 0.5 * rho_plus + 0.25 * I
}

for name, rho in test_states.items():
    is_fp = is_fixed_point(rho, 0.2)
    print(f"  {name}: {'Fixed point' if is_fp else 'Not fixed'}")


print("\n" + "=" * 70)
print("PART 6: Channel Composition")
print("=" * 70)

def composed_bit_flip_probability(p1: float, p2: float) -> float:
    """Effective probability when composing two bit-flip channels."""
    return p1 + p2 - 2*p1*p2

p1, p2 = 0.1, 0.15

p_composed = composed_bit_flip_probability(p1, p2)
print(f"\nComposing bit-flip channels:")
print(f"  p1 = {p1}, p2 = {p2}")
print(f"  Effective p = p1 + p2 - 2*p1*p2 = {p_composed:.4f}")

# Verify by comparing channel actions
rho_test = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)

# Sequential application
rho_seq = bit_flip_channel(bit_flip_channel(rho_test, p1), p2)

# Single effective channel
rho_eff = bit_flip_channel(rho_test, p_composed)

print(f"\nDifference between sequential and effective: "
      f"{np.max(np.abs(rho_seq - rho_eff)):.2e}")


print("\n" + "=" * 70)
print("PART 7: Error Detection with Repetition")
print("=" * 70)

def simulate_bit_flip_with_repetition(p: float, n_trials: int = 10000) -> dict:
    """
    Simulate bit-flip errors with 3-bit repetition.
    Returns statistics on error detection/correction.
    """
    results = {
        'no_error': 0,
        'one_error_corrected': 0,
        'multiple_errors_failed': 0
    }

    for _ in range(n_trials):
        # Generate errors on 3 bits
        errors = np.random.random(3) < p

        n_errors = sum(errors)
        if n_errors == 0:
            results['no_error'] += 1
        elif n_errors == 1:
            results['one_error_corrected'] += 1
        else:
            results['multiple_errors_failed'] += 1

    return {k: v/n_trials for k, v in results.items()}

print("\nSimulating 3-bit repetition code:")
for p in [0.01, 0.05, 0.1, 0.2]:
    results = simulate_bit_flip_with_repetition(p)
    print(f"\n  p = {p}:")
    print(f"    No error: {results['no_error']:.4f}")
    print(f"    1 error (corrected): {results['one_error_corrected']:.4f}")
    print(f"    2+ errors (failed): {results['multiple_errors_failed']:.4f}")


print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Bit-flip channel | $\mathcal{E}_X(\rho) = (1-p)\rho + pX\rho X$ |
| Kraus operators | $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p}X$ |
| Bloch transform | $(r_x, r_y, r_z) \mapsto (r_x, (1-2p)r_y, (1-2p)r_z)$ |
| Composition | $p_{\text{eff}} = p_1 + p_2 - 2p_1p_2$ |
| $n$ applications | $p_n = \frac{1 - (1-2p)^n}{2}$ |

### Main Takeaways

1. **Bit-flip channel** is the quantum analog of the classical binary symmetric channel
2. **$X$ eigenstates** ($|+\rangle$, $|-\rangle$) are fixed points
3. **Coherences decay** by factor $(1-2p)$ per application
4. **Bloch sphere contracts** toward the $x$-axis
5. Bit-flip errors are "classical" and can be corrected with repetition codes

---

## Daily Checklist

- [ ] I can write the Kraus representation for the bit-flip channel
- [ ] I understand which states are fixed points
- [ ] I can compute how coherences decay
- [ ] I understand the Bloch sphere transformation
- [ ] I know how bit-flip errors compose
- [ ] I completed the computational lab
- [ ] I attempted at least 3 practice problems

---

## Preview: Day 653

Tomorrow we study **phase-flip errors (Z)**, which:
- Are uniquely quantum (no classical analog)
- Affect coherences but not populations
- Are often harder to detect than bit-flip errors
- Model dephasing processes in real qubits

---

*"The bit-flip error is the friendly face of quantum noise—classical in nature and relatively easy to correct. Its cousin, the phase-flip, is where quantum mechanics shows its teeth."*
