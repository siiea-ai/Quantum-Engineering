# Day 843: Magic State Definition

## Week 121, Day 3 | Month 31: Fault-Tolerant QC I | Semester 2B: Fault Tolerance & Hardware

### Overview

Today we formally define magic states: quantum states that lie outside the stabilizer polytope and serve as resources for universal quantum computation. We focus on the two primary magic states, $|T\rangle$ and $|H\rangle$, exploring their geometric properties on the Bloch sphere and their relationship to non-Clifford gates. Understanding magic states is essential for fault-tolerant quantum computing, as they enable implementation of T-gates through gate teleportation.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Magic state definitions and properties |
| **Afternoon** | 2.5 hours | Stabilizer polytope and geometry |
| **Evening** | 1.5 hours | Computational lab: Magic state visualization |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define the magic state** $|T\rangle = (|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}$
2. **Define the alternative magic state** $|H\rangle$ and explain its properties
3. **Visualize magic states** on the Bloch sphere
4. **Explain the stabilizer polytope** and why magic states lie outside it
5. **Calculate magic state fidelity** and overlap with stabilizer states
6. **Understand the resource theory** of magic states

---

## Part 1: The Magic State |T⟩

### Definition

The primary magic state, denoted $|T\rangle$ (or sometimes $|A\rangle$), is defined as:

$$\boxed{|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)}$$

### Derivation from T-Gate

The magic state is created by applying the T-gate to $|+\rangle$:

$$|T\rangle = T|+\rangle = T\frac{|0\rangle + |1\rangle}{\sqrt{2}}$$

$$= \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ e^{i\pi/4} \end{pmatrix}$$

### Explicit Form

Writing $e^{i\pi/4} = \frac{1+i}{\sqrt{2}}$:

$$|T\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ \frac{1+i}{\sqrt{2}} \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1+i}{2} \end{pmatrix}$$

### Normalization Check

$$\langle T|T\rangle = |1/\sqrt{2}|^2 + |e^{i\pi/4}/\sqrt{2}|^2 = \frac{1}{2} + \frac{1}{2} = 1 \checkmark$$

### The Conjugate State

The orthogonal magic state is:

$$|T^\perp\rangle = \frac{1}{\sqrt{2}}(|0\rangle - e^{i\pi/4}|1\rangle) = T|-\rangle$$

Verify orthogonality:
$$\langle T^\perp|T\rangle = \frac{1}{2}(1 - e^{i\pi/4}e^{-i\pi/4}) = \frac{1}{2}(1-1) = 0 \checkmark$$

---

## Part 2: The Magic State |H⟩

### Definition

An alternative magic state, denoted $|H\rangle$ (for "Hadamard-type" magic), is:

$$\boxed{|H\rangle = \cos\frac{\pi}{8}|0\rangle + \sin\frac{\pi}{8}|1\rangle}$$

### Numerical Values

$$\cos(\pi/8) = \frac{\sqrt{2+\sqrt{2}}}{2} \approx 0.9239$$
$$\sin(\pi/8) = \frac{\sqrt{2-\sqrt{2}}}{2} \approx 0.3827$$

So:
$$|H\rangle \approx 0.9239|0\rangle + 0.3827|1\rangle$$

### Alternative Form

The $|H\rangle$ state can also be written as:

$$|H\rangle = \frac{1}{\sqrt{2}}(|+\rangle + e^{i\pi/4}|-\rangle)$$

### Relationship to Gates

The $|H\rangle$ state is the +1 eigenstate of $H T^\dagger H T$:

$$(HTH^\dagger)T|H\rangle = |H\rangle$$

This state enables implementation of the $\sqrt{T}$ gate.

---

## Part 3: Magic States on the Bloch Sphere

### Bloch Sphere Representation

Any pure single-qubit state can be written as:

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

The Bloch vector is $\vec{r} = (x, y, z)$ where:
$$x = \sin\theta\cos\phi, \quad y = \sin\theta\sin\phi, \quad z = \cos\theta$$

### |T⟩ Bloch Coordinates

For $|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$:

- $\cos(\theta/2) = 1/\sqrt{2}$ implies $\theta = \pi/2$ (equator)
- $\phi = \pi/4$

Bloch vector:
$$\vec{r}_T = (\sin(\pi/2)\cos(\pi/4), \sin(\pi/2)\sin(\pi/4), \cos(\pi/2))$$
$$\boxed{\vec{r}_T = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0\right)}$$

The magic state $|T\rangle$ lies on the equator at angle $\pi/4$ from the X-axis.

### |H⟩ Bloch Coordinates

For $|H\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$:

- $\theta/2 = \pi/8$ implies $\theta = \pi/4$
- $\phi = 0$

Bloch vector:
$$\vec{r}_H = (\sin(\pi/4), 0, \cos(\pi/4))$$
$$\boxed{\vec{r}_H = \left(\frac{1}{\sqrt{2}}, 0, \frac{1}{\sqrt{2}}\right)}$$

The magic state $|H\rangle$ lies at 45 degrees between the Z-axis and X-axis (in the XZ plane).

### Geometric Visualization

```
        Z (|0⟩)
          |
          |       • |H⟩ (45° from Z toward X)
          |     /
          |   /
          | /
  --------|--------Y
         /|
       /  |
     /    |
   /      |
  X       |
  (|+⟩)   |
          |
        -Z (|1⟩)

On the XY equator:
         Y
         |
    |T⟩ •|  (45° from X toward Y)
       \ |
        \|
  -------+-------> X (|+⟩)
         |
         |
```

---

## Part 4: The Stabilizer Polytope

### Stabilizer States

A state is a **stabilizer state** if it can be prepared from $|0\rangle^{\otimes n}$ using only Clifford gates.

For a single qubit, there are exactly **6 stabilizer states**:

| State | Bloch Vector | Name |
|-------|--------------|------|
| $\|0\rangle$ | (0, 0, 1) | North pole |
| $\|1\rangle$ | (0, 0, -1) | South pole |
| $\|+\rangle$ | (1, 0, 0) | +X |
| $\|-\rangle$ | (-1, 0, 0) | -X |
| $\|+i\rangle$ | (0, 1, 0) | +Y |
| $\|-i\rangle$ | (0, -1, 0) | -Y |

### The Octahedron

The 6 stabilizer states form the vertices of an **octahedron** inscribed in the Bloch sphere.

The **stabilizer polytope** for mixed states is the convex hull of these 6 points - i.e., the solid octahedron inside the Bloch sphere.

### States Inside the Polytope

A single-qubit state $\rho$ is a **stabilizer state** (possibly mixed) if and only if it can be written as:

$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

where each $|\psi_i\rangle$ is one of the 6 stabilizer states and $\sum_i p_i = 1$, $p_i \geq 0$.

Geometrically, this is any point inside the octahedron.

### Magic States Are Outside

The magic state $|T\rangle$ has Bloch vector $(1/\sqrt{2}, 1/\sqrt{2}, 0)$.

**Claim:** This lies outside the stabilizer octahedron.

**Proof:** The octahedron is defined by:
$$|x| + |y| + |z| \leq 1$$

For $|T\rangle$:
$$|x| + |y| + |z| = \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}} + 0 = \sqrt{2} > 1$$

Therefore, $|T\rangle$ is **outside** the stabilizer polytope!

$$\boxed{|T\rangle \text{ is a magic (non-stabilizer) state}}$$

### Why This Matters

- States **inside** the polytope can be simulated classically (Gottesman-Knill)
- States **outside** the polytope provide "magic" - quantum resource for universality
- The further outside, the more "magic" (more resource value)

---

## Part 5: Quantifying Magic

### Stabilizer Fidelity

The **stabilizer fidelity** of a state $|\psi\rangle$ is the maximum fidelity with any stabilizer state:

$$F_s(|\psi\rangle) = \max_{|\phi\rangle \in \text{STAB}} |\langle\phi|\psi\rangle|^2$$

For magic states, this is strictly less than 1.

### Magic State Fidelity with Stabilizers

For $|T\rangle$, the closest stabilizer states are $|+\rangle$ and $|+i\rangle$:

$$|\langle +|T\rangle|^2 = \left|\frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1 + e^{i\pi/4})\right|^2$$

$$= \frac{1}{4}|1 + e^{i\pi/4}|^2 = \frac{1}{4}\left|1 + \frac{1+i}{\sqrt{2}}\right|^2$$

$$= \frac{1}{4}\left(1 + \frac{1}{\sqrt{2}}\right)^2 + \frac{1}{4}\left(\frac{1}{\sqrt{2}}\right)^2$$

$$= \frac{1}{4}\left(1 + \frac{2}{\sqrt{2}} + \frac{1}{2} + \frac{1}{2}\right) = \frac{1}{4}(2 + \sqrt{2})$$

$$\boxed{|\langle +|T\rangle|^2 = \frac{2 + \sqrt{2}}{4} \approx 0.854}$$

### Robustness of Magic

The **robustness of magic** $\mathcal{R}(|\psi\rangle)$ quantifies how far outside the stabilizer polytope a state is:

$$\mathcal{R}(|\psi\rangle) = \min\left\{r : |\psi\rangle\langle\psi| = \frac{\sigma - r\tau}{1+r}, \sigma, \tau \in \text{STAB}\right\}$$

For $|T\rangle$:
$$\mathcal{R}(|T\rangle) = \sqrt{2} - 1 \approx 0.414$$

### Mana (Magic Monotone)

The **mana** of a state is another measure of magic:

$$\mathcal{M}(|\psi\rangle) = \log\left(\sum_P |\langle\psi|P|\psi\rangle|\right) - n$$

where the sum is over all n-qubit Paulis $P$.

For $|T\rangle$:
$$\mathcal{M}(|T\rangle) = \log(1 + \sqrt{2}) - 1 \approx 0.272$$

---

## Part 6: Magic State Variants

### The |Y⟩ Magic State

Another commonly used magic state:

$$|Y\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/2}|1\rangle) = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle) = |+i\rangle$$

Wait - this is a stabilizer state! The $|Y\rangle$ sometimes refers to different states in different contexts. Let's be precise:

The eigenstate of Y with eigenvalue +1:
$$Y|y_+\rangle = |y_+\rangle \implies |y_+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle) = |+i\rangle$$

This IS a stabilizer state.

### The |T⟩ vs |A⟩ Notation

Some papers use:
- $|T\rangle = T|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$
- $|A\rangle = |T\rangle$ (same state, different name)

### T-type vs H-type Magic

| Magic Type | State | Location on Bloch | Gate Enabled |
|------------|-------|-------------------|--------------|
| T-type | $\|T\rangle$ | Equator at $\pi/4$ | T-gate |
| H-type | $\|H\rangle$ | $45°$ latitude | Various |

Both are valid for universal computation, but have different distillation properties.

---

## Part 7: Quantum Computing Connection

### Why Magic States Enable Universality

1. **Clifford gates are transversal** on many codes
2. **Magic states can be prepared non-fault-tolerantly** at low fidelity
3. **Distillation uses only Clifford operations** to purify magic states
4. **Gate teleportation consumes magic states** to implement T-gates

This breaks the Eastin-Knill bottleneck: we don't need transversal T-gates!

### The Resource Picture

```
Classical simulation boundary:
┌─────────────────────────────────────────────────┐
│                                                 │
│   STABILIZER STATES           NON-STABILIZER    │
│   (octahedron)                STATES            │
│                                                 │
│      |0⟩ •                       • |T⟩         │
│           \                     /               │
│   |+⟩ •---•---• |-⟩          Magic             │
│           /\                  states           │
│          /  \                                  │
│      |1⟩ •    • |+i⟩                           │
│                                                 │
│   Classically simulable       Quantum power    │
│   (Gottesman-Knill)           (universality)   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Resource Cost of Magic

| Resource | Cost Estimate |
|----------|---------------|
| Raw $\|T\rangle$ preparation | 1 physical T-gate |
| 15-to-1 distillation | 15 raw states → 1 clean state |
| Surface code T-gate | ~15d² qubits, ~6d cycles |
| Typical algorithm T-count | $10^6 - 10^{12}$ |

---

## Worked Examples

### Example 1: Verify |T⟩ is Outside the Octahedron

**Problem:** Prove that $|T\rangle$ cannot be written as a convex combination of stabilizer states.

**Solution:**

The stabilizer polytope (octahedron) is defined by:
$$|x| + |y| + |z| \leq 1$$

For $|T\rangle$ with Bloch vector $\vec{r}_T = (1/\sqrt{2}, 1/\sqrt{2}, 0)$:

$$|x| + |y| + |z| = \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}} + 0 = \frac{2}{\sqrt{2}} = \sqrt{2} \approx 1.414$$

Since $\sqrt{2} > 1$, the point lies **outside** the octahedron.

Therefore, $|T\rangle$ cannot be expressed as a convex combination of stabilizer states. It is a genuine magic (non-stabilizer) state.

$$\boxed{|T\rangle \text{ is a magic state}}$$

---

### Example 2: Calculate Overlap with All Stabilizer States

**Problem:** Find $|\langle\psi|T\rangle|^2$ for all 6 stabilizer states $|\psi\rangle$.

**Solution:**

$|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$

**With $|0\rangle$:**
$$|\langle 0|T\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

**With $|1\rangle$:**
$$|\langle 1|T\rangle|^2 = \left|\frac{e^{i\pi/4}}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

**With $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:**
$$\langle +|T\rangle = \frac{1}{2}(1 + e^{i\pi/4}) = \frac{1}{2}\left(1 + \frac{1+i}{\sqrt{2}}\right)$$
$$|\langle +|T\rangle|^2 = \frac{1}{4}\left|1 + \frac{1+i}{\sqrt{2}}\right|^2 = \frac{2+\sqrt{2}}{4} \approx 0.854$$

**With $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$:**
$$\langle -|T\rangle = \frac{1}{2}(1 - e^{i\pi/4})$$
$$|\langle -|T\rangle|^2 = \frac{2-\sqrt{2}}{4} \approx 0.146$$

**With $|+i\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$:**
$$\langle +i|T\rangle = \frac{1}{2}(1 + e^{-i\pi/4}e^{i\pi/4}) = \frac{1}{2}(1 + 1) = 1$$
Wait, let me recalculate:
$$\langle +i|T\rangle = \frac{1}{2}(1 + (-i)(e^{i\pi/4})) = \frac{1}{2}(1 - ie^{i\pi/4})$$
$$= \frac{1}{2}(1 - i \cdot \frac{1+i}{\sqrt{2}}) = \frac{1}{2}(1 - \frac{i+i^2}{\sqrt{2}}) = \frac{1}{2}(1 - \frac{i-1}{\sqrt{2}})$$
$$= \frac{1}{2}(1 + \frac{1-i}{\sqrt{2}})$$
$$|\langle +i|T\rangle|^2 = \frac{1}{4}|1 + \frac{1-i}{\sqrt{2}}|^2 = \frac{2+\sqrt{2}}{4} \approx 0.854$$

**With $|-i\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$:**
Similarly: $|\langle -i|T\rangle|^2 = \frac{2-\sqrt{2}}{4} \approx 0.146$

**Summary:**
| Stabilizer | Fidelity with $\|T\rangle$ |
|------------|---------------------------|
| $\|0\rangle$ | 0.500 |
| $\|1\rangle$ | 0.500 |
| $\|+\rangle$ | 0.854 |
| $\|-\rangle$ | 0.146 |
| $\|+i\rangle$ | 0.854 |
| $\|-i\rangle$ | 0.146 |

---

### Example 3: |H⟩ State Verification

**Problem:** Verify that $|H\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$ is outside the stabilizer polytope.

**Solution:**

The Bloch vector for $|H\rangle$ is:
$$\vec{r}_H = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$$

where $\theta/2 = \pi/8$ (so $\theta = \pi/4$) and $\phi = 0$.

$$\vec{r}_H = (\sin(\pi/4), 0, \cos(\pi/4)) = \left(\frac{1}{\sqrt{2}}, 0, \frac{1}{\sqrt{2}}\right)$$

Check octahedron condition:
$$|x| + |y| + |z| = \frac{1}{\sqrt{2}} + 0 + \frac{1}{\sqrt{2}} = \sqrt{2} > 1$$

$$\boxed{|H\rangle \text{ is also a magic state}}$$

---

## Practice Problems

### Problem Set A: Direct Application

**A1.** Calculate the Bloch vector for $|T^\perp\rangle = T|-\rangle$.

**A2.** Show that $|T\rangle\langle T| + |T^\perp\rangle\langle T^\perp| = I$ (completeness).

**A3.** What is the angle between the Bloch vectors of $|T\rangle$ and $|H\rangle$?

### Problem Set B: Intermediate

**B1.** Find the stabilizer state closest to $|H\rangle$ and compute the fidelity.

**B2.** Show that $T^2|T\rangle = S|T\rangle$ produces another magic state. Where is it on the Bloch sphere?

**B3.** The state $|m\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\theta}|1\rangle)$ is magic for what values of $\theta$?

### Problem Set C: Challenging

**C1.** Prove that the convex hull of the 6 stabilizer states is exactly the octahedron $|x| + |y| + |z| \leq 1$.

**C2.** Calculate the robustness of magic for $|H\rangle$.

**C3.** **(Research-level)** For two-qubit states, describe the stabilizer polytope. How many vertices does it have?

---

## Computational Lab

```python
"""
Day 843 Computational Lab: Magic States and the Stabilizer Polytope
Visualizes magic states, stabilizer states, and the octahedron
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, List

# =============================================================================
# Part 1: State and Bloch Vector Utilities
# =============================================================================

def state_to_bloch(state: np.ndarray) -> Tuple[float, float, float]:
    """Convert quantum state |psi> to Bloch sphere coordinates."""
    state = state / np.linalg.norm(state)
    alpha, beta = state[0], state[1]

    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2

    return float(x), float(y), float(z)

def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
    """Convert Bloch coordinates to quantum state."""
    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    alpha = np.cos(theta/2)
    beta = np.exp(1j * phi) * np.sin(theta/2)

    return np.array([alpha, beta], dtype=complex)

# =============================================================================
# Part 2: Define Key States
# =============================================================================

# Computational basis
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Stabilizer states
stabilizer_states = {
    '|0⟩': ket_0,
    '|1⟩': ket_1,
    '|+⟩': (ket_0 + ket_1) / np.sqrt(2),
    '|-⟩': (ket_0 - ket_1) / np.sqrt(2),
    '|+i⟩': (ket_0 + 1j * ket_1) / np.sqrt(2),
    '|-i⟩': (ket_0 - 1j * ket_1) / np.sqrt(2),
}

# Magic states
T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

magic_states = {
    '|T⟩': T_gate @ stabilizer_states['|+⟩'],
    '|T⊥⟩': T_gate @ stabilizer_states['|-⟩'],
    '|H⟩': np.array([np.cos(np.pi/8), np.sin(np.pi/8)], dtype=complex),
}

print("=" * 60)
print("MAGIC STATES ANALYSIS")
print("=" * 60)

# =============================================================================
# Part 3: Bloch Vectors
# =============================================================================

print("\nStabilizer States (Bloch vectors):")
for name, state in stabilizer_states.items():
    x, y, z = state_to_bloch(state)
    print(f"  {name}: ({x:+.4f}, {y:+.4f}, {z:+.4f})")

print("\nMagic States (Bloch vectors):")
for name, state in magic_states.items():
    x, y, z = state_to_bloch(state)
    L1_norm = abs(x) + abs(y) + abs(z)
    print(f"  {name}: ({x:+.4f}, {y:+.4f}, {z:+.4f}), |x|+|y|+|z| = {L1_norm:.4f}")

# =============================================================================
# Part 4: Verify Octahedron Condition
# =============================================================================

print("\n" + "=" * 60)
print("STABILIZER POLYTOPE (OCTAHEDRON) CHECK")
print("=" * 60)

print("\nOctahedron condition: |x| + |y| + |z| ≤ 1")
print("\nStabilizer states:")
for name, state in stabilizer_states.items():
    x, y, z = state_to_bloch(state)
    L1 = abs(x) + abs(y) + abs(z)
    status = "INSIDE ✓" if L1 <= 1 + 1e-10 else "OUTSIDE ✗"
    print(f"  {name}: L1 = {L1:.4f} → {status}")

print("\nMagic states:")
for name, state in magic_states.items():
    x, y, z = state_to_bloch(state)
    L1 = abs(x) + abs(y) + abs(z)
    status = "INSIDE" if L1 <= 1 + 1e-10 else "OUTSIDE (MAGIC!) ✓"
    print(f"  {name}: L1 = {L1:.4f} → {status}")

# =============================================================================
# Part 5: Fidelities with Stabilizer States
# =============================================================================

print("\n" + "=" * 60)
print("FIDELITY OF |T⟩ WITH STABILIZER STATES")
print("=" * 60)

T_state = magic_states['|T⟩']
print(f"\n|T⟩ = {T_state}")

for name, stab_state in stabilizer_states.items():
    fidelity = np.abs(np.vdot(stab_state, T_state))**2
    print(f"  |⟨{name[1:-1]}|T⟩|² = {fidelity:.4f}")

max_fidelity = max(np.abs(np.vdot(s, T_state))**2 for s in stabilizer_states.values())
print(f"\n  Maximum stabilizer fidelity: {max_fidelity:.4f}")
print(f"  Magic 'distance' from stabilizers: {1 - max_fidelity:.4f}")

# =============================================================================
# Part 6: Quantifying Magic
# =============================================================================

print("\n" + "=" * 60)
print("QUANTIFYING MAGIC")
print("=" * 60)

def robustness_of_magic_pure(state: np.ndarray) -> float:
    """Calculate robustness of magic for a pure single-qubit state."""
    x, y, z = state_to_bloch(state)
    L1 = abs(x) + abs(y) + abs(z)
    if L1 <= 1:
        return 0  # Stabilizer state
    return L1 - 1

for name, state in magic_states.items():
    rom = robustness_of_magic_pure(state)
    print(f"  Robustness of magic for {name}: {rom:.4f}")

# =============================================================================
# Part 7: Visualization
# =============================================================================

fig = plt.figure(figsize=(18, 12))

# Plot 1: Bloch sphere with stabilizer states and octahedron
ax1 = fig.add_subplot(2, 3, 1, projection='3d')

# Draw sphere wireframe
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Plot stabilizer states
for name, state in stabilizer_states.items():
    x, y, z = state_to_bloch(state)
    ax1.scatter([x], [y], [z], s=150, c='blue', marker='o', edgecolors='black')

# Plot magic states
for name, state in magic_states.items():
    x, y, z = state_to_bloch(state)
    marker = '*' if 'T' in name else 'd'
    ax1.scatter([x], [y], [z], s=250, c='red', marker=marker, edgecolors='black')

# Draw octahedron
vertices = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
])

faces = [
    [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
    [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]
]

for face in faces:
    tri = [vertices[i] for i in face]
    ax1.add_collection3d(Poly3DCollection([tri], alpha=0.2, facecolor='cyan', edgecolor='blue'))

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Bloch Sphere with Stabilizer Octahedron\n(Blue=Stabilizer, Red=Magic)', fontsize=11)

# Plot 2: XY plane view (equator)
ax2 = fig.add_subplot(2, 3, 2)

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)

# Draw octahedron cross-section (diamond shape at z=0)
diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
ax2.fill(diamond[:, 0], diamond[:, 1], alpha=0.3, color='cyan', edgecolor='blue')

# Plot equator stabilizer states
for name, state in stabilizer_states.items():
    x, y, z = state_to_bloch(state)
    if abs(z) < 0.01:  # On equator
        ax2.scatter([x], [y], s=150, c='blue', marker='o', edgecolors='black', zorder=5)
        ax2.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

# Plot magic states on equator
T_x, T_y, T_z = state_to_bloch(magic_states['|T⟩'])
ax2.scatter([T_x], [T_y], s=250, c='red', marker='*', edgecolors='black', zorder=5)
ax2.annotate('|T⟩', (T_x, T_y), xytext=(5, 5), textcoords='offset points', fontsize=11, color='red')

Tp_x, Tp_y, Tp_z = state_to_bloch(magic_states['|T⊥⟩'])
ax2.scatter([Tp_x], [Tp_y], s=200, c='orange', marker='*', edgecolors='black', zorder=5)
ax2.annotate('|T⊥⟩', (Tp_x, Tp_y), xytext=(5, -15), textcoords='offset points', fontsize=10, color='orange')

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('XY Plane (Equator)\n|T⟩ outside octahedron diamond', fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: XZ plane view
ax3 = fig.add_subplot(2, 3, 3)

# Draw unit circle
ax3.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)

# Draw octahedron cross-section (diamond shape at y=0)
ax3.fill(diamond[:, 0], diamond[:, 1], alpha=0.3, color='cyan', edgecolor='blue')

# Plot XZ stabilizer states
for name, state in stabilizer_states.items():
    x, y, z = state_to_bloch(state)
    if abs(y) < 0.01:  # In XZ plane
        ax3.scatter([x], [z], s=150, c='blue', marker='o', edgecolors='black', zorder=5)
        ax3.annotate(name, (x, z), xytext=(5, 5), textcoords='offset points', fontsize=9)

# Plot |H⟩ magic state
H_x, H_y, H_z = state_to_bloch(magic_states['|H⟩'])
ax3.scatter([H_x], [H_z], s=250, c='red', marker='d', edgecolors='black', zorder=5)
ax3.annotate('|H⟩', (H_x, H_z), xytext=(5, 5), textcoords='offset points', fontsize=11, color='red')

ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_title('XZ Plane\n|H⟩ outside octahedron', fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Fidelity bar chart
ax4 = fig.add_subplot(2, 3, 4)

stab_names = list(stabilizer_states.keys())
fidelities = [np.abs(np.vdot(stabilizer_states[n], magic_states['|T⟩']))**2 for n in stab_names]

colors = ['green' if f > 0.8 else ('yellow' if f > 0.5 else 'red') for f in fidelities]
bars = ax4.bar(stab_names, fidelities, color=colors, edgecolor='black')
ax4.axhline(y=0.5, color='gray', linestyle='--', label='Random guess')
ax4.set_ylim(0, 1)
ax4.set_ylabel('Fidelity |⟨stab|T⟩|²')
ax4.set_title('|T⟩ Fidelity with Stabilizer States', fontsize=11)

for bar, f in zip(bars, fidelities):
    ax4.text(bar.get_x() + bar.get_width()/2, f + 0.02, f'{f:.3f}',
             ha='center', fontsize=9)

# Plot 5: L1 norm comparison
ax5 = fig.add_subplot(2, 3, 5)

all_states = {**stabilizer_states, **magic_states}
state_names = list(all_states.keys())
L1_norms = [sum(abs(c) for c in state_to_bloch(all_states[n])) for n in state_names]

colors = ['blue' if L1 <= 1 else 'red' for L1 in L1_norms]
bars = ax5.bar(range(len(state_names)), L1_norms, color=colors, edgecolor='black')
ax5.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Octahedron boundary')
ax5.set_xticks(range(len(state_names)))
ax5.set_xticklabels(state_names, rotation=45, ha='right')
ax5.set_ylabel('|x| + |y| + |z|')
ax5.set_title('L1 Norm: Stabilizer vs Magic\n(≤1 = inside octahedron)', fontsize=11)
ax5.legend()

# Plot 6: Magic state trajectory
ax6 = fig.add_subplot(2, 3, 6)

# Draw unit circle
ax6.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, label='Bloch sphere equator')

# Draw octahedron
ax6.fill(diamond[:, 0], diamond[:, 1], alpha=0.2, color='cyan', edgecolor='blue', label='Stabilizer region')

# Plot trajectory of T^k|+⟩ for k = 0, 1, ..., 7
plus = stabilizer_states['|+⟩']
for k in range(8):
    T_k = np.linalg.matrix_power(T_gate, k)
    state_k = T_k @ plus
    x, y, z = state_to_bloch(state_k)

    color = 'blue' if k % 2 == 0 else 'red'  # Even = Clifford, Odd = non-Clifford
    marker = 'o' if abs(x) + abs(y) + abs(z) <= 1 + 1e-10 else '*'

    ax6.scatter([x], [y], s=150, c=color, marker=marker, edgecolors='black', zorder=5)
    ax6.annotate(f'T^{k}|+⟩', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)

ax6.set_xlim(-1.5, 1.5)
ax6.set_ylim(-1.5, 1.5)
ax6.set_aspect('equal')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_title('Trajectory of T^k|+⟩ (k=0..7)\nBlue=even (Clifford), Red=odd (magic)', fontsize=11)

plt.tight_layout()
plt.savefig('day_843_magic_states.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Visualization saved to: day_843_magic_states.png")
print("=" * 60)

# =============================================================================
# Part 8: Summary
# =============================================================================

print("\n" + "=" * 60)
print("KEY RESULTS SUMMARY")
print("=" * 60)

summary = """
MAGIC STATE |T⟩:
  |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2 = T|+⟩
  Bloch vector: (1/√2, 1/√2, 0)

MAGIC STATE |H⟩:
  |H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩
  Bloch vector: (1/√2, 0, 1/√2)

STABILIZER POLYTOPE:
  Single-qubit: Octahedron |x|+|y|+|z| ≤ 1
  6 vertices: ±X, ±Y, ±Z poles

WHY MAGIC:
  |T⟩: L1 = √2 ≈ 1.414 > 1 → OUTSIDE
  Cannot be prepared by Clifford circuits
  Enables universal quantum computation

QUANTIFYING MAGIC:
  Robustness of magic R(|T⟩) = √2 - 1 ≈ 0.414
  Max stabilizer fidelity ≈ 0.854
"""
print(summary)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Magic state $\|T\rangle$ | $\frac{1}{\sqrt{2}}(\|0\rangle + e^{i\pi/4}\|1\rangle) = T\|+\rangle$ |
| Magic state $\|H\rangle$ | $\cos(\pi/8)\|0\rangle + \sin(\pi/8)\|1\rangle$ |
| $\|T\rangle$ Bloch vector | $(1/\sqrt{2}, 1/\sqrt{2}, 0)$ |
| $\|H\rangle$ Bloch vector | $(1/\sqrt{2}, 0, 1/\sqrt{2})$ |
| Octahedron condition | $\|x\| + \|y\| + \|z\| \leq 1$ |
| $\|T\rangle$ L1 norm | $\sqrt{2} > 1$ (outside octahedron) |
| Max stabilizer fidelity | $F_s(\|T\rangle) = \frac{2+\sqrt{2}}{4} \approx 0.854$ |

### Main Takeaways

1. **Magic states are non-stabilizer states** - They cannot be prepared from $|0\rangle$ using only Clifford gates

2. **The stabilizer polytope is an octahedron** - Its vertices are the 6 stabilizer states $|\pm X\rangle, |\pm Y\rangle, |\pm Z\rangle$

3. **Magic states lie outside the octahedron** - $|T\rangle$ has $|x| + |y| + |z| = \sqrt{2} > 1$

4. **Magic states enable universality** - They provide the "quantum resource" missing from Clifford circuits

5. **Two primary magic states:** $|T\rangle$ (T-type) and $|H\rangle$ (H-type), with different properties for distillation

---

## Daily Checklist

- [ ] Can write the magic state $|T\rangle$ in multiple forms
- [ ] Can compute Bloch vectors for magic states
- [ ] Understand the stabilizer polytope (octahedron)
- [ ] Can verify a state is outside the octahedron
- [ ] Know the difference between $|T\rangle$ and $|H\rangle$ magic states
- [ ] Understand why magic states enable universality
- [ ] Completed computational lab exercises

---

## Preview: Day 844

Tomorrow we learn **gate teleportation**: how to use magic states to implement the T-gate fault-tolerantly. We will:

- Construct the gate teleportation circuit
- Understand the measurement and correction operations
- Analyze why this converts a non-fault-tolerant resource into a fault-tolerant gate
- Calculate success probabilities and resource requirements

Gate teleportation is the key technique that makes magic states useful!

---

*"Magic states are the fuel that powers universal quantum computation. Distill them carefully, spend them wisely."*

---

**Day 843 Complete** | **Next: Day 844 - Gate Teleportation**
