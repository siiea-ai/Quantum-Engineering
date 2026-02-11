# Day 408: Coupled vs Uncoupled Basis

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Basis transformation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 408, you will be able to:

1. Write states in the uncoupled basis |j₁,m₁;j₂,m₂⟩
2. Write states in the coupled basis |j,m;j₁,j₂⟩
3. Understand when to use each basis
4. Describe the unitary transformation between bases
5. Count states and verify dimensions

---

## Core Content

### 1. The Uncoupled (Product) Basis

When combining two angular momenta, the **uncoupled basis** is:

$$|j_1, m_1; j_2, m_2\rangle = |j_1, m_1\rangle \otimes |j_2, m_2\rangle$$

Each subsystem is in a definite state of its own Ĵ₁² and Ĵ₁ᵤ (or Ĵ₂², Ĵ₂ᵤ).

**Dimension:** (2j₁+1)(2j₂+1)

**Example:** For j₁ = j₂ = 1/2 (two spin-1/2 particles):
- |↑↑⟩ = |1/2,1/2; 1/2,1/2⟩
- |↑↓⟩ = |1/2,1/2; 1/2,-1/2⟩
- |↓↑⟩ = |1/2,-1/2; 1/2,1/2⟩
- |↓↓⟩ = |1/2,-1/2; 1/2,-1/2⟩

### 2. The Coupled Basis

The **coupled basis** consists of eigenstates of the total angular momentum:

$$|j, m; j_1, j_2\rangle$$

where:
- Ĵ² |j,m⟩ = ℏ²j(j+1)|j,m⟩
- Ĵᵤ |j,m⟩ = ℏm|j,m⟩
- Ĵ₁² |j,m⟩ = ℏ²j₁(j₁+1)|j,m⟩
- Ĵ₂² |j,m⟩ = ℏ²j₂(j₂+1)|j,m⟩

### 3. Which Operators Commute?

**Both bases diagonalize:** Ĵ₁², Ĵ₂²

**Uncoupled basis also diagonalizes:** Ĵ₁ᵤ, Ĵ₂ᵤ

**Coupled basis also diagonalizes:** Ĵ², Ĵᵤ

Key: [Ĵ², Ĵ₁ᵤ] ≠ 0 and [Ĵ², Ĵ₂ᵤ] ≠ 0

### 4. The Basis Transformation

The two bases are related by:

$$|j, m\rangle = \sum_{m_1, m_2} \langle j_1, m_1; j_2, m_2 | j, m\rangle |j_1, m_1; j_2, m_2\rangle$$

The coefficients ⟨j₁,m₁;j₂,m₂|j,m⟩ are the **Clebsch-Gordan coefficients** (Day 409).

### 5. Selection Rule

From Ĵᵤ = Ĵ₁ᵤ + Ĵ₂ᵤ:

$$\boxed{m = m_1 + m_2}$$

This means only terms with m₁ + m₂ = m appear in the sum.

---

## Quantum Computing Connection

| Physics Basis | QC Interpretation |
|---------------|-------------------|
| Uncoupled |j₁,m₁;j₂,m₂⟩ | Product states \|q₁⟩⊗\|q₂⟩ |
| Coupled \|j,m⟩ | Entangled states |
| Basis change | Entangling gates |

The transformation from uncoupled to coupled basis creates entanglement!

---

## Worked Examples

### Example 1: Two Spin-1/2 Dimension Count

**Problem:** Verify the dimension for j₁ = j₂ = 1/2.

**Solution:**
Uncoupled basis: (2·1/2+1)(2·1/2+1) = 2×2 = 4 states

Coupled basis:
- j = 1: 2(1)+1 = 3 states (triplet)
- j = 0: 2(0)+1 = 1 state (singlet)
- Total: 3 + 1 = 4 states ✓

### Example 2: Identify Extreme States

**Problem:** Express |j=1, m=1⟩ for two spin-1/2.

**Solution:**
The maximum m = m₁ + m₂ = 1/2 + 1/2 = 1 can only come from |↑↑⟩.

Therefore:
$$|j=1, m=1\rangle = |{↑↑}\rangle$$

This is **not entangled** because it's a product state!

### Example 3: The m = 0 Subspace

**Problem:** What uncoupled states have m = 0 for two spin-1/2?

**Solution:**
m = m₁ + m₂ = 0 requires:
- m₁ = +1/2, m₂ = -1/2 → |↑↓⟩
- m₁ = -1/2, m₂ = +1/2 → |↓↑⟩

So |j=1,m=0⟩ and |j=0,m=0⟩ are both linear combinations of |↑↓⟩ and |↓↑⟩.

---

## Practice Problems

### Direct Application

1. List all uncoupled basis states for j₁ = 1, j₂ = 1/2.

2. What is the dimension of the coupled space for j₁ = 1, j₂ = 1?

3. For j₁ = 3/2, j₂ = 1/2, what are the allowed j values?

### Intermediate

4. Show that |j=j₁+j₂, m=j₁+j₂⟩ = |j₁,j₁; j₂,j₂⟩ (highest weight state).

5. For two spin-1/2, how many states have m = 0?

6. Why can't |↑↓⟩ be an eigenstate of Ĵ²?

### Challenging

7. Prove that the dimension of the coupled space equals (2j₁+1)(2j₂+1).

8. Show that |j,m⟩ and |j',m⟩ are orthogonal for j ≠ j'.

---

## Computational Lab

```python
"""
Day 408 Computational Lab: Coupled vs Uncoupled Basis
"""

import numpy as np
from itertools import product

def list_uncoupled_states(j1, j2):
    """List all uncoupled basis states."""
    m1_values = np.arange(j1, -j1-1, -1)
    m2_values = np.arange(j2, -j2-1, -1)

    states = []
    for m1 in m1_values:
        for m2 in m2_values:
            states.append((j1, m1, j2, m2))

    return states

def list_coupled_states(j1, j2):
    """List all coupled basis states using triangle rule."""
    states = []

    # Triangle rule: |j1-j2| <= j <= j1+j2
    j_min = abs(j1 - j2)
    j_max = j1 + j2

    j = j_max
    while j >= j_min:
        for m in np.arange(j, -j-1, -1):
            states.append((j, m, j1, j2))
        j -= 1

    return states

def verify_dimension(j1, j2):
    """Verify dimensions match."""
    uncoupled = list_uncoupled_states(j1, j2)
    coupled = list_coupled_states(j1, j2)

    dim_uncoupled = len(uncoupled)
    dim_coupled = len(coupled)

    print(f"\nj₁ = {j1}, j₂ = {j2}:")
    print(f"  Uncoupled dimension: {dim_uncoupled}")
    print(f"  Coupled dimension: {dim_coupled}")
    print(f"  Match: {dim_uncoupled == dim_coupled}")

    return dim_uncoupled == dim_coupled

def display_m_subspaces(j1, j2):
    """Show how states group by total m."""
    uncoupled = list_uncoupled_states(j1, j2)

    # Group by m = m1 + m2
    m_groups = {}
    for state in uncoupled:
        j1, m1, j2, m2 = state
        m = m1 + m2
        if m not in m_groups:
            m_groups[m] = []
        m_groups[m].append(state)

    print(f"\nUncoupled states grouped by m = m₁ + m₂:")
    for m in sorted(m_groups.keys(), reverse=True):
        print(f"  m = {m}: {len(m_groups[m])} states")
        for state in m_groups[m]:
            print(f"    |{state[0]},{state[1]}; {state[2]},{state[3]}⟩")

def two_spin_half_example():
    """Detailed example: two spin-1/2 particles."""
    print("\nTwo Spin-1/2 Particles (Qubits)")
    print("=" * 50)

    # Uncoupled states
    uncoupled = [
        ('|↑↑⟩', (0.5, 0.5, 0.5, 0.5), 1),
        ('|↑↓⟩', (0.5, 0.5, 0.5, -0.5), 0),
        ('|↓↑⟩', (0.5, -0.5, 0.5, 0.5), 0),
        ('|↓↓⟩', (0.5, -0.5, 0.5, -0.5), -1),
    ]

    print("\nUncoupled basis:")
    for name, state, m in uncoupled:
        print(f"  {name} has m = m₁ + m₂ = {m}")

    # Coupled states
    print("\nCoupled basis (triplet j=1 + singlet j=0):")
    print("  |1,+1⟩ = |↑↑⟩")
    print("  |1, 0⟩ = (|↑↓⟩ + |↓↑⟩)/√2  [symmetric]")
    print("  |1,-1⟩ = |↓↓⟩")
    print("  |0, 0⟩ = (|↑↓⟩ - |↓↑⟩)/√2  [antisymmetric]")

    print("\nKey insight: |0,0⟩ is the SINGLET state - maximally entangled!")

if __name__ == "__main__":
    print("Day 408: Coupled vs Uncoupled Basis")
    print("=" * 50)

    # Verify dimensions
    for j1, j2 in [(0.5, 0.5), (1, 0.5), (1, 1), (1.5, 0.5)]:
        verify_dimension(j1, j2)

    # Display m subspaces
    print("\n" + "=" * 50)
    display_m_subspaces(1, 0.5)

    # Two spin-1/2 example
    print("\n" + "=" * 50)
    two_spin_half_example()

    print("\nLab complete!")
```

---

## Summary

| Basis | Good Quantum Numbers | Use When |
|-------|---------------------|----------|
| Uncoupled | j₁, m₁, j₂, m₂ | Subsystems independent |
| Coupled | j, m, j₁, j₂ | Total angular momentum matters |

**Key:** Both bases span the same space; dimension = (2j₁+1)(2j₂+1)

---

## Daily Checklist

- [ ] I can write uncoupled basis states
- [ ] I know how to find allowed j values (triangle rule)
- [ ] I understand the selection rule m = m₁ + m₂
- [ ] I can count dimensions correctly
- [ ] I completed the computational lab

---

## Preview: Day 409

Tomorrow we calculate the Clebsch-Gordan coefficients that transform between the two bases.

---

**Next:** [Day_409_Wednesday.md](Day_409_Wednesday.md) — Clebsch-Gordan Coefficients
