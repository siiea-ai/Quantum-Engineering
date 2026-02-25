# Day 395: Ladder Operators for Angular Momentum

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Ladder operators L± |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 395, you will be able to:

1. Define ladder operators L̂± = L̂ₓ ± iL̂ᵧ
2. Derive the commutation relations [L̂ᵤ, L̂±] = ±ℏL̂±
3. Express L̂² in terms of ladder operators
4. Determine the action of L̂± on eigenstates |l,m⟩
5. Prove the bounds on m: -l ≤ m ≤ l

---

## Core Content

### 1. Definition of Ladder Operators

The raising and lowering operators are defined as:

$$\boxed{\hat{L}_+ = \hat{L}_x + i\hat{L}_y}$$
$$\boxed{\hat{L}_- = \hat{L}_x - i\hat{L}_y}$$

Note: L̂₊† = L̂₋ (they are Hermitian conjugates of each other).

Inverse relations:
$$\hat{L}_x = \frac{1}{2}(\hat{L}_+ + \hat{L}_-)$$
$$\hat{L}_y = \frac{1}{2i}(\hat{L}_+ - \hat{L}_-)$$

### 2. Key Commutation Relations

Starting from [L̂ₓ, L̂ᵧ] = iℏL̂ᵤ, we derive:

$$\boxed{[\hat{L}_z, \hat{L}_+] = +\hbar\hat{L}_+}$$
$$\boxed{[\hat{L}_z, \hat{L}_-] = -\hbar\hat{L}_-}$$

**Proof:**
$$[\hat{L}_z, \hat{L}_+] = [\hat{L}_z, \hat{L}_x + i\hat{L}_y]$$
$$= [\hat{L}_z, \hat{L}_x] + i[\hat{L}_z, \hat{L}_y]$$
$$= i\hbar\hat{L}_y + i(-i\hbar\hat{L}_x) = i\hbar\hat{L}_y + \hbar\hat{L}_x$$
$$= \hbar(\hat{L}_x + i\hat{L}_y) = \hbar\hat{L}_+$$

Also important:
$$\boxed{[\hat{L}^2, \hat{L}_\pm] = 0}$$

### 3. L̂² in Terms of Ladder Operators

$$\hat{L}^2 = \hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2$$

Using L̂₊L̂₋ and L̂₋L̂₊:

$$\hat{L}_+\hat{L}_- = (\hat{L}_x + i\hat{L}_y)(\hat{L}_x - i\hat{L}_y) = \hat{L}_x^2 + \hat{L}_y^2 + i[\hat{L}_y, \hat{L}_x]$$
$$= \hat{L}_x^2 + \hat{L}_y^2 + \hbar\hat{L}_z$$

Therefore:
$$\boxed{\hat{L}^2 = \hat{L}_+\hat{L}_- + \hat{L}_z^2 - \hbar\hat{L}_z = \hat{L}_-\hat{L}_+ + \hat{L}_z^2 + \hbar\hat{L}_z}$$

### 4. Action of Ladder Operators

Let |l,m⟩ be simultaneous eigenstate of L̂² and L̂ᵤ:
- L̂²|l,m⟩ = ℏ²l(l+1)|l,m⟩
- L̂ᵤ|l,m⟩ = ℏm|l,m⟩

**Theorem:** L̂±|l,m⟩ is eigenstate of L̂ᵤ with eigenvalue ℏ(m±1).

**Proof:**
$$\hat{L}_z(\hat{L}_+|l,m\rangle) = (\hat{L}_+\hat{L}_z + [\hat{L}_z, \hat{L}_+])|l,m\rangle$$
$$= (\hat{L}_+\hat{L}_z + \hbar\hat{L}_+)|l,m\rangle$$
$$= \hat{L}_+(\hbar m)|l,m\rangle + \hbar\hat{L}_+|l,m\rangle$$
$$= \hbar(m+1)(\hat{L}_+|l,m\rangle)$$

So L̂₊ raises m by 1, and L̂₋ lowers m by 1.

### 5. Normalization and Bounds

Since L̂± doesn't change l (because [L̂², L̂±] = 0):
$$\hat{L}_+|l,m\rangle = c_+|l,m+1\rangle$$
$$\hat{L}_-|l,m\rangle = c_-|l,m-1\rangle$$

To find c±, use L̂₋L̂₊ = L̂² - L̂ᵤ² - ℏL̂ᵤ:

$$\langle l,m|\hat{L}_-\hat{L}_+|l,m\rangle = |c_+|^2$$
$$= \langle l,m|(\hat{L}^2 - \hat{L}_z^2 - \hbar\hat{L}_z)|l,m\rangle$$
$$= \hbar^2[l(l+1) - m^2 - m] = \hbar^2[l(l+1) - m(m+1)]$$

$$\boxed{c_+ = \hbar\sqrt{l(l+1) - m(m+1)} = \hbar\sqrt{(l-m)(l+m+1)}}$$

Similarly:
$$\boxed{c_- = \hbar\sqrt{l(l+1) - m(m-1)} = \hbar\sqrt{(l+m)(l-m+1)}}$$

**Bounds on m:**

For c₊ to be real (or zero at the top): l(l+1) - m(m+1) ≥ 0 → m ≤ l

For c₋ to be real (or zero at the bottom): l(l+1) - m(m-1) ≥ 0 → m ≥ -l

$$\boxed{-l \leq m \leq l}$$

---

## Quantum Computing Connection

| Ladder Operator | Qubit Gate |
|-----------------|------------|
| L̂₊ on spin-1/2 | σ₊ = (X + iY)/2 |
| L̂₋ on spin-1/2 | σ₋ = (X - iY)/2 |
| σ₊\|↓⟩ = \|↑⟩ | Bit flip from 1 to 0 |
| σ₋\|↑⟩ = \|↓⟩ | Bit flip from 0 to 1 |

---

## Worked Examples

### Example 1: Calculate [L̂₊, L̂₋]

**Problem:** Find the commutator [L̂₊, L̂₋].

**Solution:**
$$[\hat{L}_+, \hat{L}_-] = [\hat{L}_x + i\hat{L}_y, \hat{L}_x - i\hat{L}_y]$$
$$= [\hat{L}_x, \hat{L}_x] - i[\hat{L}_x, \hat{L}_y] + i[\hat{L}_y, \hat{L}_x] - i^2[\hat{L}_y, \hat{L}_y]$$
$$= 0 - i(i\hbar\hat{L}_z) + i(-i\hbar\hat{L}_z) - 0$$
$$= \hbar\hat{L}_z + \hbar\hat{L}_z = 2\hbar\hat{L}_z$$

$$\boxed{[\hat{L}_+, \hat{L}_-] = 2\hbar\hat{L}_z}$$

### Example 2: L̂₊|1,1⟩

**Problem:** Calculate L̂₊|1,1⟩.

**Solution:**
Using c₊ = ℏ√[(l-m)(l+m+1)]:

For l=1, m=1:
$$c_+ = \hbar\sqrt{(1-1)(1+1+1)} = \hbar\sqrt{0 \cdot 3} = 0$$

$$\boxed{\hat{L}_+|1,1\rangle = 0}$$

This confirms that m=l is the maximum—we can't raise further.

### Example 3: Build |1,0⟩ from |1,1⟩

**Problem:** Express |1,0⟩ using L̂₋ acting on |1,1⟩.

**Solution:**
$$\hat{L}_-|1,1\rangle = c_-|1,0\rangle$$

$$c_- = \hbar\sqrt{(l+m)(l-m+1)} = \hbar\sqrt{(1+1)(1-1+1)} = \hbar\sqrt{2}$$

$$|1,0\rangle = \frac{1}{\hbar\sqrt{2}}\hat{L}_-|1,1\rangle$$

---

## Practice Problems

### Direct Application

1. Calculate L̂₋|2,1⟩ and find the normalization constant.

2. Verify that L̂₊L̂₋ + L̂₋L̂₊ = 2(L̂² - L̂ᵤ²).

3. Show that ⟨l,m|L̂₊|l,m⟩ = 0.

### Intermediate

4. Calculate ⟨L̂ₓ⟩ and ⟨L̂ᵧ⟩ for the state |l,m⟩.

5. For the state |ψ⟩ = (|1,1⟩ + |1,-1⟩)/√2, find ⟨L̂²⟩, ⟨L̂ᵤ⟩, and ⟨L̂ₓ⟩.

6. Prove that (L̂₊)² |l,m⟩ ∝ |l,m+2⟩ and find the coefficient.

### Challenging

7. Show that L̂ₓ² + L̂ᵧ² = (L̂₊L̂₋ + L̂₋L̂₊)/2.

8. Derive the uncertainty relation ΔLₓ · ΔLᵧ ≥ (ℏ/2)|⟨L̂ᵤ⟩| for the state |l,m⟩.

---

## Computational Lab

```python
"""
Day 395 Computational Lab: Ladder Operators
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def angular_momentum_matrices(l):
    """
    Construct L_x, L_y, L_z, L_+, L_- matrices for angular momentum l.
    Dimension: (2l+1) x (2l+1)
    Basis: |l,l>, |l,l-1>, ..., |l,-l>
    """
    dim = int(2*l + 1)
    m_values = np.arange(l, -l-1, -1)  # l, l-1, ..., -l

    # L_z is diagonal
    Lz = np.diag(m_values).astype(complex)

    # L_+ (raising operator)
    Lplus = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        m = m_values[i+1]  # m value of the ket being raised
        Lplus[i, i+1] = np.sqrt(l*(l+1) - m*(m+1))

    # L_- (lowering operator)
    Lminus = Lplus.T.conj()

    # L_x and L_y
    Lx = (Lplus + Lminus) / 2
    Ly = (Lplus - Lminus) / (2j)

    # L^2
    L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz

    return Lx, Ly, Lz, Lplus, Lminus, L2

def verify_commutation_relations(l):
    """Verify [L_z, L_+] = hbar*L_+ etc."""
    Lx, Ly, Lz, Lplus, Lminus, L2 = angular_momentum_matrices(l)
    hbar = 1  # Natural units

    print(f"\nVerifying commutation relations for l = {l}:")

    # [L_z, L_+] = hbar*L_+
    comm1 = Lz @ Lplus - Lplus @ Lz
    diff1 = np.max(np.abs(comm1 - hbar*Lplus))
    print(f"  [L_z, L_+] = ℏL_+: Error = {diff1:.2e}")

    # [L_z, L_-] = -hbar*L_-
    comm2 = Lz @ Lminus - Lminus @ Lz
    diff2 = np.max(np.abs(comm2 - (-hbar*Lminus)))
    print(f"  [L_z, L_-] = -ℏL_-: Error = {diff2:.2e}")

    # [L_+, L_-] = 2*hbar*L_z
    comm3 = Lplus @ Lminus - Lminus @ Lplus
    diff3 = np.max(np.abs(comm3 - 2*hbar*Lz))
    print(f"  [L_+, L_-] = 2ℏL_z: Error = {diff3:.2e}")

    # [L^2, L_z] = 0
    comm4 = L2 @ Lz - Lz @ L2
    diff4 = np.max(np.abs(comm4))
    print(f"  [L², L_z] = 0: Error = {diff4:.2e}")

def demonstrate_ladder_action(l):
    """Show how ladder operators raise/lower m values."""
    Lx, Ly, Lz, Lplus, Lminus, L2 = angular_momentum_matrices(l)

    dim = int(2*l + 1)
    m_values = np.arange(l, -l-1, -1)

    print(f"\nLadder operator action for l = {l}:")
    print("-" * 50)

    for i, m in enumerate(m_values):
        # Create |l,m> state
        state = np.zeros(dim, dtype=complex)
        state[i] = 1.0

        # Apply L_+
        raised = Lplus @ state
        raised_norm = np.linalg.norm(raised)

        # Apply L_-
        lowered = Lminus @ state
        lowered_norm = np.linalg.norm(lowered)

        # Theoretical norms
        c_plus = np.sqrt(l*(l+1) - m*(m+1))
        c_minus = np.sqrt(l*(l+1) - m*(m-1))

        print(f"  |{l},{m:+d}⟩:")
        print(f"    L_+: ||L_+|l,m⟩|| = {raised_norm:.4f} (theory: {c_plus:.4f})")
        print(f"    L_-: ||L_-|l,m⟩|| = {lowered_norm:.4f} (theory: {c_minus:.4f})")

def plot_ladder_structure(l_max=3):
    """Visualize the ladder structure of angular momentum states."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, l_max+1))

    for l in range(l_max+1):
        m_values = np.arange(-l, l+1)
        x_positions = m_values

        # Plot states
        ax.scatter(x_positions, [l]*len(m_values), s=200, c=[colors[l]],
                   label=f'l = {l}', zorder=5)

        # Add labels
        for m in m_values:
            ax.annotate(f'|{l},{m}⟩', (m, l), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8)

        # Draw arrows for L_+ and L_-
        for i, m in enumerate(m_values[:-1]):
            # L_+ arrow (raising)
            ax.annotate('', xy=(m+0.7, l+0.05), xytext=(m+0.3, l+0.05),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        for i, m in enumerate(m_values[1:], 1):
            # L_- arrow (lowering)
            ax.annotate('', xy=(m-0.7, l-0.05), xytext=(m-0.3, l-0.05),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax.set_xlabel('m (magnetic quantum number)', fontsize=12)
    ax.set_ylabel('l (orbital quantum number)', fontsize=12)
    ax.set_title('Angular Momentum Ladder Structure\n(Red: L₊ raises m, Blue: L₋ lowers m)',
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-l_max-1, l_max+1)
    ax.set_ylim(-0.5, l_max+0.8)

    plt.tight_layout()
    plt.savefig('ladder_structure.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Day 395: Ladder Operators for Angular Momentum")
    print("=" * 55)

    # Verify commutation relations
    for l in [1, 2]:
        verify_commutation_relations(l)

    # Demonstrate ladder action
    demonstrate_ladder_action(l=2)

    # Plot ladder structure
    print("\nPlotting ladder structure...")
    plot_ladder_structure(l_max=3)

    print("\nLab complete!")
```

---

## Summary

| Concept | Formula |
|---------|---------|
| Raising operator | L̂₊ = L̂ₓ + iL̂ᵧ |
| Lowering operator | L̂₋ = L̂ₓ - iL̂ᵧ |
| Commutator with L̂ᵤ | [L̂ᵤ, L̂±] = ±ℏL̂± |
| L̂² expression | L̂² = L̂₊L̂₋ + L̂ᵤ² - ℏL̂ᵤ |
| Ladder action | L̂±\|l,m⟩ = ℏ√[l(l+1)-m(m±1)]\|l,m±1⟩ |
| Bounds | -l ≤ m ≤ l |

---

## Daily Checklist

- [ ] I can define L̂₊ and L̂₋
- [ ] I can derive [L̂ᵤ, L̂±] = ±ℏL̂±
- [ ] I understand how ladder operators change m
- [ ] I can calculate the normalization coefficients
- [ ] I understand why m is bounded by ±l

---

## Preview: Day 396

Tomorrow we complete the eigenvalue problem: using ladder operators, we prove that l must be a non-negative integer (for orbital angular momentum), giving the complete spectrum L² = ℏ²l(l+1) and Lᵤ = ℏm with m = -l, ..., +l.

---

**Next:** [Day_396_Thursday.md](Day_396_Thursday.md) — Eigenvalue Spectrum
