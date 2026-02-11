# Day 396: Angular Momentum Eigenvalue Spectrum

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Complete eigenvalue derivation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 396, you will be able to:

1. Derive the complete eigenvalue spectrum of L̂² and L̂ᵤ
2. Prove that l must be integer or half-integer
3. Understand why orbital angular momentum requires integer l
4. Calculate the degeneracy (2l+1) for each l value
5. Construct the complete basis {|l,m⟩}

---

## Core Content

### 1. The Eigenvalue Problem

We seek simultaneous eigenstates of L̂² and L̂ᵤ (they commute):

$$\hat{L}^2|l,m\rangle = \lambda|l,m\rangle$$
$$\hat{L}_z|l,m\rangle = \mu|l,m\rangle$$

where we'll determine the allowed values of λ and μ.

### 2. Bounds on the Eigenvalues

**Physical Constraint:** L² = Lₓ² + Lᵧ² + Lᵤ² ≥ Lᵤ²

Since Lₓ² and Lᵧ² have non-negative eigenvalues:
$$\langle\hat{L}^2\rangle \geq \langle\hat{L}_z^2\rangle$$
$$\lambda \geq \mu^2$$

### 3. Ladder Operator Analysis

From Day 395, we know:
- L̂₊|l,m⟩ ∝ |l,m+1⟩ (raises m)
- L̂₋|l,m⟩ ∝ |l,m-1⟩ (lowers m)

Since λ ≥ μ², there must be maximum and minimum values of m.

**Maximum:** L̂₊|l,m_max⟩ = 0
**Minimum:** L̂₋|l,m_min⟩ = 0

### 4. Determining the Spectrum

From L̂₋L̂₊ = L̂² - L̂ᵤ² - ℏL̂ᵤ:

At the maximum:
$$\hat{L}_-\hat{L}_+|l,m_{max}\rangle = 0$$
$$(\hat{L}^2 - \hat{L}_z^2 - \hbar\hat{L}_z)|l,m_{max}\rangle = 0$$
$$\lambda - \mu_{max}^2 - \hbar\mu_{max} = 0$$

At the minimum:
$$\hat{L}_+\hat{L}_-|l,m_{min}\rangle = 0$$
$$(\hat{L}^2 - \hat{L}_z^2 + \hbar\hat{L}_z)|l,m_{min}\rangle = 0$$
$$\lambda - \mu_{min}^2 + \hbar\mu_{min} = 0$$

Subtracting:
$$\mu_{max}^2 + \hbar\mu_{max} = \mu_{min}^2 - \hbar\mu_{min}$$
$$(\mu_{max} + \mu_{min})(\mu_{max} - \mu_{min} + \hbar) = 0$$

**Solution 1:** μ_max + μ_min = 0 → μ_max = -μ_min

**Solution 2:** μ_max - μ_min = -ℏ (impossible since max > min)

Therefore: **μ_max = -μ_min**

### 5. Quantization Condition

From maximum to minimum, we apply L̂₋ exactly n times:
$$m_{min} = m_{max} - n\hbar$$

Combined with μ_max = -μ_min:
$$-m_{max} = m_{max} - n\hbar$$
$$2m_{max} = n\hbar$$
$$m_{max} = \frac{n\hbar}{2}$$

Define: **l ≡ m_max/ℏ = n/2**

Since n is a non-negative integer: **l = 0, 1/2, 1, 3/2, 2, ...**

### 6. The Complete Spectrum

$$\boxed{\hat{L}^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle}$$

$$\boxed{\hat{L}_z|l,m\rangle = \hbar m|l,m\rangle}$$

where:
- l = 0, 1/2, 1, 3/2, 2, ... (integer or half-integer)
- m = -l, -l+1, ..., l-1, l (2l+1 values)

**Degeneracy:** For each l, there are **(2l+1) states**.

### 7. Orbital vs Spin: Why l Must Be Integer

For **orbital** angular momentum L̂ = r̂ × p̂:

The eigenfunctions must be single-valued:
$$Y_l^m(\theta, \phi + 2\pi) = Y_l^m(\theta, \phi)$$

Since Y_l^m ∝ e^{imφ}, this requires e^{2πim} = 1.

Therefore **m must be integer**, which means **l must also be integer**.

Half-integer values (l = 1/2, 3/2, ...) correspond to **spin** angular momentum (Week 58).

---

## Quantum Computing Connection

| l value | # States | Physical System | QC Analog |
|---------|----------|-----------------|-----------|
| l = 0 | 1 | s orbital | — |
| l = 1 | 3 | p orbital | Qutrit |
| l = 1/2 | 2 | Spin-1/2 | **Qubit** |
| l = 1 (spin) | 3 | Spin-1 | Qutrit |

The qubit is a spin-1/2 system with l = 1/2, giving exactly 2 states.

---

## Worked Examples

### Example 1: l = 1 Subspace

**Problem:** List all states for l = 1 and verify the eigenvalues.

**Solution:**
For l = 1: m = -1, 0, +1 (three states)

States: |1,-1⟩, |1,0⟩, |1,1⟩

Eigenvalues:
- L̂²|1,m⟩ = ℏ²·1(1+1)|1,m⟩ = 2ℏ²|1,m⟩
- L̂ᵤ|1,m⟩ = ℏm|1,m⟩

Check: L̂₊|1,1⟩ = ℏ√[1(2)-1(2)]|1,2⟩ = 0 ✓

### Example 2: Matrix Representation for l = 1

**Problem:** Write L̂ᵤ and L̂± as 3×3 matrices in the basis {|1,1⟩, |1,0⟩, |1,-1⟩}.

**Solution:**

$$L_z = \hbar\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

For L̂₊: L̂₊|1,0⟩ = ℏ√2|1,1⟩, L̂₊|1,-1⟩ = ℏ√2|1,0⟩

$$L_+ = \hbar\sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

$$L_- = L_+^\dagger = \hbar\sqrt{2}\begin{pmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$

### Example 3: Total Number of States

**Problem:** How many states exist with L² ≤ 6ℏ²?

**Solution:**
L² = ℏ²l(l+1) ≤ 6ℏ² → l(l+1) ≤ 6

- l = 0: 0(1) = 0 ≤ 6 ✓ → 1 state
- l = 1: 1(2) = 2 ≤ 6 ✓ → 3 states
- l = 2: 2(3) = 6 ≤ 6 ✓ → 5 states
- l = 3: 3(4) = 12 > 6 ✗

Total: 1 + 3 + 5 = **9 states**

---

## Practice Problems

### Direct Application

1. List all allowed values of m for l = 3.

2. Calculate L̂²|2,-1⟩ and L̂ᵤ|2,-1⟩.

3. What is the degeneracy for l = 4?

### Intermediate

4. Show that the average value of L̂² over all m states (for fixed l) is ℏ²l(l+1).

5. Find the matrix representation of L̂ₓ for l = 1.

6. For l = 2, calculate ⟨2,1|L̂ₓ²|2,1⟩.

### Challenging

7. Prove that Tr(L̂ᵤ) = 0 for any l subspace.

8. Show that the eigenvalues of L̂ₓ in an l subspace are ℏm where m = -l, ..., +l (same as L̂ᵤ).

---

## Computational Lab

```python
"""
Day 396 Computational Lab: Angular Momentum Eigenvalue Spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def angular_momentum_spectrum():
    """
    Visualize the angular momentum eigenvalue spectrum.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: L^2 vs l
    l_values = np.arange(0, 6)
    L2_values = l_values * (l_values + 1)

    ax1 = axes[0]
    ax1.bar(l_values, L2_values, color='steelblue', edgecolor='black')
    ax1.set_xlabel('l (orbital quantum number)', fontsize=12)
    ax1.set_ylabel('L²/ℏ² = l(l+1)', fontsize=12)
    ax1.set_title('Angular Momentum Magnitude', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Annotate degeneracy
    for l in l_values:
        deg = 2*l + 1
        ax1.annotate(f'{deg} states', (l, l*(l+1)), textcoords="offset points",
                     xytext=(0, 5), ha='center', fontsize=10)

    # Right plot: m values for each l
    ax2 = axes[1]

    colors = plt.cm.viridis(np.linspace(0, 1, 6))

    for l in range(6):
        m_values = np.arange(-l, l+1)
        ax2.scatter([l]*len(m_values), m_values, c=[colors[l]], s=100,
                    label=f'l={l}' if l < 4 else None)

    ax2.set_xlabel('l', fontsize=12)
    ax2.set_ylabel('m', fontsize=12)
    ax2.set_title('Allowed m Values for Each l', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('angular_momentum_spectrum.png', dpi=150)
    plt.show()

def verify_eigenvalues(l):
    """
    Construct matrices and verify eigenvalues match l(l+1) and m.
    """
    dim = int(2*l + 1)
    m_values = np.arange(l, -l-1, -1)

    # Construct L_z
    Lz = np.diag(m_values).astype(complex)

    # Construct L_+ and L_-
    Lplus = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        m = m_values[i+1]
        Lplus[i, i+1] = np.sqrt(l*(l+1) - m*(m+1))

    Lminus = Lplus.T.conj()

    # Construct L_x, L_y
    Lx = (Lplus + Lminus) / 2
    Ly = (Lplus - Lminus) / (2j)

    # Construct L^2
    L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz

    # Verify L^2 eigenvalues
    L2_eigenvalues, _ = np.linalg.eigh(L2)

    print(f"\nl = {l}:")
    print(f"  Expected L²/ℏ² = {l*(l+1):.4f}")
    print(f"  Computed L²/ℏ² = {L2_eigenvalues}")
    print(f"  All equal? {np.allclose(L2_eigenvalues, l*(l+1))}")

    # Verify L_z eigenvalues
    Lz_eigenvalues = np.diag(Lz).real

    print(f"  Expected L_z/ℏ = {list(m_values)}")
    print(f"  Computed L_z/ℏ = {list(Lz_eigenvalues)}")

def visualize_vector_model(l):
    """
    Visualize the classical vector model of angular momentum.
    The magnitude |L| = sqrt(l(l+1)) * hbar, but L_z = m * hbar.
    This means L precesses around z-axis.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    L_magnitude = np.sqrt(l*(l+1))

    # Draw cone for each m value
    for m in range(-l, l+1):
        # Lz = m, |L| = sqrt(l(l+1))
        # Cone half-angle: theta = arccos(m / sqrt(l(l+1)))
        if L_magnitude > 0:
            cos_theta = m / L_magnitude
            if abs(cos_theta) <= 1:
                theta = np.arccos(cos_theta)

                # Draw circle at height z = m
                phi = np.linspace(0, 2*np.pi, 100)
                r = L_magnitude * np.sin(theta)
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                z = np.full_like(phi, m)

                ax.plot(x, y, z, 'b-', alpha=0.5)

                # Draw one L vector
                Lx = L_magnitude * np.sin(theta)
                Ly = 0
                Lz = m
                ax.quiver(0, 0, 0, Lx, Ly, Lz, color='red',
                          arrow_length_ratio=0.1, linewidth=2)

    # Draw z-axis
    ax.plot([0, 0], [0, 0], [-l-1, l+1], 'k-', linewidth=2)
    ax.text(0, 0, l+1.5, 'z', fontsize=14)

    ax.set_xlabel('Lx/ℏ')
    ax.set_ylabel('Ly/ℏ')
    ax.set_zlabel('Lz/ℏ')
    ax.set_title(f'Vector Model for l = {l}\n|L| = √{l*(l+1):.2f}ℏ, Lz = mℏ',
                 fontsize=14)

    # Set equal aspect ratio
    max_range = L_magnitude + 0.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-l-1, l+1])

    plt.tight_layout()
    plt.savefig(f'vector_model_l{l}.png', dpi=150)
    plt.show()

def degeneracy_plot():
    """
    Plot degeneracy (2l+1) vs l.
    """
    l_values = np.arange(0, 10)
    degeneracies = 2*l_values + 1

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(l_values, degeneracies, color='coral', edgecolor='black')
    ax.plot(l_values, degeneracies, 'ko-', markersize=8)

    ax.set_xlabel('Orbital quantum number l', fontsize=12)
    ax.set_ylabel('Degeneracy (2l + 1)', fontsize=12)
    ax.set_title('Number of States per l Value', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add orbital labels
    orbital_names = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    for i, (l, deg) in enumerate(zip(l_values, degeneracies)):
        ax.annotate(f'{orbital_names[i]}', (l, deg), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('degeneracy_plot.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Day 396: Angular Momentum Eigenvalue Spectrum")
    print("=" * 55)

    # Visualize spectrum
    print("\n1. Visualizing eigenvalue spectrum...")
    angular_momentum_spectrum()

    # Verify eigenvalues
    print("\n2. Verifying eigenvalues numerically...")
    for l in [1, 2, 3]:
        verify_eigenvalues(l)

    # Vector model
    print("\n3. Visualizing vector model...")
    visualize_vector_model(l=2)

    # Degeneracy
    print("\n4. Plotting degeneracy...")
    degeneracy_plot()

    print("\nLab complete!")
```

---

## Summary

| Result | Formula |
|--------|---------|
| L̂² eigenvalues | ℏ²l(l+1), l = 0,1,2,... (orbital) |
| L̂ᵤ eigenvalues | ℏm, m = -l,...,+l |
| Degeneracy | 2l + 1 states per l |
| Single-valuedness | Requires integer m, hence integer l |
| General angular momentum | l = 0, 1/2, 1, 3/2, ... allowed |

---

## Daily Checklist

- [ ] I can derive the eigenvalue spectrum using ladder operators
- [ ] I understand why l can be integer or half-integer
- [ ] I know why orbital angular momentum requires integer l
- [ ] I can calculate degeneracies for any l value
- [ ] I completed the computational lab

---

## Preview: Day 397

Tomorrow we introduce the spherical harmonics Y_l^m(θ,φ)—the explicit wave functions that are simultaneous eigenfunctions of L̂² and L̂ᵤ. These beautiful functions appear throughout physics, from atomic orbitals to gravitational waves.

---

**Next:** [Day_397_Friday.md](Day_397_Friday.md) — Spherical Harmonics I
