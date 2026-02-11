# Day 397: Spherical Harmonics I

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Spherical harmonics foundations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 397, you will be able to:

1. Define spherical harmonics as eigenfunctions of L̂² and L̂ᵤ
2. Write explicit forms for l = 0, 1, 2
3. Verify orthonormality on the unit sphere
4. Understand parity properties
5. Connect spherical harmonics to atomic orbitals

---

## Core Content

### 1. Definition and Properties

**Spherical harmonics** Y_l^m(θ,φ) are simultaneous eigenfunctions of L̂² and L̂ᵤ:

$$\hat{L}^2 Y_l^m(\theta,\phi) = \hbar^2 l(l+1) Y_l^m(\theta,\phi)$$
$$\hat{L}_z Y_l^m(\theta,\phi) = \hbar m Y_l^m(\theta,\phi)$$

### 2. Explicit Formula

$$\boxed{Y_l^m(\theta,\phi) = (-1)^m \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}}$$

where P_l^m(x) are the **associated Legendre polynomials**:

$$P_l^m(x) = \frac{(-1)^m}{2^l l!}(1-x^2)^{m/2}\frac{d^{l+m}}{dx^{l+m}}(x^2-1)^l$$

### 3. Explicit Forms for Low l

**l = 0 (s orbital):**
$$\boxed{Y_0^0 = \frac{1}{\sqrt{4\pi}}}$$

**l = 1 (p orbitals):**
$$Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta$$
$$Y_1^{\pm 1} = \mp\sqrt{\frac{3}{8\pi}}\sin\theta\, e^{\pm i\phi}$$

**l = 2 (d orbitals):**
$$Y_2^0 = \sqrt{\frac{5}{16\pi}}(3\cos^2\theta - 1)$$
$$Y_2^{\pm 1} = \mp\sqrt{\frac{15}{8\pi}}\sin\theta\cos\theta\, e^{\pm i\phi}$$
$$Y_2^{\pm 2} = \sqrt{\frac{15}{32\pi}}\sin^2\theta\, e^{\pm 2i\phi}$$

### 4. Orthonormality

$$\boxed{\int_0^{2\pi}\int_0^{\pi} Y_{l'}^{m'*}(\theta,\phi) Y_l^m(\theta,\phi) \sin\theta\, d\theta\, d\phi = \delta_{ll'}\delta_{mm'}}$$

Using the solid angle element dΩ = sinθ dθ dφ:
$$\int Y_{l'}^{m'*} Y_l^m\, d\Omega = \delta_{ll'}\delta_{mm'}$$

### 5. Completeness

Any square-integrable function on the sphere can be expanded:
$$f(\theta,\phi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l} c_{lm} Y_l^m(\theta,\phi)$$

where:
$$c_{lm} = \int Y_l^{m*}(\theta,\phi) f(\theta,\phi)\, d\Omega$$

### 6. Parity

Under inversion r̂ → -r̂ (equivalently θ → π-θ, φ → φ+π):

$$\boxed{Y_l^m(\pi-\theta, \phi+\pi) = (-1)^l Y_l^m(\theta,\phi)}$$

- l even: **even parity** (s, d, g, ...)
- l odd: **odd parity** (p, f, h, ...)

### 7. Complex Conjugate

$$Y_l^{-m}(\theta,\phi) = (-1)^m Y_l^{m*}(\theta,\phi)$$

---

## Quantum Computing Connection

| Spherical Harmonic | Physical System | Application |
|-------------------|-----------------|-------------|
| Y_0^0 | s orbital | Spherical symmetry |
| Y_1^m | p orbitals | Directional bonding |
| General Y_l^m | Angular momentum | Quantum simulation |

Spherical harmonics form a basis for representing quantum states with angular dependence on quantum computers via amplitude encoding.

---

## Worked Examples

### Example 1: Verify Normalization of Y_1^0

**Problem:** Show that ∫|Y_1^0|² dΩ = 1.

**Solution:**
$$Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta$$

$$\int |Y_1^0|^2 d\Omega = \frac{3}{4\pi}\int_0^{2\pi}d\phi\int_0^{\pi}\cos^2\theta\sin\theta\, d\theta$$

$$= \frac{3}{4\pi} \cdot 2\pi \cdot \int_0^{\pi}\cos^2\theta\sin\theta\, d\theta$$

Let u = cosθ, du = -sinθ dθ:
$$= \frac{3}{2}\int_{-1}^{1} u^2\, du = \frac{3}{2}\left[\frac{u^3}{3}\right]_{-1}^{1} = \frac{3}{2} \cdot \frac{2}{3} = 1$$

$$\boxed{\int |Y_1^0|^2 d\Omega = 1} \checkmark$$

### Example 2: Verify Orthogonality

**Problem:** Show that ∫Y_1^{0*}Y_1^1 dΩ = 0.

**Solution:**
$$Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta, \quad Y_1^1 = -\sqrt{\frac{3}{8\pi}}\sin\theta\, e^{i\phi}$$

$$\int Y_1^{0*}Y_1^1\, d\Omega = -\frac{3}{4\pi}\sqrt{\frac{1}{2}}\int_0^{2\pi}e^{i\phi}d\phi\int_0^{\pi}\cos\theta\sin^2\theta\, d\theta$$

The φ integral:
$$\int_0^{2\pi}e^{i\phi}d\phi = \left[\frac{e^{i\phi}}{i}\right]_0^{2\pi} = \frac{1}{i}(1-1) = 0$$

$$\boxed{\int Y_1^{0*}Y_1^1\, d\Omega = 0} \checkmark$$

### Example 3: Expand a Function

**Problem:** Expand f(θ,φ) = sinθ cosφ in spherical harmonics.

**Solution:**
Note that: sinθ cosφ = sinθ · (e^{iφ} + e^{-iφ})/2

Comparing with Y_1^{±1}:
$$Y_1^1 = -\sqrt{\frac{3}{8\pi}}\sin\theta\, e^{i\phi}$$
$$Y_1^{-1} = \sqrt{\frac{3}{8\pi}}\sin\theta\, e^{-i\phi}$$

So:
$$\sin\theta\cos\phi = -\sqrt{\frac{2\pi}{3}}(Y_1^1 - Y_1^{-1})$$

---

## Practice Problems

### Direct Application

1. Write Y_2^1 explicitly and verify it satisfies L̂ᵤY_2^1 = ℏY_2^1.

2. Calculate ∫|Y_2^0|² dΩ.

3. Show that Y_1^1 + Y_1^{-1} is proportional to sinθ cosφ (a real function).

### Intermediate

4. Expand f(θ,φ) = cos²θ in spherical harmonics.

5. Calculate ⟨Y_2^1|L̂₊|Y_2^0⟩.

6. Find the expansion of x/r = sinθ cosφ in spherical harmonics.

### Challenging

7. Prove the addition theorem for l = 1:
   $$\sum_{m=-1}^{1} Y_1^{m*}(\theta_1,\phi_1)Y_1^m(\theta_2,\phi_2) = \frac{3}{4\pi}\cos\gamma$$
   where γ is the angle between the two directions.

8. Show that ∫Y_{l_1}^{m_1}Y_{l_2}^{m_2}Y_{l_3}^{m_3*}dΩ is related to Clebsch-Gordan coefficients.

---

## Computational Lab

```python
"""
Day 397 Computational Lab: Spherical Harmonics Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

def plot_spherical_harmonic_3d(l, m, title=None):
    """
    Plot |Y_l^m|^2 as a 3D surface.
    """
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    # scipy.special.sph_harm uses convention sph_harm(m, l, phi, theta)
    Y = sph_harm(m, l, PHI, THETA)
    Y_abs2 = np.abs(Y)**2

    # Convert to Cartesian for 3D plot
    # Use |Y|^2 as the radial coordinate
    R = Y_abs2
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color based on sign of real part
    colors = np.real(Y)
    colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)

    ax.plot_surface(X, Y_coord, Z, facecolors=plt.cm.RdBu(colors),
                    alpha=0.8, antialiased=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if title is None:
        title = f'$|Y_{l}^{{{m}}}|^2$'
    ax.set_title(title, fontsize=14)

    # Equal aspect ratio
    max_range = np.max(np.abs([X, Y_coord, Z]))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    return fig, ax

def plot_all_harmonics(l_max=2):
    """
    Plot all spherical harmonics up to l_max.
    """
    n_cols = 2 * l_max + 1
    n_rows = l_max + 1

    fig = plt.figure(figsize=(3*n_cols, 3*n_rows))

    for l in range(l_max + 1):
        for m in range(-l, l+1):
            idx = l * n_cols + (m + l_max) + 1
            ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')

            theta = np.linspace(0, np.pi, 50)
            phi = np.linspace(0, 2*np.pi, 50)
            THETA, PHI = np.meshgrid(theta, phi)

            Y = sph_harm(m, l, PHI, THETA)
            R = np.abs(Y)

            X = R * np.sin(THETA) * np.cos(PHI)
            Y_coord = R * np.sin(THETA) * np.sin(PHI)
            Z = R * np.cos(THETA)

            colors = np.real(Y)
            norm_colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)

            ax.plot_surface(X, Y_coord, Z, facecolors=plt.cm.RdBu(norm_colors),
                            alpha=0.8, antialiased=True)

            ax.set_title(f'$Y_{l}^{{{m}}}$', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    plt.tight_layout()
    plt.savefig('all_spherical_harmonics.png', dpi=150)
    plt.show()

def verify_orthonormality(l_max=3):
    """
    Numerically verify orthonormality of spherical harmonics.
    """
    print("Verifying orthonormality of spherical harmonics:")
    print("=" * 50)

    n_theta = 100
    n_phi = 200

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    THETA, PHI = np.meshgrid(theta, phi)

    # Test orthonormality for a few pairs
    test_pairs = [
        ((0, 0), (0, 0)),  # Same state
        ((1, 0), (1, 0)),  # Same state
        ((1, 0), (1, 1)),  # Different m
        ((1, 1), (2, 1)),  # Different l
        ((2, 1), (2, 1)),  # Same state
    ]

    for (l1, m1), (l2, m2) in test_pairs:
        Y1 = sph_harm(m1, l1, PHI, THETA)
        Y2 = sph_harm(m2, l2, PHI, THETA)

        # Integrate Y1* Y2 sin(theta) dtheta dphi
        integrand = np.conj(Y1) * Y2 * np.sin(THETA)
        integral = np.sum(integrand) * dtheta * dphi

        expected = 1.0 if (l1 == l2 and m1 == m2) else 0.0

        print(f"∫ Y_{l1}^{m1}* Y_{l2}^{m2} dΩ = {integral:.6f} (expected: {expected})")

def real_spherical_harmonics():
    """
    Show the connection to real orbitals (px, py, pz, etc.)
    """
    print("\nReal Spherical Harmonics (Atomic Orbitals):")
    print("=" * 50)

    print("\nl = 1 (p orbitals):")
    print("  p_z ∝ Y_1^0 ∝ cos(θ)")
    print("  p_x ∝ (Y_1^{-1} - Y_1^1)/√2 ∝ sin(θ)cos(φ)")
    print("  p_y ∝ i(Y_1^{-1} + Y_1^1)/√2 ∝ sin(θ)sin(φ)")

    print("\nl = 2 (d orbitals):")
    print("  d_z² ∝ Y_2^0 ∝ 3cos²(θ) - 1")
    print("  d_xz ∝ (Y_2^{-1} - Y_2^1) ∝ sin(θ)cos(θ)cos(φ)")
    print("  d_yz ∝ i(Y_2^{-1} + Y_2^1) ∝ sin(θ)cos(θ)sin(φ)")
    print("  d_xy ∝ i(Y_2^{-2} + Y_2^2) ∝ sin²(θ)sin(2φ)")
    print("  d_x²-y² ∝ (Y_2^{-2} - Y_2^2) ∝ sin²(θ)cos(2φ)")

if __name__ == "__main__":
    print("Day 397: Spherical Harmonics Visualization")
    print("=" * 50)

    # Plot individual harmonic
    print("\n1. Plotting Y_2^1...")
    fig, ax = plot_spherical_harmonic_3d(2, 1)
    plt.savefig('Y_2_1.png', dpi=150)
    plt.show()

    # Plot all harmonics
    print("\n2. Plotting all harmonics up to l=2...")
    plot_all_harmonics(l_max=2)

    # Verify orthonormality
    print("\n3. Verifying orthonormality...")
    verify_orthonormality()

    # Real harmonics
    real_spherical_harmonics()

    print("\nLab complete!")
```

---

## Summary

| Quantity | Formula |
|----------|---------|
| Definition | Y_l^m(θ,φ) = eigenfunction of L̂², L̂ᵤ |
| Y_0^0 | 1/√(4π) |
| Y_1^0 | √(3/4π) cosθ |
| Orthonormality | ∫Y_{l'}^{m'*}Y_l^m dΩ = δ_{ll'}δ_{mm'} |
| Parity | Y_l^m(-r̂) = (-1)^l Y_l^m(r̂) |
| Completeness | f(θ,φ) = Σ c_{lm} Y_l^m |

---

## Daily Checklist

- [ ] I can write explicit forms of Y_l^m for l = 0, 1, 2
- [ ] I understand orthonormality on the sphere
- [ ] I know the parity of spherical harmonics
- [ ] I can expand functions in spherical harmonics
- [ ] I completed the computational lab

---

## Preview: Day 398

Tomorrow we explore more properties of spherical harmonics: the addition theorem, recursion relations, and connections to the Legendre polynomials. We'll also see how they appear in multipole expansions and quantum mechanics.

---

**Next:** [Day_398_Saturday.md](Day_398_Saturday.md) — Spherical Harmonics II
