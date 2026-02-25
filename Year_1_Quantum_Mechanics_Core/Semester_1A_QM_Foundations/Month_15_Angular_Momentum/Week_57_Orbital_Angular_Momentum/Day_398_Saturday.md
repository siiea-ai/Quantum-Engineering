# Day 398: Spherical Harmonics II — Advanced Properties

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Advanced spherical harmonic properties |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 398, you will be able to:

1. Derive and apply the addition theorem for spherical harmonics
2. Use recursion relations for efficient computation
3. Connect spherical harmonics to associated Legendre polynomials
4. Understand multipole expansions in electrostatics
5. Convert between complex and real spherical harmonics

---

## Core Content

### 1. Associated Legendre Polynomials

The spherical harmonics factor as:
$$Y_l^m(\theta,\phi) = N_{lm} P_l^m(\cos\theta) e^{im\phi}$$

where the **associated Legendre polynomials** are:

$$P_l^m(x) = (-1)^m(1-x^2)^{m/2}\frac{d^m}{dx^m}P_l(x)$$

and P_l(x) are the **Legendre polynomials**:
$$P_l(x) = \frac{1}{2^l l!}\frac{d^l}{dx^l}(x^2-1)^l$$

**First few Legendre polynomials:**
- P₀(x) = 1
- P₁(x) = x
- P₂(x) = (3x² - 1)/2
- P₃(x) = (5x³ - 3x)/2

### 2. Addition Theorem

The fundamental addition theorem relates products of spherical harmonics:

$$\boxed{P_l(\cos\gamma) = \frac{4\pi}{2l+1}\sum_{m=-l}^{l} Y_l^{m*}(\theta_1,\phi_1)Y_l^m(\theta_2,\phi_2)}$$

where γ is the angle between directions (θ₁,φ₁) and (θ₂,φ₂):
$$\cos\gamma = \cos\theta_1\cos\theta_2 + \sin\theta_1\sin\theta_2\cos(\phi_1-\phi_2)$$

**Equivalent form:**
$$\sum_{m=-l}^{l} Y_l^{m*}(\hat{r}_1)Y_l^m(\hat{r}_2) = \frac{2l+1}{4\pi}P_l(\hat{r}_1 \cdot \hat{r}_2)$$

### 3. Recursion Relations

**Three-term recursion for P_l^m:**
$$(l-m+1)P_{l+1}^m(x) = (2l+1)xP_l^m(x) - (l+m)P_{l-1}^m(x)$$

**Raising m:**
$$P_l^{m+1}(x) = \frac{2mx}{\sqrt{1-x^2}}P_l^m(x) - \sqrt{(l+m)(l-m+1)}P_l^{m-1}(x)$$

### 4. Real Spherical Harmonics

Complex Y_l^m can be combined to form real functions:

$$\boxed{Y_{lm}^{(c)} = \begin{cases} \frac{i}{\sqrt{2}}(Y_l^m - (-1)^m Y_l^{-m}) & m < 0 \\ Y_l^0 & m = 0 \\ \frac{1}{\sqrt{2}}(Y_l^{-m} + (-1)^m Y_l^m) & m > 0 \end{cases}}$$

**Examples (atomic orbital notation):**
- Y₁₀^(c) = p_z ∝ z/r = cosθ
- Y₁₁^(c) = p_x ∝ x/r = sinθ cosφ
- Y₁₋₁^(c) = p_y ∝ y/r = sinθ sinφ

### 5. Multipole Expansion

The potential from a charge distribution can be expanded:

$$\Phi(\mathbf{r}) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l} \frac{4\pi}{2l+1}\frac{q_{lm}}{r^{l+1}}Y_l^m(\theta,\phi)$$

where the **multipole moments** are:
$$q_{lm} = \int \rho(\mathbf{r}') r'^l Y_l^{m*}(\theta',\phi')\, d^3r'$$

| l | Name | Example |
|---|------|---------|
| 0 | Monopole | Total charge |
| 1 | Dipole | Polar molecule |
| 2 | Quadrupole | Nuclear shape |

---

## Quantum Computing Connection

| Concept | QC Application |
|---------|----------------|
| Spherical harmonics | Molecular orbital simulation |
| Addition theorem | Angular momentum coupling |
| Multipole expansion | Quantum chemistry algorithms |
| Real harmonics | Efficient state preparation |

---

## Worked Examples

### Example 1: Addition Theorem for l = 1

**Problem:** Verify the addition theorem for l = 1.

**Solution:**
We need to show:
$$P_1(\cos\gamma) = \frac{4\pi}{3}\sum_{m=-1}^{1} Y_1^{m*}(\theta_1,\phi_1)Y_1^m(\theta_2,\phi_2)$$

Since P₁(x) = x, the LHS is cosγ.

Computing the sum:
$$\sum_{m=-1}^{1} Y_1^{m*}Y_1^m = |Y_1^1|^2 + |Y_1^0|^2 + |Y_1^{-1}|^2 + \text{cross terms}$$

After algebra (using explicit forms):
$$= \frac{3}{4\pi}[\cos\theta_1\cos\theta_2 + \sin\theta_1\sin\theta_2\cos(\phi_1-\phi_2)]$$
$$= \frac{3}{4\pi}\cos\gamma$$

Therefore: $\frac{4\pi}{3} \cdot \frac{3}{4\pi}\cos\gamma = \cos\gamma = P_1(\cos\gamma)$ ✓

### Example 2: Real p Orbitals

**Problem:** Express px, py, pz in terms of complex spherical harmonics.

**Solution:**
- p_z = Y₁⁰ = √(3/4π) cosθ = √(3/4π) z/r
- p_x = (Y₁⁻¹ - Y₁¹)/√2 = √(3/4π) sinθ cosφ = √(3/4π) x/r
- p_y = i(Y₁⁻¹ + Y₁¹)/√2 = √(3/4π) sinθ sinφ = √(3/4π) y/r

### Example 3: Dipole Moment

**Problem:** Calculate the dipole moment q₁ₘ for a point charge q at position (a, 0, 0).

**Solution:**
$$q_{1m} = \int \rho(\mathbf{r}') r' Y_1^{m*}(\theta',\phi')\, d^3r'$$

For point charge at (a,0,0): ρ(r') = qδ³(r' - a x̂)

At (a,0,0): r' = a, θ' = π/2, φ' = 0

$$q_{10} = qa \cdot Y_1^{0*}(\pi/2, 0) = qa \cdot \sqrt{\frac{3}{4\pi}}\cos(\pi/2) = 0$$

$$q_{11} = qa \cdot Y_1^{1*}(\pi/2, 0) = qa \cdot \left(-\sqrt{\frac{3}{8\pi}}\right) \cdot 1 \cdot 1 = -qa\sqrt{\frac{3}{8\pi}}$$

---

## Practice Problems

### Direct Application

1. Calculate P₂(x) using the Rodrigues formula.

2. Verify orthogonality: ∫₋₁¹ P₁(x)P₂(x)dx = 0.

3. Write the real d orbitals in terms of complex Y₂ᵐ.

### Intermediate

4. Use the addition theorem to prove ∫Y_l^m(r̂)dΩ = √(4π)δ_{l0}δ_{m0}.

5. Expand 1/|r-r'| for |r| > |r'| using the addition theorem.

6. Find the quadrupole moment tensor for two charges +q at (0,0,±a).

### Challenging

7. Prove the addition theorem using the rotation operator and Wigner D-matrices.

8. Show that ∇²Y_l^m = 0 (i.e., r^l Y_l^m is a solid harmonic).

---

## Computational Lab

```python
"""
Day 398 Computational Lab: Advanced Spherical Harmonics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, lpmv, legendre
from mpl_toolkits.mplot3d import Axes3D

def verify_addition_theorem(l, n_points=50):
    """
    Numerically verify the addition theorem for spherical harmonics.
    """
    print(f"\nVerifying addition theorem for l = {l}:")

    # Random directions
    np.random.seed(42)
    theta1, phi1 = np.pi * np.random.rand(), 2*np.pi * np.random.rand()
    theta2, phi2 = np.pi * np.random.rand(), 2*np.pi * np.random.rand()

    # Angle between directions
    cos_gamma = (np.cos(theta1)*np.cos(theta2) +
                 np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2))

    # LHS: P_l(cos_gamma)
    P_l = legendre(l)
    lhs = P_l(cos_gamma)

    # RHS: (4π/(2l+1)) Σ Y_l^m*(θ1,φ1) Y_l^m(θ2,φ2)
    rhs = 0.0
    for m in range(-l, l+1):
        Y1 = sph_harm(m, l, phi1, theta1)
        Y2 = sph_harm(m, l, phi2, theta2)
        rhs += np.conj(Y1) * Y2

    rhs *= 4*np.pi / (2*l + 1)

    print(f"  LHS: P_{l}(cos γ) = {lhs:.6f}")
    print(f"  RHS: (4π/{2*l+1}) Σ Y_l^m* Y_l^m = {np.real(rhs):.6f}")
    print(f"  Match: {np.isclose(lhs, np.real(rhs))}")

def plot_real_spherical_harmonics(l_max=2):
    """
    Plot real spherical harmonics (atomic orbitals).
    """
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    # Conversion to Cartesian
    def to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    fig = plt.figure(figsize=(15, 10))

    # p orbitals
    orbital_data = [
        (1, 0, 'p_z'),
        (1, 1, 'p_x'),
        (1, -1, 'p_y'),
        (2, 0, 'd_z²'),
        (2, 2, 'd_x²-y²'),
        (2, 1, 'd_xz'),
    ]

    for idx, (l, m, name) in enumerate(orbital_data):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')

        if m == 0:
            Y_real = np.real(sph_harm(0, l, PHI, THETA))
        elif m > 0:
            Y_real = np.real(sph_harm(m, l, PHI, THETA) +
                             (-1)**m * sph_harm(-m, l, PHI, THETA)) / np.sqrt(2)
        else:
            Y_real = np.imag(sph_harm(-m, l, PHI, THETA) -
                             (-1)**m * sph_harm(m, l, PHI, THETA)) / np.sqrt(2)

        R = np.abs(Y_real)
        X, Y, Z = to_cartesian(R, THETA, PHI)

        # Color by sign
        colors = Y_real
        norm_colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)

        ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu(norm_colors),
                        alpha=0.8, antialiased=True)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.suptitle('Real Spherical Harmonics (Atomic Orbitals)', fontsize=14)
    plt.tight_layout()
    plt.savefig('real_spherical_harmonics.png', dpi=150)
    plt.show()

def legendre_recursion():
    """
    Demonstrate Legendre polynomial recursion.
    """
    x = np.linspace(-1, 1, 200)

    fig, ax = plt.subplots(figsize=(10, 6))

    for l in range(5):
        P_l = legendre(l)
        ax.plot(x, P_l(x), label=f'$P_{l}(x)$', linewidth=2)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('$P_l(x)$', fontsize=12)
    ax.set_title('Legendre Polynomials', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('legendre_polynomials.png', dpi=150)
    plt.show()

def multipole_expansion_demo():
    """
    Visualize multipole expansion for a dipole.
    """
    # Two charges: +q at (0,0,a), -q at (0,0,-a)
    a = 1.0
    q = 1.0

    # Create grid
    x = np.linspace(-3, 3, 100)
    z = np.linspace(-3, 3, 100)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)

    # Exact potential
    r_plus = np.sqrt(X**2 + Y**2 + (Z-a)**2)
    r_minus = np.sqrt(X**2 + Y**2 + (Z+a)**2)

    # Avoid division by zero
    r_plus[r_plus < 0.1] = np.nan
    r_minus[r_minus < 0.1] = np.nan

    V_exact = q/r_plus - q/r_minus

    # Dipole approximation: V ≈ p·r̂/r² = 2qa·cosθ/r²
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r[r < 0.1] = np.nan
    cos_theta = Z / r
    p = 2 * q * a  # dipole moment
    V_dipole = p * cos_theta / r**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Exact
    im1 = axes[0].contourf(X, Z, V_exact, levels=50, cmap='RdBu_r')
    axes[0].set_title('Exact Potential', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')
    plt.colorbar(im1, ax=axes[0])

    # Dipole approximation
    im2 = axes[1].contourf(X, Z, V_dipole, levels=50, cmap='RdBu_r')
    axes[1].set_title('Dipole Approximation', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle('Multipole Expansion: Dipole', fontsize=14)
    plt.tight_layout()
    plt.savefig('multipole_expansion.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Day 398: Advanced Spherical Harmonics")
    print("=" * 50)

    # Verify addition theorem
    for l in [1, 2, 3]:
        verify_addition_theorem(l)

    # Plot real harmonics
    print("\nPlotting real spherical harmonics...")
    plot_real_spherical_harmonics()

    # Legendre polynomials
    print("\nPlotting Legendre polynomials...")
    legendre_recursion()

    # Multipole expansion
    print("\nDemonstrating multipole expansion...")
    multipole_expansion_demo()

    print("\nLab complete!")
```

---

## Summary

| Topic | Key Result |
|-------|------------|
| Addition theorem | P_l(cosγ) = (4π/(2l+1)) Σ Y_l^{m*}(r̂₁)Y_l^m(r̂₂) |
| Legendre recursion | (l+1)P_{l+1} = (2l+1)xP_l - lP_{l-1} |
| Real harmonics | Linear combinations for atomic orbitals |
| Multipole expansion | Φ = Σ (q_{lm}/r^{l+1}) Y_l^m |

---

## Daily Checklist

- [ ] I understand the addition theorem
- [ ] I can use recursion relations
- [ ] I know how to form real spherical harmonics
- [ ] I understand multipole expansions
- [ ] I completed the computational lab

---

## Preview: Day 399

Tomorrow we review the entire week, synthesize orbital angular momentum concepts, and prepare for Week 58 on spin angular momentum.

---

**Next:** [Day_399_Sunday.md](Day_399_Sunday.md) — Week Review & Lab
