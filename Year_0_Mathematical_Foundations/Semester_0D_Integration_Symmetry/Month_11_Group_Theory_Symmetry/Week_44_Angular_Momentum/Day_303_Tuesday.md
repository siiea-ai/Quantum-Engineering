# Day 303: Spherical Harmonics and Orbital Angular Momentum

## Overview

**Month 11, Week 44, Day 2 — Tuesday**

Today we study spherical harmonics $Y_\ell^m(\theta, \phi)$ — the angular wavefunctions for orbital angular momentum. These functions appear throughout physics: atomic orbitals, multipole expansions, gravitational waves, and the cosmic microwave background. They are the eigenfunctions of $\hat{L}^2$ and $\hat{L}_z$ in coordinate space.

## Learning Objectives

1. Derive spherical harmonics from the angular momentum eigenvalue problem
2. Understand the role of associated Legendre polynomials
3. Master orthogonality and completeness relations
4. Visualize orbital angular momentum states
5. Connect to atomic physics and chemistry

---

## 1. Orbital Angular Momentum in Position Space

### The Operators

In Cartesian coordinates:
$$\hat{L}_x = -i\hbar\left(y\frac{\partial}{\partial z} - z\frac{\partial}{\partial y}\right)$$

In spherical coordinates $(r, \theta, \phi)$:

$$\boxed{\hat{L}_z = -i\hbar\frac{\partial}{\partial \phi}}$$

$$\hat{L}_\pm = \hbar e^{\pm i\phi}\left(\pm\frac{\partial}{\partial \theta} + i\cot\theta\frac{\partial}{\partial \phi}\right)$$

$$\boxed{\hat{L}^2 = -\hbar^2\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]}$$

### Key Observation

$\hat{L}_z$ and $\hat{L}^2$ involve only $\theta$ and $\phi$, not $r$. The angular momentum eigenfunctions depend only on angles.

---

## 2. The Eigenvalue Problem

### Finding the Eigenfunctions

We seek $Y(\theta, \phi)$ satisfying:
$$\hat{L}^2 Y = \hbar^2 \ell(\ell+1) Y$$
$$\hat{L}_z Y = \hbar m Y$$

### Step 1: Solve the $\phi$ Equation

From $\hat{L}_z Y = \hbar m Y$:
$$-i\hbar\frac{\partial Y}{\partial \phi} = \hbar m Y$$
$$Y(\theta, \phi) = \Theta(\theta) e^{im\phi}$$

For single-valuedness: $Y(\theta, \phi + 2\pi) = Y(\theta, \phi)$
$$e^{im \cdot 2\pi} = 1 \implies m \in \mathbb{Z}$$

**Orbital angular momentum has integer $m$ only!**

### Step 2: The $\theta$ Equation

Substituting into $\hat{L}^2 Y = \hbar^2 \ell(\ell+1) Y$:

$$\frac{1}{\sin\theta}\frac{d}{d\theta}\left(\sin\theta\frac{d\Theta}{d\theta}\right) - \frac{m^2}{\sin^2\theta}\Theta = -\ell(\ell+1)\Theta$$

With $x = \cos\theta$, this becomes the **associated Legendre equation**:

$$\frac{d}{dx}\left[(1-x^2)\frac{dP}{dx}\right] + \left[\ell(\ell+1) - \frac{m^2}{1-x^2}\right]P = 0$$

---

## 3. Associated Legendre Polynomials

### Definition

For $m \geq 0$:
$$\boxed{P_\ell^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_\ell(x)}$$

where $P_\ell(x)$ are the ordinary Legendre polynomials:
$$P_\ell(x) = \frac{1}{2^\ell \ell!}\frac{d^\ell}{dx^\ell}(x^2 - 1)^\ell$$

For $m < 0$:
$$P_\ell^{-m}(x) = (-1)^m \frac{(\ell - m)!}{(\ell + m)!} P_\ell^m(x)$$

### Examples

$$P_0^0 = 1$$
$$P_1^0 = \cos\theta, \quad P_1^1 = -\sin\theta$$
$$P_2^0 = \frac{1}{2}(3\cos^2\theta - 1), \quad P_2^1 = -3\sin\theta\cos\theta, \quad P_2^2 = 3\sin^2\theta$$

### Boundary Conditions

For solutions to be regular at $\theta = 0, \pi$ (the poles):
$$\ell = 0, 1, 2, 3, \ldots \quad \text{and} \quad |m| \leq \ell$$

---

## 4. Spherical Harmonics

### Definition

$$\boxed{Y_\ell^m(\theta, \phi) = (-1)^m \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}} P_\ell^m(\cos\theta) e^{im\phi}}$$

The $(-1)^m$ is the **Condon-Shortley phase convention**.

### Explicit Forms

$$Y_0^0 = \frac{1}{\sqrt{4\pi}}$$

$$Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta, \quad Y_1^{\pm 1} = \mp\sqrt{\frac{3}{8\pi}}\sin\theta \, e^{\pm i\phi}$$

$$Y_2^0 = \sqrt{\frac{5}{16\pi}}(3\cos^2\theta - 1)$$

$$Y_2^{\pm 1} = \mp\sqrt{\frac{15}{8\pi}}\sin\theta\cos\theta \, e^{\pm i\phi}$$

$$Y_2^{\pm 2} = \sqrt{\frac{15}{32\pi}}\sin^2\theta \, e^{\pm 2i\phi}$$

### Symmetry Properties

$$Y_\ell^{-m} = (-1)^m \left(Y_\ell^m\right)^*$$

Under parity ($\theta \to \pi - \theta$, $\phi \to \phi + \pi$):
$$Y_\ell^m(-\hat{r}) = (-1)^\ell Y_\ell^m(\hat{r})$$

---

## 5. Orthogonality and Completeness

### Orthonormality

$$\boxed{\int_0^{2\pi} d\phi \int_0^\pi \sin\theta \, d\theta \, Y_{\ell'}^{m'*}(\theta, \phi) Y_\ell^m(\theta, \phi) = \delta_{\ell\ell'}\delta_{mm'}}$$

Or in shorthand:
$$\langle Y_{\ell'}^{m'} | Y_\ell^m \rangle = \delta_{\ell\ell'}\delta_{mm'}$$

### Completeness

Any square-integrable function on the sphere can be expanded:
$$\boxed{f(\theta, \phi) = \sum_{\ell=0}^{\infty} \sum_{m=-\ell}^{\ell} c_{\ell m} Y_\ell^m(\theta, \phi)}$$

where:
$$c_{\ell m} = \int d\Omega \, Y_\ell^{m*}(\theta, \phi) f(\theta, \phi)$$

### Closure Relation

$$\sum_{\ell=0}^{\infty} \sum_{m=-\ell}^{\ell} Y_\ell^{m*}(\theta', \phi') Y_\ell^m(\theta, \phi) = \delta(\cos\theta - \cos\theta')\delta(\phi - \phi')$$

---

## 6. Real Spherical Harmonics

### Definition

For visualization and chemistry, real combinations are often used:

$$Y_{\ell m}^c = \frac{1}{\sqrt{2}}(Y_\ell^{-m} + (-1)^m Y_\ell^m) \quad (m > 0)$$
$$Y_{\ell m}^s = \frac{i}{\sqrt{2}}(Y_\ell^{-m} - (-1)^m Y_\ell^m) \quad (m > 0)$$
$$Y_{\ell 0}^c = Y_\ell^0$$

### Chemistry Notation

| Spherical | Chemistry | Angular Pattern |
|-----------|-----------|-----------------|
| $Y_0^0$ | $s$ | Sphere |
| $Y_1^0$ | $p_z$ | Dumbbell along z |
| $Y_1^{\pm 1}$ | $p_x, p_y$ | Dumbbells along x, y |
| $Y_2^0$ | $d_{z^2}$ | "Donut" + lobes |
| $Y_2^{\pm 1}$ | $d_{xz}, d_{yz}$ | Four-leaf clover |
| $Y_2^{\pm 2}$ | $d_{xy}, d_{x^2-y^2}$ | Four lobes in plane |

---

## 7. Addition Theorem

### The Formula

For angle $\gamma$ between directions $(\theta_1, \phi_1)$ and $(\theta_2, \phi_2)$:

$$\boxed{P_\ell(\cos\gamma) = \frac{4\pi}{2\ell+1} \sum_{m=-\ell}^{\ell} Y_\ell^{m*}(\theta_1, \phi_1) Y_\ell^m(\theta_2, \phi_2)}$$

### Physical Applications

- **Multipole expansion:** Potential from charge distribution
- **Scattering:** Partial wave expansion
- **CMB:** Angular power spectrum

---

## 8. Quantum Mechanics Connection

### Atomic Orbitals

The hydrogen atom wavefunction:
$$\psi_{n\ell m}(r, \theta, \phi) = R_{n\ell}(r) Y_\ell^m(\theta, \phi)$$

The spherical harmonic determines the **angular shape** of the orbital.

### Selection Rules

Electric dipole transitions require:
$$\Delta \ell = \pm 1, \quad \Delta m = 0, \pm 1$$

This follows from the integral:
$$\langle \ell', m' | \hat{\mathbf{r}} | \ell, m \rangle$$

### Probability Density

$$|\psi|^2 \propto |Y_\ell^m(\theta, \phi)|^2$$

For $m \neq 0$, the probability depends on $\phi$ — the electron is "circulating" around the z-axis.

---

## 9. Computational Lab

```python
"""
Day 303: Spherical Harmonics
"""

import numpy as np
from scipy.special import sph_harm, lpmv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Note: scipy.special.sph_harm uses convention sph_harm(m, l, phi, theta)

def spherical_harmonic(l, m, theta, phi):
    """
    Compute Y_l^m(theta, phi).
    scipy uses (m, l, phi, theta) convention.
    """
    return sph_harm(m, l, phi, theta)


def plot_spherical_harmonic(l, m, title=None):
    """
    Visualize |Y_l^m|^2 as surface coloring.
    """
    # Create grid
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    # Compute spherical harmonic
    Y = spherical_harmonic(l, m, THETA, PHI)

    # Use |Y|^2 for radius/color
    R = np.abs(Y)**2

    # Convert to Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize colors
    norm_R = R / np.max(R) if np.max(R) > 0 else R

    ax.plot_surface(X, Y_coord, Z, facecolors=plt.cm.coolwarm(norm_R),
                    alpha=0.8, linewidth=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if title is None:
        title = f'$|Y_{l}^{m}|^2$'
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.max([np.max(np.abs(X)), np.max(np.abs(Y_coord)), np.max(np.abs(Z))])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    return fig, ax


def plot_all_harmonics_up_to_l(l_max):
    """
    Plot all spherical harmonics up to l_max.
    """
    fig = plt.figure(figsize=(15, 4 * (l_max + 1)))

    for l in range(l_max + 1):
        n_cols = 2 * l + 1
        for m in range(-l, l + 1):
            # Subplot position
            idx = sum(2*k + 1 for k in range(l)) + (m + l) + 1

            ax = fig.add_subplot(l_max + 1, 2 * l_max + 1, idx, projection='3d')

            # Compute on grid
            theta = np.linspace(0, np.pi, 50)
            phi = np.linspace(0, 2*np.pi, 50)
            THETA, PHI = np.meshgrid(theta, phi)

            Y = spherical_harmonic(l, m, THETA, PHI)
            R = np.abs(Y)

            # Real part visualization
            R_real = np.abs(np.real(Y * np.exp(-1j * m * PHI / 2)))

            X = R * np.sin(THETA) * np.cos(PHI)
            Y_c = R * np.sin(THETA) * np.sin(PHI)
            Z = R * np.cos(THETA)

            # Color by sign of real part
            colors = np.where(np.real(Y) >= 0, 'blue', 'red')

            ax.plot_surface(X, Y_c, Z, alpha=0.7)
            ax.set_title(f'$Y_{l}^{{{m}}}$', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    plt.tight_layout()
    return fig


def demonstrate_orthogonality():
    """
    Verify orthogonality of spherical harmonics numerically.
    """
    print("=" * 50)
    print("SPHERICAL HARMONIC ORTHOGONALITY")
    print("=" * 50)

    l_max = 3
    n_theta = 100
    n_phi = 200

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)

    dtheta = np.pi / n_theta
    dphi = 2 * np.pi / n_phi

    # Test orthogonality
    print("\nInner products <Y_l'^m'|Y_l^m>:")
    print("-" * 50)

    test_cases = [
        ((0, 0), (0, 0)),
        ((1, 0), (1, 0)),
        ((1, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((2, 1), (2, 1)),
        ((1, 1), (2, 1)),
    ]

    for (l1, m1), (l2, m2) in test_cases:
        Y1 = spherical_harmonic(l1, m1, THETA, PHI)
        Y2 = spherical_harmonic(l2, m2, THETA, PHI)

        # Numerical integration
        integrand = np.conj(Y1) * Y2 * np.sin(THETA)
        integral = np.sum(integrand) * dtheta * dphi

        expected = 1 if (l1 == l2 and m1 == m2) else 0
        print(f"<Y_{l1}^{m1}|Y_{l2}^{m2}> = {integral.real:.6f} + {integral.imag:.6f}i  (expected: {expected})")


def demonstrate_completeness():
    """
    Expand a function in spherical harmonics.
    """
    print("\n" + "=" * 50)
    print("SPHERICAL HARMONIC EXPANSION")
    print("=" * 50)

    # Define a function on the sphere: f(theta, phi) = cos(theta)
    # This is proportional to Y_1^0

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)

    f = np.cos(THETA)  # This is sqrt(4pi/3) * Y_1^0

    dtheta = np.pi / 100
    dphi = 2 * np.pi / 200

    print("\nExpanding f(θ,φ) = cos(θ) in spherical harmonics:")
    print("-" * 50)

    coefficients = {}
    for l in range(4):
        for m in range(-l, l + 1):
            Y = spherical_harmonic(l, m, THETA, PHI)
            integrand = np.conj(Y) * f * np.sin(THETA)
            c_lm = np.sum(integrand) * dtheta * dphi

            if np.abs(c_lm) > 1e-6:
                coefficients[(l, m)] = c_lm
                print(f"c_{l},{m} = {c_lm.real:.6f}")

    # The only nonzero coefficient should be c_1,0 = sqrt(4pi/3)
    expected = np.sqrt(4 * np.pi / 3)
    print(f"\nExpected c_1,0 = sqrt(4π/3) = {expected:.6f}")


def ladder_operators_on_harmonics():
    """
    Demonstrate ladder operator action on spherical harmonics.
    """
    print("\n" + "=" * 50)
    print("LADDER OPERATORS ON SPHERICAL HARMONICS")
    print("=" * 50)

    # L+ Y_l^m = hbar * sqrt(l(l+1) - m(m+1)) * Y_l^{m+1}
    # L- Y_l^m = hbar * sqrt(l(l+1) - m(m-1)) * Y_l^{m-1}

    l = 2
    print(f"\nFor l = {l}:")
    print("-" * 50)

    for m in range(-l, l + 1):
        # L+ coefficient
        if m < l:
            coeff_plus = np.sqrt(l*(l+1) - m*(m+1))
            print(f"L+ |{l},{m:+d}> = ℏ × {coeff_plus:.4f} |{l},{m+1:+d}>")
        else:
            print(f"L+ |{l},{m:+d}> = 0")

        # L- coefficient
        if m > -l:
            coeff_minus = np.sqrt(l*(l+1) - m*(m-1))
            print(f"L- |{l},{m:+d}> = ℏ × {coeff_minus:.4f} |{l},{m-1:+d}>")
        else:
            print(f"L- |{l},{m:+d}> = 0")
        print()


def visualize_orbital_shapes():
    """
    Create publication-quality visualization of atomic orbital shapes.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    orbitals = [
        (0, 0, 's'),
        (1, 0, '$p_z$'),
        (1, 1, '$p_x$'),
        (1, -1, '$p_y$'),
        (2, 0, '$d_{z^2}$'),
        (2, 1, '$d_{xz}$'),
        (2, -1, '$d_{yz}$'),
        (2, 2, '$d_{x^2-y^2}$'),
    ]

    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 80)
    THETA, PHI = np.meshgrid(theta, phi)

    for ax, (l, m, name) in zip(axes, orbitals):
        Y = spherical_harmonic(l, m, THETA, PHI)

        # For real orbitals, take appropriate combination
        if m > 0:
            Y = (spherical_harmonic(l, -m, THETA, PHI) +
                 (-1)**m * spherical_harmonic(l, m, THETA, PHI)) / np.sqrt(2)
        elif m < 0:
            Y = 1j * (spherical_harmonic(l, -np.abs(m), THETA, PHI) -
                     (-1)**np.abs(m) * spherical_harmonic(l, np.abs(m), THETA, PHI)) / np.sqrt(2)

        R = np.abs(Y)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y_c = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        # Color by sign
        Y_real = np.real(Y)
        colors = np.where(Y_real >= 0, 1, 0)

        ax.plot_surface(X, Y_c, Z, facecolors=plt.cm.RdBu(colors),
                       alpha=0.8, linewidth=0)

        ax.set_title(name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Equal aspect ratio
        max_range = max(np.max(np.abs(X)), np.max(np.abs(Y_c)), np.max(np.abs(Z)))
        if max_range > 0:
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    plt.savefig('atomic_orbitals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: atomic_orbitals.png")


# Main execution
if __name__ == "__main__":
    demonstrate_orthogonality()
    demonstrate_completeness()
    ladder_operators_on_harmonics()
    visualize_orbital_shapes()

    # Create individual harmonic plot
    fig, ax = plot_spherical_harmonic(2, 1)
    plt.savefig('Y_2_1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: Y_2_1.png")
```

---

## 10. Practice Problems

### Problem 1: Explicit Computation

Verify that $Y_1^1 = -\sqrt{\frac{3}{8\pi}}\sin\theta \, e^{i\phi}$ satisfies:
- (a) $\hat{L}_z Y_1^1 = \hbar Y_1^1$
- (b) $\hat{L}^2 Y_1^1 = 2\hbar^2 Y_1^1$

### Problem 2: Normalization

Show that $\int |Y_2^0|^2 d\Omega = 1$ by explicit integration.

### Problem 3: Orthogonality

Prove that $\langle Y_1^0 | Y_2^0 \rangle = 0$ using the explicit forms.

### Problem 4: Addition Theorem

Use the addition theorem to show that:
$$P_1(\cos\gamma) = \cos\theta_1 \cos\theta_2 + \sin\theta_1 \sin\theta_2 \cos(\phi_1 - \phi_2)$$

### Problem 5: Selection Rules

Show that $\langle Y_2^1 | \cos\theta | Y_1^0 \rangle \neq 0$ but $\langle Y_2^0 | \cos\theta | Y_2^0 \rangle = 0$.

---

## Summary

### Spherical Harmonics

$$\boxed{Y_\ell^m(\theta, \phi) = (-1)^m \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}} P_\ell^m(\cos\theta) e^{im\phi}}$$

### Quantum Numbers for Orbital Angular Momentum

| Symbol | Name | Values | Constraint |
|--------|------|--------|------------|
| $\ell$ | Orbital quantum number | $0, 1, 2, \ldots$ | Integer only |
| $m$ | Magnetic quantum number | $-\ell, \ldots, \ell$ | Integer only |

### Key Properties

- **Orthonormal:** $\langle Y_{\ell'}^{m'} | Y_\ell^m \rangle = \delta_{\ell\ell'}\delta_{mm'}$
- **Complete:** Any function on sphere expandable
- **Parity:** $(-1)^\ell$
- **Symmetry:** $Y_\ell^{-m} = (-1)^m (Y_\ell^m)^*$

---

## Preview: Day 304

Tomorrow we explore **spin angular momentum** — intrinsic angular momentum that has no classical analog. Unlike orbital angular momentum, spin can be half-integer, leading to the remarkable two-state system of spin-1/2.
