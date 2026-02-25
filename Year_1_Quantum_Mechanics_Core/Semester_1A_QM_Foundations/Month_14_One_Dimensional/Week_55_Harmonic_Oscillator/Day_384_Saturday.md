# Day 384: QHO in Phase Space — The Wigner Function

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Phase Space Representations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 384, you will be able to:

1. Define the Wigner function and understand its properties
2. Calculate Wigner functions for Fock states and coherent states
3. Understand negativity as a signature of non-classicality
4. Visualize quantum states in phase space
5. Connect the classical limit to Wigner function behavior
6. Apply phase space methods to quantum optics and computing

---

## Core Content

### 1. Phase Space in Classical and Quantum Mechanics

#### Classical Phase Space

In classical mechanics, a system's state is specified by a point $(x, p)$ in phase space. The time evolution traces a trajectory determined by Hamilton's equations.

For the harmonic oscillator:
- Trajectories are ellipses
- Area enclosed = $2\pi E/\omega$
- Microstate = single point

#### The Quantum Problem

In quantum mechanics:
- $[\hat{x}, \hat{p}] = i\hbar$ forbids simultaneous sharp values
- Cannot have a point in phase space
- Uncertainty principle: $\Delta x \Delta p \geq \hbar/2$

**Question:** Can we still define a "probability distribution" in phase space?

**Answer:** Yes, but with modifications — the **Wigner function**.

---

### 2. The Wigner Function

The **Wigner function** (1932) is a quasi-probability distribution:

$$\boxed{W(x, p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty}\psi^*(x+y)\psi(x-y)e^{2ipy/\hbar}dy}$$

Alternatively, in terms of the density matrix $\hat{\rho}$:

$$W(x, p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty}\langle x-y|\hat{\rho}|x+y\rangle e^{2ipy/\hbar}dy$$

#### Key Properties

| Property | Mathematical Statement |
|----------|------------------------|
| Real-valued | $W(x, p) \in \mathbb{R}$ |
| Normalization | $\int\int W(x,p) dx dp = 1$ |
| Marginals | $\int W(x,p) dp = |\psi(x)|^2$ |
| | $\int W(x,p) dx = |\tilde{\psi}(p)|^2$ |
| Bounded | $|W(x,p)| \leq 2/h$ |
| Can be negative | Non-classical signature! |

**The crucial property:** $W(x,p)$ can be **negative** — this is what makes it "quasi" probability.

---

### 3. Wigner Function for Fock States

For the number state $|n\rangle$:

$$\boxed{W_n(x, p) = \frac{(-1)^n}{\pi\hbar}L_n\left(\frac{2H}{\hbar\omega}\right)e^{-2H/\hbar\omega}}$$

where:
- $H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2$ is the classical Hamiltonian
- $L_n$ is the Laguerre polynomial

In dimensionless variables $(q = x/x_0$, $\tilde{p} = p/p_0$, with $r^2 = q^2 + \tilde{p}^2$):

$$W_n(q, \tilde{p}) = \frac{(-1)^n}{\pi}L_n(2r^2)e^{-r^2}$$

#### Explicit Forms

| State | Wigner Function |
|-------|-----------------|
| $|0\rangle$ | $W_0 = \frac{1}{\pi}e^{-r^2}$ (Gaussian, positive) |
| $|1\rangle$ | $W_1 = \frac{1}{\pi}(2r^2 - 1)e^{-r^2}$ (negative at origin!) |
| $|2\rangle$ | $W_2 = \frac{1}{\pi}(2r^4 - 4r^2 + 1)e^{-r^2}$ |

**Key observation:**
- $|0\rangle$ has a positive Wigner function (Gaussian, "most classical")
- $|n\rangle$ for $n \geq 1$ has negative regions — signature of non-classicality

---

### 4. Wigner Function for Coherent States

For a coherent state $|\alpha\rangle$ with $\alpha = (q_0 + i\tilde{p}_0)/\sqrt{2}$:

$$\boxed{W_\alpha(q, \tilde{p}) = \frac{1}{\pi}\exp\left[-(q - q_0)^2 - (\tilde{p} - \tilde{p}_0)^2\right]}$$

**This is a 2D Gaussian centered at $(q_0, \tilde{p}_0)$!**

Properties:
- Always positive (no negativity)
- Circular shape (equal uncertainties in $x$ and $p$)
- Under time evolution, the center rotates: $(q_0, \tilde{p}_0) \to (q_0(t), \tilde{p}_0(t))$
- The width remains constant (minimum uncertainty preserved)

**Physical interpretation:** Coherent states are the "most classical" — their Wigner functions are positive Gaussians that follow classical trajectories.

---

### 5. Negativity as Non-classicality

**Hudson's Theorem (1974):** The only pure states with positive Wigner functions are Gaussian states (coherent states, squeezed states).

**Implications:**
- Negativity of $W(x,p)$ signals quantum behavior
- Fock states with $n \geq 1$ are non-classical
- Cat states have strong negativity (interference fringes)

#### Quantifying Non-classicality

The **Wigner negativity** is often quantified as:
$$\mathcal{N} = \int |W(x,p)| dx dp - 1$$

For classical states: $\mathcal{N} = 0$
For non-classical states: $\mathcal{N} > 0$

---

### 6. Wigner Function for Cat States

For the even cat state $|\text{cat}_+\rangle \propto |\alpha\rangle + |-\alpha\rangle$:

$$W_{\text{cat}}(q, \tilde{p}) = \mathcal{N}\left[W_\alpha + W_{-\alpha} + 2W_{\text{interference}}\right]$$

where the interference term creates **fringes** in phase space:
$$W_{\text{interference}} \propto e^{-(q^2 + \tilde{p}^2)}\cos(2\sqrt{2}q_0 \tilde{p})$$

The oscillatory fringes have:
- Period decreasing with $|\alpha|$
- Negative regions between the two Gaussian "blobs"
- Signature of quantum superposition

---

### 7. Classical Limit

What happens for large quantum numbers?

#### Large $n$ Limit

For $|n\rangle$ with $n \gg 1$:
- The Wigner function oscillates rapidly
- When coarse-grained, it approaches the classical distribution
- Classical probability: $P(x,p) \propto \delta(H - E_n)$

This is the **correspondence principle** in phase space.

#### Phase Space Area

The minimum uncertainty area is:
$$\Delta x \Delta p = \frac{\hbar}{2}$$

This defines a "quantum cell" of area $\hbar/2$ in phase space.

**Planck's constant sets the scale:** States with structure finer than $\hbar$ exhibit quantum features (negativity).

---

### 8. Quantum Computing Connection: GKP Codes and Phase Space

#### Phase Space Engineering

Modern quantum computing uses phase space representations for:

| Application | Phase Space Feature |
|-------------|---------------------|
| **GKP qubits** | Grid states with periodic Wigner function |
| **Cat qubits** | Two-blob Wigner function with interference |
| **Squeezed states** | Elliptical Wigner function |
| **Error detection** | Displacements visible in phase space |

#### GKP (Gottesman-Kitaev-Preskill) Codes

The ideal GKP code states have Wigner functions that are periodic grids:
$$W_{|0_L\rangle}(x, p) \propto \sum_{m,n}\delta(x - 2\sqrt{\pi}m)\delta(p - 2\sqrt{\pi}n)$$

This periodicity protects against small displacement errors!

#### Tomography

Measuring the Wigner function experimentally:
1. **Homodyne detection:** Measures marginals $\int W dp = |\psi(x)|^2$
2. **Heterodyne detection:** Samples from the Q-function $Q(\alpha) = \langle\alpha|\hat{\rho}|\alpha\rangle/\pi$
3. **Quantum state tomography:** Reconstruct full $W(x,p)$

---

## Worked Examples

### Example 1: Ground State Wigner Function

**Problem:** Verify that the Wigner function for $|0\rangle$ is $W_0(q, \tilde{p}) = \frac{1}{\pi}e^{-(q^2 + \tilde{p}^2)}$.

**Solution:**

The ground state wave function (in dimensionless units):
$$\psi_0(q) = \frac{1}{\pi^{1/4}}e^{-q^2/2}$$

The Wigner function:
$$W_0(q, \tilde{p}) = \frac{1}{\pi}\int_{-\infty}^{\infty}\psi_0^*(q+y)\psi_0(q-y)e^{2i\tilde{p}y}dy$$

$$= \frac{1}{\pi\sqrt{\pi}}\int_{-\infty}^{\infty}e^{-(q+y)^2/2}e^{-(q-y)^2/2}e^{2i\tilde{p}y}dy$$

Expand the exponents:
$$-\frac{(q+y)^2 + (q-y)^2}{2} = -\frac{2q^2 + 2y^2}{2} = -q^2 - y^2$$

$$W_0 = \frac{1}{\pi\sqrt{\pi}}e^{-q^2}\int_{-\infty}^{\infty}e^{-y^2 + 2i\tilde{p}y}dy$$

Complete the square: $-y^2 + 2i\tilde{p}y = -(y - i\tilde{p})^2 - \tilde{p}^2$

$$W_0 = \frac{1}{\pi\sqrt{\pi}}e^{-q^2}e^{-\tilde{p}^2}\int_{-\infty}^{\infty}e^{-(y-i\tilde{p})^2}dy$$

The integral equals $\sqrt{\pi}$ (Gaussian integral, contour shift):

$$\boxed{W_0(q, \tilde{p}) = \frac{1}{\pi}e^{-(q^2 + \tilde{p}^2)}}$$ ✓ $\blacksquare$

---

### Example 2: First Excited State Negativity

**Problem:** Find where $W_1(q, \tilde{p}) < 0$ for the first excited state $|1\rangle$.

**Solution:**

The Wigner function for $|1\rangle$:
$$W_1(q, \tilde{p}) = \frac{1}{\pi}(2r^2 - 1)e^{-r^2}$$

where $r^2 = q^2 + \tilde{p}^2$.

The sign of $W_1$ is determined by $(2r^2 - 1)$:
- $W_1 < 0$ when $2r^2 - 1 < 0$, i.e., $r^2 < 1/2$
- $W_1 > 0$ when $r^2 > 1/2$
- $W_1 = 0$ on the circle $r = 1/\sqrt{2}$

**Region of negativity:**
$$\boxed{q^2 + \tilde{p}^2 < \frac{1}{2}}$$

This is a disk of radius $1/\sqrt{2} \approx 0.707$ centered at the origin.

**Maximum negativity:** At the origin, $W_1(0,0) = -\frac{1}{\pi} \approx -0.318$. $\blacksquare$

---

### Example 3: Phase Space Area and Uncertainty

**Problem:** A coherent state $|\alpha\rangle$ has Wigner function width $\sigma_q = \sigma_{\tilde{p}} = 1/\sqrt{2}$ (in dimensionless units). Find the uncertainty product $\Delta x \cdot \Delta p$ in SI units.

**Solution:**

In dimensionless units:
$$\Delta q = \sigma_q = \frac{1}{\sqrt{2}}, \quad \Delta \tilde{p} = \sigma_{\tilde{p}} = \frac{1}{\sqrt{2}}$$

Converting to SI units:
- $\Delta x = \Delta q \cdot x_0 = \frac{1}{\sqrt{2}}\sqrt{\frac{\hbar}{m\omega}}$
- $\Delta p = \Delta \tilde{p} \cdot p_0 = \frac{1}{\sqrt{2}}\sqrt{m\omega\hbar}$

The product:
$$\Delta x \cdot \Delta p = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \cdot \sqrt{\frac{\hbar}{m\omega}} \cdot \sqrt{m\omega\hbar} = \frac{1}{2}\hbar$$

$$\boxed{\Delta x \cdot \Delta p = \frac{\hbar}{2}}$$

This confirms coherent states are minimum uncertainty states. $\blacksquare$

---

## Practice Problems

### Level 1: Direct Application

1. **Marginals:** Show that $\int_{-\infty}^{\infty}W_0(q, \tilde{p})d\tilde{p} = |\psi_0(q)|^2$ for the ground state.

2. **Normalization:** Verify $\int\int W_0(q, \tilde{p})dq\,d\tilde{p} = 1$.

3. **Coherent State Center:** A coherent state has $\alpha = 2e^{i\pi/4}$. Find the center of its Wigner function in phase space.

### Level 2: Intermediate

4. **Second Excited State:** Find the circles where $W_2(q, \tilde{p}) = 0$ for the state $|2\rangle$.
   *Hint:* $L_2(x) = \frac{1}{2}(x^2 - 4x + 2)$.

5. **Time Evolution:** If a coherent state Wigner function is centered at $(q_0, \tilde{p}_0)$ at $t=0$, find its center at time $t$.

6. **Superposition:** For the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, calculate $W(0, 0)$.

### Level 3: Challenging

7. **Cat State Fringes:** For the even cat state $|\alpha\rangle + |-\alpha\rangle$ with real $\alpha$, find the fringe spacing along the $p$-axis at $q = 0$.

8. **Wigner Negativity:** Compute the total "negative volume" $\int\int_{W<0}|W(q,\tilde{p})|dq\,d\tilde{p}$ for the $|1\rangle$ state.

9. **Quantum Optics:** The P-representation of a state is defined by $\hat{\rho} = \int P(\alpha)|\alpha\rangle\langle\alpha|d^2\alpha$. Show that for the coherent state $|\beta\rangle$, $P(\alpha) = \delta^{(2)}(\alpha - \beta)$, while for the Fock state $|1\rangle$, $P(\alpha)$ is highly singular (involves derivatives of delta functions).

---

## Computational Lab

### Objective
Compute and visualize Wigner functions for various quantum states using QuTiP.

```python
"""
Day 384 Computational Lab: Wigner Function Visualization
Quantum Harmonic Oscillator - Week 55
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, genlaguerre
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Part 1: Wigner Function Computation (Analytical)
# =============================================================================

print("=" * 70)
print("Part 1: Analytical Wigner Functions")
print("=" * 70)

def wigner_fock(n, q, p):
    """
    Wigner function for Fock state |n⟩.

    W_n(q, p) = ((-1)^n / π) L_n(2r²) exp(-r²)

    where r² = q² + p² (dimensionless coordinates)
    """
    r_squared = q**2 + p**2
    L_n = genlaguerre(n, 0)  # Laguerre polynomial
    return ((-1)**n / np.pi) * L_n(2 * r_squared) * np.exp(-r_squared)

def wigner_coherent(alpha, q, p):
    """
    Wigner function for coherent state |α⟩.

    W(q, p) = (1/π) exp(-(q - q0)² - (p - p0)²)

    where α = (q0 + i*p0)/√2
    """
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    return (1 / np.pi) * np.exp(-(q - q0)**2 - (p - p0)**2)

# Create grid
q_range = np.linspace(-4, 4, 200)
p_range = np.linspace(-4, 4, 200)
Q, P = np.meshgrid(q_range, p_range)

# =============================================================================
# Part 2: Visualize Fock State Wigner Functions
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Fock State Wigner Functions")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for n in range(6):
    ax = axes[n // 3, n % 3]
    W = wigner_fock(n, Q, P)

    # Use diverging colormap centered at 0
    vmax = np.max(np.abs(W))
    im = ax.contourf(Q, P, W, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(Q, P, W, levels=[0], colors='black', linewidths=1)  # Zero contour

    ax.set_xlabel('q (position)', fontsize=10)
    ax.set_ylabel('p (momentum)', fontsize=10)
    ax.set_title(f'|{n}⟩: W(0,0) = {W[100,100]:.3f}', fontsize=12)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Wigner Functions of Fock States |n⟩', fontsize=14)
plt.tight_layout()
plt.savefig('day_384_fock_wigner.png', dpi=150, bbox_inches='tight')
plt.show()

print("Fock state Wigner functions saved.")

# =============================================================================
# Part 3: Coherent State Wigner Functions
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Coherent State Wigner Functions")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

alpha_values = [0, 1.5, 1.5 + 1.5j]
titles = ['|0⟩ (vacuum)', '|α=1.5⟩', '|α=1.5+1.5i⟩']

for idx, (alpha, title) in enumerate(zip(alpha_values, titles)):
    ax = axes[idx]
    W = wigner_coherent(alpha, Q, P)

    im = ax.contourf(Q, P, W, levels=50, cmap='Blues')
    ax.set_xlabel('q (position)', fontsize=11)
    ax.set_ylabel('p (momentum)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Mark center
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    ax.plot(q0, p0, 'r+', markersize=15, markeredgewidth=2)

fig.suptitle('Wigner Functions of Coherent States', fontsize=14)
plt.tight_layout()
plt.savefig('day_384_coherent_wigner.png', dpi=150, bbox_inches='tight')
plt.show()

print("Coherent state Wigner functions saved.")

# =============================================================================
# Part 4: Cat State Wigner Functions
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Cat State Wigner Functions (Interference!)")
print("=" * 70)

def wigner_cat(alpha, q, p, sign=1):
    """
    Wigner function for cat state |α⟩ + sign * |-α⟩ (normalized).

    Includes interference fringes!
    """
    # Normalization
    overlap = np.exp(-2 * np.abs(alpha)**2)
    N_sq = 2 * (1 + sign * overlap)

    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)

    # Two Gaussian terms
    W_plus = np.exp(-(q - q0)**2 - (p - p0)**2)
    W_minus = np.exp(-(q + q0)**2 - (p + p0)**2)

    # Interference term
    W_interference = 2 * sign * np.exp(-q**2 - p**2) * np.cos(2 * (q * p0 - p * q0))

    return (W_plus + W_minus + W_interference) / (np.pi * N_sq)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

alpha_values = [1, 2, 3]

for idx, alpha in enumerate(alpha_values):
    # Even cat state
    ax = axes[0, idx]
    W = wigner_cat(alpha, Q, P, sign=1)
    vmax = np.max(np.abs(W))
    im = ax.contourf(Q, P, W, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(Q, P, W, levels=[0], colors='black', linewidths=0.5)
    ax.set_title(f'Even Cat: |{alpha}⟩ + |{-alpha}⟩', fontsize=12)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Odd cat state
    ax = axes[1, idx]
    W = wigner_cat(alpha, Q, P, sign=-1)
    vmax = np.max(np.abs(W))
    im = ax.contourf(Q, P, W, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(Q, P, W, levels=[0], colors='black', linewidths=0.5)
    ax.set_title(f'Odd Cat: |{alpha}⟩ - |{-alpha}⟩', fontsize=12)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Schrodinger Cat State Wigner Functions', fontsize=14)
plt.tight_layout()
plt.savefig('day_384_cat_wigner.png', dpi=150, bbox_inches='tight')
plt.show()

print("Cat state Wigner functions saved.")

# =============================================================================
# Part 5: 3D Wigner Function Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: 3D Wigner Function")
print("=" * 70)

fig = plt.figure(figsize=(15, 5))

# Ground state
ax1 = fig.add_subplot(131, projection='3d')
W0 = wigner_fock(0, Q, P)
ax1.plot_surface(Q, P, W0, cmap='viridis', alpha=0.8)
ax1.set_xlabel('q')
ax1.set_ylabel('p')
ax1.set_zlabel('W')
ax1.set_title('|0⟩: Positive Gaussian')

# First excited state
ax2 = fig.add_subplot(132, projection='3d')
W1 = wigner_fock(1, Q, P)
ax2.plot_surface(Q, P, W1, cmap='RdBu_r', alpha=0.8)
ax2.set_xlabel('q')
ax2.set_ylabel('p')
ax2.set_zlabel('W')
ax2.set_title('|1⟩: Negative at Origin')

# Cat state
ax3 = fig.add_subplot(133, projection='3d')
W_cat = wigner_cat(2, Q, P, sign=1)
ax3.plot_surface(Q, P, W_cat, cmap='RdBu_r', alpha=0.8)
ax3.set_xlabel('q')
ax3.set_ylabel('p')
ax3.set_zlabel('W')
ax3.set_title('Even Cat (α=2): Interference')

plt.tight_layout()
plt.savefig('day_384_wigner_3d.png', dpi=150, bbox_inches='tight')
plt.show()

print("3D Wigner functions saved.")

# =============================================================================
# Part 6: Time Evolution of Coherent State in Phase Space
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Time Evolution in Phase Space")
print("=" * 70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

alpha_0 = 2 + 1j
times = np.linspace(0, 2*np.pi, 8)

for idx, t in enumerate(times):
    ax = axes[idx // 4, idx % 4]

    # Time evolution: α(t) = α_0 * exp(-iωt)
    alpha_t = alpha_0 * np.exp(-1j * t)

    W = wigner_coherent(alpha_t, Q, P)
    im = ax.contourf(Q, P, W, levels=30, cmap='Blues')

    # Mark center
    q0 = np.sqrt(2) * np.real(alpha_t)
    p0 = np.sqrt(2) * np.imag(alpha_t)
    ax.plot(q0, p0, 'r+', markersize=12, markeredgewidth=2)

    # Draw trajectory
    t_traj = np.linspace(0, 2*np.pi, 100)
    alpha_traj = alpha_0 * np.exp(-1j * t_traj)
    q_traj = np.sqrt(2) * np.real(alpha_traj)
    p_traj = np.sqrt(2) * np.imag(alpha_traj)
    ax.plot(q_traj, p_traj, 'r--', alpha=0.5)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(f't = {t/np.pi:.2f}π', fontsize=11)

fig.suptitle('Coherent State Time Evolution in Phase Space', fontsize=14)
plt.tight_layout()
plt.savefig('day_384_phase_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Time evolution saved.")

# =============================================================================
# Part 7: Classical Correspondence - Large n
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Classical Limit (Large n)")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

n_values = [1, 10, 50]
q_fine = np.linspace(-12, 12, 400)
p_fine = np.linspace(-12, 12, 400)
Q_fine, P_fine = np.meshgrid(q_fine, p_fine)

for idx, n in enumerate(n_values):
    ax = axes[idx]
    W = wigner_fock(n, Q_fine, P_fine)

    # Classical energy contour
    r_classical = np.sqrt(2*n + 1)  # Classical radius

    vmax = np.max(np.abs(W))
    im = ax.contourf(Q_fine, P_fine, W, levels=50, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax)

    # Draw classical orbit
    theta = np.linspace(0, 2*np.pi, 100)
    q_class = r_classical * np.cos(theta)
    p_class = r_classical * np.sin(theta)
    ax.plot(q_class, p_class, 'k--', linewidth=2, label='Classical orbit')

    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title(f'|{n}⟩: Classical radius = {r_classical:.2f}', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()

fig.suptitle('Correspondence Principle: Wigner Function Approaches Classical Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('day_384_classical_limit.png', dpi=150, bbox_inches='tight')
plt.show()

print("Classical limit visualization saved.")

# =============================================================================
# Part 8: Quantifying Negativity
# =============================================================================

print("\n" + "=" * 70)
print("Part 8: Wigner Negativity (Non-classicality)")
print("=" * 70)

def wigner_negativity(W, dq, dp):
    """
    Compute Wigner negativity: ∫∫|W| dqdp - 1
    """
    total = np.sum(np.abs(W)) * dq * dp
    return total - 1

dq = q_range[1] - q_range[0]
dp = p_range[1] - p_range[0]

print("\nWigner negativity for various states:")
print("-" * 50)

# Fock states
for n in range(6):
    W = wigner_fock(n, Q, P)
    neg = wigner_negativity(W, dq, dp)
    print(f"|{n}⟩: Negativity = {neg:.4f}")

# Coherent states
for alpha in [0, 1, 2]:
    W = wigner_coherent(alpha, Q, P)
    neg = wigner_negativity(W, dq, dp)
    print(f"|α={alpha}⟩: Negativity = {neg:.6f} (should be ~0)")

# Cat states
for alpha in [1, 2, 3]:
    W = wigner_cat(alpha, Q, P, sign=1)
    neg = wigner_negativity(W, dq, dp)
    print(f"Cat (α={alpha}): Negativity = {neg:.4f}")

# =============================================================================
# Part 9: Cross-sections through Wigner Function
# =============================================================================

print("\n" + "=" * 70)
print("Part 9: Cross-sections of Wigner Functions")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# q = 0 cross-section
ax = axes[0]
p_line = np.linspace(-4, 4, 200)

for n in range(4):
    W_slice = wigner_fock(n, 0, p_line)
    ax.plot(p_line, W_slice, label=f'|{n}⟩', linewidth=2)

ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('p (at q=0)', fontsize=12)
ax.set_ylabel('W(0, p)', fontsize=12)
ax.set_title('Cross-section at q = 0', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# p = 0 cross-section for cat state
ax = axes[1]
q_line = np.linspace(-6, 6, 200)

for alpha in [1, 2, 3]:
    W_slice = wigner_cat(alpha, q_line, 0, sign=1)
    ax.plot(q_line, W_slice, label=f'Cat α={alpha}', linewidth=2)

ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('q (at p=0)', fontsize=12)
ax.set_ylabel('W(q, 0)', fontsize=12)
ax.set_title('Even Cat State Cross-section at p = 0', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_384_cross_sections.png', dpi=150, bbox_inches='tight')
plt.show()

print("Cross-section plots saved.")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Wigner function | $W(x,p) = \frac{1}{\pi\hbar}\int\psi^*(x+y)\psi(x-y)e^{2ipy/\hbar}dy$ |
| Fock state | $W_n = \frac{(-1)^n}{\pi}L_n(2r^2)e^{-r^2}$ |
| Coherent state | $W_\alpha = \frac{1}{\pi}e^{-(q-q_0)^2-(p-p_0)^2}$ |
| Marginal (position) | $\int W(x,p)dp = |\psi(x)|^2$ |
| Marginal (momentum) | $\int W(x,p)dx = |\tilde{\psi}(p)|^2$ |
| Non-classicality | Negativity: $\mathcal{N} = \int|W|dxdp - 1$ |

### Main Takeaways

1. **Quasi-probability:** Wigner function maps quantum states to phase space distributions
2. **Can be negative:** Unlike classical probability — signature of quantum behavior
3. **Hudson's theorem:** Only Gaussians (coherent, squeezed) have positive Wigner functions
4. **Fock states:** $|n\rangle$ with $n \geq 1$ have negative regions
5. **Cat states:** Show interference fringes in phase space
6. **Classical limit:** Large $n$ Wigner functions approach classical distributions

---

## Daily Checklist

- [ ] Read Schleich "Quantum Optics in Phase Space" Chapter 3 (or equivalent)
- [ ] Derive the Wigner function for the ground state
- [ ] Find the negative region for $|1\rangle$
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Run and understand the computational lab
- [ ] Visualize Wigner functions for superposition states

---

## Preview: Day 385

Tomorrow is the **Week Review & Lab**. We'll synthesize all QHO concepts — algebraic and analytic methods, coherent states, and phase space — with a comprehensive computational project using QuTiP.

---

*"The Wigner function is perhaps the closest we can come to a phase space description of quantum mechanics."*
— W.P. Schleich

---

**Next:** [Day_385_Sunday.md](Day_385_Sunday.md) — Week Review & Lab
