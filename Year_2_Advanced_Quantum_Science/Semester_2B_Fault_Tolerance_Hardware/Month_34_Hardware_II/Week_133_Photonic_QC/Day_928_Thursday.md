# Day 928: Continuous Variable Quantum Computing

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Quadrature operators, Gaussian states, phase space representation |
| Afternoon | 2.5 hours | Problem solving: CV gates and state transformations |
| Evening | 1.5 hours | Computational lab: Wigner function visualization |

## Learning Objectives

By the end of today, you will be able to:

1. Define quadrature operators and their commutation relations
2. Describe Gaussian states (coherent, squeezed, thermal) in phase space
3. Derive the action of CV gates: displacement, squeezing, rotation
4. Calculate Wigner functions for various quantum states
5. Explain the requirements for universal CV quantum computing
6. Implement phase space visualization using Python

## Core Content

### 1. From Discrete to Continuous Variables

**Discrete Variable (DV) Approach:**
- Qubits: 2-dimensional Hilbert space
- States: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$
- Gates: Finite-dimensional unitary matrices

**Continuous Variable (CV) Approach:**
- Qumodes: Infinite-dimensional Hilbert space
- States: Functions of continuous variables
- Gates: Infinite-dimensional unitaries (Gaussian and non-Gaussian)

**The Harmonic Oscillator:**
The quantum harmonic oscillator provides the mathematical framework:
$$\hat{H} = \hbar\omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \hbar\omega\left(\hat{n} + \frac{1}{2}\right)$$

Energy eigenstates: Fock states $|n\rangle$

### 2. Quadrature Operators

**Definition:**
The position and momentum quadratures generalize to any bosonic mode:
$$\hat{q} = \frac{1}{\sqrt{2}}(\hat{a} + \hat{a}^\dagger)$$
$$\hat{p} = \frac{i}{\sqrt{2}}(\hat{a}^\dagger - \hat{a})$$

**Commutation Relation:**
$$[\hat{q}, \hat{p}] = i$$

This is the canonical commutation relation (with $\hbar = 1$).

**Inverse Relations:**
$$\hat{a} = \frac{1}{\sqrt{2}}(\hat{q} + i\hat{p})$$
$$\hat{a}^\dagger = \frac{1}{\sqrt{2}}(\hat{q} - i\hat{p})$$

**Uncertainty Relation:**
$$\Delta q \cdot \Delta p \geq \frac{1}{2}$$

States saturating this bound are called **minimum uncertainty states**.

### 3. Phase Space Representation

**Classical Phase Space:**
A point $(q, p)$ represents the state of a classical harmonic oscillator.

**Quantum Phase Space:**
The Heisenberg uncertainty principle prevents simultaneous sharp values. We use **quasi-probability distributions**.

**The Wigner Function:**
$$W(q, p) = \frac{1}{\pi}\int_{-\infty}^{\infty} \langle q - y|\hat{\rho}|q + y\rangle e^{2ipy} dy$$

Properties:
1. Real-valued (can be negative!)
2. Normalized: $\int W(q,p) dq\, dp = 1$
3. Marginals give true probabilities: $\int W(q,p) dp = P(q)$
4. Negativity indicates non-classicality

**For a pure state $|\psi\rangle$:**
$$W(q, p) = \frac{1}{\pi}\int_{-\infty}^{\infty} \psi^*(q + y)\psi(q - y) e^{2ipy} dy$$

### 4. Gaussian States

Gaussian states have Wigner functions that are Gaussian distributions.

**Vacuum State:**
$$|0\rangle: \quad W_0(q, p) = \frac{1}{\pi}e^{-(q^2 + p^2)}$$

Circular distribution centered at origin.

**Coherent States:**
$$|\alpha\rangle = \hat{D}(\alpha)|0\rangle, \quad \alpha = q_0 + ip_0$$

$$W_\alpha(q, p) = \frac{1}{\pi}e^{-[(q-q_0)^2 + (p-p_0)^2]}$$

Displaced vacuum - same shape, different center.

**Squeezed States:**
$$|sq, r\rangle = \hat{S}(r)|0\rangle$$

$$W_{sq}(q, p) = \frac{1}{\pi}e^{-(e^{2r}q^2 + e^{-2r}p^2)}$$

Elliptical distribution: squeezed in one quadrature, anti-squeezed in the other.

**Thermal States:**
$$\hat{\rho}_{th} = \sum_{n=0}^\infty \frac{\bar{n}^n}{(\bar{n}+1)^{n+1}}|n\rangle\langle n|$$

$$W_{th}(q, p) = \frac{1}{\pi(2\bar{n}+1)}e^{-(q^2 + p^2)/(2\bar{n}+1)}$$

Broadened Gaussian (mixed state).

**Covariance Matrix:**
Gaussian states are fully characterized by:
1. Mean values: $\bar{q} = \langle\hat{q}\rangle$, $\bar{p} = \langle\hat{p}\rangle$
2. Covariance matrix:
$$\Sigma = \begin{pmatrix} \langle\Delta q^2\rangle & \langle\Delta q\Delta p + \Delta p\Delta q\rangle/2 \\ \langle\Delta q\Delta p + \Delta p\Delta q\rangle/2 & \langle\Delta p^2\rangle \end{pmatrix}$$

For vacuum: $\Sigma = \frac{1}{2}I$

### 5. Gaussian Operations

**Displacement Operator:**
$$\hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}} = e^{i\sqrt{2}(p_0\hat{q} - q_0\hat{p})}$$

Action:
$$\hat{D}^\dagger(\alpha)\hat{q}\hat{D}(\alpha) = \hat{q} + q_0$$
$$\hat{D}^\dagger(\alpha)\hat{p}\hat{D}(\alpha) = \hat{p} + p_0$$

In phase space: translation by $(q_0, p_0)$.

**Squeezing Operator:**
$$\hat{S}(r) = e^{\frac{r}{2}(\hat{a}^2 - \hat{a}^{\dagger 2})}$$

Action:
$$\hat{S}^\dagger(r)\hat{q}\hat{S}(r) = e^{-r}\hat{q}$$
$$\hat{S}^\dagger(r)\hat{p}\hat{S}(r) = e^{r}\hat{p}$$

In phase space: squeeze along $q$, stretch along $p$.

**Rotation (Phase Shift):**
$$\hat{R}(\theta) = e^{i\theta\hat{n}} = e^{i\theta\hat{a}^\dagger\hat{a}}$$

Action:
$$\hat{R}^\dagger(\theta)\hat{q}\hat{R}(\theta) = \cos\theta\,\hat{q} + \sin\theta\,\hat{p}$$
$$\hat{R}^\dagger(\theta)\hat{p}\hat{R}(\theta) = -\sin\theta\,\hat{q} + \cos\theta\,\hat{p}$$

In phase space: rotation by angle $\theta$.

**Symplectic Transformations:**
All Gaussian unitaries correspond to symplectic matrices $S$ acting on $(q, p)$:
$$\begin{pmatrix} q' \\ p' \end{pmatrix} = S \begin{pmatrix} q \\ p \end{pmatrix}, \quad S^T \Omega S = \Omega$$

where $\Omega = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$.

### 6. Non-Gaussian States and Operations

**Gaussian States Are Not Enough:**
For universal CV quantum computing, we need non-Gaussian resources.

**Gottesman-Knill Theorem (CV version):**
Gaussian states + Gaussian operations + homodyne detection = efficiently classically simulable

**Non-Gaussian States:**
- Fock states $|n\rangle$ for $n \geq 1$
- Cat states $|\alpha\rangle \pm |-\alpha\rangle$
- GKP states (tomorrow!)

**Non-Gaussian Operations:**
- Photon addition: $\hat{a}^\dagger|\psi\rangle$
- Photon subtraction: $\hat{a}|\psi\rangle$
- Kerr nonlinearity: $e^{i\chi(\hat{a}^\dagger\hat{a})^2}$
- Cubic phase gate: $e^{i\gamma\hat{q}^3}$

### 7. CV Quantum Gates

**Single-Mode Gaussian Gates:**
The set $\{\hat{D}(\alpha), \hat{S}(r), \hat{R}(\theta)\}$ generates all single-mode Gaussian unitaries.

**Two-Mode Gaussian Gates:**

*Beam Splitter:*
$$\hat{U}_{BS}(\theta) = e^{\theta(\hat{a}^\dagger\hat{b} - \hat{a}\hat{b}^\dagger)}$$

*Two-Mode Squeezing:*
$$\hat{S}_2(r) = e^{r(\hat{a}\hat{b} - \hat{a}^\dagger\hat{b}^\dagger)}$$

Creates entanglement! Output is a two-mode squeezed vacuum (EPR state).

**Universal CV Gate Set:**
Gaussian gates + Cubic phase gate = universal

The cubic phase gate:
$$\hat{V}(\gamma) = e^{i\gamma\hat{q}^3}$$

This is the CV analog of the T gate.

### 8. Homodyne and Heterodyne Detection

**Homodyne Detection:**
Measures a single quadrature (e.g., $\hat{q}$) with arbitrary precision.

POVM: $\{|q\rangle\langle q|\}$ where $|q\rangle$ is quadrature eigenstate.

For coherent state $|\alpha\rangle$:
$$P(q) = \frac{1}{\sqrt{\pi}}e^{-(q - q_0)^2}$$

**Heterodyne Detection:**
Simultaneously measures both quadratures with $\frac{1}{2}$ vacuum unit noise in each.

POVM: $\frac{1}{\pi}|\alpha\rangle\langle\alpha|$ (coherent state projection)

For any state $\hat{\rho}$:
$$P(\alpha) = \frac{1}{\pi}\langle\alpha|\hat{\rho}|\alpha\rangle = Q(\alpha)$$

This is the Husimi Q-function.

## Quantum Computing Applications

### CV Cluster States

**CV Cluster State:**
$$|\psi_{cluster}\rangle = \prod_{edges} \hat{C}_Z |+\rangle^{\otimes N}$$

where $|+\rangle$ is momentum squeezed state and $\hat{C}_Z = e^{i\hat{q}_1\hat{q}_2}$.

**One-Way CV Quantum Computing:**
1. Prepare large CV cluster state (Gaussian operation!)
2. Perform computation via homodyne measurements
3. Feed-forward classical results

**Advantage:** Deterministic entanglement generation using squeezing and beam splitters.

### Error Correction in CV Systems

CV systems face analog errors - shifts in $q$ and $p$. Error correction is fundamentally different from DV.

**Key insight:** Encode a DV qubit into CV modes (GKP encoding - tomorrow!).

## Worked Examples

### Example 1: Coherent State Quadrature Statistics

**Problem:** Calculate the mean and variance of $\hat{q}$ and $\hat{p}$ for coherent state $|\alpha\rangle$ where $\alpha = 2 + i$.

**Solution:**
For $|\alpha\rangle$ with $\alpha = q_0/\sqrt{2} + ip_0/\sqrt{2}$:
$$q_0 = \sqrt{2}\text{Re}(\alpha) = 2\sqrt{2}$$
$$p_0 = \sqrt{2}\text{Im}(\alpha) = \sqrt{2}$$

Mean values:
$$\langle\hat{q}\rangle = q_0 = 2\sqrt{2} \approx 2.83$$
$$\langle\hat{p}\rangle = p_0 = \sqrt{2} \approx 1.41$$

Variances (same as vacuum):
$$\langle\Delta q^2\rangle = \langle\Delta p^2\rangle = \frac{1}{2}$$

$$\boxed{\langle\hat{q}\rangle = 2\sqrt{2}, \quad \langle\hat{p}\rangle = \sqrt{2}, \quad \Delta q = \Delta p = \frac{1}{\sqrt{2}}}$$

### Example 2: Squeezed State Variances

**Problem:** A vacuum state is squeezed with $r = 1$ (about 8.7 dB). Calculate the quadrature variances.

**Solution:**
The squeezing operator transforms variances:
$$\langle\Delta q^2\rangle = \frac{1}{2}e^{-2r} = \frac{1}{2}e^{-2} \approx 0.068$$
$$\langle\Delta p^2\rangle = \frac{1}{2}e^{2r} = \frac{1}{2}e^{2} \approx 3.69$$

Standard deviations:
$$\Delta q \approx 0.26, \quad \Delta p \approx 1.92$$

Uncertainty product:
$$\Delta q \cdot \Delta p = \frac{1}{2}$$

Still minimum uncertainty, but redistributed!

Squeezing in dB:
$$S_{dB} = -10\log_{10}(e^{-2r}) = 10 \cdot 2r \cdot \log_{10}(e) \approx 8.7 \text{ dB}$$

### Example 3: Two-Mode Squeezed State Correlations

**Problem:** Calculate the correlation $\langle\hat{q}_1\hat{q}_2\rangle$ for a two-mode squeezed vacuum with parameter $r$.

**Solution:**
The two-mode squeezed vacuum:
$$|TMS, r\rangle = \hat{S}_2(r)|0,0\rangle = \frac{1}{\cosh r}\sum_{n=0}^\infty \tanh^n r |n,n\rangle$$

The two-mode squeezing creates correlations:
$$\hat{q}_1 + \hat{q}_2 \to e^{-r}(\hat{q}_1 + \hat{q}_2)$$ (squeezed)
$$\hat{q}_1 - \hat{q}_2 \to e^{r}(\hat{q}_1 - \hat{q}_2)$$ (anti-squeezed)

Variances:
$$\langle(\hat{q}_1 + \hat{q}_2)^2\rangle = e^{-2r}$$
$$\langle(\hat{q}_1 - \hat{q}_2)^2\rangle = e^{2r}$$

Since $\langle\hat{q}_1\rangle = \langle\hat{q}_2\rangle = 0$:
$$\langle\hat{q}_1^2\rangle + 2\langle\hat{q}_1\hat{q}_2\rangle + \langle\hat{q}_2^2\rangle = e^{-2r}$$
$$\langle\hat{q}_1^2\rangle - 2\langle\hat{q}_1\hat{q}_2\rangle + \langle\hat{q}_2^2\rangle = e^{2r}$$

Adding and subtracting:
$$\langle\hat{q}_1^2\rangle + \langle\hat{q}_2^2\rangle = \cosh(2r)$$
$$\langle\hat{q}_1\hat{q}_2\rangle = -\frac{1}{2}\sinh(2r)$$

$$\boxed{\langle\hat{q}_1\hat{q}_2\rangle = -\frac{1}{2}\sinh(2r)}$$

For large $r$: strong negative correlation (EPR-like).

## Practice Problems

### Level 1: Direct Application

1. **Quadrature Operators**

   Show that $[\hat{q}, \hat{p}] = i$ starting from $[\hat{a}, \hat{a}^\dagger] = 1$.

2. **Vacuum Wigner Function**

   Verify that $W_0(q, p) = \frac{1}{\pi}e^{-(q^2 + p^2)}$ integrates to 1.

3. **Displacement Action**

   If $|\psi\rangle = |0\rangle$ and we apply $\hat{D}(3 + 2i)$, what are the new mean values $\langle\hat{q}\rangle$ and $\langle\hat{p}\rangle$?

### Level 2: Intermediate

4. **Squeezed Coherent State**

   A coherent state $|\alpha = 1\rangle$ is squeezed with $r = 0.5$. Calculate:
   a) The new variances $\langle\Delta q^2\rangle$ and $\langle\Delta p^2\rangle$
   b) The new mean values (hint: squeezing doesn't change means for states centered at origin, but $|\alpha\rangle$ is not!)

5. **Covariance Matrix Evolution**

   Starting with vacuum ($\Sigma = \frac{1}{2}I$), apply:
   a) Squeezing with $r = 1$
   b) Rotation by $\theta = \pi/4$
   c) What is the final covariance matrix?

6. **Homodyne Measurement**

   A squeezed state with $r = 1$ squeezed along $q$ is measured with homodyne detection along $q$. What is the probability distribution of outcomes?

### Level 3: Challenging

7. **Photon-Added Coherent State**

   Calculate the Wigner function for the photon-added coherent state $\hat{a}^\dagger|\alpha\rangle$ (unnormalized). Show it has negative regions.

8. **CV Teleportation**

   In CV teleportation, Alice and Bob share a two-mode squeezed state. Alice performs Bell measurement (joint homodyne on her mode and input state). Derive the fidelity of teleportation as a function of squeezing $r$.

9. **Symplectic Eigenvalues**

   The covariance matrix of a two-mode Gaussian state is:
   $$\Sigma = \begin{pmatrix} a & 0 & c & 0 \\ 0 & a & 0 & -c \\ c & 0 & b & 0 \\ 0 & -c & 0 & b \end{pmatrix}$$
   Calculate the symplectic eigenvalues and determine when this state is entangled.

## Computational Lab: Wigner Function Visualization

```python
"""
Day 928 Computational Lab: Continuous Variable Quantum States
Wigner function calculation and visualization
"""

import numpy as np
from scipy.special import factorial, hermite
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid for phase space
def create_phase_space_grid(q_range=5, p_range=5, n_points=100):
    """Create meshgrid for phase space plotting."""
    q = np.linspace(-q_range, q_range, n_points)
    p = np.linspace(-p_range, p_range, n_points)
    Q, P = np.meshgrid(q, p)
    return Q, P, q, p


def wigner_vacuum(Q, P):
    """Wigner function of vacuum state."""
    return (1/np.pi) * np.exp(-(Q**2 + P**2))


def wigner_coherent(Q, P, alpha):
    """Wigner function of coherent state |α⟩."""
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    return (1/np.pi) * np.exp(-((Q - q0)**2 + (P - p0)**2))


def wigner_squeezed(Q, P, r, theta=0):
    """
    Wigner function of squeezed vacuum.
    r: squeezing parameter
    theta: squeezing angle
    """
    # Rotate coordinates
    Q_rot = Q * np.cos(theta) + P * np.sin(theta)
    P_rot = -Q * np.sin(theta) + P * np.cos(theta)

    # Apply squeezing
    return (1/np.pi) * np.exp(-(np.exp(2*r) * Q_rot**2 + np.exp(-2*r) * P_rot**2))


def wigner_thermal(Q, P, n_bar):
    """Wigner function of thermal state with mean photon number n_bar."""
    sigma = 2 * n_bar + 1
    return (1/(np.pi * sigma)) * np.exp(-(Q**2 + P**2)/sigma)


def wigner_fock_n(Q, P, n):
    """
    Wigner function of Fock state |n⟩.
    Uses the Laguerre polynomial formula.
    """
    from scipy.special import genlaguerre

    r2 = Q**2 + P**2
    Ln = genlaguerre(n, 0)
    W = ((-1)**n / np.pi) * Ln(2 * r2) * np.exp(-r2)
    return W


def wigner_cat_state(Q, P, alpha, phase=0):
    """
    Wigner function of cat state (|α⟩ + e^{iφ}|-α⟩)/N.
    phase: relative phase (0 for even cat, π for odd cat)
    """
    q0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)

    # Normalization
    overlap = np.exp(-2 * np.abs(alpha)**2)
    N2 = 2 * (1 + np.cos(phase) * overlap)

    # Two coherent state contributions
    W_plus = wigner_coherent(Q, P, alpha)
    W_minus = wigner_coherent(Q, P, -alpha)

    # Interference term
    W_int = (2/np.pi) * np.cos(2*p0*Q - 2*q0*P + phase) * np.exp(-(Q**2 + P**2))

    return (W_plus + W_minus + W_int) / N2


def plot_wigner(W, Q, P, title, filename, show_3d=True):
    """Plot Wigner function in 2D and optionally 3D."""
    fig = plt.figure(figsize=(14, 5))

    # 2D contour plot
    ax1 = fig.add_subplot(121)
    levels = np.linspace(W.min(), W.max(), 30)
    cf = ax1.contourf(Q, P, W, levels=levels, cmap='RdBu_r')
    ax1.contour(Q, P, W, levels=[0], colors='k', linewidths=1)
    plt.colorbar(cf, ax=ax1, label='W(q,p)')
    ax1.set_xlabel('q', fontsize=12)
    ax1.set_ylabel('p', fontsize=12)
    ax1.set_title(f'{title} - Contour', fontsize=14)
    ax1.set_aspect('equal')

    if show_3d:
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(Q, P, W, cmap='RdBu_r', alpha=0.8)
        ax2.set_xlabel('q')
        ax2.set_ylabel('p')
        ax2.set_zlabel('W(q,p)')
        ax2.set_title(f'{title} - 3D', fontsize=14)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


def demonstrate_gaussian_states():
    """Visualize various Gaussian states."""
    print("=" * 60)
    print("Gaussian State Wigner Functions")
    print("=" * 60)

    Q, P, q, p = create_phase_space_grid(4, 4, 150)

    # Vacuum
    W_vac = wigner_vacuum(Q, P)
    plot_wigner(W_vac, Q, P, 'Vacuum State', 'wigner_vacuum.png')

    # Coherent state
    alpha = 2 + 1j
    W_coh = wigner_coherent(Q, P, alpha)
    plot_wigner(W_coh, Q, P, f'Coherent State α={alpha}', 'wigner_coherent.png')

    # Squeezed state
    r = 0.7
    W_sq = wigner_squeezed(Q, P, r)
    plot_wigner(W_sq, Q, P, f'Squeezed State r={r}', 'wigner_squeezed.png')

    # Rotated squeezed state
    W_sq_rot = wigner_squeezed(Q, P, r, theta=np.pi/4)
    plot_wigner(W_sq_rot, Q, P, f'Rotated Squeezed State', 'wigner_squeezed_rotated.png')

    # Thermal state
    n_bar = 1.5
    W_th = wigner_thermal(Q, P, n_bar)
    plot_wigner(W_th, Q, P, f'Thermal State n̄={n_bar}', 'wigner_thermal.png')


def demonstrate_non_gaussian_states():
    """Visualize non-Gaussian states with Wigner negativity."""
    print("\n" + "=" * 60)
    print("Non-Gaussian State Wigner Functions")
    print("=" * 60)

    Q, P, q, p = create_phase_space_grid(4, 4, 150)

    # Fock states
    for n in [1, 2, 3]:
        W_fock = wigner_fock_n(Q, P, n)
        plot_wigner(W_fock, Q, P, f'Fock State |{n}⟩', f'wigner_fock_{n}.png')

        # Check for negativity
        min_val = W_fock.min()
        print(f"Fock |{n}⟩ minimum Wigner value: {min_val:.4f}")

    # Cat states
    alpha = 2
    for phase, name in [(0, 'even'), (np.pi, 'odd')]:
        W_cat = wigner_cat_state(Q, P, alpha, phase)
        plot_wigner(W_cat, Q, P, f'{name.capitalize()} Cat State α={alpha}',
                   f'wigner_cat_{name}.png')

        min_val = W_cat.min()
        print(f"{name.capitalize()} cat minimum Wigner value: {min_val:.4f}")


def gaussian_operations_demo():
    """Demonstrate Gaussian operations in phase space."""
    print("\n" + "=" * 60)
    print("Gaussian Operations Demonstration")
    print("=" * 60)

    Q, P, q, p = create_phase_space_grid(5, 5, 100)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Initial state: vacuum
    W0 = wigner_vacuum(Q, P)
    axes[0, 0].contourf(Q, P, W0, levels=30, cmap='RdBu_r')
    axes[0, 0].set_title('1. Vacuum |0⟩')
    axes[0, 0].set_aspect('equal')

    # After displacement
    alpha = 2 + 1j
    W1 = wigner_coherent(Q, P, alpha)
    axes[0, 1].contourf(Q, P, W1, levels=30, cmap='RdBu_r')
    axes[0, 1].set_title(f'2. After D({alpha})')
    axes[0, 1].set_aspect('equal')

    # After squeezing (on vacuum)
    r = 0.8
    W2 = wigner_squeezed(Q, P, r)
    axes[0, 2].contourf(Q, P, W2, levels=30, cmap='RdBu_r')
    axes[0, 2].set_title(f'3. After S(r={r})')
    axes[0, 2].set_aspect('equal')

    # After rotation
    W3 = wigner_squeezed(Q, P, r, theta=np.pi/3)
    axes[1, 0].contourf(Q, P, W3, levels=30, cmap='RdBu_r')
    axes[1, 0].set_title(f'4. After R(θ=π/3)')
    axes[1, 0].set_aspect('equal')

    # Displaced squeezed
    # First squeeze, then displace
    q0, p0 = 2, 1
    Q_shift = Q - q0
    P_shift = P - p0
    W4 = (1/np.pi) * np.exp(-(np.exp(2*r) * Q_shift**2 + np.exp(-2*r) * P_shift**2))
    axes[1, 1].contourf(Q, P, W4, levels=30, cmap='RdBu_r')
    axes[1, 1].set_title('5. Squeezed then Displaced')
    axes[1, 1].set_aspect('equal')

    # Different order: displace then squeeze
    # Squeezing also affects the displacement!
    q0_sq = q0 * np.exp(-r)
    p0_sq = p0 * np.exp(r)
    Q_shift2 = Q - q0_sq
    P_shift2 = P - p0_sq
    W5 = (1/np.pi) * np.exp(-(np.exp(2*r) * Q_shift2**2 + np.exp(-2*r) * P_shift2**2))
    axes[1, 2].contourf(Q, P, W5, levels=30, cmap='RdBu_r')
    axes[1, 2].set_title('6. Displaced then Squeezed')
    axes[1, 2].set_aspect('equal')

    for ax in axes.flat:
        ax.set_xlabel('q')
        ax.set_ylabel('p')

    plt.tight_layout()
    plt.savefig('gaussian_operations.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gaussian_operations.png")


def covariance_matrix_analysis():
    """Analyze covariance matrices for different states."""
    print("\n" + "=" * 60)
    print("Covariance Matrix Analysis")
    print("=" * 60)

    # Vacuum
    sigma_vac = 0.5 * np.eye(2)
    print("Vacuum covariance matrix:")
    print(sigma_vac)

    # Squeezed (r=1)
    r = 1.0
    sigma_sq = 0.5 * np.diag([np.exp(-2*r), np.exp(2*r)])
    print(f"\nSqueezed (r={r}) covariance matrix:")
    print(sigma_sq)

    # Rotated squeezed
    theta = np.pi/4
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    sigma_rot = R @ sigma_sq @ R.T
    print(f"\nRotated squeezed (θ=π/4) covariance matrix:")
    print(sigma_rot)

    # Thermal
    n_bar = 2.0
    sigma_th = (n_bar + 0.5) * np.eye(2)
    print(f"\nThermal (n̄={n_bar}) covariance matrix:")
    print(sigma_th)

    # Two-mode squeezed (just the 4x4 structure)
    r = 0.5
    c, s = np.cosh(2*r), np.sinh(2*r)
    sigma_tms = 0.5 * np.array([
        [c, 0, s, 0],
        [0, c, 0, -s],
        [s, 0, c, 0],
        [0, -s, 0, c]
    ])
    print(f"\nTwo-mode squeezed (r={r}) covariance matrix:")
    print(sigma_tms)

    # Check uncertainty relation
    print("\nUncertainty products (should be ≥ 0.25):")
    print(f"  Vacuum: {np.linalg.det(sigma_vac):.4f}")
    print(f"  Squeezed: {np.linalg.det(sigma_sq):.4f}")
    print(f"  Rotated: {np.linalg.det(sigma_rot):.4f}")
    print(f"  Thermal: {np.linalg.det(sigma_th):.4f}")


def homodyne_simulation():
    """Simulate homodyne detection outcomes."""
    print("\n" + "=" * 60)
    print("Homodyne Detection Simulation")
    print("=" * 60)

    n_samples = 10000

    # Vacuum state homodyne
    q_vacuum = np.random.normal(0, 1/np.sqrt(2), n_samples)

    # Coherent state homodyne
    alpha = 2
    q0 = np.sqrt(2) * alpha
    q_coherent = np.random.normal(q0, 1/np.sqrt(2), n_samples)

    # Squeezed state homodyne
    r = 1.0
    sigma_sq = np.exp(-r) / np.sqrt(2)
    q_squeezed = np.random.normal(0, sigma_sq, n_samples)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(q_vacuum, bins=50, density=True, alpha=0.7, label='Sampled')
    q_plot = np.linspace(-4, 4, 200)
    axes[0].plot(q_plot, np.exp(-q_plot**2) / np.sqrt(np.pi), 'r-',
                 linewidth=2, label='Theory')
    axes[0].set_xlabel('q measurement', fontsize=12)
    axes[0].set_ylabel('Probability density', fontsize=12)
    axes[0].set_title('Vacuum State Homodyne', fontsize=14)
    axes[0].legend()

    axes[1].hist(q_coherent, bins=50, density=True, alpha=0.7, label='Sampled')
    axes[1].plot(q_plot + q0, np.exp(-q_plot**2) / np.sqrt(np.pi), 'r-',
                 linewidth=2, label='Theory')
    axes[1].set_xlabel('q measurement', fontsize=12)
    axes[1].set_title(f'Coherent State α={alpha}', fontsize=14)
    axes[1].legend()

    axes[2].hist(q_squeezed, bins=50, density=True, alpha=0.7, label='Sampled')
    sigma = sigma_sq
    axes[2].plot(q_plot, np.exp(-q_plot**2 / (2*sigma**2)) /
                 (sigma * np.sqrt(2*np.pi)), 'r-', linewidth=2, label='Theory')
    axes[2].set_xlabel('q measurement', fontsize=12)
    axes[2].set_title(f'Squeezed State r={r}', fontsize=14)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('homodyne_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: homodyne_simulation.png")

    # Calculate variances
    print("\nMeasured variances:")
    print(f"  Vacuum: {np.var(q_vacuum):.4f} (theory: 0.5)")
    print(f"  Coherent: {np.var(q_coherent):.4f} (theory: 0.5)")
    print(f"  Squeezed: {np.var(q_squeezed):.4f} (theory: {0.5*np.exp(-2*r):.4f})")


def main():
    """Run all CV quantum computing demonstrations."""
    print("\n" + "=" * 60)
    print("DAY 928: CONTINUOUS VARIABLE QUANTUM COMPUTING")
    print("=" * 60)

    # Basic demonstrations
    demonstrate_gaussian_states()
    demonstrate_non_gaussian_states()
    gaussian_operations_demo()
    covariance_matrix_analysis()
    homodyne_simulation()

    print("\n" + "=" * 60)
    print("Simulations Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Quadrature operators | $\hat{q} = (\hat{a} + \hat{a}^\dagger)/\sqrt{2}$, $\hat{p} = i(\hat{a}^\dagger - \hat{a})/\sqrt{2}$ |
| Commutator | $[\hat{q}, \hat{p}] = i$ |
| Uncertainty relation | $\Delta q \cdot \Delta p \geq 1/2$ |
| Wigner function | $W(q,p) = \frac{1}{\pi}\int \langle q-y\|\hat{\rho}\|q+y\rangle e^{2ipy} dy$ |
| Displacement | $\hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$ |
| Squeezing | $\hat{S}(r) = e^{\frac{r}{2}(\hat{a}^2 - \hat{a}^{\dagger 2})}$ |
| Squeezed variance | $\langle\Delta q^2\rangle = \frac{1}{2}e^{-2r}$ |

### Key Takeaways

1. **Continuous variables** use infinite-dimensional Hilbert spaces (qumodes vs qubits)
2. **Quadrature operators** $\hat{q}$ and $\hat{p}$ are the fundamental observables
3. **Gaussian states** (vacuum, coherent, squeezed, thermal) have Gaussian Wigner functions
4. **Wigner negativity** is a signature of non-classicality, required for quantum advantage
5. **Gaussian operations** alone are efficiently classically simulable
6. **Non-Gaussian resources** (Fock states, cat states, cubic phase) enable universal CV QC

## Daily Checklist

- [ ] I can define quadrature operators and derive their commutator
- [ ] I understand Gaussian states and their phase space representation
- [ ] I can calculate Wigner functions for basic quantum states
- [ ] I understand the role of non-Gaussianity in CV quantum computing
- [ ] I completed the computational lab simulations
- [ ] I solved at least 3 practice problems

## Preview of Day 929

Tomorrow we study **GKP (Gottesman-Kitaev-Preskill) Encoding**, which encodes a discrete qubit into a continuous-variable mode. Key topics:
- Grid states in phase space
- Logical Pauli and Clifford operations
- Error correction for shift errors
- Approximate GKP states and their preparation
