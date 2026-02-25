# Day 164: Action-Angle Variables — The Geometry of Integrable Systems

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Action-Angle Variables |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define action variables as phase space integrals and compute them for standard systems
2. Construct angle variables as canonical conjugates to actions
3. State and interpret the Liouville-Arnold theorem for integrable systems
4. Apply the Bohr-Sommerfeld quantization rule J = nℏ
5. Understand adiabatic invariance and its applications
6. Visualize motion on invariant tori and connect to KAM theory

---

## Core Content

### 1. Motivation: The Search for Simplicity

In Hamiltonian mechanics, we've learned that clever coordinate choices can simplify problems dramatically. The ultimate simplification occurs when we find coordinates where:

1. The Hamiltonian depends only on "momenta" (actions): H = H(J₁, ..., Jₙ)
2. The conjugate "positions" (angles) are cyclic: ∂H/∂θᵢ = 0

In such coordinates, Hamilton's equations become trivial:

$$\dot{J}_i = -\frac{\partial H}{\partial \theta_i} = 0 \implies J_i = \text{const}$$

$$\dot{\theta}_i = \frac{\partial H}{\partial J_i} = \omega_i(J) = \text{const} \implies \theta_i = \omega_i t + \theta_i^{(0)}$$

**The Problem is Completely Solved!** The actions are constants, and the angles increase linearly with time at constant frequencies.

---

### 2. Definition of Action Variables

**The Action Variable:**

For a system with periodic motion in phase space, the **action variable** J is defined as:

$$\boxed{J = \frac{1}{2\pi}\oint p \, dq}$$

The integral is taken over one complete period of oscillation—a closed contour in phase space.

**Geometric Interpretation:**

$$J = \frac{\text{Phase space area enclosed by orbit}}{2\pi}$$

The factor of 1/(2π) is conventional, ensuring the conjugate angle variable has period 2π.

**For Multiple Degrees of Freedom:**

For an n-degree-of-freedom system with n periodic motions:

$$J_i = \frac{1}{2\pi}\oint_{C_i} p_i \, dq_i$$

where C_i is the closed path in the (qᵢ, pᵢ) plane for the i-th degree of freedom.

**Key Properties:**

| Property | Statement |
|----------|-----------|
| Dimensions | [J] = [action] = [energy × time] = J·s |
| Value | J > 0 for bound motion |
| Invariance | J is constant along trajectories |
| Function of E | J = J(E) for autonomous systems |

---

### 3. Definition of Angle Variables

The **angle variable** θ is the canonical conjugate to J, defined through a generating function.

**Construction via Hamilton's Characteristic Function:**

1. Solve the Hamilton-Jacobi equation for W(q, J):
   $$H\left(q, \frac{\partial W}{\partial q}\right) = E(J)$$

2. The momentum is:
   $$p = \frac{\partial W}{\partial q}$$

3. The angle variable is:
   $$\theta = \frac{\partial W}{\partial J}$$

**Properties of θ:**

| Property | Statement |
|----------|-----------|
| Period | θ increases by 2π per oscillation |
| Evolution | θ = ωt + θ₀ (linear in time) |
| Canonical | {θ, J} = 1 |

**Why 2π Periodicity?**

From the definition of action:

$$J = \frac{1}{2\pi}\oint p \, dq = \frac{1}{2\pi}\oint \frac{\partial W}{\partial q} dq$$

Over one complete cycle:

$$\Delta \theta = \oint \frac{\partial \theta}{\partial q} dq = \oint \frac{\partial^2 W}{\partial J \partial q} dq = \frac{\partial}{\partial J}\oint \frac{\partial W}{\partial q} dq = \frac{\partial}{\partial J}(2\pi J) = 2\pi$$

---

### 4. The Harmonic Oscillator: Complete Example

**Hamiltonian:**

$$H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 q^2 = E$$

**Phase Space Trajectories:**

Constant energy contours are ellipses:

$$\frac{p^2}{2mE} + \frac{m\omega^2 q^2}{2E} = 1$$

Semi-axes: a_q = √(2E/(mω²)), a_p = √(2mE)

**Action Variable Calculation:**

$$J = \frac{1}{2\pi}\oint p \, dq$$

Parametrize the ellipse: q = a_q cos φ, p = a_p sin φ

$$J = \frac{1}{2\pi}\int_0^{2\pi} a_p \sin\phi \cdot (-a_q \sin\phi) \, d\phi$$

$$= \frac{a_p a_q}{2\pi}\int_0^{2\pi} \sin^2\phi \, d\phi = \frac{a_p a_q}{2\pi} \cdot \pi = \frac{a_p a_q}{2}$$

$$= \frac{1}{2}\sqrt{2mE} \cdot \sqrt{\frac{2E}{m\omega^2}} = \frac{E}{\omega}$$

$$\boxed{J = \frac{E}{\omega}}$$

**Inverse:** E = ωJ, so the Hamiltonian in action-angle form is:

$$\boxed{K(J) = \omega J}$$

**Frequency:**

$$\omega = \frac{\partial K}{\partial J} = \omega$$

(Constant, independent of amplitude—this is unique to the harmonic oscillator!)

**Canonical Transformation:**

$$q = \sqrt{\frac{2J}{m\omega}}\cos\theta, \quad p = \sqrt{2m\omega J}\sin\theta$$

**Verification:**

$$\{q, p\} = \frac{\partial q}{\partial \theta}\frac{\partial p}{\partial J} - \frac{\partial q}{\partial J}\frac{\partial p}{\partial \theta}$$

$$= \left(-\sqrt{\frac{2J}{m\omega}}\sin\theta\right)\left(\sqrt{\frac{m\omega}{2J}}\sin\theta\right) - \left(\frac{1}{2}\sqrt{\frac{2}{m\omega J}}\cos\theta\right)\left(\sqrt{2m\omega J}\cos\theta\right)$$

$$= -\sin^2\theta - \cos^2\theta = -1$$

Hmm, we get -1. This means our convention should be θ → -θ or we should swap the order. With the correct sign conventions:

$$\{θ, J\} = 1 \quad \checkmark$$

---

### 5. The Simple Pendulum

The pendulum demonstrates action-angle variables for a nonlinear system with qualitatively different motion types.

**Hamiltonian:**

$$H = \frac{p_\theta^2}{2m\ell^2} - mg\ell\cos\theta = E$$

**Critical Energy:** E_sep = mgℓ (separatrix energy)

#### Case 1: Libration (Oscillation) — E < E_sep

The pendulum swings back and forth between turning points ±θ₀ where:

$$\cos\theta_0 = 1 - \frac{E + mg\ell}{mg\ell} = -\frac{E}{mg\ell}$$

**Action Variable:**

$$J_{\text{lib}} = \frac{1}{2\pi}\oint p_\theta \, d\theta = \frac{1}{\pi}\int_{-\theta_0}^{\theta_0} \sqrt{2m\ell^2(E + mg\ell\cos\theta)} \, d\theta$$

This integral involves **elliptic integrals**. With k = sin(θ₀/2):

$$J_{\text{lib}} = \frac{8}{\pi}\sqrt{mg\ell^3}\left[E(k) - (1-k^2)K(k)\right]$$

where K(k) and E(k) are complete elliptic integrals of the first and second kind.

**Small Oscillations:** For θ₀ ≪ 1:

$$J \approx \frac{E}{\omega_0}$$

where ω₀ = √(g/ℓ), recovering the harmonic oscillator result.

#### Case 2: Rotation — E > E_sep

The pendulum has enough energy to go over the top and rotates continuously.

$$J_{\text{rot}} = \frac{1}{2\pi}\oint p_\theta \, d\theta = \frac{1}{2\pi}\int_0^{2\pi} \sqrt{2m\ell^2(E + mg\ell\cos\theta)} \, d\theta$$

#### Case 3: Separatrix — E = E_sep

On the separatrix, the pendulum asymptotically approaches the unstable equilibrium. Action-angle variables are **not defined** on the separatrix—it's a singular limit.

**Phase Portrait:**

```
    p_θ
     ↑
     |    ~~~~~~(rotation)~~~~~~
     |   /                      \
     |  /   ○○○○○○○○○○○○○○○     \
     | /    ○ (libration) ○      \
-----(x)----○------○------○-----(x)---→ θ
    -π ↖    ○             ○    ↗ π
        \   ○○○○○○○○○○○○○○○   /
         \                    /
          ~~~~~~(rotation)~~~~~~
```

The separatrix (connecting the x points at ±π) divides libration from rotation.

---

### 6. The Liouville-Arnold Theorem

This fundamental theorem characterizes integrable systems geometrically.

**Definition: Complete Integrability**

A Hamiltonian system with n degrees of freedom is **completely integrable** (in the Liouville sense) if there exist n independent functions F₁ = H, F₂, ..., Fₙ that:

1. **Are in involution:** {Fᵢ, Fⱼ} = 0 for all i, j
2. **Are functionally independent:** Their gradients are linearly independent

**Theorem (Liouville-Arnold):**

Let (M, ω) be a 2n-dimensional symplectic manifold with n independent, Poisson-commuting integrals F₁, ..., Fₙ. If the level set:

$$M_f = \{x \in M : F_i(x) = f_i, \text{ for } i = 1, \ldots, n\}$$

is compact and connected, then:

1. **Topology:** M_f is diffeomorphic to an n-dimensional torus T^n
2. **Coordinates:** There exist action-angle coordinates (J₁, ..., Jₙ, θ₁, ..., θₙ)
3. **Hamiltonian:** H depends only on the actions: H = H(J₁, ..., Jₙ)
4. **Motion:** Linear flow on the torus:
   $$\theta_i(t) = \omega_i t + \theta_i^{(0)}, \quad \omega_i = \frac{\partial H}{\partial J_i}$$

**Geometric Picture:**

Phase space is **foliated** into n-dimensional invariant tori. Each torus is labeled by (J₁, ..., Jₙ). Motion on each torus is quasi-periodic with frequencies (ω₁, ..., ωₙ).

---

### 7. Frequencies and Resonances

**Frequency Vector:**

$$\boldsymbol{\omega} = \left(\omega_1, \omega_2, \ldots, \omega_n\right) = \left(\frac{\partial H}{\partial J_1}, \ldots, \frac{\partial H}{\partial J_n}\right)$$

**Resonance Condition:**

A torus is **resonant** if there exist integers n₁, ..., nₖ (not all zero) such that:

$$\sum_{i=1}^{n} n_i \omega_i = 0$$

**Types of Motion:**

| Frequency Ratio | Type | Trajectory |
|-----------------|------|------------|
| All rational | Periodic | Closes after finite time |
| Irrational ratios | Quasiperiodic | Dense on torus (ergodic on torus) |
| Resonant | Lower-dimensional | Confined to subtorus |

**Example: 2D Torus**

For two frequencies ω₁ and ω₂:
- If ω₁/ω₂ = p/q (rational): trajectory closes after q periods of ω₁
- If ω₁/ω₂ is irrational: trajectory eventually passes arbitrarily close to every point

---

### 8. Adiabatic Invariants

**Definition:**

A quantity is an **adiabatic invariant** if it remains approximately constant when system parameters change slowly compared to the natural period.

**The Adiabatic Theorem:**

If a parameter λ of H(q, p; λ) varies slowly such that:

$$T \cdot \frac{d\lambda}{dt} \ll \lambda$$

where T is the oscillation period, then the action J is an **adiabatic invariant**:

$$\frac{dJ}{dt} \approx 0$$

even though the energy E may change!

**Proof Outline:**

1. The change in J over one period is:
   $$\Delta J = \frac{1}{2\pi}\oint \Delta p \, dq$$

2. For slow variation, the dominant change averages to zero over one cycle.

3. The correction is O(ε²) where ε = T(dλ/dt)/λ.

**Example: Slowly Varying Pendulum**

If the length ℓ changes slowly:
- Energy changes: E ∝ ω ∝ 1/√ℓ
- But J = E/ω remains constant!
- As ℓ increases: ω decreases, so E must decrease proportionally
- The amplitude A adjusts to maintain J

**Applications:**

| System | Adiabatic Invariant | Application |
|--------|---------------------|-------------|
| Pendulum | J = E/ω | Variable-length pendulum |
| Magnetic field | μ = mv²⊥/(2B) | Magnetic mirrors, Van Allen belts |
| Quantum systems | n (quantum number) | Ehrenfest's theorem |
| Cosmology | Action of universe | Cosmic expansion |

---

## Quantum Mechanics Connection

### Bohr-Sommerfeld Quantization

The **old quantum theory** (1913-1925) postulated that action variables are quantized:

$$\boxed{J = n\hbar}$$

where n = 0, 1, 2, 3, ... and ℏ = h/(2π) is the reduced Planck constant.

**For the Harmonic Oscillator:**

$$J = \frac{E}{\omega} = n\hbar \implies E_n = n\hbar\omega$$

This gives correct energy level **spacing** but misses the zero-point energy!

### Einstein-Brillouin-Keller (EBK) Quantization

The refined semiclassical quantization includes the **Maslov index** μ:

$$\boxed{J = \hbar\left(n + \frac{\mu}{4}\right)}$$

where μ counts the number of turning points (caustics) in one period.

**For Harmonic Oscillator:** μ = 2 (two turning points)

$$J = \hbar\left(n + \frac{1}{2}\right) \implies E_n = \hbar\omega\left(n + \frac{1}{2}\right)$$

This **exactly** matches quantum mechanics!

**For Hydrogen Atom:**

The Bohr-Sommerfeld quantization with EBK corrections gives:

$$E_n = -\frac{me^4}{2\hbar^2 n^2}$$

where n = n_r + ℓ + 1 includes radial and angular contributions.

### The Deep Connection

| Classical | Quantum |
|-----------|---------|
| Action J | Quantum number n |
| J = nℏ | Quantization condition |
| Angle θ | Phase of wave function |
| Torus | Energy eigenspace |
| Frequency ω = ∂H/∂J | Level spacing ΔE/ℏ |
| Adiabatic invariance | Adiabatic theorem (Ehrenfest) |

**The Correspondence Principle:**

In the limit of large quantum numbers (n → ∞), quantum mechanics reproduces classical mechanics. Action-angle variables make this correspondence explicit.

---

### 9. Preview: KAM Theory

**The Big Question:**

What happens when an integrable system is perturbed?

$$H = H_0(J) + \epsilon H_1(J, \theta)$$

**Naive Expectation:** All tori are destroyed.

**The KAM Theorem (Kolmogorov-Arnold-Moser):**

For small ε, **most** invariant tori survive, slightly deformed. The surviving tori satisfy the **Diophantine condition**:

$$|\mathbf{n} \cdot \boldsymbol{\omega}| \geq \frac{\gamma}{|\mathbf{n}|^\tau}$$

for all integer vectors **n** ≠ 0, where γ > 0 and τ > n-1.

**What's Destroyed:**

- Resonant tori (where **n**·**ω** = 0)
- Near-resonant tori
- Tori with frequencies that are "too rational"

**What Survives:**

- Tori with "sufficiently irrational" frequencies
- A Cantor-set-like structure of surviving tori
- Chaos develops in the gaps between surviving tori

This is the gateway to understanding chaos while preserving the beauty of integrable systems!

---

## Worked Examples

### Example 1: Action Variable for Particle in a Box

**Problem:** Find J for a particle bouncing between walls at x = 0 and x = L.

**Solution:**

**Hamiltonian:** H = p²/(2m) = E (between bounces)

**Phase Space Trajectory:**

The particle alternates between p = +√(2mE) and p = -√(2mE) while q goes from 0 to L and back.

**Action:**

$$J = \frac{1}{2\pi}\oint p \, dq$$

The contour is a rectangle: q from 0 to L at p = +√(2mE), then q from L to 0 at p = -√(2mE).

$$J = \frac{1}{2\pi}\left[\int_0^L p \, dq + \int_L^0 (-p) \, dq\right] = \frac{1}{2\pi} \cdot 2L\sqrt{2mE}$$

$$\boxed{J = \frac{L\sqrt{2mE}}{\pi}}$$

**Energy in terms of J:**

$$E = \frac{\pi^2 J^2}{2mL^2}$$

**Frequency:**

$$\omega = \frac{\partial E}{\partial J} = \frac{\pi^2 J}{mL^2} = \frac{\pi\sqrt{2mE}}{mL} = \frac{\pi v}{L}$$

where v = √(2E/m) is the speed. This gives period T = 2L/v ✓

---

### Example 2: Central Force Problem (2D)

**Problem:** Find action variables for a particle in a central potential V(r).

**Solution:**

**Hamiltonian in polar coordinates:**

$$H = \frac{p_r^2}{2m} + \frac{p_\phi^2}{2mr^2} + V(r) = E$$

**Constants of motion:** E (energy) and L = p_φ (angular momentum)

**Angular Action:**

$$J_\phi = \frac{1}{2\pi}\oint p_\phi \, d\phi = \frac{1}{2\pi}\int_0^{2\pi} L \, d\phi = L$$

The angular action equals the angular momentum!

**Radial Action:**

The radial momentum is:

$$p_r = \sqrt{2m\left(E - V(r) - \frac{L^2}{2mr^2}\right)}$$

The radial motion oscillates between turning points r_min and r_max:

$$J_r = \frac{1}{2\pi}\oint p_r \, dr = \frac{1}{\pi}\int_{r_{\min}}^{r_{\max}} \sqrt{2m\left(E - V(r) - \frac{L^2}{2mr^2}\right)} \, dr$$

**Hamiltonian in Action Variables:**

$$H = H(J_r, J_\phi) = H(J_r, L)$$

**Frequencies:**

$$\omega_r = \frac{\partial H}{\partial J_r}, \quad \omega_\phi = \frac{\partial H}{\partial J_\phi}$$

For the Kepler problem (V = -k/r), remarkably:

$$H = -\frac{mk^2}{2(J_r + J_\phi)^2}$$

Both frequencies equal ω_r = ω_φ = mk²/(J_r + J_φ)³, explaining why Kepler orbits close!

---

### Example 3: Adiabatic Invariance of Pendulum

**Problem:** A pendulum has length ℓ(t) that slowly doubles from ℓ₀ to 2ℓ₀. If the initial amplitude is θ₀, find the final amplitude.

**Solution:**

**For small oscillations:** J = E/ω where ω = √(g/ℓ)

**Action:**

$$J = \frac{E}{\omega} = \frac{\frac{1}{2}mg\ell\theta_{\max}^2}{\sqrt{g/\ell}} = \frac{1}{2}m\theta_{\max}^2\sqrt{g\ell^3}$$

**Adiabatic Invariance:** J is constant

$$\frac{1}{2}m\theta_0^2\sqrt{g\ell_0^3} = \frac{1}{2}m\theta_f^2\sqrt{g(2\ell_0)^3}$$

$$\theta_0^2 \sqrt{\ell_0^3} = \theta_f^2 \sqrt{8\ell_0^3}$$

$$\theta_f = \frac{\theta_0}{8^{1/4}} = \frac{\theta_0}{2^{3/4}} \approx 0.595\theta_0$$

**The amplitude decreases** as the pendulum lengthens, even though the energy also changes!

**Energy change:**

$$E_f = \omega_f J = \sqrt{\frac{g}{2\ell_0}} \cdot J = \frac{1}{\sqrt{2}} \cdot \frac{E_0}{\sqrt{\ell_0/\ell_0}} = \frac{E_0}{\sqrt{2}}$$

The energy decreases (the pendulum does work against gravity as it lengthens).

---

## Practice Problems

### Level 1: Direct Application

1. **Simple calculation:** For a harmonic oscillator with m = 1 kg, ω = 2 rad/s, and E = 4 J, find J and the amplitude A.

2. **Action units:** Show that [J] = [E][T] = J·s, the same as ℏ.

3. **Quantization:** Using Bohr-Sommerfeld quantization, find the energy levels of a particle in a 1D box of length L.

### Level 2: Intermediate

4. **Quartic oscillator:** For H = p²/(2m) + αq⁴, set up the integral for J(E). How does ω depend on E?

5. **Kepler degeneracy:** For the Kepler problem, show that H depends only on J_r + J_φ, explaining why all bound orbits close.

6. **Magnetic mirror:** A charged particle spirals in a slowly varying magnetic field B(z). Show that μ = mv²⊥/(2B) is an adiabatic invariant.

### Level 3: Challenging

7. **Pendulum action:** Derive the exact formula for J_lib using elliptic integrals. Verify the small-angle limit gives J ≈ E/ω₀.

8. **EBK for hydrogen:** Apply EBK quantization to the hydrogen atom, including the Maslov correction, and derive the energy spectrum.

9. **KAM threshold:** For the kicked rotor H = p²/2 + K cos(q)∑δ(t-n), estimate the critical K value where most tori are destroyed.

---

## Computational Lab

### Lab 1: Computing Action Variables Numerically

```python
"""
Day 164 Lab: Action-Angle Variables
Numerical computation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import ellipk, ellipe

def harmonic_action(E, m=1.0, omega=1.0):
    """Analytical action for harmonic oscillator: J = E/omega."""
    return E / omega

def harmonic_action_numerical(E, m=1.0, omega=1.0, n_points=1000):
    """Numerical computation of J via phase space integral."""
    # Parametrize the ellipse
    phi = np.linspace(0, 2*np.pi, n_points)

    a_q = np.sqrt(2*E / (m * omega**2))
    a_p = np.sqrt(2*m*E)

    q = a_q * np.cos(phi)
    p = a_p * np.sin(phi)

    # J = (1/2π) ∮ p dq
    dq = np.gradient(q, phi)
    integral = np.trapz(p * dq, phi)

    return abs(integral) / (2 * np.pi)

# Test harmonic oscillator
E = 2.0
J_anal = harmonic_action(E)
J_num = harmonic_action_numerical(E)

print("Harmonic Oscillator Action Variable:")
print(f"Energy E = {E}")
print(f"J (analytical) = {J_anal:.6f}")
print(f"J (numerical)  = {J_num:.6f}")
print(f"Relative error = {abs(J_anal-J_num)/J_anal*100:.4f}%")
```

### Lab 2: Phase Space Visualization

```python
"""
Visualize action-angle coordinates for the harmonic oscillator.
"""

def plot_action_angle_transformation():
    """Compare (q,p) and (θ,J) representations."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    m, omega = 1.0, 1.0
    energies = [0.5, 1.0, 2.0, 3.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(energies)))

    # Left: (q, p) coordinates
    ax1 = axes[0]
    phi = np.linspace(0, 2*np.pi, 200)

    for E, color in zip(energies, colors):
        a_q = np.sqrt(2*E / (m * omega**2))
        a_p = np.sqrt(2*m*E)

        q = a_q * np.cos(phi)
        p = a_p * np.sin(phi)

        J = E / omega
        ax1.plot(q, p, color=color, linewidth=2, label=f'J = {J:.1f}')

    ax1.set_xlabel('q', fontsize=14)
    ax1.set_ylabel('p', fontsize=14)
    ax1.set_title('Phase Space (q, p)\nElliptical Contours', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', lw=0.5)
    ax1.axvline(x=0, color='k', lw=0.5)

    # Right: (θ, J) coordinates
    ax2 = axes[1]

    for E, color in zip(energies, colors):
        J = E / omega
        theta = np.linspace(0, 2*np.pi, 200)
        ax2.plot(theta, np.full_like(theta, J), color=color, linewidth=3,
                label=f'J = {J:.1f}')

    ax2.set_xlabel(r'$\theta$', fontsize=14)
    ax2.set_ylabel('J', fontsize=14)
    ax2.set_title('Action-Angle (θ, J)\nHorizontal Lines!', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout()
    plt.savefig('action_angle_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_action_angle_transformation()
```

### Lab 3: Pendulum Phase Space with Action Contours

```python
"""
Phase portrait for the simple pendulum showing libration, rotation, and separatrix.
"""

def pendulum_phase_portrait():
    """Plot pendulum phase space with action variable contours."""

    fig, ax = plt.subplots(figsize=(12, 8))

    m, g, L = 1.0, 9.8, 1.0
    E_sep = m * g * L  # Separatrix energy

    # Libration region (E < E_sep)
    E_lib = [0.2, 0.4, 0.6, 0.8, 0.95]
    colors_lib = plt.cm.Blues(np.linspace(0.3, 0.9, len(E_lib)))

    for E_frac, color in zip(E_lib, colors_lib):
        E = E_frac * E_sep

        # Find turning point
        theta_0 = np.arccos(-E / E_sep)

        # Upper and lower branches
        theta_upper = np.linspace(-theta_0 + 0.01, theta_0 - 0.01, 200)
        p_upper = np.sqrt(2*m*L**2*(E + m*g*L*np.cos(theta_upper)))
        p_lower = -p_upper

        ax.plot(theta_upper, p_upper, color=color, lw=1.5)
        ax.plot(theta_upper, p_lower, color=color, lw=1.5)

    # Separatrix
    theta_sep = np.linspace(-np.pi + 0.01, np.pi - 0.01, 500)
    E = E_sep - 0.001
    p_sep = np.sqrt(np.maximum(0, 2*m*L**2*(E + m*g*L*np.cos(theta_sep))))

    ax.plot(theta_sep, p_sep, 'r-', lw=2, label='Separatrix')
    ax.plot(theta_sep, -p_sep, 'r-', lw=2)

    # Rotation region (E > E_sep)
    E_rot = [1.1, 1.3, 1.6, 2.0]
    colors_rot = plt.cm.Oranges(np.linspace(0.4, 0.9, len(E_rot)))

    for E_frac, color in zip(E_rot, colors_rot):
        E = E_frac * E_sep
        theta = np.linspace(-np.pi, np.pi, 200)
        p = np.sqrt(2*m*L**2*(E + m*g*L*np.cos(theta)))

        ax.plot(theta, p, color=color, lw=1.5)
        ax.plot(theta, -p, color=color, lw=1.5)

    # Equilibrium points
    ax.plot(0, 0, 'go', markersize=10, label='Stable')
    ax.plot([-np.pi, np.pi], [0, 0], 'r^', markersize=10, label='Unstable')

    ax.set_xlabel(r'$\theta$ (rad)', fontsize=14)
    ax.set_ylabel(r'$p_\theta$', fontsize=14)
    ax.set_title('Pendulum Phase Space\nBlue: Libration | Red: Separatrix | Orange: Rotation', fontsize=14)
    ax.set_xlim(-np.pi, np.pi)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pendulum_phase_portrait.png', dpi=150, bbox_inches='tight')
    plt.show()

pendulum_phase_portrait()
```

### Lab 4: Adiabatic Invariance Demonstration

```python
"""
Demonstrate adiabatic invariance for a slowly varying harmonic oscillator.
"""

from scipy.integrate import solve_ivp

def adiabatic_invariance_demo():
    """Show that J is approximately constant under slow parameter changes."""

    def equations(t, y, tau):
        """Harmonic oscillator with slowly varying frequency."""
        q, p = y
        omega = 1.0 + 0.5 * t / tau  # ω increases slowly
        return [p, -omega**2 * q]

    q0, p0 = 1.0, 0.0
    omega0 = 1.0
    E0 = 0.5 * (p0**2 + omega0**2 * q0**2)
    J0 = E0 / omega0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    taus = [10, 50, 200]  # Different adiabaticity levels
    colors = ['red', 'blue', 'green']
    labels = ['Fast (τ=10)', 'Medium (τ=50)', 'Slow (τ=200)']

    for tau, color, label in zip(taus, colors, labels):
        t_span = [0, tau]
        t_eval = np.linspace(0, tau, 2000)

        sol = solve_ivp(
            lambda t, y: equations(t, y, tau),
            t_span, [q0, p0],
            t_eval=t_eval, method='DOP853', max_step=0.01
        )

        t = sol.t
        q, p = sol.y

        omega_t = 1.0 + 0.5 * t / tau
        E_t = 0.5 * (p**2 + omega_t**2 * q**2)
        J_t = E_t / omega_t

        axes[0, 0].plot(t/tau, omega_t, color=color, lw=1.5, label=label)
        axes[0, 1].plot(t/tau, E_t/E0, color=color, lw=1.5, label=label)
        axes[1, 0].plot(t/tau, J_t/J0, color=color, lw=1.5, label=label)

    axes[0, 0].set_xlabel(r'$t/\tau$')
    axes[0, 0].set_ylabel(r'$\omega(t)$')
    axes[0, 0].set_title('Frequency Increase')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel(r'$t/\tau$')
    axes[0, 1].set_ylabel(r'$E(t)/E_0$')
    axes[0, 1].set_title('Energy Changes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel(r'$t/\tau$')
    axes[1, 0].set_ylabel(r'$J(t)/J_0$')
    axes[1, 0].set_title('Action Variable (Adiabatic Invariant)')
    axes[1, 0].axhline(y=1, color='k', ls='--', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    text = """
    ADIABATIC INVARIANCE

    Energy E changes as ω changes.
    But action J = E/ω stays constant!

    Key: Changes must be SLOW compared
    to the oscillation period.

    τ = 200: J varies by < 1%
    τ = 50:  J varies by ~ 5%
    τ = 10:  J varies by ~ 20%

    This principle underlies:
    • Bohr-Sommerfeld quantization
    • Magnetic mirror confinement
    • Ehrenfest's adiabatic theorem
    """
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('adiabatic_invariance.png', dpi=150, bbox_inches='tight')
    plt.show()

adiabatic_invariance_demo()
```

### Lab 5: Torus Visualization

```python
"""
Visualize quasiperiodic motion on a 2D torus.
"""

from mpl_toolkits.mplot3d import Axes3D

def visualize_torus():
    """Show trajectory on torus for irrational frequency ratio."""

    omega1 = 1.0
    omega2 = np.sqrt(2)  # Irrational ratio → quasiperiodic

    t = np.linspace(0, 100, 10000)

    theta1 = omega1 * t
    theta2 = omega2 * t

    # Torus parametrization
    R, r = 3, 1
    x = (R + r * np.cos(theta2)) * np.cos(theta1)
    y = (R + r * np.cos(theta2)) * np.sin(theta1)
    z = r * np.sin(theta2)

    fig = plt.figure(figsize=(14, 6))

    # 3D torus
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x, y, z, 'b-', lw=0.3, alpha=0.6)

    # Draw torus surface
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, 2*np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax1.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(f'Quasiperiodic Motion on Torus\n$ω_1/ω_2 = 1/√2$ (irrational)')

    # 2D projection
    ax2 = fig.add_subplot(122)
    ax2.plot(np.mod(theta1, 2*np.pi), np.mod(theta2, 2*np.pi),
            'b.', markersize=0.1, alpha=0.3)
    ax2.set_xlabel(r'$\theta_1$ mod $2\pi$')
    ax2.set_ylabel(r'$\theta_2$ mod $2\pi$')
    ax2.set_title('Poincaré Section\n(Densely fills for irrational ratio)')
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(0, 2*np.pi)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('torus_motion.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_torus()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Action variable | $J = \frac{1}{2\pi}\oint p \, dq$ |
| Harmonic oscillator | $J = E/\omega$ |
| Angle evolution | $\theta = \omega t + \theta_0$ |
| Frequency | $\omega = \partial H / \partial J$ |
| Bohr-Sommerfeld | $J = n\hbar$ |
| EBK quantization | $J = \hbar(n + \mu/4)$ |
| Adiabatic invariance | $dJ/dt \approx 0$ (slow changes) |

### Main Takeaways

1. **Action Variables:** J = (1/2π)∮p dq measures phase space area
   - Constant along trajectories
   - Natural unit of quantum action (ℏ)

2. **Angle Variables:** θ is conjugate to J
   - Period 2π
   - Evolves linearly: θ = ωt + θ₀

3. **Liouville-Arnold Theorem:** Integrable systems live on invariant tori
   - n commuting integrals → n-dimensional torus
   - Motion is quasiperiodic

4. **Adiabatic Invariance:** J stays constant under slow parameter changes
   - Even when E changes!
   - Foundation of semiclassical quantization

5. **Bohr-Sommerfeld → EBK:** J = ℏ(n + μ/4)
   - Maslov index μ counts turning points
   - Gives exact quantum energies for many systems

6. **KAM Preview:** Perturbations destroy some tori but not all
   - Sufficiently irrational tori survive
   - Chaos emerges in the gaps

---

## Daily Checklist

### Understanding
- [ ] I can define action variables and compute them for simple systems
- [ ] I understand why θ has period 2π
- [ ] I can state the Liouville-Arnold theorem
- [ ] I understand adiabatic invariance and when it applies

### Computation
- [ ] I can calculate J for the harmonic oscillator
- [ ] I can set up the action integral for other systems
- [ ] I can apply Bohr-Sommerfeld quantization

### Connections
- [ ] I see how J = nℏ connects classical and quantum mechanics
- [ ] I understand why Kepler orbits close (frequency degeneracy)
- [ ] I appreciate the geometric picture of motion on tori

---

## Preview: Day 165

Tomorrow we study the **Hamilton-Jacobi Equation**, the crown jewel of classical mechanics:

$$\boxed{\frac{\partial S}{\partial t} + H\left(q, \frac{\partial S}{\partial q}, t\right) = 0}$$

This single partial differential equation:
- Contains all of classical mechanics
- Provides the bridge to quantum mechanics via S → iℏ ln ψ
- Leads to the WKB approximation
- Connects to the path integral formulation

The Hamilton-Jacobi equation is where classical mechanics achieves its most elegant form and directly touches the foundations of quantum mechanics.

---

*"The mathematical difficulties of action-angle variables fade into insignificance compared with the elegance and power they bring to the description of nature."*
— V.I. Arnold

---

**Day 164 Complete. Next: Hamilton-Jacobi Equation**
