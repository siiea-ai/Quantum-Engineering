# Day 166: Introduction to Chaos — When Determinism Meets Unpredictability

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Chaos in Hamiltonian Systems |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define chaos precisely: deterministic yet unpredictable, sensitive to initial conditions
2. Calculate and interpret Lyapunov exponents as measures of chaos
3. Analyze the Chirikov standard map and the transition from integrability to chaos
4. State the KAM theorem and explain what survives perturbations
5. Use Poincaré sections to visualize and distinguish regular from chaotic motion
6. Connect classical chaos to quantum chaos signatures

---

## Core Content

### 1. What is Chaos?

**Chaos** is one of the most profound discoveries of 20th-century science. It reveals that **deterministic systems can exhibit unpredictable behavior**.

**Definition (Kellert, 1993):**
> "Chaos theory is the qualitative study of unstable aperiodic behavior in deterministic nonlinear dynamical systems."

**Key Characteristics:**

| Property | Description |
|----------|-------------|
| **Deterministic** | Future completely determined by present |
| **Unpredictable** | Long-term prediction practically impossible |
| **Sensitive dependence** | Tiny changes → exponentially diverging outcomes |
| **Bounded** | Trajectories remain in finite region |
| **Aperiodic** | Motion never exactly repeats |

**Lorenz's Insight (1963):**

> "When the present determines the future, but the approximate present does not approximately determine the future."

**The Butterfly Effect:** Edward Lorenz's famous metaphor—a butterfly's wing flap in Brazil could trigger a tornado in Texas—captures sensitive dependence on initial conditions.

---

### 2. Sensitive Dependence and the Lyapunov Exponent

**The Defining Feature of Chaos:**

Two trajectories starting at nearby points δz(0) apart evolve as:

$$|\delta \mathbf{z}(t)| \sim |\delta \mathbf{z}(0)| e^{\lambda t}$$

**Definition of the Lyapunov Exponent:**

$$\boxed{\lambda = \lim_{t \to \infty} \lim_{|\delta \mathbf{z}_0| \to 0} \frac{1}{t} \ln \frac{|\delta \mathbf{z}(t)|}{|\delta \mathbf{z}_0|}}$$

**Interpretation:**

| Lyapunov Exponent | Behavior |
|-------------------|----------|
| λ < 0 | Trajectories converge (stable) |
| λ = 0 | Neutral stability |
| λ > 0 | **Chaos!** Exponential divergence |

**The Lyapunov Time:**

$$\tau_\lambda = \frac{1}{\lambda}$$

This is the timescale for predictability. Beyond τ_λ, prediction becomes meaningless.

**Examples:**

| System | λ (approx) | τ_λ |
|--------|------------|-----|
| Weather | ~1/(3 days) | 1-2 weeks |
| Double pendulum | ~7.5 s⁻¹ | ~0.1 s |
| Solar system (inner) | ~1/(5 Myr) | 4-5 Myr |

---

### 3. The Lyapunov Spectrum

For an n-dimensional system, there are **n Lyapunov exponents** λ₁ ≥ λ₂ ≥ ... ≥ λₙ.

**Computation from the Jacobian:**

The tangent vector δ**z** evolves via:

$$\frac{d(\delta \mathbf{z})}{dt} = \mathbf{J}(t) \cdot \delta \mathbf{z}$$

where J is the Jacobian matrix of the flow.

**Properties:**

| System Type | Constraint |
|-------------|------------|
| Hamiltonian (conservative) | ∑λᵢ = 0 |
| Volume-preserving | ∑λᵢ = 0 |
| Dissipative | ∑λᵢ < 0 |

**For Hamiltonian Systems:**

Lyapunov exponents come in pairs: if λ is an exponent, so is -λ.

**Typical Spectrum:**

- (λ, 0, 0, -λ) for 2 DOF chaotic Hamiltonian system
- The zeros correspond to conservation laws (energy, etc.)

---

### 4. The Chirikov Standard Map

The **standard map** (or Chirikov-Taylor map) is the paradigm for studying the transition to chaos in Hamiltonian systems.

**Definition:**

$$\boxed{p_{n+1} = p_n + K \sin(\theta_n)}$$
$$\boxed{\theta_{n+1} = \theta_n + p_{n+1} \pmod{2\pi}}$$

**Physical Origin: The Kicked Rotor**

A freely rotating stick that receives periodic "kicks":

$$H = \frac{p^2}{2I} + K\cos\theta \sum_{n=-\infty}^{\infty} \delta(t - nT)$$

The standard map is the **Poincaré section** of this system at times t = nT.

**Properties:**

| Property | Statement |
|----------|-----------|
| Area-preserving | det(Jacobian) = 1 |
| Twist map | ∂θ_{n+1}/∂p_n ≠ 0 |
| Integrable limit | K = 0 (horizontal lines) |

---

### 5. Phase Space Structure of the Standard Map

The parameter K controls the transition from integrability to chaos.

**K = 0: Completely Integrable**

- Phase space filled with horizontal lines (p = const)
- All motion is periodic or quasiperiodic
- All Lyapunov exponents are zero

**0 < K < K_c: Mixed Phase Space**

- **Islands of stability** (KAM tori) coexist with **chaotic seas**
- Near resonances: chains of islands surrounded by chaos
- As K increases, chaos grows, islands shrink

**K ≈ K_c ≈ 0.9716: Critical Threshold**

- The **last KAM torus** is destroyed
- This torus has the **golden mean** winding number:
  $$\omega = \frac{\sqrt{5} - 1}{2} = \phi^{-1}$$
- Golden mean is the "most irrational" number—hardest to destroy

**K > K_c: Global Chaos**

- Chaotic trajectories can diffuse throughout phase space
- No barriers to transport
- Lyapunov exponent: λ ≈ ln(K/2) for large K

---

### 6. The KAM Theorem

**The Central Question:** When an integrable system is perturbed, what survives?

**Setup:** Consider a nearly-integrable Hamiltonian:

$$H = H_0(\mathbf{J}) + \epsilon H_1(\mathbf{J}, \boldsymbol{\theta})$$

**Naive Expectation:** Any perturbation destroys all invariant tori.

**The KAM Theorem (Kolmogorov-Arnold-Moser, 1954-1963):**

For **sufficiently small** ε, **most** invariant tori survive (slightly deformed), provided their frequency vectors satisfy the **Diophantine condition**.

**Diophantine Condition:**

$$\boxed{|\mathbf{k} \cdot \boldsymbol{\omega}| \geq \frac{\gamma}{|\mathbf{k}|^\tau}}$$

for all integer vectors **k** ≠ 0, where γ > 0 and τ > n - 1.

**What This Means:**
- Frequencies must be "sufficiently irrational"
- Resonant tori (**k**·**ω** = 0) are destroyed
- Near-resonant tori are destroyed
- "Most" (by Lebesgue measure) frequencies are Diophantine

**Summary:**

| Tori | Fate Under Perturbation |
|------|------------------------|
| Sufficiently irrational | **Survive** (deformed) |
| Resonant | **Destroyed** |
| Near-resonant | Destroyed |
| Golden mean | Most robust |

---

### 7. Poincaré Sections

**Definition:** A **Poincaré section** (or surface of section) reduces a continuous flow to a discrete map by recording intersections with a lower-dimensional surface.

**Construction:**

1. Choose a surface Σ transverse to the flow
2. Record (q, p) each time trajectory crosses Σ
3. Plot successive crossings

**Interpretation:**

| Pattern | Type of Motion |
|---------|----------------|
| Single point | Periodic orbit |
| Finite set of points | Higher-period periodic orbit |
| Smooth closed curve | Quasiperiodic (KAM torus) |
| Scattered points | **Chaos** |

**Example: Double Pendulum**

For the double pendulum, take the Poincaré section at θ₁ = 0 (downward crossing). Plot (θ₂, p₂) at each crossing.

- Low energy: smooth curves (regular motion)
- High energy: scattered points (chaos)
- Transition region: islands in chaotic sea

---

### 8. Historical Examples of Chaos

#### The Three-Body Problem (Poincaré, 1889-1890)

Henri Poincaré discovered chaos while competing for the King Oscar II Prize.

**Discovery:** Stable and unstable manifolds of periodic orbits **intersect transversely**, creating **homoclinic tangles**—infinitely complex structures.

**Poincaré's remark:** He refused to draw the tangle, recognizing its infinite complexity.

**Impact:** Founded the field of dynamical systems theory.

#### The Double Pendulum

A compound pendulum with two hinged segments.

**Key Properties:**
- 4-dimensional phase space (θ₁, θ₂, p₁, p₂)
- Becomes chaotic above critical energy
- Lyapunov exponent: λ ≈ 7.5 s⁻¹ experimentally
- Easy to build and demonstrate!

#### The Lorenz System (1963)

Edward Lorenz's model of atmospheric convection:

$$\dot{x} = \sigma(y - x)$$
$$\dot{y} = rx - y - xz$$
$$\dot{z} = xy - bz$$

With σ = 10, r = 28, b = 8/3:
- First strange attractor discovered
- Fractal dimension ≈ 2.06
- Positive Lyapunov exponent ≈ 0.91

**Note:** The Lorenz system is **dissipative**, not Hamiltonian. It has attractors, which Hamiltonian systems cannot have.

---

## Quantum Mechanics Connection

### The Quantum Chaos Problem

**Central Question:** What is the quantum signature of classical chaos?

**The Problem:** Schrödinger's equation is **linear**, so quantum evolution cannot have sensitive dependence on initial conditions in the classical sense!

**Resolution:** "Quantum chaos" means studying quantum systems whose **classical limits** are chaotic.

### The Ehrenfest Time

The **Ehrenfest time** τ_E is the timescale for which quantum evolution follows classical:

$$\tau_E \sim \frac{\ln(1/\hbar)}{\lambda}$$

**For chaotic systems, τ_E is logarithmically short!**

After τ_E, quantum coherence develops over scales where classical trajectories have diverged, and quantum-classical correspondence breaks down.

### Level Spacing Statistics: The BGS Conjecture

The **Bohigas-Giannoni-Schmit conjecture** (1984) connects classical dynamics to quantum energy spectra:

| Classical Dynamics | Quantum Level Statistics |
|-------------------|-------------------------|
| Integrable | **Poisson:** P(s) = e^{-s} |
| Chaotic | **Random Matrix Theory** (GOE/GUE) |

**Poisson Distribution:** Levels are uncorrelated—can cluster arbitrarily close.

**GOE (Wigner):** P(s) ≈ (πs/2)e^{-πs²/4}

The key signature: **level repulsion**. For GOE, P(0) = 0—nearby levels "repel."

**This is how we identify quantum chaos: look at the statistics of energy levels!**

### Quantum Scars

**Discovery (Eric Heller, 1984):** Some eigenstates of chaotic quantum systems show enhanced probability density along **unstable periodic orbits**.

**Significance:**
- Contradicts naive expectation of uniform filling
- Periodic orbits (measure zero classically) leave quantum imprints
- Connected to Gutzwiller's periodic orbit theory

**Recent Development (2018):** "Many-body quantum scars" found in Rydberg atom arrays—special states that resist thermalization.

---

## Worked Examples

### Example 1: Lyapunov Exponent of the Logistic Map

**Problem:** Compute the Lyapunov exponent of the logistic map x_{n+1} = rx_n(1 - x_n) for r = 4.

**Solution:**

For a 1D map x_{n+1} = f(x_n):

$$\lambda = \lim_{N \to \infty} \frac{1}{N} \sum_{i=0}^{N-1} \ln|f'(x_i)|$$

**Step 1:** Compute the derivative.

$$f(x) = rx(1-x) \implies f'(x) = r(1 - 2x)$$

**Step 2:** For r = 4, the map is fully chaotic. The invariant measure is:

$$\rho(x) = \frac{1}{\pi\sqrt{x(1-x)}}$$

**Step 3:** Compute the average.

$$\lambda = \int_0^1 \ln|4(1-2x)| \cdot \frac{dx}{\pi\sqrt{x(1-x)}}$$

Using the substitution x = sin²θ:

$$\lambda = \frac{1}{\pi}\int_0^\pi \ln|4\cos(2\theta)| \, d\theta = \ln 2$$

$$\boxed{\lambda = \ln 2 \approx 0.693}$$

**Interpretation:** Information about initial conditions is lost at rate ln 2 per iteration.

---

### Example 2: Standard Map Fixed Points

**Problem:** Find the fixed points of the standard map and determine their stability.

**Solution:**

**Fixed points satisfy:** p* = p* + K sin(θ*), θ* = θ* + p*

From the first equation: K sin(θ*) = 0 → θ* = 0 or π

From the second equation: p* = 0 (mod 2π)

**Fixed points:** (θ*, p*) = (0, 0) and (π, 0)

**Stability Analysis:**

The Jacobian is:

$$\mathbf{M} = \begin{pmatrix} 1 & K\cos\theta \\ 1 & 1 + K\cos\theta \end{pmatrix}$$

**At (0, 0):** cos(0) = 1

$$\mathbf{M} = \begin{pmatrix} 1 & K \\ 1 & 1+K \end{pmatrix}$$

Eigenvalues: λ_± = 1 + K/2 ± √(K + K²/4)

- For 0 < K < 4: |λ| < 1 for one eigenvalue → **stable** (elliptic)
- For K > 4: both |λ| > 1 → **unstable** (hyperbolic)

**At (π, 0):** cos(π) = -1

$$\mathbf{M} = \begin{pmatrix} 1 & -K \\ 1 & 1-K \end{pmatrix}$$

- For all K > 0: one |λ| > 1 → **unstable** (hyperbolic)

---

### Example 3: Estimating K_c from Golden Mean

**Problem:** Why is the golden mean winding number most stable?

**Solution:**

The golden mean φ = (√5 - 1)/2 ≈ 0.618 has the continued fraction:

$$\phi = \cfrac{1}{1 + \cfrac{1}{1 + \cfrac{1}{1 + \cdots}}}$$

**Convergents:** 1/1, 1/2, 2/3, 3/5, 5/8, 8/13, ... (Fibonacci ratios!)

**Why most stable:** The Diophantine condition requires:

$$|n\omega - m| \geq \frac{\gamma}{|n|^\tau}$$

The golden mean has the **slowest** convergence of its continued fraction, meaning the denominators grow the slowest → it's furthest from all rationals!

**Result:** The KAM torus with golden mean winding number is destroyed **last**, at K_c ≈ 0.9716.

---

## Practice Problems

### Level 1: Direct Application

1. **Lyapunov basics:** If λ = 0.5 s⁻¹, how long before an initial error of 10⁻¹⁰ grows to 1?

2. **Standard map iteration:** Starting from (θ, p) = (0.1, 0.1), compute five iterations of the standard map with K = 0.5.

3. **Fixed point stability:** For what values of K is the origin (0, 0) of the standard map a stable fixed point?

### Level 2: Intermediate

4. **Hamiltonian Lyapunov spectrum:** For a 2-DOF Hamiltonian system, if λ₁ = 0.3, what are the other three Lyapunov exponents?

5. **KAM survival:** Which winding number is more robust to perturbation: ω = 1/3 or ω = (√5 - 1)/2? Why?

6. **Poincaré section:** For the double pendulum with small energy, what shapes do you expect to see on the Poincaré section? What about high energy?

### Level 3: Challenging

7. **Standard map diffusion:** For K > K_c, the momentum p can diffuse unboundedly. Show that ⟨p²⟩ ∝ Dt for large times. What is D(K)?

8. **Quantum level repulsion:** For a 2×2 random symmetric matrix with elements drawn from N(0, σ²), derive the level spacing distribution.

9. **Arnold diffusion:** In a 3-DOF system, why can chaos allow diffusion through phase space even when most tori survive? (This doesn't happen in 2 DOF.)

---

## Computational Lab

### Lab 1: Computing Lyapunov Exponents

```python
"""
Day 166 Lab: Lyapunov Exponents and Chaos
"""

import numpy as np
import matplotlib.pyplot as plt

def lyapunov_standard_map(K, n_iterations=100000):
    """
    Compute maximal Lyapunov exponent for the standard map.
    """
    theta = np.random.random() * 2 * np.pi
    p = np.random.random() * 2 * np.pi

    # Initialize tangent vector
    v = np.array([1.0, 0.0])

    lyap_sum = 0
    for _ in range(n_iterations):
        # Jacobian at current point
        J = np.array([[1, K * np.cos(theta)],
                      [1, 1 + K * np.cos(theta)]])

        # Evolve tangent vector
        v = J @ v
        norm_v = np.linalg.norm(v)
        lyap_sum += np.log(norm_v)
        v = v / norm_v

        # Evolve the map
        p = p + K * np.sin(theta)
        theta = (theta + p) % (2 * np.pi)

    return lyap_sum / n_iterations


# Compute λ vs K
K_values = np.linspace(0, 5, 100)
lyapunov_values = [lyapunov_standard_map(K) for K in K_values]

plt.figure(figsize=(10, 6))
plt.plot(K_values, lyapunov_values, 'b-', lw=2)
plt.axhline(y=0, color='k', ls='--', alpha=0.5)
plt.axvline(x=0.9716, color='r', ls='--', alpha=0.5, label='K_c ≈ 0.9716')
plt.xlabel('K', fontsize=14)
plt.ylabel('λ (Lyapunov exponent)', fontsize=14)
plt.title('Standard Map: Lyapunov Exponent vs K', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lyapunov_vs_K.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"λ at K=2: {lyapunov_standard_map(2):.4f}")
print(f"Theory (ln(K/2) for large K): {np.log(2/2):.4f}")
```

### Lab 2: Standard Map Phase Space

```python
"""
Visualize the standard map phase space for different K values.
"""

def iterate_standard_map(theta0, p0, K, n_iter):
    """Iterate the standard map."""
    theta, p = theta0, p0
    thetas, ps = [theta], [p]

    for _ in range(n_iter):
        p = p + K * np.sin(theta)
        theta = (theta + p) % (2 * np.pi)
        p = p % (2 * np.pi)
        thetas.append(theta)
        ps.append(p)

    return np.array(thetas), np.array(ps)


def plot_standard_map(K_values, n_orbits=50, n_iter=500):
    """Plot phase space for different K."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, K in zip(axes, K_values):
        np.random.seed(42)

        for _ in range(n_orbits):
            theta0 = np.random.random() * 2 * np.pi
            p0 = np.random.random() * 2 * np.pi
            theta, p = iterate_standard_map(theta0, p0, K, n_iter)
            ax.plot(theta, p, ',', markersize=0.5, alpha=0.5)

        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('p')
        ax.set_title(f'K = {K}')

        if K < 0.9716:
            ax.text(0.05, 0.95, 'KAM tori exist', transform=ax.transAxes,
                   fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        elif K > 1.5:
            ax.text(0.05, 0.95, 'Global chaos', transform=ax.transAxes,
                   fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.suptitle('Chirikov Standard Map: Transition to Chaos', fontsize=14)
    plt.tight_layout()
    plt.savefig('standard_map_phases.png', dpi=150, bbox_inches='tight')
    plt.show()


K_values = [0.2, 0.5, 0.9, 0.9716, 2.0, 5.0]
plot_standard_map(K_values)
```

### Lab 3: Double Pendulum Chaos

```python
"""
Double pendulum: demonstrating chaos in a physical system.
"""

from scipy.integrate import solve_ivp

def double_pendulum(t, state, L1=1, L2=1, m1=1, m2=1, g=9.81):
    """Equations of motion for double pendulum."""
    theta1, theta2, omega1, omega2 = state

    delta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    domega1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
               m2 * g * np.sin(theta2) * np.cos(delta) +
               m2 * L2 * omega2**2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(theta1)) / den1

    domega2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
               (m1 + m2) * (g * np.sin(theta1) * np.cos(delta) -
                           L1 * omega1**2 * np.sin(delta) -
                           g * np.sin(theta2))) / den2

    return [omega1, omega2, domega1, domega2]


def sensitive_dependence_demo():
    """Show sensitive dependence on initial conditions."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    T = 10
    t_eval = np.linspace(0, T, 2000)

    # Initial conditions with tiny differences
    theta1_0 = np.pi/2
    epsilons = [0, 1e-10, 1e-8]
    colors = ['blue', 'red', 'green']

    trajectories = []
    for eps in epsilons:
        state0 = [theta1_0 + eps, np.pi/2, 0, 0]
        sol = solve_ivp(double_pendulum, [0, T], state0,
                       t_eval=t_eval, method='RK45', max_step=0.01)
        trajectories.append(sol)

    # θ₁ vs time
    ax = axes[0]
    for sol, eps, color in zip(trajectories, epsilons, colors):
        label = f'ε = {eps:.0e}' if eps > 0 else 'Reference'
        ax.plot(sol.t, sol.y[0], color=color, alpha=0.7, label=label)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\theta_1$ (rad)')
    ax.set_title('Sensitive Dependence on Initial Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Divergence
    ax = axes[1]
    ref = trajectories[0].y[0]
    for sol, eps, color in zip(trajectories[1:], epsilons[1:], colors[1:]):
        diff = np.abs(sol.y[0] - ref)
        ax.semilogy(sol.t, diff + 1e-16, color=color, label=f'ε = {eps:.0e}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$|\Delta\theta_1|$')
    ax.set_title('Exponential Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tip trajectory
    ax = axes[2]
    L1, L2 = 1, 1

    for sol, color in zip(trajectories, colors):
        x2 = L1*np.sin(sol.y[0]) + L2*np.sin(sol.y[1])
        y2 = -L1*np.cos(sol.y[0]) - L2*np.cos(sol.y[1])
        ax.plot(x2, y2, color=color, alpha=0.5, lw=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Tip Trajectories Diverge')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('double_pendulum_chaos.png', dpi=150, bbox_inches='tight')
    plt.show()

sensitive_dependence_demo()
```

### Lab 4: Level Spacing Statistics

```python
"""
Quantum chaos signature: level spacing statistics.
"""

def poisson_spacing(s):
    """Poisson: P(s) = exp(-s)"""
    return np.exp(-s)

def goe_spacing(s):
    """Wigner surmise for GOE: P(s) = (π/2)s exp(-πs²/4)"""
    return (np.pi/2) * s * np.exp(-np.pi * s**2 / 4)

def random_goe_matrix(N):
    """Generate GOE random matrix."""
    H = np.random.randn(N, N)
    return (H + H.T) / np.sqrt(2)

def compute_spacings(eigenvalues):
    """Compute normalized nearest-neighbor spacings."""
    eigenvalues = np.sort(eigenvalues)
    spacings = np.diff(eigenvalues)
    return spacings / np.mean(spacings)

def level_spacing_demo():
    """Compare Poisson vs GOE statistics."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    N = 500
    n_matrices = 100

    # Poisson (uncorrelated)
    ax = axes[0]
    all_spacings = []
    for _ in range(n_matrices):
        levels = np.sort(np.random.random(N) * N)
        spacings = compute_spacings(levels)
        all_spacings.extend(spacings)

    ax.hist(all_spacings, bins=50, density=True, alpha=0.7, label='Numerical')
    s = np.linspace(0, 4, 100)
    ax.plot(s, poisson_spacing(s), 'r-', lw=2, label=r'$P(s) = e^{-s}$')
    ax.set_xlabel('s')
    ax.set_ylabel('P(s)')
    ax.set_title('Poisson (Integrable)\nLevel clustering allowed')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # GOE (chaotic)
    ax = axes[1]
    all_spacings = []
    for _ in range(n_matrices):
        H = random_goe_matrix(N)
        eigenvalues = np.linalg.eigvalsh(H)
        spacings = compute_spacings(eigenvalues)
        all_spacings.extend(spacings)

    ax.hist(all_spacings, bins=50, density=True, alpha=0.7, label='Numerical')
    ax.plot(s, goe_spacing(s), 'r-', lw=2,
            label=r'$P(s) = \frac{\pi s}{2}e^{-\pi s^2/4}$')
    ax.set_xlabel('s')
    ax.set_ylabel('P(s)')
    ax.set_title('GOE (Chaotic)\nLevel repulsion: P(0) = 0')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # Comparison
    ax = axes[2]
    ax.plot(s, poisson_spacing(s), 'b-', lw=2, label='Poisson (Integrable)')
    ax.plot(s, goe_spacing(s), 'r-', lw=2, label='GOE (Chaotic)')
    ax.fill_between(s, 0, goe_spacing(s), alpha=0.2, color='red')
    ax.annotate('Level repulsion\nP(0) = 0', xy=(0.3, 0.1), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.set_xlabel('s')
    ax.set_ylabel('P(s)')
    ax.set_title('BGS Conjecture:\nClassical chaos → Level repulsion')
    ax.legend()
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Level Spacing Statistics: The Quantum Signature of Chaos', fontsize=14)
    plt.tight_layout()
    plt.savefig('level_spacing.png', dpi=150, bbox_inches='tight')
    plt.show()

level_spacing_demo()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Lyapunov exponent | $\lambda = \lim_{t\to\infty} \frac{1}{t}\ln\frac{|\delta z(t)|}{|\delta z(0)|}$ |
| Chaos criterion | λ > 0 |
| Lyapunov time | τ_λ = 1/λ |
| Standard map | p' = p + K sin θ, θ' = θ + p' |
| Critical K | K_c ≈ 0.9716 |
| Large K Lyapunov | λ ≈ ln(K/2) |
| Diophantine condition | \|**k**·**ω**\| ≥ γ/\|**k**\|^τ |
| Poisson spacing | P(s) = e^{-s} |
| GOE spacing | P(s) = (πs/2)e^{-πs²/4} |

### Main Takeaways

1. **Chaos Defined:** Deterministic + unpredictable + sensitive dependence
   - λ > 0 is the mathematical criterion
   - Predictability limited to Lyapunov time τ_λ

2. **Standard Map:** Paradigm for Hamiltonian chaos
   - K = 0: integrable
   - K < K_c: mixed (islands + chaos)
   - K > K_c: global chaos

3. **KAM Theorem:** Most tori survive small perturbations
   - Diophantine (sufficiently irrational) frequencies survive
   - Resonant and near-resonant tori are destroyed
   - Golden mean is most robust

4. **Poincaré Sections:** Visualize chaos
   - Curves = regular motion
   - Scattered points = chaos

5. **Quantum Chaos:**
   - Level repulsion (GOE vs Poisson) is the signature
   - Quantum scars along unstable periodic orbits
   - Ehrenfest time limits quantum-classical correspondence

---

## Daily Checklist

### Understanding
- [ ] I can define chaos and explain sensitive dependence
- [ ] I understand what Lyapunov exponents measure
- [ ] I can describe the phase space structure of the standard map
- [ ] I can state the KAM theorem and its implications

### Computation
- [ ] I can compute Lyapunov exponents numerically
- [ ] I can iterate the standard map and plot phase space
- [ ] I can identify chaos in Poincaré sections

### Connections
- [ ] I understand why weather prediction has fundamental limits
- [ ] I can explain the quantum signatures of classical chaos
- [ ] I appreciate how integrable and chaotic systems differ

---

## Preview: Day 167

Tomorrow is the **Computational Lab** for Week 24, where we'll implement comprehensive numerical tools:

- Symplectic integrators for long-time Hamiltonian simulation
- Hamilton-Jacobi equation solvers
- Chaos detection and visualization
- Poincaré section generation
- Lyapunov exponent computation

This will consolidate all the computational methods from the week.

---

*"Chaos: When the present determines the future, but the approximate present does not approximately determine the future."*
— Edward Lorenz

---

**Day 166 Complete. Next: Computational Lab**
