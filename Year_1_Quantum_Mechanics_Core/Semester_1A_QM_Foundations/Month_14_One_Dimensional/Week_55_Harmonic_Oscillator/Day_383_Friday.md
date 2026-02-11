# Day 383: Coherent States — The Most Classical Quantum States

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Coherent State Properties |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 383, you will be able to:

1. Define coherent states as eigenstates of the annihilation operator
2. Expand coherent states in the Fock basis with Poissonian statistics
3. Prove that coherent states are minimum uncertainty states
4. Construct coherent states using the displacement operator
5. Analyze the time evolution of coherent states
6. Connect coherent states to laser light and quantum optics

---

## Core Content

### 1. Definition of Coherent States

Coherent states $|\alpha\rangle$ are defined as **eigenstates of the annihilation operator**:

$$\boxed{\hat{a}|\alpha\rangle = \alpha|\alpha\rangle}$$

where $\alpha \in \mathbb{C}$ is a complex number.

**Why is this remarkable?**
- $\hat{a}$ is not Hermitian, so its eigenstates need not be orthogonal
- $\alpha$ can be any complex number (continuous spectrum)
- Coherent states form an **overcomplete** basis

**Physical interpretation:** $\alpha$ encodes both the "position" and "momentum" of a classical oscillator:
$$\alpha = \frac{1}{\sqrt{2}}\left(\xi_0 + i\eta_0\right)$$
where $\xi_0 = \langle x\rangle/x_0$ and $\eta_0 = \langle p\rangle/p_0$ are dimensionless means.

---

### 2. Expansion in Fock Basis

**Theorem:** The coherent state expanded in number states is:

$$\boxed{|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle}$$

**Proof:**

Let $|\alpha\rangle = \sum_{n=0}^{\infty}c_n|n\rangle$. Applying $\hat{a}$:
$$\hat{a}|\alpha\rangle = \sum_{n=0}^{\infty}c_n\hat{a}|n\rangle = \sum_{n=1}^{\infty}c_n\sqrt{n}|n-1\rangle = \sum_{n=0}^{\infty}c_{n+1}\sqrt{n+1}|n\rangle$$

Setting this equal to $\alpha|\alpha\rangle = \sum_{n}c_n\alpha|n\rangle$:
$$c_{n+1}\sqrt{n+1} = \alpha c_n$$

Solving the recurrence:
$$c_n = \frac{\alpha^n}{\sqrt{n!}}c_0$$

Normalization: $\langle\alpha|\alpha\rangle = 1$ requires:
$$|c_0|^2\sum_{n=0}^{\infty}\frac{|\alpha|^{2n}}{n!} = |c_0|^2 e^{|\alpha|^2} = 1$$

So $c_0 = e^{-|\alpha|^2/2}$. $\blacksquare$

---

### 3. Poisson Distribution

The probability of measuring $n$ photons in a coherent state is:

$$P(n) = |\langle n|\alpha\rangle|^2 = e^{-|\alpha|^2}\frac{|\alpha|^{2n}}{n!}$$

This is a **Poisson distribution** with mean $\bar{n} = |\alpha|^2$!

$$\boxed{P(n) = e^{-\bar{n}}\frac{\bar{n}^n}{n!}, \quad \bar{n} = |\alpha|^2}$$

| Quantity | Value |
|----------|-------|
| Mean photon number | $\langle\hat{N}\rangle = |\alpha|^2$ |
| Variance | $(\Delta N)^2 = |\alpha|^2$ |
| Standard deviation | $\Delta N = |\alpha|$ |
| Mandel Q parameter | $Q = \frac{(\Delta N)^2 - \langle N\rangle}{\langle N\rangle} = 0$ (Poissonian) |

**Physical significance:**
- Fock states have $\Delta N = 0$ (number-squeezed)
- Thermal states have $Q > 0$ (super-Poissonian)
- Coherent states are exactly Poissonian ($Q = 0$)

---

### 4. Expectation Values and Uncertainty

#### Position and Momentum Means

Using $\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$:

$$\langle\alpha|\hat{x}|\alpha\rangle = \sqrt{\frac{\hbar}{2m\omega}}(\alpha + \alpha^*) = \sqrt{\frac{2\hbar}{m\omega}}\text{Re}(\alpha)$$

Similarly:
$$\langle\alpha|\hat{p}|\alpha\rangle = \sqrt{\frac{m\omega\hbar}{2}}(\alpha^* - \alpha) \cdot i = \sqrt{2m\omega\hbar}\text{Im}(\alpha)$$

**Summary:** If $\alpha = |\alpha|e^{i\phi}$:
$$\boxed{\langle x\rangle = \sqrt{\frac{2\hbar}{m\omega}}|\alpha|\cos\phi, \quad \langle p\rangle = \sqrt{2m\omega\hbar}|\alpha|\sin\phi}$$

#### Uncertainties

For **any** coherent state:
$$\langle\hat{x}^2\rangle - \langle\hat{x}\rangle^2 = \frac{\hbar}{2m\omega}$$
$$\langle\hat{p}^2\rangle - \langle\hat{p}\rangle^2 = \frac{m\omega\hbar}{2}$$

Therefore:
$$\boxed{\Delta x \cdot \Delta p = \frac{\hbar}{2}}$$

**Coherent states saturate the uncertainty principle!** They are **minimum uncertainty states**.

---

### 5. The Displacement Operator

Coherent states can be constructed from the vacuum using the **displacement operator**:

$$\boxed{\hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}}$$

**Theorem:** $|\alpha\rangle = \hat{D}(\alpha)|0\rangle$

**Properties of $\hat{D}(\alpha)$:**

| Property | Formula |
|----------|---------|
| Unitarity | $\hat{D}^\dagger(\alpha) = \hat{D}(-\alpha) = \hat{D}^{-1}(\alpha)$ |
| Composition | $\hat{D}(\alpha)\hat{D}(\beta) = e^{i\text{Im}(\alpha\beta^*)}\hat{D}(\alpha + \beta)$ |
| Action on $\hat{a}$ | $\hat{D}^\dagger(\alpha)\hat{a}\hat{D}(\alpha) = \hat{a} + \alpha$ |
| Action on $\hat{a}^\dagger$ | $\hat{D}^\dagger(\alpha)\hat{a}^\dagger\hat{D}(\alpha) = \hat{a}^\dagger + \alpha^*$ |

**Physical interpretation:** $\hat{D}(\alpha)$ "displaces" the vacuum state in phase space by $\alpha$.

#### Derivation using BCH Formula

Using the Baker-Campbell-Hausdorff formula for $[\hat{a}, \hat{a}^\dagger] = 1$:
$$\hat{D}(\alpha) = e^{-|\alpha|^2/2}e^{\alpha\hat{a}^\dagger}e^{-\alpha^*\hat{a}}$$

Acting on vacuum:
$$\hat{D}(\alpha)|0\rangle = e^{-|\alpha|^2/2}e^{\alpha\hat{a}^\dagger}e^{-\alpha^*\hat{a}}|0\rangle = e^{-|\alpha|^2/2}e^{\alpha\hat{a}^\dagger}|0\rangle$$

Using $e^{\alpha\hat{a}^\dagger}|0\rangle = \sum_n\frac{\alpha^n}{n!}(\hat{a}^\dagger)^n|0\rangle = \sum_n\frac{\alpha^n}{\sqrt{n!}}|n\rangle$:

$$= e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle = |\alpha\rangle$$ ✓

---

### 6. Time Evolution

How does a coherent state evolve under the QHO Hamiltonian?

$$|\alpha(t)\rangle = e^{-i\hat{H}t/\hbar}|\alpha\rangle = e^{-i\omega t(\hat{N} + 1/2)}|\alpha\rangle$$

Using the Fock expansion:
$$|\alpha(t)\rangle = e^{-i\omega t/2}\sum_n e^{-|\alpha|^2/2}\frac{\alpha^n}{\sqrt{n!}}e^{-in\omega t}|n\rangle$$

$$= e^{-i\omega t/2}\sum_n e^{-|\alpha|^2/2}\frac{(\alpha e^{-i\omega t})^n}{\sqrt{n!}}|n\rangle$$

$$\boxed{|\alpha(t)\rangle = e^{-i\omega t/2}|\alpha e^{-i\omega t}\rangle}$$

**The coherent state remains a coherent state!** Only its label rotates:
$$\alpha \to \alpha(t) = \alpha e^{-i\omega t}$$

**In phase space:** The center of the wave packet traces out a **classical ellipse**!

$$\langle x\rangle(t) = \sqrt{\frac{2\hbar}{m\omega}}|\alpha|\cos(\omega t - \phi)$$
$$\langle p\rangle(t) = -\sqrt{2m\omega\hbar}|\alpha|\sin(\omega t - \phi)$$

This is exactly classical motion!

---

### 7. Non-Orthogonality and Overcompleteness

#### Non-Orthogonality

Coherent states are **not orthogonal**:
$$\langle\beta|\alpha\rangle = e^{-|\alpha|^2/2}e^{-|\beta|^2/2}\sum_{n}\frac{(\beta^*)^n\alpha^n}{n!} = e^{-|\alpha|^2/2}e^{-|\beta|^2/2}e^{\beta^*\alpha}$$

$$\boxed{|\langle\beta|\alpha\rangle|^2 = e^{-|\alpha - \beta|^2}}$$

The overlap is significant when $|\alpha - \beta| \lesssim 1$.

#### Overcompleteness (Resolution of Identity)

Despite non-orthogonality, coherent states satisfy:
$$\boxed{\frac{1}{\pi}\int d^2\alpha\,|\alpha\rangle\langle\alpha| = \hat{I}}$$

where $d^2\alpha = d(\text{Re}\alpha)d(\text{Im}\alpha)$.

**Consequence:** Any state can be expanded in coherent states, though the expansion is not unique.

---

### 8. Quantum Computing Connection: Quantum Optics & Bosonic Codes

#### Laser Light

Laser light is well-approximated by a coherent state:
- Poissonian photon statistics
- Minimum uncertainty
- Classical oscillating electric field

| Light Source | Photon Statistics | $Q$ Parameter |
|--------------|-------------------|---------------|
| Thermal (incandescent) | Super-Poissonian | $Q > 0$ |
| Coherent (laser) | Poissonian | $Q = 0$ |
| Squeezed | Sub-Poissonian | $Q < 0$ |
| Fock state $|n\rangle$ | Definite number | $Q = -1$ |

#### Cat States

Superpositions of coherent states are called **cat states**:
$$|\text{cat}_\pm\rangle = \mathcal{N}(|\alpha\rangle \pm |-\alpha\rangle)$$

These are:
- **Even cat:** $|\text{cat}_+\rangle$ (even photon numbers only)
- **Odd cat:** $|\text{cat}_-\rangle$ (odd photon numbers only)

Cat states are used in:
- **Bosonic qubits:** Logical $|0_L\rangle = |\text{cat}_+\rangle$, $|1_L\rangle = |\text{cat}_-\rangle$
- **Quantum error correction:** Protected against photon loss
- **Fundamental tests:** Schrodinger's cat realized in the lab!

#### Continuous-Variable Quantum Computing

Coherent states are fundamental to CV quantum computing:
- **Gaussian operations:** Displacements, squeezing, rotations
- **Homodyne detection:** Measures field quadratures
- **Heterodyne detection:** Measures both quadratures (at a cost)
- **Boson sampling:** Coherent states interfered in linear optics

---

## Worked Examples

### Example 1: Verifying the Eigenvalue Equation

**Problem:** Verify that $|\alpha\rangle = e^{-|\alpha|^2/2}\sum_n\frac{\alpha^n}{\sqrt{n!}}|n\rangle$ satisfies $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$.

**Solution:**

$$\hat{a}|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}\hat{a}|n\rangle$$

Using $\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$:
$$= e^{-|\alpha|^2/2}\sum_{n=1}^{\infty}\frac{\alpha^n}{\sqrt{n!}}\sqrt{n}|n-1\rangle$$

$$= e^{-|\alpha|^2/2}\sum_{n=1}^{\infty}\frac{\alpha^n}{\sqrt{(n-1)!}}|n-1\rangle$$

Let $m = n-1$:
$$= e^{-|\alpha|^2/2}\sum_{m=0}^{\infty}\frac{\alpha^{m+1}}{\sqrt{m!}}|m\rangle = \alpha \cdot e^{-|\alpha|^2/2}\sum_{m=0}^{\infty}\frac{\alpha^m}{\sqrt{m!}}|m\rangle = \alpha|\alpha\rangle$$

$$\boxed{\hat{a}|\alpha\rangle = \alpha|\alpha\rangle}$$ ✓ $\blacksquare$

---

### Example 2: Photon Number Statistics

**Problem:** A coherent state has $|\alpha|^2 = 4$ (mean of 4 photons). Find:
(a) The probability of detecting 0, 1, 2, 3, 4, 5 photons
(b) The most likely photon number

**Solution:**

(a) Using $P(n) = e^{-\bar{n}}\frac{\bar{n}^n}{n!}$ with $\bar{n} = 4$:

| n | $P(n) = e^{-4}\frac{4^n}{n!}$ | Value |
|---|------------------------------|-------|
| 0 | $e^{-4}$ | 0.0183 |
| 1 | $4e^{-4}$ | 0.0733 |
| 2 | $8e^{-4}$ | 0.1465 |
| 3 | $\frac{32}{3}e^{-4}$ | 0.1954 |
| 4 | $\frac{32}{3}e^{-4}$ | 0.1954 |
| 5 | $\frac{128}{15}e^{-4}$ | 0.1563 |

(b) For a Poisson distribution with $\bar{n} = 4$:
- Mode (most likely) = $\lfloor\bar{n}\rfloor$ or $\lfloor\bar{n}\rfloor + 1$ when $\bar{n}$ is an integer
- Here: $n = 3$ and $n = 4$ are equally likely (both ~19.5%)

$$\boxed{\text{Most likely: } n = 3 \text{ or } n = 4}$$ $\blacksquare$

---

### Example 3: Time Evolution of Expectation Values

**Problem:** A coherent state $|\alpha_0\rangle$ with $\alpha_0 = 2$ (real) is prepared at $t = 0$. Find $\langle x\rangle(t)$ and $\langle p\rangle(t)$.

**Solution:**

At time $t$: $\alpha(t) = \alpha_0 e^{-i\omega t} = 2e^{-i\omega t} = 2(\cos\omega t - i\sin\omega t)$

Position expectation:
$$\langle x\rangle(t) = \sqrt{\frac{2\hbar}{m\omega}}\text{Re}(\alpha(t)) = \sqrt{\frac{2\hbar}{m\omega}} \cdot 2\cos\omega t$$
$$\boxed{\langle x\rangle(t) = 2\sqrt{\frac{2\hbar}{m\omega}}\cos\omega t}$$

Momentum expectation:
$$\langle p\rangle(t) = \sqrt{2m\omega\hbar}\text{Im}(\alpha(t)) = \sqrt{2m\omega\hbar} \cdot (-2\sin\omega t)$$
$$\boxed{\langle p\rangle(t) = -2\sqrt{2m\omega\hbar}\sin\omega t}$$

These are exactly the classical equations of motion! $\blacksquare$

---

## Practice Problems

### Level 1: Direct Application

1. **Normalization:** Verify that $\langle\alpha|\alpha\rangle = 1$ for the coherent state.

2. **Photon Statistics:** For $|\alpha|^2 = 1$, calculate $P(0)$, $P(1)$, $P(2)$, $P(3)$.

3. **Displacement Operator:** Show that $\hat{D}(0) = \hat{I}$ and $\hat{D}^\dagger(\alpha) = \hat{D}(-\alpha)$.

### Level 2: Intermediate

4. **Variance of Photon Number:** Show that $\langle\hat{N}^2\rangle - \langle\hat{N}\rangle^2 = |\alpha|^2$ for a coherent state.

5. **Phase Space Trajectory:** For $\alpha(0) = 1 + i$, find $|\alpha(t)|$ and $\arg(\alpha(t))$ as functions of time. Sketch the trajectory in the complex $\alpha$-plane.

6. **Overlap Calculation:** Calculate $|\langle\alpha|\beta\rangle|^2$ for $\alpha = 1$ and $\beta = 1 + i$. When are two coherent states approximately orthogonal?

### Level 3: Challenging

7. **Cat State Normalization:** For the even cat state $|\text{cat}_+\rangle = \mathcal{N}(|\alpha\rangle + |-\alpha\rangle)$:
   (a) Find the normalization constant $\mathcal{N}$.
   (b) Compute the photon number distribution $P(n)$.
   (c) Show that only even $n$ contribute.

8. **Squeezing Connection:** The squeezed vacuum state is $|\zeta\rangle = \hat{S}(\zeta)|0\rangle$ where $\hat{S}(\zeta) = e^{(\zeta^*\hat{a}^2 - \zeta(\hat{a}^\dagger)^2)/2}$. How does the uncertainty product $\Delta x \cdot \Delta p$ compare to that of a coherent state?

9. **P-Representation:** The Glauber-Sudarshan P-representation expresses the density matrix as:
   $$\hat{\rho} = \int P(\alpha)|\alpha\rangle\langle\alpha|d^2\alpha$$
   Find $P(\alpha)$ for:
   (a) A coherent state $|\beta\rangle\langle\beta|$
   (b) A Fock state $|n\rangle\langle n|$
   Why is the latter problematic?

---

## Computational Lab

### Objective
Construct coherent states, visualize their properties, and animate their time evolution.

```python
"""
Day 383 Computational Lab: Coherent States
Quantum Harmonic Oscillator - Week 55
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.stats import poisson
import matplotlib.animation as animation

# =============================================================================
# Part 1: Coherent State Construction
# =============================================================================

print("=" * 70)
print("Part 1: Coherent State in Fock Basis")
print("=" * 70)

def coherent_state_fock(alpha, N_max):
    """
    Coherent state |α⟩ in Fock basis.

    |α⟩ = e^{-|α|²/2} Σ (α^n/√n!) |n⟩
    """
    n = np.arange(N_max)
    coefficients = np.exp(-np.abs(alpha)**2 / 2) * alpha**n / np.sqrt(factorial(n))
    return coefficients

# Create a coherent state with α = 2
alpha = 2.0
N_max = 20
coeffs = coherent_state_fock(alpha, N_max)

print(f"\nCoherent state with α = {alpha}")
print(f"Mean photon number: |α|² = {np.abs(alpha)**2}")
print(f"\nFock state expansion coefficients (first 10):")
for n in range(10):
    print(f"  c_{n} = {coeffs[n]:.6f}, |c_{n}|² = {np.abs(coeffs[n])**2:.6f}")

# Verify normalization
norm = np.sum(np.abs(coeffs)**2)
print(f"\nNormalization: Σ|c_n|² = {norm:.6f}")

# =============================================================================
# Part 2: Poisson Distribution
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Photon Number Statistics (Poisson Distribution)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Photon number distribution for various α
ax = axes[0]
alpha_values = [1.0, 2.0, 3.0, 5.0]
n_range = np.arange(20)

for alpha in alpha_values:
    probs = np.abs(coherent_state_fock(alpha, 20))**2
    ax.bar(n_range + (alpha_values.index(alpha) - 1.5)*0.2, probs, width=0.2,
           label=f'α = {alpha}, ⟨N⟩ = {np.abs(alpha)**2}', alpha=0.7)

ax.set_xlabel('Photon number n', fontsize=12)
ax.set_ylabel('Probability P(n)', fontsize=12)
ax.set_title('Coherent State Photon Statistics', fontsize=14)
ax.legend()
ax.set_xlim(-0.5, 15)

# Compare with Poisson distribution
ax = axes[1]
alpha = 4.0
n_bar = np.abs(alpha)**2
n_range = np.arange(20)

# Quantum calculation
probs_quantum = np.abs(coherent_state_fock(alpha, 20))**2

# Poisson distribution
probs_poisson = poisson.pmf(n_range, n_bar)

ax.bar(n_range - 0.2, probs_quantum, width=0.4, label='|⟨n|α⟩|²', alpha=0.7, color='blue')
ax.bar(n_range + 0.2, probs_poisson, width=0.4, label='Poisson(n̄=16)', alpha=0.7, color='red')
ax.set_xlabel('Photon number n', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title(f'Coherent State (α={alpha}) vs Poisson Distribution', fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig('day_383_poisson_stats.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFor α = {alpha} (n̄ = {n_bar}):")
print(f"  Variance from quantum: {np.sum(n_range**2 * probs_quantum) - n_bar**2:.4f}")
print(f"  Expected (Poissonian): {n_bar}")

# =============================================================================
# Part 3: Coherent State Wave Function
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Coherent State Wave Function in Position Space")
print("=" * 70)

def hermite(n, x):
    """Compute Hermite polynomial H_n(x) using recurrence"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_prev = np.ones_like(x)
        H_curr = 2 * x
        for k in range(2, n + 1):
            H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
            H_prev = H_curr
            H_curr = H_next
        return H_curr

def psi_n(n, xi):
    """QHO wave function ψ_n(ξ)"""
    norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return norm * hermite(n, xi) * np.exp(-xi**2 / 2)

def coherent_wavefunction(alpha, xi, N_max=30):
    """
    Coherent state wave function ⟨ξ|α⟩
    """
    coeffs = coherent_state_fock(alpha, N_max)
    psi = np.zeros_like(xi, dtype=complex)
    for n in range(N_max):
        psi += coeffs[n] * psi_n(n, xi)
    return psi

# Plot coherent state wave functions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

xi = np.linspace(-8, 8, 500)

# Different α values
alpha_list = [0, 1, 2, 3]

for idx, alpha in enumerate(alpha_list):
    ax = axes[idx // 2, idx % 2]

    psi = coherent_wavefunction(alpha, xi)
    prob = np.abs(psi)**2

    ax.plot(xi, prob, 'b-', linewidth=2, label='|ψ(ξ)|²')
    ax.fill_between(xi, 0, prob, alpha=0.3, color='blue')

    # Mark the center ⟨x⟩ = √2 Re(α)
    xi_mean = np.sqrt(2) * np.real(alpha)
    ax.axvline(xi_mean, color='red', linestyle='--', label=f'⟨ξ⟩ = {xi_mean:.2f}')

    # Ground state for comparison
    psi_0 = np.abs(psi_n(0, xi - xi_mean))**2
    ax.plot(xi, psi_0, 'g--', linewidth=1, alpha=0.5, label='Displaced ground state')

    ax.set_xlabel('ξ', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title(f'α = {alpha}', fontsize=12)
    ax.legend()
    ax.set_xlim(-8, 8)

fig.suptitle('Coherent State Probability Densities', fontsize=14)
plt.tight_layout()
plt.savefig('day_383_coherent_wavefunctions.png', dpi=150, bbox_inches='tight')
plt.show()

print("Wave function plots saved.")

# =============================================================================
# Part 4: Minimum Uncertainty Verification
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Minimum Uncertainty Verification")
print("=" * 70)

def compute_uncertainty(alpha, N_max=50):
    """
    Compute Δx·Δp for coherent state |α⟩.
    In natural units (ℏ = m = ω = 1).
    """
    coeffs = coherent_state_fock(alpha, N_max)
    n = np.arange(N_max)

    # Build matrices
    from scipy.linalg import eigh

    # x = (a + a†)/√2, p = i(a† - a)/√2
    # ⟨x⟩ = √2 Re(α), ⟨p⟩ = √2 Im(α)

    # ⟨x²⟩ = ⟨(a + a†)²⟩/2 = ⟨a² + aa† + a†a + (a†)²⟩/2
    # For coherent state: ⟨a²⟩ = α², ⟨(a†)²⟩ = (α*)², ⟨aa†⟩ = |α|² + 1, ⟨a†a⟩ = |α|²

    a_exp = alpha
    a_dag_exp = np.conj(alpha)
    N_exp = np.abs(alpha)**2

    x_exp = np.sqrt(2) * np.real(alpha)
    p_exp = np.sqrt(2) * np.imag(alpha)

    x2_exp = (alpha**2 + np.conj(alpha)**2 + 2*N_exp + 1) / 2
    p2_exp = (-alpha**2 - np.conj(alpha)**2 + 2*N_exp + 1) / 2

    Delta_x = np.sqrt(np.real(x2_exp) - x_exp**2)
    Delta_p = np.sqrt(np.real(p2_exp) - p_exp**2)

    return Delta_x, Delta_p, Delta_x * Delta_p

print("\nUncertainty product for various α:")
print("-" * 50)
for alpha in [0, 1, 1+1j, 2, 3, 5]:
    dx, dp, product = compute_uncertainty(alpha)
    print(f"α = {alpha:6}: Δx = {dx:.4f}, Δp = {dp:.4f}, Δx·Δp = {product:.4f} (ℏ/2 = 0.5)")

# =============================================================================
# Part 5: Time Evolution
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Time Evolution of Coherent States")
print("=" * 70)

# Phase space trajectory
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trajectory in α-plane
ax = axes[0]
alpha_0 = 2 + 1j
omega = 1
t = np.linspace(0, 2*np.pi, 100)

alpha_t = alpha_0 * np.exp(-1j * omega * t)

ax.plot(np.real(alpha_t), np.imag(alpha_t), 'b-', linewidth=2, label='Trajectory')
ax.plot(np.real(alpha_0), np.imag(alpha_0), 'go', markersize=12, label=f'α(0) = {alpha_0}')
ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)

# Mark several times
for i, tau in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
    alpha_tau = alpha_0 * np.exp(-1j * omega * tau)
    ax.plot(np.real(alpha_tau), np.imag(alpha_tau), 'ro', markersize=8)
    ax.annotate(f't={tau/np.pi:.1f}π', (np.real(alpha_tau)+0.1, np.imag(alpha_tau)+0.1))

ax.set_xlabel('Re(α)', fontsize=12)
ax.set_ylabel('Im(α)', fontsize=12)
ax.set_title('Phase Space Trajectory', fontsize=14)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Position and momentum vs time
ax = axes[1]
x_exp = np.sqrt(2) * np.real(alpha_t)
p_exp = np.sqrt(2) * np.imag(alpha_t)

ax.plot(t/np.pi, x_exp, 'b-', linewidth=2, label='⟨x⟩/x₀')
ax.plot(t/np.pi, p_exp, 'r--', linewidth=2, label='⟨p⟩/p₀')
ax.set_xlabel('Time (π/ω)', fontsize=12)
ax.set_ylabel('Expectation value', fontsize=12)
ax.set_title('Classical Oscillation of Expectation Values', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_383_time_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Time evolution plots saved.")

# =============================================================================
# Part 6: Overlap Between Coherent States
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Non-orthogonality of Coherent States")
print("=" * 70)

def overlap(alpha, beta):
    """
    |⟨β|α⟩|² = exp(-|α - β|²)
    """
    return np.exp(-np.abs(alpha - beta)**2)

# Create overlap matrix
alpha_range = np.linspace(-3, 3, 50)
beta_range = np.linspace(-3, 3, 50)

ALPHA, BETA = np.meshgrid(alpha_range, beta_range)
OVERLAP = overlap(ALPHA, BETA)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.contourf(alpha_range, beta_range, OVERLAP, levels=20, cmap='viridis')
ax.set_xlabel('Re(α) (with β = Re(β))', fontsize=12)
ax.set_ylabel('Re(β)', fontsize=12)
ax.set_title('Overlap |⟨β|α⟩|² for real α, β', fontsize=14)
plt.colorbar(im, label='|⟨β|α⟩|²')

plt.tight_layout()
plt.savefig('day_383_overlap.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nExample overlaps:")
print(f"|⟨0|1⟩|² = {overlap(0, 1):.4f}")
print(f"|⟨1|2⟩|² = {overlap(1, 2):.4f}")
print(f"|⟨0|3⟩|² = {overlap(0, 3):.6f}")
print(f"|⟨0|5⟩|² = {overlap(0, 5):.10f}")

# =============================================================================
# Part 7: Cat States
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Schrodinger Cat States")
print("=" * 70)

def cat_state(alpha, sign, N_max=30):
    """
    Cat state: N(|α⟩ ± |-α⟩)
    sign = +1 for even cat, -1 for odd cat
    """
    coeffs_alpha = coherent_state_fock(alpha, N_max)
    coeffs_minus_alpha = coherent_state_fock(-alpha, N_max)

    coeffs = coeffs_alpha + sign * coeffs_minus_alpha

    # Normalize
    norm = np.sqrt(np.sum(np.abs(coeffs)**2))
    return coeffs / norm

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

alpha = 2.0

# Even cat state
coeffs_even = cat_state(alpha, +1)
probs_even = np.abs(coeffs_even)**2

ax = axes[0, 0]
ax.bar(np.arange(30), probs_even, color='blue', alpha=0.7)
ax.set_xlabel('Photon number n')
ax.set_ylabel('Probability')
ax.set_title(f'Even Cat State |α⟩ + |-α⟩ (α = {alpha})', fontsize=12)
ax.set_xlim(-0.5, 20)

# Odd cat state
coeffs_odd = cat_state(alpha, -1)
probs_odd = np.abs(coeffs_odd)**2

ax = axes[0, 1]
ax.bar(np.arange(30), probs_odd, color='red', alpha=0.7)
ax.set_xlabel('Photon number n')
ax.set_ylabel('Probability')
ax.set_title(f'Odd Cat State |α⟩ - |-α⟩ (α = {alpha})', fontsize=12)
ax.set_xlim(-0.5, 20)

# Wave functions
xi = np.linspace(-8, 8, 500)

def cat_wavefunction(alpha, sign, xi, N_max=30):
    """Cat state wave function"""
    coeffs = cat_state(alpha, sign, N_max)
    psi = np.zeros_like(xi, dtype=complex)
    for n in range(N_max):
        psi += coeffs[n] * psi_n(n, xi)
    return psi

psi_even = cat_wavefunction(alpha, +1, xi)
psi_odd = cat_wavefunction(alpha, -1, xi)

ax = axes[1, 0]
ax.plot(xi, np.abs(psi_even)**2, 'b-', linewidth=2)
ax.fill_between(xi, 0, np.abs(psi_even)**2, alpha=0.3, color='blue')
ax.set_xlabel('ξ')
ax.set_ylabel('|ψ(ξ)|²')
ax.set_title('Even Cat State Wave Function')
ax.set_xlim(-6, 6)

ax = axes[1, 1]
ax.plot(xi, np.abs(psi_odd)**2, 'r-', linewidth=2)
ax.fill_between(xi, 0, np.abs(psi_odd)**2, alpha=0.3, color='red')
ax.set_xlabel('ξ')
ax.set_ylabel('|ψ(ξ)|²')
ax.set_title('Odd Cat State Wave Function')
ax.set_xlim(-6, 6)

plt.tight_layout()
plt.savefig('day_383_cat_states.png', dpi=150, bbox_inches='tight')
plt.show()

print("Cat state visualizations saved.")

# =============================================================================
# Part 8: Animation of Coherent State Evolution
# =============================================================================

print("\n" + "=" * 70)
print("Part 8: Creating Animation (saved as GIF)")
print("=" * 70)

# Create animation frames
fig, ax = plt.subplots(figsize=(10, 6))

xi = np.linspace(-8, 8, 300)
alpha_0 = 3.0

def update(frame):
    ax.clear()
    t = frame * 0.1
    alpha_t = alpha_0 * np.exp(-1j * t)

    psi = coherent_wavefunction(alpha_t, xi)
    prob = np.abs(psi)**2

    ax.fill_between(xi, 0, prob, alpha=0.5, color='blue')
    ax.plot(xi, prob, 'b-', linewidth=2)

    # Potential
    V = 0.1 * xi**2
    ax.plot(xi, V, 'k--', alpha=0.5)

    # Center position
    xi_mean = np.sqrt(2) * np.real(alpha_t)
    ax.axvline(xi_mean, color='red', linestyle=':', linewidth=2)

    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('ξ', fontsize=12)
    ax.set_ylabel('|ψ|²', fontsize=12)
    ax.set_title(f'Coherent State Evolution: t = {t:.1f}, α(t) = {alpha_t:.2f}', fontsize=14)

    return ax,

# Save a few frames as static image
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
xi = np.linspace(-8, 8, 300)
times = np.linspace(0, 2*np.pi, 8)

for idx, t in enumerate(times):
    ax = axes[idx // 4, idx % 4]
    alpha_t = alpha_0 * np.exp(-1j * t)

    psi = coherent_wavefunction(alpha_t, xi)
    prob = np.abs(psi)**2

    ax.fill_between(xi, 0, prob, alpha=0.5, color='blue')
    ax.plot(xi, prob, 'b-', linewidth=2)

    V = 0.1 * xi**2
    ax.plot(xi, V, 'k--', alpha=0.5)

    xi_mean = np.sqrt(2) * np.real(alpha_t)
    ax.axvline(xi_mean, color='red', linestyle=':', linewidth=2)

    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 0.8)
    ax.set_title(f't = {t/np.pi:.2f}π')

fig.suptitle('Coherent State Time Evolution (One Period)', fontsize=14)
plt.tight_layout()
plt.savefig('day_383_evolution_frames.png', dpi=150, bbox_inches='tight')
plt.show()

print("Animation frames saved.")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Definition | $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$ |
| Fock expansion | $|\alpha\rangle = e^{-|\alpha|^2/2}\sum_n\frac{\alpha^n}{\sqrt{n!}}|n\rangle$ |
| Photon distribution | $P(n) = e^{-|\alpha|^2}\frac{|\alpha|^{2n}}{n!}$ (Poisson) |
| Mean photon number | $\langle\hat{N}\rangle = |\alpha|^2$ |
| Displacement operator | $\hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$ |
| State construction | $|\alpha\rangle = \hat{D}(\alpha)|0\rangle$ |
| Time evolution | $|\alpha(t)\rangle = e^{-i\omega t/2}|\alpha e^{-i\omega t}\rangle$ |
| Overlap | $|\langle\beta|\alpha\rangle|^2 = e^{-|\alpha-\beta|^2}$ |
| Uncertainty | $\Delta x \cdot \Delta p = \frac{\hbar}{2}$ (minimum) |

### Main Takeaways

1. **Eigenvalue equation:** Coherent states are eigenstates of $\hat{a}$, not $\hat{H}$
2. **Poissonian statistics:** Photon number follows a Poisson distribution
3. **Minimum uncertainty:** Coherent states saturate the Heisenberg limit
4. **Classical motion:** Expectation values follow classical trajectories
5. **Gaussian wave packet:** A coherent state is a displaced Gaussian that doesn't spread
6. **Overcompleteness:** Coherent states form an overcomplete basis

---

## Daily Checklist

- [ ] Read Shankar Section 7.5 on coherent states
- [ ] Verify the Fock basis expansion by acting with $\hat{a}$
- [ ] Prove that coherent states are minimum uncertainty states
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt the cat state problem (Level 3, #7)
- [ ] Run and understand the computational lab
- [ ] Watch the coherent state animation and connect to classical motion

---

## Preview: Day 384

Tomorrow we explore **phase space** representations of quantum states. We'll introduce the Wigner function, visualize quantum states as quasi-probability distributions, and understand the classical limit.

---

*"Coherent states are the quantum states that most closely resemble classical oscillations."*
— R.J. Glauber (Nobel Prize 2005)

---

**Next:** [Day_384_Saturday.md](Day_384_Saturday.md) — QHO in Phase Space
