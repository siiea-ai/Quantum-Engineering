# Day 342: Continuous Spectra

## Week 49, Day 6 - Saturday

### Month 13: Postulates & Mathematical Framework

---

## Schedule Overview (7 Hours Total)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning I** | 90 min | Position eigenstates and Dirac delta function |
| **Morning II** | 90 min | Momentum eigenstates and normalization |
| **Afternoon I** | 90 min | Completeness relations and wave function interpretation |
| **Afternoon II** | 60 min | Position-momentum overlap and Fourier connection |
| **Evening Lab** | 90 min | Computational: Wave functions and Fourier transforms |

---

## Learning Objectives

By the end of today's session, you will be able to:

1. **Define position eigenstates** $|x\rangle$ and express the eigenvalue equation $\hat{x}|x\rangle = x|x\rangle$ for continuous spectra
2. **Apply the Dirac delta function** $\delta(x-x')$ for orthonormality of continuous eigenstates
3. **Construct completeness relations** $\hat{I} = \int_{-\infty}^{\infty}|x\rangle\langle x|dx$ and explain their physical significance
4. **Interpret wave functions** as inner products $\psi(x) = \langle x|\psi\rangle$ in the position basis
5. **Derive the position-momentum overlap** $\langle x|p\rangle = (2\pi\hbar)^{-1/2}e^{ipx/\hbar}$ from first principles
6. **Connect Fourier transforms** to quantum mechanical basis transformations

---

## Core Content

### 1. Position Eigenstates and the Position Operator

In the discrete case, observables have eigenstates labeled by quantum numbers taking countable values. For position, the eigenvalue spectrum is continuous—every real number $x$ is a possible measurement outcome.

#### The Position Eigenvalue Equation

The position operator $\hat{x}$ acts on its eigenstates:

$$\boxed{\hat{x}|x\rangle = x|x\rangle}$$

where:
- $|x\rangle$ is the eigenstate corresponding to a particle localized exactly at position $x$
- $x$ (without hat) is the eigenvalue, a real number
- The spectrum is continuous: $x \in (-\infty, +\infty)$

**Physical Interpretation:** The state $|x\rangle$ represents a particle with definite position $x$. This is an idealization—no physical state can have exactly zero position uncertainty (which would require infinite momentum uncertainty by the uncertainty principle).

#### Why Continuous Spectra Require New Mathematics

For discrete eigenstates $|n\rangle$, we have:
- Kronecker delta orthonormality: $\langle m|n\rangle = \delta_{mn}$
- Completeness with summation: $\hat{I} = \sum_n |n\rangle\langle n|$

For continuous eigenstates, these become:
- Dirac delta orthonormality: $\langle x|x'\rangle = \delta(x-x')$
- Completeness with integration: $\hat{I} = \int |x\rangle\langle x|\,dx$

---

### 2. The Dirac Delta Function

The Dirac delta function $\delta(x)$ is not a function in the ordinary sense but a *distribution* (or *generalized function*). It formalizes the notion of infinite localization.

#### Defining Properties

$$\boxed{\delta(x-x') = \begin{cases} +\infty & \text{if } x = x' \\ 0 & \text{if } x \neq x' \end{cases}}$$

with the normalization constraint:

$$\boxed{\int_{-\infty}^{\infty} \delta(x-x')\,dx = 1}$$

#### The Sifting Property

The most important property of the delta function:

$$\boxed{\int_{-\infty}^{\infty} f(x)\delta(x-x')\,dx = f(x')}$$

This "sifts out" the value of $f$ at the point $x'$.

#### Representations of the Delta Function

The delta function can be understood as limits of increasingly narrow functions:

**Gaussian Representation:**
$$\delta(x) = \lim_{\epsilon \to 0^+} \frac{1}{\sqrt{2\pi\epsilon^2}}e^{-x^2/2\epsilon^2}$$

**Lorentzian Representation:**
$$\delta(x) = \lim_{\epsilon \to 0^+} \frac{1}{\pi}\frac{\epsilon}{x^2 + \epsilon^2}$$

**Sinc Representation (Fourier):**
$$\delta(x) = \lim_{L \to \infty} \frac{\sin(Lx)}{\pi x} = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{ikx}\,dk$$

#### Important Delta Function Identities

$$\delta(ax) = \frac{1}{|a|}\delta(x), \quad a \neq 0$$

$$\delta(x^2 - a^2) = \frac{1}{2|a|}[\delta(x-a) + \delta(x+a)], \quad a > 0$$

$$x\delta(x) = 0$$

$$\delta'(x) = -\frac{\delta(x)}{x} \quad \text{(distributional derivative)}$$

---

### 3. Orthonormality of Position Eigenstates

Position eigenstates satisfy the orthonormality condition using the Dirac delta:

$$\boxed{\langle x|x'\rangle = \delta(x-x')}$$

**Interpretation:** Two position eigenstates are orthogonal unless they correspond to the same position. The "infinite overlap" at $x = x'$ reflects the non-normalizable nature of position eigenstates.

**Non-Normalizability:** Unlike discrete eigenstates where $\langle n|n\rangle = 1$, position eigenstates satisfy:

$$\langle x|x\rangle = \delta(0) = \infty$$

This means $|x\rangle$ cannot represent physical states by itself—they are mathematical tools for expanding physical states.

---

### 4. Completeness Relation for Position

The completeness (or closure) relation states that position eigenstates form a complete basis:

$$\boxed{\hat{I} = \int_{-\infty}^{\infty}|x\rangle\langle x|\,dx}$$

This means any state $|\psi\rangle$ can be expanded:

$$|\psi\rangle = \hat{I}|\psi\rangle = \int_{-\infty}^{\infty}|x\rangle\langle x|\psi\rangle\,dx = \int_{-\infty}^{\infty}\psi(x)|x\rangle\,dx$$

where we define the **wave function**:

$$\boxed{\psi(x) \equiv \langle x|\psi\rangle}$$

---

### 5. The Wave Function as an Inner Product

The wave function $\psi(x)$ is the **position-space representation** of the abstract state $|\psi\rangle$:

$$\psi(x) = \langle x|\psi\rangle$$

**Physical Meaning:** $|\psi(x)|^2\,dx$ is the probability of finding the particle between $x$ and $x + dx$.

**Normalization:** Physical states satisfy:
$$\langle\psi|\psi\rangle = \int_{-\infty}^{\infty}\langle\psi|x\rangle\langle x|\psi\rangle\,dx = \int_{-\infty}^{\infty}|\psi(x)|^2\,dx = 1$$

**Inner Products:** For two states:
$$\langle\phi|\psi\rangle = \int_{-\infty}^{\infty}\phi^*(x)\psi(x)\,dx$$

---

### 6. Momentum Eigenstates

The momentum operator $\hat{p}$ also has a continuous spectrum:

$$\boxed{\hat{p}|p\rangle = p|p\rangle}$$

where $p \in (-\infty, +\infty)$.

#### Orthonormality of Momentum Eigenstates

$$\boxed{\langle p|p'\rangle = \delta(p-p')}$$

#### Completeness Relation for Momentum

$$\boxed{\hat{I} = \int_{-\infty}^{\infty}|p\rangle\langle p|\,dp}$$

#### Momentum-Space Wave Function

The **momentum-space wave function** is:

$$\tilde{\psi}(p) \equiv \langle p|\psi\rangle$$

with normalization:
$$\int_{-\infty}^{\infty}|\tilde{\psi}(p)|^2\,dp = 1$$

---

### 7. Position-Momentum Overlap: The Central Result

The overlap between position and momentum eigenstates is:

$$\boxed{\langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}}$$

This is the "plane wave" and encodes the Fourier transform relationship between position and momentum representations.

#### Derivation

Starting from the momentum operator in position representation:
$$\hat{p} = -i\hbar\frac{d}{dx}$$

Apply to $\langle x|p\rangle$:
$$\langle x|\hat{p}|p\rangle = p\langle x|p\rangle$$

In position representation:
$$-i\hbar\frac{d}{dx}\langle x|p\rangle = p\langle x|p\rangle$$

This is a first-order ODE with solution:
$$\langle x|p\rangle = A\,e^{ipx/\hbar}$$

The normalization constant $A$ is determined by requiring:
$$\langle p|p'\rangle = \int_{-\infty}^{\infty}\langle p|x\rangle\langle x|p'\rangle\,dx = |A|^2\int_{-\infty}^{\infty}e^{i(p'-p)x/\hbar}\,dx$$

Using the Fourier representation of the delta function:
$$\int_{-\infty}^{\infty}e^{ikx}\,dx = 2\pi\delta(k)$$

We get:
$$\langle p|p'\rangle = |A|^2 \cdot 2\pi\hbar\,\delta(p-p')$$

For $\langle p|p'\rangle = \delta(p-p')$, we need $|A|^2 = 1/(2\pi\hbar)$, giving:

$$A = \frac{1}{\sqrt{2\pi\hbar}}$$

**Conjugate Relation:**
$$\langle p|x\rangle = \langle x|p\rangle^* = \frac{1}{\sqrt{2\pi\hbar}}e^{-ipx/\hbar}$$

---

### 8. Fourier Transform Connection

The position-momentum overlap establishes Fourier transforms as basis transformations in quantum mechanics.

#### Position to Momentum Transformation

$$\tilde{\psi}(p) = \langle p|\psi\rangle = \int_{-\infty}^{\infty}\langle p|x\rangle\langle x|\psi\rangle\,dx$$

$$\boxed{\tilde{\psi}(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)\,dx}$$

This is the **Fourier transform** of $\psi(x)$ (with $k = p/\hbar$).

#### Momentum to Position Transformation

$$\psi(x) = \langle x|\psi\rangle = \int_{-\infty}^{\infty}\langle x|p\rangle\langle p|\psi\rangle\,dp$$

$$\boxed{\psi(x) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{ipx/\hbar}\tilde{\psi}(p)\,dp}$$

This is the **inverse Fourier transform**.

#### Physical Significance

- Fourier transform = change of basis from position to momentum
- Narrow $\psi(x)$ (well-localized) $\Leftrightarrow$ broad $\tilde{\psi}(p)$ (large momentum spread)
- This is a manifestation of the uncertainty principle

---

### 9. Rigged Hilbert Space (Gelfand Triple)

**Brief Technical Note:** Position and momentum eigenstates are not in the standard Hilbert space $\mathcal{H}$ (they're not normalizable). The mathematically rigorous framework is the **rigged Hilbert space** or **Gelfand triple**:

$$\Phi \subset \mathcal{H} \subset \Phi'$$

where:
- $\Phi$ = space of "well-behaved" test functions (e.g., Schwartz space)
- $\mathcal{H}$ = standard Hilbert space of square-integrable functions
- $\Phi'$ = space of distributions (generalized functions), containing $|x\rangle$, $|p\rangle$

For practical calculations in physics, we work with the formal Dirac notation, keeping in mind that $|x\rangle$ and $|p\rangle$ are "ideal" states used for expansion.

**Reference:** Shankar Ch. 1.10-1.11; Sakurai Ch. 1.6; Ballentine Ch. 1.4

---

## Worked Examples

### Example 1: Action of Position Operator on a Wave Function

**Problem:** Given $|\psi\rangle$ with wave function $\psi(x) = Ae^{-x^2/2\sigma^2}$, find the wave function of $\hat{x}|\psi\rangle$ and compute $\langle\psi|\hat{x}|\psi\rangle$.

**Solution:**

The wave function of $\hat{x}|\psi\rangle$ is:
$$\langle x|\hat{x}|\psi\rangle = x\langle x|\psi\rangle = x\psi(x) = Axe^{-x^2/2\sigma^2}$$

For the expectation value:
$$\langle\psi|\hat{x}|\psi\rangle = \int_{-\infty}^{\infty}\psi^*(x)\,x\,\psi(x)\,dx = |A|^2\int_{-\infty}^{\infty}x\,e^{-x^2/\sigma^2}\,dx$$

The integrand is an odd function, so:
$$\boxed{\langle\hat{x}\rangle = 0}$$

To find $|A|^2$, normalize:
$$|A|^2\int_{-\infty}^{\infty}e^{-x^2/\sigma^2}\,dx = |A|^2\sigma\sqrt{\pi} = 1$$
$$A = \frac{1}{(\pi\sigma^2)^{1/4}}$$

---

### Example 2: Fourier Transform of a Gaussian

**Problem:** Find the momentum-space wave function $\tilde{\psi}(p)$ for:
$$\psi(x) = \left(\frac{1}{\pi\sigma^2}\right)^{1/4}e^{-x^2/2\sigma^2}$$

**Solution:**

Using the Fourier transform:
$$\tilde{\psi}(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty}e^{-ipx/\hbar}\psi(x)\,dx$$

$$= \frac{1}{\sqrt{2\pi\hbar}}\left(\frac{1}{\pi\sigma^2}\right)^{1/4}\int_{-\infty}^{\infty}e^{-ipx/\hbar}e^{-x^2/2\sigma^2}\,dx$$

Complete the square in the exponent:
$$-\frac{x^2}{2\sigma^2} - \frac{ipx}{\hbar} = -\frac{1}{2\sigma^2}\left(x + \frac{ip\sigma^2}{\hbar}\right)^2 - \frac{p^2\sigma^2}{2\hbar^2}$$

The Gaussian integral gives:
$$\int_{-\infty}^{\infty}e^{-\frac{1}{2\sigma^2}(x + ip\sigma^2/\hbar)^2}\,dx = \sigma\sqrt{2\pi}$$

Therefore:
$$\tilde{\psi}(p) = \frac{\sigma\sqrt{2\pi}}{\sqrt{2\pi\hbar}}\left(\frac{1}{\pi\sigma^2}\right)^{1/4}e^{-p^2\sigma^2/2\hbar^2}$$

$$\boxed{\tilde{\psi}(p) = \left(\frac{\sigma^2}{\pi\hbar^2}\right)^{1/4}e^{-p^2\sigma^2/2\hbar^2}}$$

**Key Insight:** A Gaussian in position space with width $\sigma$ transforms to a Gaussian in momentum space with width $\hbar/\sigma$. The product of uncertainties:
$$\Delta x \cdot \Delta p = \frac{\sigma}{\sqrt{2}} \cdot \frac{\hbar}{\sigma\sqrt{2}} = \frac{\hbar}{2}$$

This is the **minimum uncertainty state** saturating the Heisenberg bound.

---

### Example 3: Delta Function Normalization Check

**Problem:** Verify that $\langle x|p\rangle = (2\pi\hbar)^{-1/2}e^{ipx/\hbar}$ gives the correct orthonormality for momentum eigenstates.

**Solution:**

Compute $\langle p|p'\rangle$ using completeness:
$$\langle p|p'\rangle = \int_{-\infty}^{\infty}\langle p|x\rangle\langle x|p'\rangle\,dx$$

$$= \int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi\hbar}}e^{-ipx/\hbar} \cdot \frac{1}{\sqrt{2\pi\hbar}}e^{ip'x/\hbar}\,dx$$

$$= \frac{1}{2\pi\hbar}\int_{-\infty}^{\infty}e^{i(p'-p)x/\hbar}\,dx$$

Let $k = (p' - p)/\hbar$:
$$= \frac{1}{2\pi\hbar}\int_{-\infty}^{\infty}e^{ikx}\,dx \cdot \hbar = \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{ikx}\,dx$$

Using $\int_{-\infty}^{\infty}e^{ikx}\,dx = 2\pi\delta(k)$:
$$= \frac{1}{2\pi} \cdot 2\pi\delta\left(\frac{p'-p}{\hbar}\right) = \delta\left(\frac{p'-p}{\hbar}\right)$$

Using $\delta(ax) = |a|^{-1}\delta(x)$ with $a = 1/\hbar$:
$$\boxed{\langle p|p'\rangle = \hbar \cdot \frac{1}{\hbar}\delta(p'-p) = \delta(p-p') \quad \checkmark}$$

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Show that $\langle x|\hat{x}^2|\psi\rangle = x^2\psi(x)$.

**Problem 1.2:** Using completeness, prove that:
$$\langle\phi|\psi\rangle = \int_{-\infty}^{\infty}\phi^*(x)\psi(x)\,dx = \int_{-\infty}^{\infty}\tilde{\phi}^*(p)\tilde{\psi}(p)\,dp$$
(Parseval's theorem)

**Problem 1.3:** Calculate $\langle x|\hat{p}|\psi\rangle$ and show it equals $-i\hbar\frac{d\psi}{dx}$.

### Level 2: Intermediate

**Problem 2.1:** For the state $|\psi\rangle = \int f(x)|x\rangle\,dx$ where $f(x) = N(a^2 - x^2)$ for $|x| < a$ and zero otherwise:
(a) Find the normalization constant $N$
(b) Calculate $\langle\hat{x}\rangle$ and $\langle\hat{x}^2\rangle$
(c) Find the momentum-space wave function $\tilde{\psi}(p)$

**Problem 2.2:** Prove the completeness relation can be written as:
$$\delta(x-x') = \frac{1}{2\pi\hbar}\int_{-\infty}^{\infty}e^{ip(x-x')/\hbar}\,dp$$

**Problem 2.3:** A particle has momentum-space wave function $\tilde{\psi}(p) = B\,e^{-|p|/p_0}$. Find:
(a) The normalization constant $B$
(b) The position-space wave function $\psi(x)$
(c) The probability density $|\psi(x)|^2$

### Level 3: Challenging

**Problem 3.1:** Starting from $[\hat{x}, \hat{p}] = i\hbar$, derive the form of $\hat{p}$ in position representation without assuming it is $-i\hbar\frac{d}{dx}$.
*Hint:* Consider $\langle x|[\hat{x}, \hat{p}]|x'\rangle$ and use the product rule.

**Problem 3.2:** The **squeeze operator** is $\hat{S}(\xi) = e^{\frac{\xi}{2}(\hat{a}^2 - \hat{a}^{\dagger 2})}$ for real $\xi$. For a squeezed Gaussian state:
$$\psi_\xi(x) = \left(\frac{1}{\pi\sigma_\xi^2}\right)^{1/4}e^{-x^2/2\sigma_\xi^2}$$
with $\sigma_\xi = \sigma_0 e^{-\xi}$:
(a) Find $\tilde{\psi}_\xi(p)$
(b) Calculate $\Delta x$ and $\Delta p$
(c) Verify $\Delta x \cdot \Delta p = \hbar/2$ for all $\xi$

**Problem 3.3:** Consider the **coherent state** $|z\rangle$ defined by $\hat{a}|z\rangle = z|z\rangle$. Show that:
$$\langle x|z\rangle = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}\exp\left[-\frac{m\omega}{2\hbar}\left(x - \sqrt{\frac{2\hbar}{m\omega}}\text{Re}(z)\right)^2 + i\sqrt{\frac{2m\omega}{\hbar}}\text{Im}(z)x\right]$$

---

## Computational Lab: Wave Functions and Fourier Transforms

```python
"""
Day 342 Computational Lab: Continuous Spectra and Fourier Transforms
====================================================================
Exploring position/momentum representations and their Fourier transform
relationship using numerical techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift

# Physical constants (using natural units with hbar = 1 for simplicity)
hbar = 1.0

# ============================================================================
# Part 1: Gaussian Wave Packet in Position and Momentum Space
# ============================================================================

def gaussian_position(x, sigma, x0=0):
    """
    Normalized Gaussian wave function in position space.
    psi(x) = (1/pi*sigma^2)^(1/4) * exp(-(x-x0)^2 / (2*sigma^2))
    """
    norm = (1.0 / (np.pi * sigma**2))**0.25
    return norm * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gaussian_momentum_analytical(p, sigma):
    """
    Analytical Fourier transform of Gaussian (momentum space wave function).
    """
    sigma_p = hbar / sigma  # Width in momentum space
    norm = (sigma**2 / (np.pi * hbar**2))**0.25
    return norm * np.exp(-p**2 * sigma**2 / (2 * hbar**2))

# Set up spatial grid
N = 2048  # Number of points (power of 2 for FFT efficiency)
x_max = 20.0
x = np.linspace(-x_max, x_max, N)
dx = x[1] - x[0]

# Gaussian parameters
sigma = 1.0  # Width in position space

# Compute position-space wave function
psi_x = gaussian_position(x, sigma)

# Compute momentum-space wave function via FFT
# Note: numpy FFT convention differs from physics convention
# We need to include proper normalization and phase factors
psi_p_numerical = fftshift(fft(psi_x)) * dx / np.sqrt(2 * np.pi * hbar)
p = fftshift(fftfreq(N, dx)) * 2 * np.pi * hbar

# Analytical result for comparison
psi_p_analytical = gaussian_momentum_analytical(p, sigma)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Position space probability density
ax1 = axes[0, 0]
ax1.plot(x, np.abs(psi_x)**2, 'b-', linewidth=2, label='$|\\psi(x)|^2$')
ax1.fill_between(x, np.abs(psi_x)**2, alpha=0.3)
ax1.set_xlabel('Position $x$', fontsize=12)
ax1.set_ylabel('Probability density', fontsize=12)
ax1.set_title('Position-Space Wave Function', fontsize=14)
ax1.set_xlim(-5, 5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Momentum space probability density
ax2 = axes[0, 1]
ax2.plot(p, np.abs(psi_p_numerical)**2, 'r-', linewidth=2,
         label='Numerical FFT')
ax2.plot(p, np.abs(psi_p_analytical)**2, 'k--', linewidth=2,
         label='Analytical')
ax2.fill_between(p, np.abs(psi_p_numerical)**2, alpha=0.3, color='red')
ax2.set_xlabel('Momentum $p$', fontsize=12)
ax2.set_ylabel('Probability density', fontsize=12)
ax2.set_title('Momentum-Space Wave Function', fontsize=14)
ax2.set_xlim(-5, 5)
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================================================
# Part 2: Uncertainty Principle Demonstration
# ============================================================================

sigmas = np.linspace(0.3, 3.0, 50)
delta_x_list = []
delta_p_list = []

for sig in sigmas:
    # Position uncertainty for Gaussian: Delta_x = sigma/sqrt(2)
    delta_x = sig / np.sqrt(2)
    # Momentum uncertainty: Delta_p = hbar/(sigma*sqrt(2))
    delta_p = hbar / (sig * np.sqrt(2))
    delta_x_list.append(delta_x)
    delta_p_list.append(delta_p)

delta_x_arr = np.array(delta_x_list)
delta_p_arr = np.array(delta_p_list)
product = delta_x_arr * delta_p_arr

ax3 = axes[1, 0]
ax3.plot(sigmas, delta_x_arr, 'b-', linewidth=2, label='$\\Delta x$')
ax3.plot(sigmas, delta_p_arr, 'r-', linewidth=2, label='$\\Delta p$')
ax3.axhline(y=hbar/2, color='gray', linestyle='--', label='$\\hbar/2$')
ax3.set_xlabel('Gaussian width $\\sigma$', fontsize=12)
ax3.set_ylabel('Uncertainty', fontsize=12)
ax3.set_title('Position and Momentum Uncertainties vs Width', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.plot(sigmas, product, 'g-', linewidth=2, label='$\\Delta x \\cdot \\Delta p$')
ax4.axhline(y=hbar/2, color='red', linestyle='--', linewidth=2,
            label='Heisenberg bound $\\hbar/2$')
ax4.set_xlabel('Gaussian width $\\sigma$', fontsize=12)
ax4.set_ylabel('$\\Delta x \\cdot \\Delta p$', fontsize=12)
ax4.set_title('Uncertainty Product (Minimum Uncertainty State)', fontsize=14)
ax4.set_ylim(0, 1.0)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day342_gaussian_uncertainty.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Part 3: Dirac Delta Function Approximations
# ============================================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

x_delta = np.linspace(-3, 3, 1000)
epsilons = [1.0, 0.5, 0.2, 0.1, 0.05]
colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))

# Gaussian approximation
ax = axes2[0]
for eps, color in zip(epsilons, colors):
    delta_gaussian = (1.0 / np.sqrt(2 * np.pi * eps**2)) * np.exp(-x_delta**2 / (2 * eps**2))
    ax.plot(x_delta, delta_gaussian, color=color, linewidth=2,
            label=f'$\\epsilon = {eps}$')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$\\delta_\\epsilon(x)$', fontsize=12)
ax.set_title('Gaussian Approximation to $\\delta(x)$', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 10)

# Lorentzian approximation
ax = axes2[1]
for eps, color in zip(epsilons, colors):
    delta_lorentz = (1.0 / np.pi) * eps / (x_delta**2 + eps**2)
    ax.plot(x_delta, delta_lorentz, color=color, linewidth=2,
            label=f'$\\epsilon = {eps}$')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$\\delta_\\epsilon(x)$', fontsize=12)
ax.set_title('Lorentzian Approximation to $\\delta(x)$', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 10)

# Sinc approximation
ax = axes2[2]
L_values = [1, 2, 5, 10, 20]
colors2 = plt.cm.plasma(np.linspace(0, 1, len(L_values)))
for L, color in zip(L_values, colors2):
    # Avoid division by zero
    delta_sinc = np.where(np.abs(x_delta) > 1e-10,
                          np.sin(L * x_delta) / (np.pi * x_delta),
                          L / np.pi)
    ax.plot(x_delta, delta_sinc, color=color, linewidth=1.5,
            label=f'$L = {L}$')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$\\delta_L(x)$', fontsize=12)
ax.set_title('Sinc Approximation to $\\delta(x)$', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-2, 8)

plt.tight_layout()
plt.savefig('day342_delta_approximations.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Part 4: Position-Momentum Overlap Visualization
# ============================================================================

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

# Plane waves for different momenta
x_pw = np.linspace(-10, 10, 500)
momenta = [0.5, 1.0, 2.0, 3.0]
colors3 = plt.cm.coolwarm(np.linspace(0, 1, len(momenta)))

ax = axes3[0, 0]
for p_val, color in zip(momenta, colors3):
    # <x|p> = (2*pi*hbar)^(-1/2) * exp(i*p*x/hbar)
    overlap = (1.0 / np.sqrt(2 * np.pi * hbar)) * np.exp(1j * p_val * x_pw / hbar)
    ax.plot(x_pw, np.real(overlap), color=color, linewidth=1.5,
            label=f'$p = {p_val}$')
ax.set_xlabel('Position $x$', fontsize=12)
ax.set_ylabel('Re$\\langle x|p\\rangle$', fontsize=12)
ax.set_title('Real Part of Position-Momentum Overlap', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes3[0, 1]
for p_val, color in zip(momenta, colors3):
    overlap = (1.0 / np.sqrt(2 * np.pi * hbar)) * np.exp(1j * p_val * x_pw / hbar)
    ax.plot(x_pw, np.imag(overlap), color=color, linewidth=1.5,
            label=f'$p = {p_val}$')
ax.set_xlabel('Position $x$', fontsize=12)
ax.set_ylabel('Im$\\langle x|p\\rangle$', fontsize=12)
ax.set_title('Imaginary Part of Position-Momentum Overlap', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Wave packet decomposition
ax = axes3[1, 0]
# Create a Gaussian wave packet
sigma_wp = 2.0
psi_wp = gaussian_position(x_pw, sigma_wp)
ax.plot(x_pw, np.real(psi_wp), 'b-', linewidth=2, label='$\\psi(x)$')
ax.fill_between(x_pw, np.real(psi_wp), alpha=0.3)

# Show decomposition into plane waves (schematic)
for i, p_val in enumerate([0.2, 0.5, 1.0]):
    weight = gaussian_momentum_analytical(p_val, sigma_wp)
    plane_wave = np.real((1.0 / np.sqrt(2 * np.pi * hbar)) * np.exp(1j * p_val * x_pw / hbar))
    ax.plot(x_pw, weight * plane_wave * 3, '--', alpha=0.7,
            label=f'$\\tilde{{\\psi}}({p_val})\\langle x|{p_val}\\rangle$')
ax.set_xlabel('Position $x$', fontsize=12)
ax.set_ylabel('Wave function', fontsize=12)
ax.set_title('Wave Packet as Superposition of Plane Waves', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Probability conservation
ax = axes3[1, 1]
# Verify normalization in both spaces
psi_x_norm = gaussian_position(x, sigma)
norm_x = np.trapezoid(np.abs(psi_x_norm)**2, x)

psi_p_norm = gaussian_momentum_analytical(p, sigma)
norm_p = np.trapezoid(np.abs(psi_p_norm)**2, p)

ax.bar(['Position space', 'Momentum space'], [norm_x, norm_p],
       color=['blue', 'red'], alpha=0.7, edgecolor='black')
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
           label='Expected: 1.0')
ax.set_ylabel('Normalization integral', fontsize=12)
ax.set_title('Parseval\'s Theorem: Norm Conservation', fontsize=14)
ax.set_ylim(0, 1.2)
ax.legend()

for i, (name, val) in enumerate([('Position', norm_x), ('Momentum', norm_p)]):
    ax.text(i, val + 0.05, f'{val:.6f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('day342_overlap_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Part 5: Interactive Exploration - Wave Packet Width vs Uncertainty
# ============================================================================

print("\n" + "="*70)
print("NUMERICAL VERIFICATION OF CONTINUOUS SPECTRUM PROPERTIES")
print("="*70)

# Test sifting property numerically
def test_sifting_property():
    """Verify delta function sifting property numerically."""
    x_test = np.linspace(-5, 5, 1001)
    dx_test = x_test[1] - x_test[0]

    # Test function
    f = lambda x: x**2 + 2*x + 1

    # Delta approximation (Gaussian)
    x_prime = 1.5  # Point to sift out
    epsilon = 0.01
    delta_approx = (1.0 / np.sqrt(2*np.pi*epsilon**2)) * np.exp(-(x_test - x_prime)**2 / (2*epsilon**2))

    # Numerical integral
    result = np.trapezoid(f(x_test) * delta_approx, x_test)
    expected = f(x_prime)

    print(f"\nSifting Property Test:")
    print(f"  f(x) = x^2 + 2x + 1, x' = {x_prime}")
    print(f"  Numerical: integral[f(x) * delta(x-x')] dx = {result:.6f}")
    print(f"  Expected:  f(x') = {expected:.6f}")
    print(f"  Error: {abs(result - expected):.2e}")

test_sifting_property()

# Verify completeness relation
def test_completeness():
    """Verify completeness relation numerically."""
    x_grid = np.linspace(-10, 10, 501)
    dx = x_grid[1] - x_grid[0]

    # Test with a specific state
    sigma = 1.0
    psi = gaussian_position(x_grid, sigma)

    # Apply completeness: I|psi> should give back |psi>
    # In position basis: integral |x><x|psi> dx = integral |x> psi(x) dx
    # which gives psi(x') = <x'|psi>

    # Verify via inner product preservation
    inner_product_direct = np.trapezoid(np.abs(psi)**2, x_grid)

    print(f"\nCompleteness Relation Test:")
    print(f"  <psi|psi> = integral |psi(x)|^2 dx = {inner_product_direct:.6f}")
    print(f"  Expected: 1.0")
    print(f"  Error: {abs(inner_product_direct - 1.0):.2e}")

test_completeness()

# Verify Fourier transform is unitary
def test_fourier_unitarity():
    """Verify Fourier transform preserves norm."""
    N = 4096
    x = np.linspace(-20, 20, N)
    dx = x[1] - x[0]

    # Non-Gaussian test function
    psi_x = np.exp(-x**2/4) * np.cos(3*x)
    psi_x = psi_x / np.sqrt(np.trapezoid(np.abs(psi_x)**2, x))  # Normalize

    # FFT to momentum space
    psi_p = fftshift(fft(psi_x)) * dx / np.sqrt(2*np.pi*hbar)
    p = fftshift(fftfreq(N, dx)) * 2*np.pi*hbar
    dp = p[1] - p[0]

    norm_x = np.trapezoid(np.abs(psi_x)**2, x)
    norm_p = np.trapezoid(np.abs(psi_p)**2, p)

    print(f"\nFourier Unitarity Test (Parseval's Theorem):")
    print(f"  ||psi||^2 in position space: {norm_x:.6f}")
    print(f"  ||psi||^2 in momentum space: {norm_p:.6f}")
    print(f"  Ratio (should be 1): {norm_p/norm_x:.6f}")

test_fourier_unitarity()

print("\n" + "="*70)
print("Lab complete. Generated visualizations saved to current directory.")
print("="*70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Position eigenvalue equation | $\hat{x}\|x\rangle = x\|x\rangle$ |
| Momentum eigenvalue equation | $\hat{p}\|p\rangle = p\|p\rangle$ |
| Position orthonormality | $\langle x\|x'\rangle = \delta(x-x')$ |
| Momentum orthonormality | $\langle p\|p'\rangle = \delta(p-p')$ |
| Position completeness | $\hat{I} = \int_{-\infty}^{\infty}\|x\rangle\langle x\|dx$ |
| Momentum completeness | $\hat{I} = \int_{-\infty}^{\infty}\|p\rangle\langle p\|dp$ |
| Wave function definition | $\psi(x) = \langle x\|\psi\rangle$ |
| Position-momentum overlap | $\langle x\|p\rangle = (2\pi\hbar)^{-1/2}e^{ipx/\hbar}$ |
| Fourier transform (QM) | $\tilde{\psi}(p) = (2\pi\hbar)^{-1/2}\int e^{-ipx/\hbar}\psi(x)\,dx$ |
| Delta sifting property | $\int f(x)\delta(x-a)\,dx = f(a)$ |

### Main Takeaways

1. **Continuous spectra** require the Dirac delta function for orthonormality, replacing Kronecker deltas.

2. **Position and momentum eigenstates** are not normalizable and live outside the standard Hilbert space (in the "rigged" extension).

3. **The wave function** $\psi(x) = \langle x|\psi\rangle$ is the projection of the abstract state onto position eigenstates.

4. **The position-momentum overlap** $\langle x|p\rangle = (2\pi\hbar)^{-1/2}e^{ipx/\hbar}$ is fundamental—it connects the two representations via Fourier transforms.

5. **Fourier transforms are unitary** basis changes between position and momentum representations, preserving norms and inner products.

6. **The uncertainty principle emerges naturally** from the Fourier relationship: narrow position distributions require broad momentum distributions.

---

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain why continuous spectra require delta function normalization
- [ ] I understand that $|x\rangle$ and $|p\rangle$ are mathematical tools, not physical states
- [ ] I can interpret the wave function as an inner product with position eigenstates
- [ ] I understand the Fourier transform as a basis change in Hilbert space

### Mathematical Skills
- [ ] I can derive the position-momentum overlap from first principles
- [ ] I can manipulate delta function integrals using the sifting property
- [ ] I can transform between position and momentum representations
- [ ] I can verify orthonormality and completeness relations

### Problem-Solving
- [ ] I solved at least 2 Level 1 problems correctly
- [ ] I attempted at least 1 Level 2 problem
- [ ] I ran and understood the computational lab code

### Connections
- [ ] I see how this formalism connects to wave mechanics from earlier courses
- [ ] I understand how the uncertainty principle emerges from Fourier analysis

---

## Preview: Day 343

**Week 49 Review: Hilbert Space Foundations**

Tomorrow we consolidate the week's material on Hilbert space, covering:
- Comprehensive review of finite vs. infinite-dimensional spaces
- Discrete vs. continuous spectra comparison
- Dirac notation mastery check
- Practice with representation transformations
- Preparation for measurement theory (Week 50)

---

## References

1. **Shankar, R.** *Principles of Quantum Mechanics*, 2nd ed. (Springer, 1994), Ch. 1.10-1.11: "Continuous Spectra—Position and Momentum"

2. **Sakurai, J.J. & Napolitano, J.** *Modern Quantum Mechanics*, 3rd ed. (Cambridge, 2020), Ch. 1.6: "Position, Momentum, and Translation"

3. **Griffiths, D.J. & Schroeter, D.F.** *Introduction to Quantum Mechanics*, 3rd ed. (Cambridge, 2018), Ch. 3.5-3.6: "Continuous Spectra"

4. **Ballentine, L.E.** *Quantum Mechanics: A Modern Development*, 2nd ed. (World Scientific, 2014), Ch. 1.4: "Rigged Hilbert Space"

---

*"The wave function is but a shadow of the state vector, projected onto the screen of position space."* — Adapted from Dirac's philosophy
