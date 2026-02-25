# Day 349: Fourier Transform Connection

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Fourier Transform in Quantum Mechanics |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving and Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: FFT and Momentum Space |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Derive the Fourier transform relationship between $\psi(x)$ and $\phi(p)$
2. Transform wave functions between position and momentum space
3. Apply Parseval's theorem to verify probability conservation
4. Interpret the momentum space wave function physically
5. Calculate expectation values in momentum representation
6. Connect the uncertainty principle to Fourier analysis

---

## Required Reading

### Primary Texts
- **Shankar, Chapter 4.3**: The Connection Between Position and Momentum Representations (pp. 181-190)
- **Sakurai, Chapter 1.7**: Wave Functions in Position and Momentum Space (pp. 55-60)
- **Griffiths, Chapter 3.4**: The Free Particle (pp. 105-110)

### Supplementary Reading
- **Cohen-Tannoudji, Chapter II.E**: Change of Representation
- **Arfken & Weber, Chapter 20**: Fourier Transforms

---

## Core Content: Theory and Concepts

### 1. The Fundamental Connection

Yesterday we established:
- Position eigenstate: $\langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$
- Momentum eigenstate: $\langle p|x\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{-ipx/\hbar}$

This is the **kernel of the Fourier transform**.

### 2. Transforming from Position to Momentum Space

The momentum space wave function is:

$$\phi(p) = \langle p|\psi\rangle$$

Insert completeness relation $\int |x\rangle\langle x| dx = \hat{I}$:

$$\phi(p) = \int_{-\infty}^{\infty} \langle p|x\rangle\langle x|\psi\rangle dx = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\hbar}}e^{-ipx/\hbar}\psi(x) dx$$

$$\boxed{\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty} e^{-ipx/\hbar}\psi(x) dx}$$

This is the **Fourier transform** of $\psi(x)$ (with $k = p/\hbar$).

### 3. Transforming from Momentum to Position Space

Similarly:

$$\psi(x) = \langle x|\psi\rangle = \int_{-\infty}^{\infty} \langle x|p\rangle\langle p|\psi\rangle dp$$

$$\boxed{\psi(x) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty} e^{ipx/\hbar}\phi(p) dp}$$

This is the **inverse Fourier transform**.

### 4. Mathematical Form of Fourier Transforms

**Standard Physics Convention:**

With $k = p/\hbar$, the transforms become:

$$\tilde{\psi}(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-ikx}\psi(x) dx$$

$$\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{ikx}\tilde{\psi}(k) dk$$

**Quantum Mechanics Convention:**

$$\phi(p) = \mathcal{F}[\psi(x)] = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x) dx$$

$$\psi(x) = \mathcal{F}^{-1}[\phi(p)] = \frac{1}{\sqrt{2\pi\hbar}}\int e^{ipx/\hbar}\phi(p) dp$$

### 5. Parseval's Theorem

**Theorem (Parseval/Plancherel):** The Fourier transform preserves the inner product:

$$\int_{-\infty}^{\infty} |\psi(x)|^2 dx = \int_{-\infty}^{\infty} |\phi(p)|^2 dp$$

**Proof:**

$$\int |\psi(x)|^2 dx = \langle\psi|\psi\rangle = \int \langle\psi|p\rangle\langle p|\psi\rangle dp = \int |\phi(p)|^2 dp$$ $\blacksquare$

**Physical Meaning:** Total probability is conserved whether we compute it in position space or momentum space:

$$\boxed{\int_{-\infty}^{\infty} |\psi(x)|^2 dx = \int_{-\infty}^{\infty} |\phi(p)|^2 dp = 1}$$

### 6. Physical Interpretation of Momentum Space

**$|\phi(p)|^2 dp$** = probability of measuring momentum between $p$ and $p + dp$

Just as $|\psi(x)|^2$ is the position probability density, $|\phi(p)|^2$ is the **momentum probability density**.

**Expectation values in momentum space:**

$$\langle p\rangle = \int_{-\infty}^{\infty} p|\phi(p)|^2 dp$$

$$\langle p^2\rangle = \int_{-\infty}^{\infty} p^2|\phi(p)|^2 dp$$

This is often **simpler** than computing in position space!

### 7. Key Fourier Transform Pairs

| Position Space $\psi(x)$ | Momentum Space $\phi(p)$ |
|--------------------------|-------------------------|
| $e^{-x^2/2a^2}$ (Gaussian) | $\sim e^{-p^2 a^2/2\hbar^2}$ (Gaussian) |
| $e^{ip_0x/\hbar}\psi(x)$ (momentum boost) | $\phi(p - p_0)$ (shifted) |
| $\psi(x - x_0)$ (position shift) | $e^{-ipx_0/\hbar}\phi(p)$ |
| $\delta(x)$ | $\frac{1}{\sqrt{2\pi\hbar}}$ (constant) |
| Constant | $\sqrt{2\pi\hbar}\delta(p)$ |
| $\frac{1}{\sqrt{a}}$ for $|x| < a/2$ (box) | $\sqrt{\frac{a}{2\pi\hbar}}\text{sinc}\left(\frac{pa}{2\hbar}\right)$ |

### 8. Fourier Transform of Gaussian Wave Packet

**Important Example:** The Gaussian wave function

$$\psi(x) = \left(\frac{1}{\pi a^2}\right)^{1/4} e^{-x^2/2a^2}$$

transforms to:

$$\phi(p) = \left(\frac{a^2}{\pi\hbar^2}\right)^{1/4} e^{-p^2 a^2/2\hbar^2}$$

**Observation:** Both are Gaussians!
- Position width: $\Delta x = a/\sqrt{2}$
- Momentum width: $\Delta p = \hbar/(a\sqrt{2})$
- Product: $\Delta x \cdot \Delta p = \hbar/2$ (minimum uncertainty)

**The Gaussian is the ONLY function that is its own Fourier transform** (up to scaling).

### 9. The Uncertainty Principle from Fourier Analysis

**Mathematical Theorem:** For any function $f$ and its Fourier transform $\tilde{f}$:

$$\Delta x \cdot \Delta k \geq \frac{1}{2}$$

where $\Delta x$ and $\Delta k$ are the standard deviations.

**Quantum Mechanics:** With $p = \hbar k$:

$$\boxed{\Delta x \cdot \Delta p \geq \frac{\hbar}{2}}$$

**Physical Interpretation:**
- A narrow position distribution ($\Delta x$ small) requires many momentum components (large $\Delta p$)
- A well-defined momentum (small $\Delta p$) requires a broad spatial extent (large $\Delta x$)
- This is a property of waves, not a limitation of measurement!

### 10. Operators in Momentum Representation

In momentum space, operators take different forms:

**Momentum operator:**
$$\hat{p}\phi(p) = p\phi(p)$$
(Just multiplication - momentum representation diagonalizes $\hat{p}$!)

**Position operator:**
$$\hat{x}\phi(p) = i\hbar\frac{d\phi}{dp}$$

**Kinetic energy:**
$$\hat{T}\phi(p) = \frac{p^2}{2m}\phi(p)$$

**Potential energy:** (convolution in momentum space)
$$\langle p|\hat{V}|\psi\rangle = \int V(p - p')\phi(p') dp'$$

where $V(p)$ is the Fourier transform of $V(x)$.

### 11. Time Evolution in Momentum Space

The Schrodinger equation in momentum space for free particle:

$$i\hbar\frac{\partial\phi}{\partial t} = \frac{p^2}{2m}\phi(p, t)$$

**Solution:**
$$\phi(p, t) = \phi(p, 0)e^{-ip^2t/2m\hbar}$$

Each momentum component evolves with its own phase!

### 12. Wave Packet Spreading

A Gaussian wave packet:
- Maintains Gaussian shape in momentum space (unchanged width)
- Spreads in position space over time

**Physical reason:** Different momentum components travel at different speeds ($v = p/m$), causing the packet to spread.

$$\Delta x(t) = \sqrt{(\Delta x_0)^2 + \frac{\hbar^2 t^2}{4m^2(\Delta x_0)^2}}$$

---

## Quantum Computing Connection

### Quantum Fourier Transform (QFT)

The **Quantum Fourier Transform** is the discrete analogue of the continuous Fourier transform.

For $N = 2^n$ basis states:

$$|j\rangle \xrightarrow{QFT} \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle$$

**In matrix form:**

$$F_N = \frac{1}{\sqrt{N}}\begin{pmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{N-1} \\
1 & \omega^2 & \omega^4 & \cdots & \omega^{2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{N-1} & \omega^{2(N-1)} & \cdots & \omega^{(N-1)^2}
\end{pmatrix}$$

where $\omega = e^{2\pi i/N}$.

### QFT Circuit

The QFT can be implemented efficiently with:
- Hadamard gates
- Controlled phase gates

For $n$ qubits: $O(n^2)$ gates (exponentially faster than classical FFT for superposition states!)

### Applications of QFT

1. **Shor's Algorithm:** Period finding via QFT leads to factoring
2. **Phase Estimation:** Extract eigenvalues of unitary operators
3. **Quantum Simulation:** Transform between position and momentum bases

### Position-Momentum Duality in Quantum Computing

In continuous-variable quantum computing:

$$\hat{X} \leftrightarrow \hat{P}$$

The QFT (implemented via squeezing and beam splitters) maps:
- Position eigenstates $|x\rangle$ to momentum eigenstates $|p\rangle$
- Homodyne measurement in one quadrature to measurement in conjugate quadrature

---

## Worked Examples

### Example 1: Fourier Transform of Gaussian

**Problem:** Find the momentum space wave function for:
$$\psi(x) = \left(\frac{1}{\pi a^2}\right)^{1/4} e^{-x^2/2a^2}$$

**Solution:**

$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty} e^{-ipx/\hbar}\left(\frac{1}{\pi a^2}\right)^{1/4} e^{-x^2/2a^2} dx$$

$$= \left(\frac{1}{\pi a^2}\right)^{1/4}\frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty} e^{-x^2/2a^2 - ipx/\hbar} dx$$

Complete the square in the exponent:

$$-\frac{x^2}{2a^2} - \frac{ipx}{\hbar} = -\frac{1}{2a^2}\left(x + \frac{ipa^2}{\hbar}\right)^2 - \frac{p^2a^2}{2\hbar^2}$$

$$\phi(p) = \left(\frac{1}{\pi a^2}\right)^{1/4}\frac{1}{\sqrt{2\pi\hbar}}e^{-p^2a^2/2\hbar^2}\int_{-\infty}^{\infty} e^{-(x + ipa^2/\hbar)^2/2a^2} dx$$

The integral is $\sqrt{2\pi}a$, so:

$$\phi(p) = \left(\frac{1}{\pi a^2}\right)^{1/4}\frac{\sqrt{2\pi}a}{\sqrt{2\pi\hbar}}e^{-p^2a^2/2\hbar^2}$$

$$\boxed{\phi(p) = \left(\frac{a^2}{\pi\hbar^2}\right)^{1/4} e^{-p^2a^2/2\hbar^2}}$$

**Verification:** This is also a normalized Gaussian in $p$.

### Example 2: Momentum Space for Boosted Gaussian

**Problem:** Find $\phi(p)$ for $\psi(x) = \left(\frac{1}{\pi a^2}\right)^{1/4} e^{-x^2/2a^2}e^{ip_0x/\hbar}$

**Solution:**

Using the Fourier shift theorem: if $\psi(x) \to \phi(p)$, then $e^{ip_0x/\hbar}\psi(x) \to \phi(p - p_0)$

From Example 1:
$$\phi(p) = \left(\frac{a^2}{\pi\hbar^2}\right)^{1/4} e^{-(p-p_0)^2a^2/2\hbar^2}$$

**Physical interpretation:** The wave packet is now centered at momentum $p_0$, corresponding to classical motion with velocity $v = p_0/m$.

### Example 3: Square Wave Function

**Problem:** A particle is confined to a region of width $L$ with uniform probability:
$$\psi(x) = \begin{cases} \frac{1}{\sqrt{L}} & |x| < L/2 \\ 0 & |x| > L/2 \end{cases}$$

Find $\phi(p)$ and $\langle p^2\rangle$.

**Solution:**

$$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int_{-L/2}^{L/2} e^{-ipx/\hbar}\frac{1}{\sqrt{L}} dx$$

$$= \frac{1}{\sqrt{2\pi\hbar L}}\left[\frac{e^{-ipx/\hbar}}{-ip/\hbar}\right]_{-L/2}^{L/2}$$

$$= \frac{1}{\sqrt{2\pi\hbar L}}\cdot\frac{\hbar}{ip}\left(e^{-ipL/2\hbar} - e^{ipL/2\hbar}\right)$$

$$= \frac{1}{\sqrt{2\pi\hbar L}}\cdot\frac{\hbar}{ip}\cdot(-2i)\sin\left(\frac{pL}{2\hbar}\right)$$

$$\boxed{\phi(p) = \sqrt{\frac{L}{2\pi\hbar}}\frac{\sin(pL/2\hbar)}{pL/2\hbar} = \sqrt{\frac{L}{2\pi\hbar}}\text{sinc}\left(\frac{pL}{2\hbar}\right)}$$

**For $\langle p^2\rangle$:**

$$\langle p^2\rangle = \int_{-\infty}^{\infty} p^2|\phi(p)|^2 dp$$

This integral diverges! The sharp edges in position space require infinite momentum components.

Alternatively, compute in position space:

$$\langle p^2\rangle = -\hbar^2\int \psi^*\frac{d^2\psi}{dx^2} dx$$

The second derivative gives delta functions at the boundaries, leading to divergence.

**Lesson:** Square wave functions have infinite kinetic energy - they're unphysical idealizations.

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Verify that $\phi(p) = \sqrt{\frac{a}{\hbar\sqrt{\pi}}}e^{-p^2a^2/\hbar^2}$ is normalized.

**Answer:** Use $\int e^{-\alpha x^2}dx = \sqrt{\pi/\alpha}$ with $\alpha = a^2/\hbar^2$.

---

**Problem 1.2:** For $\psi(x) = Ae^{-|x|/a}$, find $\phi(p)$.

*Hint: Split integral into $x < 0$ and $x > 0$ regions.*

**Answer:** $\phi(p) = \frac{\sqrt{2a^3/\pi}}{\hbar}\cdot\frac{1}{1 + p^2a^2/\hbar^2}$ (Lorentzian)

---

**Problem 1.3:** Show that if $\psi(x)$ is real and even, then $\phi(p)$ is real and even.

**Answer:** For real $\psi$: $\phi(-p) = \int e^{ipx/\hbar}\psi(x)dx = [\phi(p)]^*$. If $\psi$ even, the $\sin$ terms vanish, leaving only real $\cos$ terms.

---

### Level 2: Intermediate

**Problem 2.1:** A Gaussian wave packet has width $\sigma$ in position space. Find its width in momentum space and verify the uncertainty relation.

**Solution:**
Position uncertainty: $\Delta x = \sigma/\sqrt{2}$
From Fourier transform: momentum width is $\hbar/\sigma$
Momentum uncertainty: $\Delta p = \hbar/(\sigma\sqrt{2})$
Product: $\Delta x \cdot \Delta p = \hbar/2$ $\checkmark$

---

**Problem 2.2:** The momentum space wave function is $\phi(p) = Ne^{-|p|/p_0}$. Find:
(a) The normalization $N$
(b) The position space wave function $\psi(x)$
(c) $\langle p^2\rangle$

**Solution:**
(a) $N = 1/\sqrt{2p_0}$
(b) $\psi(x) = \sqrt{\frac{2}{\pi}}\frac{p_0/\hbar}{1 + (p_0 x/\hbar)^2}$ (Lorentzian)
(c) $\langle p^2\rangle = 2p_0^2$

---

**Problem 2.3:** Show that the Fourier transform of $\psi(x - x_0)$ is $e^{-ipx_0/\hbar}\phi(p)$.

**Solution:**
$$\mathcal{F}[\psi(x-x_0)] = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x-x_0)dx$$

Substitute $u = x - x_0$:
$$= \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ip(u+x_0)/\hbar}\psi(u)du = e^{-ipx_0/\hbar}\phi(p)$$ $\checkmark$

---

### Level 3: Challenging

**Problem 3.1:** Derive the position-momentum uncertainty relation using only properties of the Fourier transform.

**Solution:**

Define $(\Delta x)^2 = \int x^2|\psi|^2 dx$ and $(\Delta p)^2 = \int p^2|\phi|^2 dp$.

Using the Fourier derivative property: $\mathcal{F}[x\psi] = i\hbar\frac{d\phi}{dp}$

And Parseval: $\int |x\psi|^2 dx = \int \hbar^2|d\phi/dp|^2 dp$

The Cauchy-Schwarz inequality gives:
$$\left|\int p\phi^*\frac{d\phi}{dp}dp\right|^2 \leq \int |p\phi|^2 dp \cdot \int \left|\frac{d\phi}{dp}\right|^2 dp$$

After algebra (using $\int p\phi^*d\phi/dp \, dp = -\frac{1}{2}\int|\phi|^2 dp$):

$$\frac{1}{4} \leq (\Delta p)^2 \cdot \frac{(\Delta x)^2}{\hbar^2}$$

$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

---

**Problem 3.2:** A wave packet at $t = 0$ has $\phi(p, 0) = N\text{rect}(p/p_0)$ (uniform in $[-p_0/2, p_0/2]$, zero outside). Find:
(a) $\psi(x, 0)$
(b) The time evolution of $\phi(p, t)$
(c) Does $\psi(x, t)$ spread?

**Solution:**
(a) $\psi(x, 0) \propto \text{sinc}(p_0 x/2\hbar)$

(b) $\phi(p, t) = \phi(p, 0)e^{-ip^2t/2m\hbar}$ - the magnitude $|\phi|$ is unchanged!

(c) Yes, $\psi(x, t)$ spreads because different momentum components acquire different phases, causing interference patterns that broaden the envelope.

---

**Problem 3.3:** Prove that for any wave function:
$$\langle xp + px\rangle = 2\text{Re}\left[\int \psi^* x(-i\hbar\frac{d\psi}{dx})dx\right]$$

and relate this to the rate of change of $\langle x^2\rangle$.

**Solution:**

By Ehrenfest: $\frac{d\langle x^2\rangle}{dt} = \frac{1}{i\hbar}\langle[x^2, H]\rangle$

For free particle $H = p^2/2m$:
$$[x^2, p^2] = x[x, p^2] + [x, p^2]x = 2i\hbar(xp + px)$$

So $\frac{d\langle x^2\rangle}{dt} = \frac{1}{m}\langle xp + px\rangle$

---

## Computational Lab: Fourier Transforms in Quantum Mechanics

```python
"""
Day 349 Computational Lab: Fourier Transform Connection
Topics: FFT, position-momentum duality, wave packet evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift
from scipy.special import erf

# Physical constants (hbar = 1 for simplicity)
hbar = 1.0

# ============================================
# Part 1: Basic Fourier Transform of Gaussian
# ============================================

print("=" * 50)
print("Part 1: Fourier Transform of Gaussian")
print("=" * 50)

# Create position grid
N = 2048  # Number of points
L = 20.0  # Domain size
dx = L / N
x = np.linspace(-L/2, L/2, N)

# Gaussian wave function
a = 1.0  # Width parameter
psi_x = (1/(np.pi * a**2))**0.25 * np.exp(-x**2 / (2*a**2))

# Numerical Fourier transform
# Note: numpy FFT uses different convention, need to scale
dp = 2 * np.pi * hbar / L
p = fftfreq(N, dx) * 2 * np.pi * hbar  # Momentum values
p = fftshift(p)

# Compute FT (with proper normalization)
phi_p_numerical = fftshift(fft(psi_x)) * dx / np.sqrt(2 * np.pi * hbar)

# Analytical result
phi_p_analytical = (a**2 / (np.pi * hbar**2))**0.25 * np.exp(-p**2 * a**2 / (2*hbar**2))

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Position space
ax1 = axes[0, 0]
ax1.plot(x, np.abs(psi_x)**2, 'b-', linewidth=2)
ax1.fill_between(x, 0, np.abs(psi_x)**2, alpha=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel(r'$|\psi(x)|^2$')
ax1.set_title('Position Space')
ax1.set_xlim(-5, 5)
ax1.grid(True, alpha=0.3)

# Momentum space
ax2 = axes[0, 1]
ax2.plot(p, np.abs(phi_p_numerical)**2, 'b-', linewidth=2, label='Numerical FFT')
ax2.plot(p, np.abs(phi_p_analytical)**2, 'r--', linewidth=2, label='Analytical')
ax2.fill_between(p, 0, np.abs(phi_p_analytical)**2, alpha=0.3)
ax2.set_xlabel('p')
ax2.set_ylabel(r'$|\phi(p)|^2$')
ax2.set_title('Momentum Space')
ax2.set_xlim(-5/a, 5/a)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Log scale comparison
ax3 = axes[1, 0]
ax3.semilogy(x, np.abs(psi_x)**2, 'b-', linewidth=2)
ax3.set_xlabel('x')
ax3.set_ylabel(r'$|\psi(x)|^2$ (log scale)')
ax3.set_title('Position Space (log scale)')
ax3.set_xlim(-10, 10)
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.semilogy(p, np.abs(phi_p_numerical)**2, 'b-', linewidth=2)
ax4.set_xlabel('p')
ax4.set_ylabel(r'$|\phi(p)|^2$ (log scale)')
ax4.set_title('Momentum Space (log scale)')
ax4.set_xlim(-10/a, 10/a)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify Parseval's theorem
norm_x = np.sum(np.abs(psi_x)**2) * dx
norm_p = np.sum(np.abs(phi_p_numerical)**2) * dp
print(f"\nPosition space norm: {norm_x:.6f}")
print(f"Momentum space norm: {norm_p:.6f}")
print(f"Parseval's theorem verified: {np.isclose(norm_x, norm_p, rtol=0.01)}")

# ============================================
# Part 2: Width Relations and Uncertainty
# ============================================

print("\n" + "=" * 50)
print("Part 2: Width Relations and Uncertainty Principle")
print("=" * 50)

# Calculate widths for different a values
a_values = [0.5, 1.0, 2.0, 4.0]
fig, axes = plt.subplots(len(a_values), 2, figsize=(12, 3*len(a_values)))

for i, a_val in enumerate(a_values):
    # Position space
    psi = (1/(np.pi * a_val**2))**0.25 * np.exp(-x**2 / (2*a_val**2))

    # Momentum space
    phi = fftshift(fft(psi)) * dx / np.sqrt(2 * np.pi * hbar)

    # Calculate uncertainties
    delta_x = np.sqrt(np.sum(x**2 * np.abs(psi)**2 * dx))
    delta_p = np.sqrt(np.sum(p**2 * np.abs(phi)**2 * dp))
    product = delta_x * delta_p

    # Plot
    axes[i, 0].plot(x, np.abs(psi)**2, 'b-', linewidth=2)
    axes[i, 0].fill_between(x, 0, np.abs(psi)**2, alpha=0.3)
    axes[i, 0].set_xlim(-8, 8)
    axes[i, 0].set_ylabel(r'$|\psi|^2$')
    axes[i, 0].set_title(f'a = {a_val}, $\Delta x$ = {delta_x:.3f}')
    axes[i, 0].grid(True, alpha=0.3)

    axes[i, 1].plot(p, np.abs(phi)**2, 'r-', linewidth=2)
    axes[i, 1].fill_between(p, 0, np.abs(phi)**2, alpha=0.3, color='red')
    axes[i, 1].set_xlim(-8, 8)
    axes[i, 1].set_ylabel(r'$|\phi|^2$')
    axes[i, 1].set_title(f'$\Delta p$ = {delta_p:.3f}, $\Delta x \Delta p$ = {product:.4f}')
    axes[i, 1].grid(True, alpha=0.3)

    print(f"a = {a_val}: Delta_x = {delta_x:.4f}, Delta_p = {delta_p:.4f}, product = {product:.4f}")

axes[-1, 0].set_xlabel('x')
axes[-1, 1].set_xlabel('p')
plt.tight_layout()
plt.savefig('uncertainty_widths.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nMinimum uncertainty (hbar/2) = {hbar/2:.4f}")

# ============================================
# Part 3: Moving Wave Packet
# ============================================

print("\n" + "=" * 50)
print("Part 3: Wave Packet with Initial Momentum")
print("=" * 50)

a = 1.0
p0 = 3.0  # Initial momentum

# Wave packet with momentum
psi_moving = (1/(np.pi * a**2))**0.25 * np.exp(-x**2 / (2*a**2)) * np.exp(1j * p0 * x / hbar)

# Fourier transform
phi_moving = fftshift(fft(psi_moving)) * dx / np.sqrt(2 * np.pi * hbar)

# Expected: shifted Gaussian centered at p0
phi_expected = (a**2 / (np.pi * hbar**2))**0.25 * np.exp(-(p - p0)**2 * a**2 / (2*hbar**2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, psi_moving.real, 'b-', linewidth=2, label=r'Re($\psi$)')
ax1.plot(x, psi_moving.imag, 'r--', linewidth=2, label=r'Im($\psi$)')
ax1.plot(x, np.abs(psi_moving), 'k-', linewidth=1, label=r'$|\psi|$')
ax1.set_xlabel('x')
ax1.set_title(f'Position Space (p0 = {p0})')
ax1.legend()
ax1.set_xlim(-5, 5)
ax1.grid(True, alpha=0.3)

ax2.plot(p, np.abs(phi_moving)**2, 'b-', linewidth=2, label='Numerical')
ax2.plot(p, np.abs(phi_expected)**2, 'r--', linewidth=2, label='Expected')
ax2.axvline(x=p0, color='g', linestyle=':', label=f'p0 = {p0}')
ax2.set_xlabel('p')
ax2.set_title('Momentum Space (shifted Gaussian)')
ax2.legend()
ax2.set_xlim(-5, 10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('moving_wavepacket_fourier.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate mean momentum
p_mean = np.sum(p * np.abs(phi_moving)**2 * dp).real
print(f"Expected <p> = {p0}")
print(f"Calculated <p> = {p_mean:.4f}")

# ============================================
# Part 4: Time Evolution of Wave Packet
# ============================================

print("\n" + "=" * 50)
print("Part 4: Time Evolution (Wave Packet Spreading)")
print("=" * 50)

m = 1.0  # Mass
a = 1.0
p0 = 2.0

# Initial state
psi_0 = (1/(np.pi * a**2))**0.25 * np.exp(-x**2 / (2*a**2)) * np.exp(1j * p0 * x / hbar)
phi_0 = fftshift(fft(psi_0)) * dx / np.sqrt(2 * np.pi * hbar)

# Time evolution in momentum space
times = [0, 1, 2, 4, 8]
fig, axes = plt.subplots(2, len(times), figsize=(15, 6))

for i, t in enumerate(times):
    # Evolve in momentum space
    phase = np.exp(-1j * p**2 * t / (2 * m * hbar))
    phi_t = phi_0 * phase

    # Transform back to position space
    psi_t = ifft(fftshift(phi_t)) * N * dp / np.sqrt(2 * np.pi * hbar)

    # Calculate width
    prob_density = np.abs(psi_t)**2
    x_mean = np.sum(x * prob_density * dx).real
    x2_mean = np.sum(x**2 * prob_density * dx).real
    delta_x = np.sqrt(x2_mean - x_mean**2)

    # Classical position
    x_classical = p0 * t / m

    # Position space
    axes[0, i].plot(x, np.abs(psi_t)**2, 'b-', linewidth=2)
    axes[0, i].axvline(x=x_classical, color='r', linestyle='--', label='Classical')
    axes[0, i].set_xlim(-10, 30)
    axes[0, i].set_title(f't = {t}, $\Delta x$ = {delta_x:.2f}')
    axes[0, i].set_xlabel('x')
    if i == 0:
        axes[0, i].set_ylabel(r'$|\psi(x,t)|^2$')
        axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)

    # Momentum space (magnitude unchanged)
    axes[1, i].plot(p, np.abs(phi_t)**2, 'r-', linewidth=2)
    axes[1, i].set_xlim(-5, 10)
    axes[1, i].set_xlabel('p')
    if i == 0:
        axes[1, i].set_ylabel(r'$|\phi(p,t)|^2$')
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle('Wave Packet Time Evolution: Position spreads, Momentum unchanged', fontsize=12)
plt.tight_layout()
plt.savefig('wavepacket_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 5: Different Wave Functions
# ============================================

print("\n" + "=" * 50)
print("Part 5: Fourier Transforms of Various Functions")
print("=" * 50)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# 1. Square well
psi_square = np.where(np.abs(x) < 2, 1/2, 0)
phi_square = fftshift(fft(psi_square)) * dx / np.sqrt(2 * np.pi * hbar)

axes[0, 0].plot(x, np.abs(psi_square)**2, 'b-', linewidth=2)
axes[0, 0].set_title('Square Well: Position Space')
axes[0, 0].set_xlim(-5, 5)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(p, np.abs(phi_square)**2, 'r-', linewidth=2)
axes[0, 1].set_title('Square Well: Momentum Space (sinc)')
axes[0, 1].set_xlim(-10, 10)
axes[0, 1].grid(True, alpha=0.3)

# 2. Double Gaussian
psi_double = 0.5 * ((1/(np.pi * 0.5**2))**0.25 * np.exp(-(x-2)**2 / (2*0.5**2)) +
                    (1/(np.pi * 0.5**2))**0.25 * np.exp(-(x+2)**2 / (2*0.5**2)))
phi_double = fftshift(fft(psi_double)) * dx / np.sqrt(2 * np.pi * hbar)

axes[1, 0].plot(x, np.abs(psi_double)**2, 'b-', linewidth=2)
axes[1, 0].set_title('Double Gaussian: Position Space')
axes[1, 0].set_xlim(-6, 6)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(p, np.abs(phi_double)**2, 'r-', linewidth=2)
axes[1, 1].set_title('Double Gaussian: Momentum Space (interference)')
axes[1, 1].set_xlim(-10, 10)
axes[1, 1].grid(True, alpha=0.3)

# 3. Exponential decay
psi_exp = np.exp(-np.abs(x)) / np.sqrt(2)
phi_exp = fftshift(fft(psi_exp)) * dx / np.sqrt(2 * np.pi * hbar)

axes[2, 0].plot(x, np.abs(psi_exp)**2, 'b-', linewidth=2)
axes[2, 0].set_title('Exponential: Position Space')
axes[2, 0].set_xlim(-5, 5)
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(p, np.abs(phi_exp)**2, 'r-', linewidth=2)
axes[2, 1].set_title('Exponential: Momentum Space (Lorentzian)')
axes[2, 1].set_xlim(-10, 10)
axes[2, 1].grid(True, alpha=0.3)

for ax in axes[:, 0]:
    ax.set_xlabel('x')
for ax in axes[:, 1]:
    ax.set_xlabel('p')

plt.tight_layout()
plt.savefig('various_fourier_transforms.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Part 6: Quantum Fourier Transform (Discrete)
# ============================================

print("\n" + "=" * 50)
print("Part 6: Quantum Fourier Transform Matrix")
print("=" * 50)

def qft_matrix(n):
    """Generate n-dimensional QFT matrix"""
    N = n
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    return F

# 4-dimensional QFT (2 qubits)
F4 = qft_matrix(4)

print("QFT matrix for 2 qubits (N=4):")
print(np.round(F4, 3))

# Verify unitarity
print(f"\nUnitary check (F*F^dag = I): {np.allclose(F4 @ F4.conj().T, np.eye(4))}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

im1 = ax1.imshow(np.abs(F4), cmap='Blues')
ax1.set_title('|QFT Matrix| (N=4)')
ax1.set_xlabel('Input state')
ax1.set_ylabel('Output state')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(np.angle(F4), cmap='twilight')
ax2.set_title('Phase of QFT Matrix (N=4)')
ax2.set_xlabel('Input state')
ax2.set_ylabel('Output state')
plt.colorbar(im2, ax=ax2, label='Phase (radians)')

plt.tight_layout()
plt.savefig('qft_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("Lab Complete!")
print("=" * 50)
```

---

## Summary

### Key Formulas

| Transform | Forward | Inverse |
|-----------|---------|---------|
| Position to Momentum | $\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x)dx$ | $\psi(x) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{ipx/\hbar}\phi(p)dp$ |
| Parseval's Theorem | $\int|\psi(x)|^2dx = \int|\phi(p)|^2dp$ | |
| Shift Theorem | $\psi(x-x_0) \leftrightarrow e^{-ipx_0/\hbar}\phi(p)$ | $e^{ip_0x/\hbar}\psi(x) \leftrightarrow \phi(p-p_0)$ |
| Derivative | $\frac{d\psi}{dx} \leftrightarrow \frac{ip}{\hbar}\phi(p)$ | |

### Key Transform Pairs

| Position $\psi(x)$ | Momentum $\phi(p)$ |
|-------------------|-------------------|
| Gaussian $e^{-x^2/2a^2}$ | Gaussian $e^{-p^2a^2/2\hbar^2}$ |
| Delta $\delta(x)$ | Constant $1/\sqrt{2\pi\hbar}$ |
| Box function | Sinc function |
| Exponential $e^{-|x|/a}$ | Lorentzian $1/(1+p^2a^2/\hbar^2)$ |

### Key Takeaways

1. **Position and momentum wave functions** are Fourier transforms of each other.

2. **Parseval's theorem** ensures probability is conserved between representations.

3. **Gaussian is special:** It's the only function that is its own Fourier transform (minimum uncertainty).

4. **Uncertainty principle** follows from Fourier analysis mathematics.

5. **Momentum space** is often simpler for kinetic energy calculations.

6. **Wave packet spreading** occurs because different momentum components travel at different speeds.

---

## Daily Checklist

- [ ] Read Shankar 4.3 and Sakurai 1.7 on Fourier transforms
- [ ] Derive the Fourier transform of a Gaussian wave packet
- [ ] Verify Parseval's theorem for a specific example
- [ ] Work through all three examples
- [ ] Complete Level 1 and 2 practice problems
- [ ] Attempt at least one Level 3 problem
- [ ] Run computational lab and visualize transforms
- [ ] Understand the QFT connection to continuous Fourier transform
- [ ] Relate uncertainty principle to Fourier width theorem

---

## Preview: Tomorrow's Topics

**Day 350: Week 50 Review and Qiskit Lab**

Tomorrow we consolidate our understanding with:

- Comprehensive review of Week 50 material
- Practice exam covering all observable concepts
- Extensive Qiskit lab implementing:
  - Measurement in different bases
  - Born rule verification
  - Measurement statistics
  - Multi-qubit observables

**Preparation:** Review all week's material and prepare questions.

---

**References:**
- Shankar, R. (1994). Principles of Quantum Mechanics, Chapter 4
- Sakurai, J.J. (2017). Modern Quantum Mechanics, Chapter 1.7
- Griffiths, D.J. (2018). Introduction to Quantum Mechanics, Chapter 3.4
- Arfken, G.B. & Weber, H.J. (2012). Mathematical Methods for Physicists, Chapter 20
- Nielsen, M.A. & Chuang, I.L. (2010). Quantum Computation and Quantum Information, Chapter 5
