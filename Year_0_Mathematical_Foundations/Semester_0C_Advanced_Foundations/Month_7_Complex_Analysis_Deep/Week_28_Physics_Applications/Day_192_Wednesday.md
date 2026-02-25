# Day 192: Scattering Theory Applications

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: S-Matrix Analyticity |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Poles, Resonances, Levinson |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 192, you will be able to:

1. Understand the S-matrix and its analytic properties
2. Relate S-matrix poles to bound states and resonances
3. Compute resonance widths from pole positions
4. Apply Levinson's theorem to count bound states
5. Calculate scattering phase shifts from complex analysis
6. Connect to experimental observables

---

## Core Content

### 1. The S-Matrix

**Definition:** The scattering matrix relates incoming and outgoing states:
$$|\psi_{\text{out}}\rangle = S |\psi_{\text{in}}\rangle$$

For 1D scattering with transmission $t$ and reflection $r$:
$$S = \begin{pmatrix} t & r' \\ r & t' \end{pmatrix}$$

**Unitarity:** $S^\dagger S = I$ (probability conservation)

**For spherically symmetric potential (partial waves):**
$$S_\ell(k) = e^{2i\delta_\ell(k)}$$

where $\delta_\ell(k)$ is the phase shift.

### 2. Analyticity of S(k)

**The S-matrix is meromorphic** in the complex $k$-plane:

| Location | Physical Meaning |
|----------|-----------------|
| Real axis | Physical scattering |
| Positive imaginary axis | Bound states |
| Lower half-plane | Resonances |
| Branch cuts | Thresholds |

**Key property:** $S(k) S(-k) = 1$ (from time-reversal)

### 3. Bound States as Poles

**Bound state condition:** Normalizable wave function at $E < 0$.

For $E = -\hbar^2\kappa^2/2m$ (bound), $k = i\kappa$ (positive imaginary).

**Near a bound state pole:**
$$S(k) \approx \frac{i\kappa + k}{i\kappa - k} \cdot (\text{regular})$$

or equivalently:
$$S(k) \approx \frac{iR}{k - i\kappa}$$

The residue $R$ is related to the wave function normalization.

### 4. Resonances

**Resonance:** Quasi-bound state with finite lifetime.

**Pole location:** $k = k_R - i\Gamma/2$ where:
- $k_R$: resonance momentum
- $\Gamma$: width (inverse lifetime)

**Energy:** $E = E_R - i\Gamma_E/2$ where:
$$E_R = \frac{\hbar^2 k_R^2}{2m}, \quad \Gamma_E = \frac{\hbar^2 k_R \Gamma}{m}$$

**Breit-Wigner formula:** Near resonance:
$$\sigma(E) = \frac{\pi}{k^2}\frac{\Gamma_E^2/4}{(E - E_R)^2 + \Gamma_E^2/4}$$

### 5. Levinson's Theorem

**Theorem:** The phase shift at zero energy is related to bound states:
$$\boxed{\delta(0) - \delta(\infty) = n_B \pi}$$

where $n_B$ is the number of bound states.

**Proof sketch:** The phase shift $\delta(k)$ is the argument of $S(k) = e^{2i\delta(k)}$.

$$\delta(k) = \frac{1}{2i}\ln S(k)$$

The change in $\delta$ from $k = 0$ to $k = \infty$ equals $\pi$ times the number of poles crossed on the positive imaginary axis (bound states).

**Application:** Count bound states by measuring low-energy scattering!

### 6. Jost Functions

**Definition:** The Jost function $f(k)$ is analytic in upper half-plane with:
$$S(k) = \frac{f(-k)}{f(k)}$$

**Zeros of $f(k)$:**
- On positive imaginary axis → bound states
- In lower half-plane → resonances

**Advantage:** $f(k)$ has no poles, only zeros — easier to analyze.

---

## Worked Examples

### Example 1: Square Well Bound States

**Problem:** Find S-matrix poles for square well $V = -V_0$ for $|x| < a$.

**Solution:**
The S-matrix for s-wave:
$$S_0(k) = e^{2ika}\frac{k\cot(Ka) + ik}{k\cot(Ka) - ik}$$

where $K = \sqrt{k^2 + 2mV_0/\hbar^2}$.

**Bound state condition:** $S$ has pole when denominator vanishes at $k = i\kappa$:
$$i\kappa\cot(\sqrt{-\kappa^2 + 2mV_0/\hbar^2} \cdot a) = \kappa$$

This is the familiar transcendental equation for bound states.

### Example 2: Resonance Width

**Problem:** A resonance has pole at $k = 2 - 0.1i$ (units: fm$^{-1}$). Find energy and width.

**Solution:**
$k_R = 2$ fm$^{-1}$, $\Gamma/2 = 0.1$ fm$^{-1}$

$$E_R = \frac{\hbar^2 k_R^2}{2m} = \frac{(197.3 \text{ MeV·fm})^2 \times 4}{2 \times 938.3 \text{ MeV}} \approx 83 \text{ MeV}$$

$$\Gamma_E = \frac{\hbar^2 k_R \Gamma}{m} = \frac{197.3^2 \times 2 \times 0.2}{938.3} \approx 8.3 \text{ MeV}$$

The resonance has energy 83 MeV and width 8.3 MeV.

### Example 3: Levinson's Theorem Application

**Problem:** A potential has phase shift $\delta(0) = 2\pi$ and $\delta(\infty) = 0$. How many bound states?

**Solution:**
$$n_B = \frac{\delta(0) - \delta(\infty)}{\pi} = \frac{2\pi - 0}{\pi} = 2$$

Two bound states.

---

## Practice Problems

**P1.** Show that unitarity $|S|^2 = 1$ on the real axis follows from $S(k)S(-k) = 1$.

**P2.** For a delta function potential $V(x) = \lambda\delta(x)$, find the S-matrix and locate its poles.

**P3.** Derive the Breit-Wigner formula from the pole structure of $S(E)$.

**P4.** If a resonance has width $\Gamma = 100$ MeV, what is its lifetime?

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

def breit_wigner(E, E_R, Gamma):
    """Breit-Wigner resonance cross section."""
    return Gamma**2 / 4 / ((E - E_R)**2 + Gamma**2 / 4)

def S_matrix_1d_delta(k, lambda_val, m=1, hbar=1):
    """S-matrix for delta function potential."""
    return (1 + 1j * m * lambda_val / (hbar**2 * k)) / \
           (1 - 1j * m * lambda_val / (hbar**2 * k))

# Resonance visualization
E = np.linspace(0, 20, 500)
E_R, Gamma = 10, 2

sigma = breit_wigner(E, E_R, Gamma)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Breit-Wigner
axes[0, 0].plot(E, sigma, 'b-', linewidth=2)
axes[0, 0].axvline(x=E_R, color='r', linestyle='--', label=f'E_R = {E_R}')
axes[0, 0].axhline(y=0.5, color='g', linestyle=':', label='Half-maximum')
axes[0, 0].fill_between(E, 0, sigma, where=(np.abs(E-E_R) < Gamma/2), alpha=0.3)
axes[0, 0].set_xlabel('Energy')
axes[0, 0].set_ylabel('Cross section')
axes[0, 0].set_title(f'Breit-Wigner Resonance (Γ = {Gamma})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# S-matrix in complex k-plane
k_real = np.linspace(-3, 3, 100)
k_imag = np.linspace(-1, 2, 100)
K_R, K_I = np.meshgrid(k_real, k_imag)
K = K_R + 1j * K_I

# For attractive delta potential (bound state)
lambda_attractive = -2
S = S_matrix_1d_delta(K, lambda_attractive)

axes[0, 1].contourf(K_R, K_I, np.log10(np.abs(S)), levels=50, cmap='viridis')
axes[0, 1].axhline(y=0, color='w', linestyle='-', linewidth=0.5)
axes[0, 1].axvline(x=0, color='w', linestyle='-', linewidth=0.5)
# Mark bound state pole
kappa = np.abs(lambda_attractive) / 2  # Bound state for delta potential
axes[0, 1].plot([0], [kappa], 'r*', markersize=15, label='Bound state pole')
axes[0, 1].set_xlabel('Re(k)')
axes[0, 1].set_ylabel('Im(k)')
axes[0, 1].set_title('|S(k)| for Attractive Delta Potential')
axes[0, 1].legend()

# Phase shift vs k
k_vals = np.linspace(0.01, 5, 200)
S_vals = S_matrix_1d_delta(k_vals, lambda_attractive)
delta = np.angle(S_vals) / 2  # S = e^{2iδ}
delta = np.unwrap(delta)

axes[1, 0].plot(k_vals, delta / np.pi, 'b-', linewidth=2)
axes[1, 0].axhline(y=1, color='r', linestyle='--', label='δ/π = 1 (1 bound state)')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('δ/π')
axes[1, 0].set_title('Phase Shift (Levinson\'s Theorem)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Argand diagram
axes[1, 1].plot(S_vals.real, S_vals.imag, 'b-', linewidth=2)
axes[1, 1].plot([1], [0], 'go', markersize=10, label='k=0')
axes[1, 1].plot([1], [0], 'ro', markersize=10, label='k→∞')
theta = np.linspace(0, 2*np.pi, 100)
axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
axes[1, 1].set_xlabel('Re(S)')
axes[1, 1].set_ylabel('Im(S)')
axes[1, 1].set_title('S-matrix in Complex Plane (Argand Diagram)')
axes[1, 1].legend()
axes[1, 1].axis('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scattering_theory.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $S_\ell = e^{2i\delta_\ell}$ | Partial wave S-matrix |
| $\sigma \propto \frac{\Gamma^2/4}{(E-E_R)^2 + \Gamma^2/4}$ | Breit-Wigner |
| $\delta(0) - \delta(\infty) = n_B\pi$ | Levinson's theorem |
| $\tau = \hbar/\Gamma$ | Resonance lifetime |

### Main Takeaways

1. **S-matrix analyticity** encodes all scattering physics
2. **Bound states** are poles on positive imaginary axis
3. **Resonances** are poles in lower half-plane
4. **Levinson's theorem** counts bound states from phase shifts
5. Complex analysis provides **exact constraints** on scattering

---

## Preview: Day 193

Tomorrow: **Asymptotic Methods**
- Saddle point approximation
- Method of steepest descent
- WKB from complex analysis

---

*"The poles of the S-matrix tell us everything about the bound states and resonances."*
