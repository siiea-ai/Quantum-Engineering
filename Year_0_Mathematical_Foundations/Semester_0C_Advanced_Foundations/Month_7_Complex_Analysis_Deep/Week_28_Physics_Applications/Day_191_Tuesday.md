# Day 191: Dispersion Relations and Causality

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Dispersion Relations |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Causality & Sum Rules |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 191, you will be able to:

1. Derive Kramers-Kronig relations from contour integration
2. Understand the connection between analyticity and causality
3. Apply dispersion relations to physical response functions
4. Derive and interpret sum rules
5. Connect to the optical theorem in scattering
6. Apply to refractive index and conductivity

---

## Core Content

### 1. Causality and Analyticity

**Causality Principle:** Effect cannot precede cause.

For a response function $\chi(t)$ (response at time $t$ to impulse at $t=0$):
$$\chi(t) = 0 \text{ for } t < 0$$

**Fourier Transform:**
$$\tilde{\chi}(\omega) = \int_{-\infty}^{\infty} \chi(t) e^{i\omega t} dt = \int_0^{\infty} \chi(t) e^{i\omega t} dt$$

**Key Result:** For $\text{Im}(\omega) > 0$ (upper half-plane):
$$|\tilde{\chi}(\omega)| = \left|\int_0^{\infty} \chi(t) e^{i\omega t} dt\right| \leq \int_0^{\infty} |\chi(t)| e^{-\text{Im}(\omega) \cdot t} dt < \infty$$

**Conclusion:** Causal response ⟹ $\tilde{\chi}(\omega)$ analytic in upper half-plane.

### 2. Kramers-Kronig Relations

**Setup:** Let $\chi(\omega)$ be:
1. Analytic in upper half-plane
2. $\chi(\omega) \to 0$ as $|\omega| \to \infty$

**Derivation:** For real $\omega_0$, apply Cauchy's integral formula on contour closing in UHP:
$$\chi(\omega_0) = \frac{1}{2\pi i} \oint \frac{\chi(\omega)}{\omega - \omega_0} d\omega$$

Taking the principal value along the real axis plus semicircle (which vanishes):
$$\chi(\omega_0) = \frac{1}{\pi i} \mathcal{P}\int_{-\infty}^{\infty} \frac{\chi(\omega)}{\omega - \omega_0} d\omega$$

Separating real and imaginary parts ($\chi = \chi' + i\chi''$):

$$\boxed{\chi'(\omega) = \frac{1}{\pi} \mathcal{P}\int_{-\infty}^{\infty} \frac{\chi''(\omega')}{\omega' - \omega} d\omega'}$$

$$\boxed{\chi''(\omega) = -\frac{1}{\pi} \mathcal{P}\int_{-\infty}^{\infty} \frac{\chi'(\omega')}{\omega' - \omega} d\omega'}$$

### 3. Physical Interpretation

**Real part $\chi'$:** Dispersive (reactive) response — energy storage
**Imaginary part $\chi''$:** Absorptive (dissipative) response — energy loss

**Kramers-Kronig says:** Dispersion and absorption are fundamentally linked!

You cannot have absorption without dispersion (and vice versa).

### 4. Sum Rules

**Moment sum rules:** Integrate dispersion relations weighted by powers of $\omega$.

**f-sum rule (oscillator strength):**
$$\int_0^{\infty} \omega \cdot \text{Im}\,\chi(\omega) d\omega = \frac{\pi}{2} \langle [A, [H, A]] \rangle$$

For conductivity $\sigma(\omega)$:
$$\int_0^{\infty} \text{Re}\,\sigma(\omega) d\omega = \frac{\pi n e^2}{2m}$$

where $n$ is electron density.

**Thomas-Reiche-Kuhn sum rule:**
$$\sum_n f_{0n} = Z$$

where $f_{0n}$ are oscillator strengths and $Z$ is number of electrons.

### 5. Optical Theorem

**Scattering amplitude** $f(\theta)$ satisfies analyticity constraints.

**Optical Theorem:**
$$\boxed{\text{Im}\, f(0) = \frac{k}{4\pi} \sigma_{\text{tot}}}$$

The forward scattering amplitude is related to total cross section.

**Proof sketch:** Conservation of probability + unitarity of S-matrix.

**Dispersion relation for scattering:**
$$\text{Re}\, f(\omega) = \frac{2\omega^2}{\pi} \mathcal{P}\int_0^{\infty} \frac{\text{Im}\, f(\omega')}{\omega'(\omega'^2 - \omega^2)} d\omega'$$

### 6. Applications

#### Refractive Index

Complex refractive index: $\tilde{n}(\omega) = n(\omega) + i\kappa(\omega)$

Kramers-Kronig:
$$n(\omega) - 1 = \frac{2}{\pi} \mathcal{P}\int_0^{\infty} \frac{\omega'\kappa(\omega')}{\omega'^2 - \omega^2} d\omega'$$

#### Dielectric Function

$$\varepsilon(\omega) = \varepsilon_1(\omega) + i\varepsilon_2(\omega)$$

$$\varepsilon_1(\omega) - 1 = \frac{2}{\pi} \mathcal{P}\int_0^{\infty} \frac{\omega'\varepsilon_2(\omega')}{\omega'^2 - \omega^2} d\omega'$$

---

## Worked Examples

### Example 1: Lorentzian Response

**Problem:** Verify Kramers-Kronig for the Lorentzian:
$$\chi(\omega) = \frac{1}{\omega_0^2 - \omega^2 - i\gamma\omega}$$

**Solution:**
Real part: $\chi'(\omega) = \frac{\omega_0^2 - \omega^2}{(\omega_0^2 - \omega^2)^2 + \gamma^2\omega^2}$

Imaginary part: $\chi''(\omega) = \frac{\gamma\omega}{(\omega_0^2 - \omega^2)^2 + \gamma^2\omega^2}$

Kramers-Kronig relates these. The Lorentzian has a pole at:
$$\omega = \pm\sqrt{\omega_0^2 - \gamma^2/4} - i\gamma/2$$

Both poles are in lower half-plane (as required for causality). ✓

### Example 2: Drude Conductivity

**Problem:** Apply sum rule to Drude model:
$$\sigma(\omega) = \frac{ne^2/m}{1/\tau - i\omega}$$

**Solution:**
$$\text{Re}\,\sigma = \frac{ne^2\tau/m}{1 + \omega^2\tau^2}$$

$$\int_0^{\infty} \text{Re}\,\sigma(\omega) d\omega = \frac{ne^2\tau}{m} \cdot \frac{\pi}{2\tau} = \frac{\pi ne^2}{2m}$$

This verifies the f-sum rule! ✓

---

## Practice Problems

**P1.** Derive Kramers-Kronig for a function that goes as $1/\omega$ at infinity (need subtraction).

**P2.** Show that causality implies $\chi(-\omega) = \chi^*(\omega)$ for real $\omega$.

**P3.** Apply the optical theorem to find $\sigma_{\text{tot}}$ from forward scattering amplitude.

**P4.** Derive the sum rule for the polarizability $\alpha(\omega)$.

---

## Computational Lab

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def kramers_kronig_real(chi_imag, omega, omega_range):
    """Compute Re χ from Im χ using Kramers-Kronig."""
    def integrand(wp, w):
        if abs(wp - w) < 1e-10:
            return 0
        return chi_imag(wp) / (wp - w)

    result = np.zeros_like(omega)
    for i, w in enumerate(omega):
        # Principal value via splitting
        I1, _ = integrate.quad(lambda wp: integrand(wp, w),
                              omega_range[0], w - 0.01, limit=100)
        I2, _ = integrate.quad(lambda wp: integrand(wp, w),
                              w + 0.01, omega_range[1], limit=100)
        result[i] = (I1 + I2) / np.pi
    return result

# Test with Lorentzian
omega_0 = 5
gamma = 0.5

def chi_lorentzian(omega):
    return 1 / (omega_0**2 - omega**2 - 1j * gamma * omega)

omega = np.linspace(0.1, 10, 100)
chi = chi_lorentzian(omega)

# Kramers-Kronig reconstruction
chi_real_kk = kramers_kronig_real(lambda w: chi_lorentzian(w).imag, omega, (0.01, 20))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(omega, chi.real, 'b-', linewidth=2, label='Exact Re χ')
axes[0].plot(omega, chi_real_kk, 'r--', linewidth=2, label='KK from Im χ')
axes[0].axvline(x=omega_0, color='g', linestyle=':', label='Resonance')
axes[0].set_xlabel('ω')
axes[0].set_ylabel('Re χ(ω)')
axes[0].set_title('Kramers-Kronig Verification')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega, chi.imag, 'b-', linewidth=2, label='Im χ (absorption)')
axes[1].set_xlabel('ω')
axes[1].set_ylabel('Im χ(ω)')
axes[1].set_title('Absorption Spectrum')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kramers_kronig.png', dpi=150, bbox_inches='tight')
plt.show()

# Sum rule verification
area = integrate.quad(lambda w: chi_lorentzian(w).imag * w, 0, 100)[0]
print(f"∫ω·Im(χ)dω = {area:.4f}")
print(f"Expected (π/2)·1 = {np.pi/2:.4f}")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\chi'(\omega) = \frac{1}{\pi}\mathcal{P}\int \frac{\chi''(\omega')}{\omega'-\omega}d\omega'$ | Kramers-Kronig (real from imaginary) |
| $\chi''(\omega) = -\frac{1}{\pi}\mathcal{P}\int \frac{\chi'(\omega')}{\omega'-\omega}d\omega'$ | Kramers-Kronig (imaginary from real) |
| $\text{Im }f(0) = \frac{k}{4\pi}\sigma_{\text{tot}}$ | Optical theorem |

### Main Takeaways

1. **Causality ⟺ Analyticity** in upper half-plane
2. **Kramers-Kronig** links dispersion and absorption
3. **Sum rules** constrain integrated response
4. **Optical theorem** connects forward scattering to total cross section
5. These are **exact** results from complex analysis!

---

## Preview: Day 192

Tomorrow: **Scattering Theory Applications**
- S-matrix analyticity
- Poles and resonances
- Levinson's theorem

---

*"The Kramers-Kronig relations are perhaps the most beautiful example of how mathematics constrains physics."*
