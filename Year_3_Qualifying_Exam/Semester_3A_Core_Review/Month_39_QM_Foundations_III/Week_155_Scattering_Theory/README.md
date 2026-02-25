# Week 155: Scattering Theory

## Overview

**Days:** 1079-1085
**Theme:** Scattering amplitude, cross sections, partial waves, Born approximation
**Prerequisites:** Wave mechanics, angular momentum, asymptotic analysis

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Define** scattering amplitude and differential cross section
2. **Apply** the Born approximation to calculate scattering from weak potentials
3. **Perform** partial wave analysis and calculate phase shifts
4. **Use** the optical theorem to relate forward scattering to total cross section
5. **Analyze** resonance scattering using the Breit-Wigner formula
6. **Calculate** scattering from common potentials (square well, Coulomb, Yukawa)

---

## Daily Schedule

| Day | Focus | Key Concepts |
|-----|-------|--------------|
| **1079** | Scattering Amplitude | Cross sections, scattering kinematics |
| **1080** | Born Approximation | Weak scattering, Fourier transforms |
| **1081** | Partial Wave Analysis | Phase shifts, s-wave scattering |
| **1082** | Higher Partial Waves | $\ell > 0$ contributions, centrifugal barrier |
| **1083** | Optical Theorem | Unitarity, total cross section |
| **1084** | Resonances | Breit-Wigner formula, compound states |
| **1085** | Review & Oral Practice | Problem synthesis, oral Q&A |

---

## Core Concepts

### 1. Scattering Setup

A plane wave $e^{ikz}$ incident on a localized potential $V(r)$ produces an outgoing spherical wave at large $r$:

$$\psi(\mathbf{r}) \xrightarrow{r\to\infty} e^{ikz} + f(\theta, \phi)\frac{e^{ikr}}{r}$$

The **scattering amplitude** $f(\theta, \phi)$ contains all information about the scattering process.

### 2. Cross Sections

**Differential cross section:**
$$\frac{d\sigma}{d\Omega} = |f(\theta, \phi)|^2$$

**Total cross section:**
$$\sigma_{\text{tot}} = \int \frac{d\sigma}{d\Omega}d\Omega = \int_0^{2\pi}d\phi\int_0^\pi |f(\theta)|^2\sin\theta\,d\theta$$

### 3. Born Approximation

For weak scattering, the first Born approximation gives:
$$f^{(1)}(\mathbf{q}) = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r})e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r$$

where $\mathbf{q} = \mathbf{k} - \mathbf{k}'$ is the momentum transfer.

The scattering amplitude is proportional to the Fourier transform of the potential.

### 4. Partial Wave Expansion

Expand the scattering amplitude in Legendre polynomials:
$$f(\theta) = \sum_{\ell=0}^{\infty}(2\ell+1)f_\ell P_\ell(\cos\theta)$$

where the **partial wave amplitude** is:
$$f_\ell = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k} = \frac{1}{k\cot\delta_\ell - ik}$$

### 5. Phase Shifts

The phase shift $\delta_\ell$ characterizes the effect of the potential on partial wave $\ell$.

**Attractive potential:** $\delta_\ell > 0$
**Repulsive potential:** $\delta_\ell < 0$

### 6. Optical Theorem

$$\boxed{\sigma_{\text{tot}} = \frac{4\pi}{k}\text{Im}[f(0)]}$$

This fundamental result connects forward scattering to total cross section via unitarity.

---

## Key Equations

### Scattering Amplitude and Cross Section

| Equation | Description |
|----------|-------------|
| $$\frac{d\sigma}{d\Omega} = \|f(\theta)\|^2$$ | Differential cross section |
| $$\sigma_{\text{tot}} = \int\frac{d\sigma}{d\Omega}d\Omega$$ | Total cross section |
| $$\mathbf{q} = \mathbf{k}' - \mathbf{k}$$ | Momentum transfer |
| $$q = 2k\sin(\theta/2)$$ | For elastic scattering |

### Born Approximation

| Equation | Description |
|----------|-------------|
| $$f^{(1)}(\mathbf{q}) = -\frac{m}{2\pi\hbar^2}\tilde{V}(\mathbf{q})$$ | First Born |
| $$\tilde{V}(\mathbf{q}) = \int V(\mathbf{r})e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r$$ | Fourier transform |

### Partial Waves

| Equation | Description |
|----------|-------------|
| $$f(\theta) = \sum_\ell (2\ell+1)f_\ell P_\ell(\cos\theta)$$ | Expansion |
| $$f_\ell = \frac{e^{2i\delta_\ell}-1}{2ik} = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$$ | Partial wave amplitude |
| $$\sigma_\ell = \frac{4\pi(2\ell+1)}{k^2}\sin^2\delta_\ell$$ | Partial wave cross section |
| $$\sigma_{\text{tot}} = \sum_\ell \sigma_\ell$$ | Total from partial waves |

### Special Results

| Equation | Description |
|----------|-------------|
| $$\sigma_{\text{max}}^{(\ell)} = \frac{4\pi(2\ell+1)}{k^2}$$ | Unitarity limit |
| $$\sigma_{\text{tot}} = \frac{4\pi}{k}\text{Im}[f(0)]$$ | Optical theorem |
| $$f_\ell = \frac{\Gamma/2}{E - E_R + i\Gamma/2}$$ | Breit-Wigner resonance |

---

## Important Scattering Results

### Yukawa Potential

$V(r) = V_0\frac{e^{-\mu r}}{\mu r}$

$$f_{\text{Born}}(q) = -\frac{2mV_0}{\hbar^2\mu(q^2 + \mu^2)}$$

### Coulomb Potential

$V(r) = \frac{Z_1Z_2e^2}{4\pi\epsilon_0 r}$

$$\frac{d\sigma}{d\Omega} = \left(\frac{Z_1Z_2e^2}{16\pi\epsilon_0 E}\right)^2\frac{1}{\sin^4(\theta/2)}$$

This is the **Rutherford formula** - exact in both classical and quantum mechanics!

### Hard Sphere

$V(r) = \infty$ for $r < a$, zero otherwise

Low energy ($ka \ll 1$): $\sigma \approx 4\pi a^2$

High energy ($ka \gg 1$): $\sigma \approx 2\pi a^2$

---

## Study Resources

### Primary Texts
- Shankar, Chapter 19 (Scattering Theory)
- Griffiths, Chapter 11 (Scattering)
- Sakurai, Chapter 7 (Scattering)

### Supplementary
- [Physics LibreTexts - Born Approximation](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Introductory_Quantum_Mechanics_(Fitzpatrick)/14:_Scattering_Theory/14.02:_Born_Approximation)
- [UTAustin Partial Waves](https://farside.ph.utexas.edu/teaching/qmech/Quantum/node134.html)

---

## Qualifying Exam Relevance

### Typical Problem Types
1. Calculate Born approximation for given potential
2. Find phase shift from square well potential
3. Apply optical theorem
4. Identify resonance conditions
5. Low-energy scattering analysis

### Common Pitfalls
- Forgetting the $1/r$ in asymptotic form
- Sign errors in Born formula
- Wrong normalization in partial wave expansion
- Confusing scattering amplitude with cross section

---

## Week Checklist

- [ ] Understand scattering setup and cross sections
- [ ] Master Born approximation formula
- [ ] Derive partial wave expansion
- [ ] Calculate phase shifts for simple potentials
- [ ] Prove optical theorem
- [ ] Understand resonance scattering
- [ ] Complete problem sets
- [ ] Practice oral explanations

---

**Created:** February 2026
**Status:** NOT STARTED
