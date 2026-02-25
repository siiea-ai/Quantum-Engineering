# Week 155: Scattering Theory - Comprehensive Review Guide

## Table of Contents
1. [Scattering Fundamentals](#1-scattering-fundamentals)
2. [The Scattering Amplitude](#2-the-scattering-amplitude)
3. [Born Approximation](#3-born-approximation)
4. [Partial Wave Analysis](#4-partial-wave-analysis)
5. [Phase Shifts](#5-phase-shifts)
6. [The Optical Theorem](#6-the-optical-theorem)
7. [Resonance Scattering](#7-resonance-scattering)
8. [Applications](#8-applications)

---

## 1. Scattering Fundamentals

### Physical Setup

Scattering experiments are fundamental to physics - from Rutherford's discovery of the atomic nucleus to modern particle physics at the LHC. The basic setup involves:

1. **Incident beam:** Particles with well-defined momentum $\hbar\mathbf{k}$
2. **Target:** Localized potential $V(\mathbf{r})$
3. **Detector:** Measures scattered particles at angle $(\theta, \phi)$

### The Scattering Experiment

**Incident flux:** $j_{\text{inc}} = \frac{\hbar k}{m}|\psi_{\text{inc}}|^2$ (particles per unit area per unit time)

**Scattered flux into solid angle $d\Omega$:**
$$dN = j_{\text{inc}} \cdot \frac{d\sigma}{d\Omega}d\Omega$$

The **differential cross section** $d\sigma/d\Omega$ has units of area and represents the effective target area for scattering into $d\Omega$.

### Cross Section Definitions

**Differential cross section:**
$$\frac{d\sigma}{d\Omega} = \frac{\text{number scattered into } d\Omega \text{ per unit time}}{\text{incident flux}}$$

**Total cross section:**
$$\sigma_{\text{tot}} = \int\frac{d\sigma}{d\Omega}d\Omega = \int_0^{2\pi}d\phi\int_0^\pi\frac{d\sigma}{d\Omega}\sin\theta\,d\theta$$

For azimuthally symmetric scattering:
$$\sigma_{\text{tot}} = 2\pi\int_0^\pi\frac{d\sigma}{d\Omega}\sin\theta\,d\theta$$

---

## 2. The Scattering Amplitude

### Asymptotic Form

Far from the scattering center, the wavefunction has the form:
$$\psi(\mathbf{r}) \xrightarrow{r\to\infty} A\left[e^{ikz} + f(\theta,\phi)\frac{e^{ikr}}{r}\right]$$

- First term: incident plane wave
- Second term: outgoing spherical wave
- $f(\theta,\phi)$: **scattering amplitude** (has dimensions of length)

### Connection to Cross Section

The differential cross section is simply:
$$\boxed{\frac{d\sigma}{d\Omega} = |f(\theta,\phi)|^2}$$

All information about the scattering is encoded in $f(\theta,\phi)$.

### Lippmann-Schwinger Equation

The formal solution to the scattering problem:
$$|\psi\rangle = |\phi\rangle + G_0^{(+)}V|\psi\rangle$$

where:
- $|\phi\rangle$ is the free particle state
- $G_0^{(+)} = (E - H_0 + i\epsilon)^{-1}$ is the retarded Green's function

This leads to the scattering amplitude:
$$f(\mathbf{k}',\mathbf{k}) = -\frac{m}{2\pi\hbar^2}\langle\mathbf{k}'|V|\psi^{(+)}\rangle$$

---

## 3. Born Approximation

### First Born Approximation

When the scattering is weak, replace $|\psi\rangle$ by the incident wave $|\phi\rangle$:

$$f^{(1)}(\mathbf{k}',\mathbf{k}) = -\frac{m}{2\pi\hbar^2}\langle\mathbf{k}'|V|\mathbf{k}\rangle$$

For a local potential $V(\mathbf{r})$:
$$\boxed{f^{(1)}(\mathbf{q}) = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r})e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r}$$

where $\mathbf{q} = \mathbf{k}' - \mathbf{k}$ is the **momentum transfer**.

### Physical Interpretation

The Born approximation gives the scattering amplitude as the Fourier transform of the potential. This is valid when:
- The potential is weak
- The incident energy is high
- Multiple scattering can be neglected

### Validity Criterion

Born approximation is valid when:
$$\frac{m|V_0|a^2}{\hbar^2} \ll 1$$

where $V_0$ is the potential strength and $a$ is its range.

### Momentum Transfer

For elastic scattering ($|\mathbf{k}'| = |\mathbf{k}| = k$):
$$q = |\mathbf{q}| = 2k\sin(\theta/2)$$

- Forward scattering ($\theta = 0$): $q = 0$
- Back scattering ($\theta = \pi$): $q = 2k$

### Born Series

The Born series is an iterative solution:
$$f = f^{(1)} + f^{(2)} + f^{(3)} + \cdots$$

where $f^{(n)}$ involves $n$ interactions with the potential.

---

## 4. Partial Wave Analysis

### Why Partial Waves?

For central potentials $V(r)$, angular momentum is conserved. It's natural to expand in angular momentum eigenstates.

### Expansion of Plane Wave

$$e^{ikz} = \sum_{\ell=0}^{\infty}(2\ell+1)i^\ell j_\ell(kr)P_\ell(\cos\theta)$$

where $j_\ell(kr)$ are spherical Bessel functions.

### Expansion of Scattering Amplitude

$$f(\theta) = \sum_{\ell=0}^{\infty}(2\ell+1)f_\ell P_\ell(\cos\theta)$$

where $f_\ell$ is the **partial wave amplitude** for angular momentum $\ell$.

### Partial Wave Cross Section

$$\sigma_\ell = \frac{4\pi(2\ell+1)}{k^2}|f_\ell|^2$$

$$\sigma_{\text{tot}} = \sum_{\ell=0}^{\infty}\sigma_\ell$$

### Asymptotic Behavior

Far from the potential, the radial wavefunction becomes:
$$R_\ell(r) \xrightarrow{r\to\infty} \frac{A_\ell}{kr}\sin(kr - \ell\pi/2 + \delta_\ell)$$

The phase shift $\delta_\ell$ encodes the effect of the potential.

### Partial Wave Amplitude

$$\boxed{f_\ell = \frac{e^{2i\delta_\ell} - 1}{2ik} = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}}$$

---

## 5. Phase Shifts

### Physical Meaning

The phase shift $\delta_\ell$ is the change in phase of the $\ell$th partial wave due to the potential:

- **No potential:** $\psi \sim \sin(kr - \ell\pi/2)$
- **With potential:** $\psi \sim \sin(kr - \ell\pi/2 + \delta_\ell)$

### Sign Convention

- **Attractive potential:** Pulls wavefunction in, $\delta_\ell > 0$
- **Repulsive potential:** Pushes wavefunction out, $\delta_\ell < 0$

### Calculating Phase Shifts

**Method 1: Match at boundary**
Solve the radial equation inside and outside the potential, match at the boundary.

**Method 2: Scattering matrix**
Define $S_\ell = e^{2i\delta_\ell}$, then $f_\ell = (S_\ell - 1)/(2ik)$.

### Low-Energy Behavior

At low energies ($ka \ll 1$ where $a$ is the potential range):

- **s-wave dominates:** Higher $\ell$ suppressed by centrifugal barrier
- **Scattering length:** $\delta_0 \to -ka_s$ where $a_s$ is the scattering length
- **Cross section:** $\sigma \to 4\pi a_s^2$

### Effective Range Expansion

$$k\cot\delta_0 = -\frac{1}{a_s} + \frac{1}{2}r_e k^2 + \cdots$$

where $a_s$ is the scattering length and $r_e$ is the effective range.

### Levinson's Theorem

$$\delta_\ell(0) - \delta_\ell(\infty) = n_\ell\pi$$

where $n_\ell$ is the number of bound states with angular momentum $\ell$.

---

## 6. The Optical Theorem

### Statement

$$\boxed{\sigma_{\text{tot}} = \frac{4\pi}{k}\text{Im}[f(0)]}$$

The total cross section is proportional to the imaginary part of the forward scattering amplitude.

### Proof via Partial Waves

From the partial wave expansion:
$$f(0) = \sum_\ell(2\ell+1)f_\ell$$

$$\text{Im}[f(0)] = \sum_\ell(2\ell+1)\frac{\sin^2\delta_\ell}{k} = \frac{k}{4\pi}\sum_\ell\sigma_\ell = \frac{k\sigma_{\text{tot}}}{4\pi}$$

### Physical Interpretation

The optical theorem reflects **conservation of probability** (unitarity):

- Forward scattering interferes with the incident wave
- This interference removes flux from the incident beam
- The removed flux equals the total scattered flux

### Connection to Unitarity

The S-matrix must be unitary: $S^\dagger S = 1$.

For each partial wave: $|S_\ell|^2 = |e^{2i\delta_\ell}|^2 = 1$

This constraint leads to:
$$\sigma_\ell \leq \frac{4\pi(2\ell+1)}{k^2}$$

This is the **unitarity limit**.

---

## 7. Resonance Scattering

### Resonance Condition

A resonance occurs when $\delta_\ell$ passes through $\pi/2$ (or odd multiples), making $\sin^2\delta_\ell = 1$ and the partial wave cross section maximal.

### Breit-Wigner Formula

Near a resonance at energy $E_R$:
$$f_\ell = \frac{\Gamma/2}{E - E_R + i\Gamma/2}$$

where $\Gamma$ is the resonance width (related to lifetime by $\tau = \hbar/\Gamma$).

### Resonance Cross Section

$$\sigma_\ell = \frac{4\pi(2\ell+1)}{k^2}\frac{(\Gamma/2)^2}{(E-E_R)^2 + (\Gamma/2)^2}$$

This is a **Lorentzian** peaked at $E = E_R$ with full width $\Gamma$ at half maximum.

### Physical Origin

Resonances occur when the particle can temporarily form a quasi-bound state in the potential. The width $\Gamma$ is related to the tunneling rate out of this state.

### Examples

- **Nuclear resonances:** Compound nucleus formation
- **Atomic resonances:** Autoionizing states
- **Particle physics:** Resonances like $\rho$, $\omega$, $J/\psi$

---

## 8. Applications

### Yukawa Potential (Screened Coulomb)

$$V(r) = V_0\frac{e^{-\mu r}}{\mu r}$$

Born approximation:
$$f(\theta) = -\frac{2mV_0}{\hbar^2\mu(q^2 + \mu^2)}$$

where $q = 2k\sin(\theta/2)$.

Cross section:
$$\frac{d\sigma}{d\Omega} = \left(\frac{2mV_0}{\hbar^2\mu}\right)^2\frac{1}{(4k^2\sin^2(\theta/2) + \mu^2)^2}$$

### Coulomb Scattering

$$V(r) = \frac{Z_1Z_2e^2}{4\pi\epsilon_0 r}$$

**Rutherford formula:**
$$\frac{d\sigma}{d\Omega} = \left(\frac{Z_1Z_2e^2}{16\pi\epsilon_0 E}\right)^2\frac{1}{\sin^4(\theta/2)}$$

Remarkable fact: This classical result is also exact quantum mechanically (the Born series sums to the exact answer).

### Hard Sphere

$V(r) = \infty$ for $r < a$

**Low energy ($ka \ll 1$):**
$$\delta_0 \approx -ka, \quad \sigma \approx 4\pi a^2$$

**High energy ($ka \gg 1$):**
$$\sigma \to 2\pi a^2$$

The factor of 2 at high energy comes from diffraction (shadow scattering).

### Square Well

$V(r) = -V_0$ for $r < a$, zero otherwise

**Phase shift:**
$$\tan\delta_0 = \frac{k\tan(Ka) - K\tan(ka)}{K + k\tan(Ka)\tan(ka)}$$

where $K = \sqrt{2m(E+V_0)}/\hbar$.

**Ramsauer-Townsend effect:** At certain energies, $\delta_0 = n\pi$ and s-wave scattering vanishes.

---

## Summary: Key Results

### Formulas to Know

1. $\frac{d\sigma}{d\Omega} = |f(\theta)|^2$

2. $f_{\text{Born}} = -\frac{m}{2\pi\hbar^2}\tilde{V}(\mathbf{q})$

3. $f_\ell = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$

4. $\sigma_{\text{tot}} = \frac{4\pi}{k}\text{Im}[f(0)]$

5. Low-energy: $\sigma \to 4\pi a_s^2$

### Physical Insights

1. Scattering amplitude is Fourier transform of potential (Born)
2. Phase shifts encode potential's effect on each partial wave
3. Optical theorem connects forward scattering to total cross section
4. Resonances occur at quasi-bound state energies
5. Low-energy scattering dominated by s-wave

---

## References

1. Shankar, R. *Principles of Quantum Mechanics*, Chapter 19
2. Griffiths, D.J. *Introduction to Quantum Mechanics*, Chapter 11
3. Sakurai, J.J. *Modern Quantum Mechanics*, Chapter 7
4. Taylor, J.R. *Scattering Theory*
5. [Physics LibreTexts - Scattering Theory](https://phys.libretexts.org/)

---

**Word Count:** ~2300 words
