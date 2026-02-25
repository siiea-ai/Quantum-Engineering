# Week 155: Scattering Theory - Problem Set

## Instructions

This problem set contains 28 problems at PhD qualifying exam level. Problems are organized by topic and difficulty:
- **Level A (1-10):** Fundamental concepts and direct applications
- **Level B (11-20):** Intermediate problems requiring synthesis
- **Level C (21-28):** Challenging problems at qualifying exam level

---

## Part I: Scattering Fundamentals and Born Approximation

### Problem 1: Cross Section Basics

A beam of $10^6$ particles per second is incident on a target. Detectors at $\theta = 30°$ and $\theta = 60°$ (covering solid angles of $10^{-3}$ sr each) detect 100 and 25 particles per second respectively.

(a) Calculate $d\sigma/d\Omega$ at each angle.

(b) If the scattering is isotropic, what would be the total cross section?

(c) The actual scattering follows $d\sigma/d\Omega = A\cos^2\theta$. Find $A$ and $\sigma_{\text{tot}}$.

---

### Problem 2: Born Approximation Derivation

Starting from the Lippmann-Schwinger equation:
$$|\psi\rangle = |\phi\rangle + G_0^{(+)}V|\psi\rangle$$

(a) Show that the scattering amplitude is:
$$f(\mathbf{k}', \mathbf{k}) = -\frac{m}{2\pi\hbar^2}\langle\mathbf{k}'|V|\psi^{(+)}\rangle$$

(b) Derive the first Born approximation by replacing $|\psi\rangle$ with $|\phi\rangle$.

(c) State the conditions under which this approximation is valid.

---

### Problem 3: Yukawa Potential

For the Yukawa potential $V(r) = V_0\frac{e^{-\mu r}}{\mu r}$:

(a) Calculate the Fourier transform $\tilde{V}(\mathbf{q})$.

(b) Find the Born approximation scattering amplitude.

(c) Derive the differential cross section $d\sigma/d\Omega$.

(d) In the limit $\mu \to 0$, show you recover the Coulomb result.

---

### Problem 4: Gaussian Potential

For a Gaussian potential $V(r) = V_0 e^{-r^2/a^2}$:

(a) Calculate the Born scattering amplitude.

(b) Find the differential cross section.

(c) At what angle is the first minimum in the cross section?

---

### Problem 5: Square Well Born

For a spherical square well $V(r) = -V_0$ for $r < a$, zero otherwise:

(a) Calculate the Fourier transform of the potential.

(b) Find the Born scattering amplitude.

(c) Show that in the low-energy limit ($ka \ll 1$), the cross section becomes isotropic.

---

### Problem 6: Momentum Transfer

(a) Show that for elastic scattering with $|\mathbf{k}'| = |\mathbf{k}| = k$:
$$q = |\mathbf{q}| = 2k\sin(\theta/2)$$

(b) What is the maximum momentum transfer?

(c) For Yukawa scattering, at what angle is the cross section reduced to half its forward value?

---

### Problem 7: Born Validity

For a square well potential of depth $V_0$ and radius $a$:

(a) Derive the Born validity criterion.

(b) For an electron scattering from a potential with $V_0 = 10$ eV and $a = 2$ Angstroms, at what minimum energy is Born valid?

(c) How does the validity criterion change for high-energy scattering?

---

## Part II: Partial Wave Analysis

### Problem 8: Partial Wave Expansion

(a) Show that the plane wave expansion is:
$$e^{ikz} = \sum_{\ell=0}^{\infty}(2\ell+1)i^\ell j_\ell(kr)P_\ell(\cos\theta)$$

(b) Using the asymptotic form of $j_\ell(kr)$, identify the incoming and outgoing parts.

(c) Show how the outgoing part is modified by the S-matrix $S_\ell = e^{2i\delta_\ell}$.

---

### Problem 9: Phase Shift Definition

(a) Define the phase shift $\delta_\ell$ from the asymptotic behavior of the radial wavefunction.

(b) Show that the partial wave amplitude is:
$$f_\ell = \frac{e^{2i\delta_\ell} - 1}{2ik}$$

(c) Rewrite this as $f_\ell = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$.

---

### Problem 10: S-Wave Scattering

For pure s-wave scattering ($\ell = 0$ only):

(a) Show that $f(\theta) = f_0$ is independent of angle.

(b) Calculate the total cross section in terms of $\delta_0$.

(c) What is the maximum possible cross section (unitarity limit)?

---

### Problem 11: Hard Sphere Phase Shift

For a hard sphere of radius $a$:

(a) Write the boundary condition at $r = a$.

(b) Calculate the s-wave phase shift $\delta_0$.

(c) Find the low-energy ($ka \ll 1$) and high-energy ($ka \gg 1$) limits.

---

### Problem 12: Square Well Phase Shift

For a spherical square well $V(r) = -V_0$ for $r < a$:

(a) Write the radial equations inside and outside the well.

(b) Match solutions at $r = a$ to find $\delta_0$.

(c) Find the condition for $\delta_0 = \pi/2$ (resonance).

---

### Problem 13: Effective Range Theory

(a) State the effective range expansion for s-wave scattering:
$$k\cot\delta_0 = -\frac{1}{a_s} + \frac{1}{2}r_e k^2 + \cdots$$

(b) What are the physical meanings of $a_s$ (scattering length) and $r_e$ (effective range)?

(c) For a hard sphere of radius $a$, find $a_s$ and $r_e$.

---

### Problem 14: Higher Partial Waves

(a) Show that for a potential of range $a$, partial waves with $\ell > ka$ are suppressed.

(b) Explain this in terms of the centrifugal barrier.

(c) At what energy do d-waves ($\ell = 2$) become important for scattering from a potential of range 1 fm?

---

## Part III: Optical Theorem and Unitarity

### Problem 15: Optical Theorem Proof

(a) Using the partial wave expansion, show that:
$$\text{Im}[f(0)] = \frac{k}{4\pi}\sigma_{\text{tot}}$$

(b) What is the physical interpretation of this result?

(c) How does this relate to conservation of probability?

---

### Problem 16: Unitarity Limit

(a) Show that unitarity requires $|S_\ell| = 1$, i.e., $|e^{2i\delta_\ell}| = 1$.

(b) Derive the unitarity limit: $\sigma_\ell \leq \frac{4\pi(2\ell+1)}{k^2}$.

(c) At what value of $\delta_\ell$ is this limit achieved?

---

### Problem 17: Forward Scattering

(a) Calculate $f(0)$ for the Yukawa potential in Born approximation.

(b) Verify the optical theorem by calculating $\sigma_{\text{tot}}$ directly.

(c) Explain any discrepancy.

---

### Problem 18: Absorptive Scattering

If the potential can absorb particles (inelastic scattering), then $|S_\ell| < 1$.

(a) Write $S_\ell = \eta_\ell e^{2i\delta_\ell}$ with $0 \leq \eta_\ell \leq 1$. Find the elastic cross section.

(b) Find the absorption (reaction) cross section.

(c) Show that the total cross section is still given by the optical theorem.

---

## Part IV: Resonances and Special Topics

### Problem 19: Breit-Wigner Formula

Near a resonance at energy $E_R$ with width $\Gamma$:

(a) Show that $\cot\delta_\ell \approx \frac{2(E_R - E)}{\Gamma}$.

(b) Derive the Breit-Wigner form: $f_\ell = \frac{\Gamma/2}{E - E_R + i\Gamma/2}$.

(c) Calculate the cross section at resonance and at $E = E_R \pm \Gamma/2$.

---

### Problem 20: Ramsauer-Townsend Effect

At certain energies, the s-wave phase shift for electron-atom scattering passes through $\pi$, making $\sigma_0 = 0$.

(a) Explain this "Ramsauer-Townsend effect" physically.

(b) For a square well, find the condition for this to occur.

(c) Sketch $\sigma(E)$ showing the Ramsauer minimum.

---

### Problem 21: Levinson's Theorem (MIT 2019)

(a) State Levinson's theorem relating $\delta_\ell(0)$ to the number of bound states.

(b) For a square well that barely supports one s-wave bound state, what is $\delta_0(k=0)$?

(c) How does the phase shift evolve as the well depth increases?

---

### Problem 22: Coulomb Scattering (Caltech 2018)

(a) Show that the Coulomb potential presents special difficulties for scattering theory (hint: consider the asymptotic wavefunction).

(b) The Rutherford formula is:
$$\frac{d\sigma}{d\Omega} = \left(\frac{Z_1Z_2e^2}{16\pi\epsilon_0 E}\right)^2\frac{1}{\sin^4(\theta/2)}$$
Derive this using classical mechanics.

(c) Why does the quantum result agree with the classical one?

---

### Problem 23: Identical Particle Scattering (Yale 2017)

Two identical spin-0 bosons scatter from each other.

(a) What is the relation between $f(\theta)$ and $f(\pi - \theta)$ due to exchange symmetry?

(b) Calculate the differential cross section in terms of $f(\theta)$.

(c) What happens at $\theta = 90°$?

---

### Problem 24: Neutron-Proton Scattering (Princeton 2020)

Low-energy neutron-proton scattering is characterized by:
- Triplet scattering length: $a_t = 5.4$ fm
- Singlet scattering length: $a_s = -23.7$ fm

(a) Calculate the total low-energy cross section.

(b) Why is $a_s$ negative? What does this indicate about the singlet state?

(c) The deuteron binding energy is 2.2 MeV. Relate this to the triplet scattering length.

---

### Problem 25: Form Factor (Berkeley 2018)

The charge distribution of a nucleus can be probed by electron scattering. The cross section is:
$$\frac{d\sigma}{d\Omega} = \left(\frac{d\sigma}{d\Omega}\right)_{\text{Mott}}|F(q)|^2$$

where $F(q)$ is the form factor.

(a) Show that $F(q) = \int \rho(r)e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r$ where $\rho$ is the charge density.

(b) For a uniform sphere of radius $R$, calculate $F(q)$.

(c) Where are the zeros of $F(q)$? How can these be used to determine $R$?

---

### Problem 26: Resonance Lifetime (Harvard 2019)

A particle resonance has energy $E_R = 1$ GeV and width $\Gamma = 100$ MeV.

(a) Calculate the lifetime of the resonance.

(b) At what energies is the cross section half its maximum value?

(c) If the resonance decays into two particles, what is the branching ratio for partial widths $\Gamma_1 = 60$ MeV and $\Gamma_2 = 40$ MeV?

---

### Problem 27: Three-Dimensional Delta Function (Chicago 2020)

Consider scattering from a 3D delta function: $V(\mathbf{r}) = \lambda\delta^3(\mathbf{r})$.

(a) What is the Born approximation scattering amplitude?

(b) This is actually divergent. The regularized result is:
$$f = \frac{a}{1 + ika}$$
where $a = -m\lambda/(2\pi\hbar^2)$. Verify the low-energy limit.

(c) Show this satisfies the optical theorem.

---

### Problem 28: Scattering from a Step Potential (Cornell 2021)

A particle with energy $E$ scatters from a step: $V = 0$ for $x < 0$, $V = V_0$ for $x > 0$, where $E > V_0$.

(a) This is a 1D problem, but define analogous "transmission" and "reflection" coefficients.

(b) Calculate $R$ and $T$ and verify $R + T = 1$.

(c) What happens as $E \to V_0^+$?

(d) Compare to the 3D problem of scattering from a spherical step.

---

## Bonus Problems

### Problem 29: Glauber Approximation

For high-energy scattering, the Glauber approximation treats the particle as traveling in a straight line through the potential.

(a) Derive the Glauber formula for the scattering amplitude.

(b) Apply this to scattering from a Gaussian potential.

(c) When is this approximation valid compared to Born?

---

### Problem 30: Inverse Scattering

Given the phase shifts $\delta_\ell(k)$ for all $\ell$ and $k$:

(a) Is the potential uniquely determined?

(b) What additional information might be needed?

(c) This is related to the Marchenko equation. State its significance.

---

## Problem Checklist

| Problem | Topic | Status | Time | Difficulty |
|---------|-------|--------|------|------------|
| 1 | Cross section basics | [ ] | ___ min | ___/5 |
| 2 | Born derivation | [ ] | ___ min | ___/5 |
| 3 | Yukawa | [ ] | ___ min | ___/5 |
| 4 | Gaussian | [ ] | ___ min | ___/5 |
| 5 | Square well Born | [ ] | ___ min | ___/5 |
| 6 | Momentum transfer | [ ] | ___ min | ___/5 |
| 7 | Born validity | [ ] | ___ min | ___/5 |
| 8 | Partial wave expansion | [ ] | ___ min | ___/5 |
| 9 | Phase shift definition | [ ] | ___ min | ___/5 |
| 10 | S-wave | [ ] | ___ min | ___/5 |
| 11 | Hard sphere | [ ] | ___ min | ___/5 |
| 12 | Square well phase | [ ] | ___ min | ___/5 |
| 13 | Effective range | [ ] | ___ min | ___/5 |
| 14 | Higher partial waves | [ ] | ___ min | ___/5 |
| 15 | Optical theorem | [ ] | ___ min | ___/5 |
| 16 | Unitarity limit | [ ] | ___ min | ___/5 |
| 17 | Forward scattering | [ ] | ___ min | ___/5 |
| 18 | Absorption | [ ] | ___ min | ___/5 |
| 19 | Breit-Wigner | [ ] | ___ min | ___/5 |
| 20 | Ramsauer-Townsend | [ ] | ___ min | ___/5 |
| 21 | Levinson | [ ] | ___ min | ___/5 |
| 22 | Coulomb | [ ] | ___ min | ___/5 |
| 23 | Identical particles | [ ] | ___ min | ___/5 |
| 24 | n-p scattering | [ ] | ___ min | ___/5 |
| 25 | Form factor | [ ] | ___ min | ___/5 |
| 26 | Resonance lifetime | [ ] | ___ min | ___/5 |
| 27 | 3D delta | [ ] | ___ min | ___/5 |
| 28 | Step potential | [ ] | ___ min | ___/5 |

---

**Target:** Complete at least 20 problems before moving to Week 156.
