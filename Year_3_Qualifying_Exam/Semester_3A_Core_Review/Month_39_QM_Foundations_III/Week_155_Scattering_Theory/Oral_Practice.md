# Week 155: Scattering Theory - Oral Exam Practice

## Overview

Scattering theory questions are staples of qualifying exams. This document covers the most common oral exam questions with model answers.

---

## Fundamental Questions

### Q1: "What is scattering amplitude and how is it related to cross section?"

**Answer:**

"The scattering amplitude $f(\theta,\phi)$ appears in the asymptotic form of the wavefunction:
$$\psi \to e^{ikz} + f(\theta)\frac{e^{ikr}}{r}$$

The first term is the incident plane wave, the second is the outgoing scattered spherical wave.

The differential cross section is simply:
$$\frac{d\sigma}{d\Omega} = |f(\theta)|^2$$

Physical interpretation: $|f|^2$ has units of area and represents the effective target area for scattering into a given solid angle."

**Follow-up:** "Why $e^{ikr}/r$?"
> "This is the outgoing spherical wave solution to the Helmholtz equation. The $1/r$ ensures flux conservation - the probability flux through any sphere is constant."

---

### Q2: "Derive and explain the Born approximation."

**Setup:**
"The Born approximation treats the potential as a perturbation. The scattering amplitude becomes:"

$$f_{\text{Born}} = -\frac{m}{2\pi\hbar^2}\int V(\mathbf{r})e^{-i\mathbf{q}\cdot\mathbf{r}}d^3r$$

**Key insight:**
"The scattering amplitude is the Fourier transform of the potential! This means:
- Large-angle scattering probes short-range structure
- Forward scattering probes the overall strength"

**Validity:**
"Born is valid when the potential is weak or energy is high:
$$\frac{mV_0 a^2}{\hbar^2} \ll 1 \quad \text{or} \quad ka \gg 1$$"

**Example:**
"For Yukawa potential $V = V_0 e^{-\mu r}/(\mu r)$:
$$f = -\frac{2mV_0}{\hbar^2\mu(q^2+\mu^2)}$$
where $q = 2k\sin(\theta/2)$."

---

### Q3: "Explain partial wave analysis."

**Main idea:**
"For central potentials, angular momentum is conserved. We expand in angular momentum eigenstates:
$$f(\theta) = \sum_{\ell}(2\ell+1)f_\ell P_\ell(\cos\theta)$$

Each partial wave has amplitude:
$$f_\ell = \frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$$

where $\delta_\ell$ is the phase shift."

**Why useful:**
1. At low energy, only a few partial waves contribute
2. The potential affects each $\ell$ independently
3. Resonances appear clearly in single partial waves

**Cross section:**
$$\sigma_{\text{tot}} = \sum_\ell \frac{4\pi(2\ell+1)}{k^2}\sin^2\delta_\ell$$

---

### Q4: "What is a phase shift and what does it tell us?"

**Definition:**
"The phase shift $\delta_\ell$ is the change in phase of the radial wavefunction due to the potential.

Without potential: $\psi \sim \sin(kr - \ell\pi/2)$
With potential: $\psi \sim \sin(kr - \ell\pi/2 + \delta_\ell)$"

**Physical meaning:**
- Attractive potential: pulls wave in → $\delta_\ell > 0$
- Repulsive potential: pushes wave out → $\delta_\ell < 0$
- Resonance: $\delta_\ell = \pi/2$ (mod $\pi$)

**How to calculate:**
"Solve the radial Schrodinger equation, match to free solution at large $r$, extract the phase difference."

**Example - hard sphere:**
"For a hard sphere of radius $a$, the s-wave phase shift is $\delta_0 = -ka$ because the wave can't penetrate and effectively starts at $r = a$ instead of $r = 0$."

---

### Q5: "State and prove the optical theorem."

**Statement:**
$$\sigma_{\text{tot}} = \frac{4\pi}{k}\text{Im}[f(0)]$$

**Proof sketch:**
"Using partial waves:
$$f(0) = \sum_\ell(2\ell+1)\frac{e^{i\delta_\ell}\sin\delta_\ell}{k}$$

$$\text{Im}[f(0)] = \sum_\ell(2\ell+1)\frac{\sin^2\delta_\ell}{k} = \frac{k\sigma_{\text{tot}}}{4\pi}$$"

**Physical interpretation:**
"Forward scattering interferes with the incident wave. This interference removes particles from the beam. By unitarity, removed = scattered, so forward scattering determines total cross section."

**Consequence:**
"Even if all scattering is backward, $\text{Im}[f(0)] \neq 0$!"

---

## Calculation Questions

### Q6: "Calculate the scattering from a Yukawa potential."

**Setup:**
$V(r) = V_0 \frac{e^{-\mu r}}{\mu r}$

**Fourier transform:**
$$\tilde{V}(q) = \frac{4\pi V_0}{\mu(q^2 + \mu^2)}$$

**Scattering amplitude:**
$$f = -\frac{2mV_0}{\hbar^2\mu(q^2 + \mu^2)}$$

**Cross section:**
$$\frac{d\sigma}{d\Omega} = \left(\frac{2mV_0}{\hbar^2\mu}\right)^2 \frac{1}{(4k^2\sin^2(\theta/2) + \mu^2)^2}$$

**Limits:**
- $\mu \to 0$: Coulomb (Rutherford formula)
- $\mu \to \infty$: point interaction

---

### Q7: "Find the s-wave phase shift for a hard sphere."

**Boundary condition:** $\psi(r=a) = 0$

**Outside solution:** $u(r) = A\sin(kr + \delta_0)$

**Applying BC:** $\sin(ka + \delta_0) = 0 \Rightarrow \delta_0 = -ka$

**Low-energy cross section:**
$$\sigma = \frac{4\pi}{k^2}\sin^2(ka) \approx 4\pi a^2$$

**Physical picture:**
"The hard sphere pushes the wave out by distance $a$, giving a negative phase shift."

---

### Q8: "Explain resonance scattering and derive the Breit-Wigner formula."

**Physical picture:**
"A resonance occurs when the particle can temporarily form a quasi-bound state inside the potential. This shows up as rapid variation of the phase shift."

**Mathematical condition:**
Near resonance, $\delta_\ell$ passes through $\pi/2$:
$$\cot\delta_\ell = \frac{2(E_R - E)}{\Gamma}$$

**Breit-Wigner amplitude:**
$$f_\ell = \frac{\Gamma/2}{E - E_R + i\Gamma/2}$$

**Cross section:**
$$\sigma \propto \frac{1}{(E-E_R)^2 + (\Gamma/2)^2}$$

This is a Lorentzian with:
- Peak at $E = E_R$
- FWHM = $\Gamma$
- Maximum = unitarity limit

**Lifetime:**
$$\tau = \hbar/\Gamma$$

---

## Conceptual Questions

### Q9: "When does Born approximation fail?"

**Fails when:**
1. **Strong potential:** Multiple scattering important
2. **Low energy:** Potential significantly distorts wavefunction
3. **Near resonances:** Strong energy dependence
4. **Long-range potentials:** Coulomb requires special treatment

**Signs of failure:**
- Born predicts $\sigma \propto E^{-2}$ at low energy
- Actually $\sigma \to$ constant (determined by scattering length)
- Born misses resonances entirely

---

### Q10: "What is the unitarity limit?"

**Statement:**
$$\sigma_\ell \leq \frac{4\pi(2\ell+1)}{k^2}$$

**Origin:**
Unitarity requires $|S_\ell| = 1$, i.e., probability is conserved.

$S_\ell = e^{2i\delta_\ell}$ always has unit magnitude.

**When saturated:**
$\delta_\ell = \pi/2$ (resonance)

**Physical meaning:**
"There's a maximum amount any partial wave can scatter, set by the wavelength."

---

### Q11: "What happens for identical particle scattering?"

**Bosons:**
Must symmetrize: $f(\theta) + f(\pi-\theta)$
$$\frac{d\sigma}{d\Omega} = |f(\theta) + f(\pi-\theta)|^2$$

At $\theta = 90°$: enhanced by factor of 4 (constructive interference)

**Fermions:**
Must antisymmetrize: $f(\theta) - f(\pi-\theta)$
$$\frac{d\sigma}{d\Omega} = |f(\theta) - f(\pi-\theta)|^2$$

At $\theta = 90°$: vanishes (destructive interference)

---

### Q12: "Explain Levinson's theorem."

**Statement:**
$$\delta_\ell(k=0) = n_\ell \pi$$

where $n_\ell$ is the number of bound states with angular momentum $\ell$.

**Physical meaning:**
"Each bound state contributes $\pi$ to the zero-energy phase shift. The phase shift 'counts' the bound states."

**Application:**
If a potential barely supports one bound state, $\delta_0(0) = \pi$.
As the well deepens and a second bound state appears, $\delta_0(0) \to 2\pi$.

---

## Quick-Fire Questions

Be ready to answer in 30 seconds:

1. **"What does the Born approximation give you?"**
   > The Fourier transform of the potential (times constants).

2. **"What's the cross section for s-wave scattering?"**
   > $\sigma_0 = 4\pi\sin^2\delta_0/k^2$

3. **"What's the optical theorem?"**
   > $\sigma_{\text{tot}} = (4\pi/k)\text{Im}[f(0)]$

4. **"What's the unitarity limit for s-wave?"**
   > $\sigma_{\text{max}} = 4\pi/k^2$

5. **"What's the scattering length?"**
   > The low-energy limit: $\delta_0 \to -ka_s$, giving $\sigma \to 4\pi a_s^2$

6. **"What does a negative scattering length mean?"**
   > No bound state, but possible virtual state (near-threshold resonance)

---

## Practice Schedule

| Day | Focus | Duration |
|-----|-------|----------|
| 1 | Q1-Q3 with derivations | 45 min |
| 2 | Q4-Q6 calculation practice | 45 min |
| 3 | Q7-Q9 conceptual | 45 min |
| 4 | Q10-Q12 + quick-fire | 45 min |
| 5 | Full mock oral | 90 min |

---

## Exam Tips

**Draw pictures:** Sketch the potential, turning points, phase shift vs. energy

**Know the limits:** What happens at high/low energy? Strong/weak scattering?

**Connect to experiment:** How are phase shifts measured? What experiments probe scattering?

**Be ready for "why":** Why does the optical theorem hold? Why does Born fail at low energy?

---

**Remember:** Scattering theory connects quantum mechanics to experiment. The committee wants to see that you understand both the formalism and the physics.
