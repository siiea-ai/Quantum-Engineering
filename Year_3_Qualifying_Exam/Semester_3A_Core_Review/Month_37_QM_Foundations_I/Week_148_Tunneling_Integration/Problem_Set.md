# Week 148: Tunneling and WKB — Problem Set

## Instructions

25 qualifying-exam-level problems on tunneling and WKB, plus integration of Month 37 topics.

- **Level 1 (Problems 1-8):** Direct applications — 10-15 minutes each
- **Level 2 (Problems 9-17):** Intermediate — 15-25 minutes each
- **Level 3 (Problems 18-25):** Challenging — 25-40 minutes each

---

## Level 1: Direct Application

### Problem 1: Step Potential Basics

A particle with energy $$E = 2V_0$$ is incident on a step potential of height $$V_0$$.

(a) Calculate the reflection probability $$\mathcal{R}$$.

(b) Calculate the transmission probability $$\mathcal{T}$$.

(c) Verify $$\mathcal{R} + \mathcal{T} = 1$$.

---

### Problem 2: Step Potential Below Barrier

A particle with $$E = V_0/2$$ is incident on a step of height $$V_0$$.

(a) What is the penetration depth into the barrier?

(b) What fraction of the probability density at $$x = 0$$ remains at $$x = 1/\kappa$$?

(c) Why is $$\mathcal{R} = 1$$ in this case?

---

### Problem 3: Rectangular Barrier

A particle tunnels through a barrier of height $$V_0$$ and width $$a$$. Its energy is $$E = V_0/2$$.

(a) Calculate $$\kappa$$.

(b) For $$\kappa a = 2$$, estimate the transmission probability.

(c) How does $$\mathcal{T}$$ change if the barrier width doubles?

---

### Problem 4: WKB Validity

Determine whether WKB is valid for:

(a) Free particle (constant potential)

(b) Harmonic oscillator near the equilibrium point

(c) Harmonic oscillator near the classical turning point

---

### Problem 5: Simple WKB Quantization

Use WKB to find the energy levels of a particle in a box of length $$L$$.

(a) Write the quantization condition.

(b) Solve for $$E_n$$.

(c) Compare to exact result. Why is there no $$+1/2$$?

---

### Problem 6: WKB for Harmonic Oscillator

Apply WKB quantization to $$V = \frac{1}{2}m\omega^2x^2$$.

(a) Find the turning points in terms of $$E$$.

(b) Evaluate the integral $$\int_{-x_0}^{x_0} p\,dx$$.

(c) Show that $$E_n = \hbar\omega(n + 1/2)$$.

---

### Problem 7: Tunneling Rate Estimate

An electron must tunnel through a barrier of height 2 eV and width 1 nm. Its kinetic energy is 1 eV.

(a) Calculate $$\kappa$$ in nm$$^{-1}$$.

(b) Calculate the transmission probability.

(c) If the electron attempts to tunnel $$10^{15}$$ times per second, what is the tunneling rate?

---

### Problem 8: Current Conservation

For a step potential with $$E > V_0$$:

(a) Calculate the probability current $$j$$ in regions I and II.

(b) Show that $$j_{transmitted}/j_{incident} = \mathcal{T}$$.

(c) Verify current conservation.

---

## Level 2: Intermediate

### Problem 9: Transmission Resonances

For a rectangular barrier with $$E > V_0$$:

(a) Derive the condition for perfect transmission ($$\mathcal{T} = 1$$).

(b) What are the resonance energies?

(c) Explain physically why resonances occur.

---

### Problem 10: Double Barrier Tunneling

Consider two identical rectangular barriers separated by distance $$L$$.

(a) Qualitatively, what happens to transmission as a function of energy?

(b) What are the resonance conditions?

(c) This is the basis for the resonant tunneling diode. Explain.

---

### Problem 11: WKB for Linear Potential

A particle in potential $$V(x) = mgx$$ for $$x > 0$$ with $$V(x) = \infty$$ for $$x < 0$$.

(a) Find the turning point for energy $$E$$.

(b) Apply WKB quantization (note: only one turning point, hard wall at $$x = 0$$).

(c) Find $$E_n$$ and compare to Airy function result.

---

### Problem 12: Triangular Barrier

A particle encounters a triangular barrier: $$V(x) = V_0(1 - x/a)$$ for $$0 < x < a$$.

(a) Find the turning points.

(b) Calculate the WKB tunneling probability.

(c) How does $$\mathcal{T}$$ depend on $$a$$?

---

### Problem 13: Time to Tunnel

An electron is in a finite well with tunneling barrier on one side.

(a) If the tunneling probability is $$10^{-10}$$, estimate the escape time.

(b) Use $$f \sim E/h$$ for the attempt frequency.

(c) For $$E = 1$$ eV, what is the lifetime?

---

### Problem 14: WKB Phase

A particle moves from $$x = 0$$ to $$x = L$$ in a slowly varying potential.

(a) What is the WKB phase accumulated?

(b) When is this phase an integer multiple of $$2\pi$$?

(c) How does this relate to quantization?

---

### Problem 15: Reflection Above Barrier

Show that quantum reflection can occur even when $$E > V_0$$ if the potential varies rapidly.

(a) For a step potential, what is $$\mathcal{R}$$ when $$E = 1.01 V_0$$?

(b) Compare to the case $$E = 10 V_0$$.

(c) Why does classical mechanics predict no reflection for $$E > V_0$$?

---

### Problem 16: Connection Formula Application

A particle is in a potential with a single turning point at $$x = a$$.

(a) Write the WKB wavefunction in the allowed region ($$x < a$$).

(b) Use connection formulas to find the wavefunction in the forbidden region ($$x > a$$).

(c) How does the wavefunction behave as $$x \to \infty$$?

---

### Problem 17: Gamow Factor

For alpha decay from a nucleus of radius $$R$$ with Coulomb barrier:

(a) Write the potential $$V(r)$$ for $$r > R$$.

(b) Find the outer turning point $$b$$ in terms of alpha energy $$E$$.

(c) Set up (but don't evaluate) the integral for the Gamow factor.

---

## Level 3: Challenging

### Problem 18: Alpha Decay Lifetime (MIT)

An alpha particle with kinetic energy 5 MeV is emitted from a nucleus with $$Z = 84$$ and $$R = 8$$ fm.

(a) Calculate the Coulomb barrier height.

(b) Find the outer turning point.

(c) Estimate the Gamow factor and the half-life.

---

### Problem 19: WKB with Multiple Turning Points (Berkeley)

A particle is in a double-well potential with a barrier in the middle.

(a) How many turning points are there for low energies?

(b) Write the quantization condition including tunneling through the central barrier.

(c) Show that this leads to energy level splitting.

---

### Problem 20: Scanning Tunneling Microscope (Caltech)

In STM, electrons tunnel through a vacuum gap of width $$d \approx 1$$ nm. The work function is $$\phi = 4$$ eV.

(a) Calculate the decay constant $$\kappa$$ for electrons at the Fermi level.

(b) If the current is proportional to $$e^{-2\kappa d}$$, by what factor does the current change if $$d$$ increases by 0.1 nm?

(c) This extreme sensitivity enables atomic resolution. Explain.

---

### Problem 21: Complex Turning Points (Yale)

For the quartic oscillator $$V = \lambda x^4$$:

(a) Find the turning points as functions of energy.

(b) Apply WKB quantization.

(c) Find $$E_n$$ in terms of $$\lambda$$ and $$n$$.

---

### Problem 22: Field Emission (Princeton)

Electrons in a metal face a triangular barrier when a strong electric field $$\mathcal{E}$$ is applied.

(a) Write the potential $$V(x) = W - e\mathcal{E}x$$ where $$W$$ is the work function.

(b) Find the turning point.

(c) Calculate the WKB transmission probability and show $$\mathcal{T} \propto \exp(-c/\mathcal{E})$$.

---

### Problem 23: WKB Normalization

For a bound state wavefunction:

(a) Show that in WKB, $$\int |\psi|^2 dx \approx \int \frac{C^2}{p(x)}dx$$ in the allowed region.

(b) Evaluate this for the harmonic oscillator.

(c) Determine the normalization constant $$C$$.

---

### Problem 24: Above-Barrier Reflection

Consider scattering from potential $$V(x) = V_0 \text{sech}^2(x/a)$$.

(a) This is a "reflectionless" potential for certain energies. Why?

(b) For general $$E > V_0$$, calculate $$\mathcal{R}$$ using WKB.

(c) Compare to exact result.

---

### Problem 25: Comprehensive Integration

A particle is in the ground state of a harmonic oscillator. At $$t = 0$$, a constant force $$F$$ is suddenly applied.

(a) What is the new equilibrium position?

(b) Express the initial state in terms of the new energy eigenstates (hint: coherent states).

(c) Calculate $$\langle x(t) \rangle$$ and $$\langle p(t) \rangle$$.

(d) What is the probability of finding the particle in the new ground state?

(e) If the new potential has a finite barrier at large $$x$$, estimate the tunneling rate.

---

## Practice Exam Problems (Day 1035)

Choose 3 of the following 4 problems. Time: 3 hours.

### Exam Problem A
Calculate the energy levels of a particle in a half-harmonic oscillator using WKB.

### Exam Problem B
An electron in a 1D box of width $$L$$ is initially in state $$\psi(x) = Ax(L-x)$$. Find the time evolution and calculate $$\langle x(t) \rangle$$.

### Exam Problem C
Derive the transmission coefficient for a delta-function barrier and find the reflection coefficient at low and high energies.

### Exam Problem D
Use the uncertainty principle to estimate the ground state energy of the harmonic oscillator. Then verify using the exact result.

---

*Problem Set for Week 148 — Tunneling and WKB*
