# QM Integration Exam - Practice Written Exam

## Exam Information

**Duration:** 3 hours
**Instructions:** Answer 6 of the following 8 problems. Each problem is worth equal credit. Show all work and justify your answers. Partial credit will be awarded for correct reasoning even if the final answer is wrong.

**Allowed:** One 8.5x11 formula sheet (both sides), prepared in advance.

---

## Problem 1: Quantum Postulates and Measurement (25 points)

A spin-1/2 particle is prepared in the state:
$$|\psi\rangle = \frac{1}{\sqrt{3}}|+\rangle + \sqrt{\frac{2}{3}}|-\rangle$$

where $|+\rangle$ and $|-\rangle$ are eigenstates of $S_z$ with eigenvalues $+\hbar/2$ and $-\hbar/2$ respectively.

**(a)** (5 points) What are the possible results of measuring $S_z$, and what is the probability of each?

**(b)** (7 points) Calculate $\langle S_x\rangle$ and $\langle S_x^2\rangle$ for this state.

**(c)** (6 points) If $S_z$ is measured and the result is $-\hbar/2$, what is the state immediately after the measurement? If $S_x$ is then measured, what are the probabilities of each outcome?

**(d)** (7 points) Calculate the uncertainty $\Delta S_z$ and verify the uncertainty relation $\Delta S_x \Delta S_z \geq \frac{\hbar}{2}|\langle S_y\rangle|$.

---

## Problem 2: One-Dimensional Systems (25 points)

A particle of mass $m$ is in a one-dimensional harmonic oscillator potential $V(x) = \frac{1}{2}m\omega^2 x^2$.

**(a)** (6 points) At $t = 0$, the particle is in the state:
$$|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$
Find $|\psi(t)\rangle$.

**(b)** (7 points) Calculate $\langle x\rangle(t)$ and show it oscillates at frequency $\omega$.

**(c)** (6 points) Calculate $\langle x^2\rangle(t)$. Does the width of the wave packet oscillate?

**(d)** (6 points) The potential is suddenly changed to $V(x) = \frac{1}{2}m(2\omega)^2 x^2$ at $t = 0$ when the particle is in the ground state of the original oscillator. What is the probability that the particle is found in the ground state of the new oscillator?

---

## Problem 3: Angular Momentum and Spin (25 points)

Two spin-1/2 particles interact via the Hamiltonian:
$$H = A\mathbf{S}_1 \cdot \mathbf{S}_2 + B(S_{1z} + S_{2z})$$

where $A$ and $B$ are constants with dimensions of energy.

**(a)** (6 points) Show that $[H, \mathbf{S}^2] = 0$ and $[H, S_z] = 0$, where $\mathbf{S} = \mathbf{S}_1 + \mathbf{S}_2$.

**(b)** (7 points) Find the eigenvalues and eigenstates of $H$ in terms of $|S, m_s\rangle$ states.

**(c)** (6 points) If $B = 0$, what are the energy eigenvalues? Sketch the energy level diagram.

**(d)** (6 points) If $B \neq 0$, how does the triplet state split? Draw the new energy level diagram.

---

## Problem 4: Time-Independent Perturbation Theory (25 points)

A hydrogen atom is placed in a uniform electric field $\mathcal{E}$ pointing in the $z$-direction (Stark effect).

**(a)** (5 points) Write down the perturbation Hamiltonian $H'$.

**(b)** (7 points) Calculate the first-order energy correction to the ground state ($n=1$). Explain your result.

**(c)** (7 points) For the $n=2$ states, explain why you must use degenerate perturbation theory. Set up (but do not fully solve) the perturbation matrix in the basis $\{|2,0,0\rangle, |2,1,0\rangle, |2,1,1\rangle, |2,1,-1\rangle\}$.

**(d)** (6 points) Which matrix elements are non-zero? What does this tell you about the splitting pattern?

---

## Problem 5: Identical Particles and Many-Body (25 points)

**(a)** (6 points) Write the ground state wavefunction for the helium atom, including both spatial and spin parts. Explain why the spin part must be a singlet.

**(b)** (8 points) Using the variational method with trial function $\psi = (Z_{\text{eff}}^3/\pi a_0^3)e^{-Z_{\text{eff}}(r_1+r_2)/a_0}$, show that the optimal effective nuclear charge is $Z_{\text{eff}} = Z - 5/16$. (You may use the result $\langle 1/r_{12}\rangle = 5Z_{\text{eff}}/(8a_0)$ for this wavefunction.)

**(c)** (5 points) For the first excited state (1s2s configuration), why is the triplet state lower in energy than the singlet? Give a physical explanation in terms of electron correlation.

**(d)** (6 points) Write the second-quantized Hamiltonian for two electrons in terms of creation and annihilation operators, including electron-electron repulsion.

---

## Problem 6: WKB and Tunneling (25 points)

A particle of mass $m$ encounters a potential barrier:
$$V(x) = V_0\left(1 - \frac{x^2}{a^2}\right) \quad \text{for } |x| \leq a$$
and $V(x) = 0$ for $|x| > a$. The particle has energy $E < V_0$.

**(a)** (6 points) Find the classical turning points.

**(b)** (8 points) Using the WKB approximation, calculate the transmission coefficient through this barrier.

**(c)** (5 points) Evaluate your result for the limit of a thin barrier ($a$ small) and compare to the rectangular barrier result.

**(d)** (6 points) This potential models alpha decay from a nucleus. If $V_0 = 30$ MeV, $a = 10$ fm, and $m = 4$ amu, estimate the tunneling probability for an alpha particle with $E = 5$ MeV.

---

## Problem 7: Scattering Theory (25 points)

Consider scattering from a spherical square well: $V(r) = -V_0$ for $r < a$ and $V(r) = 0$ for $r > a$.

**(a)** (6 points) For s-wave scattering, write the radial Schrodinger equation inside and outside the well. What are the boundary conditions?

**(b)** (8 points) Solve for the s-wave phase shift $\delta_0$ in terms of $k$, $K$, and $a$, where $k = \sqrt{2mE}/\hbar$ and $K = \sqrt{2m(E+V_0)}/\hbar$.

**(c)** (5 points) Find the condition for a zero-energy resonance ($\delta_0 = \pi/2$ at $k \to 0$). What is the physical significance of this condition?

**(d)** (6 points) Using the Born approximation, calculate the differential cross section $d\sigma/d\Omega$. Compare the Born result with your exact partial wave result in the weak-scattering limit.

---

## Problem 8: Time-Dependent Perturbation Theory (25 points)

A hydrogen atom in its ground state is exposed to an oscillating electric field:
$$\mathbf{E}(t) = \mathcal{E}_0\cos(\omega t)\hat{z}$$

turned on at $t = 0$.

**(a)** (5 points) Write the time-dependent perturbation $H'(t)$ in terms of the dipole operator.

**(b)** (8 points) Using first-order time-dependent perturbation theory, calculate the probability of transition to the $2p$ ($m=0$) state as a function of time.

**(c)** (6 points) Identify the resonance condition. Near resonance, what is the transition probability?

**(d)** (6 points) What selection rules apply to electric dipole transitions? Which $n=2$ states can be reached from the ground state?

---

## End of Exam

**Time Management Suggestion:**
- Read all problems first (10 minutes)
- Work on 6 problems at ~25 minutes each (150 minutes)
- Review and check answers (20 minutes)

**Remember:**
- Show all work
- State any approximations made
- Check units and limiting cases
- Partial credit is available

---

## Formula Sheet Suggestions

Consider including:
- Pauli matrices
- Angular momentum commutators
- Clebsch-Gordan coefficients for spin-1/2
- Hydrogen wavefunctions (ground state, 2s, 2p)
- Harmonic oscillator ladder operators
- Standard integrals (Gaussian, exponential)
- Scattering formulas (Born, partial wave)
- WKB formulas
