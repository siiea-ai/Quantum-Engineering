# Quantum Mechanics Written Qualifying Exam

## Exam Information

**Duration:** 3 hours (180 minutes)
**Total Points:** 200
**Number of Problems:** 8
**Passing Score:** 160 points (80%)

---

## Instructions

1. **Time Management:** You have 180 minutes for 8 problems. Budget approximately 20-22 minutes per problem.

2. **Show All Work:** Partial credit is awarded for correct reasoning and setup, even if the final answer is incorrect.

3. **State Assumptions:** If you need to make approximations or assumptions, state them clearly.

4. **Physical Reasoning:** Whenever possible, explain your physical reasoning, not just mathematical manipulations.

5. **Units and Limits:** Check that your answers have correct units and behave correctly in limiting cases.

6. **No External Resources:** This is a closed-book exam. No notes, textbooks, or electronic devices.

---

## Useful Constants and Formulas

You may use the following without derivation:

- $\hbar = 1.055 \times 10^{-34}$ J$\cdot$s
- $m_e = 9.109 \times 10^{-31}$ kg
- $e = 1.602 \times 10^{-19}$ C
- $a_0 = 0.529$ \AA (Bohr radius)
- $E_0 = 13.6$ eV (Hydrogen ground state binding energy)

Pauli Matrices:
$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

---

## Problem 1: Operator Algebra and Uncertainty (25 points)

Consider two Hermitian operators $\hat{A}$ and $\hat{B}$ satisfying:

$$[\hat{A}, \hat{B}] = i\hat{C}$$

where $\hat{C}$ is also Hermitian.

**(a)** (8 points) Prove that the generalized uncertainty relation holds:

$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle \hat{C} \rangle|$$

where $\Delta A = \sqrt{\langle \hat{A}^2 \rangle - \langle \hat{A} \rangle^2}$.

**(b)** (7 points) A particle is in the state:

$$|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$$

where $|n\rangle$ are harmonic oscillator energy eigenstates. Calculate $\Delta x$ and $\Delta p$ explicitly and verify the uncertainty relation.

**(c)** (10 points) Define the operators:

$$\hat{A} = \hat{a} + \hat{a}^\dagger, \quad \hat{B} = i(\hat{a}^\dagger - \hat{a})$$

where $\hat{a}$ and $\hat{a}^\dagger$ are the harmonic oscillator ladder operators. Find $[\hat{A}, \hat{B}]$ and determine for which states the uncertainty product $\Delta A \cdot \Delta B$ is minimized.

---

## Problem 2: Finite Square Well (25 points)

A particle of mass $m$ is confined to a symmetric finite square well:

$$V(x) = \begin{cases} -V_0 & |x| < a \\ 0 & |x| > a \end{cases}$$

where $V_0 > 0$.

**(a)** (8 points) Write down the time-independent Schrodinger equation in each region. For bound states ($E < 0$), define appropriate wavevector parameters and write the general solution with correct boundary conditions as $x \to \pm\infty$.

**(b)** (10 points) Apply continuity conditions at $x = a$ for the even-parity bound states. Show that the allowed energies satisfy:

$$\kappa = k\tan(ka)$$

where you should define $k$ and $\kappa$ in terms of $E$, $V_0$, $m$, and $\hbar$.

**(c)** (7 points) Consider the limit $V_0 \to \infty$ with $a$ fixed. Show that you recover the infinite square well energy levels in this limit. What is the ground state energy for a well with $V_0 = 10\frac{\hbar^2}{2ma^2}$? (A graphical or approximate numerical answer is acceptable.)

---

## Problem 3: Spin-1/2 Dynamics (25 points)

A spin-1/2 particle is placed in a time-dependent magnetic field:

$$\vec{B}(t) = B_0\cos(\omega t)\hat{z} + B_0\sin(\omega t)\hat{x}$$

The Hamiltonian is $\hat{H} = -\gamma \vec{S} \cdot \vec{B}(t)$ where $\gamma$ is the gyromagnetic ratio.

**(a)** (8 points) Write the Hamiltonian explicitly as a $2 \times 2$ matrix in the $|+\rangle$, $|-\rangle$ basis (eigenstates of $\hat{S}_z$).

**(b)** (10 points) Transform to a rotating frame using the unitary operator:

$$\hat{U}(t) = e^{i\omega t \hat{S}_z/\hbar}$$

Find the effective Hamiltonian $\hat{H}_{eff}$ in the rotating frame, where:

$$\hat{H}_{eff} = \hat{U}^\dagger \hat{H} \hat{U} - i\hbar \hat{U}^\dagger \frac{\partial \hat{U}}{\partial t}$$

**(c)** (7 points) If the particle starts in state $|+\rangle$ at $t = 0$, find the probability of measuring $|-\rangle$ at time $t$ when $\omega = \gamma B_0$ (resonance condition). Interpret this result physically.

---

## Problem 4: Angular Momentum Addition (25 points)

Two spin-1 particles are in the state:

$$|\Psi\rangle = \frac{1}{\sqrt{2}}\left(|1,1\rangle_1|1,-1\rangle_2 - |1,-1\rangle_1|1,1\rangle_2\right)$$

where $|j,m\rangle_i$ denotes particle $i$ in state with total spin $j$ and $z$-component $m$.

**(a)** (8 points) What are the possible values of the total angular momentum $J$ when combining two spin-1 particles? Express $|\Psi\rangle$ in the $|J, M\rangle$ basis (total angular momentum eigenstates).

**(b)** (8 points) Calculate $\langle \hat{S}_{1z} \rangle$, $\langle \hat{S}_{2z} \rangle$, and $\langle \hat{S}_{1z}\hat{S}_{2z} \rangle$ in state $|\Psi\rangle$.

**(c)** (9 points) A measurement of $\hat{J}^2$ is performed. What are the possible outcomes and their probabilities? After measuring $J^2 = 2\hbar^2$, what is the state of the system?

---

## Problem 5: Time-Independent Perturbation Theory (25 points)

Consider a hydrogen atom with the perturbation:

$$\hat{V} = \lambda \frac{e^2}{a_0^3}r^2$$

where $\lambda$ is a small dimensionless parameter and $a_0$ is the Bohr radius.

**(a)** (10 points) Calculate the first-order energy correction to the ground state. You may use:

$$\psi_{100}(r) = \frac{1}{\sqrt{\pi}a_0^{3/2}}e^{-r/a_0}$$

and the integral $\int_0^\infty r^n e^{-\alpha r} dr = \frac{n!}{\alpha^{n+1}}$.

**(b)** (8 points) Without doing a detailed calculation, explain qualitatively whether the first-order correction to the $n=2$ states will lift the degeneracy. Which states remain degenerate and why?

**(c)** (7 points) Calculate the second-order correction to the ground state energy. You may approximate the sum over intermediate states by using the closure relation with an average excitation energy $\bar{E} \approx \frac{3}{4}|E_1|$. Express your answer in terms of $\lambda$, $e^2$, and $a_0$.

---

## Problem 6: Time-Dependent Perturbation Theory (25 points)

A hydrogen atom in its ground state is exposed to a uniform electric field that is suddenly turned on at $t = 0$:

$$\vec{E}(t) = E_0 \hat{z} \cdot \Theta(t)$$

where $\Theta(t)$ is the Heaviside step function.

**(a)** (8 points) Write down the perturbation Hamiltonian $\hat{V}$. Which $n = 2$ states can be excited from the ground state by this perturbation, and why? (Consider selection rules.)

**(b)** (10 points) Using first-order time-dependent perturbation theory, calculate the probability of finding the atom in the $|2,1,0\rangle$ state at time $t$. You may use:

$$\langle 2,1,0|\hat{z}|1,0,0\rangle = \frac{2^7}{3^5}\sqrt{\frac{1}{3}}a_0 \approx 0.745 a_0$$

**(c)** (7 points) Describe the long-time behavior of this transition probability. Does it grow without bound? Why is first-order perturbation theory eventually inadequate?

---

## Problem 7: Identical Particles (25 points)

Consider two identical fermions in a one-dimensional harmonic oscillator potential.

**(a)** (8 points) If both fermions are spin-1/2, write down the ground state spatial and spin wavefunction. What is the ground state energy?

**(b)** (9 points) Now suppose the fermions interact via a contact interaction:

$$\hat{V}_{int} = \lambda\delta(x_1 - x_2)$$

where $\lambda > 0$. Using first-order perturbation theory, calculate the energy shift due to this interaction for the ground state you found in part (a).

**(c)** (8 points) What is the first excited state of this two-fermion system (ignoring the interaction)? Is it degenerate? How does the contact interaction affect this excited state?

---

## Problem 8: Scattering Theory (25 points)

A particle of mass $m$ and energy $E = \frac{\hbar^2 k^2}{2m}$ scatters from a spherical potential:

$$V(r) = \begin{cases} V_0 & r < a \\ 0 & r > a \end{cases}$$

where $V_0 > 0$ (repulsive barrier).

**(a)** (8 points) In the low-energy limit ($ka \ll 1$), show that only $s$-wave ($\ell = 0$) scattering is significant. Write down the radial Schrodinger equation for $\ell = 0$ inside and outside the sphere.

**(b)** (10 points) Apply boundary conditions at $r = a$ to find the $s$-wave phase shift $\delta_0$. Express your answer in terms of $k$, $\kappa = \sqrt{2m(V_0 - E)/\hbar^2}$, and $a$.

**(c)** (7 points) Calculate the total scattering cross section $\sigma_{tot}$ in the limit $V_0 \to \infty$ (hard sphere). Show that $\sigma_{tot} = 4\pi a^2$ in the low-energy limit. Compare this to the classical geometric cross section and explain the factor of 4.

---

## End of Exam

**Checklist before submitting:**
- [ ] All problems attempted
- [ ] Work shown for each part
- [ ] Units checked where applicable
- [ ] Name on all pages

**Good luck!**

---

*This exam is modeled after PhD qualifying exams from MIT, Yale, Caltech, and Princeton physics departments.*
