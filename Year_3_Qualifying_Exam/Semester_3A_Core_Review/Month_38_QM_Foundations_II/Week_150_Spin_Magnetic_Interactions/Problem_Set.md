# Week 150: Spin and Magnetic Interactions - Problem Set

## Instructions

- **Total Problems:** 28
- **Recommended Time:** 4-5 hours (spread across the week)
- **Difficulty Levels:** Direct Application (D), Intermediate (I), Challenging (C)
- **Exam Conditions:** For problems marked with *, attempt under timed conditions (15-20 min each)

---

## Part A: Pauli Matrices and Spin Operators (Problems 1-8)

### Problem 1 (D)
Verify the following Pauli matrix properties:
(a) $\sigma_x^2 = \sigma_y^2 = \sigma_z^2 = I$
(b) $\text{Tr}(\sigma_i) = 0$ for $i = x, y, z$
(c) $\det(\sigma_i) = -1$ for $i = x, y, z$

### Problem 2 (D)
Prove the commutation relations:
(a) $[\sigma_x, \sigma_y] = 2i\sigma_z$
(b) $[\sigma_y, \sigma_z] = 2i\sigma_x$
(c) $[\sigma_z, \sigma_x] = 2i\sigma_y$

### Problem 3 (D)
Prove the anticommutation relations $\{\sigma_i, \sigma_j\} = 2\delta_{ij}I$.

### Problem 4 (I)*
Show that any $2\times 2$ matrix can be written as:
$$A = a_0 I + a_1\sigma_x + a_2\sigma_y + a_3\sigma_z$$
Find the coefficients $a_i$ in terms of $A$.

### Problem 5 (I)
Using the identity $\sigma_i\sigma_j = \delta_{ij}I + i\epsilon_{ijk}\sigma_k$, prove that:
$$(\boldsymbol{\sigma}\cdot\mathbf{a})(\boldsymbol{\sigma}\cdot\mathbf{b}) = (\mathbf{a}\cdot\mathbf{b})I + i\boldsymbol{\sigma}\cdot(\mathbf{a}\times\mathbf{b})$$

### Problem 6 (I)
Calculate the eigenvalues and normalized eigenvectors of $\sigma_x$, $\sigma_y$, and verify they are orthonormal.

### Problem 7 (C)*
Prove that $e^{i\theta(\hat{n}\cdot\boldsymbol{\sigma})} = \cos\theta\, I + i\sin\theta\,(\hat{n}\cdot\boldsymbol{\sigma})$ where $\hat{n}$ is a unit vector.

### Problem 8 (C)
Find all $2\times 2$ unitary matrices that commute with all three Pauli matrices. What does this tell you about the structure of $SU(2)$?

---

## Part B: Spin States and Measurements (Problems 9-15)

### Problem 9 (D)
A spin-1/2 particle is in the state $|\psi\rangle = \frac{1}{\sqrt{3}}|\uparrow\rangle + \sqrt{\frac{2}{3}}|\downarrow\rangle$.
(a) Calculate the probabilities of measuring $S_z = +\hbar/2$ and $S_z = -\hbar/2$.
(b) Calculate $\langle S_z\rangle$ and $\langle S_z^2\rangle$.
(c) Find $\Delta S_z$.

### Problem 10 (D)
For the state in Problem 9:
(a) What is the probability of measuring $S_x = +\hbar/2$?
(b) What is $\langle S_x\rangle$?

### Problem 11 (I)*
A spin-1/2 particle is in the state:
$$|\psi\rangle = \cos\frac{\pi}{8}|\uparrow\rangle + e^{i\pi/4}\sin\frac{\pi}{8}|\downarrow\rangle$$
(a) What are the Bloch sphere angles $(\theta, \phi)$?
(b) Calculate $\langle S_x\rangle$, $\langle S_y\rangle$, $\langle S_z\rangle$.
(c) In what direction is the spin "pointing"?

### Problem 12 (I)
Find the eigenstate of $S_n = \mathbf{S}\cdot\hat{n}$ with eigenvalue $+\hbar/2$, where:
(a) $\hat{n} = (\sin 30°, 0, \cos 30°)$
(b) $\hat{n} = (1/\sqrt{2}, 1/\sqrt{2}, 0)$

### Problem 13 (I)*
A spin-1/2 particle is prepared in the state $|\uparrow_z\rangle$. If $S_x$ is measured, what are the possible outcomes and their probabilities? After measuring $S_x = +\hbar/2$, what is the probability of subsequently measuring $S_z = +\hbar/2$?

### Problem 14 (C)
For a spin-1/2 particle in state $|\psi\rangle$, prove that:
$$\langle S_x\rangle^2 + \langle S_y\rangle^2 + \langle S_z\rangle^2 = \frac{\hbar^2}{4}$$
only if $|\psi\rangle$ is a pure state. What happens for mixed states?

### Problem 15 (C)*
A spin is initially in the state $|+z\rangle$. A measurement is made of the component of spin along a direction at angle $\theta$ from the z-axis in the xz-plane.
(a) What are the probabilities for the two possible outcomes?
(b) For what angle is the uncertainty in this measurement maximized?

---

## Part C: Stern-Gerlach Experiments (Problems 16-19)

### Problem 16 (D)
A beam of spin-1/2 particles in the state $|\uparrow_z\rangle$ enters a Stern-Gerlach apparatus with field gradient along the x-direction.
(a) What fraction emerges in the upper beam (spin-up along x)?
(b) If only the upper beam is kept and passed through another SG apparatus along z, what fractions emerge in each beam?

### Problem 17 (I)*
An unpolarized beam of silver atoms passes through three consecutive Stern-Gerlach devices:
- SG1: oriented along z, upper beam selected
- SG2: oriented at angle $\theta$ from z in the xz-plane, upper beam selected
- SG3: oriented along z

(a) What fraction of the original beam emerges from SG3 in the upper beam?
(b) For what angle $\theta$ is this fraction maximized?

### Problem 18 (I)
A modified Stern-Gerlach experiment uses three devices: SGz, SGx, SGz.
(a) If only the upper beam from each device is selected, what is the final intensity relative to the initial SGz output?
(b) What if the SGx device is removed (just two SGz devices in series)?

### Problem 19 (C)
Design a Stern-Gerlach sequence that prepares the state $|+y\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle + i|\downarrow\rangle)$ starting from an unpolarized beam. Explain why this is or isn't possible.

---

## Part D: Spin Dynamics (Problems 20-24)

### Problem 20 (D)
A spin-1/2 particle with gyromagnetic ratio $\gamma$ is placed in a uniform magnetic field $\mathbf{B} = B_0\hat{z}$.
(a) Write down the Hamiltonian.
(b) If the initial state is $|\uparrow_x\rangle$, find $|\psi(t)\rangle$.
(c) Calculate $\langle S_x\rangle(t)$ and $\langle S_z\rangle(t)$.

### Problem 21 (I)*
An electron spin is initially in state $|\uparrow_z\rangle$ and is placed in a magnetic field $\mathbf{B} = B_0\hat{x}$.
(a) Find the state $|\psi(t)\rangle$.
(b) At what time is the spin completely flipped to $|\downarrow_z\rangle$?
(c) Calculate the probability of finding $S_z = +\hbar/2$ as a function of time.

### Problem 22 (I)
A spin-1/2 particle is in a time-dependent magnetic field:
$$\mathbf{B}(t) = B_0\hat{z} + B_1(\cos\omega t\,\hat{x} + \sin\omega t\,\hat{y})$$

Write down the Hamiltonian. At resonance ($\omega = \gamma B_0$), show that in the rotating frame the effective Hamiltonian is simply $H_{\text{eff}} = -\gamma B_1 S_x$.

### Problem 23 (C)*
For the system in Problem 22 at resonance, starting from $|\uparrow_z\rangle$:
(a) Find the probability $P_{\downarrow}(t)$ of measuring spin-down along z.
(b) This is Rabi oscillation. What is the Rabi frequency?
(c) How long does a complete spin flip take?

### Problem 24 (C)
**Spin Echo Problem:** A spin is initially in state $|\uparrow_z\rangle$. It evolves in a field $\mathbf{B} = B_0\hat{z}$ for time $\tau$, then a $\pi$ pulse about x is applied, then it evolves for another time $\tau$.
(a) What is the final state?
(b) Show that this "refocuses" the spin regardless of the exact value of $B_0$. Why is this useful in NMR?

---

## Part E: Advanced Applications (Problems 25-28)

### Problem 25 (I)
Calculate the thermal equilibrium polarization of an ensemble of electron spins in a magnetic field $B = 1$ T at temperature $T = 300$ K. The polarization is defined as:
$$P = \frac{N_{\uparrow} - N_{\downarrow}}{N_{\uparrow} + N_{\downarrow}}$$

### Problem 26 (I)*
The magnetic moment of an electron is $\boldsymbol{\mu} = -g_s\mu_B\mathbf{S}/\hbar$ where $g_s \approx 2$ and $\mu_B$ is the Bohr magneton.
(a) Calculate the energy splitting between spin-up and spin-down in a 3 T field.
(b) What frequency of electromagnetic radiation causes transitions between these levels?
(c) In what part of the spectrum is this? (microwave, infrared, visible, etc.)

### Problem 27 (C)
A spin-1/2 particle is subject to a Hamiltonian $H = \mathbf{S}\cdot\mathbf{B}(t)$ where the magnetic field rotates slowly (adiabatically) from $\hat{z}$ to $-\hat{z}$ along a great circle on the unit sphere.
(a) If the spin starts in $|\uparrow_z\rangle$, what state does it end up in?
(b) What is the geometric (Berry) phase accumulated?

### Problem 28 (C)* - Qualifying Exam Style
Consider a two-level system (spin-1/2) subject to the Hamiltonian:
$$H = \frac{\hbar\omega_0}{2}\sigma_z + \frac{\hbar\Omega}{2}(\sigma_+ e^{-i\omega t} + \sigma_- e^{i\omega t})$$

where $\sigma_{\pm} = (\sigma_x \pm i\sigma_y)/2$.

(a) Transform to the interaction picture with respect to $H_0 = \frac{\hbar\omega_0}{2}\sigma_z$.
(b) Make the rotating wave approximation and find the effective Hamiltonian.
(c) For the resonant case $\omega = \omega_0$, find the probability of transition from $|\uparrow\rangle$ to $|\downarrow\rangle$ as a function of time.
(d) How does this relate to magnetic resonance experiments?

---

## Bonus Problems

### Bonus 1
The density matrix for a spin-1/2 ensemble is:
$$\rho = \frac{1}{2}(I + \mathbf{P}\cdot\boldsymbol{\sigma})$$
where $\mathbf{P}$ is the polarization vector with $|\mathbf{P}| \leq 1$.
(a) Show that $\text{Tr}(\rho) = 1$ and $\rho^{\dagger} = \rho$.
(b) When is $\rho$ a pure state?
(c) Calculate $\langle S_z\rangle$ in terms of $\mathbf{P}$.

### Bonus 2
Prove that the set of $2\times 2$ unitary matrices with determinant 1 forms the group $SU(2)$, and that every element can be written as:
$$U = e^{i(\alpha\sigma_x + \beta\sigma_y + \gamma\sigma_z)}$$
for some real $\alpha, \beta, \gamma$.

---

## Answer Key (Quick Reference)

| Problem | Key Answer |
|---------|------------|
| 9a | $P(\uparrow) = 1/3$, $P(\downarrow) = 2/3$ |
| 9b | $\langle S_z\rangle = -\hbar/6$ |
| 10a | $P(S_x = +\hbar/2) = 1/2 + \sqrt{2}/3$ |
| 11a | $\theta = \pi/4$, $\phi = \pi/4$ |
| 16a | 1/2 |
| 17a | $\frac{1}{4}\cos^2(\theta/2)$ |
| 21b | $t = \pi/(\gamma B_0)$ |
| 23b | $\Omega_R = \gamma B_1$ |
| 25 | $P \approx 0.0023$ at 300 K |
| 26b | $\nu \approx 84$ GHz |

---

**Detailed solutions are provided in Problem_Solutions.md**
