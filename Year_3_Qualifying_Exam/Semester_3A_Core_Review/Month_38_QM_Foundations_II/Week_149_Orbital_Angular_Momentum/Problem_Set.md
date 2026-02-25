# Week 149: Orbital Angular Momentum - Problem Set

## Instructions

- **Total Problems:** 27
- **Recommended Time:** 4-5 hours (spread across the week)
- **Difficulty Levels:** Direct Application (D), Intermediate (I), Challenging (C)
- **Exam Conditions:** For problems marked with *, attempt under timed conditions (15-20 min each)

---

## Part A: Angular Momentum Algebra (Problems 1-8)

### Problem 1 (D)
Verify the commutation relation $[L_x, L_y] = i\hbar L_z$ by explicit calculation using $L_x = yp_z - zp_y$ and $L_y = zp_x - xp_z$, and the canonical commutation relations.

### Problem 2 (D)
Show that $[L^2, L_z] = 0$ using the commutation relations between $L_x$, $L_y$, and $L_z$.

### Problem 3 (D)
Prove that the ladder operators satisfy:
(a) $[L_z, L_+] = \hbar L_+$
(b) $[L_z, L_-] = -\hbar L_-$
(c) $[L_+, L_-] = 2\hbar L_z$

### Problem 4 (I)*
A particle is in an eigenstate of $L^2$ with eigenvalue $6\hbar^2$.
(a) What are the possible values of $L_z$?
(b) What is the minimum uncertainty $\Delta L_x$ if the particle is in the state with maximum $L_z$?

### Problem 5 (I)
Consider the state $|\psi\rangle = \frac{1}{\sqrt{3}}|l=1, m=1\rangle + \sqrt{\frac{2}{3}}|l=1, m=0\rangle$.
(a) Calculate $\langle L_z\rangle$ and $\langle L_z^2\rangle$.
(b) Find the uncertainty $\Delta L_z$.
(c) What is the probability of measuring $L_z = -\hbar$?

### Problem 6 (I)*
For a particle in the state $|l=2, m=1\rangle$:
(a) Calculate $\langle L_x\rangle$.
(b) Calculate $\langle L_x^2\rangle$.
(c) Find the uncertainty $\Delta L_x$.

### Problem 7 (C)
Prove the generalized uncertainty relation for angular momentum:

$$\Delta L_x \cdot \Delta L_y \geq \frac{\hbar}{2}|\langle L_z\rangle|$$

Under what conditions is the equality satisfied?

### Problem 8 (C)*
An angular momentum eigenstate $|l,m\rangle$ is rotated by angle $\theta$ about the x-axis. Find the probability that a subsequent measurement of $L_z$ yields $m\hbar$. Express your answer for general $l$ and $m$, then evaluate explicitly for $l=1$, $m=1$.

---

## Part B: Ladder Operators and Matrix Elements (Problems 9-14)

### Problem 9 (D)
Using ladder operators, calculate:
(a) $L_+|2,1\rangle$
(b) $L_-|2,1\rangle$
(c) $L_+L_-|2,1\rangle$

### Problem 10 (D)
For $l=1$, construct the matrix representations of $L_x$, $L_y$, and $L_z$ in the basis $\{|1,1\rangle, |1,0\rangle, |1,-1\rangle\}$.

### Problem 11 (I)*
Find the eigenvalues and normalized eigenvectors of $L_x$ for $l=1$. Express the eigenstates in terms of the $|l,m\rangle$ basis.

### Problem 12 (I)
A particle has angular momentum $l=1$ and is in the state:

$$|\psi\rangle = \frac{1}{\sqrt{6}}\begin{pmatrix} 1 \\ 2 \\ 1 \end{pmatrix}$$

in the $\{|1,1\rangle, |1,0\rangle, |1,-1\rangle\}$ basis.

(a) Calculate $\langle L_z\rangle$ and $\langle L^2\rangle$.
(b) What is the probability of measuring $L_x = \hbar$?
(c) If $L_x$ is measured and the result is $\hbar$, what is the state immediately after measurement?

### Problem 13 (C)*
For $l=2$, a particle is in the eigenstate of $L_x$ with eigenvalue $\hbar$.
(a) Express this state as a linear combination of $|2,m\rangle$ states.
(b) What is the probability of measuring $L_z = 0$?

### Problem 14 (C)
Prove that for any angular momentum state $|l,m\rangle$:

$$\langle l,m|L_x^2 + L_y^2|l,m\rangle = \hbar^2[l(l+1) - m^2]$$

---

## Part C: Spherical Harmonics (Problems 15-19)

### Problem 15 (D)
Verify that $Y_1^0(\theta,\phi) = \sqrt{\frac{3}{4\pi}}\cos\theta$ satisfies:
(a) $L^2 Y_1^0 = 2\hbar^2 Y_1^0$
(b) $L_z Y_1^0 = 0$

### Problem 16 (D)
Calculate the following integrals:
(a) $\int Y_2^1(\theta,\phi)^* Y_1^0(\theta,\phi) d\Omega$
(b) $\int |Y_2^2(\theta,\phi)|^2 d\Omega$
(c) $\int Y_1^0(\theta,\phi) Y_1^0(\theta,\phi) Y_2^0(\theta,\phi) d\Omega$

### Problem 17 (I)*
A particle on a sphere is in the state:

$$\psi(\theta,\phi) = \frac{1}{\sqrt{3}}Y_1^0 + \sqrt{\frac{2}{3}}Y_1^1$$

(a) What is the probability of measuring $L^2 = 2\hbar^2$?
(b) What is $\langle L_z\rangle$?
(c) Calculate the expectation value $\langle\cos\theta\rangle$.

### Problem 18 (I)
Show that:

$$Y_l^m(\pi - \theta, \phi + \pi) = (-1)^l Y_l^m(\theta,\phi)$$

and interpret this result in terms of parity.

### Problem 19 (C)
Using the raising operator $L_+ = \hbar e^{i\phi}\left(\frac{\partial}{\partial\theta} + i\cot\theta\frac{\partial}{\partial\phi}\right)$, derive $Y_1^1(\theta,\phi)$ from $Y_1^0(\theta,\phi)$.

---

## Part D: Central Potentials (Problems 20-23)

### Problem 20 (D)
For a particle in a 3D isotropic harmonic oscillator $V(r) = \frac{1}{2}m\omega^2 r^2$:
(a) Write down the effective radial potential $V_{\text{eff}}(r)$ for angular momentum $l$.
(b) At what radius does $V_{\text{eff}}(r)$ have its minimum for $l=1$?

### Problem 21 (I)*
Consider an electron in a spherically symmetric potential $V(r)$. Prove that if $\psi_{nlm}(r,\theta,\phi)$ is an energy eigenstate, then $\psi_{nl,-m}(r,\theta,\phi) = (-1)^m\psi_{nlm}^*(r,\theta,\phi)$ is also an energy eigenstate with the same energy.

### Problem 22 (I)
A particle of mass $m$ moves in the potential:

$$V(r) = \begin{cases} 0 & r < a \\ \infty & r \geq a \end{cases}$$

(a) Write down the radial equation for $l=0$ and solve for the energy eigenvalues.
(b) What is the ground state energy?
(c) What is the degeneracy of the first excited state?

### Problem 23 (C)*
For a 3D isotropic harmonic oscillator, show that the energy levels are $E_N = \hbar\omega(N + 3/2)$ where $N = 2n_r + l$ with $n_r = 0, 1, 2, \ldots$ and $l = 0, 1, 2, \ldots$. Calculate the degeneracy of the $N=2$ level.

---

## Part E: Hydrogen Atom (Problems 24-27)

### Problem 24 (D)
For the hydrogen atom:
(a) What is the energy of the ground state in eV?
(b) What is the Bohr radius in Angstroms?
(c) What is the most probable radius for an electron in the ground state?

### Problem 25 (I)*
For the hydrogen atom state $|n=2, l=1, m=0\rangle$:
(a) Write down the complete wave function $\psi_{210}(r,\theta,\phi)$.
(b) At what radius is the radial probability density maximum?
(c) Calculate $\langle r\rangle$ and compare with part (b).

### Problem 26 (C)
An electron in a hydrogen atom is in the state:

$$|\psi\rangle = \frac{1}{\sqrt{3}}|2,0,0\rangle + \sqrt{\frac{2}{3}}|2,1,0\rangle$$

where $|n,l,m\rangle$ denotes the hydrogen atom eigenstate.

(a) What is the expectation value of the energy?
(b) What is $\langle L^2\rangle$?
(c) If the angular momentum magnitude is measured, what values are possible and with what probabilities?
(d) After measuring $L^2 = 2\hbar^2$, what is the probability of subsequently measuring $L_z = \hbar$?

### Problem 27 (C)* - Yale Qualifying Exam Style
Consider a hydrogen atom in its ground state. A measurement of the operator $\mathbf{L}\cdot\hat{n}$, where $\hat{n}$ is a unit vector in an arbitrary direction, is performed.

(a) What values can be obtained from this measurement, and with what probabilities?
(b) If instead the atom is in the state $|2,1,1\rangle$ and we measure $L_x$, what are the possible outcomes and their probabilities?
(c) For the state $|2,1,1\rangle$, calculate $\langle L_x^2\rangle - \langle L_x\rangle^2$.

---

## Bonus Problems (Optional - Research Level)

### Bonus Problem 1
The Casimir operators of $SO(3)$ are operators that commute with all generators. Show that $L^2$ is the only independent Casimir operator for orbital angular momentum.

### Bonus Problem 2
Derive the Wigner d-matrix elements $d^l_{mm'}(\beta)$ for $l=1$ using the rotation operator $e^{-i\beta L_y/\hbar}$.

### Bonus Problem 3
Show that the hydrogen atom has an additional symmetry beyond rotational invariance (the Runge-Lenz vector) that explains the "accidental" $l$-degeneracy.

---

## Answer Key (Quick Reference)

| Problem | Key Answer |
|---------|------------|
| 4a | $m = -2, -1, 0, 1, 2$ |
| 4b | $\Delta L_x = \hbar\sqrt{3}$ |
| 5a | $\langle L_z\rangle = \hbar/3$ |
| 5c | $P(L_z = -\hbar) = 0$ |
| 6c | $\Delta L_x = \hbar\sqrt{5/2}$ |
| 9a | $2\hbar\|2,2\rangle$ |
| 12a | $\langle L_z\rangle = 0$, $\langle L^2\rangle = 2\hbar^2$ |
| 22b | $E_0 = \pi^2\hbar^2/(2ma^2)$ |
| 24a | $E_1 = -13.6$ eV |
| 26a | $E = -3.4$ eV |

---

**Detailed solutions are provided in Problem_Solutions.md**
