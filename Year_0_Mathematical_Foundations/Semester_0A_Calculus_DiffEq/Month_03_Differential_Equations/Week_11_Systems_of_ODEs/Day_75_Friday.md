# Day 75: Week 11 Problem Set — Systems of ODEs Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Parts I-II |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Parts III-IV |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Review |

**Total Study Time: 7.5 hours**

---

## 📋 Instructions

This comprehensive problem set covers systems of ODEs. Show all work clearly.

**Total Points:** 200

---

# 📝 PART I: MATRIX FORMULATION (40 points)

**Problem 1** (8 pts): Convert to a first-order system:
$$y''' - 3y'' + 2y' = e^t$$

**Problem 2** (8 pts): Write in matrix form:
$$x' = 3x - 2y + 1$$
$$y' = x + y - t$$

**Problem 3** (8 pts): Verify that $\mathbf{x}_1 = \begin{pmatrix} e^{2t} \\ e^{2t} \end{pmatrix}$ and $\mathbf{x}_2 = \begin{pmatrix} e^{-t} \\ 2e^{-t} \end{pmatrix}$ are solutions to:
$$\mathbf{x}' = \begin{pmatrix} 4 & -1 \\ 2 & 1 \end{pmatrix}\mathbf{x}$$

**Problem 4** (8 pts): Show these solutions are linearly independent using the Wronskian.

**Problem 5** (8 pts): Write the general solution and solve the IVP with $\mathbf{x}(0) = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$

---

# 📝 PART II: EIGENVALUE METHOD (50 points)

**Problem 6** (10 pts): Solve using eigenvalues:
$$\mathbf{x}' = \begin{pmatrix} 5 & -1 \\ 3 & 1 \end{pmatrix}\mathbf{x}$$

**Problem 7** (10 pts): Solve:
$$\mathbf{x}' = \begin{pmatrix} 1 & -2 \\ 1 & 3 \end{pmatrix}\mathbf{x}$$

**Problem 8** (10 pts): Solve with complex eigenvalues:
$$\mathbf{x}' = \begin{pmatrix} 1 & -5 \\ 1 & -1 \end{pmatrix}\mathbf{x}$$

**Problem 9** (10 pts): Solve with repeated eigenvalue:
$$\mathbf{x}' = \begin{pmatrix} 3 & -4 \\ 1 & -1 \end{pmatrix}\mathbf{x}$$

**Problem 10** (10 pts): Solve the IVP:
$$\mathbf{x}' = \begin{pmatrix} 2 & -1 \\ 3 & -2 \end{pmatrix}\mathbf{x}, \quad \mathbf{x}(0) = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

---

# 📝 PART III: PHASE PORTRAITS & STABILITY (50 points)

**Problem 11** (10 pts): Classify the equilibrium and describe the phase portrait:
$$\mathbf{x}' = \begin{pmatrix} -3 & 1 \\ 1 & -3 \end{pmatrix}\mathbf{x}$$

**Problem 12** (10 pts): Classify:
$$\mathbf{x}' = \begin{pmatrix} 2 & -5 \\ 1 & -2 \end{pmatrix}\mathbf{x}$$

**Problem 13** (10 pts): For what values of $k$ is the system stable?
$$\mathbf{x}' = \begin{pmatrix} -1 & k \\ k & -1 \end{pmatrix}\mathbf{x}$$

**Problem 14** (10 pts): Sketch the phase portrait:
$$\mathbf{x}' = \begin{pmatrix} 0 & 1 \\ -4 & 0 \end{pmatrix}\mathbf{x}$$

**Problem 15** (10 pts): A system has $\text{tr}(A) = -2$ and $\det(A) = 5$. Classify the equilibrium.

---

# 📝 PART IV: APPLICATIONS (60 points)

**Problem 16** (15 pts): **Coupled Oscillators**

Two masses connected by springs satisfy:
$$m_1 x_1'' = -k_1 x_1 + k_2(x_2 - x_1)$$
$$m_2 x_2'' = -k_2(x_2 - x_1)$$

With $m_1 = m_2 = 1$, $k_1 = 3$, $k_2 = 2$:
(a) Convert to a first-order system
(b) Find the eigenvalues (normal mode frequencies)
(c) Describe the two normal modes physically

**Problem 17** (15 pts): **Predator-Prey**

Linearized Lotka-Volterra near equilibrium:
$$x' = -0.1x + 0.2y$$
$$y' = -0.5x$$
where x = deviation in prey, y = deviation in predators.
(a) Find eigenvalues
(b) Classify the equilibrium
(c) Describe the population dynamics

**Problem 18** (15 pts): **RLC Network**

Two coupled circuits share a common inductor:
$$L\frac{dI_1}{dt} = -R_1 I_1 + M \frac{dI_2}{dt}$$
$$L\frac{dI_2}{dt} = -R_2 I_2 + M \frac{dI_1}{dt}$$

With L = 1, M = 0, $R_1 = 2$, $R_2 = 3$:
(a) Write as a system
(b) Solve for $I_1(t)$ and $I_2(t)$
(c) Determine long-term behavior

**Problem 19** (15 pts): **Quantum Two-Level System**

The Schrödinger equation for a two-level atom in a field:
$$i\hbar\frac{d}{dt}\begin{pmatrix} c_1 \\ c_2 \end{pmatrix} = \begin{pmatrix} E_1 & V \\ V & E_2 \end{pmatrix}\begin{pmatrix} c_1 \\ c_2 \end{pmatrix}$$

With $\hbar = 1$, $E_1 = 1$, $E_2 = -1$, $V = 1$:
(a) Find the energy eigenvalues
(b) If $c_1(0) = 1$, $c_2(0) = 0$, find the probability $|c_2(t)|^2$
(c) What is the oscillation frequency (Rabi frequency)?

---

# ✅ ANSWER KEY

## Part I: Matrix Formulation

**1.** Let $x_1 = y$, $x_2 = y'$, $x_3 = y''$:
$$\mathbf{x}' = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & -2 & 3 \end{pmatrix}\mathbf{x} + \begin{pmatrix} 0 \\ 0 \\ e^t \end{pmatrix}$$

**2.** $\mathbf{x}' = \begin{pmatrix} 3 & -2 \\ 1 & 1 \end{pmatrix}\mathbf{x} + \begin{pmatrix} 1 \\ -t \end{pmatrix}$

**3.** Direct substitution verifies both solutions.

**4.** $W = \begin{vmatrix} e^{2t} & e^{-t} \\ e^{2t} & 2e^{-t} \end{vmatrix} = e^t \neq 0$ ✓

**5.** $\mathbf{x} = c_1\begin{pmatrix} 1 \\ 1 \end{pmatrix}e^{2t} + c_2\begin{pmatrix} 1 \\ 2 \end{pmatrix}e^{-t}$; with IC: $c_1 = 2$, $c_2 = 1$

## Part II: Eigenvalue Method

**6.** $\lambda = 4, 2$; $\mathbf{x} = c_1(1,1)^T e^{4t} + c_2(1,3)^T e^{2t}$

**7.** $\lambda = 2 \pm i$; stable spiral
$$\mathbf{x} = e^{2t}[c_1\begin{pmatrix} 2\cos t \\ \cos t - \sin t \end{pmatrix} + c_2\begin{pmatrix} 2\sin t \\ \sin t + \cos t \end{pmatrix}]$$

**8.** $\lambda = \pm 2i$; center (ellipses)

**9.** $\lambda = 1$ (repeated); $\mathbf{v} = (2,1)^T$, $\mathbf{w} = (1,0)^T$
$$\mathbf{x} = c_1\begin{pmatrix} 2 \\ 1 \end{pmatrix}e^t + c_2\left[t\begin{pmatrix} 2 \\ 1 \end{pmatrix} + \begin{pmatrix} 1 \\ 0 \end{pmatrix}\right]e^t$$

**10.** $\lambda = 1, -1$; $\mathbf{x} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}e^t$

## Part III: Phase Portraits & Stability

**11.** $\lambda = -2, -4$; **stable node**

**12.** $\lambda = \pm i$; **center** (stable, not asymptotic)

**13.** Need both eigenvalues with negative real part: $-1 < k < 1$

**14.** $\lambda = \pm 2i$; center with elliptical orbits

**15.** $\tau^2 - 4\Delta = 4 - 20 = -16 < 0$, $\tau < 0$: **stable spiral**

## Part IV: Applications

**16.** 
(a) 4×4 system with $\mathbf{x} = (x_1, x_1', x_2, x_2')^T$
(b) Normal frequencies: $\omega_1 = \sqrt{1} = 1$, $\omega_2 = \sqrt{5}$
(c) In-phase and out-of-phase oscillation modes

**17.**
(a) $\lambda = -0.05 \pm 0.3i\sqrt{11/9}$ (approximately)
(b) Stable spiral
(c) Damped oscillations around equilibrium

**18.**
(a) $\frac{d}{dt}\begin{pmatrix} I_1 \\ I_2 \end{pmatrix} = \begin{pmatrix} -2 & 0 \\ 0 & -3 \end{pmatrix}\begin{pmatrix} I_1 \\ I_2 \end{pmatrix}$
(b) $I_1 = I_{10}e^{-2t}$, $I_2 = I_{20}e^{-3t}$
(c) Both currents decay to zero

**19.**
(a) $E_\pm = \pm\sqrt{2}$
(b) $|c_2(t)|^2 = \frac{1}{2}\sin^2(\sqrt{2}t)$
(c) Rabi frequency $\Omega_R = 2\sqrt{2}$

---

## 📊 Scoring Guide

| Part | Points | Your Score |
|------|--------|------------|
| I: Matrix Formulation (5 × 8) | 40 | |
| II: Eigenvalue Method (5 × 10) | 50 | |
| III: Phase Portraits (5 × 10) | 50 | |
| IV: Applications (4 × 15) | 60 | |
| **TOTAL** | **200** | |

---

## 🔜 Tomorrow: Computational Lab

---

*"Systems of ODEs reveal how coupled components evolve together—the mathematics of interaction."*
