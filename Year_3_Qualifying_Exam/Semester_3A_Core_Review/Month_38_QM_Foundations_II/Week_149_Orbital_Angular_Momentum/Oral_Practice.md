# Week 149: Orbital Angular Momentum - Oral Exam Practice

## Introduction

PhD qualifying oral exams test your ability to explain concepts clearly, derive results on demand, and respond to follow-up questions. This guide provides practice questions with suggested response frameworks.

---

## Oral Exam Format

**Duration:** 20-30 minutes per topic
**Structure:**
1. Initial question (1-2 minutes to think)
2. Explanation/derivation (5-10 minutes)
3. Follow-up questions (5-10 minutes)
4. "What if..." variations

**Evaluation Criteria:**
- Clarity of explanation
- Physical intuition
- Mathematical rigor
- Response to probing questions

---

## Question 1: Angular Momentum Commutation Relations

### Initial Question
"Derive the commutation relations for angular momentum from the canonical commutation relations."

### Suggested Response Framework

**Opening (30 seconds):**
"Angular momentum operators are defined as $L_i = \epsilon_{ijk}r_j p_k$, or explicitly $\mathbf{L} = \mathbf{r} \times \mathbf{p}$. The fundamental commutation relations follow from $[r_i, p_j] = i\hbar\delta_{ij}$."

**Main Derivation (3-4 minutes):**
1. Write $L_x = yp_z - zp_y$, $L_y = zp_x - xp_z$
2. Compute $[L_x, L_y]$ term by term
3. Show result is $i\hbar L_z$
4. State the general result: $[L_i, L_j] = i\hbar\epsilon_{ijk}L_k$

**Physical Insight (1 minute):**
"This means we cannot simultaneously measure all three components of angular momentum. The non-commutativity reflects the geometric fact that rotations about different axes don't commute."

### Likely Follow-up Questions

**Q: "Why does $[L^2, L_z] = 0$?"**

A: "Because $L^2$ is a Casimir operator - it commutes with all generators. Physically, knowing the magnitude of angular momentum doesn't constrain which direction it points."

**Q: "What about spin? Does it satisfy the same commutation relations?"**

A: "Yes, spin operators satisfy identical commutation relations $[S_i, S_j] = i\hbar\epsilon_{ijk}S_k$. This is required by the rotation group structure, regardless of whether the angular momentum is orbital or intrinsic."

---

## Question 2: Eigenvalue Spectrum

### Initial Question
"Using ladder operators, derive the eigenvalue spectrum of $L^2$ and $L_z$."

### Suggested Response Framework

**Setup (1 minute):**
"I'll define ladder operators $L_{\pm} = L_x \pm iL_y$ and use their commutation properties to derive the spectrum."

**Key steps:**
1. Show $[L_z, L_{\pm}] = \pm\hbar L_{\pm}$
2. Therefore $L_{\pm}|l,m\rangle$ has eigenvalue $(m \pm 1)\hbar$
3. Argue that $m$ is bounded (use $\langle L_x^2 + L_y^2\rangle \geq 0$)
4. At the bounds, $L_{\pm}$ annihilates the state
5. Derive $l(l+1)\hbar^2$ from the top state

**Result:**
"The eigenvalues are $L^2 = \hbar^2 l(l+1)$ and $L_z = m\hbar$ with $l = 0, 1/2, 1, \ldots$ and $m = -l, \ldots, l$."

### Follow-up Questions

**Q: "Why are half-integer values allowed for spin but not orbital angular momentum?"**

A: "For orbital angular momentum, the wave function must be single-valued under $\phi \to \phi + 2\pi$, requiring $e^{2\pi im} = 1$, so $m$ must be an integer. Spin has no such constraint because it doesn't correspond to motion in physical space."

**Q: "What's special about the $m = l$ state?"**

A: "It's annihilated by $L_+$, meaning angular momentum is maximally aligned with the z-axis. But note $\langle L_x\rangle = \langle L_y\rangle = 0$, so it's not actually pointing in the z-direction - it's in a cone of half-angle $\theta = \cos^{-1}(l/\sqrt{l(l+1)})$."

---

## Question 3: Spherical Harmonics

### Initial Question
"What are spherical harmonics and why are they important in quantum mechanics?"

### Suggested Response Framework

**Definition (1 minute):**
"Spherical harmonics $Y_l^m(\theta,\phi)$ are the simultaneous eigenfunctions of $L^2$ and $L_z$ in position space. They form a complete orthonormal basis for functions on the sphere."

**Explicit form:**
$$Y_l^m(\theta,\phi) = N_{lm}P_l^m(\cos\theta)e^{im\phi}$$

**Importance:**
1. Appear in any central potential problem (separation of variables)
2. Describe angular distributions in atomic orbitals
3. Determine selection rules for transitions
4. Basis for multipole expansions

**Examples:**
$$Y_0^0 = \frac{1}{\sqrt{4\pi}}, \quad Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta$$

### Follow-up Questions

**Q: "What does the parity of spherical harmonics tell us about selection rules?"**

A: "Under parity, $Y_l^m \to (-1)^l Y_l^m$. For electric dipole transitions, the matrix element $\langle f|\mathbf{r}|i\rangle$ must be non-zero. Since $\mathbf{r}$ has odd parity, we need $(-1)^{l_i}(-1)(-1)^{l_f} = 1$, giving $\Delta l = \pm 1$."

**Q: "How do you normalize spherical harmonics?"**

A: "By integrating $|Y_l^m|^2$ over the unit sphere with measure $d\Omega = \sin\theta\, d\theta\, d\phi$. The result is $\int |Y_l^m|^2 d\Omega = 1$."

---

## Question 4: Hydrogen Atom

### Initial Question
"Walk me through the solution of the hydrogen atom."

### Suggested Response Framework

**Overview (30 seconds):**
"The hydrogen atom with Coulomb potential $V(r) = -e^2/r$ is separable in spherical coordinates. The angular part gives spherical harmonics; the radial part gives quantized energies."

**Key steps:**
1. Separation: $\psi(r,\theta,\phi) = R(r)Y_l^m(\theta,\phi)$
2. Radial equation with effective potential
3. Asymptotic analysis: $u(r) \sim r^{l+1}$ near origin, $e^{-\kappa r}$ at infinity
4. Series solution leads to quantization
5. Energy: $E_n = -13.6\text{ eV}/n^2$

**Degeneracy:**
"The $n$-th level has $n^2$ states (plus factor of 2 for spin). This 'accidental' degeneracy reflects hidden $SO(4)$ symmetry."

### Follow-up Questions

**Q: "Why does the energy only depend on $n$ and not on $l$?"**

A: "This is due to an additional conserved quantity, the Runge-Lenz vector $\mathbf{A} = \frac{1}{2m}(\mathbf{p} \times \mathbf{L} - \mathbf{L} \times \mathbf{p}) - \frac{e^2 \mathbf{r}}{r}$. It generates rotations in a 4D space, enlarging the symmetry from $SO(3)$ to $SO(4)$."

**Q: "What breaks this degeneracy?"**

A: "Relativistic corrections (fine structure), spin-orbit coupling, and the Lamb shift all break the $l$-degeneracy. For example, fine structure gives an energy correction proportional to $1/(l+1/2)$."

---

## Question 5: Central Potentials (General)

### Initial Question
"Explain how to solve the Schrödinger equation for a general central potential."

### Suggested Response Framework

**Method (2 minutes):**
1. Use spherical coordinates
2. Separate angular and radial parts
3. Angular: spherical harmonics (universal for all central potentials)
4. Radial: depends on specific $V(r)$

**Radial equation:**
$$-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} + V_{\text{eff}}(r)u = Eu$$

where $u = rR$ and $V_{\text{eff}} = V(r) + \frac{\hbar^2 l(l+1)}{2mr^2}$

**Physical interpretation:**
"The centrifugal barrier $\propto l(l+1)/r^2$ creates an effective repulsion that keeps particles with $l > 0$ away from the origin."

### Follow-up Questions

**Q: "Compare the 3D harmonic oscillator to the hydrogen atom."**

A: "Both have degeneracy beyond what rotational symmetry alone would predict. For the oscillator, $E_N = \hbar\omega(N + 3/2)$ with $N = 2n_r + l$. States with the same $N$ are degenerate. This reflects $SU(3)$ symmetry."

**Q: "What determines how many bound states exist?"**

A: "For potentials falling off faster than $1/r$, there are finitely many bound states. Potentials falling as $1/r$ (Coulomb) have infinitely many bound states accumulating at $E = 0$."

---

## Quick-Fire Practice Questions

Answer each in 1-2 sentences:

1. **What is the physical meaning of the quantum number $m$?**
   - The z-component of angular momentum is $m\hbar$.

2. **Why can't we measure $L_x$ and $L_y$ simultaneously with certainty?**
   - They don't commute: $[L_x, L_y] = i\hbar L_z \neq 0$.

3. **What is the maximum $L_z$ for a state with $L^2 = 12\hbar^2$?**
   - $l(l+1) = 12 \Rightarrow l = 3$, so $L_z^{max} = 3\hbar$.

4. **Why is the ground state of hydrogen spherically symmetric?**
   - It has $l = 0$, so the angular part is $Y_0^0 = const$.

5. **What happens to $L_+|l,l\rangle$?**
   - It gives zero (annihilates the state).

---

## Practice Exercise

Practice explaining these topics aloud for 5 minutes each:

1. Derive $[L_x, L_y] = i\hbar L_z$
2. Show that $l(l+1)\hbar^2$ is the $L^2$ eigenvalue
3. Calculate the hydrogen ground state energy
4. Explain the physical meaning of spherical harmonics

**Self-evaluation criteria:**
- Did you state the key result within the first minute?
- Were your intermediate steps clear?
- Did you connect math to physics?
- Could you handle reasonable follow-up questions?

---

**Preparation Time:** 2-3 hours
**Recommended Practice:** Explain topics to a study partner or record yourself
