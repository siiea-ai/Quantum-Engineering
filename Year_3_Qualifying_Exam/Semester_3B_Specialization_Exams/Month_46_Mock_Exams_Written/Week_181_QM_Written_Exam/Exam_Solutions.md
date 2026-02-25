# Quantum Mechanics Written Exam - Complete Solutions

## Solution 1: Operator Algebra and Uncertainty

### Part (a): Proving the Generalized Uncertainty Relation (8 points)

**Strategy:** Use the Cauchy-Schwarz inequality.

Define shifted operators:
$$\hat{A}' = \hat{A} - \langle \hat{A} \rangle, \quad \hat{B}' = \hat{B} - \langle \hat{B} \rangle$$

Note that $[\hat{A}', \hat{B}'] = [\hat{A}, \hat{B}] = i\hat{C}$.

Consider the state $|\phi\rangle = (\hat{A}' + i\lambda \hat{B}')|\psi\rangle$ for real $\lambda$.

The norm is non-negative:
$$\langle \phi | \phi \rangle = \langle \psi|(\hat{A}' - i\lambda \hat{B}')(\hat{A}' + i\lambda \hat{B}')|\psi\rangle \geq 0$$

Expanding:
$$\langle (\hat{A}')^2 \rangle + \lambda^2 \langle (\hat{B}')^2 \rangle + i\lambda\langle [\hat{A}', \hat{B}'] \rangle \geq 0$$

$$(\Delta A)^2 + \lambda^2 (\Delta B)^2 - \lambda\langle \hat{C} \rangle \geq 0$$

This quadratic in $\lambda$ is non-negative, so its discriminant must satisfy:
$$\langle \hat{C} \rangle^2 - 4(\Delta A)^2(\Delta B)^2 \leq 0$$

Therefore:
$$\boxed{\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle \hat{C} \rangle|}$$

---

### Part (b): Explicit Calculation for Given State (7 points)

Given: $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$

Recall for the harmonic oscillator:
$$\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger), \quad \hat{p} = i\sqrt{\frac{m\omega\hbar}{2}}(\hat{a}^\dagger - \hat{a})$$

**Calculate $\langle x \rangle$:**
$$\langle x \rangle = \sqrt{\frac{\hbar}{2m\omega}}\left(\frac{1}{\sqrt{3}}\langle 0| + \sqrt{\frac{2}{3}}\langle 1|\right)(\hat{a} + \hat{a}^\dagger)\left(\frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle\right)$$

The only non-zero terms come from $\langle 0|\hat{a}|1\rangle = 1$ and $\langle 1|\hat{a}^\dagger|0\rangle = 1$:
$$\langle x \rangle = \sqrt{\frac{\hbar}{2m\omega}} \cdot 2 \cdot \frac{1}{\sqrt{3}} \cdot \sqrt{\frac{2}{3}} = \sqrt{\frac{\hbar}{2m\omega}} \cdot \frac{2\sqrt{2}}{3}$$

**Calculate $\langle x^2 \rangle$:**
$$\hat{x}^2 = \frac{\hbar}{2m\omega}(\hat{a} + \hat{a}^\dagger)^2 = \frac{\hbar}{2m\omega}(\hat{a}^2 + (\hat{a}^\dagger)^2 + \hat{a}\hat{a}^\dagger + \hat{a}^\dagger\hat{a})$$

$$= \frac{\hbar}{2m\omega}(\hat{a}^2 + (\hat{a}^\dagger)^2 + 2\hat{N} + 1)$$

$$\langle x^2 \rangle = \frac{\hbar}{2m\omega}\left(\frac{1}{3}(2 \cdot 0 + 1) + \frac{2}{3}(2 \cdot 1 + 1) + \frac{2\sqrt{2}}{3}\langle 0|1\rangle\sqrt{2}\right)$$

The $\hat{a}^2$ and $(\hat{a}^\dagger)^2$ terms contribute from cross-terms. After calculation:
$$\langle x^2 \rangle = \frac{\hbar}{2m\omega}\left(\frac{1}{3} + 2 + \frac{4}{3}\right) = \frac{\hbar}{2m\omega} \cdot \frac{11}{3}$$

$$(\Delta x)^2 = \frac{\hbar}{2m\omega}\left(\frac{11}{3} - \frac{8}{9}\right) = \frac{\hbar}{2m\omega} \cdot \frac{25}{9}$$

$$\Delta x = \frac{5}{3}\sqrt{\frac{\hbar}{2m\omega}}$$

By similar calculation (or using symmetry):
$$\Delta p = \frac{5}{3}\sqrt{\frac{m\omega\hbar}{2}}$$

**Verification:**
$$\Delta x \cdot \Delta p = \frac{25}{9} \cdot \frac{\hbar}{2} = \frac{25\hbar}{18} > \frac{\hbar}{2} \checkmark$$

$$\boxed{\Delta x \cdot \Delta p = \frac{25\hbar}{18} \geq \frac{\hbar}{2}}$$

---

### Part (c): Minimum Uncertainty States (10 points)

Given:
$$\hat{A} = \hat{a} + \hat{a}^\dagger, \quad \hat{B} = i(\hat{a}^\dagger - \hat{a})$$

Note these are proportional to position and momentum operators.

**Calculate the commutator:**
$$[\hat{A}, \hat{B}] = [(\hat{a} + \hat{a}^\dagger), i(\hat{a}^\dagger - \hat{a})]$$

$$= i[\hat{a}, \hat{a}^\dagger] - i[\hat{a}, -\hat{a}] + i[\hat{a}^\dagger, \hat{a}^\dagger] - i[\hat{a}^\dagger, -\hat{a}]$$

$$= i \cdot 1 + i \cdot 1 = 2i$$

So $\hat{C} = 2$ (a constant!).

$$\boxed{[\hat{A}, \hat{B}] = 2i}$$

The uncertainty relation gives:
$$\Delta A \cdot \Delta B \geq 1$$

**Minimum uncertainty states** are those that saturate this bound. These are the **coherent states**:

$$|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

which satisfy $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$.

For coherent states:
$$\langle \hat{A}^2 \rangle - \langle \hat{A} \rangle^2 = 1, \quad \langle \hat{B}^2 \rangle - \langle \hat{B} \rangle^2 = 1$$

$$\boxed{\text{Coherent states } |\alpha\rangle \text{ minimize the uncertainty product with } \Delta A \cdot \Delta B = 1}$$

---

## Solution 2: Finite Square Well

### Part (a): Setting Up the Equations (8 points)

The time-independent Schrodinger equation in each region:

**Region I** ($x < -a$):
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi$$

**Region II** ($|x| < a$):
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} - V_0\psi = E\psi$$

**Region III** ($x > a$):
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi$$

For bound states, $-V_0 < E < 0$. Define:
$$k = \sqrt{\frac{2m(E + V_0)}{\hbar^2}}, \quad \kappa = \sqrt{\frac{2m|E|}{\hbar^2}} = \sqrt{\frac{-2mE}{\hbar^2}}$$

**General solutions:**

Region I: $\psi_I = Ae^{\kappa x}$ (decaying as $x \to -\infty$)

Region II: $\psi_{II} = B\cos(kx) + C\sin(kx)$

Region III: $\psi_{III} = De^{-\kappa x}$ (decaying as $x \to +\infty$)

$$\boxed{\psi = \begin{cases} Ae^{\kappa x} & x < -a \\ B\cos(kx) + C\sin(kx) & |x| < a \\ De^{-\kappa x} & x > a \end{cases}}$$

---

### Part (b): Even-Parity Condition (10 points)

For even-parity states, $\psi(-x) = \psi(x)$, so $C = 0$ and $A = D$.

$$\psi_{II} = B\cos(kx), \quad \psi_{III} = Ae^{-\kappa x}$$

**Continuity at $x = a$:**
$$B\cos(ka) = Ae^{-\kappa a}$$

**Derivative continuity at $x = a$:**
$$-kB\sin(ka) = -\kappa Ae^{-\kappa a}$$

Dividing the second equation by the first:
$$\frac{-kB\sin(ka)}{B\cos(ka)} = \frac{-\kappa Ae^{-\kappa a}}{Ae^{-\kappa a}}$$

$$k\tan(ka) = \kappa$$

$$\boxed{\kappa = k\tan(ka)}$$

where:
$$k = \sqrt{\frac{2m(E + V_0)}{\hbar^2}}, \quad \kappa = \sqrt{\frac{-2mE}{\hbar^2}}$$

Note: $k^2 + \kappa^2 = \frac{2mV_0}{\hbar^2}$, which defines a circle in the $k$-$\kappa$ plane.

---

### Part (c): Infinite Well Limit (7 points)

As $V_0 \to \infty$ with $a$ fixed:
- The circle $k^2 + \kappa^2 = \frac{2mV_0}{\hbar^2}$ becomes very large
- For finite energy, $|E| \ll V_0$, so $k \approx \sqrt{\frac{2mV_0}{\hbar^2}} \to \infty$

The transcendental equation $\kappa = k\tan(ka)$ with large $k$ and relatively small $\kappa$ requires:
$$\tan(ka) \to 0^+ \quad \Rightarrow \quad ka \to n\pi$$

for even states (including the ground state with $n$ odd).

For the infinite square well of width $2a$:
$$E_n = \frac{\hbar^2 \pi^2 n^2}{2m(2a)^2} = \frac{\hbar^2 \pi^2 n^2}{8ma^2}$$

This matches the even-parity states.

**For $V_0 = 10\frac{\hbar^2}{2ma^2}$:**

Define dimensionless variables: $z = ka$, and note $z^2 + (\kappa a)^2 = \frac{2mV_0a^2}{\hbar^2} = 10$.

The graphical intersection of:
- $\kappa a = z\tan z$ (from transcendental equation)
- $z^2 + (\kappa a)^2 = 10$ (circle of radius $\sqrt{10}$)

The ground state occurs for $z \approx 1.43$ (numerical solution).

$$E_0 = \frac{\hbar^2 k^2}{2m} - V_0 = \frac{\hbar^2}{2ma^2}(z^2 - 10) \approx \frac{\hbar^2}{2ma^2}(2.04 - 10) = -\frac{7.96\hbar^2}{2ma^2}$$

$$\boxed{E_0 \approx -7.96\frac{\hbar^2}{2ma^2} \approx -0.80 V_0}$$

---

## Solution 3: Spin-1/2 Dynamics

### Part (a): Hamiltonian Matrix (8 points)

$$\hat{H} = -\gamma \vec{S} \cdot \vec{B} = -\gamma(S_x B_x + S_z B_z)$$

$$= -\gamma\left(\frac{\hbar}{2}\sigma_x \cdot B_0\sin(\omega t) + \frac{\hbar}{2}\sigma_z \cdot B_0\cos(\omega t)\right)$$

$$= -\frac{\gamma\hbar B_0}{2}\left(\cos(\omega t)\sigma_z + \sin(\omega t)\sigma_x\right)$$

$$\boxed{\hat{H} = -\frac{\gamma\hbar B_0}{2}\begin{pmatrix} \cos(\omega t) & \sin(\omega t) \\ \sin(\omega t) & -\cos(\omega t) \end{pmatrix}}$$

Let $\omega_0 = \gamma B_0$ (Larmor frequency), so $\hat{H} = -\frac{\hbar\omega_0}{2}\begin{pmatrix} \cos(\omega t) & \sin(\omega t) \\ \sin(\omega t) & -\cos(\omega t) \end{pmatrix}$.

---

### Part (b): Rotating Frame Transformation (10 points)

The unitary operator:
$$\hat{U}(t) = e^{i\omega t S_z/\hbar} = e^{i\omega t \sigma_z/2} = \begin{pmatrix} e^{i\omega t/2} & 0 \\ 0 & e^{-i\omega t/2} \end{pmatrix}$$

**Transformed operators:**
$$\hat{U}^\dagger \sigma_z \hat{U} = \sigma_z$$

$$\hat{U}^\dagger \sigma_x \hat{U} = \sigma_x\cos(\omega t) + \sigma_y\sin(\omega t)$$

The second term in $\hat{H}_{eff}$:
$$-i\hbar \hat{U}^\dagger \frac{\partial \hat{U}}{\partial t} = -i\hbar \cdot \frac{i\omega}{2}\sigma_z = \frac{\hbar\omega}{2}\sigma_z$$

Calculating $\hat{U}^\dagger \hat{H} \hat{U}$:
$$\hat{U}^\dagger \hat{H} \hat{U} = -\frac{\hbar\omega_0}{2}\left[\cos(\omega t)\sigma_z + \sin(\omega t)(\sigma_x\cos(\omega t) + \sigma_y\sin(\omega t))\right]$$

$$= -\frac{\hbar\omega_0}{2}\left[\sigma_z\cos(\omega t) + \sigma_x\sin(\omega t)\cos(\omega t) + \sigma_y\sin^2(\omega t)\right]$$

After using $\cos(\omega t) = 1$ and $\sin(\omega t)\cos(\omega t) = \frac{1}{2}\sin(2\omega t)$ and applying rotating wave approximation (keeping only time-independent terms):

$$\hat{H}_{eff} = -\frac{\hbar\omega_0}{2}\sigma_x + \frac{\hbar\omega}{2}\sigma_z - \frac{\hbar\omega_0}{2}\sigma_z$$

$$\boxed{\hat{H}_{eff} = \frac{\hbar(\omega - \omega_0)}{2}\sigma_z - \frac{\hbar\omega_0}{2}\sigma_x = \frac{\hbar}{2}\begin{pmatrix} \omega - \omega_0 & -\omega_0 \\ -\omega_0 & -(\omega - \omega_0) \end{pmatrix}}$$

---

### Part (c): Resonance Transition Probability (7 points)

At resonance, $\omega = \omega_0 = \gamma B_0$:

$$\hat{H}_{eff} = -\frac{\hbar\omega_0}{2}\sigma_x$$

This generates rotation about the $x$-axis in the rotating frame.

The state evolves as:
$$|\psi(t)\rangle_{rot} = e^{i\omega_0 t \sigma_x/2}|+\rangle = \cos\left(\frac{\omega_0 t}{2}\right)|+\rangle + i\sin\left(\frac{\omega_0 t}{2}\right)|-\rangle$$

The probability of finding $|-\rangle$:
$$P_{+\to -}(t) = \left|i\sin\left(\frac{\omega_0 t}{2}\right)\right|^2 = \sin^2\left(\frac{\omega_0 t}{2}\right)$$

$$\boxed{P_{+\to -}(t) = \sin^2\left(\frac{\gamma B_0 t}{2}\right)}$$

**Physical interpretation:** This is **Rabi oscillation** at resonance. The spin completely flips between $|+\rangle$ and $|-\rangle$ with period $T = \frac{2\pi}{\omega_0}$. At $t = \frac{\pi}{\omega_0}$, the spin is completely inverted (this is a "$\pi$-pulse"). This is the basis of NMR/MRI and qubit control in quantum computing.

---

## Solution 4: Angular Momentum Addition

### Part (a): Expressing in $|J, M\rangle$ Basis (8 points)

For two spin-1 particles, $j_1 = j_2 = 1$.

The possible total angular momentum values:
$$J = |j_1 - j_2|, ..., j_1 + j_2 = 0, 1, 2$$

The given state has $M = m_1 + m_2 = 1 + (-1) = 0$ and $M = (-1) + 1 = 0$, so $M = 0$.

The state is antisymmetric under particle exchange, which corresponds to $J = 1$ (the triplet manifold is symmetric, singlet is antisymmetric, and for integer $j$, the $J = 1$ state with $M = 0$ has the right antisymmetry).

Using Clebsch-Gordan coefficients for $1 \otimes 1 \to J$:

$$|J=1, M=0\rangle = \frac{1}{\sqrt{2}}(|1,1\rangle|1,-1\rangle - |1,-1\rangle|1,1\rangle)$$

$$\boxed{|\Psi\rangle = |J=1, M=0\rangle}$$

---

### Part (b): Expectation Values (8 points)

**$\langle S_{1z} \rangle$:**
$$\langle S_{1z} \rangle = \frac{1}{2}\left[|\langle 1,1|_1|^2 \cdot \hbar(1) + |\langle 1,-1|_1|^2 \cdot \hbar(-1)\right] = \frac{1}{2}[\hbar - \hbar] = 0$$

$$\boxed{\langle S_{1z} \rangle = 0}$$

By symmetry: $\boxed{\langle S_{2z} \rangle = 0}$

**$\langle S_{1z}S_{2z} \rangle$:**
$$\langle S_{1z}S_{2z} \rangle = \frac{1}{2}\left[\hbar(1) \cdot \hbar(-1) + \hbar(-1) \cdot \hbar(1)\right] = \frac{1}{2}[-\hbar^2 - \hbar^2] = -\hbar^2$$

$$\boxed{\langle S_{1z}S_{2z} \rangle = -\hbar^2}$$

---

### Part (c): Measurement of $\hat{J}^2$ (9 points)

Since $|\Psi\rangle = |J=1, M=0\rangle$ is an eigenstate of $\hat{J}^2$:

$$\hat{J}^2|J=1, M=0\rangle = \hbar^2 J(J+1)|J=1, M=0\rangle = 2\hbar^2|J=1, M=0\rangle$$

**Possible outcomes:** Only $J^2 = 2\hbar^2$ (corresponding to $J = 1$)

**Probability:** 100%

$$\boxed{\text{Outcome: } J^2 = 2\hbar^2 \text{ with probability } P = 1}$$

After measuring $J^2 = 2\hbar^2$, the state remains unchanged:

$$\boxed{|\Psi\rangle_{after} = |J=1, M=0\rangle = \frac{1}{\sqrt{2}}(|1,1\rangle_1|1,-1\rangle_2 - |1,-1\rangle_1|1,1\rangle_2)}$$

---

## Solution 5: Time-Independent Perturbation Theory

### Part (a): First-Order Ground State Correction (10 points)

The perturbation: $\hat{V} = \lambda\frac{e^2}{a_0^3}r^2$

First-order correction:
$$E_1^{(1)} = \langle 1,0,0|\hat{V}|1,0,0\rangle = \lambda\frac{e^2}{a_0^3}\langle r^2 \rangle_{100}$$

$$\langle r^2 \rangle_{100} = \int_0^\infty r^2 |\psi_{100}|^2 4\pi r^2 dr = \frac{4\pi}{\pi a_0^3}\int_0^\infty r^4 e^{-2r/a_0} dr$$

Using $\int_0^\infty r^n e^{-\alpha r} dr = \frac{n!}{\alpha^{n+1}}$:

$$\langle r^2 \rangle_{100} = \frac{4}{a_0^3} \cdot \frac{4!}{(2/a_0)^5} = \frac{4}{a_0^3} \cdot \frac{24 a_0^5}{32} = 3a_0^2$$

$$E_1^{(1)} = \lambda\frac{e^2}{a_0^3} \cdot 3a_0^2 = 3\lambda\frac{e^2}{a_0}$$

Since $E_1 = -\frac{e^2}{2a_0}$ (in Gaussian units), we can write:

$$\boxed{E_1^{(1)} = 3\lambda\frac{e^2}{a_0} = -6\lambda |E_1|}$$

---

### Part (b): Degeneracy of $n=2$ States (8 points)

The $n=2$ level has 4-fold degeneracy: $|2,0,0\rangle$, $|2,1,-1\rangle$, $|2,1,0\rangle$, $|2,1,1\rangle$.

The perturbation $\hat{V} = \lambda\frac{e^2}{a_0^3}r^2$ is a **scalar** (depends only on $r$).

**Selection rules:** Scalar operators cannot change $\ell$ or $m$. Therefore:
$$\langle 2,\ell,m|\hat{V}|2,\ell',m'\rangle = 0 \text{ unless } \ell = \ell' \text{ and } m = m'$$

The perturbation matrix in the $n=2$ subspace is diagonal, so no off-diagonal mixing occurs.

**Effect on degeneracy:**
- The $|2,0,0\rangle$ state gets shifted by $\langle 2,0,0|r^2|2,0,0\rangle$
- The $|2,1,m\rangle$ states all get the same shift (since $\langle r^2 \rangle$ doesn't depend on $m$)

$$\boxed{\text{The 3-fold degeneracy of } |2,1,m\rangle \text{ states remains. Only the } \ell=0 \text{ and } \ell=1 \text{ degeneracy is lifted.}}$$

---

### Part (c): Second-Order Correction (7 points)

$$E_1^{(2)} = \sum_{n \neq 1} \frac{|\langle n,\ell,m|\hat{V}|1,0,0\rangle|^2}{E_1 - E_n}$$

Using the closure approximation with average energy denominator:
$$E_1^{(2)} \approx \frac{1}{E_1 - \bar{E}}\left[\langle 1,0,0|\hat{V}^2|1,0,0\rangle - |\langle 1,0,0|\hat{V}|1,0,0\rangle|^2\right]$$

With $\bar{E} = E_1 + \frac{3}{4}|E_1|$:
$$E_1 - \bar{E} = -\frac{3}{4}|E_1| = -\frac{3e^2}{8a_0}$$

We need $\langle r^4 \rangle_{100}$:
$$\langle r^4 \rangle_{100} = \frac{4}{a_0^3}\int_0^\infty r^6 e^{-2r/a_0} dr = \frac{4}{a_0^3} \cdot \frac{6!}{(2/a_0)^7} = \frac{4 \cdot 720 \cdot a_0^7}{128 a_0^3} = \frac{45a_0^4}{2}$$

$$\langle \hat{V}^2 \rangle = \lambda^2\frac{e^4}{a_0^6} \cdot \frac{45a_0^4}{2} = \frac{45\lambda^2 e^4}{2a_0^2}$$

$$\langle \hat{V} \rangle^2 = 9\lambda^2\frac{e^4}{a_0^2}$$

$$E_1^{(2)} \approx \frac{1}{-3e^2/(8a_0)}\left[\frac{45\lambda^2 e^4}{2a_0^2} - \frac{9\lambda^2 e^4}{a_0^2}\right] = \frac{-8a_0}{3e^2} \cdot \frac{27\lambda^2 e^4}{2a_0^2}$$

$$\boxed{E_1^{(2)} \approx -36\lambda^2\frac{e^2}{a_0}}$$

---

## Solution 6: Time-Dependent Perturbation Theory

### Part (a): Perturbation and Selection Rules (8 points)

The perturbation Hamiltonian:
$$\hat{V} = -e\vec{E} \cdot \vec{r} = -eE_0 z \cdot \Theta(t) = -eE_0 r\cos\theta \cdot \Theta(t)$$

$$\boxed{\hat{V} = -eE_0 z \cdot \Theta(t)}$$

**Selection rules** for electric dipole transitions ($\hat{z} \propto Y_1^0$):
- $\Delta \ell = \pm 1$
- $\Delta m = 0$ (for $z$-polarization)

From the ground state $|1,0,0\rangle$ ($\ell = 0, m = 0$):
- Can transition to $\ell = 1$, $m = 0$

$$\boxed{\text{Only } |2,1,0\rangle \text{ can be excited. The states } |2,0,0\rangle, |2,1,\pm 1\rangle \text{ are forbidden.}}$$

---

### Part (b): Transition Probability (10 points)

First-order time-dependent perturbation theory:
$$c_f(t) = -\frac{i}{\hbar}\int_0^t \langle f|\hat{V}(t')|i\rangle e^{i\omega_{fi}t'} dt'$$

where $\omega_{fi} = \frac{E_f - E_i}{\hbar} = \frac{E_2 - E_1}{\hbar} = \frac{3E_1}{4\hbar} < 0$ (since $E_1 < 0$).

Actually, $\omega_{21} = \frac{E_2 - E_1}{\hbar} = \frac{-E_1/4 - (-E_1)}{\hbar} = \frac{3|E_1|}{4\hbar} > 0$.

$$c_{210}(t) = -\frac{i}{\hbar}(-eE_0)\langle 2,1,0|z|1,0,0\rangle \int_0^t e^{i\omega_{21}t'} dt'$$

$$= \frac{ieE_0}{\hbar} \cdot 0.745a_0 \cdot \frac{e^{i\omega_{21}t} - 1}{i\omega_{21}}$$

$$= \frac{eE_0 \cdot 0.745a_0}{\hbar\omega_{21}}(e^{i\omega_{21}t} - 1)$$

The probability:
$$P_{1\to 210}(t) = |c_{210}(t)|^2 = \frac{e^2E_0^2(0.745a_0)^2}{\hbar^2\omega_{21}^2}|e^{i\omega_{21}t} - 1|^2$$

$$= \frac{e^2E_0^2(0.745a_0)^2}{\hbar^2\omega_{21}^2} \cdot 4\sin^2\left(\frac{\omega_{21}t}{2}\right)$$

$$\boxed{P_{1\to 210}(t) = \frac{4e^2E_0^2(0.745a_0)^2}{\hbar^2\omega_{21}^2}\sin^2\left(\frac{\omega_{21}t}{2}\right)}$$

where $\omega_{21} = \frac{3|E_1|}{4\hbar} = \frac{3e^2}{8\hbar a_0}$.

---

### Part (c): Long-Time Behavior (7 points)

The probability oscillates:
$$P(t) \propto \sin^2\left(\frac{\omega_{21}t}{2}\right)$$

This oscillates between 0 and $P_{max} = \frac{4e^2E_0^2(0.745a_0)^2}{\hbar^2\omega_{21}^2}$.

**It does NOT grow without bound.** This is characteristic of a **sudden perturbation** to a discrete final state.

**First-order perturbation theory breaks down when:**
1. $P_{max} \gtrsim 1$ (strong field regime)
2. When we need to include back-transitions from $|2,1,0\rangle$ to $|1,0,0\rangle$
3. When other states become significantly populated

$$\boxed{\text{The probability oscillates with frequency } \omega_{21}. \text{ First-order PT fails when } eE_0 a_0 \gtrsim \hbar\omega_{21}.}$$

---

## Solution 7: Identical Particles

### Part (a): Ground State of Two Fermions (8 points)

For the harmonic oscillator, single-particle energies: $E_n = \hbar\omega(n + 1/2)$.

For two fermions with spin, the Pauli exclusion principle allows both to occupy $n = 0$ if they have opposite spins.

**Ground state:**
$$\Psi_{gs}(x_1, x_2) = \psi_0(x_1)\psi_0(x_2) \cdot \chi_{singlet}$$

The spatial part is symmetric, so the spin part must be antisymmetric (singlet):
$$\chi_{singlet} = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$$

$$\boxed{\Psi_{gs} = \psi_0(x_1)\psi_0(x_2) \otimes \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)}$$

**Ground state energy:**
$$\boxed{E_{gs} = 2 \times \frac{\hbar\omega}{2} = \hbar\omega}$$

---

### Part (b): Contact Interaction Correction (9 points)

The perturbation: $\hat{V}_{int} = \lambda\delta(x_1 - x_2)$

First-order correction:
$$E^{(1)} = \langle \Psi_{gs}|\hat{V}_{int}|\Psi_{gs}\rangle = \lambda\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} |\psi_0(x_1)|^2|\psi_0(x_2)|^2\delta(x_1 - x_2)dx_1 dx_2$$

$$= \lambda\int_{-\infty}^{\infty} |\psi_0(x)|^4 dx$$

For the harmonic oscillator ground state:
$$\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}e^{-m\omega x^2/(2\hbar)}$$

$$|\psi_0(x)|^4 = \sqrt{\frac{m\omega}{\pi\hbar}}e^{-2m\omega x^2/\hbar}$$

$$\int_{-\infty}^{\infty} |\psi_0(x)|^4 dx = \sqrt{\frac{m\omega}{\pi\hbar}} \cdot \sqrt{\frac{\pi\hbar}{2m\omega}} = \sqrt{\frac{1}{2}} \cdot \frac{1}{\sqrt{\pi}}\sqrt{\frac{m\omega}{\hbar}} = \frac{1}{\sqrt{2\pi}}\sqrt{\frac{m\omega}{\hbar}}$$

$$\boxed{E^{(1)} = \frac{\lambda}{\sqrt{2\pi}}\sqrt{\frac{m\omega}{\hbar}}}$$

---

### Part (c): First Excited State (8 points)

The first excited state has one particle in $n = 0$ and one in $n = 1$.

The spatial wavefunction can be symmetric or antisymmetric:
$$\Psi_S = \frac{1}{\sqrt{2}}[\psi_0(x_1)\psi_1(x_2) + \psi_1(x_1)\psi_0(x_2)]$$
$$\Psi_A = \frac{1}{\sqrt{2}}[\psi_0(x_1)\psi_1(x_2) - \psi_1(x_1)\psi_0(x_2)]$$

For fermions:
- Symmetric spatial $\Rightarrow$ singlet spin (antisymmetric)
- Antisymmetric spatial $\Rightarrow$ triplet spin (symmetric)

**Energy (without interaction):** $E = \frac{\hbar\omega}{2} + \frac{3\hbar\omega}{2} = 2\hbar\omega$

**Degeneracy:** 4-fold ($1$ singlet + $3$ triplet states)

$$\boxed{\text{First excited state: } E = 2\hbar\omega, \text{ 4-fold degenerate}}$$

**Effect of contact interaction:**

For the triplet state ($\Psi_A$ spatial):
$$\langle \Psi_A|\delta(x_1-x_2)|\Psi_A\rangle = \frac{1}{2}\int|\psi_0(x)\psi_1(x) - \psi_1(x)\psi_0(x)|^2 dx = 0$$

The antisymmetric spatial wavefunction vanishes when $x_1 = x_2$!

For the singlet state ($\Psi_S$ spatial):
$$\langle \Psi_S|\delta(x_1-x_2)|\Psi_S\rangle = \int |\psi_0(x)|^2|\psi_1(x)|^2 dx > 0$$

$$\boxed{\text{Triplet states (antisym. spatial): unshifted. Singlet state (sym. spatial): shifted up by } \Delta E > 0}$$

---

## Solution 8: Scattering Theory

### Part (a): Low-Energy s-Wave Dominance (8 points)

For low energies, $ka \ll 1$. The angular momentum barrier is:
$$V_{eff}^{(\ell)}(r) = V(r) + \frac{\hbar^2\ell(\ell+1)}{2mr^2}$$

For $\ell > 0$, the centrifugal barrier $\propto \ell(\ell+1)/r^2$ prevents the particle from penetrating to small $r$ at low energies. Only $\ell = 0$ (s-wave) contributes significantly.

**Radial equation for $\ell = 0$:**

Let $u(r) = r R(r)$:

**Inside ($r < a$):**
$$\frac{d^2u}{dr^2} + \frac{2m}{\hbar^2}(E - V_0)u = 0$$
$$\frac{d^2u}{dr^2} - \kappa^2 u = 0 \quad \text{where } \kappa = \sqrt{\frac{2m(V_0 - E)}{\hbar^2}}$$

**Outside ($r > a$):**
$$\frac{d^2u}{dr^2} + k^2 u = 0 \quad \text{where } k = \sqrt{\frac{2mE}{\hbar^2}}$$

$$\boxed{u'' - \kappa^2 u = 0 \text{ for } r < a; \quad u'' + k^2 u = 0 \text{ for } r > a}$$

---

### Part (b): Phase Shift Calculation (10 points)

**Solutions:**
- Inside: $u(r) = A\sinh(\kappa r)$ (choosing solution regular at origin)
- Outside: $u(r) = B\sin(kr + \delta_0)$

**Boundary conditions at $r = a$:**

Continuity of $u$:
$$A\sinh(\kappa a) = B\sin(ka + \delta_0)$$

Continuity of $u'$:
$$A\kappa\cosh(\kappa a) = Bk\cos(ka + \delta_0)$$

Dividing:
$$\frac{\kappa\cosh(\kappa a)}{\sinh(\kappa a)} = \frac{k\cos(ka + \delta_0)}{\sin(ka + \delta_0)}$$

$$\kappa\coth(\kappa a) = k\cot(ka + \delta_0)$$

Solving for $\delta_0$:
$$\cot(ka + \delta_0) = \frac{\kappa}{k}\coth(\kappa a)$$

$$ka + \delta_0 = \text{arccot}\left(\frac{\kappa}{k}\coth(\kappa a)\right)$$

$$\boxed{\tan(ka + \delta_0) = \frac{k}{\kappa}\tanh(\kappa a)}$$

or equivalently:

$$\boxed{\delta_0 = \arctan\left(\frac{k}{\kappa}\tanh(\kappa a)\right) - ka}$$

---

### Part (c): Hard Sphere Limit (7 points)

As $V_0 \to \infty$: $\kappa \to \infty$, so $\tanh(\kappa a) \to 1$ and $\frac{k}{\kappa} \to 0$.

$$\tan(ka + \delta_0) \to 0 \quad \Rightarrow \quad ka + \delta_0 = n\pi$$

For the principal value with small $ka$:
$$\delta_0 = -ka$$

The total cross section for s-wave:
$$\sigma_{tot} = \frac{4\pi}{k^2}\sin^2\delta_0 = \frac{4\pi}{k^2}\sin^2(ka)$$

In the low-energy limit $ka \ll 1$:
$$\sigma_{tot} \approx \frac{4\pi}{k^2}(ka)^2 = 4\pi a^2$$

$$\boxed{\sigma_{tot} = 4\pi a^2}$$

**Comparison to classical:** The classical geometric cross section is $\sigma_{classical} = \pi a^2$.

**The factor of 4** arises from:
1. In quantum mechanics, the incoming plane wave is diffracted around the sphere
2. There's interference between the incident wave and the scattered spherical wave
3. This is related to the **optical theorem**: the forward scattering amplitude removes flux from the transmitted beam
4. The sphere's "shadow" is equal in size to its geometric cross section, and by unitarity, the total scattered flux equals the shadow

$$\boxed{\sigma_{quantum} = 4\sigma_{classical} \text{ due to diffraction and shadow scattering}}$$

---

## Summary of Solutions

| Problem | Topic | Key Results |
|---------|-------|-------------|
| 1 | Uncertainty | $\Delta A \Delta B \geq \frac{1}{2}|\langle C \rangle|$; coherent states minimize |
| 2 | Finite well | $\kappa = k\tan(ka)$; infinite limit recovers $E_n \propto n^2$ |
| 3 | Spin dynamics | Rabi oscillations at resonance: $P = \sin^2(\omega_0 t/2)$ |
| 4 | Angular momentum | $|\Psi\rangle = |J=1, M=0\rangle$; antisymmetric under exchange |
| 5 | TI perturbation | $E_1^{(1)} = 3\lambda e^2/a_0$; $\ell$ degeneracy lifted |
| 6 | TD perturbation | Selection rule $\Delta\ell = \pm 1$; oscillating probability |
| 7 | Identical particles | Singlet spin for ground state; triplet unaffected by contact |
| 8 | Scattering | $\sigma = 4\pi a^2$ for hard sphere (4x classical) |

---

*Solutions prepared based on standard graduate quantum mechanics at the PhD qualifying exam level.*
