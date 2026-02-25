# QM Integration Exam - Solutions

## Problem 1: Quantum Postulates and Measurement

### Solution

**(a) Measurement outcomes and probabilities:**

Possible results: $S_z = +\hbar/2$ or $S_z = -\hbar/2$

Probabilities:
$$P(+\hbar/2) = |\langle+|\psi\rangle|^2 = \left|\frac{1}{\sqrt{3}}\right|^2 = \boxed{\frac{1}{3}}$$

$$P(-\hbar/2) = |\langle-|\psi\rangle|^2 = \left|\sqrt{\frac{2}{3}}\right|^2 = \boxed{\frac{2}{3}}$$

**(b) Expectation values of $S_x$:**

Using $S_x = \frac{\hbar}{2}\begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}$ in the $|+\rangle, |-\rangle$ basis:

$$|\psi\rangle = \begin{pmatrix}1/\sqrt{3}\\\sqrt{2/3}\end{pmatrix}$$

$$\langle S_x\rangle = \langle\psi|S_x|\psi\rangle = \frac{\hbar}{2}\begin{pmatrix}1/\sqrt{3} & \sqrt{2/3}\end{pmatrix}\begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}\begin{pmatrix}1/\sqrt{3}\\\sqrt{2/3}\end{pmatrix}$$

$$= \frac{\hbar}{2} \cdot 2 \cdot \frac{1}{\sqrt{3}} \cdot \sqrt{\frac{2}{3}} = \boxed{\frac{\hbar\sqrt{2}}{3}}$$

For $\langle S_x^2\rangle$: Note that $S_x^2 = \frac{\hbar^2}{4}\mathbf{1}$, so:
$$\boxed{\langle S_x^2\rangle = \frac{\hbar^2}{4}}$$

**(c) Sequential measurements:**

After measuring $S_z = -\hbar/2$, state collapses to $|-\rangle$.

Express $|-\rangle$ in $S_x$ basis: $|+\rangle_x = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$, $|-\rangle_x = \frac{1}{\sqrt{2}}(|+\rangle - |-\rangle)$

$$|-\rangle = \frac{1}{\sqrt{2}}(|+\rangle_x - |-\rangle_x)$$

$$\boxed{P(S_x = +\hbar/2) = \frac{1}{2}, \quad P(S_x = -\hbar/2) = \frac{1}{2}}$$

**(d) Uncertainty relation:**

$$\langle S_z\rangle = \frac{\hbar}{2}\left(\frac{1}{3} - \frac{2}{3}\right) = -\frac{\hbar}{6}$$

$$\langle S_z^2\rangle = \frac{\hbar^2}{4}$$

$$(\Delta S_z)^2 = \langle S_z^2\rangle - \langle S_z\rangle^2 = \frac{\hbar^2}{4} - \frac{\hbar^2}{36} = \frac{8\hbar^2}{36} = \frac{2\hbar^2}{9}$$

$$\Delta S_z = \frac{\hbar\sqrt{2}}{3}$$

Similarly: $(\Delta S_x)^2 = \frac{\hbar^2}{4} - \frac{2\hbar^2}{9} = \frac{\hbar^2}{36}$, so $\Delta S_x = \frac{\hbar}{6}$

For $\langle S_y\rangle$: By symmetry and explicit calculation, $\langle S_y\rangle = 0$.

Check: $\Delta S_x \Delta S_z = \frac{\hbar}{6} \cdot \frac{\hbar\sqrt{2}}{3} = \frac{\hbar^2\sqrt{2}}{18}$

RHS: $\frac{\hbar}{2}|\langle S_y\rangle| = 0$

$$\boxed{\frac{\hbar^2\sqrt{2}}{18} \geq 0 \checkmark}$$

---

## Problem 2: One-Dimensional Systems

### Solution

**(a) Time evolution:**

$$|\psi(t)\rangle = \frac{1}{\sqrt{2}}(e^{-iE_0t/\hbar}|0\rangle + e^{-iE_1t/\hbar}|1\rangle)$$

With $E_n = (n+1/2)\hbar\omega$:

$$\boxed{|\psi(t)\rangle = \frac{e^{-i\omega t/2}}{\sqrt{2}}(|0\rangle + e^{-i\omega t}|1\rangle)}$$

**(b) $\langle x\rangle(t)$:**

Using $x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^\dagger)$:

$$\langle x\rangle = \sqrt{\frac{\hbar}{2m\omega}}\langle\psi|(a + a^\dagger)|\psi\rangle$$

Non-zero terms: $\langle 0|a|1\rangle = 1$ and $\langle 1|a^\dagger|0\rangle = 1$

$$\langle x\rangle = \sqrt{\frac{\hbar}{2m\omega}} \cdot \frac{1}{2}(e^{i\omega t} + e^{-i\omega t}) = \boxed{\sqrt{\frac{\hbar}{2m\omega}}\cos(\omega t)}$$

**(c) $\langle x^2\rangle(t)$:**

$x^2 = \frac{\hbar}{2m\omega}(a^2 + a^{\dagger 2} + aa^\dagger + a^\dagger a)$

$$\langle x^2\rangle = \frac{\hbar}{2m\omega}[\langle a^2\rangle + \langle a^{\dagger 2}\rangle + \langle aa^\dagger\rangle + \langle a^\dagger a\rangle]$$

Computing each term:
- $\langle 0|a^2|0\rangle = 0$, $\langle 1|a^2|1\rangle = 0$, cross terms vanish
- Similarly for $a^{\dagger 2}$
- $\langle aa^\dagger\rangle = \frac{1}{2}(1 + 2) = 3/2$
- $\langle a^\dagger a\rangle = \frac{1}{2}(0 + 1) = 1/2$

$$\boxed{\langle x^2\rangle = \frac{\hbar}{2m\omega} \cdot 2 = \frac{\hbar}{m\omega}}$$

This is time-independent! The width does NOT oscillate (it's a "displaced" state, not squeezed).

**(d) Sudden change:**

The particle is in $|0\rangle_{\text{old}}$ (ground state of $\omega$).

This must be expanded in eigenstates of the new oscillator ($2\omega$):

$$|0\rangle_{\text{old}} = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}e^{-m\omega x^2/(2\hbar)}$$

$$|0\rangle_{\text{new}} = \left(\frac{2m\omega}{\pi\hbar}\right)^{1/4}e^{-m\omega x^2/\hbar}$$

$$P = |\langle 0_{\text{new}}|0_{\text{old}}\rangle|^2 = \left|\int\psi_0^{\text{new}*}\psi_0^{\text{old}}dx\right|^2$$

Using Gaussian integral:
$$P = \frac{2^{1/2}\cdot 1}{(1 + 2)^{1/2}/\sqrt{2}} = \boxed{\frac{2\sqrt{2}}{3} \approx 0.943}$$

---

## Problem 3: Angular Momentum and Spin

### Solution

**(a) Commutators:**

$\mathbf{S}_1 \cdot \mathbf{S}_2 = \frac{1}{2}(\mathbf{S}^2 - \mathbf{S}_1^2 - \mathbf{S}_2^2)$

Since $\mathbf{S}^2$ commutes with itself and with $S_z$: $[H, \mathbf{S}^2] = 0$ $\checkmark$

$S_z = S_{1z} + S_{2z}$ commutes with $\mathbf{S}_1\cdot\mathbf{S}_2$ (scalar) and with itself.
$$\boxed{[H, S_z] = 0}$$

**(b) Eigenvalues:**

Using $\mathbf{S}_1\cdot\mathbf{S}_2 = \frac{1}{2}[S(S+1) - \frac{3}{4} - \frac{3}{4}]\hbar^2$:

For $S = 1$ (triplet): $\mathbf{S}_1\cdot\mathbf{S}_2 = \frac{\hbar^2}{4}$
For $S = 0$ (singlet): $\mathbf{S}_1\cdot\mathbf{S}_2 = -\frac{3\hbar^2}{4}$

Eigenvalues:
$$E_{1,m} = \frac{A\hbar^2}{4} + Bm\hbar \quad (m = -1, 0, 1)$$
$$\boxed{E_{0,0} = -\frac{3A\hbar^2}{4}}$$

**(c) $B = 0$:**

Triplet: $E_T = A\hbar^2/4$ (3-fold degenerate)
Singlet: $E_S = -3A\hbar^2/4$

For $A > 0$: singlet is lower (antiferromagnetic)
For $A < 0$: triplet is lower (ferromagnetic)

**(d) $B \neq 0$:**

The triplet splits: $E_{1,m} = \frac{A\hbar^2}{4} + Bm\hbar$

$$E_{1,1} = \frac{A\hbar^2}{4} + B\hbar$$
$$E_{1,0} = \frac{A\hbar^2}{4}$$
$$E_{1,-1} = \frac{A\hbar^2}{4} - B\hbar$$

Linear Zeeman splitting of triplet; singlet unaffected.

---

## Problem 4: Perturbation Theory (Stark Effect)

### Solution

**(a) Perturbation:**
$$\boxed{H' = e\mathcal{E}z = e\mathcal{E}r\cos\theta}$$

**(b) Ground state:**

First-order: $E_1^{(1)} = \langle 1,0,0|H'|1,0,0\rangle$

By parity: $z$ is odd, $|1,0,0|^2$ is even, so the integral vanishes.

$$\boxed{E_1^{(1)} = 0}$$

Physical reason: The ground state has no permanent electric dipole moment.

**(c) n=2 degeneracy:**

The $n=2$ level is 4-fold degenerate. We need the matrix:
$$H'_{ij} = \langle i|e\mathcal{E}z|j\rangle$$

in the basis $\{|2,0,0\rangle, |2,1,0\rangle, |2,1,1\rangle, |2,1,-1\rangle\}$.

**(d) Non-zero elements:**

Selection rules: $\Delta\ell = \pm 1$, $\Delta m = 0$ for $z$.

Non-zero: $\langle 2,0,0|z|2,1,0\rangle$ and its conjugate.

The matrix is:
$$H' = e\mathcal{E}\begin{pmatrix}
0 & \alpha & 0 & 0\\
\alpha & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0
\end{pmatrix}$$

where $\alpha = \langle 2,0,0|z|2,1,0\rangle = 3a_0$.

$|2,1,\pm 1\rangle$ states don't mix (different $m$) and have zero first-order shift.

The $2s$-$2p_0$ mixing gives $E^{(1)} = \pm 3ea_0\mathcal{E}$.

$$\boxed{\text{Linear Stark effect with splitting } \pm 3ea_0\mathcal{E}}$$

---

## Problem 5: Identical Particles

### Solution

**(a) Ground state wavefunction:**

$$\Psi = \phi_{1s}(\mathbf{r}_1)\phi_{1s}(\mathbf{r}_2) \cdot \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$$

Spin must be singlet because:
- Both electrons in same spatial orbital (symmetric)
- Total wavefunction must be antisymmetric
- Therefore spin part must be antisymmetric = singlet

**(b) Variational calculation:**

Energy: $E = \langle T\rangle + \langle V_{en}\rangle + \langle V_{ee}\rangle$

$$\langle T\rangle = 2 \cdot Z_{\text{eff}}^2 \cdot 13.6 \text{ eV}$$
$$\langle V_{en}\rangle = -2 \cdot 2 \cdot Z_{\text{eff}} \cdot 27.2 \text{ eV} = -4Z_{\text{eff}} \cdot 27.2 \text{ eV}$$
$$\langle V_{ee}\rangle = \frac{5Z_{\text{eff}}}{8} \cdot 27.2 \text{ eV}$$

$$E = 27.2\left(Z_{\text{eff}}^2 - 4Z_{\text{eff}} + \frac{5Z_{\text{eff}}}{8}\right) = 27.2\left(Z_{\text{eff}}^2 - \frac{27Z_{\text{eff}}}{8}\right)$$

$$\frac{dE}{dZ_{\text{eff}}} = 27.2\left(2Z_{\text{eff}} - \frac{27}{8}\right) = 0$$

$$\boxed{Z_{\text{eff}} = \frac{27}{16} = Z - \frac{5}{16}}$$

**(c) Triplet lower than singlet:**

Triplet has antisymmetric spatial wavefunction:
$$\psi_T \propto \phi_{1s}(1)\phi_{2s}(2) - \phi_{1s}(2)\phi_{2s}(1)$$

This vanishes when $\mathbf{r}_1 = \mathbf{r}_2$, so electrons avoid each other.

This reduces electron-electron repulsion (exchange hole), lowering energy.

**(d) Second-quantized Hamiltonian:**

$$\boxed{H = \sum_\sigma \epsilon_1 c_{1\sigma}^\dagger c_{1\sigma} + \sum_\sigma \epsilon_2 c_{2\sigma}^\dagger c_{2\sigma} + U\sum_{i}n_{i\uparrow}n_{i\downarrow} + V\sum_{\sigma,\sigma'}n_{1\sigma}n_{2\sigma'}}$$

---

## Problem 6: WKB and Tunneling

### Solution

**(a) Turning points:**

$V(x_t) = E$: $V_0(1 - x_t^2/a^2) = E$

$$x_t^2 = a^2\left(1 - \frac{E}{V_0}\right)$$

$$\boxed{x_t = \pm a\sqrt{1 - E/V_0}}$$

**(b) WKB transmission:**

$$T = \exp\left(-\frac{2}{\hbar}\int_{-x_t}^{x_t}\sqrt{2m(V-E)}dx\right)$$

Let $u = x/a$, $u_t = \sqrt{1-E/V_0}$:

$$T = \exp\left(-\frac{2a}{\hbar}\sqrt{2mV_0}\int_{-u_t}^{u_t}\sqrt{1 - u^2 - E/V_0}du\right)$$

Using $\int_{-u_t}^{u_t}\sqrt{u_t^2 - u^2}du = \frac{\pi u_t^2}{2}$:

$$\boxed{T = \exp\left(-\frac{\pi a}{\hbar}\sqrt{2m(V_0-E)}\left(1 - \frac{E}{V_0}\right)\right)}$$

**(c) Thin barrier limit:**

For small $a$: $T \approx \exp(-\pi a\sqrt{2m(V_0-E)}/\hbar)$

Compare to rectangular: $T = \exp(-2a\sqrt{2m(V_0-E)}/\hbar)$

The parabolic barrier has a smaller exponent (easier tunneling) due to its shape.

**(d) Alpha decay estimate:**

$V_0 - E = 25$ MeV, $a = 10$ fm, $m = 4 \times 931$ MeV/c$^2$

$$\kappa = \frac{\sqrt{2 \times 3724 \times 25}}{197 \text{ MeV}\cdot\text{fm}} \approx 2.2 \text{ fm}^{-1}$$

$$T \approx e^{-\pi \times 10 \times 2.2 \times (1 - 5/30)} \approx e^{-58} \approx \boxed{10^{-25}}$$

---

## Problem 7: Scattering Theory

### Solution

**(a) Radial equations:**

Inside ($r < a$): $u'' + K^2 u = 0$ where $K = \sqrt{2m(E+V_0)}/\hbar$

Outside ($r > a$): $u'' + k^2 u = 0$ where $k = \sqrt{2mE}/\hbar$

Boundary conditions:
- $u(0) = 0$ (regularity)
- $u$ and $u'$ continuous at $r = a$

**(b) Phase shift:**

Inside: $u = A\sin(Kr)$
Outside: $u = B\sin(kr + \delta_0)$

Matching: $\frac{u'}{u}|_{r=a^-} = \frac{u'}{u}|_{r=a^+}$

$$K\cot(Ka) = k\cot(ka + \delta_0)$$

$$\boxed{\tan\delta_0 = \frac{k\tan(Ka) - K\tan(ka)}{K + k\tan(Ka)\tan(ka)}}$$

**(c) Zero-energy resonance:**

At $k \to 0$: $\delta_0 \to \pi/2$ when $\cot(Ka)|_{E=0} = 0$

$$Ka = \frac{\pi}{2}, \frac{3\pi}{2}, ...$$

$$\boxed{\sqrt{\frac{2mV_0}{\hbar^2}}a = \left(n + \frac{1}{2}\right)\pi}$$

Physical significance: A new bound state is about to appear (or just appeared).

**(d) Born approximation:**

$$f_{\text{Born}} = -\frac{m}{2\pi\hbar^2}\int_0^a (-V_0)4\pi r^2 \frac{\sin(qr)}{qr}dr = \frac{2mV_0}{\hbar^2 q^3}[\sin(qa) - qa\cos(qa)]$$

For weak scattering, the exact result should match Born. This occurs when:
- $mV_0 a^2/\hbar^2 \ll 1$ (weak potential)
- $ka \gg 1$ (high energy)

---

## Problem 8: Time-Dependent Perturbation

### Solution

**(a) Perturbation:**

$$H'(t) = -\mathbf{d}\cdot\mathbf{E} = -e\mathcal{E}_0 z\cos(\omega t)$$

$$\boxed{H'(t) = -e\mathcal{E}_0 z\cos(\omega t)}$$

**(b) Transition probability:**

First-order: $c_{2p}^{(1)}(t) = -\frac{i}{\hbar}\int_0^t \langle 2p|H'(t')|1s\rangle e^{i\omega_{21}t'}dt'$

where $\omega_{21} = (E_2 - E_1)/\hbar$.

$$= \frac{ie\mathcal{E}_0 z_{21}}{2\hbar}\left[\frac{e^{i(\omega_{21}+\omega)t}-1}{\omega_{21}+\omega} + \frac{e^{i(\omega_{21}-\omega)t}-1}{\omega_{21}-\omega}\right]$$

Near resonance ($\omega \approx \omega_{21}$), second term dominates:

$$P_{1s\to 2p}(t) = |c_{2p}|^2 \approx \frac{e^2\mathcal{E}_0^2|z_{21}|^2}{\hbar^2}\frac{\sin^2[(\omega_{21}-\omega)t/2]}{(\omega_{21}-\omega)^2}$$

**(c) Resonance:**

At $\omega = \omega_{21}$:
$$\boxed{P \approx \frac{e^2\mathcal{E}_0^2|z_{21}|^2 t^2}{4\hbar^2}}$$

(grows as $t^2$ until perturbation theory breaks down)

**(d) Selection rules:**

Electric dipole: $\Delta\ell = \pm 1$, $\Delta m = 0, \pm 1$

From $1s$ ($\ell=0$): only $\ell = 1$ (p-states) accessible.

For $z$-polarization: $\Delta m = 0$

$$\boxed{\text{Only } |2,1,0\rangle \text{ accessible from } |1,0,0\rangle}$$

---

## Grading Rubric Summary

Each problem: 25 points total
- Setup and method: 7-8 points
- Calculation: 10-12 points
- Final answer: 3-5 points
- Physical insight: 3-5 points

Passing: 70+ points (6 problems)
