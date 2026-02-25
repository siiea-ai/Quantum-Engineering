# Week 154: Variational and WKB Methods - Problem Solutions

## Part I: Variational Method Solutions

### Solution 1: Variational Principle Proof

**(a) Proof:**

Let $\{|n\rangle\}$ be the complete set of energy eigenstates: $H|n\rangle = E_n|n\rangle$ with $E_0 \leq E_1 \leq E_2 \leq \cdots$

Any normalized $|\psi\rangle$ can be expanded: $|\psi\rangle = \sum_n c_n|n\rangle$ with $\sum_n|c_n|^2 = 1$.

$$\langle\psi|H|\psi\rangle = \sum_n |c_n|^2 E_n$$

Since $E_n \geq E_0$ for all $n$:
$$\sum_n |c_n|^2 E_n \geq \sum_n |c_n|^2 E_0 = E_0$$

$$\boxed{\langle\psi|H|\psi\rangle \geq E_0}$$

**(b) Equality condition:**

Equality holds iff $E_n = E_0$ for all $n$ with $c_n \neq 0$.

If the ground state is non-degenerate, this means $c_n = \delta_{n0}$, so $|\psi\rangle = |0\rangle$.

$$\boxed{\text{Equality holds iff } |\psi\rangle = |0\rangle \text{ (the exact ground state)}}$$

**(c) Extension to first excited state:**

If $\langle\psi|\psi_0\rangle = c_0 = 0$, then:
$$\langle\psi|H|\psi\rangle = \sum_{n=1}^{\infty}|c_n|^2 E_n \geq E_1\sum_{n=1}^{\infty}|c_n|^2 = E_1$$

$$\boxed{\text{If } \langle\psi|\psi_0\rangle = 0, \text{ then } \langle\psi|H|\psi\rangle \geq E_1}$$

---

### Solution 2: Hydrogen Ground State

**(a) Normalization:**

$$\int_0^\infty |\psi|^2 4\pi r^2 dr = 4\pi A^2 \int_0^\infty r^2 e^{-2\alpha r}dr = 4\pi A^2 \cdot \frac{2!}{(2\alpha)^3} = \frac{\pi A^2}{\alpha^3} = 1$$

$$\boxed{A = \sqrt{\frac{\alpha^3}{\pi}}}$$

**(b) Expectation values:**

$$\langle T\rangle = -\frac{\hbar^2}{2m}\int \psi^* \nabla^2\psi \, d^3r = \frac{\hbar^2\alpha^2}{2m}$$

Using $\langle 1/r\rangle = \alpha$ for exponential wavefunction:
$$\langle V\rangle = -\frac{e^2}{4\pi\epsilon_0}\langle 1/r\rangle = -\frac{e^2\alpha}{4\pi\epsilon_0}$$

$$\boxed{\langle T\rangle = \frac{\hbar^2\alpha^2}{2m}, \quad \langle V\rangle = -\frac{e^2\alpha}{4\pi\epsilon_0}}$$

**(c) Optimization:**

$$E(\alpha) = \frac{\hbar^2\alpha^2}{2m} - \frac{e^2\alpha}{4\pi\epsilon_0}$$

$$\frac{dE}{d\alpha} = \frac{\hbar^2\alpha}{m} - \frac{e^2}{4\pi\epsilon_0} = 0$$

$$\alpha_{\text{opt}} = \frac{me^2}{4\pi\epsilon_0\hbar^2} = \frac{1}{a_0}$$

$$\boxed{E_{\text{min}} = -\frac{me^4}{32\pi^2\epsilon_0^2\hbar^2} = -\frac{e^2}{8\pi\epsilon_0 a_0} = -13.6 \text{ eV}}$$

**(d) Comparison:**

This IS the exact ground state energy! The exponential trial function is the exact ground state wavefunction.

---

### Solution 3: Gaussian Trial for Hydrogen

**(a) Energy calculation:**

For $\psi = A e^{-\alpha r^2}$: normalization gives $A = (2\alpha/\pi)^{3/4}$

$$\langle T\rangle = \frac{3\hbar^2\alpha}{2m}$$

$$\langle V\rangle = -\frac{e^2}{4\pi\epsilon_0}\sqrt{\frac{2\alpha}{\pi}} \cdot 2 = -\frac{e^2}{4\pi\epsilon_0}\sqrt{\frac{8\alpha}{\pi}}$$

$$E(\alpha) = \frac{3\hbar^2\alpha}{2m} - \frac{e^2}{4\pi\epsilon_0}\sqrt{\frac{8\alpha}{\pi}}$$

**(b) Optimization:**

$$\frac{dE}{d\alpha} = \frac{3\hbar^2}{2m} - \frac{e^2}{4\pi\epsilon_0}\sqrt{\frac{2}{\pi\alpha}} = 0$$

Solving: $\alpha_{\text{opt}} = \frac{8}{9\pi}\frac{m^2e^4}{(4\pi\epsilon_0)^2\hbar^4}$

$$\boxed{E_{\text{min}} = -\frac{4}{3\pi}\frac{me^4}{(4\pi\epsilon_0)^2\hbar^2} = -\frac{8}{3\pi} \times 13.6 \text{ eV} \approx -11.5 \text{ eV}}$$

**(c) Why worse:**

The Gaussian misses the **cusp** at $r = 0$. The exact hydrogen wavefunction has:
$$\left.\frac{d\psi}{dr}\right|_{r=0} = -\frac{\psi(0)}{a_0} \neq 0$$

A Gaussian has zero derivative at the origin, failing to capture this Coulomb cusp.

---

### Solution 7: Helium Variational Calculation

**(a) Kinetic energy:**

Each electron in a hydrogen-like orbital with $Z_{\text{eff}}$ has $\langle T\rangle = Z_{\text{eff}}^2 \times 13.6$ eV.

$$\boxed{\langle T\rangle = 2 Z_{\text{eff}}^2 \times 13.6 \text{ eV}}$$

**(b) Electron-nucleus potential:**

$$\langle V_{en}\rangle = -2 \times \frac{Z \cdot Z_{\text{eff}}}{a_0}\frac{e^2}{4\pi\epsilon_0} = -4 Z_{\text{eff}} \times 13.6 \text{ eV}$$

(With $Z = 2$ for helium)

$$\boxed{\langle V_{en}\rangle = -4 Z_{\text{eff}} \times 13.6 \text{ eV} = -8 Z_{\text{eff}} \times 13.6 \text{ eV}}$$

Wait, let me redo this. $\langle V_{en}\rangle$ for one electron with effective charge $Z_{\text{eff}}$ seeing true nuclear charge $Z$:

$$\langle V_{en}\rangle = -\frac{Ze^2}{4\pi\epsilon_0}\langle 1/r\rangle = -\frac{Ze^2}{4\pi\epsilon_0} \cdot \frac{Z_{\text{eff}}}{a_0}$$

For two electrons: $\langle V_{en}\rangle = -2 \cdot Z \cdot Z_{\text{eff}} \times 27.2$ eV

$$\boxed{\langle V_{en}\rangle = -4 Z_{\text{eff}} \times 27.2 \text{ eV} = -108.8 Z_{\text{eff}}/Z_{\text{eff}} \times Z_{\text{eff}} = -2ZZ_{\text{eff}} \times 27.2 \text{ eV}}$$

**(c) Optimization:**

$$E(Z_{\text{eff}}) = Z_{\text{eff}}^2(27.2) - 4Z_{\text{eff}}(27.2) + \frac{5Z_{\text{eff}}}{8}(27.2) \text{ eV}$$

$$= 27.2\left(Z_{\text{eff}}^2 - 4Z_{\text{eff}} + \frac{5Z_{\text{eff}}}{8}\right) = 27.2\left(Z_{\text{eff}}^2 - \frac{27Z_{\text{eff}}}{8}\right) \text{ eV}$$

$$\frac{dE}{dZ_{\text{eff}}} = 27.2\left(2Z_{\text{eff}} - \frac{27}{8}\right) = 0$$

$$\boxed{Z_{\text{eff}} = \frac{27}{16} \approx 1.69}$$

**(d) Ground state energy:**

$$E_0 = 27.2\left(\frac{27^2}{256} - \frac{27 \times 27}{8 \times 16}\right) = 27.2 \times \frac{27^2}{256}\left(1 - 2\right) = -27.2 \times \frac{729}{256}$$

$$\boxed{E_0 = -77.5 \text{ eV}}$$

Experimental: $-78.98$ eV. Error: $\approx 2\%$.

---

## Part II: WKB Solutions

### Solution 11: Harmonic Oscillator WKB

**(a) Turning points:**

$E = \frac{1}{2}m\omega^2 x^2$ gives $x_{\pm} = \pm\sqrt{2E/(m\omega^2)}$

**(b) Phase integral:**

$$\oint p\,dx = 2\int_{-x_0}^{x_0}\sqrt{2m\left(E - \frac{1}{2}m\omega^2 x^2\right)}dx$$

Let $x = x_0\sin\theta$:
$$= 2\sqrt{2mE}\int_{-\pi/2}^{\pi/2}x_0\cos^2\theta\,d\theta = 2\sqrt{2mE} \cdot x_0 \cdot \frac{\pi}{2}$$

$$= \pi x_0\sqrt{2mE} = \pi\sqrt{\frac{2E}{m\omega^2}}\sqrt{2mE} = \frac{2\pi E}{\omega}$$

**(c) Quantization:**

$$\oint p\,dx = \left(n + \frac{1}{2}\right)h$$

$$\frac{2\pi E}{\omega} = \left(n + \frac{1}{2}\right)h$$

$$\boxed{E_n = \left(n + \frac{1}{2}\right)\hbar\omega}$$

This is exact!

---

### Solution 15: Square Barrier Tunneling

**(a) WKB transmission:**

For $V = V_0$ in $0 < x < a$, $E < V_0$:

$$\kappa = \sqrt{2m(V_0 - E)}/\hbar$$

$$T_{\text{WKB}} = \exp\left(-2\int_0^a \kappa\,dx\right) = e^{-2\kappa a}$$

$$\boxed{T_{\text{WKB}} = e^{-2\kappa a} = \exp\left(-\frac{2a}{\hbar}\sqrt{2m(V_0-E)}\right)}$$

**(b) Comparison with exact:**

Exact result for thick barrier:
$$T_{\text{exact}} \approx 16\frac{E(V_0-E)}{V_0^2}e^{-2\kappa a}$$

For $E = V_0/2$: prefactor $= 16 \times \frac{1}{2} \times \frac{1}{2} = 4$

For $\kappa a = 2$: $e^{-4} \approx 0.018$

$$T_{\text{WKB}} \approx 0.018, \quad T_{\text{exact}} \approx 4 \times 0.018 = 0.072$$

WKB underestimates by factor of 4 due to neglected prefactors.

---

### Solution 19: Alpha Decay

**(a) Turning point:**

$$E = \frac{2Ze^2}{4\pi\epsilon_0 r_0}$$

$$r_0 = \frac{2Ze^2}{4\pi\epsilon_0 E} = \frac{2 \times 90 \times (1.44 \text{ MeV}\cdot\text{fm})}{5 \text{ MeV}} = \frac{259.2}{5} \text{ fm} \approx 52 \text{ fm}$$

$$\boxed{r_0 \approx 52 \text{ fm}}$$

**(b) Gamow factor:**

$$\gamma = \frac{1}{\hbar}\int_R^{r_0}\sqrt{2m_\alpha\left(\frac{2Ze^2}{4\pi\epsilon_0 r} - E\right)}dr$$

Using the standard integral:
$$\gamma = \frac{2Ze^2}{\hbar v}\left[\cos^{-1}\sqrt{R/r_0} - \sqrt{(R/r_0)(1-R/r_0)}\right]$$

where $v = \sqrt{2E/m_\alpha}$.

For $R = 7$ fm, $r_0 = 52$ fm:
$$R/r_0 \approx 0.135, \quad \sqrt{R/r_0} \approx 0.37$$
$$\cos^{-1}(0.37) \approx 1.19 \text{ rad}, \quad \sqrt{0.135 \times 0.865} \approx 0.34$$

Numerical evaluation gives $\gamma \approx 35$.

**(c) Decay rate:**

$$T = e^{-2\gamma} \approx e^{-70} \approx 10^{-30}$$

Decay rate: $\Gamma = \nu T = 10^{21} \times 10^{-30} = 10^{-9}$ s$^{-1}$

Half-life: $t_{1/2} = \frac{\ln 2}{\Gamma} \approx \frac{0.7}{10^{-9}} \approx 7 \times 10^8$ s $\approx 22$ years

$$\boxed{t_{1/2} \sim 10^8 \text{ s} \sim 20 \text{ years}}$$

(Actual values vary enormously with energy - Geiger-Nuttall law!)

---

### Solution 22: Double Well Splitting

**(a) Energy splitting derivation:**

In WKB, each well supports states with:
$$\int_a^b p\,dx = \left(n + \frac{1}{2}\right)\pi\hbar$$

Tunneling connects the wells. The symmetric/antisymmetric combinations split by:
$$\Delta E = E_+ - E_- \approx \frac{\hbar\omega}{\pi}\exp\left(-\frac{1}{\hbar}\int_{x_1}^{x_2}\kappa\,dx\right)$$

$$\boxed{\Delta E \approx \frac{\hbar\omega}{\pi}e^{-\gamma}}$$

where $\gamma = \int \kappa dx/\hbar$ is the Gamow factor through the central barrier.

**(b) Tunneling time interpretation:**

The tunneling frequency is $\omega_{\text{tunnel}} = \Delta E/\hbar$.

Time to tunnel from one well to the other:
$$\tau = \frac{\pi}{\omega_{\text{tunnel}}} = \frac{\pi\hbar}{\Delta E} = \frac{\pi^2}{\omega}e^{\gamma}$$

**(c) Ammonia estimate:**

Inversion frequency: $\nu = 24$ GHz $= \Delta E/h$

$$\Delta E = h\nu = 6.6 \times 10^{-34} \times 24 \times 10^9 = 1.6 \times 10^{-23} \text{ J} \approx 10^{-4} \text{ eV}$$

Assuming $\hbar\omega \sim 0.1$ eV (N-H bending mode):
$$e^{-\gamma} = \frac{\pi\Delta E}{\hbar\omega} \approx \frac{\pi \times 10^{-4}}{0.1} \approx 3 \times 10^{-3}$$

$$\gamma \approx 6$$

The barrier height $V_0$ enters through $\gamma \approx \frac{2d}{\hbar}\sqrt{2mV_0}$ where $d$ is barrier width.

With $d \sim 0.4$ Angstrom, $m \sim m_{\text{N}}$:
$$V_0 \sim 0.25 \text{ eV}$$

$$\boxed{V_0 \sim 0.2-0.3 \text{ eV}}$$

---

### Solution 24: Radial WKB and Hydrogen

**(a) Radial quantization condition:**

$$\int_{r_1}^{r_2}\sqrt{2m\left(E - V(r) - \frac{\hbar^2(\ell+1/2)^2}{2mr^2}\right)}dr = \left(n_r + \frac{1}{2}\right)\pi\hbar$$

Note: The Langer correction replaces $\ell(\ell+1) \to (\ell+1/2)^2$.

**(b) Hydrogen application:**

For hydrogen: $V(r) = -e^2/(4\pi\epsilon_0 r)$

The integral can be evaluated using contour methods:
$$n_r + \ell + 1 = n = \frac{Z}{\sqrt{-2E/E_R}}$$

where $E_R = 13.6$ eV.

$$\boxed{E_n = -\frac{Z^2 E_R}{n^2} = -\frac{13.6 Z^2}{n^2} \text{ eV}}$$

**(c) Langer correction:**

The naive $\ell(\ell+1)$ gives wrong results for s-waves ($\ell=0$).

The correction $\ell(\ell+1) \to (\ell+1/2)^2$ accounts for the fact that the radial WKB wavefunction must behave correctly as $r \to 0$.

This can be derived rigorously from the 3D nature of the problem.

---

## Summary of Key Results

### Variational Method
- Always gives upper bound: $E_{\text{var}} \geq E_0$
- Hydrogen with exponential: exact result
- Hydrogen with Gaussian: ~85% of correct answer (missing cusp)
- Helium: $Z_{\text{eff}} = 27/16 \approx 1.69$, error ~2%

### WKB
- Harmonic oscillator: exact
- Bohr-Sommerfeld: $\oint p\,dx = (n+1/2)h$
- Tunneling: $T \approx e^{-2\gamma}$

### Alpha Decay
- Gamow factor determines lifetime
- Explains Geiger-Nuttall law
- Lifetimes span 20+ orders of magnitude

---

**Practice Tip:** For qualifying exams, memorize the key integrals (Gaussian, exponential) and standard results (helium $Z_{\text{eff}}$, Bohr-Sommerfeld). These appear repeatedly.
