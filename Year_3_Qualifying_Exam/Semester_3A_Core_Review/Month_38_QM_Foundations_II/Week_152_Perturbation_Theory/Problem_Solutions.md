# Week 152: Perturbation Theory - Problem Solutions

## Part A: Non-Degenerate Perturbation Theory

### Solution 1

$H_0 = \hbar\omega(a^{\dagger}a + 1/2)$, $H' = \lambda x^2 = \frac{\lambda\hbar}{2m\omega}(a + a^{\dagger})^2$

**(a)** For ground state:
$$E_0^{(1)} = \langle 0|H'|0\rangle = \frac{\lambda\hbar}{2m\omega}\langle 0|(a + a^{\dagger})^2|0\rangle$$

$(a + a^{\dagger})^2 = a^2 + a^{\dagger}a + aa^{\dagger} + (a^{\dagger})^2$

$\langle 0|a^2|0\rangle = \langle 0|(a^{\dagger})^2|0\rangle = 0$
$\langle 0|a^{\dagger}a|0\rangle = 0$
$\langle 0|aa^{\dagger}|0\rangle = 1$

$$E_0^{(1)} = \frac{\lambda\hbar}{2m\omega}$$

**(b)** For state $|n\rangle$:
$\langle n|(a + a^{\dagger})^2|n\rangle = \langle n|a^{\dagger}a + aa^{\dagger}|n\rangle = n + (n+1) = 2n + 1$

$$E_n^{(1)} = \frac{\lambda\hbar}{2m\omega}(2n + 1)$$

**(c)** The exact Hamiltonian $H_0 + H' = \frac{p^2}{2m} + \frac{1}{2}(m\omega^2 + 2\lambda)x^2$ is a harmonic oscillator with $\omega' = \sqrt{\omega^2 + 2\lambda/m}$.

Exact energies: $E_n = \hbar\omega'(n + 1/2)$

For small $\lambda$: $\omega' \approx \omega(1 + \lambda/(m\omega^2)) = \omega + \lambda/(m\omega)$

$E_n \approx \hbar\omega(n + 1/2) + \frac{\lambda\hbar}{m\omega}(n + 1/2) = E_n^{(0)} + E_n^{(1)}$ ✓

---

### Solution 3

**(a)** $H' = \lambda x^4 = \lambda\left(\frac{\hbar}{2m\omega}\right)^2(a + a^{\dagger})^4$

Using $(a + a^{\dagger})^4$ and keeping terms with equal powers of $a$ and $a^{\dagger}$:

$\langle n|(a + a^{\dagger})^4|n\rangle = 6n^2 + 6n + 3$

(This comes from terms like $a^{\dagger}a^{\dagger}aa$, $a^{\dagger}aa^{\dagger}a$, etc.)

$$E_n^{(1)} = \frac{3\lambda\hbar^2}{4m^2\omega^2}(2n^2 + 2n + 1)$$

**(b)** For the ground state, $E_0^{(2)} < 0$ because all intermediate states have $E_k > E_0$, making every term in the sum negative:

$$E_0^{(2)} = \sum_{k > 0}\frac{|\langle k|H'|0\rangle|^2}{E_0 - E_k} < 0$$

The ground state energy is always lowered by second-order effects.

---

### Solution 5

For hydrogen ground state in electric field $H' = eE_0 z$:

**(a)** $E^{(1)} = \langle 1s|eE_0 z|1s\rangle = 0$ because the ground state has even parity ($l=0$) and $z$ has odd parity.

**(b)** $E^{(2)} = \sum_{n,l,m}\frac{|\langle nlm|eE_0 z|1s\rangle|^2}{E_1 - E_n}$

Selection rule: $z$ only connects to $l=1$, $m=0$ states.

Dominant contribution from $n=2$, $l=1$, $m=0$:

$\langle 2,1,0|z|1,0,0\rangle$ is of order $a_0$.

$$E^{(2)} \sim -\frac{(eE_0 a_0)^2}{E_1 - E_2} \sim -\frac{(eE_0 a_0)^2}{13.6\text{ eV} \times (3/4)}$$

**(c)** It's quadratic because the first-order term vanishes by parity. The polarizability is $\alpha = -2E^{(2)}/(E_0^2)$.

---

## Part B: Degenerate Perturbation Theory

### Solution 9

Diagonalize $W = \begin{pmatrix} a & b \\ b^* & c \end{pmatrix}$

Eigenvalues:
$$E^{(1)} = \frac{a + c}{2} \pm \sqrt{\left(\frac{a-c}{2}\right)^2 + |b|^2}$$

Eigenvectors: For $E_+$, $|+\rangle = \cos\frac{\theta}{2}|1\rangle + e^{i\phi}\sin\frac{\theta}{2}|2\rangle$

where $\tan\theta = \frac{2|b|}{a-c}$ and $\phi = \arg(b)$.

---

### Solution 10

2D harmonic oscillator, $E_1 = 2\hbar\omega$ (degenerate), states $|1,0\rangle$ and $|0,1\rangle$.

$H' = \lambda xy = \frac{\lambda\hbar}{2m\omega}(a_x + a_x^{\dagger})(a_y + a_y^{\dagger})$

**(a)** Matrix elements:

$\langle 1,0|xy|1,0\rangle = \langle 1,0|xy|0,1\rangle = ?$

Using $x = \sqrt{\frac{\hbar}{2m\omega}}(a_x + a_x^{\dagger})$ and similarly for $y$:

$\langle 1,0|xy|0,1\rangle = \frac{\hbar}{2m\omega}\langle 1,0|(a_x + a_x^{\dagger})(a_y + a_y^{\dagger})|0,1\rangle$

$= \frac{\hbar}{2m\omega}\langle 1,0|a_x a_y^{\dagger}|0,1\rangle = \frac{\hbar}{2m\omega}$

$\langle 1,0|xy|1,0\rangle = 0$ (need $a_y^{\dagger}a_y$ term which gives 0)

$$W = \frac{\lambda\hbar}{2m\omega}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**(b)** Eigenvalues: $E^{(1)} = \pm\frac{\lambda\hbar}{2m\omega}$

**(c)** Good states: $|+\rangle = \frac{1}{\sqrt{2}}(|1,0\rangle + |0,1\rangle)$, $|-\rangle = \frac{1}{\sqrt{2}}(|1,0\rangle - |0,1\rangle)$

These are the symmetric and antisymmetric combinations.

---

### Solution 11

$n=2$ hydrogen with $H' = eE_0 z$:

**(a)** Selection rules: $z$ has $\Delta m = 0$, $\Delta l = \pm 1$. Only connects $l=0$ to $l=1$ with $m=0$.

States: $|2,0,0\rangle$ (2s), $|2,1,0\rangle$ (2p, m=0), $|2,1,\pm 1\rangle$ (2p, m=±1)

Only $\langle 2,0,0|z|2,1,0\rangle \neq 0$.

**(b)** $\langle 2,0,0|z|2,1,0\rangle = \int R_{20}R_{21}r^3 dr \int Y_0^0 \cos\theta Y_1^0 d\Omega$

Angular integral: $\sqrt{\frac{1}{4\pi}}\sqrt{\frac{3}{4\pi}}\int \cos^2\theta \sin\theta d\theta d\phi = \frac{\sqrt{3}}{4\pi}\cdot\frac{2}{3}\cdot 2\pi = \frac{1}{\sqrt{3}}$

Radial integral: $\int_0^{\infty} R_{20}R_{21}r^3 dr = -3\sqrt{3}a_0$ (after calculation)

Result: $\langle 2,0,0|z|2,1,0\rangle = -3a_0$

**(c)** The $4 \times 4$ matrix (in basis $|2,0,0\rangle$, $|2,1,0\rangle$, $|2,1,1\rangle$, $|2,1,-1\rangle$):

$$W = -3eE_0 a_0\begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

Eigenvalues: $E^{(1)} = \pm 3eE_0 a_0, 0, 0$

Good states: $|\pm\rangle = \frac{1}{\sqrt{2}}(|2,0,0\rangle \pm |2,1,0\rangle)$ for $E^{(1)} = \pm 3eE_0 a_0$

---

### Solution 14

Two spin-1/2 particles, ground spatial state, $H' = J\mathbf{S}_1 \cdot \mathbf{S}_2$

Using $\mathbf{S}_1 \cdot \mathbf{S}_2 = \frac{1}{2}(S^2 - S_1^2 - S_2^2)$ where $S = S_1 + S_2$:

For $S = 1$ (triplet): $\langle\mathbf{S}_1 \cdot \mathbf{S}_2\rangle = \frac{1}{2}(2 - 3/4 - 3/4)\hbar^2 = \frac{\hbar^2}{4}$

For $S = 0$ (singlet): $\langle\mathbf{S}_1 \cdot \mathbf{S}_2\rangle = \frac{1}{2}(0 - 3/4 - 3/4)\hbar^2 = -\frac{3\hbar^2}{4}$

**(a)** Energy shifts:
- Triplet ($m = 1, 0, -1$): $E^{(1)} = \frac{J\hbar^2}{4}$ (3-fold degenerate)
- Singlet: $E^{(1)} = -\frac{3J\hbar^2}{4}$ (non-degenerate)

**(b)** Degeneracies: 3 (triplet), 1 (singlet)

---

## Part C: Time-Dependent Perturbation

### Solution 15

$H'(t) = F_0 x$ for $t > 0$, constant.

$\langle 1|H'|0\rangle = F_0\langle 1|x|0\rangle = F_0\sqrt{\frac{\hbar}{2m\omega}}$

**(a)**
$$c_1^{(1)}(t) = -\frac{i}{\hbar}\int_0^t F_0\sqrt{\frac{\hbar}{2m\omega}}e^{i\omega t'}dt' = -\frac{iF_0}{\hbar}\sqrt{\frac{\hbar}{2m\omega}}\frac{e^{i\omega t} - 1}{i\omega}$$

$$= \frac{F_0}{\hbar\omega}\sqrt{\frac{\hbar}{2m\omega}}(e^{i\omega t} - 1)$$

**(b)**
$$P_{0\to 1}(t) = |c_1|^2 = \frac{F_0^2}{m\omega^3\hbar}|e^{i\omega t} - 1|^2 = \frac{F_0^2}{m\omega^3\hbar} \cdot 2(1 - \cos\omega t)$$

$$= \frac{4F_0^2}{m\omega^3\hbar}\sin^2\left(\frac{\omega t}{2}\right)$$

---

### Solution 19

$H'(t) = F_0 x\cos(\omega t)$ for oscillator in state $|n\rangle$.

**(a)** The operator $x = \sqrt{\frac{\hbar}{2m\omega_0}}(a + a^{\dagger})$ connects $|n\rangle$ only to $|n-1\rangle$ and $|n+1\rangle$.

**(b)** For $|n\rangle \to |n+1\rangle$:

$$c_{n+1}^{(1)}(t) = -\frac{i}{\hbar}\cdot\frac{F_0}{2}\sqrt{\frac{\hbar(n+1)}{2m\omega_0}}\left[\frac{e^{i(\omega_{n+1,n}+\omega)t}-1}{\omega_{n+1,n}+\omega} + \frac{e^{i(\omega_{n+1,n}-\omega)t}-1}{\omega_{n+1,n}-\omega}\right]$$

where $\omega_{n+1,n} = \omega_0$.

Near resonance $\omega \approx \omega_0$:

$$P_{n\to n+1}(t) \approx \frac{F_0^2(n+1)}{8m\omega_0\hbar}\frac{\sin^2[(\omega_0-\omega)t/2]}{[(\omega_0-\omega)/2]^2}$$

**(c)** Maximum at $\omega = \omega_0$ (resonance with oscillator frequency).

---

## Part D: Fermi's Golden Rule

### Solution 22

For hydrogen 2p → 1s decay:

**(a)** Fermi's golden rule: $\Gamma = \frac{2\pi}{\hbar}|M|^2\rho(E)$

For photon emission:
- $|M|^2 \propto |\langle 1s|e\mathbf{r}|2p\rangle|^2$
- $\rho(E) = \frac{\omega^2}{\pi^2\hbar c^3}$ (photon density of states)

Combined: $\Gamma = \frac{\omega^3}{3\pi\epsilon_0\hbar c^3}|\langle g|\mathbf{d}|e\rangle|^2$

**(b)** For 2p → 1s:
- $\hbar\omega = 10.2$ eV, $\omega = 1.55 \times 10^{16}$ rad/s
- $|\langle 1s|e\mathbf{r}|2p\rangle| \approx 0.74 ea_0$

$$\Gamma \approx 6.3 \times 10^8 \text{ s}^{-1}$$

$$\tau = 1/\Gamma \approx 1.6 \text{ ns}$$

---

## Part E: Adiabatic and Berry Phase

### Solution 27

Magnetic field traces a cone: $\mathbf{B} = B_0(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$

**(a)** Solid angle enclosed by a cone of half-angle $\theta$:
$$\Omega = 2\pi(1 - \cos\theta)$$

**(b)** Berry phase for spin-up:
$$\gamma = -\frac{\Omega}{2} = -\pi(1 - \cos\theta)$$

---

### Solution 28

Magnetic field rotates from $+\hat{z}$ to $-\hat{z}$ along a great circle (e.g., through $\hat{x}$).

**(a)** Dynamical phase: $\theta = -\frac{1}{\hbar}\int_0^T E_+(t)dt = -\frac{1}{\hbar}\int_0^T \frac{\gamma\hbar B_0}{2}dt = -\frac{\gamma B_0 T}{2}$

**(b)** The great circle encloses half the sphere: $\Omega = 2\pi$

Berry phase: $\gamma = -\frac{2\pi}{2} = -\pi$

**(c)** Starting in $|\uparrow_z\rangle$, the spin follows the field direction (adiabatic). At the end, the field points in $-\hat{z}$, so the eigenstate is $|\downarrow_z\rangle$.

The Berry phase of $-\pi$ means $|\psi\rangle = e^{i\pi}|\downarrow\rangle = -|\downarrow\rangle$

Final state: $|\downarrow_z\rangle$ (up to phase)

---

### Solution 30

**(a)** Adiabatic condition: $\left|\frac{\langle m|\dot{H}|n\rangle}{\hbar(E_n - E_m)^2}\right| \ll 1$

For a gap $\Delta = |E_n - E_m|$, the time scale $T$ must satisfy $T \gg \hbar/\Delta^2 \times |\dot{H}|$.

**(b)** For $H = \mathbf{R}\cdot\boldsymbol{\sigma}$:

Eigenvalues: $E_{\pm} = \pm|\mathbf{R}|$

For the lower state $|n_-(\mathbf{R})\rangle$, the Berry phase around a loop enclosing the origin:

$$\gamma = -\frac{\Omega}{2}$$

where $\Omega$ is the solid angle subtended by the loop as seen from the origin.

**(c)** At $\mathbf{R} = 0$: The gap closes ($E_+ = E_- = 0$). Adiabatic evolution fails here because the adiabatic condition requires a finite gap. The origin is a "diabolic point" or "conical intersection."

**(d)** For spin-1/2 in $\mathbf{B}$: $H = -\gamma_e\mathbf{S}\cdot\mathbf{B} = -\frac{\gamma_e\hbar}{2}\mathbf{B}\cdot\boldsymbol{\sigma}$

This has the same form with $\mathbf{R} = -\frac{\gamma_e\hbar}{2}\mathbf{B}$. Berry phase = $-\Omega/2$, confirmed.

---

**End of Solutions**
