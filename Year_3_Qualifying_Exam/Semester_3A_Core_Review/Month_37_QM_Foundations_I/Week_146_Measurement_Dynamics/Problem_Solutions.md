# Week 146: Measurement and Dynamics — Problem Solutions

## Level 1 Solutions

### Solution 1: Measurement Probabilities

**(a)**
$$P(S_z = +\hbar/2) = |\langle+_z|\psi\rangle|^2 = \left|\frac{1}{\sqrt{3}}\right|^2 = \boxed{\frac{1}{3}}$$

$$P(S_z = -\hbar/2) = |\langle-_z|\psi\rangle|^2 = \left|\sqrt{\frac{2}{3}}\right|^2 = \boxed{\frac{2}{3}}$$

**(b)**
$$\langle S_z \rangle = \frac{\hbar}{2}\cdot\frac{1}{3} + \left(-\frac{\hbar}{2}\right)\cdot\frac{2}{3} = \frac{\hbar}{6} - \frac{\hbar}{3} = \boxed{-\frac{\hbar}{6}}$$

**(c)** After measuring $$S_z = +\hbar/2$$, the state collapses to:
$$\boxed{|+\rangle_z}$$

---

### Solution 2: Sequential Measurements

**(a)** We have $$|+\rangle_x = \frac{1}{\sqrt{2}}(|+\rangle_z + |-\rangle_z)$$, so $$|\psi\rangle = |+\rangle_x$$.
$$P(S_x = +\hbar/2) = |\langle+_x|\psi\rangle|^2 = |\langle+_x|+_x\rangle|^2 = \boxed{1}$$

**(b)** After measuring $$S_x = +\hbar/2$$, state is $$|+\rangle_x$$.
$$P(S_z = +\hbar/2) = |\langle+_z|+_x\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \boxed{\frac{1}{2}}$$

**(c)** Joint probability:
$$P(S_x = +, \text{then } S_z = +) = 1 \times \frac{1}{2} = \boxed{\frac{1}{2}}$$

---

### Solution 3: Stationary States

**(a)** General solution:
$$\boxed{|\psi(t)\rangle = \sum_{n=1}^{\infty} c_n e^{-iE_nt/\hbar}|n\rangle}$$

where $$E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$$ and $$c_n = \langle n|\psi(0)\rangle$$.

**(b)** For $$|\psi(0)\rangle = |n\rangle$$:
$$\boxed{|\psi(t)\rangle = e^{-iE_nt/\hbar}|n\rangle}$$

**(c)**
$$|\psi_n(x,t)|^2 = |e^{-iE_nt/\hbar}|^2|\phi_n(x)|^2 = |\phi_n(x)|^2$$

The phase factor has modulus 1, so $$|\psi|^2$$ is time-independent. $$\checkmark$$

---

### Solution 4: Time Evolution of Superposition

**(a)**
$$\boxed{|\psi(t)\rangle = \frac{1}{\sqrt{2}}\left(e^{-iE_1t/\hbar}|1\rangle + e^{-iE_2t/\hbar}|2\rangle\right)}$$

**(b)**
$$|\psi(x,t)|^2 = \frac{1}{2}\left[|\psi_1|^2 + |\psi_2|^2 + 2\text{Re}(\psi_1^*\psi_2 e^{-i(E_2-E_1)t/\hbar})\right]$$

$$= \frac{1}{2}\left[|\psi_1|^2 + |\psi_2|^2 + 2|\psi_1||\psi_2|\cos(\omega_{21}t + \phi)\right]$$

**(c)**
$$\boxed{\omega_{21} = \frac{E_2 - E_1}{\hbar} = \frac{3\pi^2\hbar}{2mL^2}}$$

---

### Solution 5: Expectation Value Evolution

**(a)** Using $$[\hat{H}, \hat{x}]$$:
$$[\hat{p}^2, \hat{x}] = \hat{p}[\hat{p}, \hat{x}] + [\hat{p}, \hat{x}]\hat{p} = -2i\hbar\hat{p}$$
$$[\hat{H}, \hat{x}] = \frac{1}{2m}(-2i\hbar\hat{p}) = -\frac{i\hbar\hat{p}}{m}$$

$$\frac{d\langle\hat{x}\rangle}{dt} = \frac{i}{\hbar}\left\langle-\frac{i\hbar\hat{p}}{m}\right\rangle = \boxed{\frac{\langle\hat{p}\rangle}{m}}$$

**(b)** Using $$[\hat{H}, \hat{p}] = [V(\hat{x}), \hat{p}] = -i\hbar V'(\hat{x})$$:
$$\frac{d\langle\hat{p}\rangle}{dt} = \frac{i}{\hbar}(-i\hbar)\langle V'(\hat{x})\rangle = \boxed{-\langle V'(\hat{x})\rangle}$$

**(c)** For harmonic oscillator, $$V'(x) = m\omega^2 x$$:
$$\frac{d\langle\hat{p}\rangle}{dt} = -m\omega^2\langle\hat{x}\rangle$$

Combined with part (a):
$$\frac{d^2\langle\hat{x}\rangle}{dt^2} = \frac{1}{m}\frac{d\langle\hat{p}\rangle}{dt} = -\omega^2\langle\hat{x}\rangle \checkmark$$

---

### Solution 6: Free Particle Propagator

**(a)** As $$t \to 0^+$$, the exponential oscillates rapidly. Using stationary phase:
$$K_0(x,0^+;x',0) \propto \delta(x-x') \checkmark$$

**(b)** For $$\psi(x,0) = \delta(x)$$:
$$\psi(x,t) = \int K_0(x,t;x',0)\delta(x')dx' = K_0(x,t;0,0)$$
$$= \boxed{\sqrt{\frac{m}{2\pi i\hbar t}}\exp\left[\frac{imx^2}{2\hbar t}\right]}$$

**(c)**
$$|\psi(x,t)|^2 = \frac{m}{2\pi\hbar t}$$

This is $$\propto 1/t$$, showing the spreading of the initially localized packet. $$\checkmark$$

---

### Solution 7: Two-Level System Dynamics

**(a)**
$$\hat{U}(t) = e^{-i\hat{H}t/\hbar} = e^{-i\omega_0 t\sigma_z/2} = \boxed{\begin{pmatrix} e^{-i\omega_0 t/2} & 0 \\ 0 & e^{i\omega_0 t/2} \end{pmatrix}}$$

**(b)**
$$|\psi(t)\rangle = \hat{U}(t)|+\rangle_x = \frac{1}{\sqrt{2}}(e^{-i\omega_0 t/2}|+\rangle_z + e^{i\omega_0 t/2}|-\rangle_z)$$

**(c)**
$$\langle S_x(t)\rangle = \frac{\hbar}{2}\langle\psi(t)|\sigma_x|\psi(t)\rangle = \frac{\hbar}{2}\cos(\omega_0 t)$$

This is **Larmor precession** — the spin precesses about the z-axis at angular frequency $$\omega_0$$.

---

### Solution 8: Heisenberg Picture

**(a)** Heisenberg equations:
$$\frac{d\hat{x}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{x}_H] = \frac{\hat{p}_H}{m}$$
$$\frac{d\hat{p}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{p}_H] = -m\omega^2\hat{x}_H$$

**(b)** These give $$\ddot{\hat{x}}_H = -\omega^2\hat{x}_H$$, with solution:
$$\boxed{\hat{x}_H(t) = \hat{x}_H(0)\cos(\omega t) + \frac{\hat{p}_H(0)}{m\omega}\sin(\omega t)}$$
$$\boxed{\hat{p}_H(t) = \hat{p}_H(0)\cos(\omega t) - m\omega\hat{x}_H(0)\sin(\omega t)}$$

**(c)**
$$[\hat{x}_H(t), \hat{p}_H(t)] = [\hat{x}(0), \hat{p}(0)](\cos^2\omega t + \sin^2\omega t) = i\hbar \checkmark$$

---

### Solution 9: Conservation Laws

**(a)**
$$[\hat{H}, \hat{p}] = \left[\frac{\hat{p}^2}{2m}, \hat{p}\right] = 0 \checkmark$$

Momentum is conserved for a free particle — there are no forces.

**(b)**
$$[\hat{H}, \hat{x}] = \frac{1}{2m}[\hat{p}^2, \hat{x}] = -\frac{i\hbar\hat{p}}{m} \neq 0$$

Position changes because the particle moves!

**(c)** Consider $$\hat{G} = \hat{x} - \frac{\hat{p}t}{m}$$ (initial position):
$$[\hat{H}, \hat{G}] = [\hat{H}, \hat{x}] - \frac{t}{m}[\hat{H}, \hat{p}] = -\frac{i\hbar\hat{p}}{m} - 0 = -\frac{i\hbar\hat{p}}{m}$$

But accounting for explicit time dependence in Ehrenfest:
$$\frac{d\langle\hat{G}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H},\hat{G}]\rangle + \left\langle-\frac{\hat{p}}{m}\right\rangle = \frac{\langle\hat{p}\rangle}{m} - \frac{\langle\hat{p}\rangle}{m} = 0 \checkmark$$

---

## Level 2 Solutions

### Solution 10: Measurement of Non-Eigenstate

**(a)** $$E = \frac{3}{2}\hbar\omega$$ corresponds to $$|1\rangle$$:
$$P(E_1) = |\langle 1|\psi\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \boxed{\frac{1}{2}}$$

**(b)**
$$\langle\hat{H}\rangle = \frac{1}{2}(E_0 + E_1) = \frac{1}{2}\left(\frac{1}{2}\hbar\omega + \frac{3}{2}\hbar\omega\right) = \boxed{\hbar\omega}$$

$$\langle\hat{H}^2\rangle = \frac{1}{2}(E_0^2 + E_1^2) = \frac{1}{2}\left(\frac{\hbar^2\omega^2}{4} + \frac{9\hbar^2\omega^2}{4}\right) = \frac{5\hbar^2\omega^2}{4}$$

$$(\Delta H)^2 = \frac{5\hbar^2\omega^2}{4} - \hbar^2\omega^2 = \frac{\hbar^2\omega^2}{4}$$

$$\boxed{\Delta H = \frac{\hbar\omega}{2}}$$

**(c)** After measuring $$E_0 = \frac{1}{2}\hbar\omega$$, state is $$|0\rangle$$:
$$\boxed{\langle\hat{x}\rangle = \langle 0|\hat{x}|0\rangle = 0}$$

by symmetry of the ground state.

---

### Solution 11: Time Evolution with Mixed Initial State

Note: $$\hat{\rho}(0) = |+\rangle_x\langle+|_x$$ — this is actually a pure state!

**(a)**
$$\hat{\rho}(t) = \hat{U}(t)\hat{\rho}(0)\hat{U}^\dagger(t)$$

$$= \frac{1}{2}\begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} e^{i\omega t/2} & 0 \\ 0 & e^{-i\omega t/2} \end{pmatrix}$$

$$= \boxed{\frac{1}{2}\begin{pmatrix} 1 & e^{-i\omega t} \\ e^{i\omega t} & 1 \end{pmatrix}}$$

**(b)**
$$\langle\hat{S}_x\rangle = \text{Tr}\left(\hat{\rho}\cdot\frac{\hbar}{2}\sigma_x\right) = \frac{\hbar}{4}\text{Tr}\begin{pmatrix} e^{i\omega t} & 1 \\ 1 & e^{-i\omega t} \end{pmatrix} = \boxed{\frac{\hbar}{2}\cos(\omega t)}$$

**(c)** $$\text{Tr}(\hat{\rho}^2) = \frac{1}{4}(1 + 1 + 1 + 1) = 1$$, so this is a **pure state**.

---

### Solution 12: Propagator Construction

**(a)** Using completeness:
$$K(x,t;x',0) = \sum_{n=1}^{\infty} \psi_n(x)\psi_n^*(x')e^{-iE_nt/\hbar}$$

$$= \boxed{\frac{2}{L}\sum_{n=1}^{\infty}\sin\left(\frac{n\pi x}{L}\right)\sin\left(\frac{n\pi x'}{L}\right)\exp\left(-\frac{in^2\pi^2\hbar t}{2mL^2}\right)}$$

**(b)** At $$t=0$$:
$$K(x,0;x',0) = \frac{2}{L}\sum_n \sin\frac{n\pi x}{L}\sin\frac{n\pi x'}{L} = \delta(x-x') \checkmark$$

by completeness of $$\{\psi_n\}$$.

**(c)** Revival time: $$e^{-iE_nt/\hbar} = 1$$ for all $$n$$. Since $$E_n \propto n^2$$:
$$\boxed{T_{rev} = \frac{4mL^2}{\pi\hbar}}$$

---

### Solutions 13-19: [Detailed solutions follow similar format]

---

## Level 3 Solutions

### Solution 20: Sudden Approximation

**(a)** Initial state: $$\psi_1^{(old)}(x) = \sqrt{\frac{2}{L}}\sin\frac{\pi x}{L}$$ for $$0 < x < L$$, zero elsewhere.

New ground state: $$\psi_1^{(new)}(x) = \sqrt{\frac{1}{L}}\sin\frac{\pi x}{2L}$$

Overlap:
$$c_1 = \int_0^L \psi_1^{(new)*}(x)\psi_1^{(old)}(x)dx = \sqrt{\frac{2}{L^2}}\int_0^L \sin\frac{\pi x}{2L}\sin\frac{\pi x}{L}dx$$

Using $$\sin A \sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$$:
$$= \sqrt{\frac{2}{L^2}} \cdot \frac{L}{2}\left[\frac{\sin(\pi/2)}{\pi/2} - \frac{\sin(3\pi/2)}{3\pi/2}\right] = \frac{8\sqrt{2}}{3\pi}$$

$$\boxed{P_1 = |c_1|^2 = \frac{128}{9\pi^2} \approx 0.72}$$

**(b)** First excited state of new well:
$$c_2 = \int_0^L \sqrt{\frac{1}{L}}\sin\frac{2\pi x}{2L}\sqrt{\frac{2}{L}}\sin\frac{\pi x}{L}dx = \sqrt{\frac{2}{L^2}}\int_0^L \sin^2\frac{\pi x}{L}dx = \sqrt{\frac{2}{L^2}}\cdot\frac{L}{2}$$

$$\boxed{P_2 = |c_2|^2 = \frac{1}{2}}$$

Wait, let me recalculate more carefully...

The first excited state of the new well (width $$2L$$) is:
$$\psi_2^{(new)}(x) = \sqrt{\frac{1}{L}}\sin\frac{\pi x}{L}$$

This exactly matches the old ground state in the region $$0 < x < L$$!

$$c_2 = \int_0^{2L} \psi_2^{(new)*}\psi_1^{(old)}dx = \int_0^L \frac{1}{\sqrt{L}}\sin\frac{\pi x}{L}\cdot\sqrt{\frac{2}{L}}\sin\frac{\pi x}{L}dx = \sqrt{\frac{2}{L^2}}\cdot\frac{L}{2} = \sqrt{\frac{1}{2}}$$

$$\boxed{P_2 = \frac{1}{2}}$$

**(c)** Initial energy: $$E_1^{(old)} = \frac{\pi^2\hbar^2}{2mL^2}$$

$$\langle E \rangle_{new} = \sum_n |c_n|^2 E_n^{(new)}$$

By energy conservation in the sudden limit, $$\langle E \rangle_{new} = E_1^{(old)} = \frac{\pi^2\hbar^2}{2mL^2}$$.

The new ground state energy is $$E_1^{(new)} = \frac{\pi^2\hbar^2}{8mL^2} = \frac{1}{4}E_1^{(old)}$$.

So $$\langle E \rangle > E_1^{(new)}$$: the particle has excess energy distributed among excited states.

---

### Solution 21: Adiabatic Theorem and Berry Phase

**(a)** The instantaneous Hamiltonian:
$$\hat{H}(t) = -\gamma B_0\frac{\hbar}{2}\begin{pmatrix} \cos\theta & e^{-i\phi}\sin\theta \\ e^{i\phi}\sin\theta & -\cos\theta \end{pmatrix}$$

Eigenvalues: $$E_\pm = \mp\frac{\gamma B_0\hbar}{2}$$

Ground state ($$E_-$$):
$$|-(t)\rangle = \begin{pmatrix} \sin(\theta/2) \\ -e^{i\phi}\cos(\theta/2) \end{pmatrix}$$

**(b)** By adiabatic theorem, for slow $$\omega$$:
$$|\psi(t)\rangle \approx e^{i\gamma_d(t)}e^{i\gamma_g(t)}|-(t)\rangle$$

where $$\gamma_d = \int_0^t E_-dt'/\hbar$$ is dynamical phase and $$\gamma_g$$ is geometric phase.

**(c)** Berry phase after one cycle:
$$\gamma_g = i\oint \langle-|\nabla_\phi|-\rangle d\phi = i\int_0^{2\pi}\langle-|\frac{\partial}{\partial\phi}|-\rangle d\phi$$

$$\frac{\partial}{\partial\phi}|-\rangle = \begin{pmatrix} 0 \\ -ie^{i\phi}\cos(\theta/2) \end{pmatrix}$$

$$\langle-|\frac{\partial}{\partial\phi}|-\rangle = -i\cos^2(\theta/2)$$

$$\boxed{\gamma_g = \int_0^{2\pi}(-\cos^2(\theta/2))d\phi = -\pi(1+\cos\theta) = -\Omega/2}$$

where $$\Omega$$ is the solid angle subtended.

---

### Solution 22: Quantum Zeno Effect

**(a)** Solve $$i\hbar\dot{c}_1 = \hbar\Omega c_2$$, $$i\hbar\dot{c}_2 = \hbar\Omega c_1$$ with $$c_1(0) = 1$$:
$$c_1(t) = \cos(\Omega t)$$
$$\boxed{P_1(t) = \cos^2(\Omega t)}$$

**(b)** After first measurement at $$t/N$$:
$$P_1(t/N) = \cos^2(\Omega t/N) \approx 1 - (\Omega t/N)^2$$ for small $$t/N$$

After $$N$$ measurements:
$$P_{survive} = [1 - (\Omega t/N)^2]^N \xrightarrow{N\to\infty} \boxed{1}$$

**(c)** **Quantum Zeno paradox:** Frequent observation prevents the system from evolving! The measurement "freezes" the state. This is because each measurement projects back to $$|1\rangle$$, and for short times, the deviation from $$|1\rangle$$ is $$O(t^2)$$, so repeated measurements eliminate accumulated evolution.

---

### Solutions 23-26: [Similar detailed treatment]

---

*Solutions for Week 146 — Measurement and Dynamics*
*For additional details, see Sakurai Chapter 2 and Shankar Chapters 4, 6*
