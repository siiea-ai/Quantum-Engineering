# Week 150: Spin and Magnetic Interactions - Problem Solutions

## Part A: Pauli Matrices and Spin Operators

### Solution 1

**(a)** Direct calculation:
$$\sigma_x^2 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I \quad \checkmark$$

$$\sigma_y^2 = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I \quad \checkmark$$

$$\sigma_z^2 = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I \quad \checkmark$$

**(b)** $\text{Tr}(\sigma_x) = 0 + 0 = 0$, $\text{Tr}(\sigma_y) = 0 + 0 = 0$, $\text{Tr}(\sigma_z) = 1 + (-1) = 0$ $\checkmark$

**(c)** $\det(\sigma_x) = 0 - 1 = -1$, $\det(\sigma_y) = 0 - i(-i) = -1$, $\det(\sigma_z) = -1 - 0 = -1$ $\checkmark$

---

### Solution 2

**(a)**
$$[\sigma_x, \sigma_y] = \sigma_x\sigma_y - \sigma_y\sigma_x$$

$$\sigma_x\sigma_y = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = \begin{pmatrix} i & 0 \\ 0 & -i \end{pmatrix} = i\sigma_z$$

$$\sigma_y\sigma_x = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} -i & 0 \\ 0 & i \end{pmatrix} = -i\sigma_z$$

$$[\sigma_x, \sigma_y] = i\sigma_z - (-i\sigma_z) = 2i\sigma_z \quad \checkmark$$

Parts (b) and (c) follow by cyclic permutation.

---

### Solution 4

Any $2\times 2$ matrix can be written as $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$.

We want $A = a_0 I + a_1\sigma_x + a_2\sigma_y + a_3\sigma_z$:

$$= a_0\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} + a_1\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} + a_2\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} + a_3\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

$$= \begin{pmatrix} a_0 + a_3 & a_1 - ia_2 \\ a_1 + ia_2 & a_0 - a_3 \end{pmatrix}$$

Matching entries:
- $a = a_0 + a_3$
- $d = a_0 - a_3$
- $b = a_1 - ia_2$
- $c = a_1 + ia_2$

Solving:
$$a_0 = \frac{a+d}{2} = \frac{1}{2}\text{Tr}(A)$$
$$a_3 = \frac{a-d}{2} = \frac{1}{2}\text{Tr}(A\sigma_z)$$
$$a_1 = \frac{b+c}{2} = \frac{1}{2}\text{Tr}(A\sigma_x)$$
$$a_2 = \frac{c-b}{2i} = \frac{1}{2}\text{Tr}(A\sigma_y)$$

General formula: $a_i = \frac{1}{2}\text{Tr}(A\sigma_i)$ for $i = 1,2,3$.

---

### Solution 7

We need to prove $e^{i\theta(\hat{n}\cdot\boldsymbol{\sigma})} = \cos\theta\, I + i\sin\theta\,(\hat{n}\cdot\boldsymbol{\sigma})$.

Let $N = \hat{n}\cdot\boldsymbol{\sigma}$. Since $|\hat{n}| = 1$ and using the identity from Problem 5:

$$N^2 = (\hat{n}\cdot\boldsymbol{\sigma})^2 = (\hat{n}\cdot\hat{n})I = I$$

Therefore $N^{2k} = I$ and $N^{2k+1} = N$.

Expanding the exponential:
$$e^{i\theta N} = \sum_{k=0}^{\infty}\frac{(i\theta)^k}{k!}N^k$$

$$= \sum_{k=0}^{\infty}\frac{(i\theta)^{2k}}{(2k)!}N^{2k} + \sum_{k=0}^{\infty}\frac{(i\theta)^{2k+1}}{(2k+1)!}N^{2k+1}$$

$$= \sum_{k=0}^{\infty}\frac{(-1)^k\theta^{2k}}{(2k)!}I + i\sum_{k=0}^{\infty}\frac{(-1)^k\theta^{2k+1}}{(2k+1)!}N$$

$$= \cos\theta\, I + i\sin\theta\, N \quad \checkmark$$

---

## Part B: Spin States and Measurements

### Solution 9

$|\psi\rangle = \frac{1}{\sqrt{3}}|\uparrow\rangle + \sqrt{\frac{2}{3}}|\downarrow\rangle$

**(a)**
$$P(S_z = +\hbar/2) = \left|\frac{1}{\sqrt{3}}\right|^2 = \frac{1}{3}$$
$$P(S_z = -\hbar/2) = \left|\sqrt{\frac{2}{3}}\right|^2 = \frac{2}{3}$$

**(b)**
$$\langle S_z\rangle = \frac{1}{3}\cdot\frac{\hbar}{2} + \frac{2}{3}\cdot\left(-\frac{\hbar}{2}\right) = \frac{\hbar}{6} - \frac{\hbar}{3} = -\frac{\hbar}{6}$$

$$\langle S_z^2\rangle = \frac{1}{3}\cdot\frac{\hbar^2}{4} + \frac{2}{3}\cdot\frac{\hbar^2}{4} = \frac{\hbar^2}{4}$$

**(c)**
$$\Delta S_z = \sqrt{\langle S_z^2\rangle - \langle S_z\rangle^2} = \sqrt{\frac{\hbar^2}{4} - \frac{\hbar^2}{36}} = \sqrt{\frac{8\hbar^2}{36}} = \frac{2\hbar\sqrt{2}}{6} = \frac{\hbar\sqrt{2}}{3}$$

---

### Solution 10

**(a)** The eigenstates of $S_x$ are:
$$|+x\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle + |\downarrow\rangle)$$

$$\langle +x|\psi\rangle = \frac{1}{\sqrt{2}}\left(\frac{1}{\sqrt{3}} + \sqrt{\frac{2}{3}}\right) = \frac{1}{\sqrt{6}}(1 + \sqrt{2})$$

$$P(S_x = +\hbar/2) = \frac{1}{6}(1 + \sqrt{2})^2 = \frac{1}{6}(3 + 2\sqrt{2}) = \frac{1}{2} + \frac{\sqrt{2}}{3}$$

**(b)** Using the matrix form:
$$\langle S_x\rangle = \langle\psi|\frac{\hbar}{2}\sigma_x|\psi\rangle$$

$$= \frac{\hbar}{2}\begin{pmatrix} \frac{1}{\sqrt{3}} & \sqrt{\frac{2}{3}} \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} \frac{1}{\sqrt{3}} \\ \sqrt{\frac{2}{3}} \end{pmatrix}$$

$$= \frac{\hbar}{2}\begin{pmatrix} \frac{1}{\sqrt{3}} & \sqrt{\frac{2}{3}} \end{pmatrix}\begin{pmatrix} \sqrt{\frac{2}{3}} \\ \frac{1}{\sqrt{3}} \end{pmatrix} = \frac{\hbar}{2}\cdot\frac{2\sqrt{2}}{3} = \frac{\hbar\sqrt{2}}{3}$$

---

### Solution 11

$|\psi\rangle = \cos\frac{\pi}{8}|\uparrow\rangle + e^{i\pi/4}\sin\frac{\pi}{8}|\downarrow\rangle$

**(a)** Comparing with the general form $|\psi\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + e^{i\phi}\sin\frac{\theta}{2}|\downarrow\rangle$:
$$\frac{\theta}{2} = \frac{\pi}{8} \Rightarrow \theta = \frac{\pi}{4}$$
$$\phi = \frac{\pi}{4}$$

**(b)**
$$\langle S_x\rangle = \frac{\hbar}{2}\sin\theta\cos\phi = \frac{\hbar}{2}\sin\frac{\pi}{4}\cos\frac{\pi}{4} = \frac{\hbar}{2}\cdot\frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} = \frac{\hbar}{4}$$

$$\langle S_y\rangle = \frac{\hbar}{2}\sin\theta\sin\phi = \frac{\hbar}{2}\cdot\frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} = \frac{\hbar}{4}$$

$$\langle S_z\rangle = \frac{\hbar}{2}\cos\theta = \frac{\hbar}{2}\cos\frac{\pi}{4} = \frac{\hbar}{2\sqrt{2}}$$

**(c)** The spin points in direction $(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) = (1/2, 1/2, 1/\sqrt{2})$.

This is 45° from the z-axis, in the direction bisecting the x and y axes in the xy-plane.

---

### Solution 13

Initial state: $|\uparrow_z\rangle$

**Part 1:** Measuring $S_x$

$$|\uparrow_z\rangle = \frac{1}{\sqrt{2}}|+x\rangle + \frac{1}{\sqrt{2}}|-x\rangle$$

Outcomes: $S_x = +\hbar/2$ with probability 1/2, $S_x = -\hbar/2$ with probability 1/2.

**Part 2:** After measuring $S_x = +\hbar/2$, state is $|+x\rangle = \frac{1}{\sqrt{2}}(|\uparrow_z\rangle + |\downarrow_z\rangle)$

Probability of $S_z = +\hbar/2$: $|\langle\uparrow_z|+x\rangle|^2 = 1/2$

---

### Solution 15

Initial state: $|+z\rangle$
Measurement direction: $\hat{n} = (\sin\theta, 0, \cos\theta)$

**(a)** The eigenstate of $S_n$ with eigenvalue $+\hbar/2$ is:
$$|+n\rangle = \cos\frac{\theta}{2}|\uparrow\rangle + \sin\frac{\theta}{2}|\downarrow\rangle$$

Probability of $+\hbar/2$:
$$P_+ = |\langle +n|+z\rangle|^2 = \cos^2\frac{\theta}{2}$$

Probability of $-\hbar/2$:
$$P_- = \sin^2\frac{\theta}{2}$$

**(b)** Uncertainty is maximized when $P_+ = P_- = 1/2$:
$$\cos^2\frac{\theta}{2} = \frac{1}{2} \Rightarrow \theta = \frac{\pi}{2}$$

Maximum uncertainty occurs when measuring perpendicular to the spin direction.

---

## Part C: Stern-Gerlach

### Solution 17

**(a)**
- After SG1: state is $|\uparrow_z\rangle$ (50% of beam)
- SG2 at angle $\theta$: $|\uparrow_z\rangle = \cos\frac{\theta}{2}|+\rangle_{\theta} + \sin\frac{\theta}{2}|-\rangle_{\theta}$
  - Upper beam selected with probability $\cos^2(\theta/2)$
  - State becomes $|+\rangle_{\theta} = \cos\frac{\theta}{2}|\uparrow_z\rangle + \sin\frac{\theta}{2}|\downarrow_z\rangle$
- SG3 along z: probability of upper beam is $\cos^2(\theta/2)$

Total fraction in SG3 upper beam:
$$\frac{1}{2} \times \cos^2\frac{\theta}{2} \times \cos^2\frac{\theta}{2} = \frac{1}{2}\cos^4\frac{\theta}{2}$$

**(b)** To maximize, take $\frac{d}{d\theta}\cos^4(\theta/2) = 0$:
$$-4\cos^3\frac{\theta}{2}\sin\frac{\theta}{2}\cdot\frac{1}{2} = 0$$

This gives $\theta = 0$ (trivial) or $\theta = \pi$ (gives zero).

Maximum at $\theta = 0$, giving $\frac{1}{2}$.

Actually, the question asks for the upper beam of SG3 given the middle device exists. The maximum at $\theta = 0$ makes the middle device do nothing.

---

### Solution 18

**(a)** SGz → SGx → SGz:
- After SGz: $|\uparrow_z\rangle$
- After SGx (upper): $|+x\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle + |\downarrow\rangle)$ with probability 1/2
- After SGz (upper): probability 1/2

Final intensity: $(1)(1/2)(1/2) = 1/4$

**(b)** Just SGz → SGz (both upper):
- After first SGz: $|\uparrow_z\rangle$
- Second SGz: all in upper beam (probability 1)

Final intensity: 1

**Conclusion:** The intermediate measurement "destroys" the z-polarization.

---

## Part D: Spin Dynamics

### Solution 20

**(a)** $H = -\gamma\mathbf{S}\cdot\mathbf{B} = -\gamma B_0 S_z = \frac{\hbar\omega_L}{2}\sigma_z$ where $\omega_L = -\gamma B_0$

**(b)** $|\uparrow_x\rangle = \frac{1}{\sqrt{2}}(|\uparrow\rangle + |\downarrow\rangle)$

$$|\psi(t)\rangle = e^{-iHt/\hbar}|\uparrow_x\rangle = \frac{1}{\sqrt{2}}(e^{-i\omega_L t/2}|\uparrow\rangle + e^{i\omega_L t/2}|\downarrow\rangle)$$

**(c)**
$$\langle S_x\rangle(t) = \frac{\hbar}{2}\cos(\omega_L t)$$
$$\langle S_z\rangle(t) = \frac{1}{2}\cdot\frac{\hbar}{2} + \frac{1}{2}\cdot(-\frac{\hbar}{2}) = 0$$

---

### Solution 21

$\mathbf{B} = B_0\hat{x}$, $H = -\gamma B_0 S_x = \frac{\hbar\omega}{2}\sigma_x$ where $\omega = -\gamma B_0$

**(a)**
$$|\psi(t)\rangle = e^{-i\omega t\sigma_x/2}|\uparrow\rangle = \left(\cos\frac{\omega t}{2}I - i\sin\frac{\omega t}{2}\sigma_x\right)|\uparrow\rangle$$

$$= \cos\frac{\omega t}{2}|\uparrow\rangle - i\sin\frac{\omega t}{2}|\downarrow\rangle$$

**(b)** Complete flip to $|\downarrow\rangle$ when $\cos(\omega t/2) = 0$:
$$\omega t/2 = \pi/2 \Rightarrow t = \frac{\pi}{\omega} = \frac{\pi}{\gamma B_0}$$

**(c)**
$$P(\uparrow) = \cos^2\frac{\omega t}{2}$$

---

### Solution 23

At resonance with $H_{\text{eff}} = -\gamma B_1 S_x = \frac{\hbar\Omega_1}{2}\sigma_x$:

**(a)** Similar to Solution 21:
$$P_{\downarrow}(t) = \sin^2\frac{\Omega_1 t}{2}$$

**(b)** Rabi frequency: $\Omega_R = \Omega_1 = \gamma B_1$

**(c)** Complete spin flip when $\Omega_1 t/2 = \pi/2$:
$$t_{\pi} = \frac{\pi}{\Omega_1} = \frac{\pi}{\gamma B_1}$$

---

### Solution 24

**(a)**
Step 1: Evolve for time $\tau$ in field $B_0\hat{z}$:
$$|\psi(\tau)\rangle = e^{-i\omega_L\tau/2}|\uparrow\rangle$$

Step 2: Apply $\pi$ pulse about x (rotation by $\pi$):
$$R_x(\pi)|\psi(\tau)\rangle = (-i\sigma_x)(e^{-i\omega_L\tau/2}|\uparrow\rangle) = -ie^{-i\omega_L\tau/2}|\downarrow\rangle$$

Step 3: Evolve for another time $\tau$:
$$|\psi(2\tau)\rangle = e^{-iH\tau/\hbar}(-ie^{-i\omega_L\tau/2}|\downarrow\rangle) = -ie^{-i\omega_L\tau/2}e^{i\omega_L\tau/2}|\downarrow\rangle = -i|\downarrow\rangle$$

Actually, let me redo this more carefully. After the $\pi$ pulse, the state becomes $-i|\downarrow\rangle$. Evolving in $B_0\hat{z}$:
$$e^{-iH\tau/\hbar}|\downarrow\rangle = e^{i\omega_L\tau/2}|\downarrow\rangle$$

So final state: $-ie^{i\omega_L\tau/2}e^{-i\omega_L\tau/2}|\downarrow\rangle = -i|\downarrow\rangle$ (up to global phase).

**(b)** The key insight is that the phases from the two evolution periods cancel:
- First period: phase $e^{-i\omega_L\tau/2}$ for $|\uparrow\rangle$
- The $\pi$ pulse flips to $|\downarrow\rangle$
- Second period: phase $e^{+i\omega_L\tau/2}$ for $|\downarrow\rangle$

Net phase: $e^{-i\omega_L\tau/2}e^{+i\omega_L\tau/2} = 1$

This "refocusing" removes dephasing due to field inhomogeneities, which is crucial in NMR for achieving clear signals.

---

### Solution 25

Energy splitting: $\Delta E = g_s\mu_B B = 2 \times (9.274 \times 10^{-24})(1) = 1.855 \times 10^{-23}$ J

At temperature $T$:
$$\frac{N_{\uparrow}}{N_{\downarrow}} = e^{\Delta E/(k_B T)}$$

With $k_B T = (1.38 \times 10^{-23})(300) = 4.14 \times 10^{-21}$ J:

$$\frac{\Delta E}{k_B T} = \frac{1.855 \times 10^{-23}}{4.14 \times 10^{-21}} = 0.00448$$

$$P = \frac{N_{\uparrow} - N_{\downarrow}}{N_{\uparrow} + N_{\downarrow}} = \tanh\frac{\Delta E}{2k_B T} \approx \frac{\Delta E}{2k_B T} = 0.00224$$

**Answer:** Polarization $\approx 0.22\%$ at room temperature.

---

**End of Solutions**
