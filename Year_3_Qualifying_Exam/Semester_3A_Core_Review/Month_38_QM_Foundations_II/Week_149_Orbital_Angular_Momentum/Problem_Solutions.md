# Week 149: Orbital Angular Momentum - Problem Solutions

## Part A: Angular Momentum Algebra

### Solution 1

We need to verify $[L_x, L_y] = i\hbar L_z$.

Starting with $L_x = yp_z - zp_y$ and $L_y = zp_x - xp_z$:

$$[L_x, L_y] = [yp_z - zp_y, zp_x - xp_z]$$

Expanding:
$$= [yp_z, zp_x] - [yp_z, xp_z] - [zp_y, zp_x] + [zp_y, xp_z]$$

Using $[AB, CD] = A[B,C]D + AC[B,D] + [A,C]BD + C[A,D]B$:

**First term:** $[yp_z, zp_x] = y[p_z, z]p_x = y(-i\hbar)p_x = -i\hbar yp_x$

**Second term:** $[yp_z, xp_z] = 0$ (no matching position-momentum pairs)

**Third term:** $[zp_y, zp_x] = 0$ (no matching position-momentum pairs)

**Fourth term:** $[zp_y, xp_z] = x[z, p_z]p_y = x(i\hbar)p_y = i\hbar xp_y$

Therefore:
$$[L_x, L_y] = -i\hbar yp_x + i\hbar xp_y = i\hbar(xp_y - yp_x) = i\hbar L_z \quad \checkmark$$

---

### Solution 2

$$[L^2, L_z] = [L_x^2 + L_y^2 + L_z^2, L_z]$$

Since $[L_z^2, L_z] = 0$:
$$= [L_x^2, L_z] + [L_y^2, L_z]$$

Using $[A^2, B] = A[A,B] + [A,B]A$:
$$[L_x^2, L_z] = L_x[L_x, L_z] + [L_x, L_z]L_x = L_x(-i\hbar L_y) + (-i\hbar L_y)L_x = -i\hbar(L_xL_y + L_yL_x)$$

$$[L_y^2, L_z] = L_y[L_y, L_z] + [L_y, L_z]L_y = L_y(i\hbar L_x) + (i\hbar L_x)L_y = i\hbar(L_yL_x + L_xL_y)$$

Adding:
$$[L^2, L_z] = -i\hbar(L_xL_y + L_yL_x) + i\hbar(L_yL_x + L_xL_y) = 0 \quad \checkmark$$

---

### Solution 3

**(a)** $[L_z, L_+] = [L_z, L_x + iL_y] = [L_z, L_x] + i[L_z, L_y]$

Using $[L_z, L_x] = i\hbar L_y$ and $[L_z, L_y] = -i\hbar L_x$:
$$= i\hbar L_y + i(-i\hbar L_x) = i\hbar L_y + \hbar L_x = \hbar(L_x + iL_y) = \hbar L_+ \quad \checkmark$$

**(b)** $[L_z, L_-] = [L_z, L_x - iL_y] = i\hbar L_y - i(-i\hbar L_x) = i\hbar L_y - \hbar L_x = -\hbar L_- \quad \checkmark$

**(c)** $[L_+, L_-] = [L_x + iL_y, L_x - iL_y]$
$$= [L_x, L_x] - i[L_x, L_y] + i[L_y, L_x] - i^2[L_y, L_y]$$
$$= 0 - i(i\hbar L_z) + i(-i\hbar L_z) - 0 = \hbar L_z + \hbar L_z = 2\hbar L_z \quad \checkmark$$

---

### Solution 4

**(a)** Given $L^2|\psi\rangle = 6\hbar^2|\psi\rangle$, we have $l(l+1) = 6$, so $l = 2$.

Possible values of $L_z$: $m\hbar$ where $m = -2, -1, 0, 1, 2$

**Answer:** $L_z \in \{-2\hbar, -\hbar, 0, \hbar, 2\hbar\}$

**(b)** For $|l=2, m=2\rangle$, we use the uncertainty principle:

$$\Delta L_x \cdot \Delta L_y \geq \frac{1}{2}|\langle[L_x, L_y]\rangle| = \frac{1}{2}|\langle i\hbar L_z\rangle| = \frac{\hbar}{2}|\langle L_z\rangle| = \frac{\hbar}{2} \cdot 2\hbar = \hbar^2$$

By symmetry, $\Delta L_x = \Delta L_y$ for an $|l,m\rangle$ state.

Also: $\langle L_x^2 + L_y^2\rangle = \langle L^2 - L_z^2\rangle = 6\hbar^2 - 4\hbar^2 = 2\hbar^2$

Since $\langle L_x\rangle = \langle L_y\rangle = 0$ for $|l,m\rangle$ states:
$$(\Delta L_x)^2 + (\Delta L_y)^2 = 2\hbar^2$$

With $\Delta L_x = \Delta L_y$: $2(\Delta L_x)^2 = 2\hbar^2$

**Answer:** $\Delta L_x = \hbar$

---

### Solution 5

**(a)** $|\psi\rangle = \frac{1}{\sqrt{3}}|1,1\rangle + \sqrt{\frac{2}{3}}|1,0\rangle$

$$\langle L_z\rangle = \frac{1}{3}(\hbar) + \frac{2}{3}(0) = \frac{\hbar}{3}$$

$$\langle L_z^2\rangle = \frac{1}{3}(\hbar^2) + \frac{2}{3}(0) = \frac{\hbar^2}{3}$$

**(b)** $\Delta L_z = \sqrt{\langle L_z^2\rangle - \langle L_z\rangle^2} = \sqrt{\frac{\hbar^2}{3} - \frac{\hbar^2}{9}} = \sqrt{\frac{2\hbar^2}{9}} = \frac{\hbar\sqrt{2}}{3}$

**(c)** The state has no $|1,-1\rangle$ component.

**Answer:** $P(L_z = -\hbar) = 0$

---

### Solution 6

For $|l=2, m=1\rangle$:

**(a)** $\langle L_x\rangle = \frac{1}{2}\langle 2,1|L_+ + L_-|2,1\rangle$

Since $L_+|2,1\rangle \propto |2,2\rangle$ and $L_-|2,1\rangle \propto |2,0\rangle$, both are orthogonal to $|2,1\rangle$.

**Answer:** $\langle L_x\rangle = 0$

**(b)** $\langle L_x^2\rangle = \frac{1}{4}\langle 2,1|(L_+ + L_-)^2|2,1\rangle = \frac{1}{4}\langle 2,1|L_+L_- + L_-L_+|2,1\rangle$

Using $L_+L_- = L^2 - L_z^2 + \hbar L_z$ and $L_-L_+ = L^2 - L_z^2 - \hbar L_z$:

$$L_+L_- + L_-L_+ = 2(L^2 - L_z^2)$$

$$\langle L_x^2\rangle = \frac{1}{2}\langle L^2 - L_z^2\rangle = \frac{1}{2}[6\hbar^2 - \hbar^2] = \frac{5\hbar^2}{2}$$

**(c)** $\Delta L_x = \sqrt{\langle L_x^2\rangle - \langle L_x\rangle^2} = \sqrt{\frac{5\hbar^2}{2}}$

**Answer:** $\Delta L_x = \hbar\sqrt{5/2}$

---

## Part B: Ladder Operators and Matrix Elements

### Solution 9

**(a)** $L_+|2,1\rangle = \hbar\sqrt{2(3) - 1(2)}|2,2\rangle = \hbar\sqrt{4}|2,2\rangle = 2\hbar|2,2\rangle$

**(b)** $L_-|2,1\rangle = \hbar\sqrt{2(3) - 1(0)}|2,0\rangle = \hbar\sqrt{6}|2,0\rangle$

**(c)** $L_+L_-|2,1\rangle = L_+(\hbar\sqrt{6}|2,0\rangle) = \hbar\sqrt{6} \cdot \hbar\sqrt{6}|2,1\rangle = 6\hbar^2|2,1\rangle$

---

### Solution 10

For $l=1$ in basis $\{|1,1\rangle, |1,0\rangle, |1,-1\rangle\}$:

**$L_z$:**
$$L_z = \hbar\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

**$L_+$:** Non-zero elements are $\langle 1|L_+|0\rangle = \hbar\sqrt{2}$ and $\langle 0|L_+|-1\rangle = \hbar\sqrt{2}$

$$L_+ = \hbar\sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

**$L_-$:**
$$L_- = \hbar\sqrt{2}\begin{pmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$

**$L_x = \frac{L_+ + L_-}{2}$:**
$$L_x = \frac{\hbar}{\sqrt{2}}\begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

**$L_y = \frac{L_+ - L_-}{2i}$:**
$$L_y = \frac{\hbar}{\sqrt{2}}\begin{pmatrix} 0 & -i & 0 \\ i & 0 & -i \\ 0 & i & 0 \end{pmatrix}$$

---

### Solution 11

Finding eigenvalues of $L_x$ for $l=1$:

$$L_x = \frac{\hbar}{\sqrt{2}}\begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

Characteristic equation: $\det(L_x - \lambda I) = 0$

$$\det\begin{pmatrix} -\lambda & \hbar/\sqrt{2} & 0 \\ \hbar/\sqrt{2} & -\lambda & \hbar/\sqrt{2} \\ 0 & \hbar/\sqrt{2} & -\lambda \end{pmatrix} = 0$$

$$-\lambda(\lambda^2 - \hbar^2/2) - \frac{\hbar}{\sqrt{2}}(-\lambda\frac{\hbar}{\sqrt{2}}) = 0$$

$$-\lambda^3 + \lambda\hbar^2/2 + \lambda\hbar^2/2 = 0$$

$$\lambda(\lambda^2 - \hbar^2) = 0$$

**Eigenvalues:** $\lambda = 0, \pm\hbar$

**Eigenvectors:**

For $\lambda = \hbar$: $|L_x = \hbar\rangle = \frac{1}{2}|1,1\rangle + \frac{1}{\sqrt{2}}|1,0\rangle + \frac{1}{2}|1,-1\rangle$

For $\lambda = 0$: $|L_x = 0\rangle = \frac{1}{\sqrt{2}}|1,1\rangle - \frac{1}{\sqrt{2}}|1,-1\rangle$

For $\lambda = -\hbar$: $|L_x = -\hbar\rangle = \frac{1}{2}|1,1\rangle - \frac{1}{\sqrt{2}}|1,0\rangle + \frac{1}{2}|1,-1\rangle$

---

### Solution 12

Given $|\psi\rangle = \frac{1}{\sqrt{6}}\begin{pmatrix} 1 \\ 2 \\ 1 \end{pmatrix}$

**(a)**
$$\langle L_z\rangle = \frac{1}{6}[\hbar + 0 + (-\hbar)] = 0$$

$$\langle L^2\rangle = \frac{1}{6}[2\hbar^2 + 2\hbar^2 + 2\hbar^2] = 2\hbar^2$$ (since all components have $l=1$)

**(b)** From Solution 11, $|L_x = \hbar\rangle = \frac{1}{2}|1,1\rangle + \frac{1}{\sqrt{2}}|1,0\rangle + \frac{1}{2}|1,-1\rangle$

$$P(L_x = \hbar) = |\langle L_x = \hbar|\psi\rangle|^2 = \left|\frac{1}{2}\cdot\frac{1}{\sqrt{6}} + \frac{1}{\sqrt{2}}\cdot\frac{2}{\sqrt{6}} + \frac{1}{2}\cdot\frac{1}{\sqrt{6}}\right|^2$$

$$= \left|\frac{1}{2\sqrt{6}} + \frac{2}{\sqrt{12}} + \frac{1}{2\sqrt{6}}\right|^2 = \left|\frac{1}{\sqrt{6}} + \frac{2}{\sqrt{12}}\right|^2 = \left|\frac{1}{\sqrt{6}} + \frac{1}{\sqrt{3}}\right|^2$$

$$= \left|\frac{1 + \sqrt{2}}{\sqrt{6}}\right|^2 = \frac{(1+\sqrt{2})^2}{6} = \frac{3 + 2\sqrt{2}}{6}$$

**(c)** After measuring $L_x = \hbar$, the state collapses to:

$$|L_x = \hbar\rangle = \frac{1}{2}|1,1\rangle + \frac{1}{\sqrt{2}}|1,0\rangle + \frac{1}{2}|1,-1\rangle$$

---

## Part D: Central Potentials

### Solution 22

**(a)** For $l=0$, the radial equation is:

$$-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} = Eu$$

where $u(r) = rR(r)$.

Boundary conditions: $u(0) = 0$ and $u(a) = 0$.

Solution: $u(r) = A\sin(kr)$ where $k = \sqrt{2mE}/\hbar$

From $u(a) = 0$: $ka = n\pi$ for $n = 1, 2, 3, \ldots$

**Energy eigenvalues:**
$$E_n = \frac{\hbar^2 k^2}{2m} = \frac{n^2\pi^2\hbar^2}{2ma^2}$$

**(b)** Ground state ($n=1$, $l=0$):
$$E_0 = \frac{\pi^2\hbar^2}{2ma^2}$$

**(c)** First excited state can be either:
- $n=2$, $l=0$: $E = \frac{4\pi^2\hbar^2}{2ma^2}$
- $n=1$, $l=1$: Need to solve for $l=1$ case

For $l=1$, the solution involves spherical Bessel functions. The first zero of $j_1(x)$ is at $x \approx 4.49$, giving:

$$E_{l=1} = \frac{(4.49)^2\hbar^2}{2ma^2} \approx \frac{20.2\hbar^2}{2ma^2}$$

Since $4\pi^2 \approx 39.5 > 20.2$, the first excited state is $(n=1, l=1)$.

Degeneracy: For $l=1$, $m = -1, 0, 1$, so degeneracy = 3.

---

## Part E: Hydrogen Atom

### Solution 24

**(a)** $E_1 = -\frac{13.6\text{ eV}}{1^2} = -13.6\text{ eV}$

**(b)** $a_0 = \frac{4\pi\epsilon_0\hbar^2}{m_e e^2} \approx 0.529$ Å

**(c)** For the ground state $R_{10}(r) = 2(1/a_0)^{3/2}e^{-r/a_0}$

Radial probability density: $P(r) = r^2|R_{10}|^2 = 4(1/a_0)^3 r^2 e^{-2r/a_0}$

Maximum at $\frac{dP}{dr} = 0$:
$$\frac{d}{dr}(r^2 e^{-2r/a_0}) = (2r - \frac{2r^2}{a_0})e^{-2r/a_0} = 0$$

$$2r(1 - r/a_0) = 0 \Rightarrow r = a_0$$

**Answer:** Most probable radius = $a_0 = 0.529$ Å

---

### Solution 25

**(a)** $\psi_{210} = R_{21}(r)Y_1^0(\theta,\phi)$

$$R_{21}(r) = \frac{1}{2\sqrt{6}}\left(\frac{1}{a_0}\right)^{3/2}\frac{r}{a_0}e^{-r/2a_0}$$

$$Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta$$

$$\psi_{210} = \frac{1}{4\sqrt{2\pi}}\left(\frac{1}{a_0}\right)^{5/2}r\cos\theta\, e^{-r/2a_0}$$

**(b)** Radial probability density: $P(r) = r^2|R_{21}|^2 \propto r^4 e^{-r/a_0}$

$$\frac{dP}{dr} \propto (4r^3 - \frac{r^4}{a_0})e^{-r/a_0} = 0$$

$$r(4 - r/a_0) = 0 \Rightarrow r = 4a_0$$

**(c)** Using $\langle r\rangle_{nl} = \frac{a_0}{2}[3n^2 - l(l+1)]$:

$$\langle r\rangle_{21} = \frac{a_0}{2}[3(4) - 2] = \frac{a_0}{2}(10) = 5a_0$$

Note: $\langle r\rangle = 5a_0 > r_{max} = 4a_0$, due to asymmetric probability distribution.

---

### Solution 26

$|\psi\rangle = \frac{1}{\sqrt{3}}|2,0,0\rangle + \sqrt{\frac{2}{3}}|2,1,0\rangle$

**(a)** Both states have $n=2$:
$$\langle E\rangle = E_2 = -\frac{13.6\text{ eV}}{4} = -3.4\text{ eV}$$

**(b)**
$$\langle L^2\rangle = \frac{1}{3}(0) + \frac{2}{3}(2\hbar^2) = \frac{4\hbar^2}{3}$$

**(c)**
- $L^2 = 0$ with probability $1/3$ (from $|2,0,0\rangle$)
- $L^2 = 2\hbar^2$ with probability $2/3$ (from $|2,1,0\rangle$)

**(d)** After measuring $L^2 = 2\hbar^2$, the state collapses to $|2,1,0\rangle$.

For $|1,0\rangle$, the probability of $L_z = \hbar$ is $|\langle 1,1|1,0\rangle|^2 = 0$.

**Answer:** $P(L_z = \hbar | L^2 = 2\hbar^2) = 0$

---

### Solution 27

**(a)** Ground state is $|1,0,0\rangle$ with $l=0$, $m=0$.

For $l=0$, $\mathbf{L} = 0$, so $\mathbf{L}\cdot\hat{n} = 0$ always.

**Answer:** Only possible outcome is 0, with probability 1.

**(b)** For $|2,1,1\rangle$, using the $L_x$ eigenstates from Solution 11:

$$|1,1\rangle = \frac{1}{2}|L_x=\hbar\rangle - \frac{1}{\sqrt{2}}|L_x=0\rangle + \frac{1}{2}|L_x=-\hbar\rangle$$

Wait, let me invert the transformation. From Solution 11:

$|L_x = \hbar\rangle = \frac{1}{2}|1,1\rangle + \frac{1}{\sqrt{2}}|1,0\rangle + \frac{1}{2}|1,-1\rangle$

The inverse gives:
$$P(L_x = \hbar) = |{}_x\langle \hbar|1,1\rangle|^2 = 1/4$$
$$P(L_x = 0) = |{}_x\langle 0|1,1\rangle|^2 = 1/2$$
$$P(L_x = -\hbar) = |{}_x\langle -\hbar|1,1\rangle|^2 = 1/4$$

**(c)** $\langle L_x\rangle = \frac{1}{4}(\hbar) + \frac{1}{2}(0) + \frac{1}{4}(-\hbar) = 0$

$\langle L_x^2\rangle = \frac{1}{4}(\hbar^2) + \frac{1}{2}(0) + \frac{1}{4}(\hbar^2) = \frac{\hbar^2}{2}$

$$\langle L_x^2\rangle - \langle L_x\rangle^2 = \frac{\hbar^2}{2} - 0 = \frac{\hbar^2}{2}$$

---

**End of Solutions**
