# Week 149: Orbital Angular Momentum - Comprehensive Review Guide

## Introduction

Orbital angular momentum is one of the most fundamental concepts in quantum mechanics and appears on virtually every PhD qualifying examination. This review guide provides a thorough treatment of the theory, emphasizing the mathematical rigor and physical intuition expected at the doctoral level.

---

## Section 1: Classical Angular Momentum and Quantum Correspondence

### Classical Definition

In classical mechanics, the angular momentum of a particle with position $\mathbf{r}$ and momentum $\mathbf{p}$ is:

$$\mathbf{L} = \mathbf{r} \times \mathbf{p}$$

In component form:

$$L_x = yp_z - zp_y, \quad L_y = zp_x - xp_z, \quad L_z = xp_y - yp_x$$

### Quantum Mechanical Operators

Applying the correspondence principle $\mathbf{p} \to -i\hbar\nabla$, we obtain the angular momentum operators:

$$\hat{L}_x = -i\hbar\left(y\frac{\partial}{\partial z} - z\frac{\partial}{\partial y}\right)$$

$$\hat{L}_y = -i\hbar\left(z\frac{\partial}{\partial x} - x\frac{\partial}{\partial z}\right)$$

$$\hat{L}_z = -i\hbar\left(x\frac{\partial}{\partial y} - y\frac{\partial}{\partial x}\right)$$

In spherical coordinates $(r,\theta,\phi)$, these take particularly useful forms:

$$\hat{L}_z = -i\hbar\frac{\partial}{\partial\phi}$$

$$\hat{L}_{\pm} = \hbar e^{\pm i\phi}\left(\pm\frac{\partial}{\partial\theta} + i\cot\theta\frac{\partial}{\partial\phi}\right)$$

$$\hat{L}^2 = -\hbar^2\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]$$

---

## Section 2: Commutation Relations

### Fundamental Commutators

The angular momentum operators satisfy the fundamental commutation relations:

$$\boxed{[L_i, L_j] = i\hbar\epsilon_{ijk}L_k}$$

Explicitly:
- $[L_x, L_y] = i\hbar L_z$
- $[L_y, L_z] = i\hbar L_x$
- $[L_z, L_x] = i\hbar L_y$

**Derivation from position-momentum commutators:**

Starting from $[x_i, p_j] = i\hbar\delta_{ij}$, we compute:

$$[L_x, L_y] = [yp_z - zp_y, zp_x - xp_z]$$

Expanding using $[AB, CD] = A[B,C]D + [A,C]BD + CA[B,D] + C[A,D]B$:

$$= y[p_z, z]p_x + z[z, x]p_y p_z - z[p_y, x]p_z - x[z, z]p_y p_z + \ldots$$

After careful calculation:

$$= yp_x(-i\hbar) - (-i\hbar)xp_y = i\hbar(xp_y - yp_x) = i\hbar L_z$$

### Commutation with $L^2$

The total angular momentum squared commutes with all components:

$$[L^2, L_x] = [L^2, L_y] = [L^2, L_z] = 0$$

**Proof:**

$$[L^2, L_z] = [L_x^2 + L_y^2 + L_z^2, L_z]$$

$$= [L_x^2, L_z] + [L_y^2, L_z]$$

Using $[A^2, B] = A[A,B] + [A,B]A$:

$$= L_x[L_x, L_z] + [L_x, L_z]L_x + L_y[L_y, L_z] + [L_y, L_z]L_y$$

$$= L_x(-i\hbar L_y) + (-i\hbar L_y)L_x + L_y(i\hbar L_x) + (i\hbar L_x)L_y$$

$$= -i\hbar(L_xL_y + L_yL_x) + i\hbar(L_yL_x + L_xL_y) = 0$$

**Physical significance:** Since $[L^2, L_z] = 0$, we can simultaneously measure the magnitude of angular momentum and its z-component. However, since $[L_x, L_y] \neq 0$, we cannot simultaneously measure all three components.

---

## Section 3: Ladder Operators and Eigenvalue Spectrum

### Defining Ladder Operators

The raising and lowering operators are defined as:

$$L_+ = L_x + iL_y, \quad L_- = L_x - iL_y$$

These satisfy:

$$[L_z, L_{\pm}] = \pm\hbar L_{\pm}$$

$$[L_+, L_-] = 2\hbar L_z$$

$$L^2 = L_-L_+ + L_z^2 + \hbar L_z = L_+L_- + L_z^2 - \hbar L_z$$

### Eigenvalue Derivation

Let $|l,m\rangle$ be a simultaneous eigenstate of $L^2$ and $L_z$:

$$L^2|l,m\rangle = \lambda|l,m\rangle, \quad L_z|l,m\rangle = \mu|l,m\rangle$$

**Step 1: Effect of ladder operators on $L_z$ eigenvalue**

Using $[L_z, L_{\pm}] = \pm\hbar L_{\pm}$:

$$L_z(L_{\pm}|l,m\rangle) = (L_{\pm}L_z \pm \hbar L_{\pm})|l,m\rangle = (\mu \pm \hbar)(L_{\pm}|l,m\rangle)$$

Therefore $L_{\pm}|l,m\rangle$ is an eigenstate of $L_z$ with eigenvalue $\mu \pm \hbar$.

**Step 2: Bounds on the eigenvalues**

Since $L_x$ and $L_y$ are Hermitian, $L_x^2$ and $L_y^2$ are positive semi-definite:

$$\langle l,m|L_x^2 + L_y^2|l,m\rangle = \langle l,m|L^2 - L_z^2|l,m\rangle = \lambda - \mu^2 \geq 0$$

Thus $\mu^2 \leq \lambda$, meaning $\mu$ is bounded.

**Step 3: Top and bottom of the ladder**

Since $\mu$ is bounded, there exist maximum and minimum values $\mu_{max}$ and $\mu_{min}$.

At the top: $L_+|l, \mu_{max}\rangle = 0$

At the bottom: $L_-|l, \mu_{min}\rangle = 0$

**Step 4: Finding the eigenvalues**

From $L_+|l, \mu_{max}\rangle = 0$:

$$L_-L_+|l, \mu_{max}\rangle = (L^2 - L_z^2 - \hbar L_z)|l, \mu_{max}\rangle = 0$$

$$\Rightarrow \lambda = \mu_{max}^2 + \hbar\mu_{max} = \mu_{max}(\mu_{max} + \hbar)$$

Similarly from $L_+L_-|l, \mu_{min}\rangle = 0$:

$$\lambda = \mu_{min}^2 - \hbar\mu_{min} = \mu_{min}(\mu_{min} - \hbar)$$

Comparing: $\mu_{max}(\mu_{max} + \hbar) = \mu_{min}(\mu_{min} - \hbar)$

This gives $\mu_{max} = -\mu_{min}$.

Since $\mu$ changes by $\hbar$ at each step, $\mu_{max} - \mu_{min} = n\hbar$ for some non-negative integer $n$.

Thus $2\mu_{max} = n\hbar$, giving $\mu_{max} = n\hbar/2$.

**Step 5: Standard notation**

Define $l = n/2$ (where $n = 0, 1, 2, \ldots$) and $m = \mu/\hbar$. Then:

$$\boxed{L^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle}$$

$$\boxed{L_z|l,m\rangle = \hbar m|l,m\rangle}$$

where $l = 0, \frac{1}{2}, 1, \frac{3}{2}, \ldots$ and $m = -l, -l+1, \ldots, l-1, l$.

**Important:** For orbital angular momentum, $l$ must be a non-negative integer (not half-integer) due to single-valuedness of the wave function under $\phi \to \phi + 2\pi$.

---

## Section 4: Matrix Elements and Normalization

### Ladder Operator Matrix Elements

To find the normalization of $L_{\pm}|l,m\rangle$, we compute:

$$\|L_+|l,m\rangle\|^2 = \langle l,m|L_-L_+|l,m\rangle = \langle l,m|L^2 - L_z^2 - \hbar L_z|l,m\rangle$$

$$= \hbar^2[l(l+1) - m^2 - m] = \hbar^2[l(l+1) - m(m+1)]$$

Therefore:

$$\boxed{L_+|l,m\rangle = \hbar\sqrt{l(l+1) - m(m+1)}|l,m+1\rangle}$$

Similarly:

$$\boxed{L_-|l,m\rangle = \hbar\sqrt{l(l+1) - m(m-1)}|l,m-1\rangle}$$

Alternative form using $(l-m)(l+m+1) = l(l+1) - m(m+1)$:

$$L_+|l,m\rangle = \hbar\sqrt{(l-m)(l+m+1)}|l,m+1\rangle$$

$$L_-|l,m\rangle = \hbar\sqrt{(l+m)(l-m+1)}|l,m-1\rangle$$

### Matrix Representations

For $l = 1$ in the basis $\{|1,1\rangle, |1,0\rangle, |1,-1\rangle\}$:

$$L_z = \hbar\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

$$L_+ = \hbar\sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}, \quad L_- = \hbar\sqrt{2}\begin{pmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$

$$L_x = \frac{L_+ + L_-}{2} = \frac{\hbar}{\sqrt{2}}\begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

$$L_y = \frac{L_+ - L_-}{2i} = \frac{\hbar}{\sqrt{2}}\begin{pmatrix} 0 & -i & 0 \\ i & 0 & -i \\ 0 & i & 0 \end{pmatrix}$$

---

## Section 5: Spherical Harmonics

### Definition and Properties

The spherical harmonics $Y_l^m(\theta,\phi)$ are the position-space representation of $|l,m\rangle$:

$$Y_l^m(\theta,\phi) = \langle\theta,\phi|l,m\rangle$$

They satisfy the eigenvalue equations:

$$L^2 Y_l^m = \hbar^2 l(l+1) Y_l^m$$

$$L_z Y_l^m = \hbar m Y_l^m$$

### Explicit Form

The spherical harmonics are given by:

$$Y_l^m(\theta,\phi) = (-1)^m\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\phi}$$

where $P_l^m$ are the associated Legendre polynomials:

$$P_l^m(x) = \frac{1}{2^l l!}(1-x^2)^{m/2}\frac{d^{l+m}}{dx^{l+m}}(x^2-1)^l$$

### Low-Order Spherical Harmonics

**$l = 0$:**
$$Y_0^0 = \frac{1}{\sqrt{4\pi}}$$

**$l = 1$:**
$$Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta$$

$$Y_1^{\pm 1} = \mp\sqrt{\frac{3}{8\pi}}\sin\theta\, e^{\pm i\phi}$$

**$l = 2$:**
$$Y_2^0 = \sqrt{\frac{5}{16\pi}}(3\cos^2\theta - 1)$$

$$Y_2^{\pm 1} = \mp\sqrt{\frac{15}{8\pi}}\sin\theta\cos\theta\, e^{\pm i\phi}$$

$$Y_2^{\pm 2} = \sqrt{\frac{15}{32\pi}}\sin^2\theta\, e^{\pm 2i\phi}$$

### Orthonormality and Completeness

$$\int_0^{2\pi}d\phi\int_0^{\pi}\sin\theta\, d\theta\, Y_l^{m*}(\theta,\phi)Y_{l'}^{m'}(\theta,\phi) = \delta_{ll'}\delta_{mm'}$$

$$\sum_{l=0}^{\infty}\sum_{m=-l}^{l}Y_l^{m*}(\theta',\phi')Y_l^m(\theta,\phi) = \delta(\cos\theta - \cos\theta')\delta(\phi - \phi')$$

### Parity

Under parity transformation $(\theta,\phi) \to (\pi-\theta, \phi+\pi)$:

$$Y_l^m(\pi-\theta, \phi+\pi) = (-1)^l Y_l^m(\theta,\phi)$$

Thus spherical harmonics have parity $(-1)^l$.

---

## Section 6: Central Potentials

### Separation of Variables

For a central potential $V(r)$, the Hamiltonian is:

$$H = -\frac{\hbar^2}{2m}\nabla^2 + V(r)$$

In spherical coordinates:

$$\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\right) + \frac{1}{r^2}\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]$$

The angular part is $-L^2/\hbar^2$, so:

$$H = -\frac{\hbar^2}{2m}\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{d}{dr}\right) + \frac{L^2}{2mr^2} + V(r)$$

Since $[H, L^2] = [H, L_z] = 0$, we seek simultaneous eigenstates:

$$\psi(r,\theta,\phi) = R(r)Y_l^m(\theta,\phi)$$

### Radial Equation

Substituting into the Schrödinger equation:

$$\left[-\frac{\hbar^2}{2m}\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{d}{dr}\right) + \frac{\hbar^2 l(l+1)}{2mr^2} + V(r)\right]R(r) = ER(r)$$

Defining $u(r) = rR(r)$, this becomes:

$$-\frac{\hbar^2}{2m}\frac{d^2u}{dr^2} + \left[V(r) + \frac{\hbar^2 l(l+1)}{2mr^2}\right]u = Eu$$

This is a 1D Schrödinger equation with effective potential:

$$V_{\text{eff}}(r) = V(r) + \frac{\hbar^2 l(l+1)}{2mr^2}$$

The second term is the centrifugal barrier.

---

## Section 7: The Hydrogen Atom

### The Coulomb Potential

For hydrogen:

$$V(r) = -\frac{e^2}{4\pi\epsilon_0 r} = -\frac{e^2}{r}$$

(using Gaussian units for simplicity in the second form)

### Energy Eigenvalues

The bound state energies are:

$$\boxed{E_n = -\frac{m_e e^4}{2(4\pi\epsilon_0)^2\hbar^2 n^2} = -\frac{13.6\text{ eV}}{n^2}}$$

where $n = 1, 2, 3, \ldots$ is the principal quantum number.

In terms of the Bohr radius $a_0 = \frac{4\pi\epsilon_0\hbar^2}{m_e e^2} \approx 0.529$ Å:

$$E_n = -\frac{e^2}{2a_0 n^2}$$

### Quantum Numbers and Degeneracy

Each energy level $E_n$ has states characterized by:
- Principal quantum number: $n = 1, 2, 3, \ldots$
- Angular momentum: $l = 0, 1, 2, \ldots, n-1$
- Magnetic quantum number: $m = -l, -l+1, \ldots, l$

**Degeneracy:** For each $n$, the degeneracy is:

$$g_n = \sum_{l=0}^{n-1}(2l+1) = n^2$$

(Not counting spin; with spin: $2n^2$)

### Radial Wave Functions

$$R_{nl}(r) = -\sqrt{\left(\frac{2}{na_0}\right)^3\frac{(n-l-1)!}{2n[(n+l)!]^3}}e^{-r/na_0}\left(\frac{2r}{na_0}\right)^l L_{n-l-1}^{2l+1}\left(\frac{2r}{na_0}\right)$$

where $L_q^p$ are associated Laguerre polynomials.

**Key examples:**

$$R_{10}(r) = 2\left(\frac{1}{a_0}\right)^{3/2}e^{-r/a_0}$$

$$R_{20}(r) = \frac{1}{2\sqrt{2}}\left(\frac{1}{a_0}\right)^{3/2}\left(2 - \frac{r}{a_0}\right)e^{-r/2a_0}$$

$$R_{21}(r) = \frac{1}{2\sqrt{6}}\left(\frac{1}{a_0}\right)^{3/2}\frac{r}{a_0}e^{-r/2a_0}$$

### Expectation Values

$$\langle r \rangle_{nl} = \frac{a_0}{2}\left[3n^2 - l(l+1)\right]$$

$$\langle r^2 \rangle_{nl} = \frac{a_0^2 n^2}{2}\left[5n^2 + 1 - 3l(l+1)\right]$$

$$\left\langle\frac{1}{r}\right\rangle_{nl} = \frac{1}{n^2 a_0}$$

$$\left\langle\frac{1}{r^2}\right\rangle_{nl} = \frac{1}{n^3 a_0^2(l+1/2)}$$

$$\left\langle\frac{1}{r^3}\right\rangle_{nl} = \frac{1}{n^3 a_0^3 l(l+1/2)(l+1)} \quad (l \neq 0)$$

---

## Section 8: Selection Rules Preview

Angular momentum conservation leads to selection rules for electromagnetic transitions. For electric dipole (E1) transitions:

$$\Delta l = \pm 1, \quad \Delta m = 0, \pm 1$$

The $\Delta l = \pm 1$ rule follows from the parity of $Y_l^m$ and the dipole operator $\mathbf{r}$.

---

## Summary of Key Results

| Quantity | Expression |
|----------|------------|
| $L^2$ eigenvalue | $\hbar^2 l(l+1)$ |
| $L_z$ eigenvalue | $\hbar m$ |
| Allowed $l$ (orbital) | $0, 1, 2, \ldots$ |
| Allowed $m$ | $-l, -l+1, \ldots, l$ |
| $L_+$ action | $\hbar\sqrt{(l-m)(l+m+1)}\|l,m+1\rangle$ |
| $L_-$ action | $\hbar\sqrt{(l+m)(l-m+1)}\|l,m-1\rangle$ |
| Hydrogen $E_n$ | $-13.6\text{ eV}/n^2$ |
| Degeneracy | $n^2$ (or $2n^2$ with spin) |

---

## References

1. Shankar, R. *Principles of Quantum Mechanics*, Chapter 12
2. Sakurai, J.J. *Modern Quantum Mechanics*, Chapter 3
3. Griffiths, D.J. *Introduction to Quantum Mechanics*, Chapter 4
4. Cohen-Tannoudji, C. *Quantum Mechanics*, Chapter VI

---

**Word Count:** ~2500
**Last Updated:** February 9, 2026
