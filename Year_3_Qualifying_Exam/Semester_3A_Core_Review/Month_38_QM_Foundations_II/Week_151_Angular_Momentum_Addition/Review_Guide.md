# Week 151: Angular Momentum Addition - Comprehensive Review Guide

## Introduction

Adding angular momenta is fundamental to understanding atomic structure, selection rules, and quantum state spaces. This topic appears frequently on PhD qualifying exams, often in the context of atomic fine structure, nuclear physics, or multi-electron atoms. Mastery of Clebsch-Gordan coefficients is essential.

---

## Section 1: The Problem of Angular Momentum Addition

### Physical Motivation

Many physical systems involve multiple sources of angular momentum:
- **Atoms:** Orbital angular momentum $\mathbf{L}$ and spin $\mathbf{S}$
- **Multi-electron atoms:** Individual electron angular momenta
- **Nuclei:** Proton and neutron spins
- **Molecules:** Rotational and electronic angular momentum

We need to find eigenstates of the total angular momentum $\mathbf{J} = \mathbf{J}_1 + \mathbf{J}_2$.

### Mathematical Setup

Consider two angular momentum operators $\mathbf{J}_1$ and $\mathbf{J}_2$ acting on Hilbert spaces $\mathcal{H}_1$ and $\mathcal{H}_2$ respectively.

The combined system lives in the tensor product space:
$$\mathcal{H} = \mathcal{H}_1 \otimes \mathcal{H}_2$$

**Dimension:** $(2j_1 + 1)(2j_2 + 1)$

### Two Natural Bases

**Uncoupled Basis:** Eigenstates of $J_1^2$, $J_{1z}$, $J_2^2$, $J_{2z}$
$$|j_1, m_1; j_2, m_2\rangle = |j_1, m_1\rangle \otimes |j_2, m_2\rangle$$

**Coupled Basis:** Eigenstates of $J_1^2$, $J_2^2$, $J^2$, $J_z$
$$|j_1, j_2; j, m\rangle$$

Both bases span the same space and are related by a unitary transformation.

---

## Section 2: Allowed Values of Total Angular Momentum

### The Triangle Rule

When adding $\mathbf{J}_1$ and $\mathbf{J}_2$, the allowed values of total $j$ are:

$$\boxed{j = |j_1 - j_2|, |j_1 - j_2| + 1, \ldots, j_1 + j_2 - 1, j_1 + j_2}$$

This is the **triangle rule:** $j_1$, $j_2$, and $j$ must form a triangle (possibly degenerate).

### Dimension Check

For each allowed $j$, there are $(2j+1)$ values of $m$. The total count:

$$\sum_{j=|j_1-j_2|}^{j_1+j_2}(2j+1) = (2j_1+1)(2j_2+1)$$

This confirms both bases have the same dimension.

### Examples

**Two spin-1/2 particles:** $j_1 = j_2 = 1/2$
- $j = 0, 1$
- Dimension: $2 \times 2 = 4 = 1 + 3$ ✓

**Spin-1 and spin-1/2:** $j_1 = 1$, $j_2 = 1/2$
- $j = 1/2, 3/2$
- Dimension: $3 \times 2 = 6 = 2 + 4$ ✓

**Two spin-1 particles:** $j_1 = j_2 = 1$
- $j = 0, 1, 2$
- Dimension: $3 \times 3 = 9 = 1 + 3 + 5$ ✓

---

## Section 3: Clebsch-Gordan Coefficients

### Definition

The Clebsch-Gordan (CG) coefficients are the expansion coefficients relating the two bases:

$$|j,m\rangle = \sum_{m_1,m_2} \langle j_1,m_1;j_2,m_2|j,m\rangle |j_1,m_1\rangle|j_2,m_2\rangle$$

**Notation variants:**
- $\langle j_1,m_1;j_2,m_2|j,m\rangle$ (Dirac notation)
- $C^{j,m}_{j_1,m_1;j_2,m_2}$
- $(j_1,m_1,j_2,m_2|j,m)$

### Fundamental Properties

**1. Selection rule:** CG coefficients vanish unless $m = m_1 + m_2$

**2. Triangle rule:** Vanish unless $|j_1 - j_2| \leq j \leq j_1 + j_2$

**3. Reality:** CG coefficients are real (Condon-Shortley convention)

**4. Orthogonality:**

$$\sum_{m_1,m_2}\langle j_1,m_1;j_2,m_2|j,m\rangle\langle j_1,m_1;j_2,m_2|j',m'\rangle = \delta_{jj'}\delta_{mm'}$$

$$\sum_{j,m}\langle j_1,m_1;j_2,m_2|j,m\rangle\langle j_1,m_1';j_2,m_2'|j,m\rangle = \delta_{m_1m_1'}\delta_{m_2m_2'}$$

### Symmetry Properties

$$\langle j_1,m_1;j_2,m_2|j,m\rangle = (-1)^{j_1+j_2-j}\langle j_2,m_2;j_1,m_1|j,m\rangle$$

$$\langle j_1,m_1;j_2,m_2|j,m\rangle = (-1)^{j_1+j_2-j}\langle j_1,-m_1;j_2,-m_2|j,-m\rangle$$

---

## Section 4: Calculating CG Coefficients

### Method 1: Highest Weight State

Start with the maximum $m$ state for maximum $j$:

$$|j_1+j_2, j_1+j_2\rangle = |j_1,j_1\rangle|j_2,j_2\rangle$$

This is unique, so the CG coefficient is 1.

Apply the lowering operator $J_- = J_{1-} + J_{2-}$ repeatedly to generate other states.

### Method 2: Recursion Relations

Apply $J_{\pm}$ to both sides of the CG expansion:

$$\sqrt{(j\mp m)(j\pm m+1)}\langle j_1,m_1;j_2,m_2|j,m\pm 1\rangle$$
$$= \sqrt{(j_1\mp m_1+1)(j_1\pm m_1)}\langle j_1,m_1\mp 1;j_2,m_2|j,m\rangle$$
$$+ \sqrt{(j_2\mp m_2+1)(j_2\pm m_2)}\langle j_1,m_1;j_2,m_2\mp 1|j,m\rangle$$

### Method 3: Orthogonality

Use orthogonality between states of different $j$ with same $m$ to determine relative coefficients.

### Example: Two Spin-1/2 Particles

For $j_1 = j_2 = 1/2$, we couple to $j = 0, 1$.

**Triplet ($j=1$):**

$$|1,1\rangle = |\uparrow\uparrow\rangle$$

$$|1,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$$

$$|1,-1\rangle = |\downarrow\downarrow\rangle$$

**Singlet ($j=0$):**

$$|0,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$$

The CG coefficients are:
- $\langle\frac{1}{2},\frac{1}{2};\frac{1}{2},\frac{1}{2}|1,1\rangle = 1$
- $\langle\frac{1}{2},\frac{1}{2};\frac{1}{2},-\frac{1}{2}|1,0\rangle = \frac{1}{\sqrt{2}}$
- $\langle\frac{1}{2},-\frac{1}{2};\frac{1}{2},\frac{1}{2}|1,0\rangle = \frac{1}{\sqrt{2}}$
- $\langle\frac{1}{2},\frac{1}{2};\frac{1}{2},-\frac{1}{2}|0,0\rangle = \frac{1}{\sqrt{2}}$
- $\langle\frac{1}{2},-\frac{1}{2};\frac{1}{2},\frac{1}{2}|0,0\rangle = -\frac{1}{\sqrt{2}}$

---

## Section 5: Spin-Orbit Coupling

### The $\mathbf{L}\cdot\mathbf{S}$ Operator

For an electron in a central potential, spin-orbit coupling arises from relativistic effects:

$$H_{SO} = \frac{1}{2m^2c^2}\frac{1}{r}\frac{dV}{dr}\mathbf{L}\cdot\mathbf{S} = \xi(r)\mathbf{L}\cdot\mathbf{S}$$

To evaluate $\mathbf{L}\cdot\mathbf{S}$, use the total angular momentum $\mathbf{J} = \mathbf{L} + \mathbf{S}$:

$$\mathbf{J}^2 = \mathbf{L}^2 + \mathbf{S}^2 + 2\mathbf{L}\cdot\mathbf{S}$$

$$\boxed{\mathbf{L}\cdot\mathbf{S} = \frac{1}{2}(\mathbf{J}^2 - \mathbf{L}^2 - \mathbf{S}^2)}$$

### Eigenvalues

In the coupled basis $|n,l,s,j,m_j\rangle$:

$$\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$

For an electron ($s = 1/2$) with orbital angular momentum $l$:
- $j = l + 1/2$: $\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}l$
- $j = l - 1/2$: $\langle\mathbf{L}\cdot\mathbf{S}\rangle = -\frac{\hbar^2}{2}(l+1)$

### Fine Structure Splitting

The energy splitting between $j = l + 1/2$ and $j = l - 1/2$ levels is:

$$\Delta E = E_{l+1/2} - E_{l-1/2} = \frac{\hbar^2}{2}\langle\xi\rangle[(l) - (-(l+1))] = \frac{\hbar^2}{2}\langle\xi\rangle(2l+1)$$

This is proportional to $(2l+1)$, explaining the structure of fine-structure multiplets.

---

## Section 6: Selection Rules

### Electric Dipole Transitions

For electric dipole (E1) transitions, the transition amplitude involves:

$$\langle f|\mathbf{r}|i\rangle$$

Angular momentum conservation requires:

$$\boxed{\Delta j = 0, \pm 1 \quad (\text{but } j=0 \not\to j'=0)}$$

$$\boxed{\Delta m = 0, \pm 1}$$

From parity: $\Delta l = \pm 1$ (for orbital angular momentum)

Spin is unchanged: $\Delta s = 0$

### Physical Interpretation

The photon carries spin-1, so the initial + photon angular momentum must match the final state. The $j = 0 \to j' = 0$ forbidden rule arises because you cannot combine $j=0$ with spin-1 photon to get $j'=0$.

### General Rule

For any tensor operator of rank $k$:
$$\Delta j = 0, \pm 1, \ldots, \pm k$$
subject to the triangle rule $|j_i - k| \leq j_f \leq j_i + k$.

---

## Section 7: Term Symbols and Spectroscopy

### Term Symbol Notation

Atomic states are labeled by:

$$^{2S+1}L_J$$

- $S$ = total spin quantum number
- $L$ = total orbital quantum number (S, P, D, F, ... for L = 0, 1, 2, 3, ...)
- $J$ = total angular momentum quantum number
- $2S+1$ = spin multiplicity

### Examples

| Configuration | Term Symbol | Meaning |
|--------------|-------------|---------|
| $1s$ | $^2S_{1/2}$ | One electron, $L=0$, $S=1/2$, $J=1/2$ |
| $2p$ | $^2P_{1/2}$, $^2P_{3/2}$ | $L=1$, $S=1/2$, $J=1/2$ or $3/2$ |
| $3d$ | $^2D_{3/2}$, $^2D_{5/2}$ | $L=2$, $S=1/2$, $J=3/2$ or $5/2$ |
| $1s^2$ | $^1S_0$ | Closed shell, all zero |
| $1s2s$ (triplet) | $^3S_1$ | $L=0$, $S=1$, $J=1$ |
| $1s2s$ (singlet) | $^1S_0$ | $L=0$, $S=0$, $J=0$ |

### Hund's Rules

For ground state of multi-electron atoms:
1. Maximize $S$ (minimize electron repulsion via exchange)
2. Maximize $L$ (subject to rule 1)
3. $J = |L-S|$ for less than half-filled shell; $J = L+S$ for more than half-filled

---

## Section 8: Adding Three or More Angular Momenta

### Sequential Coupling

For three angular momenta, first couple two, then couple the result with the third:

$$\mathbf{J}_{12} = \mathbf{J}_1 + \mathbf{J}_2$$
$$\mathbf{J} = \mathbf{J}_{12} + \mathbf{J}_3$$

Different coupling orders (e.g., $\mathbf{J}_{13} = \mathbf{J}_1 + \mathbf{J}_3$ first) give different intermediate bases, related by **recoupling coefficients** (6j symbols, 9j symbols).

### Wigner 3j and 6j Symbols

More symmetric notation for angular momentum coupling:

**3j symbol:** Related to CG coefficients by:
$$\begin{pmatrix} j_1 & j_2 & j \\ m_1 & m_2 & -m \end{pmatrix} = \frac{(-1)^{j_1-j_2+m}}{\sqrt{2j+1}}\langle j_1,m_1;j_2,m_2|j,m\rangle$$

**6j symbol:** Describes recoupling of three angular momenta.

---

## Summary of Key Results

| Concept | Key Formula |
|---------|-------------|
| Allowed $j$ values | $\|j_1 - j_2\| \leq j \leq j_1 + j_2$ |
| CG definition | $\|j,m\rangle = \sum \langle j_1,m_1;j_2,m_2\|j,m\rangle\|j_1,m_1\rangle\|j_2,m_2\rangle$ |
| Selection rule | $m = m_1 + m_2$ |
| Spin-orbit operator | $\mathbf{L}\cdot\mathbf{S} = \frac{1}{2}(J^2 - L^2 - S^2)$ |
| E1 selection | $\Delta j = 0, \pm 1$; $\Delta m = 0, \pm 1$ |
| Term symbol | $^{2S+1}L_J$ |

---

## References

1. Sakurai, J.J. *Modern Quantum Mechanics*, Section 3.8
2. Shankar, R. *Principles of Quantum Mechanics*, Chapter 15
3. Griffiths, D.J. *Introduction to Quantum Mechanics*, Section 4.4
4. Condon, E.U. & Shortley, G.H. *The Theory of Atomic Spectra*

---

**Word Count:** ~2500
**Last Updated:** February 9, 2026
