# Week 151: Angular Momentum Addition - Problem Solutions

## Part A: Basic Addition and Counting

### Solution 1

**(a)** $j_1 = 2$, $j_2 = 1$: $j = |2-1|, ..., 2+1 = 1, 2, 3$

**(b)** $j_1 = 3/2$, $j_2 = 1$: $j = |3/2-1|, ..., 3/2+1 = 1/2, 3/2, 5/2$

**(c)** $j_1 = 5/2$, $j_2 = 3/2$: $j = |5/2-3/2|, ..., 5/2+3/2 = 1, 2, 3, 4$

**(d)** $j_1 = 2$, $j_2 = 2$: $j = 0, 1, 2, 3, 4$

---

### Solution 2

**(a)** Two spin-1 particles: $(2 \cdot 1 + 1)^2 = 9$

Allowed $j = 0, 1, 2$: $\sum(2j+1) = 1 + 3 + 5 = 9$ ✓

**(b)** Spin-3/2 and spin-1/2: $(4)(2) = 8$

Allowed $j = 1, 2$: $\sum = 3 + 5 = 8$ ✓

**(c)** Spin-2 and spin-1: $(5)(3) = 15$

Allowed $j = 1, 2, 3$: $\sum = 3 + 5 + 7 = 15$ ✓

---

### Solution 4

For 3d electron: $n=3$, $l=2$, $s=1/2$

**(a)** Possible $j$ values: $j = |l-s|, ..., l+s = 3/2, 5/2$

**(b)** For $j = 3/2$: $(2 \cdot 3/2 + 1) = 4$ states ($m_j = -3/2, -1/2, 1/2, 3/2$)

For $j = 5/2$: $(2 \cdot 5/2 + 1) = 6$ states

**(c)** Total: $4 + 6 = 10 = (2l+1)(2s+1) = 5 \cdot 2$ ✓

---

### Solution 5

**Three spin-1/2 particles:**

First couple two: $1/2 \otimes 1/2 = 0 \oplus 1$ (singlet and triplet)

Then add the third spin-1/2:
- $0 \otimes 1/2 = 1/2$
- $1 \otimes 1/2 = 1/2 \oplus 3/2$

**Total:** $j = 1/2$ (appears twice), $j = 3/2$

Dimension check: $2^3 = 8 = 2 + 2 + 4$ ✓

---

## Part B: Clebsch-Gordan Coefficients - Basic

### Solution 7

**Two spin-1/2 particles:**

**Uncoupled basis:** $|\uparrow\uparrow\rangle$, $|\uparrow\downarrow\rangle$, $|\downarrow\uparrow\rangle$, $|\downarrow\downarrow\rangle$

**Coupled basis:**

Triplet ($j=1$):
- $|1,1\rangle = |\uparrow\uparrow\rangle$
- $|1,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$
- $|1,-1\rangle = |\downarrow\downarrow\rangle$

Singlet ($j=0$):
- $|0,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$

---

### Solution 8

**(a)** $|2,2\rangle$ is the maximum state: $|j_1=1, m_1=1\rangle|j_2=1, m_2=1\rangle$

$\langle 1,1; 1,1 | 2,2 \rangle = 1$

**(b)** Apply $J_- = J_{1-} + J_{2-}$ to $|2,2\rangle$:

$J_-|2,2\rangle = \sqrt{4}|2,1\rangle = 2|2,1\rangle$

$(J_{1-} + J_{2-})|1,1\rangle|1,1\rangle = \sqrt{2}|1,0\rangle|1,1\rangle + \sqrt{2}|1,1\rangle|1,0\rangle$

Therefore: $|2,1\rangle = \frac{1}{\sqrt{2}}(|1,0\rangle|1,1\rangle + |1,1\rangle|1,0\rangle)$

$\langle 1,1; 1,0 | 2,1 \rangle = \frac{1}{\sqrt{2}}$

**(c)** Apply $J_-$ again to $|2,1\rangle$:

$J_-|2,1\rangle = \sqrt{6}|2,0\rangle$

After calculation: $|2,0\rangle = \frac{1}{\sqrt{6}}(|1,1\rangle|1,-1\rangle + 2|1,0\rangle|1,0\rangle + |1,-1\rangle|1,1\rangle)$

$\langle 1,0; 1,0 | 2,0 \rangle = \frac{2}{\sqrt{6}} = \sqrt{\frac{2}{3}}$

---

### Solution 9

For $j_1 = 1$, $j_2 = 1/2$, finding $|3/2, 1/2\rangle$:

Using the formula for adding $j_2 = 1/2$ to $j_1 = l = 1$:

$$|l + 1/2, m\rangle = \sqrt{\frac{l + m + 1/2}{2l+1}}|l, m-1/2\rangle|1/2, 1/2\rangle + \sqrt{\frac{l - m + 1/2}{2l+1}}|l, m+1/2\rangle|1/2, -1/2\rangle$$

For $m = 1/2$, $l = 1$:

$$|3/2, 1/2\rangle = \sqrt{\frac{1 + 1/2 + 1/2}{3}}|1, 0\rangle|1/2, 1/2\rangle + \sqrt{\frac{1 - 1/2 + 1/2}{3}}|1, 1\rangle|1/2, -1/2\rangle$$

$$= \sqrt{\frac{2}{3}}|1, 0\rangle|\uparrow\rangle + \sqrt{\frac{1}{3}}|1, 1\rangle|\downarrow\rangle$$

---

### Solution 11

For $j_1 = j_2 = 1$:

**(a)** $|j=2, m=1\rangle$: From Solution 8, applying lowering operator:

$$|2,1\rangle = \frac{1}{\sqrt{2}}(|1,1\rangle|1,0\rangle + |1,0\rangle|1,1\rangle)$$

**(b)** $|j=1, m=1\rangle$: Must be orthogonal to $|2,1\rangle$ in the $m=1$ subspace.

States with $m=1$: $|1,1\rangle|1,0\rangle$ and $|1,0\rangle|1,1\rangle$

Orthogonal combination:
$$|1,1\rangle = \frac{1}{\sqrt{2}}(|1,1\rangle|1,0\rangle - |1,0\rangle|1,1\rangle)$$

**(c)** Orthogonality check:
$$\langle 2,1|1,1\rangle = \frac{1}{2}(1)(1) + \frac{1}{2}(1)(-1) = 0 \checkmark$$

---

## Part D: Spin-Orbit Coupling

### Solution 18

Using $\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$:

**(a)** $^2P_{3/2}$: $l=1$, $s=1/2$, $j=3/2$

$$\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}\left[\frac{3}{2}\cdot\frac{5}{2} - 1\cdot 2 - \frac{1}{2}\cdot\frac{3}{2}\right] = \frac{\hbar^2}{2}\left[\frac{15}{4} - 2 - \frac{3}{4}\right] = \frac{\hbar^2}{2} \cdot 1 = \frac{\hbar^2}{2}$$

**(b)** $^2P_{1/2}$: $l=1$, $s=1/2$, $j=1/2$

$$\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}\left[\frac{1}{2}\cdot\frac{3}{2} - 2 - \frac{3}{4}\right] = \frac{\hbar^2}{2}\left[\frac{3}{4} - \frac{11}{4}\right] = \frac{\hbar^2}{2}(-2) = -\hbar^2$$

**(c)** $^2D_{5/2}$: $l=2$, $s=1/2$, $j=5/2$

$$\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}\left[\frac{5}{2}\cdot\frac{7}{2} - 2\cdot 3 - \frac{3}{4}\right] = \frac{\hbar^2}{2}\left[\frac{35}{4} - 6 - \frac{3}{4}\right] = \frac{\hbar^2}{2} \cdot 2 = \hbar^2$$

---

### Solution 19

For $n=2$ hydrogen:

**(a)** Possible $(l, j)$:
- $l=0$: $j = 1/2$ only ($^2S_{1/2}$)
- $l=1$: $j = 1/2, 3/2$ ($^2P_{1/2}$, $^2P_{3/2}$)

**(b)** From Solution 18:
- $^2S_{1/2}$: $\langle\mathbf{L}\cdot\mathbf{S}\rangle = 0$ (since $l=0$)
- $^2P_{1/2}$: $\langle\mathbf{L}\cdot\mathbf{S}\rangle = -\hbar^2$
- $^2P_{3/2}$: $\langle\mathbf{L}\cdot\mathbf{S}\rangle = \hbar^2/2$

**(c)** For spin-orbit coupling $H_{SO} = A(r)\mathbf{L}\cdot\mathbf{S}$ with $A > 0$:
- Negative $\langle\mathbf{L}\cdot\mathbf{S}\rangle$ gives lower energy
- **Lowest energy:** $^2P_{1/2}$ (most negative)
- But $^2S_{1/2}$ has no spin-orbit shift

Without other effects, $^2P_{1/2}$ is lowest among $l=1$ states.

---

### Solution 21

For 4f orbital: $l = 3$, $s = 1/2$

**(a)** $j = l \pm s = 5/2, 7/2$

**(b)** Landé g-factor: $g_J = 1 + \frac{j(j+1) + s(s+1) - l(l+1)}{2j(j+1)}$

For $j = 7/2$:
$$g_J = 1 + \frac{\frac{7}{2}\cdot\frac{9}{2} + \frac{3}{4} - 12}{2\cdot\frac{7}{2}\cdot\frac{9}{2}} = 1 + \frac{\frac{63}{4} + \frac{3}{4} - 12}{\frac{63}{2}} = 1 + \frac{\frac{66-48}{4}}{\frac{63}{2}} = 1 + \frac{18/4}{63/2} = 1 + \frac{18}{126} = 1 + \frac{1}{7} = \frac{8}{7}$$

For $j = 5/2$:
$$g_J = 1 + \frac{\frac{5}{2}\cdot\frac{7}{2} + \frac{3}{4} - 12}{2\cdot\frac{5}{2}\cdot\frac{7}{2}} = 1 + \frac{\frac{35}{4} + \frac{3}{4} - 12}{\frac{35}{2}} = 1 + \frac{\frac{38-48}{4}}{\frac{35}{2}} = 1 - \frac{10/4}{35/2} = 1 - \frac{10}{70} = 1 - \frac{1}{7} = \frac{6}{7}$$

**(c)** Energy shifts: $\Delta E = g_J \mu_B m_j B$

For $j = 7/2$, $m_j = -7/2, ..., 7/2$: shifts are $\frac{8}{7}\mu_B B \times m_j$

For $j = 5/2$, $m_j = -5/2, ..., 5/2$: shifts are $\frac{6}{7}\mu_B B \times m_j$

---

## Part E: Selection Rules

### Solution 23

Electric dipole selection rules: $\Delta l = \pm 1$, $\Delta j = 0, \pm 1$ (not $0 \to 0$), $\Delta s = 0$

**(a)** $^2S_{1/2} \to {}^2P_{1/2}$: $\Delta l = +1$, $\Delta j = 0$, $\Delta s = 0$ ✓ **Allowed**

**(b)** $^2S_{1/2} \to {}^2P_{3/2}$: $\Delta l = +1$, $\Delta j = +1$, $\Delta s = 0$ ✓ **Allowed**

**(c)** $^2S_{1/2} \to {}^2D_{3/2}$: $\Delta l = +2$ ✗ **Forbidden**

**(d)** $^2P_{1/2} \to {}^2P_{3/2}$: $\Delta l = 0$ ✗ **Forbidden**

**(e)** $^1S_0 \to {}^1S_0$: $\Delta l = 0$ ✗, also $j = 0 \to j' = 0$ ✗ **Forbidden**

---

### Solution 26

For $np^2$ (two equivalent $p$ electrons):

**(a)** Each electron has $l = 1$. Total $L = 0, 1, 2$

**(b)** Each electron has $s = 1/2$. Total $S = 0, 1$

**(c)** Possible terms (ignoring Pauli):
- $L=0$, $S=0$: $^1S_0$
- $L=0$, $S=1$: $^3S_1$
- $L=1$, $S=0$: $^1P_1$
- $L=1$, $S=1$: $^3P_0$, $^3P_1$, $^3P_2$
- $L=2$, $S=0$: $^1D_2$
- $L=2$, $S=1$: $^3D_1$, $^3D_2$, $^3D_3$

**(d)** For equivalent electrons, must have antisymmetric total wave function.

Spatial part: symmetric for even $L$ (0, 2), antisymmetric for odd $L$ (1)
Spin part: antisymmetric for $S=0$ (singlet), symmetric for $S=1$ (triplet)

Allowed combinations (spatial × spin = antisymmetric):
- $L$ even + $S=0$: symmetric × antisymmetric = antisymmetric ✓
- $L$ odd + $S=1$: antisymmetric × symmetric = antisymmetric ✓

**Allowed terms:** $^1S_0$, $^3P_0$, $^3P_1$, $^3P_2$, $^1D_2$

---

### Solution 27

Configuration: 3d4s (one d, one s electron)

**(a)** $l_1 = 2$, $l_2 = 0$: $L = 2$ only (D state)

$s_1 = s_2 = 1/2$: $S = 0, 1$

Terms:
- $S=0$: $^1D_2$ (one state)
- $S=1$: $^3D_1$, $^3D_2$, $^3D_3$ (three states)

**(b)** Hund's rules:
1. Max $S$: $S = 1$ (triplet)
2. Max $L$: $L = 2$ (D)
3. Less than half-filled d: $J = |L-S| = 1$

**Ground term:** $^3D_1$

**(c)** With spin-orbit coupling, $^3D$ splits into $J = 1, 2, 3$

Order depends on sign of spin-orbit constant. For normal ordering (A > 0, less than half-filled):
$^3D_1 < {}^3D_2 < {}^3D_3$

**(d)** Landé g-factor for $^3D_1$ ($L=2$, $S=1$, $J=1$):

$$g_J = 1 + \frac{1(2) + 1(2) - 2(3)}{2(1)(2)} = 1 + \frac{2 + 2 - 6}{4} = 1 - \frac{1}{2} = \frac{1}{2}$$

**(e)** For $J = 1$: $m_J = -1, 0, 1$

**Three Zeeman sublevels**

---

**End of Solutions**
