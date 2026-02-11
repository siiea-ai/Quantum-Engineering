# Week 151: Angular Momentum Addition - Oral Exam Practice

## Introduction

Angular momentum addition is a core topic that appears on almost every physics qualifying exam. Be prepared to derive CG coefficients on a whiteboard and explain spin-orbit coupling.

---

## Question 1: Angular Momentum Addition Basics

### Initial Question
"Explain how to add two angular momenta in quantum mechanics."

### Suggested Response Framework

**Opening:**
"When combining two angular momentum systems with quantum numbers $j_1$ and $j_2$, the total angular momentum $\mathbf{J} = \mathbf{J}_1 + \mathbf{J}_2$ has allowed values given by the triangle rule: $j = |j_1 - j_2|, ..., j_1 + j_2$."

**Key Points:**
1. The combined Hilbert space is the tensor product
2. Two natural bases: uncoupled $|j_1,m_1;j_2,m_2\rangle$ and coupled $|j,m\rangle$
3. Selection rule: $m = m_1 + m_2$
4. Clebsch-Gordan coefficients connect the bases

**Dimension Check:**
"The dimension $(2j_1+1)(2j_2+1)$ equals $\sum_j (2j+1)$, confirming both bases span the same space."

### Follow-up Questions

**Q: "Derive the allowed j values for two spin-1 particles."**

A: "$j_1 = j_2 = 1$, so $j = 0, 1, 2$. Check: $3 \times 3 = 9 = 1 + 3 + 5$. Physically, two spin-1 particles can combine to give spin-0 (singlet), spin-1 (triplet), or spin-2 (quintet)."

**Q: "What happens for three spin-1/2 particles?"**

A: "First combine two to get $j_{12} = 0, 1$. Then add the third: $0 \otimes 1/2 = 1/2$ and $1 \otimes 1/2 = 1/2, 3/2$. Result: $j = 1/2$ (twice) and $j = 3/2$ (once). Total: $2 + 2 + 4 = 8 = 2^3$."

---

## Question 2: Clebsch-Gordan Coefficients

### Initial Question
"What are Clebsch-Gordan coefficients and how do you calculate them?"

### Suggested Response Framework

**Definition:**
"CG coefficients are the expansion coefficients relating coupled and uncoupled angular momentum bases:

$$|j,m\rangle = \sum_{m_1,m_2} \langle j_1,m_1;j_2,m_2|j,m\rangle |j_1,m_1\rangle|j_2,m_2\rangle$$

They are non-zero only when $m = m_1 + m_2$ and the triangle rule is satisfied."

**Calculation Methods:**
1. **Highest weight:** Start with $|j_{max}, j_{max}\rangle = |j_1,j_1\rangle|j_2,j_2\rangle$
2. **Lowering operator:** Apply $J_- = J_{1-} + J_{2-}$
3. **Orthogonality:** States with same $m$ but different $j$ are orthogonal

**Example:**
"For two spin-1/2: $|1,1\rangle = |\uparrow\uparrow\rangle$, then lowering gives $|1,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$. The singlet is the orthogonal combination."

### Follow-up Questions

**Q: "What are the symmetry properties of CG coefficients?"**

A: "Swapping $j_1 \leftrightarrow j_2$ introduces a phase $(-1)^{j_1+j_2-j}$. Also, flipping all $m$ values: $\langle j_1,m_1;j_2,m_2|j,m\rangle = (-1)^{j_1+j_2-j}\langle j_1,-m_1;j_2,-m_2|j,-m\rangle$."

**Q: "When is a CG coefficient equal to 1?"**

A: "When the uncoupled state uniquely determines the coupled state. For example, $\langle j_1,j_1;j_2,j_2|j_1+j_2,j_1+j_2\rangle = 1$ because there's only one state with maximum $m$."

---

## Question 3: Spin-Orbit Coupling

### Initial Question
"Explain spin-orbit coupling and its effect on atomic energy levels."

### Suggested Response Framework

**Physical Origin:**
"An electron moving through the nuclear electric field experiences a magnetic field in its rest frame, which couples to its spin. The Hamiltonian is:

$$H_{SO} = \xi(r)\mathbf{L}\cdot\mathbf{S}$$"

**Key Operator Identity:**
"Using $\mathbf{J} = \mathbf{L} + \mathbf{S}$:

$$\mathbf{L}\cdot\mathbf{S} = \frac{1}{2}(J^2 - L^2 - S^2)$$

So in the coupled basis:
$$\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$"

**Effect on Energy:**
"For an electron ($s = 1/2$), states with $j = l + 1/2$ have higher energy than $j = l - 1/2$ (for normal ordering). This splits each $l > 0$ level into a doublet."

### Follow-up Questions

**Q: "Calculate the fine structure splitting for a 2p electron."**

A: "For $l = 1$, $s = 1/2$: $j = 1/2, 3/2$.
- $^2P_{3/2}$: $\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}[\frac{15}{4} - 2 - \frac{3}{4}] = \frac{\hbar^2}{2}$
- $^2P_{1/2}$: $\langle\mathbf{L}\cdot\mathbf{S}\rangle = \frac{\hbar^2}{2}[\frac{3}{4} - 2 - \frac{3}{4}] = -\hbar^2$

Splitting: $\Delta E = \frac{3\hbar^2}{2}\langle\xi\rangle$"

**Q: "What are the good quantum numbers in spin-orbit coupling?"**

A: "$n$, $l$, $s$, $j$, $m_j$. Importantly, $m_l$ and $m_s$ are NOT good quantum numbers because $[H_{SO}, L_z] \neq 0$ and $[H_{SO}, S_z] \neq 0$."

---

## Question 4: Selection Rules

### Initial Question
"Derive the selection rules for electric dipole transitions."

### Suggested Response Framework

**Starting Point:**
"The transition rate is proportional to $|\langle f|\mathbf{r}|i\rangle|^2$. Angular momentum conservation requires the photon's angular momentum to be absorbed."

**Key Rules:**
1. **Parity:** $\mathbf{r}$ is odd, so $\Delta l = \pm 1$
2. **Angular momentum:** Photon carries spin 1, so $\Delta j = 0, \pm 1$ (but $0 \not\to 0$)
3. **$m_j$:** $\Delta m_j = 0, \pm 1$ (corresponding to $\pi$ and $\sigma^{\pm}$ polarizations)
4. **Spin:** $\Delta s = 0$ (dipole doesn't couple to spin)

**Triangle Rule:**
"The angular momenta of initial, final, and photon must satisfy the triangle rule."

### Follow-up Questions

**Q: "Why is $j=0 \to j'=0$ forbidden?"**

A: "A photon carries angular momentum 1. You can't combine $j_i = 0$ with 1 to get $j_f = 0$ because the triangle rule requires $|0-1| \leq 0 \leq 0+1$, which fails at the lower bound."

**Q: "How many spectral lines in the sodium D transition?"**

A: "The D-lines are $3p \to 3s$. Initial: $^2P_{1/2}$, $^2P_{3/2}$. Final: $^2S_{1/2}$. Allowed transitions: $^2P_{1/2} \to {}^2S_{1/2}$ ($\Delta j = 0$) and $^2P_{3/2} \to {}^2S_{1/2}$ ($\Delta j = -1$). Two lines: the famous D1 and D2 lines."

---

## Question 5: Term Symbols

### Initial Question
"Explain term symbol notation and Hund's rules."

### Suggested Response Framework

**Notation:**
"A term symbol $^{2S+1}L_J$ encodes:
- $2S+1$: spin multiplicity
- $L$: total orbital angular momentum (S, P, D, F, ...)
- $J$: total angular momentum"

**Hund's Rules (for ground state):**
1. Maximize $S$ (exchange interaction lowers energy)
2. Maximize $L$ consistent with rule 1
3. $J = |L-S|$ for less than half-filled; $J = L+S$ for more than half-filled

**Example:**
"For nitrogen (2p³): Three $p$ electrons each with $l=1$, $s=1/2$.
- Max $S = 3/2$ (all parallel)
- Max $L = 0$ (by Pauli: $m_l = 1, 0, -1$)
- Less than half: $J = |0 - 3/2| = 3/2$

Ground state: $^4S_{3/2}$"

### Follow-up Questions

**Q: "Derive the allowed terms for the $np^2$ configuration."**

A: "For two equivalent $p$ electrons: $L = 0, 1, 2$ and $S = 0, 1$. But Pauli exclusion restricts combinations. Symmetric spatial (even $L$) requires antisymmetric spin ($S=0$). Antisymmetric spatial (odd $L$) requires symmetric spin ($S=1$). Allowed: $^1S_0$, $^3P_{0,1,2}$, $^1D_2$."

---

## Quick-Fire Questions

1. **What are the allowed $j$ values for $l=2$, $s=1/2$?**
   - $j = 3/2, 5/2$

2. **How many states in the triplet of two spin-1/2 particles?**
   - 3 states ($m = -1, 0, 1$)

3. **What is $\langle\mathbf{L}\cdot\mathbf{S}\rangle$ for an $s$ electron?**
   - Zero (since $l = 0$)

4. **Is $^3D_3 \to {}^3P_2$ allowed?**
   - Yes: $\Delta l = -1$, $\Delta j = -1$, $\Delta s = 0$

5. **What's the ground term of carbon (2p²)?**
   - $^3P_0$ (by Hund's rules)

---

## Whiteboard Exercise

Practice deriving these on a whiteboard:

1. **CG coefficients for two spin-1/2 particles**
   - Start from $|1,1\rangle = |\uparrow\uparrow\rangle$
   - Apply $J_-$ to get $|1,0\rangle$
   - Find $|0,0\rangle$ by orthogonality

2. **Energy splitting from spin-orbit coupling**
   - Write $\mathbf{L}\cdot\mathbf{S}$ in terms of $J^2, L^2, S^2$
   - Calculate for specific $l$, $s$, $j$

3. **Allowed terms for a configuration**
   - List $L$, $S$ possibilities
   - Apply Pauli for equivalent electrons
   - Determine $J$ values

---

**Preparation Time:** 2-3 hours
**Key Skill:** Whiteboard derivation of CG coefficients
