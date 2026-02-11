# Day 394: Commutation Relations of Angular Momentum

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Derivation of $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$ |
| **Afternoon** | 2.5 hours | $[\hat{L}^2, \hat{L}_i] = 0$, uncertainty relations |
| **Evening** | 1.5 hours | Computational lab: matrix representations |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** $[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$ from the canonical commutation relations
2. **Express** the general commutation relation using the Levi-Civita symbol
3. **Prove** that $\hat{L}^2$ commutes with all angular momentum components
4. **Explain** why we can only have simultaneous eigenstates of $\hat{L}^2$ and one component
5. **Apply** the angular momentum uncertainty relations
6. **Construct** matrix representations of angular momentum operators

---

## 1. The Fundamental Commutation Relations

### 1.1 Starting Point: Canonical Commutation Relations

We begin with the fundamental position-momentum commutation relations:

$$[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}$$

where $\delta_{ij}$ is the Kronecker delta. Explicitly:

$$[\hat{x}, \hat{p}_x] = i\hbar, \quad [\hat{y}, \hat{p}_y] = i\hbar, \quad [\hat{z}, \hat{p}_z] = i\hbar$$

All other combinations vanish:

$$[\hat{x}, \hat{p}_y] = [\hat{x}, \hat{y}] = [\hat{p}_x, \hat{p}_y] = 0, \quad \text{etc.}$$

### 1.2 Useful Commutator Identities

Before proceeding, we establish key identities:

**Identity 1: Linearity**
$$[\hat{A}, \hat{B} + \hat{C}] = [\hat{A}, \hat{B}] + [\hat{A}, \hat{C}]$$

**Identity 2: Product Rule**
$$[\hat{A}, \hat{B}\hat{C}] = \hat{B}[\hat{A}, \hat{C}] + [\hat{A}, \hat{B}]\hat{C}$$

**Identity 3: Alternative Product Rule**
$$[\hat{A}\hat{B}, \hat{C}] = \hat{A}[\hat{B}, \hat{C}] + [\hat{A}, \hat{C}]\hat{B}$$

**Identity 4: Antisymmetry**
$$[\hat{A}, \hat{B}] = -[\hat{B}, \hat{A}]$$

### 1.3 Derivation of $[\hat{L}_x, \hat{L}_y]$

**Goal:** Prove that $[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$.

**Step 1:** Write out the operators explicitly:
$$\hat{L}_x = \hat{y}\hat{p}_z - \hat{z}\hat{p}_y$$
$$\hat{L}_y = \hat{z}\hat{p}_x - \hat{x}\hat{p}_z$$

**Step 2:** Expand the commutator:
$$[\hat{L}_x, \hat{L}_y] = [\hat{y}\hat{p}_z - \hat{z}\hat{p}_y, \hat{z}\hat{p}_x - \hat{x}\hat{p}_z]$$

Using linearity:
$$= [\hat{y}\hat{p}_z, \hat{z}\hat{p}_x] - [\hat{y}\hat{p}_z, \hat{x}\hat{p}_z] - [\hat{z}\hat{p}_y, \hat{z}\hat{p}_x] + [\hat{z}\hat{p}_y, \hat{x}\hat{p}_z]$$

**Step 3:** Evaluate each term using the product rule.

**Term 1:** $[\hat{y}\hat{p}_z, \hat{z}\hat{p}_x]$
$$= \hat{y}[\hat{p}_z, \hat{z}\hat{p}_x] + [\hat{y}, \hat{z}\hat{p}_x]\hat{p}_z$$
$$= \hat{y}\{\hat{z}[\hat{p}_z, \hat{p}_x] + [\hat{p}_z, \hat{z}]\hat{p}_x\} + \{\hat{z}[\hat{y}, \hat{p}_x] + [\hat{y}, \hat{z}]\hat{p}_x\}\hat{p}_z$$

Using $[\hat{p}_z, \hat{p}_x] = 0$, $[\hat{p}_z, \hat{z}] = -i\hbar$, $[\hat{y}, \hat{p}_x] = 0$, $[\hat{y}, \hat{z}] = 0$:
$$= \hat{y}(-i\hbar)\hat{p}_x = -i\hbar\hat{y}\hat{p}_x$$

**Term 2:** $[\hat{y}\hat{p}_z, \hat{x}\hat{p}_z]$
$$= \hat{y}[\hat{p}_z, \hat{x}\hat{p}_z] + [\hat{y}, \hat{x}\hat{p}_z]\hat{p}_z$$
$$= \hat{y}\{\hat{x}[\hat{p}_z, \hat{p}_z] + [\hat{p}_z, \hat{x}]\hat{p}_z\} + \{\hat{x}[\hat{y}, \hat{p}_z] + [\hat{y}, \hat{x}]\hat{p}_z\}\hat{p}_z$$

Using $[\hat{p}_z, \hat{p}_z] = 0$, $[\hat{p}_z, \hat{x}] = 0$, $[\hat{y}, \hat{p}_z] = 0$, $[\hat{y}, \hat{x}] = 0$:
$$= 0$$

**Term 3:** $[\hat{z}\hat{p}_y, \hat{z}\hat{p}_x]$
$$= \hat{z}[\hat{p}_y, \hat{z}\hat{p}_x] + [\hat{z}, \hat{z}\hat{p}_x]\hat{p}_y$$

Since $[\hat{z}, \hat{z}] = 0$ and $[\hat{p}_y, \hat{z}] = 0$, $[\hat{p}_y, \hat{p}_x] = 0$:
$$= 0$$

**Term 4:** $[\hat{z}\hat{p}_y, \hat{x}\hat{p}_z]$
$$= \hat{z}[\hat{p}_y, \hat{x}\hat{p}_z] + [\hat{z}, \hat{x}\hat{p}_z]\hat{p}_y$$
$$= \hat{z}\{[\hat{p}_y, \hat{x}]\hat{p}_z + \hat{x}[\hat{p}_y, \hat{p}_z]\} + \{[\hat{z}, \hat{x}]\hat{p}_z + \hat{x}[\hat{z}, \hat{p}_z]\}\hat{p}_y$$

Using $[\hat{p}_y, \hat{x}] = 0$, $[\hat{p}_y, \hat{p}_z] = 0$, $[\hat{z}, \hat{x}] = 0$, $[\hat{z}, \hat{p}_z] = i\hbar$:
$$= \hat{x}(i\hbar)\hat{p}_y = i\hbar\hat{x}\hat{p}_y$$

**Step 4:** Combine all terms:
$$[\hat{L}_x, \hat{L}_y] = -i\hbar\hat{y}\hat{p}_x - 0 - 0 + i\hbar\hat{x}\hat{p}_y$$
$$= i\hbar(\hat{x}\hat{p}_y - \hat{y}\hat{p}_x)$$

$$\boxed{[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z}$$

---

## 2. General Commutation Relation

### 2.1 Cyclic Permutations

By cyclic permutation of indices $(x \to y \to z \to x)$:

$$\boxed{[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z}$$
$$\boxed{[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x}$$
$$\boxed{[\hat{L}_z, \hat{L}_x] = i\hbar\hat{L}_y}$$

### 2.2 Compact Notation with Levi-Civita Symbol

All three relations can be written as:

$$\boxed{[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k}$$

where summation over $k$ is implied, and $\epsilon_{ijk}$ is the **Levi-Civita symbol**:

$$\epsilon_{ijk} = \begin{cases}
+1 & \text{if } (i,j,k) \text{ is an even permutation of } (1,2,3) \\
-1 & \text{if } (i,j,k) \text{ is an odd permutation of } (1,2,3) \\
0 & \text{if any two indices are equal}
\end{cases}$$

### 2.3 The Angular Momentum Algebra

The commutation relations $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$ define the **Lie algebra** $\mathfrak{so}(3)$ of the rotation group SO(3).

**Key insight:** This algebra is isomorphic to $\mathfrak{su}(2)$, the algebra of the special unitary group SU(2), which is crucial for understanding spin.

The **structure constants** of the algebra are:

$$f_{ijk} = \epsilon_{ijk}$$

---

## 3. Commutation with $\hat{L}^2$

### 3.1 Definition of $\hat{L}^2$

The total angular momentum squared is:

$$\hat{L}^2 = \hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2$$

### 3.2 Proof that $[\hat{L}^2, \hat{L}_z] = 0$

**Goal:** Show that $\hat{L}^2$ commutes with $\hat{L}_z$.

$$[\hat{L}^2, \hat{L}_z] = [\hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2, \hat{L}_z]$$

Using linearity:
$$= [\hat{L}_x^2, \hat{L}_z] + [\hat{L}_y^2, \hat{L}_z] + [\hat{L}_z^2, \hat{L}_z]$$

The last term vanishes since any operator commutes with itself:
$$[\hat{L}_z^2, \hat{L}_z] = 0$$

For the first term, use the identity $[\hat{A}^2, \hat{B}] = \hat{A}[\hat{A}, \hat{B}] + [\hat{A}, \hat{B}]\hat{A}$:

$$[\hat{L}_x^2, \hat{L}_z] = \hat{L}_x[\hat{L}_x, \hat{L}_z] + [\hat{L}_x, \hat{L}_z]\hat{L}_x$$

Using $[\hat{L}_x, \hat{L}_z] = -[\hat{L}_z, \hat{L}_x] = -i\hbar\hat{L}_y$:
$$= \hat{L}_x(-i\hbar\hat{L}_y) + (-i\hbar\hat{L}_y)\hat{L}_x = -i\hbar(\hat{L}_x\hat{L}_y + \hat{L}_y\hat{L}_x)$$

Similarly:
$$[\hat{L}_y^2, \hat{L}_z] = \hat{L}_y[\hat{L}_y, \hat{L}_z] + [\hat{L}_y, \hat{L}_z]\hat{L}_y$$

Using $[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x$:
$$= \hat{L}_y(i\hbar\hat{L}_x) + (i\hbar\hat{L}_x)\hat{L}_y = i\hbar(\hat{L}_y\hat{L}_x + \hat{L}_x\hat{L}_y)$$

**Combining:**
$$[\hat{L}^2, \hat{L}_z] = -i\hbar(\hat{L}_x\hat{L}_y + \hat{L}_y\hat{L}_x) + i\hbar(\hat{L}_y\hat{L}_x + \hat{L}_x\hat{L}_y) = 0$$

$$\boxed{[\hat{L}^2, \hat{L}_z] = 0}$$

### 3.3 General Result

By symmetry of the derivation:

$$\boxed{[\hat{L}^2, \hat{L}_i] = 0 \quad \text{for all } i = x, y, z}$$

**Physical meaning:** $\hat{L}^2$ commutes with all components of angular momentum. This is the **Casimir operator** of the algebra.

---

## 4. Simultaneous Eigenstates

### 4.1 Compatible Observables

Two observables $\hat{A}$ and $\hat{B}$ are **compatible** (can be simultaneously measured) if:

$$[\hat{A}, \hat{B}] = 0$$

### 4.2 What Can We Measure Simultaneously?

From our commutation relations:

| Operators | Commutator | Compatible? |
|-----------|------------|-------------|
| $\hat{L}^2$, $\hat{L}_z$ | 0 | Yes |
| $\hat{L}^2$, $\hat{L}_x$ | 0 | Yes |
| $\hat{L}^2$, $\hat{L}_y$ | 0 | Yes |
| $\hat{L}_x$, $\hat{L}_y$ | $i\hbar\hat{L}_z$ | No |
| $\hat{L}_y$, $\hat{L}_z$ | $i\hbar\hat{L}_x$ | No |
| $\hat{L}_z$, $\hat{L}_x$ | $i\hbar\hat{L}_y$ | No |

**Conclusion:** We can find simultaneous eigenstates of:
- $\hat{L}^2$ and $\hat{L}_z$ (standard choice)
- $\hat{L}^2$ and $\hat{L}_x$
- $\hat{L}^2$ and $\hat{L}_y$

But NOT eigenstates of any two different components simultaneously.

### 4.3 Standard Notation

We denote simultaneous eigenstates of $\hat{L}^2$ and $\hat{L}_z$ as $|\ell, m\rangle$:

$$\hat{L}^2|\ell, m\rangle = \hbar^2\ell(\ell+1)|\ell, m\rangle$$
$$\hat{L}_z|\ell, m\rangle = \hbar m|\ell, m\rangle$$

(The eigenvalue structure will be derived on Day 396.)

---

## 5. Uncertainty Relations

### 5.1 General Uncertainty Principle

For any two observables with $[\hat{A}, \hat{B}] = i\hat{C}$:

$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle\hat{C}\rangle|$$

### 5.2 Angular Momentum Uncertainty Relations

From $[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$:

$$\boxed{\Delta L_x \cdot \Delta L_y \geq \frac{\hbar}{2}|\langle\hat{L}_z\rangle|}$$

Similarly:
$$\Delta L_y \cdot \Delta L_z \geq \frac{\hbar}{2}|\langle\hat{L}_x\rangle|$$
$$\Delta L_z \cdot \Delta L_x \geq \frac{\hbar}{2}|\langle\hat{L}_y\rangle|$$

### 5.3 Physical Interpretation

**Case 1: State with $\langle\hat{L}_z\rangle = m\hbar \neq 0$**
$$\Delta L_x \cdot \Delta L_y \geq \frac{\hbar^2}{2}|m|$$

The uncertainty product is bounded below by a nonzero value.

**Case 2: State with $\langle\hat{L}_z\rangle = 0$**

The uncertainty bound is zero, but this does NOT mean we can know $L_x$ and $L_y$ precisely. The general uncertainty relation is:

$$\Delta L_x \cdot \Delta L_y \geq \frac{1}{2}|\langle[\hat{L}_x, \hat{L}_y]\rangle| = \frac{\hbar}{2}|\langle\hat{L}_z\rangle|$$

### 5.4 Minimum Uncertainty States

For an eigenstate $|\ell, m\rangle$:
- $\Delta L_z = 0$ (eigenstate of $\hat{L}_z$)
- $\langle\hat{L}_x\rangle = \langle\hat{L}_y\rangle = 0$ (by symmetry)
- $\Delta L_x = \Delta L_y \neq 0$

The uncertainty relation $\Delta L_y \cdot \Delta L_z \geq \frac{\hbar}{2}|\langle\hat{L}_x\rangle| = 0$ is trivially satisfied.

---

## 6. Quantum Computing Connection

### 6.1 Pauli Matrices and Angular Momentum

The Pauli matrices satisfy:

$$[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$$

Comparing with $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$, we identify:

$$\hat{S}_i = \frac{\hbar}{2}\sigma_i$$

These are the **spin-1/2** angular momentum operators (Week 58).

### 6.2 SU(2) and Qubit Rotations

The rotation operator on a qubit is:

$$\hat{R}_{\mathbf{n}}(\theta) = e^{-i\theta\mathbf{n}\cdot\boldsymbol{\sigma}/2}$$

The commutation relations ensure that rotations compose correctly:

$$\hat{R}_x(\alpha)\hat{R}_y(\beta) \neq \hat{R}_y(\beta)\hat{R}_x(\alpha)$$

This non-commutativity is the geometric origin of quantum gate sequences mattering.

---

## 7. Worked Examples

### Example 1: Verify $[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x$

**Problem:** Derive $[\hat{L}_y, \hat{L}_z]$ directly from the definitions.

**Solution:**

$$\hat{L}_y = \hat{z}\hat{p}_x - \hat{x}\hat{p}_z$$
$$\hat{L}_z = \hat{x}\hat{p}_y - \hat{y}\hat{p}_x$$

$$[\hat{L}_y, \hat{L}_z] = [\hat{z}\hat{p}_x - \hat{x}\hat{p}_z, \hat{x}\hat{p}_y - \hat{y}\hat{p}_x]$$

Expanding:
$$= [\hat{z}\hat{p}_x, \hat{x}\hat{p}_y] - [\hat{z}\hat{p}_x, \hat{y}\hat{p}_x] - [\hat{x}\hat{p}_z, \hat{x}\hat{p}_y] + [\hat{x}\hat{p}_z, \hat{y}\hat{p}_x]$$

**Term 1:** $[\hat{z}\hat{p}_x, \hat{x}\hat{p}_y]$
$$= \hat{z}[\hat{p}_x, \hat{x}]\hat{p}_y = \hat{z}(-i\hbar)\hat{p}_y = -i\hbar\hat{z}\hat{p}_y$$

**Term 2:** $[\hat{z}\hat{p}_x, \hat{y}\hat{p}_x] = 0$ (no non-commuting pairs)

**Term 3:** $[\hat{x}\hat{p}_z, \hat{x}\hat{p}_y] = 0$ (no non-commuting pairs)

**Term 4:** $[\hat{x}\hat{p}_z, \hat{y}\hat{p}_x]$
$$= \hat{y}[\hat{x}, \hat{p}_x]\hat{p}_z = \hat{y}(i\hbar)\hat{p}_z = i\hbar\hat{y}\hat{p}_z$$

Wait, let me redo this more carefully:
$$[\hat{x}\hat{p}_z, \hat{y}\hat{p}_x] = \hat{x}[\hat{p}_z, \hat{y}\hat{p}_x] + [\hat{x}, \hat{y}\hat{p}_x]\hat{p}_z$$
$$= \hat{x}\cdot 0 + \hat{y}[\hat{x}, \hat{p}_x]\hat{p}_z = i\hbar\hat{y}\hat{p}_z$$

**Combining:**
$$[\hat{L}_y, \hat{L}_z] = -i\hbar\hat{z}\hat{p}_y + i\hbar\hat{y}\hat{p}_z = i\hbar(\hat{y}\hat{p}_z - \hat{z}\hat{p}_y)$$

$$\boxed{[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x} \quad \checkmark$$

---

### Example 2: Compute $\langle\hat{L}_x\rangle$ for an Eigenstate of $\hat{L}_z$

**Problem:** If $|\psi\rangle = |\ell, m\rangle$ is an eigenstate of $\hat{L}_z$ with eigenvalue $m\hbar$, find $\langle\hat{L}_x\rangle$.

**Solution:**

Consider the commutator expectation value:
$$\langle[\hat{L}_z, \hat{L}_x]\rangle = \langle\hat{L}_z\hat{L}_x\rangle - \langle\hat{L}_x\hat{L}_z\rangle$$

Since $|\ell, m\rangle$ is an eigenstate of $\hat{L}_z$:
$$\hat{L}_z|\ell, m\rangle = m\hbar|\ell, m\rangle$$
$$\langle\ell, m|\hat{L}_z = m\hbar\langle\ell, m|$$

Therefore:
$$\langle\hat{L}_z\hat{L}_x\rangle = m\hbar\langle\hat{L}_x\rangle$$
$$\langle\hat{L}_x\hat{L}_z\rangle = \langle\hat{L}_x\rangle m\hbar$$

So:
$$\langle[\hat{L}_z, \hat{L}_x]\rangle = m\hbar\langle\hat{L}_x\rangle - m\hbar\langle\hat{L}_x\rangle = 0$$

But we also know:
$$[\hat{L}_z, \hat{L}_x] = -i\hbar\hat{L}_y$$

Therefore:
$$\langle[\hat{L}_z, \hat{L}_x]\rangle = -i\hbar\langle\hat{L}_y\rangle = 0$$

This gives $\langle\hat{L}_y\rangle = 0$, not $\langle\hat{L}_x\rangle$ directly.

**Alternative approach:** Use the rotation symmetry. The state $|\ell, m\rangle$ is axially symmetric about $z$. Under rotation about $z$ by angle $\phi$:
$$\hat{L}_x \to \hat{L}_x\cos\phi - \hat{L}_y\sin\phi$$

Since $|\ell, m\rangle$ has azimuthal symmetry (it's an eigenstate of $\hat{L}_z$), the expectation value must be invariant under $z$-rotations. This requires:

$$\boxed{\langle\hat{L}_x\rangle = \langle\hat{L}_y\rangle = 0}$$

---

### Example 3: Uncertainty Product for $|\ell=1, m=1\rangle$

**Problem:** Calculate $\Delta L_x \cdot \Delta L_y$ for the state $|1, 1\rangle$.

**Solution:**

From quantum mechanics of angular momentum (detailed in Day 396):

For $|\ell, m\rangle$:
$$\langle\hat{L}^2\rangle = \hbar^2\ell(\ell+1)$$
$$\langle\hat{L}_z^2\rangle = \hbar^2 m^2$$
$$\langle\hat{L}_x^2\rangle = \langle\hat{L}_y^2\rangle = \frac{1}{2}(\langle\hat{L}^2\rangle - \langle\hat{L}_z^2\rangle)$$

For $|1, 1\rangle$:
$$\langle\hat{L}^2\rangle = 2\hbar^2$$
$$\langle\hat{L}_z^2\rangle = \hbar^2$$
$$\langle\hat{L}_x^2\rangle = \langle\hat{L}_y^2\rangle = \frac{1}{2}(2\hbar^2 - \hbar^2) = \frac{\hbar^2}{2}$$

Since $\langle\hat{L}_x\rangle = \langle\hat{L}_y\rangle = 0$:
$$(\Delta L_x)^2 = \langle\hat{L}_x^2\rangle - \langle\hat{L}_x\rangle^2 = \frac{\hbar^2}{2}$$
$$\Delta L_x = \Delta L_y = \frac{\hbar}{\sqrt{2}}$$

Therefore:
$$\boxed{\Delta L_x \cdot \Delta L_y = \frac{\hbar^2}{2}}$$

**Check against uncertainty relation:**
$$\Delta L_x \cdot \Delta L_y \geq \frac{\hbar}{2}|\langle\hat{L}_z\rangle| = \frac{\hbar}{2} \cdot \hbar = \frac{\hbar^2}{2} \quad \checkmark$$

The state $|1, 1\rangle$ saturates the uncertainty bound!

---

## 8. Practice Problems

### Level 1: Direct Application

1. **Problem 1.1:** Using the commutation relations, compute $[\hat{L}_z, \hat{L}_y]$.

2. **Problem 1.2:** Verify that $[\hat{L}_z^2, \hat{L}_z] = 0$ by explicit calculation.

3. **Problem 1.3:** Calculate $[\hat{L}^2, \hat{L}_x]$ and show it equals zero.

### Level 2: Intermediate

4. **Problem 2.1:** Prove the identity $[\hat{A}^2, \hat{B}] = \hat{A}[\hat{A}, \hat{B}] + [\hat{A}, \hat{B}]\hat{A}$.

5. **Problem 2.2:** Show that $[\hat{L}^2, \hat{\mathbf{L}}] = 0$ (vector commutator).

6. **Problem 2.3:** Calculate the uncertainty product $\Delta L_y \cdot \Delta L_z$ for the state $|2, 0\rangle$.

### Level 3: Challenging

7. **Problem 3.1:** Prove that $[\hat{L}_i, \hat{p}^2] = 0$ where $\hat{p}^2 = \hat{p}_x^2 + \hat{p}_y^2 + \hat{p}_z^2$.

8. **Problem 3.2:** Show that for a central potential $V(r)$, $[\hat{L}_i, \hat{H}] = 0$.

9. **Problem 3.3:** Derive the commutation relation $[\hat{L}_i, \hat{L}_j\hat{L}_k]$ in terms of angular momentum operators.

---

## 9. Computational Lab: Matrix Representations

```python
"""
Day 394 Computational Lab: Angular Momentum Commutation Relations
=================================================================
Matrix representations and verification of commutation relations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Set up publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 10),
    'axes.labelsize': 14
})

# =============================================================================
# Part 1: Matrix Representations of Angular Momentum
# =============================================================================

def angular_momentum_matrices(l):
    """
    Construct matrix representations of Lx, Ly, Lz for angular momentum l.

    The basis states are |l, m> for m = l, l-1, ..., -l.

    Parameters:
    -----------
    l : int or half-int
        Angular momentum quantum number

    Returns:
    --------
    Lx, Ly, Lz : numpy arrays
        (2l+1) x (2l+1) matrices
    """
    dim = int(2*l + 1)
    m_values = np.arange(l, -l-1, -1)  # m = l, l-1, ..., -l

    # Lz is diagonal in this basis
    Lz = np.diag(m_values).astype(complex)

    # L+ and L- are off-diagonal
    Lplus = np.zeros((dim, dim), dtype=complex)
    Lminus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim-1):
        m = m_values[i+1]  # m value of the lower state
        # L+ raises m to m+1, so it connects |l,m> to |l,m+1>
        # Matrix element: <l,m+1|L+|l,m> = hbar * sqrt(l(l+1) - m(m+1))
        coeff = np.sqrt(l*(l+1) - m*(m+1))
        Lplus[i, i+1] = coeff  # <m+1|L+|m>

    Lminus = Lplus.conj().T

    # Lx = (L+ + L-)/2, Ly = (L+ - L-)/(2i)
    Lx = (Lplus + Lminus) / 2
    Ly = (Lplus - Lminus) / (2j)

    return Lx, Ly, Lz


def verify_commutation_relations(l):
    """
    Verify [Li, Lj] = i * epsilon_ijk * Lk for angular momentum l.
    """
    Lx, Ly, Lz = angular_momentum_matrices(l)

    print(f"\n=== Angular Momentum l = {l} ===")
    print(f"Matrix dimension: {2*l+1} x {2*l+1}")

    # Compute commutators
    comm_xy = Lx @ Ly - Ly @ Lx
    comm_yz = Ly @ Lz - Lz @ Ly
    comm_zx = Lz @ Lx - Lx @ Lz

    # Expected results (note: we're working in units where hbar = 1)
    expected_xy = 1j * Lz
    expected_yz = 1j * Lx
    expected_zx = 1j * Ly

    # Verify
    error_xy = np.max(np.abs(comm_xy - expected_xy))
    error_yz = np.max(np.abs(comm_yz - expected_yz))
    error_zx = np.max(np.abs(comm_zx - expected_zx))

    print(f"\n[Lx, Ly] = i*Lz ? Max error: {error_xy:.2e}")
    print(f"[Ly, Lz] = i*Lx ? Max error: {error_yz:.2e}")
    print(f"[Lz, Lx] = i*Ly ? Max error: {error_zx:.2e}")

    return error_xy < 1e-10 and error_yz < 1e-10 and error_zx < 1e-10


def verify_L2_commutes(l):
    """
    Verify that L^2 commutes with all components.
    """
    Lx, Ly, Lz = angular_momentum_matrices(l)
    L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz

    comm_x = L2 @ Lx - Lx @ L2
    comm_y = L2 @ Ly - Ly @ L2
    comm_z = L2 @ Lz - Lz @ L2

    error_x = np.max(np.abs(comm_x))
    error_y = np.max(np.abs(comm_y))
    error_z = np.max(np.abs(comm_z))

    print(f"\n[L^2, Lx] = 0 ? Max error: {error_x:.2e}")
    print(f"[L^2, Ly] = 0 ? Max error: {error_y:.2e}")
    print(f"[L^2, Lz] = 0 ? Max error: {error_z:.2e}")

    # Also verify L^2 eigenvalue
    eigenvalue = l * (l + 1)
    L2_expected = eigenvalue * np.eye(int(2*l+1))
    error_eigenvalue = np.max(np.abs(L2 - L2_expected))
    print(f"\nL^2 = l(l+1)*I ? (l(l+1) = {eigenvalue})")
    print(f"Max error: {error_eigenvalue:.2e}")

    return error_x < 1e-10 and error_y < 1e-10 and error_z < 1e-10


# =============================================================================
# Part 2: Visualization of Matrices
# =============================================================================

def visualize_matrices(l):
    """
    Visualize Lx, Ly, Lz matrices for angular momentum l.
    """
    Lx, Ly, Lz = angular_momentum_matrices(l)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    matrices = [Lx, Ly, Lz]
    names = ['$L_x$', '$L_y$', '$L_z$']
    dim = int(2*l + 1)
    m_labels = [f'{m}' for m in np.arange(l, -l-1, -1)]

    for idx, (mat, name) in enumerate(zip(matrices, names)):
        # Real part
        ax_re = axes[0, idx]
        im_re = ax_re.imshow(np.real(mat), cmap='RdBu_r',
                             vmin=-l, vmax=l, aspect='equal')
        ax_re.set_title(f'Re({name})')
        ax_re.set_xticks(range(dim))
        ax_re.set_yticks(range(dim))
        ax_re.set_xticklabels(m_labels)
        ax_re.set_yticklabels(m_labels)
        ax_re.set_xlabel('$m$')
        ax_re.set_ylabel("$m'$")
        plt.colorbar(im_re, ax=ax_re)

        # Add value annotations
        for i in range(dim):
            for j in range(dim):
                val = np.real(mat[i, j])
                if np.abs(val) > 0.01:
                    ax_re.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=8)

        # Imaginary part
        ax_im = axes[1, idx]
        im_im = ax_im.imshow(np.imag(mat), cmap='RdBu_r',
                             vmin=-l, vmax=l, aspect='equal')
        ax_im.set_title(f'Im({name})')
        ax_im.set_xticks(range(dim))
        ax_im.set_yticks(range(dim))
        ax_im.set_xticklabels(m_labels)
        ax_im.set_yticklabels(m_labels)
        ax_im.set_xlabel('$m$')
        ax_im.set_ylabel("$m'$")
        plt.colorbar(im_im, ax=ax_im)

        for i in range(dim):
            for j in range(dim):
                val = np.imag(mat[i, j])
                if np.abs(val) > 0.01:
                    ax_im.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=8)

    plt.suptitle(f'Angular Momentum Matrices for $\\ell = {l}$\n'
                 f'(units of $\\hbar$)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'angular_momentum_matrices_l{l}.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 3: Commutator Visualization
# =============================================================================

def visualize_commutators(l):
    """
    Visualize commutators and verify they equal the expected values.
    """
    Lx, Ly, Lz = angular_momentum_matrices(l)
    dim = int(2*l + 1)

    # Compute commutators
    comm_xy = Lx @ Ly - Ly @ Lx
    expected_xy = 1j * Lz

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # [Lx, Ly]
    ax1 = axes[0]
    im1 = ax1.imshow(np.imag(comm_xy), cmap='RdBu_r', aspect='equal')
    ax1.set_title('Im([Lx, Ly])')
    plt.colorbar(im1, ax=ax1)
    for i in range(dim):
        for j in range(dim):
            val = np.imag(comm_xy[i, j])
            if np.abs(val) > 0.01:
                ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10)

    # i*Lz
    ax2 = axes[1]
    im2 = ax2.imshow(np.imag(expected_xy), cmap='RdBu_r', aspect='equal')
    ax2.set_title('Im(i*Lz)')
    plt.colorbar(im2, ax=ax2)
    for i in range(dim):
        for j in range(dim):
            val = np.imag(expected_xy[i, j])
            if np.abs(val) > 0.01:
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10)

    # Difference
    ax3 = axes[2]
    diff = comm_xy - expected_xy
    im3 = ax3.imshow(np.abs(diff), cmap='hot', aspect='equal')
    ax3.set_title('|[Lx, Ly] - i*Lz| (should be zero)')
    plt.colorbar(im3, ax=ax3)

    plt.suptitle(f'Verification: [Lx, Ly] = i*Lz for l = {l}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'commutator_verification_l{l}.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 4: Rotation Non-commutativity
# =============================================================================

def rotation_noncommutativity():
    """
    Demonstrate that rotations about different axes don't commute.
    """
    l = 1
    Lx, Ly, Lz = angular_momentum_matrices(l)

    # Rotation angles
    angles = [np.pi/4, np.pi/3, np.pi/2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, theta in zip(axes, angles):
        # R_x(theta) * R_y(theta) vs R_y(theta) * R_x(theta)
        Rx = expm(-1j * theta * Lx)
        Ry = expm(-1j * theta * Ly)

        RxRy = Rx @ Ry
        RyRx = Ry @ Rx

        diff = RxRy - RyRx

        im = ax.imshow(np.abs(diff), cmap='hot', aspect='equal')
        ax.set_title(f'$|R_x(\\theta)R_y(\\theta) - R_y(\\theta)R_x(\\theta)|$\n'
                     f'$\\theta = {theta/np.pi:.2f}\\pi$')
        plt.colorbar(im, ax=ax)

        # Compute Frobenius norm of difference
        norm_diff = np.linalg.norm(diff, 'fro')
        ax.set_xlabel(f'Frobenius norm: {norm_diff:.4f}')

    plt.suptitle('Rotations Do Not Commute\n'
                 '(Consequence of angular momentum commutation relations)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('rotation_noncommutativity.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 5: Uncertainty Relations
# =============================================================================

def uncertainty_verification():
    """
    Verify uncertainty relations for angular momentum.
    """
    print("\n=== Uncertainty Relation Verification ===")

    for l in [1, 2, 3]:
        Lx, Ly, Lz = angular_momentum_matrices(l)
        L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz
        dim = int(2*l + 1)

        print(f"\nl = {l}:")
        print("-" * 40)

        for m_idx, m in enumerate(np.arange(l, -l-1, -1)):
            # State |l, m> is the (m_idx)-th basis vector
            state = np.zeros(dim, dtype=complex)
            state[m_idx] = 1.0

            # Expectation values
            Lx_exp = np.real(state.conj() @ Lx @ state)
            Ly_exp = np.real(state.conj() @ Ly @ state)
            Lz_exp = np.real(state.conj() @ Lz @ state)

            Lx2_exp = np.real(state.conj() @ Lx @ Lx @ state)
            Ly2_exp = np.real(state.conj() @ Ly @ Ly @ state)

            # Uncertainties
            delta_Lx = np.sqrt(Lx2_exp - Lx_exp**2)
            delta_Ly = np.sqrt(Ly2_exp - Ly_exp**2)

            # Uncertainty product
            product = delta_Lx * delta_Ly

            # Lower bound from uncertainty relation
            lower_bound = 0.5 * np.abs(Lz_exp)

            print(f"  |{l},{m:+d}>: ΔLx·ΔLy = {product:.4f}, "
                  f"(1/2)|<Lz>| = {lower_bound:.4f}, "
                  f"Satisfied: {product >= lower_bound - 1e-10}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Day 394 Lab: Angular Momentum Commutation Relations")
    print("=" * 55)

    # Verify commutation relations for l = 1, 2, 3
    print("\n" + "="*55)
    print("Part 1: Verifying Commutation Relations")
    print("="*55)
    for l in [1, 2, 3]:
        success = verify_commutation_relations(l)
        print(f"l = {l}: {'PASSED' if success else 'FAILED'}")

    # Verify L^2 commutes with all components
    print("\n" + "="*55)
    print("Part 2: Verifying [L^2, Li] = 0")
    print("="*55)
    for l in [1, 2, 3]:
        success = verify_L2_commutes(l)
        print(f"l = {l}: {'PASSED' if success else 'FAILED'}")

    # Visualizations
    print("\n" + "="*55)
    print("Part 3: Generating Visualizations")
    print("="*55)

    print("\nVisualizing matrices for l=1...")
    visualize_matrices(1)

    print("\nVisualizing commutators for l=2...")
    visualize_commutators(2)

    print("\nDemonstrating rotation non-commutativity...")
    rotation_noncommutativity()

    # Uncertainty relations
    print("\n" + "="*55)
    print("Part 4: Uncertainty Relations")
    print("="*55)
    uncertainty_verification()

    print("\n" + "="*55)
    print("Lab complete! Figures saved to current directory.")
    print("="*55)
```

---

## 10. Summary

### Key Formulas Table

| Relation | Formula |
|----------|---------|
| Fundamental commutator | $[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$ |
| General form | $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$ |
| $\hat{L}^2$ commutation | $[\hat{L}^2, \hat{L}_i] = 0$ for all $i$ |
| Uncertainty relation | $\Delta L_x \Delta L_y \geq \frac{\hbar}{2}|\langle\hat{L}_z\rangle|$ |
| Compatible observables | $\hat{L}^2$ and $\hat{L}_z$ (or any single component) |

### Main Takeaways

1. **The commutation relations** $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$ are derived from the canonical commutation relations $[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}$

2. **$\hat{L}^2$ commutes with all components**, making it the Casimir operator of the algebra

3. **Simultaneous eigenstates** exist only for $\hat{L}^2$ and one component (conventionally $\hat{L}_z$)

4. **Uncertainty relations** prevent precise knowledge of more than one component

5. **Non-commuting rotations** are a geometric consequence of the angular momentum algebra

---

## 11. Daily Checklist

- [ ] Derived $[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$ step by step
- [ ] Understood the Levi-Civita symbol notation
- [ ] Proved $[\hat{L}^2, \hat{L}_z] = 0$
- [ ] Explained why only $\hat{L}^2$ and one component can be measured simultaneously
- [ ] Calculated uncertainty products for specific states
- [ ] Ran computational lab and verified commutation relations numerically
- [ ] Solved at least 3 practice problems

---

## 12. Preview: Day 395

Tomorrow we introduce **ladder operators** $\hat{L}_\pm = \hat{L}_x \pm i\hat{L}_y$:

- They satisfy $[\hat{L}_z, \hat{L}_\pm] = \pm\hbar\hat{L}_\pm$
- They raise/lower the $m$ quantum number by 1
- They are the key to deriving the eigenvalue spectrum
- They generalize to all angular momentum algebras

---

*Day 394 of Year 1: Quantum Mechanics Core*
*Week 57: Orbital Angular Momentum*
*QSE Self-Study Curriculum*
