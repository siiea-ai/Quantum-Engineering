# Week 145: Mathematical Framework — Review Guide

## Introduction

The mathematical framework of quantum mechanics is built on the theory of Hilbert spaces and linear operators. This review guide provides a comprehensive treatment of the essential concepts tested on PhD qualifying exams. Work through this material actively—derive the key results yourself and practice explaining them as you would in an oral exam.

---

## 1. Hilbert Spaces

### 1.1 Definition and Properties

A **Hilbert space** $$\mathcal{H}$$ is a complete inner product space over the complex numbers. The key properties are:

**Inner Product Axioms:**
For vectors $$|\psi\rangle, |\phi\rangle, |\chi\rangle \in \mathcal{H}$$ and scalars $$\alpha, \beta \in \mathbb{C}$$:

1. **Conjugate Symmetry:** $$\langle \phi | \psi \rangle = \langle \psi | \phi \rangle^*$$

2. **Linearity in Second Argument:** $$\langle \phi | \alpha\psi + \beta\chi \rangle = \alpha\langle \phi | \psi \rangle + \beta\langle \phi | \chi \rangle$$

3. **Positive Definiteness:** $$\langle \psi | \psi \rangle \geq 0$$ with equality iff $$|\psi\rangle = 0$$

**Completeness:** Every Cauchy sequence in $$\mathcal{H}$$ converges to a limit in $$\mathcal{H}$$.

### 1.2 The Norm and Distance

The **norm** is defined by:
$$\| \psi \| = \sqrt{\langle \psi | \psi \rangle}$$

This induces a metric (distance function):
$$d(\psi, \phi) = \| \psi - \phi \|$$

### 1.3 Orthonormality and Bases

An **orthonormal set** $$\{|n\rangle\}$$ satisfies:
$$\langle m | n \rangle = \delta_{mn}$$

For a **complete orthonormal basis** (countable discrete spectrum):
$$\sum_n |n\rangle\langle n| = \hat{1} \quad \text{(completeness relation)}$$

Any state can be expanded:
$$|\psi\rangle = \sum_n c_n |n\rangle, \quad c_n = \langle n | \psi \rangle$$

### 1.4 Continuous Spectra

For continuous variables (e.g., position), completeness becomes:
$$\int |x\rangle\langle x| \, dx = \hat{1}$$

With normalization:
$$\langle x | x' \rangle = \delta(x - x')$$

### 1.5 The Schwarz Inequality

For any two states:
$$|\langle \phi | \psi \rangle|^2 \leq \langle \phi | \phi \rangle \langle \psi | \psi \rangle$$

with equality iff $$|\psi\rangle = \lambda |\phi\rangle$$ for some $$\lambda \in \mathbb{C}$$.

**Proof Sketch:** Consider $$|z\rangle = |\psi\rangle - \frac{\langle \phi | \psi \rangle}{\langle \phi | \phi \rangle}|\phi\rangle$$ and use $$\langle z | z \rangle \geq 0$$.

---

## 2. Dirac Notation

### 2.1 Kets, Bras, and Brackets

**Ket:** $$|\psi\rangle$$ represents a state vector in $$\mathcal{H}$$

**Bra:** $$\langle\psi|$$ represents the dual vector in $$\mathcal{H}^*$$

**Bracket:** $$\langle\phi|\psi\rangle$$ is the inner product

The correspondence is antilinear:
$$|\alpha\psi + \beta\phi\rangle \leftrightarrow \alpha^*\langle\psi| + \beta^*\langle\phi|$$

### 2.2 Outer Products

The **outer product** $$|\phi\rangle\langle\psi|$$ is an operator:
$$(|\phi\rangle\langle\psi|)|\chi\rangle = |\phi\rangle\langle\psi|\chi\rangle = \langle\psi|\chi\rangle \cdot |\phi\rangle$$

### 2.3 Matrix Elements

For an operator $$\hat{A}$$:
$$A_{mn} = \langle m | \hat{A} | n \rangle$$

These form the matrix representation of $$\hat{A}$$ in the basis $$\{|n\rangle\}$$.

### 2.4 Change of Representation

Moving between bases $$\{|n\rangle\}$$ and $$\{|n'\rangle\}$$:
$$|\psi\rangle = \sum_n |n\rangle\langle n|\psi\rangle = \sum_{n'} |n'\rangle\langle n'|\psi\rangle$$

The transformation matrix has elements $$S_{n'n} = \langle n' | n \rangle$$.

---

## 3. Linear Operators

### 3.1 Definition

A **linear operator** $$\hat{A}: \mathcal{H} \to \mathcal{H}$$ satisfies:
$$\hat{A}(\alpha|\psi\rangle + \beta|\phi\rangle) = \alpha\hat{A}|\psi\rangle + \beta\hat{A}|\phi\rangle$$

### 3.2 The Adjoint Operator

The **adjoint** (or Hermitian conjugate) $$\hat{A}^\dagger$$ is defined by:
$$\langle \phi | \hat{A}^\dagger | \psi \rangle = \langle \psi | \hat{A} | \phi \rangle^* \quad \text{for all } |\phi\rangle, |\psi\rangle$$

**Properties:**
- $$(\hat{A}^\dagger)^\dagger = \hat{A}$$
- $$(\hat{A}\hat{B})^\dagger = \hat{B}^\dagger \hat{A}^\dagger$$
- $$(\alpha\hat{A} + \beta\hat{B})^\dagger = \alpha^*\hat{A}^\dagger + \beta^*\hat{B}^\dagger$$
- $$(|\phi\rangle\langle\psi|)^\dagger = |\psi\rangle\langle\phi|$$

### 3.3 Special Operators

**Hermitian (Self-Adjoint):** $$\hat{A}^\dagger = \hat{A}$$
- Eigenvalues are real
- Eigenvectors with different eigenvalues are orthogonal
- Represent physical observables

**Unitary:** $$\hat{U}^\dagger \hat{U} = \hat{U}\hat{U}^\dagger = \hat{1}$$
- Preserve inner products: $$\langle\hat{U}\phi|\hat{U}\psi\rangle = \langle\phi|\psi\rangle$$
- Eigenvalues have unit modulus: $$|u| = 1$$
- Represent symmetry transformations and time evolution

**Projection:** $$\hat{P}^2 = \hat{P} = \hat{P}^\dagger$$
- Project onto subspaces
- $$\hat{P}_n = |n\rangle\langle n|$$ projects onto state $$|n\rangle$$

**Anti-Hermitian:** $$\hat{A}^\dagger = -\hat{A}$$
- Eigenvalues are purely imaginary
- Example: $$\hat{A} = i\hat{H}$$ where $$\hat{H}$$ is Hermitian

### 3.4 Functions of Operators

For a Hermitian operator with spectral decomposition $$\hat{A} = \sum_n a_n |n\rangle\langle n|$$:
$$f(\hat{A}) = \sum_n f(a_n) |n\rangle\langle n|$$

**Important Case:** The exponential of an operator:
$$e^{\hat{A}} = \sum_{k=0}^{\infty} \frac{\hat{A}^k}{k!}$$

For Hermitian $$\hat{H}$$, $$e^{-i\hat{H}t/\hbar}$$ is unitary.

---

## 4. Eigenvalue Problems

### 4.1 The Eigenvalue Equation

For operator $$\hat{A}$$:
$$\hat{A}|a\rangle = a|a\rangle$$

where $$a$$ is the eigenvalue and $$|a\rangle$$ is the eigenvector (eigenstate).

### 4.2 Hermitian Operator Theorems

**Theorem 1:** Eigenvalues of Hermitian operators are real.

**Proof:**
$$\langle a | \hat{A} | a \rangle = a\langle a | a \rangle$$
$$\langle a | \hat{A}^\dagger | a \rangle = a^*\langle a | a \rangle$$
Since $$\hat{A}^\dagger = \hat{A}$$: $$a = a^*$$, so $$a \in \mathbb{R}$$.

**Theorem 2:** Eigenvectors of distinct eigenvalues are orthogonal.

**Proof:** For $$\hat{A}|a\rangle = a|a\rangle$$ and $$\hat{A}|a'\rangle = a'|a'\rangle$$ with $$a \neq a'$$:
$$\langle a' | \hat{A} | a \rangle = a\langle a' | a \rangle = a'\langle a' | a \rangle$$
$$(a - a')\langle a' | a \rangle = 0 \Rightarrow \langle a' | a \rangle = 0$$

### 4.3 Spectral Decomposition

For Hermitian $$\hat{A}$$ with orthonormal eigenbasis:
$$\hat{A} = \sum_n a_n |a_n\rangle\langle a_n|$$

This is the **spectral theorem** for Hermitian operators.

### 4.4 Degeneracy

When multiple linearly independent eigenvectors share an eigenvalue, the eigenspace is **degenerate**. The **degeneracy** $$g_n$$ is the dimension of the eigenspace.

For degenerate eigenvalues, we can choose any orthonormal basis within the degenerate subspace.

### 4.5 Simultaneous Eigenstates

**Theorem:** Two Hermitian operators $$\hat{A}$$ and $$\hat{B}$$ have a common eigenbasis if and only if they commute: $$[\hat{A}, \hat{B}] = 0$$.

**Proof (if direction):** If $$[\hat{A}, \hat{B}] = 0$$ and $$\hat{A}|a\rangle = a|a\rangle$$, then:
$$\hat{A}(\hat{B}|a\rangle) = \hat{B}\hat{A}|a\rangle = a(\hat{B}|a\rangle)$$
So $$\hat{B}|a\rangle$$ is also an eigenvector of $$\hat{A}$$ with eigenvalue $$a$$.

---

## 5. Position and Momentum Representations

### 5.1 Position Eigenstates

$$\hat{x}|x\rangle = x|x\rangle$$

The wavefunction is:
$$\psi(x) = \langle x | \psi \rangle$$

In position representation:
$$\hat{x} \to x, \quad \hat{p} \to -i\hbar\frac{d}{dx}$$

### 5.2 Momentum Eigenstates

$$\hat{p}|p\rangle = p|p\rangle$$

$$\langle x | p \rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$

The momentum-space wavefunction is:
$$\tilde{\psi}(p) = \langle p | \psi \rangle = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x)\,dx$$

---

## 6. Commutators

### 6.1 Definition and Properties

The **commutator** of operators $$\hat{A}$$ and $$\hat{B}$$:
$$[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$$

**Basic Properties:**
1. Antisymmetry: $$[\hat{A}, \hat{B}] = -[\hat{B}, \hat{A}]$$
2. Linearity: $$[\hat{A}, \alpha\hat{B} + \beta\hat{C}] = \alpha[\hat{A}, \hat{B}] + \beta[\hat{A}, \hat{C}]$$
3. Product rule: $$[\hat{A}, \hat{B}\hat{C}] = [\hat{A}, \hat{B}]\hat{C} + \hat{B}[\hat{A}, \hat{C}]$$
4. Jacobi identity: $$[\hat{A}, [\hat{B}, \hat{C}]] + [\hat{B}, [\hat{C}, \hat{A}]] + [\hat{C}, [\hat{A}, \hat{B}]] = 0$$

### 6.2 Fundamental Commutators

**Canonical commutation relation:**
$$[\hat{x}, \hat{p}] = i\hbar$$

This can be verified in position representation:
$$[\hat{x}, \hat{p}]\psi = x(-i\hbar\frac{d\psi}{dx}) - (-i\hbar\frac{d(x\psi)}{dx}) = i\hbar\psi$$

**Generalized canonical relations:**
$$[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}$$
$$[\hat{x}_i, \hat{x}_j] = 0$$
$$[\hat{p}_i, \hat{p}_j] = 0$$

### 6.3 Angular Momentum Commutators

$$[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$$

Explicitly:
$$[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$$
$$[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x$$
$$[\hat{L}_z, \hat{L}_x] = i\hbar\hat{L}_y$$

Also: $$[\hat{L}^2, \hat{L}_i] = 0$$ for all $$i$$.

### 6.4 Useful Commutator Identities

**Baker-Campbell-Hausdorff (simplified case):**
If $$[\hat{A}, [\hat{A}, \hat{B}]] = [\hat{B}, [\hat{A}, \hat{B}]] = 0$$, then:
$$e^{\hat{A}}e^{\hat{B}} = e^{\hat{A} + \hat{B} + \frac{1}{2}[\hat{A}, \hat{B}]}$$

**Hadamard lemma:**
$$e^{\hat{A}}\hat{B}e^{-\hat{A}} = \hat{B} + [\hat{A}, \hat{B}] + \frac{1}{2!}[\hat{A}, [\hat{A}, \hat{B}]] + \cdots$$

### 6.5 Complete Sets of Commuting Observables (CSCO)

A **CSCO** is a maximal set of mutually commuting Hermitian operators whose simultaneous eigenstates form a unique basis.

Example for hydrogen atom: $$\{\hat{H}, \hat{L}^2, \hat{L}_z, \hat{S}^2, \hat{S}_z\}$$

---

## 7. The Uncertainty Principle

### 7.1 Variance and Uncertainty

For observable $$\hat{A}$$ in state $$|\psi\rangle$$:

**Expectation value:**
$$\langle \hat{A} \rangle = \langle \psi | \hat{A} | \psi \rangle$$

**Variance:**
$$(\Delta A)^2 = \langle (\hat{A} - \langle \hat{A} \rangle)^2 \rangle = \langle \hat{A}^2 \rangle - \langle \hat{A} \rangle^2$$

**Uncertainty (standard deviation):**
$$\Delta A = \sqrt{(\Delta A)^2}$$

### 7.2 The Robertson Uncertainty Relation

For any two Hermitian operators $$\hat{A}$$ and $$\hat{B}$$:

$$\boxed{\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle [\hat{A}, \hat{B}] \rangle|}$$

**Derivation:**

Define $$\hat{A}' = \hat{A} - \langle \hat{A} \rangle$$ and $$\hat{B}' = \hat{B} - \langle \hat{B} \rangle$$.

Consider $$|f\rangle = (\hat{A}' + i\lambda\hat{B}')|\psi\rangle$$ for real $$\lambda$$.

Since $$\langle f | f \rangle \geq 0$$:
$$\langle \hat{A}'^2 \rangle + \lambda^2\langle \hat{B}'^2 \rangle + i\lambda\langle [\hat{A}', \hat{B}'] \rangle \geq 0$$

Note that $$[\hat{A}', \hat{B}'] = [\hat{A}, \hat{B}]$$, which is anti-Hermitian, so $$\langle [\hat{A}, \hat{B}] \rangle$$ is pure imaginary.

Write $$\langle [\hat{A}, \hat{B}] \rangle = iC$$ where $$C$$ is real. Then:
$$(\Delta A)^2 + \lambda^2(\Delta B)^2 - \lambda C \geq 0$$

This quadratic in $$\lambda$$ must be non-negative for all $$\lambda$$, so the discriminant is non-positive:
$$C^2 - 4(\Delta A)^2(\Delta B)^2 \leq 0$$
$$(\Delta A)^2(\Delta B)^2 \geq \frac{C^2}{4} = \frac{|\langle [\hat{A}, \hat{B}] \rangle|^2}{4}$$

Taking the square root gives the Robertson inequality.

### 7.3 Position-Momentum Uncertainty

Since $$[\hat{x}, \hat{p}] = i\hbar$$:
$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

### 7.4 Minimum Uncertainty States

States that saturate the uncertainty bound satisfy:
$$(\hat{A}' + i\lambda_0\hat{B}')|\psi\rangle = 0$$

where $$\lambda_0$$ minimizes the quadratic. This occurs when:
$$\lambda_0 = \frac{C}{2(\Delta B)^2} = \frac{\langle [\hat{A}, \hat{B}] \rangle / i}{2(\Delta B)^2}$$

For $$\hat{x}$$ and $$\hat{p}$$, Gaussian wavepackets are minimum uncertainty states.

### 7.5 Energy-Time Uncertainty

The energy-time uncertainty relation:
$$\Delta E \cdot \Delta t \geq \frac{\hbar}{2}$$

requires careful interpretation since time is not an operator in non-relativistic QM. Here $$\Delta t$$ is the timescale for observable change:
$$\Delta t = \frac{\Delta A}{|d\langle \hat{A} \rangle / dt|}$$

This gives a bound on how quickly quantum states can evolve.

---

## 8. Key Results Summary

### Essential Formulas

| Concept | Formula |
|---------|---------|
| Norm | $$\|\psi\| = \sqrt{\langle\psi|\psi\rangle}$$ |
| Completeness (discrete) | $$\sum_n \|n\rangle\langle n\| = \hat{1}$$ |
| Completeness (continuous) | $$\int \|x\rangle\langle x\| dx = \hat{1}$$ |
| Adjoint of product | $$(\hat{A}\hat{B})^\dagger = \hat{B}^\dagger\hat{A}^\dagger$$ |
| Canonical commutator | $$[\hat{x}, \hat{p}] = i\hbar$$ |
| Angular momentum | $$[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$$ |
| Uncertainty principle | $$\Delta A \Delta B \geq \frac{1}{2}|\langle[\hat{A},\hat{B}]\rangle|$$ |
| Position-momentum | $$\Delta x \Delta p \geq \frac{\hbar}{2}$$ |

### Proof Techniques

1. **For Hermiticity:** Show $$\langle\phi|\hat{A}|\psi\rangle = \langle\psi|\hat{A}|\phi\rangle^*$$ for all states
2. **For commutators:** Apply operators in both orders to a test function
3. **For uncertainty relations:** Use the Schwarz inequality or the Robertson derivation
4. **For eigenstates:** Apply the operator and verify the eigenvalue equation

---

## 9. Common Qualifying Exam Problem Types

1. **Prove an operator is Hermitian** (or not)
2. **Calculate commutators** involving functions of operators
3. **Find eigenvalues and eigenvectors** of given matrices
4. **Prove the uncertainty relation** for specific operators
5. **Calculate expectation values** in given states
6. **Determine if operators have simultaneous eigenstates**
7. **Work with change of basis** and unitary transformations

---

## 10. Study Questions

1. Why must observables be represented by Hermitian operators?
2. What is the physical significance of commuting operators?
3. How does the uncertainty principle limit simultaneous measurements?
4. What distinguishes a Hilbert space from an ordinary vector space?
5. Why is the spectral theorem so important in quantum mechanics?

---

*Review Guide for Week 145 — Mathematical Framework*
*Month 37: QM Foundations Review I*
