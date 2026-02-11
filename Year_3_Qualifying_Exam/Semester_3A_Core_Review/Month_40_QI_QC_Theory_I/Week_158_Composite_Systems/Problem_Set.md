# Week 158: Composite Systems - Problem Set

## Instructions

This problem set contains 30 qualifying exam-style problems on composite systems, tensor products, partial trace, Schmidt decomposition, and purification. Work through these problems without consulting solutions first.

**Difficulty Levels:**
- (E) Easy: Direct application of formulas
- (M) Medium: Multi-step reasoning required
- (H) Hard: Challenging, research-level insight needed

---

## Section A: Tensor Products (Problems 1-8)

### Problem 1 (E)
Compute the tensor product $$|+\rangle \otimes |0\rangle$$ where $$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$.

Write your answer:
(a) In Dirac notation
(b) As a column vector in the computational basis $$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$

---

### Problem 2 (E)
Compute the Kronecker product:

$$\sigma_x \otimes I = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

---

### Problem 3 (M)
Show that for any states $$|a\rangle, |b\rangle, |c\rangle, |d\rangle$$:

$$(\langle a| \otimes \langle b|)(|c\rangle \otimes |d\rangle) = \langle a|c\rangle \cdot \langle b|d\rangle$$

---

### Problem 4 (M)
Verify the mixed-product property:

$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

for the specific case $$A = C = \sigma_x$$ and $$B = D = \sigma_z$$.

---

### Problem 5 (E)
The CNOT gate has matrix:

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

Compute $$\text{CNOT}|+0\rangle$$ and $$\text{CNOT}|+1\rangle$$.

---

### Problem 6 (M)
Show that the CNOT gate can be written as:

$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

by computing the matrix representation of the right-hand side.

---

### Problem 7 (H)
Prove that if $$A$$ has eigenvalue $$\alpha$$ with eigenvector $$|a\rangle$$, and $$B$$ has eigenvalue $$\beta$$ with eigenvector $$|b\rangle$$, then $$A \otimes B$$ has eigenvalue $$\alpha\beta$$ with eigenvector $$|a\rangle \otimes |b\rangle$$.

---

### Problem 8 (H)
For two qubits, the SWAP gate exchanges the two qubits: $$\text{SWAP}|ab\rangle = |ba\rangle$$.

(a) Write the matrix for SWAP.
(b) Express SWAP in terms of CNOT gates.
(c) Show that $$\text{SWAP} = \frac{1}{2}(I \otimes I + X \otimes X + Y \otimes Y + Z \otimes Z)$$.

---

## Section B: Partial Trace (Problems 9-16)

### Problem 9 (E)
Compute $$\text{Tr}_B(|0\rangle\langle 1| \otimes |+\rangle\langle +|)$$.

---

### Problem 10 (E)
For the product state $$\rho_{AB} = |0\rangle\langle 0| \otimes |+\rangle\langle +|$$:

(a) Compute $$\rho_A = \text{Tr}_B(\rho_{AB})$$
(b) Compute $$\rho_B = \text{Tr}_A(\rho_{AB})$$
(c) Verify that $$\rho_{AB} = \rho_A \otimes \rho_B$$

---

### Problem 11 (M)
The four Bell states are:
$$|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle)$$
$$|\Psi^\pm\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

Compute the reduced density matrix $$\rho_A$$ for each Bell state.

---

### Problem 12 (M)
Consider the state:
$$|\psi\rangle = \frac{1}{2}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|10\rangle + \frac{1}{2}|11\rangle$$

(a) Show this is a product state by factoring it.
(b) Compute both reduced density matrices.
(c) Verify that both reduced states are pure.

---

### Problem 13 (M)
For the state $$|\psi\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|11\rangle$$:

(a) Compute the density matrix $$\rho_{AB}$$.
(b) Compute $$\rho_A = \text{Tr}_B(\rho_{AB})$$.
(c) Is this state entangled? Justify using the purity of $$\rho_A$$.

---

### Problem 14 (H)
Prove that for any bipartite pure state $$|\psi\rangle_{AB}$$:

$$S(\rho_A) = S(\rho_B)$$

where $$S$$ is the von Neumann entropy.

---

### Problem 15 (H)
For the density matrix:

$$\rho_{AB} = \frac{1}{4}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

(a) Compute both reduced density matrices.
(b) Is this a pure or mixed state?
(c) If mixed, can you write it as a mixture of Bell states?

---

### Problem 16 (H)
Show that the partial trace is the unique linear map $$\mathcal{E}$$ satisfying:

1. $$\mathcal{E}(A \otimes B) = A \cdot \text{Tr}(B)$$
2. $$\mathcal{E}$$ is completely positive

---

## Section C: Schmidt Decomposition (Problems 17-24)

### Problem 17 (E)
Determine the Schmidt decomposition and Schmidt rank for:

(a) $$|00\rangle$$
(b) $$|+\rangle|0\rangle$$
(c) $$\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

---

### Problem 18 (M)
Find the Schmidt decomposition of:
$$|\psi\rangle = \frac{1}{\sqrt{3}}|00\rangle + \frac{1}{\sqrt{3}}|01\rangle + \frac{1}{\sqrt{3}}|11\rangle$$

---

### Problem 19 (M)
For the state:
$$|\psi\rangle = \frac{1}{2}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|10\rangle - \frac{1}{2}|11\rangle$$

(a) Write the coefficient matrix $$C$$ where $$|\psi\rangle = \sum_{ij} C_{ij}|ij\rangle$$.
(b) Compute the SVD of $$C$$.
(c) Write the Schmidt decomposition.

---

### Problem 20 (M)
Prove that a bipartite pure state $$|\psi\rangle$$ is a product state if and only if its Schmidt rank is 1.

---

### Problem 21 (H)
For the state:
$$|\psi\rangle = \sqrt{\frac{1}{2}}|00\rangle + \sqrt{\frac{1}{3}}|11\rangle + \sqrt{\frac{1}{6}}|22\rangle$$

(a) What is the Schmidt rank?
(b) Compute the reduced density matrix $$\rho_A$$.
(c) Calculate the von Neumann entropy $$S(\rho_A)$$.

---

### Problem 22 (H)
The generalized Bell state (for dimension $$d$$) is:
$$|\Phi_d\rangle = \frac{1}{\sqrt{d}}\sum_{j=0}^{d-1}|jj\rangle$$

(a) What is the Schmidt rank?
(b) Compute the reduced density matrix.
(c) Calculate the entanglement entropy.

---

### Problem 23 (H)
Prove that the Schmidt coefficients of $$|\psi\rangle$$ are the singular values of its coefficient matrix $$C$$.

---

### Problem 24 (H)
For a random state $$|\psi\rangle \in \mathbb{C}^{d_A} \otimes \mathbb{C}^{d_B}$$ with $$d_A \leq d_B$$:

(a) What is the maximum possible Schmidt rank?
(b) What is the typical Schmidt rank for a Haar-random state?
(c) What is the expected entanglement entropy for large $$d$$?

---

## Section D: Purification (Problems 25-28)

### Problem 25 (M)
Construct a purification of the mixed state:
$$\rho = \frac{2}{3}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1|$$

Verify your answer by computing the partial trace.

---

### Problem 26 (M)
The maximally mixed qubit state is $$\rho = \frac{I}{2}$$.

(a) Construct a purification using a one-qubit ancilla.
(b) Is your purification unique? If not, give another one.
(c) Show that any purification of $$I/2$$ is a maximally entangled state.

---

### Problem 27 (H)
Prove that all purifications of a mixed state $$\rho_A$$ are related by unitaries on the ancilla system.

That is, if $$|\Psi\rangle$$ and $$|\Psi'\rangle$$ are both purifications of $$\rho_A$$, then there exists a unitary $$U_B$$ such that:
$$|\Psi'\rangle = (I_A \otimes U_B)|\Psi\rangle$$

---

### Problem 28 (H)
Consider a qubit in the mixed state:
$$\rho = \frac{3}{4}|+\rangle\langle+| + \frac{1}{4}|-\rangle\langle-|$$

(a) Find the spectral decomposition of $$\rho$$.
(b) Construct a purification in the eigenbasis.
(c) Find the Schmidt coefficients of your purification.
(d) Calculate the entanglement entropy.

---

## Section E: Advanced Topics (Problems 29-30)

### Problem 29 (H)
**Partial Transpose and PPT Criterion**

The partial transpose of $$\rho_{AB}$$ over $$B$$ swaps indices in the $$B$$ subsystem:
$$(\rho^{T_B})_{ij,kl} = \rho_{il,kj}$$

(a) Compute the partial transpose of the Bell state $$|\Phi^+\rangle\langle\Phi^+|$$.
(b) Find the eigenvalues of $$\rho^{T_B}$$.
(c) What does this tell you about entanglement?

---

### Problem 30 (H)
**Three-Qubit Systems**

For the GHZ state:
$$|\text{GHZ}\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

(a) Compute the reduced density matrix $$\rho_{AB} = \text{Tr}_C(\rho_{ABC})$$.
(b) Compute $$\rho_A = \text{Tr}_{BC}(\rho_{ABC})$$.
(c) Is $$\rho_{AB}$$ entangled? (Hint: Use PPT criterion.)
(d) Calculate $$S(\rho_A)$$, $$S(\rho_{AB})$$, and $$S(\rho_{ABC})$$. Compare to the inequalities for bipartite entropy.

---

## Bonus Challenge Problems

### Bonus 1 (Research Level)
**Operator Schmidt Decomposition**

Any operator $$O$$ on $$\mathcal{H}_A \otimes \mathcal{H}_B$$ can be written:
$$O = \sum_k c_k A_k \otimes B_k$$

For the CNOT gate:
(a) Find the operator Schmidt decomposition.
(b) What is the operator Schmidt rank?
(c) Relate this to the entangling power of CNOT.

---

### Bonus 2 (Research Level)
**Entanglement of Formation via Purification**

The entanglement of formation of a mixed state $$\rho_{AB}$$ is:
$$E_F(\rho_{AB}) = \min_{\{p_i, |\psi_i\rangle\}} \sum_i p_i S(\text{Tr}_B(|\psi_i\rangle\langle\psi_i|))$$

where the minimum is over all decompositions $$\rho_{AB} = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$.

(a) For pure states, show that $$E_F = S(\rho_A)$$.
(b) Show that $$E_F$$ is related to purifications: $$E_F(\rho_{AB}) = \min_{|\Psi\rangle} S(\rho_A)$$ where the min is over all purifications of $$\rho_{AB}$$.

---

## Problem Set Summary

| Section | Problems | Topics |
|---------|----------|--------|
| A | 1-8 | Tensor products |
| B | 9-16 | Partial trace |
| C | 17-24 | Schmidt decomposition |
| D | 25-28 | Purification |
| E | 29-30 | Advanced topics |

**Total: 30 problems + 2 bonus problems**

---

*Complete this problem set before consulting solutions. Time yourself and identify areas needing additional review.*
