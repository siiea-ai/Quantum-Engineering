# Week 158: Composite Systems - Problem Solutions

## Section A: Tensor Products

### Solution 1

$$|+\rangle \otimes |0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle$$

**(a) Dirac notation:**
$$= \frac{1}{\sqrt{2}}(|0\rangle \otimes |0\rangle + |1\rangle \otimes |0\rangle) = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

**(b) Column vector:**
Basis order: $$|00\rangle, |01\rangle, |10\rangle, |11\rangle$$

$$|+0\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 0 \\ 1 \\ 0 \end{pmatrix}$$

---

### Solution 2

$$\sigma_x \otimes I = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

$$= \begin{pmatrix} 0 \cdot I & 1 \cdot I \\ 1 \cdot I & 0 \cdot I \end{pmatrix} = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

---

### Solution 3

By definition of inner product in tensor product spaces:
$$\langle a| \otimes \langle b|$$ is the dual of $$|a\rangle \otimes |b\rangle$$

For basis states:
$$(\langle i| \otimes \langle j|)(|k\rangle \otimes |l\rangle) = \delta_{ik}\delta_{jl} = \langle i|k\rangle \cdot \langle j|l\rangle$$

By linearity, for general states:
$$(\langle a| \otimes \langle b|)(|c\rangle \otimes |d\rangle) = \langle a|c\rangle \cdot \langle b|d\rangle$$

---

### Solution 4

**LHS:** $$(σ_x \otimes σ_z)(σ_x \otimes σ_z)$$

**RHS:** $$(σ_x σ_x) \otimes (σ_z σ_z) = I \otimes I = I_4$$

**Direct computation of LHS:**
$$σ_x \otimes σ_z = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \\ 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \end{pmatrix}$$

$$(σ_x \otimes σ_z)^2 = I_4$$ ✓

---

### Solution 5

$$|+0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 0 \\ 1 \\ 0 \end{pmatrix}$$

$$\text{CNOT}|+0\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \\ 1 \\ 0 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 0 \\ 0 \\ 1 \end{pmatrix} = |\Phi^+\rangle$$

$$|+1\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |11\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ 1 \\ 0 \\ 1 \end{pmatrix}$$

$$\text{CNOT}|+1\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ 1 \\ 1 \\ 0 \end{pmatrix} = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) = |\Psi^+\rangle$$

---

### Solution 6

$$|0\rangle\langle 0| \otimes I = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

$$|1\rangle\langle 1| \otimes X = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} \otimes \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

Sum:
$$|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix} = \text{CNOT}$$ ✓

---

### Solution 7

Given: $$A|a\rangle = α|a\rangle$$ and $$B|b\rangle = β|b\rangle$$

$$(A \otimes B)(|a\rangle \otimes |b\rangle) = (A|a\rangle) \otimes (B|b\rangle)$$
$$= (α|a\rangle) \otimes (β|b\rangle)$$
$$= αβ(|a\rangle \otimes |b\rangle)$$

Therefore $$|a\rangle \otimes |b\rangle$$ is an eigenvector of $$A \otimes B$$ with eigenvalue $$αβ$$.

---

### Solution 8

**(a) SWAP matrix:**
$$\text{SWAP}|00\rangle = |00\rangle, \text{SWAP}|01\rangle = |10\rangle, \text{SWAP}|10\rangle = |01\rangle, \text{SWAP}|11\rangle = |11\rangle$$

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**(b) In terms of CNOT:**
$$\text{SWAP} = \text{CNOT}_{12} \cdot \text{CNOT}_{21} \cdot \text{CNOT}_{12}$$

**(c) Pauli expansion:**
Compute:
- $$I \otimes I = I_4$$
- $$X \otimes X = \text{antidiag}(1,1,1,1)$$
- $$Y \otimes Y = \text{antidiag}(1,-1,-1,1)$$
- $$Z \otimes Z = \text{diag}(1,-1,-1,1)$$

$$\frac{1}{2}(I \otimes I + X \otimes X + Y \otimes Y + Z \otimes Z) = \text{SWAP}$$ ✓

---

## Section B: Partial Trace

### Solution 9

$$\text{Tr}_B(|0\rangle\langle 1| \otimes |+\rangle\langle +|) = |0\rangle\langle 1| \cdot \text{Tr}(|+\rangle\langle +|)$$
$$= |0\rangle\langle 1| \cdot 1 = |0\rangle\langle 1|$$

---

### Solution 10

**(a)** $$ρ_A = \text{Tr}_B(|0\rangle\langle 0| \otimes |+\rangle\langle +|) = |0\rangle\langle 0| \cdot \text{Tr}(|+\rangle\langle +|) = |0\rangle\langle 0|$$

**(b)** $$ρ_B = \text{Tr}_A(|0\rangle\langle 0| \otimes |+\rangle\langle +|) = \text{Tr}(|0\rangle\langle 0|) \cdot |+\rangle\langle +| = |+\rangle\langle +|$$

**(c)** $$ρ_A \otimes ρ_B = |0\rangle\langle 0| \otimes |+\rangle\langle +| = ρ_{AB}$$ ✓

---

### Solution 11

For all Bell states, the reduced density matrix is the same:

$$|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle)$$

$$ρ_A = \text{Tr}_B(|\Phi^\pm\rangle\langle\Phi^\pm|)$$
$$= \frac{1}{2}(\langle 0|0\rangle|0\rangle\langle 0| + \langle 1|1\rangle|1\rangle\langle 1| \pm \langle 0|1\rangle \cdot \text{cross terms})$$
$$= \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

Similarly for $$|\Psi^\pm\rangle$$:
$$ρ_A = \frac{I}{2}$$

**All four Bell states have maximally mixed reduced states.**

---

### Solution 12

**(a)** $$|\psi\rangle = \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle + |1\rangle) = |+\rangle|+\rangle$$

Product state!

**(b)** $$ρ_A = |+\rangle\langle +|$$, $$ρ_B = |+\rangle\langle +|$$

**(c)** Both are pure: $$\text{Tr}(ρ_A^2) = \text{Tr}(ρ_B^2) = 1$$ ✓

---

### Solution 13

**(a)** $$|\psi\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|11\rangle$$

$$ρ_{AB} = |\psi\rangle\langle\psi|$$

**(b)** Coefficient matrix:
$$C = \begin{pmatrix} 1/\sqrt{2} & 1/2 \\ 0 & 1/2 \end{pmatrix}$$

$$ρ_A = CC^\dagger = \begin{pmatrix} 1/2 + 1/4 & 1/4 \\ 1/4 & 1/4 \end{pmatrix} = \begin{pmatrix} 3/4 & 1/4 \\ 1/4 & 1/4 \end{pmatrix}$$

**(c)** $$\text{Tr}(ρ_A^2) = (3/4)^2 + 2(1/4)^2 + (1/4)^2 = 9/16 + 2/16 + 1/16 = 12/16 = 3/4 < 1$$

Since $$ρ_A$$ is mixed, the state is **entangled**.

---

### Solution 14

For a bipartite pure state $$|\psi\rangle$$, use Schmidt decomposition:
$$|\psi\rangle = \sum_i λ_i |a_i\rangle|b_i\rangle$$

Then:
$$ρ_A = \sum_i λ_i^2 |a_i\rangle\langle a_i|$$
$$ρ_B = \sum_i λ_i^2 |b_i\rangle\langle b_i|$$

Both have the same eigenvalues $$\{λ_i^2\}$$, so:
$$S(ρ_A) = -\sum_i λ_i^2 \log λ_i^2 = S(ρ_B)$$

---

### Solution 15

**(a)** Direct calculation:
$$ρ_A = \text{Tr}_B(ρ_{AB}) = \frac{1}{4}\begin{pmatrix} 1+1 & 0+1 \\ 0+1 & 1+1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1/2 \\ 1/2 & 1 \end{pmatrix}$$

Wait, let me recalculate. The matrix in block form:
$$ρ_{AB} = \frac{1}{4}\begin{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} & \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \\ \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} & \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \end{pmatrix}$$

$$ρ_A = \frac{1}{4}(\text{Tr of each } 2\times2 \text{ block}) = \frac{1}{4}\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = \frac{I}{2}$$

**(b)** $$\text{Tr}(ρ_{AB}^2) = ?$$
Eigenvalues of $$ρ_{AB}$$: compute them...
Actually, $$ρ_{AB} = \frac{1}{2}(|\Phi^+\rangle\langle\Phi^+| + |\Psi^+\rangle\langle\Psi^+|)$$
This is a **mixed state** (mixture of Bell states).

**(c)** Yes: $$ρ_{AB} = \frac{1}{2}|\Phi^+\rangle\langle\Phi^+| + \frac{1}{2}|\Psi^+\rangle\langle\Psi^+|$$

---

### Solution 16

**Existence:** The partial trace defined satisfies both properties.

**Uniqueness:** Any linear map satisfying property 1 is determined on all tensor products $$A \otimes B$$. Since these span the space of all operators, the map is uniquely determined.

Complete positivity follows from the definition via sum of $$\langle j|$$ and $$|j\rangle$$ sandwiches.

---

## Section C: Schmidt Decomposition

### Solution 17

**(a)** $$|00\rangle = 1 \cdot |0\rangle|0\rangle$$
Schmidt rank = 1, $$λ_1 = 1$$

**(b)** $$|+\rangle|0\rangle = 1 \cdot |+\rangle|0\rangle$$
Schmidt rank = 1, $$λ_1 = 1$$

**(c)** $$\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = \frac{1}{\sqrt{2}}|0\rangle|0\rangle + \frac{1}{\sqrt{2}}|1\rangle|1\rangle$$
Schmidt rank = 2, $$λ_1 = λ_2 = \frac{1}{\sqrt{2}}$$

---

### Solution 18

$$|\psi\rangle = \frac{1}{\sqrt{3}}|00\rangle + \frac{1}{\sqrt{3}}|01\rangle + \frac{1}{\sqrt{3}}|11\rangle$$

Coefficient matrix:
$$C = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

SVD of $$C$$: compute $$CC^\dagger$$ and $$C^\dagger C$$...

$$CC^\dagger = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

Eigenvalues: $$λ = \frac{3 \pm \sqrt{5}}{6}$$

$$λ_1^2 = \frac{3+\sqrt{5}}{6} \approx 0.873$$, $$λ_2^2 = \frac{3-\sqrt{5}}{6} \approx 0.127$$

$$λ_1 \approx 0.934$$, $$λ_2 \approx 0.357$$

Schmidt rank = 2 (entangled state).

---

### Solution 19

**(a)** $$C = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**(b)** This is proportional to the Hadamard matrix!
$$C = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

The SVD: $$H = U \Sigma V^\dagger$$ where $$U = V = H/\sqrt{2}$$ and $$\Sigma = I$$.

So $$C = \frac{1}{2}H$$ has singular values $$\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}$$.

**(c)** Schmidt decomposition:
$$|\psi\rangle = \frac{1}{\sqrt{2}}|+\rangle|+\rangle + \frac{1}{\sqrt{2}}|-\rangle|-\rangle$$

Wait, let me verify: Actually $$|+\rangle|+\rangle + |-\rangle|-\rangle = |00\rangle + |11\rangle \neq |\psi\rangle$$.

Let me redo: $$C = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$CC^\dagger = \frac{1}{4}\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = \frac{I}{2}$$

So both singular values are $$1/\sqrt{2}$$.

Schmidt form: $$|\psi\rangle = \frac{1}{\sqrt{2}}|a_1\rangle|b_1\rangle + \frac{1}{\sqrt{2}}|a_2\rangle|b_2\rangle$$

where the bases are determined by the SVD.

---

### Solution 20

**If Schmidt rank = 1:**
$$|\psi\rangle = λ_1|a_1\rangle|b_1\rangle = |a_1\rangle|b_1\rangle$$ (since $$λ_1 = 1$$)

This is a product state.

**If product state:**
$$|\psi\rangle = |a\rangle|b\rangle$$

This is already in Schmidt form with $$r = 1$$.

---

### Solution 21

$$|\psi\rangle = \sqrt{\frac{1}{2}}|00\rangle + \sqrt{\frac{1}{3}}|11\rangle + \sqrt{\frac{1}{6}}|22\rangle$$

**(a)** Already in Schmidt form! Schmidt rank = 3.

**(b)** $$ρ_A = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1| + \frac{1}{6}|2\rangle\langle 2|$$

**(c)** $$S(ρ_A) = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{3}\log_2\frac{1}{3} - \frac{1}{6}\log_2\frac{1}{6}$$
$$= \frac{1}{2} + \frac{1}{3}(1.585) + \frac{1}{6}(2.585)$$
$$= 0.5 + 0.528 + 0.431 = 1.459$$ bits

---

### Solution 22

$$|\Phi_d\rangle = \frac{1}{\sqrt{d}}\sum_{j=0}^{d-1}|jj\rangle$$

**(a)** Schmidt rank = $$d$$ (maximally entangled)

**(b)** $$ρ_A = \frac{1}{d}\sum_j |j\rangle\langle j| = \frac{I_d}{d}$$

**(c)** $$S(ρ_A) = \log_2 d$$ (maximum for dimension $$d$$)

---

### Solution 23

Write $$|\psi\rangle = \sum_{ij} C_{ij}|i\rangle|j\rangle$$.

$$ρ_A = CC^\dagger$$

The eigenvalues of $$ρ_A$$ are $$\{σ_i^2\}$$ where $$\{σ_i\}$$ are the singular values of $$C$$.

Schmidt coefficients are $$λ_i = σ_i$$. ✓

---

### Solution 24

**(a)** Maximum Schmidt rank = $$\min(d_A, d_B) = d_A$$

**(b)** For Haar-random states, with probability 1, Schmidt rank = $$d_A$$ (full rank).

**(c)** Expected entropy $$\approx \log_2 d_A - \frac{d_A}{2d_B \ln 2}$$ for $$d_B \gg d_A$$.

---

## Section D: Purification

### Solution 25

$$ρ = \frac{2}{3}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1|$$

**Purification:**
$$|\Psi\rangle = \sqrt{\frac{2}{3}}|0\rangle|0\rangle + \sqrt{\frac{1}{3}}|1\rangle|1\rangle$$

**Verification:**
$$\text{Tr}_B(|\Psi\rangle\langle\Psi|) = \frac{2}{3}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1| = ρ$$ ✓

---

### Solution 26

**(a)** $$ρ = \frac{I}{2} = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$$

Purification: $$|\Psi\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|11\rangle = |\Phi^+\rangle$$

**(b)** Not unique! Another: $$|\Psi'\rangle = \frac{1}{\sqrt{2}}|01\rangle + \frac{1}{\sqrt{2}}|10\rangle = |\Psi^+\rangle$$

**(c)** Any purification of $$I/2$$ has Schmidt coefficients $$1/\sqrt{2}, 1/\sqrt{2}$$, which is the maximally entangled case for qubits.

---

### Solution 27

Let $$ρ_A = \sum_i p_i|i\rangle\langle i|$$ be the spectral decomposition.

Any purification has the form:
$$|\Psi\rangle = \sum_i \sqrt{p_i}|i\rangle|φ_i\rangle$$

where $$\{|φ_i\rangle\}$$ are orthonormal (for $$\text{Tr}_B$$ to give back $$ρ_A$$).

For two purifications with $$\{|φ_i\rangle\}$$ and $$\{|φ'_i\rangle\}$$:

Define $$U_B$$ by $$U_B|φ_i\rangle = |φ'_i\rangle$$ and extend to an orthonormal basis.

Then $$(I_A \otimes U_B)|\Psi\rangle = |\Psi'\rangle$$. ✓

---

### Solution 28

**(a)** $$ρ = \frac{3}{4}|+\rangle\langle+| + \frac{1}{4}|-\rangle\langle-|$$

Already diagonal in $$\{|+\rangle, |-\rangle\}$$ basis.

**(b)** Purification: $$|\Psi\rangle = \sqrt{\frac{3}{4}}|+\rangle|0\rangle + \sqrt{\frac{1}{4}}|-\rangle|1\rangle$$
$$= \frac{\sqrt{3}}{2}|+\rangle|0\rangle + \frac{1}{2}|-\rangle|1\rangle$$

**(c)** Schmidt coefficients: $$λ_1 = \frac{\sqrt{3}}{2}, λ_2 = \frac{1}{2}$$

**(d)** $$S = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4}$$
$$= \frac{3}{4}(2 - \log_2 3) + \frac{1}{4}(2) = \frac{3}{4}(0.415) + 0.5 = 0.811$$ bits

---

## Section E: Advanced Topics

### Solution 29

**(a)** $$ρ = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

Partial transpose over B:
$$ρ^{T_B} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**(b)** Eigenvalues of $$ρ^{T_B}$$: $$\{1/2, 1/2, 1/2, -1/2\}$$

**(c)** Negative eigenvalue $$\Rightarrow$$ the state is **entangled** (violates PPT criterion).

---

### Solution 30

$$|\text{GHZ}\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

**(a)** $$ρ_{AB} = \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$$

This is a classical mixture, not entangled!

**(b)** $$ρ_A = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

**(c)** $$ρ_{AB}$$ is **separable** (classical correlation, not quantum entanglement).
PPT: $$ρ_{AB}^{T_B} = ρ_{AB}$$ (already diagonal), all eigenvalues $$\geq 0$$.

**(d)**
- $$S(ρ_A) = 1$$ bit
- $$S(ρ_{AB}) = 1$$ bit (eigenvalues 1/2, 1/2)
- $$S(ρ_{ABC}) = 0$$ (pure state)

Interesting: $$S(ρ_A) = S(ρ_{AB})$$ for GHZ states!

---

*Solutions complete. Review any problems you found challenging before the oral examination.*
