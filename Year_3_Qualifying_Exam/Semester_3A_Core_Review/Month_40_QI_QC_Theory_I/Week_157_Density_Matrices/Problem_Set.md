# Week 157: Density Matrices - Problem Set

## Instructions

This problem set contains 30 qualifying exam-style problems on density matrices. Work through these problems without consulting solutions first. Time yourself - aim for 10-15 minutes per problem in exam conditions.

**Difficulty Levels:**
- (E) Easy: Direct application of formulas
- (M) Medium: Multi-step reasoning required
- (H) Hard: Challenging, research-level insight needed

---

## Section A: Pure States and Basic Properties (Problems 1-8)

### Problem 1 (E)
Construct the density matrix for the state $$|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$$.

Verify that it satisfies all three properties of a density matrix.

---

### Problem 2 (E)
For the state $$|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$:

(a) Write the density matrix in the computational basis.

(b) Calculate $$\text{Tr}(\rho)$$ and $$\text{Tr}(\rho^2)$$.

(c) Find the eigenvalues of $$\rho$$.

---

### Problem 3 (M)
Consider the state $$|\psi\rangle = \cos\theta|0\rangle + e^{i\phi}\sin\theta|1\rangle$$.

(a) Write the general form of the density matrix $$\rho = |\psi\rangle\langle\psi|$$.

(b) Express the off-diagonal elements in terms of $$\theta$$ and $$\phi$$.

(c) Under what conditions is $$\rho$$ diagonal in the computational basis?

---

### Problem 4 (M)
The density matrix for a pure state satisfies $$\rho^2 = \rho$$.

(a) Prove this property algebraically.

(b) Show that this implies all eigenvalues are either 0 or 1.

(c) Explain why a pure state density matrix has exactly one eigenvalue equal to 1.

---

### Problem 5 (E)
Compute the expectation value of $$\sigma_z$$ for the state:

$$\rho = \frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}$$

Is this a pure or mixed state? Justify your answer.

---

### Problem 6 (M)
A qubit is prepared in state $$|0\rangle$$ with probability $$2/3$$ and state $$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$ with probability $$1/3$$.

(a) Write the density matrix for this ensemble.

(b) Is this a pure or mixed state? Calculate the purity to confirm.

(c) What is the probability of measuring $$|0\rangle$$ in the computational basis?

---

### Problem 7 (H)
Prove that any density matrix can be written as a convex combination of pure state projectors:

$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

where $$p_i \geq 0$$ and $$\sum_i p_i = 1$$. The states $$|\psi_i\rangle$$ need not be orthogonal.

---

### Problem 8 (M)
For a general qubit state $$\rho$$, show that:

$$\text{det}(\rho) = \lambda_1 \lambda_2$$

where $$\lambda_1, \lambda_2$$ are the eigenvalues. Use this to derive a formula for purity in terms of the determinant.

---

## Section B: Mixed States and Ensembles (Problems 9-14)

### Problem 9 (E)
The maximally mixed state for a qubit is $$\rho = \frac{I}{2}$$.

(a) Write this as a matrix.

(b) Verify it is a valid density matrix.

(c) Calculate its von Neumann entropy.

---

### Problem 10 (M)
Show that the maximally mixed state $$\rho = I/d$$ in dimension $$d$$ can be written as an equal mixture of any orthonormal basis $$\{|i\rangle\}_{i=1}^d$$:

$$\rho = \frac{1}{d}\sum_{i=1}^d |i\rangle\langle i|$$

---

### Problem 11 (H)
Two experimentalists prepare ensembles:

- Alice: $$|0\rangle$$ with prob $$1/2$$, $$|1\rangle$$ with prob $$1/2$$
- Bob: $$|+\rangle$$ with prob $$1/2$$, $$|-\rangle$$ with prob $$1/2$$

(a) Show that both ensembles produce the same density matrix.

(b) Is there any measurement that can distinguish Alice's preparation from Bob's?

(c) Explain why this non-uniqueness is fundamentally different from classical probability.

---

### Problem 12 (M)
Consider the mixed state:

$$\rho = \frac{1}{4}|0\rangle\langle 0| + \frac{3}{4}|1\rangle\langle 1|$$

(a) Calculate the purity.

(b) Calculate the von Neumann entropy.

(c) Find the Bloch vector.

---

### Problem 13 (M)
A state has von Neumann entropy $$S = 0.5$$ bits.

(a) What are the possible eigenvalues for a qubit with this entropy?

(b) Write a general form for such a density matrix.

(c) What is the purity of this state?

---

### Problem 14 (H)
Prove that for any density matrix $$\rho$$:

$$S(\rho) \leq \log_2(\text{rank}(\rho))$$

with equality if and only if $$\rho$$ is proportional to a projector onto its support.

---

## Section C: Bloch Sphere Representation (Problems 15-20)

### Problem 15 (E)
Compute the Bloch vector for each state:

(a) $$|0\rangle$$

(b) $$|1\rangle$$

(c) $$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

(d) $$|+i\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$$

---

### Problem 16 (M)
The Bloch vector for a state is $$\vec{r} = (1/2, 0, \sqrt{3}/2)$$.

(a) Write the density matrix.

(b) Is this a pure or mixed state?

(c) Calculate the probability of measuring $$|0\rangle$$.

(d) Find the eigenvalues of $$\rho$$.

---

### Problem 17 (M)
Derive the formula for purity in terms of the Bloch vector:

$$\text{Tr}(\rho^2) = \frac{1}{2}(1 + |\vec{r}|^2)$$

---

### Problem 18 (M)
A qubit is prepared in the state with Bloch vector $$\vec{r} = (0.6, 0, 0.8)$$.

(a) Calculate the purity.

(b) Can this state be written as a pure state $$|\psi\rangle$$? If so, find $$|\psi\rangle$$.

(c) What is the probability of measuring spin-up along the x-axis?

---

### Problem 19 (H)
The Hadamard gate $$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$ acts on a state with Bloch vector $$\vec{r}$$.

(a) Find the transformation of the Bloch vector under $$H$$.

(b) Verify your answer for the specific case $$\vec{r} = (0, 0, 1)$$.

(c) Show that $$H$$ corresponds to a $$180°$$ rotation about which axis?

---

### Problem 20 (H)
A general single-qubit unitary can be written as $$U = e^{i\alpha}e^{-i\theta(\hat{n}\cdot\vec{\sigma})/2}$$ for some axis $$\hat{n}$$ and angle $$\theta$$.

(a) Show that the global phase $$e^{i\alpha}$$ has no effect on the Bloch vector.

(b) Prove that the Bloch vector rotates by angle $$\theta$$ about axis $$\hat{n}$$.

(c) For $$U = \begin{pmatrix} e^{-i\pi/8} & 0 \\ 0 & e^{i\pi/8} \end{pmatrix}$$, find the rotation axis and angle.

---

## Section D: Trace Distance and Fidelity (Problems 21-24)

### Problem 21 (M)
Compute the trace distance between:

(a) $$\rho = |0\rangle\langle 0|$$ and $$\sigma = |1\rangle\langle 1|$$

(b) $$\rho = |0\rangle\langle 0|$$ and $$\sigma = |+\rangle\langle +|$$

(c) $$\rho = |0\rangle\langle 0|$$ and $$\sigma = \frac{I}{2}$$

---

### Problem 22 (M)
Compute the fidelity between:

(a) Two identical pure states $$|\psi\rangle$$ and $$|\psi\rangle$$

(b) $$|0\rangle$$ and $$|+\rangle$$

(c) $$|0\rangle$$ and the maximally mixed state $$I/2$$

---

### Problem 23 (H)
For two qubit states with Bloch vectors $$\vec{r}_1$$ and $$\vec{r}_2$$:

(a) Prove that the trace distance is $$D = \frac{1}{2}|\vec{r}_1 - \vec{r}_2|$$.

(b) Find the maximum trace distance between a pure state and the maximally mixed state.

(c) For what states is the trace distance maximized?

---

### Problem 24 (H)
Prove the Fuchs-van de Graaf inequalities:

$$1 - \sqrt{F(\rho, \sigma)} \leq D(\rho, \sigma) \leq \sqrt{1 - F(\rho, \sigma)}$$

For which pairs of states are the bounds saturated?

---

## Section E: Partial Trace (Problems 25-28)

### Problem 25 (M)
For the two-qubit state $$|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:

(a) Write the density matrix $$\rho_{AB}$$.

(b) Compute the reduced density matrix $$\rho_A = \text{Tr}_B(\rho_{AB})$$.

(c) Compute the reduced density matrix $$\rho_B = \text{Tr}_A(\rho_{AB})$$.

(d) Is $$\rho_{AB} = \rho_A \otimes \rho_B$$? What does this tell you about entanglement?

---

### Problem 26 (M)
Consider the product state $$|\psi\rangle = |+\rangle \otimes |0\rangle$$.

(a) Write the full density matrix $$\rho_{AB}$$.

(b) Compute both reduced density matrices.

(c) Verify that $$\rho_{AB} = \rho_A \otimes \rho_B$$.

---

### Problem 27 (H)
For the state $$|\psi\rangle = \alpha|00\rangle + \beta|01\rangle + \gamma|10\rangle + \delta|11\rangle$$:

(a) Write $$\rho_A$$ in terms of $$\alpha, \beta, \gamma, \delta$$.

(b) Under what conditions is $$\rho_A$$ pure?

(c) Prove that $$\rho_A$$ is pure if and only if $$|\psi\rangle$$ is a product state.

---

### Problem 28 (H)
A three-qubit system is in state:

$$|\text{GHZ}\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

(a) Compute $$\rho_{AB} = \text{Tr}_C(\rho_{ABC})$$.

(b) Compute $$\rho_A = \text{Tr}_{BC}(\rho_{ABC})$$.

(c) Calculate the von Neumann entropy $$S(\rho_A)$$ and $$S(\rho_{AB})$$.

---

## Section F: Advanced Topics (Problems 29-30)

### Problem 29 (H)
**Purification Problem**

Given the mixed state:

$$\rho_A = \frac{2}{3}|0\rangle\langle 0| + \frac{1}{3}|1\rangle\langle 1|$$

(a) Find a purification $$|\Psi\rangle_{AB}$$ such that $$\rho_A = \text{Tr}_B(|\Psi\rangle\langle\Psi|)$$.

(b) Is the purification unique? If not, characterize all possible purifications.

(c) Find the Schmidt decomposition of your purification.

---

### Problem 30 (H)
**Extremality and Convexity**

(a) Prove that pure states are the only extremal points of the convex set of density matrices.

(b) Show that any density matrix can be written as a convex combination of at most $$d^2$$ pure states, where $$d$$ is the Hilbert space dimension.

(c) For a qubit, show geometrically that any mixed state lies on a line segment connecting two pure states on the Bloch sphere surface.

---

## Bonus Challenge Problems

### Bonus 1 (Research Level)
The quantum relative entropy is defined as:

$$S(\rho || \sigma) = \text{Tr}(\rho \log \rho) - \text{Tr}(\rho \log \sigma)$$

(a) Show that $$S(\rho || \sigma) \geq 0$$ with equality iff $$\rho = \sigma$$ (Klein's inequality).

(b) Is $$S(\rho || \sigma)$$ symmetric in its arguments?

(c) Does the triangle inequality hold for relative entropy?

---

### Bonus 2 (Research Level)
For a bipartite state $$\rho_{AB}$$, the quantum mutual information is:

$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$

(a) Show that $$I(A:B) \geq 0$$.

(b) Compute $$I(A:B)$$ for a Bell state $$|\Phi^+\rangle$$.

(c) Compute $$I(A:B)$$ for a product state.

---

## Problem Set Summary

| Section | Problems | Topics |
|---------|----------|--------|
| A | 1-8 | Pure states, basic properties |
| B | 9-14 | Mixed states, ensembles, entropy |
| C | 15-20 | Bloch sphere representation |
| D | 21-24 | Trace distance, fidelity |
| E | 25-28 | Partial trace |
| F | 29-30 | Advanced topics |

**Total: 30 problems + 2 bonus problems**

---

*Complete this problem set before consulting solutions. Time yourself and identify areas needing additional review.*
