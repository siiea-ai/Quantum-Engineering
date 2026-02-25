# Week 158: Composite Systems - Oral Examination Practice

## Introduction

This document contains typical oral examination questions on composite quantum systems. Practice explaining these concepts clearly and concisely, as if to an examination committee.

---

## Conceptual Questions

### Q1: What is a tensor product and why is it used in quantum mechanics?

**Model Answer:**

The tensor product is the mathematical operation that combines two quantum systems into a single composite system. If system A has Hilbert space $$\mathcal{H}_A$$ of dimension $$d_A$$ and system B has Hilbert space $$\mathcal{H}_B$$ of dimension $$d_B$$, the composite system lives in $$\mathcal{H}_A \otimes \mathcal{H}_B$$, which has dimension $$d_A \cdot d_B$$.

**Why tensor products?**

1. **State space construction**: The tensor product provides the correct state space for multi-particle systems. Unlike classical probability where we multiply probabilities, quantum mechanics requires us to combine amplitudes.

2. **Preserving superposition**: The tensor product structure allows for superpositions that cannot be written as products - these are entangled states.

3. **Local operations**: Operators acting on one subsystem take the form $$A \otimes I$$, naturally embedding in the composite space.

**Key properties:**
- $$(|a\rangle + |b\rangle) \otimes |c\rangle = |a\rangle \otimes |c\rangle + |b\rangle \otimes |c\rangle$$ (distributive)
- $$\langle a|\otimes\langle b| (|c\rangle \otimes |d\rangle) = \langle a|c\rangle \cdot \langle b|d\rangle$$
- Dimension multiplies: $$\dim(\mathcal{H}_A \otimes \mathcal{H}_B) = d_A \cdot d_B$$

---

### Q2: What is the partial trace and what does it mean physically?

**Model Answer:**

The partial trace is an operation that extracts the reduced density matrix of a subsystem from a composite system's density matrix.

**Definition:**
For a bipartite state $$\rho_{AB}$$, the partial trace over B gives:
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B)\rho_{AB}(I_A \otimes |j\rangle_B)$$

**Key property:** For tensor products, $$\text{Tr}_B(A \otimes B) = A \cdot \text{Tr}(B)$$.

**Physical interpretation:**

1. **Describing subsystems**: When we only have access to subsystem A, the reduced density matrix $$\rho_A$$ gives the correct statistics for any measurement on A alone.

2. **"Tracing out" degrees of freedom**: We sum over all possible states of the unobserved system.

3. **Connecting to entanglement**: For a pure entangled state, the reduced density matrices are mixed. The degree of mixing quantifies entanglement.

**Example:** For the Bell state $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:
$$\rho_A = \text{Tr}_B(|\Phi^+\rangle\langle\Phi^+|) = \frac{I}{2}$$

Complete knowledge of the whole, complete ignorance of the parts - this is the essence of quantum entanglement.

---

### Q3: State and explain the Schmidt decomposition theorem.

**Model Answer:**

**Theorem Statement:**
Any bipartite pure state $$|\psi\rangle \in \mathcal{H}_A \otimes \mathcal{H}_B$$ can be written as:
$$|\psi\rangle = \sum_{i=1}^{r} \lambda_i |a_i\rangle_A |b_i\rangle_B$$

where:
- $$\lambda_i > 0$$ are the Schmidt coefficients with $$\sum_i \lambda_i^2 = 1$$
- $$\{|a_i\rangle\}$$ are orthonormal in $$\mathcal{H}_A$$
- $$\{|b_i\rangle\}$$ are orthonormal in $$\mathcal{H}_B$$
- $$r \leq \min(d_A, d_B)$$ is the Schmidt rank

**Proof idea:**
Write the state as $$|\psi\rangle = \sum_{ij} C_{ij}|ij\rangle$$ and apply SVD to the coefficient matrix $$C = U\Sigma V^\dagger$$. The singular values become Schmidt coefficients.

**Physical significance:**

1. **Entanglement characterization**: Schmidt rank 1 means product state (not entangled). Schmidt rank > 1 means entangled.

2. **Reduced density matrices**:
   $$\rho_A = \sum_i \lambda_i^2 |a_i\rangle\langle a_i|, \quad \rho_B = \sum_i \lambda_i^2 |b_i\rangle\langle b_i|$$
   Both have the same eigenvalues $$\{\lambda_i^2\}$$.

3. **Entanglement measure**: The entanglement entropy $$E = -\sum_i \lambda_i^2 \log \lambda_i^2$$ quantifies entanglement.

---

### Q4: What is purification and why is it important?

**Model Answer:**

**Definition:**
Purification is the process of representing a mixed state $$\rho_A$$ as the reduced state of a pure state on a larger system:
$$\rho_A = \text{Tr}_B(|\Psi\rangle_{AB}\langle\Psi|)$$

**Construction:**
Given $$\rho_A = \sum_i p_i |i\rangle\langle i|$$:
$$|\Psi\rangle_{AB} = \sum_i \sqrt{p_i}|i\rangle_A|i\rangle_B$$

**Key properties:**
1. Every mixed state has a purification
2. Purifications are not unique - related by unitaries on the ancilla
3. Minimum ancilla dimension = rank of $$\rho_A$$

**Why important:**

1. **Conceptual**: Mixedness can always be understood as arising from entanglement with an inaccessible environment.

2. **Proof technique**: Many quantum information inequalities are proved by considering purifications (e.g., strong subadditivity).

3. **Channel representation**: Quantum channels can be understood through Stinespring dilation - unitary evolution on a purified system followed by partial trace.

4. **Entanglement theory**: The purification of a mixed state $$\rho_A$$ has entanglement entropy equal to $$S(\rho_A)$$.

---

### Q5: How can you tell if a bipartite pure state is entangled?

**Model Answer:**

There are several equivalent methods:

**1. Schmidt decomposition:**
- Compute the Schmidt decomposition $$|\psi\rangle = \sum_i \lambda_i |a_i\rangle|b_i\rangle$$
- Schmidt rank = 1: Product state (not entangled)
- Schmidt rank > 1: Entangled

**2. Reduced density matrix purity:**
- Compute $$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$$
- If $$\text{Tr}(\rho_A^2) = 1$$: Product state
- If $$\text{Tr}(\rho_A^2) < 1$$: Entangled

**3. Factorization test:**
- Check if $$|\psi\rangle = |a\rangle \otimes |b\rangle$$ for some states
- Can be done by checking if the coefficient matrix has rank 1

**4. Concurrence (for two qubits):**
- Compute $$C(|\psi\rangle) = |\langle\psi|\tilde{\psi}\rangle|$$
- $$C = 0$$: Product state
- $$C > 0$$: Entangled

**Physical intuition:**
For a product state, each subsystem has a definite pure state. For an entangled state, knowledge of the whole doesn't give knowledge of the parts - the reduced states are necessarily mixed.

---

## Technical Questions

### Q6: Compute the partial trace of a general two-qubit density matrix.

**Model Answer:**

Consider $$\rho_{AB}$$ in the basis $$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$:

$$\rho_{AB} = \begin{pmatrix} \rho_{00,00} & \rho_{00,01} & \rho_{00,10} & \rho_{00,11} \\ \rho_{01,00} & \rho_{01,01} & \rho_{01,10} & \rho_{01,11} \\ \rho_{10,00} & \rho_{10,01} & \rho_{10,10} & \rho_{10,11} \\ \rho_{11,00} & \rho_{11,01} & \rho_{11,10} & \rho_{11,11} \end{pmatrix}$$

**Partial trace over B:**
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_{j \in \{0,1\}} (I \otimes \langle j|)\rho_{AB}(I \otimes |j\rangle)$$

$$\rho_A = \begin{pmatrix} \rho_{00,00} + \rho_{01,01} & \rho_{00,10} + \rho_{01,11} \\ \rho_{10,00} + \rho_{11,01} & \rho_{10,10} + \rho_{11,11} \end{pmatrix}$$

**Block interpretation:**
View $$\rho_{AB}$$ as $$2 \times 2$$ blocks of $$2 \times 2$$ matrices:
$$\rho_{AB} = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$

Then $$\rho_A = \begin{pmatrix} \text{Tr}(A) & \text{Tr}(B) \\ \text{Tr}(C) & \text{Tr}(D) \end{pmatrix}$$

---

### Q7: Find the Schmidt decomposition of $$|\psi\rangle = \alpha|00\rangle + \beta|11\rangle$$.

**Model Answer:**

**Step 1:** Write coefficient matrix
$$C = \begin{pmatrix} \alpha & 0 \\ 0 & \beta \end{pmatrix}$$

**Step 2:** This is already diagonal! SVD is:
$$C = I \cdot \begin{pmatrix} |\alpha| & 0 \\ 0 & |\beta| \end{pmatrix} \cdot I$$

(assuming $$\alpha, \beta$$ are real and positive; otherwise include phases in the bases)

**Step 3:** Schmidt decomposition:
$$|\psi\rangle = |\alpha| \cdot |0\rangle|0\rangle + |\beta| \cdot |1\rangle|1\rangle$$

**Schmidt coefficients:** $$\lambda_1 = |\alpha|, \lambda_2 = |\beta|$$
**Schmidt rank:** 2 (if both $$\alpha, \beta \neq 0$$)

**Reduced density matrices:**
$$\rho_A = |\alpha|^2|0\rangle\langle 0| + |\beta|^2|1\rangle\langle 1|$$
$$\rho_B = |\alpha|^2|0\rangle\langle 0| + |\beta|^2|1\rangle\langle 1|$$

**Entanglement entropy:**
$$E = -|\alpha|^2\log|\alpha|^2 - |\beta|^2\log|\beta|^2$$

Maximum when $$|\alpha| = |\beta| = 1/\sqrt{2}$$ (Bell state): $$E = 1$$ bit.

---

### Q8: Prove that $$\text{Tr}_B(A \otimes B) = A \cdot \text{Tr}(B)$$.

**Model Answer:**

**Method 1: Direct calculation**

$$\text{Tr}_B(A \otimes B) = \sum_j (I_A \otimes \langle j|_B)(A \otimes B)(I_A \otimes |j\rangle_B)$$

$$= \sum_j A \otimes (\langle j|B|j\rangle)$$

$$= A \otimes \sum_j \langle j|B|j\rangle$$

$$= A \otimes \text{Tr}(B)$$

$$= A \cdot \text{Tr}(B)$$

(The last step uses that $$A \otimes c = cA$$ for scalar $$c$$.)

**Method 2: Matrix elements**

For basis states:
$$\langle i|[\text{Tr}_B(A \otimes B)]|k\rangle = \sum_j \langle ij|(A \otimes B)|kj\rangle$$

$$= \sum_j \langle i|A|k\rangle \langle j|B|j\rangle = \langle i|A|k\rangle \cdot \text{Tr}(B)$$

$$= \langle i|[A \cdot \text{Tr}(B)]|k\rangle$$

---

### Q9: Explain the connection between Schmidt decomposition and SVD.

**Model Answer:**

**Setting up the connection:**

Write a bipartite state $$|\psi\rangle = \sum_{i,j} C_{ij}|i\rangle_A|j\rangle_B$$ with coefficient matrix $$C$$.

**SVD of C:**
$$C = U \Sigma V^\dagger$$

where:
- $$U$$ is $$d_A \times d_A$$ unitary (left singular vectors)
- $$\Sigma$$ is $$d_A \times d_B$$ diagonal with non-negative entries $$\sigma_k$$
- $$V$$ is $$d_B \times d_B$$ unitary (right singular vectors)

**Constructing Schmidt bases:**

Define new bases:
$$|a_k\rangle = \sum_i U_{ik}^* |i\rangle_A$$
$$|b_k\rangle = \sum_j V_{jk}^* |j\rangle_B$$

These are orthonormal because $$U$$ and $$V$$ are unitary.

**Schmidt decomposition:**
$$|\psi\rangle = \sum_k \sigma_k |a_k\rangle|b_k\rangle$$

The Schmidt coefficients are exactly the singular values $$\{\sigma_k\}$$.

**Key insight:**
- Singular values of $$C$$ = Schmidt coefficients
- Number of non-zero singular values = Schmidt rank
- Left singular vectors define $$|a_k\rangle$$
- Right singular vectors define $$|b_k\rangle$$

---

### Q10: How does the Schmidt rank relate to entanglement?

**Model Answer:**

**Direct relationship:**

- **Schmidt rank = 1**: The state can be written as $$|\psi\rangle = |a\rangle|b\rangle$$, a product state. **Not entangled.**

- **Schmidt rank > 1**: The state cannot be written as a single product. **Entangled.**

- **Schmidt rank = $$\min(d_A, d_B)$$**: With equal coefficients, this is **maximally entangled**.

**Quantitative connections:**

1. **Entanglement entropy:** $$E = -\sum_i \lambda_i^2 \log \lambda_i^2$$
   - Bounded: $$0 \leq E \leq \log(\text{Schmidt rank})$$
   - Maximum for uniform $$\lambda_i = 1/\sqrt{r}$$

2. **Schmidt number:** The Schmidt rank itself is sometimes called the Schmidt number, a discrete entanglement measure.

3. **Entanglement witnesses:** Schmidt rank > $$k$$ can be detected by certain observables.

**Physical intuition:**
Higher Schmidt rank means the state is a more "complex" superposition of product states. More terms in the Schmidt sum = more ways the subsystems are correlated = more entanglement.

**Important caveat:**
The Schmidt decomposition only works for **pure bipartite** states. For mixed states, we need different separability criteria (PPT, etc.).

---

## Calculation Questions

### Q11: Find the Schmidt decomposition of $$|\psi\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|10\rangle$$.

**Model Answer:**

**Step 1:** Coefficient matrix
$$C = \begin{pmatrix} 1/\sqrt{2} & 1/2 \\ 1/2 & 0 \end{pmatrix}$$

**Step 2:** Compute $$CC^\dagger$$
$$CC^\dagger = \begin{pmatrix} 1/2 + 1/4 & 1/(2\sqrt{2}) \\ 1/(2\sqrt{2}) & 1/4 \end{pmatrix} = \begin{pmatrix} 3/4 & 1/(2\sqrt{2}) \\ 1/(2\sqrt{2}) & 1/4 \end{pmatrix}$$

**Step 3:** Find eigenvalues of $$CC^\dagger$$ (= squares of Schmidt coefficients)
$$\det(CC^\dagger - \mu I) = (\frac{3}{4} - \mu)(\frac{1}{4} - \mu) - \frac{1}{8}$$
$$= \mu^2 - \mu + \frac{3}{16} - \frac{2}{16} = \mu^2 - \mu + \frac{1}{16}$$
$$\mu = \frac{1 \pm \sqrt{1 - 1/4}}{2} = \frac{1 \pm \sqrt{3}/2}{2}$$

$$\mu_1 = \frac{2 + \sqrt{3}}{4} \approx 0.933, \quad \mu_2 = \frac{2 - \sqrt{3}}{4} \approx 0.067$$

**Step 4:** Schmidt coefficients
$$\lambda_1 = \sqrt{\mu_1} \approx 0.966, \quad \lambda_2 = \sqrt{\mu_2} \approx 0.259$$

**Schmidt rank = 2** (entangled)

---

### Q12: Construct a purification of $$\rho = 0.6|+\rangle\langle+| + 0.4|-\rangle\langle-|$$.

**Model Answer:**

**Step 1:** State is already in spectral form with eigenstates $$|+\rangle, |-\rangle$$ and eigenvalues $$0.6, 0.4$$.

**Step 2:** Purification:
$$|\Psi\rangle = \sqrt{0.6}|+\rangle|0\rangle + \sqrt{0.4}|-\rangle|1\rangle$$

**Verification:**
$$\text{Tr}_B(|\Psi\rangle\langle\Psi|) = 0.6|+\rangle\langle+| + 0.4|-\rangle\langle-| = \rho$$ ✓

**Schmidt coefficients:** $$\sqrt{0.6}, \sqrt{0.4}$$

**Entanglement entropy:**
$$S = -0.6\log_2(0.6) - 0.4\log_2(0.4)$$
$$= 0.6 \times 0.737 + 0.4 \times 1.322 = 0.442 + 0.529 = 0.971$$ bits

---

## Tips for Oral Exams

1. **Start with definitions**: Clearly state the mathematical definition before discussing properties.

2. **Give examples**: Bell states are the canonical examples for entanglement; use them liberally.

3. **Connect concepts**: Link Schmidt decomposition to SVD, partial trace to reduced states, purification to entanglement.

4. **Draw diagrams**: Sketch tensor product structures as boxes with wires.

5. **Check dimensions**: Always verify that dimensions match in your calculations.

6. **Physical interpretation**: Every mathematical concept has a physical meaning - explain it.

---

*Practice these questions until you can answer them fluently. Record yourself and review for clarity.*
