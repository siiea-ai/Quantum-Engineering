# Week 157: Density Matrices - Oral Examination Practice

## Introduction

This document contains typical oral examination questions on density matrices. Practice answering these questions aloud, as if speaking to an examination committee. Time yourself - aim for clear, complete answers in 3-5 minutes each.

---

## Conceptual Questions

### Q1: What is a density matrix and why do we need it?

**Model Answer:**

A density matrix (or density operator) is the most general way to describe the state of a quantum system. While a pure quantum state can be described by a state vector $$|\psi\rangle$$, the density matrix formalism extends this to handle:

1. **Statistical mixtures**: When we have incomplete knowledge about which pure state a system is in, we describe it as a probabilistic ensemble.

2. **Subsystems of entangled states**: When part of a larger quantum system is entangled with an environment, the subsystem alone cannot be described by any pure state.

3. **Open quantum systems**: Systems interacting with environments naturally evolve into mixed states through decoherence.

Mathematically, a density matrix $$\rho$$ is a positive semi-definite, Hermitian operator with unit trace. For a pure state, $$\rho = |\psi\rangle\langle\psi|$$, and for a mixed state, $$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$ where $$p_i$$ are classical probabilities.

The key advantage is that expectation values of any observable $$A$$ are given by the trace formula $$\langle A \rangle = \text{Tr}(\rho A)$$, regardless of whether the state is pure or mixed.

---

### Q2: How do you determine if a density matrix represents a pure or mixed state?

**Model Answer:**

There are several equivalent criteria:

**1. Purity test:**
Calculate $$\gamma = \text{Tr}(\rho^2)$$. For a pure state, $$\gamma = 1$$. For a mixed state, $$\gamma < 1$$.

**2. Idempotency:**
A pure state satisfies $$\rho^2 = \rho$$. This is because pure states are projection operators.

**3. Rank:**
A pure state has rank 1 (exactly one non-zero eigenvalue, which equals 1). Mixed states have rank greater than 1.

**4. Von Neumann entropy:**
$$S(\rho) = -\text{Tr}(\rho \log \rho) = 0$$ for pure states, $$S > 0$$ for mixed states.

**5. Bloch vector (for qubits):**
Write $$\rho = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$$. Pure states have $$|\vec{r}| = 1$$ (on the Bloch sphere surface), mixed states have $$|\vec{r}| < 1$$ (inside the ball).

In practice, calculating the purity $$\text{Tr}(\rho^2)$$ is usually the quickest check.

---

### Q3: Explain the Bloch sphere representation for qubits.

**Model Answer:**

The Bloch sphere is a geometric representation of qubit states. Any qubit density matrix can be written as:

$$\rho = \frac{1}{2}(I + r_x\sigma_x + r_y\sigma_y + r_z\sigma_z) = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$$

The Bloch vector $$\vec{r} = (r_x, r_y, r_z)$$ satisfies $$|\vec{r}| \leq 1$$ from positivity of $$\rho$$.

**Geometric interpretation:**
- The unit sphere ($$|\vec{r}| = 1$$) represents pure states
- The interior ($$|\vec{r}| < 1$$) represents mixed states
- The origin ($$\vec{r} = 0$$) is the maximally mixed state $$I/2$$

**Important pure states:**
- $$|0\rangle$$: north pole $$(0,0,1)$$
- $$|1\rangle$$: south pole $$(0,0,-1)$$
- $$|+\rangle$$: $$(1,0,0)$$
- $$|-\rangle$$: $$(-1,0,0)$$
- $$|+i\rangle$$: $$(0,1,0)$$
- $$|-i\rangle$$: $$(0,-1,0)$$

**Physical meaning:**
- $$r_z$$ relates to $$\langle\sigma_z\rangle$$, the expected measurement in the computational basis
- Unitary operations correspond to rotations of the Bloch vector
- Decoherence contracts the Bloch vector toward the origin

---

### Q4: What is the partial trace and why is it important?

**Model Answer:**

The partial trace is the mathematical operation that describes a subsystem of a larger quantum system. Given a bipartite state $$\rho_{AB}$$, the reduced density matrix of system A is:

$$\rho_A = \text{Tr}_B(\rho_{AB})$$

This is computed by "tracing out" the B degrees of freedom:
$$\rho_A = \sum_j (I_A \otimes \langle j|_B)\rho_{AB}(I_A \otimes |j\rangle_B)$$

**Why it's important:**

1. **Describing subsystems**: When we only have access to part of a quantum system, the partial trace gives us the correct description for local observations.

2. **Detecting entanglement**: For a product state $$\rho_{AB} = \rho_A \otimes \rho_B$$, the reduced states are pure if the global state is pure. For entangled states, reduced states are mixed even when the global state is pure.

3. **Quantum channels**: The partial trace arises naturally in describing open quantum systems where we trace over environmental degrees of freedom.

**Key example:**
For the Bell state $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:
$$\rho_A = \text{Tr}_B(|\Phi^+\rangle\langle\Phi^+|) = \frac{I}{2}$$

The reduced state is maximally mixed, reflecting that we have complete knowledge of the joint system but complete ignorance about the subsystem - a hallmark of entanglement.

---

### Q5: Explain the difference between a superposition and a mixture.

**Model Answer:**

This is a fundamental distinction in quantum mechanics:

**Superposition (Pure State):**
$$|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

Density matrix: $$\rho = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

The off-diagonal elements (coherences) are non-zero. The state exhibits quantum interference effects.

**Mixture (Mixed State):**
Equal classical mixture of $$|0\rangle$$ and $$|1\rangle$$ with probabilities 1/2 each.

Density matrix: $$\rho = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

No off-diagonal elements. No interference effects.

**Key differences:**

1. **Coherences**: Superpositions have off-diagonal density matrix elements; mixtures don't (in the mixture's eigenbasis).

2. **Interference**: Superpositions show interference in measurements; mixtures don't.

3. **Purity**: Superpositions are pure ($$\text{Tr}(\rho^2) = 1$$); mixtures are not.

4. **Bloch vector**: The superposition $$|+\rangle$$ has $$\vec{r} = (1,0,0)$$; the mixture has $$\vec{r} = (0,0,0)$$.

5. **State preparation**: Superpositions arise from coherent quantum operations; mixtures arise from classical randomness or decoherence.

---

## Technical Questions

### Q6: Derive the formula for trace distance between two qubit states in terms of their Bloch vectors.

**Model Answer:**

Start with the Bloch representation:
$$\rho_1 = \frac{1}{2}(I + \vec{r}_1\cdot\vec{\sigma}), \quad \rho_2 = \frac{1}{2}(I + \vec{r}_2\cdot\vec{\sigma})$$

Their difference:
$$\rho_1 - \rho_2 = \frac{1}{2}(\vec{r}_1 - \vec{r}_2)\cdot\vec{\sigma} = \frac{1}{2}\vec{d}\cdot\vec{\sigma}$$

where $$\vec{d} = \vec{r}_1 - \vec{r}_2$$.

The eigenvalues of $$\hat{n}\cdot\vec{\sigma}$$ for unit vector $$\hat{n}$$ are $$\pm 1$$, so eigenvalues of $$\vec{d}\cdot\vec{\sigma}$$ are $$\pm|\vec{d}|$$.

The trace distance is:
$$D(\rho_1, \rho_2) = \frac{1}{2}\text{Tr}|\rho_1 - \rho_2| = \frac{1}{2} \cdot \frac{1}{2}(|\vec{d}| + |-\vec{d}|) = \frac{1}{2}|\vec{d}|$$

$$\boxed{D(\rho_1, \rho_2) = \frac{1}{2}|\vec{r}_1 - \vec{r}_2|}$$

This shows the trace distance is half the Euclidean distance between Bloch vectors.

---

### Q7: Prove that the von Neumann entropy is zero if and only if the state is pure.

**Model Answer:**

$$S(\rho) = -\text{Tr}(\rho \log \rho) = -\sum_i \lambda_i \log \lambda_i$$

where $$\{\lambda_i\}$$ are eigenvalues of $$\rho$$ with $$\lambda_i \geq 0$$ and $$\sum_i \lambda_i = 1$$.

**If $$\rho$$ is pure:**
Pure states have exactly one eigenvalue equal to 1 and the rest equal to 0.
$$S = -1 \cdot \log 1 - 0 \cdot \log 0 - \ldots = 0$$
(using convention $$0 \log 0 = 0$$)

**If $$S = 0$$:**
The function $$f(x) = -x \log x$$ is non-negative for $$x \in [0,1]$$ and equals zero only at $$x = 0$$ or $$x = 1$$.

$$S = \sum_i f(\lambda_i) = 0$$

requires each term $$f(\lambda_i) = 0$$, so each $$\lambda_i \in \{0, 1\}$$.

Since $$\sum_i \lambda_i = 1$$, exactly one eigenvalue is 1 and the rest are 0. This means $$\rho$$ is a rank-1 projector, i.e., a pure state.

---

### Q8: Show that a density matrix can always be diagonalized by a unitary transformation.

**Model Answer:**

This follows from the spectral theorem for Hermitian operators.

**Key properties of density matrices:**
1. $$\rho = \rho^\dagger$$ (Hermitian)
2. $$\rho \geq 0$$ (positive semi-definite)
3. $$\text{Tr}(\rho) = 1$$

**Spectral theorem:**
Any Hermitian operator can be diagonalized:
$$\rho = U D U^\dagger$$

where $$U$$ is unitary and $$D$$ is diagonal with real entries.

**For density matrices specifically:**
$$D = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$$

where:
- $$\lambda_i \geq 0$$ (from positive semi-definiteness)
- $$\sum_i \lambda_i = 1$$ (from unit trace, since trace is invariant under unitary transformation)

**Physical interpretation:**
The eigenvectors of $$\rho$$ form an orthonormal basis in which $$\rho$$ is diagonal. The eigenvalues are the probabilities of finding the system in each eigenstate - this is the unique ensemble decomposition with orthogonal states.

---

### Q9: Explain why the same density matrix can arise from different ensemble preparations.

**Model Answer:**

This non-uniqueness is a fundamental feature of quantum mechanics that distinguishes it from classical probability.

**Mathematical reason:**
A density matrix contains less information than a full specification of an ensemble. Given $$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$:

- The ensemble is specified by $$\{(p_i, |\psi_i\rangle)\}$$ - potentially infinitely many numbers
- The density matrix has $$d^2$$ real parameters (for dimension $$d$$)

**Example: Maximally mixed qubit**
$$\rho = \frac{I}{2}$$

Can be prepared as:
1. Equal mixture of $$|0\rangle$$ and $$|1\rangle$$
2. Equal mixture of $$|+\rangle$$ and $$|-\rangle$$
3. Equal mixture of $$|+i\rangle$$ and $$|-i\rangle$$
4. Uniform mixture over all pure states (continuous)

All give identical $$\rho$$ and identical measurement statistics.

**Physical implication:**
No measurement can distinguish between these preparations. The density matrix captures all operationally accessible information about a quantum state. The specific preparation history is irretrievable - this is sometimes called "preparation non-contextuality."

**Connection to purification:**
All ensembles giving the same $$\rho$$ arise from tracing out different purifications related by unitaries on the purifying system.

---

### Q10: What are trace distance and fidelity, and how are they related?

**Model Answer:**

Both are measures of distinguishability between quantum states.

**Trace Distance:**
$$D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma|$$

Properties:
- $$0 \leq D \leq 1$$
- $$D = 0$$ iff $$\rho = \sigma$$
- $$D = 1$$ for orthogonal pure states
- Operational meaning: Maximum probability of correctly distinguishing states in a single measurement is $$\frac{1}{2}(1 + D)$$

**Fidelity:**
$$F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

For pure states: $$F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2$$

Properties:
- $$0 \leq F \leq 1$$
- $$F = 1$$ iff $$\rho = \sigma$$
- $$F = 0$$ for orthogonal states

**Fuchs-van de Graaf inequalities:**
$$1 - \sqrt{F} \leq D \leq \sqrt{1 - F}$$

These show that high fidelity implies small trace distance and vice versa. The bounds are tight for pure states.

**When to use which:**
- Trace distance is a true metric (satisfies triangle inequality)
- Fidelity has nicer multiplicativity properties for tensor products
- Both are essential tools in quantum information theory

---

## Calculation Questions

### Q11: Calculate the density matrix, Bloch vector, purity, and entropy for the state $$|\psi\rangle = \sqrt{0.8}|0\rangle + \sqrt{0.2}|1\rangle$$.

**Model Answer:**

**Density matrix:**
$$\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} 0.8 & \sqrt{0.16} \\ \sqrt{0.16} & 0.2 \end{pmatrix} = \begin{pmatrix} 0.8 & 0.4 \\ 0.4 & 0.2 \end{pmatrix}$$

**Bloch vector:**
$$r_x = \text{Tr}(\rho\sigma_x) = 2 \times 0.4 = 0.8$$
$$r_y = 0$$ (no imaginary off-diagonal)
$$r_z = \text{Tr}(\rho\sigma_z) = 0.8 - 0.2 = 0.6$$

$$\vec{r} = (0.8, 0, 0.6)$$

Check: $$|\vec{r}|^2 = 0.64 + 0 + 0.36 = 1$$ ✓ (pure state)

**Purity:**
$$\gamma = \text{Tr}(\rho^2) = \frac{1}{2}(1 + |\vec{r}|^2) = \frac{1}{2}(1 + 1) = 1$$

Or directly: $$\gamma = 1$$ for pure states.

**Entropy:**
$$S(\rho) = 0$$ for pure states.

---

### Q12: A qubit in state $$|0\rangle$$ undergoes dephasing, becoming $$\rho = (1-p)|0\rangle\langle 0| + p\frac{I}{2}$$. Find the Bloch vector and entropy as functions of $$p$$.

**Model Answer:**

**Density matrix:**
$$\rho = (1-p)\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + \frac{p}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 - p/2 & 0 \\ 0 & p/2 \end{pmatrix}$$

**Bloch vector:**
$$r_x = 0, \quad r_y = 0, \quad r_z = (1-p/2) - p/2 = 1 - p$$

$$\vec{r} = (0, 0, 1-p)$$

The state moves from north pole toward center as $$p$$ increases.

**Eigenvalues:**
$$\lambda_1 = 1 - p/2, \quad \lambda_2 = p/2$$

**Entropy:**
$$S = -\lambda_1\log_2\lambda_1 - \lambda_2\log_2\lambda_2$$
$$= -(1-p/2)\log_2(1-p/2) - (p/2)\log_2(p/2)$$

At $$p = 0$$: $$S = 0$$ (pure state $$|0\rangle$$)
At $$p = 1$$: $$S = 1$$ bit (maximally mixed)

---

## Challenging Questions

### Q13: Explain the relationship between entanglement and the purity of reduced density matrices.

**Model Answer:**

For a bipartite pure state $$|\psi\rangle_{AB}$$, entanglement is directly connected to how mixed the reduced states are.

**Key insight:**
For a pure bipartite state, if and only if the state is a product state (not entangled), the reduced density matrices are pure.

**Quantitatively:**
The entanglement entropy is defined as:
$$E(|\psi\rangle_{AB}) = S(\rho_A) = S(\rho_B)$$

This is zero for product states and maximal ($$\log_2 d$$) for maximally entangled states.

**Physical interpretation:**
- Product state: Each subsystem has a definite pure state
- Entangled state: Subsystem states are "smeared out" - we have complete knowledge of the joint system but incomplete knowledge of each part

**Schmidt decomposition connection:**
$$|\psi\rangle = \sum_i \sqrt{\lambda_i}|i\rangle_A|i\rangle_B$$

$$\rho_A = \sum_i \lambda_i |i\rangle\langle i|$$

The Schmidt coefficients directly determine the purity:
$$\gamma_A = \sum_i \lambda_i^2$$

More entanglement means more uniform $$\lambda_i$$, meaning lower purity and higher entropy.

---

### Q14: Derive the conditions for a 2x2 matrix to be a valid density matrix.

**Model Answer:**

A general $$2 \times 2$$ Hermitian matrix:
$$\rho = \begin{pmatrix} a & b \\ b^* & c \end{pmatrix}$$

with $$a, c$$ real and $$b$$ complex.

**Condition 1: Unit trace**
$$\text{Tr}(\rho) = a + c = 1$$

So $$c = 1 - a$$.

**Condition 2: Positive semi-definiteness**
Both eigenvalues must be non-negative.

Eigenvalues: $$\lambda_\pm = \frac{1}{2}\left((a+c) \pm \sqrt{(a-c)^2 + 4|b|^2}\right)$$

With $$a + c = 1$$:
$$\lambda_\pm = \frac{1}{2}\left(1 \pm \sqrt{(2a-1)^2 + 4|b|^2}\right)$$

For $$\lambda_- \geq 0$$:
$$1 \geq \sqrt{(2a-1)^2 + 4|b|^2}$$

$$(2a-1)^2 + 4|b|^2 \leq 1$$

**Summary of constraints:**
1. $$a + c = 1$$, $$a, c \in \mathbb{R}$$
2. $$(2a-1)^2 + 4|b|^2 \leq 1$$

In Bloch form with $$a = (1+r_z)/2$$, $$b = (r_x - ir_y)/2$$:
$$r_x^2 + r_y^2 + r_z^2 \leq 1$$

This is exactly the Bloch ball condition.

---

### Q15: Explain purification and give an example.

**Model Answer:**

**Definition:**
Purification is the process of representing a mixed state $$\rho_A$$ as the reduced state of a pure state on a larger system:
$$\rho_A = \text{Tr}_B(|\Psi\rangle_{AB}\langle\Psi|)$$

**Existence:**
Every mixed state has a purification. If $$\rho_A = \sum_i \lambda_i |i\rangle\langle i|$$, then:
$$|\Psi\rangle_{AB} = \sum_i \sqrt{\lambda_i}|i\rangle_A|i\rangle_B$$

is a purification (in Schmidt form).

**Example:**
Mixed state: $$\rho_A = \frac{3}{4}|0\rangle\langle 0| + \frac{1}{4}|1\rangle\langle 1|$$

Purification:
$$|\Psi\rangle_{AB} = \frac{\sqrt{3}}{2}|0\rangle_A|0\rangle_B + \frac{1}{2}|1\rangle_A|1\rangle_B$$

**Non-uniqueness:**
Purifications are not unique. Given one purification $$|\Psi\rangle$$, all others have the form:
$$|\Psi'\rangle = (I_A \otimes U_B)|\Psi\rangle$$

for some unitary $$U_B$$ on the ancilla.

**Physical interpretation:**
Mixedness can always be thought of as arising from entanglement with an inaccessible environment. The environment can be as small as needed (dimension equal to rank of $$\rho_A$$).

**Application:**
Purification is central to many quantum information protocols and proofs, including the proof that LOCC cannot increase entanglement.

---

## Tips for Oral Exams

1. **Start with the big picture**: Before diving into equations, briefly explain the physical motivation.

2. **Use diagrams**: Sketch the Bloch sphere when discussing qubit states.

3. **Check your work**: Verify properties (trace, positivity) as you go.

4. **Mention connections**: Link concepts (e.g., "This connects to entanglement because...")

5. **Handle mistakes gracefully**: If you make an error, acknowledge it and correct it.

6. **Ask clarifying questions**: If a question is ambiguous, ask what the examiner is looking for.

7. **Know the limits of your knowledge**: It's better to say "I'm not sure, but I think..." than to confidently state something wrong.

---

*Practice these questions until you can answer them fluently without notes.*
