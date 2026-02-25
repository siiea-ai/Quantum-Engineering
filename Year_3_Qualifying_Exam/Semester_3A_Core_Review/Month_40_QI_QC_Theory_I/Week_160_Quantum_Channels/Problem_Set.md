# Week 160: Quantum Channels - Problem Set

## Instructions

This problem set contains 26 qualifying exam-style problems on quantum channels. Focus on understanding both the mathematical formalism and physical interpretation.

---

## Section A: CPTP Maps and Kraus Representation (Problems 1-8)

### Problem 1 (E)
Verify that the identity channel $$\mathcal{I}(\rho) = \rho$$ is CPTP by:
(a) Writing its Kraus representation
(b) Verifying the completeness relation

---

### Problem 2 (E)
For a unitary evolution $$\mathcal{U}(\rho) = U\rho U^\dagger$$:
(a) What is the Kraus representation?
(b) Verify the completeness relation.
(c) Is this channel unital?

---

### Problem 3 (M)
The transpose map $$T(\rho) = \rho^T$$ is positive but not completely positive.
(a) Show it is positive.
(b) Show it is not completely positive by applying $$(T \otimes \mathcal{I})$$ to a Bell state.

---

### Problem 4 (M)
Given Kraus operators $$K_0 = |0\rangle\langle 0|$$ and $$K_1 = |0\rangle\langle 1|$$:
(a) Check if they satisfy the completeness relation.
(b) If not, can you modify them to create a valid channel?
(c) What does this channel physically represent?

---

### Problem 5 (M)
Derive Kraus operators from the system-environment model:
- System starts in $$\rho_S$$, environment in $$|0\rangle_E$$
- Joint unitary: $$U = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$
- Trace out environment

---

### Problem 6 (H)
Prove that if $$\{K_k\}$$ and $$\{L_j\}$$ are two Kraus representations of the same channel, then:
$$L_j = \sum_k U_{jk} K_k$$
for some unitary matrix $$U$$.

---

### Problem 7 (H)
Show that for any CPTP map, the number of Kraus operators can be chosen to be at most $$d^2$$, where $$d$$ is the dimension of the input space.

---

### Problem 8 (H)
A map is **unital** if $$\mathcal{E}(I) = I$$. Show that for a unital channel:
$$\sum_k K_k K_k^\dagger = I$$
in addition to the usual completeness relation.

---

## Section B: Depolarizing Channel (Problems 9-12)

### Problem 9 (E)
For the depolarizing channel with parameter $$p$$:
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

(a) Write the Kraus operators.
(b) Verify the completeness relation.
(c) Apply to $$|0\rangle\langle 0|$$.

---

### Problem 10 (M)
Show that the depolarizing channel can be rewritten as:
$$\mathcal{E}_p(\rho) = \left(1-\frac{4p}{3}\right)\rho + \frac{p}{3}I$$

For what value of $$p$$ does the output become maximally mixed?

---

### Problem 11 (M)
Compute the effect of the depolarizing channel on the Bloch vector:
$$\rho = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$$

Show that $$\vec{r} \to (1-\frac{4p}{3})\vec{r}$$.

---

### Problem 12 (H)
Two depolarizing channels with parameters $$p_1$$ and $$p_2$$ are applied in sequence.
(a) Find the Kraus operators for the composite channel.
(b) Show the composite is also a depolarizing channel.
(c) Find the effective parameter $$p_{\text{eff}}$$.

---

## Section C: Amplitude Damping (Problems 13-17)

### Problem 13 (E)
For the amplitude damping channel with Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

(a) Verify the completeness relation.
(b) Apply to $$|1\rangle\langle 1|$$.
(c) Apply to $$|0\rangle\langle 0|$$.

---

### Problem 14 (M)
Compute the effect of amplitude damping on a general qubit state:
$$\rho = \begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix}$$

---

### Problem 15 (M)
For amplitude damping, find the fixed point: a state $$\rho^*$$ such that $$\mathcal{E}(\rho^*) = \rho^*$$.

---

### Problem 16 (H)
Derive the Kraus operators for amplitude damping from the system-environment model:
- Environment is a single qubit initially in $$|0\rangle_E$$
- Joint Hamiltonian: $$H = g(|01\rangle\langle 10| + |10\rangle\langle 01|)$$
- Time evolution for time $$t$$ with $$\gamma = \sin^2(gt)$$

---

### Problem 17 (H)
The **generalized amplitude damping** channel models decay at non-zero temperature. Its Kraus operators are:
$$K_0 = \sqrt{p}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \sqrt{p}\begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
$$K_2 = \sqrt{1-p}\begin{pmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{pmatrix}, \quad K_3 = \sqrt{1-p}\begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}$$

(a) Verify completeness.
(b) What is the physical meaning of $$p$$?
(c) Find the fixed point.

---

## Section D: Phase Damping (Problems 18-20)

### Problem 18 (E)
For phase damping with parameter $$\lambda$$:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

(a) Verify completeness.
(b) Apply to $$|+\rangle\langle +|$$.
(c) Is this channel unital?

---

### Problem 19 (M)
Show that phase damping leaves diagonal elements unchanged but decays off-diagonal elements:
$$\rho_{01} \to (1-\lambda)\rho_{01}$$

---

### Problem 20 (M)
Compare phase damping to the phase flip channel:
$$\mathcal{E}_{\text{pf}}(\rho) = (1-p)\rho + pZ\rho Z$$

Show that phase damping is equivalent to phase flip with $$p = \lambda/2$$.

---

## Section E: Choi-Jamiolkowski Isomorphism (Problems 21-26)

### Problem 21 (E)
Compute the Choi matrix for the identity channel on a qubit.

---

### Problem 22 (M)
Compute the Choi matrix for the depolarizing channel with $$p = 3/4$$ (completely depolarizing).

---

### Problem 23 (M)
For the amplitude damping channel:
(a) Compute the Choi matrix.
(b) Verify it is positive semi-definite.
(c) Verify the trace condition for TP.

---

### Problem 24 (H)
Given the Choi matrix:
$$J = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

(a) Is this a valid CPTP map?
(b) If so, identify the channel.
(c) Find its Kraus operators.

---

### Problem 25 (H)
Prove that a map $$\mathcal{E}$$ is completely positive if and only if its Choi matrix is positive semi-definite.

---

### Problem 26 (H)
Show how to recover a channel from its Choi matrix:
$$\mathcal{E}(\rho) = d \cdot \text{Tr}_1[(I \otimes \rho^T)J(\mathcal{E})]$$

Apply this to verify for the identity channel.

---

## Bonus Problems

### Bonus 1 (Research Level)
**Degradable Channels**

A channel $$\mathcal{E}$$ is **degradable** if there exists another channel $$\mathcal{D}$$ such that:
$$\mathcal{E}^c = \mathcal{D} \circ \mathcal{E}$$

where $$\mathcal{E}^c$$ is the complementary channel.

Show that the amplitude damping channel is degradable.

---

### Bonus 2 (Research Level)
**Channel Capacity**

The quantum capacity of a channel is:
$$Q(\mathcal{E}) = \lim_{n\to\infty} \frac{1}{n} \max_\rho I_c(\rho, \mathcal{E}^{\otimes n})$$

where $$I_c$$ is the coherent information.

For the amplitude damping channel, show that $$Q > 0$$ when $$\gamma < 1/2$$.

---

## Problem Set Summary

| Section | Problems | Topics |
|---------|----------|--------|
| A | 1-8 | CPTP maps, Kraus representation |
| B | 9-12 | Depolarizing channel |
| C | 13-17 | Amplitude damping |
| D | 18-20 | Phase damping |
| E | 21-26 | Choi-Jamiolkowski |

**Total: 26 problems + 2 bonus problems**

---

*Complete this problem set before consulting solutions.*
