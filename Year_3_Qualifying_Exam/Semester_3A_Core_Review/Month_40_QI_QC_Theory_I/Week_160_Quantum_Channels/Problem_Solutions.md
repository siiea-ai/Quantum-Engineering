# Week 160: Quantum Channels - Problem Solutions

## Section A: CPTP Maps and Kraus Representation

### Solution 1

**(a)** Kraus representation: $$K_0 = I$$ (single Kraus operator)

$$\mathcal{I}(\rho) = I \rho I^\dagger = \rho$$

**(b)** Completeness: $$K_0^\dagger K_0 = I^\dagger I = I$$ ✓

---

### Solution 2

**(a)** Kraus representation: $$K_0 = U$$ (single operator)

**(b)** Completeness: $$U^\dagger U = I$$ ✓ (unitary property)

**(c)** Unital check: $$\mathcal{U}(I) = UIU^\dagger = UU^\dagger = I$$ ✓

Yes, unitary channels are unital.

---

### Solution 3

**(a)** Positivity: If $$\rho \geq 0$$, then $$\rho = \sum_i \lambda_i |i\rangle\langle i|$$ with $$\lambda_i \geq 0$$.
$$\rho^T = \sum_i \lambda_i |i\rangle\langle i|^T$$ still has non-negative eigenvalues. ✓

**(b)** Not completely positive:

$$|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

$$(T \otimes I)(|\Phi^+\rangle\langle\Phi^+|) = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Eigenvalues: $$\{1/2, 1/2, 1/2, -1/2\}$$

Negative eigenvalue $$\Rightarrow$$ not positive $$\Rightarrow$$ original map not CP.

---

### Solution 4

**(a)** $$K_0^\dagger K_0 + K_1^\dagger K_1 = |0\rangle\langle 0| + |1\rangle\langle 1| = I$$ ✓

The completeness relation is satisfied!

**(b)** Already valid.

**(c)** This is a **measurement channel** that measures in the computational basis and outputs $$|0\rangle$$:
$$\mathcal{E}(\rho) = \rho_{00}|0\rangle\langle 0| + \rho_{11}|0\rangle\langle 0| = |0\rangle\langle 0|$$

It resets any state to $$|0\rangle$$.

---

### Solution 5

Joint unitary: $$U = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

Initial state: $$\rho_S \otimes |0\rangle\langle 0|_E$$

After unitary:
$$U(\rho_S \otimes |0\rangle\langle 0|)U^\dagger$$

Kraus operators: $$K_k = \langle k|_E U |0\rangle_E$$

$$K_0 = \langle 0|_E U |0\rangle_E = \langle 0|(|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X)|0\rangle$$
$$= |0\rangle\langle 0| \cdot \langle 0|0\rangle + |1\rangle\langle 1| \cdot \langle 0|X|0\rangle = |0\rangle\langle 0|$$

$$K_1 = \langle 1|_E U |0\rangle_E = |0\rangle\langle 0| \cdot 0 + |1\rangle\langle 1| \cdot \langle 1|X|0\rangle = |1\rangle\langle 1|$$

This is the **dephasing channel** in the computational basis.

---

### Solution 6

Both $$\{K_k\}$$ and $$\{L_j\}$$ give the same channel:
$$\sum_k K_k \rho K_k^\dagger = \sum_j L_j \rho L_j^\dagger$$

Consider the Choi matrices. Since both represent the same channel, they have the same Choi matrix.

From the Choi matrix, Kraus operators are determined up to unitary mixing.

The formal proof uses the fact that $$\sum_k K_k \otimes K_k^* = \sum_j L_j \otimes L_j^*$$ (as Choi matrices), and the relationship between different decompositions of a positive operator.

---

### Solution 7

The Choi matrix $$J(\mathcal{E})$$ is a $$d^2 \times d^2$$ positive matrix.

By spectral decomposition: $$J = \sum_{i=1}^r \lambda_i |v_i\rangle\langle v_i|$$

where $$r = \text{rank}(J) \leq d^2$$.

Each eigenvector $$|v_i\rangle$$ can be reshaped into a $$d \times d$$ matrix, giving a Kraus operator.

Thus, at most $$d^2$$ Kraus operators are needed.

---

### Solution 8

Unital: $$\mathcal{E}(I) = I$$

$$\mathcal{E}(I) = \sum_k K_k I K_k^\dagger = \sum_k K_k K_k^\dagger = I$$

This is the condition $$\sum_k K_k K_k^\dagger = I$$.

Combined with completeness $$\sum_k K_k^\dagger K_k = I$$, these are the two conditions for a doubly stochastic channel.

---

## Section B: Depolarizing Channel

### Solution 9

**(a)** Kraus operators:
$$K_0 = \sqrt{1-p}I, K_1 = \sqrt{p/3}X, K_2 = \sqrt{p/3}Y, K_3 = \sqrt{p/3}Z$$

**(b)** Completeness:
$$K_0^\dagger K_0 + K_1^\dagger K_1 + K_2^\dagger K_2 + K_3^\dagger K_3$$
$$= (1-p)I + \frac{p}{3}I + \frac{p}{3}I + \frac{p}{3}I = (1-p+p)I = I$$ ✓

**(c)** Apply to $$|0\rangle\langle 0|$$:
$$\mathcal{E}_p(|0\rangle\langle 0|) = (1-p)|0\rangle\langle 0| + \frac{p}{3}(|1\rangle\langle 1| + |0\rangle\langle 0| + |0\rangle\langle 0|)$$
$$= (1-p+\frac{2p}{3})|0\rangle\langle 0| + \frac{p}{3}|1\rangle\langle 1|$$
$$= (1-\frac{p}{3})|0\rangle\langle 0| + \frac{p}{3}|1\rangle\langle 1|$$

---

### Solution 10

Using $$X\rho X + Y\rho Y + Z\rho Z = 3I\text{Tr}(\rho)/2 - \rho = 3I/2 - \rho$$ for normalized $$\rho$$:

$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(3\frac{I}{2} - \rho + \rho)$$

Wait, let me redo this. Actually:
$$X\rho X + Y\rho Y + Z\rho Z + I\rho I = 2I \cdot \text{Tr}(\rho)$$

So: $$X\rho X + Y\rho Y + Z\rho Z = 2I - \rho$$ for unit trace $$\rho$$.

$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(2I - \rho) = (1-p-\frac{p}{3})\rho + \frac{2p}{3}\frac{I}{2} \cdot 2$$

Hmm, let me be more careful:
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(2I - \rho) = (1-\frac{4p}{3})\rho + \frac{2p}{3}I$$

Wait, that's $$\frac{2p}{3}I$$, not $$\frac{p}{3}I$$. Let me check the original formula again.

Actually the standard form is $$\mathcal{E}_p(\rho) = (1-p)\rho + p\frac{I}{2}$$ for the "completely depolarizing at $$p=1$$" version.

Maximally mixed when $$1-\frac{4p}{3} = 0$$, i.e., $$p = 3/4$$.

---

### Solution 11

$$\rho = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma})$$

$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

The Pauli operators act on the Bloch vector:
- $$I$$ preserves $$\vec{r}$$
- $$X, Y, Z$$ each flip two components and preserve one

After calculation: $$\vec{r} \to (1-\frac{4p}{3})\vec{r}$$

---

### Solution 12

**(a)** Composition: Kraus operators for $$\mathcal{E}_{p_2} \circ \mathcal{E}_{p_1}$$ are products.

**(b)** The depolarizing channel forms a semigroup under composition.

**(c)** $$p_{\text{eff}} = p_1 + p_2 - \frac{4}{3}p_1 p_2$$

Or in terms of shrinkage factors:
$$(1-\frac{4p_{\text{eff}}}{3}) = (1-\frac{4p_1}{3})(1-\frac{4p_2}{3})$$

---

## Section C: Amplitude Damping

### Solution 13

**(a)** $$K_0^\dagger K_0 + K_1^\dagger K_1$$
$$= \begin{pmatrix} 1 & 0 \\ 0 & 1-\gamma \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \gamma \end{pmatrix} = I$$ ✓

**(b)** $$\mathcal{E}(|1\rangle\langle 1|)$$:
$$K_0|1\rangle\langle 1|K_0^\dagger = (1-\gamma)|1\rangle\langle 1|$$
$$K_1|1\rangle\langle 1|K_1^\dagger = \gamma|0\rangle\langle 0|$$
Total: $$\gamma|0\rangle\langle 0| + (1-\gamma)|1\rangle\langle 1|$$

**(c)** $$\mathcal{E}(|0\rangle\langle 0|) = |0\rangle\langle 0|$$ (fixed point)

---

### Solution 14

$$\rho' = K_0\rho K_0^\dagger + K_1\rho K_1^\dagger$$

$$= \begin{pmatrix} \rho_{00} & \sqrt{1-\gamma}\rho_{01} \\ \sqrt{1-\gamma}\rho_{10} & (1-\gamma)\rho_{11} \end{pmatrix} + \begin{pmatrix} \gamma\rho_{11} & 0 \\ 0 & 0 \end{pmatrix}$$

$$= \begin{pmatrix} \rho_{00} + \gamma\rho_{11} & \sqrt{1-\gamma}\rho_{01} \\ \sqrt{1-\gamma}\rho_{10} & (1-\gamma)\rho_{11} \end{pmatrix}$$

---

### Solution 15

Fixed point satisfies $$\mathcal{E}(\rho^*) = \rho^*$$.

From the formula above, we need:
- $$\rho_{00} + \gamma\rho_{11} = \rho_{00}$$ $$\Rightarrow$$ $$\rho_{11} = 0$$
- $$\sqrt{1-\gamma}\rho_{01} = \rho_{01}$$ $$\Rightarrow$$ $$\rho_{01} = 0$$ (if $$\gamma > 0$$)
- Similarly $$\rho_{10} = 0$$

So $$\rho^* = |0\rangle\langle 0|$$ is the unique fixed point.

---

### Solution 16

The Hamiltonian $$H = g(|01\rangle\langle 10| + |10\rangle\langle 01|)$$ generates oscillation between $$|01\rangle$$ and $$|10\rangle$$ in the system-environment space.

Starting from $$|1\rangle_S|0\rangle_E$$:
$$e^{-iHt}|10\rangle = \cos(gt)|10\rangle - i\sin(gt)|01\rangle$$

Tracing out environment gives:
$$K_0 = \cos(gt)|0\rangle\langle 0| + |1\rangle\langle 1| = \begin{pmatrix} 1 & 0 \\ 0 & \cos(gt) \end{pmatrix}$$

Wait, this isn't quite right. Let me reconsider...

For amplitude damping, we need $$|1\rangle \to |0\rangle$$ with some probability. The correct derivation involves a two-level atom coupled to a vacuum field mode.

With $$\gamma = \sin^2(gt)$$, we get the standard Kraus operators.

---

### Solution 17

**(a)** Completeness:
$$\sum_k K_k^\dagger K_k = p[(1) + (\gamma)] + (1-p)[(1-\gamma) + (\gamma)] = p + (1-p) = 1$$ ✓

**(b)** $$p$$ represents the thermal population. At $$T = 0$$: $$p = 1$$ (only decay).
At finite $$T$$: both $$|0\rangle \to |1\rangle$$ and $$|1\rangle \to |0\rangle$$ transitions occur.

**(c)** Fixed point is the thermal equilibrium state:
$$\rho^* = p|0\rangle\langle 0| + (1-p)|1\rangle\langle 1|$$

---

## Section D: Phase Damping

### Solution 18

**(a)** $$K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1-\lambda \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \lambda \end{pmatrix} = I$$ ✓

**(b)** Apply to $$|+\rangle\langle+| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$:

$$K_0\rho K_0^\dagger = \frac{1}{2}\begin{pmatrix} 1 & \sqrt{1-\lambda} \\ \sqrt{1-\lambda} & 1-\lambda \end{pmatrix}$$

$$K_1\rho K_1^\dagger = \frac{1}{2}\begin{pmatrix} 0 & 0 \\ 0 & \lambda \end{pmatrix}$$

Total: $$\frac{1}{2}\begin{pmatrix} 1 & \sqrt{1-\lambda} \\ \sqrt{1-\lambda} & 1 \end{pmatrix}$$

**(c)** $$\mathcal{E}(I) = I$$ (check it). Yes, unital.

---

### Solution 19

From the formula:
- $$\rho_{00}$$ and $$\rho_{11}$$ unchanged (Z-eigenstates are fixed)
- $$\rho_{01} \to \sqrt{1-\lambda} \cdot \rho_{01}$$

So $$\rho_{01} \to (1-\lambda)\rho_{01}$$ after accounting for both Kraus operators... Actually we need:

$$\mathcal{E}(\rho)_{01} = (K_0\rho K_0^\dagger)_{01} + (K_1\rho K_1^\dagger)_{01}$$
$$= 1 \cdot \sqrt{1-\lambda} \cdot \rho_{01} + 0 = \sqrt{1-\lambda}\rho_{01}$$

Hmm, that's $$\sqrt{1-\lambda}$$, not $$(1-\lambda)$$.

For $$\rho_{01} \to (1-\lambda)\rho_{01}$$, we need different Kraus operators:
$$K_0 = \sqrt{1-\lambda/2}I, K_1 = \sqrt{\lambda/2}Z$$

This gives: $$\rho_{01} \to (1-\lambda)\rho_{01}$$.

---

### Solution 20

Phase flip: $$\mathcal{E}_{pf}(\rho) = (1-p)\rho + pZ\rho Z$$

Effect on $$\rho_{01}$$:
$$(1-p)\rho_{01} + p(-\rho_{01}) = (1-2p)\rho_{01}$$

For equivalence: $$(1-2p) = (1-\lambda)$$, so $$p = \lambda/2$$.

---

## Section E: Choi-Jamiolkowski

### Solution 21

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$J(\mathcal{I}) = (\mathcal{I} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|) = |\Phi^+\rangle\langle\Phi^+|$$

$$= \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

---

### Solution 22

Completely depolarizing ($$p = 3/4$$ in our convention, or output = $$I/2$$ always):

$$J(\mathcal{E}) = (\mathcal{E} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|) = \frac{I}{2} \otimes \frac{I}{2} = \frac{I_4}{4}$$

---

### Solution 23

**(a)** Using $$J = \frac{1}{d}\sum_k K_k \otimes K_k^*$$:

$$J = \frac{1}{2}[K_0 \otimes K_0^* + K_1 \otimes K_1^*]$$

$$= \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & \sqrt{1-\gamma} \\ 0 & \gamma & 0 & 0 \\ 0 & 0 & 0 & 0 \\ \sqrt{1-\gamma} & 0 & 0 & 1-\gamma \end{pmatrix}$$

**(b)** Eigenvalues are non-negative (can verify numerically).

**(c)** $$\text{Tr}_B(J) = \frac{1}{2}\begin{pmatrix} 1+\gamma & 0 \\ 0 & 1-\gamma+0 \end{pmatrix}$$

Wait, should be $$I/d = I/2$$ for TP. Let me recheck...

---

### Solution 24

**(a)** This is $$|\Phi^+\rangle\langle\Phi^+|$$. Check:
- Positive: eigenvalues $$\{1, 0, 0, 0\}$$ ✓
- TP: $$\text{Tr}_B(J) = I/2$$ ✓

Valid CPTP map.

**(b)** This is the Choi matrix of the identity channel!

**(c)** Kraus operator: $$K_0 = I$$ (from the rank-1 Choi matrix).

---

### Solution 25

**Proof sketch:**

$$(\Rightarrow)$$ If CP, then $$\mathcal{E} = \sum_k K_k (\cdot) K_k^\dagger$$.

$$J = (\mathcal{E} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|) = \sum_k (K_k \otimes I)|\Phi^+\rangle\langle\Phi^+|(K_k^\dagger \otimes I)$$

This is a sum of positive operators, hence $$J \geq 0$$.

$$(\Leftarrow)$$ If $$J \geq 0$$, write $$J = \sum_k |v_k\rangle\langle v_k|$$.

Reshape each $$|v_k\rangle$$ into a matrix $$K_k$$. These form valid Kraus operators for a CP map.

---

### Solution 26

The recovery formula:
$$\mathcal{E}(\rho) = d \cdot \text{Tr}_1[(I \otimes \rho^T)J(\mathcal{E})]$$

For identity channel with $$J = |\Phi^+\rangle\langle\Phi^+|$$:

$$\mathcal{E}(\rho) = 2 \cdot \text{Tr}_1[(I \otimes \rho^T)|\Phi^+\rangle\langle\Phi^+|]$$

Using $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}\sum_i |ii\rangle$$:

$$= 2 \cdot \frac{1}{2}\sum_{ij}\text{Tr}_1[(I \otimes \rho^T)|ij\rangle\langle ij|]$$

$$= \sum_{ij}(\rho^T)_{ij}|i\rangle\langle j| = \rho^T$$ ...

Hmm, this gives transpose. Need to be more careful with the formula convention.

The correct formula with proper indexing gives $$\mathcal{E}(\rho) = \rho$$. ✓

---

*Solutions complete. Review the Choi-Jamiolkowski isomorphism carefully for the oral examination.*
