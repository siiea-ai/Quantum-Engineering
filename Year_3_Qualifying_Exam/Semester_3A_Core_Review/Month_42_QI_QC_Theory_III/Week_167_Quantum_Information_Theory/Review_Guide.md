# Week 167: Quantum Information Theory - Review Guide

## Introduction

Quantum information theory extends Shannon's classical information theory to the quantum domain. It provides the mathematical framework for understanding fundamental limits on quantum communication, computation, and data processing. This review covers the essential concepts for PhD qualifying examinations.

The central quantities—von Neumann entropy, quantum mutual information, and channel capacity—characterize the resources required for quantum information tasks and the fundamental limits on what quantum mechanics allows.

---

## 1. Von Neumann Entropy

### 1.1 Definition

The von Neumann entropy of a quantum state $\rho$ is:

$$\boxed{S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i}$$

where $\{\lambda_i\}$ are the eigenvalues of $\rho$.

**Convention:** $0 \log 0 = 0$ (by continuity).

### 1.2 Physical Interpretation

- **Uncertainty measure:** Quantifies uncertainty about the quantum state
- **Mixedness measure:** Pure states have $S = 0$, maximally mixed have $S = \log d$
- **Entanglement measure:** For bipartite pure states, $S(\rho_A) = S(\rho_B)$ measures entanglement

### 1.3 Key Properties

**Property 1: Non-negativity**
$$S(\rho) \geq 0$$

Equality holds iff $\rho$ is pure.

**Property 2: Maximum value**
$$S(\rho) \leq \log_2 d$$

Equality holds iff $\rho = I/d$ (maximally mixed).

**Property 3: Concavity**
$$S\left(\sum_i p_i \rho_i\right) \geq \sum_i p_i S(\rho_i)$$

Mixing increases entropy.

**Property 4: Additivity for product states**
$$S(\rho_A \otimes \rho_B) = S(\rho_A) + S(\rho_B)$$

### 1.4 Examples

**Pure state:** $|\psi\rangle\langle\psi|$
$$S = 0$$

**Maximally mixed qubit:** $\rho = I/2$
$$S = -\frac{1}{2}\log\frac{1}{2} - \frac{1}{2}\log\frac{1}{2} = 1 \text{ bit}$$

**Werner state:** $\rho_W = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$

Eigenvalues: $\frac{1+3p}{4}$ (once), $\frac{1-p}{4}$ (three times)
$$S(\rho_W) = -\frac{1+3p}{4}\log\frac{1+3p}{4} - 3\frac{1-p}{4}\log\frac{1-p}{4}$$

---

## 2. Composite Systems and Conditional Entropy

### 2.1 Joint and Marginal Entropy

For bipartite system $\rho_{AB}$:
- Joint entropy: $S(AB) = S(\rho_{AB})$
- Marginal entropies: $S(A) = S(\rho_A)$, $S(B) = S(\rho_B)$

where $\rho_A = \text{Tr}_B(\rho_{AB})$.

### 2.2 Subadditivity

$$\boxed{S(AB) \leq S(A) + S(B)}$$

**Proof idea:** Uses concavity and properties of relative entropy.

**Equality condition:** $\rho_{AB} = \rho_A \otimes \rho_B$ (product state)

### 2.3 Araki-Lieb Inequality

$$\boxed{|S(A) - S(B)| \leq S(AB)}$$

Combined with subadditivity:
$$|S(A) - S(B)| \leq S(AB) \leq S(A) + S(B)$$

### 2.4 Quantum Conditional Entropy

$$\boxed{S(A|B) = S(AB) - S(B)}$$

**Key difference from classical:** $S(A|B)$ can be negative!

**Example:** For maximally entangled state $|\Phi^+\rangle_{AB}$:
$$S(A|B) = S(AB) - S(B) = 0 - 1 = -1$$

**Interpretation:** Negative conditional entropy indicates entanglement. The "missing" entropy reflects correlations beyond classical.

### 2.5 Quantum Mutual Information

$$\boxed{I(A:B) = S(A) + S(B) - S(AB)}$$

**Properties:**
- Always non-negative: $I(A:B) \geq 0$
- Symmetric: $I(A:B) = I(B:A)$
- For product states: $I(A:B) = 0$
- Maximum for maximally entangled: $I(A:B) = 2S(A)$

**Equivalent expressions:**
$$I(A:B) = S(A) - S(A|B) = S(B) - S(B|A)$$

---

## 3. Strong Subadditivity

### 3.1 Statement

For any tripartite state $\rho_{ABC}$:

$$\boxed{S(ABC) + S(B) \leq S(AB) + S(BC)}$$

**Equivalent forms:**
$$S(A|BC) \leq S(A|B)$$
$$I(A:C|B) \geq 0$$

### 3.2 Significance

- Most important entropy inequality
- Foundation for many information-theoretic proofs
- No simple classical analog proof works; requires deep analysis

### 3.3 Proof (Lieb-Ruskai, 1973)

Uses the joint concavity of the function $f(X,Y) = \text{Tr}(X^p K Y^{1-p} K^\dagger)$ for $0 < p < 1$.

Full proof is technical but the result is fundamental.

### 3.4 Applications

- Data processing inequality: Information cannot increase under local operations
- Channel capacity proofs
- Entropy bounds in many-body physics

---

## 4. Holevo Bound

### 4.1 Setup

Alice prepares states $\{p_x, \rho_x\}$ (ensemble). Bob measures and wants to learn $X$.

**Question:** How much information can Bob extract?

### 4.2 Holevo Quantity

$$\boxed{\chi(\{p_x, \rho_x\}) = S(\rho) - \sum_x p_x S(\rho_x)}$$

where $\rho = \sum_x p_x \rho_x$ is the average state.

### 4.3 Holevo Bound Theorem

For any measurement $M$ that Bob performs:

$$\boxed{I(X:Y) \leq \chi(\{p_x, \rho_x\})}$$

where $Y$ is Bob's measurement outcome.

### 4.4 Proof Sketch

1. Consider purifications of the ensemble
2. Apply data processing inequality
3. Use properties of mutual information under measurements

### 4.5 Interpretation

- **Upper bound:** Cannot extract more than $\chi$ bits of classical information
- **For orthogonal states:** $\chi = H(p)$ (Shannon entropy of distribution)
- **For pure states:** $\chi = S(\rho)$ (von Neumann entropy)
- **For d-dimensional system:** $\chi \leq \log d$

### 4.6 Example: One Qubit

Alice sends qubit in state $\rho_0$ (prob 1/2) or $\rho_1$ (prob 1/2).

If $\rho_0 = |0\rangle\langle 0|$ and $\rho_1 = |1\rangle\langle 1|$ (orthogonal):
$$\chi = S(\frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|) - 0 = S(\frac{I}{2}) = 1$$

Bob can extract 1 bit perfectly.

If $\rho_0 = |0\rangle\langle 0|$ and $\rho_1 = |+\rangle\langle +|$ (non-orthogonal):
$$\chi = S(\rho) - 0 < 1$$

Bob cannot extract 1 full bit.

### 4.7 Holevo-Schumacher-Westmoreland (HSW) Theorem

The classical capacity of a quantum channel equals the regularized Holevo capacity:

$$C(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \chi^*(\mathcal{N}^{\otimes n})$$

where $\chi^*$ is maximized over input ensembles.

---

## 5. Quantum Channel Capacity

### 5.1 Classical Capacity

**Definition:** Maximum rate at which classical bits can be sent through quantum channel.

$$\boxed{C(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \max_{\{p_x, \rho_x^n\}} \chi(\mathcal{N}^{\otimes n})}$$

**For some channels:** Regularization is necessary (capacity is not additive in general).

**Examples:**
- Noiseless qubit channel: $C = 1$
- Depolarizing channel with $p$: $C = 1 - H(p) - p\log 3$ (for small $p$)

### 5.2 Quantum Capacity

**Definition:** Maximum rate for transmitting quantum information (qubits).

Uses coherent information:
$$I_c(\rho, \mathcal{N}) = S(\mathcal{N}(\rho)) - S((\mathcal{N} \otimes I)(\psi_{AR}))$$

where $|\psi_{AR}\rangle$ is a purification of $\rho$.

$$\boxed{Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \max_\rho I_c(\rho^n, \mathcal{N}^{\otimes n})}$$

**Key result:** Quantum capacity can be zero even when classical capacity is positive (entanglement-breaking channels).

### 5.3 Entanglement-Assisted Capacity

With pre-shared entanglement:

$$\boxed{C_E(\mathcal{N}) = \max_\rho I(A:B)_{\mathcal{N}(\rho)}}$$

**Key property:** Additive! No regularization needed.

**For noiseless channel:** $C_E = 2$ (superdense coding).

### 5.4 Private Capacity

For secure classical communication:

$$P(\mathcal{N}) \geq Q(\mathcal{N})$$

Private capacity is at least quantum capacity.

---

## 6. Quantum Data Compression

### 6.1 The Compression Problem

Alice has a source emitting states $\{p_i, |\psi_i\rangle\}$ i.i.d.
Goal: Compress $n$ emissions into $nR$ qubits with $R$ as small as possible.

### 6.2 Schumacher's Theorem

**Theorem:** Quantum source with density matrix $\rho$ can be compressed to $S(\rho)$ qubits per symbol asymptotically.

$$\boxed{\text{Compression rate} = S(\rho)}$$

### 6.3 Typical Subspace

For $n$ copies: $\rho^{\otimes n}$ has $d^n$ dimensional Hilbert space.

**Typical subspace:** States with eigenvalues close to $2^{-nS(\rho)}$.

**Dimension of typical subspace:** $\approx 2^{nS(\rho)}$

### 6.4 Compression Protocol

1. Project onto typical subspace
2. Encode typical subspace (dimension $2^{nS(\rho)}$) into $nS(\rho)$ qubits
3. Send compressed qubits
4. Decompress by reversing encoding

**Fidelity:** $F \to 1$ as $n \to \infty$.

### 6.5 Example

**Ensemble:** $\{1/2, |0\rangle\}$, $\{1/2, |+\rangle\}$

$$\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|+\rangle\langle +| = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{4}|0\rangle\langle 0| + \frac{1}{4}|1\rangle\langle 1| + \frac{1}{4}|0\rangle\langle 1| + \frac{1}{4}|1\rangle\langle 0|$$

Eigenvalues: $(1 + 1/\sqrt{2})/2$ and $(1 - 1/\sqrt{2})/2$

$S(\rho) \approx 0.60$ bits/symbol

---

## 7. Relative Entropy

### 7.1 Definition

$$\boxed{S(\rho \| \sigma) = \text{Tr}(\rho \log \rho) - \text{Tr}(\rho \log \sigma)}$$

If $\text{supp}(\rho) \not\subseteq \text{supp}(\sigma)$: $S(\rho \| \sigma) = +\infty$.

### 7.2 Properties

- **Non-negativity:** $S(\rho \| \sigma) \geq 0$ (Klein's inequality)
- **Equality:** $S(\rho \| \sigma) = 0$ iff $\rho = \sigma$
- **Not symmetric:** $S(\rho \| \sigma) \neq S(\sigma \| \rho)$ in general
- **Monotonicity:** $S(\rho \| \sigma) \geq S(\mathcal{E}(\rho) \| \mathcal{E}(\sigma))$ for any channel $\mathcal{E}$

### 7.3 Applications

- Data processing inequality proofs
- Hypothesis testing
- Thermodynamics (free energy differences)

---

## 8. Entanglement Measures

### 8.1 Entanglement Entropy

For bipartite pure state $|\psi\rangle_{AB}$:
$$E(|\psi\rangle) = S(\rho_A) = S(\rho_B)$$

This is the unique entanglement measure for pure states satisfying natural axioms.

### 8.2 Entanglement of Formation

For mixed states $\rho_{AB}$:
$$E_F(\rho) = \min_{\{p_i, |\psi_i\rangle\}} \sum_i p_i E(|\psi_i\rangle)$$

Minimum over all decompositions $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$.

### 8.3 Distillable Entanglement

$$E_D(\rho) = \sup \{R : \rho^{\otimes n} \xrightarrow{LOCC} |\Phi^+\rangle^{\otimes nR}\}$$

Maximum rate of extracting EPR pairs via LOCC.

### 8.4 Key Results

- $E_D(\rho) \leq E_F(\rho)$ (generally)
- $E_D(\rho) = E_F(\rho) = E(\psi)$ for pure states
- Bound entangled states: $E_F > 0$ but $E_D = 0$

---

## 9. Exam Preparation

### Key Theorems to Know

1. Von Neumann entropy properties (non-neg, max, concavity)
2. Subadditivity and strong subadditivity
3. Holevo bound
4. Schumacher compression theorem

### Common Exam Problems

1. Calculate entropy of specific density matrices
2. Prove or apply subadditivity
3. Apply Holevo bound to communication scenarios
4. Discuss quantum vs. classical capacity

### Important Formulas

Write these from memory:
$$S(\rho) = -\text{Tr}(\rho \log \rho)$$
$$S(A|B) = S(AB) - S(B)$$
$$I(A:B) = S(A) + S(B) - S(AB)$$
$$\chi = S(\rho) - \sum_i p_i S(\rho_i)$$

---

## 10. Summary Table

| Quantity | Formula | Range | Classical Analog |
|----------|---------|-------|------------------|
| Von Neumann entropy | $-\text{Tr}(\rho \log \rho)$ | $[0, \log d]$ | Shannon entropy |
| Conditional entropy | $S(AB) - S(B)$ | $[-\log d, \log d]$ | Always ≥ 0 classically |
| Mutual information | $S(A) + S(B) - S(AB)$ | $[0, 2\min(S(A), S(B))]$ | $\leq \min(H(A), H(B))$ classically |
| Holevo quantity | $S(\rho) - \sum_i p_i S(\rho_i)$ | $[0, \log d]$ | N/A |

---

## References

1. Nielsen, M.A. and Chuang, I.L. *Quantum Computation and Quantum Information.* Cambridge University Press, 2010.

2. Wilde, M. *Quantum Information Theory.* Cambridge University Press, 2013. (arXiv:1106.1445)

3. Preskill, J. "Lecture Notes for Physics 219: Quantum Information." California Institute of Technology.

4. Holevo, A.S. "The Capacity of a Quantum Communications Channel." *Problems Inform. Transmission* 9, 177 (1973).

5. Schumacher, B. "Quantum Coding." *Phys. Rev. A* 51, 2738 (1995).

---

**Word Count:** ~2600 words
**Created:** February 9, 2026
