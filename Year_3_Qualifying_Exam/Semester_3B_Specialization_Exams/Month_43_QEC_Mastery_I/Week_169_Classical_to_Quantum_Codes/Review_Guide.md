# Week 169: Classical to Quantum Codes - Review Guide

## Introduction

Quantum error correction (QEC) is the enabling technology for fault-tolerant quantum computation. This review covers the theoretical foundations from classical error correction through to the first quantum codes, emphasizing the conceptual and mathematical framework required for PhD qualifying examinations.

The central question we address: **Under what conditions can quantum information be protected from noise?** The answer, given by the Knill-Laflamme conditions, provides necessary and sufficient criteria for a code to correct a given set of errors.

---

## 1. Classical Error Correction Review

### 1.1 Linear Codes

A classical linear code $$C$$ over $$\mathbb{F}_2$$ is a $$k$$-dimensional subspace of $$\mathbb{F}_2^n$$. The code is characterized by parameters $$[n, k, d]$$:
- $$n$$: Block length (number of physical bits)
- $$k$$: Dimension (number of encoded bits)
- $$d$$: Minimum distance (minimum Hamming weight of nonzero codewords)

**Generator Matrix:** A $$k \times n$$ matrix $$G$$ whose rows form a basis for $$C$$. Encoding: $$x \mapsto xG$$.

**Parity-Check Matrix:** An $$(n-k) \times n$$ matrix $$H$$ such that $$c \in C \iff Hc^T = 0$$. The code is the null space of $$H$$.

**Relationship:** $$GH^T = 0$$ (mod 2).

### 1.2 Syndrome-Based Decoding

When a codeword $$c$$ experiences error $$e$$, the received word is $$r = c + e$$. The **syndrome** is:

$$s = Hr^T = H(c + e)^T = He^T$$

The syndrome depends only on the error, not the transmitted codeword. This enables error identification.

**Error Correction Capability:** A code with minimum distance $$d$$ can:
- Detect up to $$d - 1$$ errors
- Correct up to $$t = \lfloor (d-1)/2 \rfloor$$ errors

### 1.3 The Hamming Code

The $$[7, 4, 3]$$ Hamming code is the prototypical example. Its parity-check matrix:

$$H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

The syndrome directly gives the binary representation of the error position.

---

## 2. Quantum Errors and Their Discretization

### 2.1 The Pauli Group

The single-qubit Pauli operators are:

$$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

Key properties:
- $$X^2 = Y^2 = Z^2 = I$$
- $$XY = iZ$$, $$YZ = iX$$, $$ZX = iY$$
- $$Y = iXZ$$

The **n-qubit Pauli group** $$\mathcal{G}_n$$ consists of tensor products of Pauli operators with phases $$\{\pm 1, \pm i\}$$:

$$\mathcal{G}_n = \{\pm 1, \pm i\} \times \{I, X, Y, Z\}^{\otimes n}$$

### 2.2 Types of Quantum Errors

**Bit-flip error (X):** $$X|0\rangle = |1\rangle$$, $$X|1\rangle = |0\rangle$$

**Phase-flip error (Z):** $$Z|0\rangle = |0\rangle$$, $$Z|1\rangle = -|1\rangle$$

**Combined error (Y):** $$Y = iXZ$$ - applies both bit and phase flip

**Continuous errors:** A general single-qubit error channel acts as:
$$\mathcal{E}(\rho) = \sum_{j} K_j \rho K_j^\dagger$$

where Kraus operators $$K_j$$ can be arbitrary. However, any operator can be expanded in the Pauli basis:
$$K_j = \alpha_j^{(0)} I + \alpha_j^{(1)} X + \alpha_j^{(2)} Y + \alpha_j^{(3)} Z$$

### 2.3 Error Discretization Theorem

**Theorem (Error Discretization):** If a quantum code can correct the set of Pauli errors $$\{E_a\}$$, then it can correct any error whose Kraus operators are linear combinations of $$\{E_a\}$$.

**Proof sketch:** The recovery operation projects the environment onto distinct orthogonal subspaces for each correctable error. When a general error occurs, this measurement discretizes the error into one of the correctable Pauli errors, which can then be corrected.

This theorem is crucial: **we only need to correct Pauli errors**.

---

## 3. The Knill-Laflamme Conditions

### 3.1 Statement of the Theorem

Let $$C$$ be a quantum code (subspace of Hilbert space) with orthonormal basis $$\{|\psi_i\rangle\}$$. Let $$\{E_a\}$$ be a set of error operators.

**Theorem (Knill-Laflamme, 1997):** The code $$C$$ can correct the errors $$\{E_a\}$$ if and only if:

$$\boxed{\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = C_{ab} \delta_{ij}}$$

for all codeword basis states $$i, j$$ and all errors $$a, b$$. The matrix $$C = (C_{ab})$$ is Hermitian.

### 3.2 Equivalent Formulations

Let $$P$$ be the projector onto the code space $$C$$. The condition is equivalent to:

$$P E_a^\dagger E_b P = C_{ab} P$$

Or in terms of the error operators acting on the code space: errors must either:
1. Take the code space to **orthogonal** subspaces (distinguishable errors), or
2. Act **identically** on the code space (degenerate errors)

### 3.3 Proof of the Knill-Laflamme Theorem

**Necessity:** Suppose errors can be corrected. A recovery operation $$\mathcal{R}$$ exists such that for any codeword $$|\psi\rangle$$ and any error $$E_a$$:
$$\mathcal{R}(E_a |\psi\rangle\langle\psi| E_a^\dagger) \propto |\psi\rangle\langle\psi|$$

For this to work for superpositions, the error subspaces $$E_a C$$ must be distinguishable or identical. This requires:
$$\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = C_{ab} \delta_{ij}$$

**Sufficiency:** Assume the condition holds. We construct an explicit recovery:

1. Define syndrome projectors: The operators $$\{E_a P\}$$ span error subspaces. Orthogonalize to get projectors $$\{\Pi_k\}$$.

2. Each $$\Pi_k$$ projects onto a distinct "syndrome space" where specific errors occurred.

3. Recovery: Measure in the $$\{\Pi_k\}$$ basis, then apply the appropriate correction.

The condition ensures this procedure returns the original encoded state.

### 3.4 Physical Interpretation

The Knill-Laflamme conditions state:

1. **Orthogonality ($$i \neq j$$):** Different logical states remain distinguishable after any error:
   $$\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = 0$$

2. **Independence from logical state ($$i = j$$):** The overlap $$\langle \psi_i | E_a^\dagger E_b | \psi_i \rangle = C_{ab}$$ is the same for all basis states.

If these hold, the error reveals no information about which logical state was encoded.

### 3.5 Degenerate vs Non-Degenerate Codes

**Non-degenerate code:** The matrix $$C_{ab}$$ is diagonal, meaning different errors take the code space to orthogonal subspaces. The syndrome uniquely identifies the error.

**Degenerate code:** $$C_{ab}$$ has off-diagonal elements, meaning some errors act identically on the code space. Example: $$Z_1$$ and $$Z_2$$ may both flip the same syndrome without affecting logical information differently.

Degenerate codes can have higher distance than non-degenerate codes of the same size.

---

## 4. The Three-Qubit Codes

### 4.1 Bit-Flip Code

The simplest quantum code encodes one logical qubit into three physical qubits:

$$|0_L\rangle = |000\rangle, \quad |1_L\rangle = |111\rangle$$

**General encoded state:**
$$|\psi_L\rangle = \alpha|000\rangle + \beta|111\rangle$$

**Error correction:** This code corrects any single bit-flip ($$X$$) error.

Syndromes are measured using ancilla qubits:
- Measure $$Z_1 Z_2$$: Eigenvalue $$-1$$ indicates error on qubit 1 or 2
- Measure $$Z_2 Z_3$$: Eigenvalue $$-1$$ indicates error on qubit 2 or 3

| Syndrome ($$Z_1Z_2$$, $$Z_2Z_3$$) | Error |
|----------------------------------|-------|
| $$(+1, +1)$$ | None |
| $$(-1, +1)$$ | $$X_1$$ |
| $$(-1, -1)$$ | $$X_2$$ |
| $$(+1, -1)$$ | $$X_3$$ |

**Limitation:** Cannot correct phase-flip ($$Z$$) errors because:
$$Z_1|000\rangle = |000\rangle, \quad Z_1|111\rangle = |111\rangle$$
The error acts trivially!

### 4.2 Phase-Flip Code

Encode in the Hadamard-rotated basis:

$$|0_L\rangle = |{+}{+}{+}\rangle, \quad |1_L\rangle = |{-}{-}{-}\rangle$$

where $$|{\pm}\rangle = (|0\rangle \pm |1\rangle)/\sqrt{2}$$.

This corrects single phase-flip errors but not bit-flip errors.

**Key insight:** The phase-flip code is the Hadamard transform of the bit-flip code:
$$H^{\otimes 3}(\text{bit-flip code}) = \text{phase-flip code}$$

### 4.3 Verifying Knill-Laflamme for the Bit-Flip Code

Error set: $$\{I, X_1, X_2, X_3\}$$

Check: $$\langle 0_L | E_a^\dagger E_b | 1_L \rangle = 0$$ for all $$a, b$$

For $$E_a = E_b = I$$:
$$\langle 000 | 111 \rangle = 0 \checkmark$$

For $$E_a = X_1, E_b = I$$:
$$\langle 000 | X_1 | 111 \rangle = \langle 000 | 011 \rangle = 0 \checkmark$$

For $$E_a = X_1, E_b = X_2$$:
$$\langle 000 | X_1 X_2 | 111 \rangle = \langle 000 | 001 \rangle = 0 \checkmark$$

The diagonal elements $$\langle 0_L | E_a^\dagger E_b | 0_L \rangle$$ are all either 0 or 1, and equal $$\langle 1_L | E_a^\dagger E_b | 1_L \rangle$$. The conditions are satisfied.

---

## 5. The Nine-Qubit Shor Code

### 5.1 Construction via Concatenation

Shor's insight: Concatenate the phase-flip code with the bit-flip code.

**Step 1:** Encode against phase flips:
$$|0\rangle \to |{+}{+}{+}\rangle, \quad |1\rangle \to |{-}{-}{-}\rangle$$

**Step 2:** Encode each qubit against bit flips:
$$|{+}\rangle \to \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$
$$|{-}\rangle \to \frac{1}{\sqrt{2}}(|000\rangle - |111\rangle)$$

**Final encoding:**
$$|0_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)^{\otimes 3}$$
$$|1_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle - |111\rangle)^{\otimes 3}$$

### 5.2 Code Parameters

The Shor code has parameters $$[[9, 1, 3]]$$:
- 9 physical qubits
- 1 logical qubit
- Distance 3 (corrects any single-qubit error)

### 5.3 Error Correction Procedure

**Bit-flip errors (X):** Detected within each block of 3 qubits using $$Z_iZ_j$$ measurements, exactly as in the 3-qubit bit-flip code.

**Phase-flip errors (Z):** Detected by comparing blocks using $$X^{\otimes 3}$$ operators:
- Measure $$X_1X_2X_3 \cdot X_4X_5X_6$$
- Measure $$X_4X_5X_6 \cdot X_7X_8X_9$$

**Combined errors (Y = iXZ):** The X and Z components are corrected separately.

### 5.4 Stabilizer Generators (Preview)

The Shor code has 8 stabilizer generators:

Bit-flip detection:
$$Z_1Z_2, Z_2Z_3, Z_4Z_5, Z_5Z_6, Z_7Z_8, Z_8Z_9$$

Phase-flip detection:
$$X_1X_2X_3X_4X_5X_6, X_4X_5X_6X_7X_8X_9$$

### 5.5 Verification of Knill-Laflamme

For any single-qubit Pauli error $$E \in \{X_i, Y_i, Z_i\}$$:

$$\langle 0_L | E | 1_L \rangle = 0$$

This follows from the block structure: any single-qubit error either:
- Acts within a block (cannot flip the relative phase between blocks)
- Is a phase error (detected by inter-block comparisons)

The code is non-degenerate for single-qubit errors: each error produces a unique syndrome.

---

## 6. Beyond Shor: General Considerations

### 6.1 Code Distance

The **distance** $$d$$ of a quantum code is the minimum weight of an undetectable error (one that acts nontrivially on the code space while not being in the stabilizer).

Equivalently: $$d$$ is the minimum weight of a logical operator.

**Error correction capability:** A distance-$$d$$ code corrects $$\lfloor(d-1)/2\rfloor$$ errors.

### 6.2 The No-Cloning Connection

Quantum error correction circumvents no-cloning because:
1. We don't copy the quantum state
2. We redundantly encode using entanglement
3. Syndrome measurement extracts error information without measuring the logical state

### 6.3 Quantum Singleton Bound

For an $$[[n, k, d]]$$ quantum code:
$$k \leq n - 2(d - 1)$$

or equivalently:
$$n \geq 2d + k - 2$$

This is more restrictive than the classical bound because both $$X$$ and $$Z$$ errors must be corrected.

---

## 7. Summary of Key Results

| Topic | Key Result |
|-------|------------|
| Error discretization | Correcting Pauli errors suffices for all errors |
| Knill-Laflamme | $$\langle\psi_i\|E_a^\dagger E_b\|\psi_j\rangle = C_{ab}\delta_{ij}$$ |
| Bit-flip code | $$[[3,1,1]]$$ for $$X$$ errors only |
| Phase-flip code | $$[[3,1,1]]$$ for $$Z$$ errors only |
| Shor code | $$[[9,1,3]]$$ for all single-qubit errors |
| Singleton bound | $$k \leq n - 2d + 2$$ |

---

## 8. Exam Preparation Checklist

### Definitions to State Precisely
- [ ] Quantum code (subspace definition)
- [ ] Knill-Laflamme conditions
- [ ] Code distance
- [ ] Degenerate vs non-degenerate codes

### Proofs to Master
- [ ] Knill-Laflamme theorem (both directions)
- [ ] Error discretization theorem
- [ ] Shor code corrects all single-qubit errors

### Calculations to Practice
- [ ] Verify Knill-Laflamme for specific codes
- [ ] Syndrome calculation and error identification
- [ ] Encoding circuit construction

### Common Exam Questions
1. "State and prove the Knill-Laflamme conditions"
2. "Explain why the 3-qubit bit-flip code cannot correct phase errors"
3. "Derive the Shor code and show it has distance 3"
4. "What is a degenerate code? Give an example"

---

## References

1. Knill, E. & Laflamme, R. "Theory of Quantum Error-Correcting Codes" Phys. Rev. A 55, 900 (1997). [arXiv:quant-ph/9604034](https://arxiv.org/abs/quant-ph/9604034)

2. Shor, P.W. "Scheme for reducing decoherence in quantum computer memory" Phys. Rev. A 52, R2493 (1995).

3. Nielsen, M.A. & Chuang, I.L. *Quantum Computation and Quantum Information*, Chapter 10.

4. Preskill, J. "Lecture Notes for Physics 219: Quantum Computation" [Chapter 7](https://www.preskill.caltech.edu/ph229/notes/chap7.pdf)

5. Gottesman, D. "An Introduction to Quantum Error Correction" [arXiv:quant-ph/0004072](https://arxiv.org/abs/quant-ph/0004072)

---

**Word Count:** ~2400 words
**Review Guide Created:** February 9, 2026
