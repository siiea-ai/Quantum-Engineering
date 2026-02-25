# Week 169: Classical to Quantum Codes

## Overview

**Days:** 1177-1183
**Theme:** From Classical Error Correction to Quantum Error Correction

This week bridges classical coding theory with quantum error correction, establishing the theoretical foundation for all subsequent material. We develop the Knill-Laflamme conditions that characterize when quantum errors can be corrected, and construct the first quantum codes.

---

## Daily Schedule

### Day 1177 (Monday): Classical Linear Codes Review

**Topics:**
- Linear codes over finite fields
- Generator and parity-check matrices
- Syndrome-based error detection and correction
- Hamming distance and error-correcting capability

**Key Concepts:**
- Code parameters $$[n, k, d]$$: length, dimension, minimum distance
- Syndrome: $$s = Hx^T$$ reveals error pattern
- Hamming bound: $$\sum_{i=0}^{t} \binom{n}{i} \leq 2^{n-k}$$

### Day 1178 (Tuesday): Quantum Error Types

**Topics:**
- Bit-flip errors ($$X$$ operator)
- Phase-flip errors ($$Z$$ operator)
- Combined errors ($$Y = iXZ$$)
- General Pauli errors and the Pauli group
- Discretization of errors

**Key Concepts:**
- Error discretization: Any error can be expanded in Pauli basis
- $$\mathcal{E}(\rho) = \sum_j E_j \rho E_j^\dagger$$ with $$E_j \in \{I, X, Y, Z\}^{\otimes n}$$

### Day 1179 (Wednesday): Knill-Laflamme Conditions

**Topics:**
- Necessary conditions for quantum error correction
- Sufficient conditions and the recovery map
- Degeneracy in quantum codes
- Non-degenerate vs degenerate codes

**Key Result:**
$$\boxed{\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = C_{ab} \delta_{ij}}$$

### Day 1180 (Thursday): Three-Qubit Codes

**Topics:**
- Bit-flip code: $$|0\rangle \to |000\rangle$$, $$|1\rangle \to |111\rangle$$
- Phase-flip code: $$|0\rangle \to |{+}{+}{+}\rangle$$, $$|1\rangle \to |{-}{-}{-}\rangle$$
- Syndrome measurement circuits
- Limitations of simple codes

**Key Insight:** Neither code alone corrects general errors.

### Day 1181 (Friday): The Shor Code

**Topics:**
- Nine-qubit Shor code construction
- Concatenation of bit-flip and phase-flip codes
- Complete error correction for single-qubit errors
- Syndrome measurement and recovery

**Encoding:**
$$|0_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)^{\otimes 3}$$
$$|1_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle - |111\rangle)^{\otimes 3}$$

### Day 1182 (Saturday): Code Analysis Workshop

**Activities:**
- Verify Knill-Laflamme conditions for Shor code
- Calculate code parameters
- Implement syndrome measurement in Qiskit
- Error correction simulation

### Day 1183 (Sunday): Review and Integration

**Activities:**
- Comprehensive problem solving
- Oral exam practice
- Self-assessment
- Week 170 preview: Stabilizer formalism

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Compute syndromes** for classical linear codes and identify correctable errors
2. **State and prove** the Knill-Laflamme error correction conditions
3. **Construct** the 3-qubit bit-flip and phase-flip codes
4. **Derive** the 9-qubit Shor code and verify its error correction properties
5. **Implement** syndrome measurement circuits
6. **Distinguish** between degenerate and non-degenerate codes

---

## Key Formulas

| Concept | Formula |
|---------|---------|
| Classical syndrome | $$s = He^T$$ |
| Knill-Laflamme | $$P E_a^\dagger E_b P = C_{ab} P$$ |
| Bit-flip encoding | $$\|0_L\rangle = \|000\rangle$$, $$\|1_L\rangle = \|111\rangle$$ |
| Shor code distance | $$d = 3$$ (corrects any single-qubit error) |

---

## Files in This Week

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `Review_Guide.md` | Comprehensive theory review (2000+ words) |
| `Problem_Set.md` | 25-30 PhD qualifying exam level problems |
| `Problem_Solutions.md` | Complete solutions with detailed derivations |
| `Oral_Practice.md` | Oral exam questions and discussion points |
| `Self_Assessment.md` | Checklist and self-evaluation rubric |

---

## References

1. Knill, E. & Laflamme, R. "Theory of Quantum Error-Correcting Codes" [arXiv:quant-ph/9604034](https://arxiv.org/abs/quant-ph/9604034)
2. Shor, P.W. "Scheme for reducing decoherence in quantum computer memory" (1995)
3. Nielsen & Chuang, Chapter 10.1-10.3
4. Preskill, Chapter 7.1-7.3

---

**Week 169 Created:** February 9, 2026
