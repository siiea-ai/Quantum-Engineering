# Week 98: Quantum Errors

## Month 25: QEC Fundamentals I | Year 2: Advanced Quantum Science

---

## Overview

**Duration:** 7 days (Days 680-686)
**Focus:** Quantum error theory, noise channels, and first quantum codes
**Prerequisites:** Week 97 Classical Error Correction Review

---

## Status: ✅ COMPLETE

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 680 | Monday | Introduction to Quantum Errors | ✅ Complete |
| 681 | Tuesday | CPTP Maps and Kraus Operators | ✅ Complete |
| 682 | Wednesday | Depolarizing and Amplitude Damping | ✅ Complete |
| 683 | Thursday | Phase Damping and Combined Noise | ✅ Complete |
| 684 | Friday | Three-Qubit Bit-Flip Code | ✅ Complete |
| 685 | Saturday | Three-Qubit Phase-Flip Code | ✅ Complete |
| 686 | Sunday | Week Synthesis and Shor Code Preview | ✅ Complete |

---

## Learning Objectives

By the end of Week 98, you will be able to:

1. **Understand the three obstacles** to quantum error correction
2. **Work with Pauli errors** X (bit-flip), Z (phase-flip), Y (combined)
3. **Apply the Kraus representation** for quantum channels
4. **Analyze physical noise models** (depolarizing, amplitude/phase damping)
5. **Construct three-qubit repetition codes** (bit-flip and phase-flip)
6. **Understand the Shor [[9,1,3]] code** via concatenation

---

## Key Concepts

### Quantum Error Types

| Error | Operator | Action | Classical Analog |
|-------|----------|--------|------------------|
| Bit-flip | X | $\|0\rangle \leftrightarrow \|1\rangle$ | Bit flip |
| Phase-flip | Z | $\|1\rangle \to -\|1\rangle$ | None (quantum only) |
| Combined | Y = iXZ | Both flips | None |

### CPTP Maps

$$\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger, \quad \sum_k E_k^\dagger E_k = I$$

### Physical Noise Parameters

| Parameter | Physical Process | Formula |
|-----------|-----------------|---------|
| T₁ | Energy relaxation | $\gamma = 1 - e^{-t/T_1}$ |
| T₂ | Total decoherence | $1/T_2 = 1/(2T_1) + 1/T_\phi$ |
| T_φ | Pure dephasing | $\lambda = 1 - e^{-t/T_\phi}$ |

### Three-Qubit Codes

| Code | Logical States | Stabilizers | Corrects |
|------|---------------|-------------|----------|
| Bit-flip | $\|000\rangle$, $\|111\rangle$ | $Z_1Z_2$, $Z_2Z_3$ | X errors |
| Phase-flip | $\|{+}{+}{+}\rangle$, $\|{-}{-}{-}\rangle$ | $X_1X_2$, $X_2X_3$ | Z errors |

### Shor Code [[9,1,3]]

- **Construction:** Concatenate phase-flip then bit-flip
- **9 physical qubits** encoding **1 logical qubit**
- **Distance 3:** Corrects any single-qubit Pauli error
- **First complete quantum error correcting code**

---

## Daily Summary

### Day 680: Introduction to Quantum Errors
- Three obstacles: no-cloning, measurement collapse, continuous errors
- Pauli error basis: I, X, Y, Z
- Discretization under syndrome measurement
- Multi-qubit Pauli group

### Day 681: CPTP Maps and Kraus Operators
- CPTP = Completely Positive Trace-Preserving
- Kraus (operator-sum) representation
- Completeness relation
- Choi-Jamiołkowski isomorphism

### Day 682: Depolarizing and Amplitude Damping
- Depolarizing: symmetric Pauli noise
- Amplitude damping: T₁ relaxation
- Unital vs non-unital channels
- Hardware error rates

### Day 683: Phase Damping and Combined Noise
- Phase damping: T₂ dephasing
- T₂ ≤ 2T₁ constraint
- Combined noise models
- Pauli twirling

### Day 684: Three-Qubit Bit-Flip Code
- Encoding circuit
- Syndrome measurement (Z₁Z₂, Z₂Z₃)
- Error correction protocol
- Limitation: Z errors undetected

### Day 685: Three-Qubit Phase-Flip Code
- Hadamard duality with bit-flip
- X-type stabilizers
- Z error correction
- X errors undetected

### Day 686: Week Synthesis and Shor Code
- Complete week review
- Shor code construction
- 8 stabilizers, syndrome table
- Preview of stabilizer formalism

---

## Primary References

- **Nielsen & Chuang** Chapter 8 (Quantum Noise) and 10.1-10.2
- **Preskill Lecture Notes** Ph219 Chapter 7
- **Original Papers:**
  - Shor, P. "Scheme for reducing decoherence..." (1995)

---

## Computational Skills Developed

- Simulating Pauli errors
- Implementing CPTP maps with Kraus operators
- Analyzing noise channels (depolarizing, amplitude/phase damping)
- Building and simulating quantum codes
- Syndrome extraction and error correction

---

## Connection to Week 99

| Week 98 Topic | Week 99 Extension |
|---------------|-------------------|
| Three-qubit codes | Stabilizer formalism |
| Syndrome measurement | Knill-Laflamme conditions |
| Shor code | CSS code construction |
| Concatenation | General code families |

---

## What's Next: Week 99

**Week 99: Three-Qubit Codes (Days 687-693)**
- Complete stabilizer formalism
- Knill-Laflamme error correction conditions
- Steane [[7,1,3]] code introduction
- CSS code theory

---

*"Understanding quantum errors is the first step toward protecting quantum information."*

---

**Week 98 Complete!** 7/7 days (100%)
