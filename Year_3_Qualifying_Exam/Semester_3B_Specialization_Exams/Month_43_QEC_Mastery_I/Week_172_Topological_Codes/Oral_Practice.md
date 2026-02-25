# Week 172: Topological Codes - Oral Practice

## Introduction

This document provides oral examination practice for topological codes, the leading approach to fault-tolerant quantum computing. Questions cover the toric code, anyons, the surface code, and practical implementation.

---

## Short-Answer Questions (2-3 minutes each)

### Question 1: What is the Toric Code?

**Examiner asks:** "Describe the toric code and its key properties."

**Model answer:** "The toric code is a quantum error-correcting code defined on a square lattice with periodic boundary conditions, forming a torus. Qubits sit on edges, and stabilizers are defined on vertices (products of X on adjacent edges) and plaquettes (products of Z around faces). For an L-by-L lattice, the code has parameters [[2L^2, 2, L]], encoding 2 logical qubits with distance L. The ground state degeneracy of 4 is topological—it arises from the two non-contractible cycles on the torus and cannot be lifted by local perturbations."

---

### Question 2: What are Anyons?

**Examiner asks:** "Explain anyonic excitations in the toric code."

**Key points:**
- Two types: e (electric charges) and m (magnetic vortices)
- e created by Z errors, m created by X errors
- Braiding e around m gives -1 phase
- Non-trivial statistics (neither bosons nor fermions)
- Fusion: e x e = 1, m x m = 1, e x m = fermion

---

### Question 3: Surface Code vs Toric Code

**Examiner asks:** "What is the difference between the surface code and toric code?"

**Model answer:** "The surface code is essentially the toric code on a plane with boundaries instead of a torus. The boundaries break the periodicity: rough boundaries allow X-type logical operators to terminate, smooth boundaries allow Z-type to terminate. This reduces the encoded qubits from 2 to 1. The surface code is more practical because it doesn't require periodic boundary conditions and fits naturally on a finite 2D chip. Parameters are [[d^2, 1, d]] for the rotated layout, where d is the code distance."

---

### Question 4: Error Threshold

**Examiner asks:** "What is the error threshold of the surface code and why is it important?"

**Key points:**
- Threshold ~1% for circuit-level noise
- Below threshold: increasing d exponentially suppresses errors
- Above threshold: errors accumulate faster than correction
- Highest known threshold for 2D local codes
- Makes surface code leading candidate for near-term quantum computing

---

### Question 5: Logical Operations

**Examiner asks:** "How do you perform universal quantum computation on the surface code?"

**Model answer:** "The surface code has no transversal non-Clifford gates, so we need alternative approaches. Clifford gates are implemented via lattice surgery—merging and splitting code patches to perform measurements like Z1Z2 and X1X2, which can be composed into CNOT. For T gates, we use magic state distillation: prepare noisy T states, use distillation protocols to purify them, then inject into the computation. The combination of lattice surgery and magic state injection gives universal computation, though with significant overhead."

---

## Extended Explanation Questions (5-10 minutes)

### Question 6: Toric Code Ground States

**Examiner asks:** "Derive the ground state degeneracy of the toric code and explain its topological nature."

**Structure:**

1. **Stabilizer counting** (2 min):
   - L^2 vertex operators with 1 constraint
   - L^2 plaquette operators with 1 constraint
   - Total independent: 2L^2 - 2
   - Code dimension: k = 2L^2 - (2L^2 - 2) = 2

2. **Logical operators** (2 min):
   - Non-contractible cycles around torus
   - Two independent cycles (horizontal, vertical)
   - Each gives one logical qubit
   - Four ground states: |00⟩, |01⟩, |10⟩, |11⟩

3. **Topological nature** (2 min):
   - No local operator can distinguish ground states
   - Logical operators have minimum weight L (span the system)
   - Degeneracy depends on topology (genus g → 4^g)
   - Protected by energy gap, not symmetry

---

### Question 7: Surface Code Decoding

**Examiner asks:** "Explain how errors are corrected in the surface code using MWPM."

**Cover:**

1. **Error model:**
   - X errors create pairs of m excitations (Z-syndrome)
   - Z errors create pairs of e excitations (X-syndrome)
   - Errors appear as violated stabilizers

2. **Syndrome graph:**
   - Nodes at syndrome locations (violated stabilizers)
   - Edge weights proportional to error probability (distance)
   - Include boundary nodes for open strings

3. **MWPM algorithm:**
   - Find minimum weight matching of all nodes
   - Pairs syndromes optimally
   - Edmonds' algorithm: O(n^3) time

4. **Correction:**
   - For each matched pair, errors on connecting path
   - Apply recovery or update Pauli frame

5. **Threshold emergence:**
   - Below threshold: MWPM correctly identifies error class
   - Above threshold: errors percolate, overwhelm decoder

---

### Question 8: Lattice Surgery

**Examiner asks:** "Describe lattice surgery and how it implements a logical CNOT."

**Detailed explanation:**

1. **Basic operations:**
   - **Merge:** Bring patches together, measure joint stabilizers
   - **Split:** Reverse of merge, separate patches

2. **ZZ measurement:**
   - Merge patches horizontally
   - New stabilizers span the boundary
   - Measures Z_1 Z_2 (product of logical Z)
   - Outcome determines measurement result

3. **CNOT implementation:**
   - Need to measure XX on ancilla and ZZ on target
   - Sequence of merges and splits
   - Corrections based on measurement outcomes

4. **Time overhead:**
   - Each merge/split takes ~d code cycles
   - Total CNOT: ~4d cycles
   - Much slower than transversal but fault-tolerant

---

### Question 9: Magic State Distillation

**Examiner asks:** "Why is magic state distillation necessary and how does it work?"

**Cover:**

1. **The problem:**
   - Surface code has no transversal T gate
   - Need T for universal computation
   - Can't just apply T directly (not fault-tolerant)

2. **Magic states:**
   - |T⟩ = T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2
   - Consuming |T⟩ + Clifford operations = T gate
   - Need high-fidelity |T⟩ states

3. **Distillation protocol:**
   - Start with many noisy |T⟩ states (error p)
   - Encode in [[15,1,3]] code
   - Measure stabilizers
   - Post-select on correct syndrome
   - Output: fewer states with error ~p^3

4. **Overhead:**
   - 15-to-1 protocol: 15 input → 1 output
   - Multiple levels needed for high fidelity
   - Dominates resource cost of surface code QC

---

## Deep-Dive Questions (15-20 minutes)

### Question 10: Complete Surface Code Architecture

**Examiner asks:** "Design a fault-tolerant quantum computer based on the surface code."

**Full design:**

1. **Physical layer:**
   - 2D array of superconducting qubits
   - Nearest-neighbor connectivity
   - Data and measurement qubits interleaved

2. **Error correction layer:**
   - Distance d based on physical error rate
   - For p = 10^-3: d ≈ 15-20
   - Syndrome extraction every ~1 μs

3. **Decoder:**
   - MWPM or Union-Find
   - Must decode faster than syndrome rate
   - Hardware acceleration likely needed

4. **Logical operations:**
   - Lattice surgery for Clifford gates
   - Magic state factories for T gates
   - Pauli frame tracking

5. **Resource estimate:**
   - 1000 logical qubits
   - ~10^6 physical qubits (data + ancilla)
   - + ~10^7 for magic state factories
   - Total: ~10^7 physical qubits

6. **Timeline:**
   - Current (2026): ~100-1000 physical qubits
   - Near-term: First logical qubit demonstrations
   - Long-term: Full fault-tolerant systems

---

### Question 11: Threshold Analysis

**Examiner asks:** "Prove that the surface code has an error threshold."

**Proof outline:**

1. **Mapping to statistical mechanics:**
   - Decoding = finding most likely error
   - Equivalent to random-bond Ising model
   - Syndrome = domain walls in Ising model

2. **Phase transition argument:**
   - Low error: ordered phase (errors confined)
   - High error: disordered phase (errors percolate)
   - Transition at critical p = p_th

3. **Percolation perspective:**
   - Errors form clusters on the lattice
   - Below threshold: clusters remain small
   - Above threshold: infinite cluster spans system

4. **Numerical evidence:**
   - Monte Carlo simulation
   - Independent X/Z: p_th ≈ 11%
   - Depolarizing: p_th ≈ 1%
   - Circuit-level: p_th ≈ 0.5-1%

5. **Implications:**
   - For p < p_th: logical error rate → 0 as d → ∞
   - Scaling: p_L ~ (p/p_th)^{(d+1)/2}
   - Exponential suppression possible

---

### Question 12: Beyond MWPM

**Examiner asks:** "Discuss alternative decoders for the surface code."

**Comparison:**

1. **MWPM (Minimum-Weight Perfect Matching):**
   - Standard decoder
   - O(n^3) time complexity
   - Achieves ~1% threshold
   - Limitation: assumes independent errors

2. **Union-Find:**
   - Nearly linear time: O(n α(n))
   - Simpler to implement
   - Slightly lower threshold
   - Good for real-time decoding

3. **Neural network decoders:**
   - Train on simulated data
   - Can learn correlations
   - Fast inference after training
   - Challenge: generalization to new noise

4. **Optimal decoder (ML):**
   - Maximizes likelihood of correction
   - Exponentially slow (NP-hard)
   - Achieves highest threshold
   - Impractical but useful benchmark

5. **Correlated decoders:**
   - Account for correlated errors
   - Higher threshold in practice
   - More complex implementation

---

## Common Exam Mistakes

1. **Confusing toric and surface:** Remember boundaries change the encoding

2. **Forgetting measurement errors:** Real syndrome extraction is noisy

3. **Underestimating magic state cost:** T gates dominate overhead

4. **Confusing threshold types:** Circuit-level ≠ phenomenological ≠ code capacity

5. **Wrong anyon statistics:** e and m are bosons; ε = e×m is fermion

---

## Key Facts to Know

| Quantity | Value |
|----------|-------|
| Toric code parameters | [[2L^2, 2, L]] |
| Surface code parameters | [[d^2, 1, d]] |
| Circuit-level threshold | ~0.5-1% |
| Phenomenological threshold | ~3% |
| Independent X/Z threshold | ~11% |
| Braiding phase (e around m) | -1 |

---

**Oral Practice Document Created:** February 10, 2026
