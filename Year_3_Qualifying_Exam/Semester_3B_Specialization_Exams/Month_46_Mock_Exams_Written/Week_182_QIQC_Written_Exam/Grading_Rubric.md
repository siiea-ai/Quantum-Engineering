# Quantum Information & Computing Exam - Grading Rubric

## Overview

**Total Points:** 200
**Number of Problems:** 8
**Points per Problem:** 25 each
**Passing Score:** 160 points (80%)

---

## Problem 1: Density Matrices and Quantum States (25 points)

### Part (a): $\rho_{AB}$ Calculation (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Correct partial trace setup | 2 | Trace over C basis states |
| Correct grouping of terms | 2 | Identify which terms contribute |
| Final matrix correct | 2 | All 16 entries correct |

### Part (b): $\rho_A$ and Purity (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Correct trace over B | 2 | Proper partial trace |
| Get $\rho_A = I/2$ | 2 | Maximally mixed |
| Identify as mixed state | 2 | Clear justification |

### Part (c): Von Neumann Entropy (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Find eigenvalues of $\rho_A$ | 2 | Both 1/2 |
| Apply entropy formula | 3 | $-\sum \lambda_i \log \lambda_i$ |
| Interpret as entanglement | 2 | Connect to A-BC entanglement |

### Part (d): GHZ vs W Classification (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Define GHZ-type and W-type | 2 | Correct characterization |
| Analyze $\rho_{AB}$ | 2 | Check if separable |
| Correct conclusion | 2 | GHZ-type with justification |

---

## Problem 2: Entanglement and Bell Inequalities (25 points)

### Part (a): Concurrence (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Correct concurrence formula | 2 | $C = 2|\alpha\beta|$ for pure state |
| Calculate $C = \sin(2\theta)$ | 2 | Correct computation |
| Maximum at $\theta = \pi/4$ | 2 | Bell state identification |

### Part (b): CHSH Calculation (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Define Alice's observables | 2 | $Z$ and $X$ |
| Define Bob's observables | 2 | Correct rotations |
| Calculate correlators | 2 | Use Bell state formula |
| Get $S = 2\sqrt{2}$ | 2 | Correct final value |

### Part (c): Classical Bound Proof (5 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Separable state decomposition | 2 | Convex combination |
| Local hidden variable argument | 2 | $|a|, |b| \leq 1$ |
| Conclude $|S| \leq 2$ | 1 | Proper conclusion |

### Part (d): $\theta = \pi/6$ Analysis (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Calculate concurrence | 2 | $C = \sqrt{3}/2$ |
| Apply $S_{max}$ formula | 2 | $2\sqrt{1+C^2}$ |
| Correct conclusion | 2 | Still violates CHSH |

---

## Problem 3: Quantum Gates and Circuits (25 points)

### Part (a): Universality Proof (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| State the theorem | 2 | CNOT + 1-qubit is universal |
| Explain decomposition strategy | 3 | How to build any 2-qubit U |
| Show entangling capability | 3 | Generate $ZZ$ interactions |

### Part (b): SWAP Decomposition (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Draw correct circuit | 3 | Three CNOTs in correct order |
| Verify one or two basis states | 2 | Show it works |
| Express as formula | 2 | $\text{CNOT}_{01}\text{CNOT}_{10}\text{CNOT}_{01}$ |

### Part (c): $\sqrt{\text{SWAP}}$ (5 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Write correct matrix | 3 | All entries correct |
| Show it's entangling | 2 | Apply to product state |

### Part (d): $R_z(\pi/8)$ Synthesis (5 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Recognize need for synthesis | 2 | Not directly available |
| Propose approximation method | 2 | Solovay-Kitaev or similar |
| Estimate T-count | 1 | Reasonable number |

---

## Problem 4: Quantum Algorithms - Oracle Problems (25 points)

### Part (a): Classical Query Complexity (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Identify worst case | 2 | Need to check majority |
| Calculate bound | 3 | $2^{n-1} + 1$ |
| Justify | 1 | Correct reasoning |

### Part (b): Deutsch-Jozsa Algorithm (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Draw correct circuit | 2 | H gates, oracle, H gates |
| Explain initialization | 2 | Superposition + phase kickback |
| Explain oracle effect | 2 | Phase $(-1)^{f(x)}$ |
| Explain measurement | 2 | Why $|0^n\rangle$ indicates constant |

### Part (c): Correctness Proof (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Calculate amplitude of $|0^n\rangle$ | 3 | Sum of phases |
| Constant case analysis | 2 | Sum = $\pm 2^n$ |
| Balanced case analysis | 1 | Sum = 0 |

### Part (d): Simon's Problem (5 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| State the problem | 2 | Period finding mod 2 |
| Classical complexity | 2 | $\Omega(2^{n/2})$ |
| Quantum advantage | 1 | $O(n)$ |

---

## Problem 5: Quantum Algorithms - Shor and Grover (25 points)

### Part (a): Factoring Reduction (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Choose random $a$ | 1 | With $\gcd(a,N) = 1$ |
| Order finding gives $r$ | 2 | $a^r \equiv 1$ |
| Use $a^{r/2}$ | 2 | If $r$ even |
| Extract factor via gcd | 3 | $\gcd(a^{r/2} \pm 1, N)$ |

### Part (b): QFT Analysis (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Describe periodic input state | 2 | Superposition with period $r$ |
| QFT output structure | 3 | Peaks at $N\ell/r$ |
| Continued fractions | 2 | How to extract $r$ |

### Part (c): Grover Iterations (5 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Calculate optimal iterations | 3 | $\approx 804$ |
| Success probability | 2 | Close to 1 |

### Part (d): Grover Optimality (5 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| State lower bound | 2 | $\Omega(\sqrt{N})$ |
| Proof method | 3 | Polynomial or adversary argument |

---

## Problem 6: Quantum Channels and Noise (25 points)

### Part (a): Amplitude Damping Output (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Apply $K_0\rho K_0^\dagger$ | 3 | All four entries |
| Apply $K_1\rho K_1^\dagger$ | 2 | Correct contribution |
| Sum correctly | 2 | Final matrix |

### Part (b): Fixed Point (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Identify $|0\rangle$ as fixed point | 3 | Show it's invariant |
| Physical interpretation | 3 | T1 decay, ground state |

### Part (c): Dephasing Channel (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Expand Kraus form | 2 | Combine terms |
| Show equivalence | 2 | $(1-p)\rho + pZ\rho Z$ |
| Effect on coherence | 2 | Off-diagonal $\times (1-p)$ |

### Part (d): Composed Channels (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Consider composition | 2 | Order matters |
| Analyze properties | 2 | Different fixed points |
| Conclude not equivalent | 2 | Clear reasoning |

---

## Problem 7: Quantum Complexity (25 points)

### Part (a): BQP Definition (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Polynomial-time quantum circuit | 2 | Uniform family |
| Completeness condition | 2 | $\geq 2/3$ accept if yes |
| Soundness condition | 2 | $\leq 1/3$ accept if no |

### Part (b): BPP $\subseteq$ BQP (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Simulate randomness | 2 | Measure $|+\rangle$ |
| Simulate classical gates | 2 | Toffoli, reversible |
| Efficiency preserved | 2 | Polynomial overhead |

### Part (c): QMA Definition (7 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Define verifier and witness | 2 | Quantum state proof |
| Completeness/soundness | 2 | Correct bounds |
| Role of quantum witness | 1 | Can be entangled, superposed |
| QMA-complete example | 2 | Local Hamiltonian |

### Part (d): NP vs BQP (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| State oracle separation | 3 | Bennett et al. result |
| Explain implications | 2 | Suggests NP $\not\subseteq$ BQP |
| Structural reasons | 1 | Lack of structure in NP-complete |

---

## Problem 8: Quantum Protocols (25 points)

### Part (a): Teleportation (8 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Bell measurement by Alice | 2 | On her two qubits |
| Four outcomes listed | 3 | With Bob's corresponding states |
| Bob's corrections | 3 | Pauli gates based on bits |

### Part (b): Superdense Coding (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Alice's encoding | 2 | Four Pauli operations |
| Bob's decoding | 2 | Bell measurement |
| 2 bits per qubit | 2 | Clear demonstration |

### Part (c): BB84 Security (6 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Calculate error rate | 3 | ~12.5% for given scenario |
| Detection mechanism | 2 | Compare subset |
| Security conclusion | 1 | Abort if too many errors |

### Part (d): No-Cloning (5 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| State theorem | 1 | Cannot copy unknown state |
| Proof | 3 | Inner product argument |
| Importance for crypto | 1 | Eve cannot copy |

---

## Score Summary

| Problem | Topic | Score | Max |
|---------|-------|-------|-----|
| 1 | Density Matrices | ___ | 25 |
| 2 | Entanglement/CHSH | ___ | 25 |
| 3 | Gates/Circuits | ___ | 25 |
| 4 | Oracle Algorithms | ___ | 25 |
| 5 | Shor/Grover | ___ | 25 |
| 6 | Channels/Noise | ___ | 25 |
| 7 | Complexity | ___ | 25 |
| 8 | Protocols | ___ | 25 |
| **Total** | | ___ | **200** |

---

## Grade Thresholds

| Score | Percentage | Interpretation |
|-------|------------|----------------|
| 180-200 | 90-100% | Pass with Distinction |
| 160-179 | 80-89% | Pass |
| 140-159 | 70-79% | Conditional Pass |
| <140 | <70% | Did Not Pass |

---

## Common Errors by Topic

### Density Matrices
- Incorrect partial trace computation
- Confusing pure vs mixed states
- Wrong entropy formula

### Entanglement
- Sign errors in Bell states
- Wrong concurrence formula for mixed states
- Incorrect CHSH calculation

### Gates/Circuits
- Wrong order of gates in decomposition
- Forgetting gate is its own inverse
- Matrix multiplication errors

### Algorithms
- Confusing query and gate complexity
- Wrong QFT formula
- Incorrect Grover iteration count

### Channels
- Kraus operator errors
- Forgetting normalization
- Confusing channel types

### Complexity
- Imprecise definitions
- Missing completeness/soundness bounds
- Confusing BQP with QMA

### Protocols
- Wrong Bell measurement outcomes
- Missing classical communication step
- Incomplete no-cloning proof

---

*Use this rubric to identify specific areas for improvement.*
