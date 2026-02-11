# Comprehensive Qualifying Exam - Grading Rubric

## Overview

**Total Points:** 200
**Passing Score:** 160 (80%)
**Number of Problems:** 10

---

## Section A: Quantum Mechanics (60 points)

### Problem 1: Quantum Dynamics (20 points)

#### Part (a): Time Evolution (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Correct energy eigenvalues | 1 | $E_n = \hbar\omega(n+1/2)$ |
| Apply time evolution operator | 2 | $e^{-iE_n t/\hbar}$ phases |
| Final state expression | 2 | Correct relative phase |

#### Part (b): Position Expectations (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Recognize $\langle x \rangle = 0$ | 2 | Selection rules |
| Calculate $\langle x^2 \rangle$ | 2 | Include oscillating term |
| Correct time dependence | 1 | Frequency $2\omega$ |

#### Part (c): Uncertainty (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Use $\Delta x = \sqrt{\langle x^2\rangle}$ | 2 | Since $\langle x\rangle = 0$ |
| Show oscillation | 2 | "Breathing" at $2\omega$ |
| Min/max values | 1 | Correct range |

#### Part (d): Cat State Interpretation (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Explain superposition structure | 2 | Two components in phase space |
| Discuss decoherence | 3 | Environment coupling, rate |

---

### Problem 2: Angular Momentum and Perturbation (20 points)

#### Part (a): Matrices (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Correct $\hat{H}_0$ matrix | 2 | Diagonal with $\pm\hbar\omega_0$ |
| Correct $\hat{V}$ matrix | 3 | Off-diagonal structure |

#### Part (b): First-Order Corrections (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Apply formula $E^{(1)} = \langle n|V|n\rangle$ | 2 | Correct approach |
| All three corrections zero | 3 | Diagonal elements of V |

#### Part (c): Second-Order for $m=0$ (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Set up second-order formula | 2 | Sum over states |
| Identify matrix elements | 2 | $\langle m|V|0\rangle$ |
| Correct result (zero) | 1 | No coupling |

#### Part (d): Spin Squeezing (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Explain twisting mechanism | 2 | Correlation generation |
| Standard quantum limit | 2 | $\Delta S \sim \sqrt{N}$ |
| Metrological advantage | 1 | Beyond SQL |

---

### Problem 3: Scattering and Path Integrals (20 points)

#### Part (a): Delta Function Transmission (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Write solutions in regions | 2 | Plane waves |
| Apply boundary conditions | 3 | Continuity + derivative jump |
| Derive $T(E)$ | 2 | Correct formula |

#### Part (b): Limiting Cases (6 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| $\alpha \to \infty$: $T \to 0$ | 3 | Perfect reflection |
| $\alpha \to 0$: $T \to 1$ | 3 | Perfect transmission |

#### Part (c): Path Integral (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Set up path integral | 2 | Discretization |
| Evaluate Gaussian integrals | 3 | Correct technique |
| Final propagator | 2 | Correct form |

---

## Section B: Quantum Information/Computing (60 points)

### Problem 4: W State Entanglement (20 points)

#### Part (a): Reduced Density Matrix (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Correct partial trace | 3 | Over qubit C |
| Final matrix | 2 | Diagonal form |

#### Part (b): Entanglement of Formation (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Calculate concurrence | 3 | $C = 0$ |
| Conclude $E_F = 0$ | 2 | Separable |

#### Part (c): W vs GHZ Comparison (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Analyze particle loss | 3 | Both cases |
| Correct conclusion | 2 | W more robust |

#### Part (d): Circuit Design (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Valid circuit | 4 | Produces W state |
| Correct gate sequence | 1 | Logical order |

---

### Problem 5: Quantum Algorithms (20 points)

#### Part (a): Hidden Subgroup Problem (6 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Define HSP | 3 | Group, subgroup, function |
| Connect to Shor and Simon | 3 | Specific groups |

#### Part (b): Phase Estimation (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Draw QPE circuit | 3 | Correct structure |
| Calculate specific outcome | 4 | $|011\rangle$ for $\phi = 3/8$ |

#### Part (c): HHL Algorithm (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Conditions on A | 2 | Sparse, well-conditioned |
| Output is quantum state | 3 | Not full classical vector |
| Limitations | 2 | Readout, state prep |

---

### Problem 6: Quantum Channels (20 points)

#### Part (a): Quantum Capacity (6 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Coherent information formula | 4 | Correct expression |
| Regularization | 2 | Limit over $n$ uses |

#### Part (b): Amplitude Damping Capacity (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Anti-degradable property | 4 | For $\gamma \geq 1/2$ |
| Conclude $Q = 0$ | 3 | Correct reasoning |

#### Part (c): Entanglement-Assisted Capacity (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Superdense coding | 4 | 2 bits per qubit |
| Maximum advantage | 3 | Factor of 2 |

---

## Section C: Quantum Error Correction (60 points)

### Problem 7: Stabilizer Codes (20 points)

#### Part (a): Commutation and Logical Ops (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Verify commutation | 2 | Check all pairs |
| Find $\bar{X}, \bar{Z}$ | 3 | Correct operators |

#### Part (b): Y₃ Error Syndrome (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Analyze each stabilizer | 3 | Commutation with Y₃ |
| Correct syndrome | 2 | $(-1,-1,+1,+1)$ |

#### Part (c): Perfect Code (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| State Hamming bound | 3 | Quantum version |
| Verify saturation | 2 | Calculation |

#### Part (d): Non-CSS Consequences (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| No transversal CNOT | 3 | Explain why |
| Limited transversal gates | 2 | Which ones work |

---

### Problem 8: Surface Codes (20 points)

#### Part (a): Code Parameters (6 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Physical qubits | 2 | $n \approx 2d^2$ |
| Stabilizer counts | 2 | Approximately correct |
| Code rate | 2 | Tends to 0 |

#### Part (b): Distance Calculation (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Apply threshold formula | 3 | Correct equation |
| Solve for $d$ | 4 | $d \geq 23$ |

#### Part (c): Lattice Surgery (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Describe procedure | 3 | Merge/split |
| Time overhead | 2 | $O(d)$ |
| Justify acceptability | 2 | 2D locality |

---

### Problem 9: Fault Tolerance (20 points)

#### Part (a): Eastin-Knill (6 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| State theorem | 4 | No universal transversal |
| Implication | 2 | Need magic states |

#### Part (b): Magic State Distillation (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Output error formula | 4 | Cubic reduction |
| Count levels | 3 | 4 levels |

#### Part (c): Reed-Muller Code (7 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Identify problems | 4 | Low distance, overhead |
| Trade-offs | 3 | Pros and cons |

---

## Section D: Integration (20 points)

### Problem 10: VQE Design (20 points)

#### Part (a): Variational Principle (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| State principle | 2 | $\langle H \rangle \geq E_0$ |
| Explain proof | 3 | Eigenstate expansion |

#### Part (b): Hardware-Efficient Ansatz (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| Parameter count | 2 | $O(nL)$ |
| Depth | 1 | $O(L)$ |
| Barren plateaus | 2 | Exponentially small gradients |

#### Part (c): Error Mitigation (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| ZNE explanation | 2 | Noise extrapolation |
| PEC explanation | 2 | Quasi-probability |
| Limitations | 1 | Overhead, assumptions |

#### Part (d): Research Frontier (5 points)
| Component | Points | Criteria |
|-----------|--------|----------|
| QPE role | 2 | Direct eigenvalue |
| Resource estimates | 2 | Qubits, T-gates |
| Crossover point | 1 | $n \sim 50-100$ |

---

## Score Summary

### By Section

| Section | Problems | Your Score | Max |
|---------|----------|------------|-----|
| A: QM | 1-3 | ___ | 60 |
| B: QI/QC | 4-6 | ___ | 60 |
| C: QEC | 7-9 | ___ | 60 |
| D: Integration | 10 | ___ | 20 |
| **Total** | | ___ | **200** |

### By Problem

| Problem | Topic | Score | Max |
|---------|-------|-------|-----|
| 1 | Quantum dynamics | ___ | 20 |
| 2 | Angular momentum | ___ | 20 |
| 3 | Scattering/path integral | ___ | 20 |
| 4 | W state entanglement | ___ | 20 |
| 5 | Algorithms | ___ | 20 |
| 6 | Channels | ___ | 20 |
| 7 | Stabilizer codes | ___ | 20 |
| 8 | Surface codes | ___ | 20 |
| 9 | Fault tolerance | ___ | 20 |
| 10 | VQE integration | ___ | 20 |

---

## Final Assessment

### Pass/Fail Determination

| Criterion | Required | Your Score | Status |
|-----------|----------|------------|--------|
| Total Score | 160/200 | ___ | |
| Section A (QM) | 48/60 | ___ | |
| Section B (QI/QC) | 48/60 | ___ | |
| Section C (QEC) | 48/60 | ___ | |
| Section D (Integration) | 16/20 | ___ | |

### Overall Status

| Total | Percentage | Interpretation |
|-------|------------|----------------|
| 180-200 | 90-100% | Pass with Distinction |
| 160-179 | 80-89% | Pass |
| 140-159 | 70-79% | Conditional Pass |
| <140 | <70% | Did Not Pass |

---

## Month 46 Complete Assessment

| Week | Exam | Score | Status |
|------|------|-------|--------|
| 181 | QM Written | ___ | |
| 182 | QI/QC Written | ___ | |
| 183 | QEC Written | ___ | |
| 184 | Comprehensive | ___ | |
| **Average** | | ___ | |

### Readiness for Oral Exams (Month 47)

| Criterion | Rating (1-10) |
|-----------|---------------|
| Written exam performance | |
| Topic coverage completeness | |
| Integration ability | |
| Confidence level | |
| **Overall Readiness** | |

---

*This rubric provides detailed grading criteria for fair and consistent assessment.*
