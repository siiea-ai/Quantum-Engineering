# Week 179: Oral Exam Practice - NISQ Algorithms

## Overview

Practice these questions aloud, timing yourself to 3-5 minutes per response. Focus on explaining both the "what" and the "why" of each concept.

---

## Section A: VQE Questions (12 Questions)

### A1. VQE Basics

**Question:** Explain the Variational Quantum Eigensolver algorithm.

**Key Points:**
- Variational principle: $$\langle\psi|H|\psi\rangle \geq E_0$$
- Hybrid quantum-classical loop
- Parameterized quantum circuit (ansatz)
- Classical optimizer updates parameters
- Converges to ground state approximation

**Follow-up:** Why is VQE considered "NISQ-friendly"?

---

### A2. Ansatz Design

**Question:** Compare hardware-efficient and chemistry-inspired ansatze for VQE.

**Key Points:**

| Aspect | Hardware-Efficient | UCCSD |
|--------|-------------------|-------|
| Design | Native gates | Excitation operators |
| Depth | Shallow | Deep |
| Symmetries | Not preserved | Particle-conserving |
| Barren plateaus | Susceptible | More resistant |

**Follow-up:** When would you choose each?

---

### A3. Measurement Overhead

**Question:** Explain how the Hamiltonian is measured in VQE.

**Key Points:**
- Decompose $$H = \sum_i c_i P_i$$ (Pauli strings)
- Measure each Pauli term separately
- Combine with classical post-processing
- Grouping reduces measurement overhead

**Follow-up:** What is qubit-wise commuting grouping?

---

### A4. Barren Plateaus

**Question:** What are barren plateaus and why are they problematic?

**Key Points:**
- Variance of gradients vanishes exponentially
- Random deep circuits especially susceptible
- Optimization becomes impossible
- Need exponential shots to detect gradients

**Follow-up:** How can they be avoided?

---

### A5. Classical Optimization

**Question:** Discuss classical optimization strategies for VQE.

**Key Points:**
- Gradient-free: COBYLA, Nelder-Mead (noise tolerant)
- Gradient-based: L-BFGS, Adam (faster convergence)
- Parameter-shift rule for exact gradients
- SPSA for stochastic gradients
- Trade-offs: noise tolerance vs convergence speed

**Follow-up:** Why might gradient-free methods be preferred on noisy hardware?

---

### A6. VQE for Chemistry

**Question:** Describe the workflow for using VQE to find molecular ground states.

**Key Points:**
1. Choose basis set and compute integrals (classical)
2. Map to qubit Hamiltonian (Jordan-Wigner/Bravyi-Kitaev)
3. Select active space if needed
4. Run VQE with appropriate ansatz
5. Apply error mitigation
6. Post-process for final energy

**Follow-up:** What is an "active space" and why is it used?

---

### A7. ADAPT-VQE

**Question:** Explain the ADAPT-VQE algorithm and its advantages.

**Key Points:**
- Iteratively grows ansatz from operator pool
- Adds operator with largest gradient
- Re-optimizes all parameters
- Problem-adapted circuit depth
- Often shorter than UCCSD

**Follow-up:** What are typical operators in the pool?

---

### A8. VQE Limitations

**Question:** What are the main limitations of VQE?

**Key Points:**
- Barren plateaus for deep circuits
- Noise accumulates with depth
- Classical optimization can get stuck
- Measurement overhead
- Chemical accuracy remains challenging

**Follow-up:** How might error correction change this picture?

---

### A9. State Preparation

**Question:** How is the initial state prepared in VQE for chemistry?

**Key Points:**
- Start from Hartree-Fock state
- Classically computed mean-field solution
- Easy to prepare on quantum computer
- Ansatz builds correlations on top

**Follow-up:** Why start from Hartree-Fock rather than random state?

---

### A10. Orbital Optimization

**Question:** What is orbital optimization in VQE?

**Key Points:**
- Optimize molecular orbitals alongside VQE
- Reduces required active space size
- Can be done classically between VQE iterations
- Improves convergence and accuracy

**Follow-up:** How does this relate to CASSCF?

---

### A11. Excited States

**Question:** How can VQE be extended to find excited states?

**Key Points:**
- Fold excited states: modify Hamiltonian
- Variational quantum deflation: add penalty for overlap with found states
- Subspace methods: diagonalize in found state basis
- Each requires additional quantum resources

**Follow-up:** Why is finding excited states important for chemistry?

---

### A12. VQE vs QPE

**Question:** Compare VQE and Quantum Phase Estimation for chemistry.

**Key Points:**

| Aspect | VQE | QPE |
|--------|-----|-----|
| Circuit depth | Shallow | Deep |
| Precision | Limited by ansatz | Arbitrary |
| Noise tolerance | Better | Worse |
| Resources | NISQ-compatible | Requires QEC |

**Follow-up:** When will QPE become practical?

---

## Section B: QAOA Questions (10 Questions)

### B1. QAOA Basics

**Question:** Explain the Quantum Approximate Optimization Algorithm.

**Key Points:**
- Encode problem in cost Hamiltonian $$H_C$$
- Alternating cost and mixer unitaries
- $$|\gamma,\beta\rangle = \prod_p e^{-i\beta_p H_M}e^{-i\gamma_p H_C}|+\rangle$$
- Optimize parameters classically
- Measure final state for solution

**Follow-up:** Why alternate between cost and mixer?

---

### B2. MaxCut Problem

**Question:** How is MaxCut encoded in QAOA?

**Key Points:**
- Cost function: edges cut by partition
- $$H_C = \sum_{(i,j)\in E}\frac{1-Z_iZ_j}{2}$$
- Ground state = maximum cut
- ZZ interaction for each edge

**Follow-up:** Derive the cost Hamiltonian for a 3-node graph.

---

### B3. Mixer Hamiltonian

**Question:** What is the purpose of the mixer Hamiltonian?

**Key Points:**
- $$H_M = \sum_i X_i$$ (standard mixer)
- Creates superpositions between solutions
- Enables quantum exploration
- Can be modified for constraints

**Follow-up:** How would you modify the mixer for a constrained problem?

---

### B4. Approximation Ratios

**Question:** Discuss QAOA approximation ratios for MaxCut.

**Key Points:**
- $$p=1$$ on 3-regular: $$\geq 0.6924$$
- Increases with $$p$$
- $$p \rightarrow \infty$$: approaches 1 (adiabatic)
- Classical GW algorithm: 0.878

**Follow-up:** Is QAOA competitive with classical algorithms?

---

### B5. QAOA Parameter Landscape

**Question:** What is known about the QAOA parameter landscape?

**Key Points:**
- Periodic in $$\gamma$$ and $$\beta$$
- Optimal parameters concentrate for similar instances
- Enables transfer learning
- Local minima exist but often shallow

**Follow-up:** How can concentration be exploited?

---

### B6. QAOA Variants

**Question:** Describe some variants of QAOA.

**Key Points:**
- **QAOA+:** Additional single-qubit rotations
- **Warm-start QAOA:** Initialize from classical solution
- **Recursive QAOA:** Fix variables and recurse
- **Multi-angle QAOA:** Different parameters per gate

**Follow-up:** When would you use warm-starting?

---

### B7. QAOA for Optimization

**Question:** Beyond MaxCut, what problems can QAOA address?

**Key Points:**
- Portfolio optimization
- Traveling salesman
- Graph coloring
- Satisfiability
- Any QUBO problem

**Follow-up:** How do you handle constraints?

---

### B8. QAOA Depth Requirements

**Question:** How deep should a QAOA circuit be?

**Key Points:**
- Problem-dependent
- $$p = O(\sqrt{n})$$ often sufficient
- Deeper is not always better (noise)
- Trade-off: approximation vs noise

**Follow-up:** What limits the useful depth on current hardware?

---

### B9. QAOA vs Classical

**Question:** Can QAOA outperform classical algorithms?

**Key Points:**
- For some instances, yes (theoretically)
- In practice, classical often competitive
- Hardware noise limits current demonstrations
- Debate ongoing in community

**Follow-up:** What would constitute a QAOA "advantage"?

---

### B10. QAOA Implementation

**Question:** Discuss practical considerations for implementing QAOA.

**Key Points:**
- Native ZZ gates help (some platforms)
- Connectivity matters for dense problems
- Parameter optimization is costly
- Error mitigation typically needed

**Follow-up:** Which hardware platform is best for QAOA?

---

## Section C: Error Mitigation Questions (8 Questions)

### C1. Error Mitigation Overview

**Question:** Why is error mitigation important for NISQ algorithms?

**Key Points:**
- Devices are noisy ($$10^{-3}$$ to $$10^{-2}$$ error rates)
- Full error correction not yet practical
- Mitigation reduces effective error
- Enables meaningful computation despite noise

**Follow-up:** How does mitigation differ from correction?

---

### C2. Zero-Noise Extrapolation

**Question:** Explain zero-noise extrapolation.

**Key Points:**
- Measure at multiple noise levels
- Artificially scale noise (folding, stretching)
- Extrapolate to zero noise
- Model-dependent (linear, exponential)

**Follow-up:** What are the limitations of ZNE?

---

### C3. Probabilistic Error Cancellation

**Question:** How does probabilistic error cancellation work?

**Key Points:**
- Decompose noisy inverse as quasi-probability
- Sample operations with signs
- Combine results to cancel noise
- Exponential sampling overhead

**Follow-up:** When is PEC practical?

---

### C4. Symmetry Verification

**Question:** How do symmetries help with error mitigation?

**Key Points:**
- Many Hamiltonians have conserved quantities
- Errors can violate these symmetries
- Post-select on correct symmetry sector
- Reduces symmetry-breaking errors

**Follow-up:** What symmetries are useful for chemistry?

---

### C5. Virtual Distillation

**Question:** Explain virtual distillation for error mitigation.

**Key Points:**
- Use multiple copies of noisy state
- Measure $$\text{Tr}(O\rho^n)/\text{Tr}(\rho^n)$$
- Errors suppressed as $$p^n$$
- Trade-off: copies vs error reduction

**Follow-up:** What circuit is needed for this?

---

### C6. Combining Techniques

**Question:** How can error mitigation techniques be combined?

**Key Points:**
- Use complementary methods
- ZNE for systematic errors
- Symmetry for easy checks
- PEC for residual errors
- Order matters for efficiency

**Follow-up:** Give an example strategy for VQE.

---

### C7. Mitigation Overhead

**Question:** Discuss the resource overhead of error mitigation.

**Key Points:**
- ZNE: ~3-10× more runs
- PEC: Exponential in circuit depth
- Symmetry: Post-selection reduces statistics
- Virtual distillation: Multiplies qubit count

**Follow-up:** At what point is mitigation impractical?

---

### C8. Mitigation vs Correction

**Question:** Compare error mitigation and error correction.

**Key Points:**

| Aspect | Mitigation | Correction |
|--------|------------|------------|
| Overhead | Sampling | Qubits |
| Threshold | None | Yes ($$\sim 1\%$$) |
| Scaling | Exponential | Polynomial |
| Hardware | NISQ | Fault-tolerant |

**Follow-up:** When will the transition to QEC happen?

---

## Section D: Integration Questions (5 Questions)

### D1. Algorithm Selection

**Question:** Given a specific problem, how do you choose between VQE and QAOA?

**Key Points:**
- VQE: continuous optimization, physics/chemistry
- QAOA: discrete optimization, combinatorics
- Consider problem structure
- Consider hardware constraints

---

### D2. Hardware Matching

**Question:** How do you match NISQ algorithms to hardware platforms?

**Key Points:**
- Connectivity: affects SWAP overhead
- Gate fidelity: limits circuit depth
- Qubit count: problem size limit
- Native gates: some operations more natural

---

### D3. Quantum Utility

**Question:** What does "quantum utility" mean in the NISQ era?

**Key Points:**
- Quantum computer provides useful results
- Not necessarily faster than classical
- May guide classical computation
- Intermediate step toward advantage

---

### D4. Near-Term Applications

**Question:** What applications are most promising for NISQ devices?

**Key Points:**
- Molecular simulation (small molecules)
- Combinatorial optimization (specific instances)
- Quantum machine learning (exploration)
- Quantum simulation (analog mode)

---

### D5. Future Outlook

**Question:** What advances are needed for NISQ algorithms to become practical?

**Key Points:**
- Higher gate fidelities
- Better error mitigation
- More efficient classical optimization
- Larger coherent qubit counts
- Eventually: error correction

---

## Oral Exam Tips

1. **Know the algorithms:** Can you explain VQE/QAOA without notes?
2. **Understand trade-offs:** Every choice has pros and cons
3. **Be quantitative:** Use numbers and estimates
4. **Acknowledge limitations:** Be honest about what we don't know
5. **Connect to hardware:** Algorithms don't exist in vacuum
