# Week 179: Review Guide - NISQ Algorithms

## Introduction

Noisy Intermediate-Scale Quantum (NISQ) algorithms represent the practical approaches to quantum computation in the current era, where devices have tens to hundreds of noisy qubits without full error correction. This review covers the two flagship NISQ algorithms—VQE and QAOA—along with essential error mitigation techniques that enable useful computation despite hardware imperfections.

---

## Part I: Variational Quantum Eigensolver (VQE)

### 1.1 The Variational Principle

The foundation of VQE is the Rayleigh-Ritz variational principle:

$$E_0 \leq E(\vec{\theta}) = \frac{\langle\psi(\vec{\theta})|\hat{H}|\psi(\vec{\theta})\rangle}{\langle\psi(\vec{\theta})|\psi(\vec{\theta})\rangle}$$

For normalized states, the expectation value of the Hamiltonian provides an upper bound on the ground state energy. Minimizing over the parameter space yields the best approximation within the ansatz.

**Why Quantum?**

Classical variational methods struggle with:
- Exponential scaling of Hilbert space
- Sign problem in quantum Monte Carlo
- Strongly correlated systems

Quantum computers can:
- Represent exponentially many amplitudes efficiently
- Prepare and measure entangled states
- Access classically intractable correlations

### 1.2 VQE Algorithm Structure

**Hybrid Quantum-Classical Loop:**

```
1. Initialize parameters θ
2. REPEAT:
   a. Quantum: Prepare |ψ(θ)⟩
   b. Quantum: Measure ⟨H⟩ via Pauli decomposition
   c. Classical: Compute gradient (optional)
   d. Classical: Update θ using optimizer
3. UNTIL converged
4. Return E(θ*) and |ψ(θ*)⟩
```

**Measurement of Hamiltonian:**

Molecular Hamiltonians in second quantization:

$$\hat{H} = \sum_{pq} h_{pq}\hat{a}_p^\dagger\hat{a}_q + \frac{1}{2}\sum_{pqrs}g_{pqrs}\hat{a}_p^\dagger\hat{a}_q^\dagger\hat{a}_r\hat{a}_s$$

After Jordan-Wigner (or Bravyi-Kitaev) transformation:

$$\hat{H} = \sum_i c_i \hat{P}_i$$

where $$\hat{P}_i$$ are Pauli strings (e.g., $$X_1Z_2Y_3$$).

**Grouping Measurements:**

- Qubit-wise commuting (QWC): Same measurement basis for all qubits
- General commuting (GC): More flexible grouping
- Reduces measurement overhead from $$O(N^4)$$ to $$O(N^3)$$ or better

### 1.3 Ansatz Design

**Hardware-Efficient Ansatz (HEA):**

Uses native gates of the quantum device:

$$|\psi(\vec{\theta})\rangle = \prod_{l=1}^{L} \hat{U}_l(\vec{\theta}_l) |0\rangle^{\otimes n}$$

where $$\hat{U}_l = \prod_i R_Y(\theta_i)R_Z(\phi_i) \cdot \text{Entangling layer}$$

**Advantages:**
- Low circuit depth
- Matches hardware constraints
- Easy to implement

**Disadvantages:**
- May not respect physical symmetries
- Prone to barren plateaus
- Suboptimal for specific problems

**Unitary Coupled Cluster (UCC):**

$$|\psi_{UCC}\rangle = e^{\hat{T} - \hat{T}^\dagger}|HF\rangle$$

where $$\hat{T} = \hat{T}_1 + \hat{T}_2 + ...$$ (single, double, ... excitations).

For UCCSD (singles and doubles):

$$\hat{T}_1 = \sum_{ia}t_i^a \hat{a}_a^\dagger\hat{a}_i$$
$$\hat{T}_2 = \sum_{ijab}t_{ij}^{ab}\hat{a}_a^\dagger\hat{a}_b^\dagger\hat{a}_j\hat{a}_i$$

**Advantages:**
- Chemistry-motivated
- Particle number conserving
- Systematically improvable

**Disadvantages:**
- Deep circuits (O(N³-N⁴) gates)
- Complex compilation

**ADAPT-VQE:**

Iteratively grows the ansatz:

1. Start with Hartree-Fock state
2. Compute gradient of all operators in pool
3. Add operator with largest gradient
4. Re-optimize all parameters
5. Repeat until converged

**Advantages:**
- Problem-adapted circuit
- Typically shorter than UCCSD
- Avoids barren plateaus

### 1.4 Optimization Challenges

**Barren Plateaus:**

For deep random circuits, gradients vanish exponentially:

$$\text{Var}[\partial_\theta E] \sim O(1/2^n)$$

**Causes:**
- High expressibility of ansatz
- Global cost functions
- Noise-induced

**Mitigation:**
- Layerwise training
- Correlation-aware initialization
- Problem-structured ansatze
- Local cost functions

**Noise Effects:**

In the presence of noise:

$$E(\vec{\theta}) \rightarrow E_{noisy}(\vec{\theta}) = E(\vec{\theta}) + \epsilon(\vec{\theta})$$

where $$\epsilon$$ depends on noise model and circuit depth.

**Optimization Strategies:**

| Method | Gradient-Free | Noise Tolerance | Convergence |
|--------|---------------|-----------------|-------------|
| Nelder-Mead | Yes | Good | Slow |
| COBYLA | Yes | Good | Medium |
| SPSA | Stochastic | Good | Medium |
| Parameter-shift | No | Medium | Fast |
| Natural gradient | No | Medium | Fast |

### 1.5 VQE for Molecular Systems

**Workflow:**

1. **Classical preprocessing:**
   - Choose basis set (STO-3G, 6-31G, etc.)
   - Compute one- and two-electron integrals
   - Apply fermion-to-qubit mapping

2. **Active space selection:**
   - Full space: N orbitals → N qubits
   - Active space: reduce to chemically relevant orbitals
   - CAS-VQE: treat inactive orbitals classically

3. **Quantum computation:**
   - Run VQE with chosen ansatz
   - Measure energy

4. **Classical post-processing:**
   - Error mitigation
   - Correlation energy corrections

**Example: H₂ Molecule**

- 4 spin-orbitals → 4 qubits (JW mapping)
- Symmetry reduction → 2 qubits possible
- UCCSD requires ~10-20 CNOT gates
- Ground state energy: $$E_0 = -1.137$$ Hartree at equilibrium

---

## Part II: Quantum Approximate Optimization Algorithm (QAOA)

### 2.1 Problem Formulation

**Combinatorial Optimization:**

Many problems reduce to minimizing:

$$C(z) = \sum_{\alpha} C_\alpha(z)$$

where $$z \in \{0,1\}^n$$ and $$C_\alpha$$ are clauses depending on subsets of bits.

**MaxCut Example:**

Given graph $$G = (V, E)$$, maximize edges cut by partition:

$$C(z) = \sum_{(i,j) \in E} \frac{1}{2}(1 - z_iz_j)$$

where $$z_i = \pm 1$$ indicates partition assignment.

**Cost Hamiltonian:**

$$\hat{H}_C = \sum_{(i,j) \in E} \frac{1}{2}(1 - \hat{Z}_i\hat{Z}_j)$$

Ground state corresponds to optimal cut.

### 2.2 QAOA Algorithm

**Initial State:**

$$|s\rangle = |+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n}$$

Equal superposition over all bitstrings.

**Mixer Hamiltonian:**

$$\hat{H}_M = \sum_{i=1}^{n} \hat{X}_i$$

Generates transitions between computational basis states.

**QAOA Ansatz:**

$$|\gamma, \beta\rangle = \prod_{p=1}^{P} e^{-i\beta_p \hat{H}_M} e^{-i\gamma_p \hat{H}_C} |s\rangle$$

**Optimization:**

Find $$\gamma^*, \beta^*$$ to maximize:

$$F_P(\gamma, \beta) = \langle\gamma, \beta|\hat{H}_C|\gamma, \beta\rangle$$

### 2.3 QAOA Analysis

**P = 1 for MaxCut:**

For a single layer on 3-regular graphs:

$$F_1 \geq 0.6924 \cdot C_{max}$$

This is better than random (0.5) but below classical algorithms (0.878 for Goemans-Williamson).

**Depth Scaling:**

- $$P = O(1)$$: Polynomial time, constant approximation
- $$P = O(\text{poly}(n))$$: May approach optimal
- $$P \rightarrow \infty$$: Adiabatic limit (exponential time)

**Concentration Phenomenon:**

For typical instances, optimal parameters concentrate:
- Similar optimal $$\gamma^*, \beta^*$$ for graphs of same structure
- Enables warm-starting and transfer learning

### 2.4 QAOA Variants

**QAOA+:**
- Adds single-qubit rotations after each layer
- Improved performance for same $$P$$

**Warm-Start QAOA:**
- Initialize from classical solution (e.g., greedy or SDP)
- Faster convergence to good solutions

**Recursive QAOA:**
- After measuring, fix some variables
- Recurse on smaller problem
- Hybrid classical-quantum approach

**Multi-Angle QAOA:**
- Different parameters for each gate
- More expressive but harder to optimize

### 2.5 QAOA Applications

**Portfolio Optimization:**

Minimize risk subject to return constraint:

$$\min_x \frac{1}{2}x^T\Sigma x - \mu^T x$$
$$\text{s.t. } \sum_i x_i = B$$

QUBO formulation with penalty terms.

**Traveling Salesman:**

Binary encoding of city visits:

$$x_{i,t} = 1$$ if city $$i$$ visited at time $$t$$

Constraints encoded as penalty Hamiltonians.

**Job Shop Scheduling:**

Assign jobs to machines minimizing makespan.

$$\hat{H} = \hat{H}_{cost} + \lambda\hat{H}_{constraints}$$

---

## Part III: Error Mitigation

### 3.1 Zero-Noise Extrapolation (ZNE)

**Principle:**

If we can measure at multiple noise levels $$\lambda$$, we can extrapolate to $$\lambda = 0$$.

**Noise Scaling:**

Methods to artificially increase noise:
1. **Pulse stretching:** Increase gate duration
2. **Gate folding:** Replace $$G$$ with $$GG^\dagger G$$
3. **Probabilistic:** Add random Pauli gates

**Extrapolation Models:**

Linear: $$E(\lambda) = a + b\lambda$$

Exponential: $$E(\lambda) = a + be^{-c\lambda}$$

Richardson: Polynomial of appropriate degree

**Example:**

Measure at $$\lambda = 1, 2, 3$$:
$$E(1) = 0.45, E(2) = 0.42, E(3) = 0.38$$

Linear fit → $$E(0) = 0.48$$

### 3.2 Probabilistic Error Cancellation (PEC)

**Quasi-Probability Representation:**

Noisy channel $$\mathcal{N}$$ can be written as:

$$\mathcal{N}^{-1} = \sum_i \eta_i \mathcal{B}_i$$

where $$\mathcal{B}_i$$ are implementable operations and $$\eta_i$$ can be negative.

**Protocol:**

1. Sample operation $$\mathcal{B}_i$$ with probability $$|eta_i|/\gamma$$
2. Track sign $$\text{sgn}(\eta_i)$$
3. Compute weighted average

**Overhead:**

Sampling overhead: $$\gamma = \sum_i |\eta_i|$$

For single-qubit depolarizing channel: $$\gamma = \frac{1+3p/4}{1-3p/4}$$

Variance scales as $$\gamma^{2L}$$ for $$L$$ gates.

### 3.3 Symmetry Verification

**Conserved Quantities:**

Many Hamiltonians conserve:
- Particle number $$\hat{N}$$
- Spin $$\hat{S}^2$$, $$\hat{S}_z$$
- Spatial symmetries

**Post-Selection:**

Measure symmetry operator and keep only results in correct sector.

**Symmetry-Expanded Measurement:**

$$\langle\hat{O}\rangle_{sym} = \frac{\text{Tr}[\hat{O}\hat{P}_S\rho]}{\text{Tr}[\hat{P}_S\rho]}$$

where $$\hat{P}_S$$ projects onto correct symmetry sector.

### 3.4 Virtual Distillation

**Concept:**

Use multiple copies of noisy state $$\rho$$ to suppress errors.

**Two-Copy Protocol:**

Measure $$\langle\hat{O}\rangle = \frac{\text{Tr}[\hat{O}\rho^2]}{\text{Tr}[\rho^2]}$$

For $$\rho = (1-p)|\psi\rangle\langle\psi| + p\rho_{err}$$:

$$\frac{\text{Tr}[\hat{O}\rho^2]}{\text{Tr}[\rho^2]} \approx \langle\psi|\hat{O}|\psi\rangle + O(p^2)$$

Error suppression is quadratic.

**Multi-Copy:**

With $$M$$ copies: error $$\propto p^M$$

Trade-off: requires $$M$$ parallel circuit executions.

### 3.5 Combining Techniques

**Hybrid Error Mitigation:**

1. Use ZNE for systematic errors
2. Apply symmetry verification for easy checks
3. Add PEC for residual errors

**Resource Scaling:**

| Technique | Sampling Overhead | Best For |
|-----------|-------------------|----------|
| ZNE | Modest (×3-5) | Coherent errors |
| PEC | High (exponential) | Complete cancellation |
| Symmetry | Low (post-selection) | Symmetry-breaking errors |
| Distillation | Medium (×M) | General noise |

---

## Part IV: Practical Considerations

### 4.1 Algorithm-Hardware Matching

**Superconducting:**
- Fast gates → more circuit depth possible
- Limited connectivity → SWAP overhead
- Best for: shallow VQE, moderate QAOA

**Trapped Ion:**
- All-to-all connectivity → no SWAP needed
- Slower gates → fewer total operations
- Best for: chemistry problems needing entanglement

**Neutral Atom:**
- Large qubit count → bigger problems
- Native multi-qubit gates → reduced depth for some algorithms
- Best for: optimization problems, physics simulations

### 4.2 Current State of the Art

**VQE Demonstrations:**
- H₂, LiH, BeH₂ on superconducting devices
- Larger molecules with error mitigation
- Chemical accuracy remains challenging

**QAOA Results:**
- MaxCut on ~100 qubits
- Utility demonstrated for some instances (IBM, 2023)
- Classical simulability remains a concern

**Error Mitigation:**
- ZNE routine on all major platforms
- PEC demonstrated for small circuits
- Combination methods showing promise

### 4.3 Open Questions

1. **Quantum advantage:** When do NISQ algorithms outperform classical?
2. **Noise resilience:** Optimal trade-off between depth and error mitigation
3. **Optimization landscape:** Avoiding barren plateaus systematically
4. **Resource estimation:** Accurate prediction of algorithm performance

---

## Summary and Exam Preparation

### Key Derivations

1. **VQE:** Variational principle → parameter optimization
2. **QAOA:** MaxCut Hamiltonian → circuit implementation
3. **ZNE:** Noise scaling → extrapolation
4. **PEC:** Quasi-probability decomposition → overhead calculation

### Common Exam Questions

1. Derive the QAOA circuit for MaxCut on a triangle graph
2. Explain why hardware-efficient ansatze suffer from barren plateaus
3. Calculate the overhead of ZNE with linear extrapolation
4. Design an error mitigation strategy for a VQE calculation

### Quick Reference

$$\boxed{E_{VQE} = \langle\psi(\vec{\theta})|\hat{H}|\psi(\vec{\theta})\rangle \geq E_0}$$

$$\boxed{|\gamma,\beta\rangle = \prod_p e^{-i\beta_p H_M}e^{-i\gamma_p H_C}|+\rangle^{\otimes n}}$$

$$\boxed{\hat{H}_{MaxCut} = \sum_{(i,j)\in E}\frac{1-Z_iZ_j}{2}}$$

$$\boxed{\text{ZNE: } E(0) \approx E(\lambda) - \lambda\frac{dE}{d\lambda}}$$

$$\boxed{\text{PEC overhead: } \gamma = \sum_i|\eta_i| \sim e^{O(pL)}}$$
