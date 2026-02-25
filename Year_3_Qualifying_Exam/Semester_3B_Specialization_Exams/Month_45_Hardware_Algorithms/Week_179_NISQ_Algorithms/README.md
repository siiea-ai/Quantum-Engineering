# Week 179: NISQ Algorithms

## Overview

**Days:** 1247-1253
**Theme:** Variational algorithms and error mitigation for near-term quantum devices

This week covers the algorithmic approaches designed for Noisy Intermediate-Scale Quantum (NISQ) devices. We focus on the Variational Quantum Eigensolver (VQE) for chemistry, the Quantum Approximate Optimization Algorithm (QAOA) for combinatorial problems, and error mitigation techniques that enable useful computation despite hardware noise.

## Daily Schedule

| Day | Date (Day #) | Topic | Focus |
|-----|--------------|-------|-------|
| Monday | 1247 | VQE Fundamentals | Variational principle, ansatz design, optimization |
| Tuesday | 1248 | VQE for Chemistry | Molecular Hamiltonians, active spaces, orbital optimization |
| Wednesday | 1249 | QAOA Fundamentals | MaxCut, mixer Hamiltonians, approximation ratios |
| Thursday | 1250 | QAOA Applications | Portfolio optimization, scheduling, QUBO formulation |
| Friday | 1251 | Error Mitigation I | Zero-noise extrapolation, probabilistic error cancellation |
| Saturday | 1252 | Error Mitigation II | Symmetry verification, virtual distillation |
| Sunday | 1253 | Review & Integration | Algorithm-hardware matching, practical considerations |

## Learning Objectives

By the end of this week, you will be able to:

1. **Derive** the VQE algorithm from the variational principle
2. **Design** appropriate ansatze for molecular and optimization problems
3. **Implement** QAOA for MaxCut and portfolio optimization
4. **Analyze** barren plateau phenomena and mitigation strategies
5. **Apply** zero-noise extrapolation and other error mitigation techniques
6. **Evaluate** algorithm-hardware trade-offs for practical applications

## Key Concepts

### Variational Quantum Eigensolver (VQE)

**The Variational Principle:**

For any trial state $$|\psi(\vec{\theta})\rangle$$:

$$E(\vec{\theta}) = \langle\psi(\vec{\theta})|\hat{H}|\psi(\vec{\theta})\rangle \geq E_0$$

where $$E_0$$ is the true ground state energy.

**VQE Algorithm:**
1. Prepare parameterized state $$|\psi(\vec{\theta})\rangle$$ on quantum computer
2. Measure expectation value $$\langle\hat{H}\rangle$$ via Pauli decomposition
3. Classical optimizer updates $$\vec{\theta}$$ to minimize energy
4. Iterate until convergence

**Ansatz Types:**

| Ansatz | Description | Use Case |
|--------|-------------|----------|
| Hardware-Efficient | Native gates, low depth | General purpose |
| UCCSD | Chemistry-inspired, particle-conserving | Molecular systems |
| ADAPT-VQE | Iteratively built from operator pool | High accuracy |
| Hamiltonian Variational | Problem-specific structure | Physics simulations |

### Quantum Approximate Optimization Algorithm (QAOA)

**Problem Formulation:**

Encode optimization problem in cost Hamiltonian $$\hat{H}_C$$:

$$\hat{H}_C = \sum_{\langle i,j\rangle} C_{ij}\hat{Z}_i\hat{Z}_j + \sum_i h_i\hat{Z}_i$$

**QAOA Ansatz:**

$$|\gamma, \beta\rangle = \prod_{p=1}^{P} e^{-i\beta_p \hat{H}_M} e^{-i\gamma_p \hat{H}_C} |+\rangle^{\otimes n}$$

where:
- $$\hat{H}_M = \sum_i \hat{X}_i$$ (mixer Hamiltonian)
- $$\gamma_p, \beta_p$$ are variational parameters
- $$P$$ is the number of QAOA layers

**Performance:**

- $$P \rightarrow \infty$$: Approaches adiabatic limit
- Finite $$P$$: Approximation ratio depends on problem structure
- MaxCut: $$P=1$$ achieves ratio $$\geq 0.6924$$ for 3-regular graphs

### Error Mitigation Techniques

**Zero-Noise Extrapolation (ZNE):**

1. Run circuit at noise level $$\lambda$$
2. Artificially increase noise: $$\lambda, 2\lambda, 3\lambda, ...$$
3. Fit results and extrapolate to $$\lambda = 0$$

**Probabilistic Error Cancellation (PEC):**

1. Decompose noisy channel as linear combination of ideal operations
2. Run circuits with modified gates
3. Combine results with appropriate signs/weights

**Symmetry Verification:**

1. Check if output satisfies known symmetries
2. Post-select on correct symmetry sector
3. Reduces errors that break symmetries

**Virtual Distillation:**

1. Prepare multiple copies of noisy state
2. Collective measurement suppresses errors
3. Exponential improvement with copies

## Key Equations

$$\boxed{E_{VQE}(\vec{\theta}) = \langle\psi(\vec{\theta})|\hat{H}|\psi(\vec{\theta})\rangle}$$

$$\boxed{|\gamma,\beta\rangle_{QAOA} = \prod_{p=1}^P e^{-i\beta_p H_M}e^{-i\gamma_p H_C}|+\rangle^{\otimes n}}$$

$$\boxed{\hat{H}_{MaxCut} = \sum_{\langle i,j\rangle}\frac{1-\hat{Z}_i\hat{Z}_j}{2}}$$

$$\boxed{E_{ZNE}(0) = \lim_{\lambda\to 0} E(\lambda) \approx a_0 + a_1\lambda + a_2\lambda^2 + ...}$$

## Hardware Considerations

### Circuit Depth vs Noise

| Algorithm | Typical Depth | Noise Tolerance |
|-----------|---------------|-----------------|
| VQE (shallow) | 10-50 | Medium |
| VQE (UCCSD) | 100-1000 | Low |
| QAOA (p=1) | 2n | Medium |
| QAOA (p=5) | 10n | Low |

### Measurement Overhead

**Pauli Decomposition:**

Molecular Hamiltonians decompose into many Pauli strings:

$$\hat{H} = \sum_{i} c_i \hat{P}_i$$

Measurement scaling:
- Naive: $$O(N^4)$$ terms for $$N$$ orbitals
- Grouped: $$O(N^3)$$ with qubit-wise commuting groups
- Shadow tomography: $$O(\log N)$$ for specific observables

### Classical Optimization

| Optimizer | Type | Pros | Cons |
|-----------|------|------|------|
| COBYLA | Gradient-free | Noise tolerant | Slow convergence |
| SPSA | Stochastic gradient | Low overhead | Noisy gradients |
| L-BFGS | Gradient-based | Fast convergence | Needs exact gradients |
| Adam | Adaptive | Good for deep circuits | Hyperparameter tuning |

## Study Materials

### Required Reading
1. Cerezo, M. et al. "Variational quantum algorithms" Nature Reviews Physics (2021)
2. Farhi, E. et al. "A Quantum Approximate Optimization Algorithm" arXiv:1411.4028
3. Temme, K. et al. "Error mitigation for short-depth quantum circuits" PRL (2017)

### Recent Papers
- McClean, J.R. et al. "Barren plateaus in quantum neural network training landscapes"
- Kim, Y. et al. "Evidence for utility of quantum computing before fault tolerance" (2023)

### Problem Set Focus
- VQE implementation for H₂ and LiH
- QAOA for MaxCut instances
- Error mitigation calculations

## Deliverables

1. **Review Guide** - Comprehensive theory summary (2000+ words)
2. **Problem Set** - 25-30 problems with solutions
3. **Oral Practice** - Common qualifying exam questions
4. **Self-Assessment** - Conceptual understanding checks

## Assessment Criteria

| Skill | Novice | Proficient | Expert |
|-------|--------|------------|--------|
| VQE Design | Uses basic ansatz | Optimizes for problem | Develops custom ansatze |
| QAOA Analysis | Implements algorithm | Analyzes performance | Predicts approximation ratio |
| Error Mitigation | Applies techniques | Chooses appropriate method | Combines methods optimally |
