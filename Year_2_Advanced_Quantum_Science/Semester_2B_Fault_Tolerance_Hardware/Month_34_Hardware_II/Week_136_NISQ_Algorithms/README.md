# Week 136: NISQ Algorithm Design

## Overview

Week 136 focuses on algorithms designed specifically for Noisy Intermediate-Scale Quantum (NISQ) devices. We explore the characteristics and limitations of current quantum hardware, then develop variational algorithms that can operate within these constraints. The week culminates in understanding the challenges of trainability, noise-aware compilation, and the path toward quantum advantage.

**Week Number:** 136 of 312
**Month:** 34 (Hardware II)
**Semester:** 2B (Fault Tolerance & Hardware)
**Year:** 2 (Advanced Quantum Science)

## Learning Objectives

By the end of this week, you will be able to:

1. Characterize NISQ devices in terms of qubit count, coherence times, gate fidelities, and connectivity
2. Implement Variational Quantum Eigensolver (VQE) for molecular ground state estimation
3. Design and analyze Quantum Approximate Optimization Algorithm (QAOA) circuits
4. Understand and mitigate barren plateau phenomena in parameterized quantum circuits
5. Apply noise-aware compilation techniques for improved circuit execution
6. Integrate classical optimizers with quantum circuits in hybrid workflows

## Daily Schedule

| Day | Topic | Focus Areas |
|-----|-------|-------------|
| 946 | NISQ Era Characteristics | Qubit counts, coherence limits, gate fidelities, connectivity graphs |
| 947 | Variational Quantum Eigensolver | Ansatz design, parameter optimization, VQE algorithm structure |
| 948 | QAOA for Optimization | Mixer and cost Hamiltonians, depth scaling, approximation ratios |
| 949 | Barren Plateaus | Gradient vanishing, expressibility vs trainability trade-off |
| 950 | Noise-Aware Compilation | Error-adaptive mapping, noise-aware routing, pulse optimization |
| 951 | Hybrid Classical-Quantum Workflows | Optimizer selection, shot budgets, error-aware optimization |
| 952 | Month 34 Synthesis | NISQ landscape review, quantum advantage prospects, FT transition |

## Key Equations

### VQE Energy Expectation
$$E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle$$

### QAOA State Preparation
$$|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = \prod_{p=1}^{P} e^{-i\beta_p \hat{B}} e^{-i\gamma_p \hat{C}} |+\rangle^{\otimes n}$$

### Parameter Shift Rule
$$\frac{\partial}{\partial\theta}\langle\hat{O}\rangle = \frac{1}{2}\left(\langle\hat{O}\rangle_{\theta+\frac{\pi}{2}} - \langle\hat{O}\rangle_{\theta-\frac{\pi}{2}}\right)$$

### Barren Plateau Scaling
$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial\theta}\right] \leq \exp(-cn)$$

### QAOA Approximation Ratio
$$r = \frac{\langle\boldsymbol{\gamma}, \boldsymbol{\beta}|\hat{C}|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle}{C_{\max}}$$

## Prerequisites

- Quantum circuit model and gate operations (Year 1)
- Quantum Hamiltonian formalism (Semester 1B)
- Classical optimization methods (Month 33)
- Basic noise models and error characterization (Week 135)

## Required Software

```python
# Core packages
pip install qiskit qiskit-aer qiskit-nature
pip install pennylane pennylane-qiskit
pip install numpy scipy matplotlib
pip install pyscf  # For molecular simulations
```

## Hardware Considerations

This week's labs are designed to run on:
- **Simulators:** Qiskit Aer, PennyLane default.qubit
- **Cloud QPUs:** IBM Quantum (free tier sufficient)
- **Recommended:** GPU acceleration for variational optimization

## References

1. Preskill, J. "Quantum Computing in the NISQ era and beyond." Quantum 2, 79 (2018)
2. Peruzzo, A. et al. "A variational eigenvalue solver on a quantum processor." Nature Communications 5, 4213 (2014)
3. Farhi, E. et al. "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028
4. McClean, J. R. et al. "Barren plateaus in quantum neural network training landscapes." Nature Communications 9, 4812 (2018)
5. Kandala, A. et al. "Hardware-efficient variational quantum eigensolver." Nature 549, 242-246 (2017)

## Assessment

- Daily computational labs with working code
- Problem sets covering VQE, QAOA, and gradient analysis
- End-of-week synthesis project: Noise-aware VQE implementation for H₂

---

**Next Week:** Week 137 - Error Mitigation Techniques
