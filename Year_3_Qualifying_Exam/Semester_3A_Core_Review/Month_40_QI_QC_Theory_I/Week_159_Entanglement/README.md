# Week 159: Entanglement

## Overview

**Days:** 1107-1113
**Theme:** Separability, Bell Inequalities, Entanglement Measures

This week provides a comprehensive review of entanglement theory - one of the most important topics in quantum information science. We cover separability criteria, Bell state analysis, the CHSH inequality and its quantum violation, and quantitative measures of entanglement including concurrence, negativity, and entanglement entropy.

## Daily Breakdown

### Day 1107 (Monday): Separability and Entanglement
- Definition of separable and entangled states
- Pure state separability via Schmidt rank
- Mixed state separability: convex combinations of product states
- Witness-based detection

### Day 1108 (Tuesday): Bell States and Maximally Entangled States
- The four Bell states
- Properties of maximally entangled states
- Bell basis measurements
- Generalizations to higher dimensions

### Day 1109 (Wednesday): Bell Inequalities
- EPR argument and local hidden variables
- CHSH inequality derivation
- Classical bound: $$|S| \leq 2$$
- Quantum violation: Tsirelson bound $$|S| \leq 2\sqrt{2}$$

### Day 1110 (Thursday): PPT and Separability Criteria
- Partial transpose operation
- PPT (Positive Partial Transpose) criterion
- Sufficiency for $$2 \times 2$$ and $$2 \times 3$$ systems
- Entanglement witnesses

### Day 1111 (Friday): Entanglement Measures - Entropy
- Von Neumann entropy of entanglement
- Entanglement entropy for pure states
- Properties: monotonicity under LOCC
- Additivity and subadditivity

### Day 1112 (Saturday): Entanglement Measures - Concurrence and Negativity
- Concurrence for two-qubit systems
- Wootters formula
- Negativity and logarithmic negativity
- Comparison of measures

### Day 1113 (Sunday): Review and Problem Session
- Comprehensive problem solving
- Oral exam practice
- Self-assessment

## Key Formulas

### Bell States

$$\boxed{|\Phi^\pm\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle)}$$

$$\boxed{|\Psi^\pm\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)}$$

### CHSH Inequality

Classical bound: $$\boxed{|S| \leq 2}$$

where $$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$

Quantum maximum (Tsirelson bound): $$\boxed{|S|_{\max} = 2\sqrt{2}}$$

### Entanglement Entropy

For pure state with Schmidt decomposition $$|\psi\rangle = \sum_i \lambda_i |a_i\rangle|b_i\rangle$$:

$$\boxed{E(|\psi\rangle) = S(\rho_A) = -\sum_i \lambda_i^2 \log_2 \lambda_i^2}$$

### Concurrence (Two Qubits)

$$\boxed{C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)}$$

where $$\lambda_i$$ are decreasing square roots of eigenvalues of $$\rho\tilde{\rho}$$ with:

$$\boxed{\tilde{\rho} = (\sigma_y \otimes \sigma_y)\rho^*(\sigma_y \otimes \sigma_y)}$$

For pure states: $$C(|\psi\rangle) = |\langle\psi|\tilde{\psi}\rangle|$$

### Negativity

$$\boxed{\mathcal{N}(\rho) = \frac{\|\rho^{T_B}\|_1 - 1}{2} = \sum_i \frac{|\mu_i| - \mu_i}{2}}$$

where $$\mu_i$$ are eigenvalues of $$\rho^{T_B}$$.

Logarithmic negativity: $$\boxed{E_N = \log_2 \|\rho^{T_B}\|_1}$$

### PPT Criterion

$$\boxed{\rho^{T_B} \geq 0 \Rightarrow \rho \text{ may be separable}}$$
$$\boxed{\rho^{T_B} \not\geq 0 \Rightarrow \rho \text{ is entangled}}$$

## Learning Objectives

By the end of this week, you should be able to:

1. Determine if a given state is separable or entangled
2. Apply the PPT criterion to detect entanglement
3. Derive the CHSH inequality and calculate quantum violation
4. Compute entanglement entropy for pure bipartite states
5. Calculate concurrence for two-qubit states using Wootters formula
6. Compute negativity from partial transpose eigenvalues
7. Explain entanglement monotonicity and LOCC operations

## Files in This Week

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `Review_Guide.md` | Comprehensive theoretical review |
| `Problem_Set.md` | 28 qualifying exam-style problems |
| `Problem_Solutions.md` | Detailed worked solutions |
| `Oral_Practice.md` | Oral exam questions and answers |
| `Self_Assessment.md` | Diagnostic checklist and self-test |

## Prerequisites

Before starting this week, ensure mastery of:
- Density matrices (Week 157)
- Composite systems, partial trace, Schmidt decomposition (Week 158)

## References

1. Nielsen & Chuang, Chapter 12: Quantum entanglement
2. Preskill Notes, Chapter 4: Quantum entanglement
3. [CHSH Inequality - qubit.guide](https://qubit.guide/6.3-chsh-inequality)
4. [Entanglement Measures](https://www.gtoth.eu/Transparencies/QINF_Entanglement_Measures_2025.pdf)
5. Horodecki et al., "Quantum Entanglement" Rev. Mod. Phys. 81, 865 (2009)
