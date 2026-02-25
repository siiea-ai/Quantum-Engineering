# Week 74: Composite Systems

## Overview
**Days 512-518** | Month 19, Week 74 | The Mathematics of Multi-Party Quantum Systems

This week develops the essential mathematical machinery for describing quantum systems composed of multiple subsystems. These tools are fundamental to quantum information theory, enabling us to understand entanglement, describe local measurements, and analyze quantum correlations.

## Daily Schedule

| Day | Topic | Key Formula |
|-----|-------|-------------|
| 512 | Tensor Products Revisited | H_AB = H_A ⊗ H_B |
| 513 | Partial Trace | ρ_A = Tr_B(ρ_AB) |
| 514 | Reduced Density Matrices | ⟨A⟩ = Tr(ρ_A A) |
| 515 | Schmidt Decomposition | \|ψ⟩ = Σᵢ √λᵢ \|aᵢ⟩\|bᵢ⟩ |
| 516 | Entanglement Detection | Separability criteria |
| 517 | Purification | ρ_A from \|ψ⟩_AB |
| 518 | Week Review | Integration and problem solving |

## Key Concepts

### Tensor Product Structure
$$\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B, \quad \dim(\mathcal{H}_{AB}) = d_A \times d_B$$

### Partial Trace
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j \langle j|_B \rho_{AB} |j\rangle_B$$

### Schmidt Decomposition
$$|\psi\rangle_{AB} = \sum_{i=1}^{r} \sqrt{\lambda_i} |a_i\rangle_A |b_i\rangle_B$$

### Separable vs Entangled States
$$\rho_{\text{sep}} = \sum_i p_i \rho_i^A \otimes \rho_i^B \quad \text{vs} \quad \rho_{\text{ent}} \neq \sum_i p_i \rho_i^A \otimes \rho_i^B$$

## Learning Objectives

By the end of this week, you will be able to:
1. Construct the tensor product Hilbert space for composite systems
2. Compute the partial trace to obtain reduced density matrices
3. Calculate expectation values for local observables
4. Perform Schmidt decomposition and interpret Schmidt coefficients
5. Distinguish separable from entangled states using various criteria
6. Purify any mixed state using a reference system

## Progress

- [ ] Day 512: Tensor Products Revisited
- [ ] Day 513: Partial Trace
- [ ] Day 514: Reduced Density Matrices
- [ ] Day 515: Schmidt Decomposition
- [ ] Day 516: Entanglement Detection
- [ ] Day 517: Purification
- [ ] Day 518: Week Review

---

## Primary References

| Resource | Sections | Focus |
|----------|----------|-------|
| Nielsen & Chuang | Ch. 2.4-2.5 | Tensor products, Schmidt decomposition |
| Preskill Ph219 | Ch. 2 | Density matrices and entanglement |
| Wilde | Ch. 3-4 | Formal treatment |

---

**Previous:** [Week_73_Pure_vs_Mixed](../Week_73_Pure_vs_Mixed/README.md)

**Next:** [Week_75_Generalized_Measurements](../Week_75_Generalized_Measurements/README.md)
