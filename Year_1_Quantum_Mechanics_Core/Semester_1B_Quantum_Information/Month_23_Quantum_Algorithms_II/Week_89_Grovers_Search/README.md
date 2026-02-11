# Week 89: Grover's Search

## Overview
**Days 617-623** | Month 23, Week 1 | Grover's Search Algorithm

This week covers Grover's algorithm for unstructured search, providing a quadratic speedup over classical methods. We develop the oracle, diffusion operator, and geometric interpretation of amplitude amplification.

---

## Daily Schedule

| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 617 | Sunday | Unstructured Search Problem | Query complexity, oracle model |
| 618 | Monday | Grover Oracle | Phase oracle, marking states |
| 619 | Tuesday | Diffusion Operator | Reflection about mean |
| 620 | Wednesday | Amplitude Amplification Geometry | Rotation in 2D subspace |
| 621 | Thursday | Optimal Iteration Count | O(sqrt(N)) analysis |
| 622 | Friday | Multiple Solutions Case | k solutions modification |
| 623 | Saturday | Week Review | Synthesis and assessment |

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Explain** the unstructured search problem and its classical complexity
2. **Construct** the Grover oracle for arbitrary marked states
3. **Derive** the diffusion operator as reflection about the mean
4. **Analyze** Grover's algorithm geometrically as rotation
5. **Calculate** the optimal number of iterations for maximum success probability
6. **Extend** the analysis to multiple solution cases

---

## Key Concepts

### Grover Operator
$$G = D \cdot O_f = (2|\psi\rangle\langle\psi| - I)(I - 2|w\rangle\langle w|)$$

### Success Probability After k Iterations
$$P_{success}(k) = \sin^2\left((2k+1)\theta\right)$$

where $\sin\theta = \sqrt{M/N}$ for M solutions among N items.

### Optimal Iteration Count
$$k_{opt} = \left\lfloor\frac{\pi}{4}\sqrt{\frac{N}{M}}\right\rfloor$$

---

## Week Progress

| Day | Status |
|-----|--------|
| Day 617 | Not Started |
| Day 618 | Not Started |
| Day 619 | Not Started |
| Day 620 | Not Started |
| Day 621 | Not Started |
| Day 622 | Not Started |
| Day 623 | Not Started |

---

*Week 89 of 92 in Semester 1B*
