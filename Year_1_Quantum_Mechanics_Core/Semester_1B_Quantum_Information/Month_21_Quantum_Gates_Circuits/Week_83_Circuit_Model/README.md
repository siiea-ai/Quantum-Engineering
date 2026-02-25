# Week 83: Circuit Model

## Overview
**Days 575-581** | Month 21, Week 3 | Quantum Circuit Formalism

This week covers the circuit model of quantum computation—the standard graphical and mathematical framework for representing quantum algorithms as sequences of gates acting on qubits.

---

## Daily Schedule

| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 575 | Sunday | Circuit Diagrams | Wire conventions, time flows left to right, multi-qubit notation |
| 576 | Monday | Circuit Composition | Sequential gates multiply right-to-left, parallel via tensor |
| 577 | Tuesday | Measurement in Circuits | Projective measurement, mid-circuit, deferred measurement |
| 578 | Wednesday | Classical Control | If-then gates, feedforward, measure-and-apply |
| 579 | Thursday | Circuit Complexity | Depth, width, gate count, T-count |
| 580 | Friday | Circuit Optimization | Gate cancellation, commutation rules, template matching |
| 581 | Saturday | Week Review | Circuit model synthesis and assessment |

---

## Learning Objectives

By the end of this week, you will be able to:

1. **Read** and **write** quantum circuit diagrams following standard conventions
2. **Translate** between circuit diagrams and matrix/tensor expressions
3. **Incorporate** measurements at arbitrary points in a circuit
4. **Implement** classical control flow using feedforward
5. **Analyze** circuits using complexity metrics (depth, width, gate count)
6. **Optimize** circuits using algebraic identities and commutation rules

---

## Key Concepts

### Circuit Notation
```
q₀: ─────[H]─────●─────[M]
                 │
q₁: ─────────────⊕─────[M]
```

### Matrix-Circuit Correspondence
$$U_{circuit} = U_n \cdots U_2 \cdot U_1$$

(Gates applied left-to-right in diagram, multiplied right-to-left in matrix form)

### Parallel Operations
$$U_{parallel} = U_A \otimes U_B$$

### Circuit Complexity Metrics
- **Depth**: Longest path from input to output
- **Width**: Number of qubits
- **Gate count**: Total number of gates
- **T-count**: Number of T gates (important for fault tolerance)

---

## Week Progress

| Day | Status |
|-----|--------|
| Day 575 | ⬜ Not Started |
| Day 576 | ⬜ Not Started |
| Day 577 | ⬜ Not Started |
| Day 578 | ⬜ Not Started |
| Day 579 | ⬜ Not Started |
| Day 580 | ⬜ Not Started |
| Day 581 | ⬜ Not Started |

---

*Week 83 of 84 in Month 21*
