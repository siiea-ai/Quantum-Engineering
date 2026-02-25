# Week 179: Self-Assessment - NISQ Algorithms

## Instructions

Complete this assessment at the end of Week 179. Be honest about your understanding to identify areas for improvement.

---

## Part A: Conceptual Checklist

Rate each topic: **1** (Cannot explain) to **5** (Can teach others)

### VQE

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Variational principle | | | | | | |
| Hybrid quantum-classical loop | | | | | | |
| Hardware-efficient ansatze | | | | | | |
| UCCSD ansatz | | | | | | |
| Parameter-shift gradients | | | | | | |
| Barren plateaus | | | | | | |
| Hamiltonian measurement | | | | | | |
| ADAPT-VQE | | | | | | |

### QAOA

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| QAOA ansatz structure | | | | | | |
| Cost Hamiltonian construction | | | | | | |
| Mixer Hamiltonian role | | | | | | |
| MaxCut encoding | | | | | | |
| Approximation ratios | | | | | | |
| Parameter optimization | | | | | | |
| QAOA variants | | | | | | |

### Error Mitigation

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Zero-noise extrapolation | | | | | | |
| Gate folding | | | | | | |
| Probabilistic error cancellation | | | | | | |
| Symmetry verification | | | | | | |
| Virtual distillation | | | | | | |
| Mitigation overhead | | | | | | |

---

## Part B: Key Equations

Write from memory:

### 1. Variational Principle
$$E(\vec{\theta}) = $$

*Check:* $$\langle\psi(\vec{\theta})|H|\psi(\vec{\theta})\rangle \geq E_0$$

### 2. QAOA State
$$|\gamma, \beta\rangle = $$

*Check:* $$\prod_p e^{-i\beta_p H_M}e^{-i\gamma_p H_C}|+\rangle^{\otimes n}$$

### 3. MaxCut Hamiltonian
$$H_C = $$

*Check:* $$\sum_{(i,j)\in E}\frac{1-Z_iZ_j}{2}$$

### 4. Parameter-Shift Rule
$$\frac{\partial E}{\partial\theta} = $$

*Check:* $$\frac{E(\theta+\pi/2) - E(\theta-\pi/2)}{2}$$

### 5. ZNE Linear Extrapolation
$$E(0) \approx $$

*Check:* $$E(\lambda) - \lambda \cdot \frac{E(2\lambda)-E(\lambda)}{\lambda}$$

---

## Part C: Quick Problems

### 1. VQE Gradient (5 min)

For a 3-parameter ansatz, how many circuit evaluations are needed to compute the full gradient using the parameter-shift rule?

Answer: _______________

*Solution: 2 × 3 = 6*

### 2. QAOA Circuit (5 min)

For QAOA at $$p=2$$ on a 5-node MaxCut problem, how many parameterized layers are in the circuit?

Answer: _______________

*Solution: 4 (2 cost layers + 2 mixer layers)*

### 3. ZNE Extrapolation (5 min)

Given $$E(1) = 0.8$$, $$E(2) = 0.6$$, find $$E(0)$$ using linear extrapolation.

Answer: _______________

*Solution: $$E(0) = 0.8 + 0.2 = 1.0$$*

---

## Part D: Short Answer

### 1. Why is VQE considered "NISQ-friendly"?

_______________________________________________

_______________________________________________

### 2. What causes barren plateaus?

_______________________________________________

_______________________________________________

### 3. Why does QAOA alternate between cost and mixer Hamiltonians?

_______________________________________________

_______________________________________________

### 4. What is the main limitation of probabilistic error cancellation?

_______________________________________________

_______________________________________________

---

## Part E: Algorithm Design

### Scenario: You need to find the ground state of a 6-qubit molecular Hamiltonian.

1. What ansatz would you choose?

_______________________________________________

2. Estimate the circuit depth.

_______________________________________________

3. How many measurements per energy evaluation?

_______________________________________________

4. What error mitigation would you apply?

_______________________________________________

---

## Part F: Oral Exam Practice

Record yourself (3-5 min each):

### Question 1
"Explain the VQE algorithm and its components."

Self-evaluation:
- [ ] Explained variational principle
- [ ] Described hybrid loop
- [ ] Mentioned ansatz design
- [ ] Discussed classical optimization

### Question 2
"How does QAOA solve combinatorial optimization problems?"

Self-evaluation:
- [ ] Explained cost Hamiltonian
- [ ] Described mixer role
- [ ] Discussed parameter optimization
- [ ] Mentioned approximation ratios

### Question 3
"Compare different error mitigation techniques."

Self-evaluation:
- [ ] Covered ZNE
- [ ] Mentioned PEC
- [ ] Discussed symmetry verification
- [ ] Explained overhead trade-offs

---

## Part G: Problem Set Scoring

| Section | Attempted | Correct | Percentage |
|---------|-----------|---------|------------|
| VQE | /8 | | |
| QAOA | /8 | | |
| Error Mitigation | /8 | | |
| Integration | /4 | | |
| **Total** | /28 | | |

---

## Part H: Gap Identification

Top 3 areas needing improvement:

1. _______________________________________________

2. _______________________________________________

3. _______________________________________________

Action plan:

_______________________________________________

_______________________________________________

---

## Part I: Readiness Check

- [ ] Can explain VQE without notes
- [ ] Can derive QAOA for simple MaxCut
- [ ] Understand barren plateau problem
- [ ] Can calculate ZNE extrapolation
- [ ] Know when to use which error mitigation
- [ ] Can match algorithms to hardware

Confidence level for exam: ___ / 10

---

## Reflection

What surprised you about NISQ algorithms?

_______________________________________________

What is the biggest challenge for practical quantum advantage?

_______________________________________________

How has this week changed your view of near-term quantum computing?

_______________________________________________
