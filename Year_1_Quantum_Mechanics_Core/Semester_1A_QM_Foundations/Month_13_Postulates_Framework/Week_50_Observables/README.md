# Week 50: Observables and Measurement

## Overview

**Days:** 344-350 (7 days)
**Position:** Year 1, Month 13, Week 2
**Theme:** The Physical Content of Quantum Mechanics

This week introduces the measurement postulate—the bridge between mathematical formalism and physical reality. We learn how Hermitian operators represent observables and how measurement outcomes are predicted.

---

## Learning Objectives

By the end of Week 50, you will be able to:

1. State and apply the measurement postulate
2. Calculate measurement probabilities using the Born rule
3. Determine post-measurement states (collapse)
4. Compute expectation values and variances
5. Identify compatible observables via commutators
6. Transform between position and momentum representations
7. Connect abstract operators to physical measurements

---

## Daily Schedule

| Day | Date | Topic | Key Content |
|-----|------|-------|-------------|
| **344** | Mon | The Measurement Postulate | Eigenvalues as outcomes, P(a) = \|⟨a\|ψ⟩\|² |
| **345** | Tue | State Collapse | Post-measurement state, projection operators |
| **346** | Wed | Expectation Values | ⟨Â⟩ = ⟨ψ\|Â\|ψ⟩, variance, standard deviation |
| **347** | Thu | Compatible Observables | [Â,B̂] = 0, simultaneous eigenstates |
| **348** | Fri | Position and Momentum | x̂, p̂ operators, representations |
| **349** | Sat | Fourier Transform Connection | ψ(x) ↔ φ(p), momentum space |
| **350** | Sun | Week Review & Qiskit Lab | Measurements in quantum computing |

---

## The Measurement Postulates

### Postulate 3: Measurement Outcomes

> *The only possible results of measuring an observable A are the eigenvalues {a} of the corresponding operator Â.*

### Postulate 4: Born Rule (Discrete)

> *If the system is in state |ψ⟩ and we measure A, the probability of obtaining eigenvalue a is:*
> $$P(a) = |⟨a|ψ⟩|^2$$

### Postulate 5: State Collapse

> *Immediately after measurement yields result a, the system is in eigenstate |a⟩.*

For degenerate eigenvalue a with eigenstates {|a,i⟩}:
$$|ψ⟩ \xrightarrow{\text{measure } a} \frac{P̂_a|ψ⟩}{\|P̂_a|ψ⟩\|}$$

where $P̂_a = \sum_i |a,i⟩⟨a,i|$ is the projector onto the eigenspace.

---

## Key Formulas

### Measurement Probability

$$\boxed{P(a) = |⟨a|ψ⟩|^2 = ⟨ψ|P̂_a|ψ⟩}$$

### Expectation Value

$$\boxed{⟨\hat{A}⟩ = ⟨ψ|\hat{A}|ψ⟩ = \sum_a a \cdot P(a) = \sum_a a|⟨a|ψ⟩|^2}$$

### Variance

$$\boxed{(\Delta A)^2 = ⟨\hat{A}^2⟩ - ⟨\hat{A}⟩^2 = ⟨ψ|(\hat{A} - ⟨\hat{A}⟩)^2|ψ⟩}$$

### Standard Deviation

$$\boxed{\Delta A = \sqrt{⟨\hat{A}^2⟩ - ⟨\hat{A}⟩^2}}$$

### Compatible Observables

$$[\hat{A}, \hat{B}] = 0 \iff \text{A and B share a complete set of eigenstates}$$

---

## Position and Momentum

### Position Operator

In position representation:
$$\hat{x}\psi(x) = x\psi(x)$$

Eigenvalue equation:
$$\hat{x}|x'⟩ = x'|x'⟩$$

### Momentum Operator

In position representation:
$$\hat{p}\psi(x) = -i\hbar\frac{d\psi}{dx}$$

Eigenfunctions:
$$\psi_p(x) = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$

### Fourier Transform

$$\phi(p) = ⟨p|ψ⟩ = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty} e^{-ipx/\hbar}\psi(x)dx$$

$$\psi(x) = ⟨x|ψ⟩ = \frac{1}{\sqrt{2\pi\hbar}}\int_{-\infty}^{\infty} e^{ipx/\hbar}\phi(p)dp$$

---

## Quantum Computing Connection

### Measurement in Quantum Computing

| QM Concept | Qiskit Implementation |
|------------|----------------------|
| Computational basis {$\|0⟩, \|1⟩$} | `measure()` operation |
| P(0) = $\|⟨0\|ψ⟩\|²$ | Frequency from shots |
| Collapse to $\|0⟩$ or $\|1⟩$ | Classical bit result |
| Expectation value | Statistical average |

### Measuring in Different Bases

To measure in the X-basis (|+⟩, |-⟩):
1. Apply Hadamard: H|+⟩ = |0⟩, H|-⟩ = |1⟩
2. Measure in Z-basis
3. Interpret: 0 → +, 1 → -

---

## Physical Examples

### Spin Measurement (Stern-Gerlach)

For spin-1/2 particle in state |ψ⟩ = α|↑⟩ + β|↓⟩:

- Measure Sᵤ: outcomes ±ℏ/2 with probabilities |α|², |β|²
- After Sᵤ = +ℏ/2: state collapses to |↑⟩

### Energy Measurement

For particle in state |ψ⟩ = Σₙ cₙ|Eₙ⟩:

- Measure energy: outcome Eₙ with probability |cₙ|²
- After measuring Eₙ: state becomes |Eₙ⟩

---

## Problem Topics

1. Calculate measurement probabilities for given states
2. Determine post-measurement states
3. Compute expectation values and uncertainties
4. Transform between position and momentum
5. Verify commutation relations
6. Design measurement sequences in Qiskit

---

## Computational Lab (Day 350)

- Implement Born rule calculations
- Simulate quantum measurements with Qiskit
- Visualize measurement statistics
- Compare single-shot vs. expectation value

---

## Preview: Week 51

Next week explores the consequences of non-commuting observables: the uncertainty principle. The commutator [x̂, p̂] = iℏ leads to Heisenberg's famous relation Δx·Δp ≥ ℏ/2.

---

**Next:** [Day_344_Monday.md](Day_344_Monday.md) — The Measurement Postulate
