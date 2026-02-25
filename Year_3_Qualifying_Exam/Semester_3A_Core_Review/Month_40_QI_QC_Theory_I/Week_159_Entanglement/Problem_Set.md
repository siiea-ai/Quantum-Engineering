# Week 159: Entanglement - Problem Set

## Instructions

This problem set contains 28 qualifying exam-style problems on entanglement theory. Work through these problems systematically.

**Difficulty Levels:**
- (E) Easy: Direct application
- (M) Medium: Multi-step reasoning
- (H) Hard: Research-level insight

---

## Section A: Separability and Bell States (Problems 1-8)

### Problem 1 (E)
Determine whether each state is separable or entangled:

(a) $$|00\rangle$$
(b) $$\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
(c) $$\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$
(d) $$\frac{1}{\sqrt{2}}(|00\rangle + |01\rangle)$$

---

### Problem 2 (E)
Write out the four Bell states and verify that they form an orthonormal basis.

---

### Problem 3 (M)
Show that all four Bell states have the same reduced density matrices:
$$\rho_A = \rho_B = \frac{I}{2}$$

---

### Problem 4 (M)
The Bell states can be generated from $$|00\rangle$$ by:
1. Apply Hadamard to first qubit
2. Apply CNOT

Write the circuit and verify that it produces $$|\Phi^+\rangle$$.

---

### Problem 5 (M)
Show that the Bell states are related by local Pauli operations:
$$|\Phi^-\rangle = (Z \otimes I)|\Phi^+\rangle$$
$$|\Psi^+\rangle = (X \otimes I)|\Phi^+\rangle$$
$$|\Psi^-\rangle = (iY \otimes I)|\Phi^+\rangle$$

---

### Problem 6 (H)
For the state $$|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$$:

(a) For what values of $$\theta$$ is the state separable?
(b) Compute the entanglement entropy as a function of $$\theta$$.
(c) For what $$\theta$$ is entanglement maximized?

---

### Problem 7 (H)
Consider the mixed state:
$$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$$

(Werner state). For what range of $$p$$ is this state:
(a) Separable?
(b) Entangled?

---

### Problem 8 (H)
Prove that a pure bipartite state is entangled if and only if its reduced density matrix is mixed.

---

## Section B: CHSH Inequality (Problems 9-14)

### Problem 9 (M)
**Derive the CHSH inequality** from local hidden variable assumptions.

State clearly:
(a) The assumptions
(b) The definition of the correlation function $$E(a,b)$$
(c) The bound $$|S| \leq 2$$

---

### Problem 10 (M)
For the singlet state $$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$, compute:
$$E(a,b) = \langle\Psi^-|(\vec{a}\cdot\vec{\sigma}) \otimes (\vec{b}\cdot\vec{\sigma})|\Psi^-\rangle$$

where $$\vec{a}$$ and $$\vec{b}$$ are unit vectors.

---

### Problem 11 (M)
Using the result from Problem 10, compute the CHSH quantity $$S$$ for measurement angles:
- $$a = 0°$$, $$a' = 90°$$ (Alice)
- $$b = 45°$$, $$b' = 135°$$ (Bob)

Verify that $$|S| = 2\sqrt{2}$$.

---

### Problem 12 (H)
Prove the Tsirelson bound: for any quantum state and observables with outcomes $$\pm 1$$:
$$|S| \leq 2\sqrt{2}$$

---

### Problem 13 (M)
For the Bell state $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:

(a) Compute $$E(a,b) = \langle\Phi^+|(\vec{a}\cdot\vec{\sigma}) \otimes (\vec{b}\cdot\vec{\sigma})|\Phi^+\rangle$$
(b) Find optimal measurement settings to maximize $$|S|$$
(c) Calculate the maximum $$|S|$$

---

### Problem 14 (H)
Consider a mixed state $$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)|00\rangle\langle 00|$$.

(a) For what values of $$p$$ does this state violate the CHSH inequality?
(b) Is the state always entangled when it violates CHSH?

---

## Section C: PPT Criterion (Problems 15-18)

### Problem 15 (M)
Compute the partial transpose $$\rho^{T_B}$$ for:
$$\rho = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

Find its eigenvalues and determine if the state is entangled.

---

### Problem 16 (M)
Apply the PPT criterion to the Werner state:
$$\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)\frac{I}{4}$$

For what range of $$p$$ is the state PPT?

---

### Problem 17 (H)
For the two-qubit state:
$$\rho = \frac{1}{3}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

(a) Check if the state is normalized.
(b) Compute the partial transpose.
(c) Determine if the state is entangled.

---

### Problem 18 (H)
Explain why PPT is sufficient for separability in $$2 \times 2$$ and $$2 \times 3$$ systems but not in higher dimensions.

---

## Section D: Entanglement Measures (Problems 19-26)

### Problem 19 (E)
Compute the entanglement entropy for:
(a) $$|00\rangle$$
(b) $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
(c) $$|\psi\rangle = \sqrt{0.9}|00\rangle + \sqrt{0.1}|11\rangle$$

---

### Problem 20 (M)
For the state $$|\psi\rangle = \alpha|00\rangle + \beta|01\rangle + \gamma|10\rangle + \delta|11\rangle$$, derive the formula for entanglement entropy in terms of $$\alpha, \beta, \gamma, \delta$$.

---

### Problem 21 (M)
Compute the concurrence for the pure states:
(a) $$|00\rangle$$
(b) $$|\Phi^+\rangle$$
(c) $$|\psi\rangle = \frac{\sqrt{3}}{2}|00\rangle + \frac{1}{2}|11\rangle$$

---

### Problem 22 (H)
Apply Wootters formula to compute the concurrence of the Werner state:
$$\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)\frac{I}{4}$$

For what $$p$$ is $$C > 0$$?

---

### Problem 23 (M)
Compute the negativity and logarithmic negativity for:
(a) $$|\Phi^+\rangle\langle\Phi^+|$$
(b) The Werner state at $$p = 0.5$$

---

### Problem 24 (H)
Prove that for pure two-qubit states:
$$C(|\psi\rangle) = 2|ad - bc|$$

where $$|\psi\rangle = a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle$$.

---

### Problem 25 (H)
Show that the entanglement of formation for a two-qubit state is:
$$E_F(\rho) = h\left(\frac{1 + \sqrt{1-C^2}}{2}\right)$$

where $$h$$ is the binary entropy and $$C$$ is concurrence.

---

### Problem 26 (H)
Compare concurrence and negativity for the state:
$$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)|00\rangle\langle 00|$$

Plot both as functions of $$p$$ and discuss.

---

## Section E: Advanced Topics (Problems 27-28)

### Problem 27 (H)
**Monogamy of Entanglement**

For a three-qubit pure state, the Coffman-Kundu-Wootters inequality states:
$$C_{A|B}^2 + C_{A|C}^2 \leq C_{A|BC}^2$$

(a) Verify this for the GHZ state: $$|\text{GHZ}\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$
(b) Verify this for the W state: $$|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$

---

### Problem 28 (H)
**Entanglement Witnesses**

(a) Show that for any entangled state $$\rho$$, there exists an operator $$W$$ such that:
- $$\text{Tr}(W\sigma) \geq 0$$ for all separable $$\sigma$$
- $$\text{Tr}(W\rho) < 0$$

(b) Construct a witness for the Bell state $$|\Phi^+\rangle$$.

---

## Bonus Problems

### Bonus 1 (Research Level)
**Bound Entanglement**

The Horodecki state in $$3 \times 3$$ is:
$$\rho_a = \frac{1}{8a+1}\begin{pmatrix} a & 0 & 0 & 0 & a & 0 & 0 & 0 & a \\ 0 & a & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & a & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & a & 0 & 0 & 0 & 0 & 0 \\ a & 0 & 0 & 0 & a & 0 & 0 & 0 & a \\ 0 & 0 & 0 & 0 & 0 & a & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & \frac{1+a}{2} & 0 & \frac{\sqrt{1-a^2}}{2} \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & a & 0 \\ a & 0 & 0 & 0 & a & 0 & \frac{\sqrt{1-a^2}}{2} & 0 & \frac{1+a}{2} \end{pmatrix}$$

for $$0 < a < 1$$.

Show that for certain values of $$a$$:
(a) The state is PPT
(b) The state is entangled (not separable)
(c) The state is not distillable (bound entangled)

---

### Bonus 2 (Research Level)
**Squashed Entanglement**

The squashed entanglement is defined as:
$$E_{sq}(\rho_{AB}) = \inf_{\rho_{ABE}} \frac{1}{2}I(A;B|E)$$

where the infimum is over all extensions $$\rho_{ABE}$$ of $$\rho_{AB}$$.

(a) Show that $$E_{sq}$$ is an entanglement measure.
(b) Prove that $$E_{sq}(\rho) = 0$$ for separable states.
(c) Compute $$E_{sq}$$ for a Bell state.

---

## Problem Set Summary

| Section | Problems | Topics |
|---------|----------|--------|
| A | 1-8 | Separability, Bell states |
| B | 9-14 | CHSH inequality |
| C | 15-18 | PPT criterion |
| D | 19-26 | Entanglement measures |
| E | 27-28 | Advanced topics |

**Total: 28 problems + 2 bonus problems**

---

*Complete this problem set before consulting solutions.*
