# Day 577: Measurement in Circuits

## Overview
**Day 577** | Week 83, Day 3 | Year 1, Month 21 | Quantum Measurement Operations

Today we study how measurements are incorporated into quantum circuits, including projective measurements, mid-circuit measurements, and the powerful principle of deferred measurement.

---

## Learning Objectives

1. Represent projective measurements in circuit notation
2. Calculate measurement outcome probabilities from circuit analysis
3. Apply mid-circuit measurement and state collapse rules
4. Understand and apply the deferred measurement principle
5. Distinguish measurement in different bases
6. Implement measurement-based conditional operations

---

## Core Content

### Projective Measurement in Circuits

Standard measurement symbol at end of circuit:

```
         ┌───┐
q₀: ─────┤ U ├────[M]════c₀
         └───┘
```

The double line `═══` represents a **classical wire** carrying the measurement result.

### Computational Basis Measurement

For a single qubit in state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:

$$\boxed{P(0) = |\alpha|^2, \quad P(1) = |\beta|^2}$$

**Post-measurement states:**
- If outcome 0: $|\psi'\rangle = |0\rangle$
- If outcome 1: $|\psi'\rangle = |1\rangle$

### Measurement Operators

Projective measurement is described by projection operators:

$$\Pi_0 = |0\rangle\langle 0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

$$\Pi_1 = |1\rangle\langle 1| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

**Properties:**
- $\Pi_0 + \Pi_1 = I$ (completeness)
- $\Pi_i^2 = \Pi_i$ (idempotent)
- $\Pi_0 \Pi_1 = 0$ (orthogonal)

### Multi-Qubit Measurement

Measuring qubit $k$ in an $n$-qubit system:

$$\Pi_0^{(k)} = I^{\otimes k} \otimes |0\rangle\langle 0| \otimes I^{\otimes(n-k-1)}$$

**Example:** Measuring q₁ in a 3-qubit system:
$$\Pi_0^{(1)} = I \otimes |0\rangle\langle 0| \otimes I$$

### Mid-Circuit Measurement

Measurements can occur **anywhere** in a circuit, not just at the end:

```
         ┌───┐         ┌───┐
q₀: ─────┤ H ├────●────┤ H ├────
         └───┘    │    └───┘
                  │
q₁: ──────────────⊕────[M]═══c
```

**State evolution with mid-circuit measurement:**
1. Apply gates before measurement
2. Compute measurement probabilities
3. Collapse state based on outcome
4. Continue with remaining gates

### Partial Measurement

Measuring only some qubits leaves others in a conditional state.

**For Bell state** $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

Measure qubit 0:
- Outcome 0 (prob 1/2): State becomes $|00\rangle$
- Outcome 1 (prob 1/2): State becomes $|11\rangle$

After partial measurement, qubits become **classically correlated**.

### The Deferred Measurement Principle

**Theorem (Deferred Measurement):** Any mid-circuit measurement followed by classical control can be replaced by:
1. Coherent quantum control
2. Measurement at the end

```
Original:                        Equivalent:
         ┌───┐                            ┌───┐
q₀: ─────┤ H ├───[M]═══●════     q₀: ─────┤ H ├───●───[M]
         └───┘         ║                  └───┘   │
                     ┌─╨─┐                      ┌─┴─┐
q₁: ─────────────────┤ X ├───     q₁: ─────────┤ X ├───[M]
                     └───┘                      └───┘
```

**Why it works:** Classical control based on measurement is equivalent to quantum control before measurement.

### Measurement in Different Bases

To measure in a different basis, first rotate to that basis:

**X-basis measurement:**
```
         ┌───┐
q: ──────┤ H ├────[M]
         └───┘
```
Measures in $\{|+\rangle, |-\rangle\}$ basis.

**Y-basis measurement:**
```
         ┌────┐   ┌───┐
q: ──────┤ S† ├───┤ H ├────[M]
         └────┘   └───┘
```

**General basis** $\{|\phi_0\rangle, |\phi_1\rangle\}$:
Apply $U^\dagger$ where $U|0\rangle = |\phi_0\rangle$, then measure in Z-basis.

### Born Rule for Circuits

For a circuit with final state $|\psi_f\rangle$, measuring qubit $k$:

$$P(m_k = 0) = \langle\psi_f|\Pi_0^{(k)}|\psi_f\rangle = ||\Pi_0^{(k)}|\psi_f\rangle||^2$$

$$P(m_k = 1) = \langle\psi_f|\Pi_1^{(k)}|\psi_f\rangle = ||\Pi_1^{(k)}|\psi_f\rangle||^2$$

### Measurement and Density Matrices

After measuring qubit $k$ with outcome $m$:

$$\rho \to \rho' = \frac{\Pi_m^{(k)} \rho \Pi_m^{(k)}}{\text{Tr}(\Pi_m^{(k)} \rho)}$$

If we don't record the outcome (non-selective measurement):
$$\rho \to \rho' = \sum_m \Pi_m^{(k)} \rho \Pi_m^{(k)}$$

---

## Worked Examples

### Example 1: Bell State Measurement

Analyze measurement outcomes for:
```
         ┌───┐
|0⟩ ─────┤ H ├────●────[M]═══ outcome a
         └───┘    │
|0⟩ ──────────────⊕────[M]═══ outcome b
```

**Solution:**

After the circuit (before measurement):
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

**Joint probabilities:**
- $P(a=0, b=0) = |1/\sqrt{2}|^2 = 1/2$
- $P(a=0, b=1) = 0$
- $P(a=1, b=0) = 0$
- $P(a=1, b=1) = |1/\sqrt{2}|^2 = 1/2$

**Key observation:** Outcomes are perfectly correlated! $a = b$ always.

This is the signature of entanglement in measurement statistics.

### Example 2: Mid-Circuit Measurement

Analyze this circuit:
```
         ┌───┐         ┌───┐
|0⟩ ─────┤ H ├───[M]═══╪═══════ outcome m
         └───┘         ║
                     ┌─╨─┐
|0⟩ ─────────────────┤ X ├────[M]
                     └───┘
```
(X is applied to q₁ only if m=1)

**Solution:**

**Step 1:** After H on q₀:
$$|\psi_1\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

**Step 2:** Measure q₀:
- If m=0 (prob 1/2): State is $|00\rangle$
- If m=1 (prob 1/2): State is $|10\rangle$

**Step 3:** Classically controlled X on q₁:
- If m=0: No X applied, state remains $|00\rangle$
- If m=1: X applied to q₁, state becomes $|11\rangle$

**Final distribution:**
- $P(00) = 1/2$
- $P(11) = 1/2$

**This is the same as Bell state measurement!** (Deferred measurement principle)

### Example 3: X-Basis Measurement

What is the outcome distribution when measuring $|0\rangle$ in the X-basis?

**Solution:**

```
         ┌───┐
|0⟩ ─────┤ H ├────[M]
         └───┘
```

**State after H:**
$$H|0\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

**Probabilities:**
$$P(0) = |1/\sqrt{2}|^2 = 1/2$$
$$P(1) = |1/\sqrt{2}|^2 = 1/2$$

**Interpretation:** In the X-basis:
- Outcome 0 corresponds to $|+\rangle$
- Outcome 1 corresponds to $|-\rangle$

Since $|0\rangle = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$, equal probability for each.

---

## Practice Problems

### Problem 1: GHZ Measurement
What are the possible measurement outcomes (and probabilities) for:
```
GHZ state: (|000⟩ + |111⟩)/√2

All three qubits measured in Z-basis.
```

### Problem 2: Partial Measurement
After preparing $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$ and measuring q₀ with outcome 1, what is the state of q₁?

### Problem 3: Deferred Measurement
Show that these two circuits produce identical measurement statistics:
```
Circuit A:                    Circuit B:
         ┌───┐                         ┌───┐
q₀: ─────┤ H ├───[M]═●════      q₀: ───┤ H ├───●───[M]
         └───┘       ║                 └───┘   │
                   ┌─╨─┐                     ┌─┴─┐
q₁: ───────────────┤ Z ├───      q₁: ───────┤cZ ├───[M]
                   └───┘                     └───┘
```

### Problem 4: Y-Basis Measurement
Compute the measurement distribution for $|+\rangle$ measured in the Y-basis.

---

## Computational Lab

```python
"""Day 577: Measurement in Circuits"""
import numpy as np
from numpy.random import choice

# Standard gates
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
Sdg = np.array([[1, 0], [0, -1j]])  # S dagger

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

def tensor(*args):
    """Tensor product of multiple matrices"""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result

def projector(outcome, n_qubits, qubit):
    """Projector for measuring qubit in n-qubit system"""
    if outcome == 0:
        P = np.array([[1, 0], [0, 0]])
    else:
        P = np.array([[0, 0], [0, 1]])

    ops = [I] * n_qubits
    ops[qubit] = P
    return tensor(*ops)

def measure_qubit(state, qubit, n_qubits):
    """
    Measure a qubit and collapse the state.
    Returns (outcome, new_state, probability)
    """
    P0 = projector(0, n_qubits, qubit)
    P1 = projector(1, n_qubits, qubit)

    p0 = np.real(np.vdot(state, P0 @ state))
    p1 = np.real(np.vdot(state, P1 @ state))

    # Sample outcome
    outcome = choice([0, 1], p=[p0, p1])

    # Collapse state
    if outcome == 0:
        new_state = P0 @ state / np.sqrt(p0)
        return 0, new_state, p0
    else:
        new_state = P1 @ state / np.sqrt(p1)
        return 1, new_state, p1

def measure_probabilities(state, n_qubits):
    """Calculate all measurement probabilities in computational basis"""
    probs = np.abs(state)**2
    return probs

def state_string(state, n_qubits):
    """Pretty print state"""
    terms = []
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-10:
            bits = format(i, f'0{n_qubits}b')
            if np.abs(amp.imag) < 1e-10:
                terms.append(f"{amp.real:.4f}|{bits}>")
            else:
                terms.append(f"({amp:.4f})|{bits}>")
    return " + ".join(terms)

# ===== Example 1: Bell State Measurement =====
print("=" * 60)
print("Example 1: Bell State Measurement Statistics")
print("=" * 60)

# Prepare Bell state
state_00 = np.array([1, 0, 0, 0], dtype=complex)
state_bell = CNOT @ tensor(H, I) @ state_00

print(f"Bell state: {state_string(state_bell, 2)}")
print("\nMeasurement probabilities:")
probs = measure_probabilities(state_bell, 2)
for i, p in enumerate(probs):
    bits = format(i, '02b')
    print(f"  P({bits}) = {p:.4f}")

# Simulate many measurements
print("\nSimulating 10000 measurements:")
outcomes = {'00': 0, '01': 0, '10': 0, '11': 0}
for _ in range(10000):
    # Fresh Bell state each time
    state = state_bell.copy()

    # Measure both qubits
    o0, state, _ = measure_qubit(state, 0, 2)
    o1, state, _ = measure_qubit(state, 1, 2)

    key = f"{o0}{o1}"
    outcomes[key] += 1

print("  Empirical frequencies:")
for k, v in outcomes.items():
    print(f"    {k}: {v/10000:.4f}")

# ===== Example 2: Partial Measurement =====
print("\n" + "=" * 60)
print("Example 2: Partial Measurement and Collapse")
print("=" * 60)

print("\nStarting with Bell state |Phi+>")
state = state_bell.copy()

print(f"Initial state: {state_string(state, 2)}")

# Measure q0 only
print("\nMeasuring qubit 0...")
for trial in range(3):
    state_copy = state_bell.copy()
    outcome, collapsed, prob = measure_qubit(state_copy, 0, 2)
    print(f"  Trial {trial+1}: outcome={outcome}, collapsed to {state_string(collapsed, 2)}")

# ===== Example 3: Mid-Circuit Measurement =====
print("\n" + "=" * 60)
print("Example 3: Mid-Circuit Measurement")
print("=" * 60)
print("\nCircuit:")
print("         +---+")
print("|0> -----| H |---[M]---*---")
print("         +---+         |")
print("|0> -------------------X---")
print("\n(classically controlled X)")

def mid_circuit_simulation(num_trials=10000):
    """Simulate mid-circuit measurement with classical control"""
    outcomes = {'00': 0, '01': 0, '10': 0, '11': 0}

    for _ in range(num_trials):
        # Initial state |00>
        state = np.array([1, 0, 0, 0], dtype=complex)

        # Apply H to q0
        state = tensor(H, I) @ state

        # Measure q0
        m0, state, _ = measure_qubit(state, 0, 2)

        # Classically controlled X on q1
        if m0 == 1:
            state = tensor(I, X) @ state

        # Measure q1
        m1, state, _ = measure_qubit(state, 1, 2)

        key = f"{m0}{m1}"
        outcomes[key] += 1

    return outcomes

outcomes_mid = mid_circuit_simulation()
print("\nMid-circuit measurement results (10000 trials):")
for k, v in sorted(outcomes_mid.items()):
    print(f"  {k}: {v/10000:.4f}")

# ===== Example 4: Deferred Measurement Equivalence =====
print("\n" + "=" * 60)
print("Example 4: Deferred Measurement Principle")
print("=" * 60)
print("\nShowing equivalence of:")
print("  A: Mid-circuit measure + classical control")
print("  B: Quantum control + final measurement")

# Circuit A: Already done above
# Circuit B: All quantum, measure at end
def quantum_control_circuit(num_trials=10000):
    """CNOT instead of classical control"""
    outcomes = {'00': 0, '01': 0, '10': 0, '11': 0}

    for _ in range(num_trials):
        state = np.array([1, 0, 0, 0], dtype=complex)

        # H on q0
        state = tensor(H, I) @ state

        # CNOT (quantum control)
        state = CNOT @ state

        # Measure both at end
        m0, state, _ = measure_qubit(state, 0, 2)
        m1, state, _ = measure_qubit(state, 1, 2)

        key = f"{m0}{m1}"
        outcomes[key] += 1

    return outcomes

outcomes_quantum = quantum_control_circuit()
print("\nCircuit A (mid-circuit):")
for k, v in sorted(outcomes_mid.items()):
    print(f"  {k}: {v/10000:.4f}")
print("\nCircuit B (quantum control):")
for k, v in sorted(outcomes_quantum.items()):
    print(f"  {k}: {v/10000:.4f}")
print("\nStatistics are identical! (Deferred measurement principle)")

# ===== Example 5: Measurement in Different Bases =====
print("\n" + "=" * 60)
print("Example 5: Measurement in Different Bases")
print("=" * 60)

# X-basis measurement of |0>
print("\nX-basis measurement of |0>:")
state_0 = np.array([1, 0], dtype=complex)
state_after_H = H @ state_0
probs_x = np.abs(state_after_H)**2
print(f"  |0> after H: {state_string(state_after_H, 1)}")
print(f"  P(+) = P(outcome 0) = {probs_x[0]:.4f}")
print(f"  P(-) = P(outcome 1) = {probs_x[1]:.4f}")

# Y-basis measurement of |+>
print("\nY-basis measurement of |+>:")
state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
state_after_SH = H @ Sdg @ state_plus
probs_y = np.abs(state_after_SH)**2
print(f"  |+> after S^dag H: {state_string(state_after_SH, 1)}")
print(f"  P(|+i>) = P(outcome 0) = {probs_y[0]:.4f}")
print(f"  P(|-i>) = P(outcome 1) = {probs_y[1]:.4f}")

# ===== Example 6: GHZ State Measurement =====
print("\n" + "=" * 60)
print("Example 6: GHZ State Measurement")
print("=" * 60)

# Prepare GHZ state
state_000 = np.zeros(8, dtype=complex)
state_000[0] = 1

# H on q0
state = tensor(H, I, I) @ state_000

# CNOT on q0, q1
CNOT_01_3q = tensor(CNOT, I)
state = CNOT_01_3q @ state

# CNOT on q0, q2 (need to construct)
CNOT_02 = np.zeros((8, 8), dtype=complex)
for i in range(8):
    q0 = (i >> 2) & 1
    q1 = (i >> 1) & 1
    q2 = i & 1
    q2_new = q2 ^ q0  # XOR
    j = (q0 << 2) | (q1 << 1) | q2_new
    CNOT_02[j, i] = 1

state_ghz = CNOT_02 @ state

print(f"GHZ state: {state_string(state_ghz, 3)}")
print("\nMeasurement probabilities:")
probs = measure_probabilities(state_ghz, 3)
for i, p in enumerate(probs):
    if p > 1e-10:
        bits = format(i, '03b')
        print(f"  P({bits}) = {p:.4f}")

print("\nOnly |000> and |111> have non-zero probability!")
print("All three measurements are perfectly correlated.")
```

**Expected Output:**
```
============================================================
Example 1: Bell State Measurement Statistics
============================================================
Bell state: 0.7071|00> + 0.7071|11>

Measurement probabilities:
  P(00) = 0.5000
  P(01) = 0.0000
  P(10) = 0.0000
  P(11) = 0.5000

Simulating 10000 measurements:
  Empirical frequencies:
    00: 0.5023
    01: 0.0000
    10: 0.0000
    11: 0.4977
```

---

## Summary

### Measurement Notation

| Symbol | Meaning |
|--------|---------|
| `[M]` | Measurement in computational basis |
| `═══` | Classical wire (carries bit) |
| `[H]--[M]` | X-basis measurement |
| `[S†]--[H]--[M]` | Y-basis measurement |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Projector | $\Pi_m = \|m\rangle\langle m\|$ |
| Probability | $P(m) = \|\|\Pi_m\|\psi\rangle\|\|^2$ |
| Collapse | $\|\psi'\rangle = \Pi_m\|\psi\rangle / \sqrt{P(m)}$ |
| Completeness | $\sum_m \Pi_m = I$ |

### Key Takeaways

1. **Projective measurement** collapses the state to an eigenstate
2. **Partial measurement** creates classical correlation between qubits
3. **Mid-circuit measurement** allows adaptive quantum computation
4. **Deferred measurement principle**: classical control = quantum control + late measurement
5. **Basis change**: apply $U^\dagger$ before measuring to measure in $U$-rotated basis
6. **Born rule** gives probabilities as squared amplitudes

---

## Daily Checklist

- [ ] I can represent measurements in circuit notation
- [ ] I can calculate measurement probabilities using the Born rule
- [ ] I understand state collapse after measurement
- [ ] I can apply the deferred measurement principle
- [ ] I can set up measurements in X, Y, and arbitrary bases
- [ ] I ran the computational lab and verified measurement statistics

---

*Next: Day 578 — Classical Control*
