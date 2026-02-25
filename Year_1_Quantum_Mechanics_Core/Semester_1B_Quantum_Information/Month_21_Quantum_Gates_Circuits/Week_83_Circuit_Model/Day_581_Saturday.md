# Day 581: Week 83 Review — Circuit Model

## Overview
**Day 581** | Week 83, Day 7 | Year 1, Month 21 | Weekly Synthesis and Assessment

Today we consolidate our understanding of the quantum circuit model—circuit diagrams, composition rules, measurement, classical control, complexity metrics, and optimization techniques.

---

## Learning Objectives

1. Synthesize circuit model concepts from the entire week
2. Solve comprehensive problems combining multiple topics
3. Identify connections between different circuit concepts
4. Apply circuit analysis to practical quantum algorithms
5. Prepare for Week 84's universality theory
6. Self-assess mastery of circuit model fundamentals

---

## Week 83 Summary

### Day 575: Circuit Diagrams
- **Wires** represent qubits evolving in time
- **Time flows left to right**
- **Gates** are boxes/symbols on wires
- **Multi-qubit gates** use vertical connections
- **Measurement** symbol at circuit end or mid-circuit

### Day 576: Circuit Composition
- **Sequential:** $U = U_n \cdots U_2 U_1$ (reverse order)
- **Parallel:** $U = U_A \otimes U_B$
- **Mixed-product:** $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$
- **Controlled gates** cannot be written as simple tensor products

### Day 577: Measurement
- **Projective measurement** collapses state
- **Partial measurement** creates classical correlation
- **Mid-circuit measurement** enables adaptive algorithms
- **Deferred measurement principle:** classical control = quantum control + late measurement

### Day 578: Classical Control
- **Feedforward:** measurement outcomes control future gates
- **Measure-and-correct:** fundamental for error correction
- **Teleportation:** canonical example of classical control
- **Pauli frame tracking:** defer corrections classically

### Day 579: Circuit Complexity
- **Depth:** longest gate path (execution time)
- **Width:** number of qubits (memory)
- **T-count:** critical for fault tolerance
- **Trade-offs:** depth vs width, gate count vs ancillas

### Day 580: Circuit Optimization
- **Gate cancellation:** $UU^\dagger = I$
- **Rotation merging:** combine same-axis rotations
- **Commutation rules:** reorder for optimization
- **Template matching:** pattern-based simplification

---

## Key Formulas Reference

### Circuit Composition
$$U_{total} = U_n \cdots U_2 \cdot U_1$$
$$U_{parallel} = U_A \otimes U_B$$
$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

### Controlled Gate
$$C_U = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U$$

### Measurement
$$P(m) = \langle\psi|\Pi_m|\psi\rangle = ||\Pi_m|\psi\rangle||^2$$
$$|\psi'\rangle = \frac{\Pi_m|\psi\rangle}{\sqrt{P(m)}}$$

### Gate Identities
$$H^2 = X^2 = Y^2 = Z^2 = CNOT^2 = I$$
$$T^2 = S, \quad T^4 = Z, \quad T^8 = I$$
$$HXH = Z, \quad HZH = X$$

---

## Comprehensive Problems

### Problem 1: Circuit Analysis

Analyze this circuit completely:
```
         ┌───┐         ┌───┐
|0⟩ ─────┤ H ├────●────┤ S ├────[M]
         └───┘    │    └───┘
                  │
|0⟩ ──────────────⊕────[H]─────[M]
```

**Tasks:**
a) Write the matrix expression for the circuit
b) Calculate the final state before measurement
c) Find all measurement outcome probabilities
d) Compute the circuit depth, gate count

**Solution:**

a) **Matrix expression:**
$$U = (S \otimes H) \cdot CNOT \cdot (H \otimes I)$$

b) **State evolution:**
- Initial: $|00\rangle$
- After $H \otimes I$: $\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$
- After CNOT: $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
- After $S \otimes H$: $\frac{1}{\sqrt{2}}(|0\rangle \otimes |+\rangle + i|1\rangle \otimes |-\rangle)$

Expanding:
$$= \frac{1}{2}(|00\rangle + |01\rangle + i|10\rangle - i|11\rangle)$$

c) **Probabilities:**
$$P(00) = |1/2|^2 = 1/4$$
$$P(01) = |1/2|^2 = 1/4$$
$$P(10) = |i/2|^2 = 1/4$$
$$P(11) = |-i/2|^2 = 1/4$$

d) **Complexity:**
- Depth: 3 (H, CNOT, S⊗H)
- Gate count: 4 (H, CNOT, S, H)

---

### Problem 2: Optimization Challenge

Simplify this circuit:
```
         ┌───┐   ┌───┐   ┌───┐         ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ T ├───┤ H ├────●────┤ H ├───┤ T ├───
         └───┘   └───┘   └───┘    │    └───┘   └───┘
                                  │
         ┌───┐                    │    ┌───┐
q₁: ─────┤ X ├────────────────────⊕────┤ X ├───────────
         └───┘                         └───┘
```

**Solution:**

Step 1: Recognize $HTH$ pattern (doesn't simplify directly, but note structure)

Step 2: $X$ gates on q₁ sandwich CNOT:
- X before CNOT on target: X commutes through
- X after CNOT on target: also commutes

Result: X·CNOT·X on target = CNOT (X gates cancel by commutation)

Step 3: Two H gates on q₀ around T don't cancel, but the second pair of H's:
- After CNOT: H then T on q₀
- Pattern H-T-H = rotation in different basis

**Simplified form:**
```
         ┌───┐   ┌───┐         ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ T ├────●────┤ H ├───┤ T ├───
         └───┘   └───┘    │    └───┘   └───┘
                          │
q₁: ──────────────────────⊕────────────────────
```

**Reduction:** 8 gates → 5 gates (removed 2 X gates, 1 H)

---

### Problem 3: Teleportation Circuit Analysis

Complete the teleportation analysis for input $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:

```
|ψ⟩ ────────●────[H]────[M]═══════════●═════════
            │                         ║
|0⟩ ──[H]───⊕──────────[M]═══════●═══╬═════════
                                 ║   ║
|0⟩ ─────────────────────────────X═══Z═════════
```

**Find:**
a) State after Bell measurement (before classical control)
b) Correction operations needed for each measurement outcome
c) Verify output equals input

**Solution:**

a) **Bell measurement outcomes:**

The Bell measurement on qubits 1-2 projects onto Bell basis:
- $|00\rangle$: State of q₃ is $\alpha|0\rangle + \beta|1\rangle = |\psi\rangle$
- $|01\rangle$: State is $\alpha|1\rangle + \beta|0\rangle = X|\psi\rangle$
- $|10\rangle$: State is $\alpha|0\rangle - \beta|1\rangle = Z|\psi\rangle$
- $|11\rangle$: State is $\alpha|1\rangle - \beta|0\rangle = XZ|\psi\rangle$

b) **Corrections:**

| $m_1$ | $m_2$ | State | Correction |
|-------|-------|-------|------------|
| 0 | 0 | $\|\psi\rangle$ | $I$ |
| 0 | 1 | $X\|\psi\rangle$ | $X$ |
| 1 | 0 | $Z\|\psi\rangle$ | $Z$ |
| 1 | 1 | $XZ\|\psi\rangle$ | $ZX$ |

General correction: $Z^{m_1} X^{m_2}$

c) **Verification:** After correction, $Z^{m_1} X^{m_2} \cdot (\text{state}) = |\psi\rangle$ in all cases.

---

### Problem 4: Complexity Analysis

Given a circuit that implements $n$-qubit QFT:
- Gate count: $n + n(n-1)/2$ (H gates + controlled rotations)
- Each controlled rotation decomposes to ~3 CNOTs and rotations

**Find:**
a) Total gate count as $O(?)$
b) CNOT count
c) If each rotation requires 1 T gate on average, estimate T-count

**Solution:**

a) **Gate count:**
$$n + \frac{n(n-1)}{2} = \frac{n^2 + n}{2} = O(n^2)$$

b) **CNOT count:**
- $\frac{n(n-1)}{2}$ controlled rotations
- Each needs ~3 CNOTs (typical decomposition)
- Total: $\frac{3n(n-1)}{2} = O(n^2)$ CNOTs

c) **T-count:**
- Each controlled rotation needs ~2-4 T gates (after synthesis)
- With ~2 T per rotation: $n(n-1) = O(n^2)$ T gates

---

## Practice Problems

### Problem 5: Mixed Concepts
Design a circuit that:
1. Creates $|\Phi^+\rangle$ Bell state
2. Measures first qubit
3. Uses classical control to ensure second qubit is always $|0\rangle$

### Problem 6: Deferred Measurement
Show that mid-circuit measurement followed by conditional Z is equivalent to CZ followed by measurement at the end.

### Problem 7: Complexity Trade-off
A circuit has depth 100 and width 10. An alternative uses ancillas to achieve depth 20 with width 30. Calculate the volume of each and discuss trade-offs.

### Problem 8: Optimization
Find the simplified form of:
$$CNOT \cdot (H \otimes H) \cdot CNOT \cdot (H \otimes H)$$

---

## Computational Lab: Comprehensive Review

```python
"""Day 581: Week 83 Review - Circuit Model Comprehensive Lab"""
import numpy as np

# ===== Gate Definitions =====
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

def tensor(*args):
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result

def state_str(psi, n):
    terms = []
    for i, a in enumerate(psi):
        if np.abs(a) > 1e-10:
            terms.append(f"({a:.3f})|{format(i, f'0{n}b')}>")
    return " + ".join(terms)

# ===== Review Problem 1: Full Circuit Analysis =====
print("=" * 70)
print("Review Problem 1: Circuit Analysis")
print("=" * 70)
print("""
Circuit:
         +---+         +---+
|0> -----| H |----*----| S |----[M]
         +---+    |    +---+
                  |
|0> --------------X----[H]-----[M]
""")

# Build circuit
state = np.array([1, 0, 0, 0], dtype=complex)  # |00>

# Layer 1: H on q0
U1 = tensor(H, I)
state = U1 @ state
print(f"After H on q0: {state_str(state, 2)}")

# Layer 2: CNOT
state = CNOT @ state
print(f"After CNOT:    {state_str(state, 2)}")

# Layer 3: S on q0, H on q1
U3 = tensor(S, H)
state = U3 @ state
print(f"After S, H:    {state_str(state, 2)}")

# Measurement probabilities
probs = np.abs(state)**2
print(f"\nMeasurement probabilities:")
for i, p in enumerate(probs):
    print(f"  P({format(i, '02b')}) = {p:.4f}")

# ===== Review Problem 2: Optimization =====
print("\n" + "=" * 70)
print("Review Problem 2: Gate Identity Verification")
print("=" * 70)

# Verify key identities
print("\nIdentity checks:")
print(f"  H^2 = I: {np.allclose(H @ H, I)}")
print(f"  CNOT^2 = I: {np.allclose(CNOT @ CNOT, np.eye(4))}")
print(f"  T^4 = Z: {np.allclose(T @ T @ T @ T, Z)}")
print(f"  T^8 = I: {np.allclose(np.linalg.matrix_power(T, 8), I)}")
print(f"  HXH = Z: {np.allclose(H @ X @ H, Z)}")
print(f"  HZH = X: {np.allclose(H @ Z @ H, X)}")

# Mixed product property
A, B, C, D = H, X, S, T
left = tensor(A, B) @ tensor(C, D)
right = tensor(A @ C, B @ D)
print(f"  (A⊗B)(C⊗D) = (AC)⊗(BD): {np.allclose(left, right)}")

# ===== Review Problem 3: Teleportation Verification =====
print("\n" + "=" * 70)
print("Review Problem 3: Teleportation Protocol")
print("=" * 70)

def teleport_verify(alpha, beta):
    """Verify teleportation for state alpha|0> + beta|1>"""

    # Input state to teleport
    psi = np.array([alpha, beta], dtype=complex)
    psi = psi / np.linalg.norm(psi)

    # Initial state: |psi>|00>
    state = tensor(psi, np.array([1, 0]), np.array([1, 0]))

    # Create Bell pair on qubits 2,3
    # H on qubit 2
    state = tensor(np.eye(2), H, I) @ state
    # CNOT(2,3)
    CNOT_23 = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        q1 = (i >> 2) & 1
        q2 = (i >> 1) & 1
        q3 = i & 1
        q3_new = q3 ^ q2
        j = (q1 << 2) | (q2 << 1) | q3_new
        CNOT_23[j, i] = 1
    state = CNOT_23 @ state

    # CNOT(1,2) - qubit 1 controls qubit 2
    CNOT_12 = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        q1 = (i >> 2) & 1
        q2 = (i >> 1) & 1
        q3 = i & 1
        q2_new = q2 ^ q1
        j = (q1 << 2) | (q2_new << 1) | q3
        CNOT_12[j, i] = 1
    state = CNOT_12 @ state

    # H on qubit 1
    state = tensor(H, I, I) @ state

    # Check all measurement outcomes
    results = []
    for m1 in [0, 1]:
        for m2 in [0, 1]:
            # Project onto measurement outcome
            proj = np.zeros(8, dtype=complex)
            for i in range(8):
                q1 = (i >> 2) & 1
                q2 = (i >> 1) & 1
                if q1 == m1 and q2 == m2:
                    proj[i] = state[i]

            norm = np.linalg.norm(proj)
            if norm > 1e-10:
                proj = proj / norm

                # Extract qubit 3 state
                q3_state = np.array([proj[m1*4 + m2*2 + 0], proj[m1*4 + m2*2 + 1]])

                # Apply corrections
                if m2 == 1:
                    q3_state = X @ q3_state
                if m1 == 1:
                    q3_state = Z @ q3_state

                # Check if matches input
                # Account for global phase
                if np.abs(q3_state[0]) > 1e-10:
                    phase = psi[0] / q3_state[0]
                    match = np.allclose(q3_state * phase, psi)
                else:
                    match = np.allclose(np.abs(q3_state), np.abs(psi))

                results.append((m1, m2, match))

    return all(r[2] for r in results)

# Test with various states
test_states = [
    (1, 0, "|0>"),
    (0, 1, "|1>"),
    (1, 1, "|+>"),
    (1, -1, "|->"),
    (1, 1j, "|+i>"),
]

print("\nTeleportation verification for various input states:")
for alpha, beta, name in test_states:
    success = teleport_verify(alpha, beta)
    print(f"  {name}: {'PASS' if success else 'FAIL'}")

# ===== Review Problem 4: Complexity Metrics =====
print("\n" + "=" * 70)
print("Review Problem 4: Circuit Complexity Comparison")
print("=" * 70)

print("""
Compare these circuit implementations:

Circuit A: Serial CNOT chain (n qubits)
  Depth: n-1
  CNOT count: n-1
  Width: n

Circuit B: Parallel Bell pair creation + merge
  Depth: O(log n)
  CNOT count: n-1
  Width: n (or n + ancillas)

For n = 8:
""")

n = 8
print(f"  Serial:   Depth = {n-1}, Volume = {(n-1) * n}")
print(f"  Parallel: Depth ~ {int(np.ceil(np.log2(n)))}, Volume ~ {int(np.ceil(np.log2(n))) * n}")

# ===== Review Problem 5: Measurement Statistics =====
print("\n" + "=" * 70)
print("Review Problem 5: Bell State Measurement Statistics")
print("=" * 70)

# GHZ state measurement
print("\nGHZ state |GHZ> = (|000> + |111>)/sqrt(2)")
ghz = np.zeros(8, dtype=complex)
ghz[0] = 1/np.sqrt(2)  # |000>
ghz[7] = 1/np.sqrt(2)  # |111>

probs = np.abs(ghz)**2
print("Measurement probabilities:")
for i, p in enumerate(probs):
    if p > 1e-10:
        print(f"  P({format(i, '03b')}) = {p:.4f}")

print("\nNote: Only |000> and |111> have nonzero probability")
print("All three measurements are perfectly correlated!")

# ===== Summary Statistics =====
print("\n" + "=" * 70)
print("Week 83 Summary: Circuit Model Key Points")
print("=" * 70)
print("""
1. CIRCUIT DIAGRAMS
   - Time flows left to right
   - Top wire = most significant qubit
   - Gates read left-to-right, matrices multiply right-to-left

2. COMPOSITION
   - Sequential: U = Un...U2·U1
   - Parallel: U = UA ⊗ UB
   - Mixed-product: (A⊗B)(C⊗D) = (AC)⊗(BD)

3. MEASUREMENT
   - Projective: P(m) = ||Πm|ψ>||²
   - Collapse: |ψ'> = Πm|ψ> / sqrt(P(m))
   - Deferred measurement principle enables optimization

4. CLASSICAL CONTROL
   - Feedforward: measurement → conditional gate
   - Teleportation: Z^m1 · X^m2 correction
   - Pauli frame tracking for efficiency

5. COMPLEXITY
   - Depth: execution time, decoherence
   - T-count: dominant for fault tolerance
   - Trade-offs: depth vs width, gates vs ancillas

6. OPTIMIZATION
   - Gate cancellation: UU† = I
   - Rotation merging: Rz(α)Rz(β) = Rz(α+β)
   - Commutation rules enable reordering
   - T-count minimization is crucial
""")
```

---

## Self-Assessment Checklist

### Circuit Diagrams (Day 575)
- [ ] I can read and draw quantum circuit diagrams
- [ ] I understand the time-ordering convention
- [ ] I can identify multi-qubit gate notation

### Circuit Composition (Day 576)
- [ ] I can compute matrix products for sequential gates
- [ ] I can compute tensor products for parallel gates
- [ ] I understand the mixed-product property

### Measurement (Day 577)
- [ ] I can calculate measurement probabilities
- [ ] I understand state collapse after measurement
- [ ] I can apply the deferred measurement principle

### Classical Control (Day 578)
- [ ] I can implement feedforward operations
- [ ] I understand the teleportation protocol
- [ ] I can track Pauli frames

### Complexity (Day 579)
- [ ] I can calculate circuit depth and width
- [ ] I understand why T-count matters
- [ ] I can analyze depth-width trade-offs

### Optimization (Day 580)
- [ ] I can identify and apply gate cancellations
- [ ] I can merge rotations
- [ ] I understand commutation rules for optimization

---

## Preview: Week 84 — Universality

Next week we explore **universal gate sets**:

- **Day 582:** Universal gate sets ({H, T, CNOT} is universal)
- **Day 583:** Solovay-Kitaev theorem (efficient approximation)
- **Day 584:** Clifford gates and stabilizer formalism
- **Day 585:** Clifford+T and magic states
- **Day 586:** Native gate sets for different hardware
- **Day 587:** Compiling to hardware
- **Day 588:** Month 21 comprehensive review

**Key question:** What is the minimal set of gates needed to perform any quantum computation?

---

*Week 83 Complete — Proceed to Week 84: Universality*
