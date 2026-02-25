# Day 864: T-Gate Synthesis

## Week 124: Universal Fault-Tolerant Computation | Month 31: Fault-Tolerant QC I

---

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning | 2.5 hrs | Rotation decomposition theory |
| Afternoon | 2.5 hrs | Optimal synthesis algorithms |
| Evening | 2.0 hrs | RUS circuits and implementation |

---

### Learning Objectives

By the end of today, you will be able to:

1. **Decompose arbitrary Z-rotations** into Clifford+T sequences
2. **Apply the Ross-Selinger Gridsynth algorithm** for optimal T-count
3. **Design Repeat-Until-Success (RUS) circuits** for probabilistic synthesis
4. **Analyze the trade-offs** between ancilla-free and ancilla-assisted synthesis
5. **Use the number-theoretic structure** of the Clifford+T group
6. **Estimate T-counts** for common rotation angles

---

### Core Content

#### Part 1: The Synthesis Problem

**Problem Statement:**

Given a rotation angle $\theta$, find a Clifford+T circuit implementing $R_z(\theta)$ (or approximating it to precision $\epsilon$).

Since only angles of the form $\theta = m\pi/2^k$ are exactly representable, most rotations require approximation.

**The Key Observation:**

Every exactly-representable single-qubit unitary in the Clifford+T group has the form:

$$\boxed{U = \frac{1}{\sqrt{2}^k} \begin{pmatrix} u_0 & -u_1^* \\ u_1 & u_0^* \end{pmatrix}}$$

where $u_0, u_1 \in \mathbb{Z}[\omega]$ with $\omega = e^{i\pi/4}$, and $|u_0|^2 + |u_1|^2 = 2^k$.

**The Ring $\mathbb{Z}[\omega]$:**

Elements of $\mathbb{Z}[\omega]$ have the form:
$$z = a_0 + a_1\omega + a_2\omega^2 + a_3\omega^3$$

where $a_i \in \mathbb{Z}$ and $\omega^4 = i$, $\omega^8 = 1$.

Equivalently: $z = (a + b\omega)(1 + i) + c + d\omega$ for appropriate integers.

---

#### Part 2: Exact Synthesis

**When is Exact Synthesis Possible?**

$R_z(\theta)$ is exactly synthesizable if and only if $e^{i\theta/2} \in \mathbb{Z}[\omega, 1/\sqrt{2}]$.

This happens precisely when $\theta = m\pi/2^k$ for integers $m, k$.

**Theorem (Kliuchnikov-Maslov-Mosca):** For $R_z(m\pi/2^k)$:
- If $k \leq 2$: Clifford gate (0 T-gates)
- If $k = 3$: Requires 1 T-gate
- If $k \geq 3$: Requires $k - 2$ T-gates (optimal)

**Examples:**

| Angle | Gate | T-count |
|-------|------|---------|
| $\pi$ | Z | 0 |
| $\pi/2$ | S | 0 |
| $\pi/4$ | T | 1 |
| $\pi/8$ | $T^{1/2}$ | 2 |
| $\pi/16$ | $T^{1/4}$ | 3 |

**Exact Decomposition Algorithm:**

For $R_z(\pi/2^k)$ with $k \geq 3$:

$$R_z(\pi/2^k) = V_k \cdot T \cdot V_k^\dagger$$

where $V_k$ is recursively constructed.

The circuit for $R_z(\pi/8)$:
```
─────H─T─H─T─H─────
```
This gives $HTH \cdot T = R_x(\pi/4) \cdot R_z(\pi/4)$... (requires careful phase tracking)

Actually, the optimal circuit for $R_z(\pi/8)$:
```
─────T─H─T─H─────
```
Verification: $HTHT$ produces $R_z(\pi/8)$ up to Clifford corrections.

---

#### Part 3: Approximate Synthesis and Gridsynth

For arbitrary $\theta$, we need approximate synthesis. The **Gridsynth** algorithm by Ross and Selinger achieves:

$$\boxed{T\text{-count} \approx 3\log_2\left(\frac{1}{\epsilon}\right) + O(1)}$$

This is provably optimal to within an additive constant.

**The Grid Approach:**

Consider $R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$.

We seek $u \in \mathbb{Z}[\omega]$ with $|u|^2 = 2^k$ such that:

$$\left| \frac{u}{|u|} - e^{i\theta/2} \right| < \epsilon$$

**Algorithm Outline:**

1. **Enumerate T-count levels:** For $k = 0, 1, 2, \ldots$
2. **Search the grid:** Find $u \in \mathbb{Z}[\omega]$ with $|u|^2 = 2^k$ near the target phase
3. **Check approximation:** If $|u/|u| - e^{i\theta/2}| < \epsilon$, succeed
4. **Construct circuit:** Decompose $u$ into T-gates using the structure of $\mathbb{Z}[\omega]$

**Key Lemma (Grid Density):**

The set $\{u/|u| : u \in \mathbb{Z}[\omega], |u|^2 = 2^k\}$ forms a grid on the unit circle with spacing $\approx 1/\sqrt{2^k}$.

To achieve $\epsilon$-approximation, we need $k \approx 2\log_2(1/\epsilon)$.

T-count $\approx k \approx 2\log_2(1/\epsilon)$, but optimal algorithms achieve $\approx 3\log_2(1/\epsilon)$ due to the structure of the problem.

---

**Gridsynth Implementation Details:**

**Step 1: Prime Factorization in $\mathbb{Z}[\omega]$**

The norm form is $N(u) = |u|^2$. We factor $2^k$ in $\mathbb{Z}[\omega]$:
$$2 = -i(1+i)^2$$

So $2^k$ factors into $(1+i)^{2k}$ times units.

**Step 2: Search Strategy**

Rather than brute force, use:
- Continued fraction expansion of $\theta/\pi$
- Lattice reduction (LLL algorithm)
- Branch and bound on the ring structure

**Step 3: Circuit Extraction**

Given $u = a + b\omega$, the T-count equals the "sde" (smallest denominator exponent):
$$\text{sde}(u) = \min\{k : \sqrt{2}^k u \in \mathbb{Z}[\omega]\}$$

The circuit is constructed by repeated division by $(1+\omega)$, extracting T-gates at each step.

---

#### Part 4: Repeat-Until-Success (RUS) Circuits

**Motivation:**

Magic state distillation is expensive. Can we reduce T-count using measurement and classical feedback?

**RUS Principle:**

A probabilistic circuit that:
- Succeeds with probability $p$ and applies the target gate
- Fails with probability $1-p$ and signals failure
- On failure, we reset and retry

Expected T-count: $T_{\text{RUS}} / p$

If $p$ is close to 1 and $T_{\text{RUS}} < T_{\text{deterministic}}$, this wins on average.

**Example: RUS for V-gate**

The $V$-gate is $V = (I + 2iZ)/\sqrt{5}$, which requires $\approx 10$ T-gates deterministically.

**RUS Circuit for V:**

```
|ψ⟩ ──────●──────●─────── |ψ'⟩
          │      │
|0⟩ ──H───⊕──T───⊕───H─M─
                      ↓
                 if 0: done (applied V)
                 if 1: applied V†, correct or retry
```

This uses only 1 T-gate but succeeds with probability 1/2.

Expected T-count: $1 / 0.5 = 2$ T-gates (vs 10 deterministic).

**General RUS Framework:**

For target gate $U$:
1. Find ancilla circuit using few T-gates
2. Measurement outcome determines if $U$ or correction was applied
3. Design correction/retry strategy

**Success Probability Enhancement:**

Using multiple ancillas and more sophisticated circuits, success probabilities can be increased:

| Circuit | T-gates | Success Prob | Expected T |
|---------|---------|--------------|------------|
| Basic RUS | 1 | 0.50 | 2.0 |
| Enhanced | 2 | 0.75 | 2.7 |
| Catalyzed | 3 | 0.875 | 3.4 |

---

#### Part 5: Ancilla-Free vs. Ancilla-Assisted

**Ancilla-Free Synthesis:**

- Uses no additional qubits
- Deterministic
- T-count: $\approx 3\log_2(1/\epsilon)$
- Example: Gridsynth output directly

**Ancilla-Assisted Synthesis:**

- Uses ancilla qubits for measurement
- Probabilistic (RUS) or deterministic (with catalysis)
- Can achieve lower expected T-count
- Requires classical control

**Trade-off Analysis:**

| Factor | Ancilla-Free | Ancilla-Assisted |
|--------|--------------|------------------|
| Qubit overhead | None | +1 to +k ancillas |
| T-count | Higher | Lower (expected) |
| Depth | Predictable | Variable |
| Classical control | None | Required |
| Failure handling | N/A | Retry needed |

**When to Use Each:**

- **Ancilla-free:** Limited qubit budget, predictable timing needed
- **RUS:** Abundant qubits, T-gates bottleneck, can tolerate variable depth

---

#### Part 6: Practical Synthesis Tools

**Available Software:**

1. **Gridsynth (Original):** Ross-Selinger implementation in Haskell
   - Provably optimal for single-qubit z-rotations
   - Outputs Clifford+T circuit

2. **t|ket⟩:** Cambridge Quantum compiler
   - Includes synthesis pass
   - Integrates with full compilation stack

3. **Qiskit Transpiler:** IBM's compiler
   - `UnrollCustomDefinitions` + `Decompose`
   - Not optimal but practical

4. **Cirq:** Google's framework
   - `cirq.optimizers.MergeInteractions`
   - Focus on hardware-native gates

**Synthesis Quality Metrics:**

| Metric | Definition | Target |
|--------|------------|--------|
| T-count | Number of T-gates | Minimize |
| T-depth | Layers of T-gates (parallel) | Minimize |
| Clifford count | Number of Clifford gates | Secondary |
| Total depth | Circuit depth | Minimize for coherence |

---

### Algorithm Design Implications

**Optimizing T-Count in Algorithms:**

1. **Combine rotations:** $R_z(\theta_1) R_z(\theta_2) = R_z(\theta_1 + \theta_2)$
2. **Use symmetries:** If $\theta = -\phi$ appears, use $R_z(\theta)^\dagger$ instead
3. **Approximate collectively:** Synthesize sums rather than individual angles
4. **Exploit cancellation:** Adjacent $T T^\dagger$ pairs cancel

**T-Count for Common Subroutines:**

| Operation | T-count |
|-----------|---------|
| Single arbitrary rotation | $\approx 3\log_2(1/\epsilon)$ |
| Toffoli gate | 7 T-gates (optimal) |
| Controlled-rotation | $\approx 6\log_2(1/\epsilon)$ |
| QFT on n qubits | $\approx 3n^2\log_2(1/\epsilon)$ |

---

### Worked Examples

#### Example 1: Gridsynth for $R_z(\pi/5)$

**Problem:** Find Clifford+T approximation for $R_z(\pi/5)$ to precision $\epsilon = 10^{-3}$.

**Solution:**

**Step 1:** Target phase
$$e^{i\pi/10} = \cos(\pi/10) + i\sin(\pi/10) \approx 0.951 + 0.309i$$

**Step 2:** Estimate T-count
$$k \approx 3\log_2(1/\epsilon) = 3\log_2(1000) \approx 30$$

Actually, for $\epsilon = 10^{-3}$, gridsynth typically finds solutions with $k \approx 10-15$.

**Step 3:** Search grid
Find $u \in \mathbb{Z}[\omega]$ with $|u|^2 = 2^k$ such that $\arg(u) \approx \pi/10$.

For $k = 12$: We search for $u = a + b\omega + c\omega^2 + d\omega^3$ with:
- $|u|^2 = a^2 + b^2 + c^2 + d^2 + \sqrt{2}(\text{cross terms}) = 4096$
- $\arg(u) \approx 0.314$ radians

**Step 4:** One solution
$u = 63 + 7\omega$ (simplified example)
$|u|^2 = 63^2 + 7^2 + 63 \cdot 7 \cdot \sqrt{2} \approx ...$ (actual calculation needed)

**Step 5:** Circuit construction
The circuit has $\approx 12$ T-gates total.

Actual gridsynth output for this precision: approximately 10-12 T-gates.

---

#### Example 2: RUS Circuit Analysis

**Problem:** Design an RUS circuit for $R_z(\pi/5)$ and compute expected T-count.

**Solution:**

**Step 1:** Identify approximating angle

$\pi/5 = 0.628...$ radians.

Nearby "nice" angles:
- $\pi/4 = 0.785$ (1 T-gate, error 0.157)
- $\pi/8 = 0.393$ (2 T-gates, error 0.235)

**Step 2:** RUS decomposition

We can write:
$$R_z(\pi/5) = R_z(\pi/4) \cdot R_z(\pi/5 - \pi/4) = T \cdot R_z(-0.157)$$

Use RUS for the small rotation $R_z(-0.157)$.

**Step 3:** RUS for small rotation

For $R_z(\phi)$ with $|\phi| \ll 1$:

```
|ψ⟩ ─────●─────────●───── |ψ'⟩
         │         │
|0⟩ ──H──⊕──Rz(φ)──⊕──H──M
```

If measurement = 0: Applied $R_z(2\phi)$ to $|\psi\rangle$ (probability $\cos^2(\phi)$)
If measurement = 1: Applied identity (probability $\sin^2(\phi)$)

For $\phi = -0.0785$ (half target): success probability $\approx 0.994$.

**Step 4:** Expected T-count

$T_{\text{expected}} = 1 + \frac{T_{R_z(-0.157)}}{0.994} \approx 1 + \frac{6}{0.994} \approx 7$

vs. deterministic: $\approx 10$ T-gates.

---

#### Example 3: Multi-Rotation Optimization

**Problem:** Optimize T-count for the sequence $R_z(\pi/3) R_z(\pi/6) R_z(\pi/4)$.

**Solution:**

**Naive approach:**
- $R_z(\pi/3)$: $\approx 20$ T-gates (for $\epsilon = 10^{-6}$)
- $R_z(\pi/6)$: $\approx 20$ T-gates
- $R_z(\pi/4) = T$: 1 T-gate
- Total: $\approx 41$ T-gates

**Optimized:**

Combine: $\pi/3 + \pi/6 + \pi/4 = 2\pi/6 + \pi/6 + \pi/4 = 3\pi/6 + \pi/4 = \pi/2 + \pi/4 = 3\pi/4$

$$R_z(\pi/3) R_z(\pi/6) R_z(\pi/4) = R_z(3\pi/4) = S \cdot T = T^3$$

Optimized T-count: **3 T-gates** (vs. 41 naive)!

This is an exact synthesis since $3\pi/4 = 3 \cdot \pi/4$ is a multiple of $\pi/4$.

**Lesson:** Always combine rotations before synthesis when possible.

---

### Practice Problems

#### Level 1: Direct Application

**Problem 1.1:** How many T-gates are needed for exact synthesis of $R_z(\pi/32)$?

**Problem 1.2:** If gridsynth achieves $\epsilon = 10^{-4}$ with 12 T-gates, estimate the T-count for $\epsilon = 10^{-8}$.

**Problem 1.3:** An RUS circuit succeeds with probability 0.8 and uses 2 T-gates. What is the expected T-count?

---

#### Level 2: Intermediate

**Problem 2.1:** Prove that $R_z(\theta)$ is exactly synthesizable iff $e^{i\theta/2} \in \mathbb{Z}[\omega, 1/\sqrt{2}]$.

Hint: The Clifford+T unitaries are precisely those with entries in this ring.

**Problem 2.2:** Design an RUS circuit for the $\sqrt{T}$ gate using only one T-gate per attempt. What is the success probability?

**Problem 2.3:** Given $R_z(\theta_1), R_z(\theta_2), R_z(\theta_3)$ in a circuit, under what conditions can we reduce total T-count by combining?

---

#### Level 3: Challenging

**Problem 3.1:** **(Gridsynth Optimality)**
Prove that any algorithm approximating arbitrary $R_z(\theta)$ to precision $\epsilon$ using Clifford+T requires $\Omega(\log(1/\epsilon))$ T-gates.

Hint: Count the number of distinct rotations achievable with $k$ T-gates.

**Problem 3.2:** **(RUS Optimization)**
Design an RUS circuit for $R_z(\pi/6)$ that uses:
(a) 1 T-gate per attempt with success probability $p_1$
(b) 2 T-gates per attempt with success probability $p_2$

Find $p_1, p_2$ and determine which gives lower expected T-count.

**Problem 3.3:** **(Multi-Qubit Extension)**
For a controlled-$R_z(\theta)$ gate, compare:
(a) Direct synthesis of the two-qubit unitary
(b) Decomposition into single-qubit gates + CNOT, then synthesis

Which is more efficient for arbitrary $\theta$?

---

### Computational Lab

```python
"""
Day 864 Computational Lab: T-Gate Synthesis
Implementing approximation algorithms and RUS circuits
"""

import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt

# Gate definitions
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
T_dag = np.conj(T).T

omega = np.exp(1j * np.pi / 4)

def Rz(theta):
    """Z-rotation by angle theta"""
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

def matrix_distance(A, B):
    """Operator norm distance"""
    return np.linalg.norm(A - B, ord=2)

class ZRotationSynthesis:
    """
    Simple grid-based synthesis for Rz rotations.
    Not optimal like true Gridsynth, but demonstrates the concept.
    """

    def __init__(self, max_t_count=20):
        self.max_t_count = max_t_count
        self.gate_sequences = {}
        self.build_gate_table()

    def build_gate_table(self):
        """Build table of achievable rotations with their T-counts"""
        print("Building rotation synthesis table...")

        # Clifford Z-rotations
        clifford_z = [
            ('I', 0, I),
            ('Z', 0, Z),
            ('S', 0, S),
            ('S†', 0, np.conj(S).T),
        ]

        for name, tc, gate in clifford_z:
            phase = np.angle(gate[1, 1]) - np.angle(gate[0, 0])
            self.gate_sequences[name] = (phase, tc, gate)

        # Add T-based rotations
        current_gates = {
            'I': I,
            'T': T,
            'T†': T_dag,
            'S': S,
            'S†': np.conj(S).T,
        }

        # Generate more complex sequences
        for t_count in range(1, self.max_t_count + 1):
            new_gates = {}
            for name1, gate1 in current_gates.items():
                for name2, gate2 in [('T', T), ('T†', T_dag), ('H', H)]:
                    new_name = name2 + name1
                    new_gate = gate2 @ gate1

                    # Check if diagonal (Rz-like)
                    if np.abs(new_gate[0, 1]) < 1e-10 and np.abs(new_gate[1, 0]) < 1e-10:
                        phase = np.angle(new_gate[1, 1]) - np.angle(new_gate[0, 0])
                        tc = new_name.count('T') + new_name.count('T†')
                        if tc <= self.max_t_count:
                            if new_name not in self.gate_sequences:
                                self.gate_sequences[new_name] = (phase, tc, new_gate)
                                new_gates[new_name] = new_gate

            current_gates.update(new_gates)
            if t_count <= 5:
                print(f"  T-count {t_count}: {len(self.gate_sequences)} sequences")

    def synthesize(self, theta, epsilon=1e-3):
        """
        Find Clifford+T approximation for Rz(theta).

        Args:
            theta: Target rotation angle
            epsilon: Approximation tolerance

        Returns:
            (sequence_name, t_count, error, gate_matrix)
        """
        target_phase = theta  # Rz(theta) has phase difference theta
        target_phase = target_phase % (2 * np.pi)
        if target_phase > np.pi:
            target_phase -= 2 * np.pi

        best_match = None
        best_error = float('inf')
        best_tc = float('inf')

        for name, (phase, tc, gate) in self.gate_sequences.items():
            # Normalize phase
            phase_norm = phase % (2 * np.pi)
            if phase_norm > np.pi:
                phase_norm -= 2 * np.pi

            error = abs(phase_norm - target_phase)
            error = min(error, 2 * np.pi - error)  # Handle wraparound

            if error < epsilon:
                if tc < best_tc or (tc == best_tc and error < best_error):
                    best_match = name
                    best_error = error
                    best_tc = tc

        if best_match is None:
            return None, None, None, None

        return best_match, best_tc, best_error, self.gate_sequences[best_match][2]


class RUSCircuit:
    """
    Repeat-Until-Success circuit simulation for rotations.
    """

    def __init__(self, target_angle):
        self.target_angle = target_angle
        self.target_gate = Rz(target_angle)

    def basic_rus(self, ancilla_rotation):
        """
        Basic RUS circuit structure.

        Circuit:
        |ψ⟩ ───●───────●─── |ψ'⟩
               │       │
        |0⟩ ───X─Rz(φ)─X───H─M

        Returns success probability and achieved rotation on success.
        """
        phi = ancilla_rotation

        # Success probability (measurement outcome 0)
        # After the circuit, measuring 0 means we applied some rotation
        p_success = np.cos(phi / 2) ** 2

        # Rotation achieved on success
        achieved_rotation = 2 * phi  # Simplified model

        return p_success, achieved_rotation

    def analyze(self, t_count_per_attempt, success_prob):
        """Analyze expected T-count for RUS"""
        expected_attempts = 1 / success_prob
        expected_t_count = t_count_per_attempt * expected_attempts
        return expected_attempts, expected_t_count


def exact_synthesis_t_count(k):
    """T-count for exact Rz(pi/2^k) synthesis"""
    if k <= 2:
        return 0
    return k - 2


# Demonstrations
print("="*70)
print("T-Gate Synthesis Demonstrations")
print("="*70)

# Build synthesis table
synth = ZRotationSynthesis(max_t_count=12)

# Test cases
test_angles = [
    (np.pi / 4, "pi/4 (T gate)"),
    (np.pi / 8, "pi/8"),
    (np.pi / 5, "pi/5"),
    (np.pi / 3, "pi/3"),
    (0.123, "0.123 rad"),
    (1.0, "1.0 rad"),
]

print("\n" + "="*70)
print("Rotation Synthesis Results")
print("="*70)
print(f"{'Angle':<15} {'Name':<20} {'T-count':<10} {'Error':<15}")
print("-"*60)

for angle, name in test_angles:
    for eps in [1e-2, 1e-3]:
        seq, tc, err, gate = synth.synthesize(angle, epsilon=eps)
        if seq:
            print(f"{name:<15} {seq[:20]:<20} {tc:<10} {err:<15.2e}")

# Visualize achievable phases
print("\n" + "="*70)
print("Achievable Rotation Phases (T-count coloring)")
print("="*70)

phases = []
t_counts = []
for name, (phase, tc, gate) in synth.gate_sequences.items():
    if tc <= 8:
        phases.append(phase)
        t_counts.append(tc)

plt.figure(figsize=(12, 5))

# Plot 1: Phases on unit circle
ax1 = plt.subplot(121, projection='polar')
for phase, tc in zip(phases, t_counts):
    color = plt.cm.viridis(tc / 8)
    ax1.scatter(phase, 1, c=[color], s=20, alpha=0.6)
ax1.set_title('Achievable Phases\n(color = T-count)')
ax1.set_ylim(0, 1.2)

# Plot 2: Phase density histogram
ax2 = plt.subplot(122)
for tc in range(9):
    tc_phases = [p for p, t in zip(phases, t_counts) if t == tc]
    if tc_phases:
        ax2.hist(tc_phases, bins=50, alpha=0.5, label=f'T={tc}')
ax2.set_xlabel('Phase (radians)')
ax2.set_ylabel('Count')
ax2.set_title('Phase Distribution by T-count')
ax2.legend()

plt.tight_layout()
plt.savefig('synthesis_phases.png', dpi=150, bbox_inches='tight')
plt.show()

# RUS Analysis
print("\n" + "="*70)
print("RUS Circuit Analysis")
print("="*70)

rus = RUSCircuit(np.pi / 5)

# Compare deterministic vs RUS
det_t_count = 12  # Hypothetical deterministic T-count

print(f"\nTarget: Rz(pi/5)")
print(f"Deterministic T-count: {det_t_count}")
print("\nRUS variants:")
print(f"{'T/attempt':<12} {'Success Prob':<15} {'E[attempts]':<15} {'E[T-count]':<12}")
print("-"*55)

rus_configs = [
    (2, 0.6),
    (3, 0.75),
    (4, 0.85),
    (5, 0.9),
    (6, 0.95),
]

for t_per, p_succ in rus_configs:
    e_att, e_tc = rus.analyze(t_per, p_succ)
    marker = "***" if e_tc < det_t_count else ""
    print(f"{t_per:<12} {p_succ:<15.2f} {e_att:<15.2f} {e_tc:<12.2f} {marker}")

# Exact synthesis for special angles
print("\n" + "="*70)
print("Exact Synthesis for pi/2^k")
print("="*70)

for k in range(1, 10):
    angle = np.pi / (2 ** k)
    tc = exact_synthesis_t_count(k)
    print(f"Rz(pi/2^{k}) = Rz({np.degrees(angle):8.4f}°): T-count = {tc}")

# Multi-rotation optimization demo
print("\n" + "="*70)
print("Multi-Rotation Optimization")
print("="*70)

rotations = [np.pi/3, np.pi/6, np.pi/4]
epsilon = 1e-3

print(f"\nSequence: Rz(pi/3) * Rz(pi/6) * Rz(pi/4)")
print(f"Precision: epsilon = {epsilon}")

# Naive: synthesize each separately
naive_total = 0
print("\nNaive approach (synthesize each):")
for angle in rotations:
    seq, tc, err, _ = synth.synthesize(angle, epsilon=epsilon)
    if tc:
        print(f"  Rz({angle:.4f}): T-count = {tc}")
        naive_total += tc
    else:
        print(f"  Rz({angle:.4f}): Not found in table")
        naive_total += 15  # Estimate

print(f"  Total (naive): {naive_total} T-gates")

# Optimized: combine first
combined_angle = sum(rotations)
print(f"\nOptimized approach (combine angles):")
print(f"  Combined: {rotations[0]:.4f} + {rotations[1]:.4f} + {rotations[2]:.4f} = {combined_angle:.4f}")
print(f"  = {combined_angle/np.pi:.4f}*pi = {Fraction(combined_angle/np.pi).limit_denominator(100)}*pi")

seq, tc, err, _ = synth.synthesize(combined_angle, epsilon=epsilon)
if tc:
    print(f"  Rz({combined_angle:.4f}): T-count = {tc}")
else:
    # Check if exact
    if abs(combined_angle - 3*np.pi/4) < 1e-10:
        print(f"  Rz(3*pi/4) = T^3: T-count = 3 (EXACT!)")
        tc = 3

print(f"\nSavings: {naive_total} -> {tc} = {naive_total - tc} T-gates saved!")

# T-count scaling analysis
print("\n" + "="*70)
print("T-count Scaling with Precision")
print("="*70)

# Theoretical: ~3 log2(1/epsilon) for optimal synthesis
epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
theoretical_t_counts = [3 * np.log2(1/eps) for eps in epsilons]

print(f"{'Epsilon':<12} {'Theoretical T':<15} {'log2(1/eps)':<15}")
print("-"*45)
for eps, tc in zip(epsilons, theoretical_t_counts):
    print(f"{eps:<12.0e} {tc:<15.1f} {np.log2(1/eps):<15.1f}")

plt.figure(figsize=(8, 5))
plt.semilogx(epsilons, theoretical_t_counts, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Precision (epsilon)')
plt.ylabel('T-count')
plt.title('T-count Scaling: Optimal Synthesis\n(~3 log_2(1/epsilon))')
plt.grid(True, alpha=0.3)
plt.savefig('t_count_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("T-Gate Synthesis Lab Complete")
print("="*70)
```

---

### Summary

#### Key Formulas

| Concept | Formula |
|---------|---------|
| Optimal synthesis | T-count $\approx 3\log_2(1/\epsilon)$ |
| Exact angles | $R_z(m\pi/2^k)$ needs $k-2$ T-gates ($k \geq 3$) |
| Ring structure | Entries in $\mathbb{Z}[\omega, 1/\sqrt{2}]$, $\omega = e^{i\pi/4}$ |
| RUS expected T | $T_{\text{per attempt}} / p_{\text{success}}$ |
| Grid density | $\sim 1/\sqrt{2^k}$ at T-count $k$ |

#### Main Takeaways

1. **T-gate synthesis is the key compilation step** for fault-tolerant QC
2. **Gridsynth achieves optimal T-count** of $\approx 3\log_2(1/\epsilon)$
3. **Exact synthesis is possible** for angles $m\pi/2^k$
4. **RUS circuits trade qubits for T-gates** with probabilistic success
5. **Combining rotations before synthesis** can dramatically reduce T-count

---

### Daily Checklist

- [ ] Understand the synthesis problem and exactness conditions
- [ ] Can explain the Gridsynth algorithm at a high level
- [ ] Understand RUS circuits and their trade-offs
- [ ] Can estimate T-counts for common rotations
- [ ] Completed computational lab with synthesis implementations
- [ ] Worked through at least 2 practice problems per level

---

### Preview: Day 865

Tomorrow we study **Fault-Tolerant Circuit Compilation**---the full pipeline from quantum algorithms to physical implementations. We'll cover how to map logical gates to encoded operations, implement non-Clifford gates via code surgery and magic state injection, and optimize the resulting circuits for depth and resource efficiency.

---

*Day 864 provides the tools to synthesize individual gates---tomorrow we learn to compile complete fault-tolerant circuits.*
