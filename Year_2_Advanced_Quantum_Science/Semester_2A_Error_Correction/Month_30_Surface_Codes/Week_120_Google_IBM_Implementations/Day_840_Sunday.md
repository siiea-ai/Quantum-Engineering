# Day 840: Month 30 & Semester 2A Synthesis

## Week 120, Day 7 | Month 30: Surface Codes | Semester 2A: Quantum Error Correction

### Overview

Today marks the completion of Semester 2A: Quantum Error Correction. Over the past six months (Months 25-30), we have journeyed from the fundamental principles of quantum noise to state-of-the-art implementations achieving below-threshold operation. This synthesis day consolidates all key concepts, connects theoretical foundations to experimental reality, and prepares you for Semester 2B: Quantum Algorithms and Applications.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Semester review: Months 25-30 |
| **Afternoon** | 2.5 hours | Integration problems and connections |
| **Evening** | 1.5 hours | Forward look and reflection |

---

## Part 1: Semester 2A Complete Review

### Month 25: Quantum Noise and Decoherence

**Key Concepts Mastered:**

1. **Quantum Channels**
   $$\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger, \quad \sum_k E_k^\dagger E_k = I$$

2. **Pauli Channel Model**
   $$\mathcal{E}(\rho) = (1-p_x-p_y-p_z)\rho + p_x X\rho X + p_y Y\rho Y + p_z Z\rho Z$$

3. **Amplitude Damping** (T1 decay)
   $$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

4. **Phase Damping** (T2 dephasing)
   $$\rho \rightarrow \begin{pmatrix} \rho_{00} & \rho_{01}e^{-t/T_2} \\ \rho_{10}e^{-t/T_2} & \rho_{11} \end{pmatrix}$$

**Milestone:** Understanding that quantum information is fragile and requires active protection.

### Month 26: Classical Error Correction Foundations

**Key Concepts Mastered:**

1. **Linear Codes**
   $$C = \{c \in \mathbb{F}_2^n : Hc = 0\}$$

2. **Hamming Distance and Correction**
   $$d(C) = \min_{c_1 \neq c_2} d_H(c_1, c_2)$$
   $$\text{Corrects } t \text{ errors if } d \geq 2t + 1$$

3. **Syndrome Decoding**
   $$s = Hr^T = H(c + e)^T = He^T$$

4. **Shannon's Theorem**
   $$\boxed{R < C = 1 - H(p) \text{ achievable}}$$

**Milestone:** Classical codes provide the blueprint for quantum error correction.

### Month 27: Quantum Error Correction Basics

**Key Concepts Mastered:**

1. **Quantum Error Correction Conditions (Knill-Laflamme)**
   $$\boxed{\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = C_{ab} \delta_{ij}}$$

2. **9-Qubit Shor Code**
   $$|0_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)^{\otimes 3}$$

3. **7-Qubit Steane Code** (CSS construction)
   - Based on classical [7,4,3] Hamming code
   - Corrects arbitrary single-qubit errors

4. **5-Qubit Code** (Perfect code)
   - Smallest code to correct arbitrary errors
   - [[5,1,3]] parameters

**Milestone:** Quantum information CAN be protected despite no-cloning.

### Month 28: Stabilizer Formalism

**Key Concepts Mastered:**

1. **Pauli Group**
   $$\mathcal{P}_n = \{\pm 1, \pm i\} \times \{I, X, Y, Z\}^{\otimes n}$$

2. **Stabilizer Definition**
   $$|C\rangle = \{|\psi\rangle : S|\psi\rangle = |\psi\rangle, \forall S \in \mathcal{S}\}$$

3. **Syndrome Extraction**
   $$s_i = \begin{cases} 0 & \text{if } S_i E |\psi\rangle = E|\psi\rangle \\ 1 & \text{if } S_i E |\psi\rangle = -E|\psi\rangle \end{cases}$$

4. **Logical Operators**
   $$\bar{X}, \bar{Z} \in C(\mathcal{S}) \setminus \mathcal{S}$$
   - Commute with all stabilizers
   - Not in stabilizer group

**Milestone:** Stabilizers provide efficient description and manipulation of QEC codes.

### Month 29: Fault-Tolerant Quantum Computing

**Key Concepts Mastered:**

1. **Fault Tolerance Definition**
   - Single fault → at most one error per code block
   - Prevents error propagation

2. **Transversal Gates**
   $$\bar{U} = U^{\otimes n}$$
   - Naturally fault-tolerant
   - Limited gate set (Eastin-Knill theorem)

3. **Magic State Distillation**
   $$|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$
   - Enables universal computation
   - 15:1 distillation protocol

4. **Threshold Theorem**
   $$\boxed{p < p_{\text{th}} \Rightarrow \text{arbitrarily long computation possible}}$$

**Milestone:** Fault tolerance enables scalable quantum computation.

### Month 30: Surface Codes and Implementations

**Key Concepts Mastered:**

1. **Surface Code Stabilizers**
   $$A_v = \prod_{j \in v} X_j, \quad B_p = \prod_{j \in p} Z_j$$

2. **Threshold**
   $$\boxed{p_{\text{th}} \approx 1\% \text{ (circuit-level)}}$$

3. **Logical Error Scaling**
   $$p_L(d) \approx A\left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

4. **Error Suppression Factor**
   $$\lambda = \frac{p_L(d)}{p_L(d+2)} = 2.14 \pm 0.02 \text{ (Willow)}$$

5. **Decoding Algorithms**
   - Minimum Weight Perfect Matching (MWPM)
   - Union-Find (faster, near-optimal)
   - Neural network decoders

**Milestone:** Surface codes are the leading candidate for fault-tolerant QC.

---

## Part 2: Master Formula Sheet

### Quantum Channels and Noise

| Concept | Formula |
|---------|---------|
| Kraus representation | $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$ |
| Depolarizing channel | $\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$ |
| Amplitude damping | $\gamma = 1 - e^{-t/T_1}$ |
| Pure dephasing | $\rho_{01} \rightarrow \rho_{01} e^{-t/T_\phi}$ |
| T2 relation | $\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$ |

### Error Correction Codes

| Code | Parameters | Distance | Stabilizers |
|------|------------|----------|-------------|
| Shor | [[9,1,3]] | 3 | 8 |
| Steane | [[7,1,3]] | 3 | 6 |
| 5-qubit | [[5,1,3]] | 3 | 4 |
| Surface (d) | [[2d²-1,1,d]] | d | 2d²-2 |

### Stabilizer Formalism

| Concept | Formula |
|---------|---------|
| Stabilizer group | $\mathcal{S} = \langle S_1, ..., S_{n-k} \rangle$ |
| Code space dimension | $\dim(C) = 2^k = 2^{n-|\mathcal{S}|}$ |
| Syndrome | $s_i = \frac{1 - \langle S_i \rangle}{2}$ |
| Distance | $d = \min_{E \in C(\mathcal{S})\setminus\mathcal{S}} \text{weight}(E)$ |

### Fault Tolerance

| Concept | Formula |
|---------|---------|
| Threshold theorem | $p < p_{\text{th}} \Rightarrow p_L \rightarrow 0$ as $d \rightarrow \infty$ |
| Level-k error | $p^{(k)} \leq (p/p_{\text{th}})^{2^k}$ |
| Magic state overhead | $\sim 15^k$ for k levels of distillation |

### Surface Codes

| Concept | Formula |
|---------|---------|
| Physical qubits | $n = 2d^2 - 1$ |
| Logical error rate | $p_L \approx A(p/p_{\text{th}})^{(d+1)/2}$ |
| Error suppression | $\lambda = p_{\text{th}}/p$ |
| Willow result | $\lambda = 2.14 \pm 0.02$ |

---

## Part 3: Integration Problems

### Problem Set: Connecting Concepts

**Problem 1: From Noise to Threshold**

A qubit has $T_1 = 50$ μs and $T_2 = 30$ μs. The syndrome extraction cycle takes 1 μs with 4 CZ gates (0.3% error each) and measurement (0.5% error).

a) Calculate the total error per qubit per cycle
b) Is this below the ~1% surface code threshold?
c) What code distance is needed for $p_L < 10^{-6}$?

**Solution:**

a) **Error budget:**
- CZ contribution: $4 \times 0.003 = 0.012$
- Measurement: $0.005$
- T1 decay: $1\text{ μs}/50\text{ μs} = 0.02$
- T2 dephasing: $1\text{ μs}/30\text{ μs} - 0.5 \times 0.02 = 0.023$ (approximate)

Total: $p \approx 0.012 + 0.005 + 0.02 + 0.023 = 0.06 = 6\%$

b) **Above threshold!** Need improvement to operate below 1%.

c) Cannot achieve fault tolerance with current parameters. Need:
- Longer coherence: $T_1 > 200$ μs, $T_2 > 100$ μs
- Better gates: < 0.1% error
- Faster cycles: < 0.5 μs

---

**Problem 2: Stabilizer to Physical**

The 5-qubit code has stabilizers:
$$S_1 = XZZXI, \quad S_2 = IXZZX, \quad S_3 = XIXZZ, \quad S_4 = ZXIXZ$$

a) Verify these commute pairwise
b) Find the logical $\bar{X}$ and $\bar{Z}$ operators
c) Design a syndrome extraction circuit for $S_1$

**Solution:**

a) **Commutativity check for $S_1$ and $S_2$:**
- Positions where both are non-identity: 2, 3, 4
- Position 2: Z·X (anticommute)
- Position 3: Z·Z (commute)
- Position 4: X·Z (anticommute)
- Total anticommutations: 2 (even) → commute ✓

b) **Logical operators:**
- $\bar{X} = XXXXX$ (anticommutes with $Z$ stabilizers, commutes with $X$)
- $\bar{Z} = ZZZZZ$ (anticommutes with $X$ stabilizers, commutes with $Z$)

c) **Syndrome circuit for $S_1 = XZZXI$:**
```
ancilla: |0⟩ ─H─●──●──●──●──H─ Measure
              │  │  │  │
q1: ─────────X──┼──┼──┼──────
q2: ────────────Z──┼──┼──────
q3: ───────────────Z──┼──────
q4: ──────────────────X──────
q5: ─────────────────────────
```

---

**Problem 3: Decoding and Correction**

A distance-5 surface code experiences the following syndrome: X stabilizers show errors in a line pattern suggesting a Z error chain of length 2.

a) What is the minimum weight correction?
b) If the decoder chooses the wrong pairing, what logical error occurs?
c) Calculate the probability of decoder failure if physical error rate is 0.5%.

**Solution:**

a) **Minimum weight correction:**
Two Z errors in a chain → apply $Z_i Z_j$ correction where $i, j$ are the data qubits on the error chain.

b) **Wrong pairing scenario:**
If the decoder pairs syndromes with a different path that spans the code, a logical $\bar{Z}$ error occurs.

c) **Failure probability:**
For MWPM decoder at 0.5% error with d=5:
$$p_L \approx A(0.005/0.01)^3 = A \times 0.125$$

With typical $A \approx 0.1$:
$$p_L \approx 1.25\% \text{ per round}$$

---

**Problem 4: End-to-End Design**

Design a fault-tolerant system to run 1000 T gates with 99% success probability.

a) What logical error rate per T gate is needed?
b) If magic state distillation has 90% yield, how many raw T states are needed?
c) For λ = 2.5, what code distance achieves this?

**Solution:**

a) **Per-gate error budget:**
$$p_{\text{gate}} < \frac{1 - 0.99}{1000} = 10^{-5}$$

b) **Raw T states:**
With 90% yield per distillation (15:1 protocol):
$$N_{\text{raw}} = 1000 \times \frac{15}{0.9} \approx 16,700$$

c) **Code distance:**
Using $p_L(d) = 10^{-3} \times (1/2.5)^{(d-7)/2}$:
$$10^{-5} = 10^{-3} \times (0.4)^{(d-7)/2}$$
$$(0.4)^{(d-7)/2} = 0.01$$
$$\frac{d-7}{2} = \frac{\ln(0.01)}{\ln(0.4)} = 5.0$$
$$d = 17$$

---

## Part 4: Comprehensive Synthesis Problems

### Synthesis Problem A: Full Stack Analysis

**Scenario:** You are tasked with evaluating whether a new quantum processor can achieve below-threshold operation.

**Given specifications:**
- 127 qubits in heavy-hex layout
- CZ gate fidelity: 99.6%
- Single-qubit gate fidelity: 99.95%
- T1 = 200 μs, T2 = 150 μs
- Measurement fidelity: 99.2%
- Syndrome cycle time: 2 μs

**Questions:**

1. Calculate the effective error per syndrome cycle
2. Estimate the threshold margin (if any)
3. What code distances are achievable?
4. Compare to Google Willow's performance
5. Recommend improvements for below-threshold operation

### Synthesis Problem B: Algorithm-Hardware Matching

**Scenario:** Match these algorithms to optimal platforms:

| Algorithm | Logical Qubits | T Gates | Connectivity Needs |
|-----------|---------------|---------|-------------------|
| VQE chemistry | 50 | $10^4$ | High |
| Shor factoring | 2000 | $10^9$ | Medium |
| QAOA optimization | 100 | $10^6$ | Problem-dependent |
| Quantum simulation | 500 | $10^{10}$ | Local |

**Available platforms:**
- A: Superconducting (1000 qubits, λ=2.1, nearest-neighbor)
- B: Trapped ion (50 qubits, λ=1.5, all-to-all)
- C: Neutral atom (500 qubits, λ=1.3, reconfigurable)

Match each algorithm to the best platform and justify.

### Synthesis Problem C: Timeline Projection

**Scenario:** Project when each milestone can be achieved:

1. First useful quantum chemistry calculation (100 logical qubits, $p_L < 10^{-4}$)
2. Breaking RSA-2048 (4000 logical qubits, $p_L < 10^{-12}$)
3. Quantum advantage in optimization (500 logical qubits, $p_L < 10^{-6}$)

**Assumptions:**
- Current (2024): λ = 2.14, 100 physical qubits
- Physical qubits double every 2 years
- λ improves by 0.3 per year
- Cost reduces 30% per year

---

## Part 5: Semester 2A Final Assessment

### Written Examination Topics

1. **Quantum Noise Theory** (20%)
   - Channel representations
   - Coherence times and their relation
   - Noise models for different platforms

2. **Error Correction Codes** (25%)
   - CSS construction
   - Stabilizer formalism
   - Code parameters and properties

3. **Fault Tolerance** (25%)
   - Threshold theorem
   - Transversal gates
   - Magic state distillation

4. **Surface Codes** (30%)
   - Stabilizers and boundaries
   - Decoding algorithms
   - Real implementations

### Practical Skills Checklist

- [ ] Derive Kraus operators for common channels
- [ ] Construct stabilizer generators for standard codes
- [ ] Design fault-tolerant syndrome extraction circuits
- [ ] Implement MWPM decoder in Python
- [ ] Calculate resource estimates for algorithms
- [ ] Analyze experimental QEC results
- [ ] Compare platforms quantitatively

---

## Part 6: Forward Look - Semester 2B Preview

### Semester 2B: Quantum Algorithms and Applications

**Month 31: Quantum Algorithm Foundations**
- Query complexity and oracles
- Deutsch-Jozsa, Bernstein-Vazirani
- Simon's algorithm
- Phase kickback

**Month 32: Quantum Fourier Transform**
- QFT circuit construction
- Phase estimation
- Order finding
- Applications to number theory

**Month 33: Grover's Algorithm and Search**
- Amplitude amplification
- Optimal query complexity
- Applications to NP problems
- Quantum walks

**Month 34: Shor's Algorithm**
- Factoring and discrete log
- Period finding reduction
- Implementation considerations
- Post-quantum cryptography

**Month 35: Variational Algorithms**
- VQE and QAOA
- Barren plateaus
- Classical optimization
- Near-term applications

**Month 36: Quantum Machine Learning**
- Quantum kernels
- Quantum neural networks
- Quantum advantage in ML
- Practical implementations

---

## Part 7: Computational Lab - Semester Review Tool

```python
"""
Day 840 Computational Lab: Semester 2A Complete Review Tool
Comprehensive assessment and synthesis of quantum error correction
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict

# =============================================================================
# Part 1: Core Concepts Review
# =============================================================================

class QuantumChannel:
    """Base class for quantum channels."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, rho: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class DepolarizingChannel(QuantumChannel):
    """Depolarizing channel with error probability p."""

    def __init__(self, p: float):
        super().__init__(f"Depolarizing(p={p})")
        self.p = p
        self.paulis = [
            np.eye(2),  # I
            np.array([[0, 1], [1, 0]]),  # X
            np.array([[0, -1j], [1j, 0]]),  # Y
            np.array([[1, 0], [0, -1]])  # Z
        ]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        result = (1 - self.p) * rho
        for P in self.paulis[1:]:
            result += (self.p / 3) * P @ rho @ P.conj().T
        return result

class AmplitudeDampingChannel(QuantumChannel):
    """Amplitude damping channel with parameter gamma."""

    def __init__(self, gamma: float):
        super().__init__(f"AmplitudeDamping(γ={gamma})")
        self.gamma = gamma
        self.E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        self.E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])

    def apply(self, rho: np.ndarray) -> np.ndarray:
        return self.E0 @ rho @ self.E0.conj().T + self.E1 @ rho @ self.E1.conj().T

# =============================================================================
# Part 2: Stabilizer Code Library
# =============================================================================

@dataclass
class StabilizerCode:
    """Representation of a stabilizer code."""
    name: str
    n: int  # physical qubits
    k: int  # logical qubits
    d: int  # distance
    stabilizers: List[str]  # Pauli string representation
    logical_x: List[str]
    logical_z: List[str]

# Standard codes
CODES = {
    '3-bit repetition': StabilizerCode(
        name='3-bit repetition',
        n=3, k=1, d=3,
        stabilizers=['ZZI', 'IZZ'],
        logical_x=['XXX'],
        logical_z=['ZII']
    ),
    'Steane [[7,1,3]]': StabilizerCode(
        name='Steane',
        n=7, k=1, d=3,
        stabilizers=['IIIXXXX', 'IXXIIXX', 'XIXIXIX', 'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ'],
        logical_x=['XXXXXXX'],
        logical_z=['ZZZZZZZ']
    ),
    '5-qubit [[5,1,3]]': StabilizerCode(
        name='5-qubit',
        n=5, k=1, d=3,
        stabilizers=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'],
        logical_x=['XXXXX'],
        logical_z=['ZZZZZ']
    ),
    'Surface d=3': StabilizerCode(
        name='Surface d=3',
        n=17, k=1, d=3,
        stabilizers=[
            'XXXX.....',  # Plaquette operators (simplified)
            'ZZZZ.....',  # Vertex operators (simplified)
        ],
        logical_x=['X chain'],
        logical_z=['Z chain']
    )
}

def print_code_summary():
    """Print summary of all codes."""
    print("=" * 60)
    print("STABILIZER CODE LIBRARY")
    print("=" * 60)
    print(f"{'Code':<25} {'[[n,k,d]]':<12} {'Stabilizers':<12}")
    print("-" * 60)
    for name, code in CODES.items():
        params = f"[[{code.n},{code.k},{code.d}]]"
        print(f"{name:<25} {params:<12} {len(code.stabilizers):<12}")

print_code_summary()

# =============================================================================
# Part 3: Threshold Analysis
# =============================================================================

def logical_error_rate(p_physical: float, d: int, lambda_val: float,
                       p_ref: float = 0.01, d_ref: int = 7) -> float:
    """Calculate logical error rate for given parameters."""
    if p_physical >= p_ref:
        # Above threshold
        return min(1.0, p_physical * (p_physical / p_ref) ** ((d - d_ref) / 2))
    else:
        # Below threshold
        return p_ref * (1 / lambda_val) ** ((d - d_ref) / 2)

def find_threshold_distance(p_target: float, p_physical: float,
                            lambda_val: float) -> int:
    """Find minimum distance for target error rate."""
    for d in range(3, 101, 2):
        if logical_error_rate(p_physical, d, lambda_val) <= p_target:
            return d
    return -1  # Not achievable

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

# Analyze different scenarios
scenarios = [
    ("Google Willow", 0.0047, 2.14),
    ("IBM Heron (est.)", 0.004, 1.75),
    ("Future target", 0.003, 3.0),
]

print(f"\n{'Scenario':<20} {'p_phys':<10} {'λ':<8} {'d for 10^-6':<12} {'d for 10^-10':<12}")
print("-" * 65)

for name, p, lam in scenarios:
    d6 = find_threshold_distance(1e-6, p, lam)
    d10 = find_threshold_distance(1e-10, p, lam)
    print(f"{name:<20} {p:.4f}{'':<4} {lam:<8} {d6:<12} {d10:<12}")

# =============================================================================
# Part 4: Complete Error Budget Calculator
# =============================================================================

@dataclass
class ErrorBudget:
    """Complete error budget for QEC system."""
    two_qubit_error: float
    single_qubit_error: float
    measurement_error: float
    idle_error: float
    t1_us: float
    t2_us: float
    cycle_time_us: float

    def total_error_per_cycle(self) -> float:
        """Calculate total error per syndrome cycle."""
        # Gate contributions (4 CZ, 2 SQ per cycle)
        gate_error = 4 * self.two_qubit_error + 2 * self.single_qubit_error

        # Coherence contribution
        t1_error = self.cycle_time_us / self.t1_us
        t2_error = self.cycle_time_us / self.t2_us

        return gate_error + self.measurement_error + t1_error + t2_error

    def is_below_threshold(self, threshold: float = 0.01) -> bool:
        """Check if operating below threshold."""
        return self.total_error_per_cycle() < threshold

    def breakdown(self) -> Dict[str, float]:
        """Return breakdown of error contributions."""
        total = self.total_error_per_cycle()
        return {
            '2Q gates': 4 * self.two_qubit_error / total,
            '1Q gates': 2 * self.single_qubit_error / total,
            'Measurement': self.measurement_error / total,
            'T1 decay': (self.cycle_time_us / self.t1_us) / total,
            'T2 dephasing': (self.cycle_time_us / self.t2_us) / total,
        }

# Example systems
willow = ErrorBudget(
    two_qubit_error=0.0025,
    single_qubit_error=0.00035,
    measurement_error=0.007,
    idle_error=0.001,
    t1_us=68,
    t2_us=30,
    cycle_time_us=1.0
)

print("\n" + "=" * 60)
print("ERROR BUDGET: GOOGLE WILLOW")
print("=" * 60)
print(f"Total error per cycle: {willow.total_error_per_cycle()*100:.2f}%")
print(f"Below 1% threshold: {willow.is_below_threshold()}")
print("\nBreakdown:")
for source, fraction in willow.breakdown().items():
    print(f"  {source}: {fraction*100:.1f}%")

# =============================================================================
# Part 5: Semester Learning Verification
# =============================================================================

def semester_quiz():
    """Interactive quiz covering all semester topics."""

    questions = [
        {
            'topic': 'Quantum Channels',
            'question': 'What is the relationship between T1 and T2 coherence times?',
            'answer': 'T2 ≤ 2*T1 (equality only with pure dephasing)'
        },
        {
            'topic': 'Stabilizer Codes',
            'question': 'How many stabilizer generators does an [[n,k,d]] code have?',
            'answer': 'n - k generators'
        },
        {
            'topic': 'Fault Tolerance',
            'question': 'What is the threshold for surface codes?',
            'answer': 'Approximately 1% for circuit-level noise'
        },
        {
            'topic': 'Surface Codes',
            'question': 'How many physical qubits for a distance-d surface code?',
            'answer': '2d² - 1 physical qubits'
        },
        {
            'topic': 'Implementation',
            'question': 'What error suppression factor did Google Willow achieve?',
            'answer': 'λ = 2.14 ± 0.02'
        }
    ]

    print("\n" + "=" * 60)
    print("SEMESTER 2A KNOWLEDGE CHECK")
    print("=" * 60)

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['topic']}]")
        print(f"   Q: {q['question']}")
        print(f"   A: {q['answer']}")

semester_quiz()

# =============================================================================
# Part 6: Visualization - Semester Overview
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Quantum channel effects
ax1 = axes[0, 0]
t_range = np.linspace(0, 5, 100)
T1, T2 = 2, 1.5

coherence_t1 = np.exp(-t_range / T1)
coherence_t2 = np.exp(-t_range / T2)

ax1.plot(t_range, coherence_t1, 'b-', linewidth=2, label='Population (T1)')
ax1.plot(t_range, coherence_t2, 'r--', linewidth=2, label='Coherence (T2)')
ax1.set_xlabel('Time (arb.)', fontsize=10)
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.set_title('Month 25: Decoherence', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Code parameters
ax2 = axes[0, 1]
codes_data = [
    ('Rep 3', 3, 1, 3),
    ('5-qubit', 5, 1, 3),
    ('Steane', 7, 1, 3),
    ('Shor', 9, 1, 3),
    ('Surf d=3', 17, 1, 3),
    ('Surf d=5', 49, 1, 5),
    ('Surf d=7', 97, 1, 7),
]

names = [c[0] for c in codes_data]
n_vals = [c[1] for c in codes_data]
d_vals = [c[3] for c in codes_data]

x = np.arange(len(names))
ax2.bar(x - 0.2, n_vals, 0.4, label='n (physical)', color='steelblue')
ax2.bar(x + 0.2, d_vals, 0.4, label='d (distance)', color='coral')
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=45, ha='right')
ax2.set_ylabel('Value', fontsize=10)
ax2.set_title('Months 27-28: Code Parameters', fontsize=12)
ax2.legend()

# Plot 3: Threshold behavior
ax3 = axes[0, 2]
p_range = np.linspace(0.001, 0.02, 50)

for d in [5, 7, 9, 11]:
    p_L = [(p / 0.01) ** ((d + 1) / 2) if p < 0.01 else 1 for p in p_range]
    ax3.semilogy(p_range * 100, p_L, label=f'd={d}')

ax3.axvline(x=1, color='red', linestyle='--', label='Threshold')
ax3.set_xlabel('Physical Error Rate (%)', fontsize=10)
ax3.set_ylabel('Logical Error Rate', fontsize=10)
ax3.set_title('Month 29: Threshold Behavior', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Surface code layout
ax4 = axes[1, 0]
d = 3
for i in range(d):
    for j in range(d):
        ax4.scatter(i, j, s=200, c='blue', marker='o', edgecolors='black', zorder=3)

for i in range(d + 1):
    for j in range(d + 1):
        if (i + j) % 2 == 0 and 0 < i < d and 0 < j < d:
            ax4.scatter(i - 0.5, j - 0.5, s=150, c='red', marker='s',
                       edgecolors='black', zorder=3, alpha=0.7)
        elif (i + j) % 2 == 1 and not (i == 0 or i == d or j == 0 or j == d):
            ax4.scatter(i - 0.5, j - 0.5, s=150, c='green', marker='s',
                       edgecolors='black', zorder=3, alpha=0.7)

ax4.set_xlim(-0.5, d - 0.5)
ax4.set_ylim(-0.5, d - 0.5)
ax4.set_aspect('equal')
ax4.set_title('Month 30: Surface Code d=3', fontsize=12)
ax4.set_xlabel('X', fontsize=10)
ax4.set_ylabel('Y', fontsize=10)

# Plot 5: Implementation comparison
ax5 = axes[1, 1]
platforms = ['Google\nWillow', 'IBM\nHeron', 'Quantinuum\nH2', 'QuEra']
lambdas = [2.14, 1.8, 1.5, 1.3]
colors = ['#4285f4', '#0f62fe', '#00758f', '#9333ea']

bars = ax5.bar(platforms, lambdas, color=colors, edgecolor='black', linewidth=2)
ax5.axhline(y=1, color='red', linestyle='--', linewidth=2)
ax5.set_ylabel('Error Suppression λ', fontsize=10)
ax5.set_title('Week 120: Platform Comparison', fontsize=12)

for bar, val in zip(bars, lambdas):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val}', ha='center', va='bottom', fontsize=10)

# Plot 6: Scaling roadmap
ax6 = axes[1, 2]
years = np.arange(2024, 2036)
optimistic = 100 * 2 ** ((years - 2024) / 1.5)
conservative = 100 * 2 ** ((years - 2024) / 2.5)

ax6.semilogy(years, optimistic, 'g-', linewidth=2, label='Optimistic')
ax6.semilogy(years, conservative, 'orange', linewidth=2, label='Conservative')
ax6.axhline(y=1e6, color='red', linestyle='--', alpha=0.5)
ax6.text(2024.5, 1.2e6, '1M qubits', fontsize=9)

ax6.set_xlabel('Year', fontsize=10)
ax6.set_ylabel('Physical Qubits', fontsize=10)
ax6.set_title('Future: Scaling Roadmap', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_840_semester_synthesis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Visualization saved to: day_840_semester_synthesis.png")
print("=" * 60)

# =============================================================================
# Part 7: Semester Certificate
# =============================================================================

certificate = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    QUANTUM ENGINEERING CURRICULUM                            ║
║                                                                              ║
║                        CERTIFICATE OF COMPLETION                             ║
║                                                                              ║
║                    ═══════════════════════════════════                       ║
║                                                                              ║
║                          SEMESTER 2A                                         ║
║                    QUANTUM ERROR CORRECTION                                  ║
║                                                                              ║
║                    ═══════════════════════════════════                       ║
║                                                                              ║
║    This certifies completion of 6 months (180 days) of intensive study      ║
║    in quantum error correction, covering:                                    ║
║                                                                              ║
║    • Month 25: Quantum Noise and Decoherence                                ║
║    • Month 26: Classical Error Correction Foundations                        ║
║    • Month 27: Quantum Error Correction Basics                               ║
║    • Month 28: Stabilizer Formalism                                          ║
║    • Month 29: Fault-Tolerant Quantum Computing                             ║
║    • Month 30: Surface Codes and Real Implementations                        ║
║                                                                              ║
║    Key Achievements:                                                         ║
║    ✓ Mastered stabilizer formalism and code construction                    ║
║    ✓ Understood threshold theorem and fault tolerance                        ║
║    ✓ Analyzed state-of-the-art QEC implementations                          ║
║    ✓ Designed fault-tolerant quantum computing systems                       ║
║    ✓ Completed capstone project                                              ║
║                                                                              ║
║    Total Study Hours: ~1,260 hours (7 hours/day × 180 days)                 ║
║                                                                              ║
║    Completion Date: Day 840                                                  ║
║                                                                              ║
║                         [QUANTUM ENGINEERING PROGRAM]                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

print(certificate)
```

---

## Summary: Semester 2A Complete

### What You Have Mastered

| Month | Topic | Key Achievement |
|-------|-------|-----------------|
| 25 | Quantum Noise | Understanding decoherence mechanisms |
| 26 | Classical EC | Foundations for quantum codes |
| 27 | QEC Basics | Knill-Laflamme conditions, first codes |
| 28 | Stabilizers | Efficient code description and manipulation |
| 29 | Fault Tolerance | Threshold theorem, magic states |
| 30 | Surface Codes | State-of-the-art implementations |

### Core Competencies Achieved

1. **Theoretical Foundation**
   - Quantum channel formalism
   - Stabilizer theory
   - Fault tolerance requirements

2. **Practical Skills**
   - Code design and analysis
   - Decoder implementation
   - Resource estimation

3. **Research Awareness**
   - Current experimental status
   - Company roadmaps
   - Open problems

### Final Reflection Questions

1. What was the most surprising concept you learned this semester?
2. How has your understanding of quantum computing's challenges evolved?
3. What aspects would you like to explore further in research?
4. How do you see QEC evolving over the next decade?

---

## Preview: Semester 2B

**Quantum Algorithms and Applications**

With error correction mastered, Semester 2B focuses on what we can compute:

- **Query algorithms** (Deutsch-Jozsa, Simon)
- **Quantum Fourier Transform** and applications
- **Grover's search** and amplitude amplification
- **Shor's algorithm** for factoring
- **Variational algorithms** (VQE, QAOA)
- **Quantum machine learning**

The tools from Semester 2A ensure these algorithms can be run reliably at scale.

---

**Congratulations on completing Semester 2A: Quantum Error Correction!**

*You have mastered one of the most challenging and important topics in quantum information science. The path to fault-tolerant quantum computing is now clear, and you have the knowledge to contribute to making it a reality.*
