# Day 836: IBM Heavy-Hex Surface Codes

## Week 120, Day 3 | Month 30: Surface Codes | Semester 2A: Quantum Error Correction

### Overview

Today we examine IBM's distinctive approach to surface code implementation using the heavy-hex lattice geometry. Unlike Google's 4-connectivity grid, IBM uses 3-connectivity (degree-3 qubits) with flag qubit protocols to achieve fault tolerance. This architecture, deployed in processors like Heron, represents a fundamentally different engineering philosophy that offers advantages in fabrication yield and crosstalk mitigation.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Heavy-hex geometry and theory |
| **Afternoon** | 2.5 hours | Flag qubit protocols |
| **Evening** | 1.5 hours | Computational lab: Architecture comparison |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Describe the heavy-hex lattice** - Understand the geometry and qubit connectivity
2. **Explain the 3-connectivity constraint** - Analyze why IBM chose this architecture
3. **Implement flag qubit protocols** - Design fault-tolerant syndrome extraction with flags
4. **Compare connectivity requirements** - Evaluate trade-offs vs. 4-connectivity
5. **Analyze IBM processor evolution** - Trace the roadmap from Falcon to Heron
6. **Evaluate manufacturing advantages** - Understand yield and crosstalk benefits

---

## Core Content

### 1. The Heavy-Hex Lattice

#### 1.1 Geometric Structure

The heavy-hex lattice is derived from a hexagonal (honeycomb) lattice by adding qubits to each edge:

```
Standard Hexagonal:          Heavy-Hex:

    ●───●───●                    ●─○─●─○─●
   / \ / \ / \                  /│  │\│  │\
  ●───●───●───●                ○ ●─○─●─○─● ○
   \ / \ / \ /                  \│  │/│  │/
    ●───●───●                    ●─○─●─○─●

    ● = vertex qubit            ● = data qubit
                                ○ = auxiliary (measure) qubit
```

**Key Properties:**
- All qubits have degree 2 or 3 (maximum 3 connections)
- Data qubits sit at hexagon vertices
- Measure qubits sit on edges
- Regular, tileable pattern

#### 1.2 Why Degree-3?

**Manufacturing Advantages:**

1. **Reduced crosstalk**: Fewer neighbors = less unwanted coupling
2. **Higher yield**: Simpler wiring = fewer fabrication defects
3. **Better frequency allocation**: Easier to avoid collisions with fewer neighbors

**The Trade-off:**
Standard surface codes require degree-4 data qubits (each touches 4 stabilizers). With degree-3, we need additional protocols to achieve fault tolerance.

### 2. Surface Codes on Heavy-Hex

#### 2.1 Stabilizer Mapping

On a standard square lattice:
- Each X stabilizer: 4 data qubits (plaquette)
- Each Z stabilizer: 4 data qubits (vertex)

On heavy-hex, stabilizers must be adapted:

$$\boxed{\text{Heavy-hex stabilizers have weight 2-4 depending on position}}$$

**Boundary Effects:**
- Bulk stabilizers: weight 4
- Edge stabilizers: weight 2-3
- The lattice geometry creates a natural pattern

#### 2.2 Code Distance Constraints

For a heavy-hex lattice with $n$ columns and $m$ rows:

$$d_X = \lfloor (m+1)/2 \rfloor$$
$$d_Z = \lfloor (n+1)/2 \rfloor$$

The minimum distance:
$$\boxed{d = \min(d_X, d_Z)}$$

### 3. Flag Qubit Fault Tolerance

#### 3.1 The Problem with Low Connectivity

In standard syndrome extraction, a single fault (error on ancilla) can propagate to multiple data qubits:

```
Without flags (dangerous):

     ┌───┐
q1 ──┤   ├── (error here can spread to q1, q2)
     │ H │
q2 ──┤   ├──
     └───┘
```

With 4-connectivity, circuit scheduling can prevent dangerous propagation. With 3-connectivity, we need flag qubits.

#### 3.2 Flag Qubit Protocol

A flag qubit detects if an error has spread:

```
With flag (safe):

         ┌───┐     ┌───┐
q1 ─────┤   ├─────┤   ├─────
        │   │     │   │
flag ───┼───┼──■──┼───┼──■── (measures 0 if no spreading)
        │   │  │  │   │  │
q2 ─────┤   ├──●──┤   ├──●──
        └───┘     └───┘
```

**Flag Mechanism:**
1. Flag qubit starts in $|0\rangle$
2. CNOT gates interleave with stabilizer measurement
3. If error spreads, flag measures $|1\rangle$
4. Flagged measurements trigger special decoder handling

#### 3.3 Mathematical Framework

For a weight-$w$ stabilizer $S = P_1 P_2 \cdots P_w$ (each $P_i$ is X or Z):

**Standard extraction circuit depth:**
$$D_{\text{standard}} = w + 2 \text{ (for degree-4)}$$

**Flagged extraction:**
$$D_{\text{flagged}} = 2w + O(1)$$

The overhead is compensated by:
- Lower crosstalk error
- Better gate fidelities from simpler layout

### 4. IBM Processor Architecture

#### 4.1 Evolution of IBM Processors

| Processor | Year | Qubits | Connectivity | QV | Surface Code Support |
|-----------|------|--------|--------------|-----|----------------------|
| Falcon | 2020 | 27 | Heavy-hex | 64 | d=3 (limited) |
| Hummingbird | 2021 | 65 | Heavy-hex | 128 | d=5 |
| Eagle | 2022 | 127 | Heavy-hex | 256 | d=5-7 |
| Osprey | 2022 | 433 | Heavy-hex | - | d=7-9 |
| Condor | 2023 | 1121 | Heavy-hex | - | d=11+ |
| Heron | 2024 | 133 | Heavy-hex + | - | Optimized for QEC |

#### 4.2 Heron Processor Specifics

Heron represents IBM's latest architecture optimized for error correction:

**Key Specifications:**
- 133 qubits in heavy-hex arrangement
- Tunable couplers (like Google)
- Median 2Q gate error: 0.3%
- Median T1: 300 μs
- Median T2: 200 μs

**Architecture Improvements:**
- Reduced crosstalk via coupler isolation
- Improved frequency targeting
- Enhanced readout fidelity

#### 4.3 Gate Implementation

IBM primarily uses echoed cross-resonance (ECR) gates:

$$\boxed{ECR = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 0 & -i \\ 0 & 1 & -i & 0 \\ 0 & -i & 1 & 0 \\ -i & 0 & 0 & 1 \end{pmatrix}}$$

**Cross-Resonance Mechanism:**
1. Drive control qubit at target frequency
2. Creates conditional rotation on target
3. Echo pulses cancel unwanted terms

Gate time: ~300-500 ns (longer than Google's ~25 ns)

### 5. Comparative Analysis

#### 5.1 Google vs. IBM Architecture

| Feature | Google (Willow) | IBM (Heron) |
|---------|-----------------|-------------|
| Connectivity | Degree-4 grid | Degree-3 heavy-hex |
| 2Q Gate | CZ (25 ns) | ECR (400 ns) |
| 2Q Gate Error | 0.25% | 0.3% |
| Requires flags | No | Yes |
| Crosstalk | Higher | Lower |
| T1 | 68 μs | 300 μs |
| Fabrication yield | Lower | Higher |

#### 5.2 Effective Threshold Comparison

Theoretical thresholds:
- Square lattice (Google): $p_{\text{th}} \approx 1.0\%$
- Heavy-hex with flags (IBM): $p_{\text{th}} \approx 0.5-0.7\%$

The lower threshold for heavy-hex is compensated by:
1. Higher coherence times
2. Lower crosstalk errors
3. Better gate fidelities possible with simpler layout

#### 5.3 Circuit Depth Overhead

For one syndrome extraction round:

**Google (square lattice):**
$$D_{\text{Google}} = 4 \text{ CZ layers} + \text{measurement} \approx 8 \text{ time steps}$$

**IBM (heavy-hex with flags):**
$$D_{\text{IBM}} = 8 \text{ ECR layers} + \text{flags} + \text{measurement} \approx 16 \text{ time steps}$$

But IBM's longer T1 compensates:
$$\frac{T_{\text{cycle}}^{\text{IBM}}}{T_1^{\text{IBM}}} \approx \frac{T_{\text{cycle}}^{\text{Google}}}{T_1^{\text{Google}}}$$

### 6. Flag-Based Decoder Modifications

#### 6.1 Decoder Integration

Standard MWPM must be modified for flag information:

1. **Flag = 0**: Process syndrome normally
2. **Flag = 1**:
   - Possible high-weight error
   - Modify edge weights in matching graph
   - Consider additional error mechanisms

#### 6.2 Effective Error Model

With flags, the effective error model becomes:

$$p_{\text{eff}} = p_{\text{gate}} + p_{\text{flag}} \cdot p_{\text{spread}}$$

where:
- $p_{\text{gate}}$ = base gate error
- $p_{\text{flag}}$ = probability flag fails to detect
- $p_{\text{spread}}$ = probability of error spreading

Well-designed flag circuits achieve $p_{\text{flag}} \cdot p_{\text{spread}} \ll p_{\text{gate}}$.

---

## Worked Examples

### Example 1: Heavy-Hex Qubit Count

**Problem:** Calculate the number of data and measure qubits in a heavy-hex lattice with 5 columns and 4 rows of hexagonal cells.

**Solution:**

For a heavy-hex lattice:
- **Vertex qubits (data)**: Each hexagon has 6 vertices, but they're shared
- **Edge qubits (measure)**: Each hexagon has 6 edges, partially shared

For $n_c$ columns and $n_r$ rows:

**Data qubits (vertices):**
$$N_{\text{data}} = (2n_c + 1)(n_r + 1) + n_c \cdot n_r$$

For $n_c = 5$, $n_r = 4$:
$$N_{\text{data}} = (2 \times 5 + 1)(4 + 1) + 5 \times 4 = 11 \times 5 + 20 = 55 + 20 = 75$$

Wait, let me recalculate using IBM's actual formula.

**Simplified IBM counting:**
For a heavy-hex with $N$ total qubits:
- Approximately 2/3 are data qubits
- Approximately 1/3 are coupling/measure qubits

For IBM's 127-qubit Eagle:
$$N_{\text{data}} \approx 84, \quad N_{\text{measure}} \approx 43$$

$$\boxed{N_{\text{total}} = N_{\text{data}} + N_{\text{measure}}}$$

### Example 2: Flag Circuit Design

**Problem:** Design a flag circuit for a weight-4 Z stabilizer $Z_1 Z_2 Z_3 Z_4$ that detects if a single ancilla X error spreads to multiple data qubits.

**Solution:**

**Circuit with flag:**

```
d1: ─────────■─────────────────────────
             │
d2: ─────────┼────■────────────────────
             │    │
flag: ─|0⟩───┼────●────■───────────────  → measure
             │         │
d3: ─────────┼─────────●────■──────────
             │              │
d4: ─────────┼──────────────●────■─────
             │                   │
anc: ─|+⟩────●───────────────────●──── → measure
```

**How it works:**
1. Ancilla starts in $|+\rangle$ (for Z stabilizer measurement)
2. CNOTs connect ancilla to data qubits
3. Flag qubit has CNOTs interleaved
4. If ancilla has X error before d2 CNOT, it spreads to d2, d3, d4
5. The flag detects this pattern

**Flag detection condition:**
- No error: flag measures $|0\rangle$
- Single spread: flag measures $|1\rangle$

$$\boxed{\text{Flag } |1\rangle \Rightarrow \text{possible weight-2+ error}}$$

### Example 3: Threshold Comparison

**Problem:** IBM's heavy-hex achieves effective threshold $p_{\text{th}} = 0.6\%$ with physical error rate $p = 0.3\%$. Google achieves $p_{\text{th}} = 1.0\%$ with $p = 0.47\%$. Compare the error suppression factors.

**Solution:**

**IBM:**
$$\lambda_{\text{IBM}} = \frac{p_{\text{th}}}{p} = \frac{0.6\%}{0.3\%} = 2.0$$

**Google:**
$$\lambda_{\text{Google}} = \frac{p_{\text{th}}}{p} = \frac{1.0\%}{0.47\%} = 2.13$$

**Distance needed for same logical error:**

For $p_L = 10^{-4}$, starting from baseline $p_L(d=7) \approx 0.15\%$:

IBM: $d = 7 + 2 \times \lceil \log_{2.0}(0.0015/0.0001) \rceil = 7 + 2 \times 4 = 15$

Google: $d = 7 + 2 \times \lceil \log_{2.13}(0.0015/0.0001) \rceil = 7 + 2 \times 4 = 15$

Similar performance despite different architectures!

$$\boxed{\lambda_{\text{IBM}} \approx 2.0, \quad \lambda_{\text{Google}} \approx 2.1}$$

---

## Practice Problems

### Direct Application

**Problem 1:** An IBM heavy-hex processor has 127 qubits. Estimate:
a) The number of data qubits
b) The maximum surface code distance supportable
c) The number of flag qubits needed

**Problem 2:** The ECR gate takes 400 ns and has 0.3% error. The CZ gate takes 25 ns and has 0.25% error. Calculate the error rate per unit time for each.

**Problem 3:** A flag circuit adds 4 CNOT gates to detect spreading errors. If each CNOT has 0.3% error, what is the overhead from the flag circuit itself?

### Intermediate

**Problem 4:** Design the flag protocol for a weight-3 X stabilizer on heavy-hex. Show:
a) The gate sequence
b) Which errors are detected by the flag
c) The circuit depth

**Problem 5:** IBM's Condor processor has 1121 qubits. Assuming 60% are usable for a surface code:
a) What code distances are achievable?
b) Estimate the logical error rate if $\lambda = 2$
c) How many logical qubits can be implemented?

**Problem 6:** Compare the syndrome extraction cycle time for Google (1 μs, T1 = 68 μs) and IBM (hypothetical 4 μs, T1 = 300 μs). Which has better error per cycle from coherence decay?

### Challenging

**Problem 7:** The heavy-hex threshold is reduced by flag overhead. If the base threshold is 1% and flag circuits add effective error rate $\delta p = 0.2\%$, derive the effective threshold and compare to the square lattice.

**Problem 8:** IBM is developing qLDPC codes that could work better on heavy-hex geometry. Research and explain why the connectivity constraint is less severe for qLDPC codes compared to surface codes.

---

## Computational Lab: IBM Architecture Analysis

```python
"""
Day 836 Computational Lab: IBM Heavy-Hex Architecture Analysis
Compares IBM's heavy-hex approach with Google's square lattice
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.collections import PatchCollection

# =============================================================================
# Part 1: Heavy-Hex Lattice Construction
# =============================================================================

def create_heavy_hex_lattice(n_cols, n_rows):
    """
    Create a heavy-hex lattice structure.

    Parameters:
    -----------
    n_cols : int
        Number of hexagonal columns
    n_rows : int
        Number of hexagonal rows

    Returns:
    --------
    data_qubits : list of (x, y)
    measure_qubits : list of (x, y)
    connections : list of ((x1,y1), (x2,y2))
    """
    data_qubits = []
    measure_qubits = []
    connections = []

    # Hexagon dimensions
    h = np.sqrt(3)  # height of unit hexagon

    # Generate vertex positions (data qubits)
    for row in range(n_rows + 1):
        for col in range(n_cols + 1):
            x = 1.5 * col
            y = h * row + (h/2 if col % 2 == 1 else 0)
            data_qubits.append((x, y))

    # Generate edge positions (measure qubits) and connections
    for row in range(n_rows + 1):
        for col in range(n_cols):
            x1 = 1.5 * col
            y1 = h * row + (h/2 if col % 2 == 1 else 0)
            x2 = 1.5 * (col + 1)
            y2 = h * row + (h/2 if (col+1) % 2 == 1 else 0)

            # Measure qubit at midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            measure_qubits.append((mx, my))
            connections.append(((x1, y1), (mx, my)))
            connections.append(((mx, my), (x2, y2)))

    return data_qubits, measure_qubits, connections

print("=" * 60)
print("HEAVY-HEX LATTICE ANALYSIS")
print("=" * 60)

data_q, meas_q, conns = create_heavy_hex_lattice(4, 3)
print(f"4x3 Heavy-hex lattice:")
print(f"  Data qubits: {len(data_q)}")
print(f"  Measure qubits: {len(meas_q)}")
print(f"  Total qubits: {len(data_q) + len(meas_q)}")
print(f"  Connections: {len(conns)}")

# =============================================================================
# Part 2: IBM Processor Specifications
# =============================================================================

ibm_processors = {
    'Falcon': {'qubits': 27, 'year': 2020, 'qv': 64, 't1': 90, 't2': 70, 'cx_error': 0.01},
    'Hummingbird': {'qubits': 65, 'year': 2021, 'qv': 128, 't1': 100, 't2': 80, 'cx_error': 0.008},
    'Eagle': {'qubits': 127, 'year': 2022, 'qv': 256, 't1': 150, 't2': 100, 'cx_error': 0.006},
    'Osprey': {'qubits': 433, 'year': 2022, 'qv': None, 't1': 200, 't2': 150, 'cx_error': 0.005},
    'Condor': {'qubits': 1121, 'year': 2023, 'qv': None, 't1': 250, 't2': 180, 'cx_error': 0.004},
    'Heron': {'qubits': 133, 'year': 2024, 'qv': None, 't1': 300, 't2': 200, 'cx_error': 0.003}
}

print("\n" + "=" * 60)
print("IBM PROCESSOR EVOLUTION")
print("=" * 60)
print(f"{'Processor':<15} {'Year':<6} {'Qubits':<8} {'T1 (μs)':<10} {'CX Error':<10}")
print("-" * 55)
for name, specs in ibm_processors.items():
    print(f"{name:<15} {specs['year']:<6} {specs['qubits']:<8} {specs['t1']:<10} {specs['cx_error']*100:.2f}%")

# =============================================================================
# Part 3: Flag Circuit Overhead Analysis
# =============================================================================

def flag_circuit_overhead(base_depth, base_error, flag_gates=4, gate_error=0.003):
    """
    Calculate overhead from adding flag qubits.

    Parameters:
    -----------
    base_depth : int
        Syndrome extraction depth without flags
    base_error : float
        Error rate without flags
    flag_gates : int
        Additional gates for flag protocol
    gate_error : float
        Error per gate

    Returns:
    --------
    new_depth : int
        Total circuit depth
    new_error : float
        Total error rate
    overhead_factor : float
        Error increase factor
    """
    new_depth = base_depth + flag_gates
    flag_error = flag_gates * gate_error
    new_error = base_error + flag_error

    overhead_factor = new_error / base_error if base_error > 0 else float('inf')

    return new_depth, new_error, overhead_factor

print("\n" + "=" * 60)
print("FLAG CIRCUIT OVERHEAD ANALYSIS")
print("=" * 60)

# Typical values
base_depth = 8
base_error = 0.025  # 2.5% per cycle
gate_error = 0.003  # 0.3% per gate

for flag_gates in [2, 4, 6, 8]:
    new_d, new_e, overhead = flag_circuit_overhead(base_depth, base_error, flag_gates, gate_error)
    print(f"Flag gates: {flag_gates}, New depth: {new_d}, New error: {new_e*100:.2f}%, Overhead: {overhead:.2f}x")

# =============================================================================
# Part 4: Threshold Comparison
# =============================================================================

def effective_threshold(base_threshold, flag_overhead):
    """
    Calculate effective threshold with flag overhead.
    """
    return base_threshold / (1 + flag_overhead)

print("\n" + "=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

base_th_square = 0.01  # 1% for square lattice
flag_overhead = 0.3  # 30% overhead from flags

eff_th_heavy_hex = effective_threshold(base_th_square, flag_overhead)

print(f"Square lattice threshold: {base_th_square*100:.1f}%")
print(f"Heavy-hex effective threshold: {eff_th_heavy_hex*100:.2f}%")
print(f"Threshold reduction: {(1 - eff_th_heavy_hex/base_th_square)*100:.1f}%")

# =============================================================================
# Part 5: Surface Code Performance Comparison
# =============================================================================

def calculate_lambda(p_physical, p_threshold):
    """Calculate error suppression factor."""
    return p_threshold / p_physical

def project_logical_error(d, d_ref, p_ref, lam):
    """Project logical error to distance d."""
    return p_ref * (1/lam) ** ((d - d_ref) / 2)

print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON: GOOGLE vs IBM")
print("=" * 60)

# Google Willow parameters
google_p = 0.0047  # Effective physical error rate
google_th = 0.01
google_lambda = calculate_lambda(google_p, google_th)

# IBM Heron parameters (estimated)
ibm_p = 0.004  # Lower due to better coherence
ibm_th = 0.007  # Lower threshold due to flags
ibm_lambda = calculate_lambda(ibm_p, ibm_th)

print(f"\nGoogle Willow:")
print(f"  Physical error: {google_p*100:.2f}%")
print(f"  Threshold: {google_th*100:.1f}%")
print(f"  Lambda: {google_lambda:.2f}")

print(f"\nIBM Heron (estimated):")
print(f"  Physical error: {ibm_p*100:.2f}%")
print(f"  Threshold: {ibm_th*100:.2f}%")
print(f"  Lambda: {ibm_lambda:.2f}")

# Project error rates
print("\nProjected Logical Error Rates:")
print(f"{'Distance':<10} {'Google':<15} {'IBM':<15}")
print("-" * 40)

p_ref = 0.00143  # Reference at d=7

for d in range(7, 22, 2):
    p_google = project_logical_error(d, 7, p_ref, google_lambda)
    p_ibm = project_logical_error(d, 7, p_ref * (google_lambda/ibm_lambda), ibm_lambda)
    print(f"{d:<10} {p_google*100:.4f}%{'':<8} {p_ibm*100:.4f}%")

# =============================================================================
# Part 6: Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Heavy-hex lattice visualization
ax1 = axes[0, 0]
data_q, meas_q, conns = create_heavy_hex_lattice(3, 2)

# Draw connections
for (x1, y1), (x2, y2) in conns:
    ax1.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.5)

# Draw data qubits
for x, y in data_q:
    circle = Circle((x, y), 0.15, facecolor='blue', edgecolor='black', linewidth=2)
    ax1.add_patch(circle)

# Draw measure qubits
for x, y in meas_q:
    circle = Circle((x, y), 0.12, facecolor='red', edgecolor='black', linewidth=2)
    ax1.add_patch(circle)

ax1.set_xlim(-0.5, 5)
ax1.set_ylim(-0.5, 4)
ax1.set_aspect('equal')
ax1.set_title('Heavy-Hex Lattice Structure', fontsize=14)
ax1.set_xlabel('X position')
ax1.set_ylabel('Y position')
ax1.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)],
           ['Data Qubits', 'Measure Qubits'], loc='upper right')

# Plot 2: IBM processor evolution
ax2 = axes[0, 1]
years = [p['year'] for p in ibm_processors.values()]
qubits = [p['qubits'] for p in ibm_processors.values()]
names = list(ibm_processors.keys())

ax2.semilogy(years, qubits, 'bo-', markersize=10, linewidth=2)
for i, name in enumerate(names):
    ax2.annotate(name, (years[i], qubits[i]), textcoords="offset points",
                xytext=(5, 10), fontsize=9)

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Number of Qubits', fontsize=12)
ax2.set_title('IBM Quantum Processor Scaling', fontsize=14)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(2019, 2025)

# Plot 3: Lambda comparison
ax3 = axes[1, 0]
architectures = ['Google\nWillow', 'IBM\nHeron (est.)']
lambdas = [google_lambda, ibm_lambda]
colors = ['#4285f4', '#0f62fe']

bars = ax3.bar(architectures, lambdas, color=colors, edgecolor='black', linewidth=2)
ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Threshold (λ=1)')
ax3.set_ylabel('Error Suppression Factor λ', fontsize=12)
ax3.set_title('Error Suppression Factor Comparison', fontsize=14)
ax3.set_ylim(0, 3)
ax3.legend()

for bar, val in zip(bars, lambdas):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'λ = {val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 4: Logical error rate projections
ax4 = axes[1, 1]
distances = np.arange(7, 26, 2)

p_google = [project_logical_error(d, 7, p_ref, google_lambda) for d in distances]
p_ibm = [project_logical_error(d, 7, p_ref * 0.9, ibm_lambda) for d in distances]

ax4.semilogy(distances, np.array(p_google) * 100, 'b-o', label='Google Willow', linewidth=2, markersize=8)
ax4.semilogy(distances, np.array(p_ibm) * 100, 'r-s', label='IBM Heron (est.)', linewidth=2, markersize=8)

ax4.set_xlabel('Code Distance', fontsize=12)
ax4.set_ylabel('Logical Error Rate (%)', fontsize=12)
ax4.set_title('Projected Logical Error Rates', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, which='both')

# Add target lines
ax4.axhline(y=0.01, color='green', linestyle=':', alpha=0.7)
ax4.text(25.5, 0.01, '0.01%', va='center', fontsize=9)
ax4.axhline(y=0.001, color='orange', linestyle=':', alpha=0.7)
ax4.text(25.5, 0.001, '0.001%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('day_836_ibm_heavy_hex.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Visualization saved to: day_836_ibm_heavy_hex.png")
print("=" * 60)

# =============================================================================
# Part 7: Gate Time Comparison
# =============================================================================

print("\n" + "=" * 60)
print("GATE TIME AND ERROR ANALYSIS")
print("=" * 60)

# Google CZ gate
google_gate_time = 25e-9  # 25 ns
google_gate_error = 0.0025
google_error_per_ns = google_gate_error / (google_gate_time * 1e9)

# IBM ECR gate
ibm_gate_time = 400e-9  # 400 ns
ibm_gate_error = 0.003
ibm_error_per_ns = ibm_gate_error / (ibm_gate_time * 1e9)

print(f"\nGoogle CZ gate:")
print(f"  Gate time: {google_gate_time*1e9:.0f} ns")
print(f"  Gate error: {google_gate_error*100:.2f}%")
print(f"  Error per ns: {google_error_per_ns*100:.5f}%")

print(f"\nIBM ECR gate:")
print(f"  Gate time: {ibm_gate_time*1e9:.0f} ns")
print(f"  Gate error: {ibm_gate_error*100:.2f}%")
print(f"  Error per ns: {ibm_error_per_ns*100:.5f}%")

print(f"\nIBM has {google_error_per_ns/ibm_error_per_ns:.1f}x better error efficiency per time unit")

# =============================================================================
# Part 8: Coherence-limited Performance
# =============================================================================

print("\n" + "=" * 60)
print("COHERENCE-LIMITED ANALYSIS")
print("=" * 60)

# Syndrome cycle times
google_cycle = 1e-6  # 1 μs
ibm_cycle = 4e-6  # 4 μs (estimated with flag overhead)

# T1 values
google_t1 = 68e-6
ibm_t1 = 300e-6

# Coherence error per cycle
google_coh_error = google_cycle / google_t1
ibm_coh_error = ibm_cycle / ibm_t1

print(f"Google: T_cycle/T1 = {google_cycle*1e6:.1f}μs / {google_t1*1e6:.0f}μs = {google_coh_error*100:.2f}%")
print(f"IBM: T_cycle/T1 = {ibm_cycle*1e6:.1f}μs / {ibm_t1*1e6:.0f}μs = {ibm_coh_error*100:.2f}%")

print(f"\nCoherence error ratio (IBM/Google): {ibm_coh_error/google_coh_error:.2f}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Heavy-hex connectivity | Maximum degree = 3 |
| Flag detection | Flag $= \|1\rangle$ indicates possible spreading error |
| Effective threshold | $p_{\text{th}}^{\text{eff}} = p_{\text{th}}^{\text{base}} / (1 + \text{overhead})$ |
| ECR gate | $ECR = (I \otimes I + i X \otimes X)/\sqrt{2}$ |
| Coherence error | $p_{\text{coh}} \approx T_{\text{cycle}}/T_1$ |

### Main Takeaways

1. **Heavy-hex uses degree-3 connectivity** - Simpler fabrication but requires flag qubits
2. **Flag qubits detect error spreading** - Enable fault tolerance with low connectivity
3. **IBM compensates with higher coherence** - 300 μs T1 vs. Google's 68 μs
4. **Effective thresholds are similar** - ~0.7% for IBM vs ~1% for Google
5. **Both approaches achieve λ ~ 2** - Below-threshold operation is architecture-independent
6. **Manufacturing yield is higher** - Fewer connections = fewer defects

### Daily Checklist

- [ ] I understand the heavy-hex lattice geometry
- [ ] I can explain why flag qubits are needed for 3-connectivity
- [ ] I can design a basic flag circuit for stabilizer measurement
- [ ] I can compare Google and IBM approaches quantitatively
- [ ] I understand the trade-offs between connectivity and coherence
- [ ] I completed the computational lab and architecture comparison

---

## Preview: Day 837

Tomorrow we explore surface code implementations on alternative platforms: trapped ions (IonQ, Quantinuum) and neutral atoms (QuEra). These platforms offer fundamentally different advantages—all-to-all connectivity for ions and massive parallelism for neutral atoms—that may leapfrog superconducting approaches for certain applications.

**Key topics:**
- Trapped-ion surface codes with shuttling
- Neutral atom arrays with optical tweezers
- Platform comparison and trade-offs
- Hybrid approaches
