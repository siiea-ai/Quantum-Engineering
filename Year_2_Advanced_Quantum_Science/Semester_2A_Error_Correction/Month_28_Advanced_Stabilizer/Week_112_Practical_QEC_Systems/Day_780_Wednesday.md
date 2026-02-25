# Day 780: Code Switching and Gauge Fixing

## Year 2, Semester 2A: Error Correction | Month 28: Advanced Stabilizer Codes | Week 112

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Code switching theory and protocols |
| Afternoon | 2.5 hours | Gauge fixing in subsystem codes |
| Evening | 2 hours | Comparative analysis and simulations |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain the need for code switching** to achieve universal gate sets
2. **Implement Steane-to-Reed-Muller code switching** for transversal T gates
3. **Define gauge operators** and their role in subsystem codes
4. **Apply gauge fixing** to enable transversal non-Clifford gates
5. **Compare code switching versus magic state distillation** trade-offs
6. **Design hybrid protocols** combining multiple techniques

---

## Core Content

### 1. The Transversal Gate Problem

No single stabilizer code has a **complete transversal gate set**. The Eastin-Knill theorem proves:

$$\boxed{\text{No code has transversal gates forming a universal set}}$$

**Transversal gates for common codes:**

| Code | Transversal Gates |
|------|-------------------|
| Steane [[7,1,3]] | Full Clifford group |
| Surface code | CNOT, H (via folding) |
| 15-qubit Reed-Muller | T gate, CNOT |
| Color codes | CCZ (in 3D) |

**Solution approaches:**
1. Magic state distillation (Day 777)
2. Code switching (today)
3. Gauge fixing (today)
4. Code concatenation

### 2. Code Switching Fundamentals

**Code switching** transforms a logical state encoded in one code to an equivalent encoding in another code:

$$\boxed{|\bar{\psi}\rangle_{\text{Code A}} \rightarrow |\bar{\psi}\rangle_{\text{Code B}}}$$

The goal: Access a transversal gate in Code B that is not transversal in Code A.

#### Requirements for Code Switching

1. **Fault-tolerant transition**: Errors during switching must not propagate
2. **Logical state preservation**: The encoded information is unchanged
3. **Reversibility**: Can switch back after applying the desired gate

### 3. Steane to Reed-Muller Code Switching

The canonical example: switching between [[7,1,3]] Steane code (transversal Clifford) and [[15,1,3]] Reed-Muller code (transversal T).

#### Steane Code [[7,1,3]]

Stabilizer generators:
$$S_1 = IIIXXXX, \quad S_2 = IXXIIXX, \quad S_3 = XIXIXIX$$
$$S_4 = IIIZZZZ, \quad S_5 = IZZIIZZ, \quad S_6 = ZIZIZIZ$$

Logical operators:
$$\bar{X} = X^{\otimes 7}, \quad \bar{Z} = Z^{\otimes 7}$$

**Transversal T:** Not available (would give $T^{\otimes 7}$, not a valid logical operation)

#### Reed-Muller Code [[15,1,3]]

Based on the punctured first-order Reed-Muller code RM(1,4).

$$\boxed{\bar{T} = T^{\otimes 15}}$$

is a valid transversal T gate!

However, the Reed-Muller code does **not** have transversal H.

#### Code Switching Protocol

**Step 1: Encode auxiliary qubits**

Prepare 8 additional qubits in the state $|0\rangle^{\otimes 8}$.

**Step 2: Apply stabilizer measurements**

Measure the new stabilizers that relate Steane to Reed-Muller:

$$M_i = S_i^{\text{RM}} \cdot (S_j^{\text{Steane}})^\dagger$$

for appropriate pairings.

**Step 3: Apply corrections**

Based on measurement outcomes, apply Pauli corrections to complete the transition.

**Step 4: Apply transversal T**

$$\bar{T}|\bar{\psi}\rangle_{\text{RM}} = T^{\otimes 15}|\bar{\psi}\rangle_{\text{RM}}$$

**Step 5: Switch back to Steane**

Reverse the switching protocol.

#### Circuit Diagram

```
Steane (7 qubits) ─────[Switch to RM]────[T⊗15]────[Switch to Steane]─────
                            ↑                           ↓
Ancilla (8 qubits) ────[Prepare]─────[Measure]────[Discard/Reuse]─────
```

### 4. Subsystem Codes and Gauge Operators

**Subsystem codes** encode logical qubits in a subspace, leaving some degrees of freedom as **gauge qubits**.

#### Gauge Group Structure

The code is defined by:
- **Stabilizer group** $\mathcal{S}$: Operators that fix the code space
- **Gauge group** $\mathcal{G} \supset \mathcal{S}$: Includes gauge operators
- **Logical operators** $\bar{L}$: Act on encoded information

$$\boxed{\mathcal{G} = \langle \mathcal{S}, \mathcal{G}_{\text{gauge}} \rangle}$$

The gauge operators **commute with logical operators** but may not commute with each other.

#### Example: Bacon-Shor Code

The [[9,1,3]] Bacon-Shor code has:
- 2 stabilizer generators (products of gauge operators)
- 6 gauge generators
- 1 logical qubit

Gauge operators (2-qubit):
$$G_{X,ij} = X_i X_j, \quad G_{Z,kl} = Z_k Z_l$$

arranged on a 3×3 grid.

Stabilizers (4-qubit products):
$$S_X = \prod_{\text{row}} G_{X,ij}, \quad S_Z = \prod_{\text{col}} G_{Z,kl}$$

### 5. Gauge Fixing for Non-Clifford Gates

**Gauge fixing** measures gauge operators to collapse the gauge freedom, potentially enabling new transversal gates.

#### 3D Gauge Color Code

In the 3D gauge color code, by fixing different gauge choices:

$$\boxed{\text{Gauge choice A} \rightarrow \text{Transversal } T}$$
$$\boxed{\text{Gauge choice B} \rightarrow \text{Transversal } H}$$

The process:
1. Start with gauge choice A (transversal T available)
2. Apply T gate transversally
3. Measure gauge operators to switch to gauge choice B
4. Now H is transversal
5. Switch back as needed

#### Gauge Fixing Protocol

**Step 1:** Identify target gauge configuration

**Step 2:** Measure gauge operators $G_i$ not already fixed

**Step 3:** Apply corrections based on outcomes to restore code space

**Step 4:** The code now supports the desired transversal gate

### 6. Comparative Analysis

#### Code Switching vs. Magic State Distillation

| Aspect | Code Switching | Magic State Distillation |
|--------|---------------|-------------------------|
| Overhead qubits | Temporary (during switch) | Permanent factories |
| Time cost | $O(d)$ per switch | $O(d)$ per T-state |
| Failure modes | Switching errors | Distillation failures |
| Scalability | Limited by switching complexity | Highly parallelizable |
| Code requirements | Must find compatible codes | Works with any CSS code |

#### When to Use Each

**Code switching preferred when:**
- Need occasional non-Clifford gates
- Qubit budget is tight
- Compatible codes exist

**Magic states preferred when:**
- High T-gate density in circuit
- Parallelism is available
- Standard surface code architecture

### 7. Hybrid Approaches

Modern fault-tolerant architectures often combine approaches:

1. **Lattice surgery + magic states**: Surface code patches with magic state factories
2. **Code switching for CCZ + magic states for T**: Exploit 3D color code for CCZ
3. **Gauge fixing during computation**: Dynamically adjust gauge for different operations

#### Example Hybrid Protocol

For a circuit with many Cliffords and occasional T gates:

```
[Surface Code Computation (Cliffords)]
        ↓
[Inject Magic State for T]
        ↓
[Continue Surface Code Computation]
        ↓
[Code Switch to Color Code for CCZ]
        ↓
[Switch Back to Surface Code]
```

### 8. Error Analysis During Switching

Code switching introduces additional error mechanisms:

#### Switching Error Rate

$$\boxed{p_{\text{switch}} \sim (p/p_{\text{th}})^{d/2} \cdot n_{\text{switch}}}$$

where $n_{\text{switch}}$ is the number of syndrome measurement rounds during switching.

#### Maintaining Fault Tolerance

Requirements for fault-tolerant switching:
1. All intermediate states must be correctable
2. Measurement errors must not propagate catastrophically
3. Total error probability remains below threshold

---

## Quantum Mechanics Connection

### Gauge Symmetry in Physics

Gauge degrees of freedom in subsystem codes mirror **gauge symmetry** in physics:
- Gauge transformations leave physical observables unchanged
- Different gauge choices simplify different calculations
- Gauge fixing selects a convenient computational frame

In QEC, gauge freedom provides:
- Flexibility in syndrome measurement (can measure gauge ops or their products)
- Access to different transversal gate sets
- Potential for more efficient decoding

### Topological Interpretation

Code switching between 2D and 3D codes relates to:
- Dimensional reduction/lifting in topological phases
- Anyon condensation transitions
- Domain wall defects between codes

---

## Worked Examples

### Example 1: Steane to Reed-Muller Qubit Overhead

**Problem:** Calculate the qubit overhead for performing one T gate via code switching from Steane [[7,1,3]] to Reed-Muller [[15,1,3]] and back.

**Solution:**

Initial encoding: 7 physical qubits (Steane code)

To switch to Reed-Muller:
- Reed-Muller requires 15 qubits
- Additional qubits needed: $15 - 7 = 8$

Switching overhead:
- Ancilla qubits for measurements: ~15 (for syndrome extraction)
- Total during switch: $15 + 15 = 30$ qubits

Apply T gate: $T^{\otimes 15}$ (no additional qubits)

Switch back to Steane:
- Discard 8 qubits (or reuse)
- Syndrome measurement: ~7 ancillas

**Peak qubit usage:** 30 qubits
**Final encoding:** 7 qubits

$$\boxed{\text{Overhead factor} = 30/7 \approx 4.3\times}$$

Compare to magic state distillation: Requires dedicated factory (~30-50 qubits for 15-to-1), but amortized over many T gates.

### Example 2: Gauge Fixing in Bacon-Shor Code

**Problem:** In the [[9,1,3]] Bacon-Shor code arranged on a 3×3 grid, specify which gauge operators to measure to fix the code to a CSS form where $\bar{X} = X^{\otimes 9}$ is transversal.

**Solution:**

The Bacon-Shor code on a 3×3 grid:
```
1 - 2 - 3
|   |   |
4 - 5 - 6
|   |   |
7 - 8 - 9
```

X-type gauge operators (horizontal pairs):
$$G_{X,12} = X_1X_2, \quad G_{X,23} = X_2X_3, \ldots$$

Z-type gauge operators (vertical pairs):
$$G_{Z,14} = Z_1Z_4, \quad G_{Z,47} = Z_4Z_7, \ldots$$

To fix to a CSS form with $\bar{X} = X^{\otimes 9}$:

Measure all X-type gauge operators:
$$\{X_1X_2, X_2X_3, X_4X_5, X_5X_6, X_7X_8, X_8X_9\}$$

Each measurement outcome $\pm 1$ fixes the X-gauge.

After gauge fixing:
- If all outcomes are $+1$: $\bar{X} = X^{\otimes 9}$ directly
- Otherwise: Apply $X$ corrections to achieve $+1$ eigenspace

$$\boxed{\text{Fix 6 X-gauge operators to enable } \bar{X} = X^{\otimes 9}}$$

### Example 3: Hybrid Protocol Design

**Problem:** Design a fault-tolerant protocol to implement $T H T$ using code switching, starting from surface code encoding.

**Solution:**

Initial state: $|\bar{\psi}\rangle$ in surface code

**Step 1: First T gate (magic state injection)**

Surface code supports magic state injection:
1. Prepare magic state $|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$
2. Teleport logical qubit through magic state
3. Result: $T|\bar{\psi}\rangle$

**Step 2: H gate (code deformation or switch)**

Option A: Surface code H via lattice deformation
- Rotate the patch 90 degrees (swap rough/smooth boundaries)
- Takes $O(d)$ cycles

Option B: Code switch to Steane
- Switch surface → Steane (measure additional stabilizers)
- Apply $H^{\otimes 7}$
- Switch Steane → surface

**Step 3: Second T gate (magic state injection)**

Same as Step 1.

**Time estimate (Option A, $d = 11$):**
- T gate via injection: ~$d$ cycles = 11 cycles
- H via deformation: ~$d$ cycles = 11 cycles
- T gate via injection: ~$d$ cycles = 11 cycles
- Total: ~33 cycles

$$\boxed{THT \text{ via hybrid protocol: } \sim 3d \text{ cycles}}$$

---

## Practice Problems

### Level A: Direct Application

**A1.** List the transversal gates for (a) the [[7,1,3]] Steane code, (b) the [[15,1,3]] Reed-Muller code, (c) the surface code.

**A2.** For the Bacon-Shor [[9,1,3]] code, write down the stabilizer generators as products of gauge operators.

**A3.** If code switching takes $n_s = 5d$ syndrome rounds and physical error rate is $p = 10^{-3}$, estimate the switching error probability for $d = 7$.

### Level B: Intermediate Analysis

**B1.** Derive the logical T gate action on the Reed-Muller code. Show that $T^{\otimes 15}$ applies $T$ to the logical qubit.

**B2.** Design a gauge fixing protocol for the 3D color code to enable transversal CCZ. What gauge operators must be measured?

**B3.** Compare the qubit-time cost of performing 10 T gates using (a) code switching, (b) magic state distillation with a single factory.

### Level C: Research-Level Challenges

**C1.** Prove that no CSS code can have both transversal T and transversal H. (Hint: Use the structure of CSS logical operators.)

**C2.** Design a minimal code switching protocol between surface code and color code that preserves distance $d = 5$ throughout.

**C3.** Analyze how noise during gauge fixing affects the logical error rate. Derive conditions for fault-tolerant gauge fixing.

---

## Computational Lab

```python
"""
Day 780: Code Switching and Gauge Fixing
Simulation of code switching protocols and gauge fixing
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set
from itertools import product
from functools import reduce

# =============================================================================
# PAULI AND STABILIZER UTILITIES
# =============================================================================

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def tensor_product(ops: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Convert Pauli string like 'XYZII' to matrix."""
    return tensor_product([PAULIS[p] for p in pauli_str])


# =============================================================================
# STABILIZER CODE CLASS
# =============================================================================

class StabilizerCode:
    """Represent a stabilizer code."""

    def __init__(self, name: str, n: int, k: int, d: int,
                 stabilizers: List[str],
                 logical_x: str, logical_z: str):
        """
        Initialize stabilizer code.

        Args:
            name: Code name
            n: Number of physical qubits
            k: Number of logical qubits
            d: Code distance
            stabilizers: List of stabilizer generators as Pauli strings
            logical_x: Logical X operator
            logical_z: Logical Z operator
        """
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        self.stabilizers = stabilizers
        self.logical_x = logical_x
        self.logical_z = logical_z

    def get_stabilizer_matrices(self) -> List[np.ndarray]:
        """Get stabilizer generator matrices."""
        return [pauli_string_to_matrix(s) for s in self.stabilizers]

    def get_logical_x_matrix(self) -> np.ndarray:
        """Get logical X operator matrix."""
        return pauli_string_to_matrix(self.logical_x)

    def get_logical_z_matrix(self) -> np.ndarray:
        """Get logical Z operator matrix."""
        return pauli_string_to_matrix(self.logical_z)

    def encode_state(self, logical_state: np.ndarray) -> np.ndarray:
        """
        Encode a logical state (simplified - project onto code space).

        Args:
            logical_state: 2D vector representing |ψ⟩_L

        Returns:
            Encoded state vector
        """
        # Create projector onto code space
        dim = 2 ** self.n
        projector = np.eye(dim, dtype=complex)

        for stab_str in self.stabilizers:
            stab = pauli_string_to_matrix(stab_str)
            proj_stab = (np.eye(dim) + stab) / 2
            projector = projector @ proj_stab

        # Encode |0⟩_L and |1⟩_L
        # |0⟩_L is the +1 eigenstate of logical Z in code space
        # Find it by projecting computational |0...0⟩ onto code space
        psi_0 = np.zeros(dim, dtype=complex)
        psi_0[0] = 1  # |0...0⟩

        encoded_0 = projector @ psi_0
        if np.linalg.norm(encoded_0) > 1e-10:
            encoded_0 /= np.linalg.norm(encoded_0)
        else:
            # Try another basis state
            for i in range(dim):
                test = np.zeros(dim, dtype=complex)
                test[i] = 1
                encoded_0 = projector @ test
                if np.linalg.norm(encoded_0) > 1e-10:
                    encoded_0 /= np.linalg.norm(encoded_0)
                    break

        # |1⟩_L = X_L |0⟩_L
        X_L = self.get_logical_x_matrix()
        encoded_1 = X_L @ encoded_0
        encoded_1 /= np.linalg.norm(encoded_1)

        # Encode logical state
        return logical_state[0] * encoded_0 + logical_state[1] * encoded_1

    def transversal_gate(self, gate: np.ndarray) -> np.ndarray:
        """Apply transversal (tensor product) gate."""
        return tensor_product([gate] * self.n)

    def has_transversal_t(self) -> bool:
        """Check if T⊗n is a valid logical T."""
        # This is a simplified check - full verification requires
        # checking commutation with stabilizers
        return self.name in ['Reed-Muller [[15,1,3]]', '3D Color Code']


# =============================================================================
# DEFINE STANDARD CODES
# =============================================================================

def create_steane_code() -> StabilizerCode:
    """Create the [[7,1,3]] Steane code."""
    stabilizers = [
        'IIIXXXX',
        'IXXIIXX',
        'XIXIXIX',
        'IIIZZZZ',
        'IZZIIZZ',
        'ZIZIZIZ'
    ]
    return StabilizerCode(
        name='Steane [[7,1,3]]',
        n=7, k=1, d=3,
        stabilizers=stabilizers,
        logical_x='XXXXXXX',
        logical_z='ZZZZZZZ'
    )


def create_reed_muller_code() -> StabilizerCode:
    """Create the [[15,1,3]] Reed-Muller code."""
    # Reed-Muller stabilizers (subset)
    stabilizers = [
        'IIIIIIIIXXXXXXXX',  # Remove first qubit for [[15,1,3]]
        'IIIIXXXXIIIIXXXX',
        'IIXIIXIIXIIXIIXI',
        'IXIXIXIXIXIXIX',
        'IIIIIIIIZZZZZZZZ',
        'IIIIZZZZIIIIIZZZ',
        'IIZZIIZZIIZZIIZI',
        'IZIZIZIZIZIZIZIZ'
    ]
    # Simplified - actual RM code has more complex structure
    return StabilizerCode(
        name='Reed-Muller [[15,1,3]]',
        n=15, k=1, d=3,
        stabilizers=stabilizers[:8],  # Subset for demo
        logical_x='X' * 15,
        logical_z='Z' * 15
    )


# =============================================================================
# CODE SWITCHING SIMULATION
# =============================================================================

class CodeSwitcher:
    """Simulate code switching between codes."""

    def __init__(self, code_a: StabilizerCode, code_b: StabilizerCode):
        """
        Initialize code switcher.

        Args:
            code_a: First code
            code_b: Second code
        """
        self.code_a = code_a
        self.code_b = code_b

    def estimate_switching_cost(self, distance: int) -> Dict:
        """
        Estimate resources for switching.

        Args:
            distance: Effective code distance

        Returns:
            Dictionary with resource estimates
        """
        # Qubits during switch
        max_qubits = max(self.code_a.n, self.code_b.n)
        ancilla_qubits = max_qubits  # For syndrome measurement

        # Time for switching (syndrome rounds)
        syndrome_rounds = distance

        # Error probability (simplified model)
        p_phys = 1e-3
        p_switch = syndrome_rounds * p_phys

        return {
            'peak_qubits': max_qubits + ancilla_qubits,
            'syndrome_rounds': syndrome_rounds,
            'switch_error_prob': p_switch,
            'from_code': self.code_a.name,
            'to_code': self.code_b.name
        }

    def simulate_switch(self, logical_state: np.ndarray) -> np.ndarray:
        """
        Simulate ideal code switching (no errors).

        Args:
            logical_state: 2D logical state vector

        Returns:
            Encoded state in code_b
        """
        # In ideal case, just re-encode
        return self.code_b.encode_state(logical_state)


# =============================================================================
# GAUGE FIXING SIMULATION
# =============================================================================

class SubsystemCode:
    """Represent a subsystem code with gauge operators."""

    def __init__(self, name: str, n: int,
                 stabilizers: List[str],
                 x_gauges: List[str],
                 z_gauges: List[str],
                 logical_x: str, logical_z: str):
        """
        Initialize subsystem code.

        Args:
            name: Code name
            n: Number of physical qubits
            stabilizers: Stabilizer generators
            x_gauges: X-type gauge operators
            z_gauges: Z-type gauge operators
            logical_x: Logical X operator
            logical_z: Logical Z operator
        """
        self.name = name
        self.n = n
        self.stabilizers = stabilizers
        self.x_gauges = x_gauges
        self.z_gauges = z_gauges
        self.logical_x = logical_x
        self.logical_z = logical_z
        self.fixed_gauges: Dict[str, int] = {}

    def fix_gauge(self, gauge_op: str, value: int):
        """Fix a gauge operator to a specific eigenvalue (+1 or -1)."""
        self.fixed_gauges[gauge_op] = value

    def get_transversal_gates(self) -> List[str]:
        """Get available transversal gates based on gauge fixing."""
        # Simplified model
        if len(self.fixed_gauges) == 0:
            return ['CNOT']
        elif all(g.startswith('X') or 'X' in g for g in self.fixed_gauges):
            return ['CNOT', 'X^⊗n', 'CZ']
        else:
            return ['CNOT', 'Z^⊗n']


def create_bacon_shor_code() -> SubsystemCode:
    """Create the [[9,1,3]] Bacon-Shor code."""
    # 3x3 grid labeling:
    # 0 1 2
    # 3 4 5
    # 6 7 8

    x_gauges = [
        'XXIIIIIII',  # X_0 X_1
        'IXXIIIIII',  # X_1 X_2
        'IIIXXIIII',  # X_3 X_4
        'IIIIXXIII',  # X_4 X_5
        'IIIIIIXXII',  # X_6 X_7 (adjusted)
        'IIIIIIIXX'   # X_7 X_8
    ]

    z_gauges = [
        'ZIIZIIIII',  # Z_0 Z_3
        'IZIIZIIIII',  # Z_1 Z_4 (adjusted)
        'IIZIIIIZI',  # Z_2 Z_5
        'IIIZIIZII',  # Z_3 Z_6
        'IIIIZIIZI',  # Z_4 Z_7
        'IIIIIZIIZ'   # Z_5 Z_8
    ]

    stabilizers = [
        'XXXXXXIII',  # Product of row 1 X gauges
        'IIIXXXXXX',  # Product of row 2,3 X gauges
        'ZZIIZZIIZZ'  # Simplified Z stabilizer
    ]

    return SubsystemCode(
        name='Bacon-Shor [[9,1,3]]',
        n=9,
        stabilizers=stabilizers,
        x_gauges=x_gauges,
        z_gauges=z_gauges,
        logical_x='X' * 9,
        logical_z='Z' * 9
    )


# =============================================================================
# COMPARISON ANALYSIS
# =============================================================================

def compare_t_gate_methods():
    """Compare T-gate implementation methods."""

    print("=" * 60)
    print("T-GATE IMPLEMENTATION COMPARISON")
    print("=" * 60)

    distances = [5, 7, 11, 15, 21]

    results = []
    for d in distances:
        # Magic state distillation
        msd_qubits = 15 * 2 * d**2  # 15-to-1 factory
        msd_time = d  # cycles per T-state

        # Code switching (Steane to RM)
        switch_qubits = 15 + 15  # RM qubits + ancilla
        switch_time = 3 * d  # switching overhead

        results.append({
            'distance': d,
            'msd_qubits': msd_qubits,
            'msd_time': msd_time,
            'switch_qubits': switch_qubits,
            'switch_time': switch_time
        })

    # Print comparison table
    print(f"\n{'d':>3} | {'MSD Qubits':>12} | {'MSD Time':>10} | "
          f"{'Switch Qubits':>14} | {'Switch Time':>12}")
    print("-" * 65)

    for r in results:
        print(f"{r['distance']:>3} | {r['msd_qubits']:>12,} | "
              f"{r['msd_time']:>10} | {r['switch_qubits']:>14} | "
              f"{r['switch_time']:>12}")

    return results


def plot_overhead_comparison():
    """Visualize overhead comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    distances = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21])

    # Qubit overhead
    ax1 = axes[0]
    msd_qubits = 15 * 2 * distances**2
    switch_qubits = 30 * np.ones_like(distances)  # Constant for code switching

    ax1.semilogy(distances, msd_qubits, 'b-o', label='Magic State Distillation',
                 linewidth=2, markersize=8)
    ax1.semilogy(distances, switch_qubits, 'r-s', label='Code Switching',
                 linewidth=2, markersize=8)

    ax1.set_xlabel('Code Distance d', fontsize=12)
    ax1.set_ylabel('Qubits per T-gate', fontsize=12)
    ax1.set_title('Qubit Overhead for T-gate', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Time overhead
    ax2 = axes[1]
    msd_time = distances  # Amortized
    switch_time = 3 * distances

    ax2.plot(distances, msd_time, 'b-o', label='MSD (amortized)',
             linewidth=2, markersize=8)
    ax2.plot(distances, switch_time, 'r-s', label='Code Switching',
             linewidth=2, markersize=8)

    ax2.set_xlabel('Code Distance d', fontsize=12)
    ax2.set_ylabel('Time (code cycles)', fontsize=12)
    ax2.set_title('Time Overhead for T-gate', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day_780_overhead_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_gauge_fixing():
    """Demonstrate gauge fixing protocol."""

    print("\n" + "=" * 60)
    print("GAUGE FIXING DEMONSTRATION")
    print("=" * 60)

    code = create_bacon_shor_code()

    print(f"\nCode: {code.name}")
    print(f"Physical qubits: {code.n}")
    print(f"\nX-type gauge operators: {len(code.x_gauges)}")
    print(f"Z-type gauge operators: {len(code.z_gauges)}")

    print("\nInitial transversal gates:", code.get_transversal_gates())

    # Fix X gauges
    print("\nFixing X-type gauge operators to +1...")
    for gauge in code.x_gauges[:3]:
        code.fix_gauge(gauge, +1)

    print("After gauge fixing:", code.get_transversal_gates())

    print("\nGauge fixing enables transversal operations")
    print("that were not available with unfixed gauges.")


def analyze_t_gate_density():
    """Analyze when code switching vs MSD is preferred."""

    print("\n" + "=" * 60)
    print("OPTIMAL METHOD vs T-GATE DENSITY")
    print("=" * 60)

    # T-gates per 1000 logical gates
    t_densities = np.array([1, 5, 10, 50, 100, 200, 500])

    d = 11  # Fixed distance

    # Code switching: Fixed overhead per T-gate
    switch_overhead = 30  # qubits
    switch_time = 3 * d  # cycles per T

    # MSD: High fixed cost, low marginal cost
    factory_qubits = 15 * 2 * d**2  # ~3600 qubits
    msd_time = d  # amortized

    print(f"\nCode distance: d = {d}")
    print(f"Code switching: {switch_overhead} qubits, {switch_time} cycles/T")
    print(f"MSD factory: {factory_qubits} qubits, {msd_time} cycles/T (amortized)")

    print(f"\n{'T-gates/1000':>12} | {'Switch Cost':>12} | {'MSD Cost':>12} | {'Better':>10}")
    print("-" * 55)

    for density in t_densities:
        # Total qubit-cycles for 1000 gates
        switch_cost = density * switch_overhead * switch_time
        msd_cost = factory_qubits * (density * msd_time)

        better = "Switch" if switch_cost < msd_cost else "MSD"
        print(f"{density:>12} | {switch_cost:>12,} | {msd_cost:>12,} | {better:>10}")

    print("\nBreak-even point: when T-gate density exceeds ~10%")


if __name__ == "__main__":
    print("Day 780: Code Switching and Gauge Fixing")
    print("=" * 60)

    # Compare T-gate methods
    compare_t_gate_methods()

    # Demonstrate gauge fixing
    demonstrate_gauge_fixing()

    # Analyze optimal method
    analyze_t_gate_density()

    # Generate plots
    plot_overhead_comparison()

    # Create codes for inspection
    print("\n" + "=" * 60)
    print("CODE PROPERTIES")
    print("=" * 60)

    steane = create_steane_code()
    rm = create_reed_muller_code()

    print(f"\n{steane.name}")
    print(f"  Qubits: {steane.n}, Distance: {steane.d}")
    print(f"  Transversal T: {steane.has_transversal_t()}")

    print(f"\n{rm.name}")
    print(f"  Qubits: {rm.n}, Distance: {rm.d}")
    print(f"  Transversal T: {rm.has_transversal_t()}")

    # Switching cost
    switcher = CodeSwitcher(steane, rm)
    cost = switcher.estimate_switching_cost(distance=7)

    print(f"\nSwitching {steane.name} → {rm.name}:")
    print(f"  Peak qubits: {cost['peak_qubits']}")
    print(f"  Syndrome rounds: {cost['syndrome_rounds']}")
    print(f"  Switch error prob: {cost['switch_error_prob']:.4f}")
```

---

## Summary

### Key Formulas

| Concept | Formula/Result |
|---------|----------------|
| Eastin-Knill theorem | No code has universal transversal gates |
| Steane transversal | Clifford group ($H, S, \text{CNOT}$) |
| Reed-Muller transversal | $\bar{T} = T^{\otimes 15}$ |
| Gauge group | $\mathcal{G} = \langle \mathcal{S}, \mathcal{G}_{\text{gauge}} \rangle$ |
| Switching error | $p_{\text{switch}} \sim n_s \cdot p$ |
| Switching time | $O(d)$ syndrome rounds |

### Main Takeaways

1. **Code switching overcomes transversal limitations**: By switching between codes, we access different transversal gate sets
2. **Steane ↔ Reed-Muller for T gate**: Classic example of code switching protocol
3. **Gauge fixing in subsystem codes**: Different gauge choices enable different transversal gates
4. **Trade-offs with magic states**: Code switching has lower overhead for sparse T-gates
5. **Hybrid approaches are practical**: Combine multiple techniques based on circuit structure

---

## Daily Checklist

- [ ] I understand why code switching is necessary
- [ ] I can describe the Steane to Reed-Muller protocol
- [ ] I understand gauge operators in subsystem codes
- [ ] I can compare code switching vs magic state distillation
- [ ] I completed the computational lab
- [ ] I solved at least 2 practice problems from each level

---

## Preview: Day 781

Tomorrow we study **Hardware-Efficient Codes**, including bosonic codes that exploit the structure of specific physical systems:
- Cat codes with biased noise
- GKP (Gottesman-Kitaev-Preskill) codes in oscillators
- Codes tailored for superconducting and trapped-ion hardware

*"The best error correction exploits, rather than fights, the physics of your hardware."*

---

*Day 780 of 2184 | Year 2, Month 28, Week 112, Day 3*
*Quantum Engineering PhD Curriculum*
