# Day 573: Tensor Products and Circuit-Matrix Correspondence

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: Kronecker products, multi-qubit representation |
| Afternoon | 2.5 hours | Problem solving: Building circuit matrices |
| Evening | 1.5 hours | Computational lab: Circuit simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Compute Kronecker products** of matrices correctly
2. **Build multi-qubit gate matrices** from circuit diagrams
3. **Understand qubit ordering conventions** (big-endian vs little-endian)
4. **Convert between circuit diagrams and matrices** systematically
5. **Simulate quantum circuits** by matrix multiplication
6. **Handle mixed tensor products** (gates on non-adjacent qubits)

---

## Core Content

### 1. The Tensor Product (Kronecker Product)

For matrices $A \in \mathbb{C}^{m \times n}$ and $B \in \mathbb{C}^{p \times q}$:

$$\boxed{A \otimes B = \begin{pmatrix} a_{11}B & a_{12}B & \cdots & a_{1n}B \\ a_{21}B & a_{22}B & \cdots & a_{2n}B \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1}B & a_{m2}B & \cdots & a_{mn}B \end{pmatrix}}$$

Result: $(mp) \times (nq)$ matrix.

### 2. Basic Examples

**Two single-qubit gates:**
$$H \otimes Z = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

$$= \frac{1}{\sqrt{2}}\begin{pmatrix} Z & Z \\ Z & -Z \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & -1 & 0 & -1 \\ 1 & 0 & -1 & 0 \\ 0 & -1 & 0 & 1 \end{pmatrix}$$

### 3. Properties of Tensor Products

**Not commutative:**
$$A \otimes B \neq B \otimes A \text{ in general}$$

**Associative:**
$$(A \otimes B) \otimes C = A \otimes (B \otimes C)$$

**Mixed-product property:**
$$\boxed{(A \otimes B)(C \otimes D) = (AC) \otimes (BD)}$$

This is crucial for circuit analysis!

**Transpose:**
$$(A \otimes B)^T = A^T \otimes B^T$$

**Conjugate transpose:**
$$(A \otimes B)^\dagger = A^\dagger \otimes B^\dagger$$

### 4. Qubit Ordering Convention

**Convention used here:** $|q_0 q_1 ... q_{n-1}\rangle$

- $q_0$ is the **first** (leftmost) qubit in circuit diagrams
- In tensor products: $|\psi\rangle = |\psi_0\rangle \otimes |\psi_1\rangle \otimes ...$

**Example:** $|01\rangle = |0\rangle \otimes |1\rangle$

**Vector representation:**
$$|00\rangle \to \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, |01\rangle \to \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, |10\rangle \to \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, |11\rangle \to \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

### 5. Circuit to Matrix: Parallel Gates

When gates act in **parallel** (same time step):

```
q0: ───H───
q1: ───Z───
```

Matrix: $U = H \otimes Z$ (first qubit's gate first in tensor product)

### 6. Circuit to Matrix: Sequential Gates

When gates act in **sequence** (different time steps):

```
q0: ───A───B───
```

Matrix: $U = B \cdot A$ (later gates multiply from the left!)

**Combined:** For a circuit with parallel and sequential gates:
1. Build each time step as a tensor product
2. Multiply time steps from right to left

### 7. Identity Padding

If a gate acts on only some qubits:

```
q0: ───H───
q1: ───────
```

Matrix: $U = H \otimes I$

### 8. Multi-Qubit Example

**Circuit:**
```
q0: ───H───●───
           │
q1: ───────X───
```

**Step 1:** $H \otimes I$ (H on q0, nothing on q1)
**Step 2:** CNOT (controlled by q0, target q1)

$$U = \text{CNOT} \cdot (H \otimes I)$$

### 9. Gates on Non-Adjacent Qubits

For a 3-qubit system with gate U on qubits 0 and 2:

```
q0: ───●───
       │
q1: ───────
       │
q2: ───X───
```

This requires **SWAP gates** to make qubits adjacent, or a direct formula:

$$\text{CNOT}_{0\to 2} = (I \otimes \text{SWAP}) \cdot (\text{CNOT}_{0\to 1} \otimes I) \cdot (I \otimes \text{SWAP})$$

Or use the direct 8×8 matrix that swaps rows/columns appropriately.

### 10. General Rule for CNOT_{i→j}

For CNOT with control on qubit i and target on qubit j in an n-qubit system:

$$\text{CNOT}_{i\to j} = \sum_k |k\rangle\langle k| \otimes \begin{cases} X_j & \text{if } k_i = 1 \\ I_j & \text{if } k_i = 0 \end{cases}$$

where $k_i$ is the i-th bit of k.

### 11. Simulation by Matrix Multiplication

**To simulate a circuit:**
1. Initialize state: $|\psi_0\rangle = |00...0\rangle$
2. For each gate U in order: $|\psi\rangle \leftarrow U|\psi\rangle$
3. Measurement: $P(k) = |\langle k|\psi\rangle|^2$

### 12. Computational Complexity

For n qubits:
- State vector: $2^n$ complex numbers
- Gate matrix: $2^n \times 2^n$
- Matrix-vector multiplication: $O(4^n)$ operations

This exponential scaling is why quantum computers are hard to simulate classically!

---

## Quantum Computing Connection

Understanding tensor products is essential for:

1. **Circuit simulation:** Building and applying gate matrices
2. **Gate decomposition:** Breaking down complex gates
3. **Error analysis:** Propagating errors through circuits
4. **Optimization:** Identifying redundant operations
5. **Hardware mapping:** Placing gates on physical qubits

---

## Worked Examples

### Example 1: Build Bell State Circuit Matrix

**Problem:** Find the unitary matrix for the circuit that creates $|\Phi^+\rangle$:
```
q0: ───H───●───
           │
q1: ───────X───
```

**Solution:**

**Step 1:** H ⊗ I
$$H \otimes I = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}$$

**Step 2:** CNOT
$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

**Step 3:** Total unitary
$$U = \text{CNOT} \cdot (H \otimes I)$$

$$= \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}$$

$$= \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 1 & 0 & -1 \\ 1 & 0 & -1 & 0 \end{pmatrix}$$

**Verify:** $U|00\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$ ✓

### Example 2: Three-Qubit Circuit

**Problem:** Find the matrix for:
```
q0: ───H───●───────
           │
q1: ───────X───H───
q2: ───────────────
```

**Solution:**

**Step 1:** H ⊗ I ⊗ I (8×8 matrix)

**Step 2:** CNOT₀₁ ⊗ I (CNOT on qubits 0,1, identity on qubit 2)
$$= \text{CNOT} \otimes I$$

**Step 3:** I ⊗ H ⊗ I

$$U = (I \otimes H \otimes I) \cdot (\text{CNOT} \otimes I) \cdot (H \otimes I \otimes I)$$

### Example 3: Mixed Product Property

**Problem:** Simplify $(H \otimes Z)(X \otimes Y)$.

**Solution:**

Using $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$:

$$(H \otimes Z)(X \otimes Y) = (HX) \otimes (ZY)$$

Now:
- $HX = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$
- $ZY = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = \begin{pmatrix} 0 & -i \\ -i & 0 \end{pmatrix} = -iX$

So $(H \otimes Z)(X \otimes Y) = (HX) \otimes (-iX)$.

---

## Practice Problems

### Direct Application

1. Compute $X \otimes X$ explicitly.

2. Verify that $(I \otimes H)|01\rangle = |0\rangle|+\rangle$.

3. Find the 8×8 matrix for $H \otimes H \otimes H$.

### Intermediate

4. **Circuit to matrix:** Build the matrix for:
```
q0: ───X───
q1: ───H───●───
           │
q2: ───────X───
```

5. Verify the mixed-product property: $(H \otimes H)(H \otimes H) = I \otimes I$.

6. Show that $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$.

### Challenging

7. **Non-adjacent CNOT:** Build the 8×8 matrix for CNOT with control on qubit 0 and target on qubit 2:
```
q0: ───●───
       │
q1: ───│───
       │
q2: ───X───
```

8. For a general n-qubit state, show that applying a single-qubit gate U to qubit k is given by:
$$I^{\otimes k} \otimes U \otimes I^{\otimes (n-k-1)}$$

9. **Simulation:** Simulate the circuit H⊗H → CNOT → CZ on input |00⟩ and find the final state.

---

## Computational Lab: Circuit Simulation

```python
"""
Day 573: Tensor Products and Circuit Simulation
Building and simulating quantum circuits via matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

# Define basic gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
CZ = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]], dtype=complex)
SWAP = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print("=" * 60)
print("TENSOR PRODUCT BASICS")
print("=" * 60)

def tensor(*args):
    """Compute tensor product of multiple matrices."""
    return reduce(np.kron, args)

print("\n1. Basic Tensor Products:")
print(f"   X ⊗ X = \n{tensor(X, X)}")
print(f"\n   H ⊗ I = \n{tensor(H, I)}")

# Verify properties
print("\n2. Mixed-Product Property:")
A, B, C, D = H, Z, X, Y
lhs = tensor(A, B) @ tensor(C, D)
rhs = tensor(A @ C, B @ D)
print(f"   (A⊗B)(C⊗D) = (AC)⊗(BD): {np.allclose(lhs, rhs)}")

# Non-commutativity
print("\n3. Non-commutativity:")
print(f"   H⊗X = X⊗H: {np.allclose(tensor(H, X), tensor(X, H))}")

# Circuit simulation
print("\n" + "=" * 60)
print("CIRCUIT TO MATRIX CONVERSION")
print("=" * 60)

def ket(bitstring):
    """Create computational basis state from bitstring."""
    state = np.array([[1]], dtype=complex)
    for bit in bitstring:
        state = np.kron(state, ket_0 if bit == '0' else ket_1)
    return state

def measure_probs(state):
    """Get measurement probabilities in computational basis."""
    probs = np.abs(state.flatten())**2
    return probs

print("\n4. Bell State Circuit:")
print("   q0: ───H───●───")
print("             │")
print("   q1: ───────X───")

# Build circuit matrix
U_bell = CNOT @ tensor(H, I)
print(f"\n   Circuit matrix U = CNOT·(H⊗I):")
print(f"   {U_bell}")

# Apply to |00⟩
state_00 = ket('00')
bell_state = U_bell @ state_00
print(f"\n   U|00⟩ = {bell_state.flatten()}")
print(f"   = (|00⟩ + |11⟩)/√2: {np.allclose(bell_state, (ket('00') + ket('11'))/np.sqrt(2))}")

# Three-qubit example
print("\n" + "=" * 60)
print("THREE-QUBIT CIRCUIT")
print("=" * 60)

print("\n5. Three-qubit circuit:")
print("   q0: ───H───●───────")
print("             │")
print("   q1: ───H───X───●───")
print("                 │")
print("   q2: ───H───────X───")

# Build step by step
step1 = tensor(H, H, H)  # H on all qubits
step2 = tensor(CNOT, I)   # CNOT on q0,q1, identity on q2
step3 = tensor(I, CNOT)   # Identity on q0, CNOT on q1,q2

U_3qubit = step3 @ step2 @ step1
print(f"\n   Circuit matrix shape: {U_3qubit.shape}")

# Apply to |000⟩
state_000 = ket('000')
result = U_3qubit @ state_000
print(f"   Result state:")
for i, amp in enumerate(result.flatten()):
    if np.abs(amp) > 1e-10:
        bitstring = format(i, '03b')
        print(f"      {amp.real:+.4f} |{bitstring}⟩")

# Non-adjacent gates
print("\n" + "=" * 60)
print("NON-ADJACENT QUBIT GATES")
print("=" * 60)

print("\n6. CNOT with control=0, target=2 (skipping qubit 1):")
print("   q0: ───●───")
print("          │")
print("   q1: ───────")
print("          │")
print("   q2: ───X───")

# Method 1: Using SWAPs
CNOT_02_via_swap = tensor(I, SWAP) @ tensor(CNOT, I) @ tensor(I, SWAP)
print(f"\n   Via SWAPs: (I⊗SWAP)·(CNOT⊗I)·(I⊗SWAP)")

# Method 2: Direct construction
def cnot_non_adjacent(n_qubits, control, target):
    """Build CNOT matrix for non-adjacent qubits."""
    dim = 2**n_qubits
    result = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        bitstring = format(i, f'0{n_qubits}b')
        ctrl_bit = int(bitstring[control])
        tgt_bit = int(bitstring[target])

        # Output bitstring
        if ctrl_bit == 1:
            new_tgt_bit = 1 - tgt_bit
        else:
            new_tgt_bit = tgt_bit

        out_bits = list(bitstring)
        out_bits[target] = str(new_tgt_bit)
        j = int(''.join(out_bits), 2)

        result[j, i] = 1

    return result

CNOT_02_direct = cnot_non_adjacent(3, 0, 2)
print(f"   Direct construction matches SWAP method: {np.allclose(CNOT_02_via_swap, CNOT_02_direct)}")

# Verify action
test_states = ['000', '001', '100', '101']
print(f"\n   Action on basis states:")
for bs in test_states:
    input_state = ket(bs)
    output_state = CNOT_02_direct @ input_state
    for j, amp in enumerate(output_state.flatten()):
        if np.abs(amp) > 0.5:
            out_bs = format(j, '03b')
            print(f"      CNOT₀₂|{bs}⟩ = |{out_bs}⟩")

# General gate placement
print("\n" + "=" * 60)
print("GENERAL GATE PLACEMENT")
print("=" * 60)

def place_gate(gate, qubit, n_qubits):
    """Place a single-qubit gate on specified qubit in n-qubit system."""
    ops = [I] * n_qubits
    ops[qubit] = gate
    return reduce(np.kron, ops)

print("\n7. Placing single-qubit gate on qubit k in n-qubit system:")
print("   Formula: I⊗...⊗I ⊗ U ⊗ I⊗...⊗I")
print("            (k terms)    (n-k-1 terms)")

# Example: H on qubit 1 of 3-qubit system
H_on_1 = place_gate(H, 1, 3)
H_on_1_explicit = tensor(I, H, I)
print(f"\n   H on qubit 1 of 3: place_gate matches explicit: {np.allclose(H_on_1, H_on_1_explicit)}")

# Circuit simulation function
print("\n" + "=" * 60)
print("CIRCUIT SIMULATOR")
print("=" * 60)

class QuantumCircuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.gates = []

    def h(self, qubit):
        self.gates.append(('H', qubit))

    def x(self, qubit):
        self.gates.append(('X', qubit))

    def cnot(self, control, target):
        self.gates.append(('CNOT', control, target))

    def cz(self, q1, q2):
        self.gates.append(('CZ', q1, q2))

    def get_unitary(self):
        U = np.eye(self.dim, dtype=complex)
        for gate_info in self.gates:
            if gate_info[0] == 'H':
                G = place_gate(H, gate_info[1], self.n_qubits)
            elif gate_info[0] == 'X':
                G = place_gate(X, gate_info[1], self.n_qubits)
            elif gate_info[0] == 'CNOT':
                G = cnot_non_adjacent(self.n_qubits, gate_info[1], gate_info[2])
            elif gate_info[0] == 'CZ':
                # Build CZ similarly
                G = np.eye(self.dim, dtype=complex)
                for i in range(self.dim):
                    bs = format(i, f'0{self.n_qubits}b')
                    if bs[gate_info[1]] == '1' and bs[gate_info[2]] == '1':
                        G[i, i] = -1
            U = G @ U
        return U

    def simulate(self, initial_state=None):
        if initial_state is None:
            initial_state = ket('0' * self.n_qubits)
        U = self.get_unitary()
        return U @ initial_state

print("\n8. Simulating GHZ State Creation:")
print("   q0: ───H───●───●───")
print("             │   │")
print("   q1: ───────X───│───")
print("                 │")
print("   q2: ───────────X───")

qc = QuantumCircuit(3)
qc.h(0)
qc.cnot(0, 1)
qc.cnot(0, 2)

ghz_state = qc.simulate()
print(f"\n   Result (GHZ state):")
for i, amp in enumerate(ghz_state.flatten()):
    if np.abs(amp) > 1e-10:
        bs = format(i, '03b')
        print(f"      {amp.real:+.4f} |{bs}⟩")

expected_ghz = (ket('000') + ket('111')) / np.sqrt(2)
print(f"\n   Matches (|000⟩+|111⟩)/√2: {np.allclose(ghz_state, expected_ghz)}")

# Visualization
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Tensor product visualization
ax1 = axes[0, 0]
# Show H⊗I matrix
HI = tensor(H, I)
im1 = ax1.imshow(np.abs(HI), cmap='Blues', vmin=0, vmax=1)
ax1.set_title('|H ⊗ I| Matrix', fontsize=12)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
ax1.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
plt.colorbar(im1, ax=ax1)

# Plot 2: Bell circuit unitary
ax2 = axes[0, 1]
im2 = ax2.imshow(np.abs(U_bell), cmap='Purples', vmin=0, vmax=1)
ax2.set_title('Bell Circuit Unitary', fontsize=12)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
ax2.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
for i in range(4):
    for j in range(4):
        val = U_bell[i,j]
        if np.abs(val) > 0.3:
            text = f'{np.abs(val):.2f}'
            ax2.text(j, i, text, ha='center', va='center', fontsize=9)
plt.colorbar(im2, ax=ax2)

# Plot 3: Three-qubit circuit matrix (8x8)
ax3 = axes[1, 0]
im3 = ax3.imshow(np.abs(U_3qubit), cmap='Greens', vmin=0, vmax=0.5)
ax3.set_title('3-Qubit Circuit Unitary', fontsize=12)
ax3.set_xlabel('Input state index', fontsize=10)
ax3.set_ylabel('Output state index', fontsize=10)
plt.colorbar(im3, ax=ax3)

# Plot 4: Measurement probabilities
ax4 = axes[1, 1]
states_to_show = [
    ('|00⟩ after Bell', U_bell @ ket('00')),
    ('|000⟩ after GHZ', ghz_state),
]

x_positions = [0, 1]
colors = ['steelblue', 'coral']

for idx, (name, state) in enumerate(states_to_show):
    probs = measure_probs(state)
    n_qubits = int(np.log2(len(probs)))
    labels = [format(i, f'0{n_qubits}b') for i in range(len(probs))]

    bar_width = 0.35
    x = np.arange(len(probs)) + idx * bar_width
    ax4.bar(x, probs, bar_width, label=name, color=colors[idx], alpha=0.7)

ax4.set_xlabel('Basis state', fontsize=10)
ax4.set_ylabel('Probability', fontsize=10)
ax4.set_title('Measurement Probabilities', fontsize=12)
ax4.legend()
ax4.set_ylim([0, 0.6])

plt.tight_layout()
plt.savefig('tensor_products.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: tensor_products.png")

# Summary of qubit ordering
print("\n" + "=" * 60)
print("QUBIT ORDERING SUMMARY")
print("=" * 60)

print("""
Convention used:
- State |q₀q₁...qₙ₋₁⟩ has q₀ as leftmost (first) qubit
- Tensor product: |ψ⟩ = |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ ...
- Vector index: |q₀q₁...⟩ → index = q₀·2ⁿ⁻¹ + q₁·2ⁿ⁻² + ...

Example for 2 qubits:
  |00⟩ → index 0 → [1,0,0,0]ᵀ
  |01⟩ → index 1 → [0,1,0,0]ᵀ
  |10⟩ → index 2 → [0,0,1,0]ᵀ
  |11⟩ → index 3 → [0,0,0,1]ᵀ

Gate placement:
  A on q₀, B on q₁: matrix = A ⊗ B
  Sequential: later gates multiply from LEFT
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Tensor product | $(A \otimes B)_{(i,j),(k,l)} = A_{i,k} B_{j,l}$ |
| Mixed product | $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$ |
| Parallel gates | $U = A \otimes B$ for gates A, B in parallel |
| Sequential gates | $U = B \cdot A$ for gate A then B |
| Gate placement | $I^{\otimes k} \otimes U \otimes I^{\otimes (n-k-1)}$ |
| Basis vector | $\|q_0...q_{n-1}\rangle$ has index $\sum_i q_i 2^{n-1-i}$ |

### Main Takeaways

1. **Tensor product builds multi-qubit matrices:** Use ⊗ for parallel gates
2. **Order matters:** A⊗B ≠ B⊗A in general
3. **Mixed-product rule:** Key for simplifying tensor expressions
4. **Sequential gates multiply left:** Later gates go on the left in product
5. **Non-adjacent qubits:** Require SWAPs or careful indexing
6. **Exponential scaling:** Matrix dimension doubles with each qubit

---

## Daily Checklist

- [ ] I can compute Kronecker products of matrices
- [ ] I understand the mixed-product property
- [ ] I can build circuit matrices from circuit diagrams
- [ ] I know the qubit ordering convention
- [ ] I can handle gates on non-adjacent qubits
- [ ] I completed the circuit simulator lab
- [ ] I solved at least 3 practice problems

---

## Preview of Day 574

Tomorrow is the **Week 82 Review**, where we consolidate all two-qubit gate concepts. We'll work through comprehensive problems covering CNOT, CZ, SWAP, entangling power, gate identities, and tensor products.
