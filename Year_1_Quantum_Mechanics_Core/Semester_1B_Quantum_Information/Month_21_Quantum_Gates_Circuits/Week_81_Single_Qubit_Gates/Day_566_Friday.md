# Day 566: Gate Decomposition

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: Euler angles, ZYZ decomposition, universality |
| Afternoon | 2.5 hours | Problem solving: Decomposition calculations |
| Evening | 1.5 hours | Computational lab: Gate compilation algorithms |

## Learning Objectives

By the end of today, you will be able to:

1. **Decompose any single-qubit gate** using ZYZ Euler angles
2. **Derive the decomposition formulas** from matrix elements
3. **Understand gate universality** and approximation theorems
4. **Implement the Solovay-Kitaev algorithm** conceptually
5. **Compile gates** to hardware-native instruction sets
6. **Analyze decomposition efficiency** and gate counts

---

## Core Content

### 1. The Decomposition Problem

**Goal:** Express any single-qubit unitary U ∈ SU(2) as a product of simpler gates.

**Why this matters:**
- Real quantum hardware has limited native gates
- Algorithm design uses arbitrary rotations
- Gate synthesis bridges abstract algorithms and physical implementation

### 2. ZYZ Euler Decomposition

**Theorem (ZYZ Decomposition):** Any U ∈ SU(2) can be written as:

$$\boxed{U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)}$$

for some angles α, β, γ, δ ∈ ℝ.

**Proof:**

Any U ∈ SU(2) has the form:
$$U = \begin{pmatrix} a & -b^* \\ b & a^* \end{pmatrix}, \quad |a|^2 + |b|^2 = 1$$

Write $a = |a|e^{i\phi_a}$ and $b = |b|e^{i\phi_b}$.

The ZYZ product is:
$$R_z(\beta)R_y(\gamma)R_z(\delta) = \begin{pmatrix} e^{-i(\beta+\delta)/2}\cos\frac{\gamma}{2} & -e^{-i(\beta-\delta)/2}\sin\frac{\gamma}{2} \\ e^{i(\beta-\delta)/2}\sin\frac{\gamma}{2} & e^{i(\beta+\delta)/2}\cos\frac{\gamma}{2} \end{pmatrix}$$

Matching to U:
- $\cos(\gamma/2) = |a|$, $\sin(\gamma/2) = |b|$
- $\beta + \delta = -2\phi_a$
- $\beta - \delta = -2\phi_b - \pi$

Solving:
$$\gamma = 2\arccos|a|$$
$$\beta = -\phi_a - \phi_b - \frac{\pi}{2}$$
$$\delta = -\phi_a + \phi_b + \frac{\pi}{2}$$

The global phase α accounts for det(U) normalization. ∎

### 3. Extracting Euler Angles

**Algorithm for ZYZ decomposition:**

Given $U = \begin{pmatrix} u_{00} & u_{01} \\ u_{10} & u_{11} \end{pmatrix}$ in SU(2):

**Step 1:** Find γ from the magnitudes:
$$\cos\frac{\gamma}{2} = |u_{00}| = |u_{11}|$$
$$\gamma = 2\arccos|u_{00}|$$

**Step 2:** Find β and δ from the phases:
$$\frac{\beta + \delta}{2} = -\arg(u_{00})$$
$$\frac{\beta - \delta}{2} = \arg(u_{10}) - \frac{\pi}{2}$$

Therefore:
$$\beta = -\arg(u_{00}) + \arg(u_{10}) - \frac{\pi}{2}$$
$$\delta = -\arg(u_{00}) - \arg(u_{10}) + \frac{\pi}{2}$$

**Special cases:**
- If γ = 0: U = Rz(β + δ), only total rotation matters
- If γ = π: Different formula needed (gimbal lock)

### 4. Alternative Decompositions

**XYX Decomposition:**
$$U = R_x(\alpha)R_y(\beta)R_x(\gamma)$$

**ZXZ Decomposition:**
$$U = R_z(\alpha)R_x(\beta)R_z(\gamma)$$

**ABC Decomposition (for controlled gates):**
$$U = e^{i\alpha}AXBXC$$
where $ABC = I$ and $e^{i\alpha}AXBXC = U$.

### 5. Decomposition into Discrete Gates

Real hardware often has only discrete gates: {H, T, CNOT} or similar.

**The challenge:** How to approximate continuous rotations with discrete gates?

**Solovay-Kitaev Theorem:** For any ε > 0 and any U ∈ SU(2), there exists a sequence of gates from a universal set that approximates U to accuracy ε using O(log^c(1/ε)) gates, where c ≈ 3.97.

### 6. Universal Gate Sets

A gate set is **universal** if it can approximate any unitary.

**Common universal sets:**
| Set | Notes |
|-----|-------|
| {H, T} | Generates dense subgroup of SU(2) |
| {H, T, CNOT} | Universal for multi-qubit |
| {Rx, Ry, Rz, CNOT} | Continuous parameters |
| {√X, Rz, CNOT} | IBM native gates |

**Key insight:** H and T together can approximate any single-qubit gate!

### 7. The Solovay-Kitaev Algorithm

**Goal:** Find a short sequence of H and T gates approximating U.

**Basic idea:**
1. Start with a rough approximation V₀
2. Iteratively improve: $V_{n+1} = V_n \cdot \text{GKD}(V_n^\dagger U)$
3. GKD (group commutator decomposition) improves accuracy quadratically

**Group Commutator Decomposition:**
If W ≈ I with error ε, then:
$$W \approx [A, B] = ABA^{-1}B^{-1}$$
for some A, B with error √ε each.

This allows recursive construction with exponentially improving accuracy.

### 8. Native Gate Compilation

Different quantum hardware has different native gates:

| Platform | Native Single-Qubit Gates |
|----------|---------------------------|
| IBM Quantum | √X (SX), Rz(θ), X |
| Google Sycamore | √W, Rz(θ) |
| IonQ | R(θ,φ) arbitrary rotations |
| Rigetti | Rx(±π/2), Rz(θ) |

**Compilation task:** Convert ZYZ decomposition to native gates.

**Example (IBM):** $R_y(\theta) = R_z(\pi/2) \cdot \sqrt{X} \cdot R_z(\theta) \cdot \sqrt{X} \cdot R_z(-\pi/2)$

### 9. Gate Synthesis Optimization

**Metrics:**
- Gate count (minimize total gates)
- Depth (minimize circuit depth for parallelism)
- Specific gate count (minimize expensive gates like T)
- Fidelity (maximize accuracy)

**Trade-offs:** Fewer gates vs. higher accuracy

### 10. Practical Decomposition: Examples

**Hadamard:**
$$H = R_z(\pi)R_y(\pi/2) = e^{i\pi/2}R_z(\pi)R_y(\pi/2)R_z(0)$$

Euler angles: β = π, γ = π/2, δ = 0.

**T gate:**
$$T = e^{i\pi/8}R_z(\pi/4)$$

Already a Z-rotation!

**√X gate:**
$$\sqrt{X} = e^{i\pi/4}R_x(\pi/2) = e^{i\pi/4}R_z(-\pi/2)R_y(\pi/2)R_z(\pi/2)$$

---

## Quantum Computing Connection

Gate decomposition is essential for:

1. **Circuit compilation:** Translating algorithms to hardware
2. **Error analysis:** Understanding how decomposition affects noise
3. **Resource estimation:** Counting gates for fault tolerance
4. **Optimization:** Minimizing circuit depth and gate count
5. **Variational algorithms:** Efficient parameterization

---

## Worked Examples

### Example 1: ZYZ Decomposition of Hadamard

**Problem:** Find the ZYZ Euler angles for H.

**Solution:**

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Step 1:** Put H in SU(2) form (det = -1, need to adjust):

Actually det(H) = -1, so we write $H = (-1) \cdot H'$ where $H' = -H$ has det = 1.

Better approach: $H = e^{i\pi/2} \cdot H_{SU(2)}$ where:
$$H_{SU(2)} = e^{-i\pi/2}H = \frac{e^{-i\pi/2}}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} -i & -i \\ -i & i \end{pmatrix}$$

Let's compute directly:
$$R_z(\pi)R_y(\pi/2) = \begin{pmatrix} e^{-i\pi/2} & 0 \\ 0 & e^{i\pi/2} \end{pmatrix}\begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}$$

$$= \begin{pmatrix} \frac{-i}{\sqrt{2}} & \frac{i}{\sqrt{2}} \\ \frac{i}{\sqrt{2}} & \frac{i}{\sqrt{2}} \end{pmatrix}$$

This differs from H by a global phase. Let's verify:
$$e^{i\cdot 3\pi/4}R_z(\pi)R_y(\pi/2) = e^{i\cdot 3\pi/4}\begin{pmatrix} \frac{-i}{\sqrt{2}} & \frac{i}{\sqrt{2}} \\ \frac{i}{\sqrt{2}} & \frac{i}{\sqrt{2}} \end{pmatrix}$$

Actually, there's a simpler form:
$$H = R_y(\pi/2)R_z(\pi)$$

Let's verify:
$$R_y(\pi/2)R_z(\pi) = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}\begin{pmatrix} -i & 0 \\ 0 & i \end{pmatrix}$$

That's not right either. Let me use the Rz definition more carefully:
$$R_z(\pi) = \begin{pmatrix} e^{-i\pi/2} & 0 \\ 0 & e^{i\pi/2} \end{pmatrix} = \begin{pmatrix} -i & 0 \\ 0 & i \end{pmatrix}$$

For the ZYZ form: $H = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$

Using $|H_{00}| = |H_{11}| = 1/\sqrt{2}$:
$$\cos(\gamma/2) = 1/\sqrt{2} \Rightarrow \gamma = \pi/2$$

From phases: $\arg(H_{00}) = 0$, $\arg(H_{10}) = 0$

$$\beta + \delta = 0, \quad \beta - \delta = -\pi$$

So β = -π/2, δ = π/2.

**Answer:** H = e^{iπ/2} Rz(-π/2) Ry(π/2) Rz(π/2)

### Example 2: Decomposing an Arbitrary Unitary

**Problem:** Find the ZYZ decomposition of $U = \begin{pmatrix} \frac{1+i}{2} & \frac{-1+i}{2} \\ \frac{1+i}{2} & \frac{1-i}{2} \end{pmatrix}$.

**Solution:**

**Step 1:** Verify U ∈ SU(2):
- $|U_{00}|^2 + |U_{10}|^2 = \frac{1}{2} + \frac{1}{2} = 1$ ✓
- det(U) = 1 (can verify)

**Step 2:** Find γ:
$$|U_{00}| = \frac{|1+i|}{2} = \frac{\sqrt{2}}{2}$$
$$\cos(\gamma/2) = \frac{1}{\sqrt{2}} \Rightarrow \gamma = \frac{\pi}{2}$$

**Step 3:** Find β and δ:
$$\arg(U_{00}) = \arg(1+i) = \frac{\pi}{4}$$
$$\arg(U_{10}) = \arg(1+i) = \frac{\pi}{4}$$

$$\beta = -\arg(U_{00}) + \arg(U_{10}) - \frac{\pi}{2} = -\frac{\pi}{4} + \frac{\pi}{4} - \frac{\pi}{2} = -\frac{\pi}{2}$$
$$\delta = -\arg(U_{00}) - \arg(U_{10}) + \frac{\pi}{2} = -\frac{\pi}{4} - \frac{\pi}{4} + \frac{\pi}{2} = 0$$

**Answer:** $U = R_z(-\pi/2)R_y(\pi/2)R_z(0) = R_z(-\pi/2)R_y(\pi/2)$

### Example 3: T-count Optimization

**Problem:** Express Rz(π/8) using H and T gates.

**Solution:**

We have T = e^{iπ/8}Rz(π/4), so Rz(π/4) = e^{-iπ/8}T.

For Rz(π/8), we need a finer rotation. Since T gives π/4:
$$T^{1/2} = \sqrt{T} \text{ would give } R_z(\pi/8)$$

But √T is not in our gate set! We need to approximate.

Using the identity HT = THTHTodd... (Solovay-Kitaev approach):

A good approximation uses the sequence:
$$R_z(\pi/8) \approx HTHTH^{-1}T^{-1}H^{-1}...$$

The exact sequence depends on the desired precision and is found algorithmically.

---

## Practice Problems

### Direct Application

1. Find the ZYZ decomposition of the Y gate.

2. Verify that $R_z(\alpha)R_y(\beta)R_z(\gamma) = R_z(\alpha+\gamma)$ when β = 0.

3. Compute the ZYZ decomposition of the S gate: $S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$.

### Intermediate

4. **Alternative decomposition:** Derive the XYX decomposition formula analogous to ZYZ.

5. **Gate count:** Given H and T gates only, what is the minimum number of gates needed to approximate Rz(π/8) to accuracy ε = 0.01?

6. Show that $R_y(\theta) = S^\dagger \cdot H \cdot R_z(\theta) \cdot H \cdot S$. Verify for θ = π/2.

### Challenging

7. **ABC decomposition:** For a controlled-U gate, find A, B, C such that ABC = I and U = e^{iα}AXBXC, for U = T.

8. **Compilation efficiency:** IBM's native gates are {√X, Rz(θ), X}. Find the optimal decomposition of H in terms of these gates.

9. **Solovay-Kitaev:** Implement one iteration of the Solovay-Kitaev algorithm to improve the approximation of Rz(0.1) starting from the identity.

---

## Computational Lab: Gate Decomposition Algorithms

```python
"""
Day 566: Gate Decomposition
ZYZ decomposition, gate synthesis, and compilation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Rotation gates
def Rx(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*X

def Ry(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Y

def Rz(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Z

# Standard gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

print("=" * 60)
print("ZYZ EULER DECOMPOSITION")
print("=" * 60)

def zyz_decompose(U):
    """
    Decompose U ∈ SU(2) as U = e^{iα} Rz(β) Ry(γ) Rz(δ).
    Returns (α, β, γ, δ).
    """
    # Ensure U is in SU(2)
    det = np.linalg.det(U)
    global_phase = np.angle(det) / 2
    U_su2 = U / np.exp(1j * global_phase)

    # Extract γ from magnitude of (0,0) element
    cos_half_gamma = np.abs(U_su2[0, 0])
    cos_half_gamma = np.clip(cos_half_gamma, 0, 1)
    gamma = 2 * np.arccos(cos_half_gamma)

    # Handle edge cases
    if np.abs(np.sin(gamma/2)) < 1e-10:
        # gamma ≈ 0: U = Rz(β+δ)
        beta_plus_delta = -2 * np.angle(U_su2[0, 0])
        return global_phase, beta_plus_delta/2, 0, beta_plus_delta/2

    if np.abs(np.cos(gamma/2)) < 1e-10:
        # gamma ≈ π: U = Rz(β)·(-iY)·Rz(δ)
        beta_minus_delta = 2 * np.angle(U_su2[1, 0])
        return global_phase, beta_minus_delta/2, np.pi, -beta_minus_delta/2

    # General case
    # U[0,0] = cos(γ/2) exp(-i(β+δ)/2)
    # U[1,0] = sin(γ/2) exp(i(β-δ)/2)
    phase_00 = np.angle(U_su2[0, 0])
    phase_10 = np.angle(U_su2[1, 0])

    beta_plus_delta = -2 * phase_00
    beta_minus_delta = 2 * phase_10

    beta = (beta_plus_delta + beta_minus_delta) / 2
    delta = (beta_plus_delta - beta_minus_delta) / 2

    return global_phase, beta, gamma, delta

def verify_decomposition(U, alpha, beta, gamma, delta):
    """Verify ZYZ decomposition."""
    reconstructed = np.exp(1j*alpha) * Rz(beta) @ Ry(gamma) @ Rz(delta)
    return np.allclose(U, reconstructed)

# Test on various gates
print("\n1. ZYZ Decomposition of Standard Gates:")
print("-" * 60)

test_gates = [
    ('I', I),
    ('X', X),
    ('Y', Y),
    ('Z', Z),
    ('H', H),
    ('S', S),
    ('T', T),
    ('Rx(π/3)', Rx(np.pi/3)),
    ('Ry(π/4)', Ry(np.pi/4)),
    ('Rz(π/6)', Rz(np.pi/6)),
]

for name, gate in test_gates:
    alpha, beta, gamma, delta = zyz_decompose(gate)
    valid = verify_decomposition(gate, alpha, beta, gamma, delta)
    print(f"   {name:12}: α={alpha/np.pi:+.3f}π, β={beta/np.pi:+.3f}π, γ={gamma/np.pi:+.3f}π, δ={delta/np.pi:+.3f}π | Valid: {valid}")

# Alternative decompositions
print("\n" + "=" * 60)
print("ALTERNATIVE DECOMPOSITIONS")
print("=" * 60)

def xyx_decompose(U):
    """Decompose U as Rx(α)·Ry(β)·Rx(γ)."""
    # Transform to ZYZ problem: XYX = H·ZYZ·H (up to phases)
    # Actually, use rotation of coordinate system

    # The approach: note that Rx = H·Rz·H
    # So XYX = H·(ZYZ_transformed)·H
    # We can find ZYZ of H·U·H and transform back

    H_adj = H.conj().T
    U_transformed = H @ U @ H_adj

    alpha, beta, gamma, delta = zyz_decompose(U_transformed)

    return alpha, beta, gamma, delta

print("\n2. XYX Decomposition of H:")
alpha, beta, gamma, delta = xyx_decompose(H)
reconstructed = np.exp(1j*alpha) * Rx(beta) @ Ry(gamma) @ Rx(delta)
print(f"   α={alpha/np.pi:.3f}π, β={beta/np.pi:.3f}π, γ={gamma/np.pi:.3f}π, δ={delta/np.pi:.3f}π")
print(f"   Verified: {np.allclose(H, reconstructed)}")

# Native gate compilation
print("\n" + "=" * 60)
print("NATIVE GATE COMPILATION")
print("=" * 60)

# IBM native gates: √X (SX), Rz(θ), X
def SX():
    """√X gate (IBM native)."""
    return Rx(np.pi/2)

def compile_to_ibm(U):
    """
    Compile U to IBM native gates: Rz(θ), SX (√X).
    Uses: Ry(θ) = Rz(π/2)·SX·Rz(θ+π)·SX·Rz(-π/2)
    """
    alpha, beta, gamma, delta = zyz_decompose(U)

    # Ry(γ) = Rz(π/2) · SX · Rz(γ+π) · SX · Rz(-π/2)
    # So: Rz(β)·Ry(γ)·Rz(δ) = Rz(β)·Rz(π/2)·SX·Rz(γ+π)·SX·Rz(-π/2)·Rz(δ)
    #                        = Rz(β+π/2)·SX·Rz(γ+π)·SX·Rz(δ-π/2)

    gates = []
    if np.abs(beta + np.pi/2) > 1e-10:
        gates.append(('Rz', beta + np.pi/2))
    gates.append(('SX', None))
    if np.abs(gamma + np.pi) > 1e-10:
        gates.append(('Rz', gamma + np.pi))
    gates.append(('SX', None))
    if np.abs(delta - np.pi/2) > 1e-10:
        gates.append(('Rz', delta - np.pi/2))

    return alpha, gates

print("\n3. IBM Native Gate Compilation:")

for name, gate in [('H', H), ('T', T), ('S', S), ('Ry(π/4)', Ry(np.pi/4))]:
    alpha, ibm_gates = compile_to_ibm(gate)
    print(f"   {name}:")
    print(f"      Global phase: {alpha/np.pi:.3f}π")
    print(f"      Gates: ", end="")
    for g_name, g_param in ibm_gates:
        if g_param is not None:
            print(f"{g_name}({g_param/np.pi:.3f}π) ", end="")
        else:
            print(f"{g_name} ", end="")
    print()

# Gate approximation (discrete gates)
print("\n" + "=" * 60)
print("DISCRETE GATE APPROXIMATION")
print("=" * 60)

def generate_clifford_t_sequences(max_length):
    """Generate all sequences of H and T gates up to given length."""
    sequences = {0: [(I.copy(), [])]}  # (matrix, gate_list)

    for length in range(1, max_length + 1):
        sequences[length] = []
        for matrix, gates in sequences[length - 1]:
            # Add H
            new_matrix_h = H @ matrix
            sequences[length].append((new_matrix_h, gates + ['H']))
            # Add T
            new_matrix_t = T @ matrix
            sequences[length].append((new_matrix_t, gates + ['T']))
            # Add T†
            new_matrix_td = T.conj().T @ matrix
            sequences[length].append((new_matrix_td, gates + ['T†']))

    return sequences

def find_best_approximation(target, sequences):
    """Find the sequence that best approximates target."""
    best_fidelity = 0
    best_sequence = None

    for length, seq_list in sequences.items():
        for matrix, gates in seq_list:
            # Compute fidelity (up to global phase)
            fidelity = np.abs(np.trace(matrix.conj().T @ target))**2 / 4
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_sequence = (matrix, gates, length)

    return best_sequence, best_fidelity

print("\n4. Approximating Rz(π/8) with H and T gates:")

target = Rz(np.pi/8)
sequences = generate_clifford_t_sequences(8)

print(f"   Total sequences generated: {sum(len(s) for s in sequences.values())}")

best, fidelity = find_best_approximation(target, sequences)
matrix, gates, length = best

print(f"   Best approximation length: {length}")
print(f"   Gate sequence: {' → '.join(gates)}")
print(f"   Fidelity: {fidelity:.6f}")
print(f"   Error (1-F): {1-fidelity:.6f}")

# Solovay-Kitaev concepts
print("\n" + "=" * 60)
print("SOLOVAY-KITAEV CONCEPTS")
print("=" * 60)

def gate_distance(U, V):
    """Compute distance between unitaries (operator norm of difference)."""
    # Remove global phase
    phase = np.angle(np.trace(U.conj().T @ V))
    V_adj = V * np.exp(-1j * phase / 2)
    return np.linalg.norm(U - V_adj, ord=2)

def random_su2():
    """Generate random SU(2) matrix."""
    a = np.random.randn() + 1j * np.random.randn()
    b = np.random.randn() + 1j * np.random.randn()
    norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    a, b = a/norm, b/norm
    return np.array([[a, -np.conj(b)], [b, np.conj(a)]], dtype=complex)

print("\n5. Scaling of approximation error with sequence length:")

# Sample random targets and find approximations
np.random.seed(42)
n_samples = 20
lengths = range(2, 10)

avg_errors = []
for max_len in lengths:
    sequences = generate_clifford_t_sequences(max_len)
    errors = []
    for _ in range(n_samples):
        target = random_su2()
        best, fidelity = find_best_approximation(target, sequences)
        errors.append(1 - fidelity)
    avg_errors.append(np.mean(errors))
    print(f"   Length {max_len}: Avg error = {np.mean(errors):.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Error vs sequence length
axes[0].semilogy(list(lengths), avg_errors, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Maximum sequence length', fontsize=12)
axes[0].set_ylabel('Average approximation error (1-F)', fontsize=12)
axes[0].set_title('Approximation Error vs Gate Count\n(H and T gates)', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot 2: Decomposition visualization
ax2 = axes[1]

# Show ZYZ decomposition graphically for H
U = H
alpha, beta, gamma, delta = zyz_decompose(U)

# Create circuit diagram
circuit_elements = [
    f'Rz({beta/np.pi:.2f}π)',
    f'Ry({gamma/np.pi:.2f}π)',
    f'Rz({delta/np.pi:.2f}π)'
]

ax2.set_xlim(-0.5, 4)
ax2.set_ylim(-0.5, 1.5)

# Draw wire
ax2.plot([0, 3.5], [0.5, 0.5], 'k-', linewidth=2)

# Draw gates
for i, gate_text in enumerate(circuit_elements):
    rect = plt.Rectangle((i*1.2 + 0.2, 0.2), 0.8, 0.6, fill=True,
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(i*1.2 + 0.6, 0.5, gate_text, ha='center', va='center', fontsize=10)

ax2.text(1.8, 1.2, f'H = e^{{i{alpha/np.pi:.2f}π}} · Rz · Ry · Rz', ha='center', fontsize=14)
ax2.set_title('ZYZ Decomposition of Hadamard', fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.savefig('gate_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: gate_decomposition.png")

# Comparison of decomposition methods
print("\n" + "=" * 60)
print("DECOMPOSITION COMPARISON")
print("=" * 60)

print("\n6. Comparison of decomposition methods for random gates:")
print("-" * 70)
print(f"{'Gate':<15} {'ZYZ depth':<12} {'IBM gates':<12} {'H+T approx':<15}")
print("-" * 70)

for i in range(5):
    U = random_su2()

    # ZYZ depth (always 3 rotations)
    zyz_depth = 3

    # IBM native (count non-trivial gates)
    alpha, ibm_gates = compile_to_ibm(U)
    ibm_count = len(ibm_gates)

    # H+T approximation
    best, fidelity = find_best_approximation(U, generate_clifford_t_sequences(8))
    _, ht_gates, _ = best

    print(f"Random #{i+1:<9} {zyz_depth:<12} {ibm_count:<12} {len(ht_gates)} (F={fidelity:.4f})")

# T-count analysis
print("\n" + "=" * 60)
print("T-COUNT ANALYSIS")
print("=" * 60)

print("\n7. T-count for approximating standard gates:")

standard_targets = [
    ('Rz(π/8)', Rz(np.pi/8)),
    ('Rz(π/16)', Rz(np.pi/16)),
    ('Rx(π/8)', Rx(np.pi/8)),
]

sequences = generate_clifford_t_sequences(10)

for name, target in standard_targets:
    best, fidelity = find_best_approximation(target, sequences)
    _, gates, _ = best
    t_count = sum(1 for g in gates if g in ['T', 'T†'])
    h_count = sum(1 for g in gates if g == 'H')
    print(f"   {name}: T-count = {t_count}, H-count = {h_count}, Total = {len(gates)}, Fidelity = {fidelity:.4f}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| ZYZ decomposition | $U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$ |
| Angle γ | $\gamma = 2\arccos\|U_{00}\|$ |
| Angles β, δ | From phases of matrix elements |
| Solovay-Kitaev | $O(\log^c(1/\epsilon))$ gates for ε accuracy |
| Universal sets | {H, T} or {H, T, CNOT} |

### Main Takeaways

1. **Every single-qubit gate decomposes:** ZYZ (or XYX, etc.) Euler angles always exist
2. **Discrete approximation:** H and T can approximate any single-qubit gate
3. **Solovay-Kitaev efficiency:** Logarithmic gate count in precision
4. **Hardware compilation:** Translation to native gates is essential for execution
5. **Trade-offs exist:** Gate count vs. fidelity, different native sets

---

## Daily Checklist

- [ ] I can perform ZYZ decomposition on any single-qubit unitary
- [ ] I understand the Solovay-Kitaev theorem statement
- [ ] I can compile gates to different native instruction sets
- [ ] I understand why {H, T} is universal for single qubits
- [ ] I completed the gate compilation lab
- [ ] I solved at least 3 practice problems

---

## Preview of Day 567

Tomorrow is the **Week 81 Review**, where we consolidate all single-qubit gate concepts. We'll work through comprehensive problems covering Pauli gates, Hadamard, phase gates, rotations, Bloch sphere representation, and decomposition. This prepares us for Week 82's two-qubit gates.
