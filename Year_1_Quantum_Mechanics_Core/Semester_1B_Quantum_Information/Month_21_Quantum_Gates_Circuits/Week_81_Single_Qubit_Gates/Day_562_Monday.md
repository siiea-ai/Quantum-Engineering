# Day 562: The Hadamard Gate

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: Hadamard matrix, superposition creation, basis change |
| Afternoon | 2.5 hours | Problem solving: Hadamard circuits and identities |
| Evening | 1.5 hours | Computational lab: Quantum parallelism simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Write the Hadamard matrix** and verify its unitarity
2. **Create superposition states** from computational basis using H
3. **Transform between bases** (Z-basis ↔ X-basis) using H
4. **Prove H² = I** and understand its implications
5. **Derive key identities** like HXH = Z and HZH = X
6. **Apply Hadamard to multi-qubit states** using tensor products

---

## Core Content

### 1. The Hadamard Gate

The **Hadamard gate** is arguably the most important single-qubit gate in quantum computing:

$$\boxed{H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}}$$

**Key properties at a glance:**
- Creates superposition from computational basis states
- Self-inverse: $H^2 = I$
- Transforms between Z-basis and X-basis
- Essential for quantum parallelism

### 2. Action on Computational Basis

$$\boxed{H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle}$$

$$\boxed{H|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |-\rangle}$$

The Hadamard creates an equal superposition of both basis states, with a relative phase of 0 or π.

**Matrix verification:**
$$H|0\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = |+\rangle \checkmark$$

### 3. Hadamard as Basis Change

The Hadamard transforms between the **computational basis** (Z-eigenstates) and the **Hadamard basis** (X-eigenstates):

| Z-basis → X-basis | X-basis → Z-basis |
|-------------------|-------------------|
| $H\|0\rangle = \|+\rangle$ | $H\|+\rangle = \|0\rangle$ |
| $H\|1\rangle = \|-\rangle$ | $H\|-\rangle = \|1\rangle$ |

This is why H is called a **basis change** or **basis rotation** operator.

**General state transformation:**
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$
$$H|\psi\rangle = \alpha|+\rangle + \beta|-\rangle = \frac{\alpha + \beta}{\sqrt{2}}|0\rangle + \frac{\alpha - \beta}{\sqrt{2}}|1\rangle$$

### 4. Self-Inverse Property: H² = I

$$\boxed{H^2 = H \cdot H = I}$$

**Proof:**
$$H^2 = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I$$

**Implication:** H is its own inverse. Applying H twice returns to the original state:
$$H(H|\psi\rangle) = |\psi\rangle$$

This is crucial for quantum algorithms: we can "undo" superposition.

### 5. Unitarity Verification

A matrix U is unitary if $U^\dagger U = I$.

For H:
$$H^\dagger = H^T = H$$ (H is Hermitian!)

Therefore:
$$H^\dagger H = H \cdot H = H^2 = I \checkmark$$

**H is both Hermitian and unitary!** This is a special property shared with the Pauli matrices.

### 6. Key Identities: Conjugation Relations

The Hadamard "conjugates" Pauli operators, exchanging X and Z:

$$\boxed{HXH = Z}$$
$$\boxed{HZH = X}$$
$$\boxed{HYH = -Y}$$

**Proof of HXH = Z:**
$$HXH = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

First, compute HX:
$$HX = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$$

Then, compute (HX)H:
$$HXH = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = Z$$

**Physical interpretation:** Conjugating an operator by H swaps its action in the X and Z directions on the Bloch sphere.

### 7. Alternative Forms of Hadamard

**Exponential form:**
$$H = \frac{1}{\sqrt{2}}(X + Z) = e^{i\pi/2} \cdot e^{i\pi(X+Z)/2\sqrt{2}}$$

More usefully:
$$H = e^{i\pi/2} R_y(\pi/2) R_z(\pi)$$

**As a rotation:**
$$H = \frac{X + Z}{\sqrt{2}} = \frac{1}{\sqrt{2}}(\sigma_x + \sigma_z)$$

The Hadamard corresponds to a π rotation about the axis $\hat{n} = (\frac{1}{\sqrt{2}}, 0, \frac{1}{\sqrt{2}})$, which is the diagonal between X and Z on the Bloch sphere.

### 8. Eigenvalues and Eigenvectors

Since H² = I, the eigenvalues must satisfy λ² = 1, so λ = ±1.

**Eigenvalue +1:**
$$H|v_+\rangle = |v_+\rangle$$

Solving $(H - I)|v_+\rangle = 0$:
$$|v_+\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$$

**Eigenvalue -1:**
$$|v_-\rangle = \sin(\pi/8)|0\rangle - \cos(\pi/8)|1\rangle$$

These eigenstates lie along the axis of the Hadamard rotation on the Bloch sphere.

### 9. Hadamard and Quantum Parallelism

When applied to multiple qubits, Hadamard creates **superposition over all computational basis states**:

For n qubits:
$$H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$$

**Example (2 qubits):**
$$(H \otimes H)|00\rangle = |+\rangle|+\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$

This is the starting point for almost every quantum algorithm!

### 10. Hadamard in Quantum Algorithms

| Algorithm | Role of Hadamard |
|-----------|------------------|
| Deutsch-Jozsa | Creates initial superposition, enables interference |
| Bernstein-Vazirani | Initial and final basis transformation |
| Simon's Algorithm | Creates superposition, extracts hidden structure |
| Grover's Search | Initial superposition, diffusion operator |
| Quantum Fourier Transform | Modified Hadamard-like structure |
| Shor's Algorithm | Via QFT for period finding |

---

## Quantum Computing Connection

The Hadamard gate is the gateway to quantum parallelism:

1. **Superposition creation:** H transforms deterministic classical states into quantum superpositions
2. **Interference setup:** By creating coherent superpositions, H enables constructive and destructive interference
3. **Basis measurement:** Applying H before measurement allows probing X-basis properties
4. **Quantum speedup:** The exponential superposition from $H^{\otimes n}$ is what enables quantum algorithms to explore exponentially many inputs simultaneously

---

## Worked Examples

### Example 1: Creating and Measuring Superposition

**Problem:** Starting with |0⟩, apply H, then measure in the computational basis. What are the probabilities?

**Solution:**

Initial state: $|0\rangle$

After Hadamard:
$$H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$$

Measurement probabilities:
$$P(0) = |\langle 0|+\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$
$$P(1) = |\langle 1|+\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

The measurement outcome is completely random: 50% |0⟩, 50% |1⟩.

### Example 2: Hadamard Sandwich

**Problem:** Compute HZH algebraically and verify it equals X.

**Solution:**

We compute step by step:

$$HZ = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

$$(HZ)H = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 0 & 2 \\ 2 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = X$$

**Verified!** This identity is crucial: to implement X errors in the X-basis, we can use Z errors sandwiched by Hadamards.

### Example 3: Two-Qubit Superposition

**Problem:** Compute $(H \otimes H)|01\rangle$.

**Solution:**

Using the tensor product:
$$(H \otimes H)|01\rangle = (H|0\rangle) \otimes (H|1\rangle) = |+\rangle \otimes |-\rangle$$

Expanding:
$$= \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

$$= \frac{1}{2}(|0\rangle|0\rangle - |0\rangle|1\rangle + |1\rangle|0\rangle - |1\rangle|1\rangle)$$

$$= \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$$

---

## Practice Problems

### Direct Application

1. Compute $H|+\rangle$ and $H|-\rangle$ directly using matrix multiplication.

2. Verify that H is Hermitian: show $H^* = H^T = H$ (where * denotes complex conjugate).

3. Calculate $(H \otimes I)|10\rangle$ where the Hadamard acts only on the first qubit.

### Intermediate

4. **Circuit identity:** Prove that $H = \frac{1}{\sqrt{2}}(X + Z)$ by computing the right-hand side.

5. **Eigenvector verification:** Verify that $|v_+\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$ is an eigenvector of H with eigenvalue +1.

6. Prove that $HYH = -Y$ using matrix multiplication.

### Challenging

7. **General Hadamard identity:** For any single-qubit unitary U, show that $HUH^\dagger = HUH$ is another valid unitary. If U = Rx(θ), what is HRx(θ)H?

8. **Interference pattern:** Start with $|0\rangle$, apply H, then apply a phase gate $|0\rangle \mapsto |0\rangle, |1\rangle \mapsto e^{i\phi}|1\rangle$, then apply H again. Find the probability of measuring |0⟩ as a function of φ. When is this probability maximized/minimized?

9. **Hadamard transform:** The n-qubit Hadamard transform is $H^{\otimes n}$. Show that:
   $$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1}(-1)^{x \cdot y}|y\rangle$$
   where $x \cdot y = \sum_i x_i y_i \pmod{2}$ is the bitwise inner product.

---

## Computational Lab: Hadamard Gate and Quantum Parallelism

```python
"""
Day 562: Hadamard Gate Simulation
Exploring superposition creation, basis change, and quantum parallelism
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Define gates and states
I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print("=" * 60)
print("HADAMARD GATE: FUNDAMENTAL PROPERTIES")
print("=" * 60)

# 1. Basic matrix properties
print("\n1. Hadamard Matrix:")
print(f"   H = \n{H}")
print(f"\n   H is real: {np.allclose(H, H.real)}")
print(f"   H is symmetric: {np.allclose(H, H.T)}")
print(f"   H is Hermitian: {np.allclose(H, H.conj().T)}")

# 2. Verify H² = I
print("\n2. Self-inverse property H² = I:")
H_squared = H @ H
print(f"   H² = \n{H_squared}")
print(f"   H² = I: {np.allclose(H_squared, I)}")

# 3. Unitarity
print("\n3. Unitarity verification:")
print(f"   H†H = I: {np.allclose(H.conj().T @ H, I)}")
print(f"   HH† = I: {np.allclose(H @ H.conj().T, I)}")

# 4. Action on basis states
print("\n4. Action on computational basis:")
H_ket0 = H @ ket_0
H_ket1 = H @ ket_1
print(f"   H|0⟩ = {H_ket0.flatten()} = |+⟩")
print(f"   H|1⟩ = {H_ket1.flatten()} = |-⟩")

# 5. Action on Hadamard basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)
print("\n5. Action on Hadamard basis (inverse direction):")
print(f"   H|+⟩ = {(H @ ket_plus).flatten()} = |0⟩: {np.allclose(H @ ket_plus, ket_0)}")
print(f"   H|-⟩ = {(H @ ket_minus).flatten()} = |1⟩: {np.allclose(H @ ket_minus, ket_1)}")

# 6. Conjugation relations
print("\n6. Conjugation relations (Pauli transformations):")
print(f"   HXH = Z: {np.allclose(H @ X @ H, Z)}")
print(f"   HZH = X: {np.allclose(H @ Z @ H, X)}")
print(f"   HYH = -Y: {np.allclose(H @ Y @ H, -Y)}")

# 7. Eigenvalues and eigenvectors
print("\n7. Eigenvalue decomposition:")
eigenvalues, eigenvectors = np.linalg.eig(H)
print(f"   Eigenvalues: {eigenvalues}")
print(f"   (Should be ±1 since H² = I)")
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f"   Eigenvector for λ={val.real:.1f}: {vec}")
    # Verify
    print(f"      H|v⟩ = λ|v⟩: {np.allclose(H @ vec.reshape(-1,1), val * vec.reshape(-1,1))}")

# 8. Hadamard as rotation
print("\n" + "=" * 60)
print("HADAMARD AS ROTATION ON BLOCH SPHERE")
print("=" * 60)

# Axis of rotation for H: (1/√2, 0, 1/√2)
print("\nH = π rotation about axis n̂ = (1/√2, 0, 1/√2)")
print("This is the diagonal between X and Z axes.")

# Verify: H = exp(-iπ(X+Z)/(2√2)) up to global phase
# Actually H = (X + Z)/√2

H_from_XZ = (X + Z) / np.sqrt(2)
print(f"\nVerify H = (X + Z)/√2: {np.allclose(H, H_from_XZ)}")

# 9. Multi-qubit Hadamard
print("\n" + "=" * 60)
print("MULTI-QUBIT HADAMARD AND QUANTUM PARALLELISM")
print("=" * 60)

def tensor_product(*matrices):
    """Compute tensor product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def ket(bitstring):
    """Create computational basis state from bitstring."""
    state = np.array([[1]], dtype=complex)
    for bit in bitstring:
        if bit == '0':
            state = np.kron(state, ket_0)
        else:
            state = np.kron(state, ket_1)
    return state

# Two-qubit Hadamard
H2 = tensor_product(H, H)
print("\nTwo-qubit Hadamard H⊗H on |00⟩:")
state_00 = ket('00')
result = H2 @ state_00
print(f"   (H⊗H)|00⟩ = 1/2 × (|00⟩ + |01⟩ + |10⟩ + |11⟩)")
print(f"   Amplitudes: {result.flatten()}")
print(f"   All equal to 1/2: {np.allclose(result, np.ones((4,1))/2)}")

# Three-qubit Hadamard
print("\nThree-qubit Hadamard H⊗³ on |000⟩:")
H3 = tensor_product(H, H, H)
state_000 = ket('000')
result_3 = H3 @ state_000
print(f"   Creates equal superposition of all 8 basis states")
print(f"   Each amplitude = 1/√8 = {1/np.sqrt(8):.4f}")
print(f"   Actual amplitudes: {result_3.flatten()[:4]}... (first 4 shown)")

# 10. Hadamard transform formula
print("\n" + "=" * 60)
print("HADAMARD TRANSFORM FORMULA")
print("=" * 60)

def hadamard_transform_formula(x, n):
    """
    Compute H⊗n|x⟩ using the formula:
    H⊗n|x⟩ = (1/√2^n) Σ_y (-1)^(x·y) |y⟩
    """
    N = 2**n
    result = np.zeros((N, 1), dtype=complex)

    for y in range(N):
        # Compute x · y (bitwise inner product mod 2)
        dot_product = bin(x & y).count('1') % 2
        result[y] = ((-1)**dot_product) / np.sqrt(N)

    return result

print("\nVerifying H⊗n|x⟩ = (1/√2^n) Σ_y (-1)^(x·y) |y⟩")
for n in range(1, 4):
    Hn = np.eye(1)
    for _ in range(n):
        Hn = np.kron(Hn, H)

    print(f"\n   n = {n} qubits:")
    for x in range(min(4, 2**n)):
        # Create |x⟩
        x_binary = format(x, f'0{n}b')
        ket_x = ket(x_binary)

        # Compute via matrix multiplication
        result_matrix = Hn @ ket_x

        # Compute via formula
        result_formula = hadamard_transform_formula(x, n)

        match = np.allclose(result_matrix, result_formula)
        print(f"      |{x_binary}⟩: Formula matches matrix: {match}")

# 11. Interference pattern
print("\n" + "=" * 60)
print("INTERFERENCE WITH HADAMARD")
print("=" * 60)

def phase_gate(phi):
    """Phase gate |0⟩→|0⟩, |1⟩→e^(iφ)|1⟩"""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

print("\nCircuit: |0⟩ → H → Phase(φ) → H → Measure")
print("Probability of |0⟩ as function of φ:")

phi_values = np.linspace(0, 2*np.pi, 100)
prob_0_values = []

for phi in phi_values:
    P = phase_gate(phi)
    final_state = H @ P @ H @ ket_0
    prob_0 = np.abs(final_state[0, 0])**2
    prob_0_values.append(prob_0)

# Plot interference pattern
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(phi_values / np.pi, prob_0_values, 'b-', linewidth=2)
ax.set_xlabel('Phase φ (units of π)', fontsize=12)
ax.set_ylabel('P(|0⟩)', fontsize=12)
ax.set_title('Quantum Interference: H → Phase(φ) → H', fontsize=14)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Classical')
ax.set_xlim([0, 2])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
ax.legend()

# Mark special points
special_phis = [0, 0.5, 1, 1.5, 2]
for sp in special_phis:
    idx = int(sp / 2 * 99)
    ax.plot(sp, prob_0_values[idx], 'ro', markersize=8)
    ax.annotate(f'φ={sp}π\nP={prob_0_values[idx]:.2f}',
                xy=(sp, prob_0_values[idx]),
                xytext=(sp+0.1, prob_0_values[idx]+0.1),
                fontsize=9)

plt.tight_layout()
plt.savefig('hadamard_interference.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: hadamard_interference.png")

print("\nAnalytical result: P(|0⟩) = cos²(φ/2)")
print("   φ = 0: P = 1 (constructive interference)")
print("   φ = π: P = 0 (destructive interference)")

# 12. Visualization of Hadamard action on Bloch sphere
print("\n" + "=" * 60)
print("BLOCH SPHERE VISUALIZATION")
print("=" * 60)

def state_to_bloch(psi):
    """Convert pure state to Bloch coordinates."""
    psi = psi.flatten()
    # Compute density matrix
    rho = np.outer(psi, psi.conj())
    # Extract Bloch components
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    return np.array([x, y, z])

fig = plt.figure(figsize=(12, 5))

# Generate test states
test_states = []
for theta in np.linspace(0, np.pi, 7):
    for phi in np.linspace(0, 2*np.pi, 13)[:-1]:
        state = np.cos(theta/2) * ket_0 + np.exp(1j*phi) * np.sin(theta/2) * ket_1
        test_states.append(state)

# Plot 1: Original and H-transformed states
ax1 = fig.add_subplot(121, projection='3d')

# Draw Bloch sphere
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Draw rotation axis for H
axis = np.array([1, 0, 1]) / np.sqrt(2)
ax1.quiver(0, 0, 0, axis[0]*1.5, axis[1]*1.5, axis[2]*1.5,
           color='purple', linewidth=2, arrow_length_ratio=0.1, label='H rotation axis')

# Plot transformations
for state in test_states:
    orig = state_to_bloch(state)
    transformed = state_to_bloch(H @ state)
    ax1.scatter(*orig, c='blue', s=15, alpha=0.4)
    ax1.scatter(*transformed, c='red', s=15, alpha=0.4)
    # Draw thin line connecting them
    ax1.plot([orig[0], transformed[0]], [orig[1], transformed[1]],
             [orig[2], transformed[2]], 'g-', alpha=0.1, linewidth=0.5)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Hadamard: Blue→Red\n(π rotation about (1,0,1)/√2)')

# Plot 2: Effect on computational basis and Hadamard basis
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Special states
special = [
    (ket_0, '|0⟩', 'blue'),
    (ket_1, '|1⟩', 'blue'),
    (ket_plus, '|+⟩', 'red'),
    (ket_minus, '|-⟩', 'red'),
]

for state, name, color in special:
    bloch = state_to_bloch(state)
    ax2.scatter(*bloch, c=color, s=100, marker='o')
    ax2.text(bloch[0]*1.2, bloch[1]*1.2, bloch[2]*1.2, name, fontsize=10)

    # Show transformation
    transformed = H @ state
    trans_bloch = state_to_bloch(transformed)
    ax2.quiver(bloch[0], bloch[1], bloch[2],
               trans_bloch[0]-bloch[0], trans_bloch[1]-bloch[1], trans_bloch[2]-bloch[2],
               color='green', arrow_length_ratio=0.2)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('H swaps Z-basis ↔ X-basis\nBlue: Z-basis, Red: X-basis')

plt.tight_layout()
plt.savefig('hadamard_bloch.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: hadamard_bloch.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Hadamard matrix | $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ |
| On $\|0\rangle$ | $H\|0\rangle = \|+\rangle = \frac{1}{\sqrt{2}}(\|0\rangle + \|1\rangle)$ |
| On $\|1\rangle$ | $H\|1\rangle = \|-\rangle = \frac{1}{\sqrt{2}}(\|0\rangle - \|1\rangle)$ |
| Self-inverse | $H^2 = I$ |
| X-Z exchange | $HXH = Z$, $HZH = X$, $HYH = -Y$ |
| Alternative form | $H = \frac{X + Z}{\sqrt{2}}$ |
| n-qubit superposition | $H^{\otimes n}\|0\rangle^n = \frac{1}{\sqrt{2^n}}\sum_{x}\|x\rangle$ |

### Main Takeaways

1. **Superposition creator:** H is the primary tool for creating superposition from classical states
2. **Basis transformer:** H converts between Z-basis (computational) and X-basis (Hadamard)
3. **Self-inverse:** Applying H twice returns to the original state
4. **Pauli conjugation:** H swaps X ↔ Z under conjugation
5. **Quantum parallelism:** $H^{\otimes n}$ creates superposition over $2^n$ states, enabling exponential quantum advantage

---

## Daily Checklist

- [ ] I can write the Hadamard matrix from memory
- [ ] I understand H as a basis change between Z and X eigenstates
- [ ] I can verify H² = I algebraically
- [ ] I can prove HXH = Z using matrix multiplication
- [ ] I understand quantum parallelism via n-qubit Hadamard
- [ ] I completed the interference pattern simulation
- [ ] I solved at least 3 practice problems

---

## Preview of Day 563

Tomorrow we explore **phase gates** S and T. These gates rotate about the Z-axis by π/2 and π/4 respectively. We'll see how they relate to Z through square roots (S = √Z, T = √S) and discover the crucial concept of **phase kickback**, which is fundamental to quantum algorithms like phase estimation.
