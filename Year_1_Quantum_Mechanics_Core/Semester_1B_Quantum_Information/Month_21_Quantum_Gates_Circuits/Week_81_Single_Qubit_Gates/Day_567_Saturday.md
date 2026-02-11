# Day 567: Week 81 Review - Single-Qubit Gates

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Comprehensive review and concept integration |
| Afternoon | 2.5 hours | Problem solving across all topics |
| Evening | 1.5 hours | Self-assessment and preparation for Week 82 |

## Learning Objectives

By the end of today, you will be able to:

1. **Synthesize all single-qubit gate concepts** into a unified framework
2. **Solve complex problems** combining multiple topics
3. **Identify connections** between different representations
4. **Demonstrate mastery** of computational implementations
5. **Prepare conceptually** for two-qubit gates
6. **Self-assess understanding** across all Week 81 material

---

## Week 81 Concept Map

```
                    Single-Qubit Gates
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    Discrete Gates    Continuous Gates   Representations
         │                 │                 │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │         │      │         │      │         │
  Pauli    Phase   Rotation   General  Bloch   Euler
  X,Y,Z    S,T    Rx,Ry,Rz   Rn(θ)   Sphere  Angles
    │         │      │         │      │         │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                 │
         └────────────────┼─────────────────┘
                          │
                   Gate Decomposition
                   ZYZ, Solovay-Kitaev
```

---

## Comprehensive Review

### 1. Pauli Gates (Day 561)

**Matrices:**
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Key Properties:**
- $X^2 = Y^2 = Z^2 = I$
- Anti-commutation: $\{\sigma_i, \sigma_j\} = 2\delta_{ij}I$
- Products: $XY = iZ$, cyclic
- All have eigenvalues ±1

### 2. Hadamard Gate (Day 562)

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Key Properties:**
- $H^2 = I$
- $H|0\rangle = |+\rangle$, $H|1\rangle = |-\rangle$
- $HXH = Z$, $HZH = X$, $HYH = -Y$

### 3. Phase Gates (Day 563)

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \quad T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

**Key Properties:**
- $S^2 = Z$, $T^2 = S$, $T^8 = I$
- Z-axis rotations
- Phase kickback mechanism

### 4. Rotation Gates (Day 564)

$$R_j(\theta) = e^{-i\theta\sigma_j/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}\sigma_j$$

**Key Properties:**
- Rodrigues formula
- Same-axis composition: $R_j(\theta_1)R_j(\theta_2) = R_j(\theta_1 + \theta_2)$
- Periodicity: $R_j(4\pi) = I$

### 5. Bloch Sphere (Day 565)

$$U = e^{i\alpha}R_{\hat{n}}(\theta)$$

**Key Properties:**
- Every single-qubit gate is a rotation
- SU(2) → SO(3) double cover
- Axis-angle extraction algorithms

### 6. Gate Decomposition (Day 566)

$$U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$$

**Key Properties:**
- ZYZ Euler decomposition
- Solovay-Kitaev approximation
- Hardware compilation

---

## Master Formula Sheet

| Gate | Matrix | Key Identity |
|------|--------|--------------|
| X | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip, X² = I |
| Y | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ | Y = iXZ |
| Z | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Phase flip, Z² = I |
| H | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | H² = I, HXH = Z |
| S | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ | S² = Z, √Z |
| T | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$ | T² = S, √S |
| Rx(θ) | $\begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$ | X = iRx(π) |
| Ry(θ) | $\begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$ | Real matrix |
| Rz(θ) | $\begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$ | P(θ) = e^{iθ/2}Rz(θ) |

---

## Comprehensive Practice Problems

### Section A: Fundamentals (15 points each)

**A1. Pauli Algebra**

Compute the following and simplify:
a) $XYZYX$
b) $[X, [Y, Z]]$ (nested commutator)
c) $e^{i\pi X/4}Ye^{-i\pi X/4}$

**A2. Hadamard Applications**

a) Find the state $|\psi\rangle = H|+_y\rangle$ where $|+_y\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$.
b) Verify that $H = \frac{X + Z}{\sqrt{2}}$ by computing the right side.
c) What is $H^{\otimes 3}|000\rangle$ in the computational basis?

**A3. Phase Gate Chains**

a) Compute $STST$ and express as a single gate.
b) If $|\psi\rangle = |+\rangle$, find $(ST)^4|\psi\rangle$.
c) Find the smallest n such that $(TS)^n = I$.

### Section B: Rotations and Bloch Sphere (20 points each)

**B1. Rotation Composition**

a) Show that $R_x(\pi/2)R_y(\pi/2)R_x(-\pi/2) = R_z(\pi/2)$ up to global phase.
b) Find the axis and angle of the combined rotation $R_x(\pi/4)R_z(\pi/4)$.
c) Prove: $R_y(\theta) = R_z(-\pi/2)R_x(\theta)R_z(\pi/2)$.

**B2. Bloch Sphere Analysis**

For the state $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}e^{i\pi/3}|1\rangle$:

a) Find the Bloch sphere coordinates (x, y, z).
b) What single rotation takes $|0\rangle$ to $|\psi\rangle$? Give axis and angle.
c) Find $\langle X \rangle$, $\langle Y \rangle$, $\langle Z \rangle$ for this state.

**B3. Fixed Points**

a) Find all states $|\psi\rangle$ such that $H|\psi\rangle = |\psi\rangle$.
b) Show that the eigenstates of H are the fixed points of Hadamard rotation on the Bloch sphere.
c) What is the Bloch sphere angle θ from the z-axis for these eigenstates?

### Section C: Decomposition and Synthesis (25 points each)

**C1. ZYZ Decomposition**

Decompose the following into ZYZ form:
a) $U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix}$
b) The gate $V = HTHT$

**C2. Gate Compilation**

Given native gates {Rz(θ), √X} (IBM-like):
a) Express H using only these gates.
b) Express S using only these gates.
c) What is the minimum gate count for Ry(π/4)?

**C3. Universality**

a) Prove that {H, T} can generate any rotation Rz(kπ/4) for integer k.
b) Show that H and Rz(θ) for irrational θ/π can approximate any single-qubit gate.
c) Why is T non-Clifford important for universality?

### Section D: Integration Problems (30 points each)

**D1. Complete Circuit Analysis**

Consider the circuit: $U = R_z(\alpha)HR_z(\beta)H$ for arbitrary α, β.

a) Simplify this to a single rotation $R_{\hat{n}}(\theta)$.
b) For what values of α, β is U = I?
c) Find α, β such that U = X.

**D2. Quantum State Preparation**

Design a circuit using only {H, S, T, Rz(θ)} to prepare:
$$|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}e^{i\pi/6}|1\rangle$$

starting from $|0\rangle$. Minimize gate count.

**D3. Error Analysis**

A gate U is implemented with a small error: $\tilde{U} = U \cdot R_z(\epsilon)$ where ε << 1.

a) What is the fidelity $F = |\langle\psi|\tilde{U}^\dagger U|\psi\rangle|^2$ for $|\psi\rangle = |+\rangle$?
b) After n applications, what is the cumulative error?
c) How does Bloch sphere geometry help visualize this error?

---

## Solutions to Selected Problems

### Solution A1a: Compute XYZYX

$$XYZYX = XY(ZYX) = XY(-XYZ) = -XYXYZ$$

Using $XYX = -Y$:
$$= -(-Y)YZ = Y^2Z = IZ = Z$$

**Answer:** $XYZYX = Z$

### Solution B1a: Rotation Identity

We need to show $R_x(\pi/2)R_y(\pi/2)R_x(-\pi/2) = R_z(\pi/2)$ (up to phase).

Using the identity $R_y(\theta) = R_x(-\pi/2)R_z(\theta)R_x(\pi/2)$:

$$R_x(\pi/2)R_y(\pi/2)R_x(-\pi/2) = R_x(\pi/2) \cdot R_x(-\pi/2)R_z(\pi/2)R_x(\pi/2) \cdot R_x(-\pi/2)$$
$$= R_z(\pi/2)$$

**Verified!**

### Solution C1a: ZYZ of U

$$U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix}$$

**Step 1:** $|U_{00}| = 1/\sqrt{2}$, so $\gamma = \pi/2$.

**Step 2:** $\arg(U_{00}) = 0$, $\arg(U_{10}) = \pi/2$

$$\beta = -0 + \pi/2 - \pi/2 = 0$$
$$\delta = -0 - \pi/2 + \pi/2 = 0$$

**Step 3:** Check: $R_z(0)R_y(\pi/2)R_z(0) = R_y(\pi/2)$

$$R_y(\pi/2) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

This doesn't match U exactly. We need global phase:
$$U = iR_y(\pi/2) = e^{i\pi/2}R_y(\pi/2)$$

**Answer:** $U = e^{i\pi/2}R_z(0)R_y(\pi/2)R_z(0)$, i.e., α = π/2, β = 0, γ = π/2, δ = 0.

### Solution D2: State Preparation

Target: $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}e^{i\pi/6}|1\rangle$

**Step 1:** Amplitude preparation using Ry.
Need $\cos(\theta/2) = 1/2$, so $\theta = 2\arccos(1/2) = 2\pi/3$.

$R_y(2\pi/3)|0\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$

**Step 2:** Phase adjustment using Rz.
Need to add phase $e^{i\pi/6}$ to $|1\rangle$.

$R_z(\pi/3)$ gives phases $e^{-i\pi/6}|0\rangle$ and $e^{i\pi/6}|1\rangle$.

But this adds a phase to $|0\rangle$ too. Use P(π/6) instead:
$$P(\pi/6)|0\rangle = |0\rangle, \quad P(\pi/6)|1\rangle = e^{i\pi/6}|1\rangle$$

**Circuit:** $|\psi\rangle = P(\pi/6) R_y(2\pi/3)|0\rangle$

Using available gates: $P(\phi) = e^{i\phi/2}R_z(\phi) = TS^{...}$ approximately.

**Minimum circuit:** Ry(2π/3) followed by Rz(π/6) (2 gates).

---

## Computational Lab: Week Review

```python
"""
Day 567: Week 81 Review - Comprehensive Single-Qubit Gate Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gate definitions
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

def Rx(theta): return np.cos(theta/2)*I - 1j*np.sin(theta/2)*X
def Ry(theta): return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Y
def Rz(theta): return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Z
def Rn(theta, n):
    n = np.array(n) / np.linalg.norm(n)
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*(n[0]*X + n[1]*Y + n[2]*Z)

ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

def state_to_bloch(psi):
    psi = psi.flatten()
    rho = np.outer(psi, psi.conj())
    return np.array([np.real(np.trace(P @ rho)) for P in [X, Y, Z]])

print("=" * 70)
print("WEEK 81 REVIEW: COMPREHENSIVE SINGLE-QUBIT GATE ANALYSIS")
print("=" * 70)

# Problem A1: Pauli Algebra
print("\n" + "=" * 70)
print("PROBLEM A1: PAULI ALGEBRA")
print("=" * 70)

# a) XYZYX
result_a = X @ Y @ Z @ Y @ X
print("\na) XYZYX = ")
print(result_a)
print(f"   This equals Z: {np.allclose(result_a, Z)}")

# b) [X, [Y, Z]]
inner_comm = Y @ Z - Z @ Y
outer_comm = X @ inner_comm - inner_comm @ X
print(f"\nb) [X, [Y, Z]] = ")
print(outer_comm)
print(f"   This equals -4X: {np.allclose(outer_comm, -4*X)}")

# c) Rotation conjugation
rot_x = Rx(np.pi/2)
result_c = rot_x @ Y @ rot_x.conj().T
print(f"\nc) Rx(π/2) Y Rx(-π/2) = ")
print(result_c)
print(f"   This equals Z (up to sign): {np.allclose(result_c, Z) or np.allclose(result_c, -Z)}")

# Problem A2: Hadamard Applications
print("\n" + "=" * 70)
print("PROBLEM A2: HADAMARD APPLICATIONS")
print("=" * 70)

# a) H|+y>
ket_plus_y = (ket_0 + 1j*ket_1) / np.sqrt(2)
result_2a = H @ ket_plus_y
print(f"\na) H|+y⟩ = {result_2a.flatten()}")

# b) Verify H = (X+Z)/√2
H_from_XZ = (X + Z) / np.sqrt(2)
print(f"\nb) H = (X+Z)/√2: {np.allclose(H, H_from_XZ)}")

# c) H⊗3|000>
H3 = np.kron(np.kron(H, H), H)
ket_000 = np.kron(np.kron(ket_0, ket_0), ket_0)
result_2c = H3 @ ket_000
print(f"\nc) H⊗3|000⟩ = equal superposition, all amplitudes = 1/√8 = {1/np.sqrt(8):.4f}")
print(f"   Actual amplitudes: {result_2c.flatten()}")

# Problem A3: Phase Gate Chains
print("\n" + "=" * 70)
print("PROBLEM A3: PHASE GATE CHAINS")
print("=" * 70)

# a) STST
result_3a = S @ T @ S @ T
print(f"\na) STST = ")
print(result_3a)
# Check if it's a power of T
for k in range(16):
    if np.allclose(result_3a, np.linalg.matrix_power(T, k)):
        print(f"   This equals T^{k}")
        break

# b) (ST)^4 on |+>
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ST = S @ T
ST4 = np.linalg.matrix_power(ST, 4)
result_3b = ST4 @ ket_plus
print(f"\nb) (ST)⁴|+⟩ = {result_3b.flatten()}")
print(f"   Bloch vector: {state_to_bloch(result_3b)}")

# c) Find n such that (TS)^n = I
TS = T @ S
for n in range(1, 33):
    if np.allclose(np.linalg.matrix_power(TS, n), I):
        print(f"\nc) Smallest n such that (TS)^n = I: n = {n}")
        break

# Problem B2: Bloch Sphere Analysis
print("\n" + "=" * 70)
print("PROBLEM B2: BLOCH SPHERE ANALYSIS")
print("=" * 70)

psi = (1/np.sqrt(3))*ket_0 + np.sqrt(2/3)*np.exp(1j*np.pi/3)*ket_1
bloch = state_to_bloch(psi)
print(f"\nState |ψ⟩ = (1/√3)|0⟩ + √(2/3)·e^(iπ/3)|1⟩")
print(f"\na) Bloch coordinates: x={bloch[0]:.4f}, y={bloch[1]:.4f}, z={bloch[2]:.4f}")

# b) Rotation to create this state
theta_rotation = 2 * np.arccos(1/np.sqrt(3))
phi_rotation = np.pi/3
print(f"\nb) Rotation from |0⟩: Ry({theta_rotation/np.pi:.4f}π) then Rz({phi_rotation/np.pi:.4f}π)")

# c) Expectation values (same as Bloch coordinates)
print(f"\nc) ⟨X⟩ = {bloch[0]:.4f}, ⟨Y⟩ = {bloch[1]:.4f}, ⟨Z⟩ = {bloch[2]:.4f}")

# Problem B3: Fixed Points
print("\n" + "=" * 70)
print("PROBLEM B3: FIXED POINTS OF HADAMARD")
print("=" * 70)

eigenvalues, eigenvectors = np.linalg.eig(H)
print("\na) Eigenstates of H (fixed points up to phase):")
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    bloch_eig = state_to_bloch(vec.reshape(-1, 1))
    theta_from_z = np.arccos(bloch_eig[2])
    print(f"   λ = {val.real:+.3f}: |v⟩ = {vec}")
    print(f"   Bloch: ({bloch_eig[0]:.3f}, {bloch_eig[1]:.3f}, {bloch_eig[2]:.3f})")
    print(f"   Angle from z-axis: {theta_from_z/np.pi:.3f}π rad = {np.degrees(theta_from_z):.1f}°")

# Comprehensive Visualization
print("\n" + "=" * 70)
print("COMPREHENSIVE VISUALIZATION")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))

def draw_bloch_sphere(ax, title=""):
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='r', alpha=0.3, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='g', alpha=0.3, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='b', alpha=0.3, arrow_length_ratio=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

# Plot 1: All standard gate axes
ax1 = fig.add_subplot(221, projection='3d')
draw_bloch_sphere(ax1, "Rotation Axes of Standard Gates")

gate_info = [
    ('X (Pauli)', [1, 0, 0], 'red', 'o'),
    ('Y (Pauli)', [0, 1, 0], 'green', 'o'),
    ('Z (Pauli)', [0, 0, 1], 'blue', 'o'),
    ('H', [1/np.sqrt(2), 0, 1/np.sqrt(2)], 'purple', 's'),
    ('S', [0, 0, 1], 'cyan', '^'),
]

for name, axis, color, marker in gate_info:
    axis = np.array(axis)
    ax1.quiver(0, 0, 0, axis[0], axis[1], axis[2],
               color=color, linewidth=2, arrow_length_ratio=0.2)
    ax1.scatter([axis[0]], [axis[1]], [axis[2]], c=color, s=50, marker=marker, label=name)
ax1.legend(loc='upper left', fontsize=8)

# Plot 2: Gate action comparison
ax2 = fig.add_subplot(222, projection='3d')
draw_bloch_sphere(ax2, "Gate Actions on |0⟩")

# Apply various gates to |0⟩
gates_to_apply = [
    ('|0⟩', I, 'black'),
    ('X|0⟩', X, 'red'),
    ('H|0⟩', H, 'purple'),
    ('S|0⟩', S, 'cyan'),
    ('T|0⟩', T, 'orange'),
    ('Ry(π/4)|0⟩', Ry(np.pi/4), 'green'),
]

for name, gate, color in gates_to_apply:
    state = gate @ ket_0
    bloch = state_to_bloch(state)
    ax2.scatter(*bloch, c=color, s=100, label=name)
ax2.legend(loc='upper left', fontsize=8)

# Plot 3: Rotation trajectories
ax3 = fig.add_subplot(223, projection='3d')
draw_bloch_sphere(ax3, "Rotation Trajectories from |0⟩")

angles = np.linspace(0, 2*np.pi, 50)
for rot_name, rot_func, color in [('Rx', Rx, 'red'), ('Ry', Ry, 'green'), ('Rz', Rz, 'blue')]:
    trajectory = np.array([state_to_bloch(rot_func(a) @ ket_0) for a in angles])
    ax3.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
             color=color, linewidth=2, label=rot_name)
ax3.legend()

# Plot 4: Decomposition example
ax4 = fig.add_subplot(224)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)

# Circuit diagram for H = Rz(π)Ry(π/2)
ax4.plot([1, 9], [4, 4], 'k-', linewidth=2)  # Wire

def draw_gate(ax, x, name, color='lightblue'):
    rect = plt.Rectangle((x-0.4, 3.5), 0.8, 1, fill=True,
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, 4, name, ha='center', va='center', fontsize=10)

draw_gate(ax4, 2, 'Rz(π)', 'lightblue')
draw_gate(ax4, 4, 'Ry(π/2)', 'lightgreen')
ax4.text(5.5, 4, '=', fontsize=20, ha='center', va='center')
draw_gate(ax4, 7, 'H', 'lightyellow')

ax4.text(5, 2, 'ZYZ Decomposition of Hadamard\n(up to global phase)', ha='center', fontsize=12)
ax4.text(5, 1, 'H ≈ e^(iα) Rz(π) Ry(π/2) Rz(0)', ha='center', fontsize=11, family='monospace')

ax4.axis('off')
ax4.set_title('Gate Decomposition Example')

plt.tight_layout()
plt.savefig('week81_review.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: week81_review.png")

# Summary statistics
print("\n" + "=" * 70)
print("WEEK 81 SUMMARY STATISTICS")
print("=" * 70)

print("\nGate Properties Summary:")
print("-" * 50)
gates_summary = [
    ('X', X), ('Y', Y), ('Z', Z), ('H', H), ('S', S), ('T', T),
    ('Rx(π/2)', Rx(np.pi/2)), ('Ry(π/2)', Ry(np.pi/2)), ('Rz(π/2)', Rz(np.pi/2))
]

for name, gate in gates_summary:
    det = np.linalg.det(gate)
    trace = np.trace(gate)
    eigenvals = np.linalg.eigvals(gate)
    is_hermitian = np.allclose(gate, gate.conj().T)
    print(f"{name:12}: det={det:.3f}, tr={trace:.3f}, Hermitian={is_hermitian}")

print("\n" + "=" * 70)
print("SELF-ASSESSMENT CHECKLIST")
print("=" * 70)

checklist = [
    "Can write all Pauli matrices from memory",
    "Understand anti-commutation relations",
    "Can apply Hadamard for basis change",
    "Know the S, T hierarchy (T²=S, S²=Z)",
    "Can derive rotation gates from exponential form",
    "Understand Bloch sphere representation",
    "Can perform ZYZ decomposition",
    "Understand phase kickback",
    "Know what makes a gate set universal",
    "Can visualize gate actions geometrically"
]

print("\nRate your understanding (1-5) for each:")
for i, item in enumerate(checklist, 1):
    print(f"   {i:2}. [ ] {item}")

print("\n" + "=" * 70)
print("PREPARATION FOR WEEK 82: TWO-QUBIT GATES")
print("=" * 70)

print("""
Key concepts to carry forward:
1. Tensor products: |ψ⟩ ⊗ |φ⟩ for multi-qubit states
2. Matrix representation of two-qubit gates (4×4 matrices)
3. Entanglement: states that cannot be written as tensor products
4. Controlled operations: action depends on control qubit
5. The CNOT gate as the canonical two-qubit entangling gate

Coming up in Week 82:
- Day 568: CNOT Gate
- Day 569: Controlled Gates (CZ, CU)
- Day 570: SWAP and √SWAP
- Day 571: Entangling Power
- Day 572: Gate Identities
- Day 573: Tensor Products
- Day 574: Week Review
""")
```

---

## Self-Assessment Rubric

### Mastery Levels

| Level | Description | Criteria |
|-------|-------------|----------|
| Expert | Complete mastery | Solve all problems, explain concepts to others |
| Proficient | Strong understanding | Solve most problems, minor gaps |
| Developing | Partial understanding | Solve basic problems, need review |
| Beginning | Needs significant work | Struggle with fundamentals |

### Topic Checklist

Rate yourself 1-5 on each topic:

- [ ] Pauli gates (X, Y, Z): ___
- [ ] Hadamard gate: ___
- [ ] Phase gates (S, T): ___
- [ ] Rotation gates (Rx, Ry, Rz): ___
- [ ] Bloch sphere representation: ___
- [ ] ZYZ decomposition: ___
- [ ] Solovay-Kitaev concepts: ___
- [ ] Gate compilation: ___

**Total Score: ___/40**
- 36-40: Expert
- 28-35: Proficient
- 20-27: Developing
- Below 20: Beginning

---

## Summary

### Week 81 Key Achievements

1. **Pauli gates:** Foundation of quantum error types and observables
2. **Hadamard:** The superposition creator and basis transformer
3. **Phase gates:** Z-axis control and the Clifford hierarchy
4. **Rotations:** Continuous gate parameterization
5. **Bloch sphere:** Complete geometric understanding
6. **Decomposition:** Compiling arbitrary gates to native sets

### Connections to Remember

- All single-qubit gates are rotations on the Bloch sphere
- {H, T} generates a dense subgroup - universal for approximation
- ZYZ decomposition always works (Euler angle theorem)
- Phase kickback transfers eigenvalue information to control qubits

---

## Preview of Week 82

Next week we extend to **two-qubit gates**, where the magic of quantum computing really begins:

- **CNOT:** The canonical entangling gate
- **Controlled operations:** Conditional quantum logic
- **SWAP family:** Qubit exchange operations
- **Entangling power:** Quantifying non-local correlations
- **Gate identities:** Circuit optimization techniques
- **Tensor products:** Multi-qubit mathematics

The combination of single-qubit gates (Week 81) with two-qubit entangling gates (Week 82) gives us **universal quantum computation**!
