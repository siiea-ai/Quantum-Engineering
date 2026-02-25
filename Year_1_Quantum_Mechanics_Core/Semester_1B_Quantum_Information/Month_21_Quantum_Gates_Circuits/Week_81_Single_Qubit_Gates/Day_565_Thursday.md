# Day 565: Bloch Sphere Representation

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: Axis-angle representation, SU(2) structure |
| Afternoon | 2.5 hours | Problem solving: Converting between representations |
| Evening | 1.5 hours | Computational lab: Interactive Bloch sphere visualization |

## Learning Objectives

By the end of today, you will be able to:

1. **Prove any U ∈ SU(2)** can be written as $U = e^{i\alpha}R_{\hat{n}}(\theta)$
2. **Extract axis and angle** from a given unitary matrix
3. **Understand the SU(2) manifold** structure and its topology
4. **Convert between parameterizations** (matrix, axis-angle, Euler angles)
5. **Visualize gate actions** geometrically on the Bloch sphere
6. **Apply the double-cover relationship** between SU(2) and SO(3)

---

## Core Content

### 1. The Bloch Sphere: Review and Motivation

Every pure single-qubit state can be represented on the Bloch sphere:

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

This maps to the point $(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ on the unit sphere.

**Key insight:** The Bloch sphere visualizes both **states** and **operations**:
- States: Points on the sphere
- Operations: Rotations of the sphere

### 2. Single-Qubit Unitaries: The Group U(2)

The group of 2×2 unitary matrices is:
$$U(2) = \{U \in \mathbb{C}^{2\times 2} : U^\dagger U = I\}$$

Every U ∈ U(2) can be factored as:
$$U = e^{i\alpha}V$$
where V ∈ SU(2) (special unitary: det(V) = 1) and α ∈ [0, 2π).

The global phase $e^{i\alpha}$ is physically unobservable for isolated gates.

### 3. The Special Unitary Group SU(2)

$$SU(2) = \{U \in U(2) : \det(U) = 1\}$$

**Dimension:** SU(2) is a 3-dimensional manifold (3 real parameters).

**General form:** Any V ∈ SU(2) can be written as:

$$\boxed{V = \begin{pmatrix} a & -b^* \\ b & a^* \end{pmatrix}, \quad |a|^2 + |b|^2 = 1}$$

where a, b ∈ ℂ. This gives 4 real parameters constrained by 1 equation = 3 free parameters.

### 4. The Axis-Angle Theorem

**Theorem:** Every V ∈ SU(2) can be written as:

$$\boxed{V = R_{\hat{n}}(\theta) = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}(\hat{n}\cdot\vec{\sigma})}$$

for some unit vector $\hat{n} = (n_x, n_y, n_z)$ and angle $\theta \in [0, 2\pi]$.

**Proof:**

Let $V = \begin{pmatrix} a & -b^* \\ b & a^* \end{pmatrix}$ with $|a|^2 + |b|^2 = 1$.

Write $a = a_r + ia_i$ and $b = b_r + ib_i$ (real and imaginary parts).

Define:
$$\cos\frac{\theta}{2} = a_r, \quad \sin\frac{\theta}{2} = \sqrt{a_i^2 + b_r^2 + b_i^2}$$

If $\sin(\theta/2) \neq 0$, define:
$$n_x = \frac{b_i}{\sin(\theta/2)}, \quad n_y = \frac{b_r}{\sin(\theta/2)}, \quad n_z = \frac{a_i}{\sin(\theta/2)}$$

Then $|\hat{n}|^2 = 1$ (follows from $|a|^2 + |b|^2 = 1$).

Substituting into $R_{\hat{n}}(\theta)$ recovers V. ∎

### 5. Geometric Interpretation

**Every single-qubit gate is a rotation!**

For any gate U = e^{iα}V:
- V rotates the Bloch sphere by angle θ about axis $\hat{n}$
- The global phase α is unobservable

**Pauli gates:**
| Gate | Axis $\hat{n}$ | Angle θ |
|------|----------------|---------|
| X | (1, 0, 0) | π |
| Y | (0, 1, 0) | π |
| Z | (0, 0, 1) | π |

**Hadamard:**
| Gate | Axis $\hat{n}$ | Angle θ |
|------|----------------|---------|
| H | $(1/\sqrt{2}, 0, 1/\sqrt{2})$ | π |

### 6. Extracting Axis and Angle from a Matrix

Given V ∈ SU(2), extract (θ, $\hat{n}$):

**Step 1:** Find the rotation angle:
$$\cos\frac{\theta}{2} = \frac{1}{2}\text{Re}[\text{Tr}(V)]$$

$$\boxed{\theta = 2\arccos\left(\frac{\text{Re}[\text{Tr}(V)]}{2}\right)}$$

**Step 2:** Find the rotation axis (if θ ≠ 0):
$$\hat{n} \cdot \vec{\sigma} = \frac{V - V^\dagger}{-2i\sin(\theta/2)}$$

Explicitly:
$$n_x = \frac{\text{Im}[V_{01} + V_{10}]}{-2\sin(\theta/2)}$$
$$n_y = \frac{\text{Re}[V_{01} - V_{10}]}{2\sin(\theta/2)}$$
$$n_z = \frac{\text{Im}[V_{00} - V_{11}]}{-2\sin(\theta/2)}$$

**Special case:** When θ = 0, V = I and the axis is undefined.

### 7. The Double Cover: SU(2) → SO(3)

**Key relationship:** SU(2) is a **double cover** of SO(3) (3D rotation group).

This means:
- Every rotation in SO(3) corresponds to TWO elements of SU(2)
- Specifically: V and -V give the same physical rotation
- SU(2) is topologically a 3-sphere S³

**Consequence:** $R_{\hat{n}}(\theta)$ and $R_{\hat{n}}(\theta + 2\pi) = -R_{\hat{n}}(\theta)$ give the same Bloch sphere rotation.

### 8. Fixed Points and Eigenstates

For rotation $R_{\hat{n}}(\theta)$:

**Eigenstates:** The eigenvectors are the states pointing along ±$\hat{n}$:
$$|\hat{n}_+\rangle: \text{ eigenvalue } e^{-i\theta/2}$$
$$|\hat{n}_-\rangle: \text{ eigenvalue } e^{i\theta/2}$$

**Fixed points on Bloch sphere:** States along the rotation axis are unchanged (up to phase).

### 9. Composition of Rotations

**Single axis:** Compositions add angles:
$$R_{\hat{n}}(\theta_1)R_{\hat{n}}(\theta_2) = R_{\hat{n}}(\theta_1 + \theta_2)$$

**Different axes:** Use quaternion or matrix multiplication.

For rotations about orthogonal axes:
$$R_x(\alpha)R_y(\beta) = R_{\hat{n}}(\gamma)$$
where the resulting axis and angle follow the quaternion product formula.

### 10. Parameterizations of SU(2)

| Parameterization | Parameters | Formula |
|------------------|------------|---------|
| Axis-angle | θ, $\hat{n}$ (3 params) | $R_{\hat{n}}(\theta)$ |
| Euler angles | α, β, γ | $R_z(\alpha)R_y(\beta)R_z(\gamma)$ |
| Complex parameters | a, b | $\begin{pmatrix} a & -b^* \\ b & a^* \end{pmatrix}$ |
| Quaternion | q = (w, x, y, z) | w + xi + yj + zk |

**Euler angles (ZYZ):** Any V ∈ SU(2) can be written as:
$$V = R_z(\alpha)R_y(\beta)R_z(\gamma)$$
with α, γ ∈ [0, 2π) and β ∈ [0, π].

### 11. Topology of SU(2)

**SU(2) ≅ S³:** The constraint $|a|^2 + |b|^2 = 1$ defines a 3-sphere in ℂ².

**Consequences:**
- SU(2) is **simply connected** (any loop can be contracted to a point)
- This explains why 2π rotation ≠ identity but 4π rotation = identity
- The "belt trick" or "plate trick" demonstrates this topology

### 12. Applications

**Quantum control:** Designing pulse sequences corresponds to paths on SU(2)

**NMR:** Bloch sphere rotations describe spin dynamics

**Geometric phase:** Loops on SU(2) can accumulate Berry phases

**Variational circuits:** Optimizing over SU(2) for quantum machine learning

---

## Quantum Computing Connection

The Bloch sphere/SU(2) framework is essential for:

1. **Gate compilation:** Converting abstract gates to native rotations
2. **Error analysis:** Understanding how rotation errors propagate
3. **Pulse design:** Mapping desired unitaries to control signals
4. **Randomized benchmarking:** Sampling uniformly from SU(2)
5. **Tomography:** Reconstructing unknown operations

---

## Worked Examples

### Example 1: Extract Axis-Angle from Hadamard

**Problem:** Find the rotation axis and angle for the Hadamard gate.

**Solution:**

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Step 1:** Find the angle.
$$\text{Tr}(H) = \frac{1}{\sqrt{2}}(1 + (-1)) = 0$$
$$\cos\frac{\theta}{2} = \frac{0}{2} = 0 \Rightarrow \frac{\theta}{2} = \frac{\pi}{2} \Rightarrow \theta = \pi$$

**Step 2:** Find the axis.
$$H - H^\dagger = H - H = 0 \text{ (H is Hermitian)}$$

For Hermitian matrices, use:
$$\hat{n} \cdot \vec{\sigma} = \frac{H}{\sin(\pi/2)} = \sqrt{2}H$$

Wait, this doesn't work directly. Let's use:
$$H = \cos\frac{\pi}{2}I - i\sin\frac{\pi}{2}(\hat{n}\cdot\vec{\sigma})$$
$$H = -i(\hat{n}\cdot\vec{\sigma})$$
$$\hat{n}\cdot\vec{\sigma} = iH = \frac{i}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

But we need real $\hat{n}$. The issue is global phase!

Actually, H includes a global phase: $H = e^{i\pi/2}R_{\hat{n}}(\pi)$ for some phase convention.

Comparing $H = \frac{1}{\sqrt{2}}(X + Z)$ and using $X + Z = \sqrt{2}\,(\hat{n}\cdot\vec{\sigma})$:
$$\hat{n} = \frac{1}{\sqrt{2}}(1, 0, 1)$$

**Answer:** θ = π, $\hat{n} = (1/\sqrt{2}, 0, 1/\sqrt{2})$

### Example 2: Verify SU(2) Parameterization

**Problem:** Show that $V = \begin{pmatrix} \frac{1+i}{2} & \frac{-1+i}{2} \\ \frac{1+i}{2} & \frac{1-i}{2} \end{pmatrix}$ is in SU(2) and find its axis-angle.

**Solution:**

**Check det(V) = 1:**
$$\det(V) = \frac{1+i}{2} \cdot \frac{1-i}{2} - \frac{-1+i}{2} \cdot \frac{1+i}{2}$$
$$= \frac{(1+i)(1-i)}{4} - \frac{(-1+i)(1+i)}{4}$$
$$= \frac{2}{4} - \frac{-1+i+i-1}{4} = \frac{2}{4} - \frac{-2+2i}{4} = \frac{2+2-2i}{4} = 1-\frac{i}{2}$$

Let me recalculate:
$$a = \frac{1+i}{2}, \quad b = \frac{1+i}{2}$$
$$a^* = \frac{1-i}{2}, \quad -b^* = \frac{-1+i}{2}$$

Check: $|a|^2 + |b|^2 = \frac{2}{4} + \frac{2}{4} = 1$ ✓

$$\det(V) = a \cdot a^* - (-b^*) \cdot b = |a|^2 + |b|^2 = 1$$ ✓

**Find angle:**
$$\text{Tr}(V) = a + a^* = \frac{1+i}{2} + \frac{1-i}{2} = 1$$
$$\cos\frac{\theta}{2} = \frac{1}{2} \Rightarrow \theta = \frac{2\pi}{3}$$

**Find axis:**
$$\sin\frac{\theta}{2} = \frac{\sqrt{3}}{2}$$
$$n_z = \frac{\text{Im}(a)}{\sin(\theta/2)} = \frac{1/2}{\sqrt{3}/2} = \frac{1}{\sqrt{3}}$$

From b = (1+i)/2:
$$n_x = \frac{\text{Im}(b)}{\sin(\theta/2)} = \frac{1/2}{\sqrt{3}/2} = \frac{1}{\sqrt{3}}$$
$$n_y = \frac{\text{Re}(b)}{\sin(\theta/2)} = \frac{1/2}{\sqrt{3}/2} = \frac{1}{\sqrt{3}}$$

**Answer:** θ = 2π/3, $\hat{n} = \frac{1}{\sqrt{3}}(1, 1, 1)$

### Example 3: Euler Angle Decomposition

**Problem:** Write $R_x(\pi/2)$ in ZYZ Euler form.

**Solution:**

We need $R_x(\pi/2) = R_z(\alpha)R_y(\beta)R_z(\gamma)$.

$R_x(\pi/2) = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{-i}{\sqrt{2}} \\ \frac{-i}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}$

Using the identity $R_x(\theta) = R_z(-\pi/2)R_y(\theta)R_z(\pi/2)$:

$$R_x(\pi/2) = R_z(-\pi/2)R_y(\pi/2)R_z(\pi/2)$$

So: α = -π/2, β = π/2, γ = π/2.

---

## Practice Problems

### Direct Application

1. Find the rotation axis and angle for the S gate: $S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$.

2. Verify that $|\hat{n}|^2 = 1$ for the axis extracted from $R_y(\pi/3)$.

3. Show that T = $e^{i\pi/8}R_z(\pi/4)$ by computing both sides.

### Intermediate

4. **Double cover:** Show that $R_{\hat{n}}(\theta)$ and $-R_{\hat{n}}(\theta) = R_{\hat{n}}(\theta + 2\pi)$ produce the same transformation on any state $|\psi\rangle$ (up to global phase).

5. Find the fixed points (eigenstates) of $H$ on the Bloch sphere.

6. Given $R_x(\pi/2)R_y(\pi/2)$, find the equivalent single rotation $R_{\hat{n}}(\theta)$.

### Challenging

7. **Geodesics:** On the Bloch sphere, the shortest path between two states is a great circle arc. Show that this corresponds to a single rotation in SU(2).

8. **Composition formula:** Derive the formula for the axis and angle of $R_{\hat{m}}(\alpha)R_{\hat{n}}(\beta)$ in terms of $\hat{m}$, $\hat{n}$, α, β using quaternions.

9. **Berry phase:** A qubit is rotated around a closed loop on the Bloch sphere subtending solid angle Ω. Show that the state acquires a geometric phase of Ω/2.

---

## Computational Lab: Bloch Sphere and SU(2)

```python
"""
Day 565: Bloch Sphere Representation
Exploring SU(2), axis-angle representation, and gate geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Rotation gates
def Rn(theta, n):
    """Rotation by angle theta about axis n."""
    n = np.array(n, dtype=float)
    n = n / np.linalg.norm(n)
    n_dot_sigma = n[0]*X + n[1]*Y + n[2]*Z
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*n_dot_sigma

def Rx(theta): return Rn(theta, [1, 0, 0])
def Ry(theta): return Rn(theta, [0, 1, 0])
def Rz(theta): return Rn(theta, [0, 0, 1])

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print("=" * 60)
print("AXIS-ANGLE EXTRACTION")
print("=" * 60)

def extract_axis_angle(V):
    """
    Extract rotation axis and angle from SU(2) matrix V.
    Returns (theta, n_hat) where V = R_n(theta).
    """
    # Ensure we're in SU(2) (det = 1)
    det = np.linalg.det(V)
    if not np.isclose(np.abs(det), 1):
        raise ValueError("Matrix is not unitary")

    # Remove global phase to get into SU(2)
    V_su2 = V / np.sqrt(det)

    # Extract angle from trace
    trace = np.trace(V_su2)
    cos_half_theta = np.real(trace) / 2
    cos_half_theta = np.clip(cos_half_theta, -1, 1)
    half_theta = np.arccos(cos_half_theta)
    theta = 2 * half_theta

    # Extract axis
    if np.abs(np.sin(half_theta)) < 1e-10:
        # theta ≈ 0 or 2π, axis undefined
        return theta, np.array([0, 0, 1])  # Default axis

    sin_half_theta = np.sin(half_theta)

    # From V = cos(θ/2)I - i sin(θ/2)(n·σ)
    # n·σ = (V - V†) / (-2i sin(θ/2))
    # But V ∈ SU(2) means V = [[a, -b*], [b, a*]]

    a = V_su2[0, 0]
    b = V_su2[1, 0]

    n_z = np.imag(a) / sin_half_theta
    n_x = np.imag(b) / sin_half_theta
    n_y = np.real(b) / sin_half_theta

    n_hat = np.array([n_x, n_y, n_z])
    n_hat = n_hat / np.linalg.norm(n_hat)  # Normalize

    return theta, n_hat

# Test on known gates
print("\n1. Extracting axis and angle from known gates:")

test_gates = [
    ('X', X),
    ('Y', Y),
    ('Z', Z),
    ('H', np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)),
    ('S', np.array([[1, 0], [0, 1j]], dtype=complex)),
    ('T', np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)),
    ('Rx(π/3)', Rx(np.pi/3)),
    ('Ry(π/4)', Ry(np.pi/4)),
]

for name, gate in test_gates:
    try:
        theta, n_hat = extract_axis_angle(gate)
        print(f"   {name:10}: θ = {theta/np.pi:.3f}π, n̂ = ({n_hat[0]:.3f}, {n_hat[1]:.3f}, {n_hat[2]:.3f})")
    except Exception as e:
        print(f"   {name:10}: Error - {e}")

# Verify round-trip
print("\n2. Round-trip verification (extract then reconstruct):")
for name, gate in test_gates[:4]:
    theta, n_hat = extract_axis_angle(gate)
    reconstructed = Rn(theta, n_hat)
    # Check equality up to global phase
    ratio = gate[0,0] / reconstructed[0,0] if np.abs(reconstructed[0,0]) > 1e-10 else gate[1,0] / reconstructed[1,0]
    phase_diff = np.angle(ratio)
    reconstructed_phased = reconstructed * np.exp(1j * phase_diff)
    match = np.allclose(gate, reconstructed_phased)
    print(f"   {name}: Reconstructed matches original: {match}")

# SU(2) double cover
print("\n" + "=" * 60)
print("SU(2) DOUBLE COVER OF SO(3)")
print("=" * 60)

print("\n3. Double cover demonstration:")
print("   R_n(θ) and R_n(θ+2π) = -R_n(θ) give same physical transformation")

theta = np.pi / 3
n_hat = np.array([1, 1, 1]) / np.sqrt(3)

V1 = Rn(theta, n_hat)
V2 = Rn(theta + 2*np.pi, n_hat)

print(f"   R_n({theta/np.pi:.2f}π) = \n{V1}")
print(f"   R_n({(theta+2*np.pi)/np.pi:.2f}π) = \n{V2}")
print(f"   V2 = -V1: {np.allclose(V2, -V1)}")

# Apply to a state
psi = (ket_0 + 1j*ket_1) / np.sqrt(2)
psi1 = V1 @ psi
psi2 = V2 @ psi

print(f"\n   Applied to |ψ⟩:")
print(f"   V1|ψ⟩ = {psi1.flatten()}")
print(f"   V2|ψ⟩ = {psi2.flatten()}")
print(f"   V2|ψ⟩ = -V1|ψ⟩: {np.allclose(psi2, -psi1)}")
print(f"   Same physical state (differ by global phase)!")

# Euler angles
print("\n" + "=" * 60)
print("EULER ANGLE DECOMPOSITION")
print("=" * 60)

def euler_to_su2(alpha, beta, gamma):
    """Convert ZYZ Euler angles to SU(2) matrix."""
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)

def su2_to_euler(V):
    """Extract ZYZ Euler angles from SU(2) matrix."""
    # V = Rz(α)Ry(β)Rz(γ)
    # From the matrix elements

    # Ensure SU(2)
    V = V / np.sqrt(np.linalg.det(V))

    # β from |V[0,0]| or |V[1,1]}
    beta = 2 * np.arccos(np.clip(np.abs(V[0, 0]), 0, 1))

    if np.abs(np.sin(beta/2)) < 1e-10:
        # β ≈ 0: Only α+γ matters
        alpha = np.angle(V[0, 0])
        gamma = 0
    elif np.abs(np.cos(beta/2)) < 1e-10:
        # β ≈ π: Only α-γ matters
        alpha = np.angle(V[1, 0])
        gamma = 0
    else:
        # General case
        # V[0,0] = cos(β/2) exp(-i(α+γ)/2)
        # V[1,0] = sin(β/2) exp(i(α-γ)/2)
        alpha_plus_gamma = -2 * np.angle(V[0, 0])
        alpha_minus_gamma = 2 * np.angle(V[1, 0])
        alpha = (alpha_plus_gamma + alpha_minus_gamma) / 2
        gamma = (alpha_plus_gamma - alpha_minus_gamma) / 2

    return alpha, beta, gamma

print("\n4. Euler decomposition of known gates:")

for name, gate in test_gates[:6]:
    alpha, beta, gamma = su2_to_euler(gate)
    reconstructed = euler_to_su2(alpha, beta, gamma)
    # Check up to global phase
    phase = np.angle(gate[0,0] / reconstructed[0,0]) if np.abs(reconstructed[0,0]) > 1e-10 else 0
    match = np.allclose(gate, reconstructed * np.exp(1j*phase))
    print(f"   {name:10}: α={alpha/np.pi:.3f}π, β={beta/np.pi:.3f}π, γ={gamma/np.pi:.3f}π, verified: {match}")

# Visualization
print("\n" + "=" * 60)
print("BLOCH SPHERE VISUALIZATION")
print("=" * 60)

def state_to_bloch(psi):
    """Convert pure state to Bloch coordinates."""
    psi = psi.flatten()
    rho = np.outer(psi, psi.conj())
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    return np.array([x, y, z])

def draw_bloch_sphere(ax, show_axes=True):
    """Draw Bloch sphere wireframe."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

    if show_axes:
        ax.quiver(0, 0, 0, 1.3, 0, 0, color='r', alpha=0.3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1.3, 0, color='g', alpha=0.3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1.3, color='b', alpha=0.3, arrow_length_ratio=0.1)
        ax.text(1.5, 0, 0, 'X', fontsize=10, color='r')
        ax.text(0, 1.5, 0, 'Y', fontsize=10, color='g')
        ax.text(0, 0, 1.5, 'Z', fontsize=10, color='b')

fig = plt.figure(figsize=(15, 10))

# Plot 1: Rotation axes of common gates
ax1 = fig.add_subplot(221, projection='3d')
draw_bloch_sphere(ax1)

gate_axes = [
    ('X', [1, 0, 0], 'red'),
    ('Y', [0, 1, 0], 'green'),
    ('Z', [0, 0, 1], 'blue'),
    ('H', [1/np.sqrt(2), 0, 1/np.sqrt(2)], 'purple'),
    ('S', [0, 0, 1], 'cyan'),
]

for name, axis, color in gate_axes:
    axis = np.array(axis)
    ax1.quiver(0, 0, 0, axis[0]*1.2, axis[1]*1.2, axis[2]*1.2,
               color=color, linewidth=2, arrow_length_ratio=0.15, label=name)

ax1.set_title('Rotation Axes of Common Gates')
ax1.legend(loc='upper left')

# Plot 2: Effect of Hadamard (π rotation about (1,0,1)/√2)
ax2 = fig.add_subplot(222, projection='3d')
draw_bloch_sphere(ax2)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Draw rotation axis
h_axis = np.array([1, 0, 1]) / np.sqrt(2)
ax2.quiver(0, 0, 0, h_axis[0]*1.5, h_axis[1]*1.5, h_axis[2]*1.5,
           color='purple', linewidth=3, arrow_length_ratio=0.1, label='H axis')

# Show transformation of several states
test_states = [
    ('|0⟩', ket_0, 'blue'),
    ('|1⟩', ket_1, 'red'),
    ('|+⟩', (ket_0 + ket_1)/np.sqrt(2), 'green'),
    ('|+y⟩', (ket_0 + 1j*ket_1)/np.sqrt(2), 'orange'),
]

for name, state, color in test_states:
    bloch = state_to_bloch(state)
    transformed = state_to_bloch(H @ state)

    ax2.scatter(*bloch, c=color, s=100, marker='o', alpha=0.7)
    ax2.scatter(*transformed, c=color, s=100, marker='s', alpha=0.7)
    ax2.plot([bloch[0], transformed[0]], [bloch[1], transformed[1]],
             [bloch[2], transformed[2]], color=color, linestyle='--', alpha=0.5)

ax2.set_title('Hadamard Gate Action\n(circles → squares)')
ax2.legend(['H rotation axis'])

# Plot 3: Trajectory under continuous rotation
ax3 = fig.add_subplot(223, projection='3d')
draw_bloch_sphere(ax3)

# Rotate |0⟩ about axis (1,1,0)/√2
rot_axis = np.array([1, 1, 0]) / np.sqrt(2)
ax3.quiver(0, 0, 0, rot_axis[0]*1.5, rot_axis[1]*1.5, rot_axis[2]*1.5,
           color='purple', linewidth=3, arrow_length_ratio=0.1)

trajectory = []
for theta in np.linspace(0, 2*np.pi, 100):
    state = Rn(theta, rot_axis) @ ket_0
    trajectory.append(state_to_bloch(state))
trajectory = np.array(trajectory)

ax3.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
         'b-', linewidth=2, label='Trajectory')
ax3.scatter([0], [0], [1], c='green', s=150, marker='*', label='|0⟩')

ax3.set_title('Rotation of |0⟩ about (1,1,0)/√2')
ax3.legend()

# Plot 4: Composition of rotations
ax4 = fig.add_subplot(224, projection='3d')
draw_bloch_sphere(ax4)

# Compare Rx(π/2)Ry(π/2)Rz(π/2) applied to |0⟩
state = ket_0

colors = ['red', 'green', 'blue', 'purple']
labels = ['|0⟩', 'After Rz(π/2)', 'After Ry(π/2)', 'After Rx(π/2)']
gates = [I, Rz(np.pi/2), Ry(np.pi/2) @ Rz(np.pi/2), Rx(np.pi/2) @ Ry(np.pi/2) @ Rz(np.pi/2)]

bloch_points = []
for gate in gates:
    transformed = gate @ ket_0
    bloch_points.append(state_to_bloch(transformed))

for i, (point, color, label) in enumerate(zip(bloch_points, colors, labels)):
    ax4.scatter(*point, c=color, s=100, marker='o', label=label)
    if i > 0:
        ax4.plot([bloch_points[i-1][0], point[0]],
                 [bloch_points[i-1][1], point[1]],
                 [bloch_points[i-1][2], point[2]],
                 color='gray', linestyle='--', alpha=0.7)

ax4.set_title('Composition: Rx(π/2)·Ry(π/2)·Rz(π/2)|0⟩')
ax4.legend(loc='upper left', fontsize=8)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('bloch_sphere_su2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bloch_sphere_su2.png")

# Interactive demonstration of axis extraction
print("\n" + "=" * 60)
print("RANDOM SU(2) MATRICES")
print("=" * 60)

def random_su2():
    """Generate a random SU(2) matrix (Haar measure)."""
    # Generate random complex 2x2 and orthogonalize
    a = np.random.randn() + 1j * np.random.randn()
    b = np.random.randn() + 1j * np.random.randn()
    norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    a, b = a/norm, b/norm
    return np.array([[a, -np.conj(b)], [b, np.conj(a)]], dtype=complex)

print("\n5. Random SU(2) matrices - axis-angle extraction:")
for i in range(5):
    V = random_su2()
    theta, n_hat = extract_axis_angle(V)
    reconstructed = Rn(theta, n_hat)
    phase = np.angle(V[0,0] / reconstructed[0,0]) if np.abs(reconstructed[0,0]) > 1e-10 else 0
    match = np.allclose(V, reconstructed * np.exp(1j*phase))
    print(f"   Random #{i+1}: θ = {theta/np.pi:.3f}π, n̂ = ({n_hat[0]:+.3f}, {n_hat[1]:+.3f}, {n_hat[2]:+.3f}), verified: {match}")

# Gate fidelity analysis
print("\n" + "=" * 60)
print("GATE ERROR ANALYSIS")
print("=" * 60)

def gate_fidelity(U, V):
    """Compute average gate fidelity between U and V."""
    return (np.abs(np.trace(U.conj().T @ V))**2 + 2) / 6

print("\n6. Effect of small rotation errors:")
target = Rx(np.pi/2)  # Perfect Rx(π/2)

errors = [0.001, 0.01, 0.05, 0.1]
for err in errors:
    # Error in angle
    actual_angle = Rx(np.pi/2 + err)
    fid_angle = gate_fidelity(target, actual_angle)

    # Error in axis (small tilt)
    actual_axis = Rn(np.pi/2, [1, err, 0])
    fid_axis = gate_fidelity(target, actual_axis)

    print(f"   Error = {err:.3f} rad:")
    print(f"      Angle error fidelity: {fid_angle:.6f}")
    print(f"      Axis tilt fidelity:   {fid_axis:.6f}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| SU(2) general form | $V = \begin{pmatrix} a & -b^* \\ b & a^* \end{pmatrix}$, $\|a\|^2 + \|b\|^2 = 1$ |
| Axis-angle form | $V = R_{\hat{n}}(\theta) = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}(\hat{n}\cdot\vec{\sigma})$ |
| Angle extraction | $\theta = 2\arccos\left(\frac{\text{Re}[\text{Tr}(V)]}{2}\right)$ |
| Euler decomposition | $V = R_z(\alpha)R_y(\beta)R_z(\gamma)$ |
| Double cover | $R_{\hat{n}}(\theta) \sim -R_{\hat{n}}(\theta) = R_{\hat{n}}(\theta + 2\pi)$ |

### Main Takeaways

1. **Every single-qubit gate is a rotation:** U = e^{iα}R_n̂(θ) for some axis and angle
2. **SU(2) structure:** 3-dimensional manifold, topologically a 3-sphere
3. **Double cover:** SU(2) maps 2-to-1 onto SO(3), explaining 4π periodicity
4. **Multiple parameterizations:** Axis-angle, Euler angles, and matrix form all equivalent
5. **Geometric insight:** The Bloch sphere provides complete visualization of qubit operations

---

## Daily Checklist

- [ ] I can prove that any U ∈ SU(2) has the axis-angle form
- [ ] I can extract axis and angle from a given 2×2 unitary matrix
- [ ] I understand the SU(2) → SO(3) double cover
- [ ] I can convert between axis-angle and Euler angle representations
- [ ] I completed the visualization lab showing gate axes
- [ ] I solved at least 3 practice problems

---

## Preview of Day 566

Tomorrow we explore **gate decomposition** in detail. We'll prove that any single-qubit unitary can be decomposed using the ZYZ Euler angle sequence, and discuss how to approximate arbitrary gates using discrete gate sets. This is crucial for compiling quantum algorithms to real hardware.
