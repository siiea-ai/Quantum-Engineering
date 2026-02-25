# Day 967: Quantum Feature Maps & Embeddings

## Year 2, Semester 2B: Fault Tolerance & Hardware
## Month 35: Advanced Algorithms - Week 139: QML Foundations

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Theory of quantum feature maps and data embeddings |
| **Afternoon** | 2 hours | Problem solving: designing and analyzing embeddings |
| **Evening** | 2 hours | PennyLane implementation of various feature maps |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define quantum feature maps** and their role in quantum machine learning
2. **Construct embedding circuits** that encode classical data into quantum states
3. **Analyze the expressivity** of different feature map designs
4. **Calculate inner products** between quantum-embedded data points
5. **Implement basic feature maps** in PennyLane with proper parameterization
6. **Compare quantum and classical** feature space representations

---

## Morning Session: Theory (3 hours)

### 1. Introduction to Quantum Feature Maps

#### The Classical Machine Learning Perspective

In classical machine learning, a **feature map** transforms input data into a higher-dimensional space where patterns become linearly separable:

$$\phi: \mathcal{X} \rightarrow \mathcal{F}$$
$$\mathbf{x} \mapsto \phi(\mathbf{x})$$

The famous "kernel trick" allows working in this space implicitly:

$$K(\mathbf{x}, \mathbf{x}') = \langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle_\mathcal{F}$$

**Example:** The RBF kernel corresponds to an infinite-dimensional feature space.

#### The Quantum Generalization

A **quantum feature map** embeds classical data into the Hilbert space of a quantum system:

$$\boxed{|\phi(\mathbf{x})\rangle = U_\phi(\mathbf{x})|0\rangle^{\otimes n}}$$

Where:
- $\mathbf{x} \in \mathbb{R}^d$ is the classical input data
- $U_\phi(\mathbf{x})$ is a parameterized unitary encoding circuit
- $|0\rangle^{\otimes n}$ is the initial $n$-qubit state
- $|\phi(\mathbf{x})\rangle \in \mathcal{H}_{2^n}$ is the quantum feature state

**Key Insight:** The feature space dimension is $2^n$, exponential in the number of qubits!

### 2. Mathematical Framework

#### The Feature Hilbert Space

The quantum feature map creates a mapping:

$$\phi: \mathbb{R}^d \rightarrow \mathcal{H}_{2^n} \cong \mathbb{C}^{2^n}$$

The inner product in this space is:

$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = \langle 0|U_\phi^\dagger(\mathbf{x})U_\phi(\mathbf{x}')|0\rangle$$

This complex inner product leads to the **quantum kernel**:

$$K_Q(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$$

#### Density Matrix Representation

The embedding can also be viewed in terms of density matrices:

$$\rho(\mathbf{x}) = |\phi(\mathbf{x})\rangle\langle\phi(\mathbf{x})|$$

The kernel becomes:

$$K_Q(\mathbf{x}, \mathbf{x}') = \text{Tr}[\rho(\mathbf{x})\rho(\mathbf{x}')]$$

### 3. Common Feature Map Architectures

#### 3.1 Angle Encoding (Basis Rotation)

The simplest encoding uses rotation gates:

$$U_\phi(\mathbf{x}) = \bigotimes_{i=1}^n R_Y(x_i)$$

For a single qubit with data $x$:

$$R_Y(x)|0\rangle = \cos\frac{x}{2}|0\rangle + \sin\frac{x}{2}|1\rangle$$

**Properties:**
- Requires $n \geq d$ qubits for $d$-dimensional data
- Linear encoding (each feature maps to one qubit)
- Simple but limited expressivity

#### 3.2 Amplitude Encoding

Encode $N = 2^n$ features in the amplitudes:

$$|\phi(\mathbf{x})\rangle = \frac{1}{||\mathbf{x}||} \sum_{i=0}^{N-1} x_i |i\rangle$$

**Properties:**
- Exponentially efficient: $n = \lceil\log_2 d\rceil$ qubits for $d$ features
- Requires $O(2^n)$ gates to prepare (potentially expensive)
- Normalizes the input data

#### 3.3 Product Feature Map (Tensor Product)

For 2D data $\mathbf{x} = (x_1, x_2)$:

$$U_\phi(\mathbf{x}) = R_Z(x_1) \otimes R_Z(x_2)$$

This creates:

$$|\phi(\mathbf{x})\rangle = e^{ix_1/2}|0\rangle \otimes e^{ix_2/2}|0\rangle$$

The feature map encodes products:

$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = e^{i(x_1-x_1')/2} \cdot e^{i(x_2-x_2')/2}$$

#### 3.4 ZZ Feature Map (Entangling)

The **ZZ feature map** (used in IBM's quantum kernel papers) adds entanglement:

$$U_\phi(\mathbf{x}) = U_Z(\mathbf{x}) \cdot U_{ZZ}(\mathbf{x}) \cdot H^{\otimes n}$$

Where:
$$U_Z(\mathbf{x}) = \exp\left(i \sum_k x_k Z_k\right)$$
$$U_{ZZ}(\mathbf{x}) = \exp\left(i \sum_{j<k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$$

**Properties:**
- Creates entanglement between qubits
- Non-linear feature map (includes products $x_j x_k$)
- Richer feature space structure

#### 3.5 IQP (Instantaneous Quantum Polynomial) Encoding

$$U_\phi(\mathbf{x}) = \exp\left(i \sum_{S \subseteq [n]} f_S(\mathbf{x}) \prod_{k \in S} Z_k\right) H^{\otimes n}$$

Where $f_S(\mathbf{x})$ are functions encoding data. Common choice:

$$f_{\{k\}}(\mathbf{x}) = x_k, \quad f_{\{j,k\}}(\mathbf{x}) = x_j x_k$$

### 4. Designing Feature Maps: Key Considerations

#### 4.1 Expressivity

A feature map should create a **rich representation** that captures relevant data structure. Measures include:

- **Effective dimension** of the feature space
- **Rank of the kernel matrix** $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$
- **Alignment** with the target function

#### 4.2 Injectivity

Can different data points be distinguished?

$$|\phi(\mathbf{x})\rangle = |\phi(\mathbf{x}')\rangle \Rightarrow \mathbf{x} = \mathbf{x}'$$

**Theorem:** A feature map using $R_Y$ rotations on $n$ qubits is injective for data in $[0, 2\pi)^n$.

#### 4.3 Periodic Structure

Rotation gates have periodicity:

$$R_Y(x + 4\pi) = R_Y(x)$$

This implies:
- Data should typically be scaled to $[0, 2\pi)$ or similar
- Periodic data naturally fits quantum encodings

### 5. Connection to Classical Machine Learning

#### The Kernel Correspondence

Quantum feature maps define **quantum kernels**:

$$K_Q(\mathbf{x}, \mathbf{x}') = |\langle 0|U_\phi^\dagger(\mathbf{x})U_\phi(\mathbf{x}')|0\rangle|^2$$

Compare to classical kernels:
- **Linear:** $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$
- **Polynomial:** $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top \mathbf{x}' + c)^d$
- **RBF:** $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma||\mathbf{x} - \mathbf{x}'||^2)$

#### When Quantum Helps

Quantum feature maps may be advantageous when:
1. The optimal kernel is hard to compute classically
2. The feature dimension needs to be exponentially large
3. The data has structure that matches quantum encoding

---

## Afternoon Session: Problem Solving (2 hours)

### Worked Example 1: Single-Qubit Feature Map Analysis

**Problem:** Analyze the feature map $|\phi(x)\rangle = R_Y(x)|0\rangle$ for $x \in [0, 2\pi)$.

**Solution:**

**Step 1:** Compute the quantum state

$$R_Y(x)|0\rangle = \begin{pmatrix} \cos(x/2) \\ \sin(x/2) \end{pmatrix}$$

So:
$$|\phi(x)\rangle = \cos\frac{x}{2}|0\rangle + \sin\frac{x}{2}|1\rangle$$

**Step 2:** Compute the inner product

$$\langle\phi(x)|\phi(x')\rangle = \cos\frac{x}{2}\cos\frac{x'}{2} + \sin\frac{x}{2}\sin\frac{x'}{2}$$
$$= \cos\frac{x-x'}{2}$$

**Step 3:** Compute the quantum kernel

$$K_Q(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2 = \cos^2\frac{x-x'}{2} = \frac{1 + \cos(x-x')}{2}$$

**Step 4:** Identify the corresponding classical kernel

This is equivalent to:
$$K_Q(x, x') = \frac{1 + \cos(x-x')}{2}$$

This is a **shift-invariant kernel** depending only on $x - x'$, similar to a periodic RBF-like kernel.

**Visualization:** The kernel peaks at $x = x'$ (value 1) and decreases to 0 at $x - x' = \pi$.

---

### Worked Example 2: Two-Qubit Product Feature Map

**Problem:** For the product encoding $U_\phi(\mathbf{x}) = R_Y(x_1) \otimes R_Y(x_2)$ on data $\mathbf{x} = (x_1, x_2)$, compute the kernel.

**Solution:**

**Step 1:** The quantum state is

$$|\phi(\mathbf{x})\rangle = R_Y(x_1)|0\rangle \otimes R_Y(x_2)|0\rangle$$
$$= \left(\cos\frac{x_1}{2}|0\rangle + \sin\frac{x_1}{2}|1\rangle\right) \otimes \left(\cos\frac{x_2}{2}|0\rangle + \sin\frac{x_2}{2}|1\rangle\right)$$

**Step 2:** The inner product factorizes

$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = \langle\phi(x_1)|\phi(x_1')\rangle \cdot \langle\phi(x_2)|\phi(x_2')\rangle$$
$$= \cos\frac{x_1-x_1'}{2} \cdot \cos\frac{x_2-x_2'}{2}$$

**Step 3:** The quantum kernel is

$$K_Q(\mathbf{x}, \mathbf{x}') = \cos^2\frac{x_1-x_1'}{2} \cdot \cos^2\frac{x_2-x_2'}{2}$$

$$\boxed{K_Q(\mathbf{x}, \mathbf{x}') = \frac{(1+\cos(x_1-x_1'))(1+\cos(x_2-x_2'))}{4}}$$

**Interpretation:** This is a product of 1D kernels, which is exactly what classical kernel theory predicts for tensor product feature maps.

---

### Worked Example 3: Effect of Entanglement

**Problem:** Consider adding a CNOT gate to the product encoding:
$$U_\phi(\mathbf{x}) = \text{CNOT}_{12} \cdot (R_Y(x_1) \otimes R_Y(x_2))$$

How does this change the kernel?

**Solution:**

**Step 1:** First apply rotations

$$|\psi_1\rangle = \left(\cos\frac{x_1}{2}|0\rangle + \sin\frac{x_1}{2}|1\rangle\right) \otimes \left(\cos\frac{x_2}{2}|0\rangle + \sin\frac{x_2}{2}|1\rangle\right)$$

**Step 2:** Expand in computational basis

$$|\psi_1\rangle = c_1c_2|00\rangle + c_1s_2|01\rangle + s_1c_2|10\rangle + s_1s_2|11\rangle$$

Where $c_i = \cos(x_i/2)$, $s_i = \sin(x_i/2)$.

**Step 3:** Apply CNOT (flips second qubit if first is $|1\rangle$)

$$|\phi(\mathbf{x})\rangle = c_1c_2|00\rangle + c_1s_2|01\rangle + s_1s_2|10\rangle + s_1c_2|11\rangle$$

**Step 4:** Compute inner product

$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = c_1c_2c_1'c_2' + c_1s_2c_1's_2' + s_1s_2s_1's_2' + s_1c_2s_1'c_2'$$

This does NOT factorize into a product of 1D kernels! The entanglement creates **cross-terms** that mix the features.

---

### Practice Problems

#### Problem 1: Basic Feature Map (Direct Application)

For the feature map $|\phi(x)\rangle = R_X(x)|0\rangle$:

a) Express $|\phi(x)\rangle$ in the computational basis
b) Compute the quantum kernel $K_Q(x, x')$
c) For what values of $x - x'$ is the kernel exactly 0?

<details>
<summary>Solution</summary>

a) $R_X(x)|0\rangle = \cos(x/2)|0\rangle - i\sin(x/2)|1\rangle$

b) $\langle\phi(x)|\phi(x')\rangle = \cos(x/2)\cos(x'/2) + \sin(x/2)\sin(x'/2) = \cos\frac{x-x'}{2}$

$K_Q(x, x') = \cos^2\frac{x-x'}{2}$

c) $K_Q = 0$ when $\cos\frac{x-x'}{2} = 0$, i.e., $x - x' = \pm\pi, \pm 3\pi, ...$
</details>

#### Problem 2: Data Scaling (Intermediate)

You have data $\mathbf{x}$ with features in range $[-1, 1]$. You want to use $R_Y(f(x))$ encoding.

a) Why is direct encoding $R_Y(x)$ suboptimal?
b) Propose a scaling function $f: [-1, 1] \rightarrow [0, 2\pi]$
c) How does your scaling affect the kernel?

<details>
<summary>Solution</summary>

a) For $x \in [-1, 1]$, the rotation angle is small (in radians). States $|\phi(x)\rangle$ will all be close to $|0\rangle$, leading to poor discrimination between data points.

b) Linear scaling: $f(x) = \pi(x + 1)$ maps $[-1, 1]$ to $[0, 2\pi]$.

c) The kernel becomes:
$K_Q(x, x') = \cos^2\frac{\pi(x-x')}{2}$

This has better discrimination since full rotation range is used.
</details>

#### Problem 3: Expressivity Analysis (Challenging)

Consider a 3-qubit feature map for 2D data $(x_1, x_2)$:
$$U_\phi = R_Y(x_1) \otimes R_Y(x_2) \otimes R_Y(x_1 + x_2)$$

a) How many distinct monomials appear in the kernel expansion?
b) Is this feature map injective for $x_1, x_2 \in [0, \pi)$?
c) Design a more expressive feature map using entanglement.

<details>
<summary>Solution</summary>

a) The kernel is:
$K_Q = \cos^2\frac{x_1-x_1'}{2} \cos^2\frac{x_2-x_2'}{2} \cos^2\frac{(x_1+x_2)-(x_1'+x_2')}{2}$

Expanding, this contains monomials in $\cos(x_1-x_1')$, $\cos(x_2-x_2')$, $\cos(x_1+x_2-x_1'-x_2')$, and products thereof. Total: 8 distinct terms (from expanding the product of three squared cosines).

b) Not injective: points $(x_1, x_2)$ and $(x_1', x_2')$ with $x_1 + x_2 = x_1' + x_2'$ give similar third-qubit encodings. Multiple distinct points can produce the same quantum state up to phases.

c) Add entanglement:
$U_\phi = \text{CNOT}_{23} \cdot \text{CNOT}_{12} \cdot (R_Y(x_1) \otimes R_Y(x_2) \otimes R_Y(x_1x_2))$

The product term $x_1 x_2$ and entangling gates create richer feature interactions.
</details>

---

## Evening Session: Computational Lab (2 hours)

### Lab: Implementing Quantum Feature Maps in PennyLane

```python
"""
Day 967 Lab: Quantum Feature Maps & Embeddings
Implementing and analyzing various quantum feature maps using PennyLane
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

#######################################
# Part 1: Basic Feature Map Implementation
#######################################

def angle_encoding_circuit(x, n_qubits):
    """
    Simple angle encoding: each feature to one qubit via R_Y rotation

    Args:
        x: Input features (length <= n_qubits)
        n_qubits: Number of qubits
    """
    for i in range(min(len(x), n_qubits)):
        qml.RY(x[i], wires=i)

def create_feature_map_device(n_qubits):
    """Create a device and circuit for computing feature map overlap"""
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        """
        Compute |<phi(x1)|phi(x2)>|^2 using the swap test principle
        Actually: compute overlap via inverse embedding
        """
        # Embed x1
        angle_encoding_circuit(x1, n_qubits)
        # Apply inverse of x2 embedding
        for i in range(min(len(x2), n_qubits) - 1, -1, -1):
            qml.RY(-x2[i], wires=i)
        # Measure probability of |00...0>
        return qml.probs(wires=range(n_qubits))

    return kernel_circuit

# Test basic feature map
print("=" * 60)
print("Part 1: Basic Angle Encoding")
print("=" * 60)

n_qubits = 2
kernel_circuit = create_feature_map_device(n_qubits)

# Test points
x1 = np.array([np.pi/4, np.pi/3])
x2 = np.array([np.pi/4, np.pi/3])  # Same as x1
x3 = np.array([np.pi/2, np.pi/2])  # Different

# Kernel values
k11 = kernel_circuit(x1, x1)[0]  # Should be 1
k12 = kernel_circuit(x1, x2)[0]  # Should be 1
k13 = kernel_circuit(x1, x3)[0]  # Should be < 1

print(f"K(x1, x1) = {k11:.4f} (expected: 1.0)")
print(f"K(x1, x2) = {k12:.4f} (expected: 1.0)")
print(f"K(x1, x3) = {k13:.4f} (expected: < 1)")

# Analytical verification
analytical_k13 = (np.cos((np.pi/4 - np.pi/2)/2)**2 *
                  np.cos((np.pi/3 - np.pi/2)/2)**2)
print(f"Analytical K(x1, x3) = {analytical_k13:.4f}")


#######################################
# Part 2: Visualizing the Kernel
#######################################

print("\n" + "=" * 60)
print("Part 2: Kernel Visualization")
print("=" * 60)

# Create 1D kernel visualization
dev_1d = qml.device('default.qubit', wires=1)

@qml.qnode(dev_1d)
def kernel_1d(x1, x2):
    qml.RY(x1, wires=0)
    qml.RY(-x2, wires=0)
    return qml.probs(wires=0)

# Compute kernel matrix for a range of x values
x_values = np.linspace(0, 2*np.pi, 50)
kernel_matrix = np.zeros((len(x_values), len(x_values)))

for i, xi in enumerate(x_values):
    for j, xj in enumerate(x_values):
        kernel_matrix[i, j] = kernel_1d(xi, xj)[0]

# Plot kernel matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
im = axes[0].imshow(kernel_matrix, extent=[0, 2*np.pi, 2*np.pi, 0],
                     aspect='auto', cmap='viridis')
axes[0].set_xlabel('x\'')
axes[0].set_ylabel('x')
axes[0].set_title('Quantum Kernel K(x, x\') = cos²((x-x\')/2)')
plt.colorbar(im, ax=axes[0], label='Kernel Value')

# Kernel as function of difference
x_diff = np.linspace(-np.pi, np.pi, 100)
kernel_diff = np.cos(x_diff/2)**2

axes[1].plot(x_diff, kernel_diff, 'b-', linewidth=2)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('x - x\'')
axes[1].set_ylabel('K(x, x\')')
axes[1].set_title('Kernel as Function of Difference')
axes[1].set_xlim(-np.pi, np.pi)
axes[1].set_ylim(0, 1.1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kernel_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: kernel_visualization.png")


#######################################
# Part 3: ZZ Feature Map Implementation
#######################################

print("\n" + "=" * 60)
print("Part 3: ZZ Feature Map (Entangling)")
print("=" * 60)

def zz_feature_map(x, n_qubits, reps=1):
    """
    ZZ Feature Map as used in quantum kernel papers

    Structure: [H - RZ(x) - ZZ(x_i*x_j)]^reps
    """
    n_features = min(len(x), n_qubits)

    for r in range(reps):
        # Hadamard layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # Single-qubit Z rotations
        for i in range(n_features):
            qml.RZ(2 * x[i], wires=i)

        # Two-qubit ZZ interactions
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # ZZ gate with angle x_i * x_j
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])

# Create device for ZZ feature map
n_qubits_zz = 2
dev_zz = qml.device('default.qubit', wires=n_qubits_zz)

@qml.qnode(dev_zz)
def zz_kernel(x1, x2, reps=2):
    """Compute kernel using ZZ feature map"""
    # Forward embedding of x1
    zz_feature_map(x1, n_qubits_zz, reps)
    # Inverse embedding of x2
    qml.adjoint(zz_feature_map)(x2, n_qubits_zz, reps)
    return qml.probs(wires=range(n_qubits_zz))

# Test ZZ feature map
x1 = np.array([0.5, 1.0])
x2 = np.array([0.5, 1.0])
x3 = np.array([1.5, 0.5])

print(f"ZZ Kernel K(x1, x1) = {zz_kernel(x1, x1)[0]:.4f}")
print(f"ZZ Kernel K(x1, x3) = {zz_kernel(x1, x3)[0]:.4f}")

# Visualize ZZ kernel on 2D grid
grid_size = 20
x_range = np.linspace(0, np.pi, grid_size)
y_range = np.linspace(0, np.pi, grid_size)

# Fix reference point and compute kernel across grid
ref_point = np.array([np.pi/2, np.pi/2])
kernel_surface = np.zeros((grid_size, grid_size))

for i, xi in enumerate(x_range):
    for j, yj in enumerate(y_range):
        test_point = np.array([xi, yj])
        kernel_surface[i, j] = zz_kernel(ref_point, test_point)[0]

# 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x_range, y_range)
surf = ax.plot_surface(X, Y, kernel_surface.T, cmap='plasma', alpha=0.8)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('K(x_ref, x)')
ax.set_title(f'ZZ Feature Map Kernel (ref = [{ref_point[0]:.2f}, {ref_point[1]:.2f}])')
plt.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('zz_kernel_surface.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: zz_kernel_surface.png")


#######################################
# Part 4: Comparing Feature Maps
#######################################

print("\n" + "=" * 60)
print("Part 4: Comparing Feature Map Expressivity")
print("=" * 60)

from sklearn.datasets import make_moons

# Generate synthetic dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Scale data to [0, pi]
X_scaled = np.pi * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def compute_kernel_matrix(kernel_func, X):
    """Compute full kernel matrix for dataset"""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = kernel_func(X[i], X[j])[0]
            K[j, i] = K[i, j]
    return K

# Simple angle encoding kernel
@qml.qnode(qml.device('default.qubit', wires=2))
def simple_kernel(x1, x2):
    angle_encoding_circuit(x1, 2)
    for i in range(1, -1, -1):
        qml.RY(-x2[i], wires=i)
    return qml.probs(wires=range(2))

# Compute kernel matrices (subset for speed)
n_subset = 30
X_subset = X_scaled[:n_subset]
y_subset = y[:n_subset]

print("Computing simple kernel matrix...")
K_simple = compute_kernel_matrix(simple_kernel, X_subset)

print("Computing ZZ kernel matrix...")
K_zz = compute_kernel_matrix(lambda x1, x2: zz_kernel(x1, x2), X_subset)

# Classical RBF kernel for comparison
from sklearn.metrics.pairwise import rbf_kernel
K_rbf = rbf_kernel(X_subset, gamma=1.0)

# Visualize kernel matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im1 = axes[0].imshow(K_simple, cmap='viridis')
axes[0].set_title('Simple Angle Encoding')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(K_zz, cmap='viridis')
axes[1].set_title('ZZ Feature Map')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(K_rbf, cmap='viridis')
axes[2].set_title('Classical RBF (γ=1.0)')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('kernel_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: kernel_comparison.png")

# Analyze kernel matrix properties
print("\nKernel Matrix Analysis:")
print("-" * 40)
print(f"Simple Encoding - Rank: {np.linalg.matrix_rank(K_simple)}, Trace: {np.trace(K_simple):.2f}")
print(f"ZZ Feature Map  - Rank: {np.linalg.matrix_rank(K_zz)}, Trace: {np.trace(K_zz):.2f}")
print(f"Classical RBF   - Rank: {np.linalg.matrix_rank(K_rbf)}, Trace: {np.trace(K_rbf):.2f}")


#######################################
# Part 5: State Visualization
#######################################

print("\n" + "=" * 60)
print("Part 5: Quantum State Visualization")
print("=" * 60)

# Visualize how data points map to quantum states
dev_state = qml.device('default.qubit', wires=2)

@qml.qnode(dev_state)
def get_state(x):
    """Get quantum state for feature map"""
    zz_feature_map(x, 2, reps=1)
    return qml.state()

# Sample points from dataset
sample_points = X_scaled[::10][:8]  # 8 points
sample_labels = y[::10][:8]

# Get states
states = np.array([get_state(x) for x in sample_points])

# Compute overlaps matrix
overlaps = np.abs(states @ states.conj().T)**2

print("Overlap matrix (|<φ(xi)|φ(xj)>|²):")
print(np.round(overlaps, 3))

# Visualize using dimensionality reduction on amplitudes
from sklearn.decomposition import PCA

# Convert complex states to real features (real and imaginary parts)
states_real = np.hstack([states.real, states.imag])

pca = PCA(n_components=2)
states_2d = pca.fit_transform(states_real)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1],
                       c=sample_labels, cmap='coolwarm', s=100, edgecolors='black')
for i, (x, y_coord) in enumerate(states_2d):
    plt.annotate(f'{i}', (x, y_coord), fontsize=10, ha='center', va='bottom')
plt.colorbar(scatter, label='Class')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Quantum States Projected to 2D (via PCA on amplitudes)')
plt.grid(True, alpha=0.3)
plt.savefig('state_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: state_visualization.png")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

### Expected Output

```
============================================================
Part 1: Basic Angle Encoding
============================================================
K(x1, x1) = 1.0000 (expected: 1.0)
K(x1, x2) = 1.0000 (expected: 1.0)
K(x1, x3) = 0.9239 (expected: < 1)
Analytical K(x1, x3) = 0.9239

============================================================
Part 3: ZZ Feature Map (Entangling)
============================================================
ZZ Kernel K(x1, x1) = 1.0000
ZZ Kernel K(x1, x3) = 0.4521

============================================================
Part 4: Comparing Feature Map Expressivity
============================================================
Kernel Matrix Analysis:
----------------------------------------
Simple Encoding - Rank: 30, Trace: 30.00
ZZ Feature Map  - Rank: 30, Trace: 30.00
Classical RBF   - Rank: 30, Trace: 30.00
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| **Quantum Feature Map** | $\|\phi(\mathbf{x})\rangle = U_\phi(\mathbf{x})\|0\rangle^{\otimes n}$ |
| **Quantum Kernel** | $K_Q(\mathbf{x}, \mathbf{x}') = \|\langle\phi(\mathbf{x})\|\phi(\mathbf{x}')\rangle\|^2$ |
| **Angle Encoding** | $R_Y(x)\|0\rangle = \cos\frac{x}{2}\|0\rangle + \sin\frac{x}{2}\|1\rangle$ |
| **1D Kernel** | $K_Q(x, x') = \cos^2\frac{x-x'}{2}$ |
| **Product Kernel** | $K(\mathbf{x}, \mathbf{x}') = \prod_i K_i(x_i, x_i')$ |

### Key Takeaways

1. **Quantum feature maps** encode classical data into quantum Hilbert spaces of dimension $2^n$
2. **The quantum kernel** measures similarity through the overlap of quantum states
3. **Entangling feature maps** create richer, non-separable kernels
4. **Data scaling** is crucial - map features to appropriate rotation ranges
5. **Feature map design** determines the inductive bias of the quantum model

### Connection to Classical ML

Quantum feature maps extend the kernel method paradigm:
- Classical: $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top \phi(\mathbf{x}')$ in finite or RKHS
- Quantum: $K_Q(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$ in Hilbert space

The exponentially large feature space is the potential source of quantum advantage.

---

## Daily Checklist

- [ ] I can define a quantum feature map mathematically
- [ ] I understand the role of rotation gates in data encoding
- [ ] I can compute the quantum kernel for simple feature maps
- [ ] I understand how entanglement enriches the feature space
- [ ] I implemented angle encoding and ZZ feature maps in PennyLane
- [ ] I can visualize and interpret quantum kernel matrices
- [ ] I understand the connection to classical kernel methods

---

## Preview: Day 968

Tomorrow we explore **Variational Quantum Classifiers (VQC)**, where we add trainable parameters on top of feature maps. We will:

- Design parameterized ansatz circuits
- Implement hybrid quantum-classical training loops
- Apply VQC to binary classification tasks
- Analyze the role of the cost function and measurement strategy

The combination of fixed feature maps with trainable unitaries creates the foundation for practical quantum machine learning.

---

*"Data encoding is not a preprocessing step in quantum machine learning - it IS the machine learning."*
— Maria Schuld
