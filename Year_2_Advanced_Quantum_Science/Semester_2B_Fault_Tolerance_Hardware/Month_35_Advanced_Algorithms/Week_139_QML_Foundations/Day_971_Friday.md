# Day 971: Data Encoding Strategies

## Year 2, Semester 2B: Fault Tolerance & Hardware
## Month 35: Advanced Algorithms - Week 139: QML Foundations

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Theory of data encoding methods |
| **Afternoon** | 2 hours | Problem solving: encoding design and analysis |
| **Evening** | 2 hours | Implementing and comparing encoding strategies |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Compare amplitude, angle, and basis encoding** methods systematically
2. **Analyze qubit requirements** for different encoding strategies
3. **Evaluate gate complexity** of various encoding circuits
4. **Design efficient encoding schemes** for specific data types
5. **Understand trade-offs** between encoding depth and expressivity
6. **Select appropriate encodings** for different QML tasks

---

## Morning Session: Theory (3 hours)

### 1. The Data Encoding Problem

#### Fundamental Challenge

Classical data exists in $\mathbb{R}^d$ or $\mathbb{Z}^d$, but quantum computers manipulate states in $\mathcal{H}_{2^n}$. We need:

$$\boxed{\text{Encoding}: \mathbf{x} \in \mathbb{R}^d \mapsto |\phi(\mathbf{x})\rangle \in \mathcal{H}_{2^n}}$$

#### Key Considerations

1. **Qubit efficiency:** How many qubits for $d$ features?
2. **Gate complexity:** How many gates to prepare $|\phi(\mathbf{x})\rangle$?
3. **Expressivity:** How much of Hilbert space is accessible?
4. **Injectivity:** Can different $\mathbf{x}$ yield the same state?
5. **Hardware compatibility:** Native gates on target device?

### 2. Basis Encoding

#### Definition

Encode classical data as computational basis states:

$$|\mathbf{x}\rangle = |x_1 x_2 \cdots x_d\rangle$$

Where each $x_i \in \{0, 1\}$ is a bit.

#### Implementation

For binary string $\mathbf{x} = (1, 0, 1, 1)$:
$$|1011\rangle = X \otimes I \otimes X \otimes X |0000\rangle$$

**Circuit:**
```
|0⟩ ─[X]─ → |1⟩
|0⟩ ───── → |0⟩
|0⟩ ─[X]─ → |1⟩
|0⟩ ─[X]─ → |1⟩
```

#### Properties

| Property | Value |
|----------|-------|
| **Qubits needed** | $d$ (one per binary feature) |
| **Gates** | $O(d)$ (at most $d$ X gates) |
| **Depth** | 1 |
| **Expressivity** | Limited to $2^d$ basis states |

#### Use Cases

- Discrete/categorical data
- Binary classification labels
- Quantum database encoding
- Oracles in quantum algorithms

#### Limitations

- Cannot encode continuous data directly
- Requires discretization (quantization)
- No superposition for a single data point

### 3. Angle Encoding

#### Definition

Encode features as rotation angles:

$$|\phi(\mathbf{x})\rangle = \bigotimes_{i=1}^{n} R(\alpha x_i) |0\rangle$$

Where $R \in \{R_X, R_Y, R_Z\}$ and $\alpha$ is a scaling factor.

#### Common Variants

**RY Encoding:**
$$R_Y(x)|0\rangle = \cos\frac{x}{2}|0\rangle + \sin\frac{x}{2}|1\rangle$$

**RX Encoding:**
$$R_X(x)|0\rangle = \cos\frac{x}{2}|0\rangle - i\sin\frac{x}{2}|1\rangle$$

**RZ Encoding:**
$$R_Z(x)|0\rangle = e^{-ix/2}|0\rangle$$

(Note: $R_Z$ on $|0\rangle$ only adds a global phase; needs Hadamard first)

#### Dense Angle Encoding

Use multiple rotations per qubit:
$$|q_i\rangle = R_Z(x_{3i+2}) R_Y(x_{3i+1}) R_X(x_{3i}) |0\rangle$$

This encodes 3 features per qubit!

#### Properties

| Property | Value |
|----------|-------|
| **Qubits needed** | $\lceil d/k \rceil$ where $k$ = features/qubit |
| **Gates** | $O(d)$ rotations |
| **Depth** | $O(k)$ |
| **Expressivity** | Continuous, but limited to product states |

#### Scaling Considerations

Data should typically be scaled to $[0, 2\pi)$:
$$x' = 2\pi \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Or to $[0, \pi]$ for better discrimination with $R_Y$.

### 4. Amplitude Encoding

#### Definition

Encode $N = 2^n$ features as state amplitudes:

$$\boxed{|\phi(\mathbf{x})\rangle = \frac{1}{||\mathbf{x}||} \sum_{i=0}^{N-1} x_i |i\rangle}$$

#### Properties

| Property | Value |
|----------|-------|
| **Qubits needed** | $n = \lceil\log_2 d\rceil$ |
| **Gates** | $O(2^n)$ in general |
| **Depth** | $O(2^n)$ or $O(n^2)$ with optimizations |
| **Expressivity** | Full access to $n$-qubit Hilbert space |

#### Gate Complexity Analysis

Preparing an arbitrary $n$-qubit state requires:
$$O(2^n) \text{ CNOT gates and } O(2^n) \text{ single-qubit rotations}$$

**Theorem (Shende et al.):** Any $n$-qubit state can be prepared with $\frac{23}{48} \cdot 4^n$ CNOT gates asymptotically.

#### Efficient Special Cases

**Uniform superposition:**
$$|+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{i=0}^{2^n-1}|i\rangle$$

Only $n$ gates!

**Sparse vectors:** If only $k$ amplitudes are non-zero, can prepare in $O(kn)$ gates.

#### Amplitude Encoding Circuit (General)

Uses a recursive decomposition:

```
def amplitude_encoding(amplitudes, wires):
    if len(wires) == 1:
        # Base case: single rotation
        theta = 2 * arccos(amplitudes[0] / norm(amplitudes))
        RY(theta, wires[0])
    else:
        # Recursive case: split and apply controlled ops
        ...
```

### 5. IQP (Instantaneous Quantum Polynomial) Encoding

#### Definition

$$U_{\text{IQP}}(\mathbf{x}) = D(\mathbf{x}) H^{\otimes n}$$

Where $D(\mathbf{x})$ is a diagonal unitary:

$$D(\mathbf{x}) = \exp\left(i \sum_{S \subseteq [n]} f_S(\mathbf{x}) Z_S\right)$$

Here $Z_S = \prod_{k \in S} Z_k$.

#### Typical Choice

$$D(\mathbf{x}) = \exp\left(i \sum_k x_k Z_k + i \sum_{j<k} x_j x_k Z_j Z_k\right)$$

This encodes both linear and quadratic features.

#### Properties

| Property | Value |
|----------|-------|
| **Qubits needed** | $n \geq d$ for linear terms |
| **Gates** | $O(n + n^2) = O(n^2)$ |
| **Depth** | $O(n)$ with parallel ZZ gates |
| **Expressivity** | Encodes polynomial features |

#### Advantages

- All gates commute (diagonal + Hadamard)
- Conjectured classically hard to simulate
- Natural for polynomial feature encoding

### 6. Comparison of Encoding Methods

#### Summary Table

| Encoding | Qubits | Gates | Depth | Best For |
|----------|--------|-------|-------|----------|
| **Basis** | $d$ | $O(d)$ | 1 | Discrete data |
| **Angle** | $d$ | $O(d)$ | 1 | Continuous, few features |
| **Dense Angle** | $d/3$ | $O(d)$ | 3 | More features, less qubits |
| **Amplitude** | $\log_2 d$ | $O(d)$ | $O(\log d)$ to $O(d)$ | Many features |
| **IQP** | $d$ | $O(d^2)$ | $O(d)$ | Polynomial features |

#### Trade-off Visualization

```
                    Qubits
         Many ──┼──────────────┼── Few
              Angle     IQP     Amplitude
                │        │         │
                │        │         │
            Simple    Medium   Complex
                       Gates
```

### 7. Hybrid and Advanced Encodings

#### 7.1 Data Re-uploading Encoding

Repeat encoding at multiple layers:

$$|\psi\rangle = \prod_{l=1}^L W_l S(\mathbf{x}) |0\rangle$$

**Advantage:** Increases effective frequency spectrum.

#### 7.2 Hardware-Efficient Encoding

Design encoding for specific hardware:

**For superconducting qubits (IBM/Google):**
- Use native gates: $R_Z$, $\sqrt{X}$, CNOT
- Avoid decompositions that add depth

**For trapped ions (IonQ):**
- XX gates are native
- All-to-all connectivity allows dense encoding

#### 7.3 Trainable Encoding

Make the encoding itself learnable:

$$S(\mathbf{x}, \boldsymbol{\phi}) = \prod_k R(\phi_k + x_k \cdot w_k)$$

Where $\boldsymbol{\phi}, \mathbf{w}$ are trainable.

### 8. Encoding and Kernel Connection

#### Theorem (Schuld et al.)

The choice of encoding determines the implicit quantum kernel:

$$K(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$$

**Angle encoding kernel:**
$$K(\mathbf{x}, \mathbf{x}') = \prod_{i=1}^d \cos^2\frac{x_i - x_i'}{2}$$

**Amplitude encoding kernel:**
$$K(\mathbf{x}, \mathbf{x}') = \frac{|\mathbf{x}^\top \mathbf{x}'|^2}{||\mathbf{x}||^2 ||\mathbf{x}'||^2}$$

(This is the squared normalized inner product!)

**Implication:** Encoding choice is a modeling decision, not just preprocessing.

---

## Afternoon Session: Problem Solving (2 hours)

### Worked Example 1: Amplitude Encoding Circuit Design

**Problem:** Design a circuit to amplitude-encode the vector $\mathbf{x} = (1, 2, 3, 4)$ into a 2-qubit state.

**Solution:**

**Step 1: Normalize**

$$||\mathbf{x}|| = \sqrt{1^2 + 2^2 + 3^2 + 4^2} = \sqrt{30}$$

Normalized: $\tilde{\mathbf{x}} = \frac{1}{\sqrt{30}}(1, 2, 3, 4)$

Target state:
$$|\psi\rangle = \frac{1}{\sqrt{30}}(|00\rangle + 2|01\rangle + 3|10\rangle + 4|11\rangle)$$

**Step 2: First qubit rotation**

The first qubit should have:
- $P(|0\rangle) = (1^2 + 2^2)/30 = 5/30 = 1/6$
- $P(|1\rangle) = (3^2 + 4^2)/30 = 25/30 = 5/6$

$$R_Y(\theta_1)|0\rangle = \sqrt{1/6}|0\rangle + \sqrt{5/6}|1\rangle$$
$$\cos^2(\theta_1/2) = 1/6 \Rightarrow \theta_1 = 2\arccos(\sqrt{1/6}) \approx 2.31$$

**Step 3: Second qubit (controlled rotations)**

When first qubit is $|0\rangle$, second qubit needs:
- $|0\rangle$ with probability $1/5$, $|1\rangle$ with probability $4/5$
- Angle: $\theta_{2,0} = 2\arccos(\sqrt{1/5}) \approx 2.21$

When first qubit is $|1\rangle$, second qubit needs:
- $|0\rangle$ with probability $9/25$, $|1\rangle$ with probability $16/25$
- Angle: $\theta_{2,1} = 2\arccos(\sqrt{9/25}) = 2\arccos(3/5) \approx 1.85$

**Step 4: Circuit**

```
|0⟩ ─[RY(θ₁)]──●───────────●───────────
               │           │
|0⟩ ───────────[RY(θ₂₀)]──[RY(θ₂₁-θ₂₀)]─
               (if ctrl=0)  (if ctrl=1)
```

Actually, the controlled rotations are:
- $CR_Y(\theta_{2,0})$ controlled on $|0\rangle$ (anti-control)
- $CR_Y(\theta_{2,1})$ controlled on $|1\rangle$

---

### Worked Example 2: Comparing Encoding Expressivity

**Problem:** For a 4-feature dataset, compare angle encoding on 4 qubits vs. dense angle encoding on 2 qubits.

**Solution:**

**Angle Encoding (4 qubits):**
$$|\phi\rangle = R_Y(x_1)|0\rangle \otimes R_Y(x_2)|0\rangle \otimes R_Y(x_3)|0\rangle \otimes R_Y(x_4)|0\rangle$$

State space: Product states only → effective dimension = 4 (just the angles)

**Dense Angle Encoding (2 qubits):**
$$|\phi\rangle = R_Z(x_4)R_Y(x_3)R_X(x_1)|0\rangle \otimes R_Z(x_4)R_Y(x_2)R_X(x_3)|0\rangle$$

(Various arrangements possible)

State space: Still product states → effective dimension = 4

**Key Insight:** Without entanglement, both have the same expressivity! The dense encoding just uses fewer qubits.

**With entanglement (IQP-style):**
Add $e^{i x_i x_j Z_i Z_j}$ terms:

$$|\phi\rangle = U_{ZZ}(\mathbf{x}) \cdot (R_Y(x_1) \otimes R_Y(x_2) \otimes R_Y(x_3) \otimes R_Y(x_4)) \cdot H^{\otimes 4}|0000\rangle$$

Now the state space includes entangled states, dramatically increasing expressivity.

---

### Worked Example 3: Shot Budget for Amplitude Encoding

**Problem:** You want to use amplitude encoding for 256 features on 8 qubits. Estimate the circuit resources.

**Solution:**

**Qubit requirement:** $\lceil\log_2 256\rceil = 8$ qubits ✓

**Gate count:**
Using Shende et al. decomposition:
$$\text{CNOTs} \approx \frac{23}{48} \cdot 4^8 \approx 31,000$$

This is prohibitively expensive!

**Alternative: Sparse amplitude encoding**

If only $k = 16$ features are non-zero:
$$\text{Gates} = O(k \cdot n) = O(16 \cdot 8) = O(128)$$

Much more feasible.

**Practical consideration:**

Amplitude encoding is often **not practical** for large feature dimensions due to gate complexity. Use angle encoding with fewer features, or dimensionality reduction first.

---

### Practice Problems

#### Problem 1: Qubit Efficiency (Direct Application)

You have a dataset with 100 continuous features.

a) How many qubits needed for angle encoding (1 feature/qubit)?
b) How many qubits for dense angle encoding (3 features/qubit)?
c) How many qubits for amplitude encoding?
d) Which is most qubit-efficient?

<details>
<summary>Solution</summary>

a) 100 qubits

b) $\lceil 100/3 \rceil = 34$ qubits

c) $\lceil \log_2 100 \rceil = 7$ qubits (need 128 amplitudes)

d) Amplitude encoding is most qubit-efficient, but has exponential gate complexity.
</details>

#### Problem 2: Kernel Analysis (Intermediate)

For angle encoding $|\phi(x)\rangle = R_Y(x)|0\rangle$ on a single qubit:

a) Write the state explicitly
b) Compute $K(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2$
c) Compare to RBF kernel $K_{RBF}(x, x') = e^{-\gamma(x-x')^2}$
d) For what data structure might the quantum kernel be better?

<details>
<summary>Solution</summary>

a) $|\phi(x)\rangle = \cos(x/2)|0\rangle + \sin(x/2)|1\rangle$

b) $\langle\phi(x)|\phi(x')\rangle = \cos(x/2)\cos(x'/2) + \sin(x/2)\sin(x'/2) = \cos\frac{x-x'}{2}$

$K(x, x') = \cos^2\frac{x-x'}{2}$

c) The quantum kernel is **periodic** (repeats every $4\pi$), while RBF is not.

d) Quantum kernel may be better for:
- Periodic data (seasonal patterns)
- Circular/angular features
- Data naturally in $[0, 2\pi]$ range
</details>

#### Problem 3: Circuit Optimization (Challenging)

Design an efficient circuit to encode $\mathbf{x} = (x_1, x_2)$ such that the resulting kernel has the form:

$$K(\mathbf{x}, \mathbf{x}') = e^{-\gamma||\mathbf{x} - \mathbf{x}'||^2}$$

(i.e., approximate the RBF kernel quantumly)

<details>
<summary>Solution</summary>

This is a non-trivial problem! The Gaussian (RBF) kernel cannot be exactly realized with finite-depth quantum circuits.

**Approximation approach:**

Taylor expand: $e^{-u} \approx 1 - u + u^2/2 - ...$

The quantum kernel from IQP encoding has the form:
$K = \prod_i \cos^2(\cdots)$

Using $\cos^2(t) = \frac{1 + \cos(2t)}{2} \approx 1 - t^2$ for small $t$.

**Strategy:**
1. Scale data: $\mathbf{x}' = \sqrt{\gamma}\mathbf{x}$
2. Use encoding: $U(\mathbf{x}) = \exp(i(x_1 Z_1 + x_2 Z_2))H^{\otimes 2}$
3. The resulting kernel is approximately Gaussian for small differences.

Alternatively, use random Fourier features approach: sample frequencies and encode phases.
</details>

---

## Evening Session: Computational Lab (2 hours)

### Lab: Comparing Data Encoding Strategies

```python
"""
Day 971 Lab: Data Encoding Strategies
Comparing different encoding methods for QML
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

np.random.seed(42)

print("=" * 60)
print("Data Encoding Strategies Comparison")
print("=" * 60)

#######################################
# Part 1: Implementing Different Encodings
#######################################

print("\n" + "-" * 40)
print("Part 1: Encoding Implementations")
print("-" * 40)

n_qubits = 2
dev = qml.device('default.qubit', wires=n_qubits)

# 1. Angle Encoding
def angle_encoding(x, wires):
    """Simple angle encoding: one rotation per qubit"""
    for i, w in enumerate(wires):
        if i < len(x):
            qml.RY(x[i], wires=w)

# 2. Dense Angle Encoding
def dense_angle_encoding(x, wires):
    """Dense angle encoding: multiple rotations per qubit"""
    n_features = len(x)
    n_wires = len(wires)

    for i, w in enumerate(wires):
        # Up to 3 features per qubit
        idx = 3 * i
        if idx < n_features:
            qml.RX(x[idx], wires=w)
        if idx + 1 < n_features:
            qml.RY(x[idx + 1], wires=w)
        if idx + 2 < n_features:
            qml.RZ(x[idx + 2], wires=w)

# 3. IQP Encoding
def iqp_encoding(x, wires):
    """IQP-style encoding with ZZ interactions"""
    n_features = min(len(x), len(wires))

    # Hadamard layer
    for w in wires:
        qml.Hadamard(wires=w)

    # Z rotations (linear terms)
    for i in range(n_features):
        qml.RZ(2 * x[i], wires=wires[i])

    # ZZ interactions (quadratic terms)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2 * x[i] * x[j], wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])

# 4. Amplitude Encoding (for 2 features on 1 qubit)
def amplitude_encoding_2d(x, wires):
    """Encode 2D data in amplitude of single qubit"""
    # Normalize
    norm = np.sqrt(x[0]**2 + x[1]**2 + 1e-10)
    x_norm = x / norm

    # Rotation angle
    theta = 2 * np.arccos(np.clip(x_norm[0], -1, 1))
    qml.RY(theta, wires=wires[0])

# 5. Entangled Angle Encoding
def entangled_angle_encoding(x, wires):
    """Angle encoding with entanglement"""
    for i, w in enumerate(wires):
        if i < len(x):
            qml.RY(x[i], wires=w)

    # Add entanglement
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(x[0] * x[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


# Visualize circuits
print("\n1. Angle Encoding:")
@qml.qnode(dev)
def angle_circuit(x):
    angle_encoding(x, range(n_qubits))
    return qml.state()

x_example = np.array([0.5, 1.0])
print(qml.draw(angle_circuit)(x_example))

print("\n2. IQP Encoding:")
@qml.qnode(dev)
def iqp_circuit(x):
    iqp_encoding(x, range(n_qubits))
    return qml.state()

print(qml.draw(iqp_circuit)(x_example))

print("\n3. Entangled Angle Encoding:")
@qml.qnode(dev)
def entangled_circuit(x):
    entangled_angle_encoding(x, range(n_qubits))
    return qml.state()

print(qml.draw(entangled_circuit)(x_example))


#######################################
# Part 2: Kernel Analysis
#######################################

print("\n" + "-" * 40)
print("Part 2: Kernel Analysis")
print("-" * 40)

def create_kernel_function(encoding_func):
    """Create a kernel function from an encoding"""
    @qml.qnode(dev)
    def kernel(x1, x2):
        encoding_func(x1, range(n_qubits))
        qml.adjoint(encoding_func)(x2, range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    def kernel_value(x1, x2):
        return kernel(x1, x2)[0]

    return kernel_value

# Create kernel functions
kernel_angle = create_kernel_function(angle_encoding)
kernel_iqp = create_kernel_function(iqp_encoding)
kernel_entangled = create_kernel_function(entangled_angle_encoding)

# Test
x1 = np.array([0.5, 1.0])
x2 = np.array([0.5, 1.0])
x3 = np.array([1.5, 0.3])

print("Kernel values K(x1, x1):")
print(f"  Angle:     {kernel_angle(x1, x1):.4f}")
print(f"  IQP:       {kernel_iqp(x1, x1):.4f}")
print(f"  Entangled: {kernel_entangled(x1, x1):.4f}")

print("\nKernel values K(x1, x3):")
print(f"  Angle:     {kernel_angle(x1, x3):.4f}")
print(f"  IQP:       {kernel_iqp(x1, x3):.4f}")
print(f"  Entangled: {kernel_entangled(x1, x3):.4f}")


#######################################
# Part 3: Kernel Surface Visualization
#######################################

print("\n" + "-" * 40)
print("Part 3: Kernel Surface Visualization")
print("-" * 40)

# Fix reference point, vary test point
ref_point = np.array([np.pi/2, np.pi/2])
grid_size = 30
x_range = np.linspace(0, np.pi, grid_size)
y_range = np.linspace(0, np.pi, grid_size)

kernels = {
    'Angle': kernel_angle,
    'IQP': kernel_iqp,
    'Entangled': kernel_entangled
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, kernel_func) in enumerate(kernels.items()):
    K_surface = np.zeros((grid_size, grid_size))
    for i, xi in enumerate(x_range):
        for j, yj in enumerate(y_range):
            K_surface[i, j] = kernel_func(ref_point, np.array([xi, yj]))

    im = axes[idx].imshow(K_surface.T, origin='lower',
                           extent=[0, np.pi, 0, np.pi],
                           cmap='viridis', aspect='equal')
    axes[idx].scatter([ref_point[0]], [ref_point[1]], c='red', s=100,
                       marker='*', label='Reference')
    axes[idx].set_xlabel('x₁')
    axes[idx].set_ylabel('x₂')
    axes[idx].set_title(f'{name} Kernel')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig('encoding_kernels.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 4: Classification Comparison
#######################################

print("\n" + "-" * 40)
print("Part 4: Classification Comparison")
print("-" * 40)

# Generate dataset
X, y = make_moons(n_samples=80, noise=0.15, random_state=42)
y = 2 * y - 1

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

def compute_kernel_matrix(X1, X2, kernel_func):
    """Compute kernel matrix"""
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j])
    return K

def train_and_evaluate(kernel_func, name):
    """Train SVM with kernel and evaluate"""
    K_train = compute_kernel_matrix(X_train, X_train, kernel_func)
    K_test = compute_kernel_matrix(X_test, X_train, kernel_func)

    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)

    train_acc = svm.score(K_train, y_train)
    test_acc = svm.score(K_test, y_test)

    print(f"{name:15s} | Train: {train_acc:.2%} | Test: {test_acc:.2%}")
    return test_acc

print("\nSVM Classification Results:")
print("-" * 50)

results = {}
for name, kernel_func in kernels.items():
    results[name] = train_and_evaluate(kernel_func, name)

# Also compare with classical RBF
from sklearn.metrics.pairwise import rbf_kernel
K_rbf_train = rbf_kernel(X_train, gamma=1.0)
K_rbf_test = rbf_kernel(X_test, X_train, gamma=1.0)
svm_rbf = SVC(kernel='precomputed')
svm_rbf.fit(K_rbf_train, y_train)
results['RBF (γ=1)'] = svm_rbf.score(K_rbf_test, y_test)
print(f"{'RBF (γ=1)':15s} | Train: {svm_rbf.score(K_rbf_train, y_train):.2%} | Test: {results['RBF (γ=1)']:.2%}")


#######################################
# Part 5: Encoding Expressivity Analysis
#######################################

print("\n" + "-" * 40)
print("Part 5: Encoding Expressivity Analysis")
print("-" * 40)

def measure_encoding_expressivity(encoding_func, n_samples=500):
    """Measure how much of state space is covered by encoding"""
    @qml.qnode(dev)
    def get_state(x):
        encoding_func(x, range(n_qubits))
        return qml.state()

    states = []
    for _ in range(n_samples):
        x = np.random.uniform(0, np.pi, 2)
        state = get_state(x)
        states.append(state)

    states = np.array(states)

    # Measure pairwise distances
    distances = []
    for i in range(min(100, n_samples)):
        for j in range(i + 1, min(100, n_samples)):
            fidelity = np.abs(np.dot(states[i].conj(), states[j]))**2
            distances.append(1 - fidelity)

    return np.mean(distances), np.std(distances)

print("Encoding Expressivity (mean pairwise distance):")
print("-" * 50)

encodings = {
    'Angle': angle_encoding,
    'IQP': iqp_encoding,
    'Entangled': entangled_angle_encoding
}

expressivity_results = {}
for name, enc_func in encodings.items():
    mean_dist, std_dist = measure_encoding_expressivity(enc_func)
    expressivity_results[name] = mean_dist
    print(f"{name:15s} | Mean Distance: {mean_dist:.4f} +/- {std_dist:.4f}")

print("\nHigher mean distance indicates more coverage of state space.")


#######################################
# Part 6: Gate Count Analysis
#######################################

print("\n" + "-" * 40)
print("Part 6: Gate Count Analysis")
print("-" * 40)

def count_gates(encoding_func, n_features, n_qubits_count):
    """Count gates in encoding circuit"""
    dev_count = qml.device('default.qubit', wires=n_qubits_count)

    @qml.qnode(dev_count)
    def circuit(x):
        encoding_func(x, range(n_qubits_count))
        return qml.state()

    x = np.random.uniform(0, 1, n_features)
    specs = qml.specs(circuit)(x)

    return specs['resources'].num_gates

print("Gate counts for 2D data on 2 qubits:")
print("-" * 40)

for name, enc_func in encodings.items():
    try:
        n_gates = count_gates(enc_func, 2, 2)
        print(f"{name:15s} | {n_gates} gates")
    except:
        print(f"{name:15s} | (error counting)")


#######################################
# Part 7: Scaling Analysis
#######################################

print("\n" + "-" * 40)
print("Part 7: Scaling with Feature Dimension")
print("-" * 40)

def analyze_encoding_scaling():
    """Analyze how encoding resources scale with features"""

    dimensions = [2, 4, 8, 16]
    results = {
        'dimension': dimensions,
        'angle_qubits': [],
        'dense_angle_qubits': [],
        'amplitude_qubits': [],
        'iqp_gates': []
    }

    for d in dimensions:
        # Angle encoding: 1 qubit per feature
        results['angle_qubits'].append(d)

        # Dense angle: 3 features per qubit
        results['dense_angle_qubits'].append(int(np.ceil(d / 3)))

        # Amplitude: log2(d) qubits
        results['amplitude_qubits'].append(int(np.ceil(np.log2(d))))

        # IQP: O(d^2) gates for pairwise interactions
        results['iqp_gates'].append(d + d * (d - 1) // 2)

    return results

scaling = analyze_encoding_scaling()

print("Qubit requirements:")
print("-" * 60)
print(f"{'Dimension':<12} {'Angle':<10} {'Dense':<10} {'Amplitude':<10}")
print("-" * 60)
for i, d in enumerate(scaling['dimension']):
    print(f"{d:<12} {scaling['angle_qubits'][i]:<10} "
          f"{scaling['dense_angle_qubits'][i]:<10} "
          f"{scaling['amplitude_qubits'][i]:<10}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

dims = scaling['dimension']
axes[0].plot(dims, scaling['angle_qubits'], 'o-', label='Angle', linewidth=2)
axes[0].plot(dims, scaling['dense_angle_qubits'], 's-', label='Dense Angle', linewidth=2)
axes[0].plot(dims, scaling['amplitude_qubits'], '^-', label='Amplitude', linewidth=2)
axes[0].set_xlabel('Feature Dimension')
axes[0].set_ylabel('Qubits Required')
axes[0].set_title('Qubit Scaling')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(dims, scaling['iqp_gates'], 'o-', color='purple', linewidth=2)
axes[1].set_xlabel('Feature Dimension')
axes[1].set_ylabel('Number of Gates')
axes[1].set_title('IQP Encoding Gate Scaling')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('encoding_scaling.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 8: Practical Recommendations
#######################################

print("\n" + "-" * 40)
print("Part 8: Encoding Selection Guide")
print("-" * 40)

recommendations = """
ENCODING SELECTION GUIDE
========================

1. ANGLE ENCODING
   - Best for: Few continuous features (< 10)
   - Pros: Simple, shallow circuits
   - Cons: Linear qubit scaling

2. DENSE ANGLE ENCODING
   - Best for: Medium feature count, limited qubits
   - Pros: 3x qubit efficiency vs angle
   - Cons: Slightly deeper circuits

3. AMPLITUDE ENCODING
   - Best for: High-dimensional sparse data
   - Pros: Logarithmic qubit scaling
   - Cons: Exponential gate complexity

4. IQP ENCODING
   - Best for: Data with important feature interactions
   - Pros: Captures polynomial features
   - Cons: Quadratic gate scaling

5. ENTANGLED ANGLE
   - Best for: General QML applications
   - Pros: Good expressivity, moderate depth
   - Cons: More complex than simple angle

PRACTICAL TIPS:
- Start with angle encoding, add complexity if needed
- Use dimensionality reduction before encoding for large d
- Match encoding periodicity to data structure
- Consider hardware native gates
"""

print(recommendations)

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Encoding | Formula |
|----------|---------|
| **Basis** | $\|\mathbf{x}\rangle = \|x_1 x_2 \cdots x_d\rangle$ |
| **Angle** | $\|\phi(\mathbf{x})\rangle = \bigotimes_i R_Y(x_i)\|0\rangle$ |
| **Amplitude** | $\|\phi(\mathbf{x})\rangle = \sum_i \frac{x_i}{\|\|\mathbf{x}\|\|}\|i\rangle$ |
| **IQP** | $U_{\text{IQP}} = e^{i\sum x_i Z_i + i\sum x_i x_j Z_i Z_j} H^{\otimes n}$ |

### Resource Comparison

| Encoding | Qubits | Gates | Depth |
|----------|--------|-------|-------|
| Basis | $d$ | $O(d)$ | 1 |
| Angle | $d$ | $O(d)$ | 1 |
| Dense Angle | $d/3$ | $O(d)$ | 3 |
| Amplitude | $\log d$ | $O(d)$ | $O(d)$ |
| IQP | $d$ | $O(d^2)$ | $O(d)$ |

### Key Takeaways

1. **Encoding choice** determines the implicit quantum kernel
2. **Qubit efficiency** vs. **gate complexity** is a fundamental trade-off
3. **Amplitude encoding** is qubit-efficient but gate-expensive
4. **Entanglement** is necessary for non-product-state expressivity
5. **Data scaling** to appropriate ranges is crucial
6. **Hardware constraints** should influence encoding choice

---

## Daily Checklist

- [ ] I can implement basis, angle, and amplitude encodings
- [ ] I understand the resource requirements of each encoding
- [ ] I can analyze the kernel induced by an encoding
- [ ] I can select appropriate encodings for different data types
- [ ] I understand the trade-offs between encodings
- [ ] I implemented and compared multiple encodings in PennyLane

---

## Preview: Day 972

Tomorrow we tackle **Expressibility and Trainability** - the critical balance in QML:

- Formal measures of circuit expressibility
- Barren plateaus and their causes
- Strategies to avoid vanishing gradients
- The expressibility-trainability trade-off
- Practical circuit design principles

---

*"The choice of how to embed classical data into a quantum state is not a preprocessing step - it is the heart of quantum machine learning."*
— Maria Schuld
