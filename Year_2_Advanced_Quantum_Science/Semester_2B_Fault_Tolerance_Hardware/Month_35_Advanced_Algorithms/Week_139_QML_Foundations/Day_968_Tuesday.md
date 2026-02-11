# Day 968: Variational Quantum Classifiers

## Year 2, Semester 2B: Fault Tolerance & Hardware
## Month 35: Advanced Algorithms - Week 139: QML Foundations

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Theory of variational quantum classifiers |
| **Afternoon** | 2 hours | Problem solving: classifier design and analysis |
| **Evening** | 2 hours | Full VQC implementation with training |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Construct a variational quantum classifier** with encoding and ansatz layers
2. **Design cost functions** appropriate for classification tasks
3. **Implement hybrid training loops** combining quantum circuits with classical optimizers
4. **Analyze decision boundaries** created by quantum classifiers
5. **Apply VQC** to binary and multi-class classification problems
6. **Compare VQC performance** with classical methods on benchmark datasets

---

## Morning Session: Theory (3 hours)

### 1. The Variational Quantum Classifier Framework

#### Architecture Overview

A **Variational Quantum Classifier (VQC)** combines three key components:

$$\boxed{|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = W(\boldsymbol{\theta}) \cdot U_\phi(\mathbf{x}) |0\rangle^{\otimes n}}$$

Where:
- $U_\phi(\mathbf{x})$ is the **feature map** (data encoding)
- $W(\boldsymbol{\theta})$ is the **ansatz** (trainable unitary)
- $\boldsymbol{\theta}$ are the **variational parameters**

The classification is performed by measuring an observable:

$$\hat{y} = f\left(\langle\psi(\mathbf{x}, \boldsymbol{\theta})|\hat{M}|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle\right)$$

#### Circuit Diagram

```
|0⟩ ─┤ U_φ(x) ├──┤ W(θ) ├──┤ M ├── → ŷ
     ╰─────────╯  ╰──────╯  ╰───╯
     Data Encoding  Ansatz  Measurement
```

### 2. Cost Functions for Classification

#### 2.1 Binary Classification

For binary labels $y \in \{-1, +1\}$, we measure an observable with eigenvalues $\pm 1$ (e.g., $Z$ operator):

$$\langle Z \rangle = \langle\psi|\hat{Z}|\psi\rangle \in [-1, +1]$$

**Prediction:**
$$\hat{y} = \text{sign}(\langle Z \rangle)$$

**Cost Function (Squared Error):**
$$L(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N \left(y_i - \langle Z \rangle_i\right)^2$$

**Cost Function (Hinge Loss):**
$$L(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N \max(0, 1 - y_i \langle Z \rangle_i)$$

#### 2.2 Probability-Based Classification

Using measurement probabilities directly:

$$p_0 = |\langle 0|\psi\rangle|^2, \quad p_1 = |\langle 1|\psi\rangle|^2$$

**Cross-Entropy Loss:**
$$L = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log(p_1^{(i)}) + (1-y_i)\log(p_0^{(i)})\right]$$

For labels $y \in \{0, 1\}$.

#### 2.3 Multi-Class Classification

For $K$ classes, use $\lceil \log_2 K \rceil$ qubits and map basis states to classes:

$$P(\text{class } k | \mathbf{x}) = |\langle k|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle|^2$$

**Softmax Mapping:**
Alternatively, use $K$ separate expectation values and apply softmax:

$$P(\text{class } k) = \frac{e^{\langle M_k \rangle}}{\sum_{j=1}^K e^{\langle M_j \rangle}}$$

### 3. Ansatz Design

#### 3.1 Hardware-Efficient Ansatz

Designed for NISQ devices with native gate sets:

$$W(\boldsymbol{\theta}) = \prod_{l=1}^L \left[\text{Entangle} \cdot \bigotimes_{i=1}^n R(\theta_{l,i})\right]$$

**Example Layer:**
```
─[RY(θ₁)]─●──────────
          │
─[RY(θ₂)]─X──●───────
             │
─[RY(θ₃)]───X──●─────
               │
─[RY(θ₄)]─────X──────
```

#### 3.2 Strongly Entangling Layers

More expressive with all-to-all connectivity (simulated):

$$W(\boldsymbol{\theta}) = \prod_{l=1}^L \left[\prod_{i<j} \text{CNOT}_{ij} \cdot \bigotimes_i R(\theta_{l,i})\right]$$

#### 3.3 Number of Parameters

For $n$ qubits, $L$ layers, with 3-parameter rotations:
$$|\boldsymbol{\theta}| = 3nL$$

**Example:** 4 qubits, 3 layers → 36 parameters

### 4. Training: Hybrid Quantum-Classical Loop

#### The Optimization Cycle

```
1. Initialize θ randomly
2. For each iteration:
   a. Compute quantum circuit outputs for training data
   b. Calculate cost function L(θ)
   c. Compute gradients ∇_θ L
   d. Update: θ ← θ - η ∇_θ L
3. Until convergence
```

#### 4.1 Parameter-Shift Rule

For circuits with Pauli rotation gates, gradients can be computed exactly:

$$\frac{\partial}{\partial \theta_j}\langle M \rangle = \frac{1}{2}\left[\langle M \rangle_{\theta_j + \pi/2} - \langle M \rangle_{\theta_j - \pi/2}\right]$$

This requires **two circuit evaluations per parameter** per training sample.

#### 4.2 Gradient-Free Optimization

Alternatives when gradients are costly:
- **COBYLA** (Constrained Optimization BY Linear Approximation)
- **SPSA** (Simultaneous Perturbation Stochastic Approximation)
- **Nelder-Mead** (Simplex method)

SPSA uses only 2 circuit evaluations per iteration (regardless of parameter count):

$$\hat{\nabla} L = \frac{L(\boldsymbol{\theta} + c\Delta) - L(\boldsymbol{\theta} - c\Delta)}{2c\Delta}$$

### 5. Decision Boundaries

#### How VQC Creates Separations

The VQC creates a decision boundary in input space:

$$\mathcal{B} = \{\mathbf{x} : \langle\psi(\mathbf{x}, \boldsymbol{\theta})|\hat{M}|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = b\}$$

Where $b$ is the classification threshold (typically 0 for $\pm 1$ labels).

#### Complexity of Boundaries

The decision boundary complexity depends on:
1. **Feature map expressivity:** More entangled maps → more complex boundaries
2. **Ansatz depth:** Deeper circuits → more flexible boundaries
3. **Number of qubits:** More qubits → higher-dimensional feature space

**Key Insight:** Quantum classifiers can create **non-linear boundaries** in the original input space even when the quantum observable is a linear operator, because the feature map is non-linear.

### 6. Connection to Classical ML

#### VQC vs. Classical Neural Networks

| Aspect | VQC | Classical NN |
|--------|-----|--------------|
| **Feature Space** | $2^n$-dim Hilbert space | Hidden layer dimensions |
| **Parameters** | $O(nL)$ | $O(\sum_i d_i \times d_{i+1})$ |
| **Non-linearity** | Quantum feature map | Activation functions |
| **Training** | Parameter-shift rule | Backpropagation |

#### VQC as a Linear Model in Feature Space

In the quantum feature space, VQC is a linear classifier:

$$\hat{y} = \text{sign}(\mathbf{w}^\top \phi(\mathbf{x}) + b)$$

Where the "weights" are implicitly defined by $W(\boldsymbol{\theta})$ and the measurement $\hat{M}$.

---

## Afternoon Session: Problem Solving (2 hours)

### Worked Example 1: Designing a VQC for XOR Problem

**Problem:** Design a VQC to classify the XOR dataset:
- $(0, 0) \rightarrow -1$
- $(0, 1) \rightarrow +1$
- $(1, 0) \rightarrow +1$
- $(1, 1) \rightarrow -1$

**Solution:**

**Step 1: Choose Feature Map**

The XOR problem requires a non-linear separator. Use entangling feature map:

$$U_\phi(\mathbf{x}) = \text{CNOT}_{12} \cdot (H \otimes H) \cdot (R_Z(x_1) \otimes R_Z(x_2))$$

But first, scale data from $\{0, 1\}$ to $\{0, \pi\}$: $x \mapsto \pi x$.

**Step 2: Design Ansatz**

Simple ansatz with 2 layers:

$$W(\boldsymbol{\theta}) = \text{CNOT}_{12} \cdot R_Y(\theta_2) \otimes R_Y(\theta_4) \cdot \text{CNOT}_{12} \cdot R_Y(\theta_1) \otimes R_Y(\theta_3)$$

Total: 4 parameters.

**Step 3: Measurement**

Measure $Z_1$ (first qubit):
$$\hat{y} = \text{sign}(\langle Z_1 \rangle)$$

**Step 4: Training**

Minimize:
$$L(\boldsymbol{\theta}) = \frac{1}{4}\sum_{i=1}^4 (y_i - \langle Z_1 \rangle_i)^2$$

After training, the circuit should achieve 100% accuracy on this simple dataset.

---

### Worked Example 2: Gradient Calculation

**Problem:** For a single-qubit VQC $|\psi(\theta)\rangle = R_Y(\theta)|0\rangle$ with measurement $\langle Z \rangle$, compute $\frac{\partial \langle Z \rangle}{\partial \theta}$ analytically and verify the parameter-shift rule.

**Solution:**

**Step 1: Compute the state**

$$|\psi(\theta)\rangle = R_Y(\theta)|0\rangle = \cos\frac{\theta}{2}|0\rangle + \sin\frac{\theta}{2}|1\rangle$$

**Step 2: Compute expectation value**

$$\langle Z \rangle = \cos^2\frac{\theta}{2} - \sin^2\frac{\theta}{2} = \cos\theta$$

**Step 3: Analytical gradient**

$$\frac{\partial \langle Z \rangle}{\partial \theta} = -\sin\theta$$

**Step 4: Parameter-shift rule**

$$\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{1}{2}\left[\cos(\theta + \pi/2) - \cos(\theta - \pi/2)\right]$$
$$= \frac{1}{2}\left[-\sin\theta - (-(-\sin\theta))\right]$$
$$= \frac{1}{2}\left[-\sin\theta - \sin\theta\right] = -\sin\theta$$

**Verification:** Matches analytical result.

---

### Worked Example 3: Cost Function Comparison

**Problem:** Compare squared error vs. hinge loss for a binary classifier with labels $y \in \{-1, +1\}$ and predictions $\langle Z \rangle \in [-1, +1]$.

**Solution:**

Consider three scenarios for a sample with $y = +1$:

| Prediction $\langle Z \rangle$ | Correct? | Squared Error | Hinge Loss |
|------------------------------|----------|---------------|------------|
| +0.9 | Yes | $(1 - 0.9)^2 = 0.01$ | $\max(0, 1 - 0.9) = 0.1$ |
| +0.1 | Marginally | $(1 - 0.1)^2 = 0.81$ | $\max(0, 1 - 0.1) = 0.9$ |
| -0.5 | No | $(1 - (-0.5))^2 = 2.25$ | $\max(0, 1 - (-0.5)) = 1.5$ |

**Key Observations:**
1. Squared error penalizes all deviations, even correct ones
2. Hinge loss gives zero penalty for predictions with $y \cdot \langle Z \rangle \geq 1$
3. Hinge loss gradient is constant for misclassified points (better for optimization)
4. Squared error can have very large values for wrong predictions

**Recommendation:** Use squared error for regression-like behavior; hinge loss for margin-based classification.

---

### Practice Problems

#### Problem 1: Parameter Count (Direct Application)

A VQC uses:
- 6 qubits
- ZZ feature map with 2 repetitions (no trainable parameters)
- Strongly entangling ansatz with 4 layers, using $R_X, R_Y, R_Z$ per qubit

a) How many trainable parameters are there?
b) How many CNOT gates are in the ansatz (assuming linear connectivity)?
c) If gradients are computed via parameter-shift, how many circuit evaluations per training sample?

<details>
<summary>Solution</summary>

a) Parameters: $3 \times 6 \times 4 = 72$ parameters

b) CNOTs: Linear connectivity means $n-1 = 5$ CNOTs per layer, total $5 \times 4 = 20$ CNOTs

c) Circuit evaluations: $2 \times 72 = 144$ evaluations per sample (plus 1 for forward pass = 145 total)
</details>

#### Problem 2: Decision Boundary (Intermediate)

For a 1D classification problem with data $x \in [0, 2\pi]$, a VQC uses:
- Feature map: $R_Y(x)|0\rangle$
- Ansatz: $R_Y(\theta)|0\rangle$ (applied after feature map)
- Measurement: $\langle Z \rangle$

a) Express $\langle Z \rangle$ as a function of $x$ and $\theta$
b) Find the decision boundary (where $\langle Z \rangle = 0$)
c) For what $\theta$ values is the classifier trivial (always predicts same class)?

<details>
<summary>Solution</summary>

a) The state is $R_Y(\theta)R_Y(x)|0\rangle = R_Y(\theta + x)|0\rangle$

$\langle Z \rangle = \cos(\theta + x)$

b) Decision boundary: $\cos(\theta + x) = 0$

$\theta + x = \pm\frac{\pi}{2} + n\pi$

$x = -\theta \pm \frac{\pi}{2} + n\pi$ for integer $n$

c) Trivial classifier when $|\cos(\theta + x)| = 1$ for all $x$ in range, which never happens since $\cos$ varies over $[0, 2\pi]$. But if we limit to practical cases: no single $\theta$ makes it trivial over full range.
</details>

#### Problem 3: Multi-Class Extension (Challenging)

Design a VQC for 4-class classification using 2 qubits.

a) How do you map computational basis states to classes?
b) Write the probability for each class
c) What cost function would you use?
d) How would you handle imbalanced classes?

<details>
<summary>Solution</summary>

a) Mapping:
- $|00\rangle \rightarrow$ Class 0
- $|01\rangle \rightarrow$ Class 1
- $|10\rangle \rightarrow$ Class 2
- $|11\rangle \rightarrow$ Class 3

b) Probabilities:
$P(\text{class } k) = |\langle k|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle|^2$

Where $|k\rangle$ is the binary representation of $k$.

c) Cross-entropy loss:
$L = -\frac{1}{N}\sum_{i=1}^N \sum_{k=0}^3 \mathbb{1}_{y_i = k} \log P(\text{class } k | \mathbf{x}_i)$

d) Handle imbalance via:
- Class weights in loss: $L = -\sum_i w_{y_i} \log P(y_i | \mathbf{x}_i)$
- Oversampling minority classes
- Focal loss: $L = -\alpha_k (1 - P_k)^\gamma \log P_k$
</details>

---

## Evening Session: Computational Lab (2 hours)

### Lab: Implementing a Complete VQC Training Pipeline

```python
"""
Day 968 Lab: Variational Quantum Classifiers
Complete implementation with training pipeline
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set random seed
np.random.seed(42)

print("=" * 60)
print("Variational Quantum Classifier Implementation")
print("=" * 60)

#######################################
# Part 1: Data Preparation
#######################################

print("\n" + "-" * 40)
print("Part 1: Data Preparation")
print("-" * 40)

# Generate moon dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Convert labels from {0, 1} to {-1, +1}
y = 2 * y - 1

# Scale features to [0, pi]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# Visualize dataset
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class +1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='x', label='Class -1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Moons Dataset')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='blue', marker='o', label='Class +1')
plt.scatter(X_scaled[y == -1, 0], X_scaled[y == -1, 1], c='red', marker='x', label='Class -1')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('Scaled Dataset [0, π]')
plt.legend()

plt.tight_layout()
plt.savefig('vqc_data.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 2: VQC Circuit Definition
#######################################

print("\n" + "-" * 40)
print("Part 2: VQC Circuit Definition")
print("-" * 40)

n_qubits = 2
n_layers = 3

# Create quantum device
dev = qml.device('default.qubit', wires=n_qubits)

def feature_map(x):
    """
    ZZ-style feature map with data re-uploading
    """
    # First layer: Hadamard + data encoding
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # Single-qubit rotations with data
    for i in range(min(len(x), n_qubits)):
        qml.RZ(x[i], wires=i)

    # Entangling layer with data-dependent angle
    qml.CNOT(wires=[0, 1])
    qml.RZ(x[0] * x[1], wires=1)  # Product feature
    qml.CNOT(wires=[0, 1])

def variational_layer(params, layer_idx):
    """
    Single layer of the variational ansatz
    params shape: (n_qubits, 3) for RX, RY, RZ rotations
    """
    for i in range(n_qubits):
        qml.RX(params[layer_idx, i, 0], wires=i)
        qml.RY(params[layer_idx, i, 1], wires=i)
        qml.RZ(params[layer_idx, i, 2], wires=i)

    # Entangling gates (circular connectivity)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])

@qml.qnode(dev)
def vqc_circuit(x, params):
    """
    Full VQC circuit: feature map + variational layers
    Returns expectation value of Z on first qubit
    """
    feature_map(x)

    for l in range(n_layers):
        variational_layer(params, l)

    return qml.expval(qml.PauliZ(0))

# Initialize parameters
params_shape = (n_layers, n_qubits, 3)
initial_params = np.random.uniform(-np.pi, np.pi, params_shape)

print(f"Number of qubits: {n_qubits}")
print(f"Number of layers: {n_layers}")
print(f"Total parameters: {np.prod(params_shape)}")

# Visualize circuit
print("\nCircuit structure:")
print(qml.draw(vqc_circuit)(X_train[0], initial_params))


#######################################
# Part 3: Cost Function and Training
#######################################

print("\n" + "-" * 40)
print("Part 3: Training Pipeline")
print("-" * 40)

def cost_function(params, X, y):
    """
    Mean squared error cost function
    """
    predictions = np.array([vqc_circuit(x, params) for x in X])
    loss = np.mean((predictions - y) ** 2)
    return loss

def accuracy(params, X, y):
    """
    Classification accuracy
    """
    predictions = np.array([np.sign(vqc_circuit(x, params)) for x in X])
    return np.mean(predictions == y)

# Training with gradient descent
optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

# Use a subset for faster training (demonstration)
n_train_subset = 50
X_train_subset = X_train[:n_train_subset]
y_train_subset = y_train[:n_train_subset]

# Training loop
n_epochs = 50
params = initial_params.copy()

train_costs = []
train_accs = []
test_accs = []

print("\nTraining VQC...")
print("-" * 50)

for epoch in range(n_epochs):
    # Compute cost and update parameters
    params, cost = optimizer.step_and_cost(
        lambda p: cost_function(p, X_train_subset, y_train_subset),
        params
    )

    # Track metrics every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        train_acc = accuracy(params, X_train_subset, y_train_subset)
        test_acc = accuracy(params, X_test, y_test)

        train_costs.append(cost)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1:3d} | Cost: {cost:.4f} | "
              f"Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")

print("-" * 50)
print("Training complete!")


#######################################
# Part 4: Visualize Training Progress
#######################################

print("\n" + "-" * 40)
print("Part 4: Training Visualization")
print("-" * 40)

epochs_tracked = list(range(1, n_epochs + 1, 5))
epochs_tracked[0] = 1  # Fix first epoch

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cost over epochs
axes[0].plot(epochs_tracked, train_costs, 'b-o', linewidth=2, markersize=6)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cost (MSE)')
axes[0].set_title('Training Cost')
axes[0].grid(True, alpha=0.3)

# Accuracy over epochs
axes[1].plot(epochs_tracked, train_accs, 'b-o', label='Train', linewidth=2, markersize=6)
axes[1].plot(epochs_tracked, test_accs, 'r-s', label='Test', linewidth=2, markersize=6)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Classification Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.4, 1.05)

plt.tight_layout()
plt.savefig('vqc_training.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 5: Decision Boundary Visualization
#######################################

print("\n" + "-" * 40)
print("Part 5: Decision Boundary")
print("-" * 40)

def plot_decision_boundary(params, X, y, title, ax):
    """
    Plot decision boundary and data points
    """
    # Create grid
    h = 0.05
    x_min, x_max = 0, np.pi
    y_min, y_max = 0, np.pi
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([vqc_circuit(p, params) for p in grid_points])
    Z = Z.reshape(xx.shape)

    # Plot contour
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    # Plot data points
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o',
               edgecolors='black', s=50, label='Class +1')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='x',
               s=50, label='Class -1')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before training
plot_decision_boundary(initial_params, X_scaled, y, 'Before Training', axes[0])

# After training
plot_decision_boundary(params, X_scaled, y, 'After Training', axes[1])

plt.tight_layout()
plt.savefig('vqc_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 6: Comparison with Different Optimizers
#######################################

print("\n" + "-" * 40)
print("Part 6: Optimizer Comparison")
print("-" * 40)

def train_with_optimizer(optimizer, optimizer_name, n_epochs=30):
    """Train VQC with a specific optimizer"""
    params = np.random.uniform(-np.pi, np.pi, params_shape)
    costs = []

    for epoch in range(n_epochs):
        params, cost = optimizer.step_and_cost(
            lambda p: cost_function(p, X_train_subset, y_train_subset),
            params
        )
        costs.append(cost)

    final_acc = accuracy(params, X_test, y_test)
    print(f"{optimizer_name:20s} | Final Test Accuracy: {final_acc:.2%}")
    return costs

# Compare optimizers
optimizers = {
    'Gradient Descent': qml.GradientDescentOptimizer(stepsize=0.1),
    'Adam': qml.AdamOptimizer(stepsize=0.1),
    'Momentum': qml.MomentumOptimizer(stepsize=0.1, momentum=0.9),
}

plt.figure(figsize=(10, 6))

for name, opt in optimizers.items():
    costs = train_with_optimizer(opt, name, n_epochs=30)
    plt.plot(costs, label=name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('vqc_optimizer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 7: Effect of Circuit Depth
#######################################

print("\n" + "-" * 40)
print("Part 7: Effect of Circuit Depth")
print("-" * 40)

def create_vqc_with_depth(depth):
    """Create a VQC with specified depth"""
    @qml.qnode(dev)
    def circuit(x, params):
        feature_map(x)
        for l in range(depth):
            variational_layer(params, l)
        return qml.expval(qml.PauliZ(0))
    return circuit

depths = [1, 2, 3, 4, 5]
depth_results = []

for depth in depths:
    circuit = create_vqc_with_depth(depth)
    params_d = np.random.uniform(-np.pi, np.pi, (depth, n_qubits, 3))

    # Train
    optimizer = qml.AdamOptimizer(stepsize=0.1)
    for _ in range(30):
        def cost_d(p):
            preds = np.array([circuit(x, p) for x in X_train_subset])
            return np.mean((preds - y_train_subset) ** 2)
        params_d = optimizer.step(cost_d, params_d)

    # Evaluate
    preds = np.array([np.sign(circuit(x, params_d)) for x in X_test])
    acc = np.mean(preds == y_test)
    n_params = depth * n_qubits * 3

    depth_results.append({
        'depth': depth,
        'accuracy': acc,
        'n_params': n_params
    })
    print(f"Depth {depth} | Parameters: {n_params:3d} | Test Accuracy: {acc:.2%}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar([str(d['depth']) for d in depth_results],
        [d['accuracy'] for d in depth_results],
        color='steelblue', edgecolor='black')
plt.xlabel('Circuit Depth')
plt.ylabel('Test Accuracy')
plt.title('Accuracy vs. Circuit Depth')
plt.ylim(0.5, 1.0)

plt.subplot(1, 2, 2)
plt.bar([str(d['depth']) for d in depth_results],
        [d['n_params'] for d in depth_results],
        color='coral', edgecolor='black')
plt.xlabel('Circuit Depth')
plt.ylabel('Number of Parameters')
plt.title('Parameters vs. Circuit Depth')

plt.tight_layout()
plt.savefig('vqc_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 8: Gradient Magnitude Analysis
#######################################

print("\n" + "-" * 40)
print("Part 8: Gradient Analysis")
print("-" * 40)

@qml.qnode(dev)
def circuit_for_grad(x, params):
    feature_map(x)
    for l in range(n_layers):
        variational_layer(params, l)
    return qml.expval(qml.PauliZ(0))

# Compute gradients at different random initializations
n_samples = 20
gradient_norms = []

for _ in range(n_samples):
    random_params = np.random.uniform(-np.pi, np.pi, params_shape)

    # Compute gradient
    grad_fn = qml.grad(lambda p: cost_function(p, X_train_subset[:10], y_train_subset[:10]))
    grad = grad_fn(random_params)

    gradient_norms.append(np.linalg.norm(grad))

print(f"Mean gradient norm: {np.mean(gradient_norms):.4f}")
print(f"Std gradient norm: {np.std(gradient_norms):.4f}")
print(f"Min gradient norm: {np.min(gradient_norms):.4f}")
print(f"Max gradient norm: {np.max(gradient_norms):.4f}")

plt.figure(figsize=(8, 4))
plt.hist(gradient_norms, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(gradient_norms), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(gradient_norms):.3f}')
plt.xlabel('Gradient Norm')
plt.ylabel('Frequency')
plt.title('Distribution of Gradient Norms at Random Initialization')
plt.legend()
plt.savefig('vqc_gradient_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

### Expected Output Summary

```
Training VQC...
--------------------------------------------------
Epoch   1 | Cost: 1.0234 | Train Acc: 52.00% | Test Acc: 48.00%
Epoch   5 | Cost: 0.7821 | Train Acc: 64.00% | Test Acc: 62.00%
Epoch  10 | Cost: 0.4523 | Train Acc: 78.00% | Test Acc: 76.00%
Epoch  50 | Cost: 0.1234 | Train Acc: 92.00% | Test Acc: 88.00%
--------------------------------------------------
Training complete!
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| **VQC State** | $\|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = W(\boldsymbol{\theta}) U_\phi(\mathbf{x}) \|0\rangle$ |
| **Prediction** | $\hat{y} = \text{sign}(\langle Z \rangle)$ |
| **MSE Cost** | $L = \frac{1}{N}\sum_i (y_i - \langle Z \rangle_i)^2$ |
| **Parameter-Shift** | $\partial_\theta \langle M \rangle = \frac{1}{2}[\langle M \rangle_{\theta+\pi/2} - \langle M \rangle_{\theta-\pi/2}]$ |
| **Parameters** | $\|\boldsymbol{\theta}\| = 3nL$ for $n$ qubits, $L$ layers |

### Key Takeaways

1. **VQC = Feature Map + Ansatz + Measurement**: Each component plays a crucial role
2. **Hybrid training** combines quantum circuits with classical optimization
3. **Parameter-shift rule** enables exact gradient computation
4. **Circuit depth** affects both expressivity and trainability
5. **Decision boundaries** can be highly non-linear in input space
6. **Choice of optimizer** significantly impacts training dynamics

### Connection to Classical ML

- VQC is analogous to **kernel methods + neural networks**
- The feature map creates an implicit kernel
- The ansatz acts like a neural network in feature space
- Training is similar to classical backpropagation but uses quantum gradients

---

## Daily Checklist

- [ ] I understand the three components of a VQC (feature map, ansatz, measurement)
- [ ] I can design appropriate cost functions for classification
- [ ] I understand the parameter-shift rule for gradient computation
- [ ] I can implement a complete VQC training pipeline
- [ ] I can visualize decision boundaries
- [ ] I understand the trade-offs of circuit depth and parameters
- [ ] I can compare VQC with classical classifiers

---

## Preview: Day 969

Tomorrow we explore **Quantum Kernel Methods**, where instead of training a variational circuit, we use the quantum kernel directly:

$$K_Q(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$$

We will:
- Implement quantum kernels in PennyLane
- Use quantum kernels with classical SVMs
- Compare quantum and classical kernel performance
- Understand when quantum kernels may offer advantages

---

*"The variational quantum classifier is to quantum computing what the neural network is to classical computing - a flexible, trainable function approximator built from simple components."*
— Patrick Coles
