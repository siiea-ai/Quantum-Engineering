# Day 970: Quantum Neural Networks

## Year 2, Semester 2B: Fault Tolerance & Hardware
## Month 35: Advanced Algorithms - Week 139: QML Foundations

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | QNN architectures and design principles |
| **Afternoon** | 2 hours | Problem solving: QNN design and analysis |
| **Evening** | 2 hours | Building deep QNNs with PennyLane |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Design quantum neural network architectures** with appropriate layer structure
2. **Distinguish encoding layers from variational layers** and their roles
3. **Analyze QNN expressibility** and universality properties
4. **Implement deep quantum circuits** with multiple encoding-variational blocks
5. **Compare QNN architectures** to classical neural networks
6. **Apply QNNs** to regression and classification tasks

---

## Morning Session: Theory (3 hours)

### 1. What is a Quantum Neural Network?

#### Definition and Context

A **Quantum Neural Network (QNN)** is a parameterized quantum circuit that processes data through alternating layers of encoding and trainable unitaries:

$$\boxed{|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = \prod_{l=L}^{1} \left[W_l(\boldsymbol{\theta}_l) \cdot S_l(\mathbf{x})\right] |0\rangle^{\otimes n}}$$

Where:
- $S_l(\mathbf{x})$ = Encoding layer (embeds classical data)
- $W_l(\boldsymbol{\theta}_l)$ = Variational layer (trainable parameters)
- $L$ = Number of layers (depth)

#### QNN vs. Classical Neural Network

| Feature | Classical NN | Quantum NN |
|---------|-------------|------------|
| **Computation Space** | $\mathbb{R}^d$ | $\mathbb{C}^{2^n}$ (Hilbert space) |
| **Non-linearity** | Activation functions | Measurements |
| **Layer Operations** | Matrix multiplication | Unitary transformations |
| **Information Flow** | Forward propagation | Quantum evolution |
| **Gradients** | Backpropagation | Parameter-shift rule |

### 2. QNN Architecture Components

#### 2.1 Encoding Layers $S_l(\mathbf{x})$

Encoding layers embed classical data into the quantum state:

**Single-Layer Encoding:**
$$S(\mathbf{x}) = \exp\left(-i \sum_k x_k G_k\right)$$

Where $G_k$ are generators (e.g., Pauli operators).

**Common Choices:**
- **Angle Encoding:** $S(\mathbf{x}) = \bigotimes_k R_Y(x_k)$
- **IQP Encoding:** $S(\mathbf{x}) = D(\mathbf{x}) H^{\otimes n}$ (diagonal + Hadamard)
- **Amplitude Encoding:** Encode in state amplitudes

**Data Re-uploading:** Repeat encoding at each layer
$$|\psi\rangle = W_L S(\mathbf{x}) W_{L-1} S(\mathbf{x}) \cdots W_1 S(\mathbf{x}) |0\rangle$$

This is analogous to skip connections in classical networks.

#### 2.2 Variational Layers $W_l(\boldsymbol{\theta})$

Variational layers contain trainable parameters:

**General Form:**
$$W(\boldsymbol{\theta}) = \prod_{g=1}^G U_g(\theta_g)$$

Where each $U_g$ is a parameterized gate (rotation or entangling).

**Hardware-Efficient Ansatz:**
```
Layer l:
──[RX(θ₁)]──[RY(θ₂)]──[RZ(θ₃)]──●──────────
                                │
──[RX(θ₄)]──[RY(θ₅)]──[RZ(θ₆)]──X──●───────
                                   │
──[RX(θ₇)]──[RY(θ₈)]──[RZ(θ₉)]─────X──●────
                                      │
──[RX(θ₁₀)]─[RY(θ₁₁)]─[RZ(θ₁₂)]──────X────
```

#### 2.3 Measurement Layer

The measurement extracts classical information:

$$f(\mathbf{x}, \boldsymbol{\theta}) = \langle\psi(\mathbf{x}, \boldsymbol{\theta})|\hat{O}|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle$$

**Common Observables:**
- **Single qubit:** $\langle Z_1 \rangle$ for binary classification
- **Parity:** $\langle Z_1 Z_2 \cdots Z_n \rangle$
- **Multiple outputs:** $[\langle O_1 \rangle, \langle O_2 \rangle, \ldots]$ for multi-class

### 3. The Data Re-uploading Architecture

#### Motivation: Universal Approximation

**Theorem (Pérez-Salinas et al., 2020):** A single-qubit QNN with data re-uploading:

$$|\psi(x)\rangle = \prod_{l=1}^L R(\theta_l) R_X(x) |0\rangle$$

can approximate any function $f: [0, 2\pi] \rightarrow [-1, 1]$ to arbitrary precision as $L \rightarrow \infty$.

**Proof Sketch:** The output $\langle Z \rangle$ is a Fourier series in $x$, and adding layers adds higher-frequency terms.

#### Multi-Qubit Extension

$$|\psi(\mathbf{x})\rangle = \prod_{l=1}^L \left[W_l(\boldsymbol{\theta}_l) \cdot \bigotimes_{k=1}^n R(x_k)\right] |0\rangle^{\otimes n}$$

This creates a multi-dimensional Fourier series:

$$\langle O \rangle = \sum_{\omega \in \Omega} c_\omega(\boldsymbol{\theta}) e^{i \omega \cdot \mathbf{x}}$$

Where $\Omega$ is the frequency spectrum determined by the circuit structure.

### 4. Expressibility of QNNs

#### Definition

**Expressibility** measures how well a parameterized circuit can cover the space of possible unitaries.

**Formal Definition:** For a circuit ansatz $U(\boldsymbol{\theta})$, expressibility is measured by comparing the distribution of states to the Haar (uniform) distribution:

$$\mathcal{E} = D_{KL}\left(P_{U(\boldsymbol{\theta})} \| P_{\text{Haar}}\right)$$

Lower expressibility value = more expressive circuit.

#### Factors Affecting Expressibility

1. **Circuit Depth:** More layers → more expressibility
2. **Gate Set:** Non-Clifford gates (T gates) increase expressibility
3. **Entanglement:** Entangling gates are crucial
4. **Parameter Count:** More parameters → higher expressibility

#### Expressibility vs. Trainability Trade-off

**Observation:** Highly expressive circuits often suffer from **barren plateaus** (vanishing gradients).

$$\text{High Expressibility} \leftrightarrow \text{Difficult Training}$$

### 5. QNN Output Functions

#### Fourier Series Representation

**Theorem (Schuld et al., 2021):** The output of a QNN with angle encoding has the form:

$$f(\mathbf{x}) = \sum_{\omega \in \Omega} c_\omega e^{i\omega \cdot \mathbf{x}}$$

Where the frequency spectrum $\Omega$ is determined by the circuit structure.

**Key Result:** The expressibility of a QNN is fundamentally limited by its frequency spectrum.

#### Frequency Spectrum Constraints

For a circuit with $L$ layers of $R_X(x)$ gates:

$$\Omega = \{-L, -L+1, \ldots, L-1, L\}$$

**Implication:** To approximate functions with high-frequency components, need deep circuits or multiple data copies.

### 6. QNN Types and Variants

#### 6.1 Quantum Perceptron

Single-qubit QNN:
$$|\psi(x, \boldsymbol{\theta})\rangle = R_Y(\theta_2) R_Z(x) R_Y(\theta_1) |0\rangle$$

Output: $\langle Z \rangle = \sin\theta_1 \sin\theta_2 \cos x + \cos\theta_1 \cos\theta_2$

#### 6.2 Quantum Convolutional Neural Network (QCNN)

Inspired by classical CNNs:
1. **Convolutional layers:** Parameterized gates on neighboring qubits
2. **Pooling layers:** Measure some qubits, condition on results
3. **Fully connected:** Final variational layer before measurement

```
Layer 1 (Conv): [U₁][U₂][U₃][U₄]
                 ╱  ╲╱  ╲╱  ╲╱  ╲
Layer 2 (Pool):  ●───●───●───●
                  ╲ ╱   ╲ ╱
Layer 3 (FC):     [W]   [W]
                    ╲   ╱
Output:              M
```

#### 6.3 Quantum Graph Neural Network

For graph-structured data:
$$U(\mathbf{x}, G) = \prod_{(i,j) \in E(G)} U_{ij}(x_i, x_j)$$

Entangling gates respect graph structure.

### 7. Connection to Classical ML

#### Universal Approximation Comparison

| Model | Universality Condition |
|-------|----------------------|
| **Classical NN** | Single hidden layer with enough neurons |
| **QNN (single qubit)** | Sufficient depth with data re-uploading |
| **QNN (multi-qubit)** | Depth + entanglement structure |

#### Capacity Scaling

**Classical NN:** Parameters scale as $O(d \times h)$ where $d$ = input dim, $h$ = hidden dim.

**QNN:** Parameters scale as $O(n \times L)$ where $n$ = qubits, $L$ = layers, but effective dimension is $2^n$.

**Potential Advantage:** QNNs may require fewer parameters to achieve similar expressibility (not yet proven in general).

---

## Afternoon Session: Problem Solving (2 hours)

### Worked Example 1: Single-Qubit QNN Output

**Problem:** For a 2-layer single-qubit QNN:
$$|\psi(x)\rangle = R_Y(\theta_2) R_X(x) R_Y(\theta_1) R_X(x) |0\rangle$$

Express $\langle Z \rangle$ as a function of $x$, $\theta_1$, $\theta_2$.

**Solution:**

**Step 1: Apply gates sequentially**

Start: $|0\rangle$

After $R_X(x)$: $\cos(x/2)|0\rangle - i\sin(x/2)|1\rangle$

After $R_Y(\theta_1)$:
$$\begin{pmatrix} \cos(\theta_1/2) & -\sin(\theta_1/2) \\ \sin(\theta_1/2) & \cos(\theta_1/2) \end{pmatrix} \begin{pmatrix} \cos(x/2) \\ -i\sin(x/2) \end{pmatrix}$$

This gets complex. Let's use a different approach.

**Step 2: Use Fourier representation**

For angle encoding with $R_X(x)$, the output has the form:

$$\langle Z \rangle = a_0 + a_1 \cos(x) + a_2 \cos(2x) + b_1 \sin(x) + b_2 \sin(2x)$$

With 2 layers of $R_X(x)$, maximum frequency is 2.

**Step 3: Coefficients depend on $\theta_1, \theta_2$**

The general form with 2 data uploads:
$$\langle Z \rangle = c_0(\theta) + c_1(\theta)\cos(x) + c_2(\theta)\cos(2x) + s_1(\theta)\sin(x) + s_2(\theta)\sin(2x)$$

The coefficients $c_k(\theta), s_k(\theta)$ are sinusoidal functions of $\theta_1, \theta_2$.

**Key Insight:** The QNN can represent any function in the span of $\{1, \cos x, \cos 2x, \sin x, \sin 2x\}$, which are the first 5 Fourier basis functions.

---

### Worked Example 2: Counting Trainable Parameters

**Problem:** Design a QNN for a 4-dimensional input with:
- 4 qubits
- Angle encoding (one feature per qubit)
- 3 variational layers with $R_Y$, $R_Z$ rotations and circular CNOT entanglement
- Measurement of $Z_1$

How many trainable parameters? What is the frequency spectrum?

**Solution:**

**Step 1: Count parameters**

Each variational layer:
- $R_Y$ on each qubit: 4 parameters
- $R_Z$ on each qubit: 4 parameters
- CNOTs: 0 parameters

Parameters per layer: 8
Total: $3 \times 8 = 24$ parameters

**Step 2: Analyze frequency spectrum**

With single-layer angle encoding and 3 variational layers, there's no data re-uploading. The frequency spectrum is limited to:
$$\Omega = \{-1, 0, +1\}^4$$

Maximum frequency in any direction is 1.

**If we add data re-uploading** (encode before each variational layer):
$$\Omega = \{-3, -2, \ldots, +3\}^4$$

Maximum frequency in any direction is 3.

---

### Worked Example 3: QNN vs. Classical NN Parameter Efficiency

**Problem:** Compare a classical NN and QNN for approximating a 2D function.

Classical: 2 inputs → 8 hidden neurons → 1 output (with ReLU activation)
Quantum: 2 qubits, 4 layers of data re-uploading with $R_Y$ ansatz

**Solution:**

**Classical NN Parameters:**
- Input to hidden: $2 \times 8 + 8$ (weights + biases) = 24
- Hidden to output: $8 \times 1 + 1$ = 9
- Total: 33 parameters

**QNN Parameters:**
- 4 layers × 2 qubits × 3 rotations ($R_X, R_Y, R_Z$) = 24 parameters
- (Plus encoding uses data directly)
- Total: 24 parameters

**Expressibility Comparison:**

QNN with 4 layers can represent:
$$f(x_1, x_2) = \sum_{|\omega_1| \leq 4, |\omega_2| \leq 4} c_{\omega} e^{i(\omega_1 x_1 + \omega_2 x_2)}$$

This is $9 \times 9 = 81$ Fourier basis functions.

Classical NN with ReLU can approximate any continuous function on a compact domain, but the actual function class depends on weight magnitudes.

**Conclusion:** The QNN with fewer parameters has a well-defined frequency-limited expressibility, while the classical NN's expressibility depends on weight norms.

---

### Practice Problems

#### Problem 1: Layer Design (Direct Application)

You need a QNN for 3-class classification with 4 input features.

a) How many qubits are needed for output encoding?
b) Design an encoding layer for the 4 features
c) How many measurement outcomes map to each class?

<details>
<summary>Solution</summary>

a) Need at least $\lceil\log_2 3\rceil = 2$ qubits for output, but using all 4 qubits and measuring 2 works too.

b) Encoding layer options:
- Angle encoding: $R_Y(x_1) \otimes R_Y(x_2) \otimes R_Y(x_3) \otimes R_Y(x_4)$
- Or with entanglement: Add ZZ gates after rotations

c) With 2 measurement qubits:
- Class 0: $|00\rangle$
- Class 1: $|01\rangle$
- Class 2: $|10\rangle$
- (Discard or reassign $|11\rangle$)

Prediction: $\text{argmax}_k P(|k\rangle)$
</details>

#### Problem 2: Frequency Analysis (Intermediate)

A QNN encodes data $x$ using $R_Z(x)$ gates on 2 qubits with 2 layers of data re-uploading.

a) What is the maximum frequency in the output $\langle Z_1 \rangle$?
b) How many independent Fourier coefficients can be tuned?
c) Would using $R_X(x)$ instead change the answer?

<details>
<summary>Solution</summary>

a) With 2 layers of $R_Z(x)$ on 2 qubits = 4 total encoding gates. Max frequency = 4 (sum of contributions).

But actually, $R_Z$ gates on different qubits don't combine to increase frequency in the same direction. The correct analysis:
- Each qubit has 2 $R_Z(x)$ gates
- Max frequency per qubit: 2
- Total observable $\langle Z_1 \rangle$ depends only on qubit 1
- Max frequency: 2

b) Fourier basis for $\langle Z_1 \rangle$: $\{1, e^{\pm ix}, e^{\pm 2ix}\}$ = 5 independent real coefficients (as cosines and sines).

c) $R_X(x)$ behaves similarly in terms of frequency generation. The key is the number of data-encoding gates per qubit, not the axis.
</details>

#### Problem 3: QCNN Design (Challenging)

Design a QCNN for 8-qubit input with:
- 2 convolutional layers
- 1 pooling layer (reduce 8 → 4 qubits)
- 1 fully connected layer
- Final measurement

a) Draw the circuit schematic
b) Count total parameters (use $R_Y$ + entangling gates)
c) How does the effective receptive field grow with depth?

<details>
<summary>Solution</summary>

a) Schematic:
```
Conv 1:  [U][U][U][U][U][U][U][U]  (8 qubits)
          ╲╱ ╲╱ ╲╱ ╲╱ ╲╱ ╲╱ ╲╱
Conv 2:  [U][U][U][U][U][U][U][U]
          ╲    ╱╲    ╱╲    ╱╲    ╱
Pool:     M----M----M----M         (Measure 4, keep 4)
           ╲  ╱ ╲  ╱
FC:        [W][W][W][W]           (4 qubits)
             ╲  ╱
Measure:      Z

```

b) Parameters:
- Conv layers: 8 qubits × 2 rotations × 2 layers = 32
- Entangling gates between adjacent: 7 × 2 = 14 (if parameterized)
- Pooling: Controlled operations, ~4 parameters
- FC layer: 4 × 3 = 12 (if using R_X, R_Y, R_Z)
- Total: ~62 parameters

c) Receptive field:
- After Conv 1: 2 qubits (local)
- After Conv 2: 4 qubits (overlapping)
- After pooling: 8 qubits (information compressed)

The receptive field grows logarithmically with layers.
</details>

---

## Evening Session: Computational Lab (2 hours)

### Lab: Building Deep Quantum Neural Networks

```python
"""
Day 970 Lab: Quantum Neural Networks
Building and training deep QNN architectures
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

print("=" * 60)
print("Quantum Neural Networks Implementation")
print("=" * 60)

#######################################
# Part 1: Basic QNN Architecture
#######################################

print("\n" + "-" * 40)
print("Part 1: Basic QNN Architecture")
print("-" * 40)

n_qubits = 4
n_layers = 3
dev = qml.device('default.qubit', wires=n_qubits)

def encoding_layer(x, wires):
    """
    Angle encoding layer: embed data features into rotation angles
    """
    for i, w in enumerate(wires):
        if i < len(x):
            qml.RX(x[i], wires=w)

def variational_layer(params, wires):
    """
    Variational layer with rotations and entanglement
    params shape: (n_qubits, 3) for RX, RY, RZ
    """
    for i, w in enumerate(wires):
        qml.RX(params[i, 0], wires=w)
        qml.RY(params[i, 1], wires=w)
        qml.RZ(params[i, 2], wires=w)

    # Entangling layer (circular connectivity)
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])

@qml.qnode(dev)
def basic_qnn(x, params):
    """
    Basic QNN: single encoding + multiple variational layers
    """
    wires = range(n_qubits)

    # Single encoding at the start
    encoding_layer(x, wires)

    # Multiple variational layers
    for l in range(n_layers):
        variational_layer(params[l], wires)

    return qml.expval(qml.PauliZ(0))

# Initialize and test
params_shape = (n_layers, n_qubits, 3)
params = np.random.uniform(-np.pi, np.pi, params_shape)

x_test = np.array([0.5, 1.0, 1.5, 2.0])
output = basic_qnn(x_test, params)

print(f"Input: {x_test}")
print(f"Output: {output:.4f}")
print(f"Total parameters: {np.prod(params_shape)}")

# Draw circuit
print("\nCircuit structure:")
print(qml.draw(basic_qnn)(x_test, params))


#######################################
# Part 2: Data Re-uploading QNN
#######################################

print("\n" + "-" * 40)
print("Part 2: Data Re-uploading QNN")
print("-" * 40)

@qml.qnode(dev)
def reupload_qnn(x, params):
    """
    Data re-uploading QNN: interleave encoding and variational layers

    This architecture has better expressibility for function approximation.
    """
    wires = range(n_qubits)

    for l in range(n_layers):
        # Encoding layer (repeated at each level)
        encoding_layer(x, wires)
        # Variational layer
        variational_layer(params[l], wires)

    return qml.expval(qml.PauliZ(0))

output_reupload = reupload_qnn(x_test, params)
print(f"Re-uploading QNN output: {output_reupload:.4f}")

print("\nCircuit structure (data re-uploading):")
print(qml.draw(reupload_qnn)(x_test, params))


#######################################
# Part 3: Frequency Spectrum Analysis
#######################################

print("\n" + "-" * 40)
print("Part 3: Frequency Spectrum Analysis")
print("-" * 40)

# Single-qubit QNN for frequency analysis
dev_1q = qml.device('default.qubit', wires=1)

def single_qubit_qnn(x, params, n_layers):
    """Single qubit data re-uploading circuit"""
    for l in range(n_layers):
        qml.RX(x, wires=0)
        qml.RY(params[l, 0], wires=0)
        qml.RZ(params[l, 1], wires=0)
    return qml.expval(qml.PauliZ(0))

# Analyze output as function of x for different depths
x_range = np.linspace(0, 2 * np.pi, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx, n_l in enumerate([1, 2, 3, 4, 5, 6]):
    params_1q = np.random.uniform(-np.pi, np.pi, (n_l, 2))

    @qml.qnode(dev_1q)
    def circuit(x):
        return single_qubit_qnn(x, params_1q, n_l)

    outputs = [circuit(x) for x in x_range]

    ax = axes[idx // 3, idx % 3]
    ax.plot(x_range, outputs, 'b-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('⟨Z⟩')
    ax.set_title(f'{n_l} Layer(s) - Max Freq: {n_l}')
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

plt.suptitle('QNN Output Functions vs. Depth (Random Parameters)', fontsize=14)
plt.tight_layout()
plt.savefig('qnn_frequency_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Fourier analysis of a specific circuit
print("\nFourier Analysis of 4-layer QNN:")
params_4l = np.random.uniform(-np.pi, np.pi, (4, 2))

@qml.qnode(dev_1q)
def circuit_4l(x):
    return single_qubit_qnn(x, params_4l, 4)

outputs_4l = np.array([circuit_4l(x) for x in x_range])

# Compute FFT
fft_result = np.fft.fft(outputs_4l)
freqs = np.fft.fftfreq(len(x_range), d=(x_range[1] - x_range[0]) / (2 * np.pi))

# Plot Fourier components
plt.figure(figsize=(10, 4))
plt.stem(freqs[:len(freqs)//2], np.abs(fft_result[:len(freqs)//2]) / len(x_range),
         basefmt=" ", linefmt='b-', markerfmt='bo')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Fourier Spectrum of 4-Layer QNN Output')
plt.xlim(-0.5, 10)
plt.grid(True, alpha=0.3)
plt.savefig('qnn_fourier_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 4: Training QNN for Regression
#######################################

print("\n" + "-" * 40)
print("Part 4: Training QNN for Regression")
print("-" * 40)

# Target function
def target_function(x):
    return np.sin(2 * x) * np.cos(x)

# Training data
n_train = 30
x_train = np.linspace(0, 2 * np.pi, n_train)
y_train = target_function(x_train)

# QNN for regression
n_layers_reg = 4

@qml.qnode(dev_1q)
def regression_qnn(x, params):
    for l in range(n_layers_reg):
        qml.RX(x, wires=0)
        qml.RY(params[l, 0], wires=0)
        qml.RZ(params[l, 1], wires=0)
    return qml.expval(qml.PauliZ(0))

def cost_regression(params):
    predictions = np.array([regression_qnn(x, params) for x in x_train])
    return np.mean((predictions - y_train) ** 2)

# Training
params_reg = np.random.uniform(-np.pi, np.pi, (n_layers_reg, 2))
optimizer = qml.AdamOptimizer(stepsize=0.1)

costs = []
n_epochs = 100

print("Training regression QNN...")
for epoch in range(n_epochs):
    params_reg, cost = optimizer.step_and_cost(cost_regression, params_reg)
    costs.append(cost)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}: Cost = {cost:.4f}")

# Plot results
x_plot = np.linspace(0, 2 * np.pi, 100)
y_pred = [regression_qnn(x, params_reg) for x in x_plot]
y_true = target_function(x_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(costs, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_plot, y_true, 'b-', linewidth=2, label='Target')
axes[1].plot(x_plot, y_pred, 'r--', linewidth=2, label='QNN')
axes[1].scatter(x_train, y_train, c='green', s=30, zorder=5, label='Training Data')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('QNN Regression Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qnn_regression.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 5: Multi-Qubit QNN for Classification
#######################################

print("\n" + "-" * 40)
print("Part 5: Multi-Qubit QNN Classification")
print("-" * 40)

# Generate dataset
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
y = 2 * y - 1  # Convert to {-1, +1}

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Define QNN
n_qubits_class = 2
n_layers_class = 4
dev_class = qml.device('default.qubit', wires=n_qubits_class)

@qml.qnode(dev_class)
def classification_qnn(x, params):
    """QNN for 2D classification with data re-uploading"""
    for l in range(n_layers_class):
        # Data encoding
        qml.RX(x[0], wires=0)
        qml.RX(x[1], wires=1)

        # Variational layer
        qml.RY(params[l, 0], wires=0)
        qml.RY(params[l, 1], wires=1)
        qml.RZ(params[l, 2], wires=0)
        qml.RZ(params[l, 3], wires=1)

        # Entanglement
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[l, 4], wires=1)
        qml.CNOT(wires=[0, 1])

    return qml.expval(qml.PauliZ(0))

def cost_classification(params):
    predictions = np.array([classification_qnn(x, params) for x in X_train])
    return np.mean((predictions - y_train) ** 2)

def accuracy(params, X, y):
    predictions = np.array([np.sign(classification_qnn(x, params)) for x in X])
    return np.mean(predictions == y)

# Training
params_class = np.random.uniform(-np.pi, np.pi, (n_layers_class, 5))
optimizer = qml.AdamOptimizer(stepsize=0.1)

print("Training classification QNN...")
train_costs = []
train_accs = []
test_accs = []

for epoch in range(60):
    params_class, cost = optimizer.step_and_cost(cost_classification, params_class)

    if (epoch + 1) % 10 == 0:
        train_acc = accuracy(params_class, X_train, y_train)
        test_acc = accuracy(params_class, X_test, y_test)

        train_costs.append(cost)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}: Cost = {cost:.4f}, "
              f"Train Acc = {train_acc:.2%}, Test Acc = {test_acc:.2%}")


#######################################
# Part 6: Decision Boundary Visualization
#######################################

print("\n" + "-" * 40)
print("Part 6: Decision Boundary Visualization")
print("-" * 40)

def plot_decision_boundary(params, X, y, title):
    """Plot decision boundary for 2D classification"""
    h = 0.05
    x_min, x_max = 0, np.pi
    y_min, y_max = 0, np.pi

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([classification_qnn(p, params) for p in grid_points])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    plt.colorbar(label='⟨Z⟩')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o',
                edgecolors='black', s=60, label='Class +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='x',
                s=60, label='Class -1')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()

plot_decision_boundary(params_class, X_scaled, y,
                        f'QNN Decision Boundary (Test Acc: {test_accs[-1]:.0%})')
plt.savefig('qnn_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 7: Expressibility Comparison
#######################################

print("\n" + "-" * 40)
print("Part 7: Expressibility Comparison")
print("-" * 40)

def measure_expressibility(circuit_func, params_shape, n_samples=1000):
    """
    Measure expressibility by sampling random outputs

    Higher variance = more expressive (covers more of output space)
    """
    outputs = []
    for _ in range(n_samples):
        params = np.random.uniform(-np.pi, np.pi, params_shape)
        x = np.random.uniform(0, np.pi, 2)
        output = circuit_func(x, params)
        outputs.append(output)

    outputs = np.array(outputs)
    return np.std(outputs), np.min(outputs), np.max(outputs)

# Compare different architectures
architectures = []

# Shallow circuit (1 layer)
@qml.qnode(dev_class)
def shallow_qnn(x, params):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RY(params[0, 0], wires=0)
    qml.RY(params[0, 1], wires=1)
    return qml.expval(qml.PauliZ(0))

# Medium circuit (3 layers, no entanglement)
@qml.qnode(dev_class)
def medium_no_ent(x, params):
    for l in range(3):
        qml.RX(x[0], wires=0)
        qml.RX(x[1], wires=1)
        qml.RY(params[l, 0], wires=0)
        qml.RY(params[l, 1], wires=1)
    return qml.expval(qml.PauliZ(0))

# Deep circuit (6 layers with entanglement)
@qml.qnode(dev_class)
def deep_entangled(x, params):
    for l in range(6):
        qml.RX(x[0], wires=0)
        qml.RX(x[1], wires=1)
        qml.RY(params[l, 0], wires=0)
        qml.RY(params[l, 1], wires=1)
        qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

print("Measuring expressibility (may take a moment)...")
print("-" * 50)

circuits = [
    ("Shallow (1 layer)", shallow_qnn, (1, 2)),
    ("Medium (3L, no ent)", medium_no_ent, (3, 2)),
    ("Deep (6L, entangled)", deep_entangled, (6, 2)),
]

for name, circuit, shape in circuits:
    std, out_min, out_max = measure_expressibility(circuit, shape, n_samples=500)
    print(f"{name:20s} | Std: {std:.3f} | Range: [{out_min:.2f}, {out_max:.2f}]")

print("-" * 50)
print("Higher std = more expressive (covers output space better)")


#######################################
# Part 8: Comparing QNN Depths
#######################################

print("\n" + "-" * 40)
print("Part 8: Effect of QNN Depth")
print("-" * 40)

def train_qnn_with_depth(depth, n_epochs=40):
    """Train QNN with specified depth and return final accuracy"""

    @qml.qnode(dev_class)
    def qnn_depth(x, params):
        for l in range(depth):
            qml.RX(x[0], wires=0)
            qml.RX(x[1], wires=1)
            qml.RY(params[l, 0], wires=0)
            qml.RY(params[l, 1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(params[l, 2], wires=1)
            qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def cost(params):
        preds = np.array([qnn_depth(x, params) for x in X_train])
        return np.mean((preds - y_train) ** 2)

    params = np.random.uniform(-np.pi, np.pi, (depth, 3))
    opt = qml.AdamOptimizer(stepsize=0.1)

    for _ in range(n_epochs):
        params = opt.step(cost, params)

    # Evaluate
    preds_test = np.array([np.sign(qnn_depth(x, params)) for x in X_test])
    return np.mean(preds_test == y_test), depth * 3

depths = [1, 2, 3, 4, 5, 6]
results = []

print("Training QNNs with different depths...")
for d in depths:
    acc, n_params = train_qnn_with_depth(d)
    results.append((d, acc, n_params))
    print(f"Depth {d}: Accuracy = {acc:.2%}, Parameters = {n_params}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

depths_plot = [r[0] for r in results]
accs_plot = [r[1] for r in results]
params_plot = [r[2] for r in results]

axes[0].bar(depths_plot, accs_plot, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Depth')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Accuracy vs. Depth')
axes[0].set_ylim(0.5, 1.0)

axes[1].bar(depths_plot, params_plot, color='coral', edgecolor='black')
axes[1].set_xlabel('Depth')
axes[1].set_ylabel('Number of Parameters')
axes[1].set_title('Parameters vs. Depth')

plt.tight_layout()
plt.savefig('qnn_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

### Expected Output

```
============================================================
Quantum Neural Networks Implementation
============================================================

Training regression QNN...
Epoch 20: Cost = 0.1234
Epoch 40: Cost = 0.0456
Epoch 60: Cost = 0.0123
Epoch 80: Cost = 0.0067
Epoch 100: Cost = 0.0034

Training classification QNN...
Epoch 10: Cost = 0.7821, Train Acc = 65.71%, Test Acc = 60.00%
Epoch 60: Cost = 0.1234, Train Acc = 94.29%, Test Acc = 90.00%

Measuring expressibility...
Shallow (1 layer)    | Std: 0.234 | Range: [-0.45, 0.52]
Medium (3L, no ent)  | Std: 0.456 | Range: [-0.89, 0.91]
Deep (6L, entangled) | Std: 0.512 | Range: [-0.98, 0.99]
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| **QNN State** | $\|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = \prod_l W_l(\boldsymbol{\theta}_l) S_l(\mathbf{x}) \|0\rangle$ |
| **Output** | $f(\mathbf{x}) = \langle\psi\|\hat{O}\|\psi\rangle$ |
| **Fourier Form** | $f(\mathbf{x}) = \sum_\omega c_\omega(\boldsymbol{\theta}) e^{i\omega \cdot \mathbf{x}}$ |
| **Max Frequency** | $\omega_{\max} = L$ for $L$ encoding layers |
| **Parameters** | $\|\boldsymbol{\theta}\| = L \times n \times g$ for $g$ params/qubit/layer |

### Key Takeaways

1. **QNNs** combine encoding layers (data) with variational layers (trainable)
2. **Data re-uploading** enables universal approximation with a single qubit
3. **Frequency spectrum** of QNN outputs is determined by circuit structure
4. **Deeper circuits** are more expressive but potentially harder to train
5. **Entanglement** is crucial for multi-qubit expressibility
6. **QNN expressibility** trades off against trainability

### Connection to Classical ML

| QNN Component | Classical Analog |
|---------------|-----------------|
| Encoding layer | Input layer |
| Variational layer | Hidden layer |
| Data re-uploading | Skip connections |
| Measurement | Output activation |
| Entanglement | Inter-layer connections |

---

## Daily Checklist

- [ ] I can design QNN architectures with appropriate layers
- [ ] I understand the role of data re-uploading
- [ ] I can analyze the frequency spectrum of QNN outputs
- [ ] I implemented QNNs for regression and classification
- [ ] I understand how depth affects expressibility
- [ ] I can compare QNN architectures for a given task
- [ ] I understand the trade-off between expressibility and trainability

---

## Preview: Day 971

Tomorrow we dive deep into **Data Encoding Strategies**, exploring:

- Amplitude encoding vs. angle encoding trade-offs
- Basis encoding for discrete data
- Hardware-efficient encoding schemes
- Encoding depth vs. gate count optimization
- Impact of encoding on QML performance

---

*"The quantum neural network is not just a neural network run on a quantum computer - it is a fundamentally different computational primitive with its own expressibility and limitations."*
— Adrián Pérez-Salinas
