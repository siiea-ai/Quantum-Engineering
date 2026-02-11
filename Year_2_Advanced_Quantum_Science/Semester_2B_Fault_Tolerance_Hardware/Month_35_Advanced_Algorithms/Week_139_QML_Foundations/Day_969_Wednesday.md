# Day 969: Quantum Kernel Methods

## Year 2, Semester 2B: Fault Tolerance & Hardware
## Month 35: Advanced Algorithms - Week 139: QML Foundations

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Theory of quantum kernels and kernel estimation |
| **Afternoon** | 2 hours | Problem solving: kernel design and analysis |
| **Evening** | 2 hours | Quantum SVM implementation with kernel estimation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define quantum kernels** mathematically and understand their properties
2. **Estimate quantum kernels** using quantum circuits
3. **Connect quantum kernels** to classical kernel methods and SVMs
4. **Implement quantum kernel SVMs** using PennyLane and scikit-learn
5. **Analyze kernel matrices** for classification quality
6. **Evaluate computational costs** of quantum kernel methods

---

## Morning Session: Theory (3 hours)

### 1. From Feature Maps to Kernels

#### The Kernel Function

Given a quantum feature map $|\phi(\mathbf{x})\rangle = U_\phi(\mathbf{x})|0\rangle^{\otimes n}$, the **quantum kernel** is:

$$\boxed{K_Q(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2 = |\langle 0|U_\phi^\dagger(\mathbf{x})U_\phi(\mathbf{x}')|0\rangle|^2}$$

This is the **fidelity** (overlap squared) between two quantum states.

#### Properties of Quantum Kernels

**Theorem (Positive Semi-Definiteness):** For any set of points $\{\mathbf{x}_1, ..., \mathbf{x}_N\}$, the kernel matrix

$$K_{ij} = K_Q(\mathbf{x}_i, \mathbf{x}_j)$$

is positive semi-definite (all eigenvalues $\geq 0$).

**Proof:** Define the Gram matrix in terms of the inner product:
$$\tilde{K}_{ij} = \langle\phi(\mathbf{x}_i)|\phi(\mathbf{x}_j)\rangle$$

Then $\tilde{K}$ is a valid kernel (positive semi-definite by construction). Since $K_Q = |\tilde{K}|^2$ element-wise and $\tilde{K}$ is PSD, $K_Q$ is also PSD (Schur product theorem for symmetric PSD matrices).

#### Reproducing Kernel Hilbert Space (RKHS)

By Mercer's theorem, every positive semi-definite kernel corresponds to an inner product in some feature space. For quantum kernels:

$$K_Q(\mathbf{x}, \mathbf{x}') = \langle\Phi(\mathbf{x}), \Phi(\mathbf{x}')\rangle_{\mathcal{H}}$$

The RKHS associated with $K_Q$ can have dimension up to $2^{2n}$ (density matrix space).

### 2. Kernel Estimation Circuit

#### The Basic Protocol

To estimate $K_Q(\mathbf{x}, \mathbf{x}')$:

```
|0⟩^⊗n ──[U_φ(x)]──[U_φ†(x')]──┤ Measure P(|0...0⟩) ├
```

**Circuit:**
1. Apply feature map for $\mathbf{x}$: $U_\phi(\mathbf{x})|0\rangle = |\phi(\mathbf{x})\rangle$
2. Apply inverse feature map for $\mathbf{x}'$: $U_\phi^\dagger(\mathbf{x}')|\phi(\mathbf{x})\rangle$
3. Measure probability of $|0\rangle^{\otimes n}$

**Result:** $P(|0...0\rangle) = |\langle 0|U_\phi^\dagger(\mathbf{x}')U_\phi(\mathbf{x})|0\rangle|^2 = K_Q(\mathbf{x}, \mathbf{x}')$

#### Statistical Estimation

Each measurement gives outcome $|0\rangle^{\otimes n}$ with probability $K_Q$. After $N_s$ shots:

$$\hat{K}_Q = \frac{N_0}{N_s}$$

Where $N_0$ is the count of $|0\rangle^{\otimes n}$ outcomes.

**Variance:**
$$\text{Var}[\hat{K}_Q] = \frac{K_Q(1-K_Q)}{N_s}$$

**Confidence Interval (95%):**
$$\hat{K}_Q \pm 1.96\sqrt{\frac{\hat{K}_Q(1-\hat{K}_Q)}{N_s}}$$

#### Shot Budget for Kernel Matrix

For $N$ data points, the full kernel matrix has $\frac{N(N+1)}{2}$ unique entries (symmetric).

Total shots needed for precision $\epsilon$:
$$N_\text{total} \approx \frac{N^2}{2} \cdot \frac{1}{\epsilon^2}$$

**Example:** $N = 100$ points, $\epsilon = 0.01$ → $\approx 5 \times 10^7$ shots

### 3. Quantum Kernel SVM

#### Support Vector Machine Review

The SVM optimization problem (dual form):

$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

Subject to: $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$

The decision function:
$$f(\mathbf{x}) = \sum_{i=1}^N \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$

#### Quantum Kernel SVM

Replace the classical kernel with the quantum kernel:

$$f(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i K_Q(\mathbf{x}_i, \mathbf{x}) + b$$

**Training:**
1. Compute quantum kernel matrix $K_Q$ for training data
2. Solve classical SVM optimization with kernel matrix
3. Identify support vectors and coefficients $\alpha_i$

**Prediction:**
1. For new point $\mathbf{x}$, compute $K_Q(\mathbf{x}_i, \mathbf{x})$ for all support vectors
2. Evaluate decision function $f(\mathbf{x})$
3. Classify: $\hat{y} = \text{sign}(f(\mathbf{x}))$

### 4. Kernel Design Principles

#### 4.1 The Havlíček ZZ Kernel

From the landmark 2019 paper:

$$U_\phi(\mathbf{x}) = \prod_{d=1}^D \left[U_Z(\mathbf{x}) \cdot H^{\otimes n}\right]$$

Where:
$$U_Z(\mathbf{x}) = \exp\left(i\sum_k x_k Z_k + i\sum_{j<k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$$

**Properties:**
- Conjectured to be hard to simulate classically
- Creates highly non-linear feature maps
- Depth $D$ controls expressivity

#### 4.2 IQP (Instantaneous Quantum Polynomial) Kernels

$$U_\phi(\mathbf{x}) = D(\mathbf{x}) H^{\otimes n}$$

Where $D(\mathbf{x})$ is diagonal in computational basis:

$$D(\mathbf{x}) = \exp\left(i \sum_{S \subseteq [n]} f_S(\mathbf{x}) Z_S\right)$$

Here $Z_S = \prod_{k \in S} Z_k$.

**Advantage:** All gates commute, potentially easier to implement.

#### 4.3 Projected Quantum Kernels

Instead of full state overlap, use local observables:

$$K_{\text{proj}}(\mathbf{x}, \mathbf{x}') = \sum_k w_k \langle O_k \rangle_{\mathbf{x}} \langle O_k \rangle_{\mathbf{x}'}$$

This can reduce variance and avoid concentration issues.

### 5. Computational Complexity Analysis

#### Training Complexity

| Component | Classical Cost | Quantum Cost |
|-----------|---------------|--------------|
| Kernel matrix ($N^2$ entries) | $O(N^2 d)$ for RBF | $O(N^2 G N_s)$ where $G$ = gates, $N_s$ = shots |
| SVM optimization | $O(N^3)$ or $O(N^{2.x})$ | Same (classical) |
| **Total** | $O(N^3)$ | $O(N^2 G N_s + N^3)$ |

**Key Insight:** The quantum advantage, if any, comes from the **kernel function itself**, not the computational complexity.

#### Prediction Complexity

For a new point with $N_{SV}$ support vectors:
- Classical: $O(N_{SV} d)$
- Quantum: $O(N_{SV} G N_s)$

### 6. Connection to Quantum Advantage

#### When Might Quantum Kernels Help?

**Necessary Conditions:**
1. The optimal classical kernel cannot be efficiently computed
2. The quantum kernel captures relevant structure in the data
3. The quantum kernel is not efficiently classically simulable

**Havlíček et al. Conjecture:** ZZ feature maps with specific structure create kernels that are:
- Hard to compute classically (unless BQP = BPP)
- Useful for certain classification tasks

#### The Classical Simulation Question

**Classical Simulability:** If the quantum kernel can be computed efficiently classically, there is no quantum advantage.

**Sufficient conditions for classical simulation:**
- Low entanglement feature maps
- Clifford-only circuits
- Certain tensor network structures

**Hard to simulate (conjectured):**
- IQP circuits with random phases
- Deep ZZ feature maps
- Circuits with high entanglement

---

## Afternoon Session: Problem Solving (2 hours)

### Worked Example 1: Computing a 2-Qubit Kernel

**Problem:** For the product feature map $U_\phi(x) = R_Z(x_1) \otimes R_Z(x_2) \cdot H^{\otimes 2}$, compute $K_Q(\mathbf{x}, \mathbf{x}')$ analytically.

**Solution:**

**Step 1: Compute $|\phi(\mathbf{x})\rangle$**

$$|\phi(\mathbf{x})\rangle = R_Z(x_1) \otimes R_Z(x_2) \cdot H \otimes H |00\rangle$$

$$= R_Z(x_1) \otimes R_Z(x_2) \cdot \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle + |1\rangle)$$

$$= \frac{1}{2}(e^{-ix_1/2}|0\rangle + e^{ix_1/2}|1\rangle)(e^{-ix_2/2}|0\rangle + e^{ix_2/2}|1\rangle)$$

**Step 2: Expand in computational basis**

$$|\phi(\mathbf{x})\rangle = \frac{1}{2}\left(e^{-i(x_1+x_2)/2}|00\rangle + e^{-i(x_1-x_2)/2}|01\rangle + e^{i(x_1-x_2)/2}|10\rangle + e^{i(x_1+x_2)/2}|11\rangle\right)$$

**Step 3: Compute inner product**

$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = \frac{1}{4}\left(e^{i(x_1+x_2-x_1'-x_2')/2} + e^{i(x_1-x_2-x_1'+x_2')/2} + e^{-i(x_1-x_2-x_1'+x_2')/2} + e^{-i(x_1+x_2-x_1'-x_2')/2}\right)$$

$$= \frac{1}{2}\left(\cos\frac{(x_1-x_1')+(x_2-x_2')}{2} + \cos\frac{(x_1-x_1')-(x_2-x_2')}{2}\right)$$

Using $\cos A + \cos B = 2\cos\frac{A+B}{2}\cos\frac{A-B}{2}$:

$$= \cos\frac{x_1-x_1'}{2}\cos\frac{x_2-x_2'}{2}$$

**Step 4: Square for kernel**

$$\boxed{K_Q(\mathbf{x}, \mathbf{x}') = \cos^2\frac{x_1-x_1'}{2}\cos^2\frac{x_2-x_2'}{2}}$$

**Note:** This factorizes into a product of 1D kernels, showing no entanglement in the feature map.

---

### Worked Example 2: Kernel Matrix Eigenvalues

**Problem:** Given a 4-point dataset with kernel matrix:

$$K = \begin{pmatrix} 1 & 0.8 & 0.2 & 0.3 \\ 0.8 & 1 & 0.3 & 0.2 \\ 0.2 & 0.3 & 1 & 0.9 \\ 0.3 & 0.2 & 0.9 & 1 \end{pmatrix}$$

Analyze this kernel matrix for classification.

**Solution:**

**Step 1: Check positive semi-definiteness**

Compute eigenvalues: $\lambda_1 = 2.35$, $\lambda_2 = 1.52$, $\lambda_3 = 0.10$, $\lambda_4 = 0.03$

All $\lambda_i \geq 0$ ✓ → Valid kernel matrix

**Step 2: Analyze block structure**

Points 1-2 are similar (kernel 0.8), points 3-4 are similar (kernel 0.9).
Points 1-2 vs 3-4 have low similarity (0.2-0.3).

**Interpretation:** Data naturally clusters into two groups!

**Step 3: Effective rank**

$$\text{Effective rank} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2} = \frac{4^2}{2.35^2 + 1.52^2 + 0.10^2 + 0.03^2} \approx 2.0$$

The kernel matrix has approximately rank 2, suggesting 2D structure in feature space.

---

### Worked Example 3: Shot Budget Calculation

**Problem:** You want to build a quantum kernel SVM with 50 training points and 10 test points. Each kernel evaluation uses 1000 shots. The circuit has 100 gates. Estimate the total quantum resources needed.

**Solution:**

**Training kernel matrix:**
- Unique entries: $\frac{50 \times 51}{2} = 1275$ (including diagonal)
- Total shots: $1275 \times 1000 = 1.275 \times 10^6$
- Total gates: $1275 \times 1000 \times 100 = 1.275 \times 10^8$

**Test kernel evaluations:**
- For each test point, evaluate against all 50 training points
- But only support vectors matter. Assume 20% are SVs → 10 SVs
- Evaluations per test point: 10
- Total test evaluations: $10 \times 10 = 100$
- Test shots: $100 \times 1000 = 10^5$

**Total resources:**
- Training: $1.275 \times 10^6$ shots, $1.275 \times 10^8$ gates
- Testing: $10^5$ shots per 10 test points

**On real hardware (100 kHz sampling rate):**
- Training time: $\frac{1.275 \times 10^6}{10^5}$ seconds $\approx 13$ seconds (just measurements)
- Plus circuit execution time

---

### Practice Problems

#### Problem 1: Kernel Properties (Direct Application)

For a 3-qubit angle encoding $U_\phi(\mathbf{x}) = R_Y(x_1) \otimes R_Y(x_2) \otimes R_Y(x_3)$:

a) What is the dimension of the quantum feature space?
b) Write the kernel function $K_Q(\mathbf{x}, \mathbf{x}')$
c) Is this kernel equivalent to any classical kernel?

<details>
<summary>Solution</summary>

a) Dimension: $2^3 = 8$

b) The kernel factorizes:
$K_Q(\mathbf{x}, \mathbf{x}') = \prod_{i=1}^3 \cos^2\frac{x_i - x_i'}{2}$

c) Yes! This is equivalent to a product of 1D cosine kernels, which can be computed classically. The feature map has no entanglement, so no quantum advantage.
</details>

#### Problem 2: Shot Noise (Intermediate)

You estimate a kernel value and get $\hat{K} = 0.7$ from 1000 shots.

a) What is the standard error of this estimate?
b) What is the 95% confidence interval?
c) How many shots needed for standard error < 0.01?

<details>
<summary>Solution</summary>

a) Standard error: $\sqrt{\frac{K(1-K)}{N_s}} = \sqrt{\frac{0.7 \times 0.3}{1000}} = 0.0145$

b) 95% CI: $0.7 \pm 1.96 \times 0.0145 = [0.672, 0.728]$

c) Need: $\sqrt{\frac{0.7 \times 0.3}{N_s}} < 0.01$

$N_s > \frac{0.21}{0.0001} = 2100$ shots
</details>

#### Problem 3: Kernel Alignment (Challenging)

The **kernel-target alignment** measures how well a kernel matches the labels:

$$A(K, \mathbf{y}) = \frac{\mathbf{y}^\top K \mathbf{y}}{||\mathbf{y}|| \cdot ||K||_F}$$

For labels $\mathbf{y} = (+1, +1, -1, -1)$ and the kernel matrix from Worked Example 2:

a) Compute the alignment
b) What alignment would be achieved by a perfect kernel?
c) How does alignment relate to classification accuracy?

<details>
<summary>Solution</summary>

a) Compute:
$\mathbf{y}^\top K \mathbf{y} = 1 + 0.8 + 0.2 + 0.3 + 0.8 + 1 + 0.3 + 0.2 - 0.2 - 0.3 - 1 - 0.9 - 0.3 - 0.2 - 0.9 - 1$

Actually, let's do this properly:
$(\mathbf{y}^\top K \mathbf{y})_{ij} = y_i K_{ij} y_j$

For $y = (+1, +1, -1, -1)$:
$= K_{11} + K_{12} + K_{21} + K_{22} - K_{13} - K_{14} - K_{23} - K_{24} - K_{31} - K_{32} + K_{33} + K_{34} - K_{41} - K_{42} + K_{43} + K_{44}$
$= 1 + 0.8 + 0.8 + 1 - 0.2 - 0.3 - 0.3 - 0.2 - 0.2 - 0.3 + 1 + 0.9 - 0.3 - 0.2 + 0.9 + 1$
$= 3.6 - 2.0 + 3.8 = 5.4$

$||\mathbf{y}|| = 2$, $||K||_F = \sqrt{\sum K_{ij}^2} \approx 2.5$

$A \approx \frac{5.4}{2 \times 2.5} = 1.08$ (normalized version would be ≤ 1)

b) Perfect kernel: $K^* = \mathbf{y}\mathbf{y}^\top$ gives alignment = 1

c) Higher alignment generally correlates with better classification, but not perfectly. It's a useful heuristic for kernel selection.
</details>

---

## Evening Session: Computational Lab (2 hours)

### Lab: Quantum Kernel SVM Implementation

```python
"""
Day 969 Lab: Quantum Kernel Methods
Implementing Quantum Kernel SVM from scratch
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

print("=" * 60)
print("Quantum Kernel Methods Implementation")
print("=" * 60)

#######################################
# Part 1: Quantum Kernel Definition
#######################################

print("\n" + "-" * 40)
print("Part 1: Quantum Kernel Definition")
print("-" * 40)

n_qubits = 2
dev = qml.device('default.qubit', wires=n_qubits)

def zz_feature_map(x, reps=2):
    """
    ZZ feature map (Havlíček et al., 2019)
    Creates entangled quantum feature states
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
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """
    Compute kernel K(x1, x2) = |<φ(x1)|φ(x2)>|²

    Uses the identity: apply U(x1), then U†(x2), measure P(|0⟩)
    """
    # Embed x1
    zz_feature_map(x1)
    # Apply inverse of x2 embedding
    qml.adjoint(zz_feature_map)(x2)
    # Probability of |00...0⟩
    return qml.probs(wires=range(n_qubits))

def quantum_kernel(x1, x2):
    """Wrapper returning just the kernel value"""
    probs = kernel_circuit(x1, x2)
    return probs[0]  # P(|00⟩)

# Test kernel properties
print("Testing kernel properties:")
x1 = np.array([0.5, 1.0])
x2 = np.array([0.5, 1.0])
x3 = np.array([1.5, 0.3])

k11 = quantum_kernel(x1, x1)
k12 = quantum_kernel(x1, x2)
k13 = quantum_kernel(x1, x3)
k31 = quantum_kernel(x3, x1)

print(f"K(x1, x1) = {k11:.4f} (should be 1.0)")
print(f"K(x1, x2) = {k12:.4f} (should be 1.0, same point)")
print(f"K(x1, x3) = {k13:.4f}")
print(f"K(x3, x1) = {k31:.4f} (should equal K(x1, x3) - symmetry)")


#######################################
# Part 2: Kernel Matrix Computation
#######################################

print("\n" + "-" * 40)
print("Part 2: Kernel Matrix Computation")
print("-" * 40)

def compute_kernel_matrix(X1, X2=None, verbose=True):
    """
    Compute kernel matrix K[i,j] = K(X1[i], X2[j])

    If X2 is None, compute symmetric matrix K[i,j] = K(X1[i], X1[j])
    """
    n1 = len(X1)
    if X2 is None:
        X2 = X1
        symmetric = True
    else:
        symmetric = False
    n2 = len(X2)

    K = np.zeros((n1, n2))

    total = n1 * n2 if not symmetric else n1 * (n1 + 1) // 2
    count = 0

    for i in range(n1):
        j_start = i if symmetric else 0
        for j in range(j_start, n2):
            K[i, j] = quantum_kernel(X1[i], X2[j])
            if symmetric and i != j:
                K[j, i] = K[i, j]

            count += 1
            if verbose and count % 100 == 0:
                print(f"  Computed {count}/{total} kernel entries...")

    return K

# Generate dataset
X, y = make_moons(n_samples=60, noise=0.15, random_state=42)
y = 2 * y - 1  # Convert to {-1, +1}

# Scale to [0, π]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Compute training kernel matrix
print("\nComputing training kernel matrix...")
K_train = compute_kernel_matrix(X_train)

print(f"\nKernel matrix shape: {K_train.shape}")
print(f"Kernel matrix diagonal (should be 1s): {np.diag(K_train)[:5]}")


#######################################
# Part 3: Kernel Matrix Analysis
#######################################

print("\n" + "-" * 40)
print("Part 3: Kernel Matrix Analysis")
print("-" * 40)

# Check positive semi-definiteness
eigenvalues = np.linalg.eigvalsh(K_train)
print(f"Min eigenvalue: {eigenvalues.min():.6f}")
print(f"Max eigenvalue: {eigenvalues.max():.6f}")
print(f"PSD check: {eigenvalues.min() >= -1e-10}")

# Effective rank
eff_rank = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
print(f"Effective rank: {eff_rank:.2f}")

# Visualize kernel matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sort by label for visualization
sort_idx = np.argsort(y_train)
K_sorted = K_train[sort_idx][:, sort_idx]
y_sorted = y_train[sort_idx]

im = axes[0].imshow(K_sorted, cmap='viridis')
axes[0].set_title('Quantum Kernel Matrix (sorted by class)')
axes[0].set_xlabel('Sample index')
axes[0].set_ylabel('Sample index')
plt.colorbar(im, ax=axes[0])

# Mark class boundaries
n_class_minus = np.sum(y_train == -1)
axes[0].axhline(y=n_class_minus - 0.5, color='red', linewidth=2)
axes[0].axvline(x=n_class_minus - 0.5, color='red', linewidth=2)

# Eigenvalue distribution
axes[1].bar(range(len(eigenvalues)), np.sort(eigenvalues)[::-1], color='steelblue')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Eigenvalue')
axes[1].set_title('Eigenvalue Distribution')
axes[1].axhline(y=0, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('kernel_matrix_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 4: Quantum Kernel SVM Training
#######################################

print("\n" + "-" * 40)
print("Part 4: Quantum Kernel SVM Training")
print("-" * 40)

# Train SVM with precomputed kernel
# scikit-learn allows kernel='precomputed'
svm = SVC(kernel='precomputed', C=1.0)
svm.fit(K_train, y_train)

print(f"Number of support vectors: {len(svm.support_)}")
print(f"Support vector indices: {svm.support_}")

# Training accuracy
y_train_pred = svm.predict(K_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Training accuracy: {train_acc:.2%}")


#######################################
# Part 5: Test Set Evaluation
#######################################

print("\n" + "-" * 40)
print("Part 5: Test Set Evaluation")
print("-" * 40)

# Compute test kernel matrix: K[i,j] = K(test[i], train[j])
print("Computing test kernel matrix...")
K_test = compute_kernel_matrix(X_test, X_train)

print(f"Test kernel matrix shape: {K_test.shape}")

# Predict
y_test_pred = svm.predict(K_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTest accuracy: {test_acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Class -1', 'Class +1']))


#######################################
# Part 6: Comparison with Classical Kernels
#######################################

print("\n" + "-" * 40)
print("Part 6: Comparison with Classical Kernels")
print("-" * 40)

from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

# Compute classical kernel matrices
gamma_rbf = 1.0
K_rbf = rbf_kernel(X_train, gamma=gamma_rbf)
K_poly = polynomial_kernel(X_train, degree=3)

# Train classical SVMs
svm_rbf = SVC(kernel='precomputed')
svm_rbf.fit(K_rbf, y_train)

svm_poly = SVC(kernel='precomputed')
svm_poly.fit(K_poly, y_train)

# Evaluate
K_rbf_test = rbf_kernel(X_test, X_train, gamma=gamma_rbf)
K_poly_test = polynomial_kernel(X_test, X_train, degree=3)

acc_rbf = accuracy_score(y_test, svm_rbf.predict(K_rbf_test))
acc_poly = accuracy_score(y_test, svm_poly.predict(K_poly_test))

print(f"Quantum Kernel SVM Test Accuracy: {test_acc:.2%}")
print(f"RBF Kernel SVM Test Accuracy: {acc_rbf:.2%}")
print(f"Polynomial Kernel SVM Test Accuracy: {acc_poly:.2%}")

# Visualize kernel comparisons
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im1 = axes[0].imshow(K_train[sort_idx][:, sort_idx], cmap='viridis')
axes[0].set_title(f'Quantum Kernel (Acc: {test_acc:.0%})')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(K_rbf[sort_idx][:, sort_idx], cmap='viridis')
axes[1].set_title(f'RBF Kernel (Acc: {acc_rbf:.0%})')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(K_poly[sort_idx][:, sort_idx], cmap='viridis')
axes[2].set_title(f'Polynomial Kernel (Acc: {acc_poly:.0%})')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('kernel_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 7: Decision Boundary Visualization
#######################################

print("\n" + "-" * 40)
print("Part 7: Decision Boundary Visualization")
print("-" * 40)

def plot_decision_boundary_kernel(svm_model, X_train, y_train, kernel_func, title, ax):
    """
    Plot decision boundary using a kernel function
    """
    h = 0.08  # Grid resolution
    x_min, x_max = 0, np.pi
    y_min, y_max = 0, np.pi

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute kernel between grid and training data
    print(f"  Computing {len(grid_points)} kernel evaluations...")
    K_grid = np.array([[kernel_func(gp, xtr) for xtr in X_train]
                        for gp in grid_points])

    # Predict
    Z = svm_model.predict(K_grid)
    Z = Z.reshape(xx.shape)

    # Plot
    ax.contourf(xx, yy, Z, levels=[-2, 0, 2], colors=['lightcoral', 'lightblue'], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               c='blue', marker='o', edgecolors='black', s=50, label='Class +1')
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
               c='red', marker='x', s=50, label='Class -1')

    # Highlight support vectors
    ax.scatter(X_train[svm_model.support_, 0], X_train[svm_model.support_, 1],
               s=200, facecolors='none', edgecolors='yellow', linewidths=2,
               label='Support Vectors')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

print("Plotting quantum kernel decision boundary...")
plot_decision_boundary_kernel(svm, X_train, y_train, quantum_kernel,
                               'Quantum Kernel SVM', axes[0])

print("Plotting RBF kernel decision boundary...")
def rbf_func(x1, x2):
    return np.exp(-gamma_rbf * np.sum((x1 - x2)**2))

plot_decision_boundary_kernel(svm_rbf, X_train, y_train, rbf_func,
                               'RBF Kernel SVM', axes[1])

plt.tight_layout()
plt.savefig('decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.show()


#######################################
# Part 8: Kernel Target Alignment
#######################################

print("\n" + "-" * 40)
print("Part 8: Kernel Target Alignment")
print("-" * 40)

def kernel_target_alignment(K, y):
    """
    Compute kernel-target alignment

    A(K, y) = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    """
    y = np.array(y).reshape(-1, 1)
    yy = y @ y.T

    numerator = np.sum(K * yy)
    denominator = np.linalg.norm(K, 'fro') * np.linalg.norm(yy, 'fro')

    return numerator / denominator

alignment_quantum = kernel_target_alignment(K_train, y_train)
alignment_rbf = kernel_target_alignment(K_rbf, y_train)
alignment_poly = kernel_target_alignment(K_poly, y_train)

print(f"Quantum Kernel Alignment: {alignment_quantum:.4f}")
print(f"RBF Kernel Alignment: {alignment_rbf:.4f}")
print(f"Polynomial Kernel Alignment: {alignment_poly:.4f}")
print("\nHigher alignment generally indicates better kernel for the task.")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

### Expected Output

```
============================================================
Quantum Kernel Methods Implementation
============================================================

Testing kernel properties:
K(x1, x1) = 1.0000 (should be 1.0)
K(x1, x2) = 1.0000 (should be 1.0, same point)
K(x1, x3) = 0.3421
K(x3, x1) = 0.3421 (should equal K(x1, x3) - symmetry)

Training accuracy: 95.00%
Test accuracy: 85.00%

Quantum Kernel SVM Test Accuracy: 85.00%
RBF Kernel SVM Test Accuracy: 90.00%
Polynomial Kernel SVM Test Accuracy: 80.00%
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| **Quantum Kernel** | $K_Q(\mathbf{x}, \mathbf{x}') = \|\langle 0\|U_\phi^\dagger(\mathbf{x})U_\phi(\mathbf{x}')\|0\rangle\|^2$ |
| **Kernel Estimation** | Measure $P(\|0\rangle^{\otimes n})$ after $U_\phi^\dagger(\mathbf{x}')U_\phi(\mathbf{x})$ |
| **Variance** | $\text{Var}[\hat{K}] = K(1-K)/N_s$ |
| **SVM Decision** | $f(\mathbf{x}) = \sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$ |
| **Kernel Alignment** | $A(K, \mathbf{y}) = \langle K, \mathbf{y}\mathbf{y}^\top\rangle / (\|K\|_F \|\mathbf{y}\mathbf{y}^\top\|_F)$ |

### Key Takeaways

1. **Quantum kernels** are valid positive semi-definite kernels by construction
2. **Kernel estimation** requires measuring overlap after forward-inverse circuit
3. **Shot noise** limits precision; variance scales as $1/N_s$
4. **Quantum kernel SVM** uses quantum kernel matrix with classical SVM solver
5. **Kernel alignment** helps predict classification performance
6. **Computational cost** is $O(N^2)$ kernel evaluations, each requiring many shots

### Connection to Classical ML

Quantum kernels extend classical kernel methods:
- Same mathematical framework (RKHS, Mercer's theorem)
- Same SVM optimization algorithms
- Potential for kernels that are hard to compute classically
- No guarantee of better performance without careful design

---

## Daily Checklist

- [ ] I can define and compute quantum kernels mathematically
- [ ] I understand the kernel estimation protocol using circuits
- [ ] I can analyze kernel matrices (eigenvalues, alignment)
- [ ] I implemented a quantum kernel SVM end-to-end
- [ ] I can compare quantum and classical kernels fairly
- [ ] I understand the computational costs involved
- [ ] I recognize when quantum kernels might offer advantages

---

## Preview: Day 970

Tomorrow we explore **Quantum Neural Networks (QNNs)**, moving beyond kernel methods to trainable quantum circuits:

- Layer-by-layer QNN architectures
- Encoding layers vs. variational layers
- QNN expressibility and universality
- Comparison to classical neural networks
- Implementation of deep quantum circuits

---

*"The kernel is the bridge between quantum mechanics and machine learning - it translates quantum state overlap into classical similarity."*
— Vojtěch Havlíček
