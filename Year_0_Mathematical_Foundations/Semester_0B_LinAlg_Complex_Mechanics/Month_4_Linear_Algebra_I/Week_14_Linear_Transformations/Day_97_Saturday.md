# Day 97: Computational Lab — Linear Transformations in Action

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 1:00 PM | 3 hours | Lab Part 1: Matrix Representations |
| Afternoon | 3:00 PM - 6:00 PM | 3 hours | Lab Part 2: Quantum Operators |
| Evening | 7:30 PM - 9:00 PM | 1.5 hours | Lab Part 3: Visualization & Projects |

**Total Study Time: 7.5 hours**

---

## 🎯 Lab Objectives

By the end of today's lab, you should be able to:

1. Implement linear transformations as matrix operations in NumPy
2. Compute kernel and range numerically
3. Verify rank-nullity theorem computationally
4. Implement quantum gates as unitary matrices
5. Visualize 2D/3D transformations
6. Build a quantum circuit simulator foundation

---

## 💻 Lab Part 1: Matrix Representations (3 hours)

### 1.1 Linear Transformations as Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
np.set_printoptions(precision=4, suppress=True)

# ============================================
# Defining Linear Transformations
# ============================================

def is_linear(T: Callable, V_dim: int, num_tests: int = 100, tol: float = 1e-10) -> bool:
    """
    Test if a function T: R^n -> R^m is linear.
    
    A function is linear if:
    1. T(u + v) = T(u) + T(v)  (additivity)
    2. T(cv) = cT(v)          (homogeneity)
    """
    for _ in range(num_tests):
        # Random vectors and scalar
        u = np.random.randn(V_dim)
        v = np.random.randn(V_dim)
        c = np.random.randn()
        
        # Test additivity
        if not np.allclose(T(u + v), T(u) + T(v), atol=tol):
            return False
        
        # Test homogeneity
        if not np.allclose(T(c * v), c * T(v), atol=tol):
            return False
    
    return True

# Example: Linear function
def rotation_90(v):
    """Rotate by 90 degrees counterclockwise in R²"""
    return np.array([-v[1], v[0]])

# Example: Non-linear function
def translation(v):
    """Translation is NOT linear"""
    return v + np.array([1, 1])

def quadratic(v):
    """Quadratic function is NOT linear"""
    return v ** 2

print("Testing linearity:")
print(f"Rotation 90°: {is_linear(rotation_90, 2)}")
print(f"Translation: {is_linear(translation, 2)}")
print(f"Quadratic: {is_linear(quadratic, 2)}")

# ============================================
# Matrix Representation
# ============================================

def get_matrix_from_transformation(T: Callable, domain_dim: int, codomain_dim: int) -> np.ndarray:
    """
    Find the matrix representation of a linear transformation.
    
    Strategy: Apply T to each standard basis vector.
    The columns of the matrix are T(e_1), T(e_2), ..., T(e_n)
    """
    matrix = np.zeros((codomain_dim, domain_dim))
    
    for j in range(domain_dim):
        # Create j-th standard basis vector
        e_j = np.zeros(domain_dim)
        e_j[j] = 1.0
        
        # Apply transformation
        matrix[:, j] = T(e_j)
    
    return matrix

# Get matrix for rotation
R_90 = get_matrix_from_transformation(rotation_90, 2, 2)
print("\nRotation matrix (90° counterclockwise):")
print(R_90)

# Verify: matrix multiplication should equal function application
v = np.array([3, 4])
print(f"\nT(v) via function: {rotation_90(v)}")
print(f"T(v) via matrix:   {R_90 @ v}")
```

### 1.2 Common 2D Transformations

```python
# ============================================
# Library of 2D Transformations
# ============================================

def rotation_matrix(theta: float) -> np.ndarray:
    """Rotation by angle theta (radians) counterclockwise"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def reflection_matrix(axis: str) -> np.ndarray:
    """Reflection across x-axis, y-axis, or y=x line"""
    if axis == 'x':
        return np.array([[1,  0],
                         [0, -1]])
    elif axis == 'y':
        return np.array([[-1, 0],
                         [ 0, 1]])
    elif axis == 'y=x':
        return np.array([[0, 1],
                         [1, 0]])
    else:
        raise ValueError(f"Unknown axis: {axis}")

def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """Scale by sx in x-direction, sy in y-direction"""
    return np.array([[sx,  0],
                     [ 0, sy]])

def shear_matrix(direction: str, k: float) -> np.ndarray:
    """Shear transformation"""
    if direction == 'x':  # Shear in x-direction
        return np.array([[1, k],
                         [0, 1]])
    elif direction == 'y':  # Shear in y-direction
        return np.array([[1, 0],
                         [k, 1]])

def projection_matrix(onto: str) -> np.ndarray:
    """Orthogonal projection onto an axis"""
    if onto == 'x':
        return np.array([[1, 0],
                         [0, 0]])
    elif onto == 'y':
        return np.array([[0, 0],
                         [0, 1]])

# Demonstrate each transformation
print("=== 2D Transformation Matrices ===\n")

print("Rotation by 45°:")
print(rotation_matrix(np.pi/4))

print("\nReflection across y-axis:")
print(reflection_matrix('y'))

print("\nScaling (2x horizontal, 0.5x vertical):")
print(scaling_matrix(2, 0.5))

print("\nShear in x-direction (k=1):")
print(shear_matrix('x', 1))

print("\nProjection onto x-axis:")
print(projection_matrix('x'))
```

### 1.3 Kernel and Range Computation

```python
# ============================================
# Computing Kernel (Null Space)
# ============================================

def compute_kernel(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the kernel (null space) of A.
    
    Uses SVD: A = UΣV^T
    Kernel basis = columns of V corresponding to zero singular values
    """
    U, sigma, Vt = np.linalg.svd(A)
    
    # Find indices of zero (or near-zero) singular values
    null_mask = sigma < tol
    
    # Handle case where A has more columns than rows
    if len(sigma) < A.shape[1]:
        # Remaining columns of V are automatically in kernel
        null_mask = np.concatenate([null_mask, np.ones(A.shape[1] - len(sigma), dtype=bool)])
    
    # Kernel basis = rows of V^T (columns of V) for zero singular values
    kernel_basis = Vt[null_mask].T
    
    return kernel_basis

def compute_range(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the range (column space) of A.
    
    Uses SVD: A = UΣV^T
    Range basis = columns of U corresponding to nonzero singular values
    """
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Find indices of nonzero singular values
    rank = np.sum(sigma > tol)
    
    # Range basis = first 'rank' columns of U
    range_basis = U[:, :rank]
    
    return range_basis

# Example: Projection matrix
P = projection_matrix('x')
print("=== Analysis of Projection onto x-axis ===\n")
print("Matrix P:")
print(P)

ker_P = compute_kernel(P)
range_P = compute_range(P)

print(f"\nKernel dimension: {ker_P.shape[1] if ker_P.size > 0 else 0}")
if ker_P.size > 0:
    print("Kernel basis:")
    print(ker_P)

print(f"\nRange dimension: {range_P.shape[1]}")
print("Range basis:")
print(range_P)

# Verify rank-nullity theorem
print(f"\nRank-Nullity Check:")
print(f"dim(kernel) + dim(range) = {ker_P.shape[1] if ker_P.size > 0 else 0} + {range_P.shape[1]} = {(ker_P.shape[1] if ker_P.size > 0 else 0) + range_P.shape[1]}")
print(f"dim(domain) = {P.shape[1]}")
```

### 1.4 Rank-Nullity Theorem Verification

```python
# ============================================
# Comprehensive Rank-Nullity Verification
# ============================================

def analyze_transformation(A: np.ndarray, name: str = "T"):
    """Complete analysis of a linear transformation"""
    m, n = A.shape
    
    print(f"=== Analysis of {name}: R^{n} → R^{m} ===\n")
    print("Matrix:")
    print(A)
    
    # Compute rank using SVD
    rank = np.linalg.matrix_rank(A)
    nullity = n - rank
    
    print(f"\nDimensions:")
    print(f"  Domain dimension: {n}")
    print(f"  Codomain dimension: {m}")
    print(f"  Rank (dim of range): {rank}")
    print(f"  Nullity (dim of kernel): {nullity}")
    
    # Verify rank-nullity
    print(f"\nRank-Nullity Theorem:")
    print(f"  rank + nullity = {rank} + {nullity} = {rank + nullity}")
    print(f"  dim(domain) = {n}")
    print(f"  Theorem satisfied: {rank + nullity == n}")
    
    # Compute actual bases
    kernel = compute_kernel(A)
    range_space = compute_range(A)
    
    if kernel.size > 0:
        print(f"\nKernel basis (vectors that map to 0):")
        for i in range(kernel.shape[1]):
            print(f"  v_{i+1} = {kernel[:, i]}")
            # Verify it's in kernel
            assert np.allclose(A @ kernel[:, i], 0), "Kernel computation error!"
    else:
        print("\nKernel: {0} (trivial)")
    
    print(f"\nRange basis (reachable vectors):")
    for i in range(range_space.shape[1]):
        print(f"  w_{i+1} = {range_space[:, i]}")
    
    # Classification
    print(f"\nClassification:")
    if rank == n and rank == m:
        print("  Bijective (invertible) - one-to-one and onto")
    elif rank == n:
        print("  Injective (one-to-one) - trivial kernel")
    elif rank == m:
        print("  Surjective (onto) - full range")
    else:
        print("  Neither injective nor surjective")
    
    return rank, nullity

# Test various matrices
print("\n" + "="*60 + "\n")

# Example 1: Full rank square matrix
A1 = np.array([[1, 2],
               [3, 4]])
analyze_transformation(A1, "Invertible 2×2")

print("\n" + "="*60 + "\n")

# Example 2: Projection (rank deficient)
A2 = projection_matrix('x')
analyze_transformation(A2, "Projection onto x-axis")

print("\n" + "="*60 + "\n")

# Example 3: "Tall" matrix (more rows than columns)
A3 = np.array([[1, 0],
               [0, 1],
               [1, 1]])
analyze_transformation(A3, "R² → R³ embedding")

print("\n" + "="*60 + "\n")

# Example 4: "Wide" matrix (more columns than rows)
A4 = np.array([[1, 2, 3],
               [4, 5, 6]])
analyze_transformation(A4, "R³ → R² projection-like")

print("\n" + "="*60 + "\n")

# Example 5: Rank-1 matrix
A5 = np.array([[1, 2, 3],
               [2, 4, 6],
               [3, 6, 9]])
analyze_transformation(A5, "Rank-1 matrix")
```

---

## 💻 Lab Part 2: Quantum Operators (3 hours)

### 2.1 Quantum Gates as Unitary Matrices

```python
# ============================================
# Quantum Gate Library
# ============================================

# Pauli matrices
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)  # NOT gate, bit flip

Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)  # Phase flip

# Hadamard gate
H = np.array([[1, 1],
              [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gates
S = np.array([[1, 0],
              [0, 1j]], dtype=complex)  # √Z gate

T = np.array([[1, 0],
              [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # π/8 gate

def Rx(theta):
    """Rotation around X-axis by angle theta"""
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def Ry(theta):
    """Rotation around Y-axis by angle theta"""
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def Rz(theta):
    """Rotation around Z-axis by angle theta"""
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

print("=== Quantum Gates ===\n")
print("Pauli X (NOT gate):")
print(X)
print("\nHadamard gate:")
print(H)
print("\nPhase gate S:")
print(S)
```

### 2.2 Verifying Unitarity

```python
# ============================================
# Unitarity Verification
# ============================================

def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if matrix U is unitary: U†U = UU† = I
    
    Unitary matrices preserve inner products and norms.
    """
    n = U.shape[0]
    identity = np.eye(n)
    
    # U†U = I
    cond1 = np.allclose(U.conj().T @ U, identity, atol=tol)
    
    # UU† = I
    cond2 = np.allclose(U @ U.conj().T, identity, atol=tol)
    
    return cond1 and cond2

def verify_unitary_properties(U: np.ndarray, name: str):
    """Comprehensive unitary matrix analysis"""
    print(f"=== {name} ===")
    print(f"Matrix:\n{U}\n")
    
    # Check unitarity
    print(f"Is unitary: {is_unitary(U)}")
    
    # Compute U†U
    UdagU = U.conj().T @ U
    print(f"\nU†U (should be I):\n{UdagU}")
    
    # Determinant (should have |det| = 1)
    det = np.linalg.det(U)
    print(f"\nDeterminant: {det:.4f}")
    print(f"|det| = {np.abs(det):.4f} (should be 1)")
    
    # Eigenvalues (should all have |λ| = 1)
    eigenvalues = np.linalg.eigvals(U)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Magnitudes: {np.abs(eigenvalues)} (all should be 1)")
    
    print()

# Verify all gates
for gate, name in [(X, "Pauli X"), (Y, "Pauli Y"), (Z, "Pauli Z"), 
                   (H, "Hadamard"), (S, "Phase S"), (T, "T gate")]:
    verify_unitary_properties(gate, name)
```

### 2.3 Quantum State Evolution

```python
# ============================================
# Quantum State Manipulation
# ============================================

# Computational basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

def normalize(psi):
    """Normalize a quantum state"""
    return psi / np.linalg.norm(psi)

def probability(psi, basis_state):
    """Compute measurement probability"""
    amplitude = np.vdot(basis_state, psi)  # <basis|psi>
    return np.abs(amplitude)**2

def apply_gate(gate, psi):
    """Apply a quantum gate to a state"""
    return gate @ psi

def state_info(psi, name="ψ"):
    """Display quantum state information"""
    print(f"|{name}⟩ = {psi}")
    print(f"  Normalization: |⟨{name}|{name}⟩| = {np.linalg.norm(psi):.4f}")
    print(f"  P(measure |0⟩) = {probability(psi, ket_0):.4f}")
    print(f"  P(measure |1⟩) = {probability(psi, ket_1):.4f}")
    print()

# Start with |0⟩
print("=== Quantum State Evolution ===\n")
psi = ket_0.copy()
state_info(psi, "0")

# Apply Hadamard: |0⟩ → |+⟩ = (|0⟩ + |1⟩)/√2
psi = apply_gate(H, psi)
state_info(psi, "+")

# Apply Z gate: |+⟩ → |−⟩ = (|0⟩ - |1⟩)/√2
psi = apply_gate(Z, psi)
state_info(psi, "-")

# Apply Hadamard again: |−⟩ → |1⟩
psi = apply_gate(H, psi)
state_info(psi, "1")

print("Key insight: H·Z·H = X (bit flip)!")
print(f"H @ Z @ H =\n{H @ Z @ H}")
print(f"X =\n{X}")
print(f"Equal: {np.allclose(H @ Z @ H, X)}")
```

### 2.4 Two-Qubit Gates

```python
# ============================================
# Two-Qubit Systems
# ============================================

def tensor_product(A, B):
    """Compute tensor (Kronecker) product A ⊗ B"""
    return np.kron(A, B)

# Two-qubit basis states
ket_00 = tensor_product(ket_0, ket_0)
ket_01 = tensor_product(ket_0, ket_1)
ket_10 = tensor_product(ket_1, ket_0)
ket_11 = tensor_product(ket_1, ket_1)

print("=== Two-Qubit Basis States ===\n")
print(f"|00⟩ = {ket_00}")
print(f"|01⟩ = {ket_01}")
print(f"|10⟩ = {ket_10}")
print(f"|11⟩ = {ket_11}")

# CNOT gate (controlled-NOT)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

print("\n=== CNOT Gate ===")
print("CNOT matrix:")
print(CNOT)

# CNOT is unitary
print(f"\nCNOT is unitary: {is_unitary(CNOT)}")

# Demonstrate CNOT action
print("\nCNOT action on basis states:")
print(f"CNOT|00⟩ = {CNOT @ ket_00}  (no flip, control=0)")
print(f"CNOT|01⟩ = {CNOT @ ket_01}  (no flip, control=0)")
print(f"CNOT|10⟩ = {CNOT @ ket_10}  (flip! control=1)")
print(f"CNOT|11⟩ = {CNOT @ ket_11}  (flip! control=1)")

# Create Bell state
print("\n=== Creating Bell State ===")
# Start with |00⟩
psi = ket_00.copy()
print(f"Initial: |00⟩ = {psi}")

# Apply H to first qubit: H⊗I
H_I = tensor_product(H, I)
psi = H_I @ psi
print(f"After H⊗I: {psi}")

# Apply CNOT
psi = CNOT @ psi
print(f"After CNOT: {psi}")
print(f"\nThis is |Φ+⟩ = (|00⟩ + |11⟩)/√2, a maximally entangled Bell state!")

# Verify it's entangled (cannot be written as product state)
# If separable: |ψ⟩ = (a|0⟩ + b|1⟩) ⊗ (c|0⟩ + d|1⟩) = ac|00⟩ + ad|01⟩ + bc|10⟩ + bd|11⟩
# For |Φ+⟩: coefficient of |00⟩ = 1/√2, |11⟩ = 1/√2, |01⟩ = 0, |10⟩ = 0
# This requires ac = 1/√2, bd = 1/√2, ad = 0, bc = 0
# If ad = 0, either a=0 or d=0. If a=0, then ac=0≠1/√2. If d=0, then bd=0≠1/√2.
# Contradiction! So |Φ+⟩ is entangled.
```

### 2.5 Quantum Circuit Simulator

```python
# ============================================
# Simple Quantum Circuit Simulator
# ============================================

class QuantumCircuit:
    """A simple quantum circuit simulator"""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # Initialize to |00...0⟩
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.gates_applied = []
    
    def reset(self):
        """Reset to |00...0⟩"""
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.gates_applied = []
    
    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """Apply a single-qubit gate to specified qubit (0-indexed)"""
        # Build full operator using tensor products
        full_gate = np.eye(1)
        for q in range(self.num_qubits):
            if q == qubit:
                full_gate = tensor_product(full_gate, gate)
            else:
                full_gate = tensor_product(full_gate, I)
        
        self.state = full_gate @ self.state
        self.gates_applied.append(f"Gate on qubit {qubit}")
    
    def apply_cnot(self, control: int, target: int):
        """Apply CNOT with specified control and target qubits"""
        # For simplicity, only implement for adjacent qubits in small systems
        if self.num_qubits == 2 and control == 0 and target == 1:
            self.state = CNOT @ self.state
        else:
            # General implementation would go here
            raise NotImplementedError("General CNOT not implemented")
        
        self.gates_applied.append(f"CNOT({control}→{target})")
    
    def measure_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states"""
        return np.abs(self.state) ** 2
    
    def get_state_string(self) -> str:
        """Pretty print the state"""
        terms = []
        for i, amp in enumerate(self.state):
            if np.abs(amp) > 1e-10:
                basis = format(i, f'0{self.num_qubits}b')
                if np.isclose(amp.imag, 0):
                    terms.append(f"{amp.real:.4f}|{basis}⟩")
                else:
                    terms.append(f"({amp:.4f})|{basis}⟩")
        return " + ".join(terms)
    
    def display(self):
        """Display current state and probabilities"""
        print(f"State: {self.get_state_string()}")
        probs = self.measure_probabilities()
        print("Probabilities:")
        for i, p in enumerate(probs):
            if p > 1e-10:
                basis = format(i, f'0{self.num_qubits}b')
                print(f"  P(|{basis}⟩) = {p:.4f}")

# Demonstrate the simulator
print("=== Quantum Circuit Simulator ===\n")

qc = QuantumCircuit(2)
print("Initial state:")
qc.display()

print("\nApply Hadamard to qubit 0:")
qc.apply_single_qubit_gate(H, 0)
qc.display()

print("\nApply CNOT (control=0, target=1):")
qc.apply_cnot(0, 1)
qc.display()

print("\n→ We created the Bell state |Φ+⟩!")
```

---

## 💻 Lab Part 3: Visualization & Projects (1.5 hours)

### 3.1 Visualizing 2D Transformations

```python
# ============================================
# Visualization of Linear Transformations
# ============================================

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

def plot_transformation_2d(A: np.ndarray, title: str = "Linear Transformation"):
    """
    Visualize a 2D linear transformation by showing:
    1. Original unit square
    2. Transformed parallelogram
    3. Basis vectors and their images
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original points: unit square vertices
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    
    # Transform
    transformed = A @ square
    
    # Original space
    ax1 = axes[0]
    ax1.plot(square[0], square[1], 'b-', linewidth=2, label='Unit Square')
    ax1.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='red', width=0.02)
    ax1.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='green', width=0.02)
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Space')
    ax1.legend(['Unit Square', 'e₁', 'e₂'])
    
    # Transformed space
    ax2 = axes[1]
    ax2.plot(transformed[0], transformed[1], 'b-', linewidth=2, label='Transformed')
    # Transformed basis vectors (columns of A)
    ax2.quiver(0, 0, A[0,0], A[1,0], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.02, label=f'T(e₁)={A[:,0]}')
    ax2.quiver(0, 0, A[0,1], A[1,1], angles='xy', scale_units='xy', scale=1, 
               color='green', width=0.02, label=f'T(e₂)={A[:,1]}')
    
    # Set appropriate limits
    all_pts = np.hstack([transformed, [[0], [0]]])
    margin = 0.5
    ax2.set_xlim(all_pts[0].min() - margin, all_pts[0].max() + margin)
    ax2.set_ylim(all_pts[1].min() - margin, all_pts[1].max() + margin)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'After {title}')
    ax2.legend()
    
    # Add determinant info
    det = np.linalg.det(A)
    fig.suptitle(f'{title}\ndet(A) = {det:.3f} (area scaling factor)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'transform_{title.replace(" ", "_").lower()}.png', dpi=150)
    plt.show()

# Demonstrate various transformations
transformations = [
    (rotation_matrix(np.pi/4), "Rotation 45°"),
    (scaling_matrix(2, 0.5), "Scaling (2x, 0.5x)"),
    (shear_matrix('x', 1), "Shear X"),
    (reflection_matrix('y'), "Reflection across Y"),
    (projection_matrix('x'), "Projection onto X"),
]

for A, name in transformations:
    print(f"\n{name}:")
    print(f"Matrix:\n{A}")
    print(f"Determinant: {np.linalg.det(A):.4f}")
    plot_transformation_2d(A, name)
```

### 3.2 Bloch Sphere Visualization

```python
# ============================================
# Bloch Sphere Representation
# ============================================

def state_to_bloch(psi: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a qubit state |ψ⟩ = α|0⟩ + β|1⟩ to Bloch sphere coordinates (x, y, z).
    
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    Bloch vector: (sin θ cos φ, sin θ sin φ, cos θ)
    """
    # Normalize
    psi = psi / np.linalg.norm(psi)
    
    # Extract α and β
    alpha, beta = psi[0], psi[1]
    
    # Compute Bloch coordinates using density matrix
    # ρ = |ψ⟩⟨ψ| = [[|α|², αβ*], [α*β, |β|²]]
    # x = 2 Re(αβ*) = Tr(ρ X)
    # y = 2 Im(αβ*) = Tr(ρ Y)  
    # z = |α|² - |β|² = Tr(ρ Z)
    
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    
    return x, y, z

def plot_bloch_sphere(states: List[Tuple[np.ndarray, str]], title: str = "Bloch Sphere"):
    """Plot multiple quantum states on the Bloch sphere"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    # Draw axes
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='red', arrow_length_ratio=0.1, alpha=0.5)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='green', arrow_length_ratio=0.1, alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='blue', arrow_length_ratio=0.1, alpha=0.5)
    ax.text(1.4, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.4, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.4, 'Z', fontsize=12)
    
    # Mark |0⟩ and |1⟩
    ax.scatter([0], [0], [1], color='blue', s=100, marker='^')
    ax.scatter([0], [0], [-1], color='blue', s=100, marker='v')
    ax.text(0, 0, 1.1, '|0⟩', fontsize=10)
    ax.text(0, 0, -1.2, '|1⟩', fontsize=10)
    
    # Plot states
    colors = plt.cm.rainbow(np.linspace(0, 1, len(states)))
    for (psi, name), color in zip(states, colors):
        x, y, z = state_to_bloch(psi)
        ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.1, linewidth=2)
        ax.scatter([x], [y], [z], color=color, s=50)
        ax.text(x*1.1, y*1.1, z*1.1, name, fontsize=9)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_title(title)
    
    plt.savefig('bloch_sphere.png', dpi=150)
    plt.show()

# Demonstrate key states
ket_plus = normalize(ket_0 + ket_1)       # |+⟩
ket_minus = normalize(ket_0 - ket_1)      # |−⟩
ket_plus_i = normalize(ket_0 + 1j*ket_1)  # |+i⟩
ket_minus_i = normalize(ket_0 - 1j*ket_1) # |−i⟩

states = [
    (ket_0, '|0⟩'),
    (ket_1, '|1⟩'),
    (ket_plus, '|+⟩'),
    (ket_minus, '|−⟩'),
    (ket_plus_i, '|+i⟩'),
    (ket_minus_i, '|−i⟩'),
]

plot_bloch_sphere(states, "Key Qubit States on Bloch Sphere")

# Show gate evolution
print("\n=== Gate Action on Bloch Sphere ===")

# Starting from |0⟩, apply sequence of gates
psi = ket_0.copy()
trajectory = [(psi.copy(), '|0⟩')]

psi = H @ psi
trajectory.append((psi.copy(), 'H|0⟩'))

psi = S @ psi
trajectory.append((psi.copy(), 'SH|0⟩'))

psi = H @ psi
trajectory.append((psi.copy(), 'HSH|0⟩'))

plot_bloch_sphere(trajectory, "State Evolution: |0⟩ → H → S → H")
```

---

## 📝 Lab Exercises

### Exercise 1: Transformation Analysis
Implement a function that takes any matrix A and produces a complete report including:
- Dimensions and rank
- Kernel basis (with verification)
- Range basis (with verification)
- Whether it's injective/surjective/bijective
- Eigenvalue analysis

### Exercise 2: Quantum Gate Compiler
Create a function that decomposes a given 2×2 unitary matrix into a product of Rx, Ry, Rz rotations (up to global phase).

### Exercise 3: Entanglement Detection
Implement a function that determines if a two-qubit pure state is entangled or separable by attempting Schmidt decomposition.

### Exercise 4: Circuit Equivalence
Verify computationally that:
- HZH = X
- HXH = Z  
- HTH = Rx(π/4) (up to global phase)

---

## ✅ Lab Checklist

- [ ] Implemented transformation → matrix conversion
- [ ] Computed kernel and range for various matrices
- [ ] Verified rank-nullity theorem (5+ examples)
- [ ] Implemented quantum gate library
- [ ] Verified unitarity of all gates
- [ ] Created Bell state with circuit simulator
- [ ] Visualized 2D transformations
- [ ] Plotted states on Bloch sphere
- [ ] Completed at least 2 lab exercises
- [ ] Saved all visualizations

---

## 🔗 QM Connection Summary

Today's lab bridges linear algebra to quantum computing:

| Linear Algebra | Quantum Implementation |
|----------------|----------------------|
| Linear transformation | Quantum gate |
| Unitary matrix | Valid quantum operation |
| Matrix multiplication | Gate composition |
| Tensor product | Multi-qubit systems |
| Kernel of T | States annihilated by T |
| Range of T | Reachable states |

**Key insight:** The constraint that quantum gates must be unitary is because:
1. Unitary transformations preserve norms (probability conservation)
2. They're invertible (quantum operations are reversible)
3. They preserve inner products (distinguishability)

---

## 🔜 Tomorrow: Week 14 Review

Tomorrow we consolidate everything from linear transformations:
- Matrix representations
- Kernel and range
- Rank-nullity theorem
- Connection to quantum operators

**Prepare:** Review all days 92-97, identify any gaps.

---

*"The purpose of computing is insight, not numbers."*
— Richard Hamming
