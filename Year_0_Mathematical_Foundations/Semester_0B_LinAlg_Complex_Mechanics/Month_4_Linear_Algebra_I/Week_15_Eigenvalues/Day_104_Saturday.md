# Day 104: Computational Lab — Eigenvalues in Action

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 1:00 PM | 3 hours | Lab Part 1: Numerical Methods |
| Afternoon | 3:00 PM - 6:00 PM | 3 hours | Lab Part 2: Quantum Applications |
| Evening | 7:30 PM - 9:00 PM | 1.5 hours | Lab Part 3: Projects |

**Total Study Time: 7.5 hours**

---

## 🎯 Lab Objectives

1. Implement power method and QR algorithm
2. Visualize eigenvalue decompositions
3. Build quantum state evolution simulator
4. Apply PCA to data
5. Simulate quantum gates and circuits

---

## 💻 Lab Part 1: Numerical Eigenvalue Methods (3 hours)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)

# ============================================
# Power Method Implementation
# ============================================

def power_method(A, max_iter=100, tol=1e-10):
    """Find dominant eigenvalue and eigenvector"""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for i in range(max_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        
        # Rayleigh quotient for eigenvalue estimate
        lam = v_new @ A @ v_new
        
        if np.linalg.norm(v_new - v) < tol:
            return lam, v_new, i+1
        v = v_new
    
    return lam, v, max_iter

# Test
A = np.array([[4, 1], [2, 3]])
lam, v, iters = power_method(A)
print(f"Power method: λ = {lam:.6f} in {iters} iterations")
print(f"True eigenvalues: {np.linalg.eigvals(A)}")

# ============================================
# QR Algorithm
# ============================================

def qr_algorithm(A, max_iter=100, tol=1e-10):
    """Find all eigenvalues using QR iteration"""
    Ak = A.copy().astype(float)
    n = A.shape[0]
    
    eigenvalue_history = []
    
    for i in range(max_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
        
        # Track diagonal (converges to eigenvalues)
        eigenvalue_history.append(np.diag(Ak).copy())
        
        # Check convergence (off-diagonal → 0)
        off_diag = np.sum(np.abs(Ak - np.diag(np.diag(Ak))))
        if off_diag < tol:
            break
    
    return np.diag(Ak), np.array(eigenvalue_history), i+1

A = np.array([[4, 1, 1],
              [1, 3, 1],
              [1, 1, 2]])

eigenvalues, history, iters = qr_algorithm(A)
print(f"\nQR Algorithm: eigenvalues = {eigenvalues} in {iters} iterations")
print(f"True: {np.sort(np.linalg.eigvals(A))[::-1]}")

# Plot convergence
plt.figure(figsize=(10, 4))
for i in range(history.shape[1]):
    plt.plot(history[:, i], label=f'λ_{i+1}')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue estimate')
plt.title('QR Algorithm Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('qr_convergence.png', dpi=150)
plt.show()

# ============================================
# Inverse Power Method (for smallest eigenvalue)
# ============================================

def inverse_power_method(A, max_iter=100, tol=1e-10):
    """Find smallest eigenvalue using inverse iteration"""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    A_inv = np.linalg.inv(A)
    
    for i in range(max_iter):
        v_new = A_inv @ v
        v_new = v_new / np.linalg.norm(v_new)
        
        # Eigenvalue of A (not A^{-1})
        lam = v_new @ A @ v_new
        
        if np.linalg.norm(v_new - v) < tol:
            return lam, v_new, i+1
        v = v_new
    
    return lam, v, max_iter

lam_min, v_min, iters = inverse_power_method(A)
print(f"\nInverse power method: λ_min = {lam_min:.6f}")
```

---

## 💻 Lab Part 2: Quantum Applications (3 hours)

```python
# ============================================
# Quantum State Evolution Simulator
# ============================================

class QuantumSimulator:
    """Simulate quantum state evolution"""
    
    def __init__(self, dim=2):
        self.dim = dim
        self.state = np.zeros(dim, dtype=complex)
        self.state[0] = 1  # Start in |0⟩
    
    def set_state(self, psi):
        self.state = psi / np.linalg.norm(psi)
    
    def evolve(self, H, t):
        """Evolve under Hamiltonian H for time t"""
        U = expm(-1j * H * t)
        self.state = U @ self.state
    
    def apply_gate(self, U):
        """Apply unitary gate"""
        self.state = U @ self.state
    
    def measure_observable(self, A):
        """Compute expectation value of observable"""
        return np.real(np.vdot(self.state, A @ self.state))
    
    def measure_probabilities(self):
        """Get measurement probabilities in computational basis"""
        return np.abs(self.state)**2

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Simulate Rabi oscillations
sim = QuantumSimulator(2)
H_rabi = X  # Drive with σ_x

t_values = np.linspace(0, 4*np.pi, 200)
prob_0 = []
prob_1 = []

for t in t_values:
    sim.set_state(np.array([1, 0], dtype=complex))  # Reset to |0⟩
    sim.evolve(H_rabi, t)
    probs = sim.measure_probabilities()
    prob_0.append(probs[0])
    prob_1.append(probs[1])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_values, prob_0, 'b-', label='P(|0⟩)')
plt.plot(t_values, prob_1, 'r-', label='P(|1⟩)')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Rabi Oscillations (H = σ_x)')
plt.legend()
plt.grid(True, alpha=0.3)

# Bloch sphere trajectory
plt.subplot(1, 2, 2)
expect_x, expect_y, expect_z = [], [], []
for t in t_values:
    sim.set_state(np.array([1, 0], dtype=complex))
    sim.evolve(H_rabi, t)
    expect_x.append(sim.measure_observable(X))
    expect_y.append(sim.measure_observable(Y))
    expect_z.append(sim.measure_observable(Z))

plt.plot(t_values, expect_x, 'r-', label='⟨X⟩')
plt.plot(t_values, expect_y, 'g-', label='⟨Y⟩')
plt.plot(t_values, expect_z, 'b-', label='⟨Z⟩')
plt.xlabel('Time')
plt.ylabel('Expectation')
plt.title('Bloch Vector Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rabi_oscillations.png', dpi=150)
plt.show()

# ============================================
# Two-Qubit Entanglement
# ============================================

def kron(A, B):
    return np.kron(A, B)

# Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]], dtype=complex)

H_I = kron(H_gate, I)

# Start with |00⟩
psi = np.array([1, 0, 0, 0], dtype=complex)
psi = H_I @ psi  # Apply H to first qubit
psi = CNOT @ psi  # Apply CNOT

print("\n=== Bell State Creation ===")
print(f"|Φ+⟩ = {psi}")
print("Expected: (|00⟩ + |11⟩)/√2")

# Verify entanglement via partial trace
def partial_trace(rho, keep_qubit):
    """Trace out one qubit from 2-qubit density matrix"""
    rho = rho.reshape(2, 2, 2, 2)
    if keep_qubit == 0:
        return np.trace(rho, axis1=1, axis2=3)
    else:
        return np.trace(rho, axis1=0, axis2=2)

rho = np.outer(psi, psi.conj())
rho_A = partial_trace(rho, 0)
print(f"\nReduced density matrix (qubit A):\n{rho_A}")
print(f"Trace: {np.trace(rho_A):.4f}")
print(f"Purity: {np.trace(rho_A @ rho_A):.4f}")
print("(Purity < 1 indicates entanglement!)")
```

---

## 💻 Lab Part 3: Data Science Application (1.5 hours)

```python
# ============================================
# Principal Component Analysis (PCA)
# ============================================

from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Center the data
X_centered = X - X.mean(axis=0)

# Covariance matrix
C = X_centered.T @ X_centered / (X.shape[0] - 1)

# Eigendecomposition (spectral theorem!)
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Sort by decreasing eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("=== PCA via Eigendecomposition ===")
print(f"Eigenvalues: {eigenvalues}")
print(f"Variance explained: {eigenvalues / eigenvalues.sum() * 100}")

# Project to 2D
W = eigenvectors[:, :2]
X_pca = X_centered @ W

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i in range(3):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               label=iris.target_names[i], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Iris Dataset - PCA')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(4), eigenvalues / eigenvalues.sum() * 100)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.title('Scree Plot')
plt.xticks(range(4), ['PC1', 'PC2', 'PC3', 'PC4'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_iris.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

---

## 📝 Lab Exercises

1. **Implement shifted inverse iteration** to find eigenvalue closest to a given σ.

2. **Extend quantum simulator** to 3 qubits with Toffoli gate.

3. **Implement singular value decomposition** using eigenvalue methods.

4. **Quantum phase estimation:** Simulate the algorithm for a 2×2 unitary.

---

## ✅ Lab Checklist

- [ ] Implement power method
- [ ] Implement QR algorithm
- [ ] Build quantum simulator
- [ ] Create Bell state
- [ ] Verify entanglement
- [ ] Apply PCA to real data
- [ ] Complete at least 2 exercises

---

*"Computation is the new microscope."*
— Stephen Wolfram
