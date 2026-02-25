# Day 115: Unitary Operators — Quantum Gates and Time Evolution

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Unitary Operators |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define unitary operators and their matrix representation
2. Prove that unitary operators preserve inner products and norms
3. Show that eigenvalues of unitary operators lie on the unit circle
4. Understand the connection between Hermitian and unitary operators
5. Apply unitary operators to quantum gates and time evolution
6. Work with common quantum gates computationally

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 7.C**: Normal Operators and Spectral Theorem (relevant portions)
- Focus on: Unitary operators, preservation of norms

### Physics Texts
- **Shankar, Chapter 4.1**: Time evolution operator
- **Nielsen & Chuang, Chapter 2**: Quantum gates (preview)

---

## 📖 Core Content: Theory and Concepts

### 1. Definition of Unitary Operators

**Definition:** A linear operator U: V → V on an inner product space is **unitary** if:

$$\boxed{U^*U = UU^* = I}$$

Equivalently: U* = U⁻¹

**For matrices:** U is unitary iff U*U = I (columns form orthonormal set)

### 2. Equivalent Characterizations

**Theorem:** The following are equivalent for a linear operator U:

1. U is unitary (U*U = I)
2. U preserves inner products: ⟨Uv, Uw⟩ = ⟨v, w⟩ for all v, w
3. U preserves norms: ||Uv|| = ||v|| for all v
4. U maps orthonormal bases to orthonormal bases
5. Columns of U (in any orthonormal basis) form an orthonormal set
6. Rows of U form an orthonormal set

**Proof (1 ⟹ 2):**
⟨Uv, Uw⟩ = ⟨v, U*Uw⟩ = ⟨v, Iw⟩ = ⟨v, w⟩ ∎

**Proof (2 ⟹ 3):**
||Uv||² = ⟨Uv, Uv⟩ = ⟨v, v⟩ = ||v||² ∎

### 3. Eigenvalue Structure

**Theorem:** All eigenvalues of a unitary operator have absolute value 1 (lie on the unit circle).

**Proof:** Let Uv = λv with v ≠ 0.
$$||v|| = ||Uv|| = ||\lambda v|| = |\lambda| \cdot ||v||$$

Since ||v|| > 0, we have |λ| = 1. ∎

**Corollary:** Eigenvalues can be written as λ = e^(iθ) for some real θ.

### 4. Examples of Unitary Matrices

#### Example 1: Rotation in ℝ²
$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

Eigenvalues: e^(±iθ) (on unit circle!)

#### Example 2: Hadamard Gate
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

H*H = I ✓, eigenvalues: ±1

#### Example 3: Phase Gate
$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$

Eigenvalues: 1, i (both |λ| = 1)

#### Example 4: Pauli X, Y, Z
All Pauli matrices are both Hermitian AND unitary!
- σₓ² = σᵧ² = σᵤ² = I ⟹ unitary
- σₓ = σₓ*, etc. ⟹ Hermitian

### 5. The Unitary-Hermitian Connection

**Theorem:** If H is Hermitian, then U = e^(iH) is unitary.

**Proof:**
$$U^* = (e^{iH})^* = e^{-iH^*} = e^{-iH}$$
$$U^*U = e^{-iH}e^{iH} = e^{0} = I$$ ∎

**Conversely:** Every unitary U near I can be written as U = e^(iH) for some Hermitian H.

### 6. Unitary Groups

**U(n):** The group of n×n unitary matrices
- Closed under multiplication
- Contains identity
- Every element has inverse (U⁻¹ = U*)

**SU(n):** Special unitary group (det U = 1)
- Important in physics: SU(2) for spin, SU(3) for quarks

**Properties:**
- dim(U(n)) = n² (as a real manifold)
- dim(SU(n)) = n² - 1

### 7. Unitary Diagonalization

**Theorem:** A matrix A is unitarily diagonalizable (A = UDU*) iff A is **normal** (A*A = AA*).

**Corollary:** Both Hermitian and unitary matrices are unitarily diagonalizable.

---

## 🔬 Quantum Mechanics Connection

### Time Evolution

The time evolution of a quantum state is governed by:
$$|\psi(t)\rangle = U(t)|\psi(0)\rangle$$

where U(t) = e^(-iHt/ℏ) is the **time evolution operator**.

**Properties:**
- U(t) is unitary (preserves probability: ||ψ(t)|| = 1)
- U(t₁)U(t₂) = U(t₁ + t₂) (group property)
- U(0) = I
- dU/dt = -iH/ℏ · U (Schrödinger equation for operators)

### Quantum Gates

Quantum computation uses unitary operators as "gates":

| Gate | Matrix | Action |
|------|--------|--------|
| X (NOT) | [[0,1],[1,0]] | Bit flip: \|0⟩↔\|1⟩ |
| Z | [[1,0],[0,-1]] | Phase flip: \|1⟩→-\|1⟩ |
| H | [[1,1],[1,-1]]/√2 | Creates superposition |
| S | [[1,0],[0,i]] | π/2 phase |
| T | [[1,0],[0,e^(iπ/4)]] | π/4 phase |
| CNOT | 4×4 | Controlled-NOT |

### Why Unitary?

1. **Probability conservation:** ||Uψ|| = ||ψ|| = 1
2. **Reversibility:** U⁻¹ = U* always exists
3. **Information preservation:** No information lost
4. **Orthogonality preservation:** Distinguishable states remain distinguishable

### Two-Qubit Gates

**CNOT (Controlled-NOT):**
$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

Acts as: |c, t⟩ → |c, c⊕t⟩ (XOR the target with control)

---

## ✏️ Worked Examples

### Example 1: Verify Hadamard is Unitary

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Check H*H = I:**
$$H^* = H \text{ (real and symmetric)}$$
$$H^*H = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I \quad ✓$$

### Example 2: Time Evolution

Hamiltonian: H = ωσᵤ/2 (spin in magnetic field)

Time evolution: U(t) = e^(-iHt/ℏ) = e^(-iωσᵤt/2ℏ)

Using e^(iθσᵤ) = cos(θ)I + i·sin(θ)σᵤ:
$$U(t) = \cos(\omega t/2)I - i\sin(\omega t/2)\sigma_z = \begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}$$

**Action on |↑⟩:**
$$U(t)|↑\rangle = e^{-i\omega t/2}|↑\rangle$$

The state acquires a phase but probability is unchanged!

### Example 3: Eigenvalues of Rotation

$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Characteristic polynomial:**
$$\det(R_\theta - \lambda I) = (\cos\theta - \lambda)^2 + \sin^2\theta = \lambda^2 - 2\cos\theta \cdot \lambda + 1$$

**Eigenvalues:**
$$\lambda = \frac{2\cos\theta \pm \sqrt{4\cos^2\theta - 4}}{2} = \cos\theta \pm i\sin\theta = e^{\pm i\theta}$$

Both have |λ| = 1 ✓

---

## 📝 Practice Problems

### Level 1: Basic
1. Verify that S = [[1,0],[0,i]] is unitary.

2. Show that the product of two unitary matrices is unitary.

3. Compute H|0⟩ and H|1⟩ for the Hadamard gate.

### Level 2: Theory
4. Prove that if U is unitary, then det(U) has absolute value 1.

5. Show that the eigenspaces of a unitary operator are orthogonal.

6. Prove: U is unitary iff U maps every orthonormal basis to an orthonormal basis.

### Level 3: Applications
7. Find the matrix exponential e^(iπσₓ/2) and verify it's unitary.

8. Show that any 2×2 unitary matrix can be written as e^(iα)R for some rotation R.

9. Verify that CNOT is unitary and find its eigenvalues.

### Level 4: Quantum
10. A spin-1/2 particle in magnetic field B = Bẑ has H = -γBσᵤ/2. If |ψ(0)⟩ = |→⟩ = (|↑⟩+|↓⟩)/√2, find |ψ(t)⟩.

11. Show that any quantum gate can be approximated by combinations of H, T, and CNOT.

12. Prove that the set of 2×2 unitary matrices forms a group under multiplication.

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)

# ============================================
# Lab 1: Basic Unitary Gates
# ============================================

# Define standard gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)  # NOT
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
S = np.array([[1, 0], [0, 1j]], dtype=complex)  # Phase
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

def is_unitary(U, tol=1e-10):
    """Check if matrix is unitary"""
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=tol)

print("=== Unitary Gate Verification ===")
gates = {'I': I, 'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T}
for name, gate in gates.items():
    print(f"{name}: unitary = {is_unitary(gate)}")

# ============================================
# Lab 2: Eigenvalues on Unit Circle
# ============================================

print("\n=== Eigenvalues of Unitary Gates ===")
for name, gate in gates.items():
    eigvals = np.linalg.eigvals(gate)
    magnitudes = np.abs(eigvals)
    print(f"{name}: eigenvalues = {eigvals}, |λ| = {magnitudes}")

# ============================================
# Lab 3: Gate Actions on States
# ============================================

# Basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

print("\n=== Gate Actions ===")
print(f"X|0⟩ = {X @ ket_0}")  # Should be |1⟩
print(f"X|1⟩ = {X @ ket_1}")  # Should be |0⟩
print(f"H|0⟩ = {H @ ket_0}")  # Should be |+⟩
print(f"H|1⟩ = {H @ ket_1}")  # Should be |-⟩

# ============================================
# Lab 4: Time Evolution
# ============================================

def time_evolve(H, psi_0, t):
    """Evolve state under Hamiltonian H for time t"""
    U = expm(-1j * H * t)
    return U @ psi_0

# Rabi oscillations: H = σx
H_rabi = X
psi_0 = ket_0

t_values = np.linspace(0, 2*np.pi, 100)
probs_0 = []

for t in t_values:
    psi_t = time_evolve(H_rabi, psi_0, t)
    probs_0.append(np.abs(psi_t[0])**2)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(t_values, probs_0, 'b-', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('P(|0⟩)')
plt.title('Rabi Oscillations: H = σx, initial state |0⟩')
plt.grid(True, alpha=0.3)
plt.savefig('rabi_unitary.png', dpi=150)
plt.show()

# ============================================
# Lab 5: Two-Qubit Gates
# ============================================

def kron(A, B):
    return np.kron(A, B)

# CNOT gate
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

print("\n=== Two-Qubit Gates ===")
print(f"CNOT is unitary: {is_unitary(CNOT)}")
print(f"CNOT eigenvalues: {np.linalg.eigvals(CNOT)}")

# Create Bell state
ket_00 = np.array([1, 0, 0, 0], dtype=complex)
H_I = kron(H, I)  # Hadamard on first qubit

psi = H_I @ ket_00  # |+0⟩ = (|00⟩ + |10⟩)/√2
bell = CNOT @ psi   # (|00⟩ + |11⟩)/√2

print(f"\nBell state creation:")
print(f"|00⟩ → H⊗I → {psi}")
print(f"→ CNOT → {bell}")
print(f"Expected: (|00⟩ + |11⟩)/√2 = {np.array([1,0,0,1])/np.sqrt(2)}")

# ============================================
# Lab 6: Quantum Circuit Simulation
# ============================================

class QuantumCircuit:
    """Simple quantum circuit simulator"""
    
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1  # Initialize to |00...0⟩
    
    def apply_single(self, gate, qubit):
        """Apply single-qubit gate to specified qubit"""
        # Build full operator using tensor products
        ops = [I] * self.n
        ops[qubit] = gate
        full_op = ops[0]
        for op in ops[1:]:
            full_op = kron(full_op, op)
        self.state = full_op @ self.state
    
    def apply_cnot(self, control, target):
        """Apply CNOT with specified control and target"""
        # For simplicity, assume 2 qubits and control=0, target=1
        self.state = CNOT @ self.state
    
    def measure_probs(self):
        return np.abs(self.state)**2
    
    def __repr__(self):
        return f"State: {self.state}"

# Test circuit
qc = QuantumCircuit(2)
print("\n=== Quantum Circuit Simulation ===")
print(f"Initial: {qc}")
qc.apply_single(H, 0)
print(f"After H on qubit 0: {qc}")
qc.apply_cnot(0, 1)
print(f"After CNOT: {qc}")
print(f"Probabilities: {qc.measure_probs()}")

# ============================================
# Lab 7: Decomposition into Basic Gates
# ============================================

# Any single-qubit unitary can be written as:
# U = e^(iα) Rz(β) Ry(γ) Rz(δ)

def Rx(theta):
    """Rotation around X axis"""
    return expm(-1j * theta/2 * X)

def Ry(theta):
    """Rotation around Y axis"""
    return expm(-1j * theta/2 * Y)

def Rz(theta):
    """Rotation around Z axis"""
    return expm(-1j * theta/2 * Z)

print("\n=== Rotation Gates ===")
print(f"Rx(π) = {Rx(np.pi)}")
print(f"Should equal -iX: {-1j * X}")
print(f"Match: {np.allclose(Rx(np.pi), -1j * X)}")

# Verify Ry(π/2) creates superposition
print(f"\nRy(π/2)|0⟩ = {Ry(np.pi/2) @ ket_0}")
print(f"Should be (|0⟩+|1⟩)/√2: {(ket_0 + ket_1)/np.sqrt(2)}")

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Understand unitary definition U*U = I
- [ ] Prove norm preservation property
- [ ] Prove eigenvalues lie on unit circle
- [ ] Understand Hermitian-unitary connection via exponential
- [ ] Know standard quantum gates
- [ ] Complete computational lab
- [ ] Build intuition for time evolution

---

## 🔜 Preview: Tomorrow's Topics

**Day 116: Spectral Theorem and Normal Operators**

Tomorrow we unify Hermitian and unitary:
- Normal operators (A*A = AA*)
- Complete spectral theorem
- Functional calculus
- **QM Connection:** General observables and measurements

---

*"Unitary operators are the choreographers of quantum mechanics — they move states around while preserving the dance floor's geometry."*
