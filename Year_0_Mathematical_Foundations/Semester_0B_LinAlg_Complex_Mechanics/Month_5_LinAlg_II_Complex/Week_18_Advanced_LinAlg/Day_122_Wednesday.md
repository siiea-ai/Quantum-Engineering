# Day 122: Tensor Products — Building Composite Quantum Systems

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Tensor Product Foundations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define the tensor product of vector spaces
2. Compute tensor products of vectors using the Kronecker product
3. Understand the tensor product of operators
4. Work with multi-qubit computational bases
5. Distinguish between product states and entangled states
6. Apply tensor products to construct quantum gates for multi-qubit systems

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 9**: Multilinear Algebra (tensor products)

### Physics Texts
- **Nielsen & Chuang, Section 2.1.7**: Tensor products
- **Shankar, Chapter 10.1**: Multiple particles

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: Combining Quantum Systems

**Problem:** How do we describe the state of two particles?

**Classical:** State of particle 1 + State of particle 2 = 2 × (individual states)
**Quantum:** State space of combined system = Tensor product of individual spaces

If dim(ℋ₁) = d₁ and dim(ℋ₂) = d₂, then:
$$\dim(\mathcal{H}_1 \otimes \mathcal{H}_2) = d_1 \times d_2$$

**Key insight:** Combined system has MORE states than just product states!

### 2. Definition of Tensor Product (Vector Spaces)

**Definition:** Given vector spaces V and W over field 𝔽, the **tensor product** V ⊗ W is a vector space together with a bilinear map ⊗: V × W → V ⊗ W satisfying:

For v, v' ∈ V, w, w' ∈ W, and c ∈ 𝔽:
1. (v + v') ⊗ w = v ⊗ w + v' ⊗ w (linear in first argument)
2. v ⊗ (w + w') = v ⊗ w + v ⊗ w' (linear in second argument)
3. c(v ⊗ w) = (cv) ⊗ w = v ⊗ (cw) (scalar factoring)

**Universal property:** Any bilinear map factors uniquely through ⊗.

### 3. Tensor Product of Vectors

For v ∈ V with basis {eᵢ} and w ∈ W with basis {fⱼ}:

If v = Σᵢ vᵢeᵢ and w = Σⱼ wⱼfⱼ, then:
$$v \otimes w = \sum_{i,j} v_i w_j (e_i \otimes f_j)$$

**Basis for V ⊗ W:** {eᵢ ⊗ fⱼ}

**Dimension formula:** dim(V ⊗ W) = dim(V) × dim(W)

### 4. The Kronecker Product

**Matrix representation of ⊗:**

For vectors:
$$|v\rangle \otimes |w\rangle = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_m \end{pmatrix} \otimes \begin{pmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{pmatrix} = \begin{pmatrix} v_1 w_1 \\ v_1 w_2 \\ \vdots \\ v_1 w_n \\ v_2 w_1 \\ \vdots \\ v_m w_n \end{pmatrix}$$

For matrices A (m×n) and B (p×q):
$$A \otimes B = \begin{pmatrix} a_{11}B & a_{12}B & \cdots & a_{1n}B \\ a_{21}B & a_{22}B & \cdots & a_{2n}B \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1}B & a_{m2}B & \cdots & a_{mn}B \end{pmatrix}$$

Result is (mp) × (nq) matrix.

### 5. Key Kronecker Product Properties

| Property | Formula |
|----------|---------|
| Associativity | (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C) |
| Bilinearity | A ⊗ (B + C) = A ⊗ B + A ⊗ C |
| Mixed product | (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD) |
| Transpose | (A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ |
| Conjugate | (A ⊗ B)* = A* ⊗ B* |
| Inverse | (A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹ |
| Trace | tr(A ⊗ B) = tr(A) × tr(B) |
| Determinant | det(A ⊗ B) = det(A)ᵐ det(B)ⁿ |

**Critical formula (mixed product rule):**
$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

### 6. Tensor Product of Operators

For operators T: V → V' and S: W → W':

$$\boxed{(T \otimes S)(v \otimes w) = T(v) \otimes S(w)}$$

Extended by linearity to all of V ⊗ W.

**Matrix representation:** If T has matrix A and S has matrix B, then T ⊗ S has matrix A ⊗ B.

### 7. The Computational Basis

**Two qubits:** ℋ = ℂ² ⊗ ℂ² = ℂ⁴

Computational basis:
$$|00\rangle = |0\rangle \otimes |0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

$$|01\rangle = |0\rangle \otimes |1\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}$$

$$|10\rangle = |1\rangle \otimes |0\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}$$

$$|11\rangle = |1\rangle \otimes |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

**n qubits:** Computational basis {|i⟩} for i = 0, 1, ..., 2ⁿ-1 (binary representation)

### 8. Product States vs Entangled States

**Product state:** Can be written as |ψ⟩ = |a⟩ ⊗ |b⟩

Example: |+0⟩ = |+⟩ ⊗ |0⟩ = (|00⟩ + |10⟩)/√2

**Entangled state:** CANNOT be written as a product!

Example: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

**How to tell?**
- Product state ⟺ Schmidt rank = 1
- Entangled ⟺ Schmidt rank > 1

---

## 🔬 Quantum Mechanics Connection

### Multi-Qubit Quantum Gates

**Single-qubit gates on multi-qubit systems:**

Hadamard on qubit 1 of 2-qubit system:
$$H \otimes I = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}$$

**Two-qubit gates:**

CNOT (Controlled-NOT):
$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

CNOT flips the target qubit if control is |1⟩.

### Creating Entanglement

**Bell state circuit:**
1. Start with |00⟩
2. Apply H to first qubit: |00⟩ → (|0⟩+|1⟩)|0⟩/√2 = (|00⟩+|10⟩)/√2
3. Apply CNOT: → (|00⟩+|11⟩)/√2 = |Φ⁺⟩

**Key insight:** Entanglement requires a two-qubit gate like CNOT!

### Quantum Registers

**n-qubit register:** ℋ = (ℂ²)^⊗n = ℂ²ⁿ

State: |ψ⟩ = Σᵢ cᵢ |i⟩ where i ranges from 0 to 2ⁿ-1

**Exponential growth:** 50 qubits → 2⁵⁰ ≈ 10¹⁵ complex amplitudes!

This is why quantum computers can potentially solve problems classical computers cannot.

### Operators on Subsystems

**Local operators:** A ⊗ I acts only on first subsystem

**Expectation value:**
$$\langle A \otimes B \rangle = \langle \psi | (A \otimes B) | \psi \rangle$$

For product states |ψ⟩ = |a⟩ ⊗ |b⟩:
$$\langle A \otimes B \rangle = \langle a|A|a\rangle \cdot \langle b|B|b\rangle$$

For entangled states, this factorization fails!

---

## ✏️ Worked Examples

### Example 1: Kronecker Product of Vectors

Compute |0⟩ ⊗ |+⟩ where |+⟩ = (|0⟩+|1⟩)/√2.

$$|0\rangle \otimes |+\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \cdot 1 \\ 1 \cdot 1 \\ 0 \cdot 1 \\ 0 \cdot 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \\ 0 \\ 0 \end{pmatrix}$$

In Dirac notation: |0+⟩ = (|00⟩ + |01⟩)/√2

### Example 2: Kronecker Product of Matrices

Compute σₓ ⊗ σ_z.

$$\sigma_x \otimes \sigma_z = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

$$= \begin{pmatrix} 0 \cdot \sigma_z & 1 \cdot \sigma_z \\ 1 \cdot \sigma_z & 0 \cdot \sigma_z \end{pmatrix} = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \\ 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \end{pmatrix}$$

### Example 3: Mixed Product Rule

Verify (A ⊗ B)(v ⊗ w) = (Av) ⊗ (Bw).

Let A = σₓ, B = σ_z, v = |0⟩, w = |1⟩.

**Direct calculation:**
$$\sigma_x|0\rangle = |1\rangle, \quad \sigma_z|1\rangle = -|1\rangle$$
$$(Av) \otimes (Bw) = |1\rangle \otimes (-|1\rangle) = -|11\rangle$$

**Using tensor product:**
$$(\sigma_x \otimes \sigma_z)(|0\rangle \otimes |1\rangle) = (\sigma_x \otimes \sigma_z)|01\rangle$$

From the matrix in Example 2: column 2 (for |01⟩) is (0, 0, 0, -1)ᵀ = -|11⟩ ✓

### Example 4: Creating Bell State

Apply (H ⊗ I)·CNOT to |00⟩.

**Step 1:** (H ⊗ I)|00⟩
$$H|0\rangle = |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
$$|00\rangle \xrightarrow{H \otimes I} |+0\rangle = \frac{|00\rangle + |10\rangle}{\sqrt{2}}$$

**Step 2:** CNOT|+0⟩
$$\text{CNOT}|00\rangle = |00\rangle, \quad \text{CNOT}|10\rangle = |11\rangle$$
$$|+0\rangle \xrightarrow{\text{CNOT}} \frac{|00\rangle + |11\rangle}{\sqrt{2}} = |\Phi^+\rangle$$

### Example 5: Checking if State is Product

Is |ψ⟩ = (|00⟩ + |01⟩ + |10⟩)/√3 a product state?

**Test:** Can we write |ψ⟩ = (a|0⟩+b|1⟩) ⊗ (c|0⟩+d|1⟩)?

Expanding: ac|00⟩ + ad|01⟩ + bc|10⟩ + bd|11⟩

Comparing coefficients:
- ac = 1/√3
- ad = 1/√3
- bc = 1/√3
- bd = 0

From bd = 0: either b = 0 or d = 0.
If b = 0: bc = 0 ≠ 1/√3. Contradiction!
If d = 0: ad = 0 ≠ 1/√3. Contradiction!

**Conclusion:** |ψ⟩ is entangled!

---

## 📝 Practice Problems

### Level 1: Kronecker Products
1. Compute |1⟩ ⊗ |0⟩ and |0⟩ ⊗ |1⟩. Are they equal?

2. Compute σᵧ ⊗ σᵧ.

3. Verify tr(A ⊗ B) = tr(A)·tr(B) for A = σₓ, B = σ_z.

### Level 2: Quantum States
4. Write |+−⟩ = |+⟩ ⊗ |−⟩ in the computational basis.

5. Is |ψ⟩ = (|00⟩ + |11⟩ + |22⟩)/√3 (qutrit system) a product state?

6. Compute (σₓ ⊗ I)|Φ⁺⟩ where |Φ⁺⟩ = (|00⟩+|11⟩)/√2.

### Level 3: Operators
7. Show that (A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹.

8. Construct the SWAP gate matrix (exchanges two qubits).

9. Verify CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X by computing the matrix.

### Level 4: Theory
10. Prove: If |ψ⟩ is a product state, then the reduced density matrix is pure.

11. Show that controlled-U gate is CU = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U.

12. Prove the dimension formula: dim(V ⊗ W) = dim(V) × dim(W).

---

## 💻 Evening Computational Lab

```python
import numpy as np
from itertools import product as cartesian_product

np.set_printoptions(precision=4, suppress=True)

# ============================================
# Basic Kronecker Product Operations
# ============================================

def kron_vec(v, w):
    """Kronecker product of two vectors"""
    return np.kron(v, w)

def kron_mat(A, B):
    """Kronecker product of two matrices"""
    return np.kron(A, B)

# Standard quantum states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

print("=== Kronecker Products of Vectors ===")
print(f"|00⟩ = {kron_vec(ket_0, ket_0)}")
print(f"|01⟩ = {kron_vec(ket_0, ket_1)}")
print(f"|10⟩ = {kron_vec(ket_1, ket_0)}")
print(f"|11⟩ = {kron_vec(ket_1, ket_1)}")

print(f"\n|0+⟩ = |0⟩⊗|+⟩ = {kron_vec(ket_0, ket_plus)}")

# ============================================
# Two-Qubit Computational Basis
# ============================================

def computational_basis(n_qubits):
    """Generate computational basis for n qubits"""
    dim = 2**n_qubits
    basis = {}
    for i in range(dim):
        binary = format(i, f'0{n_qubits}b')
        vec = np.zeros(dim, dtype=complex)
        vec[i] = 1
        basis[binary] = vec
    return basis

basis_2q = computational_basis(2)
print("\n=== Two-Qubit Computational Basis ===")
for label, vec in basis_2q.items():
    print(f"|{label}⟩ = {vec}")

# ============================================
# Quantum Gates
# ============================================

# CNOT gate (control on first qubit)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# SWAP gate
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# Controlled-Z
CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)

print("\n=== Two-Qubit Gates ===")
print(f"CNOT =\n{CNOT}")
print(f"\nSWAP =\n{SWAP}")

# ============================================
# Creating Bell States
# ============================================

def create_bell_state(initial="00"):
    """Create Bell state from computational basis state"""
    # Start with |00⟩, |01⟩, |10⟩, or |11⟩
    psi = basis_2q[initial].copy()
    
    # Apply H ⊗ I
    H_I = kron_mat(H, I)
    psi = H_I @ psi
    
    # Apply CNOT
    psi = CNOT @ psi
    
    return psi

bell_states = {
    "Φ+": create_bell_state("00"),
    "Ψ+": create_bell_state("01"),
    "Φ-": create_bell_state("10"),
    "Ψ-": create_bell_state("11")
}

print("\n=== Bell States ===")
for name, state in bell_states.items():
    print(f"|{name}⟩ = {state}")

# ============================================
# Product vs Entangled States
# ============================================

def is_product_state(psi, dim_A=2, dim_B=2, tol=1e-10):
    """
    Check if bipartite state is a product state.
    Uses Schmidt decomposition (SVD).
    """
    C = psi.reshape(dim_A, dim_B)
    _, s, _ = np.linalg.svd(C)
    # Product state ⟺ only one non-zero singular value
    n_nonzero = np.sum(s > tol)
    return n_nonzero == 1

print("\n=== Product vs Entangled ===")
test_states = [
    ("|00⟩", basis_2q["00"]),
    ("|+0⟩", kron_vec(ket_plus, ket_0)),
    ("|Φ+⟩", bell_states["Φ+"]),
    ("(|00⟩+|01⟩+|10⟩)/√3", (basis_2q["00"] + basis_2q["01"] + basis_2q["10"])/np.sqrt(3))
]

for name, state in test_states:
    is_prod = is_product_state(state)
    status = "Product" if is_prod else "ENTANGLED"
    print(f"{name}: {status}")

# ============================================
# Tensor Product of Operators
# ============================================

print("\n=== Tensor Products of Operators ===")

# X ⊗ Z
X_Z = kron_mat(X, Z)
print(f"σx ⊗ σz =\n{X_Z}")

# Verify mixed product rule: (A⊗B)(v⊗w) = (Av)⊗(Bw)
v, w = ket_0, ket_1
Av = X @ v
Bw = Z @ w

lhs = X_Z @ kron_vec(v, w)
rhs = kron_vec(Av, Bw)

print(f"\n(σx⊗σz)|01⟩ = {lhs}")
print(f"(σx|0⟩)⊗(σz|1⟩) = {rhs}")
print(f"Equal: {np.allclose(lhs, rhs)}")

# ============================================
# Multi-Qubit Operations
# ============================================

def apply_gate_to_qubit(gate, target_qubit, n_qubits):
    """
    Apply single-qubit gate to specific qubit in n-qubit system.
    target_qubit: 0-indexed from left (most significant)
    """
    ops = [I] * n_qubits
    ops[target_qubit] = gate
    
    result = ops[0]
    for op in ops[1:]:
        result = kron_mat(result, op)
    
    return result

# Example: H on qubit 1 of 3-qubit system
H_q1_3qubits = apply_gate_to_qubit(H, 1, 3)
print(f"\n=== H on qubit 1 of 3-qubit system ===")
print(f"Shape: {H_q1_3qubits.shape}")

# ============================================
# GHZ State (3-qubit entanglement)
# ============================================

print("\n=== Creating GHZ State ===")
# |GHZ⟩ = (|000⟩ + |111⟩)/√2

# Start with |000⟩
psi = np.zeros(8, dtype=complex)
psi[0] = 1

# H on first qubit
H_I_I = kron_mat(kron_mat(H, I), I)
psi = H_I_I @ psi
print(f"After H⊗I⊗I: {psi}")

# CNOT(0,1)
CNOT_I = kron_mat(CNOT, I)
psi = CNOT_I @ psi
print(f"After CNOT(0,1): {psi}")

# CNOT(0,2) - need to construct this carefully
# CNOT with control 0, target 2
CNOT_02 = np.zeros((8, 8), dtype=complex)
for i in range(8):
    bits = list(format(i, '03b'))
    if bits[0] == '1':
        bits[2] = '1' if bits[2] == '0' else '0'
    j = int(''.join(bits), 2)
    CNOT_02[j, i] = 1

psi = CNOT_02 @ psi
print(f"After CNOT(0,2): {psi}")
print("Expected: (|000⟩ + |111⟩)/√2")

# ============================================
# Trace Properties
# ============================================

print("\n=== Trace Property: tr(A⊗B) = tr(A)·tr(B) ===")
A = np.array([[1, 2], [3, 4]], dtype=complex)
B = np.array([[5, 6], [7, 8]], dtype=complex)

tr_AB = np.trace(kron_mat(A, B))
tr_A_tr_B = np.trace(A) * np.trace(B)

print(f"tr(A⊗B) = {tr_AB}")
print(f"tr(A)·tr(B) = {tr_A_tr_B}")
print(f"Equal: {np.isclose(tr_AB, tr_A_tr_B)}")
```

---

## ✅ Daily Checklist

- [ ] Understand tensor product definition
- [ ] Compute Kronecker products of vectors and matrices
- [ ] Know the computational basis for multi-qubit systems
- [ ] Apply the mixed product rule
- [ ] Distinguish product states from entangled states
- [ ] Construct multi-qubit gates
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 🔜 Preview: Tomorrow

**Day 123: Composite Quantum Systems — Entanglement and Correlations**
- Reduced density matrices
- Partial trace operation
- Quantifying entanglement
- Bell inequalities preview
- QM Connection: Non-local correlations

---

*"The tensor product is how quantum mechanics builds big worlds from small ones."*
— Quantum Information Proverb
