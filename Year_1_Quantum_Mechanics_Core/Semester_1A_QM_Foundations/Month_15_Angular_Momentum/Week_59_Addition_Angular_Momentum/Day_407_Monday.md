# Day 407: Two Angular Momenta - The Tensor Product Space

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Total angular momentum and tensor products |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving: Constructing composite spaces |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab: Tensor product implementation |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you will be able to:

1. Define the total angular momentum operator for two subsystems
2. Construct the tensor product Hilbert space for composite systems
3. Calculate the dimension of the combined state space
4. Prove that the total angular momentum satisfies SU(2) commutation relations
5. Identify compatible observables in the composite system
6. Connect tensor products to two-qubit quantum computing systems

---

## Core Content

### 1. The Physical Problem

When dealing with composite quantum systems, we encounter situations where two angular momenta must be combined:

- **Electron in an atom**: Orbital angular momentum L and spin S combine to give total J
- **Two electrons**: Individual spins S_1 and S_2 combine to give total spin S
- **Two particles**: Angular momenta J_1 and J_2 of separate particles combine
- **Nucleus + electron**: Nuclear spin I and electron J give total F (hyperfine)

The mathematical question is: Given two quantum systems with angular momenta J_1 and J_2, how do we describe the states and operators of the combined system?

### 2. The Total Angular Momentum Operator

For two angular momentum systems, the **total angular momentum** is defined as:

$$\boxed{\hat{\mathbf{J}} = \hat{\mathbf{J}}_1 + \hat{\mathbf{J}}_2}$$

More explicitly, for each component:

$$\hat{J}_x = \hat{J}_{1x} + \hat{J}_{2x}$$
$$\hat{J}_y = \hat{J}_{1y} + \hat{J}_{2y}$$
$$\hat{J}_z = \hat{J}_{1z} + \hat{J}_{2z}$$

#### Key Observation

The operators J_1 and J_2 act on **different** Hilbert spaces:
- J_1 acts on H_1 (space of system 1)
- J_2 acts on H_2 (space of system 2)

They automatically commute because they refer to independent degrees of freedom:

$$[\hat{J}_{1i}, \hat{J}_{2j}] = 0 \quad \text{for all } i, j \in \{x, y, z\}$$

### 3. The Tensor Product Hilbert Space

The combined system lives in the **tensor product** of the individual Hilbert spaces:

$$\boxed{\mathcal{H} = \mathcal{H}_1 \otimes \mathcal{H}_2}$$

#### Dimension of the Composite Space

If system 1 has angular momentum j_1 and system 2 has angular momentum j_2:
- dim(H_1) = 2j_1 + 1
- dim(H_2) = 2j_2 + 1

The dimension of the composite space is:

$$\boxed{\dim(\mathcal{H}) = (2j_1 + 1)(2j_2 + 1)}$$

#### Product States

A **product state** (separable state) has the form:

$$|\psi\rangle = |j_1, m_1\rangle \otimes |j_2, m_2\rangle \equiv |j_1, m_1; j_2, m_2\rangle$$

The notation |j_1, m_1; j_2, m_2> is standard shorthand for the tensor product.

#### General States

A general state in H is a **linear combination** of product states:

$$|\Psi\rangle = \sum_{m_1=-j_1}^{j_1} \sum_{m_2=-j_2}^{j_2} c_{m_1,m_2} |j_1, m_1; j_2, m_2\rangle$$

States that cannot be written as a single product are called **entangled**.

### 4. Operators on the Tensor Product Space

When an operator acts only on one subsystem, we extend it to the full space:

$$\hat{J}_{1i} \to \hat{J}_{1i} \otimes \hat{I}_2$$
$$\hat{J}_{2i} \to \hat{I}_1 \otimes \hat{J}_{2i}$$

For notational simplicity, we often suppress the identity operators:

$$\hat{J}_{1i}|j_1, m_1; j_2, m_2\rangle = (\hat{J}_{1i}|j_1, m_1\rangle) \otimes |j_2, m_2\rangle$$

### 5. Proof: Total J Satisfies SU(2) Algebra

We must verify that the total angular momentum components satisfy the standard commutation relations.

**Theorem:** $[\hat{J}_i, \hat{J}_j] = i\hbar\varepsilon_{ijk}\hat{J}_k$

**Proof:**

$$[\hat{J}_i, \hat{J}_j] = [(\hat{J}_{1i} + \hat{J}_{2i}), (\hat{J}_{1j} + \hat{J}_{2j})]$$

Expanding:

$$= [\hat{J}_{1i}, \hat{J}_{1j}] + [\hat{J}_{1i}, \hat{J}_{2j}] + [\hat{J}_{2i}, \hat{J}_{1j}] + [\hat{J}_{2i}, \hat{J}_{2j}]$$

Since operators from different subsystems commute:

$$[\hat{J}_{1i}, \hat{J}_{2j}] = 0 \quad \text{and} \quad [\hat{J}_{2i}, \hat{J}_{1j}] = 0$$

Using the individual SU(2) algebras:

$$[\hat{J}_{1i}, \hat{J}_{1j}] = i\hbar\varepsilon_{ijk}\hat{J}_{1k}$$
$$[\hat{J}_{2i}, \hat{J}_{2j}] = i\hbar\varepsilon_{ijk}\hat{J}_{2k}$$

Therefore:

$$[\hat{J}_i, \hat{J}_j] = i\hbar\varepsilon_{ijk}(\hat{J}_{1k} + \hat{J}_{2k}) = i\hbar\varepsilon_{ijk}\hat{J}_k \quad \blacksquare$$

### 6. The Total Angular Momentum Squared

The operator J^2 is defined as:

$$\hat{J}^2 = \hat{J}_x^2 + \hat{J}_y^2 + \hat{J}_z^2$$

Expanding in terms of individual operators:

$$\hat{J}^2 = (\hat{\mathbf{J}}_1 + \hat{\mathbf{J}}_2) \cdot (\hat{\mathbf{J}}_1 + \hat{\mathbf{J}}_2)$$

$$\boxed{\hat{J}^2 = \hat{J}_1^2 + \hat{J}_2^2 + 2\hat{\mathbf{J}}_1 \cdot \hat{\mathbf{J}}_2}$$

where the dot product is:

$$\hat{\mathbf{J}}_1 \cdot \hat{\mathbf{J}}_2 = \hat{J}_{1x}\hat{J}_{2x} + \hat{J}_{1y}\hat{J}_{2y} + \hat{J}_{1z}\hat{J}_{2z}$$

#### Alternative Form Using Ladder Operators

Recall the ladder operators:

$$\hat{J}_{\pm} = \hat{J}_x \pm i\hat{J}_y$$

The dot product can be rewritten:

$$\hat{\mathbf{J}}_1 \cdot \hat{\mathbf{J}}_2 = \hat{J}_{1z}\hat{J}_{2z} + \frac{1}{2}(\hat{J}_{1+}\hat{J}_{2-} + \hat{J}_{1-}\hat{J}_{2+})$$

This form is extremely useful for calculations!

### 7. Compatible Observables

An important question: Which operators commute and can be simultaneously diagonalized?

#### Always Compatible

These operators always commute with everything relevant:

$$[\hat{J}_1^2, \hat{J}_2^2] = 0$$
$$[\hat{J}_1^2, \hat{J}^2] = 0$$
$$[\hat{J}_2^2, \hat{J}^2] = 0$$

The eigenvalues j_1 and j_2 are always good quantum numbers.

#### Two Choices for Additional Quantum Numbers

**Option 1: Uncoupled Basis**

Compatible set: {J_1^2, J_2^2, J_{1z}, J_{2z}}

$$[\hat{J}_{1z}, \hat{J}_{2z}] = 0$$

States: |j_1, m_1; j_2, m_2> with quantum numbers j_1, m_1, j_2, m_2

**Option 2: Coupled Basis**

Compatible set: {J_1^2, J_2^2, J^2, J_z}

$$[\hat{J}^2, \hat{J}_z] = 0$$

States: |j, m; j_1, j_2> with quantum numbers j, m, j_1, j_2

#### The Incompatibility

Crucially, J^2 does NOT commute with J_{1z} or J_{2z}:

$$[\hat{J}^2, \hat{J}_{1z}] \neq 0 \quad \text{(in general)}$$

This means we cannot simultaneously specify j, m, m_1, m_2. We must choose!

### 8. Dimension Counting: A Preview of Allowed j Values

The coupled and uncoupled bases must have the same dimension (they span the same space).

**Uncoupled basis dimension:** $(2j_1+1)(2j_2+1)$

**Coupled basis:** We need the sum over allowed j values to give the same dimension:

$$\sum_{j=j_{\min}}^{j_{\max}} (2j+1) = (2j_1+1)(2j_2+1)$$

This is satisfied by:

$$\boxed{j_{\min} = |j_1 - j_2|, \quad j_{\max} = j_1 + j_2}$$

with j taking all values in integer steps from j_min to j_max.

**Example:** j_1 = 1, j_2 = 1/2

- Uncoupled dimension: 3 x 2 = 6
- Allowed j values: j = 1/2, 3/2
- Coupled dimension: 2 + 4 = 6 (check!)

---

## Quantum Computing Connection

### Two-Qubit Systems

The tensor product structure is fundamental to quantum computing. For two qubits:

$$\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B = \mathbb{C}^2 \otimes \mathbb{C}^2 = \mathbb{C}^4$$

Each qubit is a spin-1/2 system, so:
- j_1 = j_2 = 1/2
- dim(H) = 2 x 2 = 4

The computational basis states are product states:

$$|00\rangle = |\uparrow\uparrow\rangle, \quad |01\rangle = |\uparrow\downarrow\rangle, \quad |10\rangle = |\downarrow\uparrow\rangle, \quad |11\rangle = |\downarrow\downarrow\rangle$$

### Entangled States

The Bell states are maximally entangled:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$

$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$

$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

These will connect to singlet and triplet states on Day 411!

### Two-Qubit Gates

Gates like CNOT act on the tensor product space:

$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

Understanding tensor products is essential for quantum circuit design.

---

## Worked Examples

### Example 1: Dimension of Combined Space

**Problem:** Two particles have angular momenta j_1 = 2 and j_2 = 1.
(a) What is the dimension of the combined Hilbert space?
(b) What are the allowed values of total j?
(c) Verify the dimension counting.

**Solution:**

(a) Dimension of composite space:
$$\dim(\mathcal{H}) = (2j_1+1)(2j_2+1) = (2 \cdot 2 + 1)(2 \cdot 1 + 1) = 5 \times 3 = 15$$

(b) Allowed j values using triangle rule:
$$|j_1 - j_2| \leq j \leq j_1 + j_2$$
$$|2 - 1| \leq j \leq 2 + 1$$
$$1 \leq j \leq 3$$

So j = 1, 2, 3 (integer steps).

(c) Dimension check:
$$\sum_j (2j+1) = (2 \cdot 1 + 1) + (2 \cdot 2 + 1) + (2 \cdot 3 + 1) = 3 + 5 + 7 = 15 \quad \checkmark$$

---

### Example 2: Action of Total J_z

**Problem:** For state |1, 1; 1/2, -1/2>, calculate J_z|state>.

**Solution:**

$$\hat{J}_z = \hat{J}_{1z} + \hat{J}_{2z}$$

Acting on the product state:

$$\hat{J}_z|1, 1; 1/2, -1/2\rangle = (\hat{J}_{1z} + \hat{J}_{2z})|1, 1\rangle \otimes |1/2, -1/2\rangle$$

$$= \hat{J}_{1z}|1, 1\rangle \otimes |1/2, -1/2\rangle + |1, 1\rangle \otimes \hat{J}_{2z}|1/2, -1/2\rangle$$

Using J_z|j,m> = m*hbar |j,m>:

$$= \hbar(1)|1, 1; 1/2, -1/2\rangle + \hbar(-1/2)|1, 1; 1/2, -1/2\rangle$$

$$= \hbar(1 - 1/2)|1, 1; 1/2, -1/2\rangle$$

$$\boxed{\hat{J}_z|1, 1; 1/2, -1/2\rangle = \frac{\hbar}{2}|1, 1; 1/2, -1/2\rangle}$$

The state has m = m_1 + m_2 = 1 - 1/2 = 1/2.

---

### Example 3: J^2 in Terms of Individual Operators

**Problem:** Show that J^2 can be written using the useful identity involving J_1.J_2.

**Solution:**

Starting from:
$$\hat{J}^2 = \hat{J}_1^2 + \hat{J}_2^2 + 2\hat{\mathbf{J}}_1 \cdot \hat{\mathbf{J}}_2$$

We can solve for the dot product:

$$\hat{\mathbf{J}}_1 \cdot \hat{\mathbf{J}}_2 = \frac{1}{2}(\hat{J}^2 - \hat{J}_1^2 - \hat{J}_2^2)$$

In the coupled basis |j, m; j_1, j_2>, all these operators are diagonal:

$$\langle \hat{\mathbf{J}}_1 \cdot \hat{\mathbf{J}}_2 \rangle = \frac{\hbar^2}{2}[j(j+1) - j_1(j_1+1) - j_2(j_2+1)]$$

For j_1 = j_2 = 1/2 and j = 0 (singlet):
$$\langle \hat{\mathbf{S}}_1 \cdot \hat{\mathbf{S}}_2 \rangle = \frac{\hbar^2}{2}[0 - 3/4 - 3/4] = -\frac{3\hbar^2}{4}$$

For j = 1 (triplet):
$$\langle \hat{\mathbf{S}}_1 \cdot \hat{\mathbf{S}}_2 \rangle = \frac{\hbar^2}{2}[2 - 3/4 - 3/4] = \frac{\hbar^2}{4}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Dimension calculation:** For j_1 = 3/2 and j_2 = 1/2, find (a) the dimension of the tensor product space, and (b) the allowed values of total j.

2. **Quantum number counting:** The uncoupled state |2, 1; 1, -1> has what value of total m?

3. **Commutator verification:** Show explicitly that [J_{1x}, J_{2y}] = 0 by considering their action on a product state.

4. **Triangle rule:** List all allowed j values for j_1 = 5/2, j_2 = 3/2.

### Level 2: Intermediate

5. **Operator identity:** Prove that [J^2, J_{1z} + J_{2z}] = 0 but [J^2, J_{1z}] may not equal zero.

6. **Ladder operators:** Express J_1.J_2 in terms of J_{1z}, J_{2z}, J_{1+}, J_{1-}, J_{2+}, J_{2-}.

7. **Dimension verification:** For general j_1 and j_2, verify algebraically that:
$$\sum_{j=|j_1-j_2|}^{j_1+j_2} (2j+1) = (2j_1+1)(2j_2+1)$$

8. **Three angular momenta:** If you have three angular momenta j_1, j_2, j_3, what is the dimension of the full tensor product space?

### Level 3: Challenging

9. **Explicit matrix:** For j_1 = 1 and j_2 = 1/2, write out the 6x6 matrix representation of J_z in the uncoupled basis.

10. **J^2 action:** Calculate J^2|1, 1; 1/2, 1/2> and show it is NOT an eigenstate of J^2.

11. **Entanglement criterion:** Prove that a product state |j_1, m_1> tensor |j_2, m_2> cannot be an eigenstate of J^2 with j different from j_max = j_1 + j_2 or j_min = |j_1 - j_2|.

12. **Connection to permutation symmetry:** For two identical particles with j_1 = j_2 = j, show that the dimension formula gives the correct count for symmetric and antisymmetric states.

---

## Computational Lab

### Exercise 1: Building the Tensor Product Space

```python
"""
Day 407 Computational Lab: Tensor Product Construction
Build the tensor product Hilbert space for two angular momentum systems
"""

import numpy as np
from itertools import product

def get_basis_states(j):
    """
    Generate all |j, m> states for given j
    Returns list of (j, m) tuples
    """
    m_values = np.arange(-j, j + 1, 1)
    return [(j, m) for m in m_values]

def tensor_product_basis(j1, j2):
    """
    Construct the uncoupled basis for two angular momenta

    Parameters:
    -----------
    j1 : float
        Angular momentum of first system
    j2 : float
        Angular momentum of second system

    Returns:
    --------
    basis : list of tuples
        Each element is (j1, m1, j2, m2)
    """
    states_1 = get_basis_states(j1)
    states_2 = get_basis_states(j2)

    basis = []
    for (j1, m1), (j2, m2) in product(states_1, states_2):
        basis.append((j1, m1, j2, m2))

    return basis

def count_coupled_states(j1, j2):
    """
    Count states in coupled basis using triangle rule
    """
    j_min = abs(j1 - j2)
    j_max = j1 + j2

    # j takes values j_min, j_min+1, ..., j_max
    j_values = np.arange(j_min, j_max + 1, 1)

    total = 0
    breakdown = {}
    for j in j_values:
        count = int(2*j + 1)
        breakdown[j] = count
        total += count

    return j_values, breakdown, total

# Example: j1 = 1, j2 = 1/2
print("="*60)
print("Tensor Product Space Construction")
print("="*60)

j1, j2 = 1, 0.5
print(f"\nSystem 1: j₁ = {j1}")
print(f"System 2: j₂ = {j2}")

# Build uncoupled basis
uncoupled_basis = tensor_product_basis(j1, j2)
print(f"\nUncoupled basis dimension: {len(uncoupled_basis)}")
print("\nUncoupled basis states |j₁, m₁; j₂, m₂⟩:")
for state in uncoupled_basis:
    j1, m1, j2, m2 = state
    print(f"  |{j1}, {m1:+.1f}; {j2}, {m2:+.1f}⟩  →  m = {m1+m2:+.1f}")

# Count coupled states
j_values, breakdown, total = count_coupled_states(j1, j2)
print(f"\nCoupled basis:")
print(f"  Allowed j values: {[float(j) for j in j_values]}")
print(f"  States per j:")
for j, count in breakdown.items():
    print(f"    j = {j}: {count} states")
print(f"  Total: {total}")

# Verify dimensions match
print(f"\nDimension check: {len(uncoupled_basis)} = {total} ✓" if len(uncoupled_basis) == total else "Mismatch!")
```

### Exercise 2: Angular Momentum Operators in Matrix Form

```python
"""
Build matrix representations of angular momentum operators
on the tensor product space
"""

import numpy as np

def jz_matrix(j):
    """
    Matrix representation of J_z in |j, m> basis
    Diagonal matrix with eigenvalues m*hbar (hbar = 1)
    """
    dim = int(2*j + 1)
    m_values = np.arange(j, -j-1, -1)  # j, j-1, ..., -j
    return np.diag(m_values)

def jp_matrix(j):
    """
    Matrix representation of J_+ (raising operator)
    J_+|j,m⟩ = sqrt(j(j+1) - m(m+1)) |j, m+1⟩
    """
    dim = int(2*j + 1)
    m_values = np.arange(j, -j-1, -1)

    Jp = np.zeros((dim, dim))
    for i in range(dim - 1):
        m = m_values[i + 1]  # m value of state being raised
        coeff = np.sqrt(j*(j+1) - m*(m+1))
        Jp[i, i+1] = coeff

    return Jp

def jm_matrix(j):
    """
    Matrix representation of J_- (lowering operator)
    """
    return jp_matrix(j).T

def jx_matrix(j):
    """J_x = (J_+ + J_-)/2"""
    return (jp_matrix(j) + jm_matrix(j)) / 2

def jy_matrix(j):
    """J_y = (J_+ - J_-)/(2i)"""
    return (jp_matrix(j) - jm_matrix(j)) / (2j)

def j_squared_matrix(j):
    """J² = j(j+1) * I in this subspace"""
    dim = int(2*j + 1)
    return j * (j + 1) * np.eye(dim)

def tensor_product_operator(A1, A2):
    """
    Construct A1 ⊗ A2 for the tensor product space
    """
    return np.kron(A1, A2)

# Build operators for j1 = 1, j2 = 1/2
j1, j2 = 1, 0.5
dim1, dim2 = int(2*j1 + 1), int(2*j2 + 1)
dim_total = dim1 * dim2

print("="*60)
print("Angular Momentum Operators on Tensor Product Space")
print("="*60)
print(f"j₁ = {j1}, j₂ = {j2}")
print(f"Total dimension: {dim_total}")

# J_1z and J_2z
J1z = tensor_product_operator(jz_matrix(j1), np.eye(dim2))
J2z = tensor_product_operator(np.eye(dim1), jz_matrix(j2))

# Total J_z = J_1z + J_2z
Jz_total = J1z + J2z

print(f"\nJ_z (total) matrix:")
print(Jz_total)

# Eigenvalues of J_z (should be m values)
eigenvalues_Jz = np.diag(Jz_total)
print(f"\nJ_z eigenvalues: {eigenvalues_Jz}")

# Verify: eigenvalues should be m1 + m2
print("\nExpected m = m₁ + m₂ values:")
for m1 in [1, 0, -1]:
    for m2 in [0.5, -0.5]:
        print(f"  m₁={m1:+}, m₂={m2:+.1f} → m={m1+m2:+.1f}")
```

### Exercise 3: Verifying Commutation Relations

```python
"""
Verify that total J satisfies SU(2) algebra
[J_i, J_j] = i * epsilon_ijk * J_k
"""

import numpy as np

def commutator(A, B):
    """Compute [A, B] = AB - BA"""
    return A @ B - B @ A

# Use j1 = 1, j2 = 1/2 as example
j1, j2 = 1, 0.5
dim1, dim2 = int(2*j1 + 1), int(2*j2 + 1)

# Build total angular momentum operators
I1, I2 = np.eye(dim1), np.eye(dim2)

J1x = tensor_product_operator(jx_matrix(j1), I2)
J1y = tensor_product_operator(jy_matrix(j1), I2)
J1z = tensor_product_operator(jz_matrix(j1), I2)

J2x = tensor_product_operator(I1, jx_matrix(j2))
J2y = tensor_product_operator(I1, jy_matrix(j2))
J2z = tensor_product_operator(I1, jz_matrix(j2))

# Total J components
Jx = J1x + J2x
Jy = J1y + J2y
Jz = J1z + J2z

print("="*60)
print("Verification of SU(2) Commutation Relations")
print("="*60)

# Test [J_x, J_y] = i*J_z
comm_xy = commutator(Jx, Jy)
expected = 1j * Jz

print("\n[J_x, J_y] = i*J_z ?")
print(f"Max difference: {np.max(np.abs(comm_xy - expected)):.2e}")
print("✓ Verified!" if np.allclose(comm_xy, expected) else "✗ Failed!")

# Test [J_y, J_z] = i*J_x
comm_yz = commutator(Jy, Jz)
expected = 1j * Jx

print("\n[J_y, J_z] = i*J_x ?")
print(f"Max difference: {np.max(np.abs(comm_yz - expected)):.2e}")
print("✓ Verified!" if np.allclose(comm_yz, expected) else "✗ Failed!")

# Test [J_z, J_x] = i*J_y
comm_zx = commutator(Jz, Jx)
expected = 1j * Jy

print("\n[J_z, J_x] = i*J_y ?")
print(f"Max difference: {np.max(np.abs(comm_zx - expected)):.2e}")
print("✓ Verified!" if np.allclose(comm_zx, expected) else "✗ Failed!")

# Verify J1 and J2 commute
print("\n[J_{1i}, J_{2j}] = 0 ?")
for (name1, op1), (name2, op2) in [
    (("J1x", J1x), ("J2x", J2x)),
    (("J1x", J1x), ("J2y", J2y)),
    (("J1y", J1y), ("J2z", J2z)),
]:
    comm = commutator(op1, op2)
    print(f"  [{name1}, {name2}] = 0: {np.allclose(comm, 0)}")

print("\nAll commutation relations verified!")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Total angular momentum | $\hat{\mathbf{J}} = \hat{\mathbf{J}}_1 + \hat{\mathbf{J}}_2$ |
| Tensor product space | $\mathcal{H} = \mathcal{H}_1 \otimes \mathcal{H}_2$ |
| Dimension | $\dim(\mathcal{H}) = (2j_1+1)(2j_2+1)$ |
| Total J^2 | $\hat{J}^2 = \hat{J}_1^2 + \hat{J}_2^2 + 2\hat{\mathbf{J}}_1 \cdot \hat{\mathbf{J}}_2$ |
| Allowed j values | $\|j_1 - j_2\| \leq j \leq j_1 + j_2$ |
| Uncoupled basis state | $\|j_1, m_1; j_2, m_2\rangle = \|j_1, m_1\rangle \otimes \|j_2, m_2\rangle$ |
| Total m value | $m = m_1 + m_2$ |

### Main Takeaways

1. **Tensor product structure**: Composite quantum systems live in the tensor product of individual Hilbert spaces.

2. **Total J satisfies SU(2)**: The sum J_1 + J_2 automatically satisfies angular momentum commutation relations.

3. **Two basis choices**: We can use uncoupled (m_1, m_2 good) or coupled (j, m good) bases.

4. **Dimension counting**: Both bases have the same dimension, verified by the triangle rule sum.

5. **Operators extend**: Single-system operators extend to the tensor product via A tensor I or I tensor A.

6. **Quantum computing connection**: Two qubits form a j_1 = j_2 = 1/2 tensor product system.

---

## Daily Checklist

- [ ] I can define the total angular momentum J = J_1 + J_2
- [ ] I understand the tensor product H = H_1 tensor H_2
- [ ] I can calculate the dimension (2j_1+1)(2j_2+1)
- [ ] I proved that total J satisfies SU(2) commutation relations
- [ ] I know which quantum numbers are good in each basis
- [ ] I can apply the triangle rule to find allowed j values
- [ ] I understand the connection to two-qubit systems
- [ ] I completed the computational exercises

---

## Preview: Day 408

Tomorrow we dive deep into the **coupled versus uncoupled bases**:

- Uncoupled basis: |j_1, m_1; j_2, m_2> - product states
- Coupled basis: |j, m; j_1, j_2> - eigenstates of J^2, J_z
- Both are complete orthonormal bases
- The unitary transformation connecting them involves Clebsch-Gordan coefficients

We will see explicitly how these two bases are related and why each is useful for different physical problems.

---

*Day 407 of QSE Self-Study Curriculum*
*Week 59: Addition of Angular Momenta*
*Month 15: Angular Momentum and Spin*
