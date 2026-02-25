# Day 965: Chemistry and Materials Simulation

## Schedule Overview

| Time Block | Duration | Focus |
|------------|----------|-------|
| Morning | 3.5 hours | Second quantization and fermion-qubit mappings |
| Afternoon | 2.5 hours | VQE for molecular ground states |
| Evening | 1 hour | Computational lab: H2 simulation |

## Learning Objectives

By the end of today, you will be able to:

1. Express electronic structure Hamiltonians in second quantization
2. Apply Jordan-Wigner transformation to map fermions to qubits
3. Analyze the qubit requirements for molecular simulation
4. Implement VQE with the UCCSD ansatz for small molecules
5. Understand resource estimates for practical chemistry applications
6. Identify promising applications in drug discovery and materials science

## Core Content

### 1. The Promise of Quantum Chemistry

Quantum chemistry is the "killer app" for quantum computers. The problem is clear:

**The Electronic Structure Problem:**
Given a molecular geometry, find the ground state energy:

$$E_0 = \min_\psi \langle\psi|H|\psi\rangle$$

**Why It Matters:**
- Drug discovery: Predict binding affinities
- Catalyst design: Optimize reaction pathways
- Materials science: Design new superconductors, batteries
- Fertilizer production: Understand nitrogen fixation

**Classical Limitations:**
- Full configuration interaction (FCI): Exact but $O(M!)$ scaling
- Coupled cluster (CCSD(T)): Gold standard but $O(N^7)$
- Density functional theory: Approximate, fails for strong correlation

---

### 2. The Electronic Structure Hamiltonian

For $N$ electrons in $M$ spin-orbitals:

$$\boxed{H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s}$$

**Terms:**
- **One-body terms** $h_{pq}$: Kinetic energy + nuclear attraction
- **Two-body terms** $h_{pqrs}$: Electron-electron repulsion

**Integrals (computed classically):**

$$h_{pq} = \int \phi_p^*(\mathbf{r}) \left(-\frac{\nabla^2}{2} - \sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}\right) \phi_q(\mathbf{r}) d\mathbf{r}$$

$$h_{pqrs} = \int \frac{\phi_p^*(\mathbf{r}_1)\phi_q^*(\mathbf{r}_2)\phi_r(\mathbf{r}_2)\phi_s(\mathbf{r}_1)}{|\mathbf{r}_1 - \mathbf{r}_2|} d\mathbf{r}_1 d\mathbf{r}_2$$

---

### 3. Fermionic Operators

Fermionic creation and annihilation operators satisfy:

$$\boxed{\{a_p, a_q^\dagger\} = \delta_{pq}, \quad \{a_p, a_q\} = \{a_p^\dagger, a_q^\dagger\} = 0}$$

where $\{A, B\} = AB + BA$ is the anticommutator.

**Interpretation:**
- $a_p^\dagger$: Creates an electron in orbital $p$
- $a_p$: Destroys an electron in orbital $p$
- $a_p^\dagger a_p = n_p$: Number operator (0 or 1)

**Key property:** Fermionic antisymmetry is encoded in the anticommutation relations.

---

### 4. Jordan-Wigner Transformation

The **Jordan-Wigner (JW) transformation** maps fermionic operators to Pauli operators:

$$\boxed{a_p^\dagger = \frac{1}{2}(X_p - iY_p) \prod_{q<p} Z_q}$$

$$\boxed{a_p = \frac{1}{2}(X_p + iY_p) \prod_{q<p} Z_q}$$

**The Z-string:** The product $\prod_{q<p} Z_q$ enforces fermionic antisymmetry.

**Example for 4 orbitals:**
- $a_0^\dagger = \frac{1}{2}(X_0 - iY_0)$ (no Z-string)
- $a_1^\dagger = \frac{1}{2}(X_1 - iY_1) Z_0$
- $a_2^\dagger = \frac{1}{2}(X_2 - iY_2) Z_1 Z_0$
- $a_3^\dagger = \frac{1}{2}(X_3 - iY_3) Z_2 Z_1 Z_0$

---

### 5. Qubit Hamiltonian Structure

After Jordan-Wigner transformation, the molecular Hamiltonian becomes:

$$H = \sum_k h_k P_k$$

where each $P_k$ is a Pauli string.

**Scaling:**
- Number of qubits: $M$ (number of spin-orbitals)
- Number of terms: $O(M^4)$ for general two-body interactions
- 1-norm: $\lambda = \sum_k |h_k| = O(M^4)$ typically

**Example: H2 in minimal basis (STO-3G)**
- 2 electrons, 4 spin-orbitals → 4 qubits
- Symmetry reduction → 2 qubits possible
- ~15 Pauli terms in Hamiltonian

---

### 6. Alternative Fermion Mappings

**Bravyi-Kitaev Transformation:**

Reduces Z-string length from $O(M)$ to $O(\log M)$:

$$a_p^\dagger \to \text{(Pauli string of length } O(\log M)\text{)}$$

**Advantages:**
- Lower weight Pauli strings → fewer CNOT gates
- Better for simulation algorithms

**Parity Mapping:**
- Encodes parity of occupation in qubits
- Also achieves $O(\log M)$ string length

---

### 7. Variational Quantum Eigensolver (VQE)

VQE is the leading NISQ approach for chemistry:

$$\boxed{E(\vec{\theta}) = \langle\psi(\vec{\theta})|H|\psi(\vec{\theta})\rangle}$$

**Algorithm:**
1. Prepare parameterized state $|\psi(\vec{\theta})\rangle$ on quantum computer
2. Measure $\langle H \rangle$ by sampling Pauli terms
3. Update $\vec{\theta}$ using classical optimizer
4. Repeat until convergence

**The UCCSD Ansatz:**

Unitary Coupled Cluster with Singles and Doubles:

$$|\psi_{\text{UCCSD}}\rangle = e^{T - T^\dagger} |\phi_0\rangle$$

where:
$$T = \sum_{i,a} t_i^a a_a^\dagger a_i + \sum_{i<j, a<b} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i$$

- $i, j$: Occupied orbitals
- $a, b$: Virtual orbitals
- $t$: Variational parameters

---

### 8. Resource Estimates for Chemistry

**Current estimates for fault-tolerant chemistry:**

| Molecule | Qubits | T-gates | Wall time |
|----------|--------|---------|-----------|
| FeMoCo (nitrogenase) | ~2,000-4,000 | $10^{10}$-$10^{12}$ | Days-weeks |
| Cytochrome P450 | ~1,000 | $10^9$ | Hours |
| Small catalyst | ~100-200 | $10^7$ | Minutes |

**NISQ limitations:**
- Circuit depth: 100-1000 layers
- Qubits: 50-100 (near-term)
- Molecules: H2, LiH, BeH2, H2O (small basis)

---

### 9. Applications and Impact

**Near-term (NISQ) opportunities:**
- Benchmark molecules: H2, LiH, BeH2
- Variational algorithms: VQE, ADAPT-VQE
- Hybrid classical-quantum methods

**Fault-tolerant era:**
- Nitrogen fixation (FeMoCo active site)
- Drug binding predictions
- Novel high-Tc superconductors
- Battery electrolyte design

**Economic impact:**
- Fertilizer industry: $100B+ annually
- Drug discovery: $1T+ market
- Energy storage: Critical for renewable transition

---

## Worked Examples

### Example 1: H2 Hamiltonian Construction

**Problem:** Derive the qubit Hamiltonian for H2 in minimal basis.

**Solution:**

Step 1: Identify the basis.
STO-3G basis: 2 spatial orbitals → 4 spin-orbitals:
- $\phi_0 = 1s_\alpha$ (spin up on orbital 1)
- $\phi_1 = 1s_\beta$ (spin down on orbital 1)
- $\phi_2 = 1s^*_\alpha$ (spin up on orbital 2)
- $\phi_3 = 1s^*_\beta$ (spin down on orbital 2)

Step 2: Write the general form.
$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$$

Step 3: Apply Jordan-Wigner.
After transformation (using standard integrals at equilibrium ~0.74 A):

$$H = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_2 + g_4 Z_3$$
$$+ g_5 Z_0 Z_1 + g_6 Z_0 Z_2 + g_7 Z_1 Z_2 + g_8 Z_0 Z_3 + g_9 Z_1 Z_3 + g_{10} Z_2 Z_3$$
$$+ g_{11} X_0 X_1 Y_2 Y_3 + g_{12} Y_0 Y_1 X_2 X_3 + g_{13} X_0 Y_1 Y_2 X_3 + g_{14} Y_0 X_1 X_2 Y_3$$

Typical values (Hartree):
- $g_0 \approx -0.81$
- Other coefficients: $O(0.1)$ to $O(0.01)$

Step 4: Symmetry reduction.
Using particle number and spin symmetry:
- Reduce from 4 qubits to 2 qubits
- Hamiltonian simplifies to ~5 terms

$\square$

---

### Example 2: UCCSD Gate Count

**Problem:** Estimate the gate count for UCCSD on H2.

**Solution:**

Step 1: Count excitation operators.
For H2 with 2 electrons in 4 spin-orbitals:
- Singles: $t_0^2, t_0^3, t_1^2, t_1^3$ (4 parameters, but symmetry reduces to 2)
- Doubles: $t_{01}^{23}$ (1 parameter)

Step 2: Implement each excitation.
Each excitation $e^{i\theta(a^\dagger_a a_i - a_i^\dagger a_a)}$ requires:
- Trotter decomposition into Pauli exponentials
- Each Pauli exponential: O(1) CNOTs + rotations

Step 3: Circuit structure.
For $T_1$ (singles):
- 2-4 CNOT gates per excitation
- Total: ~10 CNOTs for singles

For $T_2$ (doubles):
- 8-16 CNOT gates per double excitation
- Total: ~15 CNOTs for doubles

Step 4: Total UCCSD circuit.
- ~25-30 CNOT gates
- ~10-15 parameterized rotations
- Circuit depth: ~40-50

$\square$

---

### Example 3: VQE Measurement Strategy

**Problem:** How many measurements are needed to estimate $\langle H \rangle$ to precision $\epsilon$ for H2?

**Solution:**

Step 1: Identify Pauli terms.
H2 has ~15 Pauli terms (before grouping).

Step 2: Variance per term.
For a Pauli operator $P$, variance is bounded:
$$\text{Var}(\langle P \rangle) \leq 1$$

Step 3: Grouping strategy.
Commuting Paulis can be measured simultaneously.
For H2, terms group into ~5 measurement bases.

Step 4: Shot budget.
To achieve precision $\epsilon$ in $\langle H \rangle$:

$$N_{\text{shots}} \approx \frac{\lambda^2}{\epsilon^2}$$

where $\lambda = \sum_k |h_k| \approx 1$ Hartree for H2.

For chemical accuracy ($\epsilon = 1.6$ mHartree = 0.0016 Ha):
$$N_{\text{shots}} \approx \frac{1}{(0.0016)^2} \approx 4 \times 10^5 \text{ per basis}$$

Total: $\sim 2 \times 10^6$ measurements.

$\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Jordan-Wigner:** Apply JW transformation to $a_2^\dagger a_1$ for a 4-qubit system.

2. **Qubit count:** How many qubits are needed to simulate water (H2O) in a minimal basis with 7 spatial orbitals?

3. **Pauli terms:** For a molecule with $M = 10$ spin-orbitals, estimate the number of Pauli terms in the Jordan-Wigner Hamiltonian.

### Level 2: Intermediate Analysis

4. **Bravyi-Kitaev:** Compare the Z-string length for $a_{15}^\dagger$ under Jordan-Wigner vs. Bravyi-Kitaev transformations.

5. **UCCSD scaling:** For a molecule with $n$ electrons in $M$ spin-orbitals, how many singles and doubles parameters are there?

6. **Measurement grouping:** Show that $Z_0 Z_1$ and $Z_0 Z_2$ can be measured in the same basis, but $Z_0 Z_1$ and $X_0 X_1$ cannot.

### Level 3: Challenging Problems

7. **Active space:** Explain how active space selection reduces the qubit requirement and estimate savings for a typical transition metal complex.

8. **Noise in VQE:** Analyze how depolarizing noise affects VQE energy estimates and derive a noise threshold for chemical accuracy.

9. **Resource comparison:** Compare fault-tolerant (QPE) vs. NISQ (VQE) approaches for simulating FeMoCo, considering qubits, gates, and time.

---

## Computational Lab: H2 Simulation with VQE

### Lab Objective

Implement VQE for the H2 molecule and compute the ground state energy.

```python
"""
Day 965 Lab: H2 Molecular Simulation with VQE
Week 138: Quantum Simulation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import matplotlib.pyplot as plt

# =============================================================
# Part 1: H2 Hamiltonian (Pre-computed Integrals)
# =============================================================

print("=" * 60)
print("Part 1: H2 Molecular Hamiltonian")
print("=" * 60)

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor_product(*operators):
    """Compute tensor product of multiple operators."""
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

# H2 Hamiltonian coefficients at equilibrium (R = 0.74 Angstrom)
# Simplified 2-qubit form after symmetry reduction
def h2_hamiltonian_2qubit(g: dict) -> np.ndarray:
    """
    Build the 2-qubit H2 Hamiltonian.

    H = g0*II + g1*ZI + g2*IZ + g3*ZZ + g4*XX + g5*YY
    """
    H = (g['II'] * tensor_product(I, I) +
         g['ZI'] * tensor_product(Z, I) +
         g['IZ'] * tensor_product(I, Z) +
         g['ZZ'] * tensor_product(Z, Z) +
         g['XX'] * tensor_product(X, X) +
         g['YY'] * tensor_product(Y, Y))
    return H

# Coefficients at R = 0.74 A (Hartree)
g_eq = {
    'II': -1.0523732,
    'ZI': 0.39793742,
    'IZ': -0.39793742,
    'ZZ': -0.01128010,
    'XX': 0.18093119,
    'YY': 0.18093119
}

H_eq = h2_hamiltonian_2qubit(g_eq)

print("\nH2 Hamiltonian at equilibrium (2-qubit reduced form):")
print(f"  Coefficients: {g_eq}")

# Exact ground state energy
eigvals = np.linalg.eigvalsh(H_eq)
E_exact = eigvals[0]
print(f"\nExact ground state energy: {E_exact:.6f} Hartree")
print(f"  = {E_exact * 627.5:.2f} kcal/mol")

# =============================================================
# Part 2: VQE Ansatz
# =============================================================

print("\n" + "=" * 60)
print("Part 2: VQE Ansatz (Hardware-Efficient)")
print("=" * 60)

def hardware_efficient_ansatz(params: np.ndarray) -> np.ndarray:
    """
    Hardware-efficient ansatz for 2 qubits.

    Structure:
    - Ry(theta_0) on qubit 0
    - Ry(theta_1) on qubit 1
    - CNOT(0, 1)
    - Ry(theta_2) on qubit 0
    - Ry(theta_3) on qubit 1
    """
    def Ry(theta):
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)

    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    # Layer 1
    layer1 = tensor_product(Ry(params[0]), Ry(params[1]))

    # CNOT
    entangle = CNOT

    # Layer 2
    layer2 = tensor_product(Ry(params[2]), Ry(params[3]))

    # Total unitary
    U = layer2 @ entangle @ layer1

    return U

def prepare_state(params: np.ndarray) -> np.ndarray:
    """Prepare the variational state |psi(params)>."""
    U = hardware_efficient_ansatz(params)
    # Start from |00>
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    return U @ psi0

def energy_expectation(params: np.ndarray, H: np.ndarray) -> float:
    """Compute <psi(params)|H|psi(params)>."""
    psi = prepare_state(params)
    return np.real(np.vdot(psi, H @ psi))

# Test the ansatz
test_params = np.array([0.1, 0.2, 0.3, 0.4])
test_energy = energy_expectation(test_params, H_eq)
print(f"\nTest parameters: {test_params}")
print(f"Test energy: {test_energy:.6f} Hartree")
print(f"Energy gap from exact: {test_energy - E_exact:.6f} Hartree")

# =============================================================
# Part 3: VQE Optimization
# =============================================================

print("\n" + "=" * 60)
print("Part 3: VQE Optimization")
print("=" * 60)

def run_vqe(H: np.ndarray, n_params: int = 4,
            n_trials: int = 5) -> Tuple[float, np.ndarray]:
    """
    Run VQE with multiple random initializations.
    """
    best_energy = float('inf')
    best_params = None

    for trial in range(n_trials):
        # Random initial parameters
        x0 = np.random.randn(n_params) * np.pi

        # Optimize
        result = minimize(
            lambda p: energy_expectation(p, H),
            x0,
            method='COBYLA',
            options={'maxiter': 200}
        )

        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x

        print(f"  Trial {trial+1}: E = {result.fun:.6f} Hartree")

    return best_energy, best_params

print("\nRunning VQE optimization...")
E_vqe, params_vqe = run_vqe(H_eq)

print(f"\nVQE Result:")
print(f"  Energy: {E_vqe:.6f} Hartree")
print(f"  Error: {(E_vqe - E_exact)*1000:.3f} mHartree")
print(f"  Optimal parameters: {params_vqe}")

# Chemical accuracy threshold
chem_acc = 1.6e-3  # Hartree (1 kcal/mol)
achieved = abs(E_vqe - E_exact) < chem_acc
print(f"  Chemical accuracy ({chem_acc*1000:.1f} mHa): {'Achieved!' if achieved else 'Not achieved'}")

# =============================================================
# Part 4: Potential Energy Curve
# =============================================================

print("\n" + "=" * 60)
print("Part 4: H2 Potential Energy Curve")
print("=" * 60)

# Coefficients at various bond lengths (pre-computed)
def h2_coefficients(R: float) -> dict:
    """
    Get H2 Hamiltonian coefficients at bond length R (Angstrom).
    Simplified model with analytical fit.
    """
    # Nuclear repulsion
    V_nn = 1.0 / R

    # Simplified parametric model (approximate)
    decay = np.exp(-0.5 * (R - 0.74)**2)
    g = {
        'II': -1.0523732 * decay - 0.4 * (1 - decay) + V_nn,
        'ZI': 0.39793742 * decay,
        'IZ': -0.39793742 * decay,
        'ZZ': -0.01128010 * decay,
        'XX': 0.18093119 * decay,
        'YY': 0.18093119 * decay
    }
    return g

# Compute potential energy curve
R_values = np.linspace(0.3, 3.0, 30)
E_exact_curve = []
E_vqe_curve = []

print("\nComputing potential energy curve...")
for R in R_values:
    g = h2_coefficients(R)
    H = h2_hamiltonian_2qubit(g)

    # Exact
    eigvals = np.linalg.eigvalsh(H)
    E_exact_curve.append(eigvals[0])

    # VQE (single run for speed)
    result = minimize(
        lambda p: energy_expectation(p, H),
        params_vqe,  # Use previous optimal as initial guess
        method='COBYLA',
        options={'maxiter': 100}
    )
    E_vqe_curve.append(result.fun)
    params_vqe = result.x  # Update for next iteration

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(R_values, E_exact_curve, 'b-', linewidth=2, label='Exact (FCI)')
plt.plot(R_values, E_vqe_curve, 'ro', markersize=6, label='VQE')
plt.xlabel('Bond Length (Angstrom)', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('H$_2$ Potential Energy Curve', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
error = np.abs(np.array(E_vqe_curve) - np.array(E_exact_curve)) * 1000  # mHartree
plt.plot(R_values, error, 'g-', linewidth=2)
plt.axhline(y=1.6, color='r', linestyle='--', label='Chemical accuracy (1.6 mHa)')
plt.xlabel('Bond Length (Angstrom)', fontsize=12)
plt.ylabel('VQE Error (mHartree)', fontsize=12)
plt.title('VQE Error vs Bond Length', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_965_h2_pec.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================
# Part 5: UCCSD Ansatz (Simplified)
# =============================================================

print("\n" + "=" * 60)
print("Part 5: UCCSD Ansatz")
print("=" * 60)

def uccsd_operator(t: float) -> np.ndarray:
    """
    Simplified UCCSD operator for H2 (2 qubits).

    T - T† = t * (XYXY - YXYX) (double excitation only)

    This is the symmetry-adapted form.
    """
    # The double excitation operator in 2-qubit form
    XYXY = tensor_product(X, Y) @ tensor_product(X, Y)
    YXYX = tensor_product(Y, X) @ tensor_product(Y, X)

    # Actually, for 2-qubit H2, the relevant operator is:
    # T - T† = i*t*(X0 Y1 - Y0 X1)
    XY = tensor_product(X, Y)
    YX = tensor_product(Y, X)

    generator = 1j * t * (XY - YX)
    return expm(generator)

def uccsd_state(t: float) -> np.ndarray:
    """Prepare UCCSD state with parameter t."""
    U = uccsd_operator(t)
    # Reference state: |01> (one electron in each spatial orbital)
    psi0 = np.array([0, 1, 0, 0], dtype=complex)
    return U @ psi0

def uccsd_energy(t: float, H: np.ndarray) -> float:
    """Compute UCCSD energy."""
    psi = uccsd_state(t)
    return np.real(np.vdot(psi, H @ psi))

# Optimize UCCSD
print("\nOptimizing UCCSD...")
result_uccsd = minimize(
    lambda t: uccsd_energy(t[0], H_eq),
    [0.0],
    method='BFGS'
)

E_uccsd = result_uccsd.fun
t_optimal = result_uccsd.x[0]

print(f"  UCCSD Energy: {E_uccsd:.6f} Hartree")
print(f"  UCCSD Error: {(E_uccsd - E_exact)*1000:.4f} mHartree")
print(f"  Optimal t: {t_optimal:.4f}")

# =============================================================
# Part 6: Jordan-Wigner Transformation Demo
# =============================================================

print("\n" + "=" * 60)
print("Part 6: Jordan-Wigner Transformation")
print("=" * 60)

def jordan_wigner_creation(p: int, n_qubits: int) -> np.ndarray:
    """
    Construct a†_p in Jordan-Wigner representation.

    a†_p = (X_p - iY_p)/2 * Z_{p-1} * ... * Z_0
    """
    dim = 2**n_qubits

    # Build the Z-string
    z_string = np.eye(dim, dtype=complex)
    for q in range(p):
        z_op = np.eye(1)
        for j in range(n_qubits):
            if j == q:
                z_op = np.kron(z_op, Z)
            else:
                z_op = np.kron(z_op, I)
        z_string = z_string @ z_op

    # Build (X_p - iY_p)/2
    ladder_op = np.eye(1)
    for j in range(n_qubits):
        if j == p:
            ladder_op = np.kron(ladder_op, (X - 1j * Y) / 2)
        else:
            ladder_op = np.kron(ladder_op, I)

    return ladder_op @ z_string

def jordan_wigner_annihilation(p: int, n_qubits: int) -> np.ndarray:
    """Construct a_p in Jordan-Wigner representation."""
    return jordan_wigner_creation(p, n_qubits).conj().T

# Verify anticommutation relations
n_test = 4
print(f"\nVerifying fermionic anticommutation for {n_test} modes:")

for p in range(n_test):
    a_p = jordan_wigner_annihilation(p, n_test)
    a_p_dag = jordan_wigner_creation(p, n_test)

    # {a_p, a†_p} = 1
    anticomm = a_p @ a_p_dag + a_p_dag @ a_p
    is_identity = np.allclose(anticomm, np.eye(2**n_test))
    print(f"  {{a_{p}, a†_{p}}} = I: {is_identity}")

# Check {a_p, a_q} = 0 for p ≠ q
a_0 = jordan_wigner_annihilation(0, n_test)
a_1 = jordan_wigner_annihilation(1, n_test)
anticomm_01 = a_0 @ a_1 + a_1 @ a_0
is_zero = np.allclose(anticomm_01, np.zeros((2**n_test, 2**n_test)))
print(f"  {{a_0, a_1}} = 0: {is_zero}")

# =============================================================
# Part 7: Resource Estimates
# =============================================================

print("\n" + "=" * 60)
print("Part 7: Resource Estimates for Larger Molecules")
print("=" * 60)

def estimate_resources(n_electrons: int, n_orbitals: int) -> dict:
    """Estimate resources for molecular simulation."""
    M = 2 * n_orbitals  # Spin-orbitals (2 spins per spatial)

    # Qubits
    n_qubits = M

    # Hamiltonian terms (two-body dominant)
    n_terms = M**4 // 4  # Approximate

    # UCCSD parameters
    n_occ = n_electrons
    n_virt = M - n_electrons
    n_singles = n_occ * n_virt
    n_doubles = n_singles * (n_singles - 1) // 2

    # Circuit depth for UCCSD (approximate)
    circuit_depth = n_singles * 10 + n_doubles * 50

    return {
        'qubits': n_qubits,
        'terms': n_terms,
        'singles': n_singles,
        'doubles': n_doubles,
        'uccsd_depth': circuit_depth
    }

molecules = [
    ('H2', 2, 2),
    ('LiH', 4, 6),
    ('H2O', 10, 7),
    ('N2', 14, 10),
    ('benzene', 42, 36),
]

print("\nResource estimates:")
print("-" * 70)
print(f"{'Molecule':<12} {'Electrons':<10} {'Qubits':<8} {'Terms':<10} {'UCCSD Depth':<12}")
print("-" * 70)

for name, n_e, n_orb in molecules:
    res = estimate_resources(n_e, n_orb)
    print(f"{name:<12} {n_e:<10} {res['qubits']:<8} {res['terms']:<10} {res['uccsd_depth']:<12}")

print("\nNote: These are rough estimates. Actual resources depend on")
print("basis set, active space selection, and compilation efficiency.")

print("\nLab complete!")
print("Figure saved: day_965_h2_pec.png")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Electronic Hamiltonian | $H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$ |
| Jordan-Wigner | $a_p^\dagger = \frac{1}{2}(X_p - iY_p) \prod_{q<p} Z_q$ |
| VQE energy | $E(\vec{\theta}) = \langle\psi(\vec{\theta})|H|\psi(\vec{\theta})\rangle$ |
| UCCSD ansatz | $|\psi\rangle = e^{T - T^\dagger}|\phi_0\rangle$ |
| Qubit scaling | $M$ qubits for $M$ spin-orbitals |
| Term scaling | $O(M^4)$ Pauli terms |

### Key Takeaways

1. **Quantum chemistry** is a leading application for quantum computers.

2. **Second quantization** represents electrons using creation/annihilation operators.

3. **Jordan-Wigner** maps fermions to qubits with Z-strings encoding antisymmetry.

4. **VQE** is the primary NISQ algorithm for chemistry, using variational optimization.

5. **UCCSD** provides a physically motivated ansatz based on coupled cluster theory.

6. **Resource requirements** scale as $O(M^4)$ terms for molecular Hamiltonians.

---

## Daily Checklist

- [ ] I understand the electronic structure problem and its importance
- [ ] I can apply Jordan-Wigner transformation to simple operators
- [ ] I understand the VQE algorithm and its components
- [ ] I know the UCCSD ansatz structure
- [ ] I can estimate resources for molecular simulations
- [ ] I completed the H2 simulation lab

---

## Preview of Day 966

Tomorrow we conclude Week 138 with **Variational Quantum Simulation and Synthesis**. We will:

- Explore variational quantum simulation (VQS) for dynamics
- Learn variational imaginary time evolution (VarQITE)
- Compare near-term (NISQ) vs. fault-tolerant approaches
- Synthesize the week's learning on quantum simulation
- Discuss open problems and future directions

This synthesis day connects all the simulation methods we've learned.

---

*"The underlying physical laws necessary for the mathematical theory of a large part of physics and the whole of chemistry are thus completely known."*
*— Paul Dirac, 1929*

---

**Next:** [Day_966_Sunday.md](Day_966_Sunday.md) - Variational Quantum Simulation and Synthesis
