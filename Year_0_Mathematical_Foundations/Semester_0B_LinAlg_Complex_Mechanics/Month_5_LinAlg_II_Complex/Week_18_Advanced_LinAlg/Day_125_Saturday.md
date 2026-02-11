# Day 125: Computational Lab — Advanced Linear Algebra for Quantum Systems

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 1:00 PM | 3 hours | Lab Part 1: SVD & Tensor Products |
| Afternoon | 3:00 PM - 6:00 PM | 3 hours | Lab Part 2: Quantum System Simulation |
| Evening | 7:30 PM - 9:00 PM | 1.5 hours | Lab Part 3: Project Work |

**Total Study Time: 7.5 hours**

---

## 🎯 Lab Objectives

1. Build a comprehensive quantum simulation library
2. Implement SVD-based quantum state analysis
3. Simulate multi-qubit systems with tensor products
4. Analyze entanglement quantitatively
5. Model open quantum system dynamics
6. Create publication-quality visualizations

---

## 💻 Lab Part 1: Comprehensive Quantum Library (3 hours)

```python
"""
Advanced Quantum Simulation Library
Week 18: SVD, Tensor Products, Density Matrices
"""

import numpy as np
from scipy.linalg import expm, logm, sqrtm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product as cartesian_product

np.set_printoptions(precision=4, suppress=True)

# ============================================
# FUNDAMENTAL CONSTANTS AND MATRICES
# ============================================

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Common gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

# Standard states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# ============================================
# SVD-BASED QUANTUM ANALYSIS
# ============================================

class QuantumStateSVD:
    """Analyze quantum states using SVD"""
    
    def __init__(self, psi, dim_A, dim_B):
        """
        Initialize with bipartite state.
        psi: state vector
        dim_A, dim_B: dimensions of subsystems
        """
        self.psi = np.array(psi, dtype=complex).flatten()
        self.psi = self.psi / np.linalg.norm(self.psi)
        self.dim_A = dim_A
        self.dim_B = dim_B
        
        # Coefficient matrix
        self.C = self.psi.reshape(dim_A, dim_B)
        
        # Compute SVD (Schmidt decomposition)
        self.U, self.s, self.Vh = np.linalg.svd(self.C, full_matrices=False)
        
        # Schmidt coefficients (normalized)
        self.schmidt_coeffs = self.s
        self.schmidt_rank = np.sum(self.s > 1e-10)
    
    def is_product_state(self, tol=1e-10):
        """Check if state is a product state"""
        return self.schmidt_rank == 1
    
    def entanglement_entropy(self):
        """Compute von Neumann entropy of entanglement"""
        probs = self.schmidt_coeffs**2
        probs = probs[probs > 1e-15]
        return -np.sum(probs * np.log2(probs))
    
    def concurrence(self):
        """Compute concurrence for 2x2 systems"""
        if self.dim_A != 2 or self.dim_B != 2:
            raise ValueError("Concurrence defined only for 2x2 systems")
        
        # For pure states: C = 2|det(C)|
        return 2 * np.abs(np.linalg.det(self.C))
    
    def reduced_density_matrix_A(self):
        """Compute reduced density matrix for system A"""
        return np.diag(self.schmidt_coeffs**2)
    
    def reduced_density_matrix_B(self):
        """Compute reduced density matrix for system B"""
        return np.diag(self.schmidt_coeffs**2)
    
    def schmidt_decomposition_string(self):
        """Return human-readable Schmidt decomposition"""
        terms = []
        for i, coeff in enumerate(self.schmidt_coeffs):
            if coeff > 1e-10:
                terms.append(f"{coeff:.4f}|a_{i}⟩|b_{i}⟩")
        return " + ".join(terms)


# Test SVD analysis
print("=== SVD-Based Quantum State Analysis ===\n")

# Bell state
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
bell_svd = QuantumStateSVD(bell, 2, 2)
print(f"Bell state |Φ+⟩:")
print(f"  Schmidt coefficients: {bell_svd.schmidt_coeffs}")
print(f"  Schmidt rank: {bell_svd.schmidt_rank}")
print(f"  Product state? {bell_svd.is_product_state()}")
print(f"  Entanglement entropy: {bell_svd.entanglement_entropy():.4f} ebits")
print(f"  Concurrence: {bell_svd.concurrence():.4f}")

# Product state
product = np.kron(ket_plus, ket_0)
prod_svd = QuantumStateSVD(product, 2, 2)
print(f"\nProduct state |+0⟩:")
print(f"  Schmidt coefficients: {prod_svd.schmidt_coeffs}")
print(f"  Schmidt rank: {prod_svd.schmidt_rank}")
print(f"  Product state? {prod_svd.is_product_state()}")
print(f"  Entanglement entropy: {prod_svd.entanglement_entropy():.4f} ebits")

# Partially entangled
partial = np.array([np.sqrt(0.8), 0, 0, np.sqrt(0.2)])
part_svd = QuantumStateSVD(partial, 2, 2)
print(f"\nPartially entangled √0.8|00⟩ + √0.2|11⟩:")
print(f"  Schmidt coefficients: {part_svd.schmidt_coeffs}")
print(f"  Entanglement entropy: {part_svd.entanglement_entropy():.4f} ebits")
print(f"  Concurrence: {part_svd.concurrence():.4f}")

# ============================================
# TENSOR PRODUCT OPERATIONS
# ============================================

class TensorOperations:
    """Utilities for tensor product operations"""
    
    @staticmethod
    def kron_n(*args):
        """Kronecker product of multiple matrices"""
        result = args[0]
        for m in args[1:]:
            result = np.kron(result, m)
        return result
    
    @staticmethod
    def partial_trace(rho, dims, trace_over):
        """
        Partial trace of density matrix.
        dims: list of subsystem dimensions
        trace_over: list of indices to trace out
        """
        if isinstance(trace_over, int):
            trace_over = [trace_over]
        
        # Sort in reverse order to trace from end
        for idx in sorted(trace_over, reverse=True):
            rho = TensorOperations._partial_trace_single(rho, dims, idx)
            dims = [d for i, d in enumerate(dims) if i != idx]
        
        return rho
    
    @staticmethod
    def _partial_trace_single(rho, dims, idx):
        """Trace out single subsystem"""
        n = len(dims)
        d = dims[idx]
        
        # Reshape to tensor
        shape = dims + dims
        rho_tensor = rho.reshape(shape)
        
        # Trace
        result = np.trace(rho_tensor, axis1=idx, axis2=idx+n)
        
        # Reshape back
        new_dims = [d for i, d in enumerate(dims) if i != idx]
        new_dim = int(np.prod(new_dims))
        return result.reshape(new_dim, new_dim)
    
    @staticmethod
    def apply_local_operator(op, qubit_idx, n_qubits, state):
        """Apply single-qubit operator to specific qubit"""
        ops = [I] * n_qubits
        ops[qubit_idx] = op
        full_op = TensorOperations.kron_n(*ops)
        return full_op @ state


# Test tensor operations
print("\n=== Tensor Product Operations ===\n")

# Create 3-qubit GHZ state
ghz = np.zeros(8, dtype=complex)
ghz[0] = ghz[7] = 1/np.sqrt(2)

print(f"GHZ state: {ghz}")

# Partial traces
top = TensorOperations()
rho_ghz = np.outer(ghz, ghz.conj())

rho_AB = top.partial_trace(rho_ghz, [2, 2, 2], 2)
rho_A = top.partial_trace(rho_ghz, [2, 2, 2], [1, 2])

print(f"\nρ_AB (trace out C):\n{rho_AB}")
print(f"\nρ_A (trace out B,C):\n{rho_A}")
print(f"ρ_A eigenvalues: {np.linalg.eigvalsh(rho_A)}")

# ============================================
# DENSITY MATRIX OPERATIONS
# ============================================

class DensityMatrixOps:
    """Advanced density matrix operations"""
    
    @staticmethod
    def purity(rho):
        return np.real(np.trace(rho @ rho))
    
    @staticmethod
    def von_neumann_entropy(rho):
        eigs = np.linalg.eigvalsh(rho)
        eigs = eigs[eigs > 1e-15]
        return -np.sum(eigs * np.log2(eigs))
    
    @staticmethod
    def fidelity(rho, sigma):
        """Compute fidelity F(ρ, σ)"""
        sqrt_rho = sqrtm(rho)
        return np.real(np.trace(sqrtm(sqrt_rho @ sigma @ sqrt_rho)))**2
    
    @staticmethod
    def trace_distance(rho, sigma):
        """Compute trace distance D(ρ, σ) = ||ρ - σ||_1 / 2"""
        diff = rho - sigma
        return np.real(np.sum(np.abs(np.linalg.eigvalsh(diff)))) / 2
    
    @staticmethod
    def relative_entropy(rho, sigma):
        """Compute S(ρ||σ) = tr(ρ log ρ) - tr(ρ log σ)"""
        # Be careful with zero eigenvalues
        log_rho = logm(rho + 1e-15*np.eye(rho.shape[0]))
        log_sigma = logm(sigma + 1e-15*np.eye(sigma.shape[0]))
        return np.real(np.trace(rho @ (log_rho - log_sigma))) / np.log(2)
    
    @staticmethod
    def negativity(rho, dim_A, dim_B):
        """Compute negativity (entanglement measure for mixed states)"""
        # Partial transpose w.r.t. B
        rho_TB = DensityMatrixOps.partial_transpose(rho, dim_A, dim_B, 'B')
        eigs = np.linalg.eigvalsh(rho_TB)
        return (np.sum(np.abs(eigs)) - 1) / 2
    
    @staticmethod
    def partial_transpose(rho, dim_A, dim_B, system='B'):
        """Compute partial transpose"""
        rho_reshape = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        if system == 'B':
            rho_PT = np.transpose(rho_reshape, (0, 3, 2, 1))
        else:
            rho_PT = np.transpose(rho_reshape, (2, 1, 0, 3))
        return rho_PT.reshape(dim_A*dim_B, dim_A*dim_B)


# Test density matrix operations
print("\n=== Density Matrix Operations ===\n")

dm_ops = DensityMatrixOps()

# Compare Bell state and separable state
rho_bell = np.outer(bell, bell.conj())
rho_sep = np.diag([0.5, 0, 0, 0.5])  # Classical mixture |00⟩⟨00| + |11⟩⟨11|

print("Bell state ρ_Bell:")
print(f"  Purity: {dm_ops.purity(rho_bell):.4f}")
print(f"  Entropy: {dm_ops.von_neumann_entropy(rho_bell):.4f}")
print(f"  Negativity: {dm_ops.negativity(rho_bell, 2, 2):.4f}")

print("\nClassical mixture 0.5|00⟩⟨00| + 0.5|11⟩⟨11|:")
print(f"  Purity: {dm_ops.purity(rho_sep):.4f}")
print(f"  Entropy: {dm_ops.von_neumann_entropy(rho_sep):.4f}")
print(f"  Negativity: {dm_ops.negativity(rho_sep, 2, 2):.4f}")

print(f"\nFidelity between them: {dm_ops.fidelity(rho_bell, rho_sep):.4f}")
print(f"Trace distance: {dm_ops.trace_distance(rho_bell, rho_sep):.4f}")
```

---

## 💻 Lab Part 2: Quantum System Simulation (3 hours)

```python
# ============================================
# QUANTUM CHANNEL SIMULATION
# ============================================

class QuantumChannel:
    """General quantum channel with Kraus representation"""
    
    def __init__(self, kraus_operators, name="Channel"):
        self.kraus = [np.array(k, dtype=complex) for k in kraus_operators]
        self.name = name
        self._verify_trace_preserving()
    
    def _verify_trace_preserving(self, tol=1e-10):
        """Check Σ K†K = I"""
        total = sum(k.conj().T @ k for k in self.kraus)
        dim = self.kraus[0].shape[0]
        if not np.allclose(total, np.eye(dim), atol=tol):
            print(f"Warning: {self.name} may not be trace preserving")
    
    def apply(self, rho):
        """Apply channel to density matrix"""
        return sum(k @ rho @ k.conj().T for k in self.kraus)
    
    def compose(self, other):
        """Compose two channels: self ∘ other"""
        new_kraus = []
        for k1 in self.kraus:
            for k2 in other.kraus:
                new_kraus.append(k1 @ k2)
        return QuantumChannel(new_kraus, f"{self.name}∘{other.name}")
    
    def choi_matrix(self):
        """Compute Choi matrix (useful for channel analysis)"""
        dim = self.kraus[0].shape[0]
        # Create maximally entangled state
        max_ent = np.eye(dim).flatten() / np.sqrt(dim)
        max_ent_dm = np.outer(max_ent, max_ent.conj())
        # Apply channel to second subsystem
        # This is a simplification; full implementation needs partial application
        return sum(np.kron(np.eye(dim), k) @ max_ent_dm @ np.kron(np.eye(dim), k.conj().T) 
                   for k in self.kraus)


# Predefined channels
def depolarizing_channel(p, dim=2):
    """Depolarizing channel with probability p"""
    if dim == 2:
        coeff = np.sqrt(p/3)
        kraus = [
            np.sqrt(1-p) * I,
            coeff * X,
            coeff * Y,
            coeff * Z
        ]
    else:
        # General d-dimensional case
        coeff = np.sqrt(p/(dim**2 - 1))
        kraus = [np.sqrt(1-p) * np.eye(dim)]
        # Add generalized Pauli operators (simplified)
        for i in range(dim):
            for j in range(dim):
                if i != j or (i == j and i > 0):
                    op = np.zeros((dim, dim), dtype=complex)
                    op[i, j] = coeff
                    kraus.append(op)
    return QuantumChannel(kraus, f"Depolarizing(p={p})")

def amplitude_damping_channel(gamma):
    """Amplitude damping (T1) with decay probability gamma"""
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return QuantumChannel([K0, K1], f"AmplitudeDamping(γ={gamma})")

def phase_damping_channel(gamma):
    """Phase damping (T2 pure dephasing)"""
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]])
    return QuantumChannel([K0, K1], f"PhaseDamping(γ={gamma})")

def generalized_amplitude_damping(gamma, p):
    """Generalized amplitude damping at finite temperature"""
    K0 = np.sqrt(p) * np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.sqrt(p) * np.array([[0, np.sqrt(gamma)], [0, 0]])
    K2 = np.sqrt(1-p) * np.array([[np.sqrt(1-gamma), 0], [0, 1]])
    K3 = np.sqrt(1-p) * np.array([[0, 0], [np.sqrt(gamma), 0]])
    return QuantumChannel([K0, K1, K2, K3], f"GAD(γ={gamma},p={p})")


# Test channels
print("\n=== Quantum Channel Simulation ===\n")

rho_plus = np.outer(ket_plus, ket_plus.conj())
print(f"Initial state |+⟩: purity = {dm_ops.purity(rho_plus):.4f}")

# Apply different channels
channels = [
    depolarizing_channel(0.1),
    amplitude_damping_channel(0.1),
    phase_damping_channel(0.1)
]

for ch in channels:
    rho_out = ch.apply(rho_plus)
    print(f"{ch.name}: purity = {dm_ops.purity(rho_out):.4f}")

# ============================================
# LINDBLAD MASTER EQUATION SIMULATION
# ============================================

class LindbladSimulator:
    """Simulate open quantum system dynamics"""
    
    def __init__(self, H, L_ops, gamma_ops):
        """
        H: Hamiltonian
        L_ops: List of Lindblad (jump) operators
        gamma_ops: Corresponding decay rates
        """
        self.H = np.array(H, dtype=complex)
        self.L_ops = [np.array(L, dtype=complex) for L in L_ops]
        self.gamma = gamma_ops
        self.dim = H.shape[0]
    
    def lindblad_rhs(self, rho):
        """Compute dρ/dt = -i[H,ρ] + Σ γ (L ρ L† - {L†L, ρ}/2)"""
        # Coherent part
        drho = -1j * (self.H @ rho - rho @ self.H)
        
        # Dissipative part
        for L, g in zip(self.L_ops, self.gamma):
            Ld = L.conj().T
            LdL = Ld @ L
            drho += g * (L @ rho @ Ld - 0.5 * LdL @ rho - 0.5 * rho @ LdL)
        
        return drho
    
    def evolve(self, rho0, t_final, n_steps=100):
        """Evolve density matrix using simple Euler method"""
        dt = t_final / n_steps
        rho = np.array(rho0, dtype=complex)
        
        trajectory = [rho.copy()]
        times = [0]
        
        for i in range(n_steps):
            drho = self.lindblad_rhs(rho)
            rho = rho + dt * drho
            # Ensure valid density matrix
            rho = (rho + rho.conj().T) / 2  # Hermitianize
            rho = rho / np.trace(rho)  # Renormalize
            
            trajectory.append(rho.copy())
            times.append((i+1) * dt)
        
        return times, trajectory


# Simulate T1 decay
print("\n=== Lindblad Simulation: T1 Decay ===\n")

# Hamiltonian: ω σz/2
omega = 1.0
H_qubit = omega * Z / 2

# T1 decay
T1 = 10.0
gamma_1 = 1 / T1
L_minus = np.array([[0, 1], [0, 0]])  # Lowering operator

simulator = LindbladSimulator(H_qubit, [L_minus], [gamma_1])

# Start in excited state
rho0 = np.outer(ket_1, ket_1.conj())
times, trajectory = simulator.evolve(rho0, t_final=5*T1, n_steps=500)

# Extract populations
pop_1 = [np.real(rho[1, 1]) for rho in trajectory]
pop_0 = [np.real(rho[0, 0]) for rho in trajectory]
coherence = [np.abs(rho[0, 1]) for rho in trajectory]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(times, pop_1, 'r-', label='P(|1⟩)', linewidth=2)
axes[0].plot(times, pop_0, 'b-', label='P(|0⟩)', linewidth=2)
axes[0].plot(times, np.exp(-np.array(times)/T1), 'k--', label=f'exp(-t/T₁)', alpha=0.7)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Population')
axes[0].set_title('T₁ Decay: Population Dynamics')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Start in |+⟩ for coherence decay
rho0_plus = np.outer(ket_plus, ket_plus.conj())
_, trajectory_plus = simulator.evolve(rho0_plus, t_final=5*T1, n_steps=500)
coherence_plus = [np.abs(rho[0, 1]) for rho in trajectory_plus]

axes[1].plot(times, coherence_plus, 'g-', label='|ρ₀₁|', linewidth=2)
axes[1].plot(times, 0.5*np.exp(-np.array(times)/(2*T1)), 'k--', 
            label=f'0.5·exp(-t/2T₁)', alpha=0.7)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Coherence')
axes[1].set_title('T₁ Decay: Coherence Dynamics')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lindblad_T1_decay.png', dpi=150)
plt.show()

# ============================================
# ENTANGLEMENT DYNAMICS
# ============================================

print("\n=== Entanglement Dynamics Under Decoherence ===\n")

def simulate_entanglement_decay(initial_state, channel_func, channel_param_range, n_steps=50):
    """Simulate entanglement decay under local noise"""
    concurrences = []
    entropies = []
    
    for param in channel_param_range:
        # Apply channel to both qubits
        ch = channel_func(param)
        
        # Single qubit channel
        rho = np.outer(initial_state, initial_state.conj())
        
        # Apply to first qubit: (ch ⊗ I)(ρ)
        kraus_AB = [np.kron(k, I) for k in ch.kraus]
        rho_out = sum(k @ rho @ k.conj().T for k in kraus_AB)
        
        # Apply to second qubit: (I ⊗ ch)(ρ)
        kraus_AB = [np.kron(I, k) for k in ch.kraus]
        rho_out = sum(k @ rho_out @ k.conj().T for k in kraus_AB)
        
        # Compute entanglement
        state_svd = QuantumStateSVD(np.sqrt(np.diag(rho_out) + 1e-15), 2, 2)
        
        # For mixed states, use negativity instead
        neg = dm_ops.negativity(rho_out, 2, 2)
        
        concurrences.append(neg)
    
    return concurrences

# Test entanglement decay
param_range = np.linspace(0, 0.5, 50)

fig, ax = plt.subplots(figsize=(10, 6))

for name, channel_func in [
    ("Depolarizing", depolarizing_channel),
    ("Amplitude Damping", amplitude_damping_channel),
    ("Phase Damping", phase_damping_channel)
]:
    negs = simulate_entanglement_decay(bell, channel_func, param_range)
    ax.plot(param_range, negs, label=name, linewidth=2)

ax.set_xlabel('Noise Parameter')
ax.set_ylabel('Negativity')
ax.set_title('Entanglement Decay Under Local Noise')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('entanglement_decay.png', dpi=150)
plt.show()
```

---

## 💻 Lab Part 3: Advanced Projects (1.5 hours)

```python
# ============================================
# PROJECT 1: Quantum State Tomography
# ============================================

def simulate_tomography(rho_true, n_measurements=1000):
    """Simulate quantum state tomography"""
    
    # Measurement bases
    bases = {
        'Z': [ket_0, ket_1],
        'X': [ket_plus, ket_minus],
        'Y': [(ket_0 + 1j*ket_1)/np.sqrt(2), (ket_0 - 1j*ket_1)/np.sqrt(2)]
    }
    
    results = {}
    
    for basis_name, basis_states in bases.items():
        # Project onto basis states
        probs = [np.real(np.trace(np.outer(b, b.conj()) @ rho_true)) 
                 for b in basis_states]
        
        # Simulate measurements
        outcomes = np.random.choice(len(basis_states), n_measurements, p=probs)
        counts = [np.sum(outcomes == i) for i in range(len(basis_states))]
        results[basis_name] = np.array(counts) / n_measurements
    
    return results

def reconstruct_state(results):
    """Reconstruct density matrix from tomography results"""
    # For a qubit: ρ = (I + r·σ)/2
    # r_i = ⟨σ_i⟩ = P(+) - P(-) for each basis
    
    r_x = results['X'][0] - results['X'][1]
    r_y = results['Y'][0] - results['Y'][1]
    r_z = results['Z'][0] - results['Z'][1]
    
    rho = (I + r_x*X + r_y*Y + r_z*Z) / 2
    
    # Ensure valid density matrix
    eigs, vecs = np.linalg.eigh(rho)
    eigs = np.maximum(eigs, 0)  # Project to positive
    eigs = eigs / np.sum(eigs)  # Normalize
    rho_valid = vecs @ np.diag(eigs) @ vecs.conj().T
    
    return rho_valid

print("\n=== Quantum State Tomography Simulation ===\n")

# True state
theta = np.pi/6
phi = np.pi/4
psi_true = np.cos(theta/2)*ket_0 + np.exp(1j*phi)*np.sin(theta/2)*ket_1
rho_true = np.outer(psi_true, psi_true.conj())

print(f"True state ρ:\n{rho_true}")

# Simulate tomography
results = simulate_tomography(rho_true, n_measurements=10000)
print(f"\nMeasurement results: {results}")

# Reconstruct
rho_reconstructed = reconstruct_state(results)
print(f"\nReconstructed ρ:\n{rho_reconstructed}")

# Compute fidelity
fid = dm_ops.fidelity(rho_true, rho_reconstructed)
print(f"\nReconstruction fidelity: {fid:.4f}")

# ============================================
# PROJECT 2: Quantum Error Correction Code
# ============================================

print("\n=== Three-Qubit Bit-Flip Code ===\n")

def encode_bit_flip(psi):
    """Encode single qubit into 3-qubit bit-flip code"""
    # |0⟩ → |000⟩, |1⟩ → |111⟩
    alpha, beta = psi[0], psi[1]
    ket_000 = np.array([1,0,0,0,0,0,0,0], dtype=complex)
    ket_111 = np.array([0,0,0,0,0,0,0,1], dtype=complex)
    return alpha * ket_000 + beta * ket_111

def apply_bit_flip_error(psi_encoded, qubit, p):
    """Apply bit flip error to specific qubit with probability p"""
    if np.random.random() < p:
        # Apply X to specified qubit
        X_qubit = [I, I, I]
        X_qubit[qubit] = X
        full_X = TensorOperations.kron_n(*X_qubit)
        return full_X @ psi_encoded
    return psi_encoded

def syndrome_measurement(psi_encoded):
    """Measure error syndrome"""
    # Z₁Z₂ and Z₂Z₃ stabilizers
    Z1Z2 = TensorOperations.kron_n(Z, Z, I)
    Z2Z3 = TensorOperations.kron_n(I, Z, Z)
    
    # Expectation values give syndrome
    rho = np.outer(psi_encoded, psi_encoded.conj())
    s1 = np.sign(np.real(np.trace(Z1Z2 @ rho)))
    s2 = np.sign(np.real(np.trace(Z2Z3 @ rho)))
    
    return (int((1-s1)/2), int((1-s2)/2))

def correct_error(psi_encoded, syndrome):
    """Apply correction based on syndrome"""
    error_qubit = {
        (0, 0): None,  # No error
        (1, 0): 0,     # Error on qubit 0
        (1, 1): 1,     # Error on qubit 1
        (0, 1): 2      # Error on qubit 2
    }
    
    qubit = error_qubit[syndrome]
    if qubit is not None:
        X_correction = [I, I, I]
        X_correction[qubit] = X
        full_X = TensorOperations.kron_n(*X_correction)
        return full_X @ psi_encoded
    return psi_encoded

def decode_bit_flip(psi_encoded):
    """Decode back to single qubit"""
    # Project onto code space and extract coefficients
    ket_000 = np.array([1,0,0,0,0,0,0,0], dtype=complex)
    ket_111 = np.array([0,0,0,0,0,0,0,1], dtype=complex)
    
    alpha = np.vdot(ket_000, psi_encoded)
    beta = np.vdot(ket_111, psi_encoded)
    
    return np.array([alpha, beta])

# Test the code
psi_input = ket_plus.copy()
print(f"Input state: {psi_input}")

# Encode
psi_encoded = encode_bit_flip(psi_input)
print(f"Encoded state: {psi_encoded[:4]}... (showing first 4 amplitudes)")

# Apply error to qubit 1
psi_error = apply_bit_flip_error(psi_encoded.copy(), qubit=1, p=1.0)
print(f"After bit flip on qubit 1: {psi_error[:4]}...")

# Syndrome measurement
syndrome = syndrome_measurement(psi_error)
print(f"Syndrome: {syndrome}")

# Correction
psi_corrected = correct_error(psi_error, syndrome)
print(f"After correction: {psi_corrected[:4]}...")

# Decode
psi_output = decode_bit_flip(psi_corrected)
print(f"Decoded output: {psi_output}")
print(f"Fidelity with input: {np.abs(np.vdot(psi_input, psi_output))**2:.6f}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("LAB COMPLETE - Week 18 Advanced Linear Algebra")
print("="*60)
print("""
Implemented:
1. SVD-based quantum state analysis (Schmidt decomposition)
2. Tensor product operations and partial trace
3. Density matrix measures (purity, entropy, fidelity, negativity)
4. Quantum channels (depolarizing, amplitude/phase damping)
5. Lindblad master equation simulation
6. Entanglement dynamics under noise
7. Quantum state tomography
8. Three-qubit bit-flip error correction code

Key concepts demonstrated:
- Entanglement quantification via Schmidt coefficients
- Mixed states from partial trace of entangled states
- Decoherence as entanglement with environment
- Error correction through redundancy
""")
```

---

## 📝 Lab Exercises

### Exercise 1: Implement Shor's 9-Qubit Code
Extend the bit-flip code to protect against both bit and phase flips.

### Exercise 2: Quantum Process Tomography
Implement a function to characterize an unknown quantum channel.

### Exercise 3: Entanglement Witness
Implement an entanglement witness for detecting entanglement in mixed states.

### Exercise 4: Random Quantum States
Generate Haar-random quantum states and analyze their entanglement statistics.

---

## ✅ Lab Checklist

- [ ] Implement SVD-based quantum state analysis
- [ ] Build tensor product operations library
- [ ] Create density matrix utilities
- [ ] Implement quantum channels with Kraus operators
- [ ] Simulate Lindblad dynamics
- [ ] Analyze entanglement decay
- [ ] Complete state tomography simulation
- [ ] Implement error correction code
- [ ] Save all figures

---

## 🔜 Preview: Tomorrow

**Day 126: Week 18 Review — Advanced Linear Algebra Mastery**
- Comprehensive concept review
- SVD, tensor products, density matrices integration
- Quantum applications synthesis
- Preparation for Week 19 (Complex Analysis)

---

*"The computer doesn't understand quantum mechanics, but with linear algebra, it can simulate it."*
— Computational Quantum Physicist
