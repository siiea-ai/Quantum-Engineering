# Day 118: Computational Lab — Quantum Operators in Action

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 1:00 PM | 3 hours | Lab Part 1: Operator Foundations |
| Afternoon | 3:00 PM - 6:00 PM | 3 hours | Lab Part 2: Quantum Simulation |
| Evening | 7:30 PM - 9:00 PM | 1.5 hours | Lab Part 3: Advanced Applications |

**Total Study Time: 7.5 hours**

---

## 🎯 Lab Objectives

1. Build a comprehensive quantum operator library
2. Implement Bloch sphere visualization
3. Simulate quantum gate circuits
4. Explore decoherence and mixed states
5. Implement quantum tomography basics

---

## 💻 Lab Part 1: Quantum Operator Library (3 hours)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm

np.set_printoptions(precision=4, suppress=True)

# ============================================
# Fundamental Quantum Objects
# ============================================

class QuantumOperator:
    """Base class for quantum operators"""
    
    def __init__(self, matrix, name="Unnamed"):
        self.matrix = np.array(matrix, dtype=complex)
        self.name = name
        self.dim = self.matrix.shape[0]
    
    def __repr__(self):
        return f"QuantumOperator({self.name}, dim={self.dim})"
    
    def __matmul__(self, other):
        if isinstance(other, QuantumOperator):
            result = QuantumOperator(self.matrix @ other.matrix, 
                                     f"{self.name}·{other.name}")
            return result
        elif isinstance(other, QuantumState):
            return QuantumState(self.matrix @ other.vector)
        return NotImplemented
    
    def __add__(self, other):
        return QuantumOperator(self.matrix + other.matrix, 
                               f"({self.name}+{other.name})")
    
    def __mul__(self, scalar):
        return QuantumOperator(scalar * self.matrix, f"{scalar}·{self.name}")
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    @property
    def dagger(self):
        """Hermitian conjugate"""
        return QuantumOperator(self.matrix.conj().T, f"{self.name}†")
    
    @property
    def trace(self):
        return np.trace(self.matrix)
    
    @property
    def det(self):
        return np.linalg.det(self.matrix)
    
    def is_hermitian(self, tol=1e-10):
        return np.allclose(self.matrix, self.matrix.conj().T, atol=tol)
    
    def is_unitary(self, tol=1e-10):
        product = self.matrix @ self.matrix.conj().T
        return np.allclose(product, np.eye(self.dim), atol=tol)
    
    def is_positive(self, tol=1e-10):
        if not self.is_hermitian(tol):
            return False
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return np.all(eigenvalues >= -tol)
    
    def eigendecomposition(self):
        """Return eigenvalues and eigenvectors"""
        if self.is_hermitian():
            return np.linalg.eigh(self.matrix)
        return np.linalg.eig(self.matrix)
    
    def spectral_decomposition(self):
        """Return spectral form: eigenvalues and projectors"""
        eigenvalues, eigenvectors = self.eigendecomposition()
        projectors = []
        for i in range(self.dim):
            v = eigenvectors[:, i:i+1]
            P = QuantumOperator(v @ v.conj().T, f"P_{eigenvalues[i]:.2f}")
            projectors.append(P)
        return eigenvalues, projectors
    
    def function(self, f):
        """Apply function f to operator via spectral decomposition"""
        eigenvalues, projectors = self.spectral_decomposition()
        result = sum(f(lam) * P.matrix for lam, P in zip(eigenvalues, projectors))
        return QuantumOperator(result, f"f({self.name})")
    
    def exp(self):
        """Matrix exponential"""
        return QuantumOperator(expm(self.matrix), f"exp({self.name})")
    
    def commutator(self, other):
        """[A, B] = AB - BA"""
        return QuantumOperator(
            self.matrix @ other.matrix - other.matrix @ self.matrix,
            f"[{self.name},{other.name}]"
        )
    
    def anticommutator(self, other):
        """{A, B} = AB + BA"""
        return QuantumOperator(
            self.matrix @ other.matrix + other.matrix @ self.matrix,
            f"{{{self.name},{other.name}}}"
        )


class QuantumState:
    """Quantum state vector"""
    
    def __init__(self, vector, normalize=True):
        self.vector = np.array(vector, dtype=complex).flatten()
        if normalize:
            self.normalize()
        self.dim = len(self.vector)
    
    def normalize(self):
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm
    
    @property
    def bra(self):
        return self.vector.conj()
    
    @property
    def ket(self):
        return self.vector
    
    def inner(self, other):
        """⟨self|other⟩"""
        return np.vdot(self.vector, other.vector)
    
    def outer(self, other=None):
        """Return |self⟩⟨other| as operator"""
        if other is None:
            other = self
        return QuantumOperator(
            np.outer(self.vector, other.vector.conj()),
            f"|ψ⟩⟨φ|"
        )
    
    def expectation(self, operator):
        """⟨ψ|A|ψ⟩"""
        return np.real(self.bra @ operator.matrix @ self.ket)
    
    def probabilities(self):
        """Measurement probabilities in computational basis"""
        return np.abs(self.vector)**2
    
    def density_matrix(self):
        """Return |ψ⟩⟨ψ| as density operator"""
        return DensityMatrix(self.outer().matrix)
    
    def bloch_vector(self):
        """For qubit: return (x, y, z) on Bloch sphere"""
        if self.dim != 2:
            raise ValueError("Bloch vector only defined for qubits")
        rho = self.density_matrix()
        return rho.bloch_vector()


class DensityMatrix(QuantumOperator):
    """Density matrix for mixed states"""
    
    def __init__(self, matrix, name="ρ"):
        super().__init__(matrix, name)
        self._validate()
    
    def _validate(self):
        if not self.is_hermitian():
            print("Warning: Density matrix should be Hermitian")
        if not self.is_positive():
            print("Warning: Density matrix should be positive")
        if not np.isclose(self.trace, 1):
            print(f"Warning: Trace = {self.trace}, should be 1")
    
    @property
    def purity(self):
        """tr(ρ²) - equals 1 for pure states"""
        return np.real(np.trace(self.matrix @ self.matrix))
    
    def is_pure(self, tol=1e-10):
        return np.isclose(self.purity, 1, atol=tol)
    
    def von_neumann_entropy(self):
        """S = -tr(ρ log ρ)"""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Avoid log(0)
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def bloch_vector(self):
        """For qubit: return Bloch vector (x, y, z)"""
        if self.dim != 2:
            raise ValueError("Bloch vector only for qubits")
        x = 2 * np.real(self.matrix[0, 1])
        y = 2 * np.imag(self.matrix[1, 0])
        z = np.real(self.matrix[0, 0] - self.matrix[1, 1])
        return np.array([x, y, z])
    
    def evolve(self, U):
        """Unitary evolution: ρ → UρU†"""
        new_matrix = U.matrix @ self.matrix @ U.dagger.matrix
        return DensityMatrix(new_matrix)


# ============================================
# Standard Quantum Gates and Operators
# ============================================

# Identity
I = QuantumOperator([[1, 0], [0, 1]], "I")

# Pauli matrices
X = QuantumOperator([[0, 1], [1, 0]], "X")
Y = QuantumOperator([[0, -1j], [1j, 0]], "Y")
Z = QuantumOperator([[1, 0], [0, -1]], "Z")

# Hadamard
H = QuantumOperator([[1, 1], [1, -1]], "H") * (1/np.sqrt(2))

# Phase gates
S = QuantumOperator([[1, 0], [0, 1j]], "S")
T = QuantumOperator([[1, 0], [0, np.exp(1j*np.pi/4)]], "T")

# Rotation gates
def Rx(theta):
    """Rotation around X axis"""
    c, s = np.cos(theta/2), np.sin(theta/2)
    return QuantumOperator([[c, -1j*s], [-1j*s, c]], f"Rx({theta:.2f})")

def Ry(theta):
    """Rotation around Y axis"""
    c, s = np.cos(theta/2), np.sin(theta/2)
    return QuantumOperator([[c, -s], [s, c]], f"Ry({theta:.2f})")

def Rz(theta):
    """Rotation around Z axis"""
    return QuantumOperator([[np.exp(-1j*theta/2), 0], 
                            [0, np.exp(1j*theta/2)]], f"Rz({theta:.2f})")

# Standard states
ket_0 = QuantumState([1, 0])
ket_1 = QuantumState([0, 1])
ket_plus = QuantumState([1, 1])
ket_minus = QuantumState([1, -1])
ket_plus_i = QuantumState([1, 1j])
ket_minus_i = QuantumState([1, -1j])

print("=== Quantum Library Loaded ===")
print(f"Pauli X is Hermitian: {X.is_hermitian()}")
print(f"Pauli X is Unitary: {X.is_unitary()}")
print(f"Hadamard is Unitary: {H.is_unitary()}")
print(f"\n[X, Z] = {(X.commutator(Z)).matrix}")
```

---

## 💻 Lab Part 2: Bloch Sphere Visualization (3 hours)

```python
# ============================================
# Bloch Sphere Visualization
# ============================================

def plot_bloch_sphere(states=None, vectors=None, title="Bloch Sphere"):
    """
    Plot Bloch sphere with optional states/vectors.
    states: list of QuantumState objects
    vectors: list of (x, y, z) tuples
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere wireframe
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightblue', alpha=0.3)
    
    # Draw axes
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='r', alpha=0.5, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='g', alpha=0.5, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='b', alpha=0.5, arrow_length_ratio=0.1)
    ax.text(1.4, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.4, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.4, 'Z', fontsize=12)
    
    # Mark poles
    ax.scatter([0, 0], [0, 0], [1, -1], color='blue', s=50)
    ax.text(0, 0, 1.1, '|0⟩', fontsize=10)
    ax.text(0, 0, -1.1, '|1⟩', fontsize=10)
    
    # Plot states
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    if states:
        for i, state in enumerate(states):
            bv = state.bloch_vector()
            ax.quiver(0, 0, 0, bv[0], bv[1], bv[2], 
                     color=colors[i % len(colors)], arrow_length_ratio=0.1)
            ax.scatter([bv[0]], [bv[1]], [bv[2]], 
                      color=colors[i % len(colors)], s=50)
    
    if vectors:
        for i, v in enumerate(vectors):
            ax.quiver(0, 0, 0, v[0], v[1], v[2], 
                     color=colors[i % len(colors)], arrow_length_ratio=0.1)
            ax.scatter([v[0]], [v[1]], [v[2]], 
                      color=colors[i % len(colors)], s=50)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    
    return fig, ax

# Plot standard states
print("\n=== Bloch Sphere Visualization ===")
states = [ket_0, ket_1, ket_plus, ket_minus, ket_plus_i, ket_minus_i]
print("Standard qubit states on Bloch sphere:")
for s in [ket_0, ket_plus, ket_plus_i]:
    print(f"  {s.vector} → Bloch: {s.bloch_vector()}")

fig, ax = plot_bloch_sphere(states, title="Standard Qubit States")
plt.savefig('bloch_states.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Time Evolution on Bloch Sphere
# ============================================

def animate_evolution(H, psi_0, t_max=4*np.pi, n_steps=100):
    """
    Visualize time evolution under Hamiltonian H.
    Returns trajectory on Bloch sphere.
    """
    t_values = np.linspace(0, t_max, n_steps)
    trajectory = []
    
    for t in t_values:
        # U(t) = exp(-iHt)
        U = (-1j * t * H).exp()
        psi_t = U @ psi_0
        trajectory.append(psi_t.bloch_vector())
    
    return np.array(trajectory), t_values

# Precession around Z axis (Larmor precession)
psi_0 = ket_plus  # Start at |+⟩
trajectory_z, t_vals = animate_evolution(Z, psi_0)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
# Draw sphere
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(x, y, z, color='lightblue', alpha=0.2)

# Plot trajectory
ax1.plot(trajectory_z[:, 0], trajectory_z[:, 1], trajectory_z[:, 2], 
         'r-', linewidth=2, label='Trajectory')
ax1.scatter([trajectory_z[0, 0]], [trajectory_z[0, 1]], [trajectory_z[0, 2]], 
           color='green', s=100, label='Start')
ax1.scatter([trajectory_z[-1, 0]], [trajectory_z[-1, 1]], [trajectory_z[-1, 2]], 
           color='red', s=100, label='End')
ax1.set_title('Precession around Z (H = σz)')
ax1.legend()

# Precession around X axis (Rabi oscillations)
trajectory_x, _ = animate_evolution(X, ket_0)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(x, y, z, color='lightblue', alpha=0.2)
ax2.plot(trajectory_x[:, 0], trajectory_x[:, 1], trajectory_x[:, 2], 
         'b-', linewidth=2)
ax2.scatter([trajectory_x[0, 0]], [trajectory_x[0, 1]], [trajectory_x[0, 2]], 
           color='green', s=100)
ax2.set_title('Rabi oscillations (H = σx)')

plt.tight_layout()
plt.savefig('bloch_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Quantum Circuit Simulation
# ============================================

class QuantumCircuit:
    """Simple quantum circuit simulator"""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.gates = []
        self.state = QuantumState(np.zeros(self.dim))
        self.state.vector[0] = 1  # Initialize to |0...0⟩
    
    def reset(self):
        self.state = QuantumState(np.zeros(self.dim))
        self.state.vector[0] = 1
    
    def apply(self, gate, qubit):
        """Apply single-qubit gate to specified qubit"""
        # Build full operator using tensor products
        if qubit == 0:
            full_op = gate.matrix
        else:
            full_op = np.eye(2)
        
        for i in range(1, self.n_qubits):
            if i == qubit:
                full_op = np.kron(full_op, gate.matrix)
            else:
                full_op = np.kron(full_op, np.eye(2))
        
        self.state = QuantumState(full_op @ self.state.vector, normalize=False)
        self.gates.append((gate.name, qubit))
    
    def apply_cnot(self, control, target):
        """Apply CNOT gate"""
        CNOT = np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,0,1],
                         [0,0,1,0]])
        self.state = QuantumState(CNOT @ self.state.vector, normalize=False)
        self.gates.append(('CNOT', (control, target)))
    
    def measure(self):
        """Measure all qubits, return outcome and probabilities"""
        probs = self.state.probabilities()
        outcome = np.random.choice(self.dim, p=probs)
        return outcome, probs
    
    def get_density_matrix(self):
        return self.state.density_matrix()

# Create Bell state
print("\n=== Quantum Circuit: Bell State ===")
qc = QuantumCircuit(2)
qc.apply(H, 0)  # Hadamard on qubit 0
qc.apply_cnot(0, 1)  # CNOT

print(f"Final state: {qc.state.vector}")
print(f"Expected: (|00⟩ + |11⟩)/√2")
print(f"Probabilities: {qc.state.probabilities()}")

# Verify entanglement via partial trace
rho = qc.get_density_matrix()
print(f"\nFull density matrix:\n{rho.matrix}")
print(f"Purity of full state: {rho.purity:.4f}")
```

---

## 💻 Lab Part 3: Advanced Applications (1.5 hours)

```python
# ============================================
# Decoherence and Mixed States
# ============================================

def depolarizing_channel(rho, p):
    """
    Apply depolarizing channel: ρ → (1-p)ρ + p·I/2
    p = probability of depolarization
    """
    I_mixed = DensityMatrix(np.eye(2)/2, "I/2")
    new_matrix = (1-p) * rho.matrix + p * I_mixed.matrix
    return DensityMatrix(new_matrix)

def amplitude_damping(rho, gamma):
    """
    Amplitude damping channel (T1 decay).
    gamma = decay probability
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    
    new_matrix = K0 @ rho.matrix @ K0.conj().T + K1 @ rho.matrix @ K1.conj().T
    return DensityMatrix(new_matrix)

def phase_damping(rho, gamma):
    """
    Phase damping channel (T2 dephasing).
    """
    new_matrix = rho.matrix.copy()
    new_matrix[0, 1] *= np.sqrt(1 - gamma)
    new_matrix[1, 0] *= np.sqrt(1 - gamma)
    return DensityMatrix(new_matrix)

# Demonstrate decoherence
print("\n=== Decoherence Channels ===")
psi = QuantumState([1, 1])  # |+⟩ state
rho_pure = psi.density_matrix()

print(f"Initial state |+⟩:")
print(f"  Purity: {rho_pure.purity:.4f}")
print(f"  Bloch vector: {rho_pure.bloch_vector()}")

# Apply increasing depolarization
p_values = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
print("\nDepolarizing channel:")
for p in p_values:
    rho_depol = depolarizing_channel(rho_pure, p)
    print(f"  p={p:.1f}: Purity={rho_depol.purity:.4f}, "
          f"Bloch length={np.linalg.norm(rho_depol.bloch_vector()):.4f}")

# Visualize decoherence trajectory
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Depolarizing
gammas = np.linspace(0, 1, 50)
purities_depol = [depolarizing_channel(rho_pure, g).purity for g in gammas]
axes[0].plot(gammas, purities_depol, 'b-', linewidth=2)
axes[0].set_xlabel('Depolarization p')
axes[0].set_ylabel('Purity')
axes[0].set_title('Depolarizing Channel')
axes[0].grid(True, alpha=0.3)

# Amplitude damping
purities_amp = [amplitude_damping(rho_pure, g).purity for g in gammas]
axes[1].plot(gammas, purities_amp, 'r-', linewidth=2)
axes[1].set_xlabel('Decay γ')
axes[1].set_ylabel('Purity')
axes[1].set_title('Amplitude Damping (T1)')
axes[1].grid(True, alpha=0.3)

# Phase damping
purities_phase = [phase_damping(rho_pure, g).purity for g in gammas]
axes[2].plot(gammas, purities_phase, 'g-', linewidth=2)
axes[2].set_xlabel('Dephasing γ')
axes[2].set_ylabel('Purity')
axes[2].set_title('Phase Damping (T2)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decoherence_channels.png', dpi=150)
plt.show()

# ============================================
# Quantum State Tomography (Basic)
# ============================================

def tomography_measurements(state):
    """
    Simulate tomography: measure in X, Y, Z bases.
    Returns expectation values.
    """
    exp_x = state.expectation(X)
    exp_y = state.expectation(Y)
    exp_z = state.expectation(Z)
    return exp_x, exp_y, exp_z

def reconstruct_state(exp_x, exp_y, exp_z):
    """
    Reconstruct density matrix from Pauli expectations.
    ρ = (I + ⟨X⟩σx + ⟨Y⟩σy + ⟨Z⟩σz) / 2
    """
    rho = (I.matrix + exp_x * X.matrix + exp_y * Y.matrix + exp_z * Z.matrix) / 2
    return DensityMatrix(rho, "ρ_reconstructed")

# Test tomography
print("\n=== Quantum State Tomography ===")
test_state = QuantumState([np.cos(np.pi/8), np.exp(1j*np.pi/3)*np.sin(np.pi/8)])
print(f"Original state: {test_state.vector}")

# Measure
ex, ey, ez = tomography_measurements(test_state)
print(f"Measurements: ⟨X⟩={ex:.4f}, ⟨Y⟩={ey:.4f}, ⟨Z⟩={ez:.4f}")

# Reconstruct
rho_reconstructed = reconstruct_state(ex, ey, ez)
rho_original = test_state.density_matrix()

print(f"\nOriginal density matrix:\n{rho_original.matrix}")
print(f"\nReconstructed:\n{rho_reconstructed.matrix}")
print(f"\nFidelity: {np.abs(np.trace(rho_original.matrix @ rho_reconstructed.matrix)):.6f}")

# ============================================
# Complete Lab Summary
# ============================================

print("\n" + "="*50)
print("LAB COMPLETE - Summary of Implementations")
print("="*50)
print("""
1. QuantumOperator class:
   - Hermitian, unitary, positivity tests
   - Spectral decomposition
   - Matrix functions via spectral theorem
   - Commutators and anticommutators

2. QuantumState class:
   - Inner products, outer products
   - Expectation values
   - Bloch sphere representation

3. DensityMatrix class:
   - Purity, von Neumann entropy
   - Time evolution

4. Bloch sphere visualization:
   - Standard states
   - Time evolution trajectories

5. Quantum circuits:
   - Single-qubit gates
   - CNOT and Bell state creation

6. Decoherence channels:
   - Depolarizing
   - Amplitude damping (T1)
   - Phase damping (T2)

7. State tomography basics
""")
```

---

## 📝 Lab Exercises

### Exercise 1: Implement Toffoli Gate
Extend the QuantumCircuit class to include the Toffoli (CCNOT) gate.

### Exercise 2: Random Unitary Generation
Implement a function to generate random unitary matrices using the QR decomposition of random complex matrices.

### Exercise 3: Entanglement Entropy
Implement partial trace and compute entanglement entropy for 2-qubit states.

### Exercise 4: Gate Decomposition
Show that any single-qubit unitary can be written as U = e^(iα) Rz(β) Ry(γ) Rz(δ).

### Exercise 5: Noise Simulation
Simulate a quantum circuit with gate errors (random small rotations after each gate).

---

## ✅ Lab Checklist

- [ ] Implement QuantumOperator class with all methods
- [ ] Verify Pauli matrices are Hermitian and unitary
- [ ] Create Bloch sphere visualization
- [ ] Animate time evolution
- [ ] Build Bell state with circuit
- [ ] Implement decoherence channels
- [ ] Complete basic tomography
- [ ] Save all figures

---

## 🔜 Preview: Tomorrow

**Day 119: Week 17 Review — Hermitian & Unitary Operators Mastery**
- Comprehensive concept review
- Full problem set covering all topics
- Self-assessment and gap identification
- Preparation for Week 18

---

*"The computer is incredibly fast, accurate, and stupid. Man is incredibly slow, inaccurate, and brilliant. The marriage of the two is a force beyond calculation."*
— Leo Cherne
