# Day 111: Computational Lab — Inner Products in Quantum Computing

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 1:00 PM | 3 hours | Lab Part 1: Inner Products & Norms |
| Afternoon | 3:00 PM - 6:00 PM | 3 hours | Lab Part 2: Quantum Applications |
| Evening | 7:30 PM - 9:00 PM | 1.5 hours | Lab Part 3: Advanced Projects |

**Total Study Time: 7.5 hours**

---

## 🎯 Lab Objectives

1. Master NumPy/SciPy inner product computations
2. Implement Gram-Schmidt and verify orthonormality
3. Build quantum measurement simulator
4. Visualize inner products on Bloch sphere
5. Implement quantum state tomography basics
6. Connect to Fourier analysis

---

## 💻 Lab Part 1: Inner Products & Norms (3 hours)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr, expm
from scipy.integrate import quad
np.set_printoptions(precision=4, suppress=True)

# ============================================
# 1.1 Inner Product Fundamentals
# ============================================

class InnerProductSpace:
    """Generic inner product space implementation"""
    
    @staticmethod
    def real_ip(u, v):
        """Standard inner product on R^n"""
        return np.dot(u, v)
    
    @staticmethod
    def complex_ip(u, v):
        """Standard inner product on C^n (physics convention)"""
        return np.vdot(u, v)  # Conjugates first argument
    
    @staticmethod
    def weighted_ip(u, v, weights):
        """Weighted inner product"""
        return np.sum(weights * np.conj(u) * v)
    
    @staticmethod
    def function_ip(f, g, a, b):
        """L² inner product for functions on [a,b]"""
        real_part, _ = quad(lambda x: np.real(np.conj(f(x)) * g(x)), a, b)
        imag_part, _ = quad(lambda x: np.imag(np.conj(f(x)) * g(x)), a, b)
        return real_part + 1j * imag_part
    
    @staticmethod
    def matrix_ip(A, B):
        """Frobenius inner product for matrices"""
        return np.trace(A.conj().T @ B)

# Test all inner products
print("=" * 50)
print("INNER PRODUCT TESTS")
print("=" * 50)

# Real
u_real = np.array([1, 2, 3])
v_real = np.array([4, 5, 6])
print(f"\nReal IP: ⟨{u_real}, {v_real}⟩ = {InnerProductSpace.real_ip(u_real, v_real)}")

# Complex
u_complex = np.array([1+1j, 2-1j])
v_complex = np.array([1, 1j])
ip = InnerProductSpace.complex_ip(u_complex, v_complex)
print(f"Complex IP: ⟨{u_complex}, {v_complex}⟩ = {ip}")

# Verify antilinearity in first argument
alpha = 2 + 3j
ip1 = InnerProductSpace.complex_ip(alpha * u_complex, v_complex)
ip2 = np.conj(alpha) * InnerProductSpace.complex_ip(u_complex, v_complex)
print(f"Antilinearity check: ⟨αu|v⟩ = {ip1:.4f}, α*⟨u|v⟩ = {ip2:.4f}")

# Weighted
weights = np.array([2, 3])
ip_weighted = InnerProductSpace.weighted_ip(u_complex, v_complex, weights)
print(f"Weighted IP: {ip_weighted}")

# Function space
f = np.sin
g = np.cos
ip_func = InnerProductSpace.function_ip(f, g, 0, 2*np.pi)
print(f"Function IP ⟨sin, cos⟩ on [0,2π]: {ip_func:.6f}")

# Matrix (Frobenius)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
ip_mat = InnerProductSpace.matrix_ip(A, B)
print(f"Matrix IP (Frobenius): {ip_mat}")

# ============================================
# 1.2 Norm Computations
# ============================================

def compute_norms(v):
    """Compute various norms of a vector"""
    return {
        'L1': np.linalg.norm(v, ord=1),
        'L2': np.linalg.norm(v, ord=2),
        'Linf': np.linalg.norm(v, ord=np.inf),
        'custom': np.sqrt(np.abs(InnerProductSpace.complex_ip(v, v)))
    }

v = np.array([3+4j, 5-12j])
norms = compute_norms(v)
print(f"\nNorms of {v}:")
for name, value in norms.items():
    print(f"  {name}: {value:.4f}")

# ============================================
# 1.3 Cauchy-Schwarz Verification
# ============================================

def verify_cauchy_schwarz(u, v, n_random=1000):
    """Verify Cauchy-Schwarz for random vectors"""
    violations = 0
    max_ratio = 0
    
    for _ in range(n_random):
        u = np.random.randn(len(u)) + 1j * np.random.randn(len(u))
        v = np.random.randn(len(v)) + 1j * np.random.randn(len(v))
        
        lhs = np.abs(np.vdot(u, v))
        rhs = np.linalg.norm(u) * np.linalg.norm(v)
        
        ratio = lhs / rhs if rhs > 0 else 0
        max_ratio = max(max_ratio, ratio)
        
        if lhs > rhs + 1e-10:
            violations += 1
    
    return violations, max_ratio

violations, max_ratio = verify_cauchy_schwarz(np.zeros(5), np.zeros(5))
print(f"\nCauchy-Schwarz verification (1000 trials):")
print(f"  Violations: {violations}")
print(f"  Max ratio |⟨u,v⟩|/(||u|| ||v||): {max_ratio:.6f}")

# ============================================
# 1.4 Gram-Schmidt Implementation
# ============================================

def gram_schmidt(V, tol=1e-10):
    """
    Modified Gram-Schmidt orthonormalization
    V: columns are vectors to orthonormalize
    """
    n, k = V.shape
    Q = np.zeros((n, k), dtype=complex)
    R = np.zeros((k, k), dtype=complex)
    
    for j in range(k):
        v = V[:, j].astype(complex)
        
        for i in range(j):
            R[i, j] = np.vdot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        
        if R[j, j] < tol:
            print(f"Warning: Vector {j} is linearly dependent")
            Q[:, j] = 0
        else:
            Q[:, j] = v / R[j, j]
    
    return Q, R

# Test Gram-Schmidt
V = np.array([[1, 1, 1],
              [1, 0, 1],
              [1, 1, 0]], dtype=float).T

Q, R = gram_schmidt(V)
print("\n" + "=" * 50)
print("GRAM-SCHMIDT TEST")
print("=" * 50)
print(f"Original V:\n{V}")
print(f"\nOrthonormal Q:\n{Q}")
print(f"\nUpper triangular R:\n{R}")
print(f"\nVerify Q @ R = V:\n{Q @ R}")
print(f"\nVerify Q† @ Q = I:\n{Q.conj().T @ Q}")

# ============================================
# 1.5 Orthogonal Projection
# ============================================

def orthogonal_projection(v, W_basis):
    """
    Project v onto subspace spanned by W_basis
    W_basis: list of orthonormal vectors
    """
    proj = np.zeros_like(v, dtype=complex)
    for w in W_basis:
        proj += np.vdot(w, v) * w
    return proj

def projection_matrix(W_basis):
    """Create projection matrix onto span of orthonormal W_basis"""
    n = len(W_basis[0])
    P = np.zeros((n, n), dtype=complex)
    for w in W_basis:
        P += np.outer(w, w.conj())
    return P

# Project (1,2,3) onto xy-plane
v = np.array([1, 2, 3])
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])

proj = orthogonal_projection(v, [e1, e2])
P = projection_matrix([e1, e2])

print("\n" + "=" * 50)
print("ORTHOGONAL PROJECTION")
print("=" * 50)
print(f"v = {v}")
print(f"Projection onto xy-plane: {proj}")
print(f"Projection matrix P:\n{P}")
print(f"P @ v = {P @ v}")
print(f"P² = P? {np.allclose(P @ P, P)}")
print(f"P† = P? {np.allclose(P.conj().T, P)}")
```

---

## 💻 Lab Part 2: Quantum Applications (3 hours)

```python
# ============================================
# 2.1 Quantum State Class
# ============================================

class QuantumState:
    """Represent and manipulate quantum states"""
    
    def __init__(self, amplitudes, normalize=True):
        self.amplitudes = np.array(amplitudes, dtype=complex)
        if normalize:
            self.normalize()
    
    def normalize(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
    
    @property
    def dim(self):
        return len(self.amplitudes)
    
    def inner_product(self, other):
        """⟨self|other⟩"""
        return np.vdot(self.amplitudes, other.amplitudes)
    
    def fidelity(self, other):
        """F = |⟨self|other⟩|²"""
        return np.abs(self.inner_product(other))**2
    
    def measurement_probs(self, basis=None):
        """Get measurement probabilities in given basis"""
        if basis is None:
            # Computational basis
            return np.abs(self.amplitudes)**2
        else:
            return np.array([np.abs(np.vdot(b.amplitudes, self.amplitudes))**2 
                           for b in basis])
    
    def evolve(self, U):
        """Apply unitary evolution"""
        self.amplitudes = U @ self.amplitudes
        self.normalize()  # Should already be normalized, but ensure
    
    def measure(self, basis=None):
        """Simulate measurement, return outcome and collapse state"""
        probs = self.measurement_probs(basis)
        outcome = np.random.choice(len(probs), p=probs)
        
        if basis is None:
            self.amplitudes = np.zeros_like(self.amplitudes)
            self.amplitudes[outcome] = 1
        else:
            self.amplitudes = basis[outcome].amplitudes.copy()
        
        return outcome
    
    def expectation(self, observable):
        """⟨ψ|A|ψ⟩"""
        return np.real(np.vdot(self.amplitudes, observable @ self.amplitudes))
    
    def bloch_vector(self):
        """Get Bloch vector for qubit state"""
        if self.dim != 2:
            raise ValueError("Bloch vector only for qubits")
        
        # Pauli matrices
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        return np.array([self.expectation(X), 
                        self.expectation(Y), 
                        self.expectation(Z)])
    
    def __repr__(self):
        return f"QuantumState({self.amplitudes})"

# ============================================
# 2.2 Standard Quantum States
# ============================================

# Computational basis
ket_0 = QuantumState([1, 0])
ket_1 = QuantumState([0, 1])

# Hadamard basis
ket_plus = QuantumState([1, 1])
ket_minus = QuantumState([1, -1])

# Y-basis
ket_plus_i = QuantumState([1, 1j])
ket_minus_i = QuantumState([1, -1j])

print("\n" + "=" * 50)
print("QUANTUM STATES")
print("=" * 50)

print("\nComputational basis:")
print(f"  |0⟩ = {ket_0}")
print(f"  |1⟩ = {ket_1}")
print(f"  ⟨0|1⟩ = {ket_0.inner_product(ket_1)}")

print("\nHadamard basis:")
print(f"  |+⟩ = {ket_plus}")
print(f"  |-⟩ = {ket_minus}")
print(f"  ⟨+|-⟩ = {ket_plus.inner_product(ket_minus):.6f}")

print("\nBloch vectors:")
print(f"  |0⟩: {ket_0.bloch_vector()}")
print(f"  |1⟩: {ket_1.bloch_vector()}")
print(f"  |+⟩: {ket_plus.bloch_vector()}")
print(f"  |+i⟩: {ket_plus_i.bloch_vector()}")

# ============================================
# 2.3 Measurement Simulation
# ============================================

def simulate_measurements(state, n_shots=1000, basis=None):
    """Simulate n measurements and return histogram"""
    outcomes = []
    for _ in range(n_shots):
        # Create fresh copy
        psi = QuantumState(state.amplitudes.copy())
        outcome = psi.measure(basis)
        outcomes.append(outcome)
    
    return np.bincount(outcomes, minlength=state.dim) / n_shots

# Test state
psi = QuantumState([3, 4])  # Will be normalized to (3/5, 4/5)
print("\n" + "=" * 50)
print("MEASUREMENT SIMULATION")
print("=" * 50)
print(f"State: {psi}")
print(f"Theoretical probs: {psi.measurement_probs()}")

empirical = simulate_measurements(psi, n_shots=10000)
print(f"Empirical probs (10000 shots): {empirical}")

# ============================================
# 2.4 Quantum Gates
# ============================================

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gates
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

# Rotation gates
def Rx(theta):
    return np.cos(theta/2) * I - 1j * np.sin(theta/2) * X

def Ry(theta):
    return np.cos(theta/2) * I - 1j * np.sin(theta/2) * Y

def Rz(theta):
    return np.cos(theta/2) * I - 1j * np.sin(theta/2) * Z

print("\n" + "=" * 50)
print("QUANTUM GATES")
print("=" * 50)

# Verify unitarity
gates = {'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T}
for name, U in gates.items():
    is_unitary = np.allclose(U @ U.conj().T, I)
    print(f"{name} is unitary: {is_unitary}")

# Test Hadamard
psi = QuantumState([1, 0])  # |0⟩
psi.evolve(H)
print(f"\nH|0⟩ = {psi}")
print(f"Bloch vector: {psi.bloch_vector()}")

# ============================================
# 2.5 Multi-qubit States and Entanglement
# ============================================

def tensor_product(psi, phi):
    """Tensor product of two states"""
    return QuantumState(np.kron(psi.amplitudes, phi.amplitudes))

# Two-qubit gates
CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]], dtype=complex)

SWAP = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]], dtype=complex)

# Create Bell state
psi_00 = tensor_product(ket_0, ket_0)
print("\n" + "=" * 50)
print("BELL STATE CREATION")
print("=" * 50)
print(f"|00⟩ = {psi_00}")

# Apply H to first qubit
H_I = np.kron(H, I)
psi_00.evolve(H_I)
print(f"(H⊗I)|00⟩ = {psi_00}")

# Apply CNOT
psi_00.evolve(CNOT)
print(f"CNOT(H⊗I)|00⟩ = |Φ+⟩ = {psi_00}")

# Verify entanglement via Schmidt decomposition
def schmidt_coefficients(psi_2qubit):
    """Get Schmidt coefficients for 2-qubit state"""
    # Reshape to matrix
    M = psi_2qubit.amplitudes.reshape(2, 2)
    # SVD gives Schmidt decomposition
    U, S, Vh = np.linalg.svd(M)
    return S

schmidt = schmidt_coefficients(psi_00)
print(f"Schmidt coefficients: {schmidt}")
print(f"Entangled (>1 nonzero coeff): {np.sum(schmidt > 1e-10) > 1}")

# ============================================
# 2.6 Density Matrix and Partial Trace
# ============================================

def density_matrix(psi):
    """Pure state density matrix"""
    return np.outer(psi.amplitudes, psi.amplitudes.conj())

def partial_trace(rho, keep_qubit, dims=(2, 2)):
    """Partial trace over one qubit of 2-qubit system"""
    d1, d2 = dims
    rho = rho.reshape(d1, d2, d1, d2)
    
    if keep_qubit == 0:
        return np.trace(rho, axis1=1, axis2=3)
    else:
        return np.trace(rho, axis1=0, axis2=2)

# Bell state density matrix
rho_bell = density_matrix(psi_00)
print("\n" + "=" * 50)
print("DENSITY MATRIX")
print("=" * 50)
print(f"ρ(|Φ+⟩) =\n{rho_bell}")

# Reduced density matrix
rho_A = partial_trace(rho_bell, keep_qubit=0)
print(f"\nReduced density matrix (trace out B):\n{rho_A}")
print(f"Purity Tr(ρ²) = {np.real(np.trace(rho_A @ rho_A)):.4f}")
print("(Purity < 1 indicates entanglement!)")
```

---

## 💻 Lab Part 3: Advanced Projects (1.5 hours)

```python
# ============================================
# 3.1 Bloch Sphere Visualization
# ============================================

def plot_bloch_sphere(states_dict, title="Bloch Sphere"):
    """Plot quantum states on Bloch sphere"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sphere wireframe
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='blue')
    
    # Axes
    ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'k--', alpha=0.3)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'k--', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'k--', alpha=0.3)
    
    ax.text(1.4, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.4, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.4, 'Z', fontsize=12)
    
    # Plot states
    colors = plt.cm.rainbow(np.linspace(0, 1, len(states_dict)))
    for (name, state), color in zip(states_dict.items(), colors):
        bloch = state.bloch_vector()
        ax.scatter(*bloch, s=100, c=[color], label=name)
        ax.plot([0, bloch[0]], [0, bloch[1]], [0, bloch[2]], c=color, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    return fig, ax

# Create various states
states = {
    '|0⟩': ket_0,
    '|1⟩': ket_1,
    '|+⟩': ket_plus,
    '|-⟩': ket_minus,
    '|+i⟩': ket_plus_i,
    '|-i⟩': ket_minus_i,
    'random': QuantumState(np.random.randn(2) + 1j*np.random.randn(2))
}

fig, ax = plot_bloch_sphere(states)
plt.savefig('bloch_sphere_states.png', dpi=150)
plt.show()

# ============================================
# 3.2 State Evolution Animation
# ============================================

def animate_rotation(initial_state, axis='z', n_frames=50):
    """Animate rotation around axis on Bloch sphere"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Choose rotation generator
    if axis == 'x':
        rotation = Rx
    elif axis == 'y':
        rotation = Ry
    else:
        rotation = Rz
    
    # Compute trajectory
    angles = np.linspace(0, 2*np.pi, n_frames)
    trajectory = []
    
    for theta in angles:
        psi = QuantumState(initial_state.amplitudes.copy())
        psi.evolve(rotation(theta))
        trajectory.append(psi.bloch_vector())
    
    trajectory = np.array(trajectory)
    
    # Plot
    # Sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='blue')
    
    # Trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'r-', linewidth=2, label=f'R{axis} rotation')
    ax.scatter(*trajectory[0], s=100, c='green', label='Start')
    ax.scatter(*trajectory[-1], s=100, c='red', label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Rotation around {axis.upper()}-axis')
    ax.legend()
    
    return fig, trajectory

# Animate X rotation of |0⟩
fig, traj = animate_rotation(ket_0, axis='x')
plt.savefig('rotation_x.png', dpi=150)
plt.show()

# ============================================
# 3.3 Quantum State Tomography
# ============================================

def tomography_estimate(measurement_results):
    """
    Estimate quantum state from Pauli measurements
    measurement_results: dict with 'X', 'Y', 'Z' outcomes
    """
    # Estimate Bloch vector from measurement statistics
    # ⟨σ⟩ = P(+1) - P(-1)
    
    bloch = np.zeros(3)
    bases = ['X', 'Y', 'Z']
    
    for i, basis in enumerate(bases):
        outcomes = measurement_results[basis]
        # Convert 0,1 outcomes to +1,-1
        pm_outcomes = 2 * np.array(outcomes) - 1
        bloch[i] = np.mean(pm_outcomes)
    
    # Reconstruct state (for pure states on Bloch sphere)
    # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
    # where θ = arccos(z), φ = arctan2(y, x)
    
    x, y, z = bloch
    r = np.sqrt(x**2 + y**2 + z**2)
    
    if r > 0:
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        
        psi_estimated = QuantumState([
            np.cos(theta/2),
            np.exp(1j*phi) * np.sin(theta/2)
        ])
    else:
        psi_estimated = QuantumState([1, 0])
    
    return psi_estimated, bloch

def simulate_tomography(true_state, n_shots_per_basis=1000):
    """Simulate tomography measurements"""
    results = {}
    
    # Measurement bases
    basis_gates = {
        'Z': I,          # Measure in Z basis (computational)
        'X': H,          # H†ZH = X, so H rotates X to Z
        'Y': S.conj().T @ H  # Rotate Y to Z
    }
    
    for basis, U in basis_gates.items():
        outcomes = []
        for _ in range(n_shots_per_basis):
            psi = QuantumState(true_state.amplitudes.copy())
            psi.evolve(U)  # Rotate to computational basis
            outcome = psi.measure()  # Measure in computational basis
            outcomes.append(outcome)
        results[basis] = outcomes
    
    return results

# Test tomography
print("\n" + "=" * 50)
print("QUANTUM STATE TOMOGRAPHY")
print("=" * 50)

true_state = QuantumState([1, 1+1j])  # Some random state
print(f"True state: {true_state}")
print(f"True Bloch vector: {true_state.bloch_vector()}")

# Simulate measurements
results = simulate_tomography(true_state, n_shots_per_basis=10000)

# Estimate state
estimated_state, estimated_bloch = tomography_estimate(results)
print(f"\nEstimated state: {estimated_state}")
print(f"Estimated Bloch vector: {estimated_bloch}")

# Fidelity
fidelity = true_state.fidelity(estimated_state)
print(f"Reconstruction fidelity: {fidelity:.4f}")

# ============================================
# 3.4 Fourier Connection
# ============================================

def discrete_fourier_basis(N):
    """Generate discrete Fourier basis for C^N"""
    basis = []
    omega = np.exp(2j * np.pi / N)
    
    for k in range(N):
        ek = np.array([omega**(j*k) for j in range(N)]) / np.sqrt(N)
        basis.append(QuantumState(ek, normalize=False))
    
    return basis

def verify_orthonormality(basis):
    """Verify basis is orthonormal"""
    N = len(basis)
    gram = np.zeros((N, N), dtype=complex)
    
    for i in range(N):
        for j in range(N):
            gram[i, j] = basis[i].inner_product(basis[j])
    
    return np.allclose(gram, np.eye(N))

print("\n" + "=" * 50)
print("FOURIER BASIS")
print("=" * 50)

N = 4
fourier_basis = discrete_fourier_basis(N)
print(f"Discrete Fourier basis for C^{N}:")
for k, ek in enumerate(fourier_basis):
    print(f"  |ω_{k}⟩ = {ek.amplitudes}")

print(f"\nOrthonormal: {verify_orthonormality(fourier_basis)}")

# Quantum Fourier Transform
def QFT_matrix(N):
    """Quantum Fourier Transform matrix"""
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for k in range(N)] 
                     for j in range(N)]) / np.sqrt(N)

QFT = QFT_matrix(4)
print(f"\nQFT matrix (N=4):\n{QFT}")
print(f"Unitary: {np.allclose(QFT @ QFT.conj().T, np.eye(4))}")

print("\n" + "=" * 50)
print("LAB COMPLETE")
print("=" * 50)
```

---

## 📝 Lab Exercises

### Exercise 1: Inner Product Explorer
Implement a function that visualizes how the inner product ⟨u|v⟩ changes as v rotates around the Bloch sphere while u is fixed.

### Exercise 2: Entanglement Witness
Create a function that detects entanglement using the purity of reduced density matrices.

### Exercise 3: Quantum Random Walk
Implement a 1D quantum random walk using inner products and unitary evolution.

### Exercise 4: Error Analysis
Compare classical vs modified Gram-Schmidt for increasingly ill-conditioned matrices.

---

## ✅ Lab Checklist

- [ ] Implement and test all inner product types
- [ ] Verify Cauchy-Schwarz numerically
- [ ] Build working Gram-Schmidt
- [ ] Create quantum state class
- [ ] Simulate measurements
- [ ] Visualize Bloch sphere
- [ ] Implement basic tomography
- [ ] Connect to Fourier analysis

---

## 📓 Lab Report Questions

1. How does numerical precision affect Gram-Schmidt orthogonality?

2. What happens to tomography accuracy as n_shots decreases?

3. Why is the Fourier basis important for quantum computing?

4. How does entanglement appear in the Schmidt decomposition?

---

*"The computer is incredibly fast, accurate, and stupid. Man is incredibly slow, inaccurate, and brilliant. The marriage of the two is a force beyond calculation."*
— Leo Cherne
