# Day 124: Density Matrices — The Complete Description of Quantum States

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Density Matrix Formalism |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Distinguish pure states from mixed states
2. Construct density matrices for both pure and mixed states
3. Compute expectation values and probabilities using density matrices
4. Understand time evolution of density matrices
5. Work with quantum channels and Kraus operators
6. Model decoherence using density matrices

---

## 📚 Required Reading

### Primary Text
- **Nielsen & Chuang, Section 2.4**: The density operator
- **Sakurai, Chapter 3.4**: Density operators and pure vs mixed ensembles

### Secondary
- **Preskill's Notes, Chapter 3**: Density matrices
- **Wilde, Chapter 4**: Quantum channels

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: Why Density Matrices?

**Limitations of state vectors:**
- Cannot describe statistical mixtures (classical uncertainty)
- Cannot describe subsystems of entangled states
- Cannot describe outcomes of partial measurements

**Density matrices solve all these problems!**

### 2. Pure States as Density Matrices

For pure state |ψ⟩, the density matrix is:
$$\boxed{\rho = |\psi\rangle\langle\psi|}$$

**Example:** |+⟩ = (|0⟩+|1⟩)/√2
$$\rho_{|+\rangle} = |+\rangle\langle+| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

### 3. Mixed States

**Definition:** A mixed state is a statistical ensemble of pure states:
$$\boxed{\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|}$$

where pᵢ ≥ 0 and Σpᵢ = 1 (classical probabilities).

**Example:** 50% |0⟩ and 50% |1⟩ (classical mixture)
$$\rho_{\text{mixed}} = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{I}{2}$$

**Key distinction:**
- |+⟩ is a coherent superposition (pure)
- 50/50 mixture of |0⟩,|1⟩ is incoherent (mixed)
- Both give 50/50 outcomes for Z measurement, but differ for X measurement!

### 4. Properties of Density Matrices

A valid density matrix ρ must satisfy:

| Property | Condition | Meaning |
|----------|-----------|---------|
| Hermitian | ρ = ρ† | Real eigenvalues |
| Positive semidefinite | ρ ≥ 0 | All eigenvalues ≥ 0 |
| Normalized | tr(ρ) = 1 | Probabilities sum to 1 |

**Additional:**
- Eigenvalues are probabilities (spectral decomposition)
- ρ = Σᵢ λᵢ |eᵢ⟩⟨eᵢ| where λᵢ ≥ 0, Σλᵢ = 1

### 5. Pure vs Mixed: The Purity Test

**Purity:** γ(ρ) = tr(ρ²)

| State Type | Purity | Condition |
|------------|--------|-----------|
| Pure | 1 | ρ² = ρ |
| Mixed | < 1 | ρ² ≠ ρ |
| Maximally mixed | 1/d | ρ = I/d |

**Equivalently:** Pure ⟺ rank(ρ) = 1

### 6. Expectation Values

**For pure state:** ⟨A⟩ = ⟨ψ|A|ψ⟩

**For density matrix:**
$$\boxed{\langle A \rangle = \text{tr}(A\rho) = \text{tr}(\rho A)}$$

**Proof:** For ρ = Σpᵢ|ψᵢ⟩⟨ψᵢ|:
$$\text{tr}(A\rho) = \sum_i p_i \text{tr}(A|\psi_i\rangle\langle\psi_i|) = \sum_i p_i \langle\psi_i|A|\psi_i\rangle$$

### 7. Measurement Statistics

**Probability of outcome m:**
$$P(m) = \text{tr}(P_m \rho)$$
where Pₘ = |m⟩⟨m| is the projector onto outcome m.

**Post-measurement state:**
$$\rho' = \frac{P_m \rho P_m}{\text{tr}(P_m \rho)}$$

### 8. Time Evolution

**Schrödinger equation for density matrices:**
$$\boxed{\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho]}$$

This is the **von Neumann equation** (or quantum Liouville equation).

**Solution:**
$$\rho(t) = U(t)\rho(0)U^\dagger(t)$$
where U(t) = e^(-iHt/ℏ).

### 9. The Bloch Sphere Representation

For a qubit, any density matrix can be written as:
$$\boxed{\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})}$$

where:
- **r⃗** = (rₓ, rᵧ, r_z) is the **Bloch vector**
- **σ⃗** = (σₓ, σᵧ, σ_z) are Pauli matrices

**Properties:**
- |**r⃗**| ≤ 1 (inside unit ball)
- |**r⃗**| = 1 ⟺ pure state (on surface)
- |**r⃗**| < 1 ⟺ mixed state (interior)
- **r⃗** = 0 ⟺ maximally mixed (center)

### 10. Quantum Channels

**Definition:** A quantum channel ℰ maps density matrices to density matrices:
$$\rho \mapsto \mathcal{E}(\rho)$$

**Requirements:**
1. Completely positive (ℰ ⊗ I maps positive to positive)
2. Trace preserving: tr(ℰ(ρ)) = tr(ρ)

**Kraus representation:**
$$\boxed{\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger}$$

where Kraus operators satisfy Σₖ Kₖ†Kₖ = I.

---

## 🔬 Quantum Mechanics Connection

### Types of Quantum Channels

**1. Unitary channel:** ℰ(ρ) = UρU†
- Single Kraus operator K = U
- Reversible, no information loss

**2. Depolarizing channel:**
$$\mathcal{E}(\rho) = (1-p)\rho + p\frac{I}{2}$$
- Replaces state with maximally mixed state with probability p

**3. Amplitude damping (T₁ decay):**
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
- Models energy relaxation (excited → ground)

**4. Phase damping (T₂ dephasing):**
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix}$$
- Destroys off-diagonal coherence

### The Lindblad Master Equation

For open quantum systems coupled to environment:
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$

- First term: coherent (Hamiltonian) evolution
- Second term: dissipation (Lindblad operators Lₖ)

### T₁ and T₂ Times

**T₁ (relaxation time):** Time for excited state population to decay
**T₂ (decoherence time):** Time for off-diagonal elements to decay

**Relation:** T₂ ≤ 2T₁ (fundamental limit)

For qubits:
- Superconducting: T₁ ~ 100 μs, T₂ ~ 100 μs
- Trapped ions: T₁ ~ seconds, T₂ ~ seconds
- Nitrogen-vacancy: T₁ ~ ms, T₂ ~ ms at room temp

---

## ✏️ Worked Examples

### Example 1: Constructing a Mixed State

A source emits |0⟩ with probability 1/3 and |+⟩ with probability 2/3.

**Density matrix:**
$$\rho = \frac{1}{3}|0\rangle\langle 0| + \frac{2}{3}|+\rangle\langle+|$$

$$= \frac{1}{3}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + \frac{2}{3} \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 2/3 & 1/3 \\ 1/3 & 1/3 \end{pmatrix}$$

**Check:**
- tr(ρ) = 2/3 + 1/3 = 1 ✓
- Eigenvalues: λ = (1 ± √5/3)/2 ≈ 0.873, 0.127 (both ≥ 0) ✓

**Purity:** tr(ρ²) = (4/9 + 1/9 + 1/9 + 1/9) = 7/9 < 1 (mixed) ✓

### Example 2: Expectation Value

For ρ from Example 1, find ⟨σ_z⟩.

$$\langle\sigma_z\rangle = \text{tr}(\sigma_z \rho) = \text{tr}\left(\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 2/3 & 1/3 \\ 1/3 & 1/3 \end{pmatrix}\right)$$

$$= \text{tr}\begin{pmatrix} 2/3 & 1/3 \\ -1/3 & -1/3 \end{pmatrix} = 2/3 - 1/3 = 1/3$$

**Verify:** P(0) - P(1) = (2/3 × 1 + 2/3 × 1/2) - (2/3 × 1/2) = 2/3 - 1/3 = 1/3 ✓

### Example 3: Bloch Vector

Find the Bloch vector for ρ = [[3/4, 1/4], [1/4, 1/4]].

**Method:** Use ρ = (I + r⃗·σ⃗)/2

Expanding: ρ = (1/2)[[1+r_z, rₓ-irᵧ], [rₓ+irᵧ, 1-r_z]]

Comparing:
- (1+r_z)/2 = 3/4 → r_z = 1/2
- (1-r_z)/2 = 1/4 ✓
- (rₓ-irᵧ)/2 = 1/4 → rₓ = 1/2, rᵧ = 0

**Bloch vector:** r⃗ = (1/2, 0, 1/2)
**Magnitude:** |r⃗| = 1/√2 < 1 (mixed state)

### Example 4: Depolarizing Channel

Apply depolarizing channel with p = 0.1 to pure state |0⟩.

**Initial:** ρ₀ = |0⟩⟨0| = [[1, 0], [0, 0]]

**After channel:**
$$\rho' = (1-0.1)|0\rangle\langle 0| + 0.1 \frac{I}{2} = 0.9\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + 0.05\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

$$= \begin{pmatrix} 0.95 & 0 \\ 0 & 0.05 \end{pmatrix}$$

**Bloch vector:** r⃗' = (0, 0, 0.9) (contracted toward center)

### Example 5: Amplitude Damping

Apply amplitude damping with γ = 0.3 to |+⟩.

**Initial:** ρ = [[1/2, 1/2], [1/2, 1/2]]

**Kraus operators:**
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.7} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{0.3} \\ 0 & 0 \end{pmatrix}$$

**Apply:**
$$\rho' = K_0 \rho K_0^\dagger + K_1 \rho K_1^\dagger$$

$$K_0 \rho K_0^\dagger = \begin{pmatrix} 1/2 & \sqrt{0.7}/2 \\ \sqrt{0.7}/2 & 0.7/2 \end{pmatrix}$$

$$K_1 \rho K_1^\dagger = \begin{pmatrix} 0.3/2 & 0 \\ 0 & 0 \end{pmatrix}$$

$$\rho' = \begin{pmatrix} 0.65 & 0.42 \\ 0.42 & 0.35 \end{pmatrix}$$

Off-diagonal reduced but not eliminated (amplitude damping affects both).

---

## 📝 Practice Problems

### Level 1: Basic Density Matrices
1. Write the density matrix for |−⟩ = (|0⟩-|1⟩)/√2.

2. Show that ρ = I/2 is the unique density matrix with no preferred direction.

3. Compute tr(ρ²) for ρ = diag(1/2, 1/3, 1/6).

### Level 2: Expectation Values
4. For ρ = [[0.7, 0.2], [0.2, 0.3]], compute ⟨σₓ⟩, ⟨σᵧ⟩, ⟨σ_z⟩.

5. Find the Bloch vector for the state in problem 4.

6. A qubit is in state ρ with Bloch vector (0.3, 0.4, 0). What is the probability of measuring |0⟩?

### Level 3: Quantum Channels
7. Show that the depolarizing channel can be written with Kraus operators proportional to {I, X, Y, Z}.

8. Apply phase damping with γ = 0.5 to |+⟩. Compare purity before and after.

9. Prove that amplitude damping drives any state toward |0⟩ as γ → 1.

### Level 4: Theory
10. Prove: tr(ρ²) ≤ 1 with equality iff ρ is pure.

11. Show that the Bloch ball condition |r⃗| ≤ 1 is equivalent to ρ ≥ 0.

12. Derive the Lindblad equation for amplitude damping from Kraus operators.

---

## 💻 Evening Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)

# ============================================
# Density Matrix Class
# ============================================

class DensityMatrix:
    """Complete density matrix implementation"""
    
    def __init__(self, rho):
        self.rho = np.array(rho, dtype=complex)
        self.dim = self.rho.shape[0]
    
    @classmethod
    def from_pure_state(cls, psi):
        """Create density matrix from pure state vector"""
        psi = np.array(psi, dtype=complex).flatten()
        psi = psi / np.linalg.norm(psi)
        return cls(np.outer(psi, psi.conj()))
    
    @classmethod
    def from_ensemble(cls, states, probabilities):
        """Create mixed state from ensemble"""
        rho = sum(p * np.outer(psi, psi.conj()) 
                  for psi, p in zip(states, probabilities))
        return cls(rho)
    
    @classmethod
    def from_bloch_vector(cls, r):
        """Create qubit density matrix from Bloch vector"""
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        rho = (I + r[0]*X + r[1]*Y + r[2]*Z) / 2
        return cls(rho)
    
    @property
    def trace(self):
        return np.real(np.trace(self.rho))
    
    @property
    def purity(self):
        return np.real(np.trace(self.rho @ self.rho))
    
    @property
    def von_neumann_entropy(self):
        eigenvalues = np.linalg.eigvalsh(self.rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def is_valid(self, tol=1e-10):
        """Check if valid density matrix"""
        # Hermitian
        if not np.allclose(self.rho, self.rho.conj().T, atol=tol):
            return False, "Not Hermitian"
        # Positive
        if np.min(np.linalg.eigvalsh(self.rho)) < -tol:
            return False, "Not positive"
        # Normalized
        if not np.isclose(self.trace, 1, atol=tol):
            return False, f"Trace = {self.trace}"
        return True, "Valid"
    
    def is_pure(self, tol=1e-10):
        return np.isclose(self.purity, 1, atol=tol)
    
    def expectation(self, operator):
        """Compute ⟨A⟩ = tr(Aρ)"""
        return np.real(np.trace(operator @ self.rho))
    
    def bloch_vector(self):
        """Get Bloch vector for qubit"""
        if self.dim != 2:
            raise ValueError("Bloch vector only for qubits")
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        return np.array([self.expectation(X), 
                        self.expectation(Y), 
                        self.expectation(Z)])
    
    def evolve(self, H, t, hbar=1):
        """Unitary evolution under Hamiltonian H"""
        U = expm(-1j * H * t / hbar)
        return DensityMatrix(U @ self.rho @ U.conj().T)


from scipy.linalg import expm

# ============================================
# Quantum Channels
# ============================================

def depolarizing_channel(rho, p):
    """Depolarizing channel: ρ → (1-p)ρ + p·I/d"""
    d = rho.dim
    return DensityMatrix((1-p) * rho.rho + p * np.eye(d) / d)

def amplitude_damping(rho, gamma):
    """Amplitude damping channel"""
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    new_rho = K0 @ rho.rho @ K0.conj().T + K1 @ rho.rho @ K1.conj().T
    return DensityMatrix(new_rho)

def phase_damping(rho, gamma):
    """Phase damping channel"""
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]])
    new_rho = K0 @ rho.rho @ K0.conj().T + K1 @ rho.rho @ K1.conj().T
    return DensityMatrix(new_rho)

def bit_flip_channel(rho, p):
    """Bit flip channel: flips with probability p"""
    X = np.array([[0, 1], [1, 0]])
    new_rho = (1-p) * rho.rho + p * X @ rho.rho @ X
    return DensityMatrix(new_rho)

# ============================================
# Examples and Tests
# ============================================

print("=== Density Matrix Examples ===\n")

# Pure state
psi_plus = np.array([1, 1]) / np.sqrt(2)
rho_pure = DensityMatrix.from_pure_state(psi_plus)
print(f"|+⟩ density matrix:\n{rho_pure.rho}")
print(f"Purity: {rho_pure.purity:.4f}")
print(f"Is pure: {rho_pure.is_pure()}")
print(f"Bloch vector: {rho_pure.bloch_vector()}")

# Mixed state (ensemble)
psi_0 = np.array([1, 0])
psi_1 = np.array([0, 1])
rho_mixed = DensityMatrix.from_ensemble([psi_0, psi_1], [0.5, 0.5])
print(f"\n50/50 mixture of |0⟩,|1⟩:\n{rho_mixed.rho}")
print(f"Purity: {rho_mixed.purity:.4f}")
print(f"Is pure: {rho_mixed.is_pure()}")
print(f"Bloch vector: {rho_mixed.bloch_vector()}")

# ============================================
# Compare |+⟩ vs 50/50 mixture
# ============================================

print("\n=== Coherent vs Incoherent Superposition ===")

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

print(f"|+⟩ state:")
print(f"  ⟨Z⟩ = {rho_pure.expectation(Z):.4f}")
print(f"  ⟨X⟩ = {rho_pure.expectation(X):.4f}")

print(f"\n50/50 mixture:")
print(f"  ⟨Z⟩ = {rho_mixed.expectation(Z):.4f}")
print(f"  ⟨X⟩ = {rho_mixed.expectation(X):.4f}")

print("\nKey difference: Same Z statistics, different X statistics!")

# ============================================
# Quantum Channel Effects
# ============================================

print("\n=== Quantum Channel Effects ===")

rho_initial = DensityMatrix.from_pure_state(psi_plus)
print(f"Initial |+⟩: purity = {rho_initial.purity:.4f}, entropy = {rho_initial.von_neumann_entropy:.4f}")

# Apply channels
for p in [0.1, 0.3, 0.5]:
    rho_depol = depolarizing_channel(rho_initial, p)
    print(f"Depolarizing (p={p}): purity = {rho_depol.purity:.4f}, |r⃗| = {np.linalg.norm(rho_depol.bloch_vector()):.4f}")

print()
for gamma in [0.1, 0.3, 0.5]:
    rho_amp = amplitude_damping(rho_initial, gamma)
    print(f"Amplitude damp (γ={gamma}): purity = {rho_amp.purity:.4f}, bloch = {rho_amp.bloch_vector()}")

# ============================================
# Bloch Sphere Visualization
# ============================================

def plot_bloch_ball(states_dict, title="Bloch Ball"):
    """Plot density matrices on Bloch ball"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere wireframe
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightblue', alpha=0.2)
    
    # Plot states
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(states_dict)))
    for (name, rho), color in zip(states_dict.items(), colors):
        r = rho.bloch_vector()
        ax.scatter([r[0]], [r[1]], [r[2]], color=color, s=100, label=name)
        ax.quiver(0, 0, 0, r[0], r[1], r[2], color=color, alpha=0.7)
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    return fig, ax

# Compare different states
states = {
    "|0⟩": DensityMatrix.from_pure_state([1, 0]),
    "|+⟩": DensityMatrix.from_pure_state([1, 1]),
    "|+i⟩": DensityMatrix.from_pure_state([1, 1j]),
    "I/2": DensityMatrix(np.eye(2)/2),
    "0.7|0⟩+0.3I/2": depolarizing_channel(DensityMatrix.from_pure_state([1, 0]), 0.3)
}

fig, ax = plot_bloch_ball(states, "Various Quantum States")
plt.savefig('bloch_ball_states.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Decoherence Trajectory
# ============================================

def plot_decoherence_trajectory(rho_initial, channel_func, channel_name, n_steps=20):
    """Plot trajectory under repeated channel application"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightblue', alpha=0.2)
    
    # Apply channel repeatedly
    trajectory = [rho_initial.bloch_vector()]
    rho = rho_initial
    for _ in range(n_steps):
        rho = channel_func(rho, 0.1)
        trajectory.append(rho.bloch_vector())
    
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'r-', linewidth=2, label='Trajectory')
    ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]], 
              color='green', s=100, label='Start')
    ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]], 
              color='red', s=100, label='End')
    
    ax.set_title(f'Decoherence: {channel_name}')
    ax.legend()
    
    return fig

# Show decoherence trajectories
rho0 = DensityMatrix.from_pure_state([1, 1])

fig1 = plot_decoherence_trajectory(rho0, depolarizing_channel, 'Depolarizing')
plt.savefig('decoherence_depolarizing.png', dpi=150, bbox_inches='tight')

fig2 = plot_decoherence_trajectory(rho0, amplitude_damping, 'Amplitude Damping')
plt.savefig('decoherence_amplitude.png', dpi=150, bbox_inches='tight')

fig3 = plot_decoherence_trajectory(rho0, phase_damping, 'Phase Damping')
plt.savefig('decoherence_phase.png', dpi=150, bbox_inches='tight')

plt.show()

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Understand pure vs mixed states
- [ ] Construct density matrices from ensembles
- [ ] Compute expectation values with tr(Aρ)
- [ ] Work with Bloch sphere representation
- [ ] Apply quantum channels (Kraus operators)
- [ ] Model decoherence effects
- [ ] Complete computational lab
- [ ] Solve at least 6 practice problems

---

## 🔜 Preview: Tomorrow

**Day 125: Computational Lab — Advanced Linear Algebra Applications**
- SVD for quantum state analysis
- Tensor product simulations
- Density matrix evolution
- Entanglement measures
- Quantum channel implementations

---

*"The density matrix is the most complete description of a quantum system. It's the quantum equivalent of knowing everything there is to know."*
— Quantum Information Saying
