# Day 151: Small Oscillations — Normal Modes

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Linearization & Normal Modes |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Linearize equations of motion near equilibrium
2. Set up the eigenvalue problem for normal modes
3. Find normal mode frequencies and eigenvectors
4. Understand the general solution as superposition of modes
5. Apply to molecular vibrations and coupled systems
6. Connect to quantum mechanics (phonons, molecular spectra)

---

## 📖 Core Content

### 1. Motivation: Why Small Oscillations?

Near stable equilibrium, any system behaves like coupled harmonic oscillators!

**Applications:**
- Molecular vibrations (IR spectroscopy)
- Crystal lattice dynamics (phonons)
- Structural mechanics
- Electrical circuits

---

### 2. Equilibrium and Stability

**Equilibrium:** ∂V/∂qᵢ = 0 at q = q₀

**Stability:** V has a minimum at q₀
- All eigenvalues of ∂²V/∂qᵢ∂qⱼ positive

**Expansion near equilibrium:**
Let ηᵢ = qᵢ - q₀ᵢ be small displacements.

$$V \approx V_0 + \frac{1}{2}\sum_{i,j} V_{ij}\eta_i\eta_j$$

where V_{ij} = ∂²V/∂qᵢ∂qⱼ|₀

---

### 3. Kinetic and Potential Energy Matrices

**Kinetic energy:**
$$T = \frac{1}{2}\sum_{i,j} M_{ij}\dot{\eta}_i\dot{\eta}_j$$

where Mᵢⱼ = mass matrix at equilibrium.

**Potential energy:**
$$V = \frac{1}{2}\sum_{i,j} K_{ij}\eta_i\eta_j$$

where Kᵢⱼ = ∂²V/∂qᵢ∂qⱼ|₀ = stiffness matrix.

**Matrix form:**
$$T = \frac{1}{2}\dot{\boldsymbol{\eta}}^T \mathbf{M} \dot{\boldsymbol{\eta}}, \quad V = \frac{1}{2}\boldsymbol{\eta}^T \mathbf{K} \boldsymbol{\eta}$$

---

### 4. Equations of Motion

**Lagrangian:** L = T - V

**Euler-Lagrange:**
$$\mathbf{M}\ddot{\boldsymbol{\eta}} + \mathbf{K}\boldsymbol{\eta} = 0$$

This is a system of coupled linear ODEs!

---

### 5. Normal Mode Ansatz

**Try:** η(t) = **a** e^{iωt} (all coordinates oscillate together)

**Substituting:**
$$(-\omega^2\mathbf{M} + \mathbf{K})\mathbf{a} = 0$$

**Non-trivial solution requires:**
$$\boxed{\det(\mathbf{K} - \omega^2\mathbf{M}) = 0}$$

This is a **generalized eigenvalue problem**!

---

### 6. Normal Mode Frequencies and Vectors

**Eigenvalue equation:**
$$\mathbf{K}\mathbf{a}_n = \omega_n^2\mathbf{M}\mathbf{a}_n$$

- ω_n² are the eigenvalues (normal mode frequencies squared)
- **a**_n are eigenvectors (normal mode shapes)

**Properties:**
1. For stable equilibrium: all ω_n² > 0
2. Eigenvectors orthogonal: **a**_m^T **M** **a**_n = 0 (m ≠ n)
3. n modes for n degrees of freedom

---

### 7. General Solution

**Superposition of normal modes:**
$$\boldsymbol{\eta}(t) = \sum_{n=1}^{N} c_n\mathbf{a}_n\cos(\omega_n t + \phi_n)$$

**2N constants** (c_n, φ_n) determined by initial conditions.

**Normal coordinates:** Define Q_n such that each satisfies Q̈_n + ω_n²Q_n = 0 independently.

---

### 8. Example: Two Coupled Pendulums

**Setup:** Two identical pendulums (mass m, length L) coupled by spring k.

**Displacements:** θ₁, θ₂ from vertical

**Lagrangian:**
$$L = \frac{1}{2}mL^2(\dot{\theta}_1^2 + \dot{\theta}_2^2) - \frac{1}{2}mgL(\theta_1^2 + \theta_2^2) - \frac{1}{2}kL^2(\theta_1 - \theta_2)^2$$

**Matrices:**
$$\mathbf{M} = mL^2\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad \mathbf{K} = \begin{pmatrix} mgL + kL^2 & -kL^2 \\ -kL^2 & mgL + kL^2 \end{pmatrix}$$

**Eigenvalue problem:**
$$\det\begin{pmatrix} mgL + kL^2 - \omega^2 mL^2 & -kL^2 \\ -kL^2 & mgL + kL^2 - \omega^2 mL^2 \end{pmatrix} = 0$$

**Solutions:**
$$\omega_1^2 = \frac{g}{L}, \quad \omega_2^2 = \frac{g}{L} + \frac{2k}{m}$$

**Mode shapes:**
- Mode 1: **a**₁ = (1, 1) — in-phase oscillation
- Mode 2: **a**₂ = (1, -1) — out-of-phase oscillation

---

### 9. 🔬 Quantum Connection

**Classical → Quantum:**

| Classical | Quantum |
|-----------|---------|
| Normal mode ωₙ | Phonon/photon energy ℏωₙ |
| Amplitude aₙ | Creation/annihilation operators |
| T + V | Harmonic oscillator Hamiltonian |
| Mode superposition | Fock states |

**Molecular vibrations:**
- IR spectroscopy measures normal mode frequencies
- Selection rules from quantum mechanics
- Zero-point energy: E₀ = Σ ½ℏωₙ

---

## 🔧 Practice Problems

### Level 1
1. Find normal modes of two masses m connected by springs k-2k-k to walls.
2. A linear triatomic molecule (masses m-M-m). Find normal mode frequencies.

### Level 2
3. Three coupled pendulums in a row. Find all normal modes.
4. CO₂ molecule: Find the symmetric and antisymmetric stretch frequencies.

### Level 3
5. Prove that normal mode vectors are M-orthogonal.
6. For a circular chain of N masses and springs, find all normal modes.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def normal_modes_analysis():
    """Analyze normal modes of coupled oscillator systems."""
    
    print("=" * 70)
    print("NORMAL MODES OF COUPLED OSCILLATORS")
    print("=" * 70)
    
    # System 1: Two coupled pendulums
    print("\n1. Two Coupled Pendulums")
    print("-" * 40)
    
    m, L, g, k = 1.0, 1.0, 10.0, 2.0
    
    M = m * L**2 * np.eye(2)
    K = np.array([
        [m*g*L + k*L**2, -k*L**2],
        [-k*L**2, m*g*L + k*L**2]
    ])
    
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(K, M)
    frequencies = np.sqrt(eigenvalues)
    
    print(f"Normal mode frequencies: ω₁ = {frequencies[0]:.4f}, ω₂ = {frequencies[1]:.4f}")
    print(f"Expected: ω₁ = {np.sqrt(g/L):.4f}, ω₂ = {np.sqrt(g/L + 2*k/m):.4f}")
    print(f"\nMode shapes:")
    print(f"  Mode 1: {eigenvectors[:, 0]}")
    print(f"  Mode 2: {eigenvectors[:, 1]}")
    
    # Simulate and visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time evolution
    t = np.linspace(0, 10, 500)
    
    # Initial condition: displace first pendulum
    eta0 = np.array([1, 0])
    v0 = np.array([0, 0])
    
    # Project onto normal modes
    c1 = np.dot(eigenvectors[:, 0], M @ eta0)
    c2 = np.dot(eigenvectors[:, 1], M @ eta0)
    
    # Normalize
    n1 = np.dot(eigenvectors[:, 0], M @ eigenvectors[:, 0])
    n2 = np.dot(eigenvectors[:, 1], M @ eigenvectors[:, 1])
    c1 /= n1
    c2 /= n2
    
    eta1 = c1 * eigenvectors[0, 0] * np.cos(frequencies[0]*t) + c2 * eigenvectors[0, 1] * np.cos(frequencies[1]*t)
    eta2 = c1 * eigenvectors[1, 0] * np.cos(frequencies[0]*t) + c2 * eigenvectors[1, 1] * np.cos(frequencies[1]*t)
    
    axes[0, 0].plot(t, eta1, 'b-', lw=2, label='θ₁')
    axes[0, 0].plot(t, eta2, 'r-', lw=2, label='θ₂')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Angle')
    axes[0, 0].set_title('Coupled Pendulums: Beats!')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mode visualization
    ax = axes[0, 1]
    modes = ['In-phase\n(ω₁)', 'Out-of-phase\n(ω₂)']
    for i, mode in enumerate(modes):
        y = [1, 2]
        x = eigenvectors[:, i]
        ax.barh(y, x, height=0.3, left=i*3, label=mode)
        ax.annotate(mode, (i*3 + 0.5, 2.5), ha='center')
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Pendulum 1', 'Pendulum 2'])
    ax.set_title('Normal Mode Shapes')
    ax.grid(True, alpha=0.3)
    
    # System 2: Three masses on springs
    print("\n2. Three Masses on Springs")
    print("-" * 40)
    
    m = 1.0
    k = 1.0
    
    M3 = m * np.eye(3)
    K3 = k * np.array([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ])
    
    eigenvalues3, eigenvectors3 = eigh(K3, M3)
    frequencies3 = np.sqrt(eigenvalues3)
    
    print(f"Normal mode frequencies: {frequencies3}")
    
    # Visualize modes
    ax = axes[1, 0]
    x_pos = [0, 1, 2]
    colors = ['blue', 'green', 'red']
    
    for i in range(3):
        offset = i * 0.4
        mode = eigenvectors3[:, i]
        mode = mode / np.max(np.abs(mode)) * 0.3
        ax.plot(x_pos, [offset]*3, 'ko-', lw=2, markersize=15)
        ax.quiver(x_pos, [offset]*3, [0]*3, mode, angles='xy', scale_units='xy', 
                  scale=1, color=colors[i], width=0.02)
        ax.text(-0.5, offset, f'ω = {frequencies3[i]:.3f}', va='center')
    
    ax.set_xlim(-1, 3)
    ax.set_ylim(-0.2, 1.2)
    ax.set_title('Three-Mass System: Normal Modes')
    ax.set_xlabel('Position')
    ax.axis('off')
    
    # Dispersion relation preview
    ax = axes[1, 1]
    N = 20
    k_wave = np.linspace(0, np.pi, 100)
    omega_k = 2 * np.sqrt(k/m) * np.abs(np.sin(k_wave/2))
    
    ax.plot(k_wave, omega_k, 'b-', lw=2)
    ax.set_xlabel('Wave number k')
    ax.set_ylabel('Frequency ω')
    ax.set_title('Dispersion Relation (Long Chain)\nω = 2√(k/m)|sin(ka/2)|')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('normal_modes.png', dpi=150)
    plt.show()

normal_modes_analysis()
```

---

## 📝 Summary

### Normal Mode Analysis Procedure

1. Find equilibrium: ∂V/∂qᵢ = 0
2. Expand T and V to quadratic order
3. Form matrices **M** and **K**
4. Solve det(**K** - ω²**M**) = 0
5. Find eigenvectors **a**ₙ
6. General solution: η(t) = Σ cₙ**a**ₙ cos(ωₙt + φₙ)

### Key Properties

| Property | Formula |
|----------|---------|
| Secular equation | det(**K** - ω²**M**) = 0 |
| Orthogonality | **a**ₘᵀ**M****a**ₙ = δₘₙ |
| Stability | All ω² > 0 |

---

## ✅ Daily Checklist

- [ ] Linearize equations near equilibrium
- [ ] Set up generalized eigenvalue problem
- [ ] Find normal mode frequencies
- [ ] Find normal mode shapes
- [ ] Construct general solution
- [ ] Apply to coupled systems

---

## 🔮 Preview: Day 152

Tomorrow we introduce **Rigid Body Motion** — the kinematics and dynamics of extended objects!
