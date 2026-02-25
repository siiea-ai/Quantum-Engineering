# Day 399: Week 57 Review — Orbital Angular Momentum

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Comprehensive review |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem set |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Synthesis lab |

---

## Week 57 Summary

This week we developed the complete theory of orbital angular momentum:

### Key Concepts Mastered

| Day | Topic | Main Result |
|-----|-------|-------------|
| 393 | Classical → Quantum | L̂ = -iℏ(r × ∇) |
| 394 | Commutation Relations | [L̂ᵢ, L̂ⱼ] = iℏεᵢⱼₖL̂ₖ |
| 395 | Ladder Operators | L̂± = L̂ₓ ± iL̂ᵧ |
| 396 | Eigenvalue Spectrum | L² = ℏ²l(l+1), Lz = ℏm |
| 397 | Spherical Harmonics I | Y_l^m(θ,φ) explicit forms |
| 398 | Spherical Harmonics II | Addition theorem |

---

## Master Formula Sheet

### Angular Momentum Operators

$$\hat{L} = \hat{r} \times \hat{p} = -i\hbar(\mathbf{r} \times \nabla)$$

$$\hat{L}_z = -i\hbar\frac{\partial}{\partial\phi}$$

$$\hat{L}^2 = -\hbar^2\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right]$$

### Commutation Relations

$$[\hat{L}_i, \hat{L}_j] = i\hbar\varepsilon_{ijk}\hat{L}_k$$

$$[\hat{L}^2, \hat{L}_i] = 0$$

$$[\hat{L}_z, \hat{L}_\pm] = \pm\hbar\hat{L}_\pm$$

### Ladder Operators

$$\hat{L}_\pm = \hat{L}_x \pm i\hat{L}_y$$

$$\hat{L}^2 = \hat{L}_+\hat{L}_- + \hat{L}_z^2 - \hbar\hat{L}_z = \hat{L}_-\hat{L}_+ + \hat{L}_z^2 + \hbar\hat{L}_z$$

$$\hat{L}_\pm|l,m\rangle = \hbar\sqrt{l(l+1) - m(m\pm 1)}|l,m\pm 1\rangle$$

### Eigenvalues

$$\hat{L}^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle, \quad l = 0, 1, 2, ...$$

$$\hat{L}_z|l,m\rangle = \hbar m|l,m\rangle, \quad m = -l, ..., +l$$

### Spherical Harmonics

$$Y_l^m(\theta,\phi) = (-1)^m\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\phi}$$

**Orthonormality:**
$$\int Y_{l'}^{m'*}Y_l^m\,d\Omega = \delta_{ll'}\delta_{mm'}$$

**Addition Theorem:**
$$P_l(\cos\gamma) = \frac{4\pi}{2l+1}\sum_{m=-l}^{l}Y_l^{m*}(\hat{r}_1)Y_l^m(\hat{r}_2)$$

---

## Conceptual Understanding Check

### 1. Why do commutators matter?
[L̂ₓ, L̂ᵧ] ≠ 0 means we cannot simultaneously know Lₓ and Lᵧ precisely. We can only know L² and one component (conventionally Lᵤ).

### 2. Why integer l for orbital angular momentum?
Single-valuedness of wave functions: ψ(φ + 2π) = ψ(φ) requires e^{2πim} = 1, so m (and hence l) must be integers.

### 3. Physical meaning of ladder operators?
L̂₊ and L̂₋ change the magnetic quantum number m by ±1 while preserving l. They "step up" and "step down" through the 2l+1 states.

### 4. Why is |L| ≠ ℏl?
The magnitude is |L| = ℏ√[l(l+1)] > ℏl. This means even when m = l (maximum Lᵤ), there's still uncertainty in Lₓ and Lᵧ.

### 5. Connection to atoms?
Spherical harmonics describe the angular part of atomic wave functions. The orbital labels s, p, d, f correspond to l = 0, 1, 2, 3.

---

## Comprehensive Problem Set

### Problem 1: Commutator Calculation
Calculate [L̂², L̂ₓL̂ᵧ].

**Solution Sketch:** Use [L̂², L̂ₓ] = 0 and product rule.

### Problem 2: Matrix Representation
For l = 1, construct the 3×3 matrices for L̂ₓ, L̂ᵧ, L̂ᵤ and verify [L̂ₓ, L̂ᵧ] = iℏL̂ᵤ.

### Problem 3: Expectation Values
For the state |ψ⟩ = (|2,2⟩ + |2,0⟩ + |2,-2⟩)/√3, calculate:
a) ⟨L̂²⟩
b) ⟨L̂ᵤ⟩
c) ⟨L̂ₓ²⟩

### Problem 4: Spherical Harmonic Identity
Prove: Y_l^{-m} = (-1)^m Y_l^{m*}

### Problem 5: Hydrogen p Orbital
The 2p state of hydrogen has wave function ψ_{211} = R_{21}(r)Y_1^1(θ,φ). Calculate:
a) ⟨L̂²⟩ and ⟨L̂ᵤ⟩
b) ⟨L̂ₓ⟩
c) The probability of measuring Lᵤ = 0

---

## Computational Lab: Week Synthesis

```python
"""
Day 399 Computational Lab: Week 57 Synthesis
Complete angular momentum toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D

class OrbitalAngularMomentum:
    """
    Complete orbital angular momentum toolkit.
    """

    def __init__(self, l):
        """Initialize for given l value."""
        self.l = l
        self.dim = 2*l + 1
        self.m_values = np.arange(l, -l-1, -1)
        self._build_matrices()

    def _build_matrices(self):
        """Construct all angular momentum matrices."""
        l = self.l
        dim = self.dim

        # L_z (diagonal)
        self.Lz = np.diag(self.m_values).astype(complex)

        # L_+ (raising)
        self.Lplus = np.zeros((dim, dim), dtype=complex)
        for i in range(dim-1):
            m = self.m_values[i+1]
            self.Lplus[i, i+1] = np.sqrt(l*(l+1) - m*(m+1))

        # L_- (lowering)
        self.Lminus = self.Lplus.T.conj()

        # L_x, L_y
        self.Lx = (self.Lplus + self.Lminus) / 2
        self.Ly = (self.Lplus - self.Lminus) / (2j)

        # L^2
        self.L2 = self.Lx @ self.Lx + self.Ly @ self.Ly + self.Lz @ self.Lz

    def verify_commutators(self):
        """Verify all commutation relations."""
        hbar = 1

        results = {}

        # [Lx, Ly] = i*hbar*Lz
        comm_xy = self.Lx @ self.Ly - self.Ly @ self.Lx
        results['[Lx,Ly]=iℏLz'] = np.allclose(comm_xy, 1j*hbar*self.Lz)

        # [Ly, Lz] = i*hbar*Lx
        comm_yz = self.Ly @ self.Lz - self.Lz @ self.Ly
        results['[Ly,Lz]=iℏLx'] = np.allclose(comm_yz, 1j*hbar*self.Lx)

        # [Lz, Lx] = i*hbar*Ly
        comm_zx = self.Lz @ self.Lx - self.Lx @ self.Lz
        results['[Lz,Lx]=iℏLy'] = np.allclose(comm_zx, 1j*hbar*self.Ly)

        # [L^2, Lz] = 0
        comm_L2z = self.L2 @ self.Lz - self.Lz @ self.L2
        results['[L²,Lz]=0'] = np.allclose(comm_L2z, 0)

        # [Lz, L+] = hbar*L+
        comm_zplus = self.Lz @ self.Lplus - self.Lplus @ self.Lz
        results['[Lz,L+]=ℏL+'] = np.allclose(comm_zplus, hbar*self.Lplus)

        return results

    def expectation_value(self, state, operator):
        """Calculate expectation value <state|operator|state>."""
        state = np.array(state, dtype=complex)
        state = state / np.linalg.norm(state)  # Normalize
        return np.real(state.conj() @ operator @ state)

    def ladder_action(self, state, direction='up'):
        """Apply ladder operator to state."""
        state = np.array(state, dtype=complex)
        if direction == 'up':
            return self.Lplus @ state
        else:
            return self.Lminus @ state

def plot_angular_momentum_summary():
    """Create summary visualization."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Eigenvalue spectrum
    ax1 = fig.add_subplot(2, 2, 1)
    for l in range(4):
        m_values = np.arange(-l, l+1)
        ax1.scatter([l]*len(m_values), m_values, s=100, label=f'l={l}')
        for m in m_values:
            ax1.annotate(f'{m}', (l+0.1, m), fontsize=8)

    ax1.set_xlabel('l')
    ax1.set_ylabel('m')
    ax1.set_title('Angular Momentum Eigenvalues')
    ax1.grid(True, alpha=0.3)

    # 2. L^2 vs l
    ax2 = fig.add_subplot(2, 2, 2)
    l_vals = np.arange(0, 5)
    L2_vals = l_vals * (l_vals + 1)
    ax2.bar(l_vals, L2_vals, color='steelblue', edgecolor='black')
    ax2.set_xlabel('l')
    ax2.set_ylabel('L²/ℏ² = l(l+1)')
    ax2.set_title('Angular Momentum Magnitude')
    ax2.grid(True, alpha=0.3)

    # 3. Spherical harmonics (2D cross-section)
    ax3 = fig.add_subplot(2, 2, 3)
    theta = np.linspace(0, np.pi, 200)

    for l in range(4):
        Y = sph_harm(0, l, 0, theta)
        ax3.plot(theta * 180/np.pi, np.abs(Y)**2, label=f'$|Y_{l}^0|^2$', linewidth=2)

    ax3.set_xlabel('θ (degrees)')
    ax3.set_ylabel('$|Y_l^0|^2$')
    ax3.set_title('Spherical Harmonics (m=0) vs Polar Angle')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Ladder operator action
    ax4 = fig.add_subplot(2, 2, 4)
    L = OrbitalAngularMomentum(2)

    # Start from |2,-2> and apply L+ repeatedly
    state = np.array([0, 0, 0, 0, 1], dtype=complex)  # |2,-2>
    norms = [np.linalg.norm(state)]
    states = [state.copy()]

    for _ in range(4):
        state = L.ladder_action(state, 'up')
        norms.append(np.linalg.norm(state))
        states.append(state.copy())

    m_sequence = [-2, -1, 0, 1, 2]
    ax4.bar(m_sequence, norms, color='coral', edgecolor='black')
    ax4.set_xlabel('m value')
    ax4.set_ylabel('||L₊ⁿ|2,-2⟩||')
    ax4.set_title('Ladder Operator: Successive L₊ Applications')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week57_summary.png', dpi=150)
    plt.show()

def verify_all_relations():
    """Verify commutation relations for several l values."""
    print("Verifying Commutation Relations")
    print("=" * 50)

    for l in [1, 2, 3]:
        print(f"\nl = {l}:")
        L = OrbitalAngularMomentum(l)
        results = L.verify_commutators()
        for relation, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {relation}: {status}")

def example_calculations():
    """Demonstrate example calculations."""
    print("\nExample Calculations for l = 2")
    print("=" * 50)

    L = OrbitalAngularMomentum(2)

    # State: |2,1>
    state_21 = np.array([0, 1, 0, 0, 0], dtype=complex)

    print(f"\nState: |2,1⟩")
    print(f"  ⟨L²⟩/ℏ² = {L.expectation_value(state_21, L.L2):.4f} (theory: 6)")
    print(f"  ⟨Lz⟩/ℏ = {L.expectation_value(state_21, L.Lz):.4f} (theory: 1)")
    print(f"  ⟨Lx⟩/ℏ = {L.expectation_value(state_21, L.Lx):.4f} (theory: 0)")

    # Superposition state
    state_super = np.array([1, 0, 1, 0, 1], dtype=complex) / np.sqrt(3)

    print(f"\nState: (|2,2⟩ + |2,0⟩ + |2,-2⟩)/√3")
    print(f"  ⟨L²⟩/ℏ² = {L.expectation_value(state_super, L.L2):.4f}")
    print(f"  ⟨Lz⟩/ℏ = {L.expectation_value(state_super, L.Lz):.4f}")
    print(f"  ⟨Lx²⟩/ℏ² = {L.expectation_value(state_super, L.Lx @ L.Lx):.4f}")

if __name__ == "__main__":
    print("Day 399: Week 57 Synthesis - Orbital Angular Momentum")
    print("=" * 60)

    # Verify relations
    verify_all_relations()

    # Example calculations
    example_calculations()

    # Summary plot
    print("\nGenerating summary visualization...")
    plot_angular_momentum_summary()

    print("\nWeek 57 complete!")
```

---

## Preview: Week 58 — Spin Angular Momentum

Next week we discover that angular momentum can take **half-integer** values! This leads to:

- **Spin-1/2 particles** (electrons, protons, neutrons)
- **Pauli matrices** σₓ, σᵧ, σᵤ
- **Bloch sphere** representation
- **Direct connection to QUBITS**

The qubit is a spin-1/2 system. Week 58 is where quantum mechanics meets quantum computing most directly.

---

## Week 57 Checklist

- [ ] I can derive [L̂ᵢ, L̂ⱼ] = iℏεᵢⱼₖL̂ₖ from [x̂, p̂] = iℏ
- [ ] I understand ladder operators and their action
- [ ] I can derive the eigenvalue spectrum using algebraic methods
- [ ] I know why orbital l must be integer
- [ ] I can write and manipulate spherical harmonics
- [ ] I understand the addition theorem
- [ ] I completed all computational labs

---

**Next Week:** [Week_58_Spin/README.md](../Week_58_Spin/README.md) — Spin Angular Momentum
