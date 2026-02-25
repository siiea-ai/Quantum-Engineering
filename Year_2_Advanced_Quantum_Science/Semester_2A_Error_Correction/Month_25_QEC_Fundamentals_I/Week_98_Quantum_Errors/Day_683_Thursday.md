# Day 683: Phase Damping and Combined Noise Models

## Week 98: Quantum Errors | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Phase Damping & Combined Models |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 683, you will be able to:

1. **Understand pure dephasing (T₂) processes** and their Kraus representation
2. **Distinguish T₁, T₂, and T₂*** in real hardware
3. **Combine multiple noise channels** for realistic error models
4. **Analyze the combined amplitude-phase damping** (thermal relaxation)
5. **Select appropriate noise models** for QEC analysis
6. **Connect Pauli twirling** to depolarizing approximation

---

## Phase Damping: Pure Dephasing

### Physical Motivation

**Pure dephasing** occurs when the qubit loses phase coherence without losing energy. This happens due to:
- Fluctuating magnetic fields
- Charge noise (in charge qubits)
- Phonon scattering
- Any process that randomizes the relative phase between $|0\rangle$ and $|1\rangle$

Unlike amplitude damping, **no energy is exchanged** with the environment.

### Kraus Representation

**Form 1: Projection operators**

$$E_0 = \sqrt{1-\lambda}\,I, \quad E_1 = \sqrt{\lambda}\,|0\rangle\langle 0|, \quad E_2 = \sqrt{\lambda}\,|1\rangle\langle 1|$$

**Verify completeness:**
$$E_0^\dagger E_0 + E_1^\dagger E_1 + E_2^\dagger E_2 = (1-\lambda)I + \lambda|0\rangle\langle 0| + \lambda|1\rangle\langle 1| = I \quad \checkmark$$

**Form 2: Equivalent Z-error form**

$$E_0 = \sqrt{1-\lambda/2}\,I, \quad E_1 = \sqrt{\lambda/2}\,Z$$

This shows phase damping is related to random Z errors!

### Action on Density Matrix

For $\rho = \begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix}$:

$$\boxed{\mathcal{E}_{PD}(\rho) = \begin{pmatrix} \rho_{00} & (1-\lambda)\rho_{01} \\ (1-\lambda)\rho_{10} & \rho_{11} \end{pmatrix}}$$

**Key effects:**
- **Populations unchanged:** $P(0)$, $P(1)$ preserved
- **Coherence decays:** Off-diagonal elements shrink by factor $(1-\lambda)$

### Time Dependence and T₂

The dephasing parameter relates to T₂ time:

$$\lambda(t) = 1 - e^{-t/T_2}$$

For the coherence element:
$$\rho_{01}(t) = \rho_{01}(0) \cdot e^{-t/T_2}$$

---

## T₁, T₂, and T₂*: The Complete Picture

### Definitions

| Parameter | Physical Process | Effect |
|-----------|-----------------|--------|
| **T₁** | Energy relaxation | $\|1\rangle \to \|0\rangle$ decay |
| **T₂** | Total decoherence | Coherence decay (intrinsic) |
| **T₂*** | Inhomogeneous dephasing | Coherence decay (reversible + irreversible) |

### The T₂ Constraint

A fundamental result:

$$\boxed{T_2 \leq 2T_1}$$

**Why?** Amplitude damping causes both population decay (T₁) AND coherence decay. Even without pure dephasing, coherence decays due to T₁.

**Derivation:** Under amplitude damping alone, coherence evolves as:
$$\rho_{01}(t) = \rho_{01}(0) \cdot \sqrt{1-\gamma(t)} = \rho_{01}(0) \cdot e^{-t/2T_1}$$

This gives an effective $T_2^{(AD)} = 2T_1$.

### Separating T₁ and T₂ Contributions

Total decoherence rate:
$$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$

where $T_\phi$ is the **pure dephasing time**.

Solving for pure dephasing:
$$\frac{1}{T_\phi} = \frac{1}{T_2} - \frac{1}{2T_1}$$

---

## Combined Noise Models

### Amplitude Damping + Phase Damping

Real qubits experience both T₁ relaxation and T₂ dephasing simultaneously.

**Combined channel:**
$$\mathcal{E}_{combined} = \mathcal{E}_{PD} \circ \mathcal{E}_{AD}$$

**Effective Kraus operators (4 operators):**
$$E_0 = \sqrt{1-\lambda_\phi}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$$
$$E_1 = \sqrt{1-\lambda_\phi}\begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
$$E_2 = \sqrt{\lambda_\phi}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$
$$E_3 = \sqrt{\lambda_\phi}\begin{pmatrix} 0 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$$

### Generalized Amplitude Damping (Thermal)

At finite temperature, both decay ($|1\rangle \to |0\rangle$) and excitation ($|0\rangle \to |1\rangle$) occur:

$$E_0 = \sqrt{p}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \sqrt{p}\begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
$$E_2 = \sqrt{1-p}\begin{pmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{pmatrix}, \quad E_3 = \sqrt{1-p}\begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}$$

where $p = n_{th}/(2n_{th}+1)$ relates to thermal occupation $n_{th}$.

### Bit-Flip + Phase-Flip (Independent)

If X and Z errors occur independently with probabilities $p_x$ and $p_z$:

$$\mathcal{E}(\rho) = (1-p_x)(1-p_z)\rho + p_x(1-p_z)X\rho X + (1-p_x)p_z Z\rho Z + p_x p_z Y\rho Y$$

Note: The Y error probability is $p_x p_z$, not independent!

---

## Pauli Twirling and Effective Depolarization

### Pauli Twirling

**Pauli twirling** is a technique to convert any single-qubit noise channel into a Pauli channel:

$$\mathcal{E}_{twirled}(\rho) = \frac{1}{4}\sum_{P \in \{I,X,Y,Z\}} P\mathcal{E}(P\rho P^\dagger)P^\dagger$$

**Result:** The twirled channel has the form:
$$\mathcal{E}_{twirled}(\rho) = (1-p_x-p_y-p_z)\rho + p_x X\rho X + p_y Y\rho Y + p_z Z\rho Z$$

### Why Twirl?

1. **Simplifies analysis:** Pauli channels are diagonal in the Pauli basis
2. **Preserves average fidelity:** $\bar{F}(\mathcal{E}_{twirled}) = \bar{F}(\mathcal{E})$
3. **Worst-case bound:** If code corrects twirled errors, it corrects original errors

### Depolarizing as Worst-Case

If we additionally symmetrize over Clifford group (Clifford twirling), any noise becomes depolarizing:

$$\mathcal{E}_{dep}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

This is why depolarizing noise is used for worst-case analysis in QEC!

---

## Realistic Hardware Noise

### Superconducting Qubits

| Noise Source | Type | Typical Parameter |
|--------------|------|-------------------|
| Energy relaxation | Amplitude damping | T₁ = 50-100 μs |
| Dephasing | Phase damping | T₂ = 50-100 μs |
| Gate errors | Depolarizing-like | 0.1-0.5% |
| Leakage | Non-CPTP | 0.01-0.1% |
| Crosstalk | Correlated | Varies |

### Trapped Ions

| Noise Source | Type | Typical Parameter |
|--------------|------|-------------------|
| Motional heating | Amplitude damping | T₁ >> seconds |
| Magnetic field noise | Phase damping | T₂ = 1-100 ms |
| Gate errors | Depolarizing-like | 0.1-0.5% |
| Measurement errors | Classical | ~0.5% |

### Noise Hierarchy

For most platforms:
$$T_1 \sim T_2 \gg t_{gate}$$

This means gate errors (not idle errors) often dominate.

---

## Selecting Noise Models for QEC

### For Theoretical Analysis

| Goal | Recommended Model |
|------|-------------------|
| Worst-case threshold | Depolarizing |
| Realistic benchmark | Amplitude + phase damping |
| Analytic tractability | Pauli (bit-flip or phase-flip) |
| Numerical simulation | Full error model |

### Model Complexity vs Insight

```
Simple                                          Complex
  |                                                |
  Bit-flip → Phase-flip → Pauli → Depol → AD+PD → Full model
      ↓           ↓          ↓       ↓       ↓         ↓
  Analytic   Dual codes   CSS  Worst-  Real   Numerical
  solutions              design  case  device   only
```

### Pauli Approximation Validity

Pauli (including depolarizing) approximation is valid when:
1. **Gates are much faster than decoherence:** $t_{gate} \ll T_1, T_2$
2. **Errors are small:** $p \ll 1$
3. **No leakage:** System stays in computational subspace
4. **No correlations:** Errors on different qubits are independent

---

## Worked Examples

### Example 1: Pure Dephasing on a Superposition

**Problem:** Apply phase damping ($\lambda = 0.3$) to $|+\rangle$.

**Solution:**

$$|+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Phase damping preserves diagonals, shrinks off-diagonals:

$$\mathcal{E}_{PD}(\rho) = \frac{1}{2}\begin{pmatrix} 1 & (1-0.3) \\ (1-0.3) & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0.7 \\ 0.7 & 1 \end{pmatrix}$$

The coherence dropped from 0.5 to 0.35.

### Example 2: Determining T_φ from Measurements

**Problem:** A qubit has measured T₁ = 80 μs and T₂ = 60 μs. What is T_φ?

**Solution:**

Using $\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$:

$$\frac{1}{T_\phi} = \frac{1}{60} - \frac{1}{160} = \frac{160 - 60}{60 \times 160} = \frac{100}{9600} = \frac{1}{96}$$

$$T_\phi = 96 \,\mu\text{s}$$

Check: $\frac{1}{2 \times 80} + \frac{1}{96} = \frac{1}{160} + \frac{1}{96} = \frac{96 + 160}{160 \times 96} = \frac{256}{15360} = \frac{1}{60}$ ✓

### Example 3: Combined Error Probability

**Problem:** If X errors occur with probability $p_x = 0.01$ and Z errors with probability $p_z = 0.02$ independently, what's the Y error probability?

**Solution:**

For independent X and Z:
- Probability of X but not Z: $p_x(1-p_z) = 0.01 \times 0.98 = 0.0098$
- Probability of Z but not X: $(1-p_x)p_z = 0.99 \times 0.02 = 0.0198$
- Probability of both (gives Y = iXZ): $p_x p_z = 0.01 \times 0.02 = 0.0002$
- Probability of neither: $(1-p_x)(1-p_z) = 0.99 \times 0.98 = 0.9702$

Y error probability: **0.0002** (very small!)

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** Apply phase damping ($\lambda = 0.5$) to the state $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$.

**A.2** A qubit has T₁ = 100 μs and T_φ = 200 μs. Calculate T₂.

**A.3** For the combined bit+phase flip channel with $p_x = 0.05$, $p_z = 0.03$, write out the full operator-sum representation.

### Problem Set B: Intermediate

**B.1** Prove that phase damping is a unital channel.

**B.2** Show that for amplitude damping, the coherence decays as $e^{-t/2T_1}$, giving an effective $T_2^{(AD)} = 2T_1$.

**B.3** If we apply $n$ rounds of phase damping with parameter $\lambda$, show the effective parameter is $\lambda_{eff} = 1 - (1-\lambda)^n$.

### Problem Set C: Challenging

**C.1** Derive the Pauli twirl of amplitude damping. What are the resulting Pauli error probabilities?

**C.2** Show that the composition of amplitude damping (γ) followed by phase damping (λ) is NOT commutative: $\mathcal{E}_{PD} \circ \mathcal{E}_{AD} \neq \mathcal{E}_{AD} \circ \mathcal{E}_{PD}$.

**C.3** Design a noise model that violates T₂ ≤ 2T₁ and explain why it's unphysical.

---

## Computational Lab: Combined Noise Analysis

```python
"""
Day 683 Computational Lab: Phase Damping and Combined Noise
==========================================================

Analyzing T2 dephasing and realistic combined noise models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# =============================================================================
# Part 1: Basic Definitions
# =============================================================================

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def density_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]:
    """Convert density matrix to Bloch coordinates."""
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    return x, y, z

def bloch_to_density(x: float, y: float, z: float) -> np.ndarray:
    """Convert Bloch coordinates to density matrix."""
    return 0.5 * (I + x*X + y*Y + z*Z)

print("=" * 60)
print("PART 1: Phase Damping Channel")
print("=" * 60)

# =============================================================================
# Part 2: Phase Damping Implementation
# =============================================================================

def phase_damping_channel(rho: np.ndarray, lam: float) -> np.ndarray:
    """
    Apply phase damping with dephasing parameter lambda.
    Off-diagonal elements shrink by factor (1-lambda).
    """
    result = rho.copy()
    result[0, 1] *= (1 - lam)
    result[1, 0] *= (1 - lam)
    return result

def phase_damping_kraus(rho: np.ndarray, lam: float) -> np.ndarray:
    """Phase damping using Kraus operators."""
    E0 = np.sqrt(1 - lam) * I
    E1 = np.sqrt(lam) * np.array([[1, 0], [0, 0]], dtype=complex)
    E2 = np.sqrt(lam) * np.array([[0, 0], [0, 1]], dtype=complex)
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T + E2 @ rho @ E2.conj().T

# Test equivalence
rho_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
lam_test = 0.3

out1 = phase_damping_channel(rho_plus, lam_test)
out2 = phase_damping_kraus(rho_plus, lam_test)

print(f"\nPhase damping (λ={lam_test}) on |+⟩:")
print(f"  Direct formula matches Kraus: {np.allclose(out1, out2)}")
print(f"  Input coherence:  {np.abs(rho_plus[0,1]):.4f}")
print(f"  Output coherence: {np.abs(out1[0,1]):.4f}")
print(f"  Expected:         {(1-lam_test)*np.abs(rho_plus[0,1]):.4f}")

# Verify unital
print(f"\nUnital check (E(I) = I): {np.allclose(phase_damping_channel(I, lam_test), I)}")

# =============================================================================
# Part 3: T1, T2, T_phi Relationships
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: T1, T2, T_phi Relationships")
print("=" * 60)

def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply amplitude damping."""
    E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T

def combined_AD_PD(rho: np.ndarray, gamma: float, lam_phi: float) -> np.ndarray:
    """Combined amplitude damping then phase damping."""
    rho_ad = amplitude_damping_channel(rho, gamma)
    return phase_damping_channel(rho_ad, lam_phi)

# Example calculation
T1 = 80e-6  # 80 μs
T2 = 60e-6  # 60 μs
T_phi = 1 / (1/T2 - 1/(2*T1))

print(f"\nExample hardware parameters:")
print(f"  T1 = {T1*1e6:.1f} μs")
print(f"  T2 = {T2*1e6:.1f} μs")
print(f"  T_phi = {T_phi*1e6:.1f} μs (calculated)")
print(f"\n  Verify: 1/T2 = 1/(2T1) + 1/T_phi")
print(f"  1/{T2*1e6:.1f} = 1/{2*T1*1e6:.1f} + 1/{T_phi*1e6:.1f}")
print(f"  {1/T2:.6f} ≈ {1/(2*T1) + 1/T_phi:.6f}")

# =============================================================================
# Part 4: Time Evolution Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Coherence Decay Comparison")
print("=" * 60)

# Physical parameters
T1 = 100e-6  # 100 μs
T_phi = 150e-6  # 150 μs (pure dephasing)
T2 = 1 / (1/(2*T1) + 1/T_phi)  # ~86 μs

print(f"\nSimulation parameters:")
print(f"  T1 = {T1*1e6:.1f} μs")
print(f"  T_phi = {T_phi*1e6:.1f} μs")
print(f"  T2 = {T2*1e6:.1f} μs")

times = np.linspace(0, 5*T2, 100)
rho_init = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)  # |+⟩

# Track coherence under different noise models
coherence_T1_only = []
coherence_T2_only = []
coherence_combined = []

for t in times:
    gamma = 1 - np.exp(-t/T1)
    lam_T2 = 1 - np.exp(-t/T2)
    lam_phi = 1 - np.exp(-t/T_phi)

    # T1 only (amplitude damping)
    rho_T1 = amplitude_damping_channel(rho_init, gamma)
    coherence_T1_only.append(np.abs(rho_T1[0, 1]))

    # T2 only (pure phase damping at rate 1/T2)
    rho_T2 = phase_damping_channel(rho_init, lam_T2)
    coherence_T2_only.append(np.abs(rho_T2[0, 1]))

    # Combined (AD + PD at rate T_phi)
    rho_comb = combined_AD_PD(rho_init, gamma, lam_phi)
    coherence_combined.append(np.abs(rho_comb[0, 1]))

# Analytical predictions
coherence_T1_theory = 0.5 * np.exp(-times/(2*T1))
coherence_T2_theory = 0.5 * np.exp(-times/T2)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(times*1e6, coherence_T1_only, 'b-', label=f'T1 only (AD)', linewidth=2)
plt.plot(times*1e6, coherence_T1_theory, 'b--', alpha=0.5, label=f'Theory: exp(-t/2T1)')
plt.plot(times*1e6, coherence_combined, 'r-', label='T1 + T_phi combined', linewidth=2)
plt.plot(times*1e6, coherence_T2_theory, 'r--', alpha=0.5, label=f'Theory: exp(-t/T2)')
plt.xlabel('Time (μs)')
plt.ylabel('|ρ₀₁|')
plt.title('Coherence Decay: AD vs Combined')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0.5/np.e, color='gray', linestyle=':', alpha=0.5)

# Population decay
plt.subplot(1, 2, 2)
rho_1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩

pop_AD = []
pop_combined = []
for t in times:
    gamma = 1 - np.exp(-t/T1)
    lam_phi = 1 - np.exp(-t/T_phi)
    rho_ad = amplitude_damping_channel(rho_1, gamma)
    rho_comb = combined_AD_PD(rho_1, gamma, lam_phi)
    pop_AD.append(rho_ad[1, 1].real)
    pop_combined.append(rho_comb[1, 1].real)

plt.plot(times*1e6, pop_AD, 'b-', label='T1 only', linewidth=2)
plt.plot(times*1e6, pop_combined, 'r-', label='T1 + T_phi', linewidth=2)
plt.plot(times*1e6, np.exp(-times/T1), 'k--', alpha=0.5, label='exp(-t/T1)')
plt.xlabel('Time (μs)')
plt.ylabel('P(|1⟩)')
plt.title('Population Decay (from |1⟩)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_683_t1_t2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_683_t1_t2_comparison.png")

# =============================================================================
# Part 5: Pauli Twirling
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Pauli Twirling")
print("=" * 60)

def pauli_twirl(channel_func, rho: np.ndarray, *args) -> np.ndarray:
    """Apply Pauli twirl to a channel."""
    paulis = [I, X, Y, Z]
    result = np.zeros_like(rho)
    for P in paulis:
        # Apply P, then channel, then P†
        rho_P = P @ rho @ P.conj().T
        rho_out = channel_func(rho_P, *args)
        result += P @ rho_out @ P.conj().T
    return result / 4

# Twirl amplitude damping
gamma = 0.2
rho_test = bloch_to_density(0.6, 0.3, 0.4)

rho_AD = amplitude_damping_channel(rho_test, gamma)
rho_twirled = pauli_twirl(amplitude_damping_channel, rho_test, gamma)

print(f"\nAmplitude damping (γ={gamma}) with Pauli twirl:")
print(f"  Original AD output:")
x1, y1, z1 = density_to_bloch(rho_AD)
print(f"    Bloch: ({x1:.4f}, {y1:.4f}, {z1:.4f})")
print(f"  Twirled AD output:")
x2, y2, z2 = density_to_bloch(rho_twirled)
print(f"    Bloch: ({x2:.4f}, {y2:.4f}, {z2:.4f})")

# The twirled version should be Pauli-diagonal
print(f"\n  Twirled density matrix:")
print(f"    {np.round(rho_twirled, 4)}")

# Extract Pauli error probabilities
def extract_pauli_probabilities(rho_in: np.ndarray, rho_out: np.ndarray) -> dict:
    """Extract effective Pauli error probabilities from channel action."""
    # For Pauli channel: E(ρ) = (1-px-py-pz)ρ + px XρX + py YρY + pz ZρZ
    # Bloch vector transforms: (x,y,z) → ((1-2py-2pz)x, (1-2px-2pz)y, (1-2px-2py)z)
    x_in, y_in, z_in = density_to_bloch(rho_in)
    x_out, y_out, z_out = density_to_bloch(rho_out)

    if abs(x_in) > 0.1:
        shrink_x = x_out / x_in
    else:
        shrink_x = 1

    if abs(y_in) > 0.1:
        shrink_y = y_out / y_in
    else:
        shrink_y = 1

    if abs(z_in) > 0.1:
        shrink_z = z_out / z_in
    else:
        shrink_z = 1

    return {'shrink_x': shrink_x, 'shrink_y': shrink_y, 'shrink_z': shrink_z}

probs = extract_pauli_probabilities(rho_test, rho_twirled)
print(f"\n  Bloch shrinking factors:")
print(f"    x: {probs['shrink_x']:.4f}")
print(f"    y: {probs['shrink_y']:.4f}")
print(f"    z: {probs['shrink_z']:.4f}")

# =============================================================================
# Part 6: Combined Noise Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Combined Noise Model Visualization")
print("=" * 60)

fig = plt.figure(figsize=(15, 5))

# Sample points on Bloch sphere
np.random.seed(42)
n_points = 50
points_init = []
for _ in range(n_points):
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points_init.append((x, y, z))

# Parameters
gamma = 0.3
lam = 0.4

# Plot 1: Phase damping only
ax1 = fig.add_subplot(131)
points_PD = []
for (x, y, z) in points_init:
    rho = bloch_to_density(x, y, z)
    rho_out = phase_damping_channel(rho, lam)
    points_PD.append(density_to_bloch(rho_out))

x_init = [p[0] for p in points_init]
z_init = [p[2] for p in points_init]
x_PD = [p[0] for p in points_PD]
z_PD = [p[2] for p in points_PD]

ax1.scatter(x_init, z_init, alpha=0.3, s=20, label='Initial')
ax1.scatter(x_PD, z_PD, alpha=0.7, s=20, label='After PD')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax1.add_patch(circle)
ax1.set_xlim(-1.3, 1.3)
ax1.set_ylim(-1.3, 1.3)
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Z')
ax1.set_title(f'Phase Damping (λ={lam})\nShrinks X,Y; preserves Z')
ax1.legend()

# Plot 2: Amplitude damping only
ax2 = fig.add_subplot(132)
points_AD = []
for (x, y, z) in points_init:
    rho = bloch_to_density(x, y, z)
    rho_out = amplitude_damping_channel(rho, gamma)
    points_AD.append(density_to_bloch(rho_out))

x_AD = [p[0] for p in points_AD]
z_AD = [p[2] for p in points_AD]

ax2.scatter(x_init, z_init, alpha=0.3, s=20, label='Initial')
ax2.scatter(x_AD, z_AD, alpha=0.7, s=20, label='After AD')
ax2.scatter([0], [1], color='red', s=100, marker='*', label='|0⟩')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax2.add_patch(circle)
ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax2.set_aspect('equal')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')
ax2.set_title(f'Amplitude Damping (γ={gamma})\nShrinks + shifts to |0⟩')
ax2.legend()

# Plot 3: Combined
ax3 = fig.add_subplot(133)
points_comb = []
for (x, y, z) in points_init:
    rho = bloch_to_density(x, y, z)
    rho_out = combined_AD_PD(rho, gamma, lam)
    points_comb.append(density_to_bloch(rho_out))

x_comb = [p[0] for p in points_comb]
z_comb = [p[2] for p in points_comb]

ax3.scatter(x_init, z_init, alpha=0.3, s=20, label='Initial')
ax3.scatter(x_comb, z_comb, alpha=0.7, s=20, label='After AD+PD')
ax3.scatter([0], [1], color='red', s=100, marker='*', label='|0⟩')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax3.add_patch(circle)
ax3.set_xlim(-1.3, 1.3)
ax3.set_ylim(-1.3, 1.3)
ax3.set_aspect('equal')
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_title(f'Combined AD(γ={gamma})+PD(λ={lam})\nRealistic T1+T2 noise')
ax3.legend()

plt.tight_layout()
plt.savefig('day_683_combined_noise.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_683_combined_noise.png")

# =============================================================================
# Part 7: Independent X and Z Errors
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Independent X and Z Errors")
print("=" * 60)

def independent_XZ_channel(rho: np.ndarray, px: float, pz: float) -> np.ndarray:
    """
    Channel with independent X and Z errors.
    Y error occurs with probability px * pz (when both happen).
    """
    p_I = (1 - px) * (1 - pz)
    p_X = px * (1 - pz)
    p_Z = (1 - px) * pz
    p_Y = px * pz

    return (p_I * rho +
            p_X * X @ rho @ X +
            p_Y * Y @ rho @ Y +
            p_Z * Z @ rho @ Z)

px, pz = 0.05, 0.03
print(f"\nIndependent X (px={px}) and Z (pz={pz}) errors:")
print(f"  P(no error) = {(1-px)*(1-pz):.4f}")
print(f"  P(X only)   = {px*(1-pz):.4f}")
print(f"  P(Z only)   = {(1-px)*pz:.4f}")
print(f"  P(Y = XZ)   = {px*pz:.4f}")

# Compare to depolarizing with same total error rate
p_total = 1 - (1-px)*(1-pz)
print(f"\n  Total error probability: {p_total:.4f}")
print(f"  If depolarizing at same rate: pX = pY = pZ = {p_total/3:.4f}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Noise Model Selection Guide")
print("=" * 60)

summary = """
┌──────────────────────────────────────────────────────────────────────────┐
│                    Noise Model Selection Guide                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ PHASE DAMPING (T₂ dephasing):                                             │
│   • Off-diagonals decay: ρ₀₁ → (1-λ)ρ₀₁                                  │
│   • Populations preserved                                                 │
│   • Unital: E(I) = I                                                      │
│   • Physical: magnetic noise, charge noise                                │
│                                                                           │
│ COMBINED T₁ + T₂:                                                         │
│   • 1/T₂ = 1/(2T₁) + 1/T_φ                                               │
│   • T₂ ≤ 2T₁ always!                                                      │
│   • Use for realistic device modeling                                     │
│                                                                           │
│ NOISE MODEL HIERARCHY:                                                    │
│   Simple → Complex:                                                       │
│   Pauli → Depolarizing → AD+PD → Full model                              │
│                                                                           │
│ SELECTION RULES:                                                          │
│   • Threshold analysis: Depolarizing (worst case)                         │
│   • Device modeling: AD + PD with measured T₁, T₂                        │
│   • Analytic work: Pauli channels                                         │
│   • Accurate simulation: Full hardware model                              │
│                                                                           │
├──────────────────────────────────────────────────────────────────────────┤
│ PAULI TWIRLING: Converts any noise → Pauli noise                         │
│   • Preserves average fidelity                                            │
│   • Enables simpler analysis                                              │
│   • If code corrects twirled errors, it corrects original                │
└──────────────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("\n✅ Day 683 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Phase damping action | $\rho_{01} \to (1-\lambda)\rho_{01}$, diagonals unchanged |
| T₂ constraint | $T_2 \leq 2T_1$ |
| Decoherence decomposition | $\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$ |
| Independent X+Z errors | $p_Y = p_x \cdot p_z$ |
| Pauli twirl | $\mathcal{E}_{twirl}(\rho) = \frac{1}{4}\sum_P P\mathcal{E}(P\rho P^\dagger)P^\dagger$ |

### Main Takeaways

1. **Phase damping** preserves populations but decays coherence — pure T₂ process
2. **T₂ ≤ 2T₁** is fundamental — amplitude damping contributes to decoherence
3. **Combined models** capture realistic hardware: AD (T₁) + PD (T_φ) gives T₂
4. **Pauli twirling** converts any noise to Pauli channel, preserving fidelity
5. **Model selection** depends on goal: depolarizing for worst-case, combined for realistic

### Noise Model Comparison

| Model | Unital | Fixed Point | When to Use |
|-------|--------|-------------|-------------|
| Depolarizing | Yes | $I/2$ | Threshold analysis |
| Phase Damping | Yes | All $\rho$ | Pure dephasing |
| Amplitude Damping | No | $\|0\rangle$ | T₁ processes |
| Combined AD+PD | No | Near $\|0\rangle$ | Realistic modeling |

---

## Daily Checklist

- [ ] I understand phase damping and its effect on coherence
- [ ] I can derive T₂ from T₁ and T_φ
- [ ] I understand why T₂ ≤ 2T₁
- [ ] I can combine noise channels for realistic models
- [ ] I understand Pauli twirling and its uses

---

## Preview: Day 684

Tomorrow we begin building quantum error correcting codes:
- **Three-qubit bit-flip code:** The simplest QEC code
- **Encoding circuit:** How to prepare logical states
- **Error detection and correction:** Syndrome measurement
- **Limitations:** Why bit-flip code isn't enough

---

**Day 683 Complete!** Week 98: 4/7 days (57%)
