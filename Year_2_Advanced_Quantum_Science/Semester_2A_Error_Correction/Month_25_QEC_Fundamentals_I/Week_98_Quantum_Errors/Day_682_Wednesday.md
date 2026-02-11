# Day 682: Depolarizing and Amplitude Damping Channels

## Week 98: Quantum Errors | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Detailed Channel Analysis |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 682, you will be able to:

1. **Analyze the depolarizing channel** in multiple equivalent forms
2. **Understand physical T₁ relaxation** and its amplitude damping model
3. **Compute channel fidelity and error rates** for each noise model
4. **Connect noise parameters to physical quantities** (T₁, gate times)
5. **Visualize Bloch sphere contraction** under noise
6. **Apply these channels in quantum error correction design**

---

## The Depolarizing Channel: Deep Dive

### Definition and Equivalent Forms

The **depolarizing channel** is the most symmetric noise model, treating all three Pauli errors equally.

**Form 1: Pauli Error Model**

$$\boxed{\mathcal{E}_{dep}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)}$$

With probability $1-p$: no error. With probability $p/3$ each: X, Y, or Z error.

**Form 2: Maximally Mixed Form**

$$\boxed{\mathcal{E}_{dep}(\rho) = \left(1-\frac{4p}{3}\right)\rho + \frac{4p}{3} \cdot \frac{I}{2}}$$

This shows depolarizing as an interpolation between the state and the maximally mixed state!

**Derivation:** Using $I + X + Y + Z = 2|0\rangle\langle 0| + 2|1\rangle\langle 1| - I - I - I = 2I$ (for trace normalization):

$$I\rho I + X\rho X + Y\rho Y + Z\rho Z = 2\text{Tr}(\rho)I = 2I$$

Therefore:
$$\frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z) = \frac{p}{3}(2I - \rho) = \frac{2pI}{3} - \frac{p\rho}{3}$$

Combining:
$$(1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z) = \left(1-\frac{4p}{3}\right)\rho + \frac{2p}{3}I$$

**Form 3: Shrinking Factor**

Define $\lambda = 1 - \frac{4p}{3}$. Then:

$$\mathcal{E}_{dep}(\rho) = \lambda\rho + (1-\lambda)\frac{I}{2}$$

The Bloch vector shrinks uniformly: $\vec{r} \rightarrow \lambda\vec{r}$.

### Physical Interpretation

| Parameter | Meaning |
|-----------|---------|
| $p = 0$ | No noise, identity channel |
| $p = 3/4$ | Completely depolarizing: $\mathcal{E}(\rho) = I/2$ |
| $p = 1$ | "Overshoot": actually applies net error |
| $\lambda = 1 - 4p/3$ | Bloch sphere shrinking factor |

### Kraus Operators

$$E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{\frac{p}{3}}\,X, \quad E_2 = \sqrt{\frac{p}{3}}\,Y, \quad E_3 = \sqrt{\frac{p}{3}}\,Z$$

### Channel Fidelity

The **average fidelity** of the depolarizing channel is:

$$\bar{F} = \int d\psi \, \langle\psi|\mathcal{E}(|\psi\rangle\langle\psi|)|\psi\rangle = 1 - \frac{2p}{3}$$

For $p \ll 1$: the infidelity is approximately $2p/3$.

### Connection to Gate Error Rate

In benchmarking, the **error per gate (EPG)** relates to:

$$\text{EPG} = \frac{d-1}{d}(1 - \bar{F}) = \frac{p}{2} \quad \text{(for qubits, } d=2\text{)}$$

---

## Amplitude Damping: T₁ Relaxation

### Physical Motivation

Real qubits lose energy to their environment. An excited state $|1\rangle$ spontaneously decays to $|0\rangle$:

$$|1\rangle \xrightarrow{T_1} |0\rangle + \text{photon/phonon}$$

This is the quantum analog of **exponential decay** in classical systems.

### Kraus Operators

$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

where $\gamma$ is the **decay probability**.

**Time dependence:** $\gamma(t) = 1 - e^{-t/T_1}$

### Action on Density Matrix

For a general state $\rho = \begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix}$:

$$\mathcal{E}_{AD}(\rho) = \begin{pmatrix} \rho_{00} + \gamma\rho_{11} & \sqrt{1-\gamma}\,\rho_{01} \\ \sqrt{1-\gamma}\,\rho_{10} & (1-\gamma)\rho_{11} \end{pmatrix}$$

**Key Effects:**
1. **Population transfer:** $P(0) \rightarrow P(0) + \gamma P(1)$, $P(1) \rightarrow (1-\gamma)P(1)$
2. **Coherence decay:** $\rho_{01} \rightarrow \sqrt{1-\gamma}\,\rho_{01}$
3. **Fixed point:** $|0\rangle\langle 0|$ is the unique fixed point

### Bloch Sphere Transformation

On the Bloch sphere $(x, y, z)$:

$$\begin{aligned}
x &\rightarrow \sqrt{1-\gamma}\,x \\
y &\rightarrow \sqrt{1-\gamma}\,y \\
z &\rightarrow (1-\gamma)z + \gamma
\end{aligned}$$

The Bloch sphere **shrinks and shifts toward the north pole** ($|0\rangle$).

### Properties

1. **Non-unital:** $\mathcal{E}_{AD}(I) \neq I$. The maximally mixed state evolves toward $|0\rangle$
2. **Fixed point:** $|0\rangle$ is the unique fixed point
3. **Irreversible:** Cannot be undone by any unitary

### Amplitude Damping as Pauli Approximation

For small $\gamma$, amplitude damping resembles a specific combination of Pauli errors. However, it's fundamentally different because:
- It's **non-unital** (shifts the Bloch sphere)
- It has a **preferred direction** ($|1\rangle \to |0\rangle$, not vice versa)

---

## Generalized Amplitude Damping

### Finite Temperature

At finite temperature, the qubit equilibrates to a thermal state, not $|0\rangle$:

$$\rho_{eq} = \frac{e^{-H/k_B T}}{\text{Tr}(e^{-H/k_B T})} = \begin{pmatrix} 1-p_{th} & 0 \\ 0 & p_{th} \end{pmatrix}$$

where $p_{th} = \frac{1}{1 + e^{\hbar\omega/k_B T}}$ is the thermal population of $|1\rangle$.

### Kraus Operators (4 operators)

$$E_0 = \sqrt{p}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \sqrt{p}\begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

$$E_2 = \sqrt{1-p}\begin{pmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{pmatrix}, \quad E_3 = \sqrt{1-p}\begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}$$

Here $p = 1 - p_{th}$ is the probability of decay (vs excitation).

---

## Comparing Noise Models

### Bloch Sphere Effects

| Channel | x | y | z | Geometry |
|---------|---|---|---|----------|
| Depolarizing | $\lambda x$ | $\lambda y$ | $\lambda z$ | Uniform shrinking |
| Amplitude Damping | $\sqrt{1-\gamma} x$ | $\sqrt{1-\gamma} y$ | $(1-\gamma)z + \gamma$ | Shrink + shift to north |
| Bit-flip | $x$ | $(1-2p)y$ | $(1-2p)z$ | Contract along y, z |
| Phase-flip | $(1-2p)x$ | $(1-2p)y$ | $z$ | Contract along x, y |

### Mathematical Properties

| Property | Depolarizing | Amplitude Damping |
|----------|--------------|-------------------|
| Unital | Yes | No |
| Fixed point | $I/2$ (for $p=3/4$) | $\|0\rangle\langle 0\|$ |
| Pauli diagonal | Yes | No |
| Time-reversal symmetric | Yes | No |

### Error Rates

| Channel | Fidelity | Error Probability |
|---------|----------|-------------------|
| Depolarizing $(p)$ | $1 - 2p/3$ | $p$ |
| Amplitude Damping $(\gamma)$ | $\frac{1}{2}(1 + \sqrt{1-\gamma} + \gamma)$ | $\sim \gamma/2$ |
| Bit-flip $(p)$ | $1 - p/2$ | $p$ |

---

## Error Rates in Real Hardware

### Superconducting Qubits (Transmon)

| Parameter | Typical Value | Corresponding $\gamma$ or $p$ |
|-----------|---------------|-------------------------------|
| T₁ | 50-100 μs | $\gamma = 1 - e^{-t_{gate}/T_1}$ |
| Gate time | 20-50 ns | $\gamma_{gate} \approx 0.0005$ |
| 1Q gate error | 0.1-0.5% | $p \approx 0.001-0.005$ |
| 2Q gate error | 0.5-2% | $p \approx 0.005-0.02$ |

### Example Calculation

For a gate time $t = 30$ ns and $T_1 = 80$ μs:

$$\gamma = 1 - e^{-30 \times 10^{-9} / 80 \times 10^{-6}} = 1 - e^{-0.000375} \approx 0.000375$$

During one gate, about 0.04% probability of T₁ decay.

---

## Worked Examples

### Example 1: Depolarizing Channel on $|+\rangle$

**Problem:** Apply the depolarizing channel ($p = 0.1$) to $|+\rangle$.

**Solution:**

$$|+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Using the shrinking form with $\lambda = 1 - 4(0.1)/3 = 0.867$:

$$\mathcal{E}(\rho) = \lambda\rho + (1-\lambda)\frac{I}{2}$$

$$= 0.867 \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} + 0.133 \cdot \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 0.867 + 0.133 & 0.867 \\ 0.867 & 0.867 + 0.133 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0.867 \\ 0.867 & 1 \end{pmatrix}$$

The coherence has shrunk from 1 to 0.867.

### Example 2: Amplitude Damping Evolution

**Problem:** A qubit starts in $|1\rangle$ at $t=0$. Plot $P(1)$ as a function of time for $T_1 = 50\mu s$.

**Solution:**

$$P(1)(t) = (1-\gamma(t)) = e^{-t/T_1}$$

This is exponential decay with time constant $T_1$.

At $t = T_1$: $P(1) = e^{-1} \approx 0.368$
At $t = 2T_1$: $P(1) = e^{-2} \approx 0.135$
At $t = 5T_1$: $P(1) = e^{-5} \approx 0.007$

### Example 3: Coherence Under Both Noise Types

**Problem:** Compare coherence decay of $|+\rangle$ under depolarizing ($p = 0.1$) vs amplitude damping ($\gamma = 0.1$).

**Solution:**

For $\rho = |+\rangle\langle +|$, the off-diagonal element is $\rho_{01} = 1/2$.

**Depolarizing:** $\rho_{01}' = \lambda \cdot \rho_{01} = (1 - 4p/3) \cdot \frac{1}{2} = 0.867 \cdot 0.5 = 0.433$

**Amplitude Damping:** $\rho_{01}' = \sqrt{1-\gamma} \cdot \rho_{01} = \sqrt{0.9} \cdot 0.5 = 0.474$

Amplitude damping preserves more coherence at the same parameter value!

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** Compute the output of the depolarizing channel ($p = 0.2$) on the state $|0\rangle\langle 0|$.

**A.2** A qubit in state $|1\rangle$ undergoes amplitude damping with $\gamma = 0.4$. Find the output density matrix.

**A.3** What is the Bloch vector after depolarizing ($p = 0.15$) acts on the state at $(x, y, z) = (0.8, 0, 0.6)$?

### Problem Set B: Intermediate

**B.1** Prove that the depolarizing channel is unital by showing $\mathcal{E}_{dep}(I) = I$.

**B.2** For amplitude damping, show that the off-diagonal coherence decays as $\sqrt{1-\gamma}$ while the population decays as $(1-\gamma)$. Why is this significant for T₂ vs T₁?

**B.3** Derive the effective depolarizing parameter after $n$ sequential applications.

### Problem Set C: Challenging

**C.1** The **quantum capacity** of amplitude damping is:
$$Q(\gamma) = \max_{p \in [0,1]} [H_2((1-\gamma)p + \gamma) - H_2(\gamma)]$$
where $H_2$ is binary entropy. Show that $Q > 0$ for $\gamma < 1/2$.

**C.2** Design a noise model that's "worse" than depolarizing in the sense that it introduces correlated errors across time. How would this affect error correction?

**C.3** The **entanglement fidelity** of a channel is $F_e = \langle\Phi^+|J(\mathcal{E})|\Phi^+\rangle$ where $J$ is the Choi matrix. Compute $F_e$ for the depolarizing channel.

---

## Computational Lab: Channel Analysis and Visualization

```python
"""
Day 682 Computational Lab: Depolarizing & Amplitude Damping
===========================================================

In-depth analysis of common quantum noise channels.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

# =============================================================================
# Part 1: Pauli Matrices and State Utilities
# =============================================================================

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def bloch_to_density(x: float, y: float, z: float) -> np.ndarray:
    """Convert Bloch coordinates to density matrix."""
    return 0.5 * (I + x*X + y*Y + z*Z)

def density_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]:
    """Convert density matrix to Bloch coordinates."""
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    return x, y, z

print("=" * 60)
print("PART 1: Depolarizing Channel Analysis")
print("=" * 60)

# =============================================================================
# Part 2: Depolarizing Channel
# =============================================================================

def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply depolarizing channel with error probability p."""
    return (1-p) * rho + (p/3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)

def depolarizing_shrinking(rho: np.ndarray, p: float) -> np.ndarray:
    """Depolarizing in shrinking form."""
    lam = 1 - 4*p/3
    return lam * rho + (1 - lam) * I/2

# Verify equivalence
rho_test = bloch_to_density(0.5, 0.3, 0.6)
p_test = 0.15

out1 = depolarizing_channel(rho_test, p_test)
out2 = depolarizing_shrinking(rho_test, p_test)

print(f"\nDepolarizing channel (p={p_test}) equivalence check:")
print(f"  Pauli form matches shrinking form: {np.allclose(out1, out2)}")

# Bloch vector shrinking
x, y, z = density_to_bloch(rho_test)
x_out, y_out, z_out = density_to_bloch(out1)
lam = 1 - 4*p_test/3

print(f"\n  Original Bloch: ({x:.3f}, {y:.3f}, {z:.3f})")
print(f"  After channel:  ({x_out:.3f}, {y_out:.3f}, {z_out:.3f})")
print(f"  Shrinking factor λ = {lam:.3f}")
print(f"  Verify: λ × original = ({lam*x:.3f}, {lam*y:.3f}, {lam*z:.3f})")

# =============================================================================
# Part 3: Amplitude Damping Channel
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Amplitude Damping Channel")
print("=" * 60)

def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply amplitude damping with decay probability gamma."""
    E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T

# Test on |1⟩
rho_1 = np.array([[0, 0], [0, 1]], dtype=complex)
gamma_test = 0.3

rho_out = amplitude_damping_channel(rho_1, gamma_test)
print(f"\nAmplitude damping (γ={gamma_test}) on |1⟩:")
print(f"  Input:  P(0)=0, P(1)=1")
print(f"  Output: P(0)={rho_out[0,0].real:.3f}, P(1)={rho_out[1,1].real:.3f}")
print(f"  Expected: P(0)=γ={gamma_test}, P(1)=1-γ={1-gamma_test}")

# Test on |+⟩
rho_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
rho_out = amplitude_damping_channel(rho_plus, gamma_test)

print(f"\nAmplitude damping (γ={gamma_test}) on |+⟩:")
print(f"  Input coherence:  |ρ₀₁| = 0.5")
print(f"  Output coherence: |ρ₀₁| = {np.abs(rho_out[0,1]):.4f}")
print(f"  Expected: √(1-γ)×0.5 = {np.sqrt(1-gamma_test)*0.5:.4f}")

# =============================================================================
# Part 4: Time Evolution Comparison
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Time Evolution Comparison")
print("=" * 60)

# Parameters
T1 = 50e-6  # 50 microseconds
times = np.linspace(0, 5*T1, 100)

# Calculate gamma(t) and p(t) for equivalent error rates
gammas = 1 - np.exp(-times/T1)
p_equiv = gammas  # For comparison at same "strength"

# Track coherence of |+⟩ state
rho_init = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

coherence_AD = []
coherence_dep = []

for g, p in zip(gammas, p_equiv):
    rho_AD = amplitude_damping_channel(rho_init, g)
    rho_dep = depolarizing_channel(rho_init, min(p, 0.75))  # Cap at max
    coherence_AD.append(np.abs(rho_AD[0, 1]))
    coherence_dep.append(np.abs(rho_dep[0, 1]))

plt.figure(figsize=(14, 5))

# Plot 1: Coherence decay
plt.subplot(1, 3, 1)
plt.plot(times*1e6, np.array(coherence_AD)*2, 'b-', label='Amplitude Damping', linewidth=2)
plt.plot(times*1e6, np.array(coherence_dep)*2, 'r--', label='Depolarizing', linewidth=2)
plt.xlabel('Time (μs)')
plt.ylabel('Coherence |ρ₀₁|/|ρ₀₁(0)|')
plt.title('Coherence Decay Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Population decay for |1⟩
rho_1 = np.array([[0, 0], [0, 1]], dtype=complex)
pop_1_AD = []
pop_1_dep = []

for g, p in zip(gammas, p_equiv):
    rho_AD = amplitude_damping_channel(rho_1, g)
    rho_dep = depolarizing_channel(rho_1, min(p, 0.75))
    pop_1_AD.append(rho_AD[1, 1].real)
    pop_1_dep.append(rho_dep[1, 1].real)

plt.subplot(1, 3, 2)
plt.plot(times*1e6, pop_1_AD, 'b-', label='Amplitude Damping', linewidth=2)
plt.plot(times*1e6, pop_1_dep, 'r--', label='Depolarizing', linewidth=2)
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Mixed state')
plt.xlabel('Time (μs)')
plt.ylabel('P(|1⟩)')
plt.title('Population Decay from |1⟩')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Bloch sphere trajectories
ax = plt.subplot(1, 3, 3, projection='3d')

# Draw sphere wireframe
u = np.linspace(0, 2*np.pi, 20)
v = np.linspace(0, np.pi, 15)
x_s = np.outer(np.cos(u), np.sin(v))
y_s = np.outer(np.sin(u), np.sin(v))
z_s = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x_s, y_s, z_s, alpha=0.1, color='gray')

# Starting point: |+⟩ on equator
x0, y0, z0 = 1, 0, 0

# Trajectory for amplitude damping
x_AD, y_AD, z_AD = [x0], [y0], [z0]
for g in np.linspace(0, 0.95, 20):
    rho = amplitude_damping_channel(bloch_to_density(x0, y0, z0), g)
    x, y, z = density_to_bloch(rho)
    x_AD.append(x)
    y_AD.append(y)
    z_AD.append(z)

# Trajectory for depolarizing
x_dep, y_dep, z_dep = [x0], [y0], [z0]
for p in np.linspace(0, 0.7, 20):
    rho = depolarizing_channel(bloch_to_density(x0, y0, z0), p)
    x, y, z = density_to_bloch(rho)
    x_dep.append(x)
    y_dep.append(y)
    z_dep.append(z)

ax.plot(x_AD, y_AD, z_AD, 'b-', linewidth=2, label='Amplitude Damping')
ax.plot(x_dep, y_dep, z_dep, 'r--', linewidth=2, label='Depolarizing')
ax.scatter([x0], [y0], [z0], color='green', s=100, label='Start |+⟩')
ax.scatter([0], [0], [1], color='blue', s=50, marker='^', label='AD endpoint |0⟩')
ax.scatter([0], [0], [0], color='red', s=50, marker='o', label='Dep endpoint')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Bloch Sphere Trajectories')
ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('day_682_channel_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_682_channel_comparison.png")

# =============================================================================
# Part 5: Channel Fidelity Analysis
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Channel Fidelity Analysis")
print("=" * 60)

def average_fidelity_depolarizing(p: float) -> float:
    """Average fidelity of depolarizing channel."""
    return 1 - 2*p/3

def average_fidelity_amplitude_damping(gamma: float, n_samples: int = 1000) -> float:
    """Numerical average fidelity of amplitude damping."""
    fidelities = []
    for _ in range(n_samples):
        # Random pure state
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        state = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
        rho = np.outer(state, state.conj())

        # Apply channel
        rho_out = amplitude_damping_channel(rho, gamma)

        # Fidelity
        fid = np.real(state.conj() @ rho_out @ state)
        fidelities.append(fid)

    return np.mean(fidelities)

# Calculate fidelities
params = np.linspace(0, 0.5, 50)
fid_dep = [average_fidelity_depolarizing(p) for p in params]
fid_AD = [average_fidelity_amplitude_damping(g) for g in params]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(params, fid_dep, 'r-', label='Depolarizing', linewidth=2)
plt.plot(params, fid_AD, 'b-', label='Amplitude Damping', linewidth=2)
plt.xlabel('Error Parameter (p or γ)')
plt.ylabel('Average Fidelity')
plt.title('Average Channel Fidelity')
plt.legend()
plt.grid(True, alpha=0.3)

# Infidelity (error rate)
plt.subplot(1, 2, 2)
infid_dep = [1 - f for f in fid_dep]
infid_AD = [1 - f for f in fid_AD]
plt.semilogy(params, infid_dep, 'r-', label='Depolarizing', linewidth=2)
plt.semilogy(params, infid_AD, 'b-', label='Amplitude Damping', linewidth=2)
plt.xlabel('Error Parameter (p or γ)')
plt.ylabel('Average Infidelity (1 - F)')
plt.title('Channel Error Rate')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_682_fidelity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_682_fidelity_analysis.png")

# =============================================================================
# Part 6: T1 and Gate Error Relationship
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Physical Parameter Relationships")
print("=" * 60)

# Typical hardware parameters
T1_values = [30e-6, 50e-6, 100e-6, 200e-6]  # T1 in seconds
gate_times = [20e-9, 30e-9, 50e-9]  # Gate times in seconds

print("\nGate error rates from T1 relaxation:")
print("-" * 60)
print(f"{'T1 (μs)':<12} {'Gate (ns)':<12} {'γ per gate':<15} {'Error %':<10}")
print("-" * 60)

for T1 in T1_values:
    for t_gate in gate_times:
        gamma = 1 - np.exp(-t_gate/T1)
        error_pct = gamma * 100
        print(f"{T1*1e6:<12.0f} {t_gate*1e9:<12.0f} {gamma:<15.6f} {error_pct:<10.4f}")

# =============================================================================
# Part 7: Unital vs Non-Unital Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Unital vs Non-Unital Behavior")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Depolarizing (unital) - shrinks toward center
ax1 = axes[0]
points_init = []
for _ in range(100):
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points_init.append((x, y, z))

points_dep = []
p = 0.3
for (x, y, z) in points_init:
    rho = bloch_to_density(x, y, z)
    rho_out = depolarizing_channel(rho, p)
    points_dep.append(density_to_bloch(rho_out))

x_init = [p[0] for p in points_init]
z_init = [p[2] for p in points_init]
x_dep = [p[0] for p in points_dep]
z_dep = [p[2] for p in points_dep]

ax1.scatter(x_init, z_init, alpha=0.3, label='Initial', s=20)
ax1.scatter(x_dep, z_dep, alpha=0.5, label=f'After Depol (p={p})', s=20)
ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax1.add_patch(circle)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Z')
ax1.set_title('Depolarizing (Unital)\nShrinks toward center')
ax1.legend()

# Right: Amplitude damping (non-unital) - shifts toward |0⟩
points_AD = []
gamma = 0.5
for (x, y, z) in points_init:
    rho = bloch_to_density(x, y, z)
    rho_out = amplitude_damping_channel(rho, gamma)
    points_AD.append(density_to_bloch(rho_out))

x_AD = [p[0] for p in points_AD]
z_AD = [p[2] for p in points_AD]

ax2 = axes[1]
ax2.scatter(x_init, z_init, alpha=0.3, label='Initial', s=20)
ax2.scatter(x_AD, z_AD, alpha=0.5, label=f'After AD (γ={gamma})', s=20)
ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax2.scatter([0], [1], color='red', s=100, marker='*', label='|0⟩ (fixed point)')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax2.add_patch(circle)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.set_aspect('equal')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')
ax2.set_title('Amplitude Damping (Non-Unital)\nShrinks AND shifts toward |0⟩')
ax2.legend()

plt.tight_layout()
plt.savefig('day_682_unital_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_682_unital_comparison.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Depolarizing vs Amplitude Damping")
print("=" * 60)

summary = """
┌─────────────────────────────────────────────────────────────────────┐
│         Depolarizing vs Amplitude Damping Comparison                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ DEPOLARIZING (p):                                                    │
│   • E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)                          │
│   • Equivalent: λρ + (1-λ)I/2 where λ = 1 - 4p/3                    │
│   • Uniform Bloch sphere shrinking                                   │
│   • Unital: E(I) = I                                                 │
│   • Models: Random Pauli errors, worst-case noise                    │
│                                                                      │
│ AMPLITUDE DAMPING (γ):                                               │
│   • E₀ = [[1,0],[0,√(1-γ)]], E₁ = [[0,√γ],[0,0]]                   │
│   • Decay: |1⟩ → |0⟩ with probability γ                             │
│   • Bloch: shrink + shift toward north pole                          │
│   • Non-unital: E(I) ≠ I                                             │
│   • Models: T₁ relaxation, energy dissipation                        │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ Physical Relevance:                                                  │
│   • Depolarizing: Theoretical benchmark, worst-case analysis         │
│   • Amplitude Damping: Dominant noise in superconducting qubits      │
└─────────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("\n✅ Day 682 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Depolarizing (Pauli form) | $\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$ |
| Depolarizing (shrinking) | $\mathcal{E}(\rho) = \lambda\rho + (1-\lambda)\frac{I}{2}$, $\lambda = 1-\frac{4p}{3}$ |
| Amplitude damping | $E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$, $E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$ |
| T₁ decay | $\gamma(t) = 1 - e^{-t/T_1}$ |
| Depolarizing fidelity | $\bar{F} = 1 - \frac{2p}{3}$ |

### Main Takeaways

1. **Depolarizing channel:** Symmetric noise model, contracts Bloch sphere uniformly toward center
2. **Amplitude damping:** Physical T₁ relaxation, contracts AND shifts toward $|0\rangle$
3. **Unital vs non-unital:** Depolarizing preserves maximally mixed state; amplitude damping doesn't
4. **Physical connection:** $\gamma = 1 - e^{-t/T_1}$ connects channel parameter to hardware
5. **Error rates:** Modern hardware: ~0.1% single-qubit, ~0.5-1% two-qubit gate errors

---

## Daily Checklist

- [ ] I can write depolarizing channel in both Pauli and shrinking forms
- [ ] I understand amplitude damping Kraus operators and their physical meaning
- [ ] I can compute Bloch sphere transformations under both channels
- [ ] I understand the difference between unital and non-unital channels
- [ ] I can connect noise parameters to physical quantities like T₁

---

## Preview: Day 683

Tomorrow we complete our noise model survey:
- **Phase damping (T₂ dephasing)**
- **Combined noise models**
- **Realistic error processes**
- **Noise model selection for QEC analysis**

---

**Day 682 Complete!** Week 98: 3/7 days (43%)
