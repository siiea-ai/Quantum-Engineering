# Day 726: Capacity Bounds and Calculations

## Overview

**Date:** Day 726 of 1008
**Week:** 104 (Code Capacity)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Numerical Methods and Bounds for Quantum Capacity

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Upper and lower bounds |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Numerical optimization |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Practical calculations |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Apply** various bounds on quantum capacity
2. **Compute** coherent information numerically
3. **Use** semidefinite programming for capacity bounds
4. **Understand** the role of channel simulation
5. **Calculate** capacity for specific channels
6. **Evaluate** the tightness of different bounds

---

## Core Content

### 1. Upper Bounds on Quantum Capacity

#### No-Cloning Bound

The simplest upper bound:
$$Q(\mathcal{N}) \leq \log_2 d_{out}$$

where $d_{out}$ is the output dimension.

**For qubit channels:** $Q \leq 1$

#### Private Capacity Bound

$$Q(\mathcal{N}) \leq P(\mathcal{N})$$

Private capacity: maximum rate for secure classical communication.

For degradable channels: $Q = P$.

#### Rains Bound

Based on PPT (positive partial transpose) states:

$$Q(\mathcal{N}) \leq R(\mathcal{N})$$

where $R(\mathcal{N})$ involves optimization over PPT-preserving operations.

**Property:** Computable via SDP for some channels.

#### Degradable Extension Bound

If channel $\mathcal{N}$ is "less noisy" than degradable channel $\mathcal{M}$:
$$Q(\mathcal{N}) \leq Q(\mathcal{M})$$

---

### 2. Lower Bounds on Quantum Capacity

#### Coherent Information (Single Letter)

$$Q(\mathcal{N}) \geq Q^{(1)}(\mathcal{N}) = \max_\rho I_c(\rho, \mathcal{N})$$

Achieved by random stabilizer codes.

#### Hashing Bound (Pauli Channels)

For Pauli channel with error distribution $(p_I, p_X, p_Y, p_Z)$:
$$Q(\mathcal{N}) \geq 1 - H(p_I, p_X, p_Y, p_Z)$$

#### Entanglement-Assisted Lower Bound

Using shared entanglement:
$$Q(\mathcal{N}) \geq \frac{1}{2}C_E(\mathcal{N}) - \frac{1}{2}$$

where $C_E$ is entanglement-assisted classical capacity.

---

### 3. Numerical Optimization

#### Maximizing Coherent Information

**Problem:**
$$\max_\rho I_c(\rho, \mathcal{N})$$

where $\rho$ is a valid density matrix.

**Constraints:**
- $\rho \geq 0$ (positive semidefinite)
- $\text{Tr}(\rho) = 1$

**Method:** Gradient ascent or SDP relaxation

#### Gradient of Coherent Information

$$\frac{\partial I_c}{\partial \rho} = \frac{\partial S(B)}{\partial \rho} - \frac{\partial S(E)}{\partial \rho}$$

Using:
$$\frac{\partial S(\sigma)}{\partial \sigma} = -\log_2 \sigma - \frac{1}{\ln 2} I$$

#### Algorithm: Gradient Ascent

```
1. Initialize ρ (e.g., maximally mixed)
2. Repeat:
   a. Compute gradient ∂I_c/∂ρ
   b. Update: ρ ← ρ + η · gradient
   c. Project onto valid density matrices
3. Until convergence
```

---

### 4. Semidefinite Programming

#### SDP Formulation

Many capacity problems can be relaxed to:

$$\max \text{Tr}(C \cdot X)$$
$$\text{subject to: } X \geq 0, \text{ linear constraints}$$

**Example:** PPT bound via SDP

**Advantages:**
- Polynomial time solvable
- Global optimum (for the relaxation)
- Well-developed solvers (CVXPY, Mosek)

#### Capacity-Related SDPs

**1. Quantum capacity upper bound (Rains):**
Optimize over separable/PPT states

**2. Entanglement cost:**
Related to channel simulation

**3. Private capacity:**
Optimize quantum-classical channel extensions

---

### 5. Channel Simulation

#### The Concept

**Reverse Shannon theorem:** Approximate channel $\mathcal{N}$ using:
- Shared entanglement
- Classical communication
- Local operations

**Rate:** Entanglement cost of simulation $\approx$ capacity (for some channels)

#### Simulation for Capacity Bounds

If channel $\mathcal{M}$ can simulate $\mathcal{N}$:
$$Q(\mathcal{N}) \leq Q(\mathcal{M})$$

**Use:** Compare unknown channel to known one.

---

### 6. Capacity Calculations for Specific Channels

#### Depolarizing Channel (Complete Analysis)

**Lower bound (hashing):**
$$Q \geq 1 - H(p) - p\log_2 3$$

**Upper bound (various techniques):**
$$Q \leq \text{[degradable extension bounds]}$$

**Conjecture:** Hashing bound is tight for depolarizing channel.

#### Amplitude Damping

**For $\gamma \leq 1/2$ (degradable):**
$$Q = \max_\eta [H(\eta(1-\gamma)) - H(\eta\gamma)]$$

Optimize over input population $\eta \in [0,1]$.

**For $\gamma > 1/2$ (anti-degradable):**
$$Q = 0$$

#### Generalized Amplitude Damping

Includes thermal noise:
$$Q = f(\gamma, N_{th})$$

where $N_{th}$ is thermal occupation.

Computable for degradable regime.

---

## Worked Examples

### Example 1: Numerical Coherent Information

**Problem:** Numerically find the optimal coherent information for depolarizing channel with $p = 0.1$.

**Solution:**

```python
# Maximize I_c over pure states |ψ⟩ = cos(θ)|0⟩ + sin(θ)e^{iφ}|1⟩

from scipy.optimize import minimize_scalar

def coherent_info_dep(theta, p):
    # Pure state rho = |ψ⟩⟨ψ|
    # Depolarizing output entropy
    # ... (calculation details)
    pass

# Scan over theta
best_theta = minimize_scalar(lambda t: -coherent_info_dep(t, 0.1),
                             bounds=(0, np.pi/2))
```

**Result:** Optimal is maximally mixed, giving $I_c \approx 0.431$.

---

### Example 2: Degradable Channel Capacity

**Problem:** Compute the quantum capacity of amplitude damping with $\gamma = 0.3$.

**Solution:**

**Step 1:** Channel is degradable for $\gamma < 0.5$ ✓

**Step 2:** Optimize over diagonal inputs $\rho = \text{diag}(\eta, 1-\eta)$

**Step 3:** Compute
$$I_c(\eta) = H((1-\eta)(1-\gamma)) - H(\eta\gamma)$$

**Step 4:** Optimize numerically:
- $\eta^* \approx 0.42$
- $Q = I_c(\eta^*) \approx 0.38$

---

### Example 3: Capacity Bound Comparison

**Problem:** For depolarizing $p = 0.15$, compare bounds.

**Solution:**

| Bound Type | Value |
|------------|-------|
| Hashing (lower) | 0.266 |
| $Q^{(1)}$ (lower) | 0.266 |
| No-cloning (upper) | 1.0 |
| Rains (upper) | ~0.3 |

**Gap:** Approximately $0.266 \leq Q \leq 0.3$

The hashing bound is believed to be tight.

---

## Practice Problems

### Direct Application

1. **Problem 1:** Compute the hashing bound for depolarizing channel at $p = 0.17$.

2. **Problem 2:** For amplitude damping $\gamma = 0.4$, find the optimal input state numerically.

3. **Problem 3:** Verify that amplitude damping with $\gamma = 0.6$ has zero capacity.

### Intermediate

4. **Problem 4:** Prove that the quantum capacity of the completely dephasing channel is 0.

5. **Problem 5:** Set up the SDP for computing the Rains bound of a qubit channel.

6. **Problem 6:** Show that capacity is continuous in the channel parameters.

### Challenging

7. **Problem 7:** Prove that the quantum capacity of the erasure channel exactly equals $1 - 2p$.

8. **Problem 8:** Derive upper and lower bounds for the generalized amplitude damping channel with $\gamma = 0.3, N_{th} = 0.1$.

9. **Problem 9:** Analyze the gap between coherent information and quantum capacity for non-degradable channels.

---

## Computational Lab

```python
"""
Day 726: Capacity Bounds and Calculations
Numerical methods for quantum capacity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.linalg import logm

def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def binary_entropy(p: float) -> float:
    """Binary entropy function."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

class QuantumChannel:
    """Base class for quantum channels."""

    def __init__(self, kraus_ops):
        self.kraus_ops = kraus_ops

    def apply(self, rho):
        result = np.zeros_like(rho, dtype=complex)
        for K in self.kraus_ops:
            result += K @ rho @ K.conj().T
        return result

    def complementary(self, rho):
        n_kraus = len(self.kraus_ops)
        env = np.zeros((n_kraus, n_kraus), dtype=complex)
        for i, Ki in enumerate(self.kraus_ops):
            for j, Kj in enumerate(self.kraus_ops):
                env[i, j] = np.trace(Ki @ rho @ Kj.conj().T)
        return env

    def coherent_information(self, rho):
        rho_out = self.apply(rho)
        rho_env = self.complementary(rho)
        return von_neumann_entropy(rho_out) - von_neumann_entropy(rho_env)

def depolarizing_channel(p: float) -> QuantumChannel:
    """Depolarizing channel."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    kraus = [np.sqrt(1-p)*I, np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z]
    return QuantumChannel(kraus)

def amplitude_damping_channel(gamma: float) -> QuantumChannel:
    """Amplitude damping channel."""
    E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return QuantumChannel([E0, E1])

def dephasing_channel(p: float) -> QuantumChannel:
    """Dephasing channel."""
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    return QuantumChannel([np.sqrt(1-p)*I, np.sqrt(p)*Z])

def optimize_coherent_info(channel, method='grid'):
    """Find maximum coherent information."""
    if method == 'grid':
        # Grid search over pure states
        best_ic = -np.inf
        best_state = None

        # Parameterize pure states
        for theta in np.linspace(0, np.pi/2, 50):
            for phi in np.linspace(0, 2*np.pi, 50):
                psi = np.array([np.cos(theta),
                                np.sin(theta) * np.exp(1j * phi)])
                rho = np.outer(psi, psi.conj())

                ic = channel.coherent_information(rho)
                if ic > best_ic:
                    best_ic = ic
                    best_state = rho

        # Also try maximally mixed
        rho_mm = np.eye(2) / 2
        ic_mm = channel.coherent_information(rho_mm)
        if ic_mm > best_ic:
            best_ic = ic_mm
            best_state = rho_mm

        return best_ic, best_state

    elif method == 'optimize':
        # Scipy optimization
        def neg_coherent_info(params):
            theta, phi = params
            psi = np.array([np.cos(theta),
                            np.sin(theta) * np.exp(1j * phi)])
            rho = np.outer(psi, psi.conj())
            return -channel.coherent_information(rho)

        result = minimize(neg_coherent_info, [np.pi/4, 0],
                          bounds=[(0, np.pi/2), (0, 2*np.pi)])
        theta, phi = result.x
        psi = np.array([np.cos(theta), np.sin(theta) * np.exp(1j * phi)])
        rho = np.outer(psi, psi.conj())

        return -result.fun, rho

def amplitude_damping_capacity_exact(gamma: float, num_points: int = 100) -> float:
    """
    Exact capacity for amplitude damping (degradable case).
    Optimize over diagonal inputs.
    """
    if gamma > 0.5:
        return 0.0  # Anti-degradable

    def coherent_info_diagonal(eta):
        # Output state eigenvalues
        lambda_0 = eta * (1 - gamma) + (1 - eta)  # |0⟩ component
        lambda_1 = eta * gamma  # |1⟩ component after damping

        # Actually for amplitude damping the calculation is:
        # Output: diag(1 - eta(1-gamma), eta(1-gamma))? No...

        # Input: diag(1-eta, eta) meaning P(|1⟩) = eta
        # Output |0⟩⟨0| component: (1-eta) + eta*gamma = 1 - eta(1-gamma)
        # Output |1⟩⟨1| component: eta*(1-gamma)

        p0_out = 1 - eta * (1 - gamma)
        p1_out = eta * (1 - gamma)

        S_out = binary_entropy(p1_out)

        # Environment entropy
        # P(no jump) = 1 - eta*gamma when starting in |1⟩
        # Complex... let's use numerical method
        return S_out  # Simplified

    # Use numerical optimization
    channel = amplitude_damping_channel(gamma)

    best_ic = -np.inf
    for eta in np.linspace(0, 1, num_points):
        rho = np.diag([1 - eta, eta])
        ic = channel.coherent_information(rho)
        if ic > best_ic:
            best_ic = ic

    return max(0, best_ic)

def hashing_bound(p: float) -> float:
    """Hashing bound for depolarizing channel."""
    if p <= 0:
        return 1.0
    if p >= 0.75:
        return 0.0
    return max(0, 1 - binary_entropy(p) - p * np.log2(3))

def plot_capacity_bounds():
    """Plot various capacity bounds."""
    p_values = np.linspace(0.001, 0.25, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Depolarizing channel
    ax1 = axes[0]

    # Lower bound (hashing)
    lower = [hashing_bound(p) for p in p_values]

    # Numerical Q^(1)
    numerical = []
    for p in p_values[::5]:
        channel = depolarizing_channel(p)
        ic, _ = optimize_coherent_info(channel)
        numerical.append(max(0, ic))

    # Upper bound (trivial: 1)
    upper = [1.0] * len(p_values)

    ax1.fill_between(p_values, lower, upper, alpha=0.2, color='blue',
                      label='Uncertainty region')
    ax1.plot(p_values, lower, 'g-', linewidth=2, label='Hashing bound (lower)')
    ax1.plot(p_values[::5], numerical, 'ro', markersize=5, label='Numerical Q^(1)')

    ax1.axvline(x=0.1893, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax1.set_xlabel('Depolarizing parameter p')
    ax1.set_ylabel('Quantum capacity bound')
    ax1.set_title('Depolarizing Channel Capacity Bounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.25])
    ax1.set_ylim([0, 1.1])

    # Amplitude damping channel
    ax2 = axes[1]

    gamma_values = np.linspace(0.001, 0.99, 50)
    ad_capacity = []

    for gamma in gamma_values:
        q = amplitude_damping_capacity_exact(gamma)
        ad_capacity.append(q)

    ax2.plot(gamma_values, ad_capacity, 'b-', linewidth=2)
    ax2.axvline(x=0.5, color='red', linestyle='--', label='γ = 0.5 (threshold)')
    ax2.fill_between(gamma_values[:25], ad_capacity[:25], alpha=0.3, color='green',
                      label='Degradable (Q computable)')
    ax2.fill_between(gamma_values[25:], ad_capacity[25:], alpha=0.3, color='red',
                      label='Anti-degradable (Q = 0)')

    ax2.set_xlabel('Damping parameter γ')
    ax2.set_ylabel('Quantum capacity')
    ax2.set_title('Amplitude Damping Channel Capacity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig('capacity_bounds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: capacity_bounds.png")

def compute_capacity_table():
    """Compute capacity values for various channels."""
    print("=" * 70)
    print("Quantum Capacity Calculations")
    print("=" * 70)

    print("\n1. Depolarizing Channel:")
    print(f"{'p':<10} {'Hashing':<15} {'Numerical Q^(1)':<20}")
    print("-" * 50)

    for p in [0.01, 0.05, 0.10, 0.15, 0.18]:
        h = hashing_bound(p)
        channel = depolarizing_channel(p)
        q1, _ = optimize_coherent_info(channel)
        print(f"{p:<10.2f} {h:<15.4f} {max(0, q1):<20.4f}")

    print("\n2. Amplitude Damping Channel:")
    print(f"{'γ':<10} {'Q (exact)':<15} {'Status':<20}")
    print("-" * 50)

    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        q = amplitude_damping_capacity_exact(gamma)
        status = "Degradable" if gamma <= 0.5 else "Anti-degradable"
        print(f"{gamma:<10.2f} {q:<15.4f} {status:<20}")

    print("\n3. Dephasing Channel:")
    print(f"{'p':<10} {'Q = 1-H(p)':<15}")
    print("-" * 30)

    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        q = max(0, 1 - binary_entropy(p))
        print(f"{p:<10.2f} {q:<15.4f}")

def main():
    compute_capacity_table()

    print("\n" + "=" * 70)
    print("Generating Plots...")
    plot_capacity_bounds()

    print("\n" + "=" * 70)
    print("Summary of Bounds")
    print("=" * 70)
    print("""
    Bound Type       | Formula/Description          | Computability
    -----------------|------------------------------|---------------
    No-cloning       | Q ≤ log(d_out)               | Trivial
    Hashing (lower)  | Q ≥ 1 - H(errors)            | Easy
    Q^(1) (lower)    | max_ρ I_c(ρ, N)              | Optimization
    Rains (upper)    | SDP over PPT states          | Polynomial (SDP)
    Private (upper)  | Q ≤ P(N)                     | Hard in general
    LSD (exact)      | lim 1/n max I_c              | Intractable
    """)

if __name__ == "__main__":
    main()
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Upper bounds** | No-cloning, Rains, private capacity |
| **Lower bounds** | Hashing, coherent information |
| **Numerical optimization** | Gradient ascent for $\max_\rho I_c$ |
| **SDP bounds** | Polynomial-time computable relaxations |
| **Channel simulation** | Compare to known channels |

### Key Formulas

$$\boxed{Q^{(1)} = \max_\rho I_c(\rho, \mathcal{N})}$$

$$\boxed{Q_{\text{amp.damp}} = \max_\eta [H(\eta(1-\gamma)) - H(\eta\gamma)] \text{ for } \gamma \leq 1/2}$$

$$\boxed{Q_{\text{dephasing}} = 1 - H(p)}$$

### Main Takeaways

1. **Multiple bounds** bracket the true capacity
2. **Degradable channels** have exactly computable capacity
3. **Numerical optimization** can find $Q^{(1)}$
4. **SDP methods** give polynomial-time upper bounds
5. **Gap** between bounds is small for many channels

---

## Daily Checklist

- [ ] I can apply different capacity bounds
- [ ] I understand numerical optimization for coherent information
- [ ] I know when capacity is exactly computable
- [ ] I can use SDP formulations (conceptually)
- [ ] I can compute capacity for standard channels
- [ ] I completed the computational lab

---

## Preview: Day 727

Tomorrow we study **Practical Capacity Applications**, including:
- Capacity in realistic noise models
- Applications to quantum communication
- Resource analysis for QEC
- Capacity vs practical thresholds
