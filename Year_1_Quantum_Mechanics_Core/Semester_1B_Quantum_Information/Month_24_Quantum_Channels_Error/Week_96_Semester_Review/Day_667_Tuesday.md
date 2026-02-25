# Day 667: Month 21 Review - Open Quantum Systems

## Week 96: Semester 1B Review | Month 24: Quantum Channels & Error Introduction

---

## Review Scope

**Month 21: Open Quantum Systems (Days 561-588)**
- Week 81: System-Environment Models
- Week 82: Lindblad Master Equation
- Week 83: Decoherence Mechanisms
- Week 84: Month Review and Applications

---

## Core Concepts: Open Systems

### 1. The Open Systems Paradigm

**Closed system:** Unitary evolution $|\psi(t)\rangle = U(t)|\psi(0)\rangle$

**Open system:** System + Environment
$$H = H_S + H_E + H_{SE}$$

**Key insight:** System evolution is non-unitary when environment is traced out!

### 2. System-Environment Model

**Total state evolution:**
$$\rho_{SE}(t) = U(t)[\rho_S(0) \otimes \rho_E(0)]U^\dagger(t)$$

**Reduced system dynamics:**
$$\rho_S(t) = \text{Tr}_E[\rho_{SE}(t)]$$

**Born approximation:** Weak coupling, $\rho_{SE}(t) \approx \rho_S(t) \otimes \rho_E$

**Markov approximation:** No memory effects, environment "forgets"

### 3. Lindblad Master Equation

$$\boxed{\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \sum_k \gamma_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)}$$

**Components:**
- Hamiltonian term: $-\frac{i}{\hbar}[H, \rho]$ (coherent evolution)
- Dissipator: $\mathcal{D}[L](\rho) = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}$
- Jump operators $L_k$: Model specific decay channels
- Rates $\gamma_k$: Transition rates

### 4. GKSL Form (General)

$$\frac{d\rho}{dt} = \mathcal{L}(\rho) = -i[H, \rho] + \sum_{k} \left(L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)$$

**Properties:**
- Completely positive
- Trace preserving
- Generates valid density matrices for all $t \geq 0$

### 5. Common Dissipators

**Amplitude damping (decay):**
$$L = \sqrt{\gamma}|0\rangle\langle 1| = \sqrt{\gamma}\sigma_-$$
$$\frac{d\rho}{dt} = \gamma\left(\sigma_-\rho\sigma_+ - \frac{1}{2}\{\sigma_+\sigma_-, \rho\}\right)$$

**Dephasing:**
$$L = \sqrt{\gamma}\sigma_z$$
$$\frac{d\rho}{dt} = \gamma\left(\sigma_z\rho\sigma_z - \rho\right)$$

**Thermal equilibration:**
$$L_\downarrow = \sqrt{\gamma(n_{th}+1)}\sigma_-, \quad L_\uparrow = \sqrt{\gamma n_{th}}\sigma_+$$

---

## Core Concepts: Decoherence

### 6. T1 and T2 Times

**T1 (relaxation time):** Energy decay
$$P_1(t) = P_1(0)e^{-t/T_1}$$

**T2 (decoherence time):** Phase coherence decay
$$\rho_{01}(t) = \rho_{01}(0)e^{-t/T_2}$$

**Fundamental bound:**
$$\boxed{T_2 \leq 2T_1}$$

**Pure dephasing time $T_\phi$:**
$$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$

### 7. Decoherence Mechanisms

| Mechanism | Physical Origin | Effect |
|-----------|-----------------|--------|
| Amplitude damping | Energy loss to environment | T1 decay |
| Phase damping | Random phase kicks | T2 (pure dephasing) |
| Depolarizing | Isotropic noise | All coherences decay |

### 8. Quantum-Classical Transition

**Decoherence explains classicality:**
- Superpositions decay to mixtures
- Environment measures system continuously
- Classical behavior emerges at macroscopic scales

**Decoherence time:**
$$\tau_D \sim \frac{\hbar^2}{mk_BT(\Delta x)^2}$$

For macroscopic $\Delta x$: $\tau_D \approx 10^{-40}$ s (essentially instantaneous)

---

## Integration with Quantum Channels

### 9. Master Equation → Channel

**Solution of Lindblad equation:**
$$\rho(t) = e^{\mathcal{L}t}[\rho(0)]$$

This defines a **dynamical semigroup** $\{\mathcal{E}_t\}_{t\geq 0}$:
- $\mathcal{E}_0 = \mathcal{I}$ (identity)
- $\mathcal{E}_{t+s} = \mathcal{E}_t \circ \mathcal{E}_s$ (semigroup property)
- Each $\mathcal{E}_t$ is CPTP

### 10. Kraus Operators from Lindblad

For amplitude damping with rate $\gamma$:
$$K_0(t) = \begin{pmatrix}1 & 0\\0 & e^{-\gamma t/2}\end{pmatrix}, \quad K_1(t) = \begin{pmatrix}0 & \sqrt{1-e^{-\gamma t}}\\0 & 0\end{pmatrix}$$

**Connection:** $\gamma = 1/T_1$

### 11. Noise in Quantum Computing

**Gate errors:** Imperfect unitary + decoherence during gate

**Idle errors:** T1/T2 decay while waiting

**SPAM errors:** State preparation and measurement

---

## Practice Problems

### Problem 1: Lindblad Equation Solution

Solve the Lindblad equation for pure dephasing:
$$\frac{d\rho}{dt} = \gamma(\sigma_z\rho\sigma_z - \rho)$$

**Solution:**
In matrix form with $\rho = \begin{pmatrix}a & b\\b^* & 1-a\end{pmatrix}$:
- $\dot{a} = 0$ → $a(t) = a(0)$
- $\dot{b} = -2\gamma b$ → $b(t) = b(0)e^{-2\gamma t}$

Populations unchanged, coherences decay with $T_2 = 1/(2\gamma)$.

### Problem 2: T1/T2 Bound

Show that $T_2 \leq 2T_1$ for a qubit with amplitude damping only.

**Solution:**
For amplitude damping: $\gamma = 1/T_1$, and coherence decays as $e^{-\gamma t/2}$.
Thus $T_2 = 2/\gamma = 2T_1$.
With additional dephasing, $T_2$ can only decrease.

### Problem 3: Steady State

Find the steady state of amplitude damping.

**Solution:**
$\mathcal{L}(\rho_{ss}) = 0$

For $L = \sqrt{\gamma}\sigma_-$:
$$\gamma(\sigma_-\rho_{ss}\sigma_+ - \frac{1}{2}\{\sigma_+\sigma_-, \rho_{ss}\}) = 0$$

Solution: $\rho_{ss} = |0\rangle\langle 0|$ (ground state)

### Problem 4: Thermal State

For thermal bath at temperature T, show the steady state is:
$$\rho_{th} = \frac{e^{-H/k_BT}}{\text{Tr}(e^{-H/k_BT})}$$

**Solution:**
With Lindblad operators $L_\downarrow = \sqrt{\gamma(n+1)}\sigma_-$ and $L_\uparrow = \sqrt{\gamma n}\sigma_+$:

Detailed balance gives population ratio:
$$\frac{P_1}{P_0} = \frac{n}{n+1} = e^{-\hbar\omega/k_BT}$$

---

## Computational Lab

```python
"""Day 667: Month 21 Review - Open Quantum Systems"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)

print("Month 21 Review: Open Quantum Systems")
print("=" * 60)

# ============================================
# Part 1: Lindblad Master Equation
# ============================================
print("\nPART 1: Lindblad Master Equation")
print("-" * 40)

def commutator(A, B):
    return A @ B - B @ A

def anticommutator(A, B):
    return A @ B + B @ A

def dissipator(L, rho):
    """Lindblad dissipator D[L](ρ) = LρL† - (1/2){L†L, ρ}"""
    return L @ rho @ L.conj().T - 0.5 * anticommutator(L.conj().T @ L, rho)

def lindblad_rhs(t, rho_vec, H, jump_ops, rates):
    """Right-hand side of Lindblad equation."""
    rho = rho_vec.reshape(2, 2)

    # Hamiltonian term
    drho = -1j * commutator(H, rho)

    # Dissipator terms
    for L, gamma in zip(jump_ops, rates):
        drho += gamma * dissipator(L, rho)

    return drho.flatten()

# Example: Amplitude damping
print("\nAmplitude Damping Dynamics:")
gamma = 1.0  # Decay rate (1/T1)

# Initial state: |+⟩
rho_0 = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

# Solve
H = np.zeros((2, 2), dtype=complex)  # No Hamiltonian
L_decay = sigma_minus

t_span = (0, 5)
t_eval = np.linspace(0, 5, 100)

sol = solve_ivp(
    lambda t, y: lindblad_rhs(t, y, H, [L_decay], [gamma]),
    t_span,
    rho_0.flatten(),
    t_eval=t_eval,
    method='RK45'
)

# Extract populations and coherences
P1 = np.real(sol.y[3, :])  # ρ_11
coherence = np.abs(sol.y[1, :])  # |ρ_01|

print(f"Initial P(|1⟩) = {P1[0]:.3f}, Final P(|1⟩) = {P1[-1]:.3f}")
print(f"Initial |ρ_01| = {coherence[0]:.3f}, Final |ρ_01| = {coherence[-1]:.6f}")

# Verify T1 and T2
T1_expected = 1/gamma
T2_expected = 2/gamma

# Fit exponential decay
from scipy.optimize import curve_fit

def exp_decay(t, A, tau):
    return A * np.exp(-t/tau)

try:
    popt_P1, _ = curve_fit(exp_decay, t_eval, P1, p0=[0.5, 1])
    popt_coh, _ = curve_fit(exp_decay, t_eval[coherence > 1e-6], coherence[coherence > 1e-6], p0=[0.5, 1])
    print(f"\nFitted T1 = {popt_P1[1]:.3f} (expected {T1_expected:.3f})")
    print(f"Fitted T2 = {popt_coh[1]:.3f} (expected {T2_expected:.3f})")
except:
    print("Fitting failed, but theory values are T1 = 1, T2 = 2")

# ============================================
# Part 2: Pure Dephasing
# ============================================
print("\n" + "=" * 60)
print("PART 2: Pure Dephasing")
print("-" * 40)

gamma_phi = 0.5  # Dephasing rate
L_dephase = sigma_minus @ sigma_plus  # = |0⟩⟨0| projects

# Actually for dephasing we use σ_z
L_dephase = Z

sol_deph = solve_ivp(
    lambda t, y: lindblad_rhs(t, y, H, [L_dephase], [gamma_phi]),
    t_span,
    rho_0.flatten(),
    t_eval=t_eval,
    method='RK45'
)

P1_deph = np.real(sol_deph.y[3, :])
coh_deph = np.abs(sol_deph.y[1, :])

print(f"Initial P(|1⟩) = {P1_deph[0]:.3f}, Final P(|1⟩) = {P1_deph[-1]:.3f}")
print(f"Initial |ρ_01| = {coh_deph[0]:.3f}, Final |ρ_01| = {coh_deph[-1]:.6f}")
print("Note: Population unchanged, only coherence decays!")

# ============================================
# Part 3: Combined T1 and T2
# ============================================
print("\n" + "=" * 60)
print("PART 3: Combined T1 and T2 Decay")
print("-" * 40)

gamma_1 = 1.0  # Amplitude damping rate
gamma_phi = 0.5  # Additional dephasing rate

# For dephasing, the correct jump operator contribution
# actually the coherence decay from amplitude damping is γ/2
# and from dephasing is 2γ_φ (when L = σ_z)

sol_combined = solve_ivp(
    lambda t, y: lindblad_rhs(t, y, H, [sigma_minus, Z], [gamma_1, gamma_phi]),
    t_span,
    rho_0.flatten(),
    t_eval=t_eval,
    method='RK45'
)

P1_comb = np.real(sol_combined.y[3, :])
coh_comb = np.abs(sol_combined.y[1, :])

# Theoretical values
T1 = 1/gamma_1
T2 = 1/(gamma_1/2 + 2*gamma_phi)  # 1/T2 = 1/(2T1) + 1/T_φ where 1/T_φ = 2γ_φ
T_phi = 1/(2*gamma_phi)

print(f"Theory: T1 = {T1:.3f}, T_φ = {T_phi:.3f}")
print(f"Theory: T2 = {T2:.3f}")
print(f"Bound T2 ≤ 2T1: {T2:.3f} ≤ {2*T1:.3f} ✓")

# ============================================
# Part 4: Kraus Operators from Lindblad
# ============================================
print("\n" + "=" * 60)
print("PART 4: Kraus Operators from Lindblad")
print("-" * 40)

def amplitude_damping_kraus(t, gamma):
    """Kraus operators for amplitude damping at time t."""
    p = 1 - np.exp(-gamma * t)
    K0 = np.array([[1, 0], [0, np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return K0, K1

# Compare Lindblad solution to Kraus at final time
t_final = 2.0
K0, K1 = amplitude_damping_kraus(t_final, gamma)

# Apply Kraus
rho_kraus = K0 @ rho_0 @ K0.conj().T + K1 @ rho_0 @ K1.conj().T

# Get Lindblad solution at same time
idx = np.argmin(np.abs(t_eval - t_final))
rho_lindblad = sol.y[:, idx].reshape(2, 2)

print(f"At t = {t_final}:")
print(f"Kraus ρ_11 = {np.real(rho_kraus[1,1]):.4f}")
print(f"Lindblad ρ_11 = {np.real(rho_lindblad[1,1]):.4f}")
print(f"Match: {np.allclose(rho_kraus, rho_lindblad)}")

# ============================================
# Part 5: Thermal Equilibration
# ============================================
print("\n" + "=" * 60)
print("PART 5: Thermal Equilibration")
print("-" * 40)

# Thermal occupation number
n_th = 0.1  # Low temperature (mostly ground state)

L_down = np.sqrt(n_th + 1) * sigma_minus
L_up = np.sqrt(n_th) * sigma_plus

# Start from excited state
rho_excited = np.array([[0, 0], [0, 1]], dtype=complex)

sol_thermal = solve_ivp(
    lambda t, y: lindblad_rhs(t, y, H, [L_down, L_up], [gamma, gamma]),
    (0, 10),
    rho_excited.flatten(),
    t_eval=np.linspace(0, 10, 200),
    method='RK45'
)

P1_thermal = np.real(sol_thermal.y[3, :])
P1_eq = n_th / (2*n_th + 1)  # Thermal equilibrium

print(f"Thermal occupation n_th = {n_th}")
print(f"Initial P(|1⟩) = 1.0")
print(f"Final P(|1⟩) = {P1_thermal[-1]:.4f}")
print(f"Expected thermal P(|1⟩) = {P1_eq:.4f}")

print("\n" + "=" * 60)
print("Review Complete!")
```

---

## Summary

### Key Equations

| Concept | Formula |
|---------|---------|
| Lindblad equation | $\dot{\rho} = -i[H,\rho] + \sum_k \gamma_k \mathcal{D}[L_k](\rho)$ |
| Dissipator | $\mathcal{D}[L](\rho) = L\rho L^\dagger - \frac{1}{2}\\{L^\dagger L, \rho\\}$ |
| T1 decay | $P_1(t) = P_1(0)e^{-t/T_1}$ |
| T2 decay | $\rho_{01}(t) = \rho_{01}(0)e^{-t/T_2}$ |
| T1-T2 bound | $T_2 \leq 2T_1$ |
| T2 decomposition | $1/T_2 = 1/(2T_1) + 1/T_\phi$ |

### Physical Interpretation

- **Amplitude damping:** Energy loss, T1 process
- **Dephasing:** Phase randomization, pure T2 process
- **Decoherence:** Environment-induced classicality

### Connection to Other Months

- **Month 23-24:** Lindblad generates CPTP channels
- **Error correction:** Must overcome T1/T2 limitations

---

## Preview: Day 668

Tomorrow: **Month 22 Review** - Quantum algorithms, Deutsch-Jozsa, Grover, and quantum speedups!
