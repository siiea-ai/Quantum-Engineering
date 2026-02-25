# Day 657: Error Channels in Practice

## Week 94: Quantum Error Types | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Connect** theoretical channels to experimental parameters (T1, T2, gate errors)
2. **Model** realistic NISQ device noise
3. **Understand** the hierarchy of error timescales
4. **Apply** noise models to circuit simulation

---

## Core Content

### 1. The T1/T2 Hierarchy

Real qubits are characterized by two fundamental timescales:

**T1 (Energy Relaxation):** Time for excited state to decay
$$P_1(t) = P_1(0) e^{-t/T_1}$$

**T2 (Dephasing):** Time for coherence to decay
$$\rho_{01}(t) = \rho_{01}(0) e^{-t/T_2}$$

**Fundamental bound:** $T_2 \leq 2T_1$

When $T_2 = 2T_1$: pure amplitude damping limit
When $T_2 < 2T_1$: additional pure dephasing (T2* processes)

### 2. Combined T1/T2 Model

Total decoherence combines amplitude damping and pure dephasing:

$$\mathcal{E}_{T_1,T_2}(\rho) = \mathcal{E}_{AD}(\gamma_1) \circ \mathcal{E}_{PD}(\lambda)$$

where:
- $\gamma_1 = 1 - e^{-t/T_1}$
- $\lambda = 1 - e^{-t/T_\phi}$ with $\frac{1}{T_\phi} = \frac{1}{T_2} - \frac{1}{2T_1}$

### 3. Gate Error Models

**Single-qubit gate error:**
$$\mathcal{E}_{\text{gate}} = \mathcal{E}_{\text{dep}}(p_1) \circ \mathcal{U}$$
Typical $p_1 \approx 10^{-4}$ to $10^{-3}$

**Two-qubit gate error:**
$$\mathcal{E}_{\text{2Q}} = \mathcal{E}_{\text{dep}}(p_2) \circ \mathcal{U}_{2Q}$$
Typical $p_2 \approx 10^{-3}$ to $10^{-2}$

**Measurement error:**
- $P(\text{read } 0 | \text{state } 1) = \epsilon_{01}$
- $P(\text{read } 1 | \text{state } 0) = \epsilon_{10}$
Typical $\epsilon \approx 10^{-2}$

### 4. NISQ Noise Models

**Qiskit noise model structure:**
```python
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

noise_model = NoiseModel()
# Single-qubit depolarizing
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.001), ['u1', 'u2', 'u3'])
# Two-qubit depolarizing
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.01), ['cx'])
# Thermal relaxation during idle
noise_model.add_all_qubit_quantum_error(
    thermal_relaxation_error(t1, t2, gate_time), ['id'])
```

### 5. Error Budgets

For a circuit with depth $d$, $n$ qubits, $g_1$ single-qubit gates, $g_2$ two-qubit gates:

**Total error estimate:**
$$\epsilon_{\text{total}} \approx g_1 p_1 + g_2 p_2 + n \cdot d \cdot t_{\text{gate}}/T_2$$

This gives a rough estimate of circuit fidelity:
$$F \approx 1 - \epsilon_{\text{total}}$$

### 6. Correlated Errors

Real devices also have **correlated errors**:
- **Crosstalk:** Gates on one qubit affect neighbors
- **Leakage:** Population escapes the computational subspace
- **Cosmic rays:** Rare but catastrophic correlated errors

These are harder to model but increasingly important.

### 7. Typical Device Parameters (2024)

| Platform | T1 | T2 | 1Q Error | 2Q Error |
|----------|----|----|----------|----------|
| Superconducting | 50-100 μs | 50-150 μs | 0.01-0.1% | 0.3-1% |
| Trapped ions | 10-60 s | 1-10 s | 0.01-0.1% | 0.1-1% |
| Neutral atoms | 1-10 s | 0.1-1 s | 0.1-1% | 0.5-3% |
| Photonic | N/A | N/A | 0.01% | 1-10% |

### 8. The Error Correction Threshold

**Key question:** When is error correction beneficial?

Physical error rate $p$ must satisfy $p < p_{\text{threshold}}$:
- Surface code: $p_{\text{threshold}} \approx 1\%$
- Concatenated codes: $p_{\text{threshold}} \approx 0.01\%$

Current best devices are at or just below surface code threshold!

---

## Worked Example

**Problem:** A superconducting qubit has T1 = 50μs, T2 = 70μs, gate time = 20ns. Calculate the idle error per gate time.

**Solution:**
- $\gamma_1 = 1 - e^{-20\text{ns}/50\mu\text{s}} = 1 - e^{-0.0004} \approx 0.0004$
- $\lambda = 1 - e^{-20\text{ns}/70\mu\text{s}} \approx 0.00029$

Combined idle error per gate time: $\approx 0.0007 = 0.07\%$

For a circuit with 1000 gates: expected error $\approx 70\%$!

This is why error correction is essential for large circuits.

---

## Practice Problems

1. Calculate the coherent gate count before fidelity drops below 50% for a device with 0.1% error per gate.
2. Derive the relationship $T_2 \leq 2T_1$ from the structure of amplitude and phase damping.
3. Design a noise model for a 5-qubit system with realistic parameters.
4. Estimate the logical error rate for a distance-3 surface code with physical error rate 0.5%.

---

## Computational Lab

```python
"""Day 657: Error Channels in Practice"""

import numpy as np
import matplotlib.pyplot as plt

def t1_t2_noise(rho, t, T1, T2):
    """Combined T1/T2 noise model."""
    # Amplitude damping
    gamma = 1 - np.exp(-t/T1)
    # Pure dephasing (beyond T1 contribution)
    if T2 < 2*T1:
        T_phi = 1 / (1/T2 - 1/(2*T1))
        lam = 1 - np.exp(-t/T_phi)
    else:
        lam = 0

    # Apply amplitude damping
    K0_ad = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    K1_ad = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    rho_ad = K0_ad @ rho @ K0_ad.conj().T + K1_ad @ rho @ K1_ad.conj().T

    # Apply pure dephasing
    K0_pd = np.array([[1, 0], [0, np.sqrt(1-lam)]], dtype=complex)
    K1_pd = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    rho_out = K0_pd @ rho_ad @ K0_pd.conj().T + K1_pd @ rho_ad @ K1_pd.conj().T

    return rho_out

# Simulate realistic qubit decoherence
T1, T2 = 50, 30  # microseconds
times = np.linspace(0, 150, 100)

# Start in superposition
rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

p1_values = []
coherence_values = []

for t in times:
    rho = t1_t2_noise(rho_plus, t, T1, T2)
    p1_values.append(rho[1, 1].real)
    coherence_values.append(np.abs(rho[0, 1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(times, [0.5]*len(times), 'b--', alpha=0.5, label='Initial P(1)')
ax1.plot(times, p1_values, 'b-', linewidth=2, label='P(1) with noise')
ax1.axvline(x=T1, color='r', linestyle='--', alpha=0.5, label=f'T₁={T1}μs')
ax1.set_xlabel('Time (μs)')
ax1.set_ylabel('Population P(|1⟩)')
ax1.set_title('T₁ Decay')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(times, [0.5]*len(times), 'g--', alpha=0.5, label='Initial coherence')
ax2.plot(times, coherence_values, 'g-', linewidth=2, label='Coherence with noise')
ax2.axvline(x=T2, color='r', linestyle='--', alpha=0.5, label=f'T₂={T2}μs')
ax2.set_xlabel('Time (μs)')
ax2.set_ylabel('Coherence |ρ₀₁|')
ax2.set_title('T₂ Dephasing')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('t1_t2_combined.png', dpi=150)
plt.show()

# Circuit fidelity estimation
def circuit_fidelity(n_gates, error_per_gate):
    return (1 - error_per_gate)**n_gates

gate_counts = np.arange(1, 1001)
for error in [0.001, 0.005, 0.01]:
    fidelities = circuit_fidelity(gate_counts, error)
    plt.plot(gate_counts, fidelities, label=f'p={error}')

plt.xlabel('Number of gates')
plt.ylabel('Circuit fidelity')
plt.title('Fidelity Decay with Circuit Depth')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('circuit_fidelity.png', dpi=150)
plt.show()

print("Key insight: Fidelity decays exponentially with circuit depth!")
```

---

## Summary

- **T1**: energy relaxation time (amplitude damping)
- **T2**: total dephasing time ($T_2 \leq 2T_1$)
- **Gate errors**: typically modeled as depolarizing
- **Error budget**: count gates, estimate total error
- **Current devices**: near fault-tolerant threshold
- **Error correction**: essential for large-scale computation

---

## Preview: Day 658

Tomorrow: **Week 94 Review** - comprehensive integration of error types.
