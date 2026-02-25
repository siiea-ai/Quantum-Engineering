# Day 671: Comprehensive Problems - Semester 1B Integration

## Week 96: Semester 1B Review | Month 24: Quantum Channels & Error Introduction

---

## Overview

This day presents comprehensive problems that integrate concepts from all six months of Semester 1B:
- Month 19: Density Matrices
- Month 20: Entanglement
- Month 21: Open Systems
- Month 22: Quantum Algorithms
- Month 23: Quantum Channels
- Month 24: Error Correction

---

## Problem Set

### Problem 1: Complete Quantum Communication Protocol

**Scenario:** Alice wants to send a quantum state to Bob through a noisy channel.

**Setup:**
- Initial state: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$
- Channel: Depolarizing with $p = 0.1$
- Alice and Bob share Bell state $|\Phi^+\rangle$ for teleportation

**Tasks:**

a) Calculate the density matrix of $|\psi\rangle$ after direct transmission through the depolarizing channel.

b) If Alice uses quantum teleportation instead, but the Bell state has experienced the same depolarizing noise, what is the fidelity of the received state?

c) If Bob uses the 3-qubit bit-flip code to protect against transmission errors, and single X errors occur with probability $q = 0.05$, what is the probability of successful transmission?

**Solution:**

a) Direct transmission:
$$\rho_{out} = (1-p)|\psi\rangle\langle\psi| + p\frac{I}{2}$$
$$= 0.9|\psi\rangle\langle\psi| + 0.1\frac{I}{2}$$

Fidelity: $F = \langle\psi|\rho_{out}|\psi\rangle = 0.9 + 0.1 \times 0.5 = 0.95$

b) Depolarized Bell state:
$$\rho_{Bell} = 0.9|\Phi^+\rangle\langle\Phi^+| + 0.1\frac{I_4}{4}$$

Teleportation fidelity with imperfect Bell state:
$$F_{tele} = \frac{1}{2} + \frac{1}{2}F_{Bell} = \frac{1}{2} + \frac{1}{2}(0.9) = 0.95$$

c) With bit-flip code, failure requires 2+ errors:
$$P_{success} = (1-q)^3 + 3q(1-q)^2 = 0.95^3 + 3(0.05)(0.95)^2$$
$$= 0.857 + 0.135 = 0.992$$

---

### Problem 2: Decoherence During Computation

**Scenario:** A 3-qubit quantum circuit runs on a device with:
- T1 = 100 μs, T2 = 80 μs
- Single-qubit gate time: 20 ns
- Two-qubit gate time: 200 ns

**Circuit:**
1. Prepare $|000\rangle$
2. Apply H to qubit 1 (20 ns)
3. CNOT 1→2 (200 ns)
4. CNOT 1→3 (200 ns)
5. Measure (instant)

**Tasks:**

a) What is the total circuit time?

b) Write the Lindblad equation for amplitude damping on a single qubit.

c) Estimate the fidelity loss due to T1 and T2 decay during the circuit.

d) What is the final state in the ideal case, and what does it become after decoherence?

**Solution:**

a) Total time: 20 + 200 + 200 = 420 ns

b) Amplitude damping Lindblad:
$$\frac{d\rho}{dt} = \gamma\left(\sigma_-\rho\sigma_+ - \frac{1}{2}\{\sigma_+\sigma_-, \rho\}\right)$$
where $\gamma = 1/T_1$

c) Fidelity estimation:
- T1 decay: $1 - t/T_1 \approx 1 - 420\text{ns}/100\mu\text{s} = 1 - 0.0042 = 0.996$
- T2 decay on coherences: $e^{-t/T_2} = e^{-420/80000} \approx 0.995$
- With 3 qubits: Fidelity $\approx (0.995)^3 \approx 0.985$

d) Ideal final state: GHZ state
$$|\psi\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

After decoherence (approximate):
$$\rho \approx 0.985 |\psi\rangle\langle\psi| + 0.015 \rho_{noise}$$

---

### Problem 3: Algorithm with Errors

**Scenario:** Implementing Grover search on N=4 items with 1 marked.

**Tasks:**

a) How many Grover iterations are optimal?

b) If each oracle call has a 1% chance of applying the wrong phase (X error), what is the success probability after optimal iterations?

c) Design a simple error detection strategy.

**Solution:**

a) Optimal iterations: $k = \text{round}(\frac{\pi}{4}\sqrt{4}) = \text{round}(1.57) = 2$

After 2 iterations, success probability ≈ 1.0

b) With 1% error per oracle:
- 2 oracle calls → P(no error) = $(0.99)^2 = 0.98$
- If one error occurs, search may fail
- Approximate success: 0.98

c) Error detection:
- Run algorithm twice
- Compare results
- If different, repeat
- Acceptance threshold based on consistency

---

### Problem 4: Channel Tomography

**Scenario:** You have access to an unknown single-qubit channel $\mathcal{E}$.

**Tasks:**

a) How many parameters characterize a general single-qubit CPTP map?

b) Design a minimal tomography protocol (what input states and measurements).

c) Given the following data, identify the channel:
   - Input $|0\rangle$, output $|0\rangle\langle 0|$
   - Input $|1\rangle$, output $0.9|1\rangle\langle 1| + 0.1|0\rangle\langle 0|$
   - Input $|+\rangle$, output has coherence $|\rho_{01}| = 0.45$

**Solution:**

a) Single-qubit CPTP map: 12 parameters (16 for general map, minus 4 constraints)

b) Minimal protocol:
- Prepare states: $\{|0\rangle, |1\rangle, |+\rangle, |+i\rangle\}$
- Measure in X, Y, Z bases
- Reconstruct $\chi$ matrix or Choi matrix

c) Channel identification:
- $|0\rangle \to |0\rangle$: Ground state is fixed point
- $|1\rangle \to 0.9|1\rangle + 0.1|0\rangle$: Decay with $\gamma = 0.1$
- Coherence: $0.5 \to 0.45 = 0.5\sqrt{1-\gamma}$: $\sqrt{0.9} = 0.949$...

  Actually $0.45/0.5 = 0.9$, not $\sqrt{0.9}$

This suggests amplitude damping ($\gamma = 0.1$) plus additional dephasing.

Pure amplitude damping would give coherence $0.5\sqrt{0.9} \approx 0.474$, but we see 0.45.

So: Amplitude damping + pure dephasing.

---

### Problem 5: Entanglement in Error Correction

**Scenario:** The 3-qubit bit-flip code uses entanglement.

**Tasks:**

a) Calculate the entanglement entropy of the reduced state of qubit 1 for the encoded $|+_L\rangle$ state.

b) Show that syndrome measurement doesn't destroy the entanglement between logical qubits.

c) If the encoded state starts maximally entangled with a reference: $\frac{1}{\sqrt{2}}(|0_L\rangle|0_R\rangle + |1_L\rangle|1_R\rangle)$, what is the entanglement after a correctable X error?

**Solution:**

a) Encoded $|+_L\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$

Reduced state of qubit 1:
$$\rho_1 = \text{Tr}_{23}(|+_L\rangle\langle +_L|) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

Entanglement entropy: $S(\rho_1) = 1$ bit

b) Syndrome measurement projects onto eigenspaces of $Z_1Z_2$ and $Z_2Z_3$.

For the error-free case:
- $Z_1Z_2|+_L\rangle = +1|+_L\rangle$
- $Z_2Z_3|+_L\rangle = +1|+_L\rangle$

The state is already an eigenstate, so syndrome measurement doesn't change it!

c) After error $X_1$:
$$|\psi'\rangle = \frac{1}{\sqrt{2}}(|100\rangle|0_R\rangle + |011\rangle|1_R\rangle)$$

After syndrome measurement and correction:
$$|\psi_{corr}\rangle = \frac{1}{\sqrt{2}}(|000\rangle|0_R\rangle + |111\rangle|1_R\rangle)$$

Entanglement preserved! $E = 1$ ebit.

---

### Problem 6: Comprehensive Integration

**Scenario:** Design a fault-tolerant quantum teleportation protocol.

**Tasks:**

a) What are the resources needed (qubits, Bell states, classical bits)?

b) How does decoherence affect the protocol at each step?

c) If Alice's and Bob's halves of the Bell pair are encoded in the Shor code, how many physical qubits are needed?

d) What is the threshold error rate for the scheme to work?

**Solution:**

a) Resources:
- 1 qubit to teleport
- 1 Bell pair (2 qubits)
- 2 classical bits for measurement outcomes
- Total: 3 qubits, 2 cbits (without error correction)

b) Decoherence effects:
- Bell pair decoherence: Reduces fidelity directly
- During Alice's measurement: Can cause wrong outcome
- Classical communication: No quantum errors
- Bob's correction: Gate errors

c) With Shor code:
- Each logical qubit → 9 physical qubits
- Alice's state: 9 qubits
- Bell pair: 9 + 9 = 18 qubits
- Total: 27 physical qubits (plus ancillas for syndrome)

d) Threshold:
- Need physical error rate $\epsilon < \epsilon_{th}$
- For Shor code: $\epsilon_{th} \sim 10^{-4}$ to $10^{-3}$
- With surface code: $\epsilon_{th} \sim 1\%$

---

## Computational Lab

```python
"""Day 671: Comprehensive Problems - Semester 1B Integration"""

import numpy as np
from scipy.linalg import expm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)

def tensor(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print("Day 671: Comprehensive Problems")
print("=" * 60)

# ============================================
# Problem 1: Quantum Communication
# ============================================
print("\nPROBLEM 1: Quantum Communication")
print("-" * 40)

# Initial state |ψ⟩ = |+⟩
psi = (ket_0 + ket_1) / np.sqrt(2)
rho_psi = psi @ psi.conj().T

# Depolarizing channel
p = 0.1
rho_out = (1 - p) * rho_psi + p * I / 2

fidelity_direct = np.real((psi.conj().T @ rho_out @ psi)[0, 0])
print(f"a) Direct transmission fidelity: {fidelity_direct:.4f}")

# Teleportation with noisy Bell state
# F_tele = 1/2 + F_Bell/2 where F_Bell = 1-p for depolarizing
F_Bell = 1 - p
F_tele = 0.5 + 0.5 * F_Bell
print(f"b) Teleportation fidelity: {F_tele:.4f}")

# Bit-flip code protection
q = 0.05
P_success = (1-q)**3 + 3*q*(1-q)**2
print(f"c) Bit-flip code success probability: {P_success:.4f}")

# ============================================
# Problem 2: Decoherence During Computation
# ============================================
print("\n" + "=" * 60)
print("PROBLEM 2: Decoherence During Computation")
print("-" * 40)

T1 = 100e-6  # 100 μs
T2 = 80e-6   # 80 μs
t_circuit = 420e-9  # 420 ns

print(f"a) Total circuit time: {t_circuit*1e9:.0f} ns")

# Fidelity estimate
F_T1 = np.exp(-t_circuit / T1)
F_T2 = np.exp(-t_circuit / T2)
F_total = (F_T1 * F_T2) ** 3  # 3 qubits, rough estimate

print(f"c) Estimated fidelity: {F_total:.4f}")

# Simulate GHZ state creation
print("d) Creating GHZ state...")

# CNOT gate
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

# Initial |000⟩
psi_init = tensor(ket_0, ket_0, ket_0)

# H on qubit 1
H_II = tensor(H, I, I)
psi_1 = H_II @ psi_init

# CNOT 1→2 (expand to 3 qubits)
CNOT_I = tensor(CNOT, I)
psi_2 = CNOT_I @ psi_1

# CNOT 1→3 (need to reorder)
I_CNOT = np.zeros((8, 8), dtype=complex)
for i in range(2):
    for j in range(2):
        for k in range(2):
            i_new = i
            k_new = k ^ i  # XOR for CNOT
            I_CNOT[4*i_new + 2*j + k_new, 4*i + 2*j + k] = 1

psi_final = I_CNOT @ psi_2

print(f"   GHZ state amplitudes: {psi_final.flatten()}")
print(f"   Expected: [1/√2, 0, 0, 0, 0, 0, 0, 1/√2]")

# ============================================
# Problem 3: Grover with Errors
# ============================================
print("\n" + "=" * 60)
print("PROBLEM 3: Grover with Errors")
print("-" * 40)

N = 4
k_opt = int(np.round(np.pi / 4 * np.sqrt(N)))
print(f"a) Optimal iterations: {k_opt}")

p_error = 0.01
P_no_error = (1 - p_error) ** (2 * k_opt)  # 2 oracle calls per iteration
print(f"b) Probability of no oracle error: {P_no_error:.4f}")

# ============================================
# Problem 5: Entanglement in Error Correction
# ============================================
print("\n" + "=" * 60)
print("PROBLEM 5: Entanglement in QEC")
print("-" * 40)

# Encoded |+_L⟩
ket_0L = tensor(ket_0, ket_0, ket_0)
ket_1L = tensor(ket_1, ket_1, ket_1)
plus_L = (ket_0L + ket_1L) / np.sqrt(2)

# Reduced density matrix of qubit 1
rho_123 = plus_L @ plus_L.conj().T
rho_123_reshaped = rho_123.reshape(2, 4, 2, 4)
rho_1 = np.trace(rho_123_reshaped, axis1=1, axis2=3)

eigenvalues = np.real(np.linalg.eigvalsh(rho_1))
entropy = -sum(e * np.log2(e) if e > 1e-10 else 0 for e in eigenvalues)

print(f"a) Qubit 1 reduced state:")
print(f"   ρ_1 = {np.real(rho_1)}")
print(f"   Entanglement entropy: {entropy:.4f} bits")

# Check syndrome doesn't disturb
Z1Z2 = tensor(Z, Z, I)
Z2Z3 = tensor(I, Z, Z)

ev1 = np.real((plus_L.conj().T @ Z1Z2 @ plus_L)[0, 0])
ev2 = np.real((plus_L.conj().T @ Z2Z3 @ plus_L)[0, 0])

print(f"\nb) Syndrome eigenvalues: Z1Z2 = {ev1:+.0f}, Z2Z3 = {ev2:+.0f}")
print("   State is eigenstate, so syndrome measurement preserves it!")

# Entanglement with reference
print("\nc) Entanglement with reference after error:")
# |Φ⟩ = (|0L⟩|0R⟩ + |1L⟩|1R⟩)/√2
# After X1 error: (|100⟩|0R⟩ + |011⟩|1R⟩)/√2
# After correction: (|000⟩|0R⟩ + |111⟩|1R⟩)/√2 - same entanglement!
print("   Original: 1 ebit")
print("   After X1 error + correction: 1 ebit (preserved!)")

# ============================================
# Problem 6: Resource Counting
# ============================================
print("\n" + "=" * 60)
print("PROBLEM 6: Fault-Tolerant Teleportation")
print("-" * 40)

print("a) Basic teleportation resources:")
print("   - Qubits: 3 (1 to teleport + 2 in Bell pair)")
print("   - Classical bits: 2")

print("\nc) With Shor code [[9,1,3]]:")
print("   - Alice's state: 9 physical qubits")
print("   - Bell pair: 9 + 9 = 18 physical qubits")
print("   - Total: 27 physical qubits")
print("   - Plus ancillas for syndrome measurement")

print("\n" + "=" * 60)
print("Comprehensive Problems Complete!")
```

---

## Summary

### Cross-Topic Connections

| Topic A | Topic B | Connection |
|---------|---------|------------|
| Density matrices | Channels | Channels map density matrices |
| Entanglement | Teleportation | Bell states enable teleportation |
| Open systems | Error channels | Lindblad → Kraus operators |
| Algorithms | Error correction | Fault-tolerance enables algorithms |

### Key Integration Points

1. **Decoherence limits computation** → Need error correction
2. **Channels model all noise** → Unifying framework
3. **Entanglement is a resource** → Teleportation, QEC
4. **Stabilizers unify QEC** → Systematic code design

---

## Preview: Day 672

Tomorrow: **Semester 1B Complete** - Summary and Year 2 preview!
