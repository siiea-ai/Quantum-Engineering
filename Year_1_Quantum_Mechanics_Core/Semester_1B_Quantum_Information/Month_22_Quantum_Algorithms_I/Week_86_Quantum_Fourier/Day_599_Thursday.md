# Day 599: Phase Kickback and QFT

## Overview

**Day 599** | Week 86, Day 4 | Month 22 | Quantum Algorithms I

Today we explore the crucial connection between phase kickback and the QFT. Phase kickback is the mechanism by which eigenvalue information is encoded into ancilla qubits, forming the foundation of quantum phase estimation. Understanding this connection is essential for Shor's algorithm and many other quantum applications.

---

## Learning Objectives

1. Understand phase kickback with controlled-U operations
2. Connect phase kickback to eigenvalue problems
3. See how QFT extracts phase information
4. Derive the phase estimation intuition
5. Understand the role of QFT in period finding
6. Prepare for formal phase estimation (next week)

---

## Core Content

### Phase Kickback Review

Recall from Week 85: when a controlled-U acts on an eigenstate, the phase "kicks back" to the control qubit.

If $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$, then:

$$CU|+\rangle|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle|\psi\rangle + e^{2\pi i\phi}|1\rangle|\psi\rangle)$$
$$= \frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i\phi}|1\rangle)|\psi\rangle$$

The phase $e^{2\pi i\phi}$ appears on the control qubit, not the target!

### Multiple Controlled Operations

What if we apply controlled-$U^{2^k}$ operations?

$$CU^{2^k}|+\rangle|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 2^k \phi}|1\rangle)|\psi\rangle$$

The phase becomes $2^k \phi$!

### Phase Kickback with Multiple Ancillas

Consider $n$ ancilla qubits, each controlling a different power of $U$:

```
|0⟩ ─[H]─●──────────────────── (controls U^{2^{n-1}})
|0⟩ ─[H]─┼─●────────────────── (controls U^{2^{n-2}})
 ⋮       │ │
|0⟩ ─[H]─┼─┼─●──────────────── (controls U^2)
|0⟩ ─[H]─┼─┼─┼─●────────────── (controls U^1)
         │ │ │ │
|ψ⟩ ─────U²ⁿ⁻¹U²ⁿ⁻²...U²─U¹───
```

After all controlled operations:

$$\frac{1}{\sqrt{2^n}}\bigotimes_{j=1}^{n}(|0\rangle + e^{2\pi i \cdot 2^{n-j}\phi}|1\rangle) \otimes |\psi\rangle$$

### Recognizing the QFT Structure

The ancilla state is:
$$\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi ik\phi}|k\rangle$$

Let $\tilde{\phi} = 2^n \phi$. If $\tilde{\phi}$ is an integer, this is exactly:
$$\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi ik\tilde{\phi}/2^n}|k\rangle = QFT|\tilde{\phi}\rangle$$

**Key Insight:** The phase kickback creates the QFT of the phase (encoded in binary)!

### Inverse QFT Extracts the Phase

If we apply $QFT^{-1}$ to the ancilla register:
$$QFT^{-1}\left(\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi ik\tilde{\phi}/2^n}|k\rangle\right) = |\tilde{\phi}\rangle$$

Measuring gives us $\tilde{\phi}$, hence $\phi = \tilde{\phi}/2^n$!

This is the essence of **quantum phase estimation** (Week 87).

### Phase Kickback in Period Finding

For Shor's algorithm, we use:
$$U_a|x\rangle = |ax \mod N\rangle$$

If $a^r \equiv 1 \pmod{N}$ (period $r$), then the eigenstates are:
$$|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1} e^{-2\pi ijs/r}|a^j \mod N\rangle$$

with eigenvalues $e^{2\pi is/r}$.

Phase estimation gives us $s/r$, from which we extract $r$!

### QFT as Fourier Analysis

The QFT acts like frequency analysis:
- Input: time-domain signal (amplitudes)
- Output: frequency-domain (phase encodes period information)

When combined with phase kickback:
- Eigenvalue phases encode the "frequency"
- Inverse QFT reads out this frequency

### Binary Phase Representation

If $\phi = 0.b_1 b_2 \cdots b_n$ (binary fraction), then:

$$e^{2\pi i \cdot 2^{n-j}\phi} = e^{2\pi i \cdot 2^{n-j} \cdot 0.b_1\cdots b_n} = e^{2\pi i \cdot b_{j}.b_{j+1}\cdots b_n}$$

The integer part contributes nothing (full rotations), only fractional part matters.

This means each ancilla qubit carries information about one binary digit of $\phi$!

---

## Worked Examples

### Example 1: Single Ancilla Phase Kickback

Let $U = R_z(\theta)$ with eigenstate $|1\rangle$ and eigenvalue $e^{i\theta/2}$.

Show phase kickback for controlled-$R_z$.

**Solution:**

$R_z(\theta)|1\rangle = e^{i\theta/2}|1\rangle$

For controlled-$R_z$ with ancilla in $|+\rangle$:

$$CR_z|+\rangle|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle|1\rangle + |1\rangle R_z|1\rangle)$$
$$= \frac{1}{\sqrt{2}}(|0\rangle + e^{i\theta/2}|1\rangle)|1\rangle$$

The phase $e^{i\theta/2}$ kicked back to the ancilla!

### Example 2: Two Ancillas

With $\phi = 0.25 = 1/4$, show the state after phase kickback with 2 ancillas.

**Solution:**

$\phi = 0.01$ in binary (0.25 = 0/2 + 1/4)

After controlled-$U$ and controlled-$U^2$:
- Ancilla 1 (controls $U^2$): phase $e^{2\pi i \cdot 2 \cdot 0.25} = e^{i\pi} = -1$
- Ancilla 2 (controls $U$): phase $e^{2\pi i \cdot 1 \cdot 0.25} = e^{i\pi/2} = i$

State: $\frac{1}{2}(|0\rangle - |1\rangle) \otimes (|0\rangle + i|1\rangle)$
$= \frac{1}{2}(|00\rangle + i|01\rangle - |10\rangle - i|11\rangle)$

After inverse QFT: $|01\rangle = |1\rangle$ in decimal, so $\tilde{\phi} = 1$, $\phi = 1/4 = 0.25$ ✓

### Example 3: Non-Integer Phase

With $\phi = 0.3$ (not exactly representable in 2 bits), what happens?

**Solution:**

$2^2 \cdot 0.3 = 1.2$

The phases are:
- Ancilla 1: $e^{2\pi i \cdot 2 \cdot 0.3} = e^{1.2\pi i}$
- Ancilla 2: $e^{2\pi i \cdot 0.3} = e^{0.6\pi i}$

After inverse QFT, we don't get a clean basis state. Instead, we get a superposition peaked near $|1\rangle$ and $|2\rangle$.

Measurement gives $1$ with probability $\sim 0.85$ and $2$ with probability $\sim 0.09$.

This is the **precision limitation** of phase estimation!

---

## Practice Problems

### Problem 1: Phase Kickback Verification

For $U = Z$ with eigenstate $|1\rangle$ (eigenvalue $-1 = e^{i\pi}$), verify that:
- Two controlled-$Z$ operations give phase $(-1)^2 = 1$
- One controlled-$Z$ gives phase $-1$

### Problem 2: Three-Ancilla Example

With $\phi = 0.625 = 5/8 = 0.101$ in binary, compute the ancilla state after phase kickback with 3 ancillas.

### Problem 3: Fourier Analysis Connection

Show that if the input to inverse QFT is $\frac{1}{\sqrt{N}}\sum_k e^{2\pi ik\phi}|k\rangle$ and $\phi = j/N$ for integer $j$, then the output is exactly $|j\rangle$.

### Problem 4: Period Finding Setup

For $a = 2$, $N = 15$ (so $r = 4$), write the eigenstate $|u_1\rangle$ and its eigenvalue.

---

## Computational Lab

```python
"""Day 599: Phase Kickback and QFT"""
import numpy as np
import matplotlib.pyplot as plt

def qft_matrix(n):
    """QFT matrix for n qubits"""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for k in range(N)]
                     for j in range(N)]) / np.sqrt(N)

def inverse_qft(n):
    """Inverse QFT matrix"""
    return qft_matrix(n).conj().T

def phase_kickback_state(phi, n):
    """
    Create the state after phase kickback with n ancillas
    |ψ⟩ = (1/√2^n) Σ_k e^{2πikφ} |k⟩
    """
    N = 2**n
    state = np.array([np.exp(2j * np.pi * k * phi) for k in range(N)])
    return state / np.sqrt(N)

def demonstrate_phase_kickback(phi, n, verbose=True):
    """
    Demonstrate phase kickback and extraction via inverse QFT
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Phase Kickback Demo: φ = {phi}, n = {n} ancillas")
        print(f"{'='*50}")

    # State after phase kickback
    state_after_kickback = phase_kickback_state(phi, n)

    if verbose:
        print(f"\nPhase angles on each ancilla:")
        for j in range(n):
            power = 2**(n-1-j)
            angle = 2 * np.pi * power * phi
            print(f"  Ancilla {j+1} (controls U^{power}): "
                  f"phase = {angle/np.pi:.4f}π = {np.degrees(angle):.1f}°")

    # Apply inverse QFT
    QFT_inv = inverse_qft(n)
    state_after_qft = QFT_inv @ state_after_kickback

    # Measurement probabilities
    probs = np.abs(state_after_qft)**2

    if verbose:
        print(f"\nAfter inverse QFT, measurement probabilities:")
        for k in range(2**n):
            if probs[k] > 0.001:
                estimated_phi = k / (2**n)
                print(f"  |{k:0{n}b}⟩ = |{k}⟩: P = {probs[k]:.4f} "
                      f"(estimates φ = {k}/{2**n} = {estimated_phi:.4f})")

    # Most likely outcome
    k_max = np.argmax(probs)
    phi_estimate = k_max / (2**n)

    if verbose:
        print(f"\nMost likely outcome: {k_max} → φ ≈ {phi_estimate}")
        print(f"True φ = {phi}, Error = {abs(phi - phi_estimate):.4f}")

    return phi_estimate, probs

# Test with exact phases
print("=" * 60)
print("EXACT PHASE EXAMPLES")
print("=" * 60)

for phi in [0.25, 0.5, 0.75, 0.125]:
    demonstrate_phase_kickback(phi, 3)

# Test with non-exact phases
print("\n" + "=" * 60)
print("NON-EXACT PHASE EXAMPLES")
print("=" * 60)

for phi in [0.3, 0.7, 0.15]:
    demonstrate_phase_kickback(phi, 4)

# Precision analysis
print("\n" + "=" * 60)
print("PRECISION ANALYSIS")
print("=" * 60)

phi_true = 0.3
print(f"\nTrue phase φ = {phi_true}")
print(f"\n| n qubits | Best estimate | Error     | P(best)  |")
print(f"|----------|---------------|-----------|----------|")

for n in range(2, 9):
    phi_est, probs = demonstrate_phase_kickback(phi_true, n, verbose=False)
    error = abs(phi_true - phi_est)
    p_best = np.max(probs)
    print(f"| {n:8d} | {phi_est:13.6f} | {error:9.6f} | {p_best:8.4f} |")

# Visualize probability distributions
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

phi_values = [0.25, 0.3, 0.625, 0.7]
n = 4
N = 2**n

for ax, phi in zip(axes.flat, phi_values):
    _, probs = demonstrate_phase_kickback(phi, n, verbose=False)

    colors = ['blue' if probs[k] == max(probs) else 'lightblue'
              for k in range(N)]

    ax.bar(range(N), probs, color=colors, edgecolor='black', alpha=0.7)
    ax.axvline(x=phi * N, color='red', linestyle='--', linewidth=2,
               label=f'True: {phi}×{N}={phi*N:.1f}')
    ax.set_xlabel('Measurement outcome k')
    ax.set_ylabel('Probability')
    ax.set_title(f'φ = {phi} ({n} qubits)')
    ax.legend()
    ax.set_xticks(range(0, N, 2))

plt.suptitle('Phase Estimation via QFT: Measurement Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('phase_kickback_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Probability distributions saved to 'phase_kickback_distributions.png'")

# Phase kickback circuit demonstration
print("\n" + "=" * 60)
print("PHASE KICKBACK CIRCUIT SIMULATION")
print("=" * 60)

def simulate_controlled_U_power(ancilla_state, eigenvalue, power):
    """
    Simulate controlled-U^power operation
    ancilla_state: 2-element array [a, b] for a|0⟩ + b|1⟩
    eigenvalue: e^{2πiφ}
    power: apply U^power
    """
    phase = eigenvalue ** power
    # Controlled operation: apply phase to |1⟩ component
    return np.array([ancilla_state[0], ancilla_state[1] * phase])

def full_phase_kickback_simulation(phi, n):
    """
    Simulate the full phase kickback circuit step by step
    """
    eigenvalue = np.exp(2j * np.pi * phi)
    N = 2**n

    # Initialize ancilla qubits to |+⟩
    ancilla_states = [np.array([1, 1]) / np.sqrt(2) for _ in range(n)]

    print(f"\nSimulating phase kickback for φ = {phi} with {n} ancillas")
    print("-" * 50)

    # Apply controlled-U^{2^{n-1-j}} to ancilla j
    for j in range(n):
        power = 2**(n-1-j)
        ancilla_states[j] = simulate_controlled_U_power(
            ancilla_states[j], eigenvalue, power
        )
        phase_on_1 = np.angle(ancilla_states[j][1] / ancilla_states[j][0])
        print(f"After CU^{power:3d} on ancilla {j+1}: "
              f"phase on |1⟩ = {phase_on_1/np.pi:.4f}π")

    # Tensor product of all ancilla states
    full_state = ancilla_states[0]
    for j in range(1, n):
        full_state = np.kron(full_state, ancilla_states[j])

    print(f"\nFull ancilla state (first 4 amplitudes):")
    for k in range(min(4, N)):
        print(f"  |{k:0{n}b}⟩: {full_state[k]:.4f}")

    # Apply inverse QFT
    QFT_inv = inverse_qft(n)
    output_state = QFT_inv @ full_state

    print(f"\nAfter inverse QFT:")
    probs = np.abs(output_state)**2
    for k in range(N):
        if probs[k] > 0.01:
            print(f"  |{k:0{n}b}⟩ = |{k}⟩: amplitude = {output_state[k]:.4f}, "
                  f"P = {probs[k]:.4f}")

    return full_state, output_state

# Run simulation for exact and inexact phases
full_phase_kickback_simulation(0.25, 3)
full_phase_kickback_simulation(0.3, 3)

# Connection to period finding
print("\n" + "=" * 60)
print("CONNECTION TO PERIOD FINDING")
print("=" * 60)

def period_finding_eigenvalue(a, N, s, r):
    """
    Eigenvalue for modular exponentiation
    |u_s⟩ is eigenstate of U_a with eigenvalue e^{2πis/r}
    """
    return np.exp(2j * np.pi * s / r)

# Example: a=2, N=15, r=4
a, N, r = 2, 15, 4
print(f"\nPeriod finding for a={a} mod N={N} (period r={r})")
print(f"\nEigenvalues for different s values:")

for s in range(r):
    eigenvalue = period_finding_eigenvalue(a, N, s, r)
    phase = s / r
    print(f"  s={s}: eigenvalue = e^(2πi·{s}/{r}) = e^(2πi·{phase:.4f})")
    print(f"        = {eigenvalue:.4f}")

print(f"\nPhase estimation with 4 ancillas:")
for s in range(r):
    phi = s / r
    phi_est, probs = demonstrate_phase_kickback(phi, 4, verbose=False)
    k_max = np.argmax(probs)
    print(f"  s={s}: φ={phi:.4f} → measure {k_max}/16 = {k_max/16:.4f}")
    print(f"        s/r = {s}/{r} can be recovered via continued fractions")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Phase kickback | $CU\|+\rangle\|\psi\rangle = \frac{1}{\sqrt{2}}(\|0\rangle + e^{2\pi i\phi}\|1\rangle)\|\psi\rangle$ |
| Multiple powers | $CU^{2^k}$ gives phase $e^{2\pi i \cdot 2^k \phi}$ |
| After kickback | $\frac{1}{\sqrt{2^n}}\sum_k e^{2\pi ik\phi}\|k\rangle$ |
| Inverse QFT | Extracts $\tilde{\phi} = 2^n\phi$ (if integer) |

### Key Takeaways

1. **Phase kickback** transfers eigenvalue information to control qubits
2. **Multiple controlled operations** encode phase in binary
3. **Inverse QFT** decodes the phase information
4. **Precision** increases with number of ancilla qubits
5. **Non-exact phases** give probabilistic estimates peaked at nearest value
6. **Period finding** uses this to extract $s/r$ from eigenstates

---

## Daily Checklist

- [ ] I understand phase kickback with controlled-U
- [ ] I can trace multiple controlled operations
- [ ] I see how inverse QFT extracts the phase
- [ ] I understand precision limitations
- [ ] I see the connection to period finding
- [ ] I ran the lab and observed phase estimation behavior

---

*Next: Day 600 - Inverse QFT*
