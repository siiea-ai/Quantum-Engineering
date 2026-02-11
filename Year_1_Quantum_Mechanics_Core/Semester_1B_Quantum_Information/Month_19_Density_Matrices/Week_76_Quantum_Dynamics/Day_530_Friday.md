# Day 530: Important Quantum Channels

## Overview
**Day 530** | Week 76, Day 5 | Year 1, Month 19 | Common Noise Models

Today we study the most important quantum channels used in quantum information theory.

---

## Learning Objectives
1. Derive Kraus operators for depolarizing channel
2. Understand dephasing (phase damping) channel
3. Analyze amplitude damping (T₁ decay)
4. Compute channel fidelities
5. Visualize noise effects on Bloch sphere

---

## Core Content

### Depolarizing Channel

**Action:** Replaces state with maximally mixed with probability p

$$\mathcal{E}_{depol}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

**Kraus operators:**
- K₀ = √(1-3p/4) I
- K₁ = √(p/4) X
- K₂ = √(p/4) Y
- K₃ = √(p/4) Z

### Dephasing Channel (Phase Damping)

**Action:** Destroys off-diagonal coherences

$$\mathcal{E}_{deph}(\rho) = (1-p)\rho + p Z\rho Z$$

**Kraus operators:**
- K₀ = √(1-p) I
- K₁ = √p Z

**Effect on Bloch sphere:** Shrinks toward z-axis

### Amplitude Damping

**Action:** Models energy relaxation (T₁)

$$K_0 = |0\rangle\langle 0| + \sqrt{1-\gamma}|1\rangle\langle 1|$$
$$K_1 = \sqrt{\gamma}|0\rangle\langle 1|$$

**Effect:** |1⟩ decays to |0⟩ with probability γ

### Channel Comparison

| Channel | Coherences | Populations | Bloch Effect |
|---------|-----------|-------------|--------------|
| Depolarizing | Shrink uniformly | Mix toward I/2 | Uniform shrink |
| Dephasing | Shrink | Preserved | Shrink to z-axis |
| Amp. damping | Shrink | Decay to \|0⟩ | Shrink to north pole |

---

## Computational Lab

```python
"""Day 530: Important Quantum Channels"""
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
I = np.eye(2, dtype=complex)

def depolarizing(rho, p):
    return (1-p)*rho + p*I/2

def dephasing(rho, p):
    return (1-p)*rho + p*Z@rho@Z

def amplitude_damping(rho, gamma):
    K0 = np.array([[1,0],[0,np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0,np.sqrt(gamma)],[0,0]], dtype=complex)
    return K0@rho@K0.conj().T + K1@rho@K1.conj().T

# Compare channels on |+⟩
rho_plus = 0.5*np.array([[1,1],[1,1]], dtype=complex)

p_vals = np.linspace(0, 1, 50)
coherence_depol = [np.abs(depolarizing(rho_plus, p)[0,1]) for p in p_vals]
coherence_deph = [np.abs(dephasing(rho_plus, p)[0,1]) for p in p_vals]
coherence_amp = [np.abs(amplitude_damping(rho_plus, p)[0,1]) for p in p_vals]

plt.plot(p_vals, coherence_depol, label='Depolarizing')
plt.plot(p_vals, coherence_deph, label='Dephasing')
plt.plot(p_vals, coherence_amp, label='Amplitude damping')
plt.xlabel('Noise parameter')
plt.ylabel('|ρ₀₁| (coherence)')
plt.legend()
plt.title('Coherence Decay Under Different Channels')
plt.savefig('channels_comparison.png', dpi=150)
plt.show()
```

---

## Summary
- Depolarizing: uniform noise, shrinks Bloch ball
- Dephasing: destroys coherences, preserves populations
- Amplitude damping: energy decay toward |0⟩

---
*Next: Day 531 — Month Integration*
