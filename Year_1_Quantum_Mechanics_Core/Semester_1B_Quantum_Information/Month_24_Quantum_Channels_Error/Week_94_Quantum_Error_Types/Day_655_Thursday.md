# Day 655: Depolarizing Channel Analysis

## Week 94: Quantum Error Types | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Analyze** the depolarizing channel in depth
2. **Understand** its role as "worst-case" symmetric noise
3. **Compute** channel capacity and fidelity metrics
4. **Apply** depolarizing noise to circuit analysis

---

## Core Content

### 1. The Depolarizing Channel

$$\boxed{\mathcal{E}_{\text{dep}}(\rho) = (1-p)\rho + p\frac{I}{d}}$$

For qubits ($d=2$):
$$\mathcal{E}_{\text{dep}}(\rho) = (1-p)\rho + \frac{p}{2}I = \left(1-\frac{3p}{4}\right)\rho + \frac{p}{4}(X\rho X + Y\rho Y + Z\rho Z)$$

**Interpretation:** With probability $p$, the state is replaced by maximally mixed state.

### 2. Kraus Representation

$$K_0 = \sqrt{1-\frac{3p}{4}}I, \quad K_1 = \frac{\sqrt{p}}{2}X, \quad K_2 = \frac{\sqrt{p}}{2}Y, \quad K_3 = \frac{\sqrt{p}}{2}Z$$

**Verification:** $\sum_k K_k^\dagger K_k = (1-\frac{3p}{4})I + \frac{p}{4}(I+I+I) = I$ ✓

### 3. Bloch Sphere Effect

The depolarizing channel uniformly contracts the Bloch sphere:

$$\vec{r} \mapsto (1-p)\vec{r}$$

- Contracts by factor $(1-p)$ in ALL directions
- Most symmetric possible noise
- Fixed point: only $I/2$ (center of Bloch ball)

### 4. Purity Under Depolarizing

For pure input state ($\text{Tr}(\rho^2) = 1$):
$$\text{Tr}(\mathcal{E}_{\text{dep}}(\rho)^2) = (1-p)^2 + \frac{p^2}{2} + p(1-p) = 1 - \frac{3p}{2} + \frac{3p^2}{4}$$

For large $p$, purity approaches $1/2$ (maximally mixed).

### 5. Channel Capacity

The **quantum capacity** of the depolarizing channel:
$$Q = 1 - H_2\left(\frac{3p}{4}\right) - \frac{3p}{4}\log_2 3$$

for $p < p_{\text{threshold}}$, and $Q = 0$ for larger $p$.

**Classical capacity** is higher—quantum information is more fragile.

### 6. Average Gate Fidelity

For depolarizing noise after a gate $U$:
$$F_{\text{avg}} = \frac{d \cdot F_e + 1}{d + 1} = 1 - \frac{d}{d+1}p$$

For qubits: $F_{\text{avg}} = 1 - \frac{2p}{3}$

### 7. Depolarizing as Twirled Channel

Any channel, when Pauli-twirled, becomes depolarizing:
$$\text{Twirl}(\mathcal{E}) = \mathcal{E}_{\text{dep}}$$

with effective $p$ determined by average error rate.

### 8. Repeated Application

After $n$ depolarizing channels with parameter $p$:
$$(1-p)^n \to 0 \text{ as } n \to \infty$$

State converges to $I/2$ regardless of initial state.

---

## Worked Example

**Problem:** A qubit gate has depolarizing error $p = 0.01$. What is the fidelity after 100 gates?

**Solution:**
- Effective contraction: $(1-0.01)^{100} = 0.99^{100} \approx 0.366$
- Final Bloch vector magnitude: $0.366 |\vec{r}_0|$
- For pure initial state: fidelity with original $\approx (1 + 0.366)/2 = 0.683$

---

## Practice Problems

1. Prove the depolarizing channel is the only single-qubit channel invariant under all unitary rotations.
2. Calculate the diamond distance between depolarizing channels with $p_1$ and $p_2$.
3. Find the composition of two depolarizing channels.
4. Determine the threshold $p$ where quantum capacity becomes zero.

---

## Computational Lab

```python
"""Day 655: Depolarizing Channel Analysis"""

import numpy as np
import matplotlib.pyplot as plt

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def depolarizing_channel(rho, p):
    """Apply depolarizing channel."""
    return (1 - p) * rho + p * I / 2

def bloch_radius(rho):
    """Compute Bloch vector magnitude."""
    r_x = 2 * np.real(rho[0, 1])
    r_y = 2 * np.imag(rho[1, 0])
    r_z = np.real(rho[0, 0] - rho[1, 1])
    return np.sqrt(r_x**2 + r_y**2 + r_z**2)

# Uniform contraction visualization
p_values = np.linspace(0, 1, 50)
contractions = [1 - p for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, contractions, 'b-', linewidth=2)
plt.xlabel('Depolarizing probability p')
plt.ylabel('Bloch sphere radius')
plt.title('Depolarizing Channel: Uniform Bloch Sphere Contraction')
plt.grid(True, alpha=0.3)
plt.savefig('depolarizing_contraction.png', dpi=150)
plt.show()

# Track purity decay
rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)
p = 0.1
purities = []
rho = rho_0.copy()
for n in range(51):
    purities.append(np.real(np.trace(rho @ rho)))
    rho = depolarizing_channel(rho, p)

plt.figure(figsize=(10, 6))
plt.plot(range(51), purities, 'g-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Maximally mixed')
plt.xlabel('Number of applications')
plt.ylabel('Purity')
plt.title(f'Purity Decay Under Depolarizing (p={p})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('depolarizing_purity.png', dpi=150)
plt.show()

print("Depolarizing channel is the 'democratic' noise - treats all errors equally.")
```

---

## Summary

- **Depolarizing channel**: $(1-p)\rho + pI/d$ - uniform contraction toward maximally mixed
- **Most symmetric noise**: treats X, Y, Z errors equally
- **Unique fixed point**: $I/d$
- **Any channel twirls to depolarizing** - serves as average error model
- **Widely used** in theoretical analysis and benchmarking

---

## Preview: Day 656

Tomorrow: **Amplitude Damping** - the physical process of energy decay.
