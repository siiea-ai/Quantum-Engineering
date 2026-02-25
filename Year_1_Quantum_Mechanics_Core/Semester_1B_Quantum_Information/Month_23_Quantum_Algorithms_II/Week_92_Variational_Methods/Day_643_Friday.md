# Day 643: Barren Plateaus

## Overview
**Day 643** | Week 92, Day 6 | Year 1, Month 23 | Variational Methods

Today we study barren plateaus - the phenomenon of exponentially vanishing gradients that makes training variational quantum circuits difficult.

---

## Learning Objectives

1. Define barren plateaus mathematically
2. Understand when barren plateaus occur
3. Analyze the impact on trainability
4. Learn mitigation strategies
5. Connect to circuit expressibility
6. Appreciate the fundamental challenge

---

## Core Content

### What Are Barren Plateaus?

**Definition:** A cost function exhibits a barren plateau if:
$$\text{Var}_\theta[\partial_k C(\theta)] \leq O(e^{-cn})$$

for some constant $c > 0$ and all parameters $k$.

**Implication:** Gradients vanish exponentially with system size, making optimization infeasible.

### The McClean et al. Result (2018)

**Theorem:** For random parameterized circuits forming 2-designs, the variance of gradients vanishes as:

$$\text{Var}[\partial_k C] = O\left(\frac{1}{2^n}\right)$$

**Consequence:** For $n = 50$ qubits, gradients are $\sim 10^{-15}$ - undetectable!

### When Do Barren Plateaus Occur?

1. **Deep random circuits:** High expressibility → barren plateaus
2. **Global cost functions:** $C = \text{Tr}(O\rho)$ with global $O$
3. **Hardware noise:** Even shallow circuits can exhibit
4. **Entanglement:** Too much entanglement hurts

### Impact on Training

**Without barren plateau:**
- Gradients $\sim O(1)$
- Converge in polynomial iterations

**With barren plateau:**
- Need $O(2^n)$ measurements to estimate gradient
- Training becomes classically hard

### Mitigation Strategies

**1. Local Cost Functions:**
Use cost functions that only involve local observables.
$$C = \sum_i \text{Tr}(O_i \rho_i)$$

where $O_i$ acts on few qubits.

**2. Shallow Circuits:**
Limit depth to $O(\log n)$ to avoid barren plateaus.

**3. Structured Ansatze:**
Problem-specific circuits often avoid the issue.

**4. Layer-wise Training:**
Train one layer at a time, then fine-tune.

**5. Identity Initialization:**
Start near identity, grow circuit gradually.

### The Expressibility-Trainability Tradeoff

$$\text{More Expressive} \Rightarrow \text{Barren Plateaus} \Rightarrow \text{Hard to Train}$$

$$\text{Less Expressive} \Rightarrow \text{No Barren Plateaus} \Rightarrow \text{May Not Reach Solution}$$

Finding the sweet spot is an active research area.

---

## Computational Lab

```python
"""Day 643: Barren Plateaus"""
import numpy as np
import matplotlib.pyplot as plt

def random_unitary(n):
    """Generate random unitary (Haar measure)."""
    Z = np.random.randn(2**n, 2**n) + 1j * np.random.randn(2**n, 2**n)
    Q, R = np.linalg.qr(Z)
    return Q

def compute_gradient_variance(n_qubits, n_samples=100, observable='global'):
    """Estimate gradient variance for random circuits."""
    gradients = []

    for _ in range(n_samples):
        # Random parameterized circuit (simplified as random unitary)
        U = random_unitary(n_qubits)

        # Initial state |0...0⟩
        psi_0 = np.zeros(2**n_qubits)
        psi_0[0] = 1

        # State after circuit
        psi = U @ psi_0

        # Observable
        if observable == 'global':
            # Projector onto |0...0⟩
            O = np.zeros((2**n_qubits, 2**n_qubits))
            O[0, 0] = 1
        else:
            # Local: Z on first qubit
            Z = np.array([[1, 0], [0, -1]])
            O = np.kron(Z, np.eye(2**(n_qubits-1)))

        # "Gradient" approximated as variation in expectation
        expval = np.real(psi.conj() @ O @ psi)
        gradients.append(expval)

    return np.var(gradients)

# Study variance scaling with system size
n_range = range(2, 9)
variances_global = []
variances_local = []

for n in n_range:
    var_g = compute_gradient_variance(n, 200, 'global')
    var_l = compute_gradient_variance(n, 200, 'local')
    variances_global.append(var_g)
    variances_local.append(var_l)
    print(f"n={n}: Global variance = {var_g:.6f}, Local = {var_l:.6f}")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(list(n_range), variances_global, 'bo-', label='Global observable', linewidth=2)
plt.semilogy(list(n_range), variances_local, 'rs-', label='Local observable', linewidth=2)

# Expected scaling
n_arr = np.array(list(n_range))
plt.semilogy(n_arr, 1/2**n_arr, 'b--', alpha=0.5, label='O(1/2ⁿ) scaling')

plt.xlabel('Number of Qubits n', fontsize=12)
plt.ylabel('Gradient Variance', fontsize=12)
plt.title('Barren Plateau: Gradient Variance vs System Size', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('barren_plateau.png', dpi=150)
plt.show()

print("\nKey insight: Global observables → exponentially vanishing gradients")
print("Mitigation: Use local cost functions, shallow circuits, structured ansatze")
```

---

## Summary

### Key Results

| Setting | Gradient Variance |
|---------|------------------|
| Random deep circuit + global cost | $O(1/2^n)$ |
| Shallow circuit + local cost | $O(1/\text{poly}(n))$ |
| QAOA (structured) | Often avoids BP |

### Mitigation Strategies

1. Local cost functions
2. Shallow circuits
3. Problem-specific ansatze
4. Layer-wise training
5. Careful initialization

---

## Daily Checklist

- [ ] I understand what barren plateaus are
- [ ] I know when they occur
- [ ] I understand the impact on training
- [ ] I know mitigation strategies
- [ ] I ran the computational lab

---

*Next: Day 644 — Month Review*
