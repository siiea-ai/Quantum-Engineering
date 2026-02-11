# Day 654: General Pauli Errors

## Week 94: Quantum Error Types | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Define** the general Pauli channel with arbitrary error probabilities
2. **Understand** Pauli twirling and its applications
3. **Analyze** the effect of combined X, Y, Z errors
4. **Apply** the Pauli error model to quantum circuits

---

## Core Content

### 1. The Pauli Channel

The **Pauli channel** applies Pauli operators with specified probabilities:

$$\boxed{\mathcal{E}_{\text{Pauli}}(\rho) = p_I \rho + p_X X\rho X + p_Y Y\rho Y + p_Z Z\rho Z}$$

where $p_I + p_X + p_Y + p_Z = 1$.

**Kraus operators:** $K_0 = \sqrt{p_I}I$, $K_1 = \sqrt{p_X}X$, $K_2 = \sqrt{p_Y}Y$, $K_3 = \sqrt{p_Z}Z$

### 2. Special Cases

| Channel | Probabilities |
|---------|--------------|
| Identity | $p_I = 1$ |
| Bit-flip | $p_I = 1-p$, $p_X = p$ |
| Phase-flip | $p_I = 1-p$, $p_Z = p$ |
| Bit-phase-flip | $p_I = 1-p$, $p_Y = p$ |
| Depolarizing | $p_I = 1-p$, $p_X = p_Y = p_Z = p/3$ |

### 3. Effect on Bloch Vector

The Pauli channel transforms the Bloch vector as:

$$\vec{r} \mapsto \begin{pmatrix}\lambda_x & 0 & 0\\0 & \lambda_y & 0\\0 & 0 & \lambda_z\end{pmatrix}\vec{r}$$

where:
- $\lambda_x = p_I + p_X - p_Y - p_Z$
- $\lambda_y = p_I - p_X + p_Y - p_Z$
- $\lambda_z = p_I - p_X - p_Y + p_Z$

### 4. Pauli Twirling

**Pauli twirling** converts any channel into a Pauli channel:

$$\mathcal{E}_{\text{twirled}}(\rho) = \frac{1}{4}\sum_{P \in \{I,X,Y,Z\}} P^\dagger \mathcal{E}(P\rho P^\dagger) P$$

**Why twirl?**
- Simplifies error analysis
- Pauli channels are easier to correct
- Preserves average fidelity

### 5. The Y Error

The $Y$ error ($Y = iXZ$) combines bit-flip and phase-flip:

$$Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$$

**Effect:** Flips both bit AND phase (up to global phase).

### 6. Error Probability Distribution

For a general Pauli channel, the **error probability** is:
$$p_{\text{error}} = 1 - p_I = p_X + p_Y + p_Z$$

The **error distribution** specifies which type of error occurred.

---

## Worked Example

**Problem:** A channel has $p_I = 0.9$, $p_X = 0.04$, $p_Y = 0.02$, $p_Z = 0.04$. Find the Bloch sphere contraction factors.

**Solution:**
- $\lambda_x = 0.9 + 0.04 - 0.02 - 0.04 = 0.88$
- $\lambda_y = 0.9 - 0.04 + 0.02 - 0.04 = 0.84$
- $\lambda_z = 0.9 - 0.04 - 0.02 + 0.04 = 0.88$

---

## Practice Problems

1. Show that any Pauli channel is self-adjoint: $\mathcal{E} = \mathcal{E}^\dagger$.
2. Find the fixed points of a general Pauli channel.
3. Prove that Pauli twirling preserves the trace distance to the identity channel.
4. Design a Pauli channel that contracts only the $z$-component.

---

## Computational Lab

```python
"""Day 654: General Pauli Errors"""

import numpy as np
import matplotlib.pyplot as plt

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def pauli_channel(rho, p_I, p_X, p_Y, p_Z):
    """Apply general Pauli channel."""
    return (p_I * rho + p_X * X @ rho @ X +
            p_Y * Y @ rho @ Y + p_Z * Z @ rho @ Z)

def pauli_twirl(rho, channel_func):
    """Apply Pauli twirling to any channel."""
    paulis = [I, X, Y, Z]
    result = np.zeros_like(rho)
    for P in paulis:
        result += P @ channel_func(P @ rho @ P) @ P
    return result / 4

# Test Pauli channel
rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
rho_out = pauli_channel(rho_plus, 0.9, 0.04, 0.02, 0.04)
print(f"Pauli channel output:\n{rho_out}")

# Verify Bloch contraction
def bloch_vector(rho):
    return [2*np.real(rho[0,1]), 2*np.imag(rho[1,0]), np.real(rho[0,0]-rho[1,1])]

r_in = bloch_vector(rho_plus)
r_out = bloch_vector(rho_out)
print(f"\nBloch vector: {r_in} -> {r_out}")
print(f"Contraction: x={r_out[0]/r_in[0]:.2f}, y={r_out[1]/r_in[1] if r_in[1]!=0 else 'N/A'}, z={r_out[2]/r_in[2] if r_in[2]!=0 else 'N/A'}")
```

---

## Summary

- **Pauli channel** generalizes bit-flip, phase-flip, and Y errors
- **Bloch transformation** is diagonal with eigenvalues $\lambda_x, \lambda_y, \lambda_z$
- **Pauli twirling** converts any channel to a Pauli channel
- Y error = combined bit-flip AND phase-flip

---

## Preview: Day 655

Tomorrow: **Depolarizing Channel Analysis** - the most symmetric error model.
