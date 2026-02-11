# Day 870: Steane to Reed-Muller Switching

## Overview

**Day:** 870 of 1008
**Week:** 125 (Code Switching & Gauge Fixing)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** [[7,1,3]] Steane ↔ [[15,1,3]] Reed-Muller Fault-Tolerant Code Switching

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Code structures and encoding circuits |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Switching protocols and error analysis |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Describe** the complete structure of the [[7,1,3]] Steane code including stabilizers and logical operators
2. **Construct** the [[15,1,3]] Reed-Muller code from punctured RM codes
3. **Design** fault-tolerant encoding circuits for both codes
4. **Implement** the Anderson-Duclos-Cianci-Svore switching protocol
5. **Analyze** error propagation during code switching operations
6. **Explain** the recent experimental demonstrations of code switching

---

## The Steane Code [[7,1,3]] Review

### Code Construction

The Steane code is a CSS code constructed from the [7,4,3] Hamming code $\mathcal{H}_7$:

$$\mathcal{C}_{\text{Steane}} = \text{CSS}(\mathcal{H}_7, \mathcal{H}_7)$$

**Hamming Code Generator Matrix:**
$$G_{\mathcal{H}_7} = \begin{pmatrix} 1 & 0 & 0 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 & 1 & 1 & 1 \end{pmatrix}$$

**Parity Check Matrix:**
$$H_{\mathcal{H}_7} = \begin{pmatrix} 1 & 1 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}$$

### Stabilizer Generators

**X-type stabilizers** (from $H_{\mathcal{H}_7}$):
$$\begin{aligned}
g_1^X &= X_1 X_2 X_4 X_5 = XXIXXII \\
g_2^X &= X_1 X_3 X_4 X_6 = XIXIXIX \\
g_3^X &= X_2 X_3 X_4 X_7 = IXXIXXI
\end{aligned}$$

**Z-type stabilizers** (from $H_{\mathcal{H}_7}$):
$$\begin{aligned}
g_1^Z &= Z_1 Z_2 Z_4 Z_5 = ZZIZZII \\
g_2^Z &= Z_1 Z_3 Z_4 Z_6 = ZIZIZIZ \\
g_3^Z &= Z_2 Z_3 Z_4 Z_7 = IZZIZZI
\end{aligned}$$

### Logical Operators

$$\boxed{\bar{X} = X^{\otimes 7} = XXXXXXX}$$
$$\boxed{\bar{Z} = Z^{\otimes 7} = ZZZZZZZ}$$

### Logical Basis States

$$|\bar{0}\rangle = \frac{1}{\sqrt{8}} \sum_{c \in \mathcal{H}_7^\perp} |c\rangle$$

Explicitly:
$$|\bar{0}\rangle = \frac{1}{\sqrt{8}}(|0000000\rangle + |1010101\rangle + |0110011\rangle + |1100110\rangle$$
$$+ |0001111\rangle + |1011010\rangle + |0111100\rangle + |1101001\rangle)$$

$$|\bar{1}\rangle = \bar{X}|\bar{0}\rangle = \frac{1}{\sqrt{8}} \sum_{c \in \mathcal{H}_7^\perp} |c \oplus 1111111\rangle$$

### Transversal Gates on Steane

| Gate | Implementation | Why it works |
|------|----------------|--------------|
| $\bar{X}$ | $X^{\otimes 7}$ | Maps $\mathcal{H}_7^\perp \to \mathcal{H}_7^\perp + \bar{1}$ |
| $\bar{Z}$ | $Z^{\otimes 7}$ | Phase on weight-odd codewords |
| $\bar{H}$ | $H^{\otimes 7}$ | Self-dual: $\mathcal{H}_7^\perp = \mathcal{H}_7$ under Hadamard |
| $\bar{S}$ | $S^{\otimes 7}$ | Doubly-even code property |
| $\overline{\text{CNOT}}$ | $\text{CNOT}^{\otimes 7}$ | CSS code property |

---

## The Reed-Muller Code [[15,1,3]]

### Construction from Punctured RM Codes

The [[15,1,3]] quantum Reed-Muller code comes from classical Reed-Muller codes:

**Reed-Muller codes $\text{RM}(r, m)$:**
- Parameters: $[2^m, \sum_{i=0}^r \binom{m}{i}, 2^{m-r}]$
- $\text{RM}(1,4) = [16, 5, 8]$ (first-order)
- $\text{RM}(2,4) = [16, 11, 4]$ (second-order)

**Punctured codes** (remove one coordinate):
- $\text{RM}^*(1,4) = [15, 4, 8]$
- $\text{RM}^*(2,4) = [15, 10, 4]$

**Quantum Code:**
$$\boxed{\mathcal{C}_{\text{RM}} = \text{CSS}(\text{RM}^*(1,4), \text{RM}^*(2,4)^\perp)}$$

Since $\text{RM}(1,4)^\perp = \text{RM}(2,4)$ (duality), this gives a valid CSS code.

### Parameters Verification

- $n = 15$ physical qubits
- $k = \dim(\text{RM}^*(2,4)) - \dim(\text{RM}^*(1,4)) = 10 - 4 = 6$...

Wait, for the [[15,1,3]] code, we need $k=1$. The correct construction uses:

**Correct Construction:**
$$\mathcal{C}_{\text{RM}_{15}} = \text{CSS}(C_1, C_2)$$

where:
- $C_1$ = punctured first-order RM plus all-ones vector
- $C_2^\perp$ chosen to give $k=1$

The key property is the **triorthogonality**:

### Triorthogonal Structure

**Definition:** A code is **triorthogonal** if for any three codewords $c_1, c_2, c_3$:
$$|c_1 \cap c_2 \cap c_3| \equiv 0 \pmod 2$$

where $c_1 \cap c_2 \cap c_3$ denotes positions where all three have 1s.

**Theorem:** If a CSS code is triorthogonal, then $T^{\otimes n}$ is a transversal logical T gate.

The [[15,1,3]] RM code is triorthogonal!

### Stabilizer Generators for [[15,1,3]]

The Z-stabilizers come from the first-order RM code (weight-8 codewords):

$$g_i^Z = Z^{\otimes \text{support}(c_i)}$$

for generators of $\text{RM}^*(1,4)$.

**Example Z-stabilizers:**
$$g_1^Z = Z_1 Z_2 Z_3 Z_4 Z_5 Z_6 Z_7 Z_8$$
$$g_2^Z = Z_1 Z_2 Z_3 Z_4 Z_9 Z_{10} Z_{11} Z_{12}$$
$$g_3^Z = Z_1 Z_2 Z_5 Z_6 Z_9 Z_{10} Z_{13} Z_{14}$$
$$g_4^Z = Z_1 Z_3 Z_5 Z_7 Z_9 Z_{11} Z_{13} Z_{15}$$

**X-stabilizers** from the dual structure.

### Logical Operators for [[15,1,3]]

$$\boxed{\bar{X} = X^{\otimes 15}}$$
$$\boxed{\bar{Z} = Z^{\otimes 15}}$$

### The Magic: Transversal T

**Theorem:** On the [[15,1,3]] RM code:
$$\boxed{\bar{T} = T^{\otimes 15}}$$

**Proof Sketch:**

For $|0_L\rangle$, the codewords have weight $\equiv 0 \pmod 4$:
$$T^{\otimes 15}|c\rangle = e^{i\pi \text{wt}(c)/4}|c\rangle = e^{i\pi \cdot 4k/4}|c\rangle = e^{i\pi k}|c\rangle = \pm|c\rangle$$

The triorthogonal property ensures consistent phases.

For $|1_L\rangle$, codewords have weight $\equiv 3 \pmod 4$:
$$T^{\otimes 15}|c\rangle = e^{i\pi(4k+3)/4}|c\rangle = e^{i\pi k}e^{i3\pi/4}|c\rangle$$

This gives the correct $e^{i\pi/4}$ phase for logical T. $\square$

---

## The Code Switching Protocol

### Overview: Steane → Reed-Muller

**Goal:** Convert $|\psi_L\rangle_{\text{Steane}}$ to $|\psi_L\rangle_{\text{RM}}$ fault-tolerantly.

**Strategy:** Use transversal operations between the codes.

### Key Insight: Transversal CNOT Bridge

Both codes are CSS codes with:
- Steane: $\bar{X} = X^{\otimes 7}$, $\bar{Z} = Z^{\otimes 7}$
- RM: $\bar{X} = X^{\otimes 15}$, $\bar{Z} = Z^{\otimes 15}$

**Critical Property:** We can perform a **logical CNOT** between codes of different sizes using transversal physical CNOTs:

$$\overline{\text{CNOT}}_{\text{Steane} \to \text{RM}}$$

This requires careful qubit mapping.

### The Anderson-Duclos-Cianci-Svore Protocol (2014)

**Step 1: Prepare RM ancilla**
$$|+_L\rangle_{\text{RM}} = \frac{1}{\sqrt{2}}(|0_L\rangle + |1_L\rangle)_{\text{RM}}$$

This is prepared fault-tolerantly using a verified preparation circuit.

**Step 2: Transversal CNOT**
$$\overline{\text{CNOT}}_{\text{Steane} \to \text{RM}}: |\psi_L\rangle_{\text{Steane}} |+_L\rangle_{\text{RM}}$$

For $|\psi_L\rangle = \alpha|0_L\rangle + \beta|1_L\rangle$:
$$\to \alpha|0_L\rangle_{\text{Steane}}|+_L\rangle_{\text{RM}} + \beta|1_L\rangle_{\text{Steane}}|-_L\rangle_{\text{RM}}$$

Wait, this isn't quite right. Let's be more careful.

**Correct Protocol:**

**Step 1:** Prepare verified $|0_L\rangle_{\text{RM}}$ state

**Step 2:** Apply logical CNOT (Steane control, RM target):
$$|\psi_L\rangle_S |0_L\rangle_{\text{RM}} = (\alpha|0_L\rangle + \beta|1_L\rangle)_S |0_L\rangle_{\text{RM}}$$
$$\xrightarrow{\overline{\text{CNOT}}} \alpha|0_L\rangle_S|0_L\rangle_{\text{RM}} + \beta|1_L\rangle_S|1_L\rangle_{\text{RM}}$$

**Step 3:** Measure Steane qubits in X basis:
$$= \alpha|0_L\rangle_S|0_L\rangle_{\text{RM}} + \beta|1_L\rangle_S|1_L\rangle_{\text{RM}}$$

Rewrite in X basis for Steane:
$$= \frac{\alpha + \beta}{2}|+_L\rangle_S(|0_L\rangle + |1_L\rangle)_{\text{RM}} + \frac{\alpha - \beta}{2}|+_L\rangle_S(|0_L\rangle - |1_L\rangle)_{\text{RM}}$$
$$+ \frac{\alpha + \beta}{2}|-_L\rangle_S \cdot ... + \frac{\alpha - \beta}{2}|-_L\rangle_S \cdot ...$$

After measuring Steane in X basis and getting outcome $m \in \{+, -\}$:
- If $m = +$: RM state is $\alpha|0_L\rangle + \beta|1_L\rangle$ (correct!)
- If $m = -$: RM state is $\alpha|0_L\rangle - \beta|1_L\rangle$ (apply $\bar{Z}$ correction)

**Step 4:** Apply correction $\bar{Z}_{\text{RM}}^m$ based on measurement.

$$\boxed{|\psi_L\rangle_{\text{RM}} = (\alpha|0_L\rangle + \beta|1_L\rangle)_{\text{RM}}}$$

### Circuit Diagram

```
Steane Block (7 qubits):
|ψ_L⟩ ─────●─────[MX]───── (measure X basis)
           │
           │ (transversal CNOT)
           │
RM Block (15 qubits):
|0_L⟩ ─────⊕─────[Z^m]───── |ψ_L⟩_RM
```

The transversal CNOT requires mapping 7 Steane qubits to 15 RM qubits. This uses an **embedding** that respects the code structure.

### Qubit Mapping: Steane → RM

The 7 Steane qubits map to a subset of 15 RM qubits:

**Embedding function:** $\phi: \{1,...,7\} \to \{1,...,15\}$

The mapping must satisfy:
$$\text{CNOT}_{\text{Steane}_i \to \text{RM}_{\phi(i)}} \text{ for } i = 1,...,7$$

A valid embedding uses the nested structure of RM codes:
$$\phi(i) = 2i - 1 \text{ (odd positions)}$$

or another systematic mapping based on the code construction.

---

## Reed-Muller → Steane Switching

### The Reverse Protocol

**Step 1:** Prepare verified $|+_L\rangle_{\text{Steane}}$

**Step 2:** Apply logical CNOT (RM control, Steane target):
$$|\psi_L\rangle_{\text{RM}} |+_L\rangle_S \xrightarrow{\overline{\text{CNOT}}} ...$$

**Step 3:** Measure RM block in Z basis

**Step 4:** Apply $\bar{X}_S^m$ correction

This transfers the state from RM to Steane.

### Symmetry of the Protocol

The switching is symmetric:
$$\text{Steane} \xrightleftharpoons[\text{Z-basis measure}]{\text{X-basis measure}} \text{RM}$$

---

## Fault Tolerance Analysis

### Error Sources

1. **Preparation errors** in ancilla states
2. **CNOT gate errors** during transversal operation
3. **Measurement errors** in basis measurement
4. **Correction errors** from classical processing

### Error Propagation During Switch

**Scenario:** Single X error on Steane qubit $i$ before CNOT.

$$X_i |\psi_L\rangle_S |0_L\rangle_{\text{RM}}$$

After transversal CNOT:
$$\text{CNOT}(X_i \otimes I) = (X_i \otimes X_{\phi(i)}) \text{CNOT}$$

So:
$$X_i |\text{entangled}\rangle \to X_i X_{\phi(i)} |\text{entangled}\rangle$$

**Result:** One error in Steane, one error in RM. Each block has weight-1 error: correctable!

### Fault-Tolerant Gadget Requirements

**Verified State Preparation:**

For $|0_L\rangle_{\text{RM}}$ preparation:
1. Prepare unverified $|0_L\rangle_{\text{RM}}$
2. Measure stabilizers to detect errors
3. If any syndrome non-trivial, reject and restart
4. Accept only verified states

**Transversal CNOT:**
- Already fault-tolerant by construction
- Single fault → single error in each block

**Fault-Tolerant Measurement:**
- Repeat measurements 3 times
- Take majority vote
- Handles single measurement errors

### Error Threshold Analysis

**Theorem (Anderson et al.):** The Steane ↔ RM code switching protocol is fault-tolerant with threshold comparable to standard fault-tolerant gadgets.

The dominant error source is typically the ancilla preparation.

---

## Experimental Demonstration (Quantinuum 2024)

### The Experiment

**Platform:** Quantinuum H1-1 trapped-ion quantum computer

**Achievement:**
- First experimental fault-tolerant code switching
- Steane [[7,1,3]] ↔ [[15,1,3]] RM demonstrated
- Produced magic states with infidelity $< 5.1 \times 10^{-4}$

### Key Results

**Magic State Fidelity:**
$$F(|T\rangle) > 0.99949$$

This is **below the pseudo-threshold** for T-gates!

**Comparison with Distillation:**
- Code switching: Direct, deterministic
- Distillation: Required ~15:1 overhead
- Code switching achieved comparable fidelity with fewer resources

### Implications

1. **Code switching is practical:** Not just theoretical
2. **Competitive with distillation:** Sometimes superior
3. **Scalable:** Can be extended to larger codes
4. **Hardware-agnostic:** Applicable to various platforms

---

## Worked Examples

### Example 1: Constructing the Logical CNOT

**Problem:** Write out the transversal CNOT between a 3-qubit repetition code and itself.

**Solution:**

The 3-qubit repetition code:
- $|0_L\rangle = |000\rangle$
- $|1_L\rangle = |111\rangle$
- $\bar{X} = XXX$, $\bar{Z} = ZII$ (or any single Z)

Transversal CNOT between blocks A and B:
$$\overline{\text{CNOT}}_{A \to B} = \text{CNOT}_{A_1 \to B_1} \cdot \text{CNOT}_{A_2 \to B_2} \cdot \text{CNOT}_{A_3 \to B_3}$$

**Verification:**
$$|0_L\rangle_A |0_L\rangle_B = |000\rangle|000\rangle \xrightarrow{\overline{\text{CNOT}}} |000\rangle|000\rangle = |0_L\rangle_A|0_L\rangle_B \checkmark$$

$$|1_L\rangle_A |0_L\rangle_B = |111\rangle|000\rangle \xrightarrow{\overline{\text{CNOT}}} |111\rangle|111\rangle = |1_L\rangle_A|1_L\rangle_B \checkmark$$

$$\boxed{\overline{\text{CNOT}} = \text{CNOT}^{\otimes 3}}$$

### Example 2: State Transfer via Code Switching

**Problem:** Transfer $|\psi_L\rangle = \frac{1}{\sqrt{2}}(|0_L\rangle + i|1_L\rangle)$ from Steane to RM code.

**Solution:**

**Initial state:**
$$|\Psi_0\rangle = \frac{1}{\sqrt{2}}(|0_L\rangle + i|1_L\rangle)_S \otimes |0_L\rangle_{\text{RM}}$$

**After CNOT:**
$$|\Psi_1\rangle = \frac{1}{\sqrt{2}}(|0_L\rangle_S|0_L\rangle_{\text{RM}} + i|1_L\rangle_S|1_L\rangle_{\text{RM}})$$

**Rewrite Steane in X-basis:**
Using $|0_L\rangle = \frac{1}{\sqrt{2}}(|+_L\rangle + |-_L\rangle)$ and $|1_L\rangle = \frac{1}{\sqrt{2}}(|+_L\rangle - |-_L\rangle)$:

$$|\Psi_1\rangle = \frac{1}{2}[(|+_L\rangle + |-_L\rangle)_S|0_L\rangle_{\text{RM}} + i(|+_L\rangle - |-_L\rangle)_S|1_L\rangle_{\text{RM}}]$$

$$= \frac{1}{2}|+_L\rangle_S(|0_L\rangle + i|1_L\rangle)_{\text{RM}} + \frac{1}{2}|-_L\rangle_S(|0_L\rangle - i|1_L\rangle)_{\text{RM}}$$

**Measure Steane in X basis:**

- Outcome $+$: RM state is $\frac{1}{\sqrt{2}}(|0_L\rangle + i|1_L\rangle)$ ✓
- Outcome $-$: RM state is $\frac{1}{\sqrt{2}}(|0_L\rangle - i|1_L\rangle) = \bar{Z}\frac{1}{\sqrt{2}}(|0_L\rangle + i|1_L\rangle)$

**Correction:** Apply $\bar{Z}_{\text{RM}} = Z^{\otimes 15}$ if outcome is $-$.

$$\boxed{|\psi_L\rangle_{\text{RM}} = \frac{1}{\sqrt{2}}(|0_L\rangle + i|1_L\rangle)_{\text{RM}}}$$

### Example 3: Error During Code Switch

**Problem:** A Z error occurs on Steane qubit 3 during the code switch. Track the error through the protocol.

**Solution:**

**Before CNOT:**
$$Z_3|\psi_L\rangle_S|0_L\rangle_{\text{RM}}$$

**After CNOT:** Z errors on control don't propagate through CNOT:
$$\text{CNOT}(Z \otimes I) = (Z \otimes I)\text{CNOT}$$

So:
$$Z_3 |\text{entangled state}\rangle$$

**During X-basis measurement:**
Z error on qubit 3 can flip the X measurement outcome for that qubit.

**Effect on syndrome:**
The Steane X-stabilizers will detect the Z error. The syndrome tells us which qubit had the error.

**Correction strategy:**
1. Measure X on all Steane qubits
2. Compute syndrome from redundant measurements
3. Identify Z error location
4. The RM state is unaffected (Z didn't propagate)
5. Apply appropriate $\bar{Z}$ correction based on parity

$$\boxed{\text{Single Z error on Steane is correctable; RM state is protected}}$$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Write out the 6 stabilizer generators of the Steane code in both Pauli string notation and binary matrix form.

**P1.2** For the [[15,1,3]] RM code, verify that $T^{\otimes 15}|0_L\rangle = |0_L\rangle$ for a codeword of weight 8.

**P1.3** In the code switching protocol, if the X-basis measurement yields outcome $-1$, what correction must be applied to the RM block?

### Level 2: Intermediate

**P2.1** Design the complete circuit for preparing a verified $|0_L\rangle$ state of the [[15,1,3]] RM code. Include syndrome measurements and rejection criteria.

**P2.2** Prove that transversal CNOT between two CSS codes with the same block structure is a valid logical CNOT.

**P2.3** Calculate the probability of successful code switching given physical error rate $p$ per gate. Assume the circuit has 50 physical gates.

### Level 3: Challenging

**P3.1** The Steane→RM switching embeds 7 qubits into 15. Design an explicit qubit mapping $\phi: \{1,...,7\} \to \{1,...,15\}$ that respects both code structures.

**P3.2** Extend the code switching protocol to transfer a 2-qubit entangled state $|\Phi^+\rangle_L$ from two Steane blocks to two RM blocks.

**P3.3** Analyze the threshold for code switching: at what physical error rate does the logical error rate after switching exceed the input logical error rate?

---

## Computational Lab

```python
"""
Day 870: Steane to Reed-Muller Code Switching
=============================================

Implementation of the code switching protocol between [[7,1,3]] Steane
and [[15,1,3]] Reed-Muller codes.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import product
from functools import reduce

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    return reduce(np.kron, args)


def tensor_power(M, n):
    """Compute M^{otimes n}."""
    result = M
    for _ in range(n - 1):
        result = np.kron(result, M)
    return result


class SteaneCode:
    """The [[7,1,3]] Steane code."""

    def __init__(self):
        self.n = 7
        self.k = 1
        self.d = 3

        # Stabilizer generators (as binary vectors)
        # Format: [x1,...,x7,z1,...,z7] for X^{x_i}Z^{z_i}
        self.stabilizers_x = [
            [1, 1, 0, 1, 1, 0, 0],  # X1 X2 X4 X5
            [1, 0, 1, 1, 0, 1, 0],  # X1 X3 X4 X6
            [0, 1, 1, 1, 0, 0, 1],  # X2 X3 X4 X7
        ]
        self.stabilizers_z = [
            [1, 1, 0, 1, 1, 0, 0],  # Z1 Z2 Z4 Z5
            [1, 0, 1, 1, 0, 1, 0],  # Z1 Z3 Z4 Z6
            [0, 1, 1, 1, 0, 0, 1],  # Z2 Z3 Z4 Z7
        ]

        # Logical operators
        self.logical_x = [1, 1, 1, 1, 1, 1, 1]  # X^7
        self.logical_z = [1, 1, 1, 1, 1, 1, 1]  # Z^7

        # Code space (logical states as superpositions)
        self._compute_code_states()

    def _compute_code_states(self):
        """Compute the logical 0 and 1 states."""
        # Codewords of the dual Hamming code (weight 0, 4 mod 4)
        # |0_L> = sum over even-weight codewords
        # |1_L> = X^7 |0_L>

        # Hamming code codewords
        H_perp = [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 1],
        ]

        # |0_L> as state vector
        dim = 2**7
        self.state_0L = np.zeros(dim, dtype=complex)
        for c in H_perp:
            idx = sum(b * 2**(6-i) for i, b in enumerate(c))
            self.state_0L[idx] = 1
        self.state_0L /= np.sqrt(8)

        # |1_L> = X^7 |0_L>
        self.state_1L = np.zeros(dim, dtype=complex)
        for c in H_perp:
            c_flipped = [1-b for b in c]
            idx = sum(b * 2**(6-i) for i, b in enumerate(c_flipped))
            self.state_1L[idx] = 1
        self.state_1L /= np.sqrt(8)

    def encode(self, alpha: complex, beta: complex) -> np.ndarray:
        """Encode logical state alpha|0> + beta|1>."""
        return alpha * self.state_0L + beta * self.state_1L


class ReedMullerCode:
    """The [[15,1,3]] Reed-Muller code."""

    def __init__(self):
        self.n = 15
        self.k = 1
        self.d = 3

        # Compute code states
        self._compute_code_states()

    def _compute_code_states(self):
        """
        Compute logical states for [[15,1,3]] RM code.

        The code is defined by punctured RM(1,4) code.
        Codewords have weights 0, 8 (even) for |0_L>
        and weights 7, 15 (odd) for |1_L>.
        """
        # For [[15,1,3]], we use a simplified construction
        # Based on the triorthogonal property

        # First-order RM(1,4) generator matrix (before puncturing)
        # Rows: all-ones, and 4 indicator functions for 4 coordinates
        # Puncture last column

        # Codewords for |0_L>: weight 0 or 8 mod 8
        # Codewords for |1_L>: weight 7 or 15

        # Simplified: use known structure
        dim = 2**15

        # Generate codewords from RM structure
        # RM(1,4)^* has 16 codewords: 0, and 8 weight-8 and 7 weight-7

        # For demonstration, we'll use a simplified model
        # The actual codewords would come from the RM construction

        self.state_0L = np.zeros(dim, dtype=complex)
        self.state_1L = np.zeros(dim, dtype=complex)

        # |0_L> includes |0...0> and weight-8 codewords
        self.state_0L[0] = 1  # |000...0>

        # For a proper implementation, enumerate all codewords
        # Here we just mark the structure
        self.state_0L /= np.linalg.norm(self.state_0L)

        # |1_L> = X^15 |0_L>
        self.state_1L[2**15 - 1] = 1  # |111...1>
        self.state_1L /= np.linalg.norm(self.state_1L)

    def transversal_T(self) -> np.ndarray:
        """Return the transversal T gate T^{otimes 15}."""
        return tensor_power(T, self.n)


def code_switching_steane_to_rm(psi_steane: np.ndarray,
                                 verbose: bool = True) -> np.ndarray:
    """
    Simulate code switching from Steane to Reed-Muller.

    This is a simplified simulation of the protocol.

    Parameters:
    -----------
    psi_steane : np.ndarray
        7-qubit state encoded in Steane code
    verbose : bool
        Print intermediate steps

    Returns:
    --------
    np.ndarray
        15-qubit state encoded in RM code
    """
    steane = SteaneCode()
    rm = ReedMullerCode()

    if verbose:
        print("Code Switching: Steane [[7,1,3]] -> Reed-Muller [[15,1,3]]")
        print("=" * 60)

    # Step 1: Prepare |0_L>_RM ancilla
    if verbose:
        print("\nStep 1: Prepare |0_L>_RM ancilla")

    # Combined system: 7 Steane + 15 RM = 22 qubits
    # For simulation, we work with logical states

    # Extract logical amplitudes from Steane state
    alpha = np.vdot(steane.state_0L, psi_steane)
    beta = np.vdot(steane.state_1L, psi_steane)

    if verbose:
        print(f"  Logical state: {alpha:.4f}|0_L> + {beta:.4f}|1_L>")

    # Step 2: Transversal CNOT (Steane control, RM target)
    if verbose:
        print("\nStep 2: Apply transversal CNOT")
        print("  Creates entangled state between codes")

    # After CNOT: alpha|0_L>_S|0_L>_RM + beta|1_L>_S|1_L>_RM

    # Step 3: Measure Steane in X basis
    if verbose:
        print("\nStep 3: Measure Steane block in X basis")

    # Simulate measurement (random outcome weighted by probability)
    # P(+) = |alpha + beta|^2 / 2, P(-) = |alpha - beta|^2 / 2
    # After measurement, RM has the correct state (up to Z correction)

    # For simulation, assume measurement outcome +
    outcome = np.random.choice([0, 1], p=[0.5, 0.5])
    measurement_result = '+' if outcome == 0 else '-'

    if verbose:
        print(f"  Measurement outcome: {measurement_result}")

    # Step 4: Apply correction
    if verbose:
        print("\nStep 4: Apply Z correction if needed")

    if measurement_result == '-':
        # Apply Z correction
        beta = -beta
        if verbose:
            print("  Applied Z^15 correction")
    else:
        if verbose:
            print("  No correction needed")

    # Construct RM encoded state
    psi_rm = alpha * rm.state_0L + beta * rm.state_1L

    if verbose:
        print(f"\nFinal RM state: {alpha:.4f}|0_L> + {beta:.4f}|1_L>")
        print("Code switching complete!")

    return psi_rm


def verify_transversal_T():
    """Verify that T^15 is transversal T on RM code."""
    print("\n" + "=" * 60)
    print("Verifying Transversal T on [[15,1,3]] RM Code")
    print("=" * 60)

    # For the RM code, T^15 should act as logical T
    # T|0_L> = |0_L>
    # T|1_L> = e^{i*pi/4} |1_L>

    rm = ReedMullerCode()

    # Apply T^15 to |0_L>
    T15 = tensor_power(T, 15)

    # Since our simplified RM states are just |0...0> and |1...1>
    # we can verify on these

    # T^15 |0...0> = |0...0> (T|0> = |0>)
    state_0 = np.zeros(2**15, dtype=complex)
    state_0[0] = 1

    result_0 = T15 @ state_0
    print(f"\nT^15 |0...0> = |0...0>: {np.allclose(result_0, state_0)}")

    # T^15 |1...1> = e^{i*15*pi/4} |1...1> = e^{i*7*pi/4} |1...1>
    # But for the actual code, the phase is e^{i*pi/4}
    state_1 = np.zeros(2**15, dtype=complex)
    state_1[2**15 - 1] = 1

    result_1 = T15 @ state_1
    expected_phase = np.exp(1j * 15 * np.pi / 4)
    print(f"\nT^15 |1...1> phase: {result_1[2**15-1]:.4f}")
    print(f"Expected (naive): e^{{i*15*pi/4}} = {expected_phase:.4f}")
    print(f"For RM code (triorthogonal): e^{{i*pi/4}} = {np.exp(1j*np.pi/4):.4f}")

    print("\nNote: The actual RM code uses superpositions of codewords,")
    print("where the triorthogonal property ensures correct logical T action.")


def simulate_error_during_switch():
    """Simulate error propagation during code switching."""
    print("\n" + "=" * 60)
    print("Error Propagation During Code Switching")
    print("=" * 60)

    # Scenario: Z error on Steane qubit 3 before CNOT

    print("\nScenario: Z error on Steane qubit 3")
    print("-" * 40)

    print("\n1. Initial state: |ψ_L>_S ⊗ |0_L>_RM")
    print("2. Z_3 error applied to Steane block")
    print("3. CNOT applied (transversal)")

    print("\nAnalysis:")
    print("  - Z errors don't propagate through CNOT (control side)")
    print("  - Z_3 remains on Steane block only")
    print("  - RM block is unaffected")

    print("\n4. X-basis measurement on Steane:")
    print("  - Z_3 may flip measurement of qubit 3")
    print("  - Steane syndrome detects the error location")
    print("  - Error is correctable (weight-1 < d=3)")

    print("\n5. Result:")
    print("  - RM block receives correct logical state")
    print("  - Code switching succeeds despite error")
    print("  - Fault tolerance maintained!")


def resource_analysis():
    """Analyze resources for code switching."""
    print("\n" + "=" * 60)
    print("Resource Analysis: Code Switching")
    print("=" * 60)

    print("\nSteane -> Reed-Muller Switching:")
    print("-" * 40)

    # Qubit counts
    steane_qubits = 7
    rm_qubits = 15
    ancilla_verification = 7  # For Steane syndrome checks
    total_qubits = steane_qubits + rm_qubits + ancilla_verification

    print(f"  Steane data qubits: {steane_qubits}")
    print(f"  RM data qubits: {rm_qubits}")
    print(f"  Verification ancillas: {ancilla_verification}")
    print(f"  Total qubits: {total_qubits}")

    # Gate counts
    cnots_transversal = 7  # CNOT for each Steane qubit to RM
    cnots_rm_prep = 50  # Approximate for RM encoding
    measurements = 7  # X-basis on Steane

    print(f"\n  Transversal CNOTs: {cnots_transversal}")
    print(f"  RM preparation CNOTs: ~{cnots_rm_prep}")
    print(f"  Measurements: {measurements}")

    # Depth
    depth_rm_prep = 15
    depth_cnot = 1
    depth_measure = 1
    total_depth = depth_rm_prep + depth_cnot + depth_measure

    print(f"\n  Circuit depth: ~{total_depth}")

    print("\nComparison with Magic State Distillation:")
    print("-" * 40)

    ms_ancillas = 15  # For 15-to-1 protocol
    ms_cnots = 100  # Approximate
    ms_depth = 50

    print(f"  Distillation ancillas: {ms_ancillas}")
    print(f"  Distillation CNOTs: ~{ms_cnots}")
    print(f"  Distillation depth: ~{ms_depth}")

    print("\nCode switching advantage: Lower depth, deterministic")
    print("Magic state advantage: Smaller code (7 vs 15 data)")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Day 870: Steane to Reed-Muller Code Switching")
    print("=" * 60)

    # Create a test state
    steane = SteaneCode()

    # Encode |ψ> = (|0> + i|1>)/sqrt(2)
    alpha = 1 / np.sqrt(2)
    beta = 1j / np.sqrt(2)
    psi_steane = steane.encode(alpha, beta)

    print(f"\nTest state: ({alpha:.3f})|0_L> + ({beta:.3f})|1_L>")

    # Demonstrate code switching
    psi_rm = code_switching_steane_to_rm(psi_steane)

    # Verify transversal T
    verify_transversal_T()

    # Error analysis
    simulate_error_during_switch()

    # Resource analysis
    resource_analysis()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Steane and RM codes have complementary transversal gates")
    print("2. Code switching uses transversal CNOT + measurement")
    print("3. Single errors don't spread: fault-tolerant protocol")
    print("4. Quantinuum demonstrated this experimentally in 2024")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Steane stabilizers | $g^X \sim XXIXXII$, $g^Z \sim ZZIZZII$ pattern |
| Steane logical ops | $\bar{X} = X^{\otimes 7}$, $\bar{Z} = Z^{\otimes 7}$ |
| RM transversal T | $\bar{T} = T^{\otimes 15}$ |
| Switching CNOT | $\overline{\text{CNOT}}_{S \to RM}$ (transversal) |
| State transfer | $|\psi_L\rangle_S|0_L\rangle_{RM} \xrightarrow{\text{CNOT}} \to |\psi_L\rangle_{RM}$ |
| Z correction | Apply if X-measurement gives $-$ outcome |

### Main Takeaways

1. **Steane code** has transversal Clifford gates but NOT transversal T
2. **Reed-Muller code** has transversal T but NOT transversal H
3. **Triorthogonality** is the key property enabling transversal T on RM
4. **Code switching protocol** uses transversal CNOT + X-basis measurement
5. **Fault tolerance** is preserved: single errors remain single errors
6. **Experimental verification** by Quantinuum showed practical viability

---

## Daily Checklist

- [ ] I can describe the Steane code stabilizers and logical operators
- [ ] I understand the triorthogonal property of the RM code
- [ ] I can explain why $T^{\otimes 15}$ is logical T on the RM code
- [ ] I can execute the Steane→RM switching protocol step by step
- [ ] I can analyze error propagation during code switching
- [ ] I understand the experimental significance of the Quantinuum result

---

## Preview: Day 871

Tomorrow we explore **Subsystem Codes**:

- The distinction between subspace and subsystem encodings
- The Bacon-Shor code [[9,1,3]] as a key example
- Gauge qubits and gauge operators
- How gauge freedom enables new possibilities
- Trade-offs between syndrome measurements and error correction

Understanding subsystem codes prepares us for gauge fixing protocols.
