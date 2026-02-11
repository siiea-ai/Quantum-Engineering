# Day 732: GF(4) Representation

## Overview

**Day:** 732 of 1008
**Week:** 105 (Binary Representation & F₂ Linear Algebra)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Quantum Codes over GF(4)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | GF(4) field theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Quantum codes over GF(4) |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Construct** the finite field GF(4) and its operations
2. **Map** single-qubit Paulis to GF(4) elements
3. **Use** the trace function to define inner products
4. **Convert** between binary and GF(4) representations
5. **Understand** additive codes over GF(4)
6. **Connect** classical coding theory to stabilizer codes

---

## Core Content

### The Finite Field GF(4)

**Construction:**
GF(4) is the unique field with 4 elements, constructed as:
$$\text{GF}(4) = \mathbb{F}_2[x] / (x^2 + x + 1)$$

**Elements:**
$$\text{GF}(4) = \{0, 1, \omega, \bar{\omega}\}$$

where $\omega$ is a root of $x^2 + x + 1 = 0$, so:
$$\omega^2 = \omega + 1 = \bar{\omega}$$

**Alternative notation:** $\omega = \alpha$, $\bar{\omega} = \alpha^2 = \alpha + 1$

### GF(4) Arithmetic

**Addition Table:**
| + | 0 | 1 | ω | ω̄ |
|---|---|---|---|---|
| 0 | 0 | 1 | ω | ω̄ |
| 1 | 1 | 0 | ω̄ | ω |
| ω | ω | ω̄ | 0 | 1 |
| ω̄ | ω̄ | ω | 1 | 0 |

**Multiplication Table:**
| × | 0 | 1 | ω | ω̄ |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | ω | ω̄ |
| ω | 0 | ω | ω̄ | 1 |
| ω̄ | 0 | ω̄ | 1 | ω |

**Key Properties:**
- Characteristic 2: $a + a = 0$ for all $a$
- $\omega^3 = 1$ (ω is a primitive cube root of unity)
- $\bar{\omega} = \omega^2 = \omega^{-1}$
- Conjugation: $\bar{a}$ swaps ω ↔ ω̄, fixes 0, 1

### Pauli-to-GF(4) Mapping

**The fundamental correspondence:**

$$\boxed{I \leftrightarrow 0, \quad X \leftrightarrow 1, \quad Z \leftrightarrow \omega, \quad Y \leftrightarrow \bar{\omega}}$$

This is a group isomorphism (up to phases):
$$\mathcal{P}_1 / \{\pm 1, \pm i\} \cong (\text{GF}(4), +)$$

**Verification:**
- $X \cdot X = I \leftrightarrow 1 + 1 = 0$ ✓
- $Z \cdot Z = I \leftrightarrow \omega + \omega = 0$ ✓
- $X \cdot Z = iY \leftrightarrow 1 + \omega = \bar{\omega}$ ✓ (ignoring phase)
- $Y \cdot Y = I \leftrightarrow \bar{\omega} + \bar{\omega} = 0$ ✓

### Multi-Qubit Extension

An n-qubit Pauli (ignoring phase) corresponds to a vector in GF(4)^n:
$$P_1 \otimes P_2 \otimes \cdots \otimes P_n \leftrightarrow (g_1, g_2, \ldots, g_n) \in \text{GF}(4)^n$$

**Example:**
$$X \otimes Z \otimes Y \otimes I \leftrightarrow (1, \omega, \bar{\omega}, 0)$$

### Binary to GF(4) Conversion

**Formula:**
For binary vector $(a|b) \in \mathbb{F}_2^{2n}$:
$$g_i = a_i + \omega b_i \in \text{GF}(4)$$

**Conversion table:**
| $a_i$ | $b_i$ | $g_i = a_i + \omega b_i$ | Pauli |
|-------|-------|--------------------------|-------|
| 0 | 0 | 0 | I |
| 1 | 0 | 1 | X |
| 0 | 1 | ω | Z |
| 1 | 1 | 1 + ω = ω̄ | Y |

**Inverse:**
$$a_i = g_i + \bar{g}_i, \quad b_i = \omega g_i + \bar{\omega}\bar{g}_i$$

(where $\bar{g}$ denotes conjugation in GF(4))

### The Trace Function

**Definition:**
The trace from GF(4) to GF(2):
$$\text{tr}(x) = x + \bar{x}$$

**Values:**
- tr(0) = 0
- tr(1) = 0
- tr(ω) = ω + ω̄ = 1
- tr(ω̄) = ω̄ + ω = 1

**Properties:**
- tr is F₂-linear: tr(x + y) = tr(x) + tr(y)
- tr(αx) = α·tr(x) for α ∈ F₂

### Hermitian Inner Product

**Definition:**
For vectors $u, v \in \text{GF}(4)^n$:
$$\langle u, v \rangle_H = \sum_{i=1}^n u_i \bar{v}_i$$

**Trace Inner Product:**
$$\langle u, v \rangle_{\text{tr}} = \text{tr}(\langle u, v \rangle_H) = \text{tr}\left(\sum_{i=1}^n u_i \bar{v}_i\right)$$

### Connection to Symplectic Product

**Theorem:**
For Paulis P, Q with GF(4) representations u, v:
$$\boxed{\langle u, v \rangle_{\text{tr}} = \langle \text{bin}(u), \text{bin}(v) \rangle_s \pmod{2}}$$

The trace inner product equals the symplectic inner product!

**Proof:**
Let $u_i = a_i + \omega b_i$ and $v_i = c_i + \omega d_i$.

$$u_i \bar{v}_i = (a_i + \omega b_i)(c_i + \bar{\omega} d_i)$$
$$= a_i c_i + a_i \bar{\omega} d_i + \omega b_i c_i + \omega \bar{\omega} b_i d_i$$
$$= a_i c_i + a_i \bar{\omega} d_i + \omega b_i c_i + b_i d_i$$

Taking trace:
$$\text{tr}(u_i \bar{v}_i) = \text{tr}(a_i c_i) + \text{tr}(a_i \bar{\omega} d_i) + \text{tr}(\omega b_i c_i) + \text{tr}(b_i d_i)$$

Since $a_i, b_i, c_i, d_i \in \{0, 1\} = \mathbb{F}_2$:
- tr(0) = 0, tr(1) = 0
- tr(ω) = 1, tr(ω̄) = 1

$$\text{tr}(u_i \bar{v}_i) = 0 + a_i d_i + b_i c_i + 0 = a_i d_i + b_i c_i$$

Summing over i:
$$\langle u, v \rangle_{\text{tr}} = \sum_i (a_i d_i + b_i c_i) = \mathbf{a} \cdot \mathbf{d} + \mathbf{b} \cdot \mathbf{c} = \langle (a|b), (c|d) \rangle_s$$

### Additive Codes over GF(4)

**Definition:**
An additive code C over GF(4) is an additive subgroup of GF(4)^n (closed under addition but not necessarily scalar multiplication).

**Self-Orthogonal:**
C is self-orthogonal if $C \subseteq C^{\perp_H}$ under the Hermitian inner product.

**Connection to Stabilizer Codes:**
$$\text{Self-orthogonal additive codes over GF(4)} \longleftrightarrow \text{Stabilizer codes}$$

### The CSS Construction in GF(4)

**CSS codes** have a special structure in GF(4):
- X-type stabilizers: vectors in {0, 1}^n ⊂ GF(4)^n
- Z-type stabilizers: vectors in {0, ω}^n ⊂ GF(4)^n

For classical codes C₁, C₂ with C₂⊥ ⊆ C₁:
$$\text{CSS}(C_1, C_2) \leftrightarrow \{u + \omega v : u \in C_2^{\perp}, v \in C_1^{\perp}\}$$

### Weight in GF(4)

**Definition:**
The weight of $u \in \text{GF}(4)^n$:
$$\text{wt}(u) = |\{i : u_i \neq 0\}|$$

This equals the Pauli weight.

**Symplectic Weight:**
$$\text{wt}_s((a|b)) = |\{i : a_i \neq 0 \text{ or } b_i \neq 0\}| = \text{wt}(u)$$

### Standard Form for Additive Codes

A self-orthogonal additive code over GF(4) can be put in standard form:
$$G = \begin{pmatrix}
I_r & A & B + \omega C \\
0 & D & E + \omega F
\end{pmatrix}$$

where various constraints ensure self-orthogonality.

---

## Worked Examples

### Example 1: Basic GF(4) Operations

Compute in GF(4):
a) $(1 + \omega) \cdot \omega$
b) $\omega^5$
c) $\bar{\omega} + \omega$

**Solutions:**

a) $1 + \omega = \bar{\omega}$, so $(1 + \omega) \cdot \omega = \bar{\omega} \cdot \omega = 1$

b) $\omega^3 = 1$, so $\omega^5 = \omega^3 \cdot \omega^2 = 1 \cdot \bar{\omega} = \bar{\omega}$

c) $\bar{\omega} + \omega = 1$ (from addition table, or note $\omega + \bar{\omega} = \omega + \omega^2 = \omega(1 + \omega) = \omega \cdot \bar{\omega} = 1$... wait that's multiplication)

Actually: $\omega + \bar{\omega} = \omega + (\omega + 1) = 1$ (since $\omega + \omega = 0$) ✓

### Example 2: Pauli to GF(4) Conversion

Convert $P = X_1 Z_2 Y_3 I_4$ to GF(4).

**Binary representation:**
$P \leftrightarrow (1, 0, 1, 0 | 0, 1, 1, 0)$

**GF(4) conversion:**
- Position 1: $a_1 = 1, b_1 = 0 \Rightarrow g_1 = 1 + \omega \cdot 0 = 1$
- Position 2: $a_2 = 0, b_2 = 1 \Rightarrow g_2 = 0 + \omega \cdot 1 = \omega$
- Position 3: $a_3 = 1, b_3 = 1 \Rightarrow g_3 = 1 + \omega \cdot 1 = \bar{\omega}$
- Position 4: $a_4 = 0, b_4 = 0 \Rightarrow g_4 = 0$

**GF(4) vector:**
$$P \leftrightarrow (1, \omega, \bar{\omega}, 0)$$

### Example 3: Checking Commutation in GF(4)

Do $P = X_1 Z_2$ and $Q = Z_1 X_2$ commute?

**GF(4) representations:**
- $P = X_1 Z_2 \leftrightarrow u = (1, \omega)$
- $Q = Z_1 X_2 \leftrightarrow v = (\omega, 1)$

**Hermitian inner product:**
$$\langle u, v \rangle_H = u_1 \bar{v}_1 + u_2 \bar{v}_2 = 1 \cdot \bar{\omega} + \omega \cdot 1 = \bar{\omega} + \omega = 1$$

**Trace:**
$$\langle u, v \rangle_{\text{tr}} = \text{tr}(1) = 0$$

Since $\langle u, v \rangle_{\text{tr}} = 0$, the operators **commute**.

**Verification with binary:**
$u_{\text{bin}} = (1, 0 | 0, 1)$, $v_{\text{bin}} = (0, 1 | 1, 0)$
$\langle u_{\text{bin}}, v_{\text{bin}} \rangle_s = 1 \cdot 1 + 0 \cdot 0 + 0 \cdot 0 + 1 \cdot 1 = 0$ ✓

### Example 4: Steane Code in GF(4)

The [[7,1,3]] Steane code stabilizers:
- $S_1 = IIIXXXX \leftrightarrow (0,0,0,1,1,1,1)$
- $S_2 = IXXIIXX \leftrightarrow (0,1,1,0,0,1,1)$
- $S_3 = XIXIXIX \leftrightarrow (1,0,1,0,1,0,1)$
- $S_4 = IIIZZZZ \leftrightarrow (0,0,0,\omega,\omega,\omega,\omega)$
- $S_5 = IZZIIZZ \leftrightarrow (0,\omega,\omega,0,0,\omega,\omega)$
- $S_6 = ZIZIZIZ \leftrightarrow (ω,0,\omega,0,\omega,0,\omega)$

**Generator matrix in GF(4):**
$$G = \begin{pmatrix}
0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 & 1 \\
0 & 0 & 0 & \omega & \omega & \omega & \omega \\
0 & \omega & \omega & 0 & 0 & \omega & \omega \\
\omega & 0 & \omega & 0 & \omega & 0 & \omega
\end{pmatrix}$$

---

## Practice Problems

### Level 1: Direct Application

1. **GF(4) Arithmetic:** Compute:
   a) $\omega + \bar{\omega} + 1$
   b) $\omega^{10}$
   c) $(\omega + 1)^2$
   d) $1 / \omega$ (multiplicative inverse)

2. **Pauli Conversion:** Convert to GF(4) notation:
   a) $Y \otimes X \otimes Z$
   b) $X_1 X_2 X_3 X_4$
   c) $Z_1 Z_2 Z_3 Z_4$

3. **Trace Calculation:** Compute tr(x) for:
   a) $x = \omega + 1$
   b) $x = \omega \cdot \bar{\omega}$
   c) $x = \omega^2 + \omega + 1$

### Level 2: Intermediate

4. **Hermitian Inner Product:** Compute $\langle u, v \rangle_H$ and $\langle u, v \rangle_{\text{tr}}$ for:
   a) $u = (1, \omega)$, $v = (1, \bar{\omega})$
   b) $u = (\bar{\omega}, \bar{\omega}, \bar{\omega})$, $v = (1, 1, 1)$

5. **Self-Orthogonality:** Check if the following code is self-orthogonal:
   $$C = \{(0,0), (1,1), (\omega, \omega), (\bar{\omega}, \bar{\omega})\}$$

6. **Binary Conversion:** Convert the GF(4) vector $(\omega, \bar{\omega}, 1, 0)$ to binary form $(a|b)$.

### Level 3: Challenging

7. **CSS Structure:** For the classical [4,2,2] code with generator matrix:
   $$G = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix}$$
   Construct the GF(4) generator matrix for the CSS code.

8. **Additive Code Construction:** Prove that the set:
   $$C = \text{span}_{\mathbb{F}_2}\{(1, \omega, \bar{\omega}), (\omega, 1, \bar{\omega})\}$$
   is an additive self-orthogonal code over GF(4).

9. **Weight Distribution:** For the [[5,1,3]] code, find the weight distribution of the code (number of codewords of each weight).

---

## Solutions

### Level 1 Solutions

1. **GF(4) Arithmetic:**
   a) $\omega + \bar{\omega} + 1 = 1 + 1 = 0$
   b) $\omega^{10} = \omega^{9+1} = (\omega^3)^3 \cdot \omega = 1 \cdot \omega = \omega$
   c) $(\omega + 1)^2 = \bar{\omega}^2 = \omega^4 = \omega$
   d) $1/\omega = \omega^{-1} = \omega^2 = \bar{\omega}$

2. **Pauli Conversion:**
   a) $Y \otimes X \otimes Z \leftrightarrow (\bar{\omega}, 1, \omega)$
   b) $XXXX \leftrightarrow (1, 1, 1, 1)$
   c) $ZZZZ \leftrightarrow (\omega, \omega, \omega, \omega)$

3. **Trace:**
   a) tr($\bar{\omega}$) = $\bar{\omega} + \omega = 1$
   b) tr(1) = 1 + 1 = 0
   c) tr($\omega^2 + \omega + 1$) = tr(0) = 0

### Level 2 Solutions

4. **Hermitian Inner Product:**
   a) $\langle(1,\omega), (1,\bar{\omega})\rangle_H = 1 \cdot \bar{1} + \omega \cdot \bar{\bar{\omega}} = 1 + \omega \cdot \omega = 1 + \bar{\omega}$
      tr$(1 + \bar{\omega}) = $ tr$(1) + $ tr$(\bar{\omega}) = 0 + 1 = 1$

   b) $\langle(\bar{\omega},\bar{\omega},\bar{\omega}), (1,1,1)\rangle_H = 3\bar{\omega} \cdot 1 = \bar{\omega} + \bar{\omega} + \bar{\omega} = \bar{\omega}$
      tr$(\bar{\omega}) = 1$

---

## Computational Lab

```python
"""
Day 732: GF(4) Representation of Stabilizer Codes
=================================================
Implementation of GF(4) arithmetic and quantum code representation.
"""

import numpy as np
from typing import List, Tuple, Dict

class GF4:
    """
    Represents an element of GF(4) = {0, 1, ω, ω̄}.

    Internal representation:
    - 0 → 0
    - 1 → 1
    - ω → 2
    - ω̄ → 3
    """

    # Addition table (row + col)
    ADD_TABLE = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ])

    # Multiplication table (row × col)
    MUL_TABLE = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ])

    # Conjugation: 0→0, 1→1, ω→ω̄, ω̄→ω
    CONJ = np.array([0, 1, 3, 2])

    # Symbols for display
    SYMBOLS = ['0', '1', 'ω', 'ω̄']

    def __init__(self, val: int):
        """Initialize with value 0, 1, 2 (ω), or 3 (ω̄)."""
        self.val = val % 4

    @classmethod
    def zero(cls):
        return cls(0)

    @classmethod
    def one(cls):
        return cls(1)

    @classmethod
    def omega(cls):
        return cls(2)

    @classmethod
    def omega_bar(cls):
        return cls(3)

    @classmethod
    def from_pauli(cls, p: str):
        """Convert single Pauli character to GF(4)."""
        mapping = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
        return cls(mapping[p.upper()])

    def to_pauli(self) -> str:
        """Convert to Pauli character."""
        return ['I', 'X', 'Z', 'Y'][self.val]

    def __add__(self, other: 'GF4') -> 'GF4':
        return GF4(self.ADD_TABLE[self.val, other.val])

    def __mul__(self, other: 'GF4') -> 'GF4':
        return GF4(self.MUL_TABLE[self.val, other.val])

    def __neg__(self) -> 'GF4':
        """In characteristic 2, -a = a."""
        return GF4(self.val)

    def conjugate(self) -> 'GF4':
        """Return conjugate: ω ↔ ω̄, fixes 0, 1."""
        return GF4(self.CONJ[self.val])

    def trace(self) -> int:
        """Compute trace to F₂: tr(x) = x + x̄."""
        sum_val = self + self.conjugate()
        # Trace is 0 for {0, 1}, 1 for {ω, ω̄}
        return 1 if self.val >= 2 else 0

    def inverse(self) -> 'GF4':
        """Multiplicative inverse (only for non-zero)."""
        if self.val == 0:
            raise ValueError("Cannot invert zero")
        # 1⁻¹ = 1, ω⁻¹ = ω̄, ω̄⁻¹ = ω
        inv_map = {1: 1, 2: 3, 3: 2}
        return GF4(inv_map[self.val])

    def __eq__(self, other) -> bool:
        if isinstance(other, GF4):
            return self.val == other.val
        return self.val == other

    def __repr__(self):
        return self.SYMBOLS[self.val]

    def __hash__(self):
        return hash(self.val)


class GF4Vector:
    """Vector over GF(4)."""

    def __init__(self, elements: List[GF4]):
        self.elements = elements
        self.n = len(elements)

    @classmethod
    def from_list(cls, vals: List[int]):
        """Create from list of integers (0,1,2,3)."""
        return cls([GF4(v) for v in vals])

    @classmethod
    def from_pauli(cls, pauli_str: str):
        """Create from Pauli string like 'XYZII'."""
        return cls([GF4.from_pauli(p) for p in pauli_str])

    @classmethod
    def from_binary(cls, a: np.ndarray, b: np.ndarray):
        """Create from binary vectors a (X-part) and b (Z-part)."""
        n = len(a)
        elements = []
        for i in range(n):
            # g = a + ωb
            # a=0,b=0 → 0; a=1,b=0 → 1; a=0,b=1 → ω; a=1,b=1 → ω̄
            val = a[i] + 2*b[i]
            if a[i] == 1 and b[i] == 1:
                val = 3  # ω̄ = 1 + ω
            elements.append(GF4(val))
        return cls(elements)

    def to_binary(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to binary (a|b) representation."""
        a = np.zeros(self.n, dtype=int)
        b = np.zeros(self.n, dtype=int)
        for i, g in enumerate(self.elements):
            if g.val == 1:    # X
                a[i] = 1
            elif g.val == 2:  # Z = ω
                b[i] = 1
            elif g.val == 3:  # Y = ω̄
                a[i] = 1
                b[i] = 1
        return a, b

    def to_pauli(self) -> str:
        """Convert to Pauli string."""
        return ''.join(g.to_pauli() for g in self.elements)

    def __add__(self, other: 'GF4Vector') -> 'GF4Vector':
        return GF4Vector([a + b for a, b in zip(self.elements, other.elements)])

    def weight(self) -> int:
        """Number of non-zero elements."""
        return sum(1 for g in self.elements if g.val != 0)

    def hermitian_inner_product(self, other: 'GF4Vector') -> GF4:
        """Compute <u, v>_H = Σ u_i * conj(v_i)."""
        result = GF4.zero()
        for u_i, v_i in zip(self.elements, other.elements):
            result = result + u_i * v_i.conjugate()
        return result

    def trace_inner_product(self, other: 'GF4Vector') -> int:
        """Compute tr(<u, v>_H)."""
        hip = self.hermitian_inner_product(other)
        return hip.trace()

    def __repr__(self):
        return f"({', '.join(str(g) for g in self.elements)})"


def binary_to_gf4(a: np.ndarray, b: np.ndarray) -> List[int]:
    """Convert binary (a|b) to GF(4) values."""
    n = len(a)
    result = []
    for i in range(n):
        if a[i] == 0 and b[i] == 0:
            result.append(0)
        elif a[i] == 1 and b[i] == 0:
            result.append(1)
        elif a[i] == 0 and b[i] == 1:
            result.append(2)
        else:  # a[i] == 1 and b[i] == 1
            result.append(3)
    return result


def gf4_to_binary(gf4_vals: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert GF(4) values to binary (a|b)."""
    n = len(gf4_vals)
    a = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    for i, g in enumerate(gf4_vals):
        if g == 1:
            a[i] = 1
        elif g == 2:
            b[i] = 1
        elif g == 3:
            a[i] = 1
            b[i] = 1
    return a, b


def is_self_orthogonal_gf4(vectors: List[GF4Vector]) -> bool:
    """Check if GF4 vectors form a self-orthogonal set."""
    for i, u in enumerate(vectors):
        for v in vectors[i:]:
            if u.trace_inner_product(v) != 0:
                return False
    return True


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 732: GF(4) Representation")
    print("=" * 60)

    # Example 1: Basic GF(4) arithmetic
    print("\n1. GF(4) Arithmetic")
    print("-" * 40)

    omega = GF4.omega()
    omega_bar = GF4.omega_bar()
    one = GF4.one()

    print(f"ω + ω̄ = {omega + omega_bar}")
    print(f"ω × ω = {omega * omega}")
    print(f"ω × ω̄ = {omega * omega_bar}")
    print(f"ω^3 = {omega * omega * omega}")
    print(f"conj(ω) = {omega.conjugate()}")
    print(f"tr(ω) = {omega.trace()}")
    print(f"tr(1) = {one.trace()}")

    # Example 2: Pauli to GF(4) conversion
    print("\n2. Pauli ↔ GF(4) Conversion")
    print("-" * 40)

    paulis = ['XYZII', 'XXXX', 'ZZZZ', 'YZYX']
    for p in paulis:
        v = GF4Vector.from_pauli(p)
        print(f"{p} → {v}")

    # Example 3: Binary ↔ GF(4) conversion
    print("\n3. Binary ↔ GF(4) Conversion")
    print("-" * 40)

    # XZYI = (1,0,1,0 | 0,1,1,0)
    a = np.array([1, 0, 1, 0])
    b = np.array([0, 1, 1, 0])
    v = GF4Vector.from_binary(a, b)
    print(f"Binary (a|b) = ({a}|{b})")
    print(f"GF(4): {v}")
    print(f"Pauli: {v.to_pauli()}")

    # Convert back
    a_back, b_back = v.to_binary()
    print(f"Back to binary: ({a_back}|{b_back})")

    # Example 4: Commutation check via trace inner product
    print("\n4. Commutation via Trace Inner Product")
    print("-" * 40)

    test_pairs = [
        ('XZ', 'ZX'),
        ('XX', 'ZZ'),
        ('XY', 'YX'),
        ('XYZ', 'ZXY'),
    ]

    for p1, p2 in test_pairs:
        v1 = GF4Vector.from_pauli(p1)
        v2 = GF4Vector.from_pauli(p2)
        hip = v1.hermitian_inner_product(v2)
        tip = v1.trace_inner_product(v2)
        status = "commute" if tip == 0 else "anticommute"
        print(f"{p1}, {p2}: <u,v>_H = {hip}, tr = {tip} → {status}")

    # Example 5: [[5,1,3]] code in GF(4)
    print("\n5. [[5,1,3]] Code in GF(4)")
    print("-" * 40)

    stabilizers_513 = [
        'XZZXI',
        'IXZZX',
        'XIXZZ',
        'ZXIXZ'
    ]

    vecs_513 = [GF4Vector.from_pauli(s) for s in stabilizers_513]

    print("Stabilizers in GF(4):")
    for s, v in zip(stabilizers_513, vecs_513):
        print(f"  {s} → {v}")

    print(f"\nSelf-orthogonal: {is_self_orthogonal_gf4(vecs_513)}")

    # Verify trace inner products
    print("\nTrace inner product matrix:")
    for i, vi in enumerate(vecs_513):
        row = [vi.trace_inner_product(vj) for vj in vecs_513]
        print(f"  {row}")

    # Example 6: Steane code structure
    print("\n6. [[7,1,3]] Steane Code CSS Structure")
    print("-" * 40)

    steane_X = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX']
    steane_Z = ['IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']

    print("X-type stabilizers (in {0,1} ⊂ GF(4)):")
    for s in steane_X:
        v = GF4Vector.from_pauli(s)
        print(f"  {s} → {v}")

    print("\nZ-type stabilizers (in {0,ω} ⊂ GF(4)):")
    for s in steane_Z:
        v = GF4Vector.from_pauli(s)
        print(f"  {s} → {v}")

    # Example 7: Weight calculation
    print("\n7. Weight in GF(4)")
    print("-" * 40)

    test_vecs = ['XYZII', 'XXXXX', 'IIIIZ', 'YYYYY']
    for p in test_vecs:
        v = GF4Vector.from_pauli(p)
        print(f"wt({p}) = {v.weight()}")

    print("\n" + "=" * 60)
    print("End of Day 732 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| GF(4) elements | $\{0, 1, \omega, \bar{\omega}\}$, $\omega^2 + \omega + 1 = 0$ |
| Pauli mapping | $I \to 0, X \to 1, Z \to \omega, Y \to \bar{\omega}$ |
| Binary-GF(4) | $g_i = a_i + \omega b_i$ |
| Trace | $\text{tr}(x) = x + \bar{x}$ |
| Hermitian IP | $\langle u, v \rangle_H = \sum u_i \bar{v}_i$ |
| Commutation | $\text{tr}(\langle u, v \rangle_H) = \langle u, v \rangle_s$ |

### Main Takeaways

1. **GF(4)** provides a compact representation: one element per qubit
2. **Pauli multiplication** becomes GF(4) addition
3. **The trace inner product** equals the symplectic inner product
4. **Stabilizer codes** correspond to self-orthogonal additive codes over GF(4)
5. **CSS codes** have special structure: X-generators in {0,1}, Z-generators in {0,ω}

---

## Daily Checklist

- [ ] I can perform GF(4) arithmetic
- [ ] I can convert Paulis to GF(4) and back
- [ ] I understand the trace function
- [ ] I can compute the Hermitian inner product
- [ ] I know how the trace IP relates to commutation
- [ ] I understand the connection to additive codes

---

## Preview: Day 733

Tomorrow we study **Parity Check Matrices** for stabilizer codes:
- Constructing H from stabilizer generators
- Standard form for stabilizer codes
- Finding encoding circuits from H
- Syndrome extraction
- Examples: Steane, Shor, and [[5,1,3]] codes
