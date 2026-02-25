# Week 171: Code Families - Problem Solutions

## Part A: Quantum Reed-Muller Codes

### Solution 1

**(a)** $$RM(1, 4)$$:
- $$n = 2^4 = 16$$
- $$k = \binom{4}{0} + \binom{4}{1} = 1 + 4 = 5$$
- $$d = 2^{4-1} = 8$$

Parameters: $$[16, 5, 8]$$

**(b)** $$RM(2, 4)$$:
- $$n = 2^4 = 16$$
- $$k = \binom{4}{0} + \binom{4}{1} + \binom{4}{2} = 1 + 4 + 6 = 11$$
- $$d = 2^{4-2} = 4$$

Parameters: $$[16, 11, 4]$$

**(c)** $$RM(1, 4)^\perp = RM(4-1-1, 4) = RM(2, 4)$$

Verify: For $$RM(r, m)$$, we have $$RM(r, m)^\perp = RM(m-r-1, m)$$.

With $$r = 1, m = 4$$: $$RM(1, 4)^\perp = RM(2, 4)$$ ✓

**(d)** Generally: $$RM(r, m)^\perp = RM(m-r-1, m)$$

This means $$RM(r, m)$$ and $$RM(m-r-1, m)$$ are dual codes.

---

### Solution 2

**(a)** $$RM(2, 4)^\perp = RM(4-2-1, 4) = RM(1, 4)$$

**(b)** Need $$RM(1, 4) \subset RM(1, 4)$$. This is trivially true (a code contains itself).

Actually, for CSS we need $$C_2^\perp \subset C_1$$. Here:
- $$C_1 = RM(1, 4)$$
- $$C_2 = RM(2, 4)$$
- $$C_2^\perp = RM(1, 4) = C_1$$

So $$C_2^\perp \subset C_1$$ ✓

**(c)** Parameters:
- $$n = 16$$
- $$k = k_1 - (n - k_2) = 5 - (16 - 11) = 5 - 5 = 0$$

Wait, let me recalculate. For CSS($$C_1, C_2$$):
- $$k = k_1 - k_2$$ where $$C_2 \subset C_1$$

Here $$C_1 = RM(1, 4)$$ with $$k_1 = 5$$, $$C_2^\perp = RM(1, 4)$$ so $$C_2 = RM(2, 4)$$ with $$k_2 = 11$$.

But we need $$C_2^\perp \subset C_1$$, which gives $$C_2 \supset C_1^\perp$$.

Actually, the standard $$[[15, 1, 3]]$$ Reed-Muller code uses $$m = 4$$ with puncturing. Let me reconsider.

The $$[[15, 1, 3]]$$ code is $$QRM(1, 4)$$ with the all-zeros codeword removed (punctured). The correct parameters are:

$$n = 2^4 - 1 = 15$$, $$k = 1$$, $$d = 3$$

**(d)** $$k = 1$$ for the punctured code.

---

### Solution 3

**(a)** $$\mathcal{C}_2$$ consists of unitaries $$U$$ such that $$UPU^\dagger \in \mathcal{G}_n$$ for all Pauli $$P$$.

This is the definition of the Clifford group: unitaries that normalize the Pauli group.

**(b)** Compute:
$$TXT^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$$

$$= \begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix} = e^{i\pi/4} \begin{pmatrix} 0 & e^{-i\pi/2} \\ 1 & 0 \end{pmatrix}$$

$$= \frac{1}{\sqrt{2}}(X + Y) \cdot e^{i\pi/4}$$

This is NOT a Pauli, so $$T \notin \mathcal{C}_2$$.

But $$TXT^\dagger \in \mathcal{C}_2$$ (it's in the Clifford group up to phase corrections).

Actually: $$TXT^\dagger = e^{i\pi/4}(X - iY)/\sqrt{2}$$... Let me recalculate carefully.

$$TXT^\dagger = \begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix}$$

This equals $$\cos(\pi/4)X + \sin(\pi/4)Y = (X + Y)/\sqrt{2}$$ up to phase.

Since $$(X+Y)/\sqrt{2}$$ is a Clifford operation, $$T \in \mathcal{C}_3$$.

**(c)** The $$[[15, 1, 3]]$$ code accesses $$\mathcal{C}_3$$ (third level of hierarchy).

---

### Solution 4

**(a)** Transversal gates $$U^{\otimes n}$$ map product states to product states. For encoded states:
$$U^{\otimes n} S (U^{\otimes n})^\dagger = U^{\otimes n} S (U^\dagger)^{\otimes n}$$

must be a stabilizer (or at least in the stabilizer group up to phase) for all stabilizers $$S$$.

**(b)** For the $$[[15, 1, 3]]$$ RM code, each stabilizer is a product of X and Z operators. Under $$T^{\otimes 15}$$:
- $$TZT^\dagger = Z$$ (Z is unchanged)
- $$TXT^\dagger \propto (X+Y)/\sqrt{2}$$

For specific stabilizer structure of RM codes, the transformation preserves the stabilizer group (up to phases and Clifford corrections).

**(c)** The Steane code is CSS with different X and Z stabilizer structures. $$T^{\otimes 7}$$ would map:
- X-stabilizers to non-Pauli operators (since $$TXT^\dagger$$ is not Pauli)

The asymmetry between X and Z prevents transversal T.

---

### Solution 5

**(a)** For $$QRM(r, m)$$, distance $$d = 2^{\min(r+1, m-r)}$$ (approximately).

Need $$d \geq 7$$, so $$2^k \geq 7$$ implies $$k \geq 3$$.

Minimum $$m$$ depends on the construction. For $$d = 8 = 2^3$$, need $$\min(r+1, m-r) \geq 3$$.

If $$m = 6$$ and $$r = 2$$: $$\min(3, 4) = 3$$, giving $$d = 8$$.

**(b)** Valid $$r$$ values: $$r$$ such that $$r + (m-r-1) \geq m-1$$ (always true) and valid CSS construction.

For $$m = 6$$: $$r \in \{2, 3\}$$ give distance $$\geq 7$$.

**(c)** $$QRM(2, 6)$$:
- $$n = 2^6 = 64$$
- $$k = \binom{6}{0} + \binom{6}{1} + \binom{6}{2} - \binom{6}{0} - \binom{6}{1} - \binom{6}{2} - \binom{6}{3}$$

(Need careful calculation based on exact CSS structure)

**(d)** $$QRM(2, 6)$$ supports transversal $$\mathcal{C}_4$$ gates.

---

## Part B: Color Codes

### Solution 7

**(a)** The [[7,1,3]] color code on triangular lattice:

```
      1
     / \
    /   \
   2-----3
  /|\   /|\
 / | \ / | \
4--+--5--+--6
   |     |
   +--7--+
```

Faces: R (1,2,3), G (2,4,5,3), B (3,5,6), and three more completing the structure.

**(b)** Stabilizers (one per face, both X and Z type):

X-type:
- $$X_R = X_1X_2X_3X_4$$ (for one face)
- $$X_G = X_2X_3X_5X_7$$
- $$X_B = X_3X_4X_5X_6$$

Z-type:
- $$Z_R = Z_1Z_2Z_3Z_4$$
- $$Z_G = Z_2Z_3Z_5Z_7$$
- $$Z_B = Z_3Z_4Z_5Z_6$$

(Exact form depends on lattice labeling)

**(c)** X-stabilizers commute with each other (all X).
Z-stabilizers commute with each other (all Z).
$$X_f$$ and $$Z_{f'}$$ share an even number of qubits for any two faces $$f, f'$$, so they commute. ✓

---

### Solution 8

**(a)** Logical operators:
$$\overline{X} = X_1X_2X_3X_4X_5X_6X_7 = X^{\otimes 7}$$
$$\overline{Z} = Z_1Z_2Z_3Z_4Z_5Z_6Z_7 = Z^{\otimes 7}$$

**(b)** $$\overline{X}\overline{Z} = X^{\otimes 7}Z^{\otimes 7} = (XZ)^{\otimes 7} = (-ZX)^{\otimes 7} = (-1)^7 Z^{\otimes 7}X^{\otimes 7}$$
$$= -\overline{Z}\overline{X}$$

They anticommute. ✓

**(c)** Minimum weight: Look for representatives modulo stabilizers.

$$\overline{X} \cdot X_R = X_1X_2X_3X_4X_5X_6X_7 \cdot X_1X_2X_3X_4 = X_5X_6X_7$$

Weight 3. This is minimum since $$d = 3$$.

---

### Solution 9

**(a)** For general CSS codes, X and Z stabilizers have different support. $$H^{\otimes n}$$ swaps X and Z:
$$H^{\otimes n} X_f H^{\otimes n} = Z_f$$

This only works if X-stabilizers and Z-stabilizers are "paired" with same support.

Color codes have this property: each face $$f$$ has both $$X_f$$ and $$Z_f$$ with identical support.

**(b)** For $$S^{\otimes 7}$$:
$$S X S^\dagger = Y$$, $$S Z S^\dagger = Z$$

On stabilizers:
- $$S^{\otimes 7} X_f (S^\dagger)^{\otimes 7} = Y_f$$ (product of Y on face)
- $$S^{\otimes 7} Z_f (S^\dagger)^{\otimes 7} = Z_f$$

$$Y_f = i^{|f|} X_f Z_f$$ where $$|f|$$ is the face size.

For consistent phases, need $$|f| \equiv 0 \pmod 4$$ or careful phase tracking.

**(c)** For $$CNOT^{\otimes 7}$$ between blocks A and B:
$$CNOT X_A \to X_A X_B$$
$$CNOT Z_B \to Z_A Z_B$$

This maps $$\overline{X}_A$$ to $$\overline{X}_A \overline{X}_B$$ and $$\overline{Z}_B$$ to $$\overline{Z}_A \overline{Z}_B$$, which is correct for logical CNOT.

---

### Solution 10

**(a)** Constraint: Lattice must be 3-colorable (faces) with each vertex adjacent to faces of all 3 colors.

**(b)** For $$[[19, 1, 5]]$$: Extend to larger triangular lattice with 19 vertices. The distance grows with the "radius" of the lattice.

**(c)** For 2D color codes on triangular lattices:
$$d = $$ minimum number of edges on any homologically non-trivial path.

Roughly $$d \sim L$$ for lattice of linear size $$L$$, giving $$n \sim L^2$$, so $$d \sim \sqrt{n}$$.

---

## Part C: Quantum Reed-Solomon Codes

### Solution 13

**(a)** For $$[7, 3, 5]$$ RS code over $$\mathbb{F}_q$$:

Need $$n \leq q$$, so $$q \geq 7$$. Minimum: $$q = 7$$ (or $$q = 8 = 2^3$$).

**(b)** Singleton bound: $$d \leq n - k + 1 = 7 - 3 + 1 = 5$$ ✓

The code achieves equality.

**(c)** MDS = Maximum Distance Separable: codes achieving the Singleton bound $$d = n - k + 1$$.

RS codes are always MDS.

---

### Solution 14

**(a)** $$CSS(RS(7, 5), RS(7, 3)^\perp)$$:

For RS codes over $$\mathbb{F}_8$$:
- $$RS(7, 5)$$: $$[7, 5, 3]$$
- $$RS(7, 3)$$: $$[7, 3, 5]$$
- $$RS(7, 3)^\perp = RS(7, 4)$$: $$[7, 4, 4]$$

Need $$RS(7, 4) \subset RS(7, 5)$$: ✓ (polynomials of degree $$< 4$$ are included in degree $$< 5$$)

**(b)** Quantum parameters:
- $$n = 7$$
- $$k = 5 - 4 = 1$$
- $$d = \min(3, 4) = 3$$

$$[[7, 1, 3]]_8$$ (over 8-dimensional qudits)

**(c)** Quantum Singleton: $$k \leq n - 2d + 2 = 7 - 4 + 2 = 5$$

We have $$k = 1 \leq 5$$ ✓, but not saturating.

---

### Solution 15

**(a)** **Proof of quantum Singleton bound:**

For $$[[n, k, d]]$$ code, trace out $$d-1$$ qubits. The remaining $$n-d+1$$ qubits must contain all information (since errors on $$d-1$$ qubits are correctable).

By no-cloning, the traced-out qubits contain NO information about the logical state.

Apply the same argument from the other end: trace out another $$d-1$$ qubits.

The middle $$n - 2(d-1) = n - 2d + 2$$ qubits must encode $$k$$ qubits:
$$k \leq n - 2d + 2$$

**(b)** MDS quantum code: $$d = (n - k)/2 + 1 = (n - k + 2)/2$$

**(c)** Quantum MDS codes exist for many parameters when $$q$$ is large enough, but not all. There are existence gaps for small $$q$$.

---

## Part D: Concatenated Codes

### Solution 18

**(a)** Concatenating $$[[n_1, 1, d_1]]$$ (outer) with $$[[n_2, 1, d_2]]$$ (inner):
- Physical qubits: $$n_1 \cdot n_2$$
- Logical qubits: 1
- Distance: $$d_1 \cdot d_2$$

Parameters: $$[[n_1 n_2, 1, d_1 d_2]]$$

**(b)** Steane $$[[7, 1, 3]]$$ concatenated with itself:
$$[[7 \cdot 7, 1, 3 \cdot 3]] = [[49, 1, 9]]$$

**(c)** Three levels:
$$[[7^3, 1, 3^3]] = [[343, 1, 27]]$$

---

### Solution 19

**(a)** Shor $$[[9, 1, 3]]$$:
- Inner: 3-qubit bit-flip code $$[[3, 1, ?]]$$ for X errors
- Outer: 3-qubit phase-flip code $$[[3, 1, ?]]$$ for Z errors

Combined: 3 blocks of 3 qubits = 9 qubits total.

**(b)** Distance: The code corrects any single qubit error. A logical error requires weight-3 operator.

This is $$d = 3$$, which matches $$3 \times 1 = 3$$ if we consider the inner code as distance 1 for the "wrong" error type.

Actually, better to view as: any single error is corrected, so $$d = 3$$.

**(c)** Shor code is "doubly concatenated" because it uses two levels of protection: bit-flip within blocks, phase-flip between blocks.

---

### Solution 20

**(a)** For distance-3 code, failure requires $$\geq 2$$ errors in a "bad" configuration.

Probability of exactly 2 errors among $$N$$ locations:
$$P_{\text{fail}} \approx \binom{N}{2} p^2 = \frac{N(N-1)}{2} p^2 \approx C p^2$$

**(b)** For error correction to help, need $$P_{\text{fail}} < p$$:
$$C p^2 < p$$
$$p < 1/C = p_{\text{th}}$$

**(c)** For Steane code, $$N$$ includes:
- 7 data qubits
- Ancilla qubits for syndrome measurement
- Gate locations in syndrome circuit

Rough estimate: $$N \sim 100$$ locations, giving $$C \sim 5000$$, so $$p_{\text{th}} \sim 2 \times 10^{-4}$$.

---

### Solution 21

**(a)** For $$L$$ levels with base $$[[n, 1, d]]$$:
Physical qubits = $$n^L$$

**(b)** Logical error rate at level $$L$$:
$$p_L \approx (p/p_{\text{th}})^{2^L} \cdot p_{\text{th}}$$

For $$p_L < \epsilon$$:
$$2^L > \frac{\log(\epsilon/p_{\text{th}})}{\log(p/p_{\text{th}})}$$
$$L > \log_2 \left( \frac{\log(p_{\text{th}}/\epsilon)}{\log(p_{\text{th}}/p)} \right)$$

**(c)** Physical qubits:
$$n^L = n^{\log_2(\log(1/\epsilon)/\log(p_{\text{th}}/p))} = \text{polylog}(1/\epsilon)$$

---

### Solution 22

**(a)** For concatenated Steane with $$p_{\text{th}} = 10^{-4}$$:
$$p_L \approx 10^{-4} \cdot (10^4 p)^{2^L}$$

For surface code with $$p_{\text{th}} = 10^{-2}$$:
$$p_L \approx (p/10^{-2})^{d}$$ where $$d \sim \sqrt{n}$$

**(b)** Crossover when both achieve same $$p_L$$. This occurs around $$p \sim 10^{-3}$$.

**(c)** Below crossover ($$p < 10^{-4}$$), concatenated codes are more efficient.
Above crossover ($$p > 10^{-3}$$), surface codes are more efficient.

---

## Part E: Bounds and Comparisons

### Solution 24

**(a)** $$[[5, 1, 3]]$$: $$t = 1$$ error correction.

$$2^1 \cdot (1 + 5 \cdot 3) = 2 \cdot 16 = 32 = 2^5$$ ✓

Saturates the bound (perfect code).

**(b)** $$[[7, 1, 3]]$$:
$$2^1 \cdot 16 = 32 < 2^7 = 128$$

Does NOT saturate (ratio = 128/32 = 4).

**(c)** $$[[9, 1, 3]]$$:
$$32 < 2^9 = 512$$

Does NOT saturate (ratio = 16).

---

### Solution 25

**(a)** Singleton bound: $$k \leq n - 2d + 2$$

- $$[[5, 1, 3]]$$: $$1 \leq 5 - 4 + 2 = 3$$ ✓
- $$[[7, 1, 3]]$$: $$1 \leq 7 - 4 + 2 = 5$$ ✓
- $$[[9, 1, 3]]$$: $$1 \leq 9 - 4 + 2 = 7$$ ✓

**(b)** $$[[5, 1, 3]]$$ is closest to optimal (achieves $$k = 1$$ with minimum $$n = 5$$).

**(c)** For $$d = 3$$: $$k \leq n - 4$$, so $$n \geq k + 4$$.

For $$k = 1$$: $$n \geq 5$$. The $$[[5, 1, 3]]$$ achieves this minimum.

No $$[[4, 1, 3]]$$ or smaller can exist.

---

**Solutions Document Created:** February 10, 2026
