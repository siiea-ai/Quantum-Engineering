# Week 159: Entanglement - Problem Solutions

## Section A: Separability and Bell States

### Solution 1

**(a) $$|00\rangle$$:** Separable (product state $$|0\rangle \otimes |0\rangle$$)

**(b) $$\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:** Entangled (Bell state $$|\Phi^+\rangle$$, Schmidt rank 2)

**(c) $$\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$:**
$$= \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle + |1\rangle) = |+\rangle|+\rangle$$
Separable (product state)

**(d) $$\frac{1}{\sqrt{2}}(|00\rangle + |01\rangle)$$:**
$$= \frac{1}{\sqrt{2}}|0\rangle(|0\rangle + |1\rangle) = |0\rangle|+\rangle$$
Separable (product state)

---

### Solution 2

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

**Orthonormality check:**

$$\langle\Phi^+|\Phi^+\rangle = \frac{1}{2}(1 + 1) = 1$$ ✓

$$\langle\Phi^+|\Phi^-\rangle = \frac{1}{2}(1 - 1) = 0$$ ✓

$$\langle\Phi^+|\Psi^+\rangle = \frac{1}{2}(0 + 0) = 0$$ ✓

(All other inner products similarly give 0 or 1 as required)

---

### Solution 3

For $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:

$$\rho = \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

$$\rho_A = \text{Tr}_B(\rho) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

By symmetry, $$\rho_B = \frac{I}{2}$$.

Same calculation for other Bell states (the cross terms vanish upon partial trace).

---

### Solution 4

Circuit: $$|00\rangle \xrightarrow{H \otimes I} \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

$$\xrightarrow{\text{CNOT}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$ ✓

---

### Solution 5

$$Z \otimes I$$ on $$|\Phi^+\rangle$$:
$$(Z \otimes I)\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle) = |\Phi^-\rangle$$ ✓

$$X \otimes I$$ on $$|\Phi^+\rangle$$:
$$(X \otimes I)\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = \frac{1}{\sqrt{2}}(|10\rangle + |01\rangle) = |\Psi^+\rangle$$ ✓

$$iY \otimes I$$ on $$|\Phi^+\rangle$$:
$$iY|0\rangle = |1\rangle$$, $$iY|1\rangle = -|0\rangle$$
$$(iY \otimes I)\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = \frac{1}{\sqrt{2}}(|10\rangle - |01\rangle) = |\Psi^-\rangle$$ ✓

---

### Solution 6

$$|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$$

**(a)** Separable when Schmidt rank = 1, i.e., $$\sin\theta = 0$$ or $$\cos\theta = 0$$:
$$\theta = 0, \pi/2, \pi, 3\pi/2, ...$$ (multiples of $$\pi/2$$)

**(b)** Schmidt coefficients: $$\lambda_1 = \cos\theta$$, $$\lambda_2 = \sin\theta$$

$$E(\theta) = -\cos^2\theta\log_2(\cos^2\theta) - \sin^2\theta\log_2(\sin^2\theta)$$

This is the binary entropy $$H_2(\cos^2\theta)$$.

**(c)** Maximum at $$\theta = \pi/4$$ (or $$45°$$), where $$\cos^2\theta = \sin^2\theta = 1/2$$.
$$E_{\max} = 1$$ ebit.

---

### Solution 7

Werner state: $$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$$

**(a) Separable:** For $$p \leq 1/3$$

**(b) Entangled:** For $$p > 1/3$$

(This can be shown via PPT criterion or by computing the CHSH violation.)

---

### Solution 8

**Proof:**

$$(\Leftarrow)$$ If $$|\psi\rangle$$ is a product state $$|a\rangle|b\rangle$$:
$$\rho_A = |a\rangle\langle a|$$ is pure.

$$(\Rightarrow)$$ If $$\rho_A$$ is pure, then $$\rho_A = |a\rangle\langle a|$$ for some $$|a\rangle$$.
Schmidt decomposition has only one term (since $$\rho_A$$ has rank 1).
Thus $$|\psi\rangle = |a\rangle|b\rangle$$ is a product state.

Contrapositive: entangled $$\Leftrightarrow$$ reduced state is mixed.

---

## Section B: CHSH Inequality

### Solution 9

**(a) Assumptions:**
1. Realism: Outcomes predetermined by hidden variable $$\lambda$$
2. Locality: Alice's outcome doesn't depend on Bob's setting

**(b) Correlation function:**
$$E(a,b) = \int d\lambda \, p(\lambda) A(a,\lambda) B(b,\lambda)$$

where $$A, B = \pm 1$$.

**(c) Derivation:**
$$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$
$$= \int d\lambda \, p(\lambda)[A_a(B_b - B_{b'}) + A_{a'}(B_b + B_{b'})]$$

Since $$B_b, B_{b'} = \pm 1$$: either $$(B_b - B_{b'}) = 0, (B_b + B_{b'}) = \pm 2$$ or vice versa.

Thus $$|A_a(B_b - B_{b'}) + A_{a'}(B_b + B_{b'})| = 2$$ always.

$$|S| \leq \int d\lambda \, p(\lambda) \cdot 2 = 2$$

---

### Solution 10

For $$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$:

$$E(a,b) = \langle\Psi^-|(\vec{a}\cdot\vec{\sigma}) \otimes (\vec{b}\cdot\vec{\sigma})|\Psi^-\rangle$$

Using the identity for the singlet:
$$E(a,b) = -\vec{a} \cdot \vec{b} = -\cos\theta_{ab}$$

where $$\theta_{ab}$$ is the angle between $$\vec{a}$$ and $$\vec{b}$$.

---

### Solution 11

Settings: $$a = 0°$$, $$a' = 90°$$, $$b = 45°$$, $$b' = 135°$$

Angles between vectors:
- $$\theta_{ab} = 45°$$: $$E(a,b) = -\cos 45° = -\frac{1}{\sqrt{2}}$$
- $$\theta_{ab'} = 135°$$: $$E(a,b') = -\cos 135° = \frac{1}{\sqrt{2}}$$
- $$\theta_{a'b} = 45°$$: $$E(a',b) = -\cos 45° = -\frac{1}{\sqrt{2}}$$
- $$\theta_{a'b'} = 45°$$: $$E(a',b') = -\cos 45° = -\frac{1}{\sqrt{2}}$$

$$S = -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}} \times (-1) = -\frac{4}{\sqrt{2}} = -2\sqrt{2}$$

Wait, let me recalculate:
$$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$
$$= -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} + (-\frac{1}{\sqrt{2}}) + (-\frac{1}{\sqrt{2}})$$

Hmm, that gives $$-4/\sqrt{2}$$. Let me check the formula again.

Actually $$E(a,b') = -\cos(135°) = -(-\frac{1}{\sqrt{2}}) = \frac{1}{\sqrt{2}}$$

$$S = -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} = -\frac{4}{\sqrt{2}} = -2\sqrt{2}$$

$$|S| = 2\sqrt{2} \approx 2.83$$ ✓

---

### Solution 12

**Tsirelson bound proof sketch:**

Observables $$A_a, A_{a'}, B_b, B_{b'}$$ satisfy $$(A_a)^2 = (B_b)^2 = I$$ (outcomes $$\pm 1$$).

The CHSH operator:
$$\mathcal{S} = A_a \otimes (B_b - B_{b'}) + A_{a'} \otimes (B_b + B_{b'})$$

$$\mathcal{S}^2 = 4I - [A_a, A_{a'}] \otimes [B_b, B_{b'}]$$

Since $$\|[A,A']\| \leq 2$$ and $$\|[B,B']\| \leq 2$$:
$$\|\mathcal{S}^2\| \leq 4 + 4 = 8$$
$$\|\mathcal{S}\| \leq 2\sqrt{2}$$

---

### Solution 13

For $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:

**(a)** $$E(a,b) = \vec{a} \cdot \vec{b} = \cos\theta_{ab}$$

(Note: opposite sign compared to singlet!)

**(b)** Optimal: $$a = 0°$$, $$a' = 90°$$, $$b = 45°$$, $$b' = -45°$$

**(c)** $$|S| = 2\sqrt{2}$$

---

### Solution 14

$$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)|00\rangle\langle 00|$$

**(a)** CHSH violation occurs when $$|S| > 2$$.
For this state, violation requires $$p > \frac{1}{\sqrt{2}} \approx 0.707$$.

**(b)** The state can be entangled for smaller $$p$$ (e.g., any $$p > 0$$).
So CHSH violation is sufficient but not necessary for entanglement.

---

## Section C: PPT Criterion

### Solution 15

$$\rho = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

Partial transpose swaps indices 2 and 4 in $$\rho_{ij,kl}$$:

$$\rho^{T_B} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Eigenvalues: $$\{1/2, 1/2, 1/2, -1/2\}$$

Negative eigenvalue $$\Rightarrow$$ **entangled**.

---

### Solution 16

$$\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)\frac{I}{4}$$

After partial transpose, eigenvalues are:
$$\{(1+p)/4, (1+p)/4, (1+p)/4, (1-3p)/4\}$$

PPT requires $$(1-3p)/4 \geq 0$$, i.e., $$p \leq 1/3$$.

For $$p > 1/3$$: entangled (NPT).

---

### Solution 17

$$\rho = \frac{1}{3}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**(a)** $$\text{Tr}(\rho) = \frac{1}{3}(1 + 1 + 1 + 0) = 1$$ ✓

**(b)** Partial transpose:
$$\rho^{T_B} = \frac{1}{3}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix}$$

**(c)** Eigenvalues of $$\rho^{T_B}$$: $$\{2/3, 1/3, 1/3, -1/3\}$$

Negative eigenvalue $$\Rightarrow$$ **entangled**.

---

### Solution 18

For $$2 \times 2$$ and $$2 \times 3$$: Any entangled state has an entanglement witness that detects it via the partial transpose. This is related to the fact that all positive maps in these dimensions can be decomposed as sums of completely positive maps and transposition.

For higher dimensions: There exist **bound entangled states** that are PPT but not separable. The partial transpose doesn't capture all entanglement.

---

## Section D: Entanglement Measures

### Solution 19

**(a) $$|00\rangle$$:** $$E = 0$$ (product state)

**(b) $$|\Phi^+\rangle$$:**
$$\rho_A = I/2$$, eigenvalues $$1/2, 1/2$$
$$E = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = 1$$ ebit

**(c) $$|\psi\rangle = \sqrt{0.9}|00\rangle + \sqrt{0.1}|11\rangle$$:**
$$E = -0.9\log_2(0.9) - 0.1\log_2(0.1)$$
$$= 0.9 \times 0.152 + 0.1 \times 3.322 = 0.137 + 0.332 = 0.469$$ ebits

---

### Solution 20

Coefficient matrix: $$C = \begin{pmatrix} \alpha & \beta \\ \gamma & \delta \end{pmatrix}$$

$$\rho_A = CC^\dagger = \begin{pmatrix} |\alpha|^2 + |\beta|^2 & \alpha\gamma^* + \beta\delta^* \\ \gamma\alpha^* + \delta\beta^* & |\gamma|^2 + |\delta|^2 \end{pmatrix}$$

Eigenvalues: $$\lambda_\pm = \frac{1 \pm \sqrt{1 - 4|\alpha\delta - \beta\gamma|^2}}{2}$$

(using $$|\alpha|^2 + |\beta|^2 + |\gamma|^2 + |\delta|^2 = 1$$)

$$E = -\lambda_+\log_2\lambda_+ - \lambda_-\log_2\lambda_-$$

---

### Solution 21

**(a) $$|00\rangle$$:**
$$|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)|00\rangle = |11\rangle$$
$$C = |\langle 00|11\rangle| = 0$$

**(b) $$|\Phi^+\rangle$$:**
$$|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)(|00\rangle + |11\rangle)/\sqrt{2} = (|11\rangle + |00\rangle)/\sqrt{2}$$
$$C = |\langle\Phi^+|\Phi^+\rangle| = 1$$

**(c) $$|\psi\rangle = \frac{\sqrt{3}}{2}|00\rangle + \frac{1}{2}|11\rangle$$:**
$$|\tilde{\psi}\rangle = \frac{\sqrt{3}}{2}|11\rangle + \frac{1}{2}|00\rangle$$
$$C = |\frac{\sqrt{3}}{2} \cdot \frac{1}{2} + \frac{1}{2} \cdot \frac{\sqrt{3}}{2}| = \frac{\sqrt{3}}{2}$$

---

### Solution 22

For Werner state $$\rho_W = p|\Psi^-\rangle\langle\Psi^-| + (1-p)\frac{I}{4}$$:

$$\tilde{\rho}_W = (\sigma_y \otimes \sigma_y)\rho_W^*(\sigma_y \otimes \sigma_y)$$

After calculation:
$$C = \max(0, \frac{3p-1}{2})$$

$$C > 0$$ when $$p > 1/3$$.

---

### Solution 23

**(a) Bell state:**
From Solution 15: eigenvalues of $$\rho^{T_B}$$ are $$\{1/2, 1/2, 1/2, -1/2\}$$

$$\mathcal{N} = |-1/2| = 1/2$$
$$E_N = \log_2(1 + 2 \times 1/2) = \log_2(2) = 1$$

**(b) Werner at $$p = 0.5$$:**
Eigenvalue $$(1-3p)/4 = -0.25/4 = -1/8$$
$$\mathcal{N} = 1/8$$
$$E_N = \log_2(1 + 1/4) = \log_2(1.25) \approx 0.32$$

---

### Solution 24

For $$|\psi\rangle = a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle$$:

$$(\sigma_y \otimes \sigma_y)|\psi^*\rangle = -a^*|11\rangle + b^*|10\rangle + c^*|01\rangle - d^*|00\rangle$$

$$\langle\psi|\tilde{\psi}\rangle = -ad^* + bc^* + cb^* - da^* = 2(bc - ad)^*$$

$$C = |2(ad - bc)|$$

(Note: this assumes real coefficients; for complex, it's $$2|ad - bc|$$)

---

### Solution 25

The entanglement of formation is:
$$E_F = \min \sum_i p_i E(|\psi_i\rangle)$$

over decompositions $$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$.

Wootters showed this equals:
$$E_F = h\left(\frac{1 + \sqrt{1-C^2}}{2}\right)$$

Proof uses the convexity of the roof construction and properties of the spin-flip operation.

---

### Solution 26

For $$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)|00\rangle\langle 00|$$:

**Concurrence:** $$C = \max(0, p - \sqrt{(1-p)p})$$ (from Wootters formula)

**Negativity:** Computed from partial transpose eigenvalues.

Both are zero for small $$p$$ and increase with $$p$$, but they're not proportional.

---

## Section E: Advanced Topics

### Solution 27

**(a) GHZ state:**
$$\rho_{AB} = \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$$ (classical mixture)
$$C_{A|B} = 0$$

$$\rho_{AC}$$ similarly classical, $$C_{A|C} = 0$$

$$C_{A|BC}$$: The state $$|\text{GHZ}\rangle$$ from A's perspective is maximally entangled with BC as a unit.
$$C_{A|BC} = 1$$

Check: $$0 + 0 \leq 1$$ ✓

**(b) W state:**
$$|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$

After tracing out C: $$\rho_{AB}$$ is entangled with $$C_{A|B}^2 = 2/3$$ (can be computed).

By symmetry: $$C_{A|C}^2 = 2/3$$

$$C_{A|BC}^2 = 4/3$$ (from the pure state)

Check: $$2/3 + 2/3 = 4/3$$ - saturates the inequality!

---

### Solution 28

**(a)** By the Hahn-Banach separation theorem, any point outside a convex set can be separated by a hyperplane. The set of separable states is convex. An entangled state lies outside, so there exists a separating hyperplane, which defines the witness $$W$$.

**(b)** For $$|\Phi^+\rangle$$:
$$W = \frac{1}{2}I - |\Phi^+\rangle\langle\Phi^+|$$

For separable states: $$\text{Tr}(W\rho_{sep}) \geq 0$$
For $$|\Phi^+\rangle$$: $$\text{Tr}(W|\Phi^+\rangle\langle\Phi^+|) = 1/2 - 1 = -1/2 < 0$$

---

*Solutions complete. Review challenging problems before the oral examination.*
