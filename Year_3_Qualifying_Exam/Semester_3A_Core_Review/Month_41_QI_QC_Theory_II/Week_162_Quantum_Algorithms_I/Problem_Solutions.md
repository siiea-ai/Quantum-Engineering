# Week 162: Quantum Algorithms I - Complete Problem Solutions

---

## Section A: Deutsch-Jozsa and Bernstein-Vazirani Solutions

### Solution 1: Deutsch's Algorithm (Single Qubit)

**Part a) All four functions:**

| $x$ | $f_1$ (const 0) | $f_2$ (const 1) | $f_3$ (balanced) | $f_4$ (balanced) |
|-----|-----------------|-----------------|------------------|------------------|
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0 | 1 | 1 | 0 |

- $f_1, f_2$: Constant
- $f_3, f_4$: Balanced

**Part b) Circuit:**

```
|0⟩ --H--[Oracle]--H-- Measure
|1⟩ --H--[      ]-----
```

**Part c) Final states:**

Initial: $|0\rangle|1\rangle$

After H: $|+\rangle|-\rangle = \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle - |1\rangle)$

After oracle: $\frac{1}{2}\sum_x (-1)^{f(x)}|x\rangle|{-}\rangle$

For $f_1$: $|+\rangle|-\rangle$, after final H: $|0\rangle|-\rangle$
For $f_2$: $-|+\rangle|-\rangle$, after final H: $-|0\rangle|-\rangle$
For $f_3$: $\frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)|-\rangle = |-\rangle|-\rangle$, after final H: $|1\rangle|-\rangle$
For $f_4$: $\frac{1}{\sqrt{2}}(-|0\rangle + |1\rangle)|-\rangle$, after final H: $-|1\rangle|-\rangle$

**Part d) Verification:**
- Constant: First qubit is $|0\rangle$ $\checkmark$
- Balanced: First qubit is $|1\rangle$ $\checkmark$

---

### Solution 2: Deutsch-Jozsa State Analysis

**Part a) State after Hadamards:**

$$H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_{x \in \{0,1\}^n}|x\rangle$$

$$H|1\rangle = |{-}\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

Combined:
$$\boxed{|\psi_1\rangle = \frac{1}{\sqrt{2^{n+1}}}\sum_{x}|x\rangle(|0\rangle - |1\rangle)}$$

**Part b) After oracle:**

Oracle: $O_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$

$$O_f|x\rangle|{-}\rangle = O_f|x\rangle\frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = \frac{1}{\sqrt{2}}|x\rangle(|f(x)\rangle - |1 \oplus f(x)\rangle)$$

If $f(x) = 0$: $|0\rangle - |1\rangle$
If $f(x) = 1$: $|1\rangle - |0\rangle = -(|0\rangle - |1\rangle)$

So: $O_f|x\rangle|{-}\rangle = (-1)^{f(x)}|x\rangle|{-}\rangle$

$$\boxed{|\psi_2\rangle = \frac{1}{\sqrt{2^n}}\sum_x (-1)^{f(x)}|x\rangle|{-}\rangle}$$

**Part c) Amplitude of $|0\rangle^{\otimes n}$:**

After final Hadamards:
$$H^{\otimes n}\left[\frac{1}{\sqrt{2^n}}\sum_x (-1)^{f(x)}|x\rangle\right] = \sum_z \left[\frac{1}{2^n}\sum_x (-1)^{f(x)+x \cdot z}\right]|z\rangle$$

Amplitude of $|0\rangle^{\otimes n}$ (where $z = 0$):
$$\boxed{\alpha_0 = \frac{1}{2^n}\sum_x (-1)^{f(x)}}$$

For constant $f$: $\alpha_0 = \pm 1$
For balanced $f$: $\alpha_0 = 0$ (equal positive and negative terms)

---

### Solution 3: Specific Oracle Implementation

**Part a) Classification:**

$f(x_1, x_2, x_3) = x_1 \oplus x_2$

Counting outputs:
- $f(000) = 0$, $f(001) = 0$, $f(010) = 1$, $f(011) = 1$
- $f(100) = 1$, $f(101) = 1$, $f(110) = 0$, $f(111) = 0$

Four 0s, four 1s: **Balanced** $\checkmark$

**Part b) Oracle circuit:**

$$O_f:|x_1, x_2, x_3, y\rangle \to |x_1, x_2, x_3, y \oplus x_1 \oplus x_2\rangle$$

Circuit:
```
|x₁⟩ -------●-------
|x₂⟩ -----------●---
|x₃⟩ ---------------
|y⟩  ----⊕-----⊕----
```
(Two CNOT gates: $x_1$ controls first, $x_2$ controls second)

**Part c) Algorithm trace:**

Initial: $|000\rangle|1\rangle$

After H: $\frac{1}{\sqrt{8}}\sum_{x_1,x_2,x_3}|x_1 x_2 x_3\rangle|{-}\rangle$

After oracle: $\frac{1}{\sqrt{8}}\sum_{x_1,x_2,x_3}(-1)^{x_1 \oplus x_2}|x_1 x_2 x_3\rangle|{-}\rangle$

Final state first register:
$$\frac{1}{\sqrt{8}}[(1-1)(|000\rangle + |001\rangle) + (-1+1)(|010\rangle + |011\rangle) + ...]$$

After H on first register: $|110\rangle$ (non-zero due to balanced structure)

**Part d) Verification:**

Measure NOT $|000\rangle$, correctly identifying balanced function. $\checkmark$

---

### Solution 4: Bernstein-Vazirani with $s = 1011$

**Part a) Function values:**

$f(x) = 1011 \cdot x = x_0 \oplus x_2 \oplus x_3$ (using $x = x_3 x_2 x_1 x_0$)

| $x$ | $f(x)$ | | $x$ | $f(x)$ |
|-----|--------|---|-----|--------|
| 0000 | 0 | | 1000 | 1 |
| 0001 | 1 | | 1001 | 0 |
| 0010 | 0 | | 1010 | 1 |
| 0011 | 1 | | 1011 | 0 |
| 0100 | 1 | | 1100 | 0 |
| 0101 | 0 | | 1101 | 1 |
| 0110 | 1 | | 1110 | 0 |
| 0111 | 0 | | 1111 | 1 |

**Part b) Oracle circuit:**

```
|x₃⟩ ----●---------
|x₂⟩ --------●-----
|x₁⟩ --------------
|x₀⟩ ------------●-
|y⟩  --⊕----⊕----⊕-
```

**Part c) Algorithm output:**

After H-Oracle-H, the state is exactly $|1011\rangle$ because:
$$H^{\otimes 4}\left[\frac{1}{4}\sum_x (-1)^{s \cdot x}|x\rangle\right] = |s\rangle = |1011\rangle$$

Probability 1. $\checkmark$

**Part d) Classical approach:**

Query with $e_0 = 0001$: $f(e_0) = 1 = s_0$
Query with $e_1 = 0010$: $f(e_1) = 0 = s_1$
Query with $e_2 = 0100$: $f(e_2) = 1 = s_2$
Query with $e_3 = 1000$: $f(e_3) = 1 = s_3$

Requires 4 queries (in general, $n$ queries).

---

## Section B: Simon's Algorithm Solutions

### Solution 9: Simon's Algorithm Basics

**Part a) Example function with $s = 101$:**

| $x$ | $f(x)$ |
|-----|--------|
| 000 | 001 |
| 101 | 001 |
| 001 | 010 |
| 100 | 010 |
| 010 | 011 |
| 111 | 011 |
| 011 | 100 |
| 110 | 100 |

**Part b) Pairs mapping to same output:**

$(000, 101)$, $(001, 100)$, $(010, 111)$, $(011, 110)$

**Part c) Linear equations:**

Measurements give $y$ with $y \cdot s = y \cdot 101 = y_0 \oplus y_2 = 0$

Possible $y$: $\{000, 011, 100, 111\}$

Each measurement gives one equation of form $y_0 + y_2 = 0 \pmod 2$

**Part d) Number of iterations:**

Need 2 independent equations to uniquely determine $s$ (up to 0).
Expected: $O(n) = O(3)$ iterations for high probability.

---

### Solution 10: State Analysis in Simon's Algorithm

**Part a) State after oracle:**

$$|\psi\rangle = \frac{1}{\sqrt{2^n}}\sum_x |x\rangle|f(x)\rangle$$

**Part b) After measurement of second register:**

Suppose $f(x_0)$ is observed. The first register collapses to:
$$|\psi_1\rangle = \frac{1}{\sqrt{2}}(|x_0\rangle + |x_0 \oplus s\rangle)$$

**Part c) After Hadamard:**

$$H^{\otimes n}|\psi_1\rangle = \frac{1}{\sqrt{2}}\left[\frac{1}{\sqrt{2^n}}\sum_z (-1)^{x_0 \cdot z}|z\rangle + \frac{1}{\sqrt{2^n}}\sum_z (-1)^{(x_0 \oplus s) \cdot z}|z\rangle\right]$$

$$= \frac{1}{\sqrt{2^{n+1}}}\sum_z (-1)^{x_0 \cdot z}(1 + (-1)^{s \cdot z})|z\rangle$$

This is non-zero only when $s \cdot z = 0$, giving amplitude $\frac{1}{\sqrt{2^{n-1}}}(-1)^{x_0 \cdot z}$.

**Part d) Uniform distribution:**

All $z$ with $s \cdot z = 0$ have equal probability $1/2^{n-1}$.
There are $2^{n-1}$ such strings, so distribution is uniform over valid $z$.

---

### Solution 11: Linear Algebra in Simon's

**Part a) Probability of full rank:**

After $m$ measurements, vectors $y_1, \ldots, y_m$ are random from subspace $S = \{y : y \cdot s = 0\}$.

$\dim(S) = n-1$.

Probability that $m$ random vectors span $S$:
$$P \geq 1 - 2^{n-1-m}$$

For $m = n$: $P \geq 1 - 2^{-1} = 1/2$
For $m = 2n$: $P \geq 1 - 2^{-n}$

**Part b) Finding $s$:**

Solve the system $Y \cdot s = 0$ where $Y$ is the matrix of measurements.
The solution space is 1-dimensional (spanned by $s$).
Choose the non-zero solution.

**Part c) Redundant measurement:**

Discard it (or keep for error checking) and continue sampling.

**Part d) Expected iterations:**

$$E[\text{iterations}] = \sum_{k=0}^{n-2}\frac{2^{n-1}}{2^{n-1} - 2^k} \approx n + O(1)$$

Approximately $n$ iterations expected.

---

### Solution 12: Simon vs. Classical

**Part a) Classical lower bound:**

Any classical algorithm sees function values $f(x_1), f(x_2), \ldots$

To find a collision ($f(x_i) = f(x_j)$ with $x_i \neq x_j$), by birthday paradox need $\Omega(\sqrt{2^n})$ queries.

Without a collision, cannot distinguish $s = 0$ from $s \neq 0$.

**Part b) Birthday paradox:**

With $k$ random elements from set of size $N$, probability of collision is approximately $1 - e^{-k^2/(2N)}$.

For $N = 2^{n-1}$ distinct function values, need $k \approx 2^{n/2}$ for collision.

**Part c) Quantum difference:**

Quantum algorithm doesn't find collisions!
Instead, it uses interference to directly sample from the orthogonal subspace to $s$.
No collision finding is needed.

---

## Section C: Quantum Fourier Transform Solutions

### Solution 15: QFT Computation ($n = 2$)

**Part a) QFT matrix:**

$N = 4$, $\omega = e^{2\pi i/4} = i$

$$\text{QFT} = \frac{1}{2}\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & i & -1 & -i \\ 1 & -1 & 1 & -1 \\ 1 & -i & -1 & i \end{pmatrix}$$

**Part b) QFT on basis states:**

$$\text{QFT}|0\rangle = \frac{1}{2}(|0\rangle + |1\rangle + |2\rangle + |3\rangle)$$

$$\text{QFT}|1\rangle = \frac{1}{2}(|0\rangle + i|1\rangle - |2\rangle - i|3\rangle)$$

$$\text{QFT}|2\rangle = \frac{1}{2}(|0\rangle - |1\rangle + |2\rangle - |3\rangle)$$

$$\text{QFT}|3\rangle = \frac{1}{2}(|0\rangle - i|1\rangle - |2\rangle + i|3\rangle)$$

**Part c) Unitarity:**

$\text{QFT}^\dagger \text{QFT} = I$ (verify by matrix multiplication or note orthonormal rows)

**Part d) Inverse:**

$$\text{QFT}^{-1} = \text{QFT}^\dagger = \frac{1}{2}\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & -i & -1 & i \\ 1 & -1 & 1 & -1 \\ 1 & i & -1 & -i \end{pmatrix}$$

---

### Solution 16: QFT Circuit Derivation ($n = 3$)

**Part a) Product representation:**

For $|j\rangle = |j_1 j_2 j_3\rangle$ (binary):

$$\text{QFT}|j_1 j_2 j_3\rangle = \frac{1}{\sqrt{8}}\left(|0\rangle + e^{2\pi i(0.j_3)}|1\rangle\right) \otimes \left(|0\rangle + e^{2\pi i(0.j_2 j_3)}|1\rangle\right) \otimes \left(|0\rangle + e^{2\pi i(0.j_1 j_2 j_3)}|1\rangle\right)$$

**Part b) Gates needed:**

- First qubit: $H$ gives $(|0\rangle + e^{2\pi i(0.j_1)}|1\rangle)/\sqrt{2}$
  - Then controlled-$R_2$ from $j_2$ adds phase $0.0j_2$
  - Then controlled-$R_3$ from $j_3$ adds phase $0.00j_3$

- Second qubit: $H$, then controlled-$R_2$ from $j_3$

- Third qubit: $H$ only

**Part c) Circuit (before reversal):**

```
|j₁⟩ --H--R₂--R₃---------------
|j₂⟩ -----●------H--R₂---------
|j₃⟩ ----------●-----●----H----
```

Then SWAP to reverse order.

**Part d) Gate count:**

- H gates: 3
- Controlled-R gates: 3 (R₂ twice, R₃ once)
- SWAP: 1 (for bit reversal)

Total: 7 gates

---

### Solution 17: Controlled Phase Gates

**Part a) $R_2$ in terms of $T$:**

$$R_2 = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/2} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = S = T^2$$

**Part b) Controlled-$R_3$ decomposition:**

$R_3 = e^{i\pi/4} = T$, so controlled-$R_3$ = controlled-$T$.

Decomposition using Clifford+T:
$$\text{C-}T = (I \otimes T^{1/2}) \cdot \text{CNOT} \cdot (I \otimes T^{-1/2}) \cdot \text{CNOT} \cdot (I \otimes T^{1/2})$$

Wait, $T^{1/2}$ is not in Clifford+T. Need magic state approach.

**Part c) T-count for controlled-$R_k$:**

For exact implementation: Not possible with finite T gates (irrational phases for $k \geq 4$).

For approximate: $O(\log(1/\epsilon))$ T gates per controlled-$R_k$.

**Part d) Total T-count for n-qubit QFT:**

Approximately $O(n^2 \log(1/\epsilon))$ for approximate QFT to precision $\epsilon$.

---

## Section D: Phase Estimation Solutions

### Solution 21: Phase Estimation Basics

**Part a) Phase $\theta$:**

$U|0\rangle = e^{-i\pi/8}|0\rangle$

Writing $e^{-i\pi/8} = e^{2\pi i \theta}$:
$$\theta = -\frac{1}{16} = \frac{15}{16} \pmod 1$$

$\boxed{\theta = 15/16 = 0.1111_2}$

**Part b) Ancilla qubits:**

For 4-bit precision: 4 ancilla qubits.

$15/16$ is exactly representable in 4 bits as $1111_2$.

**Part c) Algorithm trace:**

1. Initial: $|0000\rangle|0\rangle$
2. Hadamard: $|++++\rangle|0\rangle$
3. Controlled-$U^{2^j}$:
   - $U^1 = R_z(\pi/4)$, $U^2 = R_z(\pi/2)$, etc.
   - Phases accumulated: $e^{2\pi i \cdot 15k/16}$ for $k$-th ancilla
4. State: $\frac{1}{4}\sum_{k=0}^{15} e^{2\pi i \cdot 15k/16}|k\rangle|0\rangle$
5. Inverse QFT: $|1111\rangle|0\rangle = |15\rangle|0\rangle$

**Part d) Measurement outcomes:**

Measure $|1111\rangle = 15$ with probability 1 (exact phase).

---

### Solution 24: Precision Analysis

**Part a) Exact phase case:**

If $\theta = 0.j_1 j_2 \ldots j_t$ exactly, then after controlled-U operations:
$$|\psi\rangle = \frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1} e^{2\pi ik\theta}|k\rangle = \text{QFT}|j_1 j_2 \ldots j_t\rangle$$

Inverse QFT gives $|j_1 j_2 \ldots j_t\rangle$ with probability 1.

**Part b) Non-exact phase:**

Let $2^t \theta = m + \delta$ with $|\delta| < 1/2$.

Amplitude of measuring $j$:
$$\alpha_j = \frac{1}{2^t}\sum_{k=0}^{2^t-1} e^{2\pi i(2^t\theta - j)k/2^t} = \frac{1 - e^{2\pi i(m+\delta-j)}}{2^t(1 - e^{2\pi i(m+\delta-j)/2^t})}$$

For $j = m$:
$$|\alpha_m|^2 = \frac{\sin^2(\pi\delta)}{2^{2t}\sin^2(\pi\delta/2^t)} \geq \frac{4}{\pi^2}$$

**Part c) Probability bound:**

$$P(|j - 2^t\theta| \leq 1) = P(j \in \{m-1, m, m+1\}) \geq \frac{8}{\pi^2} \approx 0.81$$

**Part d) Extra ancilla qubits:**

With $t + \log_2(2 + 1/(2\epsilon))$ qubits:
$$P(\text{error} > \epsilon) < \epsilon$$

---

### Solution 27: Complete Algorithm Design

**Part a) Eigenvalues of $H$:**

$$H = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 2 \end{pmatrix}$$

Characteristic polynomial: $(1-\lambda)(2-\lambda) - 0.25 = \lambda^2 - 3\lambda + 1.75 = 0$

$$\lambda = \frac{3 \pm \sqrt{9 - 7}}{2} = \frac{3 \pm \sqrt{2}}{2}$$

$$\lambda_0 = \frac{3 - \sqrt{2}}{2} \approx 0.793, \quad \lambda_1 = \frac{3 + \sqrt{2}}{2} \approx 2.207$$

Eigenvectors: Standard calculation gives normalized vectors.

**Part b) Phase estimation setup:**

$U = e^{-iH}$

Eigenvalues of $U$: $e^{-i\lambda_0} \approx e^{-0.793i}$ and $e^{-i\lambda_1} \approx e^{-2.207i}$

Phases: $\theta_0 = -\lambda_0/(2\pi) \pmod 1$, $\theta_1 = -\lambda_1/(2\pi) \pmod 1$

**Part c) Ancilla count:**

For 3-bit precision ($\epsilon = 1/8$ in phase):
- Energy precision: $2\pi/8 \approx 0.79$
- Need 3 ancilla qubits minimum

For better energy resolution, increase $t$ or scale $H$.

**Part d) Expected outcomes:**

If eigenstate $|0\rangle$ of $H$ is prepared (not exact eigenstate), measurements will give superposition outcomes corresponding to projections onto eigenstates.

---

### Solution 28: VQE vs. QPE Comparison

**Part a) Key differences:**

| Aspect | VQE | QPE |
|--------|-----|-----|
| Approach | Variational optimization | Direct eigenvalue readout |
| Classical processing | Optimization loop | Minimal (phase interpretation) |
| Quantum resources | Shallow circuits | Deep circuits (controlled-U) |
| Precision | Limited by ansatz | Scales with ancilla count |

**Part b) Resource comparison:**

- **Qubits:** VQE uses fewer (no ancilla for phase encoding)
- **Depth:** QPE much deeper (controlled-$U^{2^k}$ operations)
- **Measurements:** VQE needs many more (for gradient estimation)

**Part c) NISQ suitability:**

VQE is more suitable because:
1. Shallower circuits
2. Tolerates some noise through variational optimization
3. No coherent QFT required

**Part d) VQE to initialize QPE:**

Yes! VQE can find approximate ground state, which improves QPE success probability (good overlap with true eigenstate).

This is a hybrid approach: VQE for state prep, QPE for precision energy.

---

*This completes the solutions for Week 162. For oral practice, see Oral_Practice.md.*
