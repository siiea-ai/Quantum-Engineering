# Week 162: Quantum Algorithms I - Comprehensive Review Guide

## Foundational Algorithms: From Deutsch-Jozsa to Phase Estimation

---

## 1. The Oracle Model of Computation

### 1.1 Black-Box Functions

In the oracle (or query) model, we have access to a function $f$ only through a "black box" that computes $f(x)$ on input $x$. We measure algorithm efficiency by the number of queries (oracle calls) needed.

**Quantum Oracle Convention:**
$$O_f |x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$$

This is reversible and can be implemented as a unitary operation.

**Phase Oracle (Alternative):**
$$\tilde{O}_f |x\rangle = (-1)^{f(x)}|x\rangle$$

For binary $f$, the phase oracle relates to the standard oracle: apply to $|x\rangle|{-}\rangle$.

### 1.2 Query Complexity Classes

**Classical Query Complexity:** Minimum queries needed by any deterministic/randomized algorithm.

**Quantum Query Complexity:** Minimum queries needed by any quantum algorithm.

**Known Separations:**
- Polynomial: Grover's algorithm ($\sqrt{N}$ vs $N$)
- Exponential: Simon's algorithm, period finding

---

## 2. The Deutsch-Jozsa Algorithm

### 2.1 Problem Statement

**Promise:** $f:\{0,1\}^n \to \{0,1\}$ is either:
- **Constant:** $f(x) = c$ for all $x$ (same output for all inputs)
- **Balanced:** $f(x) = 0$ for exactly half the inputs, $f(x) = 1$ for the other half

**Task:** Determine which case holds.

### 2.2 Classical Complexity

**Deterministic:** Must query $2^{n-1} + 1$ inputs in worst case (need to see more than half)

**Randomized:** High confidence with $O(1)$ queries, but never certain

### 2.3 Quantum Algorithm

**Circuit:**
1. Initialize: $|0\rangle^{\otimes n}|1\rangle$
2. Apply Hadamard to all qubits: $|{+}\rangle^{\otimes n}|{-}\rangle$
3. Apply oracle: $\sum_x |x\rangle|{-} \oplus f(x)\rangle = \sum_x (-1)^{f(x)}|x\rangle|{-}\rangle$
4. Apply Hadamard to first $n$ qubits
5. Measure first $n$ qubits

**Analysis:**

After step 2:
$$|\psi_1\rangle = \frac{1}{\sqrt{2^n}}\sum_{x \in \{0,1\}^n}|x\rangle \cdot |{-}\rangle$$

After step 3 (using phase kickback):
$$|\psi_2\rangle = \frac{1}{\sqrt{2^n}}\sum_{x}(-1)^{f(x)}|x\rangle \cdot |{-}\rangle$$

After step 4:
$$|\psi_3\rangle = \sum_{z} \left[\frac{1}{2^n}\sum_{x}(-1)^{f(x) + x \cdot z}\right]|z\rangle \cdot |{-}\rangle$$

**Key Result:**

The amplitude of $|0\rangle^{\otimes n}$ is:
$$\alpha_0 = \frac{1}{2^n}\sum_{x}(-1)^{f(x)} = \begin{cases} \pm 1 & \text{if } f \text{ is constant} \\ 0 & \text{if } f \text{ is balanced} \end{cases}$$

**Conclusion:** Measure $|0\rangle^{\otimes n}$ with probability 1 if constant, probability 0 if balanced.

### 2.4 The Speedup

| Algorithm | Query Complexity | Type |
|-----------|------------------|------|
| Classical deterministic | $2^{n-1} + 1$ | Worst-case |
| Classical randomized | $O(1)$ | Probabilistic |
| Quantum | 1 | Exact |

The quantum algorithm is **exponentially better** than classical deterministic, and provides **certainty** unlike classical randomized.

---

## 3. The Bernstein-Vazirani Algorithm

### 3.1 Problem Statement

**Given:** Oracle for $f(x) = s \cdot x = \bigoplus_{i} s_i x_i$ (mod 2 inner product)

**Find:** The secret string $s \in \{0,1\}^n$

### 3.2 Algorithm

The circuit is identical to Deutsch-Jozsa. After applying H-Oracle-H:

$$|\psi_{\text{final}}\rangle = |s\rangle$$

**Why?** The Hadamard transform of $(-1)^{s \cdot x}$ picks out exactly $|s\rangle$:

$$H^{\otimes n}\left[\frac{1}{\sqrt{2^n}}\sum_x (-1)^{s \cdot x}|x\rangle\right] = |s\rangle$$

### 3.3 Complexity Comparison

- **Classical:** $n$ queries (query with $e_i$ to get $s_i$)
- **Quantum:** 1 query

---

## 4. Simon's Algorithm

### 4.1 Problem Statement

**Given:** Oracle for $f:\{0,1\}^n \to \{0,1\}^n$ where:
$$f(x) = f(y) \Leftrightarrow x \oplus y \in \{0^n, s\}$$

for some unknown $s \in \{0,1\}^n$.

**Find:** The period $s$.

If $s = 0^n$, the function is one-to-one. Otherwise, it's two-to-one with period $s$.

### 4.2 Algorithm

**Single Iteration:**
1. Prepare $|0\rangle^{\otimes n}|0\rangle^{\otimes n}$
2. Apply $H^{\otimes n}$ to first register: $\frac{1}{\sqrt{2^n}}\sum_x |x\rangle|0\rangle$
3. Apply oracle: $\frac{1}{\sqrt{2^n}}\sum_x |x\rangle|f(x)\rangle$
4. Apply $H^{\otimes n}$ to first register
5. Measure first register, get string $y$

**Analysis:**

After oracle, measuring the second register collapses to:
$$\frac{1}{\sqrt{2}}(|x_0\rangle + |x_0 \oplus s\rangle)|f(x_0)\rangle$$

for some $x_0$.

After Hadamard on first register:
$$\frac{1}{\sqrt{2^{n+1}}}\sum_z ((-1)^{x_0 \cdot z} + (-1)^{(x_0 \oplus s) \cdot z})|z\rangle|f(x_0)\rangle$$

This vanishes unless $(-1)^{s \cdot z} = 1$, i.e., $s \cdot z = 0$.

**Key Result:** Each measurement yields a random $y$ with $y \cdot s = 0$.

### 4.3 Classical Post-Processing

After $O(n)$ iterations, we have $n$ equations $y_i \cdot s = 0$.

With high probability, these form a system of rank $n-1$, determining $s$ up to the trivial solution $s = 0$.

Check both solutions by querying the oracle.

### 4.4 Complexity Analysis

| Algorithm | Query Complexity |
|-----------|------------------|
| Classical (best known) | $\Omega(2^{n/2})$ (birthday attack) |
| Quantum | $O(n)$ |

**Exponential separation!** This was the first example (Simon, 1994).

### 4.5 Connection to Shor

Simon's algorithm inspired Shor's factoring algorithm. The key insight: both problems involve finding a hidden period, but over different groups:

- **Simon:** Period in $(\mathbb{Z}_2)^n$ (XOR group)
- **Shor:** Period in $\mathbb{Z}_N$ (addition mod N)

---

## 5. The Quantum Fourier Transform

### 5.1 Definition

For an $n$-qubit system ($N = 2^n$ dimensional), the QFT is:

$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle$$

In matrix form:
$$\text{QFT} = \frac{1}{\sqrt{N}}\begin{pmatrix} 1 & 1 & 1 & \cdots & 1 \\ 1 & \omega & \omega^2 & \cdots & \omega^{N-1} \\ 1 & \omega^2 & \omega^4 & \cdots & \omega^{2(N-1)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & \omega^{N-1} & \omega^{2(N-1)} & \cdots & \omega^{(N-1)^2} \end{pmatrix}$$

where $\omega = e^{2\pi i/N}$.

### 5.2 Product Representation

The QFT can be written as a tensor product:

$$\text{QFT}|j_1 j_2 \cdots j_n\rangle = \frac{1}{\sqrt{2^n}}\bigotimes_{l=1}^{n}\left(|0\rangle + e^{2\pi i \cdot 0.j_{n-l+1}\cdots j_n}|1\rangle\right)$$

where $0.j_k j_{k+1} \cdots j_n = \sum_{m=k}^{n} j_m 2^{-(m-k+1)}$ is the binary fraction.

### 5.3 Circuit Construction

The product representation leads directly to a circuit.

**Key Gates:**
- **Hadamard:** Creates $|0\rangle + e^{2\pi i \cdot 0.j_k}|1\rangle$ for the leading bit
- **Controlled-$R_k$:** Adds phase $e^{2\pi i/2^k}$ controlled by subsequent bits

$$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^k} \end{pmatrix}$$

**Algorithm (for n qubits):**
1. Apply H to qubit 1
2. Apply controlled-$R_2$ (control: qubit 2, target: qubit 1)
3. Apply controlled-$R_3$ (control: qubit 3, target: qubit 1)
4. ...continue until controlled-$R_n$
5. Apply H to qubit 2, then controlled-$R_2$ through $R_{n-1}$
6. ...repeat pattern
7. SWAP qubits to reverse bit order

### 5.4 Complexity Analysis

**Gate Count:**
- Hadamards: $n$
- Controlled-$R_k$ gates: $\frac{n(n-1)}{2}$
- SWAPs: $\lfloor n/2 \rfloor$

**Total:** $O(n^2)$ gates

**Compare to Classical FFT:** $O(N \log N) = O(n \cdot 2^n)$ operations

The QFT is **exponentially faster** in terms of input size!

### 5.5 Approximate QFT

**Key Observation:** Phases $e^{2\pi i/2^k}$ for large $k$ are nearly 1.

**Approximation:** Omit controlled-$R_k$ gates for $k > m$.

**Result:** $O(n \cdot m)$ gates with error $O(n/2^m)$.

For fixed precision: $O(n \log n)$ gates suffice.

---

## 6. Phase Estimation Algorithm

### 6.1 Problem Statement

**Given:**
- A unitary $U$ with eigenstate $|u\rangle$ and eigenvalue $e^{2\pi i\theta}$
- Ability to perform controlled-$U^{2^j}$ operations
- A copy of $|u\rangle$

**Find:** The phase $\theta \in [0, 1)$ to $t$ bits of precision.

### 6.2 Algorithm Structure

**Registers:**
- $t$ ancilla qubits (control register)
- Eigenstate register containing $|u\rangle$

**Circuit:**
1. Initialize ancilla: $|0\rangle^{\otimes t}|u\rangle$
2. Apply $H^{\otimes t}$ to ancilla: $\frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1}|k\rangle|u\rangle$
3. Apply controlled-$U^{2^j}$ for each ancilla qubit $j$:
   $$\frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1}e^{2\pi ik\theta}|k\rangle|u\rangle$$
4. Apply inverse QFT to ancilla
5. Measure ancilla

### 6.3 Analysis

After controlled-$U$ operations, the ancilla state is:
$$\frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1}e^{2\pi ik\theta}|k\rangle$$

This is precisely the QFT of $|\tilde{\theta}\rangle$ where $\tilde{\theta} = 2^t \theta$ (rounded).

Applying inverse QFT:
$$\text{QFT}^{-1}\left[\frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1}e^{2\pi ik\theta}|k\rangle\right] \approx |2^t\theta\rangle$$

### 6.4 Precision and Success Probability

**Exact Case:** If $\theta = m/2^t$ for integer $m$, measurement gives $m$ with probability 1.

**General Case:** If $2^t\theta = m + \delta$ where $|\delta| < 1/2$:

$$P(\text{measure } m) \geq \frac{4}{\pi^2} \approx 0.405$$

**Higher Success:** Use more ancilla qubits. With $t + \log(2 + 1/(2\epsilon))$ ancillas:
$$P(\text{error} < \epsilon) \geq 1 - \epsilon$$

### 6.5 Controlled-$U^{2^j}$ Implementation

**Option 1:** Repeated squaring - apply $U$ a total of $2^j$ times.
- Cost: $O(2^j)$ per controlled operation
- Total: $O(2^t)$ applications of $U$

**Option 2:** If $U = e^{-iHt}$, then $U^{2^j} = e^{-iH\cdot 2^j t}$.
- May allow efficient implementation through Hamiltonian simulation

### 6.6 Applications

1. **Factoring (Shor):** Find period $r$ where $a^r \equiv 1 \pmod{N}$
2. **Quantum Chemistry:** Estimate ground state energy
3. **Eigenvalue Problems:** General spectral analysis
4. **Cryptography:** Breaking RSA, Diffie-Hellman

---

## 7. The Hidden Subgroup Problem Framework

### 7.1 General Formulation

**Given:** A group $G$, a subgroup $H \leq G$, and a function $f: G \to S$ that is:
- Constant on cosets of $H$
- Distinct on different cosets

**Find:** A generating set for $H$.

### 7.2 Algorithm Mapping

| Problem | Group $G$ | Hidden Subgroup $H$ |
|---------|-----------|---------------------|
| Deutsch-Jozsa | $(\mathbb{Z}_2)^n$ | $\{0^n\}$ or $(\mathbb{Z}_2)^n$ |
| Simon | $(\mathbb{Z}_2)^n$ | $\{0^n, s\}$ |
| Shor (period finding) | $\mathbb{Z}$ | $r\mathbb{Z}$ |
| Discrete log | $\mathbb{Z}_p^* \times \mathbb{Z}_{p-1}$ | Specific subgroup |

### 7.3 Quantum Algorithm Structure

For Abelian groups, the general algorithm is:
1. Create uniform superposition over $G$
2. Apply oracle to compute $f$ in a register
3. Apply QFT over $G$ (generalized Fourier transform)
4. Measure and classically post-process

**Abelian HSP:** Solved efficiently for all Abelian groups!

**Non-Abelian HSP:** Much harder. Graph isomorphism is a famous open case.

---

## 8. Summary: Key Results for Qualifying Exams

### Essential Algorithms

| Algorithm | Problem | Quantum Queries | Classical Queries |
|-----------|---------|-----------------|-------------------|
| Deutsch-Jozsa | Constant vs. balanced | 1 | $2^{n-1}+1$ |
| Bernstein-Vazirani | Find secret string | 1 | $n$ |
| Simon | Find period $s$ | $O(n)$ | $\Omega(2^{n/2})$ |
| QFT | Fourier transform | $O(n^2)$ gates | $O(n 2^n)$ ops |
| Phase Estimation | Estimate eigenvalue | $O(2^t)$ | N/A |

### Key Formulas

$$\boxed{\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}e^{2\pi ijk/N}|k\rangle}$$

$$\boxed{\text{Deutsch-Jozsa: } \langle 0^n|\psi_{\text{final}}\rangle = \frac{1}{2^n}\sum_x (-1)^{f(x)}}$$

$$\boxed{\text{Phase Estimation Precision: } t + \lceil\log(2 + 1/(2\epsilon))\rceil \text{ qubits for error } < \epsilon}$$

### Proof Techniques

1. **Interference:** Constructive/destructive interference in amplitudes
2. **Phase Kickback:** $U|x\rangle|u\rangle = e^{i\phi}|x\rangle|u\rangle$
3. **Fourier Sampling:** Measuring after QFT reveals period information
4. **Linear Algebra:** Post-processing to solve linear systems (Simon)

---

*This review guide covers the foundational quantum algorithms. For problem practice, see Problem_Set.md; for oral exam preparation, see Oral_Practice.md.*
