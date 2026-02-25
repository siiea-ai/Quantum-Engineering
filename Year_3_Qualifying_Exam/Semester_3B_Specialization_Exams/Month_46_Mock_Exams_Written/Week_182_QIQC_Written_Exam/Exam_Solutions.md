# Quantum Information & Computing Exam - Complete Solutions

## Solution 1: Density Matrices and Quantum States

### Part (a): Reduced Density Matrix $\rho_{AB}$ (6 points)

Given:
$$|\psi\rangle = \frac{1}{2}(|000\rangle + |011\rangle + |101\rangle + |110\rangle)$$

The full density matrix is $\rho_{ABC} = |\psi\rangle\langle\psi|$.

To trace out C, we sum over the computational basis of C:
$$\rho_{AB} = \text{Tr}_C(\rho_{ABC}) = \langle 0|_C \rho_{ABC} |0\rangle_C + \langle 1|_C \rho_{ABC} |1\rangle_C$$

Grouping terms by C:
- C = 0: $\frac{1}{2}(|00\rangle + |10\rangle)$
- C = 1: $\frac{1}{2}(|01\rangle + |11\rangle)$

$$\rho_{AB} = \frac{1}{4}(|00\rangle + |10\rangle)(\langle 00| + \langle 10|) + \frac{1}{4}(|01\rangle + |11\rangle)(\langle 01| + \langle 11|)$$

$$= \frac{1}{4}(|00\rangle\langle 00| + |00\rangle\langle 10| + |10\rangle\langle 00| + |10\rangle\langle 10|)$$
$$+ \frac{1}{4}(|01\rangle\langle 01| + |01\rangle\langle 11| + |11\rangle\langle 01| + |11\rangle\langle 11|)$$

In matrix form (basis order: $|00\rangle, |01\rangle, |10\rangle, |11\rangle$):

$$\boxed{\rho_{AB} = \frac{1}{4}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix}}$$

---

### Part (b): Reduced Density Matrix $\rho_A$ (6 points)

Trace out B from $\rho_{AB}$:
$$\rho_A = \text{Tr}_B(\rho_{AB})$$

From the structure of $\rho_{AB}$:
$$\rho_A = \frac{1}{4}[(1 + 1)|0\rangle\langle 0| + (1 + 1)|1\rangle\langle 1|] = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|)$$

$$\boxed{\rho_A = \frac{1}{2}I = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}}$$

Since $\rho_A^2 = \frac{1}{4}I \neq \rho_A$, qubit A is in a **mixed state**.

Alternatively, $\text{Tr}(\rho_A^2) = \frac{1}{2} < 1$, confirming it's mixed.

$$\boxed{\text{Qubit A is in a maximally mixed state (mixed, not pure)}}$$

---

### Part (c): Von Neumann Entropy (7 points)

$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$

Eigenvalues of $\rho_A = \frac{1}{2}I$ are $\lambda_1 = \lambda_2 = \frac{1}{2}$.

$$S(\rho_A) = -\sum_i \lambda_i \log_2 \lambda_i = -2 \times \frac{1}{2}\log_2\frac{1}{2} = -\log_2\frac{1}{2} = 1$$

$$\boxed{S(\rho_A) = 1 \text{ bit}}$$

**Interpretation:** Since $|\psi\rangle$ is a pure state of ABC, the entropy $S(\rho_A) = 1$ equals the entanglement entropy between A and BC. Qubit A is maximally entangled with the BC system.

---

### Part (d): GHZ vs W Classification (6 points)

**GHZ-type states** (like $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$):
- When one qubit is traced out, the remaining two become separable (unentangled)
- All entanglement is destroyed

**W-type states** (like $\frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$):
- When one qubit is traced out, remaining two are still entangled
- Entanglement is more robust

For our state $|\psi\rangle$, checking $\rho_{AB}$:

The eigenvalues of $\rho_{AB}$ are $\{0, 0, \frac{1}{2}, \frac{1}{2}\}$ (rank 2).

The non-zero eigenstates are:
- $\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) = |+\rangle|0\rangle$
- $\frac{1}{\sqrt{2}}(|01\rangle + |11\rangle) = |+\rangle|1\rangle$

These are product states! So $\rho_{AB}$ is separable.

$$\boxed{|\psi\rangle \text{ is GHZ-type: tracing out C leaves AB in a separable mixed state}}$$

Actually, this is a special state: it's the $|GHZ_4\rangle$ state up to local unitaries (a 4-term superposition with specific structure).

---

## Solution 2: Entanglement and Bell Inequalities

### Part (a): Concurrence Calculation (6 points)

For a pure state $|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$:

The concurrence is:
$$C = |2\langle\psi|\tilde{\psi}\rangle| = 2|\alpha\beta| = 2|\cos\theta \sin\theta| = |\sin(2\theta)|$$

where $|\tilde{\psi}\rangle = (Y \otimes Y)|\psi^*\rangle$.

$$\boxed{C(\psi) = |\sin(2\theta)|}$$

**Maximum entanglement:** $C = 1$ when $\sin(2\theta) = \pm 1$, i.e., $\theta = \pi/4$ or $\theta = 3\pi/4$.

$$\boxed{\text{Maximum at } \theta = \pi/4 \text{ (Bell state)}}$$

---

### Part (b): CHSH Parameter Calculation (8 points)

For $\theta = \pi/4$: $|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$

**Alice's observables:**
- $A_1 = Z$ (measures in $\{|0\rangle, |1\rangle\}$)
- $A_2 = X$ (measures in $\{|+\rangle, |-\rangle\}$)

**Bob's observables** (rotated by $\pm\pi/8$):
- $B_1 = \cos(\pi/8)Z + \sin(\pi/8)X$
- $B_2 = \cos(\pi/8)Z - \sin(\pi/8)X$

For the Bell state $|\Phi^+\rangle$:
$$\langle A \otimes B \rangle = \langle \Phi^+|A \otimes B|\Phi^+\rangle$$

Using the correlation formula for Bell states:
$$\langle \vec{a} \cdot \vec{\sigma} \otimes \vec{b} \cdot \vec{\sigma} \rangle = \vec{a} \cdot \vec{b}$$

- $\langle A_1 B_1 \rangle = \langle Z \otimes B_1 \rangle = \cos(\pi/8)$
- $\langle A_1 B_2 \rangle = \langle Z \otimes B_2 \rangle = \cos(\pi/8)$
- $\langle A_2 B_1 \rangle = \langle X \otimes B_1 \rangle = \sin(\pi/8)$
- $\langle A_2 B_2 \rangle = \langle X \otimes B_2 \rangle = -\sin(\pi/8)$

$$S = \cos(\pi/8) + \cos(\pi/8) + \sin(\pi/8) - (-\sin(\pi/8))$$
$$= 2\cos(\pi/8) + 2\sin(\pi/8) = 2\sqrt{2}\sin(\pi/8 + \pi/4) = 2\sqrt{2}\cos(\pi/8)$$

Actually, more directly:
$$S = 2(\cos(\pi/8) + \sin(\pi/8)) = 2\sqrt{2}\sin(\pi/8 + \pi/4) = 2\sqrt{2}\sin(3\pi/8)$$

Since $\sin(3\pi/8) = \cos(\pi/8) \approx 0.924$:
$$S = 2\sqrt{2} \times 0.924 \approx 2.61$$

But the maximum is achieved with optimal angles:
$$\boxed{S = 2\sqrt{2} \approx 2.83}$$

This violates the classical bound of 2.

---

### Part (c): CHSH for Separable States (5 points)

For a separable state $\rho_{sep} = \sum_i p_i \rho_A^{(i)} \otimes \rho_B^{(i)}$:

$$\langle A_j B_k \rangle = \text{Tr}(\rho_{sep}(A_j \otimes B_k)) = \sum_i p_i \langle A_j \rangle_i \langle B_k \rangle_i$$

where $\langle A_j \rangle_i = \text{Tr}(\rho_A^{(i)} A_j)$.

For each product state $\rho_A^{(i)} \otimes \rho_B^{(i)}$, the local expectations satisfy $|\langle A_j \rangle_i| \leq 1$ and $|\langle B_k \rangle_i| \leq 1$.

Define $a_j^{(i)} = \langle A_j \rangle_i$ and $b_k^{(i)} = \langle B_k \rangle_i$.

For each $i$:
$$|a_1^{(i)}b_1^{(i)} + a_1^{(i)}b_2^{(i)} + a_2^{(i)}b_1^{(i)} - a_2^{(i)}b_2^{(i)}|$$
$$= |a_1^{(i)}(b_1^{(i)} + b_2^{(i)}) + a_2^{(i)}(b_1^{(i)} - b_2^{(i)})| \leq 2$$

by the algebraic CHSH bound for classical variables with $|a|, |b| \leq 1$.

Taking the convex combination:
$$|S| = \left|\sum_i p_i (\cdots)\right| \leq \sum_i p_i \times 2 = 2$$

$$\boxed{|S| \leq 2 \text{ for separable states (classical bound)}}$$

---

### Part (d): $\theta = \pi/6$ Case (6 points)

For $|\psi\rangle = \cos(\pi/6)|00\rangle + \sin(\pi/6)|11\rangle = \frac{\sqrt{3}}{2}|00\rangle + \frac{1}{2}|11\rangle$

Concurrence: $C = \sin(2\pi/6) = \sin(\pi/3) = \frac{\sqrt{3}}{2}$

The maximum CHSH violation for a partially entangled state is:
$$S_{max} = 2\sqrt{1 + C^2} = 2\sqrt{1 + 3/4} = 2\sqrt{7/4} = \sqrt{7} \approx 2.65$$

Since $S_{max} = \sqrt{7} \approx 2.65 > 2$:

$$\boxed{\text{Yes, the state can violate CHSH with } S_{max} = \sqrt{7} \approx 2.65}$$

---

## Solution 3: Quantum Gates and Circuits

### Part (a): Universality of CNOT + Single-Qubit (8 points)

**To prove:** CNOT + all single-qubit gates is universal for two-qubit unitaries.

**Strategy:** Any two-qubit unitary can be decomposed into:
1. Single-qubit gates
2. CNOTs

**The key result** (Barenco et al.): Any two-qubit unitary $U$ can be written as:

$$U = (A_1 \otimes B_1) \cdot \text{CNOT} \cdot (A_2 \otimes B_2) \cdot \text{CNOT} \cdot (A_3 \otimes B_3) \cdot \text{CNOT} \cdot (A_4 \otimes B_4)$$

where $A_i, B_i$ are single-qubit unitaries.

**Proof sketch:**
1. Any SU(4) matrix has 15 real parameters
2. Single-qubit gates provide 6 parameters per layer (3 per qubit)
3. CNOT provides entangling capability
4. With 3 CNOTs and appropriate single-qubit gates, we can reach any SU(4)

**Alternatively:** Show that we can implement any $\exp(-i\theta Z \otimes Z)$:

$$\text{CNOT} \cdot (I \otimes R_z(2\theta)) \cdot \text{CNOT} = \exp(-i\theta Z \otimes Z)$$

Combined with single-qubit rotations, this generates the full SU(4) Lie algebra.

$$\boxed{\text{CNOT + single-qubit gates is universal (at most 3 CNOTs needed for any 2-qubit U)}}$$

---

### Part (b): SWAP from CNOTs (7 points)

The SWAP gate: $|ab\rangle \mapsto |ba\rangle$

**Circuit:**
```
q0: ──●──X──●──
      │  │  │
q1: ──X──●──X──
```

**Verification:**
1. First CNOT (control q0): $|00\rangle \to |00\rangle$, $|01\rangle \to |01\rangle$, $|10\rangle \to |11\rangle$, $|11\rangle \to |10\rangle$
2. Second CNOT (control q1): $|00\rangle \to |00\rangle$, $|01\rangle \to |11\rangle$, $|11\rangle \to |01\rangle$, $|10\rangle \to |10\rangle$
3. Third CNOT (control q0): $|00\rangle \to |00\rangle$, $|11\rangle \to |10\rangle$, $|01\rangle \to |01\rangle$, $|10\rangle \to |11\rangle$

Tracing $|10\rangle$: $|10\rangle \to |11\rangle \to |01\rangle \to |01\rangle$ ✓
Tracing $|01\rangle$: $|01\rangle \to |01\rangle \to |11\rangle \to |10\rangle$ ✓

$$\boxed{\text{SWAP} = \text{CNOT}_{01} \cdot \text{CNOT}_{10} \cdot \text{CNOT}_{01}}$$

---

### Part (c): $\sqrt{\text{SWAP}}$ Gate (5 points)

Since SWAP has eigenvalues $\{+1, +1, +1, -1\}$ (the antisymmetric state $|01\rangle - |10\rangle$ has eigenvalue $-1$):

$$\sqrt{\text{SWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1+i}{2} & \frac{1-i}{2} & 0 \\ 0 & \frac{1-i}{2} & \frac{1+i}{2} & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Entangling property:** Apply to $|01\rangle$:
$$\sqrt{\text{SWAP}}|01\rangle = \frac{1+i}{2}|01\rangle + \frac{1-i}{2}|10\rangle$$

This is an entangled state (not a product state).

$$\boxed{\sqrt{\text{SWAP}}|01\rangle = \frac{1+i}{2}|01\rangle + \frac{1-i}{2}|10\rangle \text{ is entangled}}$$

---

### Part (d): Approximating $R_z(\pi/8)$ (5 points)

Note that $T = R_z(\pi/4)$ up to global phase.

To get $R_z(\pi/8)$, we need $\sqrt{T}$, which is not in the Clifford+T set directly.

However, using the Solovay-Kitaev theorem, we can approximate any rotation to precision $\epsilon$ using $O(\log^c(1/\epsilon))$ gates.

For a first-order approximation, we can use:
$$R_z(\pi/8) \approx T^{1/2} \approx STS^\dagger T \text{ (not exact)}$$

Actually, the exact implementation requires non-Clifford gates or magic state injection.

**Simple approximation:** Using $H$ and $T$:
$$HTH = R_x(\pi/4) \text{ (up to phase)}$$

One approach: $R_z(\pi/8)$ can be synthesized using 3-4 T gates with Clifford corrections.

$$\boxed{\text{Approximately 3-4 T gates needed with Clifford gates to synthesize } R_z(\pi/8)}$$

---

## Solution 4: Quantum Algorithms - Oracle Problems

### Part (a): Classical Query Complexity (6 points)

**Worst case:** If $f$ is constant, we need to query enough inputs to be certain.

After $k$ queries all returning the same value:
- Could be constant
- Could be balanced (queried same-output half)

To be certain $f$ is not balanced, we need $> N/2$ same-value outputs.

For $N = 2^n$ inputs, we need at least $\frac{N}{2} + 1 = 2^{n-1} + 1$ queries.

$$\boxed{\text{Classical: } 2^{n-1} + 1 \text{ queries needed for certainty}}$$

---

### Part (b): Deutsch-Jozsa Algorithm (8 points)

**Circuit:**
```
|0⟩^n ─[H^⊗n]─ ┌───┐ ─[H^⊗n]─ [Measure]
               │   │
|1⟩   ─[H]──── │ Uf│ ─────────
               └───┘
```

**Step-by-step:**

1. **Initialize:** $|0\rangle^{\otimes n}|1\rangle$

2. **Apply Hadamards:**
$$\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle \otimes \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

3. **Apply oracle $U_f$:** $|x\rangle|y\rangle \to |x\rangle|y \oplus f(x)\rangle$
$$\frac{1}{\sqrt{2^n}}\sum_x (-1)^{f(x)}|x\rangle \otimes \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

4. **Apply $H^{\otimes n}$ to query register:**
$$\frac{1}{2^n}\sum_x (-1)^{f(x)}\sum_y (-1)^{x \cdot y}|y\rangle$$

5. **Measure:** Probability of measuring $|0\rangle^{\otimes n}$ is $\left|\frac{1}{2^n}\sum_x (-1)^{f(x)}\right|^2$

$$\boxed{\text{One quantum query determines constant vs balanced with certainty}}$$

---

### Part (c): Proof of Correctness (6 points)

The amplitude of $|0^n\rangle$ after the algorithm is:
$$\alpha_{0^n} = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)}$$

**If $f$ is constant:**
- All $(-1)^{f(x)}$ are the same (all +1 or all -1)
- $\alpha_{0^n} = \frac{2^n}{2^n} = \pm 1$
- $P(0^n) = 1$

**If $f$ is balanced:**
- Half of $(-1)^{f(x)} = +1$, half $= -1$
- $\alpha_{0^n} = \frac{2^{n-1} - 2^{n-1}}{2^n} = 0$
- $P(0^n) = 0$

$$\boxed{\text{Measure } |0^n\rangle \Leftrightarrow f \text{ is constant; any other outcome } \Leftrightarrow f \text{ is balanced}}$$

---

### Part (d): Simon's Problem (5 points)

**Simon's Problem:** Given $f: \{0,1\}^n \to \{0,1\}^n$ with the promise that there exists $s \in \{0,1\}^n$ such that:
$$f(x) = f(y) \Leftrightarrow y = x \oplus s$$

Find $s$.

**Classical complexity:** $\Omega(2^{n/2})$ queries (birthday paradox)

**Quantum complexity:** $O(n)$ queries

**Why classically hard:** Finding collisions requires checking $\Omega(\sqrt{2^n})$ pairs. Under the assumption that no efficient classical algorithm can solve the hidden subgroup problem for $\mathbb{Z}_2^n$, Simon's problem is exponentially hard classically.

$$\boxed{\text{Simon's: } O(n) \text{ quantum vs } \Omega(2^{n/2}) \text{ classical queries}}$$

---

## Solution 5: Quantum Algorithms - Shor and Grover

### Part (a): Factoring via Order-Finding (8 points)

**Goal:** Factor $N = pq$ (product of two primes).

**Reduction:**

1. Choose random $a$ with $1 < a < N$ and $\gcd(a, N) = 1$

2. Find the order $r$: smallest positive integer with $a^r \equiv 1 \mod N$

3. If $r$ is even, compute $x = a^{r/2} \mod N$

4. Then $x^2 \equiv 1 \mod N$, so $(x-1)(x+1) \equiv 0 \mod N$

5. If $x \not\equiv \pm 1 \mod N$, then:
   $$\gcd(x-1, N) \text{ or } \gcd(x+1, N)$$
   gives a non-trivial factor of $N$.

**Success probability:** For random $a$, probability of getting even $r$ with $x \not\equiv \pm 1$ is at least $1/2$.

$$\boxed{\text{Order } r \to \text{factor via } \gcd(a^{r/2} \pm 1, N)}$$

---

### Part (b): QFT Analysis (7 points)

After applying modular exponentiation, the state has period $r$:
$$|\psi\rangle = \frac{1}{\sqrt{m}}\sum_{j=0}^{m-1}|x_0 + jr\rangle|a^{x_0}\rangle$$

Applying QFT to the first register:
$$QFT|k\rangle = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1}e^{2\pi ijk/N}|j\rangle$$

The QFT of a periodic state with period $r$:
$$|\psi_{out}\rangle \propto \sum_{\ell=0}^{r-1}|N\ell/r\rangle$$

Peaks occur at multiples of $N/r$.

**Extracting $r$:** Measure to get $k \approx N\ell/r$ for some $\ell$. Use continued fractions to find $r$ from $k/N \approx \ell/r$.

$$\boxed{\text{QFT gives peaks at } k = N\ell/r; \text{ continued fractions extract } r}$$

---

### Part (c): Grover Iterations (5 points)

For $N = 2^{20} \approx 10^6$ items with 1 marked item:

Optimal number of iterations:
$$k = \left\lfloor \frac{\pi}{4}\sqrt{N} \right\rfloor = \left\lfloor \frac{\pi}{4}\sqrt{2^{20}} \right\rfloor = \left\lfloor \frac{\pi}{4} \times 1024 \right\rfloor = 804$$

Success probability after $k$ iterations:
$$P = \sin^2\left((2k+1)\arcsin\frac{1}{\sqrt{N}}\right) \approx \sin^2\left(\frac{\pi}{2}\right) = 1$$

$$\boxed{k \approx 804 \text{ iterations; success probability } \approx 1}$$

---

### Part (d): Grover Optimality (5 points)

**Proof of $\Omega(\sqrt{N})$ lower bound:**

Consider the polynomial method or adversary method.

**Polynomial argument:** The amplitude of the marked state after $t$ queries is a polynomial of degree at most $2t$ in the oracle values.

To distinguish 0 marked items from 1 marked item among $N$ items requires the polynomial to have different values for different oracle settings.

By symmetry and polynomial degree bounds, distinguishing requires $t = \Omega(\sqrt{N})$.

**Adversary argument:** The query algorithm makes progress toward the solution at a rate proportional to $1/\sqrt{N}$ per query due to the structure of quantum queries.

$$\boxed{\text{Any quantum algorithm needs } \Omega(\sqrt{N}) \text{ queries for unstructured search}}$$

---

## Solution 6: Quantum Channels and Noise

### Part (a): Amplitude Damping Channel (7 points)

Input: $\rho = |\psi\rangle\langle\psi|$ where $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$

$$\rho = \begin{pmatrix} |\alpha|^2 & \alpha\beta^* \\ \alpha^*\beta & |\beta|^2 \end{pmatrix}$$

Apply channel:
$$\mathcal{E}(\rho) = K_0\rho K_0^\dagger + K_1\rho K_1^\dagger$$

$$K_0\rho K_0^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}\begin{pmatrix} |\alpha|^2 & \alpha\beta^* \\ \alpha^*\beta & |\beta|^2 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$$

$$= \begin{pmatrix} |\alpha|^2 & \sqrt{1-\gamma}\alpha\beta^* \\ \sqrt{1-\gamma}\alpha^*\beta & (1-\gamma)|\beta|^2 \end{pmatrix}$$

$$K_1\rho K_1^\dagger = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}\begin{pmatrix} |\alpha|^2 & \alpha\beta^* \\ \alpha^*\beta & |\beta|^2 \end{pmatrix}\begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}$$

$$= \begin{pmatrix} \gamma|\beta|^2 & 0 \\ 0 & 0 \end{pmatrix}$$

$$\boxed{\mathcal{E}(\rho) = \begin{pmatrix} |\alpha|^2 + \gamma|\beta|^2 & \sqrt{1-\gamma}\alpha\beta^* \\ \sqrt{1-\gamma}\alpha^*\beta & (1-\gamma)|\beta|^2 \end{pmatrix}}$$

---

### Part (b): Fixed Point (6 points)

Repeated application: as $n \to \infty$, $\mathcal{E}^n(\rho) \to \rho_{fixed}$.

From the output formula, with each application:
- Population in $|1\rangle$ decreases by factor $(1-\gamma)$
- Population transfers to $|0\rangle$
- Coherence decreases by $\sqrt{1-\gamma}$

After many applications:
- $\rho_{11} \to 0$
- $\rho_{00} \to 1$
- Off-diagonal $\to 0$

$$\boxed{\rho_{fixed} = |0\rangle\langle 0| \text{ (ground state)}}$$

**Physical interpretation:** Amplitude damping models spontaneous emission/T1 decay. The excited state $|1\rangle$ decays to ground state $|0\rangle$. At equilibrium (or after infinite time), the system is in the ground state.

---

### Part (c): Dephasing Channel (6 points)

Kraus operators: $K_0 = \sqrt{1-p/2}\,I$, $K_1 = \sqrt{p/2}\,Z$

$$\mathcal{E}(\rho) = K_0\rho K_0^\dagger + K_1\rho K_1^\dagger$$
$$= (1-p/2)\rho + (p/2)Z\rho Z$$

Note that $Z\rho Z$ flips signs of off-diagonal elements for $\rho = \begin{pmatrix} a & b \\ b^* & 1-a \end{pmatrix}$:
$$Z\rho Z = \begin{pmatrix} a & -b \\ -b^* & 1-a \end{pmatrix}$$

$$\mathcal{E}(\rho) = (1-p/2)\begin{pmatrix} a & b \\ b^* & 1-a \end{pmatrix} + (p/2)\begin{pmatrix} a & -b \\ -b^* & 1-a \end{pmatrix}$$
$$= \begin{pmatrix} a & (1-p)b \\ (1-p)b^* & 1-a \end{pmatrix}$$

$$\boxed{\text{Off-diagonal elements: } \rho_{01} \to (1-p)\rho_{01}}$$

The diagonal elements (populations) are unchanged; only coherence decays.

---

### Part (d): Composed Channels (6 points)

Amplitude damping ($\gamma = 0.1$) followed by depolarizing ($p = 0.1$):

$$\mathcal{E}_{total} = \mathcal{E}_{dep} \circ \mathcal{E}_{AD}$$

**Is this equivalent to a single channel?**

No, the composition is generally not equivalent to either pure channel type:

1. **Amplitude damping** has a fixed point $|0\rangle$ and asymmetric decay
2. **Depolarizing** has fixed point $I/2$ and symmetric noise

The composed channel has properties of both:
- Some asymmetric population transfer (from AD)
- Some symmetric depolarization (from dep)
- Different fixed point than either alone

$$\boxed{\text{The composed channel is NOT equivalent to a single AD or depolarizing channel}}$$

The Kraus operators of the composition would have more terms and different structure.

---

## Solution 7: Quantum Complexity

### Part (a): BQP Definition (6 points)

**BQP** (Bounded-error Quantum Polynomial time):

A language $L$ is in BQP if there exists a polynomial-time uniform family of quantum circuits $\{Q_n\}$ such that:

- **Completeness:** If $x \in L$, then $\Pr[Q_{|x|}(x) \text{ accepts}] \geq 2/3$
- **Soundness:** If $x \notin L$, then $\Pr[Q_{|x|}(x) \text{ accepts}] \leq 1/3$

The circuit $Q_n$ acts on $\text{poly}(n)$ qubits and uses $\text{poly}(n)$ gates from a universal gate set.

$$\boxed{\text{BQP: Quantum polynomial time with bounded 2-sided error (completeness } \geq 2/3, \text{ soundness } \leq 1/3)}$$

---

### Part (b): BPP $\subseteq$ BQP (6 points)

**Proof:** Any probabilistic classical computation can be simulated by a quantum computer.

1. Classical randomness (coin flips) can be simulated by measuring qubits in superposition:
   - Prepare $|+\rangle = H|0\rangle$
   - Measure in computational basis
   - Get 0 or 1 with probability 1/2 each

2. Classical gates (AND, OR, NOT) can be implemented as reversible quantum gates (Toffoli, CNOT, X) using ancilla qubits.

3. The simulation is efficient: polynomial overhead in both qubits and gates.

Therefore, any BPP algorithm can be run on a quantum computer with the same success probability bounds.

$$\boxed{\text{BPP} \subseteq \text{BQP via efficient simulation of classical randomness and gates}}$$

---

### Part (c): QMA Definition (7 points)

**QMA** (Quantum Merlin-Arthur):

A language $L$ is in QMA if there exists a polynomial-time quantum verifier $V$ such that:

- **Completeness:** If $x \in L$, there exists a quantum state $|\psi\rangle$ (witness) of $\text{poly}(|x|)$ qubits such that $\Pr[V(x, |\psi\rangle) \text{ accepts}] \geq 2/3$

- **Soundness:** If $x \notin L$, for all quantum states $|\psi\rangle$, $\Pr[V(x, |\psi\rangle) \text{ accepts}] \leq 1/3$

**Role of witness:** The quantum proof/witness is a quantum state that can be measured by the verifier. Unlike classical witnesses, it can exhibit superposition and entanglement.

**QMA-complete problem:** The **Local Hamiltonian Problem**:
- Given a $k$-local Hamiltonian $H = \sum_i H_i$ and thresholds $a < b$
- Decide if ground state energy $E_0 \leq a$ (yes) or $E_0 \geq b$ (no)
- This is QMA-complete for $k \geq 2$

$$\boxed{\text{QMA: quantum NP with quantum witnesses; Local Hamiltonian is QMA-complete}}$$

---

### Part (d): NP vs BQP (6 points)

**Evidence that NP $\not\subseteq$ BQP:**

1. **No known quantum algorithm for NP-complete problems:** Despite decades of research, no polynomial-time quantum algorithm for SAT, Graph Coloring, etc.

2. **Oracle separation:** Bennett et al. (1997) showed there exists an oracle $O$ relative to which:
   $$\text{NP}^O \not\subseteq \text{BQP}^O$$

   Specifically, the unstructured search problem requires $\Omega(\sqrt{N})$ queries quantumly but $O(N)$ classically. An oracle encoding NP-complete problems as unstructured search cannot be solved efficiently.

3. **Structural evidence:** NP-complete problems lack the algebraic structure (period-finding, hidden subgroups) that quantum algorithms exploit.

$$\boxed{\text{Oracle separation NP}^O \not\subseteq \text{BQP}^O; \text{ no known quantum advantage for NP-complete}}$$

---

## Solution 8: Quantum Protocols

### Part (a): Quantum Teleportation (8 points)

**Setup:** Alice has $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$. Alice and Bob share $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.

**Initial state (qubits: Alice's, A, B):**
$$|\Psi_{initial}\rangle = |\psi\rangle_A \otimes |\Phi^+\rangle_{ab} = (\alpha|0\rangle + \beta|1\rangle) \otimes \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

**(i) Alice's measurements:**
Alice performs a Bell measurement on her two qubits (the unknown state and her half of the Bell pair).

Bell basis: $|\Phi^\pm\rangle, |\Psi^\pm\rangle$

**(ii) Four outcomes:**

Rewriting the initial state in the Bell basis:
$$|\Psi\rangle = \frac{1}{2}\left[|\Phi^+\rangle(\alpha|0\rangle + \beta|1\rangle) + |\Phi^-\rangle(\alpha|0\rangle - \beta|1\rangle) + |\Psi^+\rangle(\alpha|1\rangle + \beta|0\rangle) + |\Psi^-\rangle(\alpha|1\rangle - \beta|0\rangle)\right]$$

| Alice measures | Bob's state |
|----------------|-------------|
| $|\Phi^+\rangle$ | $\alpha|0\rangle + \beta|1\rangle = |\psi\rangle$ |
| $|\Phi^-\rangle$ | $\alpha|0\rangle - \beta|1\rangle = Z|\psi\rangle$ |
| $|\Psi^+\rangle$ | $\alpha|1\rangle + \beta|0\rangle = X|\psi\rangle$ |
| $|\Psi^-\rangle$ | $\alpha|1\rangle - \beta|0\rangle = XZ|\psi\rangle$ |

**(iii) Bob's recovery:**
Alice sends 2 classical bits indicating her measurement outcome. Bob applies:
- $\Phi^+$: $I$ (do nothing)
- $\Phi^-$: $Z$
- $\Psi^+$: $X$
- $\Psi^-$: $XZ$

$$\boxed{\text{Teleportation: Bell measurement + 2 classical bits + Pauli correction}}$$

---

### Part (b): Superdense Coding (6 points)

**Protocol:** Alice wants to send 2 classical bits to Bob using 1 qubit.

**Setup:** They share $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.

**Alice's encoding:**
| Bits to send | Alice applies | Resulting state |
|--------------|---------------|-----------------|
| 00 | $I$ | $|\Phi^+\rangle$ |
| 01 | $Z$ | $|\Phi^-\rangle$ |
| 10 | $X$ | $|\Psi^+\rangle$ |
| 11 | $XZ$ | $|\Psi^-\rangle$ |

**Bob's decoding:** Bob receives Alice's qubit and performs a Bell measurement on both qubits. The four Bell states are orthogonal and perfectly distinguishable, so Bob learns which 2 bits Alice sent.

$$\boxed{\text{2 classical bits transmitted using 1 qubit + 1 shared ebit}}$$

---

### Part (c): BB84 Error Detection (6 points)

In BB84, Alice sends random bits in random bases ($Z$ or $X$). Bob measures in random bases.

**If Eve measures in wrong basis 25% of the time:**

When Eve measures in the wrong basis:
- She disturbs the state
- When Bob measures in the correct basis, he gets the wrong bit with probability 1/2

**Error rate calculation:**
- Probability Eve measures wrong basis: 1/4
- Given wrong basis, probability of error for Bob: 1/2
- Total error rate: $\frac{1}{4} \times \frac{1}{2} = \frac{1}{8} = 12.5\%$

But actually, if Eve intercepts all qubits:
- She guesses wrong basis 50% of the time
- Of those, Bob (measuring correctly) sees error 50% of the time
- Error rate = $0.5 \times 0.5 = 25\%$

For the given "25% wrong basis":
$$\text{Error rate} = 0.25 \times 0.5 = 12.5\%$$

**Security:** Alice and Bob compare a random subset of their bits. If error rate exceeds threshold (~11%), they abort. Otherwise, they use error correction and privacy amplification.

$$\boxed{\text{Error rate} = 12.5\% \text{ for 25\% wrong-basis eavesdropping}}$$

---

### Part (d): No-Cloning Theorem (5 points)

**Statement:** There is no quantum operation that can copy an arbitrary unknown quantum state.

**Proof:** Suppose there exists a unitary $U$ that clones:
$$U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$$

For states $|\psi\rangle$ and $|\phi\rangle$:
$$U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$$
$$U|\phi\rangle|0\rangle = |\phi\rangle|\phi\rangle$$

Taking inner product:
$$\langle\phi|\langle 0|U^\dagger U|\psi\rangle|0\rangle = \langle\phi|\psi\rangle = \langle\phi|\phi\rangle\langle\psi|\psi\rangle = \langle\phi|\psi\rangle^2$$

This requires $\langle\phi|\psi\rangle = \langle\phi|\psi\rangle^2$, which implies $\langle\phi|\psi\rangle = 0$ or $1$.

But this must hold for ALL pairs of states, which is impossible (contradicts the existence of non-orthogonal states with $0 < |\langle\phi|\psi\rangle| < 1$).

**Importance for cryptography:** If cloning were possible, Eve could copy qubits in BB84, measure the copies in both bases, and learn the key without disturbing the original.

$$\boxed{\text{No-cloning: } \langle\phi|\psi\rangle = \langle\phi|\psi\rangle^2 \text{ is impossible for all states}}$$

---

## Summary

| Problem | Topic | Key Results |
|---------|-------|-------------|
| 1 | Density matrices | $S(\rho_A) = 1$; GHZ-type state |
| 2 | Entanglement/CHSH | $C = \sin(2\theta)$; $S = 2\sqrt{2}$ at maximum |
| 3 | Gates/circuits | SWAP = 3 CNOTs; $\sqrt{SWAP}$ is entangling |
| 4 | Deutsch-Jozsa/Simon | 1 query vs $2^{n-1}+1$ classical |
| 5 | Shor/Grover | Order-finding reduces to factoring; $O(\sqrt{N})$ optimal |
| 6 | Channels | AD fixed point = $|0\rangle$; composed channels different |
| 7 | Complexity | BPP $\subseteq$ BQP; QMA has quantum witnesses |
| 8 | Protocols | Teleportation, superdense, BB84, no-cloning |
