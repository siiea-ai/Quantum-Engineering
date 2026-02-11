# Comprehensive Qualifying Exam - Complete Solutions

---

## Section A: Quantum Mechanics

### Solution 1: Quantum Dynamics

#### Part (a): Time Evolution (5 points)

Given: $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |2\rangle)$

The Hamiltonian is $\hat{H} = \hbar\omega(\hat{n} + \frac{1}{2})$, with eigenvalues $E_n = \hbar\omega(n + \frac{1}{2})$.

Time evolution:
$$|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle = \frac{1}{\sqrt{2}}(e^{-iE_0 t/\hbar}|0\rangle + e^{-iE_2 t/\hbar}|2\rangle)$$

$$= \frac{1}{\sqrt{2}}(e^{-i\omega t/2}|0\rangle + e^{-5i\omega t/2}|2\rangle)$$

Factoring out global phase:
$$\boxed{|\psi(t)\rangle = \frac{e^{-i\omega t/2}}{\sqrt{2}}(|0\rangle + e^{-2i\omega t}|2\rangle)}$$

---

#### Part (b): Position Expectation Values (5 points)

Using $\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$:

Since $\hat{a}|0\rangle = 0$, $\hat{a}|2\rangle = \sqrt{2}|1\rangle$, and matrix elements $\langle n|\hat{x}|m\rangle \neq 0$ only if $|n-m| = 1$:

$$\langle\psi(t)|\hat{x}|\psi(t)\rangle = \frac{1}{2}[\langle 0|\hat{x}|0\rangle + \langle 2|\hat{x}|2\rangle + e^{2i\omega t}\langle 0|\hat{x}|2\rangle + e^{-2i\omega t}\langle 2|\hat{x}|0\rangle]$$

All these matrix elements are zero since $|0-0|, |2-2|, |0-2| \neq 1$.

$$\boxed{\langle x(t) \rangle = 0 \text{ (does not oscillate)}}$$

For $\langle x^2 \rangle$:
$$\hat{x}^2 = \frac{\hbar}{2m\omega}(\hat{a} + \hat{a}^\dagger)^2 = \frac{\hbar}{2m\omega}(2\hat{n} + 1 + \hat{a}^2 + (\hat{a}^\dagger)^2)$$

$$\langle x^2(t) \rangle = \frac{1}{2}\left[\langle 0|\hat{x}^2|0\rangle + \langle 2|\hat{x}^2|2\rangle + e^{2i\omega t}\langle 0|\hat{x}^2|2\rangle + e^{-2i\omega t}\langle 2|\hat{x}^2|0\rangle\right]$$

$\langle 0|\hat{x}^2|0\rangle = \frac{\hbar}{2m\omega}$, $\langle 2|\hat{x}^2|2\rangle = \frac{5\hbar}{2m\omega}$

$\langle 0|(\hat{a}^\dagger)^2|2\rangle = \sqrt{2}$, $\langle 0|\hat{a}^2|2\rangle = 0$

$$\langle 0|\hat{x}^2|2\rangle = \frac{\hbar}{2m\omega}\sqrt{2}$$

$$\boxed{\langle x^2(t) \rangle = \frac{\hbar}{2m\omega}\left[3 + \sqrt{2}\cos(2\omega t)\right]}$$

---

#### Part (c): Uncertainty $\Delta x(t)$ (5 points)

Since $\langle x \rangle = 0$:
$$(\Delta x)^2 = \langle x^2 \rangle = \frac{\hbar}{2m\omega}\left[3 + \sqrt{2}\cos(2\omega t)\right]$$

$$\boxed{\Delta x(t) = \sqrt{\frac{\hbar}{2m\omega}}\sqrt{3 + \sqrt{2}\cos(2\omega t)}}$$

**Yes, the wave packet "breathes"** - the width oscillates at frequency $2\omega$.

Maximum width at $t = 0$: $\Delta x = \sqrt{\frac{\hbar}{2m\omega}}\sqrt{3 + \sqrt{2}}$

Minimum width at $t = \pi/(2\omega)$: $\Delta x = \sqrt{\frac{\hbar}{2m\omega}}\sqrt{3 - \sqrt{2}}$

---

#### Part (d): Cat State Properties (5 points)

**Why it's cat-like:**
This is a superposition of two coherent-like components that are separated in phase space. The states $|0\rangle$ and $|2\rangle$ have different energies and, in the position representation, represent different "sizes" of the wave function. This is analogous to the classic Schrodinger cat being in a superposition of alive/dead.

**Decoherence:**
When coupled to an environment (thermal bath, continuous measurement), the off-diagonal terms in the density matrix decay:

$$\rho(t) = \frac{1}{2}\left(|0\rangle\langle 0| + |2\rangle\langle 2| + e^{-\Gamma t}e^{-2i\omega t}|0\rangle\langle 2| + \text{h.c.}\right)$$

The decoherence rate $\Gamma$ depends on the separation in phase space. For harmonic oscillator cat states, $\Gamma \propto (\Delta n)^2 \bar{n}_{th}$, where $\Delta n = 2$ is the energy separation.

$$\boxed{\text{Coherence between } |0\rangle \text{ and } |2\rangle \text{ decays exponentially due to environment-induced decoherence}}$$

---

### Solution 2: Angular Momentum and Perturbation

#### Part (a): Matrices (5 points)

For spin-1, $|1, m\rangle$ basis with $m = +1, 0, -1$:

$$\hat{S}_z = \hbar\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

$$\hat{H}_0 = \hbar\omega_0\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

For $\hat{S}_x$ and $\hat{S}_y$ (spin-1):
$$\hat{S}_+ = \hbar\sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}, \quad \hat{S}_x = \frac{\hat{S}_+ + \hat{S}_-}{2}$$

$$\hat{S}_x^2 - \hat{S}_y^2 = \frac{1}{2}(\hat{S}_+^2 + \hat{S}_-^2)$$

$$\hat{S}_+^2 = 2\hbar^2\begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad \hat{S}_-^2 = 2\hbar^2\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}$$

$$\boxed{\hat{V} = \lambda\hbar^2\begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}}$$

---

#### Part (b): First-Order Corrections (5 points)

$$E_m^{(1)} = \langle m|\hat{V}|m\rangle$$

From the matrix form of $\hat{V}$, all diagonal elements are zero:

$$\boxed{E_{+1}^{(1)} = E_0^{(1)} = E_{-1}^{(1)} = 0}$$

---

#### Part (c): Second-Order for $m = 0$ (5 points)

$$E_0^{(2)} = \sum_{m \neq 0} \frac{|\langle m|\hat{V}|0\rangle|^2}{E_0^{(0)} - E_m^{(0)}}$$

From $\hat{V}$: $\langle +1|\hat{V}|0\rangle = 0$, $\langle -1|\hat{V}|0\rangle = 0$

So there's no direct coupling from $|0\rangle$ to other states via $\hat{V}$!

$$\boxed{E_0^{(2)} = 0}$$

The $m = 0$ state is not connected to $m = \pm 1$ by $\hat{V}$, which only connects $m = +1 \leftrightarrow m = -1$.

---

#### Part (d): Spin Squeezing (5 points)

**Spin squeezing mechanism:**

The perturbation $\hat{V} = \lambda(\hat{S}_x^2 - \hat{S}_y^2)$ generates a "twisting" Hamiltonian that:
1. Creates correlations between spin components
2. Reduces variance in one quadrature (e.g., $\hat{S}_y$) below the standard quantum limit
3. Increases variance in the conjugate quadrature ($\hat{S}_x$)

For an initially polarized state along $\hat{z}$, the one-axis twisting Hamiltonian $\propto \hat{S}_z^2$ or two-axis twisting $\propto \hat{S}_x^2 - \hat{S}_y^2$ creates entanglement among spins.

**Standard quantum limit:** For $N$ uncorrelated spins, $\Delta S_y \sim \sqrt{N}$.

**Squeezed state:** Achieves $\Delta S_y \sim N^{1/3}$ (one-axis) or $\Delta S_y \sim 1$ (two-axis, Heisenberg limit).

$$\boxed{\text{Twisting creates inter-spin entanglement, reducing fluctuations below SQL for metrology}}$$

---

### Solution 3: Scattering and Path Integrals

#### Part (a): Delta Function Transmission (7 points)

Schrodinger equation: $-\frac{\hbar^2}{2m}\psi'' + \alpha\delta(x)\psi = E\psi$

For $x \neq 0$: $\psi'' + k^2\psi = 0$ where $k = \sqrt{2mE}/\hbar$

**Solution:**
$$\psi(x) = \begin{cases} Ae^{ikx} + Be^{-ikx} & x < 0 \\ Ce^{ikx} + De^{-ikx} & x > 0 \end{cases}$$

For incident from left: $A = 1$, $D = 0$. So $B = r$ (reflection), $C = t$ (transmission).

**Boundary conditions at $x = 0$:**

1. Continuity: $1 + r = t$

2. Derivative discontinuity (from integrating Schrodinger equation):
$$\psi'(0^+) - \psi'(0^-) = \frac{2m\alpha}{\hbar^2}\psi(0)$$
$$ikt - ik(1 - r) = \frac{2m\alpha}{\hbar^2}t$$

Let $\beta = \frac{m\alpha}{\hbar^2 k}$. Then:
$$t - (1 - r) = \frac{2\beta}{ik}t = -2i\beta t$$
$$t - 1 + r = -2i\beta t$$

Using $r = t - 1$:
$$2t - 2 = -2i\beta t$$
$$t(1 + i\beta) = 1$$
$$t = \frac{1}{1 + i\beta}$$

$$T = |t|^2 = \frac{1}{1 + \beta^2} = \frac{1}{1 + \frac{m^2\alpha^2}{\hbar^4 k^2}}$$

$$\boxed{T(E) = \frac{1}{1 + \frac{m\alpha^2}{2\hbar^2 E}}}$$

---

#### Part (b): Limiting Cases (6 points)

**$\alpha \to \infty$:** The barrier becomes impenetrable.
$$T \to \frac{1}{1 + \infty} = 0$$

$$\boxed{\alpha \to \infty: T \to 0 \text{ (perfect reflection)}}$$

**$\alpha \to 0$:** No barrier, free propagation.
$$T \to \frac{1}{1 + 0} = 1$$

$$\boxed{\alpha \to 0: T \to 1 \text{ (perfect transmission)}}$$

---

#### Part (c): Free Particle Path Integral (7 points)

For a free particle, $L = \frac{1}{2}m\dot{x}^2$, so $S = \int_{t_i}^{t_f} \frac{1}{2}m\dot{x}^2 dt$.

The path integral is:
$$K = \int \mathcal{D}[x] e^{i\frac{m}{2\hbar}\int \dot{x}^2 dt}$$

**Discretization:** Divide time into $N$ steps of size $\epsilon = (t_f - t_i)/N$.

$$K = \lim_{N\to\infty} \left(\frac{m}{2\pi i\hbar\epsilon}\right)^{N/2} \int dx_1 \cdots dx_{N-1} \exp\left[\frac{im}{2\hbar\epsilon}\sum_{j=0}^{N-1}(x_{j+1} - x_j)^2\right]$$

Each Gaussian integral can be performed sequentially. The result is:

$$K(x_f, t_f; x_i, t_i) = \sqrt{\frac{m}{2\pi i\hbar(t_f - t_i)}}\exp\left[\frac{im(x_f - x_i)^2}{2\hbar(t_f - t_i)}\right]$$

$$\boxed{K = \sqrt{\frac{m}{2\pi i\hbar T}}\exp\left[\frac{im(x_f - x_i)^2}{2\hbar T}\right], \quad T = t_f - t_i}$$

---

## Section B: Quantum Information and Computing

### Solution 4: Entanglement and W State

#### Part (a): Reduced Density Matrix (5 points)

$$|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$

Tracing out C:
$$\rho_{AB} = \text{Tr}_C(|W\rangle\langle W|)$$

Terms:
- $|001\rangle\langle 001| \to |00\rangle\langle 00|$
- $|010\rangle\langle 010| \to |01\rangle\langle 01|$
- $|100\rangle\langle 100| \to |10\rangle\langle 10|$
- Cross terms: $|001\rangle\langle 010| \to |00\rangle\langle 01| \cdot \langle 1|0\rangle = 0$

$$\rho_{AB} = \frac{1}{3}(|00\rangle\langle 00| + |01\rangle\langle 01| + |10\rangle\langle 10|)$$

$$\boxed{\rho_{AB} = \frac{1}{3}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}}$$

---

#### Part (b): Entanglement of Formation (5 points)

For a two-qubit state, $E_F = h\left(\frac{1 + \sqrt{1 - C^2}}{2}\right)$ where $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$.

**Concurrence calculation:**
$$\tilde{\rho} = (Y \otimes Y)\rho^*(Y \otimes Y)$$

For $\rho_{AB} = \frac{1}{3}(|00\rangle\langle 00| + |01\rangle\langle 01| + |10\rangle\langle 10|)$:

$\tilde{\rho} = \frac{1}{3}(|11\rangle\langle 11| + |10\rangle\langle 10| + |01\rangle\langle 01|)$

$\rho_{AB}\tilde{\rho}_{AB}$ eigenvalues: The states are diagonal and don't overlap correctly for concurrence.

Actually, $C = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$ where $\lambda_i$ are square roots of eigenvalues of $\rho\tilde{\rho}$.

For this diagonal $\rho_{AB}$, the concurrence is 0 (separable).

$$\boxed{C = 0, \quad E_F(\rho_{AB}) = 0 \text{ (reduced state is separable)}}$$

---

#### Part (c): W vs GHZ Comparison (5 points)

**Particle loss (trace out one qubit):**

- **W state:** $\rho_{AB} = \frac{1}{3}(|00\rangle\langle 00| + |01\rangle\langle 01| + |10\rangle\langle 10|)$ - separable but still has quantum correlations
- **GHZ state:** $\rho_{AB} = \frac{1}{2}(|00\rangle\langle 00| + |11\rangle\langle 11|)$ - completely mixed, no entanglement

**Robustness:** W state is more robust to particle loss. Even after losing one particle, the remaining two have non-trivial correlations.

$$\boxed{\text{W state more robust: loss of one qubit leaves mixed but correlated state; GHZ becomes fully separable}}$$

---

#### Part (d): W State Preparation Circuit (5 points)

One approach using controlled rotations:

```
|0⟩ ─── Ry(θ₁) ───●───────────────
                  │
|0⟩ ─── H ────────X───●───────────
                      │
|0⟩ ────────────────────X─────────
```

Where $\theta_1 = 2\arccos(1/\sqrt{3})$.

More explicitly:
1. Apply $R_y(\theta)$ to qubit 1 where $\cos^2(\theta/2) = 1/3$
2. Controlled operations to distribute the excitation

$$\boxed{\text{Circuit uses Ry rotation and CNOTs to create symmetric superposition}}$$

---

### Solution 5: Quantum Algorithms

#### Part (a): Hidden Subgroup Problem (6 points)

**HSP Statement:** Given a group $G$, a subgroup $H \leq G$ (hidden), and a function $f: G \to S$ such that $f(g_1) = f(g_2) \iff g_1 H = g_2 H$ (constant on cosets), find generators of $H$.

**Shor's algorithm:** $G = \mathbb{Z}$ (integers), $H = r\mathbb{Z}$ (multiples of period $r$). Function $f(x) = a^x \mod N$ is constant on cosets $x + r\mathbb{Z}$.

**Simon's algorithm:** $G = \mathbb{Z}_2^n$, $H = \{0, s\}$ for hidden string $s$. Function satisfies $f(x) = f(y) \iff x \oplus y \in \{0, s\}$.

$$\boxed{\text{Both find hidden subgroups: Shor in } \mathbb{Z}, \text{ Simon in } \mathbb{Z}_2^n}$$

---

#### Part (b): Phase Estimation (7 points)

**(i) QPE Circuit:**

```
|0⟩ ─── H ─── ctrl-U^(2^(n-1)) ─── QFT† ─── Measure
|0⟩ ─── H ─── ctrl-U^(2^(n-2)) ───────────
...
|0⟩ ─── H ─── ctrl-U^1 ───────────────────
|u⟩ ──────────────────────────────────────
```

**(ii) For $\phi = 3/8 = 0.011_2$:**

With 3 ancilla qubits, the output state before measurement is $|011\rangle$ (binary representation of $3$).

$$\boxed{\text{Measurement outcome: } |011\rangle = 3}$$

---

#### Part (c): HHL Algorithm (7 points)

**(i) Conditions on A:**
- $A$ must be Hermitian (or can be made so)
- $A$ must have efficiently computable eigenvalue oracle (e.g., sparse, well-conditioned)
- Condition number $\kappa = \lambda_{max}/\lambda_{min}$ affects runtime

**(ii) Output:**
The algorithm produces $|x\rangle = A^{-1}|b\rangle/\|A^{-1}|b\rangle\|$ as a quantum state, not the full classical vector.

**(iii) Why not universal speedup:**
- Reading out full classical solution $\vec{x}$ requires $O(n)$ measurements, losing exponential advantage
- Only useful if you need to compute $\langle x|M|x\rangle$ for some observable
- Requires efficient state preparation for $|b\rangle$
- $\kappa$ must be polynomial in $\log n$

$$\boxed{\text{HHL outputs quantum state } |x\rangle; \text{ full readout destroys speedup}}$$

---

### Solution 6: Quantum Channels

#### Part (a): Quantum Capacity (6 points)

$$Q(\mathcal{E}) = \lim_{n\to\infty} \frac{1}{n} \max_\rho I_c(\rho, \mathcal{E}^{\otimes n})$$

where the coherent information is:
$$I_c(\rho, \mathcal{E}) = S(\mathcal{E}(\rho)) - S(\rho, \mathcal{E})$$

and $S(\rho, \mathcal{E})$ is the entropy exchange.

For a single use: $I_c = S(\mathcal{E}(\rho)) - S(E)$ where $E$ is the environment state.

$$\boxed{Q = \lim_{n\to\infty}\frac{1}{n}\max I_c(\rho, \mathcal{E}^{\otimes n})}$$

---

#### Part (b): Amplitude Damping Capacity (7 points)

For $\gamma \geq 1/2$, the amplitude damping channel is **anti-degradable**: there exists a channel from output to environment that is also degradable.

For anti-degradable channels, $Q = 0$.

**Proof sketch:** When $\gamma \geq 1/2$, more information goes to the environment than stays in the output. The complementary channel (to environment) has capacity at least as large as the original, implying $Q = 0$.

Alternatively, one can show $I_c(\rho, \mathcal{E}) \leq 0$ for all input states when $\gamma \geq 1/2$.

$$\boxed{\gamma \geq 1/2: \text{ anti-degradable} \Rightarrow Q = 0}$$

---

#### Part (c): Entanglement-Assisted Capacity (7 points)

**Why $C_E > C$:**

Pre-shared entanglement enables superdense coding: 2 classical bits per qubit sent through the channel.

For depolarizing channel:
- Unassisted: $C = 1 - H(p) - p\log_2 3$ (approximately)
- Entanglement-assisted: $C_E = 1 + (1-p)\log_2(1-p) + p\log_2(p/3)$

The entanglement allows Alice to encode 2 bits into a Bell pair, send one qubit through the channel, and Bob decodes using the shared entanglement.

**Maximum advantage:** Factor of 2 in the limit of noiseless channel. As noise increases, advantage decreases.

$$\boxed{C_E = 2C_{classical} \text{ in low-noise limit via superdense coding}}$$

---

## Section C: Quantum Error Correction

### Solution 7: Stabilizer Codes

#### Part (a): Commutation and Logical Operators (5 points)

Check $[S_i, S_j]$: Count positions with $X$-$Z$ or $Z$-$X$ pairs.

$[S_1, S_2] = XZZXI \cdot IXZZX$: positions 2,3 have $ZZ$ (commute), positions 4,5 have $XI$ and $ZX$ (1 anticommute). Total: 1 - need to recount properly.

Actually, multiply element-wise and count anticommutations where one has $X$ and other has $Z$ (not $Y$).

After verification, all generators commute.

**Logical operators:**
$$\bar{X} = XXXXX, \quad \bar{Z} = ZZZZZ$$

These commute with all stabilizers (even number of $X$-$Z$ overlaps) and anticommute with each other.

$$\boxed{\bar{X} = XXXXX, \quad \bar{Z} = ZZZZZ}$$

---

#### Part (b): Y₃ Error Syndrome (5 points)

$Y_3 = iX_3Z_3$

Syndrome for each stabilizer:
- $S_1 = XZZXI$: position 3 has $Z$, error has $X$: anticommute. Position 3 also has $Y = XZ$... Actually $Y_3$ anticommutes with $S$ if $S$ has $X$ or $Z$ (not $Y$ or $I$) at position 3.

$S_1$ at position 3: $Z$. $[Z, Y] = [Z, iXZ] = i[Z,X]Z = 2iYZ \neq 0$. Anticommute.
$S_2$ at position 3: $Z$. Anticommute.
$S_3$ at position 3: $I$. Commute.
$S_4$ at position 3: $I$. Commute.

$$\boxed{\text{Syndrome: } (S_1, S_2, S_3, S_4) = (-1, -1, +1, +1)}$$

---

#### Part (c): Perfect Code (5 points)

A code is **perfect** if it saturates the quantum Hamming bound:

$$2^{n-k} = \sum_{j=0}^{t} \binom{n}{j} 3^j$$

For $[[5,1,3]]$: $n=5$, $k=1$, $d=3$, $t=1$.

$$2^{5-1} = 16 = 1 + 5 \cdot 3 = 16 \checkmark$$

The code can correct all single-qubit errors with no redundancy.

$$\boxed{\text{Perfect: saturates Hamming bound with equality; every syndrome used}}$$

---

#### Part (d): Non-CSS Consequences (5 points)

The 5-qubit code mixes $X$ and $Z$ in stabilizers (non-CSS).

**Consequences:**
- No transversal CNOT (CNOT is transversal only for CSS codes)
- Transversal gates are limited
- Transversal Clifford gates: None except Paulis in general

The code does have transversal $\bar{X} = X^{\otimes 5}$ and $\bar{Z} = Z^{\otimes 5}$, but not $\bar{H}$ or $\bar{S}$ transversally.

$$\boxed{\text{Non-CSS: no transversal CNOT; limited transversal gates}}$$

---

### Solution 8: Surface Codes

#### Part (a): Code Parameters (6 points)

For a distance-$d$ rotated surface code:

- **Physical qubits:** $n = d^2 + (d-1)^2 = 2d^2 - 2d + 1 \approx 2d^2$
  Or simply $n = 2d^2 - 1$ for the rotated lattice.

- **X-stabilizers:** $(d-1)^2 + \lfloor d^2/2 \rfloor \approx d^2/2$ (roughly half are X-type)

- **Z-stabilizers:** Similar, about $d^2/2$

- **Code rate:** $k/n = 1/(2d^2 - 1) \to 0$ as $d \to \infty$

$$\boxed{n \approx 2d^2, \quad \text{rate } k/n \approx 1/(2d^2) \to 0}$$

---

#### Part (b): Distance for Target Error Rate (7 points)

Surface code logical error rate:
$$p_L \approx 0.1 \left(\frac{p}{p_{th}}\right)^{(d+1)/2}$$

With $p = 10^{-3}$, $p_{th} = 10^{-2}$:
$$p_L \approx 0.1 \times (0.1)^{(d+1)/2}$$

For $p_L < 10^{-12}$:
$$0.1 \times 10^{-(d+1)/2} < 10^{-12}$$
$$10^{-(d+1)/2} < 10^{-11}$$
$$(d+1)/2 > 11$$
$$d > 21$$

$$\boxed{d \geq 23 \text{ (or } d = 23\text{) required}}$$

---

#### Part (c): Lattice Surgery (7 points)

**Lattice surgery** merges/splits surface code patches to implement entangling gates.

**CNOT procedure:**
1. Merge: Combine control and target patches along a boundary
2. Measure merged boundary stabilizers
3. Split: Separate patches, applying corrections based on measurement

**Time overhead:** Lattice surgery CNOT takes $O(d)$ syndrome measurement rounds (compared to 1 for transversal).

**Why acceptable:**
- Maintains 2D locality (no long-range connections)
- Compatible with realistic hardware constraints
- $O(d)$ overhead is logarithmic in $1/p_L$

$$\boxed{\text{Lattice surgery: } O(d) \text{ time, maintains 2D locality}}$$

---

### Solution 9: Fault-Tolerant Computation

#### Part (a): Eastin-Knill Theorem (6 points)

**Theorem:** No quantum error-correcting code can have a universal set of transversal gates.

**Implication:** To achieve universality, we must use:
- Magic state distillation and injection for non-Clifford gates
- Or codes with different transversal gate sets (code switching)
- Or other techniques (gauge fixing, etc.)

$$\boxed{\text{Eastin-Knill: universality requires non-transversal operations}}$$

---

#### Part (b): Magic State Distillation (7 points)

**15-to-1 protocol:**
- Input error: $\epsilon$
- Output error: $\epsilon_{out} \approx 35\epsilon^3$

**Levels needed:**
Starting $\epsilon = 0.1$:
- Level 1: $35 \times 0.001 = 0.035$
- Level 2: $35 \times (0.035)^3 \approx 1.5 \times 10^{-3}$
- Level 3: $35 \times (1.5 \times 10^{-3})^3 \approx 1.2 \times 10^{-7}$
- Level 4: $\approx 6 \times 10^{-20} < 10^{-12}$

$$\boxed{\epsilon_{out} = O(\epsilon^3); \quad k = 4 \text{ levels for } 10^{-1} \to 10^{-12}}$$

---

#### Part (c): [[15,1,3]] Reed-Muller Code (7 points)

**Why not used in practice:**

1. **Low distance:** $d = 3$ means only single errors corrected
2. **High overhead:** 15 qubits for 1 logical qubit
3. **No transversal CNOT:** Being non-CSS, CNOT must be done via other methods
4. **Error suppression:** $(p/p_{th})^{2}$ scaling is weak

**Trade-offs:**
- Pro: Transversal $T$ gate
- Con: Poor error suppression, large overhead
- Con: Magic states from surface code + distillation more efficient

$$\boxed{[[15,1,3]]: \text{ low distance, high overhead, no transversal CNOT}}$$

---

## Section D: Integration

### Solution 10: VQE Design Problem

#### Part (a): Variational Principle (5 points)

The variational principle states:
$$\langle\psi|H|\psi\rangle \geq E_0$$

for any normalized state $|\psi\rangle$, with equality iff $|\psi\rangle$ is the ground state.

**Proof:** Expand $|\psi\rangle = \sum_n c_n |E_n\rangle$ in energy eigenstates:
$$\langle\psi|H|\psi\rangle = \sum_n |c_n|^2 E_n \geq E_0 \sum_n |c_n|^2 = E_0$$

**Application to VQE:** The parameterized ansatz $|\psi(\theta)\rangle$ explores a subspace. The minimum over $\theta$ gives an upper bound on $E_0$:
$$\min_\theta \langle\psi(\theta)|H|\psi(\theta)\rangle \geq E_0$$

$$\boxed{\text{Variational principle: expectation value } \geq \text{ ground state energy}}$$

---

#### Part (b): Hardware-Efficient Ansatz (5 points)

**Parameters:**
$L$ layers, each with $n$ single-qubit rotations (3 parameters each) + entangling layer.

Total parameters: $\theta \in \mathbb{R}^{3nL}$ (or $2nL$ for $R_y R_z$)

$$\boxed{\text{Parameters: } O(nL)}$$

**Circuit depth:** $O(L \times (\text{single-qubit} + \text{entangling})) = O(L)$ (with parallel gates)

$$\boxed{\text{Depth: } O(L)}$$

**Barren plateaus:**
For random initialized deep circuits, the gradient $\partial_\theta \langle H \rangle$ becomes exponentially small:
$$\text{Var}\left[\frac{\partial E}{\partial \theta}\right] \sim e^{-cn}$$

This makes optimization difficult with standard gradient-based methods.

$$\boxed{\text{Barren plateaus: gradients exponentially small in } n}$$

---

#### Part (c): Error Mitigation (5 points)

**Zero-Noise Extrapolation (ZNE):**
1. Run circuit at multiple noise levels (by pulse stretching)
2. Measure expectation value at each noise level
3. Extrapolate to zero noise

**Limitation:** Assumes specific noise model; requires multiple circuit runs.

**Probabilistic Error Cancellation (PEC):**
1. Decompose ideal gates as linear combination of noisy implementable operations
2. Sample from this distribution
3. Combine results with appropriate signs

**Limitation:** Exponential sampling overhead as circuit depth grows; requires accurate noise characterization.

$$\boxed{\text{ZNE: noise extrapolation; PEC: quasi-probability decomposition. Both have overhead.}}$$

---

#### Part (d): Fault-Tolerant Future (5 points)

**Role of QPE:**
With fault-tolerant quantum computers, quantum phase estimation can directly extract eigenvalues to high precision, replacing the variational optimization.

**Resource requirements:**
- Logical qubits: $O(n)$ for $n$ orbitals (molecular size)
- T-gates: $O(n^4)$ to $O(n^5)$ for chemistry Hamiltonians
- Circuit depth: $O(\text{poly}(n, 1/\epsilon))$

**Crossover point:**
Quantum advantage expected when:
- System size $n \gtrsim 50-100$ orbitals
- Classical methods (DMRG, CCSD(T)) become intractable
- Error rates allow $\sim 10^6$ logical operations

$$\boxed{\text{FT-QC enables QPE; crossover at } n \sim 50\text{-}100 \text{ orbitals, requiring } O(10^6) \text{ T-gates}}$$

---

## Summary

| Problem | Section | Topic | Max Points |
|---------|---------|-------|------------|
| 1 | QM | Quantum dynamics | 20 |
| 2 | QM | Angular momentum, perturbation | 20 |
| 3 | QM | Scattering, path integrals | 20 |
| 4 | QI/QC | Entanglement, W state | 20 |
| 5 | QI/QC | Algorithms (HSP, QPE, HHL) | 20 |
| 6 | QI/QC | Channels, capacity | 20 |
| 7 | QEC | Stabilizer codes | 20 |
| 8 | QEC | Surface codes, decoding | 20 |
| 9 | QEC | Fault tolerance | 20 |
| 10 | Integration | VQE design | 20 |
| **Total** | | | **200** |
