# QI/QC Theory Integration Exam - Solutions

## Section A: Density Matrices and Entanglement

### Solution A1

**a) Density matrix (2 pts):**

$$|\psi\rangle = \frac{1}{2}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{\sqrt{2}}|11\rangle$$

$$\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} 1/4 & 1/4 & 0 & 1/(2\sqrt{2}) \\ 1/4 & 1/4 & 0 & 1/(2\sqrt{2}) \\ 0 & 0 & 0 & 0 \\ 1/(2\sqrt{2}) & 1/(2\sqrt{2}) & 0 & 1/2 \end{pmatrix}$$

**b) Reduced density matrix (2 pts):**

$$\rho_A = \text{Tr}_B(\rho) = \begin{pmatrix} \rho_{00,00} + \rho_{01,01} & \rho_{00,10} + \rho_{01,11} \\ \rho_{10,00} + \rho_{11,01} & \rho_{10,10} + \rho_{11,11} \end{pmatrix}$$

$$\rho_A = \begin{pmatrix} 1/4 + 1/4 & 1/(2\sqrt{2}) \\ 1/(2\sqrt{2}) & 0 + 1/2 \end{pmatrix} = \begin{pmatrix} 1/2 & 1/(2\sqrt{2}) \\ 1/(2\sqrt{2}) & 1/2 \end{pmatrix}$$

**c) Eigenvalues and entropy (2 pts):**

$\text{Tr}(\rho_A) = 1$, $\det(\rho_A) = 1/4 - 1/8 = 1/8$

Eigenvalues: $\lambda_{\pm} = \frac{1 \pm \sqrt{1-1/2}}{2} = \frac{1 \pm 1/\sqrt{2}}{2}$

$\lambda_+ \approx 0.854$, $\lambda_- \approx 0.146$

$$S(\rho_A) = -0.854\log_2(0.854) - 0.146\log_2(0.146) \approx 0.60 \text{ bits}$$

**d) Entanglement (2 pts):**

Since $S(\rho_A) > 0$ and the state is pure, $|\psi\rangle$ is entangled.

The Schmidt decomposition has two non-zero Schmidt coefficients $\sqrt{\lambda_\pm}$, confirming entanglement.

---

### Solution A2

**a) Eigenvalues (2 pts):**

In Bell basis, $\rho_W$ is diagonal:
$$\rho_W = \frac{1+3p}{4}|\Phi^+\rangle\langle\Phi^+| + \frac{1-p}{4}(|\Phi^-\rangle\langle\Phi^-| + |\Psi^+\rangle\langle\Psi^+| + |\Psi^-\rangle\langle\Psi^-|)$$

Eigenvalues: $\frac{1+3p}{4}$ (multiplicity 1), $\frac{1-p}{4}$ (multiplicity 3)

**b) Entanglement range (2 pts):**

Using PPT criterion: $\rho_W$ is entangled iff $p > 1/3$.

The partial transpose of $\rho_W$ has eigenvalue $(1-3p)/4$, which is negative for $p > 1/3$.

**c) Conditional entropy at p=1 (2 pts):**

At $p = 1$: $\rho_W = |\Phi^+\rangle\langle\Phi^+|$

$S(AB) = 0$ (pure state)
$\rho_B = I/2$, so $S(B) = 1$

$S(A|B) = S(AB) - S(B) = 0 - 1 = -1$ bit

---

### Solution A3

**a) CHSH inequality (3 pts):**

For observables $A, A'$ (Alice) and $B, B'$ (Bob):
$$S = E(A,B) - E(A,B') + E(A',B) + E(A',B')$$

Classical bound: $|S| \leq 2$
Quantum bound: $|S| \leq 2\sqrt{2}$

CHSH violation proves entanglement because separable states satisfy the classical bound.

**b) Maximum violation (3 pts):**

For $|\Phi^+\rangle$ with measurements in X-Z plane at angles $\theta$:
$$E(\theta_A, \theta_B) = -\cos(\theta_A - \theta_B)$$

Wait, for $|\Phi^+\rangle$: $E(\theta_A, \theta_B) = \cos(2(\theta_A - \theta_B))$ for measurement in the equator.

Optimal angles: $A = 0°$, $A' = 45°$, $B = 22.5°$, $B' = 67.5°$

$$S = \cos(45°) + \cos(45°) + \cos(45°) - \cos(135°) = 4 \times \frac{\sqrt{2}}{2} = 2\sqrt{2}$$

---

## Section B: Quantum Channels

### Solution B1

**a) Kraus operators (2 pts):**

$$\mathcal{D}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

Kraus operators: $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p/3}X$, $K_2 = \sqrt{p/3}Y$, $K_3 = \sqrt{p/3}Z$

Verify: $\sum_i K_i^\dagger K_i = (1-p)I + \frac{p}{3}(I + I + I) = I$ ✓

**b) Choi matrix (3 pts):**

$$J_\mathcal{D} = (\mathcal{D} \otimes I)(|\Phi^+\rangle\langle\Phi^+|)$$

$$= (1-p)|\Phi^+\rangle\langle\Phi^+| + \frac{p}{3}(|\Psi^+\rangle\langle\Psi^+| + |\Psi^-\rangle\langle\Psi^-| + |\Phi^-\rangle\langle\Phi^-|)$$

Eigenvalues: $(1-p) + p/3 = 1 - 2p/3$ and $p/3$ (three times).

All non-negative for $p \in [0,1]$, so positive semidefinite. ✓

**c) Entanglement-breaking (3 pts):**

Channel is entanglement-breaking when $p \geq 1/2$ (Choi matrix becomes separable).

For $p \geq 1/2$, the depolarizing channel can be written as measure-and-prepare.

---

### Solution B2

**a) Output for |+⟩ (2 pts):**

$|+\rangle\langle +| = \frac{1}{2}\begin{pmatrix}1 & 1\\1 & 1\end{pmatrix}$

$$\mathcal{A}(|+\rangle\langle +|) = K_0|+\rangle\langle +|K_0^\dagger + K_1|+\rangle\langle +|K_1^\dagger$$

$$= \frac{1}{2}\begin{pmatrix}1 & \sqrt{1-\gamma}\\\sqrt{1-\gamma} & 1-\gamma\end{pmatrix} + \frac{\gamma}{2}|0\rangle\langle 0|$$

$$= \frac{1}{2}\begin{pmatrix}1+\gamma & \sqrt{1-\gamma}\\\sqrt{1-\gamma} & 1-\gamma\end{pmatrix}$$

**b) Fixed points (2 pts):**

Fixed point satisfies $\mathcal{A}(\rho) = \rho$.

For $\gamma < 1$: Only fixed point is $|0\rangle\langle 0|$.
For $\gamma = 1$: All states become $|0\rangle\langle 0|$, which is the unique fixed point.

**c) Quantum capacity for γ=1/2 (3 pts):**

Coherent information: $I_c(\rho, \mathcal{A}) = S(\mathcal{A}(\rho)) - S_e(\rho, \mathcal{A})$

where $S_e$ is exchange entropy.

For amplitude damping with $\gamma = 1/2$, the quantum capacity is:
$$Q \approx 1 - H_2(1/2) = 0$$

Actually, $Q > 0$ for $\gamma < 1/2$ and $Q = 0$ for $\gamma \geq 1/2$.

At $\gamma = 1/2$: $Q = 0$ (channel is too noisy to transmit quantum information).

---

## Section C: Quantum Gates and Algorithms

### Solution C1

**a) H, T generate dense set (2 pts):**

$H$ generates $\pi/2$ rotations, $T = \begin{pmatrix}1 & 0\\0 & e^{i\pi/4}\end{pmatrix}$ generates $\pi/8$ rotations.

$HTH$ generates rotation about a different axis. The group generated is dense in SU(2) because $\pi/8$ is irrational multiple of $\pi$.

**b) Solovay-Kitaev (2 pts):**

Solovay-Kitaev theorem: Any SU(2) element can be approximated to precision $\epsilon$ using $O(\log^c(1/\epsilon))$ gates from a universal set, where $c \approx 3.97$.

For Clifford+T: approximately $3\log_2(1/\epsilon)$ T gates.

**c) T gate importance (2 pts):**

T gate cannot be implemented transversally on stabilizer codes (by Eastin-Knill theorem). It must be implemented using magic state distillation, making it the "expensive" gate in fault-tolerant computing.

---

### Solution C2

**a) Grover diffusion (2 pts):**

$$D = 2|s\rangle\langle s| - I$$

where $|s\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle$ is the uniform superposition.

**b) Optimal iterations (2 pts):**

State after $k$ iterations: $\sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|s'\rangle$

where $\sin\theta = 1/\sqrt{N}$ and $|s'\rangle$ is orthogonal to marked state.

Maximum at $(2k+1)\theta = \pi/2$, giving $k \approx \frac{\pi}{4}\sqrt{N}$.

**c) Optimality proof (3 pts):**

Polynomial method: After $T$ queries, acceptance probability is a polynomial of degree $\leq 2T$ in the input bits.

For OR (finding marked item): Must distinguish all-zeros from one-marked.

Using Chebyshev polynomial bounds, any polynomial distinguishing these cases needs degree $\Omega(\sqrt{N})$.

Therefore $T \geq \Omega(\sqrt{N})$.

---

### Solution C3

**a) Gate count (2 pts):**

QFT uses $n$ Hadamards and $n(n-1)/2$ controlled-phase gates.

Total: $O(n^2)$ gates.

**b) Gate decomposition (2 pts):**

$$\text{QFT} = \prod_{j=1}^{n}\left(H_j \prod_{k=j+1}^{n} CR_k\right)$$

where $CR_k$ is controlled-$R_{2^{k-j}}$ phase gate.

**c) Shor's algorithm (3 pts):**

In Shor's algorithm, we estimate the eigenvalue of the modular exponentiation unitary:
$$U|y\rangle = |ay \mod N\rangle$$

The eigenvalue is $e^{2\pi i s/r}$ where $r$ is the order of $a$ mod $N$.

QFT extracts $s/r$ to sufficient precision to determine $r$ via continued fractions.

---

## Section D: Quantum Complexity

### Solution D1

**a) BQP definition (3 pts):**

$L \in$ BQP if there exists a uniform family of polynomial-size quantum circuits $\{C_n\}$ such that:
- For $x \in L$: $\Pr[C_{|x|}(x) = 1] \geq 2/3$
- For $x \notin L$: $\Pr[C_{|x|}(x) = 1] \leq 1/3$

**b) BQP ⊆ PSPACE (3 pts):**

PSPACE algorithm to simulate quantum circuit:

1. Write acceptance probability as path sum:
$$\Pr[\text{accept}] = \sum_{y: y_1=1} |\langle y|C|0^n\rangle|^2$$

2. Each amplitude is sum over $d^{\text{gates}}$ paths.

3. Enumerate paths one at a time, keeping running sum.

4. Each path computation uses polynomial space.

5. Total: polynomial space.

**c) BQP vs NP (2 pts):**

Neither containment is known:
- BQP ⊆ NP is unlikely (factoring is in BQP but not known in NP)
- NP ⊆ BQP is unlikely (Grover gives only quadratic speedup for SAT)

Oracle evidence exists for separation in both directions.

---

### Solution D2

**a) k-local Hamiltonian (3 pts):**

Input: $H = \sum_i H_i$ where each $H_i$ acts on ≤ $k$ qubits, thresholds $a < b$ with $b - a \geq 1/\text{poly}(n)$.

Promise: Either $\lambda_{\min}(H) \leq a$ (YES) or $\lambda_{\min}(H) \geq b$ (NO).

Output: Determine which case.

**b) QMA-completeness (2 pts):**

$k$-local Hamiltonian is QMA-complete for $k \geq 2$.

Proven by Kitaev (k=5), improved to k=2 by Kempe-Kitaev-Regev.

**c) Relationship to SAT (2 pts):**

Local Hamiltonian is the quantum analog of MAX-SAT. Just as SAT asks about satisfiability of boolean constraints, Local Hamiltonian asks about ground state energy of quantum constraints. The ground state plays the role of the satisfying assignment.

---

## Section E: Quantum Protocols

### Solution E1

**a) Teleportation derivation (4 pts):**

Initial: $|\psi\rangle_A \otimes |\Phi^+\rangle_{BC}$

Expand in Bell basis on AB:
$$= \frac{1}{2}\sum_{ij} |\Phi_{ij}\rangle_{AB} \otimes \sigma_{ij}|\psi\rangle_C$$

Bell measurement gives outcome $ij$, Bob has $\sigma_{ij}|\psi\rangle$.

Bob applies $\sigma_{ij}^{-1}$ to recover $|\psi\rangle$.

**b) Without classical bits (2 pts):**

Bob's state is $\rho_C = \text{Tr}_{AB}(|\Psi\rangle\langle\Psi|) = I/2$.

Maximally mixed—no information about $|\psi\rangle$.

**c) Noisy teleportation fidelity (2 pts):**

For $\rho = (1-\epsilon)|\Phi^+\rangle\langle\Phi^+| + \epsilon\frac{I}{4}$:

$$F = (1-\epsilon) \cdot 1 + \epsilon \cdot \frac{1}{2} = 1 - \frac{\epsilon}{2}$$

---

### Solution E2

**a) BB84 protocol (2 pts):**

1. Alice: random bits and bases (Z or X), prepares corresponding states
2. Alice sends qubits to Bob
3. Bob: random basis measurement
4. Public basis comparison, keep matching
5. Sacrifice subset for error estimation
6. Privacy amplification

**b) Intercept-resend error rate (3 pts):**

Eve measures in random basis:
- Correct basis (prob 1/2): no error
- Wrong basis (prob 1/2): 50% error rate

When Alice-Bob bases match:
Error = (prob Eve wrong) × (prob Bob gets wrong bit) = $\frac{1}{2} \times \frac{1}{2} = 25\%$

**c) Maximum error rate (2 pts):**

Asymptotic key rate: $R = 1 - 2H(e)$

$R = 0$ when $H(e) = 0.5$, i.e., $e \approx 11\%$.

---

## Section F: Information Theory

### Solution F1

**a) Holevo bound (2 pts):**

For ensemble $\{p_x, \rho_x\}$ and measurement outcome $Y$:
$$I(X:Y) \leq S(\rho) - \sum_x p_x S(\rho_x) = \chi$$

where $\rho = \sum_x p_x \rho_x$.

**b) Holevo quantity calculation (3 pts):**

Ensemble: $\{1/2, |0\rangle\langle 0|\}$, $\{1/2, |+\rangle\langle +|\}$

$$\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|+\rangle\langle +| = \begin{pmatrix}3/4 & 1/4\\1/4 & 1/4\end{pmatrix}$$

Eigenvalues: $\frac{1 \pm 1/\sqrt{2}}{2}$

$S(\rho) \approx 0.60$ bits

$S(|0\rangle\langle 0|) = S(|+\rangle\langle +|) = 0$

$\chi = 0.60$ bits

**c) Superdense coding (3 pts):**

Bob receives Bell states $\{|\Phi^+\rangle, |\Phi^-\rangle, |\Psi^+\rangle, |\Psi^-\rangle\}$ each with prob 1/4.

$\rho = I/4$, $S(\rho) = 2$ bits

Each Bell state is pure: $S(\rho_i) = 0$

$\chi = 2 - 0 = 2$ bits ✓

---

### Solution F2

**a) Strong subadditivity (2 pts):**

$$S(ABC) + S(B) \leq S(AB) + S(BC)$$

Equivalently: $I(A:C|B) \geq 0$

**b) Subadditivity from SSA (3 pts):**

Take $C$ to be trivial (1-dimensional). Then:
- $S(C) = 0$
- $S(ABC) = S(AB)$
- $S(BC) = S(B)$

SSA becomes: $S(AB) + S(B) \leq S(AB) + S(B)$ (trivial)

Better: Use $I(A:C|B) \geq 0$ form. For trivial $C$:
$S(A|B) \leq S(A|BC) = S(A)$

This gives $S(AB) - S(B) \leq S(A)$, i.e., $S(AB) \leq S(A) + S(B)$. ✓

**c) Pure bipartite state (2 pts):**

For pure $|\psi\rangle_{AB}$:
$$S(A) = S(B)$$

This follows from Schmidt decomposition: both reduced states have the same eigenvalues.

The entropy $S(A) = S(B)$ is the entanglement entropy, measuring entanglement of the pure state.

---

## Grading Summary

| Section | Topic | Points | Your Score |
|---------|-------|--------|------------|
| A | Density Matrices | 20 | ___ |
| B | Channels | 15 | ___ |
| C | Gates/Algorithms | 20 | ___ |
| D | Complexity | 15 | ___ |
| E | Protocols | 15 | ___ |
| F | Information Theory | 15 | ___ |
| **Total** | | **100** | ___ |

**Passing: 80/100**

---

**Created:** February 9, 2026
