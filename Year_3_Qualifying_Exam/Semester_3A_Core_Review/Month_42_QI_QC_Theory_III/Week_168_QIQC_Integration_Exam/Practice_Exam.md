# QI/QC Theory Integration Exam

## Examination Instructions

**Time Allowed:** 3 hours
**Total Points:** 100
**Passing Score:** 80/100

### Rules
1. No notes, books, or electronic devices
2. Show all work for full credit
3. State theorems before applying them
4. Partial credit awarded for correct reasoning
5. Allocate time wisely: ~18 minutes per section

### Sections
- Section A: Density Matrices and Entanglement (20 points)
- Section B: Quantum Channels (15 points)
- Section C: Quantum Gates and Algorithms (20 points)
- Section D: Quantum Complexity (15 points)
- Section E: Quantum Protocols (15 points)
- Section F: Information Theory (15 points)

---

## Section A: Density Matrices and Entanglement (20 points)

### Problem A1 (8 points)

Consider the two-qubit state:
$$|\psi\rangle = \frac{1}{2}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{\sqrt{2}}|11\rangle$$

a) **(2 pts)** Calculate the density matrix $\rho = |\psi\rangle\langle\psi|$ (write in the computational basis).

b) **(2 pts)** Calculate the reduced density matrix $\rho_A = \text{Tr}_B(\rho)$.

c) **(2 pts)** Find the eigenvalues of $\rho_A$ and determine the von Neumann entropy $S(\rho_A)$.

d) **(2 pts)** Is $|\psi\rangle$ an entangled state? Justify your answer using the Schmidt decomposition or entanglement entropy.

### Problem A2 (6 points)

The Werner state is defined as:
$$\rho_W = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I_4}{4}$$

where $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.

a) **(2 pts)** Find the eigenvalues of $\rho_W$.

b) **(2 pts)** For what range of $p$ is $\rho_W$ entangled? (State the criterion you use.)

c) **(2 pts)** Calculate the conditional entropy $S(A|B) = S(AB) - S(B)$ for $p = 1$.

### Problem A3 (6 points)

a) **(3 pts)** State the CHSH inequality and explain its significance for entanglement detection.

b) **(3 pts)** Calculate the maximum quantum violation of CHSH for the state $|\Phi^+\rangle$. What measurement angles achieve this maximum?

---

## Section B: Quantum Channels (15 points)

### Problem B1 (8 points)

The depolarizing channel is defined as:
$$\mathcal{D}_p(\rho) = (1-p)\rho + p\frac{I}{2}$$

a) **(2 pts)** Express this channel in Kraus operator form.

b) **(3 pts)** Calculate the Choi matrix of $\mathcal{D}_p$ and verify it is positive semidefinite.

c) **(3 pts)** Determine for which values of $p$ the channel is entanglement-breaking.

### Problem B2 (7 points)

Consider the amplitude damping channel with Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

a) **(2 pts)** Calculate the output state when the input is $\rho = |+\rangle\langle +|$.

b) **(2 pts)** Find the fixed point(s) of this channel (states $\rho$ such that $\mathcal{A}(\rho) = \rho$).

c) **(3 pts)** Calculate the quantum capacity of this channel for $\gamma = 1/2$. (You may use the formula $Q = \max_\rho I_c(\rho, \mathcal{A})$ without derivation, but explain the coherent information.)

---

## Section C: Quantum Gates and Algorithms (20 points)

### Problem C1 (6 points)

a) **(2 pts)** Show that $\{H, T\}$ generates a dense set of single-qubit unitaries.

b) **(2 pts)** How many T gates are required to approximate any single-qubit unitary to precision $\epsilon$? State the relevant theorem.

c) **(2 pts)** Why is the T gate important for fault-tolerant quantum computing?

### Problem C2 (7 points)

Consider Grover's algorithm for searching a database of $N = 2^n$ items with exactly one marked item.

a) **(2 pts)** Write the Grover diffusion operator in Dirac notation.

b) **(2 pts)** How many Grover iterations are optimal? Derive this using geometric analysis.

c) **(3 pts)** Prove that the query complexity of Grover's algorithm is optimal, i.e., $\Omega(\sqrt{N})$ queries are necessary. (Sketch the polynomial method proof.)

### Problem C3 (7 points)

The Quantum Fourier Transform on $n$ qubits is:
$$\text{QFT}|j\rangle = \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}|k\rangle$$

a) **(2 pts)** How many gates does the standard QFT circuit use? Express in terms of $n$.

b) **(2 pts)** Show that QFT can be written as a product of controlled-phase gates and Hadamards.

c) **(3 pts)** In Shor's algorithm, the QFT is used for phase estimation. Explain specifically what eigenvalue is being estimated and of what unitary.

---

## Section D: Quantum Complexity (15 points)

### Problem D1 (8 points)

a) **(3 pts)** Define BQP precisely using the quantum circuit model.

b) **(3 pts)** Prove that $\text{BQP} \subseteq \text{PSPACE}$. Your proof should explain how to simulate a quantum circuit using polynomial space.

c) **(2 pts)** What is known about the relationship between BQP and NP? Is either contained in the other? What evidence exists?

### Problem D2 (7 points)

a) **(3 pts)** Define the $k$-local Hamiltonian problem and state the promise gap condition.

b) **(2 pts)** For what values of $k$ is the $k$-local Hamiltonian problem QMA-complete? State the theorem precisely.

c) **(2 pts)** Explain intuitively why the Local Hamiltonian problem is the "natural" QMA-complete problem. What is its relationship to classical satisfiability?

---

## Section E: Quantum Protocols (15 points)

### Problem E1 (8 points)

a) **(4 pts)** Derive the quantum teleportation protocol completely. Start with the initial state $|\psi\rangle_A \otimes |\Phi^+\rangle_{BC}$ and show the final state for each Bell measurement outcome.

b) **(2 pts)** Why is classical communication necessary for teleportation? What would Bob's state be without it?

c) **(2 pts)** If Alice and Bob share the noisy state $\rho = (1-\epsilon)|\Phi^+\rangle\langle\Phi^+| + \epsilon\frac{I}{4}$, what is the teleportation fidelity?

### Problem E2 (7 points)

In the BB84 protocol:

a) **(2 pts)** Describe the complete protocol, including basis reconciliation and error estimation.

b) **(3 pts)** Calculate the error rate introduced by an intercept-resend attack where Eve measures every qubit in a random basis.

c) **(2 pts)** What is the maximum tolerable error rate for secure key generation? State the formula for the asymptotic key rate.

---

## Section F: Information Theory (15 points)

### Problem F1 (8 points)

a) **(2 pts)** State the Holevo bound precisely.

b) **(3 pts)** Consider the ensemble $\{1/2, |0\rangle\langle 0|\}$, $\{1/2, |+\rangle\langle +|\}$. Calculate the Holevo quantity $\chi$.

c) **(3 pts)** In superdense coding, Alice can send 2 classical bits using 1 qubit and shared entanglement. Calculate the Holevo quantity for Bob's received states and verify it equals 2 bits.

### Problem F2 (7 points)

a) **(2 pts)** State the strong subadditivity inequality for a tripartite quantum system.

b) **(3 pts)** Prove subadditivity $S(AB) \leq S(A) + S(B)$ from strong subadditivity.

c) **(2 pts)** For a bipartite pure state $|\psi\rangle_{AB}$, what can you say about $S(A)$ and $S(B)$? How does this relate to entanglement?

---

## End of Examination

**Time Check:** You should have completed all sections within 3 hours.

**Self-Grading:** Use the Exam_Solutions.md file to grade your work. Be honest in your assessment—the goal is to identify areas for improvement before the actual qualifying exam.

---

**Total Points Available:** 100
**Passing Score:** 80
**Time Allowed:** 3 hours

**Good luck!**
