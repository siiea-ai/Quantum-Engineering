# Flash Card Sets for Final Review

## Overview

This document contains essential flash cards organized by topic for comprehensive review. Each card follows the format:

```
FRONT: Question/Prompt
BACK: Answer/Formula/Explanation
```

These cards should be used with spaced repetition software (Anki recommended) or physical cards.

---

## Set 1: Quantum Mechanics Foundations

### Card 1.1
**FRONT:** State the time-dependent Schrodinger equation.

**BACK:**
$$i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

For wave functions: $$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V\psi$$

---

### Card 1.2
**FRONT:** What is the generalized uncertainty principle?

**BACK:**
$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$$

For position-momentum: $$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

---

### Card 1.3
**FRONT:** What is the commutator of position and momentum?

**BACK:**
$$[\hat{x}, \hat{p}] = i\hbar$$

More generally: $$[x_i, p_j] = i\hbar\delta_{ij}$$

---

### Card 1.4
**FRONT:** How does an operator evolve in the Heisenberg picture?

**BACK:**
$$\frac{d\hat{A}_H}{dt} = \frac{1}{i\hbar}[\hat{A}_H, \hat{H}] + \left(\frac{\partial \hat{A}}{\partial t}\right)_H$$

If $$\hat{A}$$ has no explicit time dependence and $$[\hat{A}, \hat{H}] = 0$$, then $$\hat{A}$$ is a constant of motion.

---

### Card 1.5
**FRONT:** State the completeness relation for a discrete orthonormal basis.

**BACK:**
$$\sum_n |n\rangle\langle n| = \hat{I}$$

For continuous basis: $$\int |x\rangle\langle x| dx = \hat{I}$$

---

### Card 1.6
**FRONT:** What are the eigenvalues of the harmonic oscillator Hamiltonian?

**BACK:**
$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, \ldots$$

Ground state energy (zero-point energy): $$E_0 = \frac{\hbar\omega}{2}$$

---

### Card 1.7
**FRONT:** How do the ladder operators act on harmonic oscillator states?

**BACK:**
$$\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$$
$$\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

With: $$[\hat{a}, \hat{a}^\dagger] = 1$$

---

### Card 1.8
**FRONT:** State the angular momentum commutation relations.

**BACK:**
$$[J_i, J_j] = i\hbar\epsilon_{ijk}J_k$$

Explicitly:
- $$[J_x, J_y] = i\hbar J_z$$
- $$[J_y, J_z] = i\hbar J_x$$
- $$[J_z, J_x] = i\hbar J_y$$

Also: $$[J^2, J_i] = 0$$ for all $$i$$

---

### Card 1.9
**FRONT:** What are the eigenvalues of $$J^2$$ and $$J_z$$?

**BACK:**
$$J^2|j,m\rangle = \hbar^2 j(j+1)|j,m\rangle$$
$$J_z|j,m\rangle = \hbar m|j,m\rangle$$

Where: $$j = 0, \frac{1}{2}, 1, \frac{3}{2}, \ldots$$ and $$m = -j, -j+1, \ldots, j-1, j$$

---

### Card 1.10
**FRONT:** Write the Pauli matrices.

**BACK:**
$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

Properties:
- $$\sigma_i^2 = I$$
- $$\sigma_i\sigma_j = i\epsilon_{ijk}\sigma_k$$ (for $$i \neq j$$)
- $$\{\sigma_i, \sigma_j\} = 2\delta_{ij}I$$

---

## Set 2: Perturbation Theory

### Card 2.1
**FRONT:** First-order energy correction in non-degenerate perturbation theory?

**BACK:**
$$E_n^{(1)} = \langle n^{(0)}|\hat{H}'|n^{(0)}\rangle$$

The first-order correction is the expectation value of the perturbation in the unperturbed state.

---

### Card 2.2
**FRONT:** Second-order energy correction formula?

**BACK:**
$$E_n^{(2)} = \sum_{m \neq n} \frac{|\langle m^{(0)}|\hat{H}'|n^{(0)}\rangle|^2}{E_n^{(0)} - E_m^{(0)}}$$

Note: For ground state, all terms are negative (energy lowered).

---

### Card 2.3
**FRONT:** First-order state correction formula?

**BACK:**
$$|n^{(1)}\rangle = \sum_{m \neq n} \frac{\langle m^{(0)}|\hat{H}'|n^{(0)}\rangle}{E_n^{(0)} - E_m^{(0)}} |m^{(0)}\rangle$$

---

### Card 2.4
**FRONT:** How do you handle degenerate perturbation theory?

**BACK:**
1. Construct perturbation matrix $$W_{ij} = \langle i^{(0)}|H'|j^{(0)}\rangle$$ in degenerate subspace
2. Diagonalize $$W$$ to find eigenvalues (first-order corrections) and eigenvectors ("good" states)
3. The good states are the correct zeroth-order states

---

### Card 2.5
**FRONT:** State Fermi's Golden Rule.

**BACK:**
$$\Gamma_{i \to f} = \frac{2\pi}{\hbar}|\langle f|\hat{H}'|i\rangle|^2 \rho(E_f)$$

Transition rate from initial state $$|i\rangle$$ to final states $$|f\rangle$$ with density of states $$\rho(E_f)$$.

---

### Card 2.6
**FRONT:** What is the adiabatic theorem?

**BACK:**
If a system starts in the $$n$$-th eigenstate and the Hamiltonian changes slowly enough, the system remains in the $$n$$-th instantaneous eigenstate.

"Slowly enough" means: $$\frac{|\langle m|\dot{H}|n\rangle|}{|E_m - E_n|^2} \ll 1$$ for all $$m \neq n$$

---

## Set 3: Quantum Information

### Card 3.1
**FRONT:** Define the density matrix for a pure state.

**BACK:**
$$\rho = |\psi\rangle\langle\psi|$$

Properties of density matrices:
1. $$\rho = \rho^\dagger$$ (Hermitian)
2. $$\text{Tr}(\rho) = 1$$ (normalized)
3. $$\rho \geq 0$$ (positive semi-definite)
4. For pure states: $$\rho^2 = \rho$$, $$\text{Tr}(\rho^2) = 1$$

---

### Card 3.2
**FRONT:** What is the Bloch sphere representation of a qubit?

**BACK:**
$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$$

where $$\vec{r} = (r_x, r_y, r_z)$$ is the Bloch vector.

- Pure states: $$|\vec{r}| = 1$$ (on surface)
- Mixed states: $$|\vec{r}| < 1$$ (inside sphere)
- Maximally mixed: $$\vec{r} = 0$$ (center)

---

### Card 3.3
**FRONT:** Define the Von Neumann entropy.

**BACK:**
$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

where $$\lambda_i$$ are eigenvalues of $$\rho$$.

Properties:
- $$S \geq 0$$
- $$S = 0$$ iff pure state
- $$S_{max} = \log_2 d$$ for maximally mixed state in $$d$$ dimensions

---

### Card 3.4
**FRONT:** Write the four Bell states.

**BACK:**
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

These form a maximally entangled basis for two qubits.

---

### Card 3.5
**FRONT:** What is the CHSH inequality and its quantum violation?

**BACK:**
Classical bound: $$|S| \leq 2$$

where $$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$

Quantum maximum (Tsirelson bound): $$|S| \leq 2\sqrt{2} \approx 2.83$$

Achieved with Bell states and appropriate measurement angles.

---

### Card 3.6
**FRONT:** How does quantum teleportation work?

**BACK:**
1. Alice and Bob share Bell state $$|\Phi^+\rangle_{AB}$$
2. Alice has unknown state $$|\psi\rangle_C$$ to teleport
3. Alice performs Bell measurement on qubits C and A
4. Alice sends 2 classical bits (measurement result) to Bob
5. Bob applies correction: $$I, X, Z,$$ or $$XZ$$ based on result
6. Bob now has $$|\psi\rangle$$

No faster-than-light communication: classical bits required!

---

### Card 3.7
**FRONT:** What is the no-cloning theorem?

**BACK:**
It is impossible to create a perfect copy of an arbitrary unknown quantum state.

Proof: If $$U|0\rangle|\psi\rangle = |\psi\rangle|\psi\rangle$$ for all $$|\psi\rangle$$, then by linearity for superposition $$|\phi\rangle = \alpha|\psi_1\rangle + \beta|\psi_2\rangle$$, cloning would require non-linear evolution.

---

### Card 3.8
**FRONT:** Define quantum fidelity between two states.

**BACK:**
For general states:
$$F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

For pure states:
$$F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2$$

Properties:
- $$0 \leq F \leq 1$$
- $$F = 1$$ iff states identical
- Symmetric: $$F(\rho, \sigma) = F(\sigma, \rho)$$

---

## Set 4: Quantum Gates and Circuits

### Card 4.1
**FRONT:** Write the Hadamard gate and its action on computational basis.

**BACK:**
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$H|0\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$
$$H|1\rangle = |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

Note: $$H^2 = I$$

---

### Card 4.2
**FRONT:** Write the CNOT gate matrix.

**BACK:**
$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

Action: $$|a,b\rangle \to |a, a \oplus b\rangle$$

First qubit is control, second is target.

---

### Card 4.3
**FRONT:** What gates form a universal gate set?

**BACK:**
Several options:
1. $$\{H, T, \text{CNOT}\}$$ - most common
2. $$\{H, \text{Toffoli}\}$$
3. Any entangling 2-qubit gate + all single-qubit gates

The Clifford group $$\{H, S, \text{CNOT}\}$$ alone is NOT universal (efficiently simulable by Gottesman-Knill).

---

### Card 4.4
**FRONT:** What is the T gate and why is it important?

**BACK:**
$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

Importance:
- Completes universal gate set when added to Clifford
- Cannot be implemented transversally in stabilizer codes
- Requires magic state distillation for fault tolerance
- $$T^2 = S$$ (phase gate)

---

### Card 4.5
**FRONT:** Write the rotation gates $$R_x(\theta), R_y(\theta), R_z(\theta)$$.

**BACK:**
$$R_x(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}X = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_y(\theta) = e^{-i\theta Y/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

---

## Set 5: Quantum Algorithms

### Card 5.1
**FRONT:** What is the complexity of Grover's search algorithm?

**BACK:**
$$O(\sqrt{N})$$ queries to find marked item among $$N$$ items.

Optimal number of iterations: $$k \approx \frac{\pi}{4}\sqrt{N}$$

Quadratic speedup over classical $$O(N)$$.

---

### Card 5.2
**FRONT:** What is the complexity of Shor's algorithm?

**BACK:**
$$O((\log N)^2 (\log \log N)(\log \log \log N))$$

for factoring an $$N$$-bit number.

Exponential speedup over best known classical algorithm.

Key insight: Reduces factoring to period finding, solved efficiently by QFT.

---

### Card 5.3
**FRONT:** Write the Quantum Fourier Transform formula.

**BACK:**
$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i jk/N}|k\rangle$$

In circuit form (for $$n$$ qubits):
$$|j_1 j_2 \cdots j_n\rangle \to \frac{1}{2^{n/2}}(|0\rangle + e^{2\pi i 0.j_n}|1\rangle) \otimes \cdots \otimes (|0\rangle + e^{2\pi i 0.j_1 j_2 \cdots j_n}|1\rangle)$$

---

### Card 5.4
**FRONT:** Describe the Grover diffusion operator.

**BACK:**
$$D = 2|s\rangle\langle s| - I$$

where $$|s\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle$$ is the uniform superposition.

In computational basis: $$D_{ij} = \frac{2}{N} - \delta_{ij}$$

Effect: Reflection about the mean amplitude.

---

### Card 5.5
**FRONT:** What problem does Simon's algorithm solve?

**BACK:**
Given oracle access to $$f: \{0,1\}^n \to \{0,1\}^n$$ with promise that $$f(x) = f(y) \iff x \oplus y \in \{0^n, s\}$$ for some hidden $$s$$, find $$s$$.

Complexity: $$O(n)$$ quantum queries vs $$O(2^{n/2})$$ classical.

Exponential speedup! Precursor to Shor's algorithm.

---

## Set 6: Quantum Error Correction

### Card 6.1
**FRONT:** State the Knill-Laflamme error correction conditions.

**BACK:**
A code with projector $$P$$ onto code space can correct errors $$\{E_i\}$$ iff:

$$PE_i^\dagger E_j P = \alpha_{ij} P$$

for some Hermitian matrix $$\alpha$$.

Equivalently: errors are either detectable or act identically on code space.

---

### Card 6.2
**FRONT:** What do the parameters $$[[n, k, d]]$$ mean for a quantum code?

**BACK:**
- $$n$$ = number of physical qubits
- $$k$$ = number of logical (encoded) qubits
- $$d$$ = code distance (minimum weight of undetectable error)

Can correct up to $$t = \lfloor(d-1)/2\rfloor$$ errors.

---

### Card 6.3
**FRONT:** Define the stabilizer group and stabilizer state.

**BACK:**
Stabilizer group $$S$$: Abelian subgroup of Pauli group with $$-I \notin S$$

Stabilizer state: $$|\psi\rangle$$ such that $$g|\psi\rangle = |\psi\rangle$$ for all $$g \in S$$

For $$n$$ qubits with $$k$$ encoded qubits: $$|S| = 2^{n-k}$$ (need $$n-k$$ independent generators)

---

### Card 6.4
**FRONT:** What is the threshold theorem?

**BACK:**
If physical error rate $$p < p_{th}$$ (threshold), then arbitrarily low logical error rates can be achieved with polynomial overhead.

Typical thresholds:
- Concatenated codes: $$p_{th} \sim 10^{-4} - 10^{-2}$$
- Surface code: $$p_{th} \sim 1\%$$

---

### Card 6.5
**FRONT:** What is magic state distillation?

**BACK:**
Protocol to create high-fidelity "magic states" from many noisy copies.

Magic state for T gate: $$|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

Process:
1. Start with many noisy $$|T\rangle$$ states
2. Use Clifford operations (which are transversal/cheap)
3. Output fewer, higher-fidelity magic states
4. Use for non-Clifford gate implementation

---

### Card 6.6
**FRONT:** Write the stabilizer generators for the 7-qubit Steane code.

**BACK:**
X-type stabilizers:
- $$g_1 = IIIXXXX$$
- $$g_2 = IXXIIXX$$
- $$g_3 = XIXIXIX$$

Z-type stabilizers:
- $$g_4 = IIIZZZZ$$
- $$g_5 = IZZIIZZ$$
- $$g_6 = ZIZIZIZ$$

This is a CSS code based on the classical [7,4,3] Hamming code.

---

### Card 6.7
**FRONT:** How do you find the syndrome for an error?

**BACK:**
1. For each stabilizer generator $$g_i$$, compute $$s_i = 0$$ if error commutes with $$g_i$$, $$s_i = 1$$ if anticommutes
2. Syndrome vector $$\vec{s} = (s_1, s_2, \ldots, s_{n-k})$$

Example: For Steane code, X error on qubit 5:
- Anticommutes with $$g_4, g_5$$ (Z-type generators with Z on qubit 5)
- Syndrome points to error location in binary

---

### Card 6.8
**FRONT:** What is the surface code and why is it important?

**BACK:**
2D topological code on a square lattice:
- X-stabilizers on faces (plaquettes)
- Z-stabilizers on vertices

Key properties:
- High threshold (~1%)
- Local stabilizer measurements
- Scalable to large distances
- Planar layout (practical for hardware)

Distance $$d$$ requires $$O(d^2)$$ physical qubits for 1 logical qubit.

---

## Set 7: Hardware Platforms

### Card 7.1
**FRONT:** What is a transmon qubit?

**BACK:**
Superconducting qubit operating in regime $$E_J/E_C \gg 1$$

- $$E_J$$ = Josephson energy
- $$E_C$$ = charging energy

Properties:
- Reduced sensitivity to charge noise (vs charge qubit)
- Slightly reduced anharmonicity
- Typical $$T_1, T_2 \sim 10-100 \mu s$$
- Gate times $$\sim 10-50$$ ns

---

### Card 7.2
**FRONT:** How do trapped ion quantum computers implement two-qubit gates?

**BACK:**
Molmer-Sorensen (MS) gate or equivalent:

$$U_{MS} = \exp\left(-i\frac{\pi}{4}\sigma_x \otimes \sigma_x\right)$$

Mechanism:
1. Apply bichromatic laser field
2. Creates spin-dependent force via motional coupling
3. Ions follow closed loop in phase space
4. Accumulates geometric phase dependent on spin state

High fidelity >99.9% demonstrated.

---

### Card 7.3
**FRONT:** What is the Rydberg blockade mechanism?

**BACK:**
When two neutral atoms are both excited to Rydberg states (high principal quantum number $$n$$):

$$U_{dd} = \frac{C_6}{r^6}$$ (van der Waals interaction)

This large interaction energy blocks double excitation within blockade radius $$r_b$$.

Enables two-qubit gates:
- If one atom in $$|1\rangle$$, other cannot be excited
- Creates controlled-phase type operations

---

## Review Schedule

### Spaced Repetition Intervals
- New cards: Review daily for first week
- Known cards: 1 day → 3 days → 7 days → 14 days → 30 days
- Failed cards: Reset to 1 day

### Daily Review Target
- 50-100 cards per session
- 2-3 sessions per day during gap remediation week

### Priority Order
1. Cards for identified weak areas (from Gap Analysis)
2. Formula cards
3. Conceptual cards
4. Hardware/application cards

---

**Total Cards in This Set:** 40 core cards
**Recommended Software:** Anki with LaTeX support
**Daily Review Time:** 30-45 minutes per session
