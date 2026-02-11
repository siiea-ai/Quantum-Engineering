# Week 145: Mathematical Framework — Problem Set

## Instructions

This problem set contains 27 qualifying-exam-level problems on the mathematical framework of quantum mechanics. Problems are organized by difficulty:

- **Level 1 (Problems 1-10):** Direct applications — should take 10-15 minutes each
- **Level 2 (Problems 11-20):** Intermediate — should take 15-25 minutes each
- **Level 3 (Problems 21-27):** Challenging — should take 25-40 minutes each

**Time yourself** to build exam stamina. On a qualifying exam, you typically have 45-60 minutes per problem.

---

## Level 1: Direct Application

### Problem 1: Inner Product Properties

Consider the inner product space $$\mathbb{C}^2$$ with standard inner product.

(a) Verify that $$|u\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$$ is normalized.

(b) Find a vector $$|v\rangle$$ orthogonal to $$|u\rangle$$ and normalize it.

(c) Express $$|w\rangle = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ in the basis $$\{|u\rangle, |v\rangle\}$$.

---

### Problem 2: Hermitian Matrices

Determine whether each matrix is Hermitian:

(a) $$A = \begin{pmatrix} 1 & i \\ -i & 2 \end{pmatrix}$$

(b) $$B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

(c) $$C = \begin{pmatrix} 1 & 1+i \\ 1-i & 0 \end{pmatrix}$$

For those that are Hermitian, find all eigenvalues and verify they are real.

---

### Problem 3: Basic Commutators

Evaluate the following commutators:

(a) $$[\hat{x}^2, \hat{p}]$$

(b) $$[\hat{x}, \hat{p}^2]$$

(c) $$[\hat{x}^2, \hat{p}^2]$$

Express your answers in terms of $$\hat{x}$$, $$\hat{p}$$, and $$\hbar$$.

---

### Problem 4: Projection Operators

Let $$|+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ and $$|-\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$.

(a) Construct the projection operators $$\hat{P}_+$$ and $$\hat{P}_-$$.

(b) Verify that $$\hat{P}_+^2 = \hat{P}_+$$ and $$\hat{P}_-^2 = \hat{P}_-$$.

(c) Show that $$\hat{P}_+ + \hat{P}_- = \hat{1}$$ and $$\hat{P}_+ \hat{P}_- = 0$$.

---

### Problem 5: Unitary Matrices

Consider the matrix $$U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$ (Hadamard gate).

(a) Verify that $$U$$ is unitary.

(b) Find the eigenvalues of $$U$$ and verify they have unit modulus.

(c) If $$|\psi\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$, show that $$\langle U\psi | U\psi \rangle = \langle \psi | \psi \rangle$$.

---

### Problem 6: Dirac Notation Practice

A system has orthonormal basis states $$|1\rangle, |2\rangle, |3\rangle$$. The state is:
$$|\psi\rangle = \frac{1}{2}|1\rangle + \frac{i}{\sqrt{2}}|2\rangle + \frac{1}{2}|3\rangle$$

(a) Verify that $$|\psi\rangle$$ is normalized.

(b) What is the probability of measuring the system in state $$|2\rangle$$?

(c) What is $$\langle\psi| 3\rangle$$?

---

### Problem 7: Expectation Values

For a spin-1/2 particle in state $$|\psi\rangle = \cos(\theta/2)|+\rangle + e^{i\phi}\sin(\theta/2)|-\rangle$$:

(a) Calculate $$\langle \hat{S}_z \rangle$$.

(b) Calculate $$\langle \hat{S}_x \rangle$$.

(c) Calculate $$\langle \hat{S}_y \rangle$$.

Use $$\hat{S}_i = \frac{\hbar}{2}\sigma_i$$ where $$\sigma_i$$ are Pauli matrices.

---

### Problem 8: Completeness Relations

Using the completeness relation $$\int |x\rangle\langle x| \, dx = \hat{1}$$:

(a) Show that $$\langle\phi|\psi\rangle = \int \phi^*(x)\psi(x) \, dx$$.

(b) Show that $$\langle\phi|\hat{A}|\psi\rangle = \int\int \phi^*(x)A(x,x')\psi(x') \, dx \, dx'$$ where $$A(x,x') = \langle x|\hat{A}|x'\rangle$$.

---

### Problem 9: Eigenvalue Problem

Find the eigenvalues and normalized eigenvectors of:
$$\hat{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

Verify that the eigenvectors are orthogonal.

---

### Problem 10: Trace Properties

For operators $$\hat{A}$$ and $$\hat{B}$$ represented by finite matrices:

(a) Show that $$\text{Tr}(\hat{A}\hat{B}) = \text{Tr}(\hat{B}\hat{A})$$.

(b) Show that $$\text{Tr}(\hat{A}^\dagger) = [\text{Tr}(\hat{A})]^*$$.

(c) If $$\hat{H}$$ is Hermitian, show that $$\text{Tr}(\hat{H})$$ is real.

---

## Level 2: Intermediate

### Problem 11: Proving Hermiticity

(a) Prove that if $$\hat{A}$$ and $$\hat{B}$$ are Hermitian, then $$\hat{A}\hat{B} + \hat{B}\hat{A}$$ is Hermitian.

(b) Prove that $$i[\hat{A}, \hat{B}]$$ is Hermitian if $$\hat{A}$$ and $$\hat{B}$$ are Hermitian.

(c) Prove that $$\hat{A}\hat{B}$$ is generally NOT Hermitian unless $$[\hat{A}, \hat{B}] = 0$$.

---

### Problem 12: Commutator with Functions

(a) For $$[\hat{x}, \hat{p}] = i\hbar$$, prove that $$[\hat{x}^n, \hat{p}] = i\hbar n\hat{x}^{n-1}$$ by induction.

(b) Show that for any polynomial $$f(\hat{x})$$:
$$[\,f(\hat{x}), \hat{p}] = i\hbar \frac{df}{d\hat{x}}$$

(c) Generalize: What is $$[\hat{x}, f(\hat{p})]$$?

---

### Problem 13: Spectral Decomposition

The Hamiltonian of a two-level system is:
$$\hat{H} = E_0 \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} + V \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

(a) Find the eigenvalues $$E_\pm$$.

(b) Find the normalized eigenstates $$|+\rangle$$ and $$|-\rangle$$.

(c) Write $$\hat{H}$$ in spectral form: $$\hat{H} = E_+|+\rangle\langle+| + E_-|-\rangle\langle-|$$.

---

### Problem 14: Uncertainty Calculation

A particle is in the state $$\psi(x) = \left(\frac{2\alpha}{\pi}\right)^{1/4} e^{-\alpha x^2}$$.

(a) Calculate $$\langle \hat{x} \rangle$$ and $$\langle \hat{x}^2 \rangle$$.

(b) Calculate $$\langle \hat{p} \rangle$$ and $$\langle \hat{p}^2 \rangle$$.

(c) Calculate $$\Delta x$$ and $$\Delta p$$, and verify that $$\Delta x \Delta p = \hbar/2$$.

---

### Problem 15: Simultaneous Eigenstates

Consider the operators $$\hat{A} = \hat{L}^2$$ and $$\hat{B} = \hat{L}_z$$.

(a) Verify that $$[\hat{L}^2, \hat{L}_z] = 0$$.

(b) The simultaneous eigenstates $$|l, m\rangle$$ satisfy $$\hat{L}^2|l,m\rangle = l(l+1)\hbar^2|l,m\rangle$$ and $$\hat{L}_z|l,m\rangle = m\hbar|l,m\rangle$$. Explain why $$m$$ ranges from $$-l$$ to $$+l$$.

(c) Why can we NOT simultaneously specify $$L_x$$, $$L_y$$, and $$L_z$$?

---

### Problem 16: Operator Functions

For the Pauli matrix $$\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$:

(a) Calculate $$e^{i\theta\sigma_z/2}$$ using the power series definition.

(b) Verify your answer by noting that $$\sigma_z^2 = I$$.

(c) Show that this is a rotation operator on the Bloch sphere.

---

### Problem 17: Change of Basis

The states $$|+\rangle_z$$ and $$|-\rangle_z$$ are eigenstates of $$\hat{S}_z$$.

(a) Find the eigenstates $$|+\rangle_x$$ and $$|-\rangle_x$$ of $$\hat{S}_x$$ in terms of $$|+\rangle_z$$ and $$|-\rangle_z$$.

(b) Write the matrix representation of $$\hat{S}_z$$ in the $$\{|+\rangle_x, |-\rangle_x\}$$ basis.

(c) Verify that the eigenvalues of $$\hat{S}_z$$ are unchanged by the basis transformation.

---

### Problem 18: Schwarz Inequality Application

(a) Prove the Schwarz inequality: $$|\langle\phi|\psi\rangle|^2 \leq \langle\phi|\phi\rangle\langle\psi|\psi\rangle$$.

(b) Use this to prove that for any Hermitian operator $$\hat{A}$$ and state $$|\psi\rangle$$:
$$|\langle\psi|\hat{A}|\psi\rangle|^2 \leq \langle\psi|\psi\rangle\langle\psi|\hat{A}^2|\psi\rangle$$

(c) When does equality hold?

---

### Problem 19: Anti-Hermitian Operators

(a) Show that if $$\hat{A}$$ is anti-Hermitian ($$\hat{A}^\dagger = -\hat{A}$$), then $$e^{\hat{A}}$$ is unitary.

(b) Conversely, show that any unitary operator can be written as $$e^{i\hat{H}}$$ for some Hermitian $$\hat{H}$$.

(c) How does this relate to time evolution $$\hat{U}(t) = e^{-i\hat{H}t/\hbar}$$?

---

### Problem 20: Position-Momentum Eigenstates

(a) Verify that $$\langle x|p\rangle = \frac{1}{\sqrt{2\pi\hbar}}e^{ipx/\hbar}$$.

(b) Using $$\int |x\rangle\langle x| dx = \hat{1}$$, derive the momentum-space wavefunction:
$$\tilde{\psi}(p) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x) dx$$

(c) Show that $$\int |\tilde{\psi}(p)|^2 dp = \int |\psi(x)|^2 dx = 1$$ (Parseval's theorem).

---

## Level 3: Challenging

### Problem 21: Generalized Uncertainty Derivation (MIT)

*This problem has appeared on MIT qualifying exams.*

(a) For Hermitian operators $$\hat{A}$$ and $$\hat{B}$$, define $$\hat{A}' = \hat{A} - \langle\hat{A}\rangle$$ and similarly for $$\hat{B}'$$. Show that:
$$[\hat{A}', \hat{B}'] = [\hat{A}, \hat{B}]$$

(b) Consider the state $$|\chi(\lambda)\rangle = (\hat{A}' + i\lambda\hat{B}')|\psi\rangle$$ for real $$\lambda$$. Show that $$\langle\chi|\chi\rangle \geq 0$$ leads to:
$$(\Delta A)^2 + \lambda^2(\Delta B)^2 - \lambda\langle\hat{C}\rangle \geq 0$$
where $$i[\hat{A}, \hat{B}] = \hat{C}$$ (which is Hermitian).

(c) By minimizing over $$\lambda$$, derive the Robertson inequality:
$$\Delta A \Delta B \geq \frac{1}{2}|\langle [\hat{A}, \hat{B}] \rangle|$$

(d) When is the inequality saturated? Find the condition on $$|\psi\rangle$$.

---

### Problem 22: Coherent States (Berkeley)

*Based on Berkeley qualifying exam problems.*

Define the creation and annihilation operators for a harmonic oscillator:
$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\hat{x} + i\frac{\hat{p}}{\sqrt{2m\omega\hbar}}, \quad \hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\hat{x} - i\frac{\hat{p}}{\sqrt{2m\omega\hbar}}$$

A coherent state $$|\alpha\rangle$$ is an eigenstate of $$\hat{a}$$: $$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$$.

(a) Show that $$[\hat{a}, \hat{a}^\dagger] = 1$$.

(b) For $$|\alpha\rangle$$, calculate $$\langle\hat{x}\rangle$$, $$\langle\hat{p}\rangle$$, $$\Delta x$$, and $$\Delta p$$.

(c) Show that coherent states are minimum uncertainty states: $$\Delta x \Delta p = \hbar/2$$.

(d) Express $$|\alpha\rangle$$ in the number basis $$\{|n\rangle\}$$.

---

### Problem 23: Baker-Campbell-Hausdorff (Caltech)

*Common on Caltech and MIT exams.*

(a) Prove the Hadamard lemma: If $$[\hat{A}, [\hat{A}, \hat{B}]] = 0$$, then:
$$e^{\hat{A}}\hat{B}e^{-\hat{A}} = \hat{B} + [\hat{A}, \hat{B}]$$

(b) Use this to show that for the displacement operator $$\hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$$:
$$\hat{D}^\dagger(\alpha)\hat{a}\hat{D}(\alpha) = \hat{a} + \alpha$$

(c) Prove that $$|\alpha\rangle = \hat{D}(\alpha)|0\rangle$$ is a coherent state.

---

### Problem 24: Density Matrix Formalism (Yale)

*From Yale qualifying exam archives.*

A spin-1/2 system is prepared in state $$|+\rangle_z$$ with probability $$p$$ and $$|-\rangle_z$$ with probability $$1-p$$.

(a) Write the density matrix $$\hat{\rho}$$ for this mixed state.

(b) Calculate $$\text{Tr}(\hat{\rho})$$ and $$\text{Tr}(\hat{\rho}^2)$$. Under what condition is the state pure?

(c) Calculate $$\langle\hat{S}_z\rangle = \text{Tr}(\hat{\rho}\hat{S}_z)$$ and $$\langle\hat{S}_x\rangle = \text{Tr}(\hat{\rho}\hat{S}_x)$$.

(d) Find the eigenvalues of $$\hat{\rho}$$ and interpret physically.

---

### Problem 25: Symmetry and Conservation (Princeton)

*Based on Princeton exam problems.*

The Hamiltonian of a particle is $$\hat{H} = \frac{\hat{p}^2}{2m} + V(|\hat{\mathbf{r}}|)$$ (central potential).

(a) Show that $$[\hat{H}, \hat{L}_z] = 0$$ by explicitly computing the commutator.

(b) Use Ehrenfest's theorem to show that $$\langle\hat{L}_z\rangle$$ is constant in time.

(c) If the system starts in an eigenstate of $$\hat{L}_z$$, what can you say about measurements of $$L_z$$ at later times?

(d) Does $$[\hat{H}, \hat{L}_x] = 0$$? What does this imply about simultaneous eigenstates?

---

### Problem 26: Stone's Theorem

*Graduate-level functional analysis.*

Stone's theorem states that every strongly continuous one-parameter unitary group $$\hat{U}(t)$$ can be written as $$\hat{U}(t) = e^{-i\hat{H}t}$$ for some self-adjoint operator $$\hat{H}$$.

(a) Verify that $$\hat{U}(t) = e^{-i\hat{H}t/\hbar}$$ satisfies the group property: $$\hat{U}(t_1)\hat{U}(t_2) = \hat{U}(t_1 + t_2)$$.

(b) Show that $$\hat{U}^\dagger(t) = \hat{U}(-t)$$.

(c) Derive the Schrödinger equation from $$i\hbar\frac{d}{dt}\hat{U}(t) = \hat{H}\hat{U}(t)$$ acting on states.

(d) Explain physically why time evolution must be unitary.

---

### Problem 27: Complete Set of Commuting Observables

*Conceptual and computational.*

For the hydrogen atom, consider the operators $$\{\hat{H}, \hat{L}^2, \hat{L}_z\}$$.

(a) Verify that all pairs commute.

(b) The eigenstates are $$|n, l, m\rangle$$. What are the allowed values of $$n$$, $$l$$, and $$m$$?

(c) How many states exist for $$n = 3$$? List them all.

(d) When spin is included, we need $$\hat{S}^2$$ and $$\hat{S}_z$$. Why can't we use $$\hat{L}_x$$ or $$\hat{S}_x$$ in a CSCO?

(e) For $$n = 2$$, counting spin, how many degenerate states are there? What perturbation lifts this degeneracy?

---

## Exam Strategy Notes

### Time Management
- Level 1: 10-15 min each (warm-up)
- Level 2: 20-25 min each (core competency)
- Level 3: 30-45 min each (challenge problems)

### Key Techniques
1. **Dirac notation:** Avoid component calculations when possible
2. **Commutator algebra:** Use identities to simplify
3. **Spectral decomposition:** Powerful for operator functions
4. **Dimensional analysis:** Check all answers
5. **Limiting cases:** Verify results in known limits

### Common Pitfalls
- Sign errors in commutators
- Forgetting the 1/2 in the uncertainty relation
- Confusing Hermitian conjugate with transpose
- Incorrectly ordering operators

---

*Problem Set for Week 145 — Mathematical Framework*
*Solutions available in Problem_Solutions.md*
