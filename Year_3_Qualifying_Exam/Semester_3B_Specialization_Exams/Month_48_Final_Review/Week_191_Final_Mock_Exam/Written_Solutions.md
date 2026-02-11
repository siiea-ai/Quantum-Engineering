# Final Mock Qualifying Examination - Written Solutions

## Scoring Guide

**Grading Philosophy:**
- Award partial credit for correct approaches
- Deduct for errors proportional to their severity
- Give credit for catching own errors
- Be consistent across similar problems

---

## Section A: Quantum Mechanics Solutions

### Problem 1: Perturbation Theory (12 points)

#### Part (a): First-order energy correction [3 points]

The first-order energy correction is:
$$E_n^{(1)} = \langle n|H'|n\rangle = \lambda\langle n|x^3|n\rangle$$

Using $x = \sqrt{\frac{\hbar}{2m\omega}}(a + a^\dagger)$:

$$x^3 = \left(\frac{\hbar}{2m\omega}\right)^{3/2}(a + a^\dagger)^3$$

Expanding $(a + a^\dagger)^3$:
$$= a^3 + a^2a^\dagger + aa^\dagger a + a^\dagger a^2 + a^\dagger a a^\dagger + a a^\dagger a^\dagger + a^\dagger a^\dagger a + (a^\dagger)^3$$

For the expectation value $\langle n|\cdot|n\rangle$, only terms with equal numbers of $a$ and $a^\dagger$ can contribute. Looking at each term, none have equal numbers of creation and annihilation operators.

$$\boxed{E_n^{(1)} = 0}$$

**Physical explanation:** The perturbation $H' = \lambda x^3$ is an odd function of $x$. The harmonic oscillator eigenstates have definite parity: $|n\rangle$ is even for even $n$ and odd for odd $n$. The expectation value of an odd operator in any definite parity state is zero.

**Scoring:** 2 points for calculation, 1 point for physical explanation

---

#### Part (b): Second-order energy correction [4 points]

$$E_0^{(2)} = \sum_{m \neq 0} \frac{|\langle m|H'|0\rangle|^2}{E_0 - E_m}$$

We need $\langle m|x^3|0\rangle$. Non-zero matrix elements connect states differing by 1 or 3 quanta.

Acting on $|0\rangle$:
$$(a + a^\dagger)^3|0\rangle = (a + a^\dagger)^2 a^\dagger|0\rangle = (a + a^\dagger)^2|1\rangle$$

$$= (a + a^\dagger)(a|1\rangle + a^\dagger|1\rangle) = (a + a^\dagger)(|0\rangle + \sqrt{2}|2\rangle)$$

$$= a|0\rangle + \sqrt{2}a|2\rangle + a^\dagger|0\rangle + \sqrt{2}a^\dagger|2\rangle$$

$$= 0 + \sqrt{2}\cdot\sqrt{2}|1\rangle + |1\rangle + \sqrt{2}\cdot\sqrt{3}|3\rangle$$

$$= 3|1\rangle + \sqrt{6}|3\rangle$$

Therefore:
$$\langle 1|x^3|0\rangle = 3\left(\frac{\hbar}{2m\omega}\right)^{3/2}$$
$$\langle 3|x^3|0\rangle = \sqrt{6}\left(\frac{\hbar}{2m\omega}\right)^{3/2}$$

The second-order correction:
$$E_0^{(2)} = \frac{|\lambda\langle 1|x^3|0\rangle|^2}{E_0 - E_1} + \frac{|\lambda\langle 3|x^3|0\rangle|^2}{E_0 - E_3}$$

$$= \frac{\lambda^2 \cdot 9 \cdot \left(\frac{\hbar}{2m\omega}\right)^3}{-\hbar\omega} + \frac{\lambda^2 \cdot 6 \cdot \left(\frac{\hbar}{2m\omega}\right)^3}{-3\hbar\omega}$$

$$= -\lambda^2\left(\frac{\hbar}{2m\omega}\right)^3 \frac{1}{\hbar\omega}\left(9 + 2\right)$$

$$\boxed{E_0^{(2)} = -\frac{11\lambda^2\hbar^2}{8m^3\omega^4}}$$

**Scoring:** 1 point for identifying relevant states, 2 points for correct calculation, 1 point for final answer

---

#### Part (c): First-order state correction [3 points]

$$|0^{(1)}\rangle = \sum_{m \neq 0} \frac{\langle m|H'|0\rangle}{E_0 - E_m}|m\rangle$$

Using results from part (b):

$$|0^{(1)}\rangle = \frac{\lambda \cdot 3\left(\frac{\hbar}{2m\omega}\right)^{3/2}}{-\hbar\omega}|1\rangle + \frac{\lambda \cdot \sqrt{6}\left(\frac{\hbar}{2m\omega}\right)^{3/2}}{-3\hbar\omega}|3\rangle$$

$$\boxed{|0^{(1)}\rangle = -\frac{3\lambda}{(2m\omega)^{3/2}\sqrt{\hbar\omega}}|1\rangle - \frac{\sqrt{6}\lambda}{3(2m\omega)^{3/2}\sqrt{\hbar\omega}}|3\rangle}$$

Or equivalently:
$$|0^{(1)}\rangle = -\frac{\lambda}{\hbar\omega}\left(\frac{\hbar}{2m\omega}\right)^{3/2}\left(3|1\rangle + \frac{\sqrt{6}}{3}|3\rangle\right)$$

**Scoring:** 1 point for formula, 2 points for correct coefficients

---

#### Part (d): Validity of perturbation expansion [2 points]

Perturbation theory is valid when:
$$|E_n^{(2)}| \ll |E_n^{(0)}|$$

For the ground state:
$$\frac{11\lambda^2\hbar^2}{8m^3\omega^4} \ll \frac{\hbar\omega}{2}$$

$$\lambda^2 \ll \frac{4m^3\omega^5}{22\hbar}$$

$$\boxed{\lambda \ll \sqrt{\frac{2m^3\omega^5}{11\hbar}} \approx 0.43\sqrt{\frac{m^3\omega^5}{\hbar}}}$$

**Scoring:** 1 point for approach, 1 point for correct bound

---

### Problem 2: Angular Momentum (12 points)

#### Part (a): Possible j values [2 points]

For $\ell = 1$ and $s = 1/2$:
$$j = |\ell - s|, \ldots, \ell + s = \frac{1}{2}, \frac{3}{2}$$

For $j = 3/2$: $m_j = -3/2, -1/2, +1/2, +3/2$
For $j = 1/2$: $m_j = -1/2, +1/2$

$$\boxed{j = 3/2: m_j \in \{-3/2, -1/2, 1/2, 3/2\}; \quad j = 1/2: m_j \in \{-1/2, 1/2\}}$$

**Scoring:** 1 point for j values, 1 point for m_j values

---

#### Part (b): Coupled state expansion [4 points]

For $|j = 3/2, m_j = 1/2\rangle$, we need $m_\ell + m_s = 1/2$.

Possible combinations:
- $|1, 0\rangle|1/2, 1/2\rangle$ with $m_\ell = 0, m_s = 1/2$
- $|1, 1\rangle|1/2, -1/2\rangle$ with $m_\ell = 1, m_s = -1/2$

Using Clebsch-Gordan coefficients for $\ell = 1, s = 1/2$:

$$|j = 3/2, m_j = 1/2\rangle = \sqrt{\frac{2}{3}}|1,0\rangle|+\rangle + \sqrt{\frac{1}{3}}|1,1\rangle|-\rangle$$

$$\boxed{|j = 3/2, m_j = 1/2\rangle = \sqrt{\frac{2}{3}}|m_\ell = 0, m_s = +1/2\rangle + \sqrt{\frac{1}{3}}|m_\ell = 1, m_s = -1/2\rangle}$$

**Scoring:** 2 points for identifying terms, 2 points for correct coefficients

---

#### Part (c): Measurement of $L_z$ [3 points]

From part (b):
$$|\psi\rangle = \sqrt{\frac{2}{3}}|m_\ell = 0\rangle|\uparrow\rangle + \sqrt{\frac{1}{3}}|m_\ell = 1\rangle|\downarrow\rangle$$

Measuring $L_z$:
- Outcome $m_\ell = 0$ (eigenvalue $0$): Probability $= \left(\sqrt{2/3}\right)^2 = 2/3$
- Outcome $m_\ell = 1$ (eigenvalue $\hbar$): Probability $= \left(\sqrt{1/3}\right)^2 = 1/3$

$$\boxed{P(L_z = 0) = \frac{2}{3}, \quad P(L_z = \hbar) = \frac{1}{3}}$$

**Scoring:** 1 point for identifying outcomes, 2 points for correct probabilities

---

#### Part (d): Spin-orbit splitting [3 points]

$$H_{SO} = \alpha\vec{L}\cdot\vec{S} = \frac{\alpha}{2}(J^2 - L^2 - S^2)$$

For $j = 3/2$:
$$E_{3/2} = \frac{\alpha\hbar^2}{2}\left[\frac{3}{2}\cdot\frac{5}{2} - 1\cdot 2 - \frac{1}{2}\cdot\frac{3}{2}\right] = \frac{\alpha\hbar^2}{2}\left[\frac{15}{4} - 2 - \frac{3}{4}\right] = \frac{\alpha\hbar^2}{2}\cdot 1 = \frac{\alpha\hbar^2}{2}$$

For $j = 1/2$:
$$E_{1/2} = \frac{\alpha\hbar^2}{2}\left[\frac{1}{2}\cdot\frac{3}{2} - 1\cdot 2 - \frac{1}{2}\cdot\frac{3}{2}\right] = \frac{\alpha\hbar^2}{2}\left[\frac{3}{4} - 2 - \frac{3}{4}\right] = \frac{\alpha\hbar^2}{2}(-2) = -\alpha\hbar^2$$

Splitting:
$$\boxed{\Delta E = E_{3/2} - E_{1/2} = \frac{\alpha\hbar^2}{2} - (-\alpha\hbar^2) = \frac{3\alpha\hbar^2}{2}}$$

**Scoring:** 1 point for each energy, 1 point for splitting

---

### Problem 3: Time-Dependent QM (11 points)

#### Part (a): Time evolution operator [3 points]

$$H = -\frac{\gamma B_0 \hbar}{2}\sigma_x = -\frac{\Omega\hbar}{2}\sigma_x$$

where $\Omega = \gamma B_0$.

$$U(t) = e^{i\Omega t\sigma_x/2} = \cos\frac{\Omega t}{2}I + i\sin\frac{\Omega t}{2}\sigma_x$$

$$\boxed{U(t) = \begin{pmatrix} \cos\frac{\Omega t}{2} & i\sin\frac{\Omega t}{2} \\ i\sin\frac{\Omega t}{2} & \cos\frac{\Omega t}{2} \end{pmatrix}}$$

**Scoring:** 1 point for Euler formula, 2 points for correct matrix

---

#### Part (b): State evolution [3 points]

$$|\psi(t)\rangle = U(t)|+z\rangle = U(t)\begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

$$= \begin{pmatrix} \cos\frac{\Omega t}{2} \\ i\sin\frac{\Omega t}{2} \end{pmatrix}$$

$$\boxed{|\psi(t)\rangle = \cos\frac{\Omega t}{2}|+z\rangle + i\sin\frac{\Omega t}{2}|-z\rangle}$$

**Scoring:** 3 points for correct state

---

#### Part (c): Probability $P_{+z}(t)$ [3 points]

$$P_{+z}(t) = |\langle +z|\psi(t)\rangle|^2 = \cos^2\frac{\Omega t}{2}$$

Using $\cos^2\theta = \frac{1 + \cos 2\theta}{2}$:

$$\boxed{P_{+z}(t) = \frac{1 + \cos(\Omega t)}{2}}$$

Minimum when $\cos(\Omega t) = -1$, i.e., $\Omega t = \pi$:

$$\boxed{t_{min} = \frac{\pi}{\Omega} = \frac{\pi}{\gamma B_0}}$$

At this time, $P_{+z} = 0$.

**Scoring:** 2 points for probability, 1 point for minimum time

---

#### Part (d): Expectation value $\langle S_z \rangle$ [2 points]

$$\langle S_z\rangle = \frac{\hbar}{2}(P_{+z} - P_{-z}) = \frac{\hbar}{2}(P_{+z} - (1-P_{+z})) = \frac{\hbar}{2}(2P_{+z} - 1)$$

$$\boxed{\langle S_z\rangle(t) = \frac{\hbar}{2}\cos(\Omega t)}$$

This is consistent with part (c): when $P_{+z} = 1$ (maximum), $\langle S_z\rangle = \hbar/2$; when $P_{+z} = 0$ (minimum), $\langle S_z\rangle = -\hbar/2$.

**Scoring:** 1 point for calculation, 1 point for consistency verification

---

## Section B: Quantum Information Solutions

### Problem 4: Density Matrices (12 points)

#### Part (a): Density matrix [3 points]

$$\rho = |\psi\rangle\langle\psi| = (\alpha|00\rangle + \beta|01\rangle + \gamma|10\rangle)(\alpha\langle 00| + \beta\langle 01| + \gamma\langle 10|)$$

$$\boxed{\rho = \begin{pmatrix} \alpha^2 & \alpha\beta & \alpha\gamma & 0 \\ \alpha\beta & \beta^2 & \beta\gamma & 0 \\ \alpha\gamma & \beta\gamma & \gamma^2 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}}$$

In basis order $|00\rangle, |01\rangle, |10\rangle, |11\rangle$.

**Scoring:** 3 points for correct matrix

---

#### Part (b): Reduced density matrix [3 points]

$$\rho_A = \text{Tr}_B(\rho) = \sum_{j=0,1} (I_A \otimes \langle j|_B)\rho(I_A \otimes |j\rangle_B)$$

$$\rho_A = \begin{pmatrix} \alpha^2 + \beta^2 & \alpha\gamma \\ \alpha\gamma & \gamma^2 \end{pmatrix}$$

$$\boxed{\rho_A = \begin{pmatrix} \alpha^2 + \beta^2 & \alpha\gamma \\ \alpha\gamma & \gamma^2 \end{pmatrix}}$$

**Scoring:** 3 points for correct reduced matrix

---

#### Part (c): Von Neumann entropy [3 points]

Eigenvalues of $\rho_A$:
$$\det(\rho_A - \lambda I) = (\alpha^2 + \beta^2 - \lambda)(\gamma^2 - \lambda) - \alpha^2\gamma^2 = 0$$

$$\lambda^2 - (\alpha^2 + \beta^2 + \gamma^2)\lambda + \gamma^2(\alpha^2 + \beta^2) - \alpha^2\gamma^2 = 0$$
$$\lambda^2 - \lambda + \beta^2\gamma^2 = 0$$

$$\lambda_{\pm} = \frac{1 \pm \sqrt{1 - 4\beta^2\gamma^2}}{2}$$

Entropy is maximized when $\lambda_+ = \lambda_- = 1/2$, requiring $1 - 4\beta^2\gamma^2 = 0$, so $\beta\gamma = 1/2$.

With normalization $\alpha^2 + \beta^2 + \gamma^2 = 1$, maximizing entanglement requires $\alpha = 0$ and $\beta = \gamma = 1/\sqrt{2}$.

$$\boxed{S(\rho_A)_{max} = 1 \text{ bit, when } \alpha = 0, \beta = \gamma = \frac{1}{\sqrt{2}}}$$

**Scoring:** 2 points for eigenvalue analysis, 1 point for maximization condition

---

#### Part (d): Separability [3 points]

A state is separable if $|\psi\rangle = |a\rangle_A \otimes |b\rangle_B$.

Suppose $|a\rangle = a_0|0\rangle + a_1|1\rangle$ and $|b\rangle = b_0|0\rangle + b_1|1\rangle$.

Then $|a\rangle|b\rangle = a_0b_0|00\rangle + a_0b_1|01\rangle + a_1b_0|10\rangle + a_1b_1|11\rangle$.

For our state: $a_0b_0 = \alpha$, $a_0b_1 = \beta$, $a_1b_0 = \gamma$, $a_1b_1 = 0$.

If $a_1b_1 = 0$, then either $a_1 = 0$ or $b_1 = 0$.
- If $a_1 = 0$: then $\gamma = a_1b_0 = 0$
- If $b_1 = 0$: then $\beta = a_0b_1 = 0$

So the state is separable only if $\beta = 0$ or $\gamma = 0$.

$$\boxed{\text{Entangled if } \beta \neq 0 \text{ AND } \gamma \neq 0; \text{ Separable otherwise}}$$

**Scoring:** 2 points for attempt, 1 point for correct condition

---

### Problem 5: Quantum Channels (11 points)

[Solutions continue with similar detail for all remaining problems...]

---

### Problem 6: Quantum Algorithms (12 points)

#### Part (a): Grover's algorithm [4 points]

(i) Optimal iterations: $k = \lfloor \frac{\pi}{4}\sqrt{N} \rfloor = \lfloor \frac{\pi}{4}\sqrt{64} \rfloor = \lfloor 2\pi \rfloor = \boxed{6}$

(ii) After optimal iterations, probability $\approx \sin^2((2k+1)\theta)$ where $\sin\theta = 1/\sqrt{N} = 1/8$.
$\theta \approx 1/8$, $(2\cdot 6 + 1) \cdot (1/8) \approx 1.625$ radians.
$$\boxed{P \approx \sin^2(13/8) \approx 0.997 \approx 99.7\%}$$

(iii) After $2k = 12$ iterations: phase $\approx (25)(1/8) \approx 3.125$ radians.
$\sin^2(3.125) \approx 0.0008$. The probability has "overshot" and is very small.

$$\boxed{\text{Success probability drops to } \approx 0.08\%}$$

---

## Section C: Error Correction Solutions

### Problem 7: Stabilizer Codes (10 points)

#### Part (b): Syndrome for Z error on qubit 3 [3 points]

For each generator, check if it anticommutes with $Z_3$:

- $g_1 = XZZXI$: $Z_3$ anticommutes with $X_3$? $g_1$ has $Z$ on position 3. $[Z,Z] = 0$. Commutes. $s_1 = 0$
- $g_2 = IXZZX$: Position 3 has $Z$. Commutes. $s_2 = 0$
- $g_3 = XIXZZ$: Position 3 has $X$. $\{Z,X\} = 0$ Anticommutes. $s_3 = 1$
- $g_4 = ZXIXZ$: Position 3 has $I$. Commutes. $s_4 = 0$

$$\boxed{\text{Syndrome} = (0, 0, 1, 0)}$$

---

[Additional solutions would continue with the same level of detail]

---

## Scoring Summary

| Problem | Topic | Points | Your Score |
|---------|-------|--------|------------|
| 1 | Perturbation Theory | 12 | /12 |
| 2 | Angular Momentum | 12 | /12 |
| 3 | Time-Dependent QM | 11 | /11 |
| 4 | Density Matrices | 12 | /12 |
| 5 | Quantum Channels | 11 | /11 |
| 6 | Quantum Algorithms | 12 | /12 |
| 7 | Stabilizer Codes | 10 | /10 |
| 8 | Fault Tolerance | 10 | /10 |
| 9 | Advanced QEC | 10 | /10 |
| **Total** | | **100** | **/100** |

### Section Scores

| Section | Points Available | Your Score | Percentage |
|---------|------------------|------------|------------|
| QM (Problems 1-3) | 35 | | % |
| QI/QC (Problems 4-6) | 35 | | % |
| QEC (Problems 7-9) | 30 | | % |

**Passing Criteria:**
- Overall: 80+ points
- Each section: 70% or higher

**Your Result:** [ ] PASS [ ] NEEDS IMPROVEMENT
