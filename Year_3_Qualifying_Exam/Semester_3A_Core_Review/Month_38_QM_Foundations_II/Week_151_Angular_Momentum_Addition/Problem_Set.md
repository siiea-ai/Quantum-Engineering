# Week 151: Angular Momentum Addition - Problem Set

## Instructions

- **Total Problems:** 27
- **Recommended Time:** 4-5 hours (spread across the week)
- **Difficulty Levels:** Direct Application (D), Intermediate (I), Challenging (C)
- **Exam Conditions:** For problems marked with *, attempt under timed conditions (15-20 min each)

---

## Part A: Basic Addition and Counting (Problems 1-6)

### Problem 1 (D)
What are the allowed values of total angular momentum $j$ when combining:
(a) $j_1 = 2$ and $j_2 = 1$
(b) $j_1 = 3/2$ and $j_2 = 1$
(c) $j_1 = 5/2$ and $j_2 = 3/2$
(d) $j_1 = 2$ and $j_2 = 2$

### Problem 2 (D)
Verify the dimension count $(2j_1+1)(2j_2+1) = \sum_j (2j+1)$ for:
(a) Two spin-1 particles
(b) One spin-3/2 and one spin-1/2 particle
(c) One spin-2 and one spin-1 particle

### Problem 3 (D)
For a state with $j_1 = 1$, $m_1 = 0$ and $j_2 = 1/2$, $m_2 = 1/2$:
(a) What is the value of $m$ for the total angular momentum?
(b) What are the possible values of $j$ for this state?

### Problem 4 (I)
An electron in the $n=3$, $l=2$ state (3d orbital):
(a) What are the possible values of total angular momentum $j$?
(b) How many distinct states are there for each value of $j$?
(c) What is the total degeneracy including spin?

### Problem 5 (I)*
Three spin-1/2 particles are coupled. What are the allowed values of total angular momentum $j$? Explain your reasoning step by step.

### Problem 6 (I)
A particle has orbital angular momentum $l=3$ and spin $s=1$.
(a) What are all possible values of total $j$?
(b) What term symbols describe these states?
(c) Which state has the largest and smallest $j$?

---

## Part B: Clebsch-Gordan Coefficients - Basic (Problems 7-12)

### Problem 7 (D)
For two spin-1/2 particles, write down all four basis states in both the uncoupled $|m_1, m_2\rangle$ and coupled $|j, m\rangle$ bases.

### Problem 8 (D)
Calculate the following CG coefficients directly:
(a) $\langle 1,1; 1,1 | 2,2 \rangle$
(b) $\langle 1,1; 1,0 | 2,1 \rangle$
(c) $\langle 1,0; 1,0 | 2,0 \rangle$

### Problem 9 (I)*
For $j_1 = 1$ and $j_2 = 1/2$, express the coupled state $|j=3/2, m=1/2\rangle$ in terms of uncoupled states.

### Problem 10 (I)
Using the CG coefficient formula for adding $j_2 = 1/2$, find:
(a) $|l + 1/2, m\rangle$ for $l=1$, $m=1/2$
(b) $|l - 1/2, m\rangle$ for $l=1$, $m=1/2$

### Problem 11 (I)*
For two spin-1 particles ($j_1 = j_2 = 1$):
(a) Express $|j=2, m=1\rangle$ in the uncoupled basis
(b) Express $|j=1, m=1\rangle$ in the uncoupled basis
(c) Verify these are orthogonal

### Problem 12 (I)
Prove that for two spin-1/2 particles, the singlet state $|0,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$ is antisymmetric under particle exchange.

---

## Part C: Clebsch-Gordan Coefficients - Advanced (Problems 13-17)

### Problem 13 (I)*
Using the recursion relation, derive the CG coefficients for $|j=1, m=0\rangle$ when coupling two spin-1 particles.

### Problem 14 (C)
For $j_1 = 3/2$ and $j_2 = 1/2$:
(a) List all allowed $j$ values
(b) Calculate all CG coefficients for $m = 1$
(c) Verify orthonormality of your results

### Problem 15 (C)*
An electron in a hydrogen atom is in the state:
$$|\psi\rangle = \frac{1}{\sqrt{2}}|l=1, m_l=0\rangle|s=1/2, m_s=1/2\rangle + \frac{1}{\sqrt{2}}|l=1, m_l=1\rangle|s=1/2, m_s=-1/2\rangle$$

(a) What values of $j$ and $m_j$ are possible upon measuring $\mathbf{J}^2$ and $J_z$?
(b) Calculate the probability for each outcome.

### Problem 16 (C)
Prove the symmetry property:
$$\langle j_1,m_1;j_2,m_2|j,m\rangle = (-1)^{j_1+j_2-j}\langle j_2,m_2;j_1,m_1|j,m\rangle$$

### Problem 17 (C)*
For the addition $j_1 = 2$ and $j_2 = 1$:
(a) Construct the $|j=3, m=2\rangle$ state using the lowering operator method
(b) Find all CG coefficients with $m=2$

---

## Part D: Spin-Orbit Coupling (Problems 18-22)

### Problem 18 (D)
Calculate $\langle\mathbf{L}\cdot\mathbf{S}\rangle$ for an electron in:
(a) The $^2P_{3/2}$ state
(b) The $^2P_{1/2}$ state
(c) The $^2D_{5/2}$ state

### Problem 19 (I)*
For a hydrogen atom in the $n=2$ level:
(a) List all possible $(l, j)$ combinations
(b) Calculate $\langle\mathbf{L}\cdot\mathbf{S}\rangle$ for each
(c) Which state has the lowest energy due to spin-orbit coupling?

### Problem 20 (I)
The spin-orbit coupling Hamiltonian is $H_{SO} = A\mathbf{L}\cdot\mathbf{S}$ where $A > 0$.
(a) For $l = 2$, find the energy splitting between the $j = 5/2$ and $j = 3/2$ levels
(b) Express this splitting in terms of $A$ and $\hbar$

### Problem 21 (C)*
An atom has a single valence electron in the $4f$ orbital ($l=3$).
(a) What are the possible values of $j$?
(b) Calculate the Land\'e g-factor: $g_J = 1 + \frac{j(j+1) + s(s+1) - l(l+1)}{2j(j+1)}$
(c) In a magnetic field $B$, what is the energy shift $\Delta E = g_J \mu_B m_j B$ for each level?

### Problem 22 (C)
Consider an electron with $l=2$ in a spin-orbit potential plus a magnetic field:
$$H = A\mathbf{L}\cdot\mathbf{S} + \mu_B B(L_z + 2S_z)$$

(a) What are the good quantum numbers when $A \gg \mu_B B$?
(b) What are they when $\mu_B B \gg A$?
(c) Calculate the energy levels in both limits

---

## Part E: Selection Rules and Applications (Problems 23-27)

### Problem 23 (D)
Which of the following electric dipole transitions are allowed?
(a) $^2S_{1/2} \to {}^2P_{1/2}$
(b) $^2S_{1/2} \to {}^2P_{3/2}$
(c) $^2S_{1/2} \to {}^2D_{3/2}$
(d) $^2P_{1/2} \to {}^2P_{3/2}$
(e) $^1S_0 \to {}^1S_0$

### Problem 24 (I)*
For the hydrogen Lyman-$\alpha$ transition ($2p \to 1s$):
(a) What are the initial and final term symbols?
(b) List all allowed transitions and their $\Delta m_j$ values
(c) How many distinct spectral lines would you see without fine structure? With fine structure?

### Problem 25 (I)
The sodium D-lines arise from $3p \to 3s$ transitions.
(a) What are the term symbols for the initial and final states?
(b) How many distinct transitions are allowed?
(c) What causes the doublet structure?

### Problem 26 (C)*
Consider a multi-electron atom with configuration $np^2$ (two $p$ electrons).
(a) Using the addition rules, what values of total $L$ are possible?
(b) What values of total $S$ are possible?
(c) List all allowed term symbols including $J$ values
(d) Which terms are allowed by the Pauli principle for equivalent electrons?

### Problem 27 (C)* - Qualifying Exam Style
An atom has two valence electrons, one in a $3d$ state and one in a $4s$ state.

(a) What are the possible term symbols for this configuration? List all of them.

(b) According to Hund's rules, which term has the lowest energy?

(c) If spin-orbit coupling is included, how does the ground term split?

(d) For the ground state term, calculate the Landé g-factor.

(e) In a weak magnetic field, how many Zeeman sublevels does the ground state have?

---

## Bonus Problems

### Bonus 1
The Wigner 3j symbol is defined as:
$$\begin{pmatrix} j_1 & j_2 & j \\ m_1 & m_2 & m \end{pmatrix} = \frac{(-1)^{j_1-j_2-m}}{\sqrt{2j+1}}\langle j_1,m_1;j_2,m_2|j,-m\rangle$$

Prove the symmetry:
$$\begin{pmatrix} j_1 & j_2 & j \\ m_1 & m_2 & m \end{pmatrix} = \begin{pmatrix} j_2 & j & j_1 \\ m_2 & m & m_1 \end{pmatrix}$$

### Bonus 2
For three spin-1/2 particles, there are two $j=1/2$ multiplets. Express both $|j=1/2, m=1/2\rangle$ states in terms of the uncoupled basis $|m_1, m_2, m_3\rangle$.

### Bonus 3
Derive the Land\'e g-factor formula using the projection theorem:
$$g_J = g_L\frac{j(j+1) + l(l+1) - s(s+1)}{2j(j+1)} + g_S\frac{j(j+1) + s(s+1) - l(l+1)}{2j(j+1)}$$

---

## Answer Key (Quick Reference)

| Problem | Key Answer |
|---------|------------|
| 1a | $j = 1, 2, 3$ |
| 4a | $j = 3/2, 5/2$ |
| 5 | $j = 1/2, 1/2, 3/2$ |
| 9 | See solutions |
| 18a | $\frac{\hbar^2}{2}$ |
| 18b | $-\hbar^2$ |
| 19c | $^2S_{1/2}$ |
| 23 | a,b allowed; c,d,e forbidden |
| 26d | $^1S_0$, $^1D_2$, $^3P_0$, $^3P_1$, $^3P_2$ |

---

**Detailed solutions in Problem_Solutions.md**
