# Month 45: Hardware & Algorithms Practice Exam

## Exam Instructions

**Duration:** 3 hours (180 minutes)
**Total Points:** 100

**Rules:**
- Closed book, closed notes
- Calculator allowed for numerical calculations
- Show all work for partial credit
- Write clearly and organize your answers

**Time Allocation Suggestion:**
- Part A: 30 minutes
- Part B: 60 minutes
- Part C: 60 minutes
- Part D: 30 minutes

---

# Part A: Short Answer (20 points)

*Answer each question in 2-4 sentences. 2 points each.*

## Question A1
What is the physical origin of anharmonicity in the transmon qubit, and why is it essential for qubit operation?

---

## Question A2
Explain the Rydberg blockade mechanism in one paragraph.

---

## Question A3
Why does the transmon operate in the regime $$E_J/E_C \gg 1$$ rather than $$E_J/E_C \sim 1$$?

---

## Question A4
What is the Lamb-Dicke parameter, and what condition defines the Lamb-Dicke regime?

---

## Question A5
Describe the key difference between GKP and cat qubit error correction strategies.

---

## Question A6
What is a barren plateau in the context of variational quantum algorithms?

---

## Question A7
Explain why the Mølmer-Sørensen gate is robust to thermal motion of the ions.

---

## Question A8
What is zero-noise extrapolation, and when does it fail?

---

## Question A9
Compare the connectivity of superconducting and trapped ion quantum computers.

---

## Question A10
What does the approximation ratio measure in QAOA, and what is its value for MaxCut at $$p=1$$ on 3-regular graphs?

---

# Part B: Derivations (30 points)

*Show all steps clearly. Partial credit for correct approach.*

## Question B1 (10 points)
**Transmon Hamiltonian**

Starting from a parallel LC circuit with a Josephson junction of critical current $$I_c$$ and shunt capacitance $$C$$:

(a) Write the classical Lagrangian in terms of the node flux $$\Phi$$. (3 points)

(b) Find the conjugate momentum and derive the classical Hamiltonian. (3 points)

(c) Quantize the Hamiltonian and express it in terms of $$E_J$$, $$E_C$$, and the number operator $$\hat{n}$$. (4 points)

---

## Question B2 (10 points)
**Rydberg Blockade Radius**

(a) The van der Waals interaction between two Rydberg atoms is $$V(R) = C_6/R^6$$. For two atoms driven by a laser with Rabi frequency $$\Omega$$, derive the blockade radius $$R_b$$. (4 points)

(b) For Rb atoms in the $$|70S\rangle$$ state with $$C_6/h = 500$$ GHz·μm$$^6$$ and $$\Omega/2\pi = 2$$ MHz, calculate $$R_b$$. (3 points)

(c) How many atoms can fit in a blockade sphere if arranged on a simple cubic lattice with 4 μm spacing? (3 points)

---

## Question B3 (10 points)
**QAOA for MaxCut**

Consider the triangle graph with vertices 1, 2, 3 and edges (1,2), (2,3), (1,3).

(a) Write the cost Hamiltonian $$\hat{H}_C$$ for MaxCut on this graph. (3 points)

(b) Write the full QAOA ansatz for $$p=1$$. (3 points)

(c) The optimal cut value is 2. If QAOA at $$p=1$$ achieves an expectation value of 1.5, what is the approximation ratio? (2 points)

(d) What is the physical interpretation of the mixer Hamiltonian $$\hat{H}_M = \sum_i X_i$$? (2 points)

---

# Part C: Problem Solving (30 points)

*Show all calculations. Include units where appropriate.*

## Question C1 (10 points)
**Circuit QED Parameters**

A transmon qubit with frequency $$\omega_q/2\pi = 5.0$$ GHz and anharmonicity $$\alpha/2\pi = -250$$ MHz is coupled to a resonator with frequency $$\omega_r/2\pi = 7.0$$ GHz. The coupling strength is $$g/2\pi = 100$$ MHz.

(a) Is the system in the dispersive regime? Calculate the relevant ratio. (2 points)

(b) Calculate the dispersive shift $$\chi$$. (3 points)

(c) If the resonator linewidth is $$\kappa/2\pi = 2$$ MHz, calculate $$\chi/\kappa$$. Is this a good operating point for readout? (2 points)

(d) Calculate the Purcell-limited T1 using $$T_1^{Purcell} = \Delta^2/(\kappa g^2)$$. (3 points)

---

## Question C2 (10 points)
**VQE Error Analysis**

A VQE circuit has 15 single-qubit gates and 10 two-qubit gates. The device has single-qubit gate error $$\epsilon_1 = 0.1\%$$ and two-qubit gate error $$\epsilon_2 = 1\%$$.

(a) Estimate the total gate error probability. (2 points)

(b) If the circuit runs for 2 μs and T2 = 50 μs, estimate the dephasing error. (2 points)

(c) You apply ZNE with noise scaling factors 1, 2, 3. The measured energies are:
$$E(1) = -1.05$$ Ha, $$E(2) = -0.95$$ Ha, $$E(3) = -0.85$$ Ha.
Using linear extrapolation, estimate $$E(0)$$. (3 points)

(d) If the true ground state energy is -1.20 Ha, what is the remaining error after ZNE? Suggest a reason for this error. (3 points)

---

## Question C3 (10 points)
**Platform Comparison**

You need to run a quantum algorithm with the following requirements:
- 50 qubits
- Circuit depth: 200 two-qubit gates
- Target output fidelity: > 50%

Available platforms:
- **Platform A (Superconducting):** 100 qubits, 2-qubit error 0.5%, nearest-neighbor connectivity
- **Platform B (Trapped Ion):** 32 qubits, 2-qubit error 0.1%, all-to-all connectivity
- **Platform C (Neutral Atom):** 200 qubits, 2-qubit error 2%, reconfigurable connectivity

(a) For Platform A, estimate the circuit depth after accounting for SWAP gates (assume average distance 5 between interacting qubits). (3 points)

(b) Calculate the expected output fidelity for each platform assuming:
- Platform A: effective depth 600 gates at 0.5% error
- Platform B: 200 gates at 0.1% error (but only 32 qubits)
- Platform C: 200 gates at 2% error
(4 points)

(c) Which platform would you choose and why? What error mitigation would you recommend? (3 points)

---

# Part D: Essay/Analysis (20 points)

*Write a well-organized essay of 400-600 words.*

## Question D1 (20 points)

**Prompt:**

You are advising a chemistry research group that wants to use quantum computing to study a transition metal catalyst with 30 electrons in the active site. They have access to both a 100-qubit superconducting quantum computer (IBM) and a 30-qubit trapped ion system (Quantinuum).

Discuss the following:

1. What is the minimum number of qubits needed after choosing an appropriate active space?

2. Which hardware platform would you recommend for this problem and why?

3. What ansatz would you choose for VQE on this system?

4. What are the main sources of error, and what mitigation strategies would you employ?

5. What accuracy can realistically be achieved, and how does this compare to classical methods?

Your answer should demonstrate integration of hardware knowledge, algorithmic understanding, and practical judgment.

---

# End of Exam

**Before Submitting:**
- [ ] Reviewed all answers
- [ ] Checked calculations
- [ ] Answered all parts of each question
- [ ] Writing is legible and organized

**Time Remaining: _______**
