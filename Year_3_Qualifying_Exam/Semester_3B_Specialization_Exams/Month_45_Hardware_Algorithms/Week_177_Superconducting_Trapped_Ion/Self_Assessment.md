# Week 177: Self-Assessment - Superconducting & Trapped Ion Systems

## Instructions

Complete this self-assessment at the end of Week 177. Be honest about your understanding level. Use the results to guide further study before the qualifying exam.

---

## Part A: Conceptual Understanding Checklist

Rate your confidence on each topic: **1** (Cannot explain) to **5** (Can teach others)

### Superconducting Qubits

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Josephson junction physics and the two Josephson relations | | | | | | |
| Cooper pair box Hamiltonian and charge noise | | | | | | |
| Transmon design: $$E_J/E_C$$ ratio and its implications | | | | | | |
| Transmon energy levels and anharmonicity | | | | | | |
| Circuit quantization procedure | | | | | | |
| Dispersive readout mechanism | | | | | | |
| Cross-resonance gate operation | | | | | | |
| Tunable coupler principle | | | | | | |
| iSWAP and CZ gate implementations | | | | | | |
| Flux qubit double-well potential | | | | | | |
| Fluxonium advantages and operation | | | | | | |
| T1 and T2 limiting mechanisms | | | | | | |
| Purcell effect and Purcell filters | | | | | | |

### Trapped Ion Qubits

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Paul trap pseudopotential | | | | | | |
| Mathieu equation and stability | | | | | | |
| Hyperfine vs optical qubit encodings | | | | | | |
| Laser-ion interaction Hamiltonian | | | | | | |
| Lamb-Dicke parameter and regime | | | | | | |
| Carrier, red, and blue sideband transitions | | | | | | |
| Doppler cooling limits | | | | | | |
| Resolved sideband cooling | | | | | | |
| Normal modes in ion chains | | | | | | |
| Mølmer-Sørensen gate mechanism | | | | | | |
| Geometric phase in MS gate | | | | | | |
| QCCD architecture | | | | | | |
| Photonic interconnects for ions | | | | | | |

### Platform Comparison

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Gate time comparison | | | | | | |
| Gate fidelity comparison | | | | | | |
| Coherence time comparison | | | | | | |
| Connectivity differences | | | | | | |
| Scalability trade-offs | | | | | | |
| Application suitability | | | | | | |

---

## Part B: Key Equation Recall

Without looking at notes, write the following equations:

### 1. Transmon Hamiltonian
$$\hat{H} = $$

*Check:* $$4E_C\hat{n}^2 - E_J\cos\hat{\phi}$$ or $$\hbar\omega_{01}\hat{a}^\dagger\hat{a} + (\alpha/2)\hat{a}^\dagger\hat{a}^\dagger\hat{a}\hat{a}$$

### 2. Transmon frequency approximation
$$\omega_{01} \approx $$

*Check:* $$\sqrt{8E_JE_C} - E_C$$

### 3. Dispersive shift
$$\chi = $$

*Check:* $$g^2/\Delta$$ or $$g^2\alpha/[\Delta(\Delta+\alpha)]$$

### 4. Lamb-Dicke parameter
$$\eta = $$

*Check:* $$k\sqrt{\hbar/(2m\omega)}$$

### 5. MS gate unitary
$$\hat{U}_{MS} = $$

*Check:* $$\exp(-i\theta\hat{S}_\phi^2)$$

### 6. Ion equilibrium separation
$$d = $$

*Check:* $$(e^2/(2\pi\epsilon_0 m\omega^2))^{1/3}$$

---

## Part C: Quick Derivation Practice

Time yourself on these mini-derivations (5 minutes each):

### 1. Show that charge noise sensitivity is suppressed in the transmon
Starting point: $$H = 4E_C(n-n_g)^2 - E_J\cos\phi$$

Your derivation:

_______________________________________________

_______________________________________________

_______________________________________________

### 2. Derive the normal mode frequencies for two ions
Starting point: Coupled harmonic oscillators with Coulomb interaction

Your derivation:

_______________________________________________

_______________________________________________

_______________________________________________

### 3. Calculate the dispersive shift from perturbation theory
Starting point: Jaynes-Cummings Hamiltonian

Your derivation:

_______________________________________________

_______________________________________________

_______________________________________________

---

## Part D: Problem-Solving Assessment

Attempt these problems without solutions, then check your work:

### Problem 1 (5 minutes)
A transmon has $$E_J/h = 20$$ GHz and $$E_C/h = 300$$ MHz. Calculate:
- $$\omega_{01}$$
- Anharmonicity $$\alpha$$
- Is it in the transmon regime?

Your answer:

_______________________________________________

### Problem 2 (5 minutes)
A Yb-171 ion is in a trap with $$\omega_z/2\pi = 1.5$$ MHz. Calculate:
- Ground state extent $$z_0$$
- Lamb-Dicke parameter for 369 nm laser

Your answer:

_______________________________________________

### Problem 3 (10 minutes)
Compare running a 20-qubit circuit with 100 CNOT gates on:
- A superconducting processor with nearest-neighbor connectivity (line topology)
- A trapped ion processor with all-to-all connectivity

Estimate the number of physical gates needed on each platform.

Your answer:

_______________________________________________

---

## Part E: Oral Exam Simulation

Record yourself answering these questions (3-5 minutes each):

### Question 1
"Explain how the transmon qubit works and why it was developed."

Self-evaluation criteria:
- [ ] Mentioned charge noise problem in CPB
- [ ] Explained $$E_J/E_C$$ design choice
- [ ] Discussed anharmonicity trade-off
- [ ] Connected to circuit QED

### Question 2
"Describe the Mølmer-Sørensen gate and explain why it's robust to thermal motion."

Self-evaluation criteria:
- [ ] Explained bichromatic field setup
- [ ] Described spin-dependent force
- [ ] Explained geometric phase
- [ ] Discussed motional state returning to origin

### Question 3
"Compare superconducting and trapped ion platforms for quantum computing."

Self-evaluation criteria:
- [ ] Covered gate times and fidelities
- [ ] Discussed connectivity differences
- [ ] Mentioned scalability considerations
- [ ] Gave application-specific recommendations

---

## Part F: Gap Identification

Based on your performance above, identify your top 3 knowledge gaps:

1. _______________________________________________

2. _______________________________________________

3. _______________________________________________

**Action plan to address these gaps:**

_______________________________________________

_______________________________________________

_______________________________________________

---

## Part G: Practice Problem Scoring

Score your Problem Set performance:

| Section | Problems Attempted | Problems Correct | Percentage |
|---------|-------------------|------------------|------------|
| A: Josephson Physics | /5 | | |
| B: Circuit QED | /7 | | |
| C: Flux Qubits | /4 | | |
| D: Trapped Ion Basics | /6 | | |
| E: MS Gate | /5 | | |
| F: Integration | /3 | | |
| **Total** | /30 | | |

**Sections needing review:** _______________________________________________

---

## Part H: Readiness Assessment

Complete this checklist honestly:

### Ready for Qualifying Exam?

- [ ] I can derive the transmon Hamiltonian from scratch
- [ ] I can explain dispersive readout at the whiteboard
- [ ] I understand why $$E_J/E_C \gg 1$$ is chosen
- [ ] I can derive the MS gate mechanism
- [ ] I can calculate Lamb-Dicke parameters
- [ ] I can compare platforms for specific applications
- [ ] I can discuss recent hardware milestones (2024-2025)
- [ ] I can handle follow-up questions in oral format

### Confidence Level

On a scale of 1-10, my readiness for hardware questions on the qualifying exam:

**Score: ___ / 10**

### Recommended Actions

If score < 5:
- [ ] Review lecture notes and textbook chapters
- [ ] Redo all problem set problems
- [ ] Schedule study group session

If score 5-7:
- [ ] Focus on identified weak areas
- [ ] Practice oral explanations with a partner
- [ ] Review recent hardware papers

If score 8-10:
- [ ] Practice timed problem solving
- [ ] Prepare for challenging follow-up questions
- [ ] Help others to reinforce your understanding

---

## Reflection

What was the most challenging concept this week?

_______________________________________________

What surprised you about quantum hardware?

_______________________________________________

How would you explain the transmon to a non-physicist?

_______________________________________________

---

*Complete this assessment before moving to Week 178.*
