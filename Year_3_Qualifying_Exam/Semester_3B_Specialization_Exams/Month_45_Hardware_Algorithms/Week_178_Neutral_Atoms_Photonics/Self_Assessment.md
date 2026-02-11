# Week 178: Self-Assessment - Neutral Atoms & Photonics

## Instructions

Complete this self-assessment at the end of Week 178. Rate your understanding honestly to identify areas for further study.

---

## Part A: Conceptual Understanding Checklist

Rate each topic: **1** (Cannot explain) to **5** (Can teach others)

### Rydberg Physics

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Rydberg state properties and scaling laws | | | | | | |
| van der Waals interaction ($$C_6/R^6$$) | | | | | | |
| Rydberg blockade mechanism | | | | | | |
| Blockade radius calculation | | | | | | |
| CZ gate pulse sequence | | | | | | |
| Native CCZ gate implementation | | | | | | |
| Error sources in Rydberg gates | | | | | | |

### Neutral Atom Arrays

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Optical dipole trap physics | | | | | | |
| Tweezer array generation (SLM, AOD) | | | | | | |
| Stochastic loading and rearrangement | | | | | | |
| Ground-state qubit encoding | | | | | | |
| Coherence times and limiting factors | | | | | | |
| Mid-circuit measurement techniques | | | | | | |
| Analog vs digital operation modes | | | | | | |

### Bosonic Codes

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| GKP code grid state structure | | | | | | |
| GKP error correction via syndrome | | | | | | |
| Cat qubit encoding | | | | | | |
| Biased noise in cat qubits | | | | | | |
| Kerr-cat stabilization Hamiltonian | | | | | | |
| Logical gates in bosonic codes | | | | | | |
| Concatenation with surface codes | | | | | | |

### Photonic Quantum Computing

| Topic | 1 | 2 | 3 | 4 | 5 | Notes |
|-------|---|---|---|---|---|-------|
| Photonic qubit encodings | | | | | | |
| Linear optical gates | | | | | | |
| KLM protocol and probabilistic gates | | | | | | |
| Measurement-based quantum computing | | | | | | |
| Cluster state preparation | | | | | | |
| Fusion-based approach | | | | | | |
| Gaussian boson sampling | | | | | | |
| Photon sources and detectors | | | | | | |

---

## Part B: Key Equation Recall

Write these equations from memory:

### 1. Blockade Radius
$$R_b = $$

*Check:* $$(C_6/\hbar\Omega)^{1/6}$$

### 2. C6 Scaling
$$C_6 \propto $$

*Check:* $$n^{11}$$

### 3. GKP Logical States
$$|0_L\rangle \propto $$

*Check:* $$\sum_s |2s\sqrt{\pi}\rangle$$

### 4. Cat State Overlap
$$\langle\alpha|-\alpha\rangle = $$

*Check:* $$e^{-2|\alpha|^2}$$

### 5. KLM CZ Success Probability
$$P_{CZ} = $$

*Check:* $$1/16$$ (basic)

---

## Part C: Quick Calculations

Time yourself (5 minutes each):

### 1. Blockade Radius
Given: $$C_6/h = 200$$ GHz·μm$$^6$$, $$\Omega/2\pi = 1$$ MHz

Calculate $$R_b$$:

_______________________________________________

*Answer: $$R_b = (200 \times 10^9 / 10^6)^{1/6} = (2 \times 10^5)^{1/6} \approx 7.7$$ μm*

### 2. Cat Qubit Bit-Flip Suppression
Given: $$|\alpha|^2 = 6$$

Calculate $$\langle\alpha|-\alpha\rangle$$:

_______________________________________________

*Answer: $$e^{-12} \approx 6 \times 10^{-6}$$*

### 3. Photon Probability
A squeezed state with $$r = 0.5$$. Calculate $$P(n=0) = 1/\cosh(0.5)$$:

_______________________________________________

*Answer: $$1/\cosh(0.5) = 1/1.128 = 0.89$$*

---

## Part D: Conceptual Questions

Answer in 2-3 sentences each:

### 1. Why does $$C_6$$ scale as $$n^{11}$$?

_______________________________________________

_______________________________________________

### 2. Why are cat qubits said to have "biased noise"?

_______________________________________________

_______________________________________________

### 3. Why is measurement-based QC well-suited for photonics?

_______________________________________________

_______________________________________________

### 4. What advantage do neutral atoms have over trapped ions for large-scale computing?

_______________________________________________

_______________________________________________

---

## Part E: Platform Comparison

Complete this comparison table from memory:

| Feature | Neutral Atoms | Photonics |
|---------|---------------|-----------|
| Qubit count (2024) | | |
| Two-qubit gate fidelity | | |
| Connectivity | | |
| Operating temperature | | |
| Native gate advantage | | |
| Main scalability challenge | | |

---

## Part F: Oral Exam Simulation

Record yourself (3-5 minutes each):

### Question 1
"Explain the Rydberg blockade and how it enables entangling gates."

Self-evaluation:
- [ ] Defined blockade radius
- [ ] Explained interaction mechanism
- [ ] Described gate protocol
- [ ] Mentioned fidelity limitations

### Question 2
"Compare GKP and cat codes for quantum error correction."

Self-evaluation:
- [ ] Described both encodings
- [ ] Explained error correction mechanisms
- [ ] Discussed biased vs unbiased noise
- [ ] Mentioned hardware implementations

### Question 3
"What are the main challenges for scaling photonic quantum computers?"

Self-evaluation:
- [ ] Discussed probabilistic gates
- [ ] Mentioned photon loss
- [ ] Explained detector requirements
- [ ] Discussed proposed solutions

---

## Part G: Problem Set Scoring

| Section | Problems Attempted | Problems Correct | Percentage |
|---------|-------------------|------------------|------------|
| A: Rydberg Physics | /7 | | |
| B: Neutral Atom Arrays | /6 | | |
| C: Bosonic Codes | /6 | | |
| D: Photonics | /6 | | |
| E: Platform Comparison | /3 | | |
| **Total** | /28 | | |

---

## Part H: Gap Identification

Top 3 areas needing improvement:

1. _______________________________________________

2. _______________________________________________

3. _______________________________________________

Action plan:

_______________________________________________

_______________________________________________

---

## Part I: Readiness Check

### Checklist

- [ ] Can calculate blockade radius from parameters
- [ ] Can explain GKP error correction step-by-step
- [ ] Understand cat qubit biased noise
- [ ] Know KLM protocol basics
- [ ] Can compare platforms for specific applications
- [ ] Aware of 2024-2025 milestones

### Confidence Level

Readiness for exam questions on these topics: ___ / 10

---

## Reflection

What was the most surprising aspect of neutral atom or photonic QC?

_______________________________________________

Which platform do you think is most promising? Why?

_______________________________________________

How do bosonic codes change our thinking about error correction?

_______________________________________________
