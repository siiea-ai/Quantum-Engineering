# Sample Proposals: Annotated Examples for Quantum Research

## Introduction

This document provides annotated examples of successful proposal components. Study these examples to understand how effective proposals communicate scientific ideas, methodology, and impact. All examples are inspired by funded quantum research proposals.

---

## Part 1: Project Summary Examples

### Example 1: Quantum Error Correction (NSF Style)

> **Title:** Tailored Surface Codes for Fault-Tolerant Quantum Computing with Superconducting Qubits
>
> **Overview:** Quantum computers promise transformative capabilities for cryptography, optimization, and materials simulation, but remain fundamentally limited by hardware errors. This project develops new quantum error correction codes specifically designed for the asymmetric noise characteristics of superconducting quantum processors. The research combines theoretical code design with numerical simulation and experimental validation on cloud-accessible quantum hardware. By exploiting the bias in superconducting qubit noise, where dephasing errors dominate over bit-flip errors by ratios of 10:1 to 100:1, we target a 5x reduction in the physical qubit overhead required for fault-tolerant computation compared to conventional surface codes.
>
> **Intellectual Merit:** This research advances quantum error correction theory by moving beyond symmetric noise assumptions that poorly match real hardware. The project will: (1) develop a systematic framework for designing stabilizer codes optimized for biased noise, (2) create machine learning-based decoders that leverage noise structure, and (3) establish the first experimental benchmarks of tailored codes on commercial quantum hardware. The theoretical contributions include new bounds on logical error rates for asymmetric codes and a general methodology for hardware-aware code optimization. The experimental contributions provide critical validation that translates theoretical advantages into practical improvements.
>
> **Broader Impacts:** This project directly trains two PhD students and four undergraduate researchers in quantum computing, with active recruitment from underrepresented groups through partnerships with minority-serving institutions. All developed codes and decoders will be released as open-source software on GitHub, lowering barriers to fault-tolerant quantum computing research. The PI will integrate research results into a new graduate course on practical quantum error correction and deliver annual public lectures at local science museums. Success will accelerate the timeline for practically useful quantum computing, with direct applications to drug discovery, materials design, and optimization problems of national importance.

**Annotation:**

| Element | Analysis |
|---------|----------|
| Opening hook | Establishes importance immediately (cryptography, optimization, materials) |
| Problem statement | Clear: current codes assume symmetric noise, but real noise is biased |
| Approach | Specific: code design + ML decoders + hardware validation |
| Quantitative target | "5x reduction" - concrete and measurable |
| IM: Innovation | Three specific advances (framework, decoders, benchmarks) |
| IM: Significance | Both theoretical (bounds) and experimental (validation) contributions |
| BI: Training | Specific numbers (2 PhD, 4 UG), specific mechanism (MSI partnership) |
| BI: Dissemination | Open-source, course development, public lectures |
| BI: Impact | Connects to societal needs (drug discovery, materials) |

---

### Example 2: Quantum Sensing (DOE Style)

> **Title:** Entanglement-Enhanced Magnetometry for Materials Characterization
>
> **Abstract:** This project develops entangled nitrogen-vacancy (NV) center sensors for high-resolution magnetic imaging of quantum materials, targeting 10x improvement in sensitivity compared to single-NV approaches. Magnetic imaging at the nanoscale is critical for understanding exotic magnetic phases, topological materials, and high-temperature superconductors relevant to DOE energy missions. Current NV magnetometry approaches are limited by the standard quantum limit (SQL), leaving substantial sensitivity gains untapped. We will create protocols for generating and utilizing entanglement between multiple NV centers to achieve Heisenberg-limited sensing, validated through imaging of model magnetic systems. Partnership with Argonne National Laboratory provides access to advanced materials synthesis and characterization facilities essential for benchmark measurements.
>
> **Objectives:**
> 1. Develop entanglement generation protocols for NV center arrays achieving >90% fidelity
> 2. Demonstrate sensitivity enhancement of 5x beyond SQL in controlled experiments
> 3. Image magnetic structure in candidate quantum materials with 50 nm resolution
> 4. Establish design principles for scalable entangled sensor arrays
>
> **DOE Mission Relevance:** Advanced magnetic imaging directly supports DOE's mission in materials discovery for energy applications. Understanding magnetic properties of quantum materials—including those relevant to superconductivity, spintronics, and quantum computing substrates—requires characterization tools that exceed current capabilities. This project develops enabling technology that will accelerate materials discovery across multiple DOE priority areas.

**Annotation:**

| Element | DOE-Specific Feature |
|---------|---------------------|
| Title | Technical and specific |
| Abstract | More technical depth than NSF summary |
| Target | Quantitative (10x improvement) |
| Mission relevance | Explicit connection to DOE priorities |
| Lab partnership | Featured prominently (ANL) |
| Objectives | Numbered, measurable, specific |
| Applications | DOE-relevant (energy, materials) |

---

## Part 2: Specific Aims Page Examples

### Example 3: Complete Specific Aims Page

> **Novel Approaches to Quantum Algorithm Development for Materials Simulation**
>
> Quantum computers offer exponential speedups for simulating quantum many-body systems, promising breakthroughs in materials design, drug discovery, and fundamental physics. Despite remarkable progress in quantum hardware—with devices now exceeding 1000 qubits—current quantum algorithms cannot exploit this scale due to prohibitive circuit depths and limited connectivity. **The central challenge is developing quantum algorithms that provide advantage on near-term hardware with realistic noise and topology constraints.**
>
> Current variational quantum eigensolver (VQE) algorithms suffer from barren plateaus that prevent convergence for systems beyond ~20 qubits. Alternative approaches like quantum phase estimation require error-corrected qubits not available for 5-10 years. Recent advances in machine learning offer potential solutions: neural network quantum states can efficiently represent complex ground states, and learned optimizers can navigate loss landscapes where gradient-based methods fail. However, **no systematic framework exists for combining classical machine learning with quantum circuits in a way that exploits the strengths of each while mitigating their limitations.**
>
> We hypothesize that hybrid classical-quantum algorithms, where neural networks prepare approximate states that quantum circuits refine, can solve classically intractable problems on near-term hardware. Our preliminary work demonstrates 50x reduction in circuit depth for small molecules using neural network initialization. This project scales this approach to chemically relevant systems while developing the theoretical framework for understanding when hybrid methods succeed.
>
> **Aim 1: Develop neural network quantum state initialization protocols for VQE.** We will create architectures that output parameterized quantum circuit states optimized for molecular Hamiltonians, benchmarking against random and Hartree-Fock initialization on systems up to 50 qubits.
>
> **Aim 2: Design noise-aware optimization strategies using learned surrogate models.** We will train neural networks to predict VQE energy surfaces from noisy hardware measurements, enabling efficient navigation of the optimization landscape without additional quantum resources.
>
> **Aim 3: Validate hybrid algorithms on cloud quantum hardware for benchmark molecular systems.** We will implement optimized algorithms on IBM and IonQ systems, targeting the first demonstration of advantage for molecules beyond 30 qubits.
>
> Success will establish a practical pathway for near-term quantum advantage in chemistry, directly impacting pharmaceutical development and materials discovery. The developed methods will be released as open-source extensions to major quantum computing frameworks, enabling broad adoption by the research community.

**Annotation:**

| Section | Content Analysis | Line Count |
|---------|-----------------|------------|
| Hook (Para 1) | Big picture importance, specific problem | 4 sentences |
| Gap (Para 2) | Why current approaches fail, what's needed | 4 sentences |
| Solution (Para 3) | Hypothesis, preliminary data, approach | 3 sentences |
| Aims | Action verbs, specific metrics | 3 aims |
| Impact | Significance, dissemination | 2 sentences |

**Key Features:**
- Problem: Clear and specific (current algorithms don't scale)
- Innovation: Well-defined (ML + quantum hybrid)
- Preliminary data: Mentioned (50x reduction)
- Aims: Measurable (50 qubits, 30 qubits)
- Dissemination: Specific (open-source)

---

## Part 3: Background and Significance Examples

### Example 4: Background Section (2 pages)

> **2. Background and Significance**
>
> **2.1 The Promise and Challenge of Quantum Simulation**
>
> Quantum computers excel at simulating quantum systems—a task that becomes exponentially difficult for classical computers as system size grows [1]. For strongly correlated materials like high-temperature superconductors, classical methods scale as O(e^N), limiting calculations to approximately 50 electrons [2]. Quantum computers, in principle, can simulate N-electron systems with O(N^4) resources, enabling the study of materials beyond classical reach [3].
>
> The past decade has witnessed remarkable hardware progress. IBM's superconducting processors have grown from 5 qubits in 2016 to over 1000 qubits in 2023 [4]. Google demonstrated quantum supremacy in 2019 [5], and claims of useful quantum advantage continue to emerge [6]. However, **near-term devices remain noisy, with two-qubit gate error rates of 0.1-1%, limiting circuit depth to approximately 100-1000 layers** [7].
>
> **2.2 Variational Quantum Algorithms: State of the Art**
>
> Variational quantum eigensolvers (VQE) emerged as the leading approach for near-term quantum chemistry [8]. VQE uses a parameterized quantum circuit to prepare trial wavefunctions, measuring the energy expectation value on the quantum computer while optimizing parameters classically. This hybrid approach limits circuit depth, making VQE compatible with noisy hardware.
>
> Significant VQE demonstrations include:
> - H2 molecule (2 qubits): IBM, 2017 [9]
> - BeH2 molecule (6 qubits): IBM, 2017 [10]
> - H2O molecule (12 qubits): Google, 2020 [11]
> - Hydrogen chains (up to 12 atoms): Various groups, 2021-2023 [12-15]
>
> Despite progress, VQE faces fundamental challenges that limit scaling:
>
> *Barren Plateaus:* For random or deep circuits, the variance of parameter gradients decreases exponentially with system size [16]. This means optimization landscapes become flat, preventing gradient-based methods from finding good solutions. McArdle et al. showed barren plateaus emerge for circuits exceeding O(log N) depth for local observables [17].
>
> *Hardware Noise:* Gate errors accumulate across circuit depth, limiting the maximum expressible fidelity. For error rate ε and circuit depth D, the effective fidelity scales as (1-ε)^D [18]. Current hardware limits useful depth to ~100 gates for 0.5% error rates.
>
> *Optimization Complexity:* VQE optimization is NP-hard in general [19], and landscape features including local minima, saddle points, and narrow gorges trap gradient-based optimizers [20].
>
> **2.3 The Gap: Scalable Near-Term Algorithms**
>
> Current approaches to addressing VQE limitations fall short:
>
> | Approach | Limitation |
> |----------|------------|
> | Error mitigation | O(e^D) sampling overhead [21] |
> | Problem-specific ansatze | Limited generalization |
> | Adaptive VQE | Still suffers barren plateaus |
> | Quantum phase estimation | Requires fault tolerance |
>
> **No existing method has demonstrated scalable quantum advantage for chemistry on near-term hardware.** The gap between hardware capability (1000+ qubits) and algorithm utility (~20 qubits) continues to widen.
>
> **2.4 Our Approach: Neural Network Quantum States**
>
> We propose addressing this gap by leveraging recent advances in machine learning for quantum systems. Neural network quantum states (NQS) use classical neural networks to represent complex wavefunctions [22]. The key insight is that neural networks can capture quantum correlations efficiently when trained on appropriate data.
>
> Recent work demonstrates:
> - NQS achieve state-of-the-art classical accuracy for frustrated magnets [23]
> - Variational Monte Carlo with NQS scales to 100+ sites [24]
> - Neural network optimizers can escape barren plateaus [25]
>
> Our innovation combines NQS with VQE: use neural networks to initialize and guide quantum circuits rather than replace them entirely. Preliminary results show this hybrid approach reduces circuit depth by 50x for small molecules while maintaining chemical accuracy (see Preliminary Data).
>
> **2.5 Significance**
>
> Successfully scaling hybrid NQS-VQE algorithms would:
>
> 1. **Enable practical quantum chemistry:** First useful quantum advantage for molecular simulation
> 2. **Leverage current hardware:** Use 100+ qubit systems meaningfully
> 3. **Inform hardware development:** Identify critical architecture requirements
> 4. **Accelerate materials discovery:** Enable simulation of catalysts, battery materials, superconductors
>
> The proposed research directly addresses the most significant barrier to quantum computing utility: the algorithm-hardware gap.

**Annotation:**

| Element | Technique Used |
|---------|----------------|
| Citations | Extensive (25 references in 2 pages) |
| Structure | Clear subsections with progression |
| State of the art | Table summarizing approaches |
| Gap statement | Bold, explicit |
| Our approach | Connects to existing methods |
| Significance | Numbered list of impacts |

---

## Part 4: Research Plan Examples

### Example 5: Single Aim Section

> **Aim 2: Design noise-aware optimization strategies using learned surrogate models**
>
> **2.1 Rationale**
>
> VQE optimization requires repeated quantum circuit evaluations, each consuming significant quantum resources. On noisy hardware, measurements are corrupted by gate errors, readout errors, and decoherence. Standard optimization approaches (COBYLA, L-BFGS-B) treat each measurement as ground truth, leading to optimization on noisy landscapes that differ from the true energy surface.
>
> We hypothesize that neural networks trained on noisy VQE measurements can learn to denoise the energy landscape, enabling efficient optimization with fewer quantum resources. This "learned surrogate" approach has proven effective in Bayesian optimization for expensive black-box functions [26], but has not been applied to VQE optimization.
>
> **2.2 Approach**
>
> **2.2.1 Surrogate Model Architecture**
>
> We will develop neural network architectures that predict VQE energies from circuit parameters:
>
> $$E_{\text{pred}}(\boldsymbol{\theta}) = f_{\text{NN}}(\boldsymbol{\theta}; \mathbf{w})$$
>
> where $\boldsymbol{\theta}$ are variational parameters and $\mathbf{w}$ are learned network weights.
>
> Key design choices:
> - **Input representation:** Parameter vectors (100-500 dimensions for target systems)
> - **Architecture:** Deep ensemble of MLPs with uncertainty quantification
> - **Training data:** Pairs of parameters and measured energies from hardware
> - **Output:** Energy prediction with uncertainty estimate
>
> We will explore three architectures:
> 1. Standard deep ensemble (5 networks, uncertainty from disagreement)
> 2. Bayesian neural network with variational inference
> 3. Neural network with dropout-based uncertainty
>
> **2.2.2 Training Protocol**
>
> The surrogate model training follows Algorithm 1:
>
> ```
> Algorithm 1: Surrogate Model Training
> Input: Initial parameters θ₀, quantum circuit U(θ), Hamiltonian H
> Output: Trained surrogate model f_NN
>
> 1. Initialize training dataset D = {}
> 2. For iteration i = 1 to N_init:
>    a. Sample θ_i uniformly from parameter space
>    b. Measure E_i = <ψ(θ_i)|H|ψ(θ_i)> on quantum hardware
>    c. Add (θ_i, E_i) to D
> 3. Train initial surrogate f_NN on D
> 4. For iteration i = N_init to N_total:
>    a. Use acquisition function to select θ_i
>    b. Measure E_i on quantum hardware
>    c. Add (θ_i, E_i) to D
>    d. Retrain f_NN on D
> 5. Return f_NN
> ```
>
> The acquisition function balances exploration (high uncertainty) and exploitation (low predicted energy):
>
> $$\alpha(\boldsymbol{\theta}) = -\mu(\boldsymbol{\theta}) + \beta \sigma(\boldsymbol{\theta})$$
>
> where $\mu$ and $\sigma$ are the surrogate's mean and standard deviation predictions.
>
> **2.2.3 Optimization on Surrogate**
>
> Once trained, the surrogate enables rapid optimization:
>
> 1. Optimize $\boldsymbol{\theta}^*$ on the surrogate using gradient descent
> 2. Evaluate $\boldsymbol{\theta}^*$ on quantum hardware for verification
> 3. Refine surrogate with new measurement
> 4. Repeat until convergence
>
> This reduces quantum evaluations by ~100x in our preliminary simulations.
>
> **2.3 Validation and Benchmarks**
>
> We will validate the approach using:
>
> | System | Qubits | Classical Baseline | Target |
> |--------|--------|-------------------|--------|
> | H₂ | 4 | Exact solution | <1 mHa error |
> | LiH | 12 | CCSD(T) | <5 mHa error |
> | H₂O | 14 | CCSD(T) | <5 mHa error |
> | N₂ | 20 | CCSD(T) | <10 mHa error |
> | Fe₂ | 30 | DMRG | <20 mHa error |
>
> Hardware validation will use IBM Quantum and IonQ systems.
>
> **2.4 Expected Outcomes**
>
> - Trained surrogate models for molecular Hamiltonians up to 30 qubits
> - Open-source implementation in Qiskit and Cirq
> - Demonstrated 10-100x reduction in hardware evaluations
> - Publication: "Learned Surrogates for Variational Quantum Eigensolver Optimization"
>
> **2.5 Potential Pitfalls and Alternatives**
>
> *Pitfall 1: Surrogate fails to capture landscape features*
> - Likelihood: Medium
> - Detection: Validated predictions differ from hardware by >10%
> - Mitigation: Increase training data, adjust architecture, use physics-informed constraints
>
> *Pitfall 2: Hardware noise is non-stationary*
> - Likelihood: Low-Medium
> - Detection: Surrogate accuracy degrades over time
> - Mitigation: Continuous retraining, time-dependent noise modeling
>
> *Alternative Approach:* If surrogate models prove insufficient, we will pivot to reinforcement learning for optimization, which has shown promise in related quantum control problems [27].
>
> **Milestones:**
> - M2.1 (Month 8): Surrogate architecture validated on 4-12 qubit systems
> - M2.2 (Month 14): Demonstrate 10x quantum resource reduction on hardware
> - M2.3 (Month 18): Scale to 30-qubit systems with full validation

**Annotation:**

| Section | Purpose | Content |
|---------|---------|---------|
| Rationale | Why this aim matters | 3 sentences |
| Approach | How you'll do it | Detailed methods with equations |
| Algorithm | Reproducible protocol | Pseudocode with steps |
| Validation | How you'll know it works | Table of benchmarks |
| Outcomes | Deliverables | 4 specific items |
| Pitfalls | Risk awareness | 2 risks with mitigation |
| Alternative | Backup plan | Specific pivot strategy |
| Milestones | Timeline | 3 dated milestones |

---

## Part 5: Timeline Examples

### Example 6: Gantt Chart

```
PROJECT TIMELINE: 3-Year Research Plan

                    Year 1              Year 2              Year 3
Task               Q1 Q2 Q3 Q4  Q1 Q2 Q3 Q4  Q1 Q2 Q3 Q4
─────────────────────────────────────────────────────────────────
AIM 1: NQS Initialization
 1.1 Architecture   ██ ██
 1.2 Training       ── ██ ██
 1.3 Benchmarking         ██ ██
 1.4 Publication              ██ ██
                          ▲M1

AIM 2: Surrogate Optimization
 2.1 Model design      ── ██ ██
 2.2 Training             ── ██ ██ ██
 2.3 Hardware test              ── ██ ██ ██
 2.4 Publication                      ██ ██ ██
                                   ▲M2

AIM 3: Hardware Validation
 3.1 IBM experiments                  ── ██ ██ ██
 3.2 IonQ experiments                       ── ██ ██ ██
 3.3 Analysis                                   ── ██ ██ ██
 3.4 Publication                                      ██ ██ ██
                                                         ▲M3

CROSS-CUTTING
 Software release                  ██          ██          ██
 Workshops/outreach     ██    ██    ██    ██    ██    ██    ██
 Thesis writing                                    ── ██ ██ ██

MILESTONES
 M1: NQS reduces circuit depth 10x (simulation)     Month 10
 M2: Surrogate reduces hardware evals 10x           Month 18
 M3: Demonstrate advantage on 30+ qubit system      Month 30

DELIVERABLES
 D1: NQS initialization codebase                    Month 12
 D2: Surrogate optimization toolkit                 Month 20
 D3: Hardware validation dataset                    Month 32
 D4: Final open-source software release             Month 35
```

### Example 7: Milestone Table

| ID | Milestone | Success Criterion | Date |
|----|-----------|-------------------|------|
| M1.1 | NQS architecture designed | >90% fidelity on 20-qubit benchmark | Month 4 |
| M1.2 | Training protocol validated | Converges in <1000 iterations | Month 8 |
| M1.3 | Circuit depth reduction | 10x reduction for H2O verified | Month 10 |
| M1.4 | First publication | Submitted to Physical Review X | Month 12 |
| M2.1 | Surrogate model trained | <5% prediction error on test set | Month 10 |
| M2.2 | Optimization validated | 10x reduction in hardware evals | Month 16 |
| M2.3 | Hardware demonstration | Working on IBM/IonQ systems | Month 18 |
| M2.4 | Second publication | Submitted to Nature Computational Science | Month 20 |
| M3.1 | Large-scale simulation | 50-qubit simulation complete | Month 22 |
| M3.2 | Hardware scaling | 30-qubit system characterized | Month 28 |
| M3.3 | Advantage demonstration | Performance exceeds classical | Month 30 |
| M3.4 | Third publication | Submitted to Nature | Month 34 |
| D1 | NQS codebase | Released on GitHub | Month 12 |
| D2 | Surrogate toolkit | Released on GitHub | Month 20 |
| D3 | Hardware dataset | Released on Zenodo | Month 32 |
| D4 | Final software | Full package release | Month 35 |

---

## Part 6: Broader Impacts Examples

### Example 8: Strong Broader Impacts Section

> **5. Broader Impacts**
>
> **5.1 Training the Quantum Workforce**
>
> This project directly supports the development of a diverse quantum workforce through structured training at multiple levels:
>
> *Graduate Students (2):* Two PhD students will be fully supported, receiving training in quantum algorithms, machine learning, and experimental quantum computing. Both students will:
> - Complete a structured curriculum in quantum information science
> - Conduct summer internships at partner national laboratories (confirmed letters attached)
> - Present at major conferences (APS, QIP, ACS)
> - Graduate with expertise bridging theory and experiment
>
> *Undergraduate Researchers (4):* Four undergraduates will participate through:
> - REU supplements (Years 2-3): Two students per summer from partner HBCUs
> - Academic year research: Two students per year from [University]
> - Mentored project experience leading to co-authorship
>
> *Postdoctoral Mentoring:* One postdoc (supported by separate funds) will receive career development:
> - Co-mentoring of graduate students
> - Independent project leadership
> - Grant-writing experience
> - Interview preparation
>
> **5.2 Broadening Participation**
>
> We actively recruit and support underrepresented groups in quantum science:
>
> *Recruiting Partnerships:*
> - Established MOU with [HBCU] for REU recruiting
> - Participation in APS Bridge Program
> - Collaboration with [University] Women in Physics
>
> *Retention Support:*
> - Monthly cohort meetings for supported students
> - Mentoring pairs with senior researchers
> - Conference travel for all students
> - Professional development workshops
>
> *Track Record:* The PI's group currently includes 40% women and 25% underrepresented minorities, exceeding field averages.
>
> **5.3 Educational Materials and Curriculum**
>
> Research outcomes will be integrated into education:
>
> *New Course Module:* "Practical Quantum Error Correction"
> - 4-week module for graduate quantum computing course
> - Includes Jupyter notebooks with working code
> - Available on MIT OpenCourseWare (committed)
>
> *Undergraduate Capstone:* Updated capstone project options incorporating project methods
>
> *K-12 Outreach:*
> - Annual "Quantum Computing Day" at [Local Science Museum]
> - High school workshop series (2/year)
> - Materials developed for teacher training
>
> **5.4 Open Science and Dissemination**
>
> All project outputs will be openly accessible:
>
> *Open-Source Software:*
> - Code released on GitHub under MIT license
> - Documentation and tutorials included
> - Integration with Qiskit and Cirq
> - DOI-registered releases on Zenodo
>
> *Open Data:*
> - Hardware measurement datasets released
> - Processed data on Zenodo
> - Raw data available on request
>
> *Open Access Publications:*
> - All papers published open-access
> - Preprints on arXiv simultaneous with submission
>
> **5.5 Broader Scientific Impact**
>
> Beyond the immediate research community, this project impacts:
>
> *Industry:* Results directly applicable to commercial quantum computing (IBM, IonQ, Google). Industry advisory board member [confirmed letter] will facilitate technology transfer.
>
> *National Priorities:* Advances DOE/NSF National Quantum Initiative goals. Contributes to U.S. competitiveness in quantum technology.
>
> *Other Fields:* Quantum simulation methods applicable to drug discovery, materials science, and fundamental physics beyond chemistry applications.

---

## Part 7: Common Weaknesses in Proposals

### Examples of Weak vs. Strong Writing

**Specific Aims:**

| Weak | Strong |
|------|--------|
| "We will study quantum error correction" | "We will develop asymmetric surface codes achieving 3x reduction in qubit overhead for bias ratios >10" |
| "We will explore new algorithms" | "We will implement NQS-VQE on molecules up to 30 qubits, targeting chemical accuracy (<5 mHa)" |
| "We will validate on hardware" | "We will demonstrate 10x resource reduction on IBM/IonQ systems for H2O, LiH, and N2" |

**Methodology:**

| Weak | Strong |
|------|--------|
| "Standard simulation techniques will be used" | "We will use density matrix renormalization group (DMRG) with bond dimension χ=200 using ITensor" |
| "Machine learning will optimize the results" | "A 5-layer MLP with 256 hidden units will predict energies from 100-dimensional parameter vectors" |
| "We will analyze the data carefully" | "Statistical significance will be assessed using bootstrap resampling with 10,000 iterations" |

**Broader Impacts:**

| Weak | Strong |
|------|--------|
| "Students will be trained" | "Two PhD students and four REU undergraduates will receive structured training, with active recruitment from partner HBCUs" |
| "Results will be shared" | "All code released open-source on GitHub; all papers open-access; annual public lecture at [Museum]" |
| "This research is important for society" | "Improved quantum simulation enables drug discovery, directly impacting pharmaceutical development timelines" |

---

## Part 8: Learning from These Examples

### Key Takeaways

1. **Be specific:** Replace vague language with concrete details
2. **Quantify:** Include numbers wherever possible
3. **Show, don't tell:** Use examples, figures, tables
4. **Acknowledge limitations:** Address pitfalls proactively
5. **Connect aims:** Show how pieces fit together
6. **Make impact clear:** Why should anyone care?

### Exercise: Analyze Your Own

For each section of your proposal draft, ask:
- Would a reviewer know exactly what I'm doing?
- Are my success criteria measurable?
- Have I justified why this approach?
- What could go wrong, and what's my backup?
- Why does this matter beyond my field?

---

*"The best proposals make complex ideas seem obvious. That simplicity is the result of deep understanding and careful writing."*
