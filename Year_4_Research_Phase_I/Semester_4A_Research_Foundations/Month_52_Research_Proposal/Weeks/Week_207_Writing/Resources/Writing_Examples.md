# Writing Examples: Before and After

## Introduction

This document provides examples of weak and strong proposal writing. Study these examples to understand how to transform rough drafts into compelling prose.

---

## Part 1: Opening Sentences

### Example 1: Field Introduction

**Weak:**
> Quantum computing is an important and rapidly advancing field of research.

**Why it's weak:**
- Generic (could describe any field)
- No specifics
- Tells, doesn't show

**Strong:**
> Quantum computers can solve in hours problems that would take classical supercomputers longer than the age of the universe, promising breakthroughs in drug discovery, materials science, and cryptography.

**Why it's strong:**
- Specific and vivid
- Demonstrates importance through example
- Creates intrigue

---

### Example 2: Problem Statement

**Weak:**
> Current approaches have significant limitations.

**Why it's weak:**
- Vague
- No specifics
- Doesn't explain what or why

**Strong:**
> Current quantum error correction approaches require approximately 1,000 physical qubits to protect a single logical qubit, making practical quantum computation impossible with today's 100-qubit devices.

**Why it's strong:**
- Specific numbers
- Clear problem definition
- Implies solution space

---

### Example 3: Research Introduction

**Weak:**
> We propose to study quantum error correction.

**Why it's weak:**
- "Study" is vague
- No indication of approach
- Passive and weak

**Strong:**
> We will develop new quantum error correction codes that exploit the inherent noise asymmetry in superconducting qubits, targeting a 5-fold reduction in the physical resources required for fault-tolerant computation.

**Why it's strong:**
- Specific action ("develop")
- Clear approach (exploit noise asymmetry)
- Quantitative target (5-fold reduction)

---

## Part 2: Specific Aims

### Example 4: Complete Specific Aims Page

**Weak Version:**

> Quantum computers are becoming increasingly important. However, they still have problems with errors. We plan to work on quantum error correction to make quantum computers work better.
>
> The current approaches to error correction have some issues. They use too many qubits and don't work well with real hardware. We want to develop new methods that will be better.
>
> Our specific aims are:
> 1. Study quantum error correction codes
> 2. Develop new algorithms
> 3. Test on quantum computers
>
> If successful, this research will advance the field of quantum computing.

**Why it's weak:**
- No hook or urgency
- Vague throughout
- Generic aims
- No quantitative targets

---

**Strong Version:**

> Quantum computers promise exponential speedups for cryptography, drug discovery, and materials simulation, yet remain fundamentally limited by hardware errors. Despite billion-dollar investments, no quantum computer can yet perform useful computations because physical error rates of 0.1-1% require error correction that demands millions of qubits—far beyond current capabilities. **The path to practical quantum computing requires not just better qubits, but fundamentally new approaches to error correction that match the realities of available hardware.**
>
> Current error correction assumes symmetric noise—equal probability of different error types—but real superconducting qubits exhibit strongly biased noise where phase errors dominate bit-flip errors by factors of 10-100. Standard surface codes waste resources protecting against rare errors while under-protecting against common ones. Recent theoretical work shows that codes tailored to biased noise can dramatically reduce overhead, but **no systematic framework exists for designing, optimizing, and validating such codes for realistic hardware.**
>
> We hypothesize that quantum error correction codes optimized for hardware-specific noise characteristics can reduce physical qubit requirements by 5-10x while maintaining equivalent logical error rates. Our approach combines genetic algorithm optimization of code structures with machine learning decoders trained on realistic noise, validated on cloud-accessible quantum processors. Preliminary simulations demonstrate 3x overhead reduction for bias ratios of 10:1, establishing feasibility.
>
> **Aim 1: Develop asymmetric stabilizer codes optimized for biased noise.** We will create a computational framework for designing codes that exploit noise structure, targeting 3x overhead reduction for bias ratios exceeding 10:1.
>
> **Aim 2: Design machine learning decoders that leverage noise bias.** We will train neural network decoders on realistic noise models, achieving 50% improvement in logical error rates compared to standard approaches.
>
> **Aim 3: Validate tailored codes on cloud quantum hardware.** We will implement optimized codes on IBM and IonQ systems, providing the first experimental demonstration of bias-aware error correction advantage.
>
> This research will establish practical pathways for near-term fault-tolerant quantum computing, directly impacting the roadmaps of major technology companies and national laboratories. By releasing all code open-source, we will accelerate progress across the quantum computing community.

**Why it's strong:**
- Compelling hook with specific stakes
- Clear gap identification
- Quantitative targets throughout
- Specific, measurable aims
- Impact statement

---

## Part 3: Background Sections

### Example 5: State of the Art

**Weak:**
> Many researchers have worked on quantum error correction. Some important papers include [1-20]. These papers have made contributions to the field but there is still more work to be done.

**Why it's weak:**
- No synthesis of prior work
- Citations without context
- No critical analysis
- No connection to proposed work

**Strong:**
> Quantum error correction has advanced dramatically since Shor's foundational 9-qubit code [1]. The discovery of stabilizer codes [2] and particularly the surface code [3] established a path to fault tolerance with error thresholds near 1%—achievable with current hardware. Recent experiments have demonstrated repetition codes [4], distance-2 surface codes [5], and preliminary evidence of logical qubit protection [6].
>
> However, these demonstrations share a critical limitation: they assume symmetric noise. Table 1 summarizes the best-performing experiments and their implicit noise assumptions.
>
> | Experiment | Qubits | Code | Noise Model | Limitation |
> |------------|--------|------|-------------|------------|
> | Google [4] | 11 | Repetition | Bit-flip only | No phase errors |
> | IBM [5] | 17 | Surface d=2 | Symmetric Pauli | Bias ratio 1:1 |
> | Quantinuum [6] | 12 | Color | Symmetric | Bias ratio 1:1 |
>
> Real superconducting qubits violate these assumptions. Measurements by multiple groups [7-10] show bias ratios (Z:X error) ranging from 10:1 to over 100:1, yet no experiment has exploited this structure. This gap between theory and experiment motivates our proposed research.

**Why it's strong:**
- Synthesizes literature
- Uses table to organize information
- Critical analysis identifies gap
- Connects to proposed work

---

### Example 6: Innovation Statement

**Weak:**
> Our approach is novel and innovative. We will use new methods that have not been tried before.

**Why it's weak:**
- Claims without evidence
- No specifics
- Tells rather than shows

**Strong:**
> Our approach combines three innovations that distinguish it from prior work:
>
> 1. **Bias-aware code design:** While biased-noise codes have been studied theoretically [11-13], no systematic optimization framework exists. We develop genetic algorithms that search the space of stabilizer codes with fitness functions weighted by measured noise characteristics.
>
> 2. **Noise-adaptive decoding:** Standard MWPM decoders assume uniform edge weights. We train neural network decoders on hardware-calibrated noise distributions, enabling adaptation to device-specific error patterns without manual tuning.
>
> 3. **Hardware-in-the-loop validation:** Most error correction research relies on simulation with assumed noise models. We validate our codes on real hardware, using measured noise characteristics as optimization targets.
>
> The combination creates a closed loop from noise characterization through code optimization to hardware validation—a systematic approach not previously demonstrated.

**Why it's strong:**
- Specific innovations listed
- Contrast with prior work
- Evidence of novelty
- Integration explained

---

## Part 4: Methodology Sections

### Example 7: Method Description

**Weak:**
> We will use machine learning to train a decoder. The decoder will be implemented using standard techniques. We will test it on various datasets.

**Why it's weak:**
- No specifics (which ML technique?)
- No parameters
- "Standard techniques" is meaningless
- No success criteria

**Strong:**
> We will train a neural network decoder to predict error corrections from syndrome measurements:
>
> **Architecture:** A 5-layer fully connected network (256-512-512-512-256 hidden units) with ReLU activation and dropout (p=0.2) for regularization. Input: binary syndrome vector (dimension n-k for [[n,k,d]] code). Output: most likely error class (size 4^n for full Pauli basis, reduced by symmetry).
>
> **Training Protocol:**
> 1. Generate 10^6 error configurations from hardware-calibrated noise model
> 2. Compute syndromes and optimal corrections (via brute force for small codes, MWPM for large)
> 3. Train using Adam optimizer (lr=10^-4) with batch size 256 for 100 epochs
> 4. Validate on held-out test set (10% of data)
>
> **Success Criteria:** Classification accuracy >99% on test set; logical error rate within 10% of optimal maximum-likelihood decoder.
>
> **Software:** Implemented in PyTorch 2.0, training on 4x NVIDIA A100 GPUs (available via university cluster). Estimated training time: 4 hours per code configuration.

**Why it's strong:**
- Specific architecture
- Clear protocol
- Measurable success criteria
- Implementation details

---

### Example 8: Experimental Protocol

**Weak:**
> We will run experiments on quantum computers. The experiments will test our codes. We will measure the error rates.

**Why it's weak:**
- No protocol
- No specifics
- No controls

**Strong:**
> **Hardware Validation Protocol:**
>
> *Platform:* IBM Quantum Eagle processor (127 qubits, heavy-hex topology). Access via IBM Quantum Network (confirmed partnership letter attached).
>
> *Pre-experiment Characterization:*
> 1. Run single-qubit T1/T2 measurements (1000 shots each, 10 delay points)
> 2. Execute randomized benchmarking (depths 1-100, 30 random circuits per depth)
> 3. Extract noise bias via process tomography on 10 representative qubits
> 4. Select subset with highest T2 and bias ratio >10 for experiments
>
> *Code Implementation:*
> 1. Prepare logical |0⟩_L via encoding circuit (depth ~10 for d=3 codes)
> 2. Apply identity (memory experiment) or logical gate
> 3. Execute d rounds of syndrome extraction
> 4. Measure all data qubits in Z basis
> 5. Record raw measurement outcomes (10,000 shots per configuration)
>
> *Controls:*
> - Standard surface code (same qubits, same error rate): SQL baseline
> - Random circuits (no coherent code structure): Classical baseline
> - Bare physical qubit (no encoding): Decoherence reference
>
> *Analysis:*
> - Decode syndromes using trained neural network decoder
> - Compute logical error rate with 95% confidence intervals
> - Compare tailored vs. standard codes using paired t-test (α=0.05)
>
> **Success Criterion:** Tailored codes achieve logical error rate at least 2x lower than standard codes at matched physical error rate.

**Why it's strong:**
- Platform specified
- Complete protocol
- Controls included
- Statistical analysis specified
- Success criterion defined

---

## Part 5: Pitfall Sections

### Example 9: Risk Discussion

**Weak:**
> If our method doesn't work, we will try something else.

**Why it's weak:**
- Not specific about what could fail
- No analysis
- Vague alternative

**Strong:**
> **Potential Pitfall: Neural network decoder fails to generalize from simulation to hardware**
>
> The noise distributions used for training are necessarily approximations of actual hardware behavior. If the mismatch is large, decoder performance may degrade on real devices.
>
> *Detection:* We will compare decoder accuracy on simulated test data versus early hardware measurements. A gap >20% indicates generalization failure.
>
> *Mitigation:*
> 1. Train on diverse noise models spanning the uncertainty in characterization
> 2. Use domain adaptation techniques (fine-tuning on small hardware dataset)
> 3. Implement online learning that updates decoder from accumulated hardware data
>
> *Alternative Approach:* If learning-based decoding proves unreliable, we will revert to MWPM with bias-adjusted edge weights. This approach is mathematically guaranteed to succeed (for correctable errors) and has been extensively validated [ref]. The tradeoff is ~20% higher logical error rate compared to optimal ML decoder, but still demonstrates the advantage of our tailored codes.

**Why it's strong:**
- Specific failure mode
- Detection criteria
- Multiple mitigation strategies
- Concrete alternative with tradeoffs

---

## Part 6: Broader Impacts

### Example 10: Training Statement

**Weak:**
> Students will be trained in the research.

**Why it's weak:**
- No specifics
- No numbers
- No mechanism

**Strong:**
> **Graduate Student Training:** This project supports two PhD students full-time for three years. Student 1 (Years 1-3) will focus on theoretical code design and simulation, developing expertise in quantum information theory and high-performance computing. Student 2 (Years 2-4, with Year 1 supported by teaching assistantship) will specialize in experimental validation and machine learning. Both students will:
> - Complete structured coursework in quantum computing and ML
> - Present at national conferences (APS, QIP) annually
> - Conduct 10-week summer internships at partner national laboratories (confirmed letters from Argonne and Sandia attached)
> - Co-author at least two peer-reviewed publications
> - Graduate with skills directly applicable to industry or academic positions

**Why it's strong:**
- Specific numbers
- Clear activities
- Measurable outcomes
- Evidence of planning

---

### Example 11: Broadening Participation

**Weak:**
> We will recruit diverse students.

**Why it's weak:**
- Vague
- No mechanism
- No commitment

**Strong:**
> **Broadening Participation:** We actively recruit and support students from underrepresented groups:
>
> *Recruiting Mechanisms:*
> - REU students selected primarily from partner MSIs (Howard University, Spelman College) through established pipeline from PI's previous NSF REU site
> - Participation in APS Bridge Program, connecting with students from predominantly undergraduate institutions
> - Targeted outreach at SACNAS and ABRCMS conferences
>
> *Support and Retention:*
> - Monthly cohort meetings connecting all project students
> - Peer mentoring pairs between senior and junior students
> - Conference travel support for all students (budgeted)
> - Professional development workshops on graduate applications, presentation skills
>
> *Track Record:* The PI's research group currently includes 45% women and 30% underrepresented minorities, exceeding field averages of 20% and 10%, respectively.

**Why it's strong:**
- Specific mechanisms
- Multiple strategies
- Support beyond recruiting
- Evidence of commitment (track record)

---

## Part 7: Common Phrase Improvements

| Weak Phrase | Strong Alternative |
|-------------|-------------------|
| "It is important to note that" | (delete, make point directly) |
| "We will try to" | "We will" |
| "This research will hopefully" | "This research will" |
| "Novel and innovative" | (show innovation through specifics) |
| "State-of-the-art" | (specify what makes it leading) |
| "In the future" | (give timeline) |
| "Various" | (list specifically) |
| "Significant" | (quantify) |
| "As mentioned above" | (just make the point again) |
| "The fact that" | (delete) |
| "In order to" | "To" |
| "Utilize" | "Use" |
| "Methodology" (when you mean method) | "Method" |
| "Prior to" | "Before" |
| "Subsequent to" | "After" |

---

## Part 8: Paragraph Structure

### Example 12: Well-Structured Paragraph

**Topic Sentence (main point):**
> Surface codes are the leading approach to fault-tolerant quantum computing due to their high error threshold and simple structure.

**Supporting Detail 1:**
> The threshold of approximately 1% [ref] is achievable with current superconducting hardware, where two-qubit gate errors of 0.5% have been demonstrated [ref].

**Supporting Detail 2:**
> The local connectivity requirement—each qubit interacts only with its neighbors—matches the planar topology of superconducting chips, enabling efficient implementation without complex routing.

**Supporting Detail 3:**
> Efficient decoders based on minimum-weight perfect matching run in polynomial time, allowing real-time error correction.

**Transition to Next Topic:**
> However, standard surface codes assume symmetric noise, leaving significant room for improvement when this assumption is violated.

---

## Summary: The Before-After Pattern

When revising, apply this pattern:

1. **Find vague words** → Replace with specifics
2. **Find passive voice** → Convert to active
3. **Find claims without evidence** → Add support
4. **Find weak verbs** → Use strong action verbs
5. **Find long sentences** → Break into shorter ones
6. **Find missing numbers** → Quantify
7. **Find "we hope/try"** → Assert confidently

---

*"Good writing doesn't happen. It's made through relentless revision."*
