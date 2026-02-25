# Key Review Articles for Quantum Computing Research

## Overview

This curated list contains essential review articles for surveying the quantum computing research landscape. Reviews are organized by area and annotated with scope, reading time, and key takeaways. All papers are available on arXiv or through standard academic access.

**Reading Strategy:** Start with "Foundation" reviews, then move to area-specific reviews based on your interests.

---

## Foundation Reviews (Read First)

### 1. Quantum Computing in the NISQ Era and Beyond
**Author:** John Preskill
**Year:** 2018 | **arXiv:** 1801.00862
**Reading Time:** 2-3 hours

**Scope:** Seminal paper defining the NISQ era and outlining the path to fault tolerance

**Key Takeaways:**
- NISQ = Noisy Intermediate-Scale Quantum (50-100 qubits, no error correction)
- Near-term applications must tolerate noise
- Quantum supremacy as a milestone
- Long-term vision: fault-tolerant quantum computing

**Why Essential:** Defines the vocabulary and framework the entire field uses

---

### 2. Noisy Intermediate-Scale Quantum Algorithms
**Authors:** Bharti et al.
**Year:** 2022 | **arXiv:** 2101.08448
**Reading Time:** 4-5 hours (comprehensive)

**Scope:** Comprehensive survey of all NISQ algorithm classes

**Key Takeaways:**
- Variational algorithms (VQE, QAOA) dominate NISQ
- Error mitigation vs. error correction trade-offs
- Application-specific algorithm design
- Limitations and open problems

**Why Essential:** Best single-source overview of NISQ algorithms

---

### 3. Quantum Error Correction: An Introductory Guide
**Authors:** Roffe
**Year:** 2019 | **arXiv:** 1907.11157
**Reading Time:** 3-4 hours

**Scope:** Pedagogical introduction to QEC concepts

**Key Takeaways:**
- Stabilizer formalism explained clearly
- Surface code introduction
- Threshold theorem intuition
- Practical decoder concepts

**Why Essential:** Accessible QEC foundation before diving into advanced topics

---

## Quantum Algorithms Reviews

### 4. Variational Quantum Eigensolver (VQE) Review
**Authors:** Tilly et al.
**Year:** 2022 | **arXiv:** 2111.05176
**Reading Time:** 3-4 hours

**Scope:** Comprehensive VQE survey including variants and applications

**Topics Covered:**
- VQE algorithm structure
- Ansatz design strategies
- Optimization landscapes
- Error mitigation for VQE
- Application to chemistry

---

### 5. Quantum Approximate Optimization Algorithm (QAOA)
**Authors:** Blekos et al.
**Year:** 2024 | **arXiv:** 2306.09198
**Reading Time:** 3-4 hours

**Scope:** Complete QAOA survey from theory to applications

**Topics Covered:**
- QAOA formulation and variants
- Performance analysis
- Comparison with classical solvers
- Implementation considerations

---

### 6. Quantum Machine Learning: What Quantum Computing Means for Data Mining
**Authors:** Schuld & Petruccione
**Year:** 2018 (Book) + 2022 review papers
**Reading Time:** Variable

**Scope:** QML foundations and current state

**Key Resources:**
- Schuld: "Machine Learning with Quantum Computers" (book)
- Review: "Supervised quantum machine learning models" (2018)
- Cerezo et al.: "Variational Quantum Algorithms" (2021)

---

### 7. Quantum Algorithms for Quantum Chemistry
**Authors:** McArdle et al.
**Year:** 2020 | **Reviews of Modern Physics** 92, 015003
**Reading Time:** 5-6 hours

**Scope:** Definitive review of quantum chemistry algorithms

**Topics Covered:**
- Electronic structure problem
- VQE for chemistry
- Quantum phase estimation approaches
- Resource requirements
- Near-term vs. fault-tolerant approaches

---

## Quantum Hardware Reviews

### 8. Superconducting Qubits: Current State of Play
**Authors:** Kjaergaard et al.
**Year:** 2020 | **Annual Review of Condensed Matter Physics**
**Reading Time:** 3-4 hours

**Scope:** Comprehensive superconducting qubit review

**Topics Covered:**
- Transmon and alternative qubit designs
- Gate implementations
- Error mechanisms
- Scaling challenges
- State of the art (2020)

---

### 9. Trapped-Ion Quantum Computing: Progress and Challenges
**Authors:** Bruzewicz et al.
**Year:** 2019 | **Applied Physics Reviews** 6, 021314
**Reading Time:** 4-5 hours

**Scope:** Comprehensive trapped-ion review

**Topics Covered:**
- Ion trap architectures
- Gate mechanisms
- Scaling approaches
- Error sources
- Comparison with other platforms

---

### 10. Neutral Atom Quantum Computing
**Authors:** Henriet et al.
**Year:** 2020 | **Quantum** 4, 327
**Reading Time:** 3-4 hours

**Scope:** Neutral atom quantum computing and simulation

**Topics Covered:**
- Optical tweezer arrays
- Rydberg interactions
- Gate implementations
- Analog quantum simulation
- Scaling prospects

---

### 11. Photonic Quantum Computing
**Authors:** Slussarenko & Pryde
**Year:** 2019 | **Applied Physics Reviews** 6, 041303
**Reading Time:** 3-4 hours

**Scope:** Photonic quantum information processing

**Topics Covered:**
- Linear optical quantum computing
- Photon sources and detectors
- Integrated photonics
- Boson sampling
- Measurement-based approaches

---

### 12. Spin Qubits in Silicon
**Authors:** Burkard et al.
**Year:** 2023 | **Reviews of Modern Physics** 95, 025003
**Reading Time:** 4-5 hours

**Scope:** Silicon spin qubit review

**Topics Covered:**
- Quantum dot spin qubits
- Donor-based qubits
- Gate mechanisms
- Decoherence sources
- Manufacturing considerations

---

## Quantum Error Correction Reviews

### 13. Roads Towards Fault-Tolerant Universal Quantum Computation
**Authors:** Campbell, Terhal, Vuillot
**Year:** 2017 | **Nature** 549, 172
**Reading Time:** 2-3 hours

**Scope:** Overview of paths to fault-tolerant QC

**Topics Covered:**
- Threshold theorem
- Surface codes
- Magic state distillation
- Resource overhead
- Alternative approaches

---

### 14. Surface Codes: Towards Practical Large-Scale Quantum Computation
**Authors:** Fowler et al.
**Year:** 2012 | **Physical Review A** 86, 032324
**Reading Time:** 4-5 hours

**Scope:** Definitive surface code reference

**Topics Covered:**
- Surface code construction
- Logical operations
- Threshold calculations
- Resource estimates
- Implementation considerations

**Note:** Foundational paper, supplement with recent experimental results

---

### 15. Quantum Low-Density Parity-Check Codes
**Authors:** Breuckmann & Eberhardt
**Year:** 2021 | **PRX Quantum** 2, 040101
**Reading Time:** 3-4 hours

**Scope:** Introduction to qLDPC codes

**Topics Covered:**
- Classical LDPC background
- Quantum LDPC challenges
- Good qLDPC code constructions
- Comparison with surface codes
- Open problems

---

### 16. Bosonic Quantum Error Correction
**Authors:** Terhal, Conrad, Vuillot
**Year:** 2020 | **Quantum Science and Technology** 5, 043001
**Reading Time:** 3-4 hours

**Scope:** Bosonic code survey

**Topics Covered:**
- Cat codes
- GKP codes
- Binomial codes
- Hardware implementations
- Comparison and trade-offs

---

### 17. Decoding Quantum Error Correction Codes
**Authors:** Review various; key: Battistel et al. 2023
**Year:** 2023 | Various
**Reading Time:** Variable

**Scope:** Decoder algorithms for QEC

**Key References:**
- MWPM decoder: Higgott "PyMatching" (2021)
- Union-find: Delfosse & Nickerson (2021)
- Neural network decoders: Torlai et al. (2017)
- Real-time decoding: Skoric et al. (2023)

---

## Applications Reviews

### 18. Quantum Computing for Finance
**Authors:** Herman et al.
**Year:** 2023 | **Nature Reviews Physics** 5, 450
**Reading Time:** 2-3 hours

**Scope:** Quantum finance applications survey

**Topics Covered:**
- Option pricing
- Portfolio optimization
- Risk analysis
- Monte Carlo methods
- Near-term prospects

---

### 19. Quantum Optimization: Potential, Challenges, and Outlook
**Authors:** Abbas et al. (IBM)
**Year:** 2023 | **arXiv:** 2312.02279
**Reading Time:** 3-4 hours

**Scope:** Realistic assessment of quantum optimization

**Topics Covered:**
- QAOA analysis
- Comparison with classical solvers
- Application benchmarks
- Resource requirements
- Path forward

---

### 20. Quantum Machine Learning: Concepts, Challenges, and Opportunities
**Authors:** Cerezo et al.
**Year:** 2022 | **Nature Reviews Physics** 4, 567
**Reading Time:** 3-4 hours

**Scope:** QML state of the field

**Topics Covered:**
- QML paradigms
- Barren plateaus
- Trainability challenges
- Potential advantages
- Open problems

---

## Theoretical Foundations

### 21. Quantum Complexity Theory
**Authors:** Survey: Watrous
**Year:** Ongoing updates
**Reading Time:** Variable

**Key Resources:**
- Watrous: "Quantum Computational Complexity" (2008 survey)
- Aaronson: Lecture notes and blog
- Vidick & Watrous: "Quantum Proofs" (2016)

---

### 22. Quantum Entanglement
**Authors:** Horodecki et al.
**Year:** 2009 | **Reviews of Modern Physics** 81, 865
**Reading Time:** Long (reference text)

**Scope:** Comprehensive entanglement theory review

**Use As:** Reference rather than cover-to-cover reading

---

## Industry Roadmaps and Perspectives

### 23. IBM Quantum Roadmaps
**Source:** IBM Quantum Blog and Publications
**Updated:** Annually

**Key Documents:**
- IBM Quantum Development Roadmap
- Qiskit papers and documentation
- Utility-scale quantum computing papers

---

### 24. Google Quantum AI Publications
**Source:** Google AI Quantum Team
**Key Papers:**
- Quantum supremacy (2019)
- Error correction experiments (2021-2024)
- Willow processor papers

---

### 25. Industry Surveys and Reports
**Sources:**
- McKinsey Quantum Technology Reports
- BCG Quantum Computing Reports
- Quantum Computing Report (website)

---

## Reading Schedule Suggestion

### Week 193 Priority Reading

**Day 1-2:**
- Preskill NISQ paper (#1)
- Roffe QEC introduction (#3)

**Day 3:**
- Bharti et al. NISQ algorithms (#2) - focus on introduction and conclusions

**Day 4-5:**
- One hardware review for your preferred platform (#8-12)
- Campbell et al. fault tolerance overview (#13)

**Day 6-7:**
- Area-specific reviews based on emerging interests
- Industry perspectives

---

## How to Use This Resource

1. **Initial Survey:** Read Foundation reviews (#1-3) completely
2. **Area Exploration:** Skim relevant area reviews, focus on open problems sections
3. **Deep Dive:** Return for thorough reading during Week 194
4. **Reference:** Use as ongoing reference throughout Month 49

---

## Staying Current

**Set Up Alerts For:**
- Key author names (Preskill, Terhal, Aaronson, etc.)
- PRX Quantum new articles
- Nature Physics quantum articles
- QIP conference proceedings

**Regular Checks:**
- arXiv quant-ph daily
- Google Scholar recommendations weekly
- Review article citations monthly

---

*Last Updated: Month 49, Week 193*
