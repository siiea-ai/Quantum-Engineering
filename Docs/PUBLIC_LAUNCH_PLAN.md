# Quantum Engineering: Public Launch & Development Plan

**Date:** February 18, 2026
**Author:** Imran Ali | Siiea Innovations, LLC
**Status:** Ready for execution

---

## Executive Summary

This document outlines the complete strategy for making the Quantum Engineering curriculum public, expanding it with interactive Jupyter notebooks and Apple MLX-optimized labs, and positioning it as the definitive open-source quantum science self-study resource.

After extensive competitive research across 35+ repositories and platforms, **nothing like this project exists anywhere.** No repository on GitHub offers a structured day-by-day, multi-year, PhD-level quantum science curriculum. This project occupies a category of one.

---

## Part 1: Competitive Landscape Analysis

### The Market (What Exists)

#### Tier 1: Curated Link Lists (High Stars, No Content)

| Repository | Stars | What It Is |
|-----------|-------|-----------|
| awesome-quantum-computing | ~3,100 | Link directory — no original content |
| awesome-quantum-software (QOSF) | ~1,900 | Software catalog — not educational |
| awesome-quantum-machine-learning | ~3,300 | Paper list for QML — no curriculum |

#### Tier 2: Corporate/Institutional (Narrow Scope)

| Repository | Stars | What It Is |
|-----------|-------|-----------|
| Microsoft QuantumKatas | ~4,800 | Q# exercises — **ARCHIVED Aug 2024** |
| Qiskit Textbook (IBM) | ~680 | Jupyter notebooks — vendor-locked to Qiskit/IBM |
| Qiskit Tutorials | ~2,500 | SDK documentation — not a curriculum |
| NVIDIA CUDA-Q Academic | ~270 | GPU quantum labs — vendor-specific |
| PennyLane QML Demos | ~650 | QML demos — assumes existing knowledge |
| Microsoft quantum-curriculum-samples | ~80 | 14-week Q# course — single semester only |

#### Tier 3: Open-Source Degree Programs (Different Domain)

| Repository | Stars | What It Is |
|-----------|-------|-----------|
| OSSU Computer Science | ~201,000 | Self-taught CS degree — **no quantum track** |
| ForrestKnight open-source-cs | ~21,000 | CS course links — YouTube-driven popularity |
| Open-Source-Physics-Curriculum | ~24 | Physics link list — no original content |

#### Tier 4: MLX + Quantum (Empty Space)

| Finding | Details |
|---------|---------|
| MLX quantum repositories | **Zero** educational repos combine MLX with quantum computing |
| osxQuantum (QNeura) | Only MLX quantum simulator found — commercial, not educational |
| Opportunity | **First mover advantage** in MLX + quantum education |

### What We Do That Nobody Else Does

1. **2,016 day-by-day lesson files** — no other repo has daily structure
2. **Full mathematical foundations (Year 0)** — everyone else assumes prerequisites
3. **6-year PhD-equivalent scope** — nothing else attempts a full PhD curriculum
4. **Integrated research training (Years 4-5)** — thesis writing, defense prep
5. **Deep error correction coverage (Years 2-3)** — most repos mention QEC briefly
6. **Qualifying exam simulation (Year 3)** — unique to this project
7. **Framework-agnostic Python** — not locked to Qiskit, Cirq, or PennyLane
8. **Self-contained** — all content is in the repository, no broken external links

### Strategic Position

**We are "OSSU for Quantum Science & Engineering."** OSSU has 201K stars and no quantum track. There is literally zero competition in the structured quantum self-study space.

---

## Part 2: What We Have Today

### Current Repository Stats

| Metric | Count |
|--------|-------|
| Total markdown files | ~1,999 |
| Day lesson files (Years 0-2) | 1,008 |
| Support files (Years 3-5) | ~1,000 |
| Python files | 9 |
| Jupyter notebooks | 0 |
| Total project size | ~52 MB |
| Git commits | 3 |

### Current Structure

```
Quantum Engineering/
├── Year_0_Mathematical_Foundations/     # 336 day files
├── Year_1_Quantum_Mechanics_Core/      # 336 day files
├── Year_2_Advanced_Quantum_Science/    # 336 day files
├── Year_3_Qualifying_Exam/             # Review guides + problem sets
├── Year_4_Research_Phase_I/            # Guides + templates + code
├── Year_5_Research_Phase_II/           # Guides + templates
├── Docs/                               # Planning documents
├── Archive/                            # Old planning docs
├── README.md                           # Public overview
├── ARTICLE_SUBSTACK.md                 # Publication article
├── LICENSE                             # CC BY-NC-SA 4.0
└── CLAUDE.md                           # Development instructions
```

### What's Missing (Gaps to Fill)

| Gap | Impact | Priority |
|-----|--------|----------|
| No Jupyter notebooks | Can't run any code interactively | HIGH |
| No MLX integration | Missing unique differentiator | HIGH |
| No GitHub topics/tags | Poor discoverability | HIGH |
| Not listed on awesome-quantum-computing | Missing 3,100+ stargazer visibility | HIGH |
| No CONTRIBUTING.md | Blocks community contributions | MEDIUM |
| No requirements.txt / pyproject.toml | No reproducible environment | MEDIUM |
| 14,000+ hours claim inconsistency | README says ~10,000+, homepage says 14,000+ | LOW |

---

## Part 3: Hardware Advantage

### Apple Mac Studio (M-series, 512GB Unified Memory)

This is one of the most powerful personal machines available for scientific computing. It unlocks capabilities most researchers can't access locally.

#### Quantum Simulation Scale

| Qubits | State Vector Size | 512GB Mac Studio |
|--------|-------------------|------------------|
| 20 | 16 MB | Trivial |
| 25 | 512 MB | Easy |
| 30 | 16 GB | Comfortable |
| 33 | 128 GB | Feasible |
| 35 | 512 GB | At the limit |

Most laptops max out at ~20-25 qubits. With 512GB, we can push to **33+ qubits**, enabling simulations that compete with cluster-scale computation.

#### MLX Framework Advantages

- **Unified memory** — no CPU-GPU data transfer overhead
- **Metal acceleration** — native GPU compute on Apple Silicon
- **NumPy-compatible API** — minimal learning curve
- **Lazy evaluation** — efficient memory use for large tensors
- **Automatic differentiation** — built-in for ML training loops

#### What This Enables

1. **Large-scale quantum state simulation** — simulate real quantum algorithms at meaningful scale
2. **Quantum ML training** — train variational circuits and neural decoders locally
3. **Error correction research** — decode surface codes with ML decoders on real hardware
4. **Publication-quality results** — generate figures and data for research papers

---

## Part 4: Development Roadmap

### Phase 1: Public Launch Preparation (Current — Week of Feb 18)

**Goal:** Make the repository public-ready and publish.

- [x] Clean all sensitive information (Conexly, personal paths, API keys)
- [x] Update README with notebook roadmap and competitive positioning
- [x] Update CLAUDE.md with MLX/Jupyter development direction
- [x] Create this launch plan document
- [ ] Add `requirements.txt` for Python environment
- [ ] Add `.github/` directory with issue templates
- [ ] Add GitHub repository topics (quantum-computing, physics, self-study, curriculum, jupyter, mlx, education, open-source)
- [ ] Verify all links in README work
- [ ] Final commit and push to public repository

### Phase 2: Foundation Notebooks (Weeks 1-4)

**Goal:** Create the first batch of interactive Jupyter notebooks.

#### Week 1: Environment & Template

- [ ] Create `notebooks/` directory structure
- [ ] Create `requirements.txt` and `pyproject.toml`
- [ ] Build a template notebook with standard structure:
  - Theory overview (markdown cells with LaTeX)
  - Guided implementation (code cells with instructions)
  - Visualization (matplotlib/plotly outputs)
  - Exercises (empty cells for learner to fill)
  - Solutions (collapsed/hidden cells)
- [ ] Create `notebooks/README.md` explaining how to use them

#### Week 2-3: Year 0 Pilot Notebooks (Months 1-3)

- [ ] `Week_01_Limits.ipynb` — epsilon-delta visualization, limit convergence
- [ ] `Week_02_Derivatives.ipynb` — derivative as slope, physics applications
- [ ] `Week_03_Integration.ipynb` — Riemann sums, FTC visualization
- [ ] `Week_04_Series.ipynb` — Taylor series convergence, approximation
- [ ] `Week_05_Vectors.ipynb` — 3D vector operations, coordinate systems
- [ ] `Week_06_Partial_Derivatives.ipynb` — gradient fields, directional derivatives
- [ ] `Week_07_Multiple_Integrals.ipynb` — double/triple integrals, Jacobians
- [ ] `Week_08_Vector_Calculus.ipynb` — curl, divergence, Stokes' theorem
- [ ] `Week_09_Vector_Spaces.ipynb` — basis, dimension, linear independence
- [ ] `Week_10_Linear_Transformations.ipynb` — matrix ops, null space, range
- [ ] `Week_11_Systems.ipynb` — Gaussian elimination, LU decomposition
- [ ] `Week_12_Inner_Products.ipynb` — Gram-Schmidt, orthogonal projections

#### Week 4: MLX Lab Prototype

- [ ] `MLX_Labs/00_setup_and_basics.ipynb` — MLX installation, array operations, Metal acceleration
- [ ] `MLX_Labs/01_quantum_state_simulation.ipynb` — state vectors and gates using MLX
- [ ] `MLX_Labs/02_large_scale_simulation.ipynb` — push to 30+ qubits on Mac Studio

### Phase 3: Quantum Notebooks + MLX Deep Dive (Weeks 5-12)

**Goal:** Year 1 notebooks and advanced MLX labs.

#### Year 1 Notebooks (Priority Selections)

- [ ] `Week_49_Hilbert_Space.ipynb` — complex vector spaces, Dirac notation
- [ ] `Week_50_Measurement.ipynb` — measurement simulation, Born rule
- [ ] `Week_51_Time_Evolution.ipynb` — Schrodinger equation numerical solver
- [ ] `Week_53_Wave_Packets.ipynb` — wave packet dynamics and spreading
- [ ] `Week_55_Harmonic_Oscillator.ipynb` — ladder operators, coherent states
- [ ] `Week_57_Angular_Momentum.ipynb` — spherical harmonics visualization
- [ ] `Week_58_Spin.ipynb` — Bloch sphere, Pauli matrices, Stern-Gerlach
- [ ] `Week_73_Density_Matrices.ipynb` — pure vs mixed states, partial trace
- [ ] `Week_76_Bell_States.ipynb` — CHSH inequality, entanglement
- [ ] `Week_81_Quantum_Gates.ipynb` — Qiskit circuit builder
- [ ] `Week_85_Deutsch_Jozsa.ipynb` — first quantum algorithm
- [ ] `Week_89_Shor_Algorithm.ipynb` — period finding, factoring

#### MLX Advanced Labs

- [ ] `MLX_Labs/03_quantum_neural_decoder.ipynb` — train neural network for QEC decoding
- [ ] `MLX_Labs/04_variational_quantum_eigensolver.ipynb` — VQE with MLX optimization
- [ ] `MLX_Labs/05_quantum_kernel_methods.ipynb` — quantum-enhanced classification
- [ ] `MLX_Labs/06_barren_plateau_analysis.ipynb` — trainability landscapes

### Phase 4: Community & Visibility (Weeks 8-16)

**Goal:** Build discoverability and community.

- [ ] Submit to [awesome-quantum-computing](https://github.com/desireevl/awesome-quantum-computing) list
- [ ] Submit to [awesome-quantum-software](https://github.com/qosf/awesome-quantum-software) list
- [ ] Publish Substack article (ARTICLE_SUBSTACK.md)
- [ ] Post on r/QuantumComputing, r/Physics, r/learnprogramming
- [ ] Share on Quantum Computing Stack Exchange
- [ ] Create GitHub Discussions for community Q&A
- [ ] Add CONTRIBUTING.md with contribution guidelines
- [ ] Set up GitHub Actions for notebook validation (nbval or pytest-notebook)

### Phase 5: Platform Integration (Months 3-6)

**Goal:** Connect with siiea.ai platform.

- [ ] Deploy curriculum on siiea.ai with interactive notebook viewer
- [ ] Enable progress tracking for notebook completion
- [ ] Launch first study cohort
- [ ] Integrate with IBM Quantum for real hardware labs
- [ ] Create video walkthroughs for key notebooks

---

## Part 5: Technical Specifications

### Python Environment

```
# requirements.txt
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
sympy>=1.12
jupyter>=1.0
jupyterlab>=4.0

# Quantum computing
qiskit>=1.0
qiskit-aer>=0.13
qutip>=5.0
pennylane>=0.34

# Apple Silicon (optional)
mlx>=0.6

# Visualization
plotly>=5.18
ipywidgets>=8.1

# Testing
pytest>=8.0
nbval>=0.11
```

### Notebook Template Structure

```python
# Cell 1: Title & Overview (Markdown)
"""
# Week XX: [Topic Name]
## Quantum Engineering Curriculum — Year X, Month XX

**Learning Objectives:**
- Objective 1
- Objective 2

**Prerequisites:** [Previous notebook link]
"""

# Cell 2: Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
# ... topic-specific imports

# Cell 3-N: Theory + Implementation (alternating markdown/code)

# Cell N+1: Exercises
"""
### Exercises
1. [Exercise description]
2. [Exercise description]
"""

# Cell N+2: Solutions (collapsed by default)
# ... solution code
```

### MLX Lab Template

```python
# Cell 1: MLX Setup Verification
import mlx.core as mx
import mlx.nn as nn
print(f"MLX device: {mx.default_device()}")
print(f"Available memory: estimate based on system")

# Cell 2: Quantum State Representation in MLX
# Use mx.array for state vectors, mx.matmul for gate application

# Cell 3-N: Experiment implementation

# Cell N+1: Performance Comparison
# Compare MLX vs NumPy for same computation
```

---

## Part 6: Success Metrics

### Launch Metrics (Month 1)

| Metric | Target |
|--------|--------|
| GitHub stars | 100+ |
| Forks | 20+ |
| Unique visitors | 1,000+ |
| awesome-quantum-computing listing | Submitted |

### Growth Metrics (Months 2-6)

| Metric | Target |
|--------|--------|
| GitHub stars | 1,000+ |
| Active notebook users | 100+ |
| Community discussions | 50+ threads |
| Substack article reads | 10,000+ |
| Contributors | 5+ |

### Long-Term Vision (Year 1)

| Metric | Target |
|--------|--------|
| GitHub stars | 5,000+ |
| Complete notebook coverage | Years 0-2 (288 notebooks) |
| MLX Labs | 20+ advanced notebooks |
| Study cohort members | 50+ active learners |
| Listed on | awesome-quantum-computing, awesome-quantum-software, OSSU resources |

---

## Part 7: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Low initial visibility | Medium | Medium | Submit to awesome lists, Substack article, Reddit posts |
| Notebook quality concerns | Low | High | Thorough testing, nbval CI, peer review |
| MLX framework changes | Low | Low | Pin versions, abstract MLX calls |
| Qiskit API breaking changes | Medium | Medium | Pin versions, test regularly |
| Scope creep on notebooks | High | Medium | Prioritize Year 0-1 first, template-driven approach |
| Community management overhead | Medium | Low | GitHub Discussions (async), clear CONTRIBUTING.md |

---

## Part 8: Immediate Action Items

### Right Now (This Session)

1. ~~Clean sensitive info from both projects~~ DONE
2. ~~Update README with notebook roadmap~~ DONE
3. ~~Update CLAUDE.md with development direction~~ DONE
4. ~~Create this launch plan~~ DONE
5. Add `requirements.txt` to repository
6. Final review and commit

### This Week

1. Publish Substack article
2. Push to public GitHub
3. Add GitHub topics and description
4. Submit to awesome-quantum-computing
5. Begin Phase 2 (first notebooks)

### This Month

1. Complete 12 Year 0 pilot notebooks
2. Complete 3 MLX lab prototypes
3. Set up notebook CI testing
4. Engage quantum computing communities

---

*This plan was developed through comprehensive competitive analysis of 35+ quantum education repositories and platforms, combined with technical assessment of the Apple Mac Studio M-series (512GB) hardware capabilities.*

*Quantum Engineering: Where no one else has gone.*

**Created by Imran Ali | Siiea Innovations, LLC**
**February 2026**
