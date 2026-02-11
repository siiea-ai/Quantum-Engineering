# Standard QM/QIT References

## Purpose

This document provides a curated list of authoritative references for quantum mechanics and quantum information theory background sections. Use these for verification, citation, and deeper reading.

---

## Tier 1: Essential Textbooks

These are the standard references that every quantum information thesis should cite.

### Nielsen & Chuang (2010)
**Title:** *Quantum Computation and Quantum Information* (10th Anniversary Edition)

**Citation:**
```bibtex
@book{nielsen2010quantum,
  author    = {Nielsen, Michael A. and Chuang, Isaac L.},
  title     = {Quantum Computation and Quantum Information},
  publisher = {Cambridge University Press},
  year      = {2010},
  edition   = {10th Anniversary},
  isbn      = {978-1107002173}
}
```

**Use for:**
- Quantum mechanics foundations (Chapter 2)
- Quantum operations and channels (Chapter 8)
- Quantum error correction basics (Chapter 10)
- Standard notation and conventions

**Key sections to cite:**
- Section 2.1: Linear algebra review
- Section 2.2: Postulates of quantum mechanics
- Section 8.2: Quantum operations
- Section 8.3: Operator-sum representation

---

### Preskill Lecture Notes
**Title:** *Lecture Notes on Quantum Computation*

**Citation:**
```bibtex
@misc{preskill1998lecture,
  author = {Preskill, John},
  title  = {Lecture Notes on Quantum Computation},
  year   = {1998--present},
  url    = {http://theory.caltech.edu/~preskill/ph229/}
}
```

**Use for:**
- Clear explanations of fundamentals
- Quantum error correction (Chapter 7)
- Fault-tolerant quantum computation (Chapter 9)
- Accessible yet rigorous treatment

**Key chapters:**
- Chapter 2: Foundations of quantum theory
- Chapter 3: Quantum entanglement
- Chapter 7: Quantum error correction (essential for Week 267)

---

### Wilde (2017)
**Title:** *Quantum Information Theory* (2nd Edition)

**Citation:**
```bibtex
@book{wilde2017quantum,
  author    = {Wilde, Mark M.},
  title     = {Quantum Information Theory},
  publisher = {Cambridge University Press},
  year      = {2017},
  edition   = {2nd},
  isbn      = {978-1107176164}
}
```

**Use for:**
- Rigorous information theory
- Entropy and information measures
- Quantum channel capacity
- Mathematical depth and precision

**Key chapters:**
- Chapter 3: The Noiseless Quantum Theory
- Chapter 5: The von Neumann Entropy
- Chapter 10: Classical Communication

---

## Tier 2: Specialized References

### Watrous (2018)
**Title:** *The Theory of Quantum Information*

**Citation:**
```bibtex
@book{watrous2018theory,
  author    = {Watrous, John},
  title     = {The Theory of Quantum Information},
  publisher = {Cambridge University Press},
  year      = {2018},
  isbn      = {978-1107180567}
}
```

**Use for:**
- Mathematical rigor
- Semidefinite programming formulations
- Quantum channels and operations
- Advanced information-theoretic concepts

---

### Cover & Thomas (2006)
**Title:** *Elements of Information Theory* (2nd Edition)

**Citation:**
```bibtex
@book{cover2006elements,
  author    = {Cover, Thomas M. and Thomas, Joy A.},
  title     = {Elements of Information Theory},
  publisher = {Wiley-Interscience},
  year      = {2006},
  edition   = {2nd},
  isbn      = {978-0471241959}
}
```

**Use for:**
- Classical information theory background
- Shannon entropy and properties
- Channel capacity concepts
- Foundation for quantum extensions

---

### Sakurai & Napolitano (2017)
**Title:** *Modern Quantum Mechanics* (2nd Edition)

**Citation:**
```bibtex
@book{sakurai2017modern,
  author    = {Sakurai, J. J. and Napolitano, Jim},
  title     = {Modern Quantum Mechanics},
  publisher = {Cambridge University Press},
  year      = {2017},
  edition   = {2nd},
  isbn      = {978-1108422413}
}
```

**Use for:**
- Traditional quantum mechanics formalism
- Spin systems and angular momentum
- Time evolution and perturbation theory
- Standard physics presentation

---

## Tier 3: Review Articles

### Horodecki et al. (2009) - Entanglement Review
**Title:** "Quantum Entanglement"

**Citation:**
```bibtex
@article{horodecki2009quantum,
  author  = {Horodecki, Ryszard and Horodecki, Pawe{\l} and Horodecki, Micha{\l} and Horodecki, Karol},
  title   = {Quantum entanglement},
  journal = {Reviews of Modern Physics},
  volume  = {81},
  pages   = {865--942},
  year    = {2009},
  doi     = {10.1103/RevModPhys.81.865}
}
```

**Use for:**
- Comprehensive entanglement theory
- Separability criteria
- Entanglement measures
- Distillation and manipulation

---

### Bengtsson & Życzkowski (2017)
**Title:** *Geometry of Quantum States* (2nd Edition)

**Citation:**
```bibtex
@book{bengtsson2017geometry,
  author    = {Bengtsson, Ingemar and {\.Z}yczkowski, Karol},
  title     = {Geometry of Quantum States},
  publisher = {Cambridge University Press},
  year      = {2017},
  edition   = {2nd},
  isbn      = {978-1107026254}
}
```

**Use for:**
- Geometric perspectives on quantum states
- Bloch sphere and generalizations
- Entanglement geometry
- Mathematical depth

---

## Reference by Topic

### Quantum States

| Topic | Primary Reference | Page/Section |
|-------|------------------|--------------|
| Dirac notation | Nielsen & Chuang | Section 2.1 |
| Density matrices | Nielsen & Chuang | Section 2.4 |
| Bloch sphere | Nielsen & Chuang | Box 2.2 |
| Partial trace | Nielsen & Chuang | Section 2.4.3 |
| Purification | Wilde | Section 5.1 |

### Quantum Operations

| Topic | Primary Reference | Page/Section |
|-------|------------------|--------------|
| Unitary gates | Nielsen & Chuang | Section 4.2 |
| CPTP maps | Nielsen & Chuang | Section 8.2 |
| Kraus representation | Nielsen & Chuang | Section 8.2.4 |
| Choi-Jamiołkowski | Watrous | Chapter 2 |
| Depolarizing channel | Nielsen & Chuang | Section 8.3.4 |

### Quantum Measurement

| Topic | Primary Reference | Page/Section |
|-------|------------------|--------------|
| Projective measurement | Nielsen & Chuang | Section 2.2.5 |
| POVM formalism | Nielsen & Chuang | Section 2.2.6 |
| Born rule | Preskill | Chapter 2 |
| Generalized measurements | Wilde | Section 4.2 |

### Entanglement

| Topic | Primary Reference | Page/Section |
|-------|------------------|--------------|
| Bell states | Nielsen & Chuang | Section 1.3.6 |
| Schmidt decomposition | Nielsen & Chuang | Section 2.5 |
| Entanglement entropy | Preskill | Chapter 3 |
| LOCC | Horodecki et al. | Section V |
| Monogamy | Horodecki et al. | Section IX.D |

### Information Theory

| Topic | Primary Reference | Page/Section |
|-------|------------------|--------------|
| Von Neumann entropy | Wilde | Chapter 5 |
| Entropy properties | Nielsen & Chuang | Section 11.3 |
| Relative entropy | Wilde | Section 5.2 |
| Mutual information | Wilde | Section 5.4 |
| Holevo bound | Nielsen & Chuang | Section 12.1 |

---

## Notation Standards

Follow these notation conventions (from Nielsen & Chuang):

| Symbol | Meaning | Alternative |
|--------|---------|-------------|
| $\|0\rangle$, $\|1\rangle$ | Computational basis states | $\|↑\rangle$, $\|↓\rangle$ for spin |
| $\rho$ | Density operator | $\varrho$ (less common) |
| $\mathcal{E}$ | Quantum channel | $\Phi$ (Watrous convention) |
| $X$, $Y$, $Z$ | Pauli operators | $\sigma_x$, $\sigma_y$, $\sigma_z$ |
| $H$ | Hadamard gate | — |
| $S$, $T$ | Phase gates | $P$, $T$ |
| $\text{CNOT}$ | Controlled-NOT | $\text{CX}$, $\Lambda(X)$ |
| $I$ | Identity operator | $\mathbb{1}$, $\mathbf{1}$ |
| $\text{Tr}$ | Trace | $\text{tr}$ |
| $\text{Tr}_B$ | Partial trace over $B$ | — |
| $\log$ | Base-2 logarithm | $\log_2$ (explicit) |
| $\ln$ | Natural logarithm | — |
| $S(\rho)$ | Von Neumann entropy | $H(\rho)$ (older texts) |

---

## Citation Best Practices

### When to Cite

**Always cite for:**
- Specific theorems or results (e.g., "The Stinespring dilation theorem [Stinespring 1955]")
- Historical attributions (e.g., "Bell's theorem [Bell 1964]")
- Definitions that might vary (e.g., "We follow the convention of [Nielsen & Chuang 2010]")

**Don't over-cite:**
- Basic definitions everyone agrees on
- Standard mathematical facts
- Your own notation choices

### Citation Format

Use consistent citation style. Example:

```
The von Neumann entropy of a quantum state \rho is defined as
S(\rho) = -\Tr(\rho \log \rho) [Nielsen & Chuang 2010, Section 11.3].
```

Or with numbered references:
```
The von Neumann entropy is defined as S(\rho) = -\Tr(\rho \log \rho)~\cite{nielsen2010quantum}.
```

---

## Online Resources

### arXiv.org
- Category: quant-ph
- Essential for recent papers
- Many textbooks have arXiv versions

### Qiskit Textbook
- URL: learn.qiskit.org
- Good for implementation perspectives
- Cite sparingly (non-peer-reviewed)

### Quantum Computing Stack Exchange
- URL: quantumcomputing.stackexchange.com
- Good for resolving technical questions
- Don't cite directly; find primary sources

---

## BibTeX Template

```bibtex
% Standard references for QM/QIT background

@book{nielsen2010quantum,
  author    = {Nielsen, Michael A. and Chuang, Isaac L.},
  title     = {Quantum Computation and Quantum Information},
  publisher = {Cambridge University Press},
  year      = {2010},
  edition   = {10th Anniversary}
}

@misc{preskill1998lecture,
  author = {Preskill, John},
  title  = {Lecture Notes on Quantum Computation},
  year   = {1998},
  url    = {http://theory.caltech.edu/~preskill/ph229/}
}

@book{wilde2017quantum,
  author    = {Wilde, Mark M.},
  title     = {Quantum Information Theory},
  publisher = {Cambridge University Press},
  year      = {2017},
  edition   = {2nd}
}

@book{watrous2018theory,
  author    = {Watrous, John},
  title     = {The Theory of Quantum Information},
  publisher = {Cambridge University Press},
  year      = {2018}
}

@article{horodecki2009quantum,
  author  = {Horodecki, Ryszard and Horodecki, Pawe{\l} and Horodecki, Micha{\l} and Horodecki, Karol},
  title   = {Quantum entanglement},
  journal = {Reviews of Modern Physics},
  volume  = {81},
  pages   = {865--942},
  year    = {2009}
}

@book{cover2006elements,
  author    = {Cover, Thomas M. and Thomas, Joy A.},
  title     = {Elements of Information Theory},
  publisher = {Wiley-Interscience},
  year      = {2006},
  edition   = {2nd}
}
```

---

## Further Reading

For specialized topics beyond the basics:

- **Quantum computing models:** Kitaev, Shen, Vyalyi (2002)
- **Quantum algorithms:** Montanaro (2016) review
- **Quantum complexity:** Watrous (2009) review
- **Quantum thermodynamics:** Goold et al. (2016)
- **Quantum metrology:** Giovannetti et al. (2011)

These may be relevant depending on your thesis focus.
