# Quantum Error Correction References

## Purpose

This document provides a comprehensive reference list for quantum error correction background sections. Organized by topic with BibTeX entries and usage notes.

---

## Foundational Papers

### Shor's First Quantum Code (1995)

**Citation:**
```bibtex
@article{shor1995scheme,
  author  = {Shor, Peter W.},
  title   = {Scheme for reducing decoherence in quantum computer memory},
  journal = {Physical Review A},
  volume  = {52},
  pages   = {R2493--R2496},
  year    = {1995},
  doi     = {10.1103/PhysRevA.52.R2493}
}
```

**Use for:** Historical context, first demonstration of QEC possibility

---

### Steane Code (1996)

**Citation:**
```bibtex
@article{steane1996error,
  author  = {Steane, Andrew M.},
  title   = {Error correcting codes in quantum theory},
  journal = {Physical Review Letters},
  volume  = {77},
  pages   = {793--797},
  year    = {1996},
  doi     = {10.1103/PhysRevLett.77.793}
}
```

**Use for:** CSS codes, transversal gates, practical code example

---

### Calderbank-Shor Construction (1996)

**Citation:**
```bibtex
@article{calderbank1996good,
  author  = {Calderbank, A. Robert and Shor, Peter W.},
  title   = {Good quantum error-correcting codes exist},
  journal = {Physical Review A},
  volume  = {54},
  pages   = {1098--1105},
  year    = {1996},
  doi     = {10.1103/PhysRevA.54.1098}
}
```

**Use for:** CSS construction from classical codes, good code families

---

### Gottesman's Thesis (1997)

**Citation:**
```bibtex
@phdthesis{gottesman1997stabilizer,
  author = {Gottesman, Daniel},
  title  = {Stabilizer Codes and Quantum Error Correction},
  school = {California Institute of Technology},
  year   = {1997},
  note   = {arXiv:quant-ph/9705052}
}
```

**Use for:** Stabilizer formalism, comprehensive treatment, standard reference

---

### Knill-Laflamme Conditions (1997)

**Citation:**
```bibtex
@article{knill1997theory,
  author  = {Knill, Emanuel and Laflamme, Raymond},
  title   = {Theory of quantum error-correcting codes},
  journal = {Physical Review A},
  volume  = {55},
  pages   = {900--911},
  year    = {1997},
  doi     = {10.1103/PhysRevA.55.900}
}
```

**Use for:** Necessary and sufficient conditions for QEC, fundamental theory

---

## Fault Tolerance

### Shor's Fault-Tolerant Computation (1996)

**Citation:**
```bibtex
@inproceedings{shor1996fault,
  author    = {Shor, Peter W.},
  title     = {Fault-tolerant quantum computation},
  booktitle = {Proceedings of 37th FOCS},
  pages     = {56--65},
  year      = {1996},
  doi       = {10.1109/SFCS.1996.548464}
}
```

**Use for:** Fault-tolerant constructions, concatenated codes

---

### Threshold Theorem - Aharonov-Ben-Or (1997)

**Citation:**
```bibtex
@inproceedings{aharonov1997fault,
  author    = {Aharonov, Dorit and Ben-Or, Michael},
  title     = {Fault-tolerant quantum computation with constant error},
  booktitle = {Proceedings of 29th STOC},
  pages     = {176--188},
  year      = {1997},
  doi       = {10.1145/258533.258579}
}
```

**Use for:** Threshold theorem, theoretical foundations

---

### Threshold Theorem - Knill et al. (1998)

**Citation:**
```bibtex
@article{knill1998resilient,
  author  = {Knill, Emanuel and Laflamme, Raymond and Zurek, Wojciech H.},
  title   = {Resilient quantum computation},
  journal = {Science},
  volume  = {279},
  pages   = {342--345},
  year    = {1998},
  doi     = {10.1126/science.279.5349.342}
}
```

**Use for:** Threshold theorem, alternative approach

---

### Aliferis-Gottesman-Preskill (2006)

**Citation:**
```bibtex
@article{aliferis2006quantum,
  author  = {Aliferis, Panos and Gottesman, Daniel and Preskill, John},
  title   = {Quantum accuracy threshold for concatenated distance-3 codes},
  journal = {Quantum Information and Computation},
  volume  = {6},
  pages   = {97--165},
  year    = {2006}
}
```

**Use for:** Rigorous threshold analysis, concatenated codes

---

### Magic State Distillation (2005)

**Citation:**
```bibtex
@article{bravyi2005universal,
  author  = {Bravyi, Sergey and Kitaev, Alexei},
  title   = {Universal quantum computation with ideal {C}lifford gates and noisy ancillas},
  journal = {Physical Review A},
  volume  = {71},
  pages   = {022316},
  year    = {2005},
  doi     = {10.1103/PhysRevA.71.022316}
}
```

**Use for:** Magic states, universal fault-tolerant computation

---

### Eastin-Knill Theorem (2009)

**Citation:**
```bibtex
@article{eastin2009restrictions,
  author  = {Eastin, Bryan and Knill, Emanuel},
  title   = {Restrictions on transversal encoded quantum gate sets},
  journal = {Physical Review Letters},
  volume  = {102},
  pages   = {110502},
  year    = {2009},
  doi     = {10.1103/PhysRevLett.102.110502}
}
```

**Use for:** No-go theorem for transversal universal gates

---

## Topological Codes

### Kitaev's Toric Code (2003)

**Citation:**
```bibtex
@article{kitaev2003fault,
  author  = {Kitaev, A. Yu.},
  title   = {Fault-tolerant quantum computation by anyons},
  journal = {Annals of Physics},
  volume  = {303},
  pages   = {2--30},
  year    = {2003},
  doi     = {10.1016/S0003-4916(02)00018-0}
}
```

**Use for:** Toric code, anyons, topological protection

---

### Dennis et al. Threshold (2002)

**Citation:**
```bibtex
@article{dennis2002topological,
  author  = {Dennis, Eric and Kitaev, Alexei and Landahl, Andrew and Preskill, John},
  title   = {Topological quantum memory},
  journal = {Journal of Mathematical Physics},
  volume  = {43},
  pages   = {4452--4505},
  year    = {2002},
  doi     = {10.1063/1.1499754}
}
```

**Use for:** Surface code threshold, statistical mechanics mapping

---

### Fowler et al. Surface Code Review (2012)

**Citation:**
```bibtex
@article{fowler2012surface,
  author  = {Fowler, Austin G. and Mariantoni, Matteo and Martinis, John M. and Cleland, Andrew N.},
  title   = {Surface codes: Towards practical large-scale quantum computation},
  journal = {Physical Review A},
  volume  = {86},
  pages   = {032324},
  year    = {2012},
  doi     = {10.1103/PhysRevA.86.032324}
}
```

**Use for:** Comprehensive surface code review, practical implementation

---

### Bombin Color Codes (2006)

**Citation:**
```bibtex
@article{bombin2006topological,
  author  = {Bombin, H. and Martin-Delgado, M. A.},
  title   = {Topological quantum distillation},
  journal = {Physical Review Letters},
  volume  = {97},
  pages   = {180501},
  year    = {2006},
  doi     = {10.1103/PhysRevLett.97.180501}
}
```

**Use for:** Color codes, transversal non-Clifford gates

---

### Lattice Surgery (2012)

**Citation:**
```bibtex
@article{horsman2012surface,
  author  = {Horsman, Clare and Fowler, Austin G. and Devitt, Simon and Van Meter, Rodney},
  title   = {Surface code quantum computing by lattice surgery},
  journal = {New Journal of Physics},
  volume  = {14},
  pages   = {123011},
  year    = {2012},
  doi     = {10.1088/1367-2630/14/12/123011}
}
```

**Use for:** Lattice surgery, code deformation, logical gates

---

## Decoding

### MWPM Decoding (2010)

**Citation:**
```bibtex
@article{edmonds2010optimal,
  author  = {Fowler, Austin G.},
  title   = {Minimum weight perfect matching of fault-tolerant topological quantum error correction in average {$O(1)$} parallel time},
  journal = {Quantum Information and Computation},
  volume  = {15},
  pages   = {145--158},
  year    = {2015}
}
```

**Use for:** MWPM decoder, practical implementation

---

### Union-Find Decoder (2017)

**Citation:**
```bibtex
@article{delfosse2017almost,
  author  = {Delfosse, Nicolas and Nickerson, Naomi H.},
  title   = {Almost-linear time decoding algorithm for topological codes},
  journal = {arXiv:1709.06218},
  year    = {2017}
}
```

**Use for:** Fast decoding, linear-time algorithms

---

### Neural Network Decoding (2017)

**Citation:**
```bibtex
@article{torlai2017neural,
  author  = {Torlai, Giacomo and Melko, Roger G.},
  title   = {Neural decoder for topological codes},
  journal = {Physical Review Letters},
  volume  = {119},
  pages   = {030501},
  year    = {2017},
  doi     = {10.1103/PhysRevLett.119.030501}
}
```

**Use for:** Machine learning decoders

---

## Review Articles

### Terhal QEC Review (2015)

**Citation:**
```bibtex
@article{terhal2015quantum,
  author  = {Terhal, Barbara M.},
  title   = {Quantum error correction for quantum memories},
  journal = {Reviews of Modern Physics},
  volume  = {87},
  pages   = {307--346},
  year    = {2015},
  doi     = {10.1103/RevModPhys.87.307}
}
```

**Use for:** Comprehensive modern QEC review

---

### Campbell et al. Roadmap (2017)

**Citation:**
```bibtex
@article{campbell2017roads,
  author  = {Campbell, Earl T. and Terhal, Barbara M. and Vuillot, Christophe},
  title   = {Roads towards fault-tolerant universal quantum computation},
  journal = {Nature},
  volume  = {549},
  pages   = {172--179},
  year    = {2017},
  doi     = {10.1038/nature23460}
}
```

**Use for:** Fault-tolerance roadmap, universal computation strategies

---

### Roffe Tutorial (2019)

**Citation:**
```bibtex
@article{roffe2019quantum,
  author  = {Roffe, Joschka},
  title   = {Quantum error correction: an introductory guide},
  journal = {Contemporary Physics},
  volume  = {60},
  pages   = {226--245},
  year    = {2019},
  doi     = {10.1080/00107514.2019.1667078}
}
```

**Use for:** Accessible introduction, pedagogical

---

## Specialized Topics

### Biased Noise (2018-2020)

```bibtex
@article{tuckett2018ultrahigh,
  author  = {Tuckett, David K. and Bartlett, Stephen D. and Flammia, Steven T.},
  title   = {Ultrahigh Error Threshold for Surface Codes with Biased Noise},
  journal = {Physical Review Letters},
  volume  = {120},
  pages   = {050505},
  year    = {2018},
  doi     = {10.1103/PhysRevLett.120.050505}
}

@article{tuckett2020tailoring,
  author  = {Tuckett, David K. and Bartlett, Stephen D. and Flammia, Steven T. and Brown, Benjamin J.},
  title   = {Tailoring Surface Codes for Highly Biased Noise},
  journal = {Physical Review X},
  volume  = {10},
  pages   = {041020},
  year    = {2020},
  doi     = {10.1103/PhysRevX.10.041020}
}
```

---

### Bosonic Codes

```bibtex
@article{gottesman2001encoding,
  author  = {Gottesman, Daniel and Kitaev, Alexei and Preskill, John},
  title   = {Encoding a qubit in an oscillator},
  journal = {Physical Review A},
  volume  = {64},
  pages   = {012310},
  year    = {2001},
  doi     = {10.1103/PhysRevA.64.012310}
}

@article{albert2018performance,
  author  = {Albert, Victor V. and others},
  title   = {Performance and structure of single-mode bosonic codes},
  journal = {Physical Review A},
  volume  = {97},
  pages   = {032346},
  year    = {2018},
  doi     = {10.1103/PhysRevA.97.032346}
}
```

---

### LDPC Codes

```bibtex
@article{breuckmann2021quantum,
  author  = {Breuckmann, Nikolas P. and Eberhardt, Jens Niklas},
  title   = {Quantum low-density parity-check codes},
  journal = {PRX Quantum},
  volume  = {2},
  pages   = {040101},
  year    = {2021},
  doi     = {10.1103/PRXQuantum.2.040101}
}
```

---

## Experimental Demonstrations

### Google Quantum AI (2023-2024)

```bibtex
@article{google2023suppressing,
  author  = {{Google Quantum AI}},
  title   = {Suppressing quantum errors by scaling a surface code logical qubit},
  journal = {Nature},
  volume  = {614},
  pages   = {676--681},
  year    = {2023},
  doi     = {10.1038/s41586-022-05434-1}
}
```

---

### IBM Quantum (2023-2024)

```bibtex
@article{kim2023evidence,
  author  = {Kim, Youngseok and others},
  title   = {Evidence for the utility of quantum computing before fault tolerance},
  journal = {Nature},
  volume  = {618},
  pages   = {500--505},
  year    = {2023},
  doi     = {10.1038/s41586-023-06096-3}
}
```

---

## Quick Reference by Topic

| Topic | Primary References |
|-------|-------------------|
| QEC foundations | Shor 1995, Steane 1996, Knill-Laflamme 1997 |
| Stabilizer formalism | Gottesman 1997 |
| CSS codes | Calderbank-Shor 1996, Steane 1996 |
| Threshold theorem | Aharonov-Ben-Or 1997, Knill et al. 1998 |
| Fault tolerance | Shor 1996, Aliferis et al. 2006 |
| Magic states | Bravyi-Kitaev 2005 |
| Surface codes | Kitaev 2003, Dennis et al. 2002, Fowler et al. 2012 |
| Color codes | Bombin 2006 |
| Decoding | Fowler 2015 (MWPM), Delfosse 2017 (UF) |
| Reviews | Terhal 2015, Campbell et al. 2017 |

---

## Template BibTeX File

Save this as `qec_references.bib` and add to your thesis:

```bibtex
% ============================================================================
% QUANTUM ERROR CORRECTION REFERENCES
% ============================================================================

% === FOUNDATIONAL ===
% [Add entries from above]

% === FAULT TOLERANCE ===
% [Add entries from above]

% === TOPOLOGICAL CODES ===
% [Add entries from above]

% === DECODING ===
% [Add entries from above]

% === REVIEWS ===
% [Add entries from above]

% === SPECIALIZED ===
% [Add your specialized area references]

% === EXPERIMENTAL ===
% [Add recent experimental papers]
```
