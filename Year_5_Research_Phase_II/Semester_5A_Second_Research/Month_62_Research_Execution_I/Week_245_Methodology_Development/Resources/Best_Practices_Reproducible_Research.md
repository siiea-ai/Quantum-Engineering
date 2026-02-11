# Best Practices for Reproducible Quantum Research

## Overview

Reproducibility is the cornerstone of scientific progress. In quantum research, achieving reproducibility presents unique challenges due to the stochastic nature of quantum measurements, the complexity of quantum systems, and the rapid evolution of quantum hardware and software. This guide provides comprehensive best practices for ensuring your quantum research can be verified, extended, and built upon by the scientific community.

---

## 1. The Reproducibility Crisis and Quantum Science

### 1.1 Why Reproducibility Matters

**Scientific Validity**: Results that cannot be reproduced cannot be trusted or built upon.

**Resource Efficiency**: Reproducible research prevents duplication of effort across the community.

**Career Impact**: Reproducible work is cited more frequently and builds lasting reputation.

**Funding Requirements**: Major funding agencies increasingly mandate reproducibility measures.

### 1.2 Unique Challenges in Quantum Research

**Quantum Stochasticity**
- Measurement outcomes are inherently probabilistic
- Single-shot experiments are not reproducible by design
- Statistical reproducibility requires careful definition

**Hardware Variability**
- Quantum processors vary chip-to-chip
- Calibration parameters drift over time
- Access to specific hardware may be limited

**Rapid Evolution**
- Quantum software frameworks evolve quickly
- Hardware generations change rapidly
- Today's cutting-edge is tomorrow's obsolete

**Complex Dependencies**
- Quantum experiments depend on classical control systems
- Multi-layer software stacks
- Environmental sensitivity

---

## 2. Documentation Standards

### 2.1 The Documentation Pyramid

```
                    ┌───────────────┐
                    │   Summary     │  ← Quick reference
                    ├───────────────┤
                    │   Methods     │  ← How and why
                    ├───────────────┤
                    │   Protocols   │  ← Step-by-step
                    ├───────────────┤
                    │   Raw Notes   │  ← Everything
                    └───────────────┘
```

### 2.2 Essential Documentation Elements

**Research Narrative**
- What question are you answering?
- Why does it matter?
- What approach did you take?
- What did you find?

**Technical Specification**
- Exact parameter values with units
- Equipment specifications
- Software versions
- Environmental conditions

**Procedural Details**
- Step-by-step protocols
- Decision criteria
- Contingency procedures
- Quality checks

### 2.3 Living Documentation

Documentation should be:
- **Version-controlled**: Track all changes
- **Timestamped**: Know when things changed
- **Linked**: Connect related documents
- **Accessible**: Easy to find and read
- **Maintained**: Updated as methods evolve

---

## 3. Code and Software Best Practices

### 3.1 Code Organization

**Project Structure**
```
quantum_research_project/
├── README.md                 # Project overview
├── LICENSE                   # Usage terms
├── CHANGELOG.md             # Version history
├── setup.py / pyproject.toml # Package configuration
├── environment.yml          # Environment specification
├── src/                     # Source code
│   ├── __init__.py
│   ├── circuits/           # Quantum circuits
│   ├── analysis/           # Analysis code
│   ├── utils/              # Utilities
│   └── visualization/      # Plotting
├── tests/                   # Test suite
│   ├── test_circuits.py
│   └── test_analysis.py
├── notebooks/               # Jupyter notebooks
│   ├── exploration/        # Exploratory analysis
│   └── figures/            # Figure generation
├── data/                    # Data directory
│   ├── raw/                # Immutable raw data
│   ├── processed/          # Processed data
│   └── results/            # Final results
├── docs/                    # Documentation
│   ├── methodology.md
│   └── protocols/
└── configs/                 # Configuration files
    ├── experiment_config.yaml
    └── analysis_config.yaml
```

### 3.2 Coding Standards

**Style Guidelines**
```python
"""
Module docstring explaining purpose.

Author: Your Name
Date: Creation date
Version: Current version
"""

from typing import List, Tuple, Optional
import numpy as np

# Constants with clear naming
PLANCK_CONSTANT = 6.62607015e-34  # J·s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


def calculate_fidelity(
    rho: np.ndarray,
    sigma: np.ndarray,
    tolerance: float = 1e-10
) -> float:
    """
    Calculate quantum state fidelity between two density matrices.

    The fidelity is defined as F(ρ,σ) = (Tr√(√ρ σ √ρ))².

    Args:
        rho: First density matrix (n x n complex array)
        sigma: Second density matrix (n x n complex array)
        tolerance: Numerical tolerance for eigenvalue cleanup

    Returns:
        Fidelity value between 0 and 1

    Raises:
        ValueError: If matrices have incompatible dimensions

    Example:
        >>> rho = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
        >>> sigma = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+⟩⟨+|
        >>> calculate_fidelity(rho, sigma)
        0.5

    References:
        [1] Nielsen & Chuang, Chapter 9
    """
    if rho.shape != sigma.shape:
        raise ValueError(f"Shape mismatch: {rho.shape} vs {sigma.shape}")

    # Implementation...
    pass
```

### 3.3 Version Control Practices

**Commit Messages**
```
type(scope): brief description

Detailed explanation of changes:
- What was changed
- Why it was changed
- Any notable effects

References: #issue_number
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Branch Strategy**
```
main ─────────────────────────────────────►
        │                       ▲
        └─── feature/new-analysis ──────┘
```

### 3.4 Environment Management

**Conda Environment**
```yaml
name: quantum_research
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - numpy=1.24.3
  - scipy=1.11.2
  - matplotlib=3.7.2
  - jupyter=1.0.0
  - pip:
    - qiskit==0.45.0
    - pennylane==0.32.0
```

**Docker Container**
```dockerfile
FROM continuumio/miniconda3:23.5.2-0

COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml

COPY . /app
WORKDIR /app

ENTRYPOINT ["conda", "run", "-n", "quantum_research"]
CMD ["python", "main.py"]
```

---

## 4. Data Management

### 4.1 The FAIR Principles

**Findable**
- Persistent identifiers (DOI)
- Rich metadata
- Indexed in searchable resources

**Accessible**
- Retrievable by identifier
- Open, free protocol
- Metadata always accessible

**Interoperable**
- Formal, shared language
- FAIR vocabularies
- Qualified references

**Reusable**
- Clear usage license
- Detailed provenance
- Community standards

### 4.2 Data Organization

**Naming Conventions**
```
[PROJECT]_[EXPERIMENT]_[DATE]_[CONDITION]_[RUN].[EXT]

Examples:
QST_BELLSTATE_20240115_T10mK_R001.h5
VQE_H2_20240115_D6_COBYLA_001.json
```

**Metadata Standards**
```yaml
# metadata.yaml
experiment:
  name: "Bell State Tomography"
  id: "QST_BELLSTATE_20240115_001"
  date: "2024-01-15"
  researcher: "Your Name"

hardware:
  device: "IBM Washington"
  qubits: [0, 1]
  calibration_date: "2024-01-15"

parameters:
  shots: 8192
  optimization_level: 3
  measurement_basis: ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]

environment:
  qiskit_version: "0.45.0"
  python_version: "3.10.12"
```

### 4.3 Data Integrity

**Checksums**
```python
import hashlib

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()

# Record hash with data
data_hash = compute_file_hash("data/raw/experiment_001.h5")
```

**Validation**
```python
def validate_measurement_data(data: dict) -> bool:
    """Validate measurement data structure and values."""
    required_keys = ['counts', 'shots', 'metadata']

    # Check structure
    if not all(key in data for key in required_keys):
        return False

    # Check consistency
    total_counts = sum(data['counts'].values())
    if total_counts != data['shots']:
        return False

    # Check valid bitstrings
    n_qubits = len(list(data['counts'].keys())[0])
    for bitstring in data['counts']:
        if not all(b in '01' for b in bitstring):
            return False
        if len(bitstring) != n_qubits:
            return False

    return True
```

---

## 5. Quantum-Specific Reproducibility

### 5.1 Handling Quantum Stochasticity

**Statistical Reproducibility**
```python
def statistically_equivalent(
    counts1: dict,
    counts2: dict,
    alpha: float = 0.05
) -> bool:
    """
    Test if two measurement outcome distributions are
    statistically equivalent using chi-squared test.
    """
    from scipy.stats import chi2_contingency

    # Align keys
    all_keys = set(counts1.keys()) | set(counts2.keys())
    observed = np.array([
        [counts1.get(k, 0) for k in all_keys],
        [counts2.get(k, 0) for k in all_keys]
    ])

    chi2, p_value, dof, expected = chi2_contingency(observed)

    return p_value > alpha
```

**Seed Management**
```python
class ReproducibleQuantumExperiment:
    """Base class for reproducible quantum experiments."""

    def __init__(self, seed: int):
        self.seed = seed
        self._initialize_rngs()

    def _initialize_rngs(self):
        """Initialize all random number generators."""
        import numpy as np
        import random

        # NumPy RNG
        self.np_rng = np.random.default_rng(self.seed)

        # Python RNG
        random.seed(self.seed)

        # Quantum framework RNG (example for Qiskit)
        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = self.seed

    def get_deterministic_seed(self, label: str) -> int:
        """Generate deterministic sub-seed from label."""
        import hashlib
        combined = f"{self.seed}_{label}"
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        return int(hash_val[:8], 16)
```

### 5.2 Hardware Reproducibility

**Calibration Recording**
```python
def record_calibration_data(backend) -> dict:
    """Record complete calibration snapshot."""
    props = backend.properties()

    calibration = {
        'timestamp': props.last_update_date.isoformat(),
        'backend_name': backend.name(),
        'backend_version': backend.version,
        'qubits': {}
    }

    for qubit in range(backend.num_qubits):
        calibration['qubits'][qubit] = {
            'T1': props.t1(qubit),
            'T2': props.t2(qubit),
            'frequency': props.frequency(qubit),
            'readout_error': props.readout_error(qubit),
            'gate_errors': {
                gate.gate: props.gate_error(gate.gate, qubit)
                for gate in props.gates if len(gate.qubits) == 1
            }
        }

    return calibration
```

**Noise Model Specification**
```python
def create_reproducible_noise_model(calibration: dict):
    """Create noise model from calibration data."""
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.errors import thermal_relaxation_error

    noise_model = NoiseModel()

    for qubit, data in calibration['qubits'].items():
        # Thermal relaxation
        t1, t2 = data['T1'], data['T2']
        gate_time = 50e-9  # Typical gate time

        error = thermal_relaxation_error(t1, t2, gate_time)
        noise_model.add_quantum_error(error, ['u1', 'u2', 'u3'], [qubit])

    return noise_model
```

### 5.3 Cross-Platform Reproducibility

**Abstract Interface**
```python
from abc import ABC, abstractmethod

class QuantumBackend(ABC):
    """Abstract interface for quantum backends."""

    @abstractmethod
    def execute_circuit(self, circuit, shots: int) -> dict:
        """Execute circuit and return counts."""
        pass

    @abstractmethod
    def get_backend_info(self) -> dict:
        """Return backend specifications."""
        pass

class QiskitBackend(QuantumBackend):
    """Qiskit implementation."""
    pass

class PennyLaneBackend(QuantumBackend):
    """PennyLane implementation."""
    pass
```

---

## 6. Testing and Validation

### 6.1 Test Hierarchy

**Unit Tests**
```python
def test_pauli_string_multiplication():
    """Test Pauli string multiplication rules."""
    from src.utils import pauli_multiply

    # I * X = X
    assert pauli_multiply('I', 'X') == ('X', 1)

    # X * Y = iZ
    assert pauli_multiply('X', 'Y') == ('Z', 1j)

    # Y * X = -iZ
    assert pauli_multiply('Y', 'X') == ('Z', -1j)
```

**Integration Tests**
```python
def test_vqe_h2_ground_state():
    """Test VQE finds H2 ground state energy."""
    from src.algorithms import VQE

    # Known ground state energy
    expected_energy = -1.137  # Hartree

    vqe = VQE(molecule='H2', ansatz='UCCSD')
    result = vqe.run()

    assert abs(result.energy - expected_energy) < 0.01
```

**Benchmark Tests**
```python
def test_scaling_behavior():
    """Verify expected computational scaling."""
    import time

    times = []
    for n in [2, 4, 6, 8]:
        start = time.time()
        run_simulation(n_qubits=n)
        times.append(time.time() - start)

    # Check exponential scaling O(2^n)
    ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
    expected_ratio = 4  # 2^2
    for ratio in ratios:
        assert 2 < ratio < 8  # Allow some variance
```

### 6.2 Continuous Integration

**.github/workflows/tests.yml**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## 7. Publication and Sharing

### 7.1 Code Availability

**Repository Structure for Publication**
```
published_code/
├── README.md           # How to use
├── LICENSE             # Clear licensing
├── CITATION.cff        # How to cite
├── requirements.txt    # Dependencies
├── reproduce.py        # One-click reproduction
├── src/               # Clean source code
├── data/              # Minimal example data
└── figures/           # Figure generation
```

**CITATION.cff**
```yaml
cff-version: 1.2.0
message: "If you use this code, please cite:"
authors:
  - family-names: "Your"
    given-names: "Name"
    orcid: "https://orcid.org/0000-0000-0000-0000"
title: "Your Code Title"
version: 1.0.0
doi: 10.5281/zenodo.XXXXXXX
date-released: 2024-01-15
```

### 7.2 Data Sharing

**Data Repository Options**
- Zenodo (general purpose, DOI assignment)
- Figshare (institutional)
- Dryad (peer-reviewed data)
- Domain-specific repositories

**Data Documentation**
```
README_DATA.md

# Dataset: [Title]

## Description
[What does this data represent?]

## Collection
[How was data collected?]

## Format
[File formats and structure]

## Variables
| Name | Type | Units | Description |
|------|------|-------|-------------|
|      |      |       |             |

## Usage
[How to load and use the data]

## License
[Data usage license]

## Citation
[How to cite this dataset]
```

---

## 8. Checklists

### Pre-Publication Checklist

- [ ] Code runs on clean environment
- [ ] All figures reproducible from code
- [ ] Dependencies fully specified
- [ ] Tests pass
- [ ] Documentation complete
- [ ] License chosen
- [ ] DOI obtained
- [ ] Citation file created

### Pre-Experiment Checklist

- [ ] Protocols documented
- [ ] Calibration current
- [ ] Data storage prepared
- [ ] Environment recorded
- [ ] Seeds set
- [ ] Baseline collected

---

## Resources

### Guidelines and Standards
- FAIR Data Principles: https://www.go-fair.org/
- Reproducible Research in Computational Science (Science, 2011)
- Nature Methods: Reporting Standards

### Tools
- Git/GitHub: Version control
- Zenodo: Data archival
- Docker: Environment containerization
- Pytest: Testing framework

### Community Standards
- Qiskit Contribution Guidelines
- Open Quantum Safe
- Quantum Software Manifesto

---

*Reproducibility is not a burden—it is an investment in the integrity and impact of your research.*
