# Data Collection Best Practices for Quantum Research

## Overview

High-quality data collection is the foundation of reliable research conclusions. In quantum research, where measurements are inherently statistical and systems are sensitive to numerous environmental factors, rigorous data collection practices are essential. This guide provides comprehensive best practices for collecting, validating, and managing research data.

---

## 1. Principles of Quality Data Collection

### 1.1 The Data Quality Pyramid

```
                    ┌───────────────┐
                    │  ACTIONABLE   │  Data drives decisions
                    ├───────────────┤
                    │   ACCURATE    │  Reflects true values
                    ├───────────────┤
                    │   COMPLETE    │  No missing elements
                    ├───────────────┤
                    │  CONSISTENT   │  Internally coherent
                    ├───────────────┤
                    │    VALID      │  Correct format/type
                    └───────────────┘
```

### 1.2 Core Principles

**FAIR Data Principles**

| Principle | Application in Quantum Research |
|-----------|--------------------------------|
| **Findable** | Use consistent naming, comprehensive metadata, persistent identifiers |
| **Accessible** | Store in accessible formats, document access procedures |
| **Interoperable** | Use standard file formats, well-defined schemas |
| **Reusable** | Include complete provenance, clear licensing |

**Research Integrity Principles**

- **Honesty**: Record what you observe, not what you expect
- **Objectivity**: Apply consistent standards regardless of outcome
- **Transparency**: Document everything, including failures
- **Accountability**: Maintain clear authorship and responsibility

---

## 2. Data Collection Planning

### 2.1 Pre-Collection Preparation

**Define Data Requirements**

Before collecting any data, clearly specify:

1. **What** data is needed (variables, measurements)
2. **How much** data is needed (sample size, repetitions)
3. **How precise** the data must be (uncertainty requirements)
4. **What format** the data should take (file types, schemas)
5. **Where** the data will be stored (primary and backup locations)

**Sample Size Determination**

For quantum measurements, the required number of shots depends on:

$$N = \frac{z_{\alpha/2}^2 \cdot \sigma^2}{\epsilon^2}$$

where:
- $z_{\alpha/2}$ is the critical value (1.96 for 95% confidence)
- $\sigma^2$ is the variance of the measured quantity
- $\epsilon$ is the desired margin of error

**Example Calculation:**

For measuring a probability with 1% precision at 95% confidence:
- Maximum variance: $\sigma^2 = 0.25$ (for $p = 0.5$)
- Margin of error: $\epsilon = 0.01$
- Required shots: $N = \frac{1.96^2 \times 0.25}{0.01^2} \approx 9604$

### 2.2 Data Schema Design

**Define Your Schema Early**

```yaml
# experiment_data_schema.yaml
schema_version: "1.0"
experiment_type: "quantum_state_tomography"

measurements:
  type: object
  properties:
    basis:
      type: string
      enum: ["X", "Y", "Z", "XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
    shots:
      type: integer
      minimum: 1
    counts:
      type: object
      additionalProperties:
        type: integer
        minimum: 0
    timestamp:
      type: string
      format: date-time
  required: ["basis", "shots", "counts", "timestamp"]

metadata:
  type: object
  properties:
    backend_name:
      type: string
    backend_version:
      type: string
    calibration_id:
      type: string
    researcher:
      type: string
  required: ["backend_name", "researcher"]
```

---

## 3. Active Data Collection

### 3.1 Real-Time Quality Monitoring

**Automated Monitoring System**

```python
class RealTimeMonitor:
    """Monitor data quality during collection."""

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        self.history = []
        self.alerts = []

    def check_measurement(self, measurement: dict) -> dict:
        """Check single measurement against quality criteria."""
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }

        # Check shot count
        if measurement['shots'] < self.thresholds['min_shots']:
            quality_report['checks']['shots'] = 'FAIL: Insufficient shots'
        else:
            quality_report['checks']['shots'] = 'PASS'

        # Check count consistency
        total_counts = sum(measurement['counts'].values())
        if total_counts != measurement['shots']:
            quality_report['checks']['consistency'] = f'FAIL: Sum={total_counts}'
        else:
            quality_report['checks']['consistency'] = 'PASS'

        # Check for anomalous distribution
        max_prob = max(measurement['counts'].values()) / measurement['shots']
        if max_prob > self.thresholds['max_single_outcome']:
            quality_report['checks']['distribution'] = f'WARNING: max_prob={max_prob:.3f}'
        else:
            quality_report['checks']['distribution'] = 'PASS'

        # Add to history
        self.history.append(quality_report)

        # Generate alert if needed
        if any('FAIL' in str(v) for v in quality_report['checks'].values()):
            self.alerts.append(quality_report)

        return quality_report
```

**Key Metrics to Monitor**

| Metric | Formula | Healthy Range | Action if Violated |
|--------|---------|---------------|-------------------|
| Total counts | $\sum_i n_i$ | = shots | Investigate immediately |
| Entropy | $-\sum_i p_i \log_2 p_i$ | Context-dependent | Check preparation |
| Max probability | $\max_i p_i$ | < 0.99 (usually) | Verify state |
| Drift rate | $\|x_t - x_{t-1}\|/\Delta t$ | < threshold | Check stability |

### 3.2 Systematic Data Recording

**Structured Data Entry**

Always record:
1. **What**: Exact measurement parameters and outcomes
2. **When**: Precise timestamps (use UTC)
3. **Where**: Equipment, location, backend
4. **Who**: Researcher and any assistants
5. **How**: Protocol reference, any deviations
6. **Why**: Purpose of this specific measurement

**Metadata Completeness**

```python
def create_measurement_record(
    circuit_id: str,
    measurement_basis: str,
    counts: dict,
    shots: int,
    backend
) -> dict:
    """Create complete measurement record with metadata."""

    return {
        # Core data
        'measurement': {
            'circuit_id': circuit_id,
            'basis': measurement_basis,
            'counts': counts,
            'shots': shots
        },

        # Timing
        'timing': {
            'collected_at': datetime.utcnow().isoformat() + 'Z',
            'timezone': 'UTC',
            'execution_time_s': None  # Fill from job
        },

        # Hardware
        'hardware': {
            'backend_name': backend.name(),
            'backend_version': getattr(backend, 'version', 'unknown'),
            'qubits_used': circuit.qubits,
            'backend_properties': extract_properties(backend)
        },

        # Software
        'software': {
            'framework': 'qiskit',
            'framework_version': qiskit.__version__,
            'python_version': sys.version,
            'script_version': __version__
        },

        # Provenance
        'provenance': {
            'protocol_id': PROTOCOL_ID,
            'experiment_id': EXPERIMENT_ID,
            'researcher': RESEARCHER_NAME
        }
    }
```

### 3.3 Handling Special Cases

**Missing Data**

When data cannot be collected:
1. Document the reason clearly
2. Distinguish between missing-at-random and systematic missingness
3. Never fabricate or impute raw data
4. Record the intended measurement for completeness

**Outliers**

When encountering outliers:
1. Document the outlier before any action
2. Investigate the cause
3. Apply pre-defined criteria for inclusion/exclusion
4. Never remove data without documented justification
5. Report both with-outlier and without-outlier results

**Equipment Failures**

During equipment malfunctions:
1. Stop data collection immediately
2. Document the failure mode
3. Mark potentially affected data
4. Complete troubleshooting before resuming
5. Verify system stability before continuing

---

## 4. Data Validation

### 4.1 Immediate Validation

**Level 1: Format Validation**

Check immediately after collection:
- File created successfully
- File readable and uncorrupted
- Data structure matches schema
- Required fields present

```python
def validate_format(filepath: str, schema: dict) -> Tuple[bool, List[str]]:
    """Validate data file format against schema."""
    errors = []

    try:
        data = load_data(filepath)
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    # Check against schema
    from jsonschema import validate, ValidationError

    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        errors.append(f"Schema violation: {e.message}")

    return len(errors) == 0, errors
```

**Level 2: Value Validation**

Check data values:
- Values within physically meaningful ranges
- Probabilities sum to 1
- Counts are non-negative integers
- Timestamps are valid and sequential

### 4.2 Statistical Validation

**Consistency Checks**

```python
def statistical_validation(measurements: List[dict]) -> dict:
    """Perform statistical validation on measurement series."""

    results = {}

    # Extract outcome probabilities
    all_probs = []
    for m in measurements:
        total = sum(m['counts'].values())
        probs = {k: v/total for k, v in m['counts'].items()}
        all_probs.append(probs)

    # Check for drift
    if len(all_probs) > 5:
        first_half = all_probs[:len(all_probs)//2]
        second_half = all_probs[len(all_probs)//2:]

        for outcome in all_probs[0].keys():
            first_probs = [p.get(outcome, 0) for p in first_half]
            second_probs = [p.get(outcome, 0) for p in second_half]

            # t-test for drift
            t_stat, p_value = stats.ttest_ind(first_probs, second_probs)
            if p_value < 0.01:
                results[f'drift_{outcome}'] = {
                    'detected': True,
                    'p_value': p_value,
                    'first_mean': np.mean(first_probs),
                    'second_mean': np.mean(second_probs)
                }

    # Check for anomalous variance
    for outcome in all_probs[0].keys():
        outcome_probs = [p.get(outcome, 0) for p in all_probs]
        expected_var = np.mean(outcome_probs) * (1 - np.mean(outcome_probs)) / measurements[0]['shots']
        observed_var = np.var(outcome_probs)

        ratio = observed_var / expected_var if expected_var > 0 else np.inf
        if ratio > 2 or ratio < 0.5:
            results[f'variance_{outcome}'] = {
                'anomalous': True,
                'expected': expected_var,
                'observed': observed_var,
                'ratio': ratio
            }

    return results
```

### 4.3 Cross-Validation

**Internal Consistency**

Check that related measurements are consistent:
- Complementary bases should give complementary results
- Conservation laws should be satisfied
- Derived quantities should match direct measurements

**External Validation**

Compare with:
- Known theoretical predictions
- Previously published results
- Independent measurements

---

## 5. Data Storage and Management

### 5.1 File Organization

**Directory Structure**

```
project/
├── data/
│   ├── raw/                      # Immutable raw data
│   │   ├── 2024/
│   │   │   ├── 01/
│   │   │   │   ├── 15/
│   │   │   │   │   ├── QST_001.h5
│   │   │   │   │   ├── QST_001_metadata.json
│   │   │   │   │   └── QST_001_log.txt
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── processed/                # Processed/cleaned data
│   │   └── [similar structure]
│   ├── results/                  # Analysis results
│   │   └── [similar structure]
│   └── archive/                  # Long-term archive
│       └── [similar structure]
├── metadata/
│   ├── schemas/                  # Data schemas
│   ├── calibrations/             # Calibration records
│   └── inventory.json            # Data inventory
└── checksums/
    └── [matching structure with .sha256 files]
```

### 5.2 Naming Conventions

**File Naming Pattern**

```
[PROJECT]_[TYPE]_[DATE]_[PARAMS]_[VERSION].[EXT]

Examples:
QST_BELL_20240115_T100mK_Q01Q02_v1.h5
VQE_H2_20240115_UCCSD_D6_COBYLA_v2.json
RB_1Q_20240115_Q03_v1.csv
```

**Version Control for Data**

- Use version numbers for processed data
- Raw data should never be versioned (immutable)
- Document all transformations between versions

### 5.3 Data Integrity

**Checksum Management**

```python
import hashlib
from pathlib import Path

def compute_checksum(filepath: str, algorithm: str = 'sha256') -> str:
    """Compute cryptographic checksum of file."""
    hash_func = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()

def verify_checksum(filepath: str, expected: str, algorithm: str = 'sha256') -> bool:
    """Verify file against stored checksum."""
    actual = compute_checksum(filepath, algorithm)
    return actual == expected

def create_checksum_manifest(directory: str) -> dict:
    """Create checksum manifest for directory."""
    manifest = {
        'created': datetime.utcnow().isoformat() + 'Z',
        'algorithm': 'sha256',
        'files': {}
    }

    for path in Path(directory).rglob('*'):
        if path.is_file() and not path.suffix == '.sha256':
            relative_path = str(path.relative_to(directory))
            manifest['files'][relative_path] = compute_checksum(str(path))

    return manifest
```

### 5.4 Backup Strategy

**3-2-1 Rule**

- **3** copies of data
- **2** different storage media
- **1** offsite backup

**Backup Schedule**

| Data Type | Backup Frequency | Retention |
|-----------|------------------|-----------|
| Raw data | Immediately after collection | Permanent |
| Working files | Daily | 30 days |
| Processed data | After each processing run | 1 year |
| Results | After each analysis | Permanent |

---

## 6. Quantum-Specific Considerations

### 6.1 Handling Shot Noise

**Shot Noise Uncertainty**

For quantum measurements:
$$\sigma_p = \sqrt{\frac{p(1-p)}{N}}$$

**Reporting Requirements**

Always report:
- Number of shots
- Raw counts (not just probabilities)
- Statistical uncertainty
- Any corrections applied

### 6.2 Calibration Data

**Essential Calibration Records**

| Calibration Type | Frequency | Data to Record |
|-----------------|-----------|----------------|
| Qubit frequency | Daily | Frequency, uncertainty, method |
| Gate parameters | Daily | Pulse amplitude, duration, phase |
| Readout | Daily | Confusion matrix |
| Coherence times | Weekly | T1, T2, T2*, measurement conditions |

### 6.3 Environmental Data

**Correlated Measurements**

Record alongside quantum measurements:
- Cryostat temperature
- Magnetic field readings
- Electrical noise levels
- Timestamp (for drift correlation)

---

## 7. Common Pitfalls and Solutions

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| Inconsistent naming | Cannot find/link data | Establish conventions early |
| Missing metadata | Cannot reproduce | Automate metadata capture |
| No checksums | Undetected corruption | Compute immediately after save |
| Single storage | Data loss | Implement backup immediately |
| Manual transcription | Transcription errors | Automate data flow |
| Ignoring outliers | Biased results | Document and justify all exclusions |
| Undocumented processing | Irreproducible | Log all transformations |

---

## 8. Checklists

### Pre-Collection Checklist

- [ ] Data requirements defined
- [ ] Sample size calculated
- [ ] Schema designed
- [ ] Storage location prepared
- [ ] Backup system verified
- [ ] Naming conventions established
- [ ] Monitoring systems active

### During-Collection Checklist

- [ ] Real-time monitoring active
- [ ] All measurements documented
- [ ] Metadata captured automatically
- [ ] Anomalies flagged
- [ ] Periodic validation performed

### Post-Collection Checklist

- [ ] Data files verified
- [ ] Checksums computed
- [ ] Backup completed
- [ ] Validation passed
- [ ] Documentation complete
- [ ] Data inventory updated

---

## Resources

### Standards and Guidelines
- FAIR Guiding Principles for scientific data management
- Research Data Alliance data management standards
- NIST data quality guidelines

### Tools
- HDF5: Hierarchical data format
- JSON Schema: Data validation
- Git LFS: Large file versioning
- DVC: Data version control

### Community Resources
- Quantum software best practices
- Open quantum data repositories
- Research data management courses

---

*Quality data collection is not overhead—it is the foundation of scientific credibility.*
