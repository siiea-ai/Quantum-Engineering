# Initial Investigation Guide: Experimental Design and Data Collection

## Introduction

This guide provides comprehensive methodology for conducting initial investigations in quantum engineering research. Whether running simulations, performing experiments, or validating theoretical predictions, these principles ensure your first data collection phase yields high-quality, actionable results.

---

## Part 1: Philosophy of Initial Investigation

### 1.1 Purpose of Initial Investigations

Initial investigations serve multiple purposes:

1. **Validation:** Confirm methodology works as designed
2. **Discovery:** Identify unexpected behavior or phenomena
3. **Calibration:** Refine understanding of system parameters
4. **Foundation:** Generate data for subsequent analysis
5. **Learning:** Develop operational expertise and intuition

### 1.2 The Investigation Mindset

**Be Systematic:** Follow protocols consistently while remaining alert to deviations.

**Be Observant:** Notice and document everything, especially surprises.

**Be Skeptical:** Question your results, especially if they seem "too good."

**Be Flexible:** Adapt to unexpected findings without abandoning rigor.

**Be Patient:** Quality data takes time; rushing leads to errors.

### 1.3 Balancing Exploration and Rigor

```
             Rigorous Protocol
                    ↑
                    │
         [Sweet Spot for Initial]
         [      Investigation   ]
                    │
Open ←──────────────┼──────────────→ Constrained
Exploration         │                Execution
                    │
                    ↓
            Unstructured Play
```

Initial investigations should be in the upper-middle region: structured enough to yield reliable data, flexible enough to follow interesting findings.

---

## Part 2: Experimental Design Fundamentals

### 2.1 Question-Driven Investigation

Every experiment should answer specific questions:

**Primary Question:** [What is the main thing you're trying to learn?]
**Secondary Questions:** [What else can you learn from these measurements?]
**Validation Questions:** [How will you verify results are correct?]

**Example for Qubit Characterization:**
- Primary: What is the T1 relaxation time?
- Secondary: How does T1 depend on measurement power?
- Validation: Does repeated measurement give consistent results?

### 2.2 Hypothesis Formulation

Even exploratory investigations benefit from hypotheses:

**Strong Hypothesis:**
"Increasing the gate amplitude beyond the optimal value will decrease fidelity due to leakage to non-computational states."

**Weak Hypothesis:**
"Gate fidelity depends on amplitude."

The strong hypothesis is falsifiable and guides measurement design.

### 2.3 Variable Classification

| Variable Type | Definition | Treatment |
|---------------|------------|-----------|
| Independent | What you control/vary | Systematic variation |
| Dependent | What you measure | Careful measurement |
| Controlled | Kept constant | Monitor and document |
| Nuisance | Unavoidable variation | Randomize or block |
| Confounding | Correlated with independent | Identify and separate |

### 2.4 Control Experiment Design

**Positive Control:** Known to produce effect
- Verifies measurement system works
- Example: Measure known coherent state to verify tomography

**Negative Control:** Known to produce no effect
- Verifies you're measuring signal, not artifact
- Example: Measure with drive off to verify background

**Calibration Control:** Known quantitative result
- Verifies quantitative accuracy
- Example: Measure standard sample with known value

---

## Part 3: Data Collection Strategies

### 3.1 Sampling Strategies

#### Uniform Grid Sampling
**Use when:** Parameter space is unknown, need complete coverage

```python
import numpy as np

def uniform_grid(param_ranges, n_points_per_dim):
    """
    Create uniform grid over parameter space.

    Parameters:
    -----------
    param_ranges : dict
        {'param_name': (min, max), ...}
    n_points_per_dim : int or dict
        Number of points per dimension

    Returns:
    --------
    points : list of dicts
        List of parameter dictionaries
    """
    if isinstance(n_points_per_dim, int):
        n_points_per_dim = {k: n_points_per_dim for k in param_ranges}

    grids = []
    names = list(param_ranges.keys())

    for name in names:
        low, high = param_ranges[name]
        n = n_points_per_dim[name]
        grids.append(np.linspace(low, high, n))

    mesh = np.meshgrid(*grids, indexing='ij')
    flat = [m.flatten() for m in mesh]

    points = []
    for i in range(len(flat[0])):
        point = {name: flat[j][i] for j, name in enumerate(names)}
        points.append(point)

    return points
```

#### Adaptive Sampling
**Use when:** Need to focus on interesting regions

```python
def adaptive_refinement(data, threshold, param_ranges):
    """
    Identify regions needing refinement based on gradient.

    Parameters:
    -----------
    data : dict
        {(param1, param2): result, ...}
    threshold : float
        Gradient threshold for refinement
    param_ranges : dict
        Valid parameter ranges

    Returns:
    --------
    new_points : list
        Points to measure next
    """
    new_points = []

    # Convert to array for gradient calculation
    params = np.array(list(data.keys()))
    values = np.array(list(data.values()))

    # Find points with high local gradient
    for i, (p, v) in enumerate(zip(params, values)):
        # Find neighbors
        neighbors = find_neighbors(p, params)
        if len(neighbors) > 0:
            gradients = np.abs(values[neighbors] - v) / distance(params[neighbors], p)
            if np.max(gradients) > threshold:
                # Add midpoints for refinement
                for n in neighbors[gradients > threshold]:
                    midpoint = (p + params[n]) / 2
                    if in_bounds(midpoint, param_ranges):
                        new_points.append(midpoint)

    return new_points
```

#### Importance Sampling
**Use when:** Some regions matter more than others

```python
def importance_sample(param_ranges, n_samples, importance_func):
    """
    Sample more densely where importance is higher.

    Parameters:
    -----------
    param_ranges : dict
        {'param_name': (min, max), ...}
    n_samples : int
        Total number of samples
    importance_func : callable
        Function returning importance weight for parameters

    Returns:
    --------
    samples : list of dicts
        Parameter dictionaries
    """
    from scipy.stats import qmc

    # Generate many candidate points
    n_candidates = n_samples * 10
    sampler = qmc.Halton(d=len(param_ranges))
    candidates = sampler.random(n=n_candidates)

    # Scale to parameter ranges
    scaled = scale_to_ranges(candidates, param_ranges)

    # Compute importance weights
    weights = np.array([importance_func(p) for p in scaled])
    weights /= weights.sum()

    # Select samples proportional to weights
    indices = np.random.choice(n_candidates, size=n_samples,
                               replace=False, p=weights)

    return [scaled[i] for i in indices]
```

### 3.2 Replication Strategies

**Why Replicate:**
1. Estimate measurement uncertainty
2. Detect systematic drift
3. Increase confidence in results
4. Enable statistical analysis

**How Many Replications:**
| Purpose | Minimum | Recommended |
|---------|---------|-------------|
| Uncertainty estimate | 3 | 5-10 |
| Statistical significance | 10 | 30+ |
| Trend detection | 5 | 10+ per point |
| Publication quality | 20 | 50+ |

**When to Replicate:**
- All key measurements
- Points showing unexpected results
- Boundary conditions
- Reference calibrations

### 3.3 Randomization

**Why Randomize:**
- Break correlation between parameters and time
- Average out systematic drifts
- Enable valid statistical tests
- Reduce experimenter bias

**Randomization Schemes:**

```python
import numpy as np

def randomize_experiment_order(parameter_points, seed=42):
    """
    Randomize order of experiments to break temporal correlations.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(parameter_points))
    rng.shuffle(indices)
    return [parameter_points[i] for i in indices]

def blocked_randomization(parameter_points, block_size=10, seed=42):
    """
    Randomize within blocks to balance local and global randomization.
    Ensures all parameter values appear in each block.
    """
    rng = np.random.default_rng(seed)
    n_blocks = len(parameter_points) // block_size + 1

    randomized = []
    for block in range(n_blocks):
        block_points = parameter_points[block*block_size:(block+1)*block_size]
        indices = np.arange(len(block_points))
        rng.shuffle(indices)
        randomized.extend([block_points[i] for i in indices])

    return randomized
```

---

## Part 4: Measurement Best Practices

### 4.1 Pre-Measurement Checklist

```markdown
## Pre-Measurement Protocol

### Equipment Status
- [ ] All devices powered and warmed up (___min warmup)
- [ ] Calibrations current (last calibration: _______)
- [ ] Connections verified
- [ ] No error indicators

### System State
- [ ] Temperature stable: ______ ± ______
- [ ] Pressure/vacuum: ______
- [ ] Magnetic field: ______
- [ ] Other environmental: ______

### Software Status
- [ ] Data acquisition running
- [ ] Analysis pipeline tested
- [ ] Storage space available: ______ GB
- [ ] Backup system connected

### Documentation Ready
- [ ] Experiment log open
- [ ] Metadata template ready
- [ ] Previous data backed up
```

### 4.2 During Measurement Protocol

**Continuous Monitoring:**
- Watch for anomalies in real-time data
- Monitor equipment indicators
- Note environmental changes
- Record unexpected events

**Periodic Checks:**
- Data file integrity (every hour)
- Calibration stability (every 2-4 hours)
- Reference measurements (daily)
- Backup verification (end of session)

**Documentation Discipline:**
- Timestamp all entries
- Record deviations from protocol
- Note observations immediately
- Never trust memory for details

### 4.3 Post-Measurement Checklist

```markdown
## Post-Measurement Protocol

### Data Verification
- [ ] All data files present
- [ ] File sizes as expected
- [ ] Quick-look visualization completed
- [ ] No obvious artifacts

### Metadata Completion
- [ ] All parameters recorded
- [ ] Equipment states documented
- [ ] Deviations noted
- [ ] Observer notes added

### Backup and Archive
- [ ] Data copied to secondary storage
- [ ] Backup verified readable
- [ ] Archive location recorded
- [ ] Cloud sync completed (if applicable)

### System Shutdown (if applicable)
- [ ] Proper shutdown sequence followed
- [ ] Equipment state logged
- [ ] Issues noted for next session
```

### 4.4 Dealing with Measurement Failures

**Immediate Response:**
1. Document the failure completely
2. Do not modify failed data
3. Note what was happening when failure occurred
4. Check equipment status

**Recovery Protocol:**
1. Identify failure mode
2. Verify equipment function
3. Run test measurement
4. Resume from last known good point
5. Document the gap in data

**Partial Data:**
- Label clearly as incomplete
- Record what portion is valid
- Note any concerns about reliability
- Include in dataset with appropriate flags

---

## Part 5: Quality Assurance

### 5.1 Real-Time Quality Metrics

**For Quantum State Measurements:**

```python
def check_state_quality(measured_state):
    """
    Check quality metrics for measured quantum state.
    """
    quality = {}

    # Normalization
    trace = np.trace(measured_state)
    quality['trace'] = float(np.abs(trace))
    quality['trace_ok'] = np.isclose(trace, 1.0, atol=0.01)

    # Positivity (for density matrix)
    eigenvalues = np.linalg.eigvalsh(measured_state)
    quality['min_eigenvalue'] = float(np.min(eigenvalues))
    quality['positive'] = np.all(eigenvalues >= -1e-10)

    # Purity
    purity = np.trace(measured_state @ measured_state)
    quality['purity'] = float(np.real(purity))

    # Hermiticity
    hermitian_dev = np.max(np.abs(measured_state - measured_state.conj().T))
    quality['hermitian_deviation'] = float(hermitian_dev)
    quality['hermitian'] = hermitian_dev < 1e-10

    return quality
```

**For Time Series Data:**

```python
def check_timeseries_quality(data, expected_range=None):
    """
    Check quality of time series measurement.
    """
    quality = {}

    # Basic statistics
    quality['mean'] = float(np.mean(data))
    quality['std'] = float(np.std(data))
    quality['n_points'] = len(data)

    # Check for NaN/Inf
    quality['n_nan'] = int(np.sum(np.isnan(data)))
    quality['n_inf'] = int(np.sum(np.isinf(data)))
    quality['clean'] = quality['n_nan'] == 0 and quality['n_inf'] == 0

    # Check range
    if expected_range is not None:
        min_val, max_val = expected_range
        in_range = np.logical_and(data >= min_val, data <= max_val)
        quality['pct_in_range'] = float(np.mean(in_range) * 100)

    # Check for obvious artifacts
    diff = np.diff(data)
    quality['max_jump'] = float(np.max(np.abs(diff)))
    quality['suspicious_jumps'] = int(np.sum(np.abs(diff) > 5 * np.std(diff)))

    return quality
```

### 5.2 Statistical Process Control

**Control Chart for Measurement Stability:**

```python
import numpy as np
import matplotlib.pyplot as plt

class MeasurementControlChart:
    """
    Control chart for monitoring measurement stability.
    """

    def __init__(self, reference_data):
        """
        Initialize with reference measurements.

        Parameters:
        -----------
        reference_data : array-like
            Baseline measurements for establishing control limits
        """
        self.mean = np.mean(reference_data)
        self.std = np.std(reference_data)
        self.ucl = self.mean + 3 * self.std  # Upper control limit
        self.lcl = self.mean - 3 * self.std  # Lower control limit
        self.measurements = list(reference_data)

    def add_measurement(self, value):
        """Add new measurement and check control status."""
        self.measurements.append(value)
        in_control = self.lcl <= value <= self.ucl
        return in_control

    def check_runs(self, n_consecutive=7):
        """Check for runs (consecutive points on one side of mean)."""
        if len(self.measurements) < n_consecutive:
            return True

        recent = self.measurements[-n_consecutive:]
        all_above = all(x > self.mean for x in recent)
        all_below = all(x < self.mean for x in recent)

        return not (all_above or all_below)

    def plot(self, save_path=None):
        """Generate control chart plot."""
        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(self.measurements))
        ax.plot(x, self.measurements, 'bo-', markersize=4)
        ax.axhline(self.mean, color='g', linestyle='-', label='Mean')
        ax.axhline(self.ucl, color='r', linestyle='--', label='UCL')
        ax.axhline(self.lcl, color='r', linestyle='--', label='LCL')

        ax.set_xlabel('Measurement Number')
        ax.set_ylabel('Value')
        ax.set_title('Measurement Control Chart')
        ax.legend()

        if save_path:
            fig.savefig(save_path)
        return fig
```

### 5.3 Uncertainty Estimation

**Type A Uncertainty (Statistical):**

```python
def type_a_uncertainty(measurements, confidence=0.95):
    """
    Calculate Type A (statistical) uncertainty.

    Parameters:
    -----------
    measurements : array-like
        Repeated measurements
    confidence : float
        Confidence level for interval

    Returns:
    --------
    dict with mean, std, sem, and confidence interval
    """
    from scipy import stats

    n = len(measurements)
    mean = np.mean(measurements)
    std = np.std(measurements, ddof=1)  # Sample standard deviation
    sem = std / np.sqrt(n)  # Standard error of mean

    # t-value for confidence interval
    t_val = stats.t.ppf((1 + confidence) / 2, df=n-1)
    ci = (mean - t_val * sem, mean + t_val * sem)

    return {
        'mean': mean,
        'std': std,
        'sem': sem,
        'n': n,
        'confidence': confidence,
        'ci_low': ci[0],
        'ci_high': ci[1]
    }
```

**Combined Uncertainty:**

$$u_c = \sqrt{\sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 u^2(x_i)}$$

---

## Part 6: Data Management

### 6.1 Data Organization Structure

```
investigation_data/
├── raw/                          # Unmodified original data
│   ├── 2026-02-15/
│   │   ├── experiment_001/
│   │   │   ├── data.npz
│   │   │   ├── metadata.yaml
│   │   │   └── log.txt
│   │   └── experiment_002/
│   └── 2026-02-16/
├── processed/                    # Derived data products
│   ├── calibrated/
│   ├── filtered/
│   └── aggregated/
├── analysis/                     # Analysis outputs
│   ├── figures/
│   ├── tables/
│   └── reports/
├── metadata/                     # Cross-cutting metadata
│   ├── equipment_log.csv
│   ├── calibration_history.csv
│   └── experiment_index.csv
└── docs/                         # Documentation
    ├── protocols/
    └── notes/
```

### 6.2 Metadata Schema

```yaml
# Standard metadata schema for investigation data
$schema: "investigation_metadata_v1"

experiment:
  id: "EXP-2026-02-15-001"
  type: "parameter_sweep"
  date: "2026-02-15"
  time_start: "10:30:00"
  time_end: "14:45:00"
  researcher: "Your Name"

parameters:
  primary:
    coupling_strength:
      value: 0.5
      unit: "MHz"
      uncertainty: 0.01
  secondary:
    temperature:
      value: 15
      unit: "mK"
      uncertainty: 0.5
  fixed:
    n_qubits: 4
    measurement_basis: "Z"

equipment:
  main_system:
    model: "System Model X"
    serial: "SN-12345"
    calibration_date: "2026-02-01"
  auxiliary:
    - device: "Signal Generator"
      model: "Model Y"
      settings:
        frequency: "5.0 GHz"
        power: "-20 dBm"

data:
  files:
    - filename: "raw_data.npz"
      type: "numpy_compressed"
      size_bytes: 1024000
      checksum: "sha256:abc123..."
  format:
    time_axis: "first"
    units: "dimensionless"
  quality:
    n_points: 10000
    missing_fraction: 0.0
    artifacts_noted: false

notes: |
  Initial sweep of coupling strength parameter.
  Observed slight drift in calibration around 13:00.
  Recalibrated before continuing.

references:
  protocol: "protocols/parameter_sweep_v2.md"
  previous: "EXP-2026-02-14-003"
```

### 6.3 Backup Protocol

**3-2-1 Backup Rule:**
- 3 copies of data
- 2 different storage media
- 1 offsite location

**Implementation:**

```bash
#!/bin/bash
# Daily backup script

DATE=$(date +%Y-%m-%d)
SOURCE="/path/to/investigation_data"
LOCAL_BACKUP="/path/to/local_backup"
REMOTE_BACKUP="user@server:/path/to/remote_backup"

# Local backup
rsync -av --delete "$SOURCE" "$LOCAL_BACKUP/$DATE/"

# Remote backup
rsync -av --delete "$SOURCE" "$REMOTE_BACKUP/$DATE/"

# Verify
echo "Verifying backups..."
local_count=$(find "$LOCAL_BACKUP/$DATE" -type f | wc -l)
remote_count=$(ssh user@server "find /path/to/remote_backup/$DATE -type f | wc -l")
source_count=$(find "$SOURCE" -type f | wc -l)

if [ "$local_count" -eq "$source_count" ] && [ "$remote_count" -eq "$source_count" ]; then
    echo "Backup verified: $source_count files"
else
    echo "WARNING: Backup verification failed!"
fi
```

---

## Part 7: Common Investigation Patterns

### 7.1 Parameter Sweep Pattern

```python
class ParameterSweep:
    """
    Standard parameter sweep investigation pattern.
    """

    def __init__(self, experiment_func, param_name, param_values, **fixed_params):
        self.experiment = experiment_func
        self.param_name = param_name
        self.param_values = param_values
        self.fixed_params = fixed_params
        self.results = []

    def run(self, n_reps=1, randomize=True, progress=True):
        """Execute parameter sweep with replications."""
        points = []
        for val in self.param_values:
            for rep in range(n_reps):
                points.append((val, rep))

        if randomize:
            np.random.shuffle(points)

        for i, (val, rep) in enumerate(points):
            if progress:
                print(f"Progress: {i+1}/{len(points)}", end='\r')

            params = self.fixed_params.copy()
            params[self.param_name] = val

            result = self.experiment(**params)

            self.results.append({
                'param_value': val,
                'rep': rep,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })

        return self.aggregate_results()

    def aggregate_results(self):
        """Aggregate results by parameter value."""
        aggregated = {}
        for r in self.results:
            val = r['param_value']
            if val not in aggregated:
                aggregated[val] = []
            aggregated[val].append(r['result'])

        summary = {}
        for val, results in aggregated.items():
            summary[val] = {
                'mean': np.mean(results),
                'std': np.std(results),
                'n': len(results),
                'raw': results
            }
        return summary
```

### 7.2 Convergence Study Pattern

```python
def convergence_study(experiment_func, resolution_param, resolutions,
                       reference_func=None, **fixed_params):
    """
    Study convergence with resolution parameter.

    Parameters:
    -----------
    experiment_func : callable
        Function to run experiment
    resolution_param : str
        Name of resolution parameter
    resolutions : list
        Resolution values to test (should be increasing)
    reference_func : callable, optional
        Function returning reference value
    fixed_params : dict
        Other parameters held fixed

    Returns:
    --------
    dict with convergence analysis
    """
    results = []

    for res in resolutions:
        params = fixed_params.copy()
        params[resolution_param] = res
        result = experiment_func(**params)
        results.append({
            'resolution': res,
            'result': result
        })

    # Analyze convergence
    resolutions = np.array([r['resolution'] for r in results])
    values = np.array([r['result'] for r in results])

    # Richardson extrapolation estimate
    if len(values) >= 3:
        # Assuming power-law convergence
        extrapolated = richardson_extrapolate(resolutions[-3:], values[-3:])
    else:
        extrapolated = values[-1]

    # Compute errors relative to extrapolated/reference
    if reference_func is not None:
        reference = reference_func(**fixed_params)
    else:
        reference = extrapolated

    errors = np.abs(values - reference) / np.abs(reference)

    return {
        'resolutions': resolutions.tolist(),
        'values': values.tolist(),
        'errors': errors.tolist(),
        'extrapolated': extrapolated,
        'reference': reference,
        'converged': errors[-1] < 0.01  # 1% threshold
    }


def richardson_extrapolate(h, f, order=2):
    """Richardson extrapolation assuming h^order convergence."""
    r = h[1] / h[2]  # Refinement ratio
    return (r**order * f[2] - f[1]) / (r**order - 1)
```

### 7.3 Stability Test Pattern

```python
def stability_test(experiment_func, n_measurements, interval_seconds,
                   **params):
    """
    Test measurement stability over time.

    Parameters:
    -----------
    experiment_func : callable
        Function returning measurement
    n_measurements : int
        Number of measurements
    interval_seconds : float
        Time between measurements
    params : dict
        Experiment parameters

    Returns:
    --------
    dict with stability analysis
    """
    import time

    measurements = []
    timestamps = []

    for i in range(n_measurements):
        result = experiment_func(**params)
        measurements.append(result)
        timestamps.append(time.time())

        if i < n_measurements - 1:
            time.sleep(interval_seconds)

    measurements = np.array(measurements)
    timestamps = np.array(timestamps) - timestamps[0]

    # Analyze stability
    mean = np.mean(measurements)
    std = np.std(measurements)

    # Linear drift
    slope, intercept = np.polyfit(timestamps, measurements, 1)
    drift_per_hour = slope * 3600

    # Allan deviation for different averaging times
    allan_devs = compute_allan_deviation(measurements)

    return {
        'timestamps': timestamps.tolist(),
        'measurements': measurements.tolist(),
        'mean': float(mean),
        'std': float(std),
        'std_percent': float(std / np.abs(mean) * 100),
        'drift_per_hour': float(drift_per_hour),
        'drift_percent_per_hour': float(drift_per_hour / np.abs(mean) * 100),
        'allan_deviation': allan_devs,
        'stable': std / np.abs(mean) < 0.01  # 1% stability criterion
    }


def compute_allan_deviation(data):
    """Compute Allan deviation for power-of-2 averaging times."""
    n = len(data)
    max_m = int(np.log2(n // 2))

    allan_devs = {}
    for m in [2**i for i in range(max_m + 1)]:
        n_groups = n // m
        if n_groups < 2:
            break

        # Average in groups
        grouped = data[:n_groups * m].reshape(n_groups, m).mean(axis=1)

        # Allan variance
        diffs = np.diff(grouped)
        allan_var = np.mean(diffs**2) / 2
        allan_devs[m] = float(np.sqrt(allan_var))

    return allan_devs
```

---

## Part 8: Troubleshooting Common Issues

### 8.1 Data Collection Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Inconsistent results | Drift, poor randomization | Add calibration checks, randomize order |
| Missing data points | Acquisition failure | Check logs, implement retry logic |
| Corrupted files | Storage issues | Verify checksums, improve backup |
| Wrong parameters saved | Metadata bug | Review logging code, add verification |

### 8.2 Equipment Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Drift over time | Temperature, aging | More frequent calibration |
| Intermittent failures | Connection, power | Check connections, monitor power |
| Noise increase | EMI, ground loops | Check shielding, verify grounding |
| Unexpected offsets | Calibration error | Recalibrate, check references |

### 8.3 Software Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Crashes during run | Memory, bugs | Profile memory, add error handling |
| Slow execution | Inefficient code | Profile, optimize bottlenecks |
| Incorrect output | Logic error | Add assertions, verify against known cases |
| Reproducibility issues | Random state | Set and record all seeds |

---

## Part 9: Preparing for Analysis

### 9.1 End-of-Week Data Review

Before moving to analysis week:

1. **Completeness Check**
   - All planned measurements completed?
   - Sufficient replications?
   - Coverage of parameter space adequate?

2. **Quality Check**
   - All quality metrics acceptable?
   - Anomalies documented and understood?
   - Control experiments successful?

3. **Documentation Check**
   - All metadata complete?
   - Experiment log current?
   - File organization clear?

4. **Preliminary Assessment**
   - Any obvious trends?
   - Unexpected features?
   - Questions for analysis?

### 9.2 Analysis Preparation

```markdown
## Week 218 to Week 219 Handoff

### Data Summary
- Total experiments: ___
- Total data points: ___
- Parameter ranges covered: [list]
- Data quality: [assessment]

### Key Observations
1. [Observation 1]
2. [Observation 2]
3. [Observation 3]

### Analysis Priorities
1. [Priority 1]: [Why important]
2. [Priority 2]: [Why important]
3. [Priority 3]: [Why important]

### Questions to Answer
1. [Question]
2. [Question]

### Concerns/Caveats
1. [Concern about data quality or coverage]
2. [Limitation to note]

### Files and Locations
- Raw data: [path]
- Processed data: [path]
- Metadata: [path]
- Experiment log: [path]
```

---

## Conclusion

Initial investigation is the foundation of your research. The data collected this week will drive all subsequent analysis and conclusions. Key principles:

1. **Systematic execution** ensures complete coverage
2. **Real-time documentation** prevents information loss
3. **Quality monitoring** catches problems early
4. **Proper organization** enables efficient analysis
5. **Preliminary observation** guides analysis strategy

Remember: You cannot analyze data you don't have, and you cannot trust analysis of poor-quality data. Invest time now in careful data collection.

---

*Initial Investigation Guide - Week 218*
*Month 55: Research Execution I*
