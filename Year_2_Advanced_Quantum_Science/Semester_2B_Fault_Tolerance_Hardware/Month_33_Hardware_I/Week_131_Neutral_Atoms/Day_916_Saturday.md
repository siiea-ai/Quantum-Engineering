# Day 916: Atom Sorting and Array Preparation

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Defect detection, rearrangement algorithms, loading optimization |
| **Afternoon** | 2 hours | Problem solving: array preparation efficiency |
| **Evening** | 2 hours | Computational lab: sorting simulations |

## Learning Objectives

By the end of this day, you will be able to:

1. **Analyze defect detection** via fluorescence imaging techniques
2. **Implement atom rearrangement algorithms** for defect-free array preparation
3. **Calculate loading efficiency** and optimization strategies
4. **Design real-time sorting sequences** using AOD control
5. **Evaluate tradeoffs** between array size, preparation time, and success probability
6. **Simulate array preparation** including realistic constraints

## Core Content

### 1. The Array Loading Problem

#### Stochastic Single-Atom Loading

When loading atoms from a MOT into optical tweezers, the process is inherently stochastic:

**Light-assisted collisions** in tightly confined tweezers:
- Two atoms in same trap undergo rapid loss
- Either 0 or 1 atom remains (parity projection)
- Loading probability per site: $p \approx 0.5$

For an array of $N$ sites, the probability of perfect loading:
$$P_{perfect} = p^N = (0.5)^N$$

For $N = 100$: $P_{perfect} = 10^{-30}$ — essentially impossible!

#### Array Preparation Challenge

To prepare a useful quantum register:
1. Load atoms into a large **reservoir array**
2. **Image** to determine which sites are occupied
3. **Rearrange** atoms into a compact **target array**

The reservoir must be larger than the target to ensure enough atoms.

### 2. Fluorescence Imaging for Defect Detection

#### State-Selective Imaging

Atoms are detected via resonant fluorescence:
$$\Gamma_{fluor} = \frac{\Gamma}{2}\frac{s}{1 + s + (2\Delta/\Gamma)^2}$$

where $s = I/I_{sat}$ is the saturation parameter.

**Imaging requirements:**
- High photon collection efficiency
- Good signal-to-noise ratio
- Minimal heating during imaging

#### Photon Budget

Collected photons per atom:
$$N_{ph} = \eta \cdot \Gamma_{fluor} \cdot t_{image}$$

where $\eta$ is the collection efficiency (typically 0.01-0.1).

For reliable detection with Poisson statistics:
$$\text{SNR} = \frac{N_{ph}}{\sqrt{N_{ph} + N_{bg}}} > 3$$

**Typical parameters:**
- Collection efficiency: $\eta \approx 0.05$ (NA = 0.5)
- Imaging time: $t_{image} \approx 10-50$ ms
- Photons collected: $N_{ph} \approx 1000-5000$

#### Heating During Imaging

Photon recoil deposits energy:
$$E_{recoil} = \frac{\hbar^2 k^2}{2m}$$

Heating rate:
$$\dot{T} = \frac{E_{recoil} \cdot \Gamma_{sc}}{k_B}$$

For Rb at saturation ($\Gamma_{sc} \approx 3$ MHz):
$$\dot{T} \approx 1\,\text{mK/ms}$$

**Mitigation strategies:**
- Gray molasses cooling during imaging
- Short imaging pulses with recooling
- Low saturation imaging

### 3. Atom Rearrangement Technology

#### Dynamic Tweezer Control

Atoms can be moved by steering the trapping laser:

**AOD-based moving:**
- Change RF frequency → deflection angle changes
- Atom follows trap adiabatically
- Speed limited by trap frequency: $v_{max} \approx \omega_{trap} \times \sigma_{trap}$

**SLM-based moving:**
- Update hologram to change trap positions
- Slower (limited by SLM refresh rate)
- More flexible for complex rearrangements

#### Adiabaticity Condition

For an atom to follow its trap without heating:
$$\left|\frac{d\omega_{trap}}{dt}\right| \ll \omega_{trap}^2$$

and
$$\left|\frac{dv}{dt}\right| \ll \omega_{trap} v$$

For a trap with $\omega_{trap} = 2\pi \times 100$ kHz:
$$a_{max} \approx \omega_{trap}^2 \sigma_{trap} \approx 10^{6}\,\text{m/s}^2$$

This is very high, so acceleration is rarely limiting.

**Practical speed limit:**
Moving 10 μm in 100 μs → $v = 0.1$ m/s, $a = 2000$ m/s²

This is easily achievable while maintaining adiabaticity.

### 4. Sorting Algorithms

#### Greedy Algorithm (Nearest-Neighbor)

Simple approach:
1. Find closest occupied reservoir site to first target
2. Move that atom to the target
3. Repeat for remaining targets

**Complexity:** $O(N \cdot M)$ where $N$ = target sites, $M$ = reservoir sites

**Suboptimality:** May require longer total path than optimal

#### Hungarian Algorithm (Optimal Assignment)

Finds minimum total distance assignment:
1. Construct cost matrix: $C_{ij}$ = distance from reservoir site $i$ to target $j$
2. Apply Hungarian algorithm
3. Execute moves in optimal sequence

**Complexity:** $O(\max(N,M)^3)$

**Advantage:** Provably optimal total distance

#### Parallel Sorting with Collision Avoidance

Multiple atoms can be moved simultaneously if paths don't collide:

**Graph-based approach:**
1. Construct dependency graph (edges = potential collisions)
2. Find maximum independent set for parallel moves
3. Execute in parallel batches

**Speedup:** Factor of 2-5× over sequential sorting

### 5. Loading Efficiency Optimization

#### Required Reservoir Size

To achieve target occupancy with probability $P_{success}$:

For $n$ target sites and loading probability $p$:
$$P(k \geq n | M, p) = \sum_{k=n}^{M} \binom{M}{k} p^k (1-p)^{M-k}$$

For $p = 0.5$ and $P_{success} = 0.99$:
- $n = 10$: Need $M \approx 25$ sites
- $n = 100$: Need $M \approx 220$ sites
- $n = 1000$: Need $M \approx 2100$ sites

The overhead scales as $\approx 2n + 4\sqrt{n}$ for 99% success.

#### Enhanced Loading Techniques

**Blue-detuned push-out beam:**
Remove excess atoms before parity projection
→ Achieve $p > 0.9$ per site

**Optical reservoir with continuous loading:**
Replenish atoms during sorting
→ Effectively unlimited supply

**Dual-species loading:**
Use auxiliary species as replaceable reservoir

### 6. Preparation Time Budget

#### Sequential Steps

| Step | Duration | Notes |
|------|----------|-------|
| MOT loading | 100-500 ms | Chamber-dependent |
| Transfer to tweezers | 10-50 ms | Adiabatic handoff |
| First imaging | 10-30 ms | Detect occupancy |
| Atom sorting | 10-100 ms | Depends on array size |
| Cooling/recool | 5-20 ms | After each move |
| Final verification | 10-30 ms | Confirm success |

**Total:** 150-750 ms per preparation cycle

#### Repetition Rate

For 99% success probability:
- Preparation rate: 1-5 Hz
- Quantum circuit time: <1 ms typically
- **Duty cycle:** <1% for computation

This is a major overhead for neutral atom QC!

#### Pipelining Strategies

**Zone architecture:**
- Preparation zone (loading, sorting)
- Computation zone (gates)
- Measurement zone

Atoms move between zones, enabling parallel operation.

**Expected improvement:** 10× or more in effective computation rate

### 7. Defect-Free Array Architectures

#### Fixed Target Array

Simplest approach:
- Define fixed target positions
- Sort into exactly these positions
- Any defects require full reload

#### Flexible Target with Minimum Size

Alternative:
- Accept any compact region of size $\geq n$
- More flexible, higher success rate
- Requires adaptive algorithm execution

#### Defect-Tolerant Codes

For error-corrected computation:
- Use codes that tolerate missing qubits
- Flag defects to decoder
- Reduced preparation requirements

## Worked Examples

### Example 1: Reservoir Sizing Calculation

**Problem:** Calculate the reservoir array size needed to prepare a 100-qubit target array with 95% success probability, assuming 50% loading probability per site.

**Solution:**

**Step 1: Model the problem**
Number of loaded atoms follows binomial distribution:
$$k \sim \text{Binomial}(M, 0.5)$$

Need $P(k \geq 100) \geq 0.95$.

**Step 2: Normal approximation**
For large $M$, use normal approximation:
$$k \approx N(\mu = 0.5M, \sigma = \sqrt{0.25M})$$

We need:
$$P\left(\frac{k - 0.5M}{\sqrt{0.25M}} \geq \frac{100 - 0.5M}{\sqrt{0.25M}}\right) \geq 0.95$$

The 5th percentile of standard normal is $z_{0.05} = -1.645$.

$$\frac{100 - 0.5M}{\sqrt{0.25M}} \leq -1.645$$

**Step 3: Solve quadratic**
$$100 - 0.5M = -1.645 \times 0.5\sqrt{M}$$
$$100 - 0.5M = -0.8225\sqrt{M}$$

Let $x = \sqrt{M}$:
$$100 - 0.5x^2 = -0.8225x$$
$$0.5x^2 - 0.8225x - 100 = 0$$
$$x^2 - 1.645x - 200 = 0$$

$$x = \frac{1.645 + \sqrt{1.645^2 + 800}}{2} = \frac{1.645 + 28.3}{2} = 15.0$$

$$M = x^2 = 225$$

**Step 4: Verify**
For $M = 225$, $\mu = 112.5$, $\sigma = 7.5$.
$$P(k \geq 100) = P\left(z \geq \frac{100-112.5}{7.5}\right) = P(z \geq -1.67) = 0.95$$

**Answer:** Need approximately 225 reservoir sites for 100-qubit target.

---

### Example 2: Sorting Time Estimate

**Problem:** Estimate the time to sort 100 atoms from a 225-site reservoir into a 10×10 target array using AOD-based rearrangement. Assume atom transport speed of 50 μm/ms and average move distance of 20 μm.

**Solution:**

**Step 1: Average path length**
With 100 atoms to move and average distance 20 μm:
Total sequential path: $100 \times 20\,\mu\text{m} = 2000\,\mu\text{m}$

**Step 2: Sequential sorting time**
$$t_{seq} = \frac{2000\,\mu\text{m}}{50\,\mu\text{m/ms}} = 40\,\text{ms}$$

**Step 3: Parallel sorting**
With 2D AOD, can move row-by-row or column-by-column.
Assuming 10 parallel moves per batch:
$$t_{parallel} \approx \frac{40\,\text{ms}}{5} = 8\,\text{ms}$$

(Factor of 5 rather than 10 due to collision avoidance overhead)

**Step 4: Add overhead**
- Initial imaging: 20 ms
- Recooling after moves: 10 ms
- Final verification: 15 ms

**Total:** 8 + 20 + 10 + 15 = 53 ms sorting time

---

### Example 3: Imaging Signal-to-Noise

**Problem:** Calculate the signal-to-noise ratio for detecting a single Rb-87 atom with 30 ms imaging at saturation parameter $s = 1$, using an objective with NA = 0.6. The camera has 10 counts/photon and 5 counts background noise.

**Solution:**

**Step 1: Scattering rate**
At $s = 1$ on resonance:
$$\Gamma_{sc} = \frac{\Gamma}{2} \cdot \frac{s}{1+s} = \frac{6\,\text{MHz}}{2} \cdot \frac{1}{2} = 1.5\,\text{MHz}$$

**Step 2: Photons scattered**
$$N_{scattered} = \Gamma_{sc} \times t = 1.5 \times 10^6 \times 0.030 = 45000$$

**Step 3: Collection efficiency**
$$\eta = \frac{1}{2}(1 - \cos\theta_{max}) = \frac{1}{2}(1 - \sqrt{1 - NA^2})$$
$$\eta = \frac{1}{2}(1 - \sqrt{1 - 0.36}) = \frac{1}{2}(1 - 0.8) = 0.1$$

**Step 4: Detected photons**
$$N_{detected} = \eta \times N_{scattered} \times QE$$

Assuming quantum efficiency QE = 0.8:
$$N_{detected} = 0.1 \times 45000 \times 0.8 = 3600$$

**Step 5: Camera counts**
Signal counts: $S = 3600 \times 10 = 36000$
Background counts: $B = 5$ (per pixel, assume 10 pixels): $B_{total} = 50$

**Step 6: SNR**
$$\text{SNR} = \frac{S}{\sqrt{S + B_{total}}} = \frac{36000}{\sqrt{36050}} = 190$$

This is excellent SNR, enabling reliable single-shot detection.

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** What reservoir size gives 99.9% probability of having at least 50 atoms if loading probability is 50%?

**Problem 1.2:** An atom is transported 30 μm in 500 μs. What is the average velocity and is it adiabatic for a 100 kHz trap?

**Problem 1.3:** Calculate the photon recoil energy for Rb-87 at 780 nm and the temperature increase from scattering 10,000 photons.

### Level 2: Intermediate Analysis

**Problem 2.1:** Design an imaging protocol for a 200-site array that achieves >0.999 fidelity single-shot detection while keeping heating below 100 μK. Specify imaging time and saturation parameter.

**Problem 2.2:** Compare greedy vs Hungarian algorithm for a 5×5 target extracted from a 7×7 reservoir with random 60% initial occupancy. Simulate 100 trials and calculate average path length.

**Problem 2.3:** Calculate the duty cycle for quantum computation given:
- Preparation time: 400 ms
- Circuit depth: 50 two-qubit gates at 500 ns each
- Success probability: 0.98 per preparation

### Level 3: Challenging Problems

**Problem 3.1:** Derive the optimal reservoir size as a function of target size $n$ and loading probability $p$ that minimizes total overhead (preparation time + wasted atoms).

**Problem 3.2:** Design a parallel sorting algorithm that maximizes throughput while respecting collision avoidance. Calculate the speedup over sequential sorting as a function of array dimensions.

**Problem 3.3:** Analyze the effect of atom loss during sorting on the final success probability. If loss probability per move is $\epsilon = 0.1\%$, what is the maximum array size sortable with >95% final success?

## Computational Lab: Array Sorting Simulations

### Lab 1: Reservoir and Loading Statistics

```python
"""
Day 916 Lab: Atom Sorting and Array Preparation
Simulating stochastic loading and sorting algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import brentq

def loading_success_probability(n_target, n_reservoir, p_load=0.5):
    """
    Calculate probability of having at least n_target atoms
    when loading n_reservoir sites with probability p_load.
    """
    return 1 - binom.cdf(n_target - 1, n_reservoir, p_load)

def minimum_reservoir(n_target, p_success, p_load=0.5):
    """
    Find minimum reservoir size for given success probability.
    """
    # Search for minimum M
    for M in range(n_target, 10 * n_target):
        if loading_success_probability(n_target, M, p_load) >= p_success:
            return M
    return None

# Analyze reservoir requirements
n_targets = np.arange(10, 1001, 10)
success_probs = [0.90, 0.95, 0.99, 0.999]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for p_success in success_probs:
    reservoirs = [minimum_reservoir(n, p_success) for n in n_targets]
    axes[0].plot(n_targets, reservoirs, label=f'P = {p_success}', linewidth=2)
    axes[1].plot(n_targets, np.array(reservoirs)/n_targets,
                label=f'P = {p_success}', linewidth=2)

axes[0].set_xlabel('Target array size')
axes[0].set_ylabel('Required reservoir size')
axes[0].set_title('Reservoir Sizing')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Target array size')
axes[1].set_ylabel('Reservoir / Target ratio')
axes[1].set_title('Reservoir Overhead')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=2, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('reservoir_requirements.png', dpi=150, bbox_inches='tight')
plt.show()

# Monte Carlo simulation
print("=== Monte Carlo Loading Simulation ===")

n_target = 100
n_reservoir = 225
p_load = 0.5
n_trials = 10000

# Simulate loading
loaded_counts = np.random.binomial(n_reservoir, p_load, n_trials)
success_count = np.sum(loaded_counts >= n_target)

print(f"Target: {n_target} atoms")
print(f"Reservoir: {n_reservoir} sites")
print(f"Loading probability: {p_load}")
print(f"Trials: {n_trials}")
print(f"Successes: {success_count} ({100*success_count/n_trials:.1f}%)")
print(f"Theoretical: {100*loading_success_probability(n_target, n_reservoir):.1f}%")

# Histogram
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(loaded_counts, bins=30, density=True, alpha=0.7, color='steelblue',
        label='Simulation')

# Theoretical distribution
k = np.arange(0, n_reservoir+1)
p_k = binom.pmf(k, n_reservoir, p_load)
ax.plot(k, p_k, 'r-', linewidth=2, label='Binomial theory')

ax.axvline(x=n_target, color='green', linestyle='--', linewidth=2,
           label=f'Target = {n_target}')
ax.fill_between(k[k >= n_target], p_k[k >= n_target], alpha=0.3, color='green')

ax.set_xlabel('Number of loaded atoms')
ax.set_ylabel('Probability density')
ax.set_title(f'Loading Distribution (M={n_reservoir}, p={p_load})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loading_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 2: Sorting Algorithms

```python
"""
Lab 2: Implement and compare sorting algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def generate_random_loading(n_rows, n_cols, p_load=0.5):
    """Generate random occupancy pattern."""
    return np.random.random((n_rows, n_cols)) < p_load

def get_occupied_positions(occupancy):
    """Get list of occupied site coordinates."""
    return list(zip(*np.where(occupancy)))

def get_target_positions(n_target_rows, n_target_cols, offset=(0, 0)):
    """Generate target array positions (compact square)."""
    targets = []
    for i in range(n_target_rows):
        for j in range(n_target_cols):
            targets.append((i + offset[0], j + offset[1]))
    return targets

def distance(pos1, pos2):
    """Euclidean distance between positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def greedy_assignment(sources, targets):
    """
    Greedy (nearest neighbor) assignment algorithm.

    Returns list of (source, target) pairs and total distance.
    """
    sources = list(sources)
    targets = list(targets)
    assignments = []
    total_distance = 0

    remaining_sources = sources.copy()

    for target in targets:
        if not remaining_sources:
            break

        # Find closest source
        distances = [distance(s, target) for s in remaining_sources]
        min_idx = np.argmin(distances)
        best_source = remaining_sources.pop(min_idx)

        assignments.append((best_source, target))
        total_distance += distances[min_idx]

    return assignments, total_distance

def hungarian_assignment(sources, targets):
    """
    Optimal assignment using Hungarian algorithm.

    Returns list of (source, target) pairs and total distance.
    """
    sources = list(sources)
    targets = list(targets)

    if len(sources) < len(targets):
        raise ValueError("Not enough sources for targets")

    # Build cost matrix
    cost_matrix = np.zeros((len(sources), len(targets)))
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            cost_matrix[i, j] = distance(s, t)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = [(sources[i], targets[j]) for i, j in zip(row_ind, col_ind)]
    total_distance = cost_matrix[row_ind, col_ind].sum()

    return assignments, total_distance

def visualize_sorting(occupancy, assignments, target_positions, title=""):
    """Visualize the sorting assignment."""
    fig, ax = plt.subplots(figsize=(10, 10))

    n_rows, n_cols = occupancy.shape

    # Draw grid
    for i in range(n_rows + 1):
        ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(n_cols + 1):
        ax.axvline(x=j-0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Mark target region
    for t in target_positions:
        ax.add_patch(plt.Rectangle((t[1]-0.5, t[0]-0.5), 1, 1,
                                    fill=True, alpha=0.2, color='green'))

    # Mark all occupied sites
    occupied = get_occupied_positions(occupancy)
    used_sources = [a[0] for a in assignments]
    for pos in occupied:
        if pos in used_sources:
            ax.plot(pos[1], pos[0], 'bo', markersize=15)
        else:
            ax.plot(pos[1], pos[0], 'ro', markersize=10, alpha=0.5)

    # Draw assignment arrows
    for source, target in assignments:
        ax.annotate('', xy=(target[1], target[0]), xytext=(source[1], source[0]),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title)

    return fig, ax

# Compare algorithms
np.random.seed(42)

n_res = 8  # 8x8 reservoir
n_target = 5  # 5x5 target
p_load = 0.6

print("=== Sorting Algorithm Comparison ===\n")

# Generate random loading
occupancy = generate_random_loading(n_res, n_res, p_load)
n_loaded = np.sum(occupancy)

print(f"Reservoir: {n_res}x{n_res} = {n_res**2} sites")
print(f"Loaded atoms: {n_loaded}")
print(f"Target: {n_target}x{n_target} = {n_target**2} sites")

if n_loaded < n_target**2:
    print("ERROR: Not enough atoms loaded!")
else:
    sources = get_occupied_positions(occupancy)
    targets = get_target_positions(n_target, n_target, offset=(1, 1))

    # Greedy algorithm
    assign_greedy, dist_greedy = greedy_assignment(sources, targets)
    print(f"\nGreedy algorithm:")
    print(f"  Total distance: {dist_greedy:.2f} site units")

    # Hungarian algorithm
    assign_hungarian, dist_hungarian = hungarian_assignment(sources, targets)
    print(f"\nHungarian algorithm:")
    print(f"  Total distance: {dist_hungarian:.2f} site units")

    print(f"\nImprovement: {100*(dist_greedy - dist_hungarian)/dist_greedy:.1f}%")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plt.sca(axes[0])
    visualize_sorting(occupancy, assign_greedy, targets,
                      f"Greedy (d = {dist_greedy:.2f})")

    plt.sca(axes[1])
    visualize_sorting(occupancy, assign_hungarian, targets,
                      f"Hungarian (d = {dist_hungarian:.2f})")

    plt.tight_layout()
    plt.savefig('sorting_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Statistical comparison over many trials
n_trials = 100
greedy_distances = []
hungarian_distances = []

for _ in range(n_trials):
    occupancy = generate_random_loading(n_res, n_res, p_load)
    if np.sum(occupancy) >= n_target**2:
        sources = get_occupied_positions(occupancy)
        targets = get_target_positions(n_target, n_target, offset=(1, 1))

        _, d_g = greedy_assignment(sources, targets)
        _, d_h = hungarian_assignment(sources, targets)

        greedy_distances.append(d_g)
        hungarian_distances.append(d_h)

print(f"\n=== Statistical Comparison ({len(greedy_distances)} trials) ===")
print(f"Greedy: {np.mean(greedy_distances):.2f} ± {np.std(greedy_distances):.2f}")
print(f"Hungarian: {np.mean(hungarian_distances):.2f} ± {np.std(hungarian_distances):.2f}")
print(f"Average improvement: {100*(1 - np.mean(hungarian_distances)/np.mean(greedy_distances)):.1f}%")
```

### Lab 3: Sorting Time and Success Rate

```python
"""
Lab 3: Analyze sorting time and overall preparation success
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def sorting_time(assignments, move_speed=50, spacing=4):
    """
    Calculate sequential sorting time.

    Parameters:
    -----------
    assignments : list of (source, target) tuples
        Position assignments
    move_speed : float
        Moving speed in μm/ms
    spacing : float
        Lattice spacing in μm
    """
    total_distance = sum(np.sqrt((s[0]-t[0])**2 + (s[1]-t[1])**2)
                        for s, t in assignments)
    distance_um = total_distance * spacing
    return distance_um / move_speed

def parallel_sorting_time(assignments, move_speed=50, spacing=4, parallelism=4):
    """
    Estimate parallel sorting time.
    Assumes some fraction of moves can be parallelized.
    """
    sequential_time = sorting_time(assignments, move_speed, spacing)
    # Simple model: effective speedup proportional to parallelism
    # but limited by dependencies
    effective_speedup = min(parallelism, len(assignments) / 2)
    return sequential_time / effective_speedup

def full_preparation_cycle(n_target_rows, n_target_cols, n_reservoir,
                           p_load=0.5, move_speed=50, spacing=4):
    """
    Simulate full preparation cycle.

    Returns:
    --------
    success : bool
    total_time : float (ms)
    details : dict
    """
    n_target = n_target_rows * n_target_cols

    # Simulate loading
    loaded = np.random.binomial(n_reservoir, p_load)

    if loaded < n_target:
        return False, 0, {'loaded': loaded, 'needed': n_target, 'success': False}

    # Generate random positions
    n_res_side = int(np.ceil(np.sqrt(n_reservoir)))
    occupancy = np.zeros((n_res_side, n_res_side), dtype=bool)
    positions = np.random.choice(n_reservoir, loaded, replace=False)
    for p in positions:
        occupancy[p // n_res_side, p % n_res_side] = True

    sources = list(zip(*np.where(occupancy)))[:loaded]
    targets = [(i, j) for i in range(n_target_rows) for j in range(n_target_cols)]

    # Calculate sorting distance (simplified)
    avg_distance = np.mean([np.sqrt((s[0]-t[0])**2 + (s[1]-t[1])**2)
                           for s, t in zip(sources[:n_target], targets)])
    total_distance = avg_distance * n_target

    # Timing
    t_image = 20  # ms
    t_sort = total_distance * spacing / move_speed
    t_recool = 10  # ms
    t_verify = 15  # ms

    total_time = t_image + t_sort + t_recool + t_verify

    details = {
        'loaded': loaded,
        'needed': n_target,
        'success': True,
        't_image': t_image,
        't_sort': t_sort,
        't_recool': t_recool,
        't_verify': t_verify
    }

    return True, total_time, details

# Analyze preparation statistics
np.random.seed(42)

# Fixed target, vary reservoir
n_target = 10  # 10x10 = 100 atoms
reservoir_multipliers = np.linspace(1.5, 3.0, 20)
n_trials = 500

success_rates = []
mean_times = []

for mult in reservoir_multipliers:
    n_reservoir = int(mult * n_target**2)
    successes = 0
    times = []

    for _ in range(n_trials):
        success, t, _ = full_preparation_cycle(n_target, n_target, n_reservoir)
        if success:
            successes += 1
            times.append(t)

    success_rates.append(successes / n_trials)
    mean_times.append(np.mean(times) if times else 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(reservoir_multipliers, np.array(success_rates)*100, 'b-', linewidth=2)
axes[0].axhline(y=99, color='r', linestyle='--', label='99% threshold')
axes[0].set_xlabel('Reservoir / Target ratio')
axes[0].set_ylabel('Success rate (%)')
axes[0].set_title('Preparation Success Rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(reservoir_multipliers, mean_times, 'g-', linewidth=2)
axes[1].set_xlabel('Reservoir / Target ratio')
axes[1].set_ylabel('Mean preparation time (ms)')
axes[1].set_title('Preparation Time')
axes[1].grid(True, alpha=0.3)

# Effective preparation rate
effective_rate = np.array(success_rates) / (np.array(mean_times) + 1)
axes[2].plot(reservoir_multipliers, effective_rate * 1000, 'r-', linewidth=2)
axes[2].set_xlabel('Reservoir / Target ratio')
axes[2].set_ylabel('Effective rate (arrays/s)')
axes[2].set_title('Effective Preparation Rate')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preparation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Scaling with target size
print("\n=== Scaling with Target Size ===")

target_sizes = [5, 7, 10, 12, 15, 20]
results = {}

for n in target_sizes:
    n_reservoir = int(2.2 * n**2)  # Fixed 2.2x overhead

    successes = 0
    times = []

    for _ in range(200):
        success, t, _ = full_preparation_cycle(n, n, n_reservoir)
        if success:
            successes += 1
            times.append(t)

    results[n] = {
        'success_rate': successes / 200,
        'mean_time': np.mean(times) if times else 0,
        'reservoir': n_reservoir
    }

    print(f"{n}x{n} ({n**2} qubits): "
          f"success = {100*results[n]['success_rate']:.1f}%, "
          f"time = {results[n]['mean_time']:.0f} ms")

# Plot scaling
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sizes = [n**2 for n in target_sizes]
times = [results[n]['mean_time'] for n in target_sizes]
success = [results[n]['success_rate']*100 for n in target_sizes]

axes[0].plot(sizes, times, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Target array size (qubits)')
axes[0].set_ylabel('Mean preparation time (ms)')
axes[0].set_title('Preparation Time Scaling')
axes[0].grid(True, alpha=0.3)

axes[1].plot(sizes, success, 'go-', linewidth=2, markersize=8)
axes[1].axhline(y=95, color='r', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Target array size (qubits)')
axes[1].set_ylabel('Success rate (%)')
axes[1].set_title('Success Rate (2.2x reservoir)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Summary

### Key Formulas Table

| Quantity | Formula | Typical Value |
|----------|---------|---------------|
| Loading probability | $P_{perfect} = p^N \approx 0.5^N$ | ~0 for N>20 |
| Reservoir overhead | $M/N \approx 2 + 4/\sqrt{N}$ | 2-2.5× |
| Imaging SNR | $SNR = N_{ph}/\sqrt{N_{ph} + N_{bg}}$ | >100 |
| Move time | $t = d/v$ | ~0.5 ms per 25 μm |
| Total prep time | $t_{total} = t_{image} + t_{sort} + t_{cool}$ | 50-200 ms |

### Main Takeaways

1. **Stochastic loading** with ~50% probability per site necessitates reservoir arrays 2-2.5× larger than the target for high success probability.

2. **Fluorescence imaging** enables high-fidelity single-atom detection in ~20 ms with proper cooling strategies.

3. **Sorting algorithms** trade complexity for efficiency; Hungarian algorithm gives optimal paths but greedy is often sufficient.

4. **Preparation overhead** dominates total experiment time, with duty cycles <1% for quantum computation.

5. **Scaling considerations** favor larger arrays (reservoir overhead decreases) but preparation time increases.

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain why stochastic loading limits direct preparation
- [ ] I understand the imaging requirements for defect detection
- [ ] I can describe sorting algorithm tradeoffs
- [ ] I know the factors limiting preparation rate

### Mathematical Skills
- [ ] I can calculate reservoir sizes for target success rates
- [ ] I can estimate sorting distances and times
- [ ] I can analyze imaging SNR requirements

### Computational Skills
- [ ] I can simulate loading statistics
- [ ] I can implement sorting algorithms
- [ ] I can analyze preparation scaling

## Preview: Day 917

Tomorrow we explore **Mid-Circuit Measurement** in neutral atom systems, including:
- Dual-species arrays for non-destructive measurement
- Ancilla atoms for syndrome extraction
- Real-time classical feedback
- Applications to error correction

Mid-circuit measurement capability is essential for fault-tolerant quantum computing, enabling syndrome extraction and real-time error correction.
