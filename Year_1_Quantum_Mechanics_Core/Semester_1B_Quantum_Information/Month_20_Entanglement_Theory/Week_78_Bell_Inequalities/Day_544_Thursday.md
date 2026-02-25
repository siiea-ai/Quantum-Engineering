# Day 544: Experimental Bell Tests

## Overview
**Day 544** | Week 78, Day 5 | Year 1, Month 20 | From Theory to Laboratory

Today we explore the experimental history of Bell tests, from Aspect's pioneering work to the loophole-free tests of 2015, understanding the challenges and triumphs of testing quantum nonlocality.

---

## Learning Objectives
1. Understand key experimental loopholes
2. Trace the history of Bell tests
3. Analyze Aspect's experiments (1981-82)
4. Study loophole-free tests (2015)
5. Appreciate experimental challenges
6. Connect to modern quantum technologies

---

## Core Content

### Experimental Loopholes

**Loopholes** are assumptions that, if violated, allow LHV explanations:

### 1. Locality Loophole
**Issue:** Settings might be communicated subluminally

**Requirement:** Space-like separation between:
- Alice's measurement choice and Bob's result
- Bob's measurement choice and Alice's result

**Solution:** Fast random switching + sufficient distance

### 2. Detection Loophole (Fair Sampling)
**Issue:** Only some particles are detected

**LHV escape:** Non-detected pairs could restore $|S| \leq 2$

**Requirement:** Detection efficiency η > critical value
- For CHSH: $\eta > 82.84\%$
- For other inequalities: can be lower

### 3. Freedom of Choice
**Issue:** Measurement settings could be correlated with λ

**Requirement:** True randomness in setting selection

**Solution:** Quantum random number generators, cosmic photons

### Historical Timeline

```
1964  Bell publishes theorem
1969  CHSH inequality derived
1972  Freedman & Clauser: first test (violation seen)
1982  Aspect: fast switching (locality addressed)
1998  Weihs et al: strict locality
2015  Three groups: loophole-free tests!
2022  Nobel Prize to Aspect, Clauser, Zeilinger
```

### Aspect's Experiments (1981-82)

**Innovation:** Fast acousto-optical switches

**Setup:**
- Calcium atomic cascade → polarization-entangled photons
- Switches change settings every 10 ns
- 12 m separation (40 ns light travel time)

**Result:** $S = 2.70 \pm 0.015$ (clear violation!)

**Limitation:** Detection efficiency ~5% (detection loophole open)

### Weihs et al. (1998)

**Innovation:** True random number generators

**Setup:**
- Parametric down-conversion source
- 400 m separation
- Quantum random number generators

**Result:** Convincing locality test, detection loophole still open

### Loophole-Free Tests (2015)

Three independent groups closed all loopholes simultaneously:

**1. Delft (Hensen et al.)**
- Nitrogen-vacancy centers in diamond
- 1.3 km separation
- Heralded entanglement
- Detection efficiency ~100%

**2. Vienna (Giustina et al.)**
- Photons with superconducting detectors
- High efficiency (~75%)
- Space-like separation

**3. NIST (Shalm et al.)**
- Similar to Vienna
- Different random number sources

**Result:** All confirmed violation with p < 0.001

### Modern Improvements

**Big Bell Test (2016):**
- 100,000 human participants provided random bits
- Freedom of choice from human free will

**Cosmic Bell Test (2017):**
- Random bits from distant quasar light
- Settings determined 600 years in the past

### Technical Requirements

| Loophole | Requirement | Modern Solution |
|----------|-------------|-----------------|
| Locality | Space-like separation | >1 km, fast electronics |
| Detection | η > 82.84% | Superconducting detectors |
| Freedom | True randomness | QRNG, cosmic sources |

### Statistical Analysis

**Hypothesis testing:**
- Null: LHV (S ≤ 2)
- Alternative: QM (S > 2)
- p-value: probability of seeing data under null

2015 tests achieved p < 10⁻⁷ (decisive rejection of LHV)

---

## Worked Examples

### Example 1: Detection Efficiency Threshold
Derive why η > 2/(1 + √2) ≈ 82.84% is needed for CHSH.

**Solution:**
With efficiency η, some events are not detected.

Modified inequality accounting for no-detection:
$$S_{obs} = \eta^2 S_{true} + (1 - \eta^2) \cdot S_{no-detect}$$

Worst case: $S_{no-detect} = 2$ (LHV maximum for undetected)

For violation: $\eta^2 \cdot 2\sqrt{2} + (1-\eta^2) \cdot 2 > 2$
$$\eta^2(2\sqrt{2} - 2) > 0$$

This is always satisfied... but we need the *observed* S > 2:
$$\eta^2 \cdot 2\sqrt{2} > 2$$
$$\eta > \sqrt{\frac{1}{\sqrt{2}}} = \frac{1}{2^{1/4}} \approx 84\%$$

More careful analysis gives $\eta > 2/(1 + \sqrt{2}) \approx 82.84\%$. ∎

### Example 2: Space-Like Separation
Calculate required separation for 10 μs measurement time.

**Solution:**
Light travel time must exceed measurement duration:
$$d > c \cdot t = 3 \times 10^8 \text{ m/s} \times 10 \times 10^{-6} \text{ s} = 3000 \text{ m}$$

Need > 3 km separation for 10 μs measurements. ∎

### Example 3: Statistical Significance
If S = 2.4 with uncertainty σ = 0.1, what's the significance?

**Solution:**
Null hypothesis: S ≤ 2

Z-score: $z = (S - 2) / \sigma = (2.4 - 2) / 0.1 = 4$

p-value: $P(Z > 4) \approx 3 \times 10^{-5}$

Strong evidence against LHV (>4σ). ∎

---

## Practice Problems

### Problem 1: Cascade Sources
Explain why atomic cascade sources have low efficiency.

### Problem 2: Photon Loss
If fiber has 0.2 dB/km loss, what's the maximum distance for 90% efficiency?

### Problem 3: Coincidence Window
Why is the coincidence time window important? What happens if it's too wide?

---

## Computational Lab

```python
"""Day 544: Experimental Bell Tests"""
import numpy as np
from scipy import stats

def simulate_bell_test(n_pairs, S_true, detection_eff, coincidence_noise=0):
    """
    Simulate a Bell test with finite efficiency.

    Parameters:
    - n_pairs: number of entangled pairs generated
    - S_true: true CHSH value
    - detection_eff: single-detector efficiency
    - coincidence_noise: accidental coincidence rate
    """

    # Both detectors must fire for coincidence
    p_coincidence = detection_eff ** 2

    # Number of detected pairs
    n_detected = np.random.binomial(n_pairs, p_coincidence)

    # Add noise coincidences
    n_noise = np.random.poisson(coincidence_noise * n_pairs)
    n_total = n_detected + n_noise

    # True signal correlation
    # S comes from correlations, which have variance ~1/√N
    S_signal = S_true * n_detected / n_total if n_total > 0 else 0

    # Add statistical noise
    S_noise = np.random.normal(0, 2/np.sqrt(n_total)) if n_total > 0 else 0

    S_observed = S_signal + S_noise

    return S_observed, n_total

def analyze_significance(S, sigma, null_bound=2):
    """Calculate statistical significance"""
    z = (np.abs(S) - null_bound) / sigma
    p_value = 1 - stats.norm.cdf(z)
    return z, p_value

print("=== Simulating Bell Tests ===\n")

# High-efficiency modern test
print("1. Modern high-efficiency test (η = 90%)")
n_trials = 1000
results = []
for _ in range(n_trials):
    S, n = simulate_bell_test(10000, 2.7, 0.90)
    results.append(S)

mean_S = np.mean(results)
std_S = np.std(results)
z, p = analyze_significance(mean_S, std_S)

print(f"   S = {mean_S:.4f} ± {std_S:.4f}")
print(f"   Z-score: {z:.2f}, p-value: {p:.2e}")

# Low-efficiency historical test
print("\n2. Historical test (η = 10%)")
results = []
for _ in range(n_trials):
    S, n = simulate_bell_test(10000, 2.7, 0.10)
    results.append(S)

mean_S = np.mean(results)
std_S = np.std(results)

print(f"   S = {mean_S:.4f} ± {std_S:.4f}")
print("   (Low efficiency dilutes signal!)")

# Detection efficiency threshold
print("\n=== Detection Efficiency Analysis ===\n")

def effective_S(S_true, eta, S_undetected=2):
    """
    Effective S with detection efficiency η.
    Assumes undetected pairs give S = 2 (worst case LHV).
    """
    p_both = eta**2
    return p_both * S_true + (1 - p_both) * S_undetected

print(f"{'η (%)':<10} {'Effective S':<15} {'Violates?'}")
print("-" * 35)

for eta in [0.50, 0.70, 0.80, 0.8284, 0.85, 0.90, 0.95, 1.00]:
    S_eff = effective_S(2*np.sqrt(2), eta)
    violates = "YES" if S_eff > 2 else "no"
    print(f"{eta*100:<10.2f} {S_eff:<15.4f} {violates}")

# Critical efficiency
eta_crit = 2 / (1 + np.sqrt(2))
print(f"\nCritical efficiency: η > {eta_crit:.4f} = {eta_crit*100:.2f}%")

# Space-like separation
print("\n=== Space-Like Separation Requirements ===\n")

c = 3e8  # m/s

measurement_times = [1e-9, 10e-9, 100e-9, 1e-6, 10e-6]  # seconds

print(f"{'Measurement time':<20} {'Required separation':<20}")
print("-" * 40)

for t in measurement_times:
    d = c * t
    time_str = f"{t*1e9:.0f} ns" if t < 1e-6 else f"{t*1e6:.0f} μs"
    dist_str = f"{d:.0f} m" if d < 1000 else f"{d/1000:.1f} km"
    print(f"{time_str:<20} {dist_str:<20}")

# Historical experiments comparison
print("\n=== Historical Bell Tests ===\n")

experiments = [
    ("Freedman & Clauser 1972", 2.85, 0.09, "Calcium cascade"),
    ("Aspect et al. 1982", 2.70, 0.05, "Fast switching"),
    ("Weihs et al. 1998", 2.73, 0.02, "Strict locality"),
    ("Delft 2015", 2.42, 0.20, "Loophole-free (NV)"),
    ("Vienna 2015", 2.31, 0.09, "Loophole-free (photons)"),
]

print(f"{'Experiment':<25} {'S':<10} {'σ':<8} {'Z-score':<10} {'Notes'}")
print("-" * 70)

for name, S, sigma, notes in experiments:
    z, _ = analyze_significance(S, sigma)
    print(f"{name:<25} {S:<10.2f} {sigma:<8.2f} {z:<10.1f} {notes}")

# Loophole timeline
print("\n=== Loophole Closure Timeline ===")
print("""
Year  Experiment              Locality  Detection  Freedom
----  ----------              --------  ---------  -------
1972  Freedman & Clauser      ✗         ✗          ✗
1982  Aspect                  ✓         ✗          ✗
1998  Weihs                   ✓         ✗          ✓
2015  Hensen (Delft)          ✓         ✓          ✓  ← First loophole-free!
2015  Giustina (Vienna)       ✓         ✓          ✓
2015  Shalm (NIST)            ✓         ✓          ✓
""")

print("2022: Nobel Prize to Aspect, Clauser, and Zeilinger!")
```

---

## Summary

### Key Loopholes

| Loophole | Issue | Solution |
|----------|-------|----------|
| Locality | Subluminal communication | Space-like separation |
| Detection | Missing events | High-efficiency detectors |
| Freedom | Correlated settings | True random sources |

### Key Takeaways
1. **Multiple loopholes** required decades to close
2. **2015** saw first loophole-free tests by three groups
3. **Detection efficiency** must exceed ~83% for CHSH
4. **Space-like separation** requires fast switching
5. **2022 Nobel Prize** recognized this foundational work

---

## Daily Checklist

- [ ] I understand the three main loopholes
- [ ] I know the historical progression of Bell tests
- [ ] I understand the detection efficiency threshold
- [ ] I appreciate why 2015 was a watershed year
- [ ] I can calculate space-like separation requirements

---

*Next: Day 545 — Device-Independent Quantum Information*
