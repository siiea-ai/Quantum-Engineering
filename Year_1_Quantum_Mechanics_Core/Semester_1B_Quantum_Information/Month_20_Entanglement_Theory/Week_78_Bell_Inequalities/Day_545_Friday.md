# Day 545: Device-Independent Quantum Information

## Overview
**Day 545** | Week 78, Day 6 | Year 1, Month 20 | Security from Bell Violations

Today we explore device-independent quantum information—protocols whose security relies only on observed Bell violation, not on trusting the devices themselves.

---

## Learning Objectives
1. Define device-independence precisely
2. Understand DIQKD (device-independent QKD)
3. Connect Bell violation to randomness certification
4. Explore self-testing of quantum states
5. Analyze the security advantages
6. Study practical implementations

---

## Core Content

### What is Device-Independence?

**Standard QKD (e.g., BB84):**
- Trust that devices prepare and measure correct states
- Manufacturer could be compromised
- Side-channel attacks possible

**Device-Independent:**
- Treat devices as black boxes
- Security from Bell violation alone
- No assumptions about internal workings

### The DI Paradigm

```
  ┌─────────────┐     ┌─────────────┐
  │  Black Box  │     │  Black Box  │
  │    Alice    │     │     Bob     │
  │   (input x) │     │   (input y) │
  │  (output a) │     │  (output b) │
  └─────────────┘     └─────────────┘
        │                   │
        └───────┬───────────┘
                │
         Correlations
          P(a,b|x,y)
```

If correlations violate Bell inequality → can extract:
1. **Randomness** (cannot be predetermined)
2. **Secret key** (cannot be known to adversary)

### Device-Independent QKD (DIQKD)

**Protocol:**
1. Alice and Bob share entangled pairs
2. Randomly choose measurement settings
3. Publicly announce settings
4. Compute CHSH S from results
5. If S > 2: extract secret key
6. Apply privacy amplification

**Security proof:** Bell violation certifies:
- State must be entangled
- Outcomes cannot be predetermined
- Eavesdropper has limited information

### Key Rate Formula

For DIQKD with observed S:
$$r \geq 1 - h\left(\frac{1 + \sqrt{(S/2)^2 - 1}}{2}\right)$$

where $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$ is binary entropy.

### Randomness Certification

**Problem:** How to get certified random numbers?

**Classical:** Cannot prove randomness (could be pseudorandom)

**Quantum DI:** Bell violation proves outcomes couldn't be predetermined!

**Min-entropy bound:**
$$H_{min}(A|E) \geq f(S)$$

where E is any adversary's knowledge.

### Self-Testing

**Theorem:** Maximum CHSH violation ($S = 2\sqrt{2}$) uniquely certifies:
1. State is (locally equivalent to) $|\Phi^+\rangle$
2. Measurements are (locally equivalent to) Pauli X and Z

**Implication:** Can verify quantum state and measurements from statistics alone!

### Robust Self-Testing

For $S = 2\sqrt{2} - \epsilon$ (near-maximal):
$$\|\rho - |\Phi^+\rangle\langle\Phi^+|\|_1 \leq O(\sqrt{\epsilon})$$

Small deviation from max violation → state close to Bell state.

### Practical Challenges

| Challenge | Issue | Mitigation |
|-----------|-------|------------|
| Detection loophole | Low efficiency → no violation | High-efficiency detectors |
| Finite statistics | Uncertainty in S | Large sample sizes |
| Memory attacks | Devices store info | Fresh randomness |
| Timing attacks | Correlations in time | Strict protocols |

### State-of-the-Art

**2022:** First experimental DIQKD over 400m (Oxford group)
- Used trapped ions
- Closed all loopholes
- Generated secure key

### Comparison: DI vs Standard QKD

| Property | Standard QKD | DIQKD |
|----------|-------------|-------|
| Trust in devices | Required | Not required |
| Efficiency needed | Low (~1%) | High (>80%) |
| Key rate | Higher | Lower |
| Security level | Implementation | Fundamental |

---

## Worked Examples

### Example 1: Key Rate from S
Calculate key rate for S = 2.6.

**Solution:**
$$\frac{S}{2} = 1.3$$
$$\sqrt{(1.3)^2 - 1} = \sqrt{0.69} \approx 0.831$$
$$p = \frac{1 + 0.831}{2} = 0.916$$
$$h(0.916) \approx 0.40$$
$$r \geq 1 - 0.40 = 0.60$$

About 0.6 bits of key per round! ∎

### Example 2: Self-Testing Statement
What can we certify from S = 2.8?

**Solution:**
Max is $2\sqrt{2} \approx 2.828$, so $\epsilon = 0.028$.

Self-testing bound:
$$\|\rho_{actual} - |\Phi^+\rangle\langle\Phi^+|\|_1 \lesssim \sqrt{0.028} \approx 0.17$$

The state is within trace distance 0.17 of a perfect Bell state! ∎

### Example 3: Min-Entropy Bound
For S = 2.4, bound the adversary's knowledge.

**Solution:**
Using standard bounds for CHSH:
$$H_{min}(A|E) \geq 1 - \log_2(1 + \sqrt{2 - S^2/4})$$
$$= 1 - \log_2(1 + \sqrt{2 - 1.44})$$
$$= 1 - \log_2(1 + 0.748) \approx 0.19$$

Adversary misses at least 0.19 bits per measurement. ∎

---

## Practice Problems

### Problem 1: Critical S for DIQKD
What minimum S is needed for positive key rate?

### Problem 2: Randomness Expansion
Design a protocol to expand n random bits to more bits using Bell tests.

### Problem 3: Multi-Party DI
How would three-party DI protocols work with GHZ states?

---

## Computational Lab

```python
"""Day 545: Device-Independent Quantum Information"""
import numpy as np
from scipy.optimize import minimize_scalar

def binary_entropy(p):
    """h(p) = -p log₂ p - (1-p) log₂ (1-p)"""
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def diqkd_key_rate(S):
    """
    Lower bound on key rate for DIQKD.
    Based on Pironio et al. 2009 / Acin et al. 2007.
    """
    if S <= 2:
        return 0  # No violation, no key

    # Quantum bound check
    S = min(S, 2*np.sqrt(2))

    # Error rate from S
    sqrt_term = np.sqrt((S/2)**2 - 1)
    p = (1 + sqrt_term) / 2

    # Key rate = 1 - h(error)
    return max(0, 1 - binary_entropy(p))

def min_entropy_bound(S):
    """
    Min-entropy H_min(A|E) from CHSH value.
    Adversary's ignorance about Alice's outcome.
    """
    if S <= 2:
        return 0

    S = min(S, 2*np.sqrt(2))

    # Standard bound
    return 1 - np.log2(1 + np.sqrt(2 - S**2/4))

def self_testing_bound(S):
    """
    Trace distance to ideal Bell state from observed S.
    """
    S_max = 2 * np.sqrt(2)
    epsilon = S_max - S

    if epsilon <= 0:
        return 0

    # Robust self-testing: distance scales as sqrt(epsilon)
    return np.sqrt(epsilon)

print("=== Device-Independent Key Rate ===\n")

print(f"{'S value':<10} {'Key rate':<12} {'Min-entropy':<12} {'Self-test dist'}")
print("-" * 50)

for S in [2.0, 2.2, 2.4, 2.6, 2.7, 2.8, 2*np.sqrt(2)]:
    kr = diqkd_key_rate(S)
    me = min_entropy_bound(S)
    st = self_testing_bound(S)
    print(f"{S:<10.4f} {kr:<12.4f} {me:<12.4f} {st:<12.4f}")

# Find minimum S for positive key rate
print("\n=== Critical CHSH Value ===")

def negative_key_rate(S):
    return -diqkd_key_rate(S)

# Search for zero crossing
for S in np.linspace(2.0, 2.5, 100):
    if diqkd_key_rate(S) > 0.001:
        S_crit = S
        break

print(f"Minimum S for positive key rate: S > {S_crit:.4f}")
print(f"(Theoretical: S > 2√2 × cos(π/8)² ≈ 2.28)")

# Compare standard vs DI QKD
print("\n=== Standard vs Device-Independent QKD ===\n")

print("Standard QKD (BB84):")
print("  - Works with ~1% detection efficiency")
print("  - Key rate ~ 0.5 bits/pulse (ideal)")
print("  - Requires trusted devices")

print("\nDevice-Independent QKD:")
print("  - Needs >80% detection efficiency")
print(f"  - Key rate ~ {diqkd_key_rate(2.6):.2f} bits/round (for S=2.6)")
print("  - No trust in devices required")

# Randomness certification
print("\n=== Randomness Certification ===\n")

def certified_randomness(S, n_rounds):
    """Certified random bits from n rounds of Bell test"""
    if S <= 2:
        return 0
    h_min = min_entropy_bound(S)
    return h_min * n_rounds

print("Certified random bits from Bell test:")
print(f"{'Rounds':<10} {'S=2.4':<12} {'S=2.6':<12} {'S=2.8':<12}")
print("-" * 50)

for n in [100, 1000, 10000, 100000]:
    r24 = certified_randomness(2.4, n)
    r26 = certified_randomness(2.6, n)
    r28 = certified_randomness(2.8, n)
    print(f"{n:<10} {r24:<12.1f} {r26:<12.1f} {r28:<12.1f}")

# Self-testing simulation
print("\n=== Self-Testing: State Certification ===\n")

print("From observed CHSH, we can bound state fidelity:")
print(f"{'S observed':<12} {'ε = 2√2 - S':<15} {'Max distance':<15}")
print("-" * 45)

for S in [2.82, 2.80, 2.75, 2.70, 2.60]:
    eps = 2*np.sqrt(2) - S
    dist = self_testing_bound(S)
    print(f"{S:<12.2f} {eps:<15.4f} {dist:<15.4f}")

# Protocol sketch
print("\n=== DIQKD Protocol Sketch ===")
print("""
1. PREPARATION
   - Alice and Bob establish entanglement
   - Use high-efficiency detection system

2. MEASUREMENT ROUNDS (repeat N times)
   - Alice: randomly choose x ∈ {0, 1}
   - Bob: randomly choose y ∈ {0, 1}
   - Record outcomes a, b

3. PARAMETER ESTIMATION
   - Publicly share all (x, y) settings
   - Compute CHSH S from correlations
   - Abort if S < threshold

4. KEY EXTRACTION
   - Use rounds where x = y = 0 for raw key
   - Apply error correction
   - Apply privacy amplification

5. OUTPUT
   - Secure key bits certified by Bell violation
""")

# Security analysis
print("=== Security Guarantee ===")
print(f"""
If we observe S = 2.7 over 10⁶ rounds:
- Key rate: {diqkd_key_rate(2.7):.3f} bits/round
- Total key: ~{diqkd_key_rate(2.7) * 1e6 / 1e3:.0f}k bits
- Security: Information-theoretic (against any attack!)
- Trust needed: None (black-box devices)
""")
```

---

## Summary

### Key Concepts

| Concept | Definition |
|---------|------------|
| Device-independence | Security from correlations only |
| DIQKD | QKD using Bell violation |
| Self-testing | Certifying states from statistics |
| Randomness certification | Proving unpredictability |

### Key Takeaways
1. **Bell violation** certifies genuine quantum behavior
2. **DIQKD** removes need to trust device manufacturers
3. **Self-testing** uniquely identifies quantum states
4. **Practical** DIQKD now demonstrated experimentally
5. **Trade-off:** Higher security requires higher efficiency

---

## Daily Checklist

- [ ] I understand device-independence concept
- [ ] I can explain how Bell tests certify security
- [ ] I know the key rate formula for DIQKD
- [ ] I understand self-testing of quantum states
- [ ] I appreciate practical challenges

---

*Next: Day 546 — Week Review*
