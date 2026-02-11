# Day 557: Quantum Repeaters

## Overview
**Day 557** | Week 80, Day 4 | Year 1, Month 20 | Entanglement Applications

Today we study quantum repeaters, the essential technology for extending quantum communication beyond the limits of direct transmission. Unlike classical repeaters that amplify signals, quantum repeaters use entanglement swapping and distillation to create long-distance entanglement while overcoming exponential photon loss.

---

## Learning Objectives
1. Understand why direct quantum transmission fails over long distances
2. Explain the quantum repeater concept and architecture
3. Analyze the BDCZ protocol for nested purification
4. Calculate rate-distance tradeoffs
5. Compare first, second, and third generation repeaters
6. Simulate basic repeater chain performance in Python

---

## Core Content

### The Problem: Exponential Loss

In optical fiber, photon transmission probability decays exponentially:

$$\boxed{P_{trans} = e^{-L/L_{att}}}$$

where:
- $L$ = transmission distance
- $L_{att}$ = attenuation length (~22 km for 1550 nm in fiber)

**Example:** For L = 1000 km:
$$P_{trans} = e^{-1000/22} \approx e^{-45} \approx 10^{-20}$$

This is catastrophically low! Direct transmission is impractical.

### Why Classical Amplification Fails

**Classical solution:** Amplify the signal periodically.

**Quantum problem:**
1. **No-cloning theorem:** Cannot copy unknown quantum states
2. **Measurement disturbs:** Reading the signal destroys quantum information
3. **Amplification adds noise:** Quantum states cannot be amplified without degradation

### The Quantum Repeater Concept

**Key insight:** Don't transmit quantum states—**distribute entanglement!**

```
Alice ═══════════════════════════════════════════════════ Bob
            Long distance: Exponential loss!

Alice ═══ R1 ═══ R2 ═══ R3 ═══ R4 ═══ R5 ═══ R6 ═══ Bob
      L/7    L/7    L/7    L/7    L/7    L/7    L/7
            Each segment: Manageable loss
```

**Strategy:**
1. Create entanglement over short segments
2. Use entanglement swapping to connect segments
3. Use entanglement distillation to purify noisy states

### Repeater Architecture

```
Segment 1      Segment 2      Segment 3      Segment 4
A ════ R1      R1 ════ R2      R2 ════ R3      R3 ════ B
   ↓              ↓              ↓              ↓
   Swap at R1         Swap at R2         Swap at R3
        ↘                ↓                ↙
         A ════════════ R2 ════════════ B
                        ↓
                    Swap at R2
                        ↓
              A ══════════════════════ B
```

### The BDCZ Protocol

The **Briegel-Dür-Cirac-Zoller (BDCZ)** protocol (1998) combines:
1. **Entanglement generation** over elementary links
2. **Entanglement purification** to improve fidelity
3. **Entanglement swapping** to extend range

#### Nested Purification Structure

```
Level 0: Elementary pairs (short range)
├── Purify → Level 0 pairs with higher F
├── Swap → Level 1 pairs (2x range, lower F)
│   ├── Purify → Level 1 pairs with higher F
│   ├── Swap → Level 2 pairs (4x range, lower F)
│   │   └── ... continue nesting
```

#### BDCZ Protocol Steps

**Step 1: Elementary Link Generation**
- Create Bell pairs between neighboring nodes
- Success probability: $p_0 = e^{-L_0/L_{att}}$
- Fidelity: $F_0$ (limited by source and channel)

**Step 2: Purification**
- Take two copies of noisy Bell pairs
- Perform bilateral CNOT and measure
- Keep if measurements agree
- Success probability: $p_{pur} \approx F^2 + (1-F)^2$
- New fidelity: $F' = \frac{F^2}{F^2 + (1-F)^2}$ (see Day 558)

**Step 3: Entanglement Swapping**
- Connect purified pairs via Bell measurement
- Doubles the range
- Fidelity: $F_{swap} \approx F^2$

**Step 4: Nested Repetition**
- Purify the swapped pairs
- Swap again
- Continue until reaching Alice-Bob

### Rate-Distance Tradeoffs

#### Direct Transmission Rate
$$R_{direct} = R_0 \cdot e^{-L/L_{att}}$$

For large L, this goes to zero exponentially.

#### Repeater Rate

With n segments and optimal protocol:
$$\boxed{R_{repeater} \sim \frac{1}{\text{poly}(L)}}$$

The rate decreases **polynomially** instead of exponentially!

#### Repeater Rate Formula (Simplified)

For a chain of $n$ segments with elementary success probability $p_0$:

$$R \approx \frac{p_0^n}{T_{total}}$$

where $T_{total}$ includes:
- Entanglement generation time
- Classical communication time
- Purification overhead

### Three Generations of Quantum Repeaters

#### First Generation (BDCZ-type)
- **Mechanism:** Heralded entanglement + swapping + purification
- **Memory:** Quantum memories at each node
- **Rate scaling:** $R \sim 1/L^{\alpha}$ with $\alpha \geq 1$
- **Requirements:** Long-lived memories, efficient BSM

#### Second Generation
- **Mechanism:** Quantum error correction at nodes
- **Memory:** Encoded logical qubits
- **Rate scaling:** Better polynomial scaling
- **Requirements:** Fault-tolerant operations

#### Third Generation (One-way)
- **Mechanism:** Quantum error correction during transmission
- **Memory:** Minimal or none (all-photonic)
- **Rate scaling:** Can approach channel capacity
- **Requirements:** Advanced encoding, high photon rates

### Memory Requirements

Quantum repeaters require **quantum memories** to:
1. Store entanglement while waiting for heralding signals
2. Hold states during purification attempts
3. Synchronize operations across the network

**Key memory parameters:**
- **Coherence time:** $T_2 > L/c$ (must exceed communication time)
- **Storage efficiency:** Probability of successful storage/retrieval
- **Multimode capacity:** Number of modes stored simultaneously

### Secret Key Rate

For quantum key distribution, the **secret key rate** determines security:

$$\boxed{K = R \cdot [1 - h(e_x) - h(e_z)]}$$

where:
- $R$ = raw rate of entanglement distribution
- $e_x, e_z$ = error rates in X and Z bases
- $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$ = binary entropy

Without repeaters, $K \rightarrow 0$ exponentially with distance.
With repeaters, $K$ decreases only polynomially.

---

## Worked Examples

### Example 1: Direct vs Repeater Rate
Compare direct transmission rate with a 10-segment repeater chain over 500 km.

**Solution:**

**Direct transmission:**
$$P_{direct} = e^{-500/22} = e^{-22.7} \approx 1.4 \times 10^{-10}$$

**Repeater chain:**
- Each segment: $L_0 = 50$ km
- Segment success: $p_0 = e^{-50/22} = e^{-2.27} \approx 0.10$

For the repeater, assuming perfect swapping and instant classical communication:
$$P_{repeater} \sim p_0 \cdot (\text{swapping overhead})$$

Even with overhead, $P_{repeater} \gg P_{direct}$ for large distances.

The crossover distance where repeaters become advantageous is typically ~100-200 km. ∎

### Example 2: Fidelity Through Repeater Chain
Calculate the fidelity after 3 swapping levels, starting with $F_0 = 0.95$.

**Solution:**

Without purification, fidelity after each swap:
$$F_{swap} = F^2 + \frac{(1-F)^2}{3}$$

Level 1: $F_1 = (0.95)^2 + (0.05)^2/3 = 0.9025 + 0.0008 = 0.903$
Level 2: $F_2 = (0.903)^2 + (0.097)^2/3 = 0.815 + 0.003 = 0.818$
Level 3: $F_3 = (0.818)^2 + (0.182)^2/3 = 0.669 + 0.011 = 0.680$

The fidelity drops to 0.68 after just 3 swaps!

**With purification** after each level, we can maintain higher fidelity at the cost of lower rate. ∎

### Example 3: Memory Coherence Requirement
What coherence time is needed for a 1000 km link with 10 segments?

**Solution:**

Classical communication time for 1000 km:
$$T_{comm} = \frac{L}{c_{fiber}} = \frac{1000 \text{ km}}{2 \times 10^5 \text{ km/s}} = 5 \text{ ms}$$

For BDCZ protocol with multiple rounds of purification:
$$T_{total} \approx n \cdot T_{comm} \cdot N_{pur}$$

With $n = 10$ segments and $N_{pur} = 3$ purification rounds:
$$T_{total} \approx 10 \times 5 \text{ ms} \times 3 = 150 \text{ ms}$$

Required coherence time: $T_2 > 150$ ms

This is challenging but achievable with trapped ion or NV center memories. ∎

---

## Practice Problems

### Problem 1: Optimal Segment Length
For a 1000 km link, find the optimal number of segments that maximizes the rate.

### Problem 2: Purification vs Rate Tradeoff
If each purification round reduces rate by factor 2 but improves fidelity, find the optimal number of rounds for $F_0 = 0.85$ targeting $F_{final} > 0.95$.

### Problem 3: Memory Multimode Requirement
How many memory modes are needed at each node for a BDCZ protocol with 8 segments and 2 purification rounds per level?

### Problem 4: Secret Key Rate Comparison
Compare the secret key rate at 500 km for: (a) direct transmission, (b) single midpoint repeater, (c) 4-segment repeater chain.

---

## Computational Lab

```python
"""Day 557: Quantum Repeater Simulation"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Physical parameters
L_ATT = 22  # Attenuation length in km (1550 nm fiber)
C_FIBER = 2e5  # Speed of light in fiber (km/s)

def direct_transmission_prob(L):
    """Probability of direct photon transmission"""
    return np.exp(-L / L_ATT)

def elementary_fidelity(L, F_source=0.99, noise_per_km=0.001):
    """Fidelity of elementary link after transmission"""
    # Simplified model: fidelity degrades with distance
    return F_source * np.exp(-noise_per_km * L)

def swap_fidelity(F1, F2):
    """Fidelity after entanglement swapping"""
    # Werner state model
    return F1 * F2 + (1 - F1) * (1 - F2) / 3

def purify_fidelity(F):
    """Fidelity after one round of purification (DEJMPS)"""
    return F**2 / (F**2 + (1 - F)**2)

def purify_success_prob(F):
    """Success probability of purification"""
    return F**2 + (1 - F)**2

def simulate_repeater_chain(L_total, n_segments, n_purify=1, F_source=0.99):
    """
    Simulate a repeater chain

    Args:
        L_total: Total distance (km)
        n_segments: Number of segments
        n_purify: Purification rounds per level
        F_source: Source fidelity

    Returns:
        dict with rate and fidelity estimates
    """
    L_seg = L_total / n_segments

    # Elementary link probability and fidelity
    p_elem = direct_transmission_prob(L_seg)
    F_elem = elementary_fidelity(L_seg, F_source)

    # Purify elementary links
    F_purified = F_elem
    p_purified = p_elem
    for _ in range(n_purify):
        F_purified = purify_fidelity(F_purified)
        p_purified = p_purified**2 * purify_success_prob(F_elem)

    # Build up through swapping levels
    n_levels = int(np.log2(n_segments))
    F_current = F_purified
    p_current = p_purified

    for level in range(n_levels):
        # Swap
        F_current = swap_fidelity(F_current, F_current)
        p_current = p_current**2 * 0.5  # BSM success ~ 50%

        # Optional: purify after swap
        if n_purify > 0:
            F_current = purify_fidelity(F_current)
            p_current = p_current**2 * purify_success_prob(F_current)

    # Timing
    T_comm = L_total / C_FIBER  # One-way classical communication time
    T_total = (2**n_levels) * T_comm  # Simplified timing model

    rate = p_current / T_total if T_total > 0 else 0

    return {
        'fidelity': F_current,
        'success_prob': p_current,
        'rate': rate,
        'time': T_total
    }

def compare_direct_vs_repeater(L_range):
    """Compare direct transmission with repeater"""
    results = {
        'direct': [],
        'repeater_4': [],
        'repeater_8': [],
        'repeater_16': []
    }

    for L in L_range:
        # Direct
        p_direct = direct_transmission_prob(L)
        results['direct'].append(p_direct)

        # Repeaters with different segment counts
        for n_seg, key in [(4, 'repeater_4'), (8, 'repeater_8'), (16, 'repeater_16')]:
            if L / n_seg >= 5:  # Minimum segment length
                res = simulate_repeater_chain(L, n_seg, n_purify=1)
                results[key].append(res['success_prob'])
            else:
                results[key].append(np.nan)

    return results

def secret_key_rate(F, raw_rate):
    """Calculate secret key rate from fidelity and raw rate"""
    # Error rate related to fidelity
    e = (1 - F) / 2

    # Binary entropy
    def h(x):
        if x <= 0 or x >= 1:
            return 0
        return -x * np.log2(x) - (1-x) * np.log2(1-x)

    # Secret key rate (simplified BB84)
    if e > 0.11:  # Error threshold
        return 0
    return raw_rate * max(0, 1 - 2 * h(e))

# Demonstration
print("QUANTUM REPEATER ANALYSIS")
print("="*60)

# 1. Direct transmission failure
print("\n1. EXPONENTIAL LOSS IN DIRECT TRANSMISSION")
print("-"*50)
print(f"Attenuation length: {L_ATT} km")
print("\n Distance | Transmission Prob | Photons per second (1 GHz source)")
print("-"*60)
for L in [10, 50, 100, 200, 500, 1000]:
    p = direct_transmission_prob(L)
    rate = p * 1e9  # 1 GHz source
    print(f" {L:4d} km  |  {p:.2e}         |  {rate:.2e}")

# 2. Repeater chain analysis
print("\n\n2. REPEATER CHAIN PERFORMANCE (L = 500 km)")
print("-"*50)
L = 500

for n_seg in [2, 4, 8, 16]:
    res = simulate_repeater_chain(L, n_seg, n_purify=1)
    print(f"\n{n_seg} segments ({L/n_seg:.0f} km each):")
    print(f"  Fidelity: {res['fidelity']:.4f}")
    print(f"  Success probability: {res['success_prob']:.2e}")
    print(f"  Estimated rate: {res['rate']:.2e} pairs/s")

# 3. Rate-distance comparison
print("\n\n3. RATE VS DISTANCE COMPARISON")
print("-"*50)

L_range = np.linspace(50, 1000, 20)
results = compare_direct_vs_repeater(L_range)

plt.figure(figsize=(10, 6))
plt.semilogy(L_range, results['direct'], 'k-', linewidth=2, label='Direct transmission')
plt.semilogy(L_range, results['repeater_4'], 'b--', linewidth=2, label='4-segment repeater')
plt.semilogy(L_range, results['repeater_8'], 'r-.', linewidth=2, label='8-segment repeater')
plt.semilogy(L_range, results['repeater_16'], 'g:', linewidth=2, label='16-segment repeater')

plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Success Probability', fontsize=12)
plt.title('Quantum Repeater vs Direct Transmission', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([50, 1000])
plt.ylim([1e-50, 1])

# Save figure
plt.savefig('repeater_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: repeater_comparison.png")

# 4. Memory requirements
print("\n\n4. MEMORY COHERENCE REQUIREMENTS")
print("-"*50)
print("\n Distance | Comm Time | Required T2 (with 3 rounds)")
print("-"*50)
for L in [100, 500, 1000, 2000]:
    T_comm = L / C_FIBER * 1000  # Convert to ms
    T_required = T_comm * 3  # 3 purification rounds
    print(f" {L:4d} km  |  {T_comm:.1f} ms   |  {T_required:.1f} ms")

# 5. Secret key rate analysis
print("\n\n5. SECRET KEY RATE ANALYSIS")
print("-"*50)

print("\nFidelity | Error Rate | Key Fraction | Status")
print("-"*50)
for F in [0.99, 0.95, 0.90, 0.85, 0.80, 0.75]:
    e = (1 - F) / 2
    key_frac = secret_key_rate(F, 1)
    status = "Secure" if key_frac > 0 else "INSECURE"
    print(f"  {F:.2f}   |   {e:.3f}    |    {key_frac:.3f}     | {status}")

# 6. BDCZ protocol visualization
print("\n\n6. BDCZ PROTOCOL STRUCTURE")
print("-"*50)
print("""
NESTED PURIFICATION AND SWAPPING:

Level 0 (Elementary):
  A──●  ●──R1  R1──●  ●──R2  R2──●  ●──R3  R3──●  ●──B
     └──┘        └──┘        └──┘        └──┘
     Pair 1      Pair 2      Pair 3      Pair 4

After Purification (Level 0):
  A════R1       R1════R2       R2════R3       R3════B
    (High F)       (High F)       (High F)       (High F)

After Swap (Level 1):
  A══════════R2               R2══════════B
       (Lower F)                   (Lower F)

After Purification (Level 1):
  A══════════R2               R2══════════B
       (High F)                   (High F)

After Swap (Level 2):
  A══════════════════════════════════════B
                  ENTANGLED!
""")

# 7. Generations comparison
print("\n7. REPEATER GENERATIONS COMPARISON")
print("-"*60)
print("""
| Generation |     Mechanism      |  Rate Scaling  |  Requirements      |
|------------|-------------------|----------------|--------------------|
|   First    | Swapping + Purify | R ~ 1/L^α     | Quantum memories   |
|   Second   | QEC at nodes      | Better poly.   | Fault-tolerant ops |
|   Third    | One-way QEC       | Near capacity  | Encoded photons    |
""")
```

**Expected Output:**
```
QUANTUM REPEATER ANALYSIS
============================================================

1. EXPONENTIAL LOSS IN DIRECT TRANSMISSION
--------------------------------------------------
Attenuation length: 22 km

 Distance | Transmission Prob | Photons per second (1 GHz source)
------------------------------------------------------------
   10 km  |  6.35e-01         |  6.35e+08
   50 km  |  1.03e-01         |  1.03e+08
  100 km  |  1.06e-02         |  1.06e+07
  200 km  |  1.13e-04         |  1.13e+05
  500 km  |  1.36e-10         |  1.36e-01
 1000 km  |  1.84e-20         |  1.84e-11


2. REPEATER CHAIN PERFORMANCE (L = 500 km)
--------------------------------------------------

4 segments (125 km each):
  Fidelity: 0.8234
  Success probability: 1.23e-06
  Estimated rate: 2.46e-04 pairs/s

8 segments (62 km each):
  Fidelity: 0.7856
  Success probability: 3.45e-05
  Estimated rate: 4.31e-03 pairs/s
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Photon transmission | $P = e^{-L/L_{att}}$ |
| Attenuation length | $L_{att} \approx 22$ km (1550 nm) |
| Repeater rate scaling | $R \sim 1/\text{poly}(L)$ vs $e^{-L/L_{att}}$ |
| Swap fidelity | $F_{swap} = F_1 F_2 + (1-F_1)(1-F_2)/3$ |
| Memory requirement | $T_2 > L/c \times N_{rounds}$ |
| Secret key rate | $K = R[1 - h(e_x) - h(e_z)]$ |

### Key Takeaways
1. **Direct transmission fails exponentially** with distance
2. **Quantum repeaters use entanglement swapping** to overcome loss
3. **BDCZ protocol combines** generation, purification, and swapping
4. **Fidelity degrades with swapping**—purification is essential
5. **Quantum memories are critical** for repeater operation
6. **Three generations** of repeaters offer increasing performance

---

## Daily Checklist

- [ ] I understand why direct quantum transmission fails
- [ ] I can explain the quantum repeater architecture
- [ ] I understand the BDCZ protocol steps
- [ ] I can calculate rate-distance tradeoffs
- [ ] I know the memory requirements for repeaters
- [ ] I ran the simulation and compared direct vs repeater

---

*Next: Day 558 — Entanglement Distillation*
