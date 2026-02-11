# Day 879: Flag Fault-Tolerant Error Correction — Complete Protocols

## Month 32: Fault-Tolerant Quantum Computing II | Week 126: Flag Qubits & Syndrome Extraction

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Complete Flag-FT Protocols |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving: Lookup Tables |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 879, you will be able to:

1. Construct complete flag-FT error correction protocols
2. Build syndrome-flag lookup tables for decoding
3. Implement adaptive correction based on flag outcomes
4. Analyze protocols requiring multiple syndrome extraction rounds
5. Compare threshold behavior of flag-FT vs traditional methods
6. Handle concurrent faults during correction

---

## Core Content

### 1. The Complete Flag-FT Protocol

Flag circuits are just one component. A complete fault-tolerant error correction cycle includes:

1. **Syndrome extraction** (with flags)
2. **Classical processing** (lookup table)
3. **Correction** (conditional Pauli operations)
4. **Verification** (optional repeat)

**Protocol Structure:**

```
┌─────────────────────────────────────────────────────────┐
│                   FLAG-FT EC ROUND                       │
├─────────────────────────────────────────────────────────┤
│  1. Extract all syndromes with flag circuits             │
│  2. Collect (syndrome, flag) pairs                       │
│  3. If any flag = 1:                                     │
│       → Use flag-aware decoder                           │
│       → May require additional syndrome rounds           │
│  4. If all flags = 0:                                    │
│       → Use standard decoder                             │
│  5. Apply correction operator                            │
│  6. (Optional) Verify correction                         │
└─────────────────────────────────────────────────────────┘
```

---

### 2. Syndrome-Flag Lookup Tables

The decoder maps (syndrome, flags) to correction operators:

$$\boxed{\text{Lookup}: (\mathbf{s}, \mathbf{f}) \mapsto R}$$

where:
- $\mathbf{s}$ = syndrome vector (one bit per stabilizer)
- $\mathbf{f}$ = flag vector (one bit per flag)
- $R$ = correction Pauli operator

**Table Construction Algorithm:**

```
FOR each correctable error pattern E:
    1. Simulate E on encoded state
    2. Extract syndrome s = measure_stabilizers(E)
    3. For each possible circuit fault:
        a. Determine flag outcome f
        b. Determine actual data error E'
        c. Store: lookup[(s, f)] = E' (or equivalent)
    4. Verify all entries have unique corrections
```

**Example: [[7,1,3]] Steane Code with Flags**

The code has 6 stabilizers, so syndrome $\mathbf{s} \in \{0,1\}^6$.

With 6 flag circuits (one per stabilizer), flag vector $\mathbf{f} \in \{0,1\}^6$.

**Unflagged case ($\mathbf{f} = 000000$):**

Standard syndrome decoding - each syndrome corresponds to at most weight-1 error.

| Syndrome | Correction |
|----------|------------|
| 000000 | I (no error) |
| 100100 | $X_1$ |
| 010010 | $X_2$ |
| ... | ... |

**Flagged cases:**

When flag $f_i = 1$, the error on qubits in stabilizer $S_i$ might be weight-2.

---

### 3. Adaptive Correction Strategies

When flags trigger, we have multiple options:

**Strategy 1: Extended Lookup Table**

Pre-compute all (syndrome, flag) → correction mappings:

$$\text{Table size: } 2^{n_{\text{syndromes}} + n_{\text{flags}}}$$

For [[7,1,3]]: $2^{12} = 4096$ entries (manageable).

**Strategy 2: Two-Stage Decoding**

```
IF all flags = 0:
    correction = standard_decode(syndrome)
ELSE:
    # Flags indicate possible high-weight error
    # Run additional syndrome round with different circuit
    syndrome2 = extract_syndrome_alt()
    correction = combined_decode(syndrome, syndrome2, flags)
```

**Strategy 3: Repeat Until Unflagged**

```
REPEAT:
    (syndrome, flags) = extract_syndrome_with_flags()
    IF all flags = 0:
        BREAK
UNTIL max_rounds reached
correction = standard_decode(syndrome)
```

This works because each round with flags = 0 confirms no high-weight error occurred.

---

### 4. Multiple Round Protocols

**Why Multiple Rounds?**

A single syndrome extraction might have:
- Measurement errors (wrong syndrome readout)
- Flag errors (wrong flag outcome)
- Time-correlated errors

Multiple rounds provide redundancy.

**Standard Protocol (DiVincenzo-Aliferis):**

For distance-$d$ code, perform $d$ rounds of syndrome extraction.

**Flag-FT Protocol (Chamberland-Cross):**

```
Round 1: Extract all syndromes with flags
IF any flag triggered:
    Round 2: Extract flagged stabilizers again (possibly with different circuit)
    Combine information from both rounds
ELSE:
    Single round suffices
```

**Error Accumulation Analysis:**

Let $p$ = physical error rate per gate.

Single round failure probability: $P_1 \approx c_1 p^2$ (two faults needed)

With $r$ rounds: $P_r \approx r \cdot c_1 p^2$ (errors can accumulate)

**Optimal round count** balances fresh errors vs. syndrome reliability.

---

### 5. Flag-FT Threshold Analysis

**Threshold Definition:**

The threshold $p_{\text{th}}$ is the physical error rate below which increasing code distance reduces logical error rate.

**Flag-FT Threshold Characteristics:**

For well-designed flag circuits:

$$\boxed{p_{\text{th}}^{\text{flag}} \approx p_{\text{th}}^{\text{Shor}} \times (1 - \delta)}$$

where $\delta$ is a small penalty (typically 10-30%) due to:
- Less redundancy than cat states
- Possible decoder suboptimality

**Typical Values:**

| Method | Approximate Threshold |
|--------|----------------------|
| Shor-style | ~0.3% |
| Steane-style | ~0.1% |
| Flag-based | ~0.2% |
| Surface code (MWPM) | ~1% |

**Trade-off:**

Flag methods sacrifice some threshold for massive resource reduction.

---

### 6. Handling Concurrent Faults

**Challenge:** Multiple faults in one EC cycle.

**Fault Counting:**

For t-flag FT with distance $d = 2t + 1$:
- Up to $t$ faults should be correctable
- $t + 1$ faults may cause logical error

**Probability Analysis:**

$$P_{\text{logical}} \approx \binom{N}{t+1} p^{t+1}$$

where $N$ = number of fault locations in EC cycle.

**For flag circuits:**

$N = O(w \cdot n_{\text{stab}})$ (much smaller than Shor-style)

This actually *improves* the prefactor despite similar threshold.

---

### 7. Complete Protocol Example: [[7,1,3]] Steane Code

**Code Parameters:**
- 7 data qubits
- 6 stabilizers (3 X-type, 3 Z-type)
- Distance 3 (corrects 1 error)

**Flag-FT Protocol:**

**Step 1: Prepare flag circuits**

6 flag circuits, one per stabilizer. Each uses 2 ancillas (1 syndrome + 1 flag).

**Step 2: First extraction round**

```python
syndromes = []
flags = []
for stabilizer in all_stabilizers:
    s, f = extract_with_flag(stabilizer)
    syndromes.append(s)
    flags.append(f)
```

**Step 3: Decode based on flags**

```python
if all(f == 0 for f in flags):
    # Standard decoding
    correction = standard_decoder(syndromes)
else:
    # Flag-aware decoding
    flagged_stabs = [i for i, f in enumerate(flags) if f == 1]

    # Option A: Use extended lookup table
    correction = flag_lookup_table[(syndromes, flags)]

    # Option B: Second round for flagged stabilizers
    for i in flagged_stabs:
        s2, f2 = extract_with_flag_alt(stabilizers[i])
        syndromes[i] = majority(syndromes[i], s2)
    correction = standard_decoder(syndromes)
```

**Step 4: Apply correction**

```python
apply_pauli(correction, data_qubits)
```

**Step 5: (Optional) Verify**

Extract syndrome once more; should be trivial if correction worked.

---

### 8. Lookup Table Construction Details

**For [[7,1,3]] with X-error correction:**

X errors produce Z-syndrome patterns.

**Weight-0 (no error):**
- Syndrome: 000
- Flag: 000
- Correction: I

**Weight-1 errors (7 cases):**

| Error | Z-Syndrome | Flag | Correction |
|-------|------------|------|------------|
| $X_1$ | 110 | 000 | $X_1$ |
| $X_2$ | 011 | 000 | $X_2$ |
| $X_3$ | 101 | 000 | $X_3$ |
| $X_4$ | 111 | 000 | $X_4$ |
| $X_5$ | 100 | 000 | $X_5$ |
| $X_6$ | 010 | 000 | $X_6$ |
| $X_7$ | 001 | 000 | $X_7$ |

**Circuit faults causing weight-2 errors:**

| Fault | Syndrome | Flag | Actual Error | Correction |
|-------|----------|------|--------------|------------|
| X on syndrome of $S_1$ before flag | 100 | 100 | $X_1 X_3$ | Defer to 2nd round |
| ... | ... | ... | ... | ... |

**Key Insight:**

The same syndrome with different flags corresponds to different errors:
- Syndrome 100, Flag 000: Error $X_5$ (weight-1)
- Syndrome 100, Flag 100: Error $X_1 X_3$ or $X_1 X_5$ (weight-2, from fault)

The flag disambiguates!

---

## Practical Applications

### Real Device Implementation

**IBM Quantum (2024-2025):**

Flag circuits have been demonstrated on IBM hardware:
- Heavy-hex connectivity supports flag patterns
- Mid-circuit measurement enables flag readout
- Real-time classical processing for conditional correction

**Key Challenges:**
1. Measurement errors (2-5% on current hardware)
2. Crosstalk during parallel operations
3. Limited connectivity requires SWAP overhead

### Comparison with Surface Code

| Aspect | Flag-FT (Steane) | Surface Code |
|--------|------------------|--------------|
| Threshold | ~0.2% | ~1% |
| Qubits for d=3 | 9 (7+2) | 17 (d² + ancillas) |
| Transversal gates | H, S, CNOT | None |
| Decoder complexity | Lookup table | MWPM algorithm |

**When to use flags:**

- Near-term devices with limited qubits
- Applications requiring transversal gates
- Moderate error rates (~0.1%)

---

## Worked Examples

### Example 1: Complete Lookup Table for [[5,1,3]] Code

**Problem:** Construct the syndrome-flag lookup table for X-error correction on the [[5,1,3]] perfect code.

**Solution:**

The [[5,1,3]] code has 4 stabilizers. For X-errors, we measure Z-type syndromes.

**Stabilizers (Z-parts):**
- $S_1 = XZZXI$ → Z syndrome from $Z_2 Z_3$
- $S_2 = IXZZX$ → Z syndrome from $Z_3 Z_4$
- $S_3 = XIXZZ$ → Z syndrome from $Z_4 Z_5$
- $S_4 = ZXIXZ$ → Z syndrome from $Z_1 Z_5$

Wait, the [[5,1,3]] code has mixed stabilizers. Let me reconsider.

The stabilizers are: $S_1 = XZZXI$, $S_2 = IXZZX$, $S_3 = XIXZZ$, $S_4 = ZXIXZ$.

For X-errors on data qubits, the syndrome bits are:
- $s_i = 1$ if X anticommutes with $S_i$

**Single X errors:**

| Error | Anticommutes with | Syndrome |
|-------|-------------------|----------|
| $X_1$ | $S_1, S_4$ | 1001 |
| $X_2$ | $S_1, S_2$ | 1100 |
| $X_3$ | $S_2, S_3$ | 0110 |
| $X_4$ | $S_3, S_4$ | 0011 |
| $X_5$ | $S_1, S_3$ | 1010 |

Each single X error has unique syndrome → standard decoding works.

**Flag-triggered cases:**

With flags, circuit faults causing $X_i X_j$ (weight-2) will trigger the flag on the relevant stabilizer.

The lookup table entries:

| Syndrome | Flag | Correction | Notes |
|----------|------|------------|-------|
| 0000 | 0000 | I | No error |
| 1001 | 0000 | $X_1$ | Standard |
| 1100 | 0000 | $X_2$ | Standard |
| 0110 | 0000 | $X_3$ | Standard |
| 0011 | 0000 | $X_4$ | Standard |
| 1010 | 0000 | $X_5$ | Standard |
| 0101 | 1000 | $X_2 X_5$ | Flag on $S_1$ |
| ... | ... | ... | ... |

**Result:** Complete table has ~32 entries for distance-3. $\square$

---

### Example 2: Two-Round Protocol Analysis

**Problem:** Analyze a two-round flag-FT protocol. What is the logical error probability as a function of physical error rate $p$?

**Solution:**

**Round 1:**
- Extract syndromes with flags
- Probability of single fault: $\approx N_1 p$ where $N_1$ = fault locations
- If flag = 1, proceed to round 2

**Round 2 (if flagged):**
- Extract again with alternative circuit
- Fresh faults possible: probability $\approx N_2 p$

**Logical error occurs when:**
1. Two faults in round 1 create weight ≥ 2 error undetected, OR
2. One fault in round 1 + one in round 2 combine badly, OR
3. Data error + circuit fault combine to logical error

**Leading order:**

$$P_L \approx A \cdot p^2$$

where $A$ depends on:
- Number of fault locations: $A \propto N^2$
- Decoder effectiveness
- Flag circuit design

**For [[7,1,3]] with flag-FT:**

Typical values: $A \approx 50-200$

$$P_L \approx 100 \cdot p^2$$

At $p = 10^{-3}$: $P_L \approx 10^{-4}$ (logical error rate improved by 10×)

**Threshold estimate:**

Set $P_L < p$ (encoding should help):
$$100 p^2 < p \Rightarrow p < 10^{-2}$$

So threshold $p_{\text{th}} \approx 1\%$ for this simple model.

**Result:** Flag-FT achieves fault tolerance with threshold around 0.5-1%. $\square$

---

### Example 3: Decoder Decision Tree

**Problem:** Implement a decision-tree decoder for the [[7,1,3]] code with flags.

**Solution:**

```
FUNCTION flag_decode(syndrome, flags):
    x_syndrome = syndrome[0:3]  # From Z stabilizers
    z_syndrome = syndrome[3:6]  # From X stabilizers
    x_flags = flags[0:3]
    z_flags = flags[3:6]

    correction = I

    # Handle X errors (detected by Z stabilizers)
    IF all(x_flags == 0):
        # Standard X error decoding
        correction *= standard_x_decode(x_syndrome)
    ELSE:
        # Flag indicates possible weight-2 X error
        # Use combined syndrome-flag lookup
        correction *= flag_x_decode(x_syndrome, x_flags)

    # Handle Z errors (detected by X stabilizers)
    IF all(z_flags == 0):
        correction *= standard_z_decode(z_syndrome)
    ELSE:
        correction *= flag_z_decode(z_syndrome, z_flags)

    RETURN correction
```

**Standard decoder subroutine:**

```
FUNCTION standard_x_decode(syndrome):
    table = {
        (0,0,0): I,
        (1,1,0): X_1,
        (0,1,1): X_2,
        (1,0,1): X_3,
        (1,1,1): X_4,
        (1,0,0): X_5,
        (0,1,0): X_6,
        (0,0,1): X_7
    }
    RETURN table[syndrome]
```

**Flag-aware decoder subroutine:**

```
FUNCTION flag_x_decode(syndrome, flags):
    # Extended table including flag information
    IF flags == (1,0,0):  # Flag on first Z-stabilizer
        # Possible weight-2 errors involving qubits 1,3,5,7
        RETURN consult_extended_table(syndrome, flags)
    ELIF flags == (0,1,0):
        # Possible weight-2 errors involving qubits 2,3,6,7
        ...
    ELSE:
        # Multiple flags - very rare, conservative decode
        RETURN standard_x_decode(syndrome)
```

**Result:** Decision tree provides efficient real-time decoding. $\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Lookup table:** Complete the X-error lookup table for [[7,1,3]] with flags = 000.

2. **Protocol flow:** Trace through the flag-FT protocol for a single X error on qubit 3 of the Steane code.

3. **Round counting:** For a distance-5 code, how many syndrome rounds does the standard DiVincenzo-Aliferis protocol require?

### Level 2: Intermediate

4. **Extended table:** Add entries to the [[7,1,3]] lookup table for cases where exactly one flag is triggered.

5. **Threshold estimation:** If a flag-FT protocol has $N = 50$ fault locations and requires 2 faults for logical error, estimate the threshold.

6. **Concurrent faults:** Analyze what happens when two X faults occur in the same syndrome extraction circuit.

### Level 3: Challenging

7. **Prove correctness:** Show that the two-round flag-FT protocol for [[7,1,3]] correctly handles all single faults.

8. **Optimal rounds:** Derive the optimal number of syndrome extraction rounds as a function of measurement error rate $p_m$ and gate error rate $p_g$.

9. **Decoder design:** Design a decoder for a hypothetical [[15,1,5]] code with flags. How large is the lookup table?

---

## Computational Lab

### Objective
Implement complete flag-FT error correction protocol with lookup tables.

```python
"""
Day 879 Computational Lab: Flag-FT Error Correction Protocols
Week 126: Flag Qubits & Syndrome Extraction
"""

import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Steane Code Definition
# =============================================================================

print("=" * 70)
print("Part 1: [[7,1,3]] Steane Code Setup")
print("=" * 70)

class SteaneCode:
    """Implementation of the [[7,1,3]] Steane code."""

    def __init__(self):
        self.n = 7  # Physical qubits
        self.k = 1  # Logical qubits
        self.d = 3  # Distance

        # X-type stabilizers (detect Z errors)
        self.x_stabilizers = [
            [0, 2, 4, 6],  # X on qubits 1,3,5,7 (0-indexed)
            [1, 2, 5, 6],  # X on qubits 2,3,6,7
            [3, 4, 5, 6],  # X on qubits 4,5,6,7
        ]

        # Z-type stabilizers (detect X errors)
        self.z_stabilizers = [
            [0, 2, 4, 6],  # Z on qubits 1,3,5,7
            [1, 2, 5, 6],  # Z on qubits 2,3,6,7
            [3, 4, 5, 6],  # Z on qubits 4,5,6,7
        ]

    def syndrome_from_x_error(self, error_qubits):
        """Compute syndrome from X error(s)."""
        syndrome = []
        for stab in self.z_stabilizers:
            # X anticommutes with Z if they share odd # of qubits
            overlap = len(set(error_qubits) & set(stab))
            syndrome.append(overlap % 2)
        return tuple(syndrome)

    def syndrome_from_z_error(self, error_qubits):
        """Compute syndrome from Z error(s)."""
        syndrome = []
        for stab in self.x_stabilizers:
            overlap = len(set(error_qubits) & set(stab))
            syndrome.append(overlap % 2)
        return tuple(syndrome)

code = SteaneCode()

print("\nSteane Code Stabilizers:")
print("X-type (detect Z errors):", code.x_stabilizers)
print("Z-type (detect X errors):", code.z_stabilizers)

# =============================================================================
# Part 2: Standard Syndrome Lookup Table
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Standard Lookup Table (No Flags)")
print("=" * 70)

def build_standard_lookup(code):
    """Build standard syndrome → correction lookup table."""
    x_table = {(0, 0, 0): []}  # No error
    z_table = {(0, 0, 0): []}

    # Single X errors
    for q in range(code.n):
        syndrome = code.syndrome_from_x_error([q])
        x_table[syndrome] = [q]

    # Single Z errors
    for q in range(code.n):
        syndrome = code.syndrome_from_z_error([q])
        z_table[syndrome] = [q]

    return x_table, z_table

x_lookup, z_lookup = build_standard_lookup(code)

print("\nX-error lookup table (syndrome → correction):")
print("-" * 40)
for syndrome, correction in sorted(x_lookup.items()):
    corr_str = f"X_{correction[0]+1}" if correction else "I"
    print(f"  {syndrome} → {corr_str}")

# =============================================================================
# Part 3: Flag-Aware Lookup Table
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Flag-Aware Lookup Table")
print("=" * 70)

def build_flag_lookup(code):
    """Build extended lookup table including flag information."""
    # Structure: (syndrome, flags) → correction

    flag_table = {}

    # No flags case - use standard table
    for q in range(code.n + 1):  # 0 to 7 (7 = no error)
        if q < code.n:
            syndrome = code.syndrome_from_x_error([q])
            flag_table[(syndrome, (0, 0, 0))] = [q]
        else:
            flag_table[((0, 0, 0), (0, 0, 0))] = []

    # Flagged cases - from circuit faults
    # When flag i is triggered, error might be weight-2 on stabilizer i's qubits

    for stab_idx in range(3):
        stab_qubits = code.z_stabilizers[stab_idx]

        # Possible weight-2 errors from single fault on this stabilizer circuit
        for q1, q2 in combinations(stab_qubits, 2):
            syndrome = code.syndrome_from_x_error([q1, q2])
            flag = [0, 0, 0]
            flag[stab_idx] = 1
            flag = tuple(flag)

            # Store the correction (apply X on both qubits)
            # But we need to be careful - might collide with weight-1 error
            key = (syndrome, flag)
            if key not in flag_table:
                flag_table[key] = [q1, q2]
            else:
                # Collision - need to distinguish by additional syndrome round
                flag_table[key] = ('ambiguous', [q1, q2], flag_table[key])

    return flag_table

flag_lookup = build_flag_lookup(code)

print("\nFlag-aware lookup table:")
print("-" * 60)
print(f"{'Syndrome':<15} {'Flag':<15} {'Correction'}")
print("-" * 60)

for (syndrome, flag), correction in sorted(flag_lookup.items()):
    if isinstance(correction, tuple):
        print(f"{str(syndrome):<15} {str(flag):<15} AMBIGUOUS")
    else:
        corr_str = "I" if not correction else "X_" + "_".join(str(q+1) for q in correction)
        print(f"{str(syndrome):<15} {str(flag):<15} {corr_str}")

# =============================================================================
# Part 4: Complete Flag-FT Protocol
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Complete Flag-FT Protocol")
print("=" * 70)

class FlagFTProtocol:
    """Complete flag-FT error correction protocol."""

    def __init__(self, code):
        self.code = code
        self.x_lookup, self.z_lookup = build_standard_lookup(code)
        self.flag_lookup = build_flag_lookup(code)

    def extract_syndrome_with_flags(self, data_state, error_rate=0.01):
        """
        Simulate syndrome extraction with flag circuits.

        Args:
            data_state: Dict with 'x_errors' and 'z_errors' (lists of qubit indices)
            error_rate: Probability of fault during extraction

        Returns:
            (x_syndrome, z_syndrome, x_flags, z_flags)
        """
        # Get ideal syndrome
        x_syndrome = list(self.code.syndrome_from_x_error(data_state.get('x_errors', [])))
        z_syndrome = list(self.code.syndrome_from_z_error(data_state.get('z_errors', [])))

        # Simulate extraction faults
        x_flags = [0, 0, 0]
        z_flags = [0, 0, 0]

        for i in range(3):
            # Random fault on X-stabilizer circuit
            if np.random.random() < error_rate:
                z_flags[i] = 1
                # Fault might flip syndrome or cause data error

            # Random fault on Z-stabilizer circuit
            if np.random.random() < error_rate:
                x_flags[i] = 1

        return tuple(x_syndrome), tuple(z_syndrome), tuple(x_flags), tuple(z_flags)

    def decode(self, x_syndrome, z_syndrome, x_flags, z_flags):
        """
        Decode syndrome and flags to correction.

        Returns:
            Dict with 'x_correction' and 'z_correction'
        """
        correction = {'x_correction': [], 'z_correction': []}

        # X error correction (from Z stabilizer syndrome)
        if all(f == 0 for f in x_flags):
            # Standard decoding
            if x_syndrome in self.x_lookup:
                correction['x_correction'] = self.x_lookup[x_syndrome]
        else:
            # Flag-aware decoding
            key = (x_syndrome, x_flags)
            if key in self.flag_lookup:
                result = self.flag_lookup[key]
                if isinstance(result, tuple) and result[0] == 'ambiguous':
                    # Need additional round - for now, use first option
                    correction['x_correction'] = result[1]
                else:
                    correction['x_correction'] = result

        # Z error correction (from X stabilizer syndrome)
        if all(f == 0 for f in z_flags):
            if z_syndrome in self.z_lookup:
                correction['z_correction'] = self.z_lookup[z_syndrome]

        return correction

    def run_ec_cycle(self, data_errors, verbose=False):
        """
        Run one complete EC cycle.

        Args:
            data_errors: Dict with 'x_errors' and 'z_errors'
            verbose: Print debug info

        Returns:
            (success, residual_errors)
        """
        # Extract syndrome
        x_syn, z_syn, x_flags, z_flags = self.extract_syndrome_with_flags(data_errors)

        if verbose:
            print(f"  X syndrome: {x_syn}, X flags: {x_flags}")
            print(f"  Z syndrome: {z_syn}, Z flags: {z_flags}")

        # Decode
        correction = self.decode(x_syn, z_syn, x_flags, z_flags)

        if verbose:
            print(f"  Correction: {correction}")

        # Apply correction (in simulation, XOR with existing errors)
        residual_x = set(data_errors.get('x_errors', [])) ^ set(correction['x_correction'])
        residual_z = set(data_errors.get('z_errors', [])) ^ set(correction['z_correction'])

        # Check if residual is correctable (weight ≤ 1) or logical error
        success = len(residual_x) <= 1 and len(residual_z) <= 1

        return success, {'x_errors': list(residual_x), 'z_errors': list(residual_z)}

# Test the protocol
protocol = FlagFTProtocol(code)

print("\nTesting EC cycle with single X error on qubit 3:")
data_errors = {'x_errors': [2], 'z_errors': []}  # 0-indexed
success, residual = protocol.run_ec_cycle(data_errors, verbose=True)
print(f"  Success: {success}, Residual: {residual}")

print("\nTesting EC cycle with no error:")
data_errors = {'x_errors': [], 'z_errors': []}
success, residual = protocol.run_ec_cycle(data_errors, verbose=True)
print(f"  Success: {success}, Residual: {residual}")

# =============================================================================
# Part 5: Threshold Simulation
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Threshold Simulation")
print("=" * 70)

def simulate_logical_error_rate(code, error_rates, n_trials=1000):
    """Simulate logical error rate as function of physical error rate."""
    protocol = FlagFTProtocol(code)
    results = []

    for p in error_rates:
        successes = 0

        for _ in range(n_trials):
            # Generate random single-qubit error
            data_errors = {'x_errors': [], 'z_errors': []}

            for q in range(code.n):
                if np.random.random() < p:
                    if np.random.random() < 0.5:
                        data_errors['x_errors'].append(q)
                    else:
                        data_errors['z_errors'].append(q)

            # Run EC cycle
            success, _ = protocol.run_ec_cycle(data_errors)
            if success:
                successes += 1

        logical_error_rate = 1 - successes / n_trials
        results.append(logical_error_rate)

    return results

print("\nSimulating logical error rate vs physical error rate...")
error_rates = np.linspace(0.001, 0.15, 20)
logical_rates = simulate_logical_error_rate(code, error_rates, n_trials=500)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(error_rates, logical_rates, 'bo-', label='Flag-FT [[7,1,3]]', linewidth=2)
ax.semilogy(error_rates, error_rates, 'k--', label='No encoding (p_L = p)', linewidth=1)

# Find approximate threshold
for i in range(len(error_rates) - 1):
    if logical_rates[i] < error_rates[i] and logical_rates[i+1] >= error_rates[i+1]:
        threshold = (error_rates[i] + error_rates[i+1]) / 2
        ax.axvline(x=threshold, color='red', linestyle=':', label=f'Threshold ≈ {threshold:.3f}')
        break

ax.set_xlabel('Physical Error Rate p')
ax.set_ylabel('Logical Error Rate p_L')
ax.set_title('Flag-FT Error Correction Threshold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('day_879_threshold.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_879_threshold.png'")

# =============================================================================
# Part 6: Multi-Round Protocol
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Multi-Round Protocol")
print("=" * 70)

class MultiRoundProtocol:
    """Flag-FT protocol with multiple syndrome rounds."""

    def __init__(self, code, max_rounds=3):
        self.code = code
        self.max_rounds = max_rounds
        self.x_lookup, self.z_lookup = build_standard_lookup(code)

    def run_multi_round(self, data_errors, verbose=False):
        """Run multi-round EC with flag-based abort."""
        syndromes = []
        all_unflagged = False

        for round_num in range(self.max_rounds):
            # Extract syndrome
            x_syn = self.code.syndrome_from_x_error(data_errors.get('x_errors', []))
            z_syn = self.code.syndrome_from_z_error(data_errors.get('z_errors', []))

            # Simulate flags (random for now)
            x_flags = tuple(np.random.random(3) < 0.1)
            z_flags = tuple(np.random.random(3) < 0.1)

            syndromes.append({
                'x_syndrome': x_syn,
                'z_syndrome': z_syn,
                'x_flags': x_flags,
                'z_flags': z_flags
            })

            if verbose:
                print(f"  Round {round_num + 1}: x_syn={x_syn}, x_flags={x_flags}")

            # Check if unflagged
            if all(f == 0 for f in x_flags) and all(f == 0 for f in z_flags):
                all_unflagged = True
                break

        # Decode using last unflagged round or majority vote
        if all_unflagged:
            final_x_syn = syndromes[-1]['x_syndrome']
            final_z_syn = syndromes[-1]['z_syndrome']
        else:
            # Majority vote across rounds
            final_x_syn = tuple(
                1 if sum(s['x_syndrome'][i] for s in syndromes) > len(syndromes) / 2 else 0
                for i in range(3)
            )
            final_z_syn = tuple(
                1 if sum(s['z_syndrome'][i] for s in syndromes) > len(syndromes) / 2 else 0
                for i in range(3)
            )

        # Apply correction
        x_correction = self.x_lookup.get(final_x_syn, [])
        z_correction = self.z_lookup.get(final_z_syn, [])

        return x_correction, z_correction, len(syndromes)

# Test multi-round
multi_protocol = MultiRoundProtocol(code)
print("\nTesting multi-round protocol:")
x_corr, z_corr, n_rounds = multi_protocol.run_multi_round(
    {'x_errors': [3], 'z_errors': []}, verbose=True
)
print(f"  Correction: X on {x_corr}, Z on {z_corr}")
print(f"  Rounds used: {n_rounds}")

# =============================================================================
# Part 7: Resource Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Resource Analysis")
print("=" * 70)

def analyze_resources(code, method='flag'):
    """Analyze resource requirements for different methods."""
    n = code.n
    n_stab = 2 * 3  # X and Z stabilizers for Steane code

    if method == 'shor':
        # Shor-style: w ancillas per stabilizer (w=4 for Steane)
        ancillas_per_stab = 4 + 2  # cat state + verification
        total_ancillas = n_stab * ancillas_per_stab
        circuit_depth = 4 * 3  # Per stabilizer, sequential

    elif method == 'steane':
        # Steane-style: full encoded ancilla block
        total_ancillas = n  # One encoded block
        circuit_depth = 3  # Transversal + decode

    elif method == 'flag':
        # Flag-style: 2 ancillas per stabilizer
        total_ancillas = n_stab * 2
        circuit_depth = 4 + 2  # CNOT chain + flag connections

    return {
        'ancillas': total_ancillas,
        'depth': circuit_depth,
        'data_qubits': n,
        'total_qubits': n + total_ancillas
    }

print("\nResource comparison for [[7,1,3]] Steane code:")
print("-" * 60)
print(f"{'Method':<15} {'Ancillas':<12} {'Depth':<10} {'Total Qubits'}")
print("-" * 60)

for method in ['shor', 'steane', 'flag']:
    resources = analyze_resources(code, method)
    print(f"{method:<15} {resources['ancillas']:<12} {resources['depth']:<10} {resources['total_qubits']}")

# =============================================================================
# Part 8: Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary: Flag-FT Error Correction Protocols")
print("=" * 70)

print("""
Complete Protocol Components:

1. SYNDROME EXTRACTION: Flag circuits for each stabilizer
   - 2 ancillas per stabilizer (syndrome + flag)
   - Mid-circuit measurement of flags

2. LOOKUP TABLE: Maps (syndrome, flags) → correction
   - Standard entries for flag = 0
   - Extended entries for flag = 1

3. DECODING STRATEGY:
   - All flags 0: Standard decoder
   - Any flag 1: Flag-aware decoder or additional round

4. MULTI-ROUND OPTION:
   - Repeat until unflagged round
   - Majority vote across rounds
   - Handles measurement errors

5. THRESHOLD: ~0.2-0.5% for typical flag-FT implementations
   - Below Shor-style but acceptable
   - Compensated by huge resource savings

Key Insight: Flag-FT trades some threshold for dramatic resource reduction.
""")

print("=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Lookup table | $(\mathbf{s}, \mathbf{f}) \mapsto R$ |
| Logical error rate | $P_L \approx A \cdot p^{t+1}$ for t-correcting code |
| Threshold condition | $P_L < p$ for encoding to help |
| Multi-round majority | $\text{syndrome} = \text{majority}(s_1, s_2, \ldots, s_r)$ |

### Main Takeaways

1. **Complete protocol** = extraction + lookup + correction + (optional) verification
2. **Lookup tables** map (syndrome, flag) pairs to corrections
3. **Adaptive decoding** uses different strategies based on flag outcomes
4. **Multiple rounds** provide redundancy against measurement errors
5. **Threshold** around 0.2-0.5% with significant resource savings

---

## Daily Checklist

- [ ] Describe the complete flag-FT EC protocol
- [ ] Build a lookup table for a simple code
- [ ] Explain when multiple rounds are needed
- [ ] Calculate logical error rate for given physical rate
- [ ] Complete Level 1 practice problems
- [ ] Run computational lab simulation
- [ ] Compare flag-FT threshold to other methods

---

## Preview: Day 880

Tomorrow we apply flag techniques to specific codes: the Steane [[7,1,3]] code, surface codes, and color codes. Each code family has unique features that affect flag circuit design.

---

*"The art of fault tolerance is in the details of the protocol."*

---

**Next:** [Day_880_Friday.md](Day_880_Friday.md) — Flags on Various Codes
