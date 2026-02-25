# Day 659: Classical Error Correction Review

## Week 95: Error Detection/Correction Intro | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Review** classical error correction fundamentals
2. **Understand** parity checks and syndrome decoding
3. **Analyze** repetition codes and Hamming codes
4. **Prepare** for quantum generalizations

---

## Core Content

### 1. The Error Correction Problem

**Goal:** Transmit information reliably through a noisy channel.

**Key Idea:** Add redundancy so errors can be detected and corrected.

**Trade-off:** Rate (information per transmitted bit) vs reliability.

### 2. The Repetition Code

**Simplest error correcting code:** Repeat each bit 3 times.

**Encoding:**
$$0 \to 000, \quad 1 \to 111$$

**Decoding:** Majority vote
- $000, 001, 010, 100 \to 0$
- $111, 110, 101, 011 \to 1$

**Can correct:** Any single bit flip
**Cannot correct:** Two or more bit flips

**Rate:** $R = 1/3$ (1 information bit per 3 physical bits)

### 3. Parity Check Matrix

The repetition code is defined by parity checks:
$$H = \begin{pmatrix}1 & 1 & 0\\1 & 0 & 1\end{pmatrix}$$

**Codewords** satisfy $Hc = 0$ (mod 2):
- $H \cdot (0,0,0)^T = (0,0)^T$ ✓
- $H \cdot (1,1,1)^T = (0,0)^T$ ✓

### 4. Syndrome Decoding

**Syndrome:** $s = He$ where $e$ is the error pattern.

| Error | Syndrome | Correction |
|-------|----------|------------|
| None | $(0,0)$ | None |
| Bit 1 | $(1,1)$ | Flip bit 1 |
| Bit 2 | $(1,0)$ | Flip bit 2 |
| Bit 3 | $(0,1)$ | Flip bit 3 |

**Key property:** Syndrome identifies the error without revealing the codeword!

### 5. Hamming Codes

**[7,4,3] Hamming code:**
- 7 bits total, 4 information bits, distance 3
- Corrects any single bit error
- Rate $R = 4/7 \approx 0.57$

**Parity check matrix:**
$$H = \begin{pmatrix}1&1&1&0&1&0&0\\1&1&0&1&0&1&0\\1&0&1&1&0&0&1\end{pmatrix}$$

### 6. Code Parameters [n, k, d]

- **n:** Block length (total bits)
- **k:** Information bits
- **d:** Minimum distance (minimum Hamming weight of nonzero codeword)

**Error correction capability:** Can correct $t = \lfloor(d-1)/2\rfloor$ errors.

**Singleton bound:** $k \leq n - d + 1$

### 7. Linear Codes

A **linear code** is a subspace of $\mathbb{F}_2^n$.

**Generator matrix G:** Codewords are rows of G
$$c = mG$$
where $m$ is the message.

**Relationship:** $HG^T = 0$

### 8. Challenges for Quantum

Classical techniques face obstacles in quantum:
1. **No cloning:** Can't copy quantum states
2. **Measurement disturbs:** Can't directly check states
3. **Continuous errors:** Phase can rotate continuously
4. **New error type:** Phase errors don't exist classically

**Solution:** Measure syndromes without measuring data!

---

## Worked Example

**Problem:** Using the 3-bit repetition code, decode the received word $101$.

**Solution:**
1. Compute syndrome: $s = H \cdot (1,0,1)^T = (1,0)^T$
2. Look up: Syndrome $(1,0)$ indicates error on bit 2
3. Correct: $101 \oplus 010 = 111$
4. Decode: $111 \to 1$

---

## Practice Problems

1. Construct the generator matrix for the 3-bit repetition code.
2. Verify all syndromes for the Hamming [7,4,3] code.
3. Prove that a code with minimum distance $d$ can detect up to $d-1$ errors.
4. Design a simple parity check code that detects (but doesn't correct) single errors.

---

## Computational Lab

```python
"""Day 659: Classical Error Correction"""

import numpy as np

def encode_repetition(bit):
    """Encode single bit using 3-bit repetition."""
    return np.array([bit, bit, bit])

def syndrome_repetition(received):
    """Compute syndrome for 3-bit repetition code."""
    H = np.array([[1, 1, 0], [1, 0, 1]])
    return H @ received % 2

def decode_repetition(received):
    """Decode 3-bit repetition code."""
    syndrome = syndrome_repetition(received)

    # Syndrome lookup table
    if np.array_equal(syndrome, [0, 0]):
        error_pos = None
    elif np.array_equal(syndrome, [1, 1]):
        error_pos = 0
    elif np.array_equal(syndrome, [1, 0]):
        error_pos = 1
    else:  # [0, 1]
        error_pos = 2

    # Correct error
    corrected = received.copy()
    if error_pos is not None:
        corrected[error_pos] = 1 - corrected[error_pos]

    # Decode (majority vote)
    return corrected[0]

# Test
print("3-bit Repetition Code Demo")
print("=" * 40)

for message in [0, 1]:
    codeword = encode_repetition(message)
    print(f"\nMessage: {message} -> Codeword: {codeword}")

    # Simulate errors
    for error_pos in [None, 0, 1, 2]:
        received = codeword.copy()
        if error_pos is not None:
            received[error_pos] = 1 - received[error_pos]

        decoded = decode_repetition(received)
        syndrome = syndrome_repetition(received)

        status = "✓" if decoded == message else "✗"
        print(f"  Received: {received}, Syndrome: {syndrome}, Decoded: {decoded} {status}")
```

---

## Summary

- **Classical error correction** adds redundancy to detect/correct errors
- **Parity checks** define code structure: $Hc = 0$
- **Syndromes** identify errors without revealing data
- **Repetition code:** Simple, low rate, corrects single errors
- **Hamming codes:** More efficient, still corrects single errors
- **Quantum challenge:** Can't copy, can't measure directly, phase errors exist

---

## Preview: Day 660

Tomorrow: **Quantum Error Correction Conditions** - the Knill-Laflamme theorem that tells us when QEC is possible.
