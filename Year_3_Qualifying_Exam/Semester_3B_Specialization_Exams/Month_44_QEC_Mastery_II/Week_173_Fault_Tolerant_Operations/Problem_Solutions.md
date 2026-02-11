# Week 173: Fault-Tolerant Operations - Problem Solutions

## Section 1: Error Models and Fault Tolerance Definitions

### Solution 1

**(a)** Starting with:
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

Add and subtract $$\frac{p}{3}\rho$$:
$$= (1-p)\rho + \frac{p}{3}\rho - \frac{p}{3}\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$
$$= \left(1-p+\frac{p}{3}\right)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z + \rho) - \frac{p}{3}\rho$$

Actually, the cleaner derivation:
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

We can write this as a sum over all Paulis:
$$= \left(1-\frac{4p}{3}\right)\rho + \frac{p}{3}(I\rho I + X\rho X + Y\rho Y + Z\rho Z)$$

Check: $$1 - \frac{4p}{3} + \frac{p}{3} = 1 - p$$, and we have $$\frac{p}{3}$$ for each of X, Y, Z. $$\checkmark$$

**(b)** Probability of Pauli error = $$\frac{p}{3} + \frac{p}{3} + \frac{p}{3} = p$$

**(c)** For $$p = 0.01$$:
Probability of no error = $$1 - p = 0.99 = 99\%$$

---

### Solution 2

**Definition of Fault-Tolerant Operation for Distance-3 Code:**

An operation on encoded qubits is *fault-tolerant* if:
1. A single fault anywhere in the operation results in at most one error in any code block at the output
2. If the input has $$t$$ errors per block (where $$t \leq \lfloor(d-1)/2\rfloor = 1$$), the output has at most $$t+1$$ errors per block

**Why this is crucial:**

For a distance-3 code, we can correct 1 error. If a single fault could cause 2 or more errors, then:
- Single fault $$\to$$ 2+ errors $$\to$$ uncorrectable $$\to$$ logical error

With fault tolerance:
- Single fault $$\to$$ 1 error $$\to$$ correctable
- Two faults needed $$\to$$ probability $$\propto p^2$$

This quadratic suppression is the foundation of the threshold theorem.

---

### Solution 3

**(a)** Naive syndrome extraction problem:

For stabilizer $$Z_1Z_2Z_3Z_4$$, a naive circuit uses:
```
Ancilla: |0⟩ —H—●—●—●—●—H—M
              |  |  |  |
Data:        Z  Z  Z  Z
```

If an X error occurs on the ancilla between any two CNOT gates, it propagates to all remaining data qubits. A single fault can cause 2, 3, or 4 errors—violating fault tolerance.

**(b)** Shor-style syndrome extraction:

1. Prepare cat state $$|0000\rangle + |1111\rangle$$ using:
   - Prepare $$|+\rangle$$ on first ancilla
   - CNOT to remaining 3 ancillas
   - Verify by measuring $$Z_1Z_2$$, $$Z_2Z_3$$, $$Z_3Z_4$$ (discard if any give -1)

2. Use each ancilla qubit to interact with one data qubit

3. Measure all ancillas in X basis and take majority vote

**Why it's fault-tolerant:** Each ancilla qubit touches only one data qubit, so a single ancilla error affects only one data qubit.

**(c)** Ancilla qubits needed: 4 for the cat state (one per data qubit in the stabilizer), plus additional ancillas for verification. Total: typically 4-7 ancillas per stabilizer measurement.

---

### Solution 4

**(a)** Minimum malignant set size for distance-3: **2 faults**

With distance 3, we correct 1 error. A single fault causes ≤1 error (fault-tolerance), which is correctable. Two faults can cause a weight-2 error pattern that may be uncorrectable.

**(b)** Upper bound on malignant pairs:
$$\binom{100}{2} = \frac{100 \times 99}{2} = 4950$$

**(c)** Level-1 failure probability:
$$p^{(1)} \leq 4950 \times (10^{-4})^2 = 4950 \times 10^{-8} = 4.95 \times 10^{-5}$$

---

### Solution 5

**Proof:**

For a distance-$$d$$ code, the minimum number of errors causing logical failure is $$\lceil d/2 \rceil$$.

With fault-tolerant operations, a single fault causes at most 1 error per block. Therefore:
- $$k$$ faults cause at most $$k$$ errors
- Logical failure requires ≥$$\lceil d/2 \rceil$$ errors
- Thus, logical failure requires ≥$$\lceil d/2 \rceil$$ faults

The number of malignant sets of size exactly $$\lceil d/2 \rceil$$ is at most:
$$A_d = \binom{n_{\text{loc}}}{\lceil d/2 \rceil} \leq \frac{n_{\text{loc}}^{\lceil d/2 \rceil}}{(\lceil d/2 \rceil)!}$$

Since each fault occurs independently with probability $$p$$:
$$p^{(1)} \leq A_d \cdot p^{\lceil d/2 \rceil}$$

$$\square$$

---

### Solution 6

**(a) Independent depolarizing noise:** Each qubit independently experiences depolarizing channel. Simple model, allows analytic threshold calculations.

**(b) Circuit-level noise:** Errors at each gate, preparation, measurement. More realistic as it captures the actual structure of quantum circuits.

**(c) Correlated noise:** Errors on different qubits are not independent (e.g., crosstalk causes correlated errors on neighboring qubits).

**Most conservative threshold:** Correlated/adversarial noise gives the lowest threshold, followed by circuit-level, then independent depolarizing.

**Reason:** Independent noise allows errors to "cancel" probabilistically, while correlated noise can conspire to create worst-case error patterns. Circuit-level noise is intermediate because it captures error propagation through gates.

---

## Section 2: Threshold Theorem

### Solution 7

Given $$p_{\text{th}} = 10^{-2}$$ and $$p = 10^{-3}$$:

**(a)** Level-3 logical error rate:
$$p^{(L)} = p_{\text{th}} \left(\frac{p}{p_{\text{th}}}\right)^{2^L}$$
$$p^{(3)} = 10^{-2} \times \left(\frac{10^{-3}}{10^{-2}}\right)^{8} = 10^{-2} \times (0.1)^8 = 10^{-2} \times 10^{-8} = 10^{-10}$$

**(b)** For $$\delta = 10^{-15}$$:
$$10^{-2} \times (0.1)^{2^L} \leq 10^{-15}$$
$$(0.1)^{2^L} \leq 10^{-13}$$
$$2^L \geq 13$$
$$L \geq \log_2(13) \approx 3.7$$

So $$L = 4$$ levels.

**(c)** Physical qubits at level 4:
$$n^L = 7^4 = 2401$$ physical qubits per logical qubit.

---

### Solution 8

**(a)** Explicit scaling derivation:

Number of levels: $$L = O(\log\log(1/\delta))$$
Physical qubits per logical: $$n^L = n^{O(\log\log(1/\delta))}$$

Using $$n^{\log\log(1/\delta)} = (1/\delta)^{\log\log n \cdot \frac{1}{\log(1/\delta)}} = (1/\delta)^{\frac{\log\log n}{\log(1/\delta)}}$$

For large $$1/\delta$$: $$n^L = O(\text{poly}\log(1/\delta))$$

**(b)** More precisely:
$$L = \frac{\log\log(1/\delta) - \log\log(1/p_{\text{th}})}{\log 2}$$
$$n^L = n^{L} = \exp(L \log n) = \exp\left(\frac{\log n}{\log 2}(\log\log(1/\delta) - C)\right)$$

This is $$O(\log^{\alpha}(1/\delta))$$ where $$\alpha = \frac{\log n}{\log 2}$$.

For $$n = 7$$: $$\alpha = \log_2(7) \approx 2.81$$

**(c)** Surface codes: $$O(d^2)$$ physical qubits, $$d = O(\log(1/\delta))$$
Thus $$O(\log^2(1/\delta))$$ physical qubits.

This is better than concatenated codes ($$\alpha \approx 2.81$$) for most parameters.

---

### Solution 9

**Proof of Lemma:**

*Statement:* For a distance-3 code with fault-tolerant gadgets having $$n_{\text{loc}}$$ locations, if each location fails with probability $$p$$:
$$p_{\text{fail}} \leq \binom{n_{\text{loc}}}{2} p^2$$

*Proof:*

1. **Fault tolerance property:** By definition, a single fault causes at most 1 error in the output block.

2. **Distance-3 correction:** A distance-3 code corrects any single error. Thus, 1 error does not cause logical failure.

3. **Failure requires 2+ faults:** Logical failure requires an error of weight ≥2, which requires ≥2 faults.

4. **Probability bound:** The probability of having at least 2 faults is:
$$P[\geq 2 \text{ faults}] = 1 - P[0 \text{ faults}] - P[1 \text{ fault}]$$
$$= 1 - (1-p)^{n_{\text{loc}}} - n_{\text{loc}} p (1-p)^{n_{\text{loc}}-1}$$

For small $$p$$:
$$P[\geq 2 \text{ faults}] \approx \binom{n_{\text{loc}}}{2} p^2$$

5. **Not all 2-fault combinations cause failure:** Some 2-fault patterns still result in correctable errors. Thus:
$$p_{\text{fail}} \leq P[\geq 2 \text{ faults}] \leq \binom{n_{\text{loc}}}{2} p^2$$

$$\square$$

---

### Solution 10

**(a)** Threshold estimates:

Code A: $$p_{\text{th}}^A \approx \frac{1}{\binom{50}{2}} = \frac{2}{50 \times 49} \approx \frac{1}{1225} \approx 8.2 \times 10^{-4}$$

Code B: $$p_{\text{th}}^B \approx \frac{1}{\binom{500}{4}}$$ (distance 7, need 4 faults)

$$\binom{500}{4} = \frac{500 \times 499 \times 498 \times 497}{24} \approx 2.6 \times 10^9$$

$$p_{\text{th}}^B \approx 3.9 \times 10^{-10}$$

Wait, this seems too low. Let me recalculate considering the exponent:

For distance-7 code: $$p^{(1)} \leq A_4 p^4$$ where $$A_4 = \binom{500}{4}$$

Threshold condition: $$A_4 p^3 = 1$$ (so that $$A_4 p^4 = p$$)
$$p_{\text{th}}^B = A_4^{-1/3} = (2.6 \times 10^9)^{-1/3} \approx 7 \times 10^{-4}$$

**(b)** At $$p = 10^{-4}$$:

Code A: $$p^{(1)}_A \leq 1225 \times (10^{-4})^2 = 1.225 \times 10^{-5}$$

Code B: $$p^{(1)}_B \leq 2.6 \times 10^9 \times (10^{-4})^4 = 2.6 \times 10^{-7}$$

**Code B gives lower level-1 error rate.**

**(c)** Equal error rates when:
$$1225 \times p^2 = 2.6 \times 10^9 \times p^4$$
$$p^2 = \frac{1225}{2.6 \times 10^9} = 4.7 \times 10^{-7}$$
$$p = 6.9 \times 10^{-4}$$

---

### Solution 11

**Relationships:**

1. **Distance $$d$$** determines error correction capability (corrects $$\lfloor(d-1)/2\rfloor$$ errors) and the exponent in error suppression.

2. **Locations $$n_{\text{loc}}$$** increases with code complexity and distance. More locations means more potential fault combinations.

3. **Threshold $$p_{\text{th}}$$** scales roughly as:
   $$p_{\text{th}} \sim n_{\text{loc}}^{-2/(d-1)}$$

   Higher distance can increase threshold (stronger suppression) but also requires more complex gadgets (more $$n_{\text{loc}}$$), creating a trade-off.

4. **Concatenation level $$L$$** for target error $$\delta$$:
   $$L = \frac{\log\log(1/\delta)}{\log((d+1)/2)} + O(1)$$

   Higher distance reduces required $$L$$.

---

### Solution 12

**(a)** Why surface code threshold is higher:

1. **Local operations:** Surface code syndrome extraction uses only local (nearest-neighbor) operations, limiting error propagation
2. **No concatenation overhead:** Threshold is computed for a single level, not recursive gadgets
3. **Favorable geometry:** 2D layout matches realistic hardware constraints
4. **Efficient decoding:** MWPM decoder is near-optimal for independent noise

**(b)** When concatenated codes are preferable:

1. **Very low physical error rates:** Below $$\sim 10^{-6}$$, concatenated codes can outperform surface codes in qubit count
2. **Non-local connectivity:** If long-range gates are available, concatenated codes don't pay the locality overhead
3. **Small computations:** For few logical qubits, concatenated codes have lower minimum overhead

**(c)** Crossover point:

Surface code: $$n_{\text{surf}} \approx d^2 = O(\log^2(1/\delta))$$
Concatenated: $$n_{\text{conc}} \approx 7^L = O(\log^{2.8}(1/\delta))$$

Crossover when $$\log^2(1/\delta) = \log^{2.8}(1/\delta)$$, which occurs at moderate $$\delta$$. Detailed analysis depends on specific threshold values.

---

## Section 3: Transversal Gates

### Solution 13

**(a)** Verifying $$H^{\otimes 7}$$ implements logical Hadamard:

The logical operators are:
$$\overline{X} = X^{\otimes 7}, \quad \overline{Z} = Z^{\otimes 7}$$

Under $$H^{\otimes 7}$$:
$$H^{\otimes 7} \overline{X} (H^{\otimes 7})^\dagger = H^{\otimes 7} X^{\otimes 7} H^{\otimes 7} = Z^{\otimes 7} = \overline{Z}$$
$$H^{\otimes 7} \overline{Z} (H^{\otimes 7})^\dagger = H^{\otimes 7} Z^{\otimes 7} H^{\otimes 7} = X^{\otimes 7} = \overline{X}$$

This is exactly the action of logical Hadamard: $$\overline{H}\overline{X}\overline{H}^\dagger = \overline{Z}$$. $$\checkmark$$

**(b)** Checking $$S^{\otimes 7}$$:

The phase gate $$S$$ acts as: $$SXS^\dagger = Y$$, $$SZS^\dagger = Z$$

$$S^{\otimes 7} \overline{X} (S^{\otimes 7})^\dagger = S^{\otimes 7} X^{\otimes 7} S^{\otimes 7} = Y^{\otimes 7} = i^7 X^{\otimes 7} Z^{\otimes 7} = -i \overline{X}\overline{Z}$$

But $$\overline{S}\overline{X}\overline{S}^\dagger = \overline{Y} = i\overline{X}\overline{Z}$$

We get $$-i$$ instead of $$i$$, so $$S^{\otimes 7} \neq \overline{S}$$.

**(c)** What $$S^{\otimes 7}$$ implements:

$$S^{\otimes 7} = \overline{S}^\dagger = \overline{S^{-1}}$$

This is the inverse of the logical S gate.

---

### Solution 14

**(a)** Proof that $$\text{CNOT}^{\otimes n}$$ is logical CNOT for CSS codes:

For CSS code with logical operators:
- $$\overline{X}_1 = X_{S_1}$$ (X on some set $$S_1$$)
- $$\overline{Z}_1 = Z_{T_1}$$ (Z on some set $$T_1$$)

CNOT action: $$\text{CNOT} X_c I_t \text{CNOT}^\dagger = X_c X_t$$

$$\text{CNOT}^{\otimes n} (\overline{X}_1 \otimes I) (\text{CNOT}^{\otimes n})^\dagger = \overline{X}_1 \otimes \overline{X}_2$$

Similarly for Z. This matches logical CNOT action. $$\checkmark$$

**(b)** Condition for $$H^{\otimes n}$$ to be valid:

$$H^{\otimes n}$$ swaps X and Z stabilizers. For this to map code to itself, we need the code to be self-dual: the X and Z stabilizers must have the same structure.

Condition: $$C_1 = C_2^\perp$$ and the code is self-dual CSS.

**(c)** Example where $$H^{\otimes n}$$ fails:

The $$[[4,2,2]]$$ code is CSS but not self-dual. $$H^{\otimes 4}$$ would swap the X and Z stabilizers, producing a different code.

---

### Solution 15

**(a)** Consistency with Eastin-Knill:

Eastin-Knill says no single code has ALL gates transversal. The $$[[15,1,3]]$$ code having transversal T but not H is perfectly consistent—it just means the transversal gate set is not universal.

**(b)** Code switching for universality:

1. Start in Steane code (has transversal Clifford)
2. Perform Clifford operations transversally
3. When T gate needed:
   - Decode from Steane to physical qubit
   - Re-encode into $$[[15,1,3]]$$ code
   - Apply transversal T
   - Decode back to physical
   - Re-encode into Steane code
4. Continue computation

**(c)** Overhead comparison:

Code switching overhead: $$O(1)$$ per T gate (just encoding/decoding operations)

Magic state injection: Requires distillation, which costs $$O(\log^{\gamma}(1/\epsilon))$$ magic states

For high-precision computation, code switching can be more efficient, but it requires more complex circuitry.

---

### Solution 16

**Proof that transversal gates form a finite group:**

1. **Stabilizer code structure:** An $$[[n,k,d]]$$ stabilizer code has a $$2^k$$-dimensional code space, spanned by logical computational basis states $$|\overline{x}\rangle$$ for $$x \in \{0,1\}^k$$.

2. **Transversal gate action:** A transversal gate $$U = U_1 \otimes \cdots \otimes U_n$$ must preserve the code space and act as some logical unitary $$\overline{U}$$ on it.

3. **Normalizer condition:** $$U$$ must normalize the stabilizer group $$S$$: $$USU^\dagger = S$$.

4. **Finite normalizer:** The normalizer of $$S$$ in the $$n$$-qubit Pauli group is the Clifford group, which is finite (for fixed $$n$$).

5. **Logical action:** The map $$U \mapsto \overline{U}$$ from transversal gates to logical gates has image in the normalizer of the logical Pauli group, which is the $$k$$-qubit Clifford group (finite).

6. **Conclusion:** The set of transversal gates, modulo phases and stabilizers, is finite.

$$\square$$

---

### Solution 17

**(a)** Transversal Clifford on Steane code:

- $$\overline{H} = H^{\otimes 7}$$ (verified in Problem 13)
- $$\overline{S} = S^{\dagger \otimes 7}$$ (note the inverse)
- $$\overline{\text{CNOT}} = \text{CNOT}^{\otimes 7}$$ between two blocks

**(b)** Composition remains transversal:

If $$U = U_1 \otimes \cdots \otimes U_n$$ and $$V = V_1 \otimes \cdots \otimes V_n$$, then:
$$UV = (U_1V_1) \otimes \cdots \otimes (U_nV_n)$$

This is still transversal (tensor product form preserved).

**(c)** Conclusion:

Since $$\{H, S, \text{CNOT}\}$$ generate the Clifford group, and all generators have transversal implementations, and compositions of transversal gates are transversal, all Clifford gates have transversal implementations on the Steane code.

---

## Section 4: Eastin-Knill Theorem

### Solution 18

**Eastin-Knill Theorem Statement:**

*Theorem:* For any quantum error-correcting code with distance $$d \geq 2$$, the set of transversal logical gates cannot form a universal gate set.

**Key Assumptions:**
1. The code has distance $$d \geq 2$$ (can detect at least one error)
2. "Transversal" means tensor product form $$U = U_1 \otimes \cdots \otimes U_n$$
3. "Universal" means generating a dense subgroup of $$SU(2^k)$$ for $$k$$ logical qubits

---

### Solution 19

**(a)** Why continuous $$U(\theta)$$ cannot exist:

Suppose $$U(\theta) = U_1(\theta) \otimes \cdots \otimes U_n(\theta)$$ is a continuous family of transversal gates.

For small $$\epsilon$$, $$U(\epsilon)$$ is close to identity: $$U(\epsilon) = I + i\epsilon H + O(\epsilon^2)$$

where $$H = H_1 \otimes I \otimes \cdots + I \otimes H_2 \otimes \cdots + \cdots$$ is a sum of local terms.

But this infinitesimal generator cannot be detected by the code's error correction (it looks like small rotations on individual qubits), yet it implements a non-trivial logical rotation.

**(b)** Property violated:

The Knill-Laflamme conditions require that errors $$E_i$$ satisfying $$P E_i P \propto P$$ (where $$P$$ projects onto code space) should not affect logical information. But $$U(\epsilon) - I \approx i\epsilon H$$ would be such an undetectable "error" that changes logical state.

**(c)** Failure for $$d = 1$$:

A $$d = 1$$ code has no error correction (0 correctable errors). The Knill-Laflamme conditions are vacuous, so there's no contradiction from having continuous transversal gates.

---

### Solution 20

**Completed proof of Eastin-Knill:**

*Steps 1-4 from problem statement, plus:*

**Step 5:** Contradiction with error correction:

Suppose $$\mathcal{T}$$ is the set of transversal gates and it's universal.

Then $$\mathcal{T}$$ generates a dense subgroup of $$SU(2^k)$$.

This means for any $$\epsilon > 0$$, there exists $$U \in \mathcal{T}$$ with $$\|U - I\| < \epsilon$$.

Such $$U = U_1 \otimes \cdots \otimes U_n$$ can be written as:
$$U = I^{\otimes n} + \sum_j \epsilon_j P_j + O(\epsilon^2)$$

where $$P_j$$ are local operators acting on few qubits.

For a code with $$d \geq 2$$, any weight-1 error $$E$$ satisfies $$\langle \overline{0}|E|\overline{1}\rangle = 0$$ (Knill-Laflamme).

But $$U$$ close to identity implies its deviation from identity is a sum of low-weight terms, which by Knill-Laflamme cannot change logical information.

Contradiction: $$U$$ is non-trivial (generates universal set) but cannot change logical state.

Therefore, $$\mathcal{T}$$ cannot be universal.

$$\square$$

---

### Solution 21

**(a) Magic State Injection:**

Magic states are prepared separately (not transversally) and "injected" to implement non-Clifford gates. The non-transversal operation is state preparation, not a gate. Eastin-Knill restricts gates, not state preparation.

**(b) Code Switching:**

Different codes have different transversal gates. By switching between codes:
- Steane code $$\to$$ transversal Clifford
- Reed-Muller code $$\to$$ transversal T

Together, these form a universal set. No single code has universal transversal gates, satisfying Eastin-Knill.

**(c) Gauge Fixing:**

Subsystem codes have gauge degrees of freedom. Different gauge fixings effectively give different codes with different transversal gates. By dynamically changing the gauge, universality is achieved without violating Eastin-Knill (which applies to fixed codes).

---

## Section 5: Magic States

### Solution 22

**(a)** Verify $$|T\rangle = T|+\rangle$$:

$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

$$T|+\rangle = \frac{1}{\sqrt{2}}(T|0\rangle + T|1\rangle) = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle) = |T\rangle$$ $$\checkmark$$

**(b)** In $$\{|+\rangle, |-\rangle\}$$ basis:

$$|0\rangle = \frac{1}{\sqrt{2}}(|+\rangle + |-\rangle)$$
$$|1\rangle = \frac{1}{\sqrt{2}}(|+\rangle - |-\rangle)$$

$$|T\rangle = \frac{1}{\sqrt{2}}\left[\frac{1}{\sqrt{2}}(|+\rangle + |-\rangle) + e^{i\pi/4} \cdot \frac{1}{\sqrt{2}}(|+\rangle - |-\rangle)\right]$$

$$= \frac{1}{2}\left[(1 + e^{i\pi/4})|+\rangle + (1 - e^{i\pi/4})|-\rangle\right]$$

$$= \frac{1}{2}\left[(1 + \frac{1+i}{\sqrt{2}})|+\rangle + (1 - \frac{1+i}{\sqrt{2}})|-\rangle\right]$$

**(c)** Calculations:

$$\langle T|T\rangle = 1$$ (normalized state)

$$\langle T|S|T\rangle = \frac{1}{2}\langle 0| + \frac{1}{2}e^{-i\pi/4}\langle 1|) S (|0\rangle + e^{i\pi/4}|1\rangle)$$

$$S|0\rangle = |0\rangle, \quad S|1\rangle = i|1\rangle$$

$$= \frac{1}{2}(1 + e^{-i\pi/4} \cdot i \cdot e^{i\pi/4}) = \frac{1}{2}(1 + i) = \frac{1+i}{2}$$

---

### Solution 23

**(a)** Gate injection circuit:

```
|ψ⟩ ——●——————————X^m———— T|ψ⟩
      |
|T⟩ ——⊕——M_Z → m
```

**(b)** Proof of correctness:

Let $$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

After CNOT:
$$|\psi\rangle|T\rangle \to \frac{1}{\sqrt{2}}[\alpha|0\rangle(|0\rangle + e^{i\pi/4}|1\rangle) + \beta|1\rangle(|1\rangle + e^{i\pi/4}|0\rangle)]$$

$$= \frac{1}{\sqrt{2}}[(\alpha|0\rangle + \beta e^{i\pi/4}|1\rangle)|0\rangle + (e^{i\pi/4}\alpha|0\rangle + \beta|1\rangle)|1\rangle]$$

Measure second qubit:
- If $$m = 0$$: state is $$\alpha|0\rangle + \beta e^{i\pi/4}|1\rangle = T|\psi\rangle$$ $$\checkmark$$
- If $$m = 1$$: state is $$e^{i\pi/4}\alpha|0\rangle + \beta|1\rangle = XT|\psi\rangle$$

Apply $$X$$ correction when $$m = 1$$: $$X \cdot XT|\psi\rangle = T|\psi\rangle$$ $$\checkmark$$

**(c)** Effect of magic state error:

If $$|T\rangle$$ has error $$\epsilon$$ (e.g., $$|T'\rangle = \sqrt{1-\epsilon}|T\rangle + \sqrt{\epsilon}|T^\perp\rangle$$), the output state has error $$O(\epsilon)$$.

The error propagates directly: $$T'|\psi\rangle$$ differs from $$T|\psi\rangle$$ by $$O(\epsilon)$$.

---

### Solution 24

**(a)** The H-state enables the gate:
$$\pi/8$$-rotation about X-axis, or equivalently $$\sqrt{T}$$ up to Clifford corrections.

**(b)** Relationship between H and T states:

$$|H\rangle = \frac{1}{\sqrt{2}}(|T\rangle + |T^*\rangle)$$ where $$|T^*\rangle$$ is the complex conjugate.

They are related by Clifford gates: $$|H\rangle = e^{i\pi/8}H|T\rangle$$ approximately.

**(c)** T-state is more commonly used because:

1. T gate is the standard non-Clifford gate in most gate synthesis algorithms
2. Solovay-Kitaev and other decompositions use $$\{H, T\}$$ as the universal set
3. T-state distillation protocols are better optimized in the literature

---

## Section 6: Magic State Distillation

### Solution 25

**(a)** With $$\epsilon_{\text{in}} = 0.01$$:
$$\epsilon_{\text{out}} = 35 \times (0.01)^3 = 35 \times 10^{-6} = 3.5 \times 10^{-5}$$

**(b)** Reaching $$\epsilon < 10^{-10}$$:

Round 1: $$35 \times (10^{-2})^3 = 3.5 \times 10^{-5}$$
Round 2: $$35 \times (3.5 \times 10^{-5})^3 = 35 \times 4.3 \times 10^{-14} = 1.5 \times 10^{-12}$$

After 2 rounds: $$\epsilon \approx 1.5 \times 10^{-12} < 10^{-10}$$ $$\checkmark$$

**2 rounds needed.**

**(c)** Total input states:

Round 1: $$15$$ states $$\to$$ 1 state
Round 2: $$15$$ round-1 outputs $$\to$$ 1 final state

Total: $$15 \times 15 = 225$$ initial magic states.

---

### Solution 26

**(a)** Code used: $$[[15,1,3]]$$ punctured Reed-Muller code

Parameters: 15 physical qubits, 1 logical qubit, distance 3.

**(b)** Cubic error suppression:

- Distance 3 means 1 error is correctable
- Single error on input magic states: detected and rejected
- Two errors: most patterns also detected (8 stabilizers)
- Only weight-2 errors in the code's $$\mathbb{Z}_2$$ kernel cause logical errors
- Probability of such patterns: $$\propto \epsilon^3$$ (need specific 3-error conspiracy)

**(c)** Acceptance probability:

For small $$\epsilon$$:
$$P[\text{accept}] = 1 - P[\text{any stabilizer triggered}] \approx 1 - 15\epsilon + O(\epsilon^2)$$

For $$\epsilon = 0.01$$: $$P[\text{accept}] \approx 0.85$$

**(d)** Expected input states per output:

$$\langle N_{\text{in}} \rangle = \frac{15}{P[\text{accept}]} \approx \frac{15}{0.85} \approx 17.6$$

---

### Solution 27

**(a)** Derivation of $$\gamma$$ for 15-to-1:

After $$k$$ rounds: $$\epsilon_k \approx 35^{(3^k-1)/2} \epsilon_0^{3^k}$$

To reach $$\epsilon_{\text{target}}$$:
$$\epsilon_0^{3^k} \approx \epsilon_{\text{target}}$$
$$3^k = \frac{\log(1/\epsilon_{\text{target}})}{\log(1/\epsilon_0)}$$
$$k = \log_3\left(\frac{\log(1/\epsilon_{\text{target}})}{\log(1/\epsilon_0)}\right)$$

Number of input states: $$N = 15^k$$

$$\log N = k \log 15 = \log_3\log(1/\epsilon_{\text{target}}) \cdot \log 15 + O(1)$$

$$N = O\left(\log^{\gamma}(1/\epsilon_{\text{target}})\right)$$

where $$\gamma = \frac{\log 15}{\log 3} = \log_3 15 \approx 2.46$$

**(b)** General $$k$$-to-1 with $$\epsilon^m$$ suppression:

Rounds needed: $$k = \log_m \log(1/\epsilon_{\text{target}})$$
Input states: $$N = k^{\log_m \log(1/\epsilon)}$$

$$\gamma = \frac{\log k}{\log m}$$

**(c)** Theoretical minimum: $$\gamma = 0$$ (constant overhead)

**Yes, achieved in 2025** using asymptotically good QLDPC codes (Panteleev-Kalachev construction).

---

### Solution 28

**(a)** How QLDPC enables $$\gamma = 0$$:

Asymptotically good QLDPC codes have:
- Constant rate: $$k/n = \Theta(1)$$
- Linear distance: $$d = \Theta(n)$$

For magic state distillation:
- Encode $$\Theta(n)$$ magic states into code
- Linear distance gives error suppression $$\epsilon^{\Theta(n)}$$
- Constant rate means overhead is $$O(1)$$ per output magic state
- One round achieves any polynomial suppression
- Total overhead: $$O(1)$$ regardless of target error $$\to \gamma = 0$$

**(b)** Practical challenges:

1. **Non-local connectivity:** QLDPC codes require non-planar qubit connectivity
2. **Syndrome extraction:** Low-density checks are spatially distributed
3. **Decoding complexity:** QLDPC decoders are more complex than surface code decoders
4. **Code size:** Asymptotic benefits require large code blocks ($$n \gtrsim 10^4$$)
5. **Threshold:** QLDPC codes may have lower thresholds than surface codes

**(c)** Crossover point:

Standard 15-to-1: $$N \propto \log^{2.46}(1/\epsilon)$$
QLDPC: $$N \propto$$ constant (but large constant)

Crossover when:
$$C_1 \log^{2.46}(1/\epsilon) = C_2$$

For typical constants: crossover at $$\epsilon_{\text{target}} \sim 10^{-15}$$ to $$10^{-20}$$

For very high precision computation (e.g., Shor's algorithm for large numbers), QLDPC becomes advantageous.
