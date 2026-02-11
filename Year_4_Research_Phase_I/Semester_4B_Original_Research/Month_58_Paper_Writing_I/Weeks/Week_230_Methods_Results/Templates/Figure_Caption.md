# Figure Caption Template and Examples

## Caption Structure

Every figure caption should follow this structure:

```
Figure N. [Title: What the figure shows - one sentence]

(a) [Description of panel a]
(b) [Description of panel b]
...

[Symbol and color definitions]
[Key parameter values]
[Additional context if needed]
```

---

## Caption Elements Checklist

Each caption must include:

- [ ] **Figure number and title:** "Figure N. [Descriptive title]"
- [ ] **Panel descriptions:** What each panel shows
- [ ] **Data description:** What symbols/lines represent
- [ ] **Error bar meaning:** "Error bars represent 1σ statistical uncertainty"
- [ ] **Symbol definitions:** Define all symbols and abbreviations
- [ ] **Parameter values:** Key experimental/simulation parameters
- [ ] **Theory comparison:** Describe theory curves if present

---

## Template: Schematic Figure

```
Figure 1. Experimental setup and device schematic.

(a) Simplified circuit diagram of the superconducting qubit system.
The transmon qubit (Q) is capacitively coupled to a readout resonator
(R) and driven through a dedicated control line. The flux line
provides DC bias and fast flux pulses for two-qubit gates.

(b) Optical micrograph of the device. The qubit (center) is connected
to the readout resonator (meandering line, left) and flux bias line
(right). Scale bar: 100 μm.

(c) Energy level diagram showing the qubit transition frequency
ωq/2π = 5.12 GHz and anharmonicity α/2π = -340 MHz. The readout
resonator at ωr/2π = 7.23 GHz is dispersively coupled with
g/2π = 85 MHz.
```

---

## Template: Data Figure

```
Figure 2. Qubit coherence characterization.

(a) Energy relaxation measurement. Blue circles: measured excited
state population versus delay time. Solid line: exponential fit
yielding T1 = 85 ± 5 μs. Each point represents 10,000 single-shot
measurements; error bars (smaller than markers) show 1σ statistical
uncertainty.

(b) Ramsey oscillation measurement at detuning Δ/2π = 1 MHz.
Red squares: measured ground state probability. Solid line: fit
to damped cosine yielding T2* = 45 ± 3 μs. Inset: spin-echo
measurement showing T2E = 120 ± 10 μs.

(c) Noise power spectral density extracted from dynamical decoupling
sequences with varying filter order n (colors). Solid line:
1/f model with amplitude Sφ = 3.2 μΦ0/√Hz.
```

---

## Template: Comparison Figure

```
Figure 3. Gate fidelity optimization.

(a) CZ gate fidelity versus pulse duration. Red circles: experimental
data from randomized benchmarking with 100 random Clifford sequences
per point. Blue solid line: numerical simulation using independently
measured system parameters (see text). Gray dashed line: 99% fidelity
threshold. Error bars represent 1σ uncertainty from bootstrap analysis.

(b) Theoretical prediction for fidelity-duration tradeoff. Contours
show predicted fidelity as a function of gate duration and flux
amplitude. Star indicates experimental operating point. The optimal
region (>99.5% fidelity) is shaded in blue.

(c) Comparison of experimental (red) and simulated (blue) fidelity
for three gate implementations: CZ (circles), iSWAP (squares), and
√iSWAP (triangles). Agreement within uncertainty confirms that
decoherence, not control errors, limits performance.

Experimental parameters: ωq/2π = 5.12 GHz, T1 = 85 μs, T2E = 120 μs.
```

---

## Template: Multi-Panel Summary Figure

```
Figure 4. Comprehensive gate benchmarking results.

(a) Single-qubit randomized benchmarking survival curves for X
(blue), Y (orange), and Z (green) basis gates. Curves offset
vertically for clarity. Extracted error per gate: εX = 0.12%,
εY = 0.11%, εZ = 0.09%.

(b) Two-qubit randomized benchmarking. Red circles: standard RB
with full Clifford group. Purple squares: interleaved RB isolating
the CZ gate. Solid lines: exponential fits. Extracted CZ error:
εCZ = 0.28 ± 0.05%.

(c) Gate set tomography reconstruction fidelity. Bar heights
indicate process fidelity for each gate in the set. Error bars
show 1σ uncertainty from maximum likelihood estimation.

(d) Error budget breakdown. Pie chart showing fractional
contributions to total gate error: coherent error (blue, 15%),
T1 decay (orange, 35%), T2 dephasing (green, 40%), leakage
(red, 10%).

All measurements performed at base temperature 10 mK with
1000 repetitions per data point.
```

---

## Template: Colormap/Heatmap Figure

```
Figure 5. Parameter space exploration.

(a) Measured CZ gate fidelity as a function of flux pulse amplitude
(horizontal) and duration (vertical). Color scale indicates fidelity
from 90% (dark blue) to 99.9% (yellow). White contour marks the
99% threshold. Black cross indicates optimal operating point
(Φ/Φ0 = 0.35, τ = 35 ns) achieving 99.7% fidelity.

(b) Simulated leakage probability for the same parameter range.
Color scale from 0% (dark) to 5% (bright). The high-fidelity
region in (a) corresponds to low leakage (<0.5%) in (b).

(c) Line cuts through optimal duration (τ = 35 ns). Blue: fidelity
(left axis). Red: leakage (right axis). Shaded region indicates
the operating range maintaining >99.5% fidelity.

Data acquired over 12-hour measurement session with periodic
recalibration every 2 hours.
```

---

## Common Caption Mistakes and Corrections

### Mistake 1: Too Brief

**Bad:**
```
Figure 3. Gate fidelity data.
```

**Good:**
```
Figure 3. CZ gate fidelity versus pulse duration.
Red circles: experimental randomized benchmarking data.
Blue line: numerical simulation. Optimal point (star)
achieves 99.7 ± 0.1% at 35 ns duration.
```

### Mistake 2: No Error Bar Definition

**Bad:**
```
Figure 2. Measurement results with error bars.
```

**Good:**
```
Figure 2. Measurement results. Error bars represent
1σ statistical uncertainty from 1000 repetitions.
Systematic uncertainty (estimated 0.1%) is not shown.
```

### Mistake 3: Undefined Symbols

**Bad:**
```
Figure 4. Comparison of different approaches.
Red shows ours, blue shows previous work.
```

**Good:**
```
Figure 4. Comparison of gate performance.
Red circles: this work (randomized benchmarking).
Blue squares: Ref. [15] (process tomography).
Green triangles: Ref. [23] (cross-entropy benchmarking).
```

### Mistake 4: Missing Context

**Bad:**
```
Figure 5. Phase diagram.
```

**Good:**
```
Figure 5. Quantum phase diagram of the transverse-field
Ising model on a 10×10 square lattice. Color indicates
the order parameter ⟨σz⟩ computed by DMRG (χ = 200).
White line: analytically known phase boundary (h/J = 1).
Gray dashed line: finite-size crossover extracted from
susceptibility peak.
```

---

## Caption Writing Process

1. **State what the figure shows** (title sentence)
2. **Describe each panel** systematically
3. **Define all visual elements** (colors, symbols, lines)
4. **Specify error bars** and their meaning
5. **Include relevant parameters** for context
6. **Read caption without looking at figure** - can you understand it?

---

## Final Checklist

For each figure caption:

- [ ] Title clearly describes figure content
- [ ] Each panel is described
- [ ] All colors/symbols/line styles defined
- [ ] Error bars explained (or stated not shown)
- [ ] Key parameter values included
- [ ] Caption is self-contained (understandable alone)
- [ ] Abbreviations defined (first use in caption)
- [ ] Data source clear (experimental vs. simulation)

---

*Use this template alongside the main Guide.md and Figure_Guidelines.md for complete figure preparation.*
