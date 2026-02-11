# Limitations Framework

## Purpose

This framework guides the systematic identification, analysis, and presentation of limitations in your research. Acknowledging limitations honestly demonstrates scientific maturity and helps readers properly interpret your findings.

---

## The Philosophy of Limitations

### Why Acknowledge Limitations?

1. **Scientific Integrity**: Honest reporting of what research can and cannot tell us
2. **Reader Trust**: Transparency builds credibility
3. **Proper Interpretation**: Helps readers avoid over-interpreting results
4. **Future Guidance**: Identifies areas for improvement
5. **Scholarly Maturity**: Demonstrates doctoral-level critical thinking

### The Balance Challenge

**Too Much Self-Criticism:**
- Undermines your contributions
- Creates impression work is flawed
- May cause readers to dismiss valid findings

**Too Little Self-Criticism:**
- Appears naive or overconfident
- Reviewers/examiners will identify overlooked limitations
- Damages credibility

**Right Balance:**
- Honest about significant limitations
- Proportionate emphasis (major vs. minor)
- Constructive framing (how to address)
- Confident about valid conclusions

---

## Limitation Categories

### 1. Methodological Limitations

**Definition**: Constraints arising from the methods used to conduct the research

**Subcategories:**

#### 1.1 Measurement Technique Limitations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Accuracy | Does the technique measure what we think? | RB measures average fidelity, not coherent error details |
| Precision | What is the measurement resolution? | SNR limits fidelity precision to ±0.1% |
| Bias | Are there systematic errors? | SPAM errors bias QPT fidelities low |
| Artifacts | What artifacts might appear? | Pulse distortion creates artificial decay |

**Template:**

> The [measurement technique] provides [what it measures] but has limitations:
>
> First, [limitation 1 description]. This means that [consequence]. We mitigate this by [mitigation], but residual effects at the [level] may remain.
>
> Second, [limitation 2 description]. As discussed in Section N.3.X, we addressed this through [approach], though [remaining concern].

#### 1.2 Analysis Method Limitations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Model assumptions | What does the model assume? | Assumes Markovian noise |
| Fitting | How sensitive to fitting choices? | Results depend on binning choice |
| Statistical framework | Are statistics appropriate? | Frequentist CIs have coverage issues |

**Template:**

> Our analysis assumes [key assumptions]. These assumptions are justified when [conditions], as discussed in Section N.3.X. Violations of these assumptions would [consequences].
>
> The sensitivity of our conclusions to analysis choices was tested by [sensitivity analysis]. Results are robust to [what variations], but [specific conclusion] depends on [specific assumption].

#### 1.3 Experimental Control Limitations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Control parameters | What couldn't we control perfectly? | Temperature fluctuations ±2 mK |
| Environmental factors | What environmental factors affect results? | Magnetic field drifts during data collection |
| Reproducibility | How reproducible were conditions? | Calibrations drift between runs |

**Template:**

> Experimental conditions were controlled to [level], with residual variations in [parameters]. These variations contribute [estimated effect] to the uncertainty budget (Table N.X).
>
> [Specific uncontrolled factor] could not be fully controlled, potentially affecting [results] at the [level]. This systematic effect [is/is not] included in our quoted uncertainties.

---

### 2. Sample/System Limitations

**Definition**: Constraints arising from the specific sample, device, or system studied

#### 2.1 Sample Specificity

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Representativeness | Is this sample typical? | Single device from one fabrication run |
| Selection | How was sample selected? | Best device from batch |
| Variability | How variable are samples? | Device-to-device variation unknown |

**Template:**

> Results were obtained from [sample description]. This sample [is/is not] representative of [broader category] because [reasoning].
>
> Sample selection was based on [criteria], which [does/does not] introduce bias because [explanation].
>
> Generalization to other samples requires [caution/confidence] because [reasoning]. Future work should characterize [number] additional samples to establish [generalizability].

#### 2.2 Operating Regime Limitations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Parameter range | What range was studied? | Only optimal operating point characterized |
| Conditions | Under what conditions? | Low-power regime only |
| Duration | What timescales? | Short-term behavior only |

**Template:**

> All measurements were performed at [operating conditions]. Extrapolation to [other conditions] should be done with caution because [reason].
>
> We characterized [parameter] over the range [range]. Behavior outside this range [is/is not] expected to differ because [reasoning].

#### 2.3 Platform Limitations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Platform-specific | What's unique to this platform? | Superconducting qubits have specific noise |
| Generalizability | Does this apply to other platforms? | May not apply to ion traps |
| Scalability | Does this scale? | Demonstrated on 2 qubits only |

**Template:**

> These results are specific to [platform] and may not directly apply to [other platforms] due to [fundamental differences].
>
> Scaling beyond [current scale] was not demonstrated. Extrapolation suggests [scaling prediction], but this requires experimental validation.

---

### 3. Scope Limitations

**Definition**: Constraints on what questions the research addressed

#### 3.1 Research Focus Limitations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Scope boundaries | What was outside scope? | Did not study multi-qubit scaling |
| Depth vs. breadth | What tradeoffs were made? | Focused on one gate type |
| Time/resource constraints | What couldn't be done? | Long-term stability not characterized |

**Template:**

> This study focused on [specific scope]. The following related questions were not addressed:
>
> 1. [Unaddressed question 1]: Important for [reason], not included due to [constraint]
> 2. [Unaddressed question 2]: Would require [resources/time], planned for future work
>
> These scope limitations mean that [what conclusions cannot be drawn].

#### 3.2 Generalizability Limitations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Context dependence | How context-specific? | Results specific to this Hamiltonian |
| External validity | Does this apply elsewhere? | May not apply to different noise environments |
| Population | What population does this represent? | N=1 device limits generalization |

**Template:**

> The generalizability of these findings is limited by [factors]. Our results apply to [specific context] but may not extend to [other contexts] because [reasoning].
>
> Readers should not interpret these results as evidence that [overgeneralization]. Establishing generality requires [future work].

---

### 4. Interpretation Limitations

**Definition**: Constraints on the conclusions that can be drawn

#### 4.1 Alternative Explanations

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Confounds | What alternative explanations exist? | Could be sample variation, not method |
| Causality | Is causation established? | Correlation doesn't prove mechanism |
| Completeness | Are all factors accounted for? | Unknown unknowns may exist |

**Template:**

> Our interpretation that [conclusion] is supported by [evidence]. However, alternative explanations cannot be fully ruled out:
>
> 1. [Alternative 1]: [Description]. This is [un]likely because [reasoning].
> 2. [Alternative 2]: [Description]. Distinguishing from our interpretation requires [test].
>
> Based on available evidence, [favored interpretation] is most probable, but certainty is limited to [confidence level].

#### 4.2 Model Dependence

| Aspect | Questions to Ask | Example Limitation |
|--------|-----------------|-------------------|
| Model assumptions | What does interpretation assume? | Assumes standard decoherence model |
| Model validity | Is model valid here? | Near regime boundary |
| Parameter dependence | How sensitive to model parameters? | Interpretation changes with different T1 |

**Template:**

> Our conclusions depend on the validity of [model/framework]. If [alternative model] is correct, the interpretation would change to [alternative interpretation].
>
> The sensitivity to model choice was assessed by [analysis]. Core conclusions are robust, but [specific conclusion] depends on [specific assumption].

---

## Limitation Presentation Framework

### Step 1: Identify All Limitations

Complete this inventory:

| Category | Limitation | Severity | Impact | Can Address? |
|----------|-----------|----------|--------|--------------|
| Methodological | | Low/Med/High | | Yes/No |
| Sample | | Low/Med/High | | Yes/No |
| Scope | | Low/Med/High | | Yes/No |
| Interpretation | | Low/Med/High | | Yes/No |

### Step 2: Prioritize Limitations

**High priority** (must discuss in main text):
- Significantly affects interpretation
- Readers would notice if omitted
- Limits major conclusions

**Medium priority** (discuss briefly or in context):
- Affects specific results
- Standard for the field
- Can be addressed with caveats

**Low priority** (may omit or mention briefly):
- Minor impact
- Universal to all studies
- Obvious to experts

### Step 3: Assess Impact

For each significant limitation:

| Limitation | Affected Results | Magnitude of Effect | Conclusion Still Valid? |
|------------|-----------------|--------------------|-----------------------|
| [Limitation] | [Which results] | [Quantitative if possible] | [Yes/No/Partially] |

### Step 4: Frame Constructively

For each limitation, identify:
- How was impact minimized?
- What would address it?
- Is it fundamental or practical?

---

## Writing Limitations Well

### Effective Phrases

**Acknowledging:**
- "Several limitations should be acknowledged..."
- "These results should be interpreted in light of..."
- "Readers should note that..."
- "An important caveat is..."

**Contextualizing:**
- "This limitation is common to studies of this type..."
- "While this constrains our conclusions, it does not invalidate..."
- "The impact of this limitation is bounded by..."

**Constructive:**
- "Future work could address this by..."
- "This limitation motivates [future research]..."
- "Establishing [X] would require..."

### Ineffective Phrases

**Too Defensive:**
- "A major problem with our study..."
- "Unfortunately, we were unable to..."
- "We failed to..."

**Too Dismissive:**
- "This minor limitation does not affect..."
- "While technically a limitation, this is not important..."
- "Critics might note... but they would be wrong..."

---

## Limitation Section Organization

### Option 1: By Category

```
N.5.4 Limitations

N.5.4.1 Methodological Limitations
    - Measurement technique constraints
    - Analysis assumptions

N.5.4.2 Sample Limitations
    - Sample specificity
    - Operating regime constraints

N.5.4.3 Scope Limitations
    - What was not studied
    - Generalizability constraints

N.5.4.4 Interpretation Caveats
    - Alternative explanations
    - Model dependence
```

### Option 2: By Severity

```
N.5.4 Limitations

N.5.4.1 Significant Limitations
    - [Major limitation 1 across categories]
    - [Major limitation 2]

N.5.4.2 Moderate Limitations
    - [Medium-impact limitations]

N.5.4.3 Minor Caveats
    - [Brief mention of lesser issues]
```

### Option 3: By Result

```
N.5.4 Limitations

N.5.4.1 Limitations Affecting [Result 1]
    - [Specific limitations]

N.5.4.2 Limitations Affecting [Result 2]
    - [Specific limitations]

N.5.4.3 General Limitations
    - [Cross-cutting issues]
```

---

## Checklist

### Identification
- [ ] All significant limitations identified
- [ ] Prioritized by severity
- [ ] Impact assessed
- [ ] Nothing obvious missed

### Presentation
- [ ] Proportionate emphasis
- [ ] Constructive framing
- [ ] Clear writing
- [ ] Organized logically

### Balance
- [ ] Honest about significant issues
- [ ] Not self-undermining
- [ ] Confident about valid conclusions
- [ ] Appropriate for thesis context

### Integration
- [ ] Connects to methods section
- [ ] Informs future directions
- [ ] Guides interpretation in discussion
- [ ] Addresses what readers would wonder

---

## Examples of Well-Written Limitations

### Example 1: Methodological

> "Randomized benchmarking, while providing robust estimates of average gate fidelity, does not distinguish between coherent and incoherent error channels. This limitation means that our reported fidelity of 99.2% could arise from various combinations of error types. For applications requiring coherent error suppression (e.g., error-detection codes), additional characterization using gate set tomography or related methods would be needed. However, for surface code implementations where average error rates are the primary figure of merit, randomized benchmarking remains the appropriate metric."

### Example 2: Sample

> "All measurements were performed on a single device (Device QC-17). While this device's parameters fall within the typical range for our fabrication process (Table N.2), we cannot definitively establish that our results generalize to other devices without additional measurements. The device-to-device variation in our platform is approximately 20% for coherence times based on historical data, suggesting that gate fidelities may vary by up to 0.1% across devices. Establishing the reproducibility of our results across multiple devices is an important direction for future work."

### Example 3: Scope

> "This study focused on two-qubit gate performance and did not characterize behavior in multi-qubit circuits. As the number of qubits increases, additional error mechanisms (e.g., crosstalk, frequency crowding) become important. Extrapolating our two-qubit results to predict performance at scale requires assumptions about how errors combine that have not been validated experimentally. The scaling of our approach is therefore an open question that will be addressed in the subsequent device generation (Chapter 7)."
