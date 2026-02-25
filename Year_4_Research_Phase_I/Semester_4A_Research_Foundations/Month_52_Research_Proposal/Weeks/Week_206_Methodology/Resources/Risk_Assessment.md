# Risk Assessment for Research Proposals

## Introduction

Risk assessment is not about predicting failure—it's about demonstrating to reviewers that you've thought carefully about what could go wrong and have concrete plans to address it. This guide provides frameworks for systematic risk identification, evaluation, and mitigation.

---

## Part 1: Why Risk Assessment Matters

### Reviewer Perspective

Reviewers evaluate proposals based on likelihood of success. They ask:
- Is this project feasible?
- What could prevent success?
- Has the PI thought through potential problems?
- Are there backup plans?

**Proposals without risk assessment signal:**
- Naivety about research challenges
- Overconfidence that may lead to failure
- Lack of planning for contingencies

**Proposals with strong risk assessment signal:**
- Experience with research realities
- Thoughtful planning
- Resilience to setbacks
- Higher likelihood of producing results

### Balance: Confidence and Realism

```
TOO LITTLE RISK AWARENESS          BALANCED                TOO MUCH RISK AWARENESS
"Everything will work              "X may fail,             "Everything might fail,
 perfectly as planned"              but we have              we're not sure this
                                    backup Y"                will work at all"
        ↓                              ↓                           ↓
   Seems naive                   Seems prepared              Seems uncertain
```

---

## Part 2: Risk Categories

### Technical Risks

Problems with the science or methods:

| Risk Type | Example | Detection | Mitigation |
|-----------|---------|-----------|------------|
| Method failure | Algorithm doesn't converge | Performance metrics | Alternative algorithm |
| Measurement limitation | Resolution insufficient | Preliminary tests | Different technique |
| Model inadequacy | Theory doesn't match data | Validation experiments | Revised model |
| Complexity barriers | Problem is NP-hard | Computational scaling | Approximations |

### Resource Risks

Problems with what you need to do the work:

| Risk Type | Example | Detection | Mitigation |
|-----------|---------|-----------|------------|
| Equipment failure | Cryostat breaks | Downtime | Backup system, vendor support |
| Computing shortage | HPC allocation runs out | Usage monitoring | Cloud backup, optimize code |
| Material unavailability | Custom sample delayed | Lead time tracking | Multiple suppliers |
| Funding shortfall | Budget cut | Award notification | Prioritize core work |

### Personnel Risks

Problems with the research team:

| Risk Type | Example | Detection | Mitigation |
|-----------|---------|-----------|------------|
| Student leaves | Graduation, personal | Enrollment tracking | Overlap periods, documentation |
| Expertise gap | New skill needed | Progress review | Training, collaborator |
| PI unavailability | Sabbatical, illness | Calendar | Co-PI backup |
| Collaborator drops | Priority change | Communication | Written agreements |

### External Risks

Problems outside your control:

| Risk Type | Example | Detection | Mitigation |
|-----------|---------|-----------|------------|
| Facility access | User facility overbooked | Wait times | Multiple facilities |
| Hardware availability | Cloud quantum down | Uptime monitoring | Multiple platforms |
| Competing group | Scooped on key result | Literature monitoring | Differentiation, speed |
| Regulatory change | Export controls | Policy tracking | Legal consultation |

### Timeline Risks

Problems with schedule:

| Risk Type | Example | Detection | Mitigation |
|-----------|---------|-----------|------------|
| Task takes longer | Optimization slow | Progress tracking | Buffer time |
| Dependencies delay | Aim 2 needs Aim 1 | Milestone review | Parallel paths |
| Publication delays | Review takes months | Submission timing | Multiple journals |
| External review | IRB/IACUC slow | Application lead time | Early submission |

---

## Part 3: Risk Assessment Framework

### Step 1: Systematic Identification

Use structured brainstorming for each aim:

**Technical checklist:**
- [ ] What if primary method doesn't work?
- [ ] What if measurements aren't precise enough?
- [ ] What if model assumptions are wrong?
- [ ] What if computational resources are insufficient?
- [ ] What if preliminary data doesn't generalize?

**Resource checklist:**
- [ ] What equipment could break?
- [ ] What materials might be unavailable?
- [ ] What computing resources might be insufficient?
- [ ] What funding might be delayed or reduced?

**Personnel checklist:**
- [ ] What if key team members leave?
- [ ] What expertise gaps exist?
- [ ] What collaboration dependencies exist?

**External checklist:**
- [ ] What facility access issues might arise?
- [ ] What could competitors do?
- [ ] What policy changes could affect us?

### Step 2: Likelihood Assessment

Rate each risk's probability:

| Rating | Definition | Frequency |
|--------|------------|-----------|
| Low | Unlikely to occur | <10% chance |
| Medium | Possible, not certain | 10-50% chance |
| High | More likely than not | >50% chance |

**Criteria for rating:**
- Past experience with similar situations
- Inherent uncertainty in the approach
- Dependencies on external factors
- Complexity of the task

### Step 3: Impact Assessment

Rate each risk's consequence:

| Rating | Definition | Effect on Project |
|--------|------------|-------------------|
| Low | Minor inconvenience | Delay < 1 month, no scope change |
| Medium | Significant setback | Delay 1-6 months, or reduced scope |
| High | Project-threatening | Could prevent achieving aims |

### Step 4: Priority Matrix

Combine likelihood and impact:

```
              IMPACT
           Low  Med  High
         ┌────┬────┬────┐
    Low  │ 1  │ 2  │ 3  │
LIKE-    ├────┼────┼────┤
LIHOOD Med│ 2  │ 3  │ 4  │
         ├────┼────┼────┤
    High │ 3  │ 4  │ 5  │
         └────┴────┴────┘

Priority: 1-2 = Monitor, 3 = Plan, 4-5 = Critical (address in proposal)
```

### Step 5: Mitigation Planning

For each medium+ priority risk:

```
RISK MITIGATION TEMPLATE

Risk ID: R[X]
Description: [What could go wrong]
Category: Technical / Resource / Personnel / External / Timeline
Likelihood: Low / Medium / High
Impact: Low / Medium / High
Priority: [1-5]

Detection Plan:
- How will we know this is happening?
- What are the early warning signs?
- How often will we check?

Prevention Plan:
- What can we do to reduce likelihood?
- What preparations can we make?
- What resources should we secure in advance?

Contingency Plan:
- What will we do if this happens?
- What is the alternative approach?
- What is the impact on timeline/scope?

Owner: [Who is responsible for monitoring/responding]
```

---

## Part 4: Common Risks in Quantum Research

### Quantum Hardware Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Device calibration drift | High | Medium | Frequent recalibration; interleaved controls |
| Coherence time insufficient | Medium | High | Multiple qubit types; pulse optimization |
| Gate fidelity below threshold | Medium | High | Error mitigation; alternative gates |
| Qubit count limits scope | Low | High | Focus on algorithmic insights; emulation |
| Hardware access disrupted | Low | Medium | Multiple platforms; local simulation |

### Quantum Algorithm Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Barren plateaus | High | High | Structured ansatze; layer-wise training |
| Optimization stuck | Medium | Medium | Multiple optimizers; restarts |
| Classical simulation impossible | Medium | Medium | Tensor networks; approximate methods |
| Noise destroys quantum advantage | Medium | High | Error mitigation; fault-tolerant design |
| Competitor publishes first | Low | Medium | Focus on unique aspects; rapid publication |

### Quantum Materials Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Sample quality issues | Medium | High | Multiple growth methods; collaborators |
| Characterization ambiguity | Medium | Medium | Complementary techniques; theory support |
| Measurement artifacts | Medium | Medium | Multiple setups; systematic controls |
| Cryogenic system failure | Low | High | Maintenance contract; backup dewar |
| Material not as predicted | Medium | Medium | Theory iteration; parameter exploration |

### Quantum Theory Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Key assumption wrong | Medium | High | Numerical validation; relaxed assumptions |
| Proof technique fails | Medium | Medium | Alternative proof strategies |
| Problem harder than expected | Medium | Medium | Focus on special cases; bounds |
| Results not publishable | Low | Medium | Pivot to computational/experimental test |
| Scooped on theorem | Low | Medium | Unique applications; generalizations |

---

## Part 5: Writing Risk Sections

### Proposal Format

In NSF/DOE proposals, risk assessment typically appears within each aim:

```
AIM X: [Title]

X.1 Rationale
X.2 Technical Approach
X.3 Expected Outcomes
X.4 Potential Pitfalls and Alternatives  ← RISK SECTION
```

### Structure for Each Risk

**Short form (for minor risks):**
> *Pitfall: [Risk description].* We will mitigate by [strategy]. If necessary, we will pivot to [alternative].

**Long form (for major risks):**
> **Potential Pitfall: [Risk title]**
>
> [Risk description]. This could affect [scope/timeline] by [specific impact].
>
> *Detection:* We will monitor [indicator] with [frequency]. Warning signs include [specifics].
>
> *Mitigation:* To reduce likelihood, we will [strategy]. Preparations include [specifics].
>
> *Alternative Approach:* If this risk materializes, we will [backup plan]. This alternative achieves [subset of goals] with [tradeoffs]. [Reference showing alternative is feasible.]

### Examples

**Example 1: Technical Risk**

> **Potential Pitfall: Machine learning decoder fails to generalize**
>
> The neural network decoder trained on simulated noise may not perform well on actual hardware noise distributions. This could limit the demonstrated advantage in Aim 3.
>
> *Detection:* We will track validation accuracy on held-out hardware data. If accuracy drops below 90% (vs. 98% on simulated data), generalization is failing.
>
> *Mitigation:* We will train on diverse noise models including hardware-calibrated distributions. We will use domain adaptation techniques to transfer from simulation to hardware.
>
> *Alternative Approach:* If learning-based decoding fails, we will use minimum-weight perfect matching (MWPM) with bias-adjusted edge weights. MWPM is guaranteed to find correct decoding (for correctable errors) and has been extensively validated on hardware. The tradeoff is ~20% lower threshold compared to optimal ML decoder, but results would still demonstrate advantage for our tailored codes.

**Example 2: Resource Risk**

> **Potential Pitfall: Cloud quantum access insufficient**
>
> Our experiments require approximately 500K circuit executions per code configuration. If cloud access is limited or queue times excessive, we may not complete all planned measurements.
>
> *Detection:* We will track queue wait times and monthly usage. If average wait exceeds 4 hours or monthly usage drops below 50K shots, we are at risk.
>
> *Mitigation:* We have secured IBM Quantum Research access (letter attached) and applied for IonQ academic credits. We will also batch circuits efficiently and run during off-peak hours.
>
> *Alternative Approach:* If hardware access remains limiting, we will: (1) prioritize highest-value experiments, (2) use hardware emulators for parameter sweeps, and (3) focus on demonstrating proof-of-principle rather than comprehensive characterization. This still allows us to validate the core hypothesis with reduced statistical power.

**Example 3: Personnel Risk**

> **Potential Pitfall: Graduate student transitions**
>
> This project supports two PhD students over 3 years. If a student leaves (graduation, personal reasons), project continuity could be affected.
>
> *Mitigation:* Students will maintain detailed lab notebooks and code documentation. We will have 3-month overlap periods between student cohorts. The postdoc provides continuity across student transitions.
>
> *Alternative Approach:* If a student position becomes vacant, we will: (1) accelerate current student's work on highest-priority tasks, (2) recruit from our established pipeline of rotation students, and (3) temporarily shift effort to the PI and postdoc. Our 10% timeline buffer accommodates 2-month transition periods.

---

## Part 6: Risk Assessment Template

Use this template to systematically assess risks for your proposal:

### Risk Register

| ID | Description | Category | L | I | Priority | Mitigation | Alternative |
|----|-------------|----------|---|---|----------|------------|-------------|
| R1 | | | | | | | |
| R2 | | | | | | | |
| R3 | | | | | | | |
| ... | | | | | | | |

**L = Likelihood:** L (Low), M (Medium), H (High)
**I = Impact:** L (Low), M (Medium), H (High)
**Priority:** 1-5 (from matrix)

### Risk Details (for Priority 3+ risks)

**Risk R[X]: [Title]**

*Description:*
_____________________________________________________________

*Category:* Technical / Resource / Personnel / External / Timeline

*Likelihood:* Low / Medium / High
*Justification:* ____________________________________________

*Impact:* Low / Medium / High
*Affected aims:* ____________________________________________
*Specific consequences:* _____________________________________

*Detection plan:*
- Indicator: ______________________________________________
- Check frequency: ________________________________________
- Warning threshold: ______________________________________

*Prevention plan:*
- Action 1: ______________________________________________
- Action 2: ______________________________________________
- Resources needed: _______________________________________

*Contingency plan:*
- Alternative approach: ____________________________________
- Timeline impact: ________________________________________
- Scope impact: __________________________________________
- Feasibility evidence: ____________________________________

---

## Part 7: Common Mistakes

### Too Vague

**Weak:**
> "Things may not work as planned."

**Strong:**
> "The genetic algorithm may converge to local optima rather than global solutions, particularly for codes exceeding 50 qubits."

### No Alternatives

**Weak:**
> "We will need to be careful with the measurements."

**Strong:**
> "If noise exceeds threshold, we will use error mitigation via zero-noise extrapolation (Richardson extrapolation with 3 noise scale factors)."

### Underestimating Risks

**Weak:**
> "Unlikely to be a problem." (for a common issue)

**Strong:**
> "Hardware calibration drift is common on current devices. We will interleave calibration measurements every 100 circuits and post-select on stable periods."

### Catastrophizing

**Weak:**
> "This could completely derail the project with no recovery."

**Strong:**
> "This risk could delay Aim 2 by 3 months. Our timeline buffer and parallel work structure allow absorption of this delay."

---

## Part 8: Integration with Proposal

### Location in Proposal

```
PROJECT DESCRIPTION
├── Section 1: Introduction/Specific Aims
├── Section 2: Background and Significance
├── Section 3: Research Plan
│   ├── Aim 1
│   │   ├── 1.1 Rationale
│   │   ├── 1.2 Approach
│   │   ├── 1.3 Expected Outcomes
│   │   └── 1.4 Pitfalls and Alternatives  ← HERE
│   ├── Aim 2
│   │   └── ...
│   └── Aim 3
│       └── ...
├── Section 4: Timeline
└── Section 5: Broader Impacts
```

### Word Budget

Typical allocation for a 15-page proposal:
- Risk assessment: 1-1.5 pages total
- Per aim: 0.3-0.5 pages
- Focus on 2-3 highest-priority risks per aim

### Tone

- Confident but realistic
- Specific about problems and solutions
- Evidence-based (cite alternatives working elsewhere)
- Forward-looking (not dwelling on what could go wrong)

---

## Summary Checklist

Before submitting your proposal, verify:

- [ ] Identified risks for each aim (technical, resource, personnel, external)
- [ ] Assessed likelihood and impact systematically
- [ ] Prioritized risks (focus on medium-high priority)
- [ ] Developed specific mitigation strategies
- [ ] Provided concrete alternative approaches
- [ ] Included evidence that alternatives are feasible
- [ ] Distributed risk discussion appropriately in proposal
- [ ] Maintained confident but realistic tone

---

*"Showing you've planned for failure is the best evidence you're prepared for success."*
