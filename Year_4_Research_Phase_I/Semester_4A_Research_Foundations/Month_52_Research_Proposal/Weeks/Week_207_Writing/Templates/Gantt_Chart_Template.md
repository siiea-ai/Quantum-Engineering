# Gantt Chart and Timeline Template

## Instructions

Use this template to create your project timeline. The Gantt chart provides visual representation of task scheduling, while the milestone table defines success criteria.

---

## Part 1: Project Phases Overview

### High-Level Timeline

**Project Duration:** _____ years (_____ months)

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 1 | Months 1-___ | _________________________ |
| Phase 2 | Months ___-___ | _________________________ |
| Phase 3 | Months ___-___ | _________________________ |
| Phase 4 | Months ___-___ | _________________________ |

---

## Part 2: Detailed Task List

### Aim 1: [Title]

| Task ID | Task Description | Start | End | Dependencies | Personnel |
|---------|-----------------|-------|-----|--------------|-----------|
| 1.1 | _________________ | M__ | M__ | None | _________ |
| 1.2 | _________________ | M__ | M__ | 1.1 | _________ |
| 1.3 | _________________ | M__ | M__ | 1.1, 1.2 | _________ |
| 1.4 | _________________ | M__ | M__ | 1.2 | _________ |
| 1.5 | _________________ | M__ | M__ | 1.3, 1.4 | _________ |

**Milestone:** _________________________________ (Month ___)

---

### Aim 2: [Title]

| Task ID | Task Description | Start | End | Dependencies | Personnel |
|---------|-----------------|-------|-----|--------------|-----------|
| 2.1 | _________________ | M__ | M__ | 1.2 | _________ |
| 2.2 | _________________ | M__ | M__ | 2.1 | _________ |
| 2.3 | _________________ | M__ | M__ | 2.1, 2.2 | _________ |
| 2.4 | _________________ | M__ | M__ | 2.3 | _________ |

**Milestone:** _________________________________ (Month ___)

---

### Aim 3: [Title]

| Task ID | Task Description | Start | End | Dependencies | Personnel |
|---------|-----------------|-------|-----|--------------|-----------|
| 3.1 | _________________ | M__ | M__ | 1.5, 2.2 | _________ |
| 3.2 | _________________ | M__ | M__ | 3.1 | _________ |
| 3.3 | _________________ | M__ | M__ | 3.1, 3.2 | _________ |
| 3.4 | _________________ | M__ | M__ | 3.3 | _________ |

**Milestone:** _________________________________ (Month ___)

---

### Cross-Cutting Activities

| Task ID | Task Description | Start | End | Frequency |
|---------|-----------------|-------|-----|-----------|
| X.1 | Project management | M1 | M__ | Continuous |
| X.2 | Publication preparation | M__ | M__ | As completed |
| X.3 | Conference presentations | M__ | M__ | Annual |
| X.4 | Progress reporting | M1 | M__ | Quarterly |
| X.5 | Software releases | M__ | M__ | Annual |

---

## Part 3: Gantt Chart

### ASCII Gantt Chart

```
                    Year 1                  Year 2                  Year 3
Task               Q1   Q2   Q3   Q4   Q1   Q2   Q3   Q4   Q1   Q2   Q3   Q4
                   M1-3 M4-6 M7-9 M10-12 M13-15 M16-18 M19-21 M22-24 M25-27 M28-30 M31-33 M34-36
──────────────────────────────────────────────────────────────────────────────────────────────────
AIM 1: [Title]
 1.1 [Task]        ████
 1.2 [Task]        ──── ████████
 1.3 [Task]             ──── ████████
 1.4 [Task]                  ████████
 1.5 [Task]                       ████████
                                      ◆ M1

AIM 2: [Title]
 2.1 [Task]                       ████████
 2.2 [Task]                            ──── ████████████
 2.3 [Task]                                      ████████████
 2.4 [Task]                                           ████████
                                                          ◆ M2

AIM 3: [Title]
 3.1 [Task]                                           ████████████
 3.2 [Task]                                                ──── ████████████
 3.3 [Task]                                                          ████████████
 3.4 [Task]                                                               ████████
                                                                               ◆ M3

CROSS-CUTTING
 Publications      ────────── ◇ ──────────────── ◇ ──────────────── ◇ ──────── ◇
 Presentations          ▽                   ▽                   ▽
 Software                    ■                        ■                        ■

──────────────────────────────────────────────────────────────────────────────────────────────────
LEGEND:
████ = Active work period
──── = Preparation/setup period
◆    = Major milestone
◇    = Publication
▽    = Conference presentation
■    = Software release
```

---

### Alternative Format: Markdown Table Gantt

| Task | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:---:|
| **Aim 1** | | | | | | | | | | | | |
| 1.1 [Name] | X | X | X | | | | | | | | | |
| 1.2 [Name] | | | X | X | X | X | | | | | | |
| 1.3 [Name] | | | | | X | X | X | | | | | |
| **Aim 2** | | | | | | | | | | | | |
| 2.1 [Name] | | | | | | X | X | X | | | | |
| 2.2 [Name] | | | | | | | | X | X | X | | |
| **Aim 3** | | | | | | | | | | X | X | X |

Continue for Year 2, Year 3...

---

## Part 4: Milestone Definitions

### Milestone Table

| ID | Milestone | Success Criteria | Target Date | Verification Method |
|----|-----------|------------------|-------------|---------------------|
| M1 | __________ | _________________ | Month ___ | __________________ |
| M2 | __________ | _________________ | Month ___ | __________________ |
| M3 | __________ | _________________ | Month ___ | __________________ |
| M4 | __________ | _________________ | Month ___ | __________________ |
| M5 | __________ | _________________ | Month ___ | __________________ |
| M6 | __________ | _________________ | Month ___ | __________________ |

---

### Milestone Details

**Milestone M1: [Title]**
- Target Date: Month ___
- Description: _____________________________________________________________
- Success Criteria:
  1. ______________________________________________________________________
  2. ______________________________________________________________________
  3. ______________________________________________________________________
- Verification: How will you demonstrate achievement?
  _________________________________________________________________________
- Dependencies: What must be completed first?
  _________________________________________________________________________
- Risk if delayed: What is the impact on subsequent work?
  _________________________________________________________________________

---

**Milestone M2: [Title]**
- Target Date: Month ___
- Description: _____________________________________________________________
- Success Criteria:
  1. ______________________________________________________________________
  2. ______________________________________________________________________
- Verification: _____________________________________________________________
- Dependencies: _____________________________________________________________
- Risk if delayed: __________________________________________________________

---

**Milestone M3: [Title]**
- Target Date: Month ___
- Description: _____________________________________________________________
- Success Criteria:
  1. ______________________________________________________________________
  2. ______________________________________________________________________
- Verification: _____________________________________________________________
- Dependencies: _____________________________________________________________
- Risk if delayed: __________________________________________________________

---

## Part 5: Deliverables Schedule

### Deliverables Table

| ID | Deliverable | Type | Target Date | Recipient |
|----|-------------|------|-------------|-----------|
| D1 | ____________ | Publication | Month ___ | Journal: _______ |
| D2 | ____________ | Software | Month ___ | GitHub/Zenodo |
| D3 | ____________ | Dataset | Month ___ | Repository: _____ |
| D4 | ____________ | Publication | Month ___ | Journal: _______ |
| D5 | ____________ | Report | Month ___ | Funding agency |
| D6 | ____________ | Thesis | Month ___ | University |

---

### Deliverable Details

**D1: [Publication Title]**
- Type: Journal article / Conference paper / Preprint
- Target Journal/Conference: _________________________________________________
- Target Date: Month ___
- Authors: __________________________________________________________________
- Status: Outline / Draft / Submitted / In revision / Published

---

## Part 6: Personnel Timeline

### Effort Distribution

| Team Member | Role | Y1 Effort | Y2 Effort | Y3 Effort | Primary Aims |
|-------------|------|-----------|-----------|-----------|--------------|
| PI | Lead | ___% | ___% | ___% | All |
| Postdoc | Research | ___% | ___% | ___% | Aims __ |
| Student 1 | PhD | ___% | ___% | ___% | Aims __ |
| Student 2 | PhD | ___% | ___% | ___% | Aims __ |
| Collaborator | Advisor | ___% | ___% | ___% | Aim __ |

---

### Personnel Transitions

| Event | Date | Impact | Mitigation |
|-------|------|--------|------------|
| Student 1 starts | Month ___ | Ramp-up period | PI coverage |
| Postdoc starts | Month ___ | Expertise addition | Training plan |
| Student 2 starts | Month ___ | Capacity increase | Mentoring |
| Student 1 transitions | Month ___ | Aim X handoff | 3-month overlap |
| Postdoc transitions | Month ___ | Aim Y handoff | Documentation |

---

## Part 7: Timeline Risk Assessment

### Schedule Risks

| Risk | Affected Tasks | Likelihood | Impact | Buffer |
|------|----------------|------------|--------|--------|
| Task 1.2 delay | 1.3, 1.4, 2.1 | Medium | High | 2 months |
| Hardware access | 3.1, 3.2 | Low | Medium | 1 month |
| Publication delay | D1, D2 | Medium | Low | 3 months |
| Personnel transition | 2.3, 2.4 | Low | Medium | 1 month |

---

### Critical Path

The critical path is the longest sequence of dependent tasks:

```
Task 1.1 → Task 1.2 → Task 2.1 → Task 2.3 → Task 3.1 → Task 3.3 → Task 3.4
```

**Minimum project duration:** ___ months

**Buffer included:** ___ months

**Parallel paths that reduce risk:**
1. ________________________________________________________________________
2. ________________________________________________________________________

---

## Part 8: Timeline Narrative

*Use this section to write the timeline narrative for your proposal (0.5-1 page)*

### Draft Timeline Narrative

The proposed research will be conducted over ___ years, organized around ___ major aims and ___ key milestones.

**Year 1** focuses on _______________________________________________. During Q1-Q2, we will _______________________________________________. By Month ___, we expect to achieve Milestone 1: _______________________________________________. Q3-Q4 will be devoted to _______________________________________________.

**Year 2** transitions to _______________________________________________. Building on the results from Aim 1, we will _______________________________________________. Milestone 2, targeting _______________________________________________, is scheduled for Month ___. Parallel work on Aim 3 begins in Q3, including _______________________________________________.

**Year 3** emphasizes _______________________________________________. The primary focus is _______________________________________________, culminating in Milestone 3 at Month ___. Final activities include _______________________________________________.

This timeline includes ___-month buffers at each major transition to accommodate unexpected delays. Tasks are structured so that _______________________________________________, reducing overall project risk.

---

## Timeline Checklist

Before finalizing your timeline:

### Completeness
- [ ] All tasks from methodology are included
- [ ] All milestones are defined with success criteria
- [ ] All deliverables are scheduled
- [ ] Personnel assignments are clear

### Feasibility
- [ ] Task durations are realistic
- [ ] Dependencies are correctly identified
- [ ] Parallel activities don't exceed capacity
- [ ] Buffer time is included (10-15%)

### Clarity
- [ ] Gantt chart is readable
- [ ] Milestones are specific and measurable
- [ ] Timeline narrative explains the logic
- [ ] Critical path is identified

### Alignment
- [ ] Timeline matches methodology
- [ ] Personnel effort matches task assignments
- [ ] Resources (computing, equipment) are available when needed
- [ ] Milestones align with progress reports

---

*"A realistic timeline shows you understand the work. Include buffer, or reality will provide it for you."*
