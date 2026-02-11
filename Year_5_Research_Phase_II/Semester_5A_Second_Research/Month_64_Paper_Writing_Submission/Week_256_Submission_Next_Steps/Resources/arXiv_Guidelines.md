# arXiv Guidelines for Quantum Computing Papers

## Complete Reference for arXiv Preprint Submission

---

## What is arXiv?

arXiv.org is a free, open-access repository of electronic preprints (e-prints) for scientific research. Founded in 1991, it has become the standard venue for sharing physics, mathematics, computer science, and related research before (or alongside) peer-reviewed publication.

### Why Submit to arXiv?

1. **Priority establishment**: Date-stamped record of your research
2. **Rapid dissemination**: Available within 1-2 days
3. **Open access**: Free for all readers
4. **Community feedback**: Get input before formal publication
5. **Visibility**: Widely read in physics/quantum community
6. **Citation**: arXiv papers are citable

---

## arXiv Categories for Quantum Computing

### Primary Categories

**quant-ph (Quantum Physics)**
- Most common for quantum computing papers
- Covers: quantum information, quantum computation, quantum algorithms, quantum error correction, quantum communication
- Audience: Broad quantum information science community

**cond-mat.mes-hall (Mesoscopic Systems and Quantum Hall Effect)**
- For papers on physical implementations
- Covers: superconducting qubits, semiconductor qubits, topological systems
- Audience: Experimental condensed matter + device physics

**cond-mat.str-el (Strongly Correlated Electrons)**
- For papers on many-body quantum systems
- Covers: quantum simulation, tensor networks, strongly correlated systems
- Audience: Condensed matter theory community

**cond-mat.stat-mech (Statistical Mechanics)**
- For papers involving thermodynamics or statistical physics
- Covers: quantum thermodynamics, open quantum systems
- Audience: Statistical physics community

**physics.atom-ph (Atomic Physics)**
- For papers on atomic/molecular implementations
- Covers: trapped ions, neutral atoms, molecular qubits
- Audience: AMO physics community

**cs.IT (Information Theory)**
- For papers with heavy information theory content
- Covers: quantum channel capacity, quantum coding theory
- Audience: Information/coding theory community

### Cross-Listing Strategy

**When to cross-list**:
- Paper is relevant to multiple communities
- You want visibility in related fields
- Different aspects appeal to different audiences

**Recommended cross-listings for quantum computing**:

| Primary Category | Common Cross-Lists |
|-----------------|-------------------|
| quant-ph | cond-mat.mes-hall, physics.atom-ph, cs.IT |
| cond-mat.mes-hall | quant-ph, physics.ins-det |
| cond-mat.str-el | quant-ph, cond-mat.stat-mech |
| physics.atom-ph | quant-ph, physics.optics |
| cs.IT | quant-ph, math.IT |

**Maximum cross-lists**: Generally 3-4 is reasonable

---

## Endorsement System

### Who Needs Endorsement?

First-time submitters to a category need endorsement from an existing author in that category.

**You DO NOT need endorsement if**:
- You have previously submitted to that category
- You are affiliated with an institution that has automatic endorsement
- Your co-author has submission rights in that category

**You DO need endorsement if**:
- First submission to category
- No institutional affiliation recognized by arXiv
- No endorsed co-author

### Getting Endorsed

**Step 1: Identify potential endorsers**
- Co-authors with arXiv history
- Colleagues at your institution
- Researchers whose work you cite
- People who might be interested in your work

**Step 2: Request endorsement**
- Contact potential endorser professionally
- Explain your research briefly
- Provide arXiv with their information
- Endorser receives automated email

**Step 3: Endorser action**
- Endorser verifies you are a researcher
- Endorser clicks link in email to endorse
- You receive notification

**Endorsement request template**:

```
Subject: arXiv Endorsement Request for [Category]

Dear [Name],

I am a PhD student at [Institution] working on [brief topic].
I am preparing to submit my first paper to arXiv in the
[category] category and need an endorsement to do so.

Would you be willing to endorse my submission? The paper
is about [one sentence description]. A draft is available
at [link or attached].

If you're willing, I can initiate the endorsement request
through arXiv's system, which will send you an email with
a simple verification link.

Thank you for considering this request.

Best regards,
[Your name]
```

---

## Submission Format Guidelines

### Source Files (Preferred)

arXiv strongly prefers source files (LaTeX) over PDF.

**Benefits of source submission**:
- arXiv can generate high-quality PDF
- Figures rendered at optimal resolution
- Full-text search enabled
- Format updated as technologies improve

**Required files**:
- Main .tex file
- All included packages (.sty) if non-standard
- All figures in supported formats
- Bibliography file (.bib)
- Any custom class files

**File naming**:
- Use simple characters (a-z, 0-9, -, _)
- No spaces in file names
- Keep file names short
- Be consistent with case (arXiv is case-sensitive)

### PDF Submission (Alternative)

Submit PDF only if:
- Complex formatting that doesn't compile on arXiv
- Non-LaTeX source (Word, etc.)
- Special circumstances

**PDF requirements**:
- PDF/A format preferred
- All fonts embedded
- No security restrictions
- File size under limit

### Size Limits

| Item | Limit |
|------|-------|
| Total submission | 50 MB (expandable on request) |
| Individual file | 50 MB |
| Ancillary files | 10 MB each |

---

## Metadata Best Practices

### Title

**Do**:
- Use exact title from paper
- Include LaTeX math for equations: "Dynamics of $\ket{\psi}$ in..."
- Use standard capitalization

**Don't**:
- Add formatting not in paper
- Use "Draft" or "Preliminary" in title
- Use abbreviations not in paper

### Authors

**Format**: First Last, First Last, First Last

**Best practices**:
- List all authors
- Match order in paper
- Use consistent name format
- Consider ORCID integration

**Do NOT include**:
- Affiliations (not displayed)
- Email addresses
- Footnotes or special characters

### Abstract

**arXiv abstracts appear**:
- On the abstract page
- In email announcements
- In RSS feeds
- In search results

**Best practices**:
- Self-contained (no citations)
- LaTeX math supported ($...$)
- Keep under 1920 characters
- No HTML formatting

**Math in abstract**:
```
We demonstrate fidelity $\mathcal{F} > 99\%$ for the
controlled-$Z$ gate using superconducting qubits.
```

### Comments

**Useful comments include**:
- Page count: "15 pages"
- Figure count: "4 figures"
- Submission status: "Submitted to Physical Review Letters"
- Version changes: "v2: Corrected Eq. (5)"
- Related papers: "This paper supersedes arXiv:XXXX.XXXXX"

### Report Number

**Include if**:
- Your institution assigns report numbers
- Funding agency requires it

**Examples**:
- "MIT-CTP-5432"
- "CERN-TH-2024-001"

---

## Submission Timeline

### Daily Schedule

arXiv operates on Eastern Time (ET):

| Deadline | Action |
|----------|--------|
| 2:00 PM ET | Cutoff for next-day announcement |
| 8:00 PM ET | New papers announced |

**Weekday flow**:
- Submit by 2:00 PM ET Monday → Announced Monday 8:00 PM ET
- Submit at 3:00 PM ET Monday → Announced Tuesday 8:00 PM ET

**Weekend flow**:
- Papers submitted Saturday/Sunday → Announced Monday 8:00 PM ET

### Moderation

All submissions undergo moderation:
- Most papers: 1 business day
- Some papers: 2-3 business days
- Rarely: Placed on hold for review

**Moderation checks**:
- Appropriate category
- Scientific content
- Formatting compliance
- Not spam/duplicate

---

## Updating Submissions

### Versions

arXiv keeps all versions accessible:
- v1: Original submission
- v2, v3, etc.: Updates

**When to update**:
- Errors discovered
- Revisions after review
- Adding journal reference
- Significant changes

**How to update**:
1. Go to your arXiv user page
2. Select the submission
3. Click "Add a new version"
4. Upload new files
5. Add comments explaining changes

### Journal Reference

After publication, add journal reference:
- Updates paper's metadata
- Does not require new version
- Links to published version

**To add journal reference**:
1. Go to your arXiv user page
2. Select the submission
3. Click "Add journal reference"
4. Enter citation details and DOI

---

## Common Issues and Solutions

### Compilation Failures

| Error | Solution |
|-------|----------|
| "Package not found" | Include .sty file or use standard package |
| "File not found" | Check file names (case-sensitive) |
| "Font not found" | Use standard Computer Modern fonts |
| "Runaway argument" | Check for unclosed braces |
| "Missing $ inserted" | Check math mode delimiters |

### Moderation Holds

**Reasons for holds**:
- Category mismatch
- Content concerns
- Duplicate submission
- Formatting issues

**Resolution**:
- Wait (usually resolved in 1-2 days)
- Respond if contacted
- Contact arXiv if extended delay

### Category Issues

**If placed in wrong category**:
- Can request recategorization
- Explain why your choice is more appropriate
- Provide evidence (e.g., related papers)

---

## arXiv and Journal Policies

### Journal Compatibility

Most physics journals accept arXiv preprints:
- APS journals: Fully compatible
- Nature portfolio: Allowed
- Science: Allowed
- IEEE: Allowed with some restrictions

**Check journal policy before submission**

### Updating After Publication

**Best practice**:
1. Post initial version before/at journal submission
2. Update with revised version after review
3. Add journal reference upon publication
4. Keep arXiv version as close to published as copyright allows

---

## Tips for Quantum Computing Papers

### Visibility Optimization

1. **Choose primary category carefully**: quant-ph has highest visibility in quantum computing community

2. **Cross-list strategically**: Include relevant cond-mat or cs categories

3. **Time announcement**: Monday/Tuesday announcements get more attention than Friday

4. **Write clear abstract**: First few sentences are crucial for browsing readers

### Building Your arXiv Profile

1. **Use consistent author name**: Same format across all papers

2. **Link ORCID**: Improves discoverability

3. **Maintain author list**: Keep track of arXiv author ID

4. **Update regularly**: Keep versions current with journal

---

## Useful arXiv Links

- **arXiv homepage**: https://arxiv.org
- **Submit new paper**: https://arxiv.org/submit
- **User account**: https://arxiv.org/user
- **Help documentation**: https://info.arxiv.org/help
- **Category taxonomy**: https://arxiv.org/category_taxonomy
- **RSS feeds**: https://arxiv.org/list/quant-ph/new
- **arXiv API**: https://info.arxiv.org/help/api

---

*arXiv is a cornerstone of modern physics research. Proper use establishes priority, enables rapid dissemination, and builds your research profile.*
