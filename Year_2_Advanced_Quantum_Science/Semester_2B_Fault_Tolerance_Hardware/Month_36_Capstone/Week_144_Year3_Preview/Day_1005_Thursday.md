# Day 1005: Literature Review Methodology

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Systematic search strategies, reading techniques |
| Afternoon | 2 hours | Paper organization, annotation methods |
| Evening | 2 hours | Synthesis and writing, practical literature review |

## Learning Objectives

By the end of today, you will be able to:

1. **Execute** systematic literature searches using multiple databases
2. **Apply** the three-pass reading method to efficiently extract information
3. **Organize** papers using reference management tools and annotation systems
4. **Synthesize** literature into coherent narratives
5. **Write** effective literature review sections
6. **Maintain** an ongoing reading and annotation practice

## Core Content

### 1. The Purpose of Literature Review

A literature review serves multiple functions in research:

**For Understanding:**
- Map the current state of knowledge
- Identify key concepts and definitions
- Understand methodological approaches
- Recognize open problems and gaps

**For Positioning:**
- Place your work in context
- Justify your research questions
- Demonstrate your expertise
- Avoid duplicating existing work

**For the Qualifying Exam:**
- Show breadth of knowledge
- Support oral exam discussions
- Provide evidence for proposals
- Demonstrate research readiness

$$\boxed{\text{Good Review} = \text{Comprehensive Search} + \text{Critical Analysis} + \text{Synthesis}}$$

### 2. Systematic Search Strategy

A systematic search ensures you find all relevant literature, not just what you already know.

#### Step 1: Define Scope

Before searching, define:
- **Topic boundaries**: What is included/excluded?
- **Time range**: Recent only or historical?
- **Publication types**: Journals, conferences, preprints?
- **Keywords**: Primary and alternative terms

**Example Scope Definition:**
```
Topic: Neural network decoders for quantum error correction
Boundaries: Focus on surface/topological codes, include classical ML
Time range: 2015-2026 (post-deep learning)
Types: Journal articles, conference papers, arXiv preprints
Keywords:
  Primary: neural network decoder, machine learning QEC
  Secondary: deep learning surface code, MWPM alternatives
```

#### Step 2: Search Databases

**Primary Databases for Quantum Computing:**

| Database | Strengths | Access |
|----------|-----------|--------|
| **arXiv quant-ph** | Preprints, most current | Free |
| **Google Scholar** | Broad coverage, citations | Free |
| **Web of Science** | Citation analysis | Institutional |
| **IEEE Xplore** | Engineering/CS | Institutional |
| **APS Journals** | Physics publications | Institutional |

**Search Strategy:**
```
Database: arXiv
Search:
  ("neural network" OR "machine learning" OR "deep learning")
  AND
  ("decoder" OR "decoding")
  AND
  ("surface code" OR "topological" OR "quantum error correction")
Date: 2015-2026
```

#### Step 3: Snowball Expansion

Once you have initial papers:
1. **Forward snowballing**: Who cited this paper?
2. **Backward snowballing**: What did this paper cite?
3. **Author tracking**: What else have key authors written?

#### Step 4: Filter and Prioritize

**Filtering Criteria:**
- Relevance to your specific question
- Publication venue quality
- Citation count (with recency adjustment)
- Author reputation
- Recency for rapidly evolving fields

### 3. The Three-Pass Reading Method

Not every paper needs deep reading. Use a three-pass approach:

#### Pass 1: Survey (5-10 minutes)

**Goal:** Decide if paper is relevant

**Read:**
- Title and abstract
- Introduction (first paragraph)
- Section headings
- Figures and captions
- Conclusions (first paragraph)

**Questions to answer:**
- What is the main contribution?
- Is it relevant to my work?
- Should I read it in depth?

**Outcome:** Keep, discard, or save for later

#### Pass 2: Comprehension (30-60 minutes)

**Goal:** Understand main content without details

**Read:**
- Full introduction
- Methods overview (not details)
- Results and figures
- Full discussion
- Conclusions

**Ignore:**
- Mathematical derivations
- Implementation details
- Appendices

**Questions to answer:**
- What problem does it solve?
- What is the approach?
- What are the key results?
- What are the limitations?

**Outcome:** Annotated summary

#### Pass 3: Mastery (2-4 hours)

**Goal:** Complete understanding, could reproduce

**Read:**
- Everything
- Verify proofs and derivations
- Understand all implementation details
- Study supplementary materials

**Actions:**
- Take detailed notes
- Work through key equations
- Identify assumptions
- Consider extensions

**When to use:** Only for papers central to your research

### 4. Paper Annotation System

Develop a consistent annotation system for efficient review:

#### Annotation Categories

| Category | Symbol | Meaning |
|----------|--------|---------|
| Key result | **!** | Important finding |
| Methodology | **M** | Approach or technique |
| Definition | **D** | Term or concept defined |
| Citation needed | **[?]** | Check referenced paper |
| Question | **?** | Unclear or needs verification |
| Connection | **→** | Links to your work |
| Critique | **X** | Limitation or error |

#### Summary Template

For each paper, create a structured summary:

```markdown
## Paper: [Title]
**Authors:** [Names] | **Year:** [Year] | **Venue:** [Journal/Conference]

### Summary (2-3 sentences)
[What the paper does and main finding]

### Key Contributions
1. [Contribution 1]
2. [Contribution 2]
3. [Contribution 3]

### Methodology
[Brief description of approach]

### Key Results
- [Result 1]
- [Result 2]

### Limitations
- [Limitation 1]
- [Limitation 2]

### Connection to My Research
[How it relates to your work]

### Key Quotes
> "[Important quote]" (p. X)

### Citations to Follow
- [Paper to read next]
```

### 5. Reference Management

Use tools to organize your growing library:

#### Tool Options

| Tool | Strengths | Cost |
|------|-----------|------|
| **Zotero** | Open source, browser integration | Free |
| **Mendeley** | PDF annotation, cloud sync | Free/Paid |
| **Papers** | macOS native, beautiful UI | Paid |
| **EndNote** | Institutional standard | Paid |
| **BibDesk** | BibTeX native, lightweight | Free |

#### Organization Strategies

**Folder Structure:**
```
Literature/
├── By Topic/
│   ├── Neural_Decoders/
│   ├── Surface_Codes/
│   ├── Fault_Tolerance/
│   └── Hardware/
├── By Project/
│   ├── Proposal/
│   ├── Paper_1/
│   └── Thesis/
└── To_Read/
    ├── Priority_High/
    └── Priority_Low/
```

**Tagging System:**
- `#core` - Central to your research
- `#background` - General knowledge
- `#methods` - Useful techniques
- `#data` - Datasets or benchmarks
- `#review` - Survey papers
- `#toread` - Not yet processed

### 6. Synthesizing Literature

Moving from individual papers to coherent narrative:

#### Synthesis Approaches

**Chronological:**
Order by development of ideas over time.
- Good for: Historical context, evolution of field
- Example: "Early decoders (2016-2018) focused on... More recent work (2022-2025) has..."

**Thematic:**
Group by topic or approach.
- Good for: Comparing methods, identifying gaps
- Example: "Three main approaches exist: supervised learning, reinforcement learning, and hybrid methods..."

**Methodological:**
Organize by research methods used.
- Good for: Methodology sections, justifying approach
- Example: "Previous studies have used either simulated data or hardware experiments..."

**Theoretical:**
Structure by conceptual frameworks.
- Good for: Theory-heavy areas
- Example: "Information-theoretic bounds constrain decoder performance, while complexity theory limits computational approaches..."

#### Synthesis Matrix

Create a matrix to compare papers:

| Paper | Method | Dataset | Results | Limitations |
|-------|--------|---------|---------|-------------|
| Smith 2023 | CNN | Simulated | 0.5% threshold | Small codes only |
| Jones 2024 | Transformer | Hardware | Real-time | High resource |
| Lee 2025 | GNN | Both | Generalizable | Training cost |

### 7. Writing the Literature Review

Transform synthesis into polished prose:

#### Structure Options

**Funnel Structure:**
Broad context → Specific topic → Your contribution

**Controversy Structure:**
Competing views → Evidence for each → Your position

**Problem-Solution Structure:**
Problem description → Attempted solutions → Remaining gaps

#### Writing Techniques

**1. Use Topic Sentences**
Each paragraph starts with its main point:
> "Neural network decoders have achieved significant improvements in decoding accuracy over traditional methods."

**2. Compare and Contrast**
Show relationships between works:
> "While Smith et al. focused on accuracy, Jones et al. prioritized latency, revealing a fundamental tradeoff."

**3. Identify Patterns**
Highlight trends across papers:
> "A common theme across recent work is the tension between decoder performance and hardware constraints."

**4. Maintain Critical Voice**
Don't just summarize - evaluate:
> "Although these results are promising, they rely on simplified noise models that may not capture realistic conditions."

**5. Connect to Your Work**
Show why this matters for your research:
> "These limitations motivate our approach, which addresses both accuracy and latency within realistic noise budgets."

### 8. Staying Current

Research is ongoing - maintain your literature awareness:

**Daily/Weekly Practices:**
- arXiv email alerts (daily digest)
- Google Scholar alerts for key terms
- Follow key researchers on social media/ResearchGate
- QIP and other conference proceedings

**Monthly Practices:**
- Review journal tables of contents
- Update reading list
- Revisit and update annotations
- Identify new search terms

## Connections to Year 2 Knowledge

### Literature in QEC Research

Your Year 2 knowledge helps evaluate QEC papers:

**Key Papers You Should Know:**
1. Kitaev's surface code paper (1997)
2. Dennis et al. threshold analysis (2002)
3. Fowler et al. surface code review (2012)
4. Google's error correction experiments (2023-2025)

**Critical Reading Questions:**
- What noise model is assumed?
- Is the threshold realistic for hardware?
- How does decoder latency compare to error rates?

### Literature in Algorithm Research

For algorithm papers, evaluate:
- What problem instance sizes are tested?
- Is comparison to classical fair?
- Are resources (qubits, gates, depth) realistic?
- What noise model is used?

## Practical Exercises

### Exercise 1: Systematic Search

Conduct a systematic search on a topic:

**Topic:** _______________________

**Step 1: Define scope**
- Boundaries: _______________________
- Time range: _______________________
- Types: _______________________
- Keywords: _______________________

**Step 2: Search results**
| Database | Query | Results |
|----------|-------|---------|
| arXiv | | |
| Google Scholar | | |

**Step 3: Top 10 papers identified**
1. _______________________
2. _______________________
...

### Exercise 2: Three-Pass Reading

Apply three-pass method to one paper:

**Paper Title:** _______________________

**Pass 1 (5 min):**
- Main contribution: _______________________
- Relevant? (Y/N): _______
- Deep read? (Y/N): _______

**Pass 2 (30 min):**
- Problem: _______________________
- Approach: _______________________
- Key results: _______________________
- Limitations: _______________________

### Exercise 3: Literature Synthesis

Create a synthesis matrix for 5 papers:

| Paper | Method | Contribution | Limitation | Connection |
|-------|--------|--------------|------------|------------|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

**Synthesis paragraph:**
_______________________

### Exercise 4: Review Section Draft

Write a 500-word literature review section:

**Topic:** _______________________

**Draft:**
_______________________

## Computational Lab: Literature Management Tools

```python
"""
Day 1005 Computational Lab: Literature Review Tools
Search, organization, and synthesis utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json

# =============================================================================
# Part 1: Paper Database
# =============================================================================

@dataclass
class Paper:
    """Represent a research paper."""
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    doi: str = ""
    arxiv_id: str = ""
    citations: int = 0
    read_status: str = "unread"  # unread, pass1, pass2, pass3
    relevance: int = 0  # 1-5 scale
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate brief summary string."""
        authors_str = self.authors[0] + " et al." if len(self.authors) > 1 else self.authors[0]
        return f"{authors_str} ({self.year}): {self.title[:50]}..."

    def bibtex_key(self) -> str:
        """Generate BibTeX citation key."""
        first_author = self.authors[0].split()[-1].lower()
        return f"{first_author}{self.year}"

class LiteratureDatabase:
    """Manage a collection of papers."""

    def __init__(self):
        self.papers: Dict[str, Paper] = {}
        self.tags: Dict[str, List[str]] = {}  # tag -> list of paper IDs

    def add_paper(self, paper: Paper) -> str:
        """Add paper to database, return ID."""
        paper_id = paper.bibtex_key()
        # Handle duplicates
        counter = 1
        while paper_id in self.papers:
            paper_id = f"{paper.bibtex_key()}{chr(ord('a') + counter)}"
            counter += 1

        self.papers[paper_id] = paper

        # Update tag index
        for tag in paper.tags:
            if tag not in self.tags:
                self.tags[tag] = []
            self.tags[tag].append(paper_id)

        return paper_id

    def search(self, query: str, field: str = "all") -> List[Paper]:
        """Search papers by query."""
        results = []
        query_lower = query.lower()

        for paper in self.papers.values():
            match = False
            if field in ["all", "title"]:
                if query_lower in paper.title.lower():
                    match = True
            if field in ["all", "abstract"]:
                if query_lower in paper.abstract.lower():
                    match = True
            if field in ["all", "keywords"]:
                if any(query_lower in kw.lower() for kw in paper.keywords):
                    match = True
            if field in ["all", "authors"]:
                if any(query_lower in author.lower() for author in paper.authors):
                    match = True

            if match:
                results.append(paper)

        return results

    def get_by_tag(self, tag: str) -> List[Paper]:
        """Get all papers with a specific tag."""
        paper_ids = self.tags.get(tag, [])
        return [self.papers[pid] for pid in paper_ids if pid in self.papers]

    def get_unread(self) -> List[Paper]:
        """Get papers not yet read."""
        return [p for p in self.papers.values() if p.read_status == "unread"]

    def statistics(self) -> Dict:
        """Get database statistics."""
        papers = list(self.papers.values())
        return {
            'total_papers': len(papers),
            'by_status': {
                'unread': sum(1 for p in papers if p.read_status == 'unread'),
                'pass1': sum(1 for p in papers if p.read_status == 'pass1'),
                'pass2': sum(1 for p in papers if p.read_status == 'pass2'),
                'pass3': sum(1 for p in papers if p.read_status == 'pass3'),
            },
            'by_year': {},
            'by_relevance': {i: sum(1 for p in papers if p.relevance == i)
                           for i in range(1, 6)},
            'total_tags': len(self.tags)
        }

    def visualize_statistics(self):
        """Create visualization of database statistics."""
        stats = self.statistics()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Reading status
        ax1 = axes[0, 0]
        statuses = list(stats['by_status'].keys())
        counts = list(stats['by_status'].values())
        colors = ['lightgray', 'lightyellow', 'lightblue', 'lightgreen']
        ax1.pie(counts, labels=statuses, colors=colors, autopct='%1.1f%%',
               startangle=90)
        ax1.set_title('Reading Progress')

        # Relevance distribution
        ax2 = axes[0, 1]
        relevances = list(stats['by_relevance'].keys())
        counts = list(stats['by_relevance'].values())
        ax2.bar(relevances, counts, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Relevance (1-5)')
        ax2.set_ylabel('Number of Papers')
        ax2.set_title('Relevance Distribution')

        # Papers by year
        ax3 = axes[1, 0]
        years = {}
        for p in self.papers.values():
            years[p.year] = years.get(p.year, 0) + 1
        sorted_years = sorted(years.keys())
        ax3.bar(sorted_years, [years[y] for y in sorted_years],
               color='coral', alpha=0.7)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Number of Papers')
        ax3.set_title('Papers by Year')

        # Tag cloud (simplified as bar chart)
        ax4 = axes[1, 1]
        tag_counts = [(tag, len(papers)) for tag, papers in self.tags.items()]
        tag_counts.sort(key=lambda x: x[1], reverse=True)
        top_tags = tag_counts[:10]
        if top_tags:
            tags, counts = zip(*top_tags)
            ax4.barh(range(len(tags)), counts, color='seagreen', alpha=0.7)
            ax4.set_yticks(range(len(tags)))
            ax4.set_yticklabels(tags)
            ax4.set_xlabel('Number of Papers')
            ax4.set_title('Top Tags')
        else:
            ax4.text(0.5, 0.5, 'No tags yet', ha='center', va='center',
                    transform=ax4.transAxes)
            ax4.set_title('Top Tags')

        plt.suptitle(f'Literature Database: {stats["total_papers"]} Papers',
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('literature_statistics.png', dpi=150, bbox_inches='tight')
        plt.show()

        return fig

# =============================================================================
# Part 2: Reading Tracker
# =============================================================================

class ReadingTracker:
    """Track reading progress and time."""

    def __init__(self):
        self.sessions: List[Dict] = []

    def log_reading(self, paper_id: str, duration_minutes: int,
                   pass_level: int, notes: str = ""):
        """Log a reading session."""
        self.sessions.append({
            'paper_id': paper_id,
            'duration': duration_minutes,
            'pass_level': pass_level,
            'notes': notes,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'timestamp': datetime.now().isoformat()
        })

    def weekly_summary(self) -> Dict:
        """Get reading summary for past week."""
        from datetime import timedelta
        week_ago = datetime.now() - timedelta(days=7)

        week_sessions = [s for s in self.sessions
                        if datetime.fromisoformat(s['timestamp']) > week_ago]

        return {
            'sessions': len(week_sessions),
            'total_minutes': sum(s['duration'] for s in week_sessions),
            'papers_started': len(set(s['paper_id'] for s in week_sessions)),
            'by_pass': {
                1: sum(1 for s in week_sessions if s['pass_level'] == 1),
                2: sum(1 for s in week_sessions if s['pass_level'] == 2),
                3: sum(1 for s in week_sessions if s['pass_level'] == 3)
            }
        }

    def reading_goal_progress(self, weekly_goal_papers: int = 5,
                             weekly_goal_hours: float = 10) -> Dict:
        """Check progress toward weekly reading goals."""
        summary = self.weekly_summary()
        return {
            'papers_progress': f"{summary['papers_started']}/{weekly_goal_papers}",
            'hours_progress': f"{summary['total_minutes']/60:.1f}/{weekly_goal_hours}",
            'on_track_papers': summary['papers_started'] >= weekly_goal_papers,
            'on_track_hours': summary['total_minutes']/60 >= weekly_goal_hours
        }

# =============================================================================
# Part 3: Synthesis Tools
# =============================================================================

class SynthesisMatrix:
    """Create and manage synthesis matrices."""

    def __init__(self, dimensions: List[str]):
        """
        Initialize with comparison dimensions.

        Parameters:
        -----------
        dimensions : list of str
            Comparison criteria (e.g., ['method', 'dataset', 'results'])
        """
        self.dimensions = dimensions
        self.entries: Dict[str, Dict] = {}  # paper_id -> dimension values

    def add_entry(self, paper_id: str, values: Dict[str, str]):
        """Add paper to matrix."""
        self.entries[paper_id] = {d: values.get(d, "") for d in self.dimensions}

    def get_matrix(self) -> str:
        """Generate markdown table."""
        header = "| Paper | " + " | ".join(self.dimensions) + " |"
        separator = "|" + "|".join(["---"] * (len(self.dimensions) + 1)) + "|"

        rows = []
        for paper_id, values in self.entries.items():
            row = f"| {paper_id} | " + " | ".join(values.get(d, "") for d in self.dimensions) + " |"
            rows.append(row)

        return "\n".join([header, separator] + rows)

    def find_patterns(self) -> Dict:
        """Identify patterns across papers."""
        patterns = {dim: {} for dim in self.dimensions}

        for values in self.entries.values():
            for dim, val in values.items():
                if val:
                    patterns[dim][val] = patterns[dim].get(val, 0) + 1

        return patterns

class ThematicOrganizer:
    """Organize papers into themes."""

    def __init__(self):
        self.themes: Dict[str, List[str]] = {}  # theme -> paper_ids

    def add_theme(self, theme: str, description: str = ""):
        """Add a new theme."""
        if theme not in self.themes:
            self.themes[theme] = []

    def assign_paper(self, theme: str, paper_id: str):
        """Assign paper to theme."""
        if theme not in self.themes:
            self.add_theme(theme)
        if paper_id not in self.themes[theme]:
            self.themes[theme].append(paper_id)

    def generate_outline(self) -> str:
        """Generate thematic literature review outline."""
        outline = "# Literature Review Outline\n\n"

        for i, (theme, papers) in enumerate(self.themes.items(), 1):
            outline += f"## {i}. {theme}\n"
            for paper in papers:
                outline += f"   - {paper}\n"
            outline += "\n"

        return outline

# =============================================================================
# Part 4: Citation Network Analyzer
# =============================================================================

class CitationNetwork:
    """Analyze citation relationships."""

    def __init__(self):
        self.citations: Dict[str, List[str]] = {}  # paper -> papers it cites
        self.cited_by: Dict[str, List[str]] = {}   # paper -> papers citing it

    def add_citation(self, citing_paper: str, cited_paper: str):
        """Record that citing_paper cites cited_paper."""
        if citing_paper not in self.citations:
            self.citations[citing_paper] = []
        if cited_paper not in self.citations[citing_paper]:
            self.citations[citing_paper].append(cited_paper)

        if cited_paper not in self.cited_by:
            self.cited_by[cited_paper] = []
        if citing_paper not in self.cited_by[cited_paper]:
            self.cited_by[cited_paper].append(citing_paper)

    def key_papers(self, top_n: int = 10) -> List[tuple]:
        """Identify most-cited papers."""
        citation_counts = [(paper, len(citers))
                          for paper, citers in self.cited_by.items()]
        citation_counts.sort(key=lambda x: x[1], reverse=True)
        return citation_counts[:top_n]

    def forward_snowball(self, paper_id: str) -> List[str]:
        """Find papers that cite this paper."""
        return self.cited_by.get(paper_id, [])

    def backward_snowball(self, paper_id: str) -> List[str]:
        """Find papers this paper cites."""
        return self.citations.get(paper_id, [])

# =============================================================================
# Part 5: Demo
# =============================================================================

print("=" * 70)
print("Literature Review Methodology Tools")
print("=" * 70)

# Create database with sample papers
db = LiteratureDatabase()

sample_papers = [
    Paper(
        title="Neural Network Decoders for Surface Codes",
        authors=["Torlai, G.", "Melko, R."],
        year=2017,
        venue="Physical Review Letters",
        keywords=["neural network", "surface code", "decoder"],
        citations=150,
        tags=["neural_decoder", "core"],
        relevance=5
    ),
    Paper(
        title="Deep Reinforcement Learning for Quantum Error Correction",
        authors=["Nautrup, H. P.", "Delfosse, N.", "Briegel, H."],
        year=2019,
        venue="Physical Review X",
        keywords=["reinforcement learning", "QEC"],
        citations=80,
        tags=["neural_decoder", "rl"],
        relevance=4
    ),
    Paper(
        title="Scalable Neural Decoder for Topological Quantum Codes",
        authors=["Chamberland, C.", "Ronagh, P."],
        year=2018,
        venue="Physical Review X",
        keywords=["scalable", "topological", "decoder"],
        citations=90,
        tags=["neural_decoder", "scalability"],
        relevance=5
    ),
    Paper(
        title="Machine Learning for Quantum Error Correction",
        authors=["Sweke, R.", "Kesselring, M.", "van Nieuwenburg, E.", "Eisert, J."],
        year=2021,
        venue="Machine Learning: Science and Technology",
        keywords=["review", "machine learning", "QEC"],
        citations=60,
        tags=["review", "core"],
        relevance=4
    ),
    Paper(
        title="Transformer Neural Networks for Quantum Error Correction",
        authors=["Chen, M.", "Park, H.", "Zhang, K."],
        year=2023,
        venue="Nature Communications",
        keywords=["transformer", "attention", "decoder"],
        citations=30,
        tags=["neural_decoder", "transformer"],
        relevance=5
    ),
]

for paper in sample_papers:
    paper_id = db.add_paper(paper)
    print(f"Added: {paper_id}")

# Statistics
print("\n" + "=" * 70)
print("Database Statistics")
print("=" * 70)

stats = db.statistics()
print(f"Total papers: {stats['total_papers']}")
print(f"Reading status: {stats['by_status']}")
print(f"Total tags: {stats['total_tags']}")

# Search demo
print("\n" + "=" * 70)
print("Search Demo")
print("=" * 70)

results = db.search("neural", field="title")
print(f"Papers with 'neural' in title: {len(results)}")
for p in results:
    print(f"  - {p.summary()}")

# Synthesis matrix
print("\n" + "=" * 70)
print("Synthesis Matrix")
print("=" * 70)

matrix = SynthesisMatrix(['Method', 'Dataset', 'Performance', 'Limitation'])
matrix.add_entry('torlai2017', {
    'Method': 'RBM',
    'Dataset': 'Simulated',
    'Performance': 'Near-optimal',
    'Limitation': 'Small codes'
})
matrix.add_entry('nautrup2019', {
    'Method': 'RL',
    'Dataset': 'Simulated',
    'Performance': 'Learns strategies',
    'Limitation': 'Training time'
})
matrix.add_entry('chen2023', {
    'Method': 'Transformer',
    'Dataset': 'Simulated + HW',
    'Performance': 'Scalable',
    'Limitation': 'Complexity'
})

print(matrix.get_matrix())

# Thematic organization
print("\n" + "=" * 70)
print("Thematic Organization")
print("=" * 70)

organizer = ThematicOrganizer()
organizer.add_theme("Supervised Learning Approaches")
organizer.add_theme("Reinforcement Learning Approaches")
organizer.add_theme("Architecture Innovations")
organizer.add_theme("Scalability Solutions")

organizer.assign_paper("Supervised Learning Approaches", "torlai2017")
organizer.assign_paper("Supervised Learning Approaches", "chamberland2018")
organizer.assign_paper("Reinforcement Learning Approaches", "nautrup2019")
organizer.assign_paper("Architecture Innovations", "chen2023")

print(organizer.generate_outline())

# Reading tracker
print("\n" + "=" * 70)
print("Reading Tracker Demo")
print("=" * 70)

tracker = ReadingTracker()
tracker.log_reading("torlai2017", 45, 2, "Key paper on RBM decoders")
tracker.log_reading("nautrup2019", 30, 1, "Survey read")
tracker.log_reading("chen2023", 60, 2, "Recent transformer approach")

progress = tracker.reading_goal_progress(weekly_goal_papers=5, weekly_goal_hours=10)
print(f"Papers progress: {progress['papers_progress']}")
print(f"Hours progress: {progress['hours_progress']}")

# Uncomment to generate visualization:
# db.visualize_statistics()

print("\n" + "=" * 70)
print("Literature review tools ready!")
print("=" * 70)
```

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Systematic Search | Comprehensive, replicable literature discovery |
| Three-Pass Reading | Survey → Comprehension → Mastery |
| Synthesis | Moving from papers to narrative |
| Reference Management | Organize with tools and tags |
| Staying Current | Alerts, feeds, regular updates |

### Reading Efficiency Framework

```
All Papers
    ↓ (Filter by relevance)
Pass 1: Survey (5-10 min)
    ↓ (30% proceed)
Pass 2: Comprehension (30-60 min)
    ↓ (10% proceed)
Pass 3: Mastery (2-4 hours)
```

### Main Takeaways

1. **Systematic search** is essential - don't rely only on papers you already know

2. **Not all papers need deep reading** - triage with the three-pass method

3. **Annotation systems save time** - develop consistent practices

4. **Synthesis is more than summary** - identify patterns and gaps

5. **Stay current** - research moves fast in quantum computing

## Daily Checklist

- [ ] I understand systematic search methodology
- [ ] I can apply the three-pass reading method
- [ ] I have set up a reference management system
- [ ] I have created an annotation template
- [ ] I understand synthesis approaches (chronological, thematic, etc.)
- [ ] I know how to write a literature review section
- [ ] I have identified 5+ papers relevant to my research interest
- [ ] I have run the literature management tools

## Preview: Day 1006

Tomorrow we focus on **Identifying Research Directions**. We will:
- Map your interests to research opportunities
- Evaluate potential research directions
- Consider advisor and resource alignment
- Develop preliminary research statements
- Plan next steps toward research initiation

Tomorrow's work helps you identify where you want to contribute to the field.

---

*"Reading is not just absorbing information. It's learning to identify what matters, what's wrong, and what's missing. That's where research begins."*

---

| Navigation | Link |
|------------|------|
| Previous Day | [Day 1004: Research Proposals](./Day_1004_Wednesday.md) |
| Next Day | [Day 1006: Research Directions](./Day_1006_Friday.md) |
| Week Overview | [Week 144 README](./README.md) |
