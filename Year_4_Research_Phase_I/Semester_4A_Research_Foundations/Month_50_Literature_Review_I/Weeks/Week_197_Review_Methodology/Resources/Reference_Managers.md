# Reference Management Systems: Complete Setup Guide

## Overview

Effective reference management is the backbone of systematic literature review. This guide covers setup, configuration, and advanced usage of professional reference management tools, with a focus on Zotero as the recommended choice for academic research in quantum computing.

---

## Part 1: Choosing a Reference Manager

### Comparison of Major Tools

| Feature | Zotero | Mendeley | EndNote | JabRef |
|---------|--------|----------|---------|--------|
| **Cost** | Free | Free (Premium: $55/yr) | $274.95 | Free |
| **Open Source** | Yes | No | No | Yes |
| **Storage (Free)** | 300 MB | 2 GB | None | Local |
| **PDF Annotation** | Yes (built-in) | Yes | Yes | Via plugins |
| **Browser Extension** | Excellent | Good | Good | Limited |
| **LaTeX/BibTeX** | Yes (Better BibTeX) | Yes | Yes | Native |
| **Word Integration** | Yes | Yes | Yes | Limited |
| **Group Collaboration** | Yes | Yes | Yes | Via Git |
| **Offline Access** | Full | Full | Full | Full |
| **AI Features** | Emerging | Emerging | Limited | None |

### Recommendation for Quantum Computing Research

**Primary Choice: Zotero**

Reasons:
1. **Free and open-source** - No subscription required
2. **Better BibTeX plugin** - Superior LaTeX integration
3. **Academic community** - Strong support in physics/CS
4. **Flexible** - Extensive plugin ecosystem
5. **Privacy** - Self-hostable, data ownership

**Secondary Choice: Mendeley**

Consider if:
- You need more free cloud storage
- Your institution provides premium access
- You value social/discovery features

---

## Part 2: Zotero Complete Setup

### 2.1 Installation

#### Desktop Application

1. **Download:** Visit [zotero.org/download](https://www.zotero.org/download)
2. **Install:** Run installer for your OS (Windows, macOS, Linux)
3. **Launch:** Open Zotero, create account when prompted

#### Browser Connector

1. **Chrome:** [Zotero Connector for Chrome](https://chrome.google.com/webstore/detail/zotero-connector)
2. **Firefox:** [Zotero Connector for Firefox](https://addons.mozilla.org/firefox/addon/zotero-connector/)
3. **Safari:** Included with desktop app (macOS)
4. **Edge:** [Zotero Connector for Edge](https://microsoftedge.microsoft.com/addons/detail/zotero-connector)

#### Word/LibreOffice Plugin

- **Automatic:** Installed with desktop app
- **Verify:** Open Word → Check for "Zotero" tab

### 2.2 Account and Sync Setup

```
Settings → Sync → Create Account / Link Account
- Username: [your_username]
- Sync automatically: ✓
- Sync full-text content: ✓
```

**Sync Settings:**
- File Syncing: Zotero (free tier) or WebDAV
- For large libraries: Consider self-hosted WebDAV

### 2.3 Essential Plugins

#### Better BibTeX (CRITICAL for LaTeX users)

1. Download: [retorque.re/zotero-better-bibtex](https://retorque.re/zotero-better-bibtex/)
2. Install: Zotero → Tools → Add-ons → Install from file
3. Configure: Tools → Better BibTeX Preferences

**Key Settings:**
```
Citation key format: [auth:lower][year][shorttitle:lower:skipwords:select=1,1]
Example: smith2023quantum

Auto-export: Set up automatic .bib file export
```

#### ZotFile (PDF Management)

1. Download: [zotfile.com](http://zotfile.com/)
2. Install: Same as above
3. Configure: Tools → ZotFile Preferences

**Recommended Settings:**
```
Location of Files: Custom location (e.g., ~/Papers/)
Renaming Rules: {%a_}{%y_}{%t}
Example: Smith_2023_Quantum_Error_Correction.pdf
```

#### Zotero PDF Translate (Useful for non-English papers)

1. Download from Zotero plugin repository
2. Supports multiple translation services

### 2.4 Folder Structure for Quantum Computing Research

```
My Library/
├── _Inbox/                          # New papers to process
├── _Archive/                        # Completed reviews
│
├── 01_Research_Direction/           # Your specific topic
│   ├── Core_Papers/                 # Must-read papers
│   ├── Supporting_Work/             # Related but not central
│   └── Methods/                     # Methodology papers
│
├── 02_QEC_Fundamentals/             # Quantum Error Correction
│   ├── Surface_Codes/
│   ├── Color_Codes/
│   ├── LDPC_Codes/
│   └── Decoders/
│
├── 03_Quantum_Algorithms/           # Algorithm research
│   ├── VQE_QAOA/
│   ├── Quantum_ML/
│   └── Fault_Tolerant/
│
├── 04_Hardware/                     # Physical implementations
│   ├── Superconducting/
│   ├── Ion_Trap/
│   └── Photonic/
│
├── 05_Background/                   # Foundational material
│   ├── Textbooks/
│   ├── Review_Articles/
│   └── Tutorials/
│
└── 06_To_Review/                    # Systematic review papers
    ├── To_Screen/
    ├── Included/
    └── Excluded/
```

### 2.5 Tagging System

**Hierarchical Tags:**

```
Priority Tags:
- priority:high         # Core papers
- priority:medium       # Important
- priority:low          # Background

Status Tags:
- status:to_read        # Not yet read
- status:reading        # Currently reading
- status:read           # Completed
- status:to_reread      # Need to revisit

Content Tags:
- type:empirical        # Experimental/simulation
- type:theoretical      # Theory papers
- type:review           # Review articles
- type:tutorial         # Educational

Quality Tags:
- quality:high          # High-quality study
- quality:medium        # Standard quality
- quality:low           # Lower quality/preliminary

Methodology Tags:
- method:simulation     # Numerical simulation
- method:experiment     # Physical experiment
- method:analysis       # Theoretical analysis
- method:ml             # Machine learning
```

### 2.6 PDF Annotation Workflow

**Built-in Zotero Reader:**

1. **Open PDF:** Double-click in Zotero
2. **Highlight Types:**
   - Yellow: Key findings
   - Red: Important limitations
   - Green: Methodology
   - Blue: Future work/ideas
3. **Notes:** Add inline notes with context
4. **Extract:** Export annotations to Zotero notes

**Annotation Template:**
```
## Key Contribution
[Yellow highlights]

## Methodology
[Green highlights]

## Results
[Key findings]

## Limitations
[Red highlights]

## Relevance to My Research
[Your assessment]

## Questions/Ideas
[Blue highlights + your notes]
```

---

## Part 3: Advanced Zotero Features

### 3.1 Citation Export

**For LaTeX:**
```
1. Right-click collection → Export Collection
2. Format: Better BibTeX
3. Keep updated: ✓ (auto-export)
4. Use in LaTeX: \bibliography{exported.bib}
```

**Citation Key Patterns:**
```
Default: auth2023
With title: auth2023quantum
Custom: Configure in Better BibTeX
```

### 3.2 Report Generation

**Generate reading list:**
```
1. Select papers
2. Right-click → Generate Report from Items
3. Export as HTML/print
```

### 3.3 Saved Searches (Smart Collections)

Create dynamic collections based on criteria:

```
Example: "High Priority Unread"
Conditions:
- Tag contains "priority:high"
- AND Tag contains "status:to_read"
```

```
Example: "Recent Papers (2023+)"
Conditions:
- Date is after 2023-01-01
- AND Collection is not "Archive"
```

### 3.4 Group Libraries

For collaborative reviews:
```
1. Create group: zotero.org → Groups → Create New Group
2. Invite members
3. Set permissions (read/write)
4. Sync shared library
```

---

## Part 4: Integration Workflows

### 4.1 Browser to Library

```
Workflow:
1. Find paper on arXiv/journal
2. Click Zotero Connector icon
3. Metadata auto-imported
4. PDF auto-downloaded (if available)
5. Process in _Inbox folder
```

**Troubleshooting:**
- If connector fails: Use "Save to Zotero (Web Page)" then fix metadata
- For arXiv: Use arXiv URL for best metadata

### 4.2 arXiv Integration

**Best Practices:**
1. Save from arXiv abstract page (not PDF)
2. Check metadata accuracy
3. Add arXiv ID to "Extra" field: `arXiv: 2301.12345`
4. Link to published version if available

### 4.3 Google Scholar Integration

**Workaround for Scholar:**
1. Enable Scholar library links: Scholar Settings → Library Links → Add Zotero
2. Click "Import into Zotero" when available
3. Or: Copy paper title → Search in Zotero → Import from DOI

### 4.4 LaTeX/Overleaf Workflow

**Option 1: Better BibTeX Auto-Export**
```
1. Set up auto-export to project folder
2. Overleaf: Upload .bib file
3. Reference: \cite{key}
```

**Option 2: Zotero Integration in Overleaf**
```
1. Link Zotero account in Overleaf
2. Import directly from Zotero library
3. Auto-sync updates
```

### 4.5 Note-Taking Integration

**Zotero → Obsidian/Notion:**
1. Export annotations as Markdown
2. Use Zotero integration plugins
3. Link notes bidirectionally

**Recommended Plugins:**
- Obsidian: "Citations" or "Zotero Integration"
- Notion: Manual export or Zapier automation

---

## Part 5: Best Practices

### Daily Workflow

```
Morning:
1. Check _Inbox for new papers
2. Process: Add tags, move to folders
3. Quick-scan abstracts

Reading Session:
1. Select paper from reading queue
2. Open in Zotero reader
3. Annotate with color system
4. Export notes when finished
5. Update status tag

End of Day:
1. Sync library
2. Back up local database
3. Update reading progress
```

### Weekly Maintenance

```
1. Review _Inbox (clear backlog)
2. Update tags and organization
3. Export bibliography for writing
4. Check sync status
5. Review annotation backlog
```

### Backup Strategy

```
1. Enable Zotero sync (automatic)
2. Export full library monthly (Better BibTeX)
3. Back up data directory:
   - macOS: ~/Zotero/
   - Windows: C:\Users\<User>\Zotero\
   - Linux: ~/Zotero/
```

---

## Part 6: Troubleshooting

### Common Issues

**Connector not working:**
- Restart browser
- Reinstall connector
- Check for conflicts with other extensions

**Sync failures:**
- Check internet connection
- Verify account credentials
- Check storage quota

**Missing PDFs:**
- Configure auto-attachment in preferences
- Check ZotFile settings
- Manually attach from "Locate" menu

**Duplicate entries:**
- Use "Duplicate Items" saved search
- Merge duplicates: Right-click → Merge Items

**Citation style issues:**
- Update style: Preferences → Cite → Get additional styles
- For quantum computing: APS, IEEE, or Nature styles

---

## Part 7: Reference Manager Alternatives

### Mendeley Setup (Brief)

```
1. Download from mendeley.com
2. Create account
3. Install desktop app
4. Install browser extension (Web Importer)
5. Configure folder watching for PDFs
```

**Mendeley Pros:**
- More free cloud storage (2GB)
- Social features (follow researchers)
- Elsevier integration

**Mendeley Cons:**
- Owned by Elsevier (privacy concerns)
- Less flexible than Zotero
- Weaker BibTeX support

### JabRef for Pure BibTeX

If you only use LaTeX:
```
1. Download from jabref.org
2. Open/create .bib file
3. Manage entries directly
4. Use with Git for version control
```

**Best for:** LaTeX-only workflows, minimal overhead

---

## Quick Reference Card

### Zotero Keyboard Shortcuts

| Action | macOS | Windows |
|--------|-------|---------|
| New Item | Cmd+Shift+N | Ctrl+Shift+N |
| Add by Identifier | Cmd+Shift+I | Ctrl+Shift+I |
| New Note | Cmd+Shift+O | Ctrl+Shift+O |
| Focus Search | Cmd+F | Ctrl+F |
| Open PDF | Enter | Enter |
| Show Item in Collection | Opt+Click | Alt+Click |

### Essential Settings Checklist

- [ ] Account created and synced
- [ ] Browser connector installed
- [ ] Better BibTeX configured
- [ ] ZotFile configured (optional)
- [ ] Folder structure created
- [ ] Tag system defined
- [ ] Word processor plugin verified
- [ ] Auto-export configured (LaTeX users)
- [ ] Backup location identified

---

## Resources

### Official Documentation
- [Zotero Documentation](https://www.zotero.org/support/)
- [Better BibTeX Documentation](https://retorque.re/zotero-better-bibtex/)
- [Zotero Forums](https://forums.zotero.org/)

### Video Tutorials
- Search "Zotero tutorial academic" on YouTube
- Many universities provide institutional guides

### Community
- Zotero Forums
- r/Zotero on Reddit
- Academic Twitter #Zotero

---

*"Your reference library is your research memory. Treat it with the care it deserves."*
