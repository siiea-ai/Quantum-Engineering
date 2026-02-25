# Research Organization Tools

## Software and Systems for Thesis Materials Management

---

## Overview

This document provides a comprehensive guide to tools for organizing research materials during thesis writing. Effective tooling can significantly accelerate the writing process and reduce frustration from disorganized materials.

---

## Reference Management Systems

### Zotero (Recommended - Free, Open Source)

**Website**: https://www.zotero.org/

**Key Features**:
- Free and open-source
- Browser connector for easy import
- PDF storage and annotation
- Group libraries for collaboration
- BibTeX export
- Word and LibreOffice plugins
- iOS app available

**Best Practices for Thesis Use**:
```
Zotero Library/
├── Thesis/
│   ├── Chapter 1 - Introduction/
│   ├── Chapter 2 - Background/
│   ├── Chapter 3 - Research 1/
│   ├── Chapter 4 - Research 2/
│   └── Chapter 5 - Discussion/
└── General Reading/
```

**Plugins to Install**:
- Better BibTeX: Enhanced BibTeX export with stable citation keys
- Zotfile: Advanced PDF management and renaming
- DOI Manager: Retrieve missing DOIs

**BibTeX Export Settings**:
- Install Better BibTeX plugin
- Set citation key format: `[auth:lower][year]`
- Enable auto-export for continuous BibTeX file updates

---

### Mendeley

**Website**: https://www.mendeley.com/

**Key Features**:
- Free basic tier (2GB cloud storage)
- PDF annotation tools
- Social/discovery features
- Word plugin
- Mobile apps
- Citation metrics

**Best For**:
- Heavy PDF annotation workflows
- Discovering related papers
- Collaboration with Mendeley users

**Limitations**:
- Elsevier-owned (privacy considerations)
- Limited BibTeX customization
- Cloud storage limits on free tier

---

### EndNote

**Website**: https://endnote.com/

**Key Features**:
- Industry standard in many fields
- Excellent Word integration
- Institutional site licenses often available
- Powerful search and organization
- Cite While You Write feature

**Best For**:
- Institutions with site licenses
- Heavy Word users
- Teams using EndNote

**Limitations**:
- Expensive without institutional license
- Less flexible BibTeX export
- Steeper learning curve

---

### Paperpile

**Website**: https://paperpile.com/

**Key Features**:
- Web-based, cloud-native
- Excellent Google Docs integration
- Fast PDF management
- Clean interface
- BibTeX export

**Best For**:
- Google Docs users
- Chrome-primary workflows
- Quick setup needed

**Limitations**:
- Subscription-based ($3/month academic)
- No offline desktop app
- Less Word integration

---

### BibDesk (Mac Only)

**Website**: https://bibdesk.sourceforge.io/

**Key Features**:
- Native Mac application
- Direct BibTeX file editing
- PDF attachment and preview
- Smart groups and searches
- Free and open-source

**Best For**:
- Mac users with LaTeX workflow
- Direct BibTeX management
- Minimal, focused tool

---

### JabRef (Cross-Platform)

**Website**: https://www.jabref.org/

**Key Features**:
- Java-based, runs anywhere
- Native BibTeX/BibLaTeX support
- DOI/ISBN import
- Free and open-source
- Database synchronization

**Best For**:
- BibTeX power users
- Cross-platform needs
- Direct .bib file management

---

## File Organization and Sync

### Cloud Storage Options

| Service | Free Storage | Academic Pricing | Best For |
|---------|--------------|------------------|----------|
| Google Drive | 15 GB | 100GB @ ~$2/mo | Google ecosystem |
| Dropbox | 2 GB | Education Plus | Cross-platform |
| OneDrive | 5 GB | Often included | Microsoft/Office |
| iCloud | 5 GB | 50GB @ $1/mo | Apple ecosystem |
| Box | 10 GB | Institutional | Enterprise features |

### Recommended Folder Structure

```
Thesis_Project/
├── 00_Administration/
│   ├── Timeline/
│   ├── Committee/
│   └── Forms/
├── 01_Writing/
│   ├── Main_Document/
│   │   ├── thesis_v1.tex
│   │   └── chapters/
│   ├── Drafts/
│   └── Archive/
├── 02_Figures/
│   ├── Chapter_1/
│   ├── Chapter_2/
│   ├── ...
│   └── Source_Files/
├── 03_Data/
│   ├── Raw/
│   ├── Processed/
│   └── Analysis_Scripts/
├── 04_Code/
│   ├── Main_Projects/
│   └── Scripts/
├── 05_Bibliography/
│   ├── thesis.bib
│   └── PDFs/
└── 06_Notes/
    ├── Research_Notes/
    └── Meeting_Notes/
```

---

## Version Control

### Git for Thesis Writing

**Why Use Git**:
- Track all changes to documents
- Revert to any previous version
- Branch for major revisions
- Collaborate with advisor/committee
- Backup via GitHub/GitLab

**Basic Setup**:
```bash
# Initialize thesis repository
cd Thesis_Project
git init

# Create .gitignore
echo "*.aux
*.log
*.out
*.synctex.gz
*.pdf
.DS_Store" > .gitignore

# Initial commit
git add .
git commit -m "Initial thesis structure"

# Add remote (GitHub/GitLab)
git remote add origin https://github.com/username/thesis.git
git push -u origin main
```

**Recommended Commit Practices**:
- Commit at end of each writing session
- Use descriptive commit messages
- Tag major milestones (v0.1, v1.0-draft, etc.)

**Git Clients for Non-Command-Line Users**:
- GitHub Desktop: https://desktop.github.com/
- Sourcetree: https://www.sourcetreeapp.com/
- GitKraken: https://www.gitkraken.com/

---

## Note-Taking and Knowledge Management

### Obsidian

**Website**: https://obsidian.md/

**Key Features**:
- Markdown-based notes
- Bidirectional linking
- Graph view of connections
- Local storage (privacy)
- Extensive plugin ecosystem
- Free for personal use

**Thesis Use Cases**:
- Research notes with connections
- Literature review organization
- Idea development
- Daily writing journals

### Notion

**Website**: https://www.notion.so/

**Key Features**:
- Flexible databases and pages
- Team collaboration
- Templates available
- Cross-platform apps
- Free for personal use

**Thesis Use Cases**:
- Project management
- Task tracking
- Timeline planning
- Collaboration with advisor

### Logseq

**Website**: https://logseq.com/

**Key Features**:
- Outliner-style notes
- Bidirectional linking
- Local-first storage
- Open source
- PDF annotation integration

---

## Writing and Editing

### LaTeX Editors

| Editor | Platform | Key Feature | Best For |
|--------|----------|-------------|----------|
| Overleaf | Web | Collaboration, templates | Team writing, beginners |
| TeXstudio | Cross | Full-featured, free | Desktop LaTeX |
| VS Code + LaTeX Workshop | Cross | Customizable, modern | Developers |
| Texmaker | Cross | Simple, reliable | Straightforward editing |
| MacTeX + TexShop | Mac | Native Mac experience | Mac users |

### Overleaf (Recommended for Thesis)

**Website**: https://www.overleaf.com/

**Advantages for Thesis**:
- No local LaTeX installation needed
- Thesis templates available
- Track changes built-in
- Easy advisor collaboration
- Version history
- Git integration

**Free Tier Limitations**:
- 1 collaborator per project
- No tracked changes
- No Git integration

**Academic Tier** (often provided by institutions):
- Unlimited collaborators
- Full history
- Git sync
- Priority compile

---

## Data Analysis and Visualization

### Python Ecosystem

| Tool | Use Case |
|------|----------|
| Jupyter Notebook | Interactive analysis, thesis figures |
| Matplotlib | Publication-quality plots |
| Seaborn | Statistical visualizations |
| Plotly | Interactive figures |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| SciPy | Scientific computing |

### Visualization Best Practices for Thesis

```python
import matplotlib.pyplot as plt

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
})

# Colorblind-friendly palette
colors = ['#0077BB', '#33BBEE', '#009988',
          '#EE7733', '#CC3311', '#EE3377']
```

---

## Project Management

### Task Tracking Options

| Tool | Best For | Cost |
|------|----------|------|
| Todoist | Simple task lists | Free tier |
| Trello | Visual Kanban boards | Free |
| Asana | Complex project management | Free tier |
| Notion | All-in-one workspace | Free |
| Linear | Developer-friendly | Free for individuals |
| GitHub Projects | Code-integrated | Free |

### Writing Progress Tracking

**Daily Writing Log Template**:
```markdown
# Writing Log

## [Date]
- Words written: ___
- Time spent: ___
- Section worked on: ___
- Tomorrow's goal: ___
```

**Word Count Tracking**:
```bash
# For LaTeX (bash script)
texcount -sum thesis.tex

# For markdown
wc -w *.md
```

---

## Backup Strategy

### 3-2-1 Backup Rule

- **3** copies of data
- **2** different storage types
- **1** offsite backup

### Implementation

| Location | Type | Sync Frequency |
|----------|------|----------------|
| Working computer | Local | Real-time |
| External HDD | Local backup | Weekly |
| Cloud storage | Remote | Continuous |
| Git repository | Remote + versioned | Per commit |
| Institutional storage | Remote | Weekly |

---

## Recommended Setup Summary

### Minimal Setup

1. **Reference manager**: Zotero (free)
2. **Cloud sync**: Google Drive or Dropbox
3. **Writing**: Overleaf (free tier)
4. **Backup**: Cloud + external HDD

### Comprehensive Setup

1. **Reference manager**: Zotero + Better BibTeX
2. **Cloud sync**: Cloud storage with auto-sync
3. **Version control**: Git + GitHub/GitLab
4. **Writing**: Overleaf or local LaTeX
5. **Notes**: Obsidian or Notion
6. **Tasks**: Todoist or Notion
7. **Backup**: 3-2-1 strategy implemented

---

## Getting Started Checklist

- [ ] Choose and set up reference manager
- [ ] Create thesis folder structure
- [ ] Set up cloud sync
- [ ] Initialize Git repository
- [ ] Configure backup system
- [ ] Choose LaTeX editor
- [ ] Set up note-taking system
- [ ] Create task tracking system

---

*The best tool is the one you'll actually use consistently. Start simple and add complexity as needed.*
