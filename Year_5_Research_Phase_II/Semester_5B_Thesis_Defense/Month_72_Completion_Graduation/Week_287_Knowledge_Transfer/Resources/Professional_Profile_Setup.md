# Professional Profile Setup Guide

## Overview

This guide provides detailed instructions for setting up professional online profiles that maximize your academic visibility and career opportunities.

---

## Google Scholar

### Purpose
Google Scholar is the most widely used academic search engine. A complete profile ensures your work is discoverable and your citation metrics are accurate.

### Setup Steps

1. **Access**
   - Go to https://scholar.google.com/citations
   - Sign in with your Google account
   - Click "My profile"

2. **Basic Information**
   - **Name:** Use your publishing name consistently
   - **Affiliation:** Current institution and department
   - **Email:** Use institutional email for verification
   - **Homepage:** Link to personal website

3. **Research Interests**
   Add 5-7 keywords that describe your research:
   ```
   Example: Quantum Computing, Error Correction, Superconducting Qubits,
   Fault Tolerance, Quantum Algorithms
   ```

4. **Profile Photo**
   - Use a professional headshot
   - Consistent with other profiles
   - High resolution (at least 250x250 pixels)

5. **Claiming Publications**
   - Review suggested articles
   - Click "Add" to confirm your publications
   - Search for missing publications manually
   - Remove incorrectly attributed papers

6. **Managing Duplicates**
   - Click "Merge" on duplicate entries
   - Keep the version with most citations
   - Verify the merged entry is correct

7. **Settings**
   - Enable "Make my profile public"
   - Enable email alerts for new citations
   - Enable alerts for new articles in your areas

### Best Practices
- Review profile monthly for new citations
- Merge duplicates promptly
- Keep affiliation current
- Link to preprints and published versions

---

## ORCID

### Purpose
ORCID provides a persistent digital identifier that distinguishes you from other researchers. Increasingly required by journals and funders.

### Setup Steps

1. **Registration**
   - Go to https://orcid.org/register
   - Enter your information
   - Create password
   - Verify email
   - Note your ORCID iD: 0000-0000-0000-0000

2. **Name Variations**
   - Add all name variations
   - Include maiden names if applicable
   - Add transliterations if relevant

3. **Biography**
   ```
   [Name] is a [position] at [institution] specializing in [field].
   Their research focuses on [specific topics]. They received their
   PhD in [field] from [university] in [year].
   ```

4. **Education**
   - PhD, [Field], [University], [Start-End Years]
   - MS/MA, [Field], [University], [Start-End Years]
   - BS/BA, [Field], [University], [Start-End Years]

5. **Employment**
   Add all relevant positions:
   - Current position
   - Previous academic positions
   - Industry experience if relevant

6. **Works (Publications)**
   Import publications via:
   - **CrossRef:** Enter DOIs
   - **DataCite:** For data publications
   - **Scopus:** Link account
   - **Web of Science:** Link account
   - **Manual entry:** For items without DOIs

7. **Funding**
   - Add grants you've received
   - Link to funder registries

8. **Privacy Settings**
   | Section | Recommended |
   |---------|-------------|
   | Biography | Public |
   | Education | Public |
   | Employment | Public |
   | Works | Public |
   | Funding | Public |
   | Email | Trusted parties only |

### Integration
- Link ORCID to publisher accounts
- Add to journal submission profiles
- Include in grant applications
- Use ORCID authentication when available

---

## Personal Academic Website

### Purpose
A personal website gives you complete control over how you present yourself and your research to the world.

### Platform Options

| Platform | Pros | Cons | Cost |
|----------|------|------|------|
| GitHub Pages | Free, version controlled | Technical setup | Free |
| Hugo Academic | Beautiful, fast | Learning curve | Free |
| WordPress.com | Easy, flexible | Ads on free tier | $4-25/mo |
| Squarespace | Beautiful templates | Less flexible | $12-18/mo |
| Wix | Drag-and-drop | Can be slow | $14-23/mo |
| Google Sites | Simple, free | Limited design | Free |

### Essential Pages

1. **Home Page**
   - Professional photo (high quality)
   - Name and current position
   - Brief research description (2-3 sentences)
   - Quick links to key sections

2. **About/Bio Page**
   - Extended biography (3-4 paragraphs)
   - Research interests
   - Education summary
   - Contact information

3. **Research Page**
   - Research areas with descriptions
   - Current projects
   - Past projects
   - Research impact/vision

4. **Publications Page**
   - Complete publication list
   - Organized by year or type
   - PDF downloads (where permitted)
   - BibTeX citations
   - Links to DOIs, preprints

5. **CV Page**
   - Embedded PDF viewer
   - Download link
   - Last updated date

6. **Contact Page**
   - Email address
   - Office location
   - Links to profiles

### GitHub Pages Setup (Free)

```bash
# 1. Create repository named username.github.io

# 2. Clone and add content
git clone https://github.com/username/username.github.io
cd username.github.io

# 3. Create basic index.html
echo "Hello World" > index.html

# 4. Push to GitHub
git add .
git commit -m "Initial site"
git push origin main

# 5. Access at https://username.github.io
```

### Hugo Academic Theme Setup

```bash
# Install Hugo
brew install hugo  # macOS
# Or download from gohugo.io

# Create site with Academic theme
hugo new site mysite
cd mysite
git init
git submodule add https://github.com/wowchemy/starter-hugo-academic themes/academic

# Configure in config/_default/
# See documentation at wowchemy.com

# Preview locally
hugo server -D

# Build for production
hugo --minify
```

### SEO Best Practices

1. **Page Titles**
   - Unique, descriptive titles
   - Include your name
   - Example: "Research | Jane Smith, PhD"

2. **Meta Descriptions**
   ```html
   <meta name="description" content="Jane Smith is a quantum
   computing researcher at MIT specializing in error correction.">
   ```

3. **Image Alt Text**
   ```html
   <img src="headshot.jpg" alt="Dr. Jane Smith, quantum computing researcher">
   ```

4. **Structured Data**
   ```json
   {
     "@context": "https://schema.org",
     "@type": "Person",
     "name": "Jane Smith",
     "jobTitle": "Postdoctoral Researcher",
     "affiliation": {
       "@type": "Organization",
       "name": "MIT"
     },
     "url": "https://janesmith.github.io"
   }
   ```

5. **Submit to Google**
   - Register with Google Search Console
   - Submit sitemap.xml
   - Monitor indexing status

---

## LinkedIn

### Purpose
LinkedIn is essential for professional networking, especially for industry connections and career opportunities.

### Profile Sections

1. **Header**
   - **Photo:** Professional headshot
   - **Banner:** Research-related image
   - **Headline:** More than just your title
     ```
     PhD Candidate in Quantum Computing | Error Correction Researcher
     ```

2. **About Section**
   ```
   I am a [position] at [institution] where I [main activity].

   My research focuses on [specific area], with the goal of [impact].
   I have published [X] papers in [venues] and contributed to [achievements].

   Key areas: [Area 1] | [Area 2] | [Area 3]

   Open to: [collaborations, speaking, consulting, etc.]
   ```

3. **Experience**
   - Current and past positions
   - Brief descriptions with achievements
   - Link to company pages

4. **Education**
   - All degrees
   - Research focus for graduate degrees
   - Awards and honors

5. **Skills**
   - Add 10+ relevant skills
   - Request endorsements
   - Prioritize top skills

6. **Publications**
   - Add key publications
   - Link to full text when possible

7. **Featured**
   - Pin important content
   - Papers, talks, media coverage

### Best Practices
- Custom URL: linkedin.com/in/janesmithphd
- Connect with colleagues and field experts
- Share research updates monthly
- Engage with others' content
- Join relevant groups

---

## Twitter/X for Academics

### Purpose
Academic Twitter is valuable for real-time engagement with your research community.

### Profile Setup

1. **Handle:** @JaneSmithPhD (professional, memorable)
2. **Name:** Jane Smith, PhD
3. **Bio:**
   ```
   Quantum computing researcher @MIT | Error correction & fault tolerance
   | She/her | Views my own | janesmith.github.io
   ```
4. **Photo:** Same professional headshot
5. **Banner:** Research-related image
6. **Website:** Link to personal site

### Content Strategy

| Content Type | Frequency |
|--------------|-----------|
| Paper announcements | As published |
| Thread explanations | Monthly |
| Interesting papers | Weekly |
| Conference live-tweeting | As attending |
| Engagement | Daily |

### Thread Structure for Papers

```
🧵 New paper out! "Title of Paper"

We show that [main finding].

A thread on what we did and why it matters:

[1/N]
---
The problem: [Context]

Previous approaches [limitation].

[2/N]
---
Our approach: [Innovation]

[Figure]

[3/N]
---
Key results:
• [Finding 1]
• [Finding 2]
• [Finding 3]

[4/N]
---
This matters because [implications].

Paper: [DOI]
Code: [GitHub]

Thanks to @collaborator1 @collaborator2
Supported by @funder

[5/5]
```

---

## ResearchGate

### Purpose
Academic social network for sharing and discovering research.

### Setup
1. Create profile
2. Add publications
3. Upload full-text PDFs (where permitted)
4. Follow researchers in your field
5. Answer questions in your expertise

### Best Practices
- Keep publication list updated
- Respond to questions
- Share updates on projects
- Monitor RG Score

---

## Maintaining Your Presence

### Regular Schedule

| Timeframe | Actions |
|-----------|---------|
| Weekly | Check Google Scholar alerts |
| | Post on Twitter |
| Monthly | Update any new publications |
| | Review LinkedIn |
| Quarterly | Audit all profiles |
| | Update bios and photos if needed |
| Annually | Comprehensive review |
| | Major website updates |

### After Major Events

**New Publication:**
- Add to Google Scholar
- Add to ORCID
- Update website
- Post on social media
- Update ResearchGate

**Position Change:**
- Update all affiliations
- Update email on all profiles
- Announce on LinkedIn
- Update website bio

---

## Resources

### Tools
- Canva: Create professional graphics
- Gravatar: Consistent profile photos
- Bitly: Track link clicks
- Buffer: Schedule social posts

### Templates
- wowchemy.com: Hugo academic themes
- academicpages.github.io: GitHub Pages template

### Guides
- "Building Your Academic Online Presence" - The Professor Is In
- Twitter for Scientists (Nature article)

---

*A strong online presence amplifies the impact of your research.*
