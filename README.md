# India Runs — Track 1: Intelligent Candidate Discovery & Ranking

**Hackathon:** India Runs by Redrob AI × Hack2skill  
**Track:** Track 1 — Data & AI Challenge  
**Role:** Senior AI Engineer — Founding Team @ Redrob AI  
**Candidate pool:** 100,000 candidates  
**Output:** Top 100 ranked candidates with per-candidate reasoning  

---

## Approach

This ranker goes beyond keyword matching. The core insight from the JD is that Redrob explicitly warns against keyword stuffers — a Marketing Manager with all the right AI skills listed is not a fit. A Backend Engineer whose career history shows they built a recommendation system at a product company *is* a fit, even if their skill list looks sparse.

The pipeline has five scoring components combined with a behavioral signal multiplier.

### 1. Skill match score (35%)

Skills are scored using proficiency level, months of actual usage, peer endorsements, and Redrob platform assessment scores. Required skills (embeddings, vector DBs, retrieval systems, evaluation frameworks, Python) are weighted at full value. Nice-to-have skills (LLM fine-tuning, distributed systems, open source) contribute a smaller bonus.

The key anti-keyword-stuffer mechanism: a skill listed as "expert" with 0 months of usage gets a near-zero weight. Duration and endorsements act as a trust multiplier — they catch candidates who pad their skill list without real experience.

### 2. Career & title score (30%)

Scores experience years against the JD's 5–9 year sweet spot, title relevance to AI/ML/search roles, and career history quality. Explicitly penalises:

- Consulting-only backgrounds (TCS, Infosys, Wipro, Accenture, etc.) per the JD's stated preference
- Irrelevant titles (Marketing Manager, Accountant, Civil Engineer) regardless of listed skills
- Candidates whose only recent AI experience is calling hosted LLM APIs

### 3. TF-IDF semantic match (15%)

Builds a TF-IDF vector from each candidate's headline, summary, career descriptions, and skill names, then computes cosine similarity against a distilled JD text vector. This catches hidden gems — candidates who don't use the exact keywords "RAG" or "Pinecone" in their skills section but whose career descriptions show they built retrieval systems in production.

Runs entirely in Python stdlib — no external libraries, no API calls.

### 4. Career trajectory score (10%)

Detects whether a candidate is moving *toward* AI/ML over their career history. A Backend Engineer → ML Engineer trajectory is a strong positive signal. A Marketing Manager → AI Content Writer is not. Uses chronological role scoring to compute an upward/downward delta.

### 5. Redrob assessment score (10%)

Candidates who completed Redrob's own skill assessments are up-weighted — these are verified scores, not self-reported proficiency. AI-relevant assessments (NLP, ML, Python, system design) are weighted 2× over general assessments. Taking multiple assessments signals platform engagement.

### Behavioral signal multiplier

The profile score is multiplied by a behavioral composite built from all 23 Redrob signals. Key signals:

| Signal | Weight | Rationale |
|---|---|---|
| Recency (days since last active) | 20% | Inactive >6 months = not actually hireable |
| Open to work flag | 15% | Direct availability signal |
| Recruiter response rate | 15% | Will they reply if contacted? |
| Interview completion rate | 10% | Will they show up? |
| GitHub activity score | 10% | Verified technical engagement |
| Notice period | 8% | JD prefers ≤30 days |
| Response time | 8% | Speed of engagement |
| Salary fit | 5% | Mid-range vs senior founding team comp |
| Location / relocation | 5% | Pune/Noida preferred per JD |
| Profile completeness | 2% | Seriousness signal |
| Saved by recruiters 30d | 2% | Market validation |

### Honeypot detection

The dataset contains ~80 honeypot candidates with subtly impossible profiles. We detect them using four checks:

1. Future start dates in career history
2. `duration_months` claiming far more time than dates allow (>24 month tolerance)
3. Claimed years of experience significantly exceeding sum of career history
4. Three or more "expert" skills with 0 months of usage

**65 honeypots detected and excluded** from ranking.

---

## Running the ranker

### Requirements

- Python 3.11+
- No external packages — stdlib only

### Setup

```bash
git clone https://github.com/mohana28sri/india-runs-track1.git
cd india-runs-track1
```

Place `candidates.jsonl` in the same directory (not committed — 465MB).

### Run

```bash
python rank.py
```

Expected runtime: ~60–70 seconds on a 16GB CPU machine.  
Memory usage: ~2–3GB peak.

### Validate

```bash
python validate_submission.py submission.csv
```

### Output

`submission.csv` — 100 rows, columns: `candidate_id, rank, score, reasoning`

---

## Results summary

| Metric | Value |
|---|---|
| Total candidates processed | 100,000 |
| Honeypots detected & excluded | 65 |
| Runtime | ~65 seconds |
| Top score | 0.8387 |
| Rank 100 score | 0.6077 |

### Top 10

| Rank | Candidate | Title | Exp | Core skill hits | Score |
|---|---|---|---|---|---|
| 1 | CAND_0081846 | Lead AI Engineer | 6.7 yrs | 10 | 0.8387 |
| 2 | CAND_0086022 | Senior Applied Scientist | 5.3 yrs | 8 | 0.8303 |
| 3 | CAND_0077337 | Staff ML Engineer | 7.0 yrs | 10 | 0.8302 |
| 4 | CAND_0008425 | Senior NLP Engineer | 7.8 yrs | 9 | 0.8301 |
| 5 | CAND_0088025 | Staff ML Engineer | 8.6 yrs | 7 | 0.8245 |
| 6 | CAND_0080766 | Staff ML Engineer | 8.8 yrs | 9 | 0.8007 |
| 7 | CAND_0018499 | Senior ML Engineer | 7.2 yrs | 8 | 0.7906 |
| 8 | CAND_0071974 | Senior AI Engineer | 7.8 yrs | 9 | 0.7826 |
| 9 | CAND_0002025 | Senior AI Engineer | 5.9 yrs | 8 | 0.7697 |
| 10 | CAND_0011687 | Senior NLP Engineer | 7.8 yrs | 5 | 0.7465 |

---

## Design decisions

**Why no LLM API calls?** The compute constraint (CPU only, no network, ≤5 min) reflects a real production requirement — a system calling GPT-4 per candidate cannot scale to 200K candidates. TF-IDF cosine similarity gives semantic matching at O(n) cost with zero network dependency.

**Why penalise consulting-only backgrounds?** The JD explicitly states this. Our career scorer checks every company in the candidate's history against a list of major Indian IT services firms and applies a 0.3× multiplier if the entire career is consulting-only.

**Why is trajectory scored separately from career?** Career score measures *what* you've done. Trajectory measures *which direction* you're moving. A candidate with 3 years of AI/ML work at the end of a 7-year non-AI career is often a stronger hire than someone with 7 flat years of adjacent work — they made a deliberate move.

---

## Repository structure

```
india-runs-track1/
├── rank.py                        # Main ranker — run this
├── submission.csv                 # Output: top 100 ranked candidates
├── submission_metadata.yaml       # Submission metadata
├── README.md                      # This file
└── validate_submission.py         # Official format validator (from bundle)
```

---

## Reproduce command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

*(candidates.jsonl not committed to repo due to file size)*
