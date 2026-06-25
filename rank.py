#!/usr/bin/env python3
"""
India Runs — Track 1: Intelligent Candidate Discovery & Ranking
Senior AI Engineer @ Redrob AI
CPU-only, no network, <5 min, <16GB RAM
"""

import json
import csv
import math
import sys
from datetime import datetime, date
from pathlib import Path

CANDIDATES_FILE = "candidates.jsonl"
OUTPUT_FILE = "submission.csv"
REFERENCE_DATE = date(2026, 6, 25)
TOP_N = 100

# ─── JD signals ───────────────────────────────────────────────────────────────

# Must-have skills (highest weight)
REQUIRED_SKILLS = {
    "embeddings", "sentence transformers", "openai embeddings", "bge", "e5",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss", "chroma", "pgvector",
    "semantic search", "hybrid search", "dense retrieval", "sparse retrieval",
    "ndcg", "mrr", "map", "ranking", "learning to rank", "retrieval", "reranking",
    "python", "information retrieval", "rag", "retrieval augmented generation",
    "recommendation system", "search", "nlp", "natural language processing",
}

# Nice-to-have skills
BONUS_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "lightgbm", "llm", "large language model", "transformer",
    "bert", "gpt", "t5", "hugging face", "huggingface",
    "distributed systems", "kubernetes", "docker", "fastapi", "pytorch", "tensorflow",
    "spark", "kafka", "redis", "postgresql", "aws", "gcp", "azure",
    "a/b testing", "ab testing", "mlops", "ml pipeline", "feature store",
    "open source", "github",
}

# Red-flag skills (not wrong, but not relevant for this role)
IRRELEVANT_TITLES = {
    "marketing manager", "content writer", "graphic designer", "accountant",
    "civil engineer", "mechanical engineer", "hr manager", "human resources",
    "sales executive", "customer support", "customer service",
    "project manager", "business analyst", "operations manager",
    "administrative", "finance manager", "data entry",
}

# Consulting-only companies (penalise per JD)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "ltimindtree", "birlasoft", "mindtree", "persistent", "mastech",
}

# Good locations for this role (Pune/Noida preferred, other Tier-1 OK)
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurugram", "gurgaon", "hyderabad",
    "bengaluru", "bangalore", "mumbai", "chennai", "india",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return s.lower().strip()

def skill_hit(skill_name: str, skill_set: set) -> bool:
    n = normalize(skill_name)
    return any(req in n or n in req for req in skill_set)

def days_since(date_str: str) -> int:
    try:
        d = date.fromisoformat(date_str)
        return (REFERENCE_DATE - d).days
    except Exception:
        return 9999

# ─── Honeypot detection ───────────────────────────────────────────────────────

def is_honeypot(c: dict) -> bool:
    """
    Detect impossible profiles. Dataset has ~80 honeypots.
    Checks:
      1. Company tenure > company age (impossible timeline)
      2. Expert skill with 0 months used
      3. Total claimed experience >> sum of career history
      4. Skills count > 40 (unrealistic)
      5. future start dates in career
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    # Check 1: future dates in career
    for job in career:
        sd = job.get("start_date", "")
        try:
            if date.fromisoformat(sd) > REFERENCE_DATE:
                return True
        except Exception:
            pass

    # Check 2: duration_months mismatch with dates
    for job in career:
        dm = job.get("duration_months", 0)
        sd = job.get("start_date", "")
        ed = job.get("end_date")
        try:
            start = date.fromisoformat(sd)
            end = date.fromisoformat(ed) if ed else REFERENCE_DATE
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            # Allow ±6 month tolerance; flag extreme mismatches
            if dm > actual_months + 24:
                return True
        except Exception:
            pass

    # Check 3: claimed experience vs career history total
    claimed_yrs = profile.get("years_of_experience", 0)
    career_months = sum(j.get("duration_months", 0) for j in career)
    career_years = career_months / 12
    if claimed_yrs > 0 and career_years > 0:
        # More than 30% gap indicates honeypot
        if claimed_yrs > career_years * 1.8 and claimed_yrs - career_years > 5:
            return True

    # Check 4: expert skill with 0 duration
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 3:
        return True

    # Check 5: too many skills
    if len(skills) > 50:
        return True

    return False

# ─── Scoring functions ────────────────────────────────────────────────────────

def score_skills(c: dict) -> tuple[float, int, int]:
    """Returns (skill_score 0-1, required_hits, bonus_hits)"""
    skills = c.get("skills", [])
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})

    req_score = 0.0
    bonus_score = 0.0
    req_hits = 0
    bonus_hits = 0

    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)
        assess_score = assessments.get(name, -1)

        # Proficiency weight
        prof_w = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(prof, 0.3)

        # Duration trust: cap at 60 months
        dur_w = min(duration / 60, 1.0) if duration > 0 else 0.1

        # Endorsement trust: cap at 50
        end_w = min(endorsements / 50, 1.0) * 0.3 + 0.7

        # Assessment bonus
        assess_w = 1.0
        if assess_score >= 0:
            assess_w = 0.7 + 0.3 * (assess_score / 100)

        skill_w = prof_w * dur_w * end_w * assess_w

        if skill_hit(name, REQUIRED_SKILLS):
            req_score += skill_w
            req_hits += 1
        elif skill_hit(name, BONUS_SKILLS):
            bonus_score += skill_w * 0.4
            bonus_hits += 1

    # Normalise: required capped at 8 hits (most relevant skills)
    req_norm = min(req_score / 8.0, 1.0)
    bonus_norm = min(bonus_score / 5.0, 0.3)

    return min(req_norm + bonus_norm, 1.0), req_hits, bonus_hits


def score_career(c: dict) -> float:
    """Score career history — product company exp, AI/ML titles, tenure."""
    career = c.get("career_history", [])
    profile = c.get("profile", {})
    education = c.get("education", [])

    yrs_exp = profile.get("years_of_experience", 0)
    current_title = normalize(profile.get("current_title", ""))
    current_industry = normalize(profile.get("current_industry", ""))

    # Experience years vs JD requirement (5-9 ideal)
    if yrs_exp < 2:
        exp_score = 0.1
    elif yrs_exp < 4:
        exp_score = 0.4
    elif yrs_exp <= 9:
        exp_score = 0.9 + (yrs_exp - 5) * 0.02 if yrs_exp >= 5 else 0.6
    else:
        # Over 9 years: still OK but JD says they prefer the 5-9 range
        exp_score = max(0.7, 0.9 - (yrs_exp - 9) * 0.02)

    # Current title relevance
    ai_titles = {
        "ai engineer", "ml engineer", "machine learning engineer",
        "data scientist", "nlp engineer", "research engineer",
        "applied scientist", "senior engineer", "software engineer",
        "backend engineer", "search engineer", "recommendations engineer",
        "retrieval", "embedding", "llm", "ranking engineer"
    }
    title_score = 0.0
    for t in ai_titles:
        if t in current_title:
            title_score = 0.9
            break
    if title_score == 0.0:
        if any(t in current_title for t in ["engineer", "developer", "architect"]):
            title_score = 0.5
        elif any(t in current_title for t in ["scientist", "analyst", "researcher"]):
            title_score = 0.4
        elif any(t in current_title for t in IRRELEVANT_TITLES):
            title_score = 0.05

    # Career history quality
    career_score = 0.0
    consulting_only = True
    product_company_months = 0

    for job in career:
        co = normalize(job.get("company", ""))
        title = normalize(job.get("title", ""))
        industry = normalize(job.get("industry", ""))
        dm = job.get("duration_months", 0)
        size = job.get("company_size", "1-10")

        is_consulting = any(cf in co for cf in CONSULTING_FIRMS)
        if not is_consulting:
            consulting_only = False
            product_company_months += dm

        # AI/ML industry bonus
        ai_industries = {"artificial intelligence", "machine learning", "technology", "software", "it", "saas", "fintech"}
        if any(ai in industry for ai in ai_industries):
            career_score += dm * 0.02
        else:
            career_score += dm * 0.005

        # Role relevance
        for t in ai_titles:
            if t in title:
                career_score += dm * 0.015
                break

    # Normalise career score (cap at ~60 months of perfect experience)
    career_norm = min(career_score / 1.2, 1.0)

    # Penalise consulting-only background (per JD)
    if consulting_only and len(career) > 0:
        career_norm *= 0.3

    # Education tier bonus
    edu_score = 0.5
    for edu in education:
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            edu_score = 1.0
            break
        elif tier == "tier_2":
            edu_score = max(edu_score, 0.8)
        elif tier == "tier_3":
            edu_score = max(edu_score, 0.6)

    # Combine
    return (exp_score * 0.35 + title_score * 0.35 + career_norm * 0.2 + edu_score * 0.1)


def score_behavioral(c: dict) -> float:
    """Behavioral signal multiplier — 0.3 to 1.2"""
    sig = c.get("redrob_signals", {})

    # Recency — how recently active
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    if days_inactive < 7:
        recency = 1.0
    elif days_inactive < 30:
        recency = 0.95
    elif days_inactive < 90:
        recency = 0.80
    elif days_inactive < 180:
        recency = 0.60
    else:
        recency = 0.35

    # Open to work
    open_w = 1.1 if sig.get("open_to_work_flag") else 0.85

    # Response rate
    rr = sig.get("recruiter_response_rate", 0)
    response_w = 0.6 + 0.4 * rr

    # Response time (lower is better; cap at 48h ideal)
    rt = sig.get("avg_response_time_hours", 999)
    if rt <= 4:
        rt_w = 1.0
    elif rt <= 24:
        rt_w = 0.9
    elif rt <= 72:
        rt_w = 0.75
    else:
        rt_w = 0.55

    # Interview completion
    icr = sig.get("interview_completion_rate", 0.5)
    interview_w = 0.5 + 0.5 * icr

    # GitHub activity (AI role — this matters)
    github = sig.get("github_activity_score", -1)
    if github < 0:
        github_w = 0.85  # no GitHub linked, slight penalty
    elif github < 20:
        github_w = 0.9
    elif github < 50:
        github_w = 1.0
    else:
        github_w = 1.05

    # Notice period (JD wants ≤30 days)
    notice = sig.get("notice_period_days", 60)
    if notice <= 15:
        notice_w = 1.05
    elif notice <= 30:
        notice_w = 1.0
    elif notice <= 60:
        notice_w = 0.92
    elif notice <= 90:
        notice_w = 0.85
    else:
        notice_w = 0.75

    # Salary fit (JD is a senior founding-team role; assume ~30-60 LPA range)
    sal = sig.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0)
    sal_max = sal.get("max", 999)
    sal_mid = (sal_min + sal_max) / 2
    if 25 <= sal_mid <= 70:
        salary_w = 1.0
    elif sal_mid < 20:
        salary_w = 0.9  # might be junior
    elif sal_mid > 100:
        salary_w = 0.85  # might be overpriced
    else:
        salary_w = 0.95

    # Location preference
    location = normalize(c.get("profile", {}).get("location", ""))
    country = normalize(c.get("profile", {}).get("country", ""))
    willing_relocate = sig.get("willing_to_relocate", False)
    work_mode = sig.get("preferred_work_mode", "flexible")

    if any(loc in location for loc in PREFERRED_LOCATIONS) or "india" in country:
        location_w = 1.0
    elif willing_relocate:
        location_w = 0.9
    else:
        location_w = 0.7

    # Profile completeness
    completeness = sig.get("profile_completeness_score", 50) / 100
    complete_w = 0.7 + 0.3 * completeness

    # Saved by recruiters (market signal)
    saved = min(sig.get("saved_by_recruiters_30d", 0), 20)
    saved_w = 1.0 + saved * 0.003

    # Combine behavioral signals
    behavioral = (
        recency * 0.20 +
        open_w * 0.15 +
        response_w * 0.15 +
        rt_w * 0.08 +
        interview_w * 0.10 +
        github_w * 0.10 +
        notice_w * 0.08 +
        salary_w * 0.05 +
        location_w * 0.05 +
        complete_w * 0.02 +
        saved_w * 0.02
    )

    return behavioral


JD_TEXT = """
senior ai engineer embeddings retrieval ranking vector database search
nlp natural language processing python production deployment
sentence transformers faiss pinecone weaviate qdrant milvus elasticsearch
opensearch hybrid search dense retrieval sparse retrieval
ndcg mrr map evaluation framework learning to rank reranking
recommendation system information retrieval rag retrieval augmented generation
lora qlora fine tuning peft transformer bert llm large language model
product company startup scrappy ship ranker inference optimization
""".strip()


def score_trajectory(c: dict) -> float:
    """Detects upward AI/ML career trajectory."""
    career = c.get("career_history", [])
    if len(career) < 2:
        return 0.5

    def get_start(job):
        try:
            return date.fromisoformat(job.get("start_date", "2000-01-01"))
        except Exception:
            return date(2000, 1, 1)

    sorted_career = sorted(career, key=get_start)

    ai_role_keywords = {
        "ai", "ml", "machine learning", "nlp", "data scientist",
        "research engineer", "applied scientist", "search engineer",
        "ranking", "retrieval", "embedding", "llm", "deep learning",
        "computer vision", "recommendation"
    }
    non_ai_keywords = {
        "marketing", "sales", "hr", "finance", "accounting",
        "civil", "mechanical", "graphic", "content", "support"
    }

    def role_ai_score(title: str) -> float:
        t = title.lower()
        if any(k in t for k in ai_role_keywords):
            return 1.0
        elif any(k in t for k in ["engineer", "developer", "scientist", "architect"]):
            return 0.6
        elif any(k in t for k in non_ai_keywords):
            return 0.1
        return 0.4

    scores = [role_ai_score(j.get("title", "")) for j in sorted_career]

    if len(scores) >= 2:
        early_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        recent_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        delta = recent_avg - early_avg

        if delta > 0.3:
            return 0.9
        elif delta > 0.1:
            return 0.75
        elif delta > -0.1:
            return 0.6
        else:
            return 0.35
    return 0.5


def score_assessments(c: dict) -> float:
    """Boost candidates with strong Redrob assessments."""
    sig = c.get("redrob_signals", {})
    assessments = sig.get("skill_assessment_scores", {})

    if not assessments:
        return 0.4

    ai_assessment_keys = {
        "nlp", "machine learning", "python", "deep learning",
        "information retrieval", "search", "embeddings", "sql",
        "data structures", "algorithms", "system design"
    }

    total_score = 0.0
    total_weight = 0.0

    for skill, score in assessments.items():
        skill_lower = skill.lower()
        weight = 2.0 if any(k in skill_lower for k in ai_assessment_keys) else 1.0
        total_score += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.4

    avg = total_score / total_weight
    normalized = avg / 100.0
    count_bonus = min(len(assessments) / 5, 1.0) * 0.1

    return min(normalized + count_bonus, 1.0)


def compute_tfidf_scores(candidates_file: str = "candidates.jsonl") -> dict:
    """Build TF-IDF similarity scores between each candidate and the JD."""
    import re

    def tokenize(text: str) -> list:
        return re.findall(r"[a-z0-9]+", text.lower())

    jd_tokens = set(tokenize(JD_TEXT))

    print("  [TF-IDF] Computing document frequencies...")
    df = {}
    total_docs = 0

    with open(candidates_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            total_docs += 1
            text = _candidate_text(c)
            seen = set(tokenize(text))
            for token in seen:
                if token in jd_tokens:
                    df[token] = df.get(token, 0) + 1

    idf = {}
    for token, count in df.items():
        idf[token] = math.log((total_docs + 1) / (count + 1)) + 1

    print("  [TF-IDF] Scoring candidates...")
    scores = {}
    jd_vec = {t: idf.get(t, 1.0) for t in jd_tokens}
    jd_norm = math.sqrt(sum(v**2 for v in jd_vec.values()))

    with open(candidates_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue

            cid = c["candidate_id"]
            text = _candidate_text(c)
            tokens = tokenize(text)

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            total = len(tokens) or 1

            dot = 0.0
            cand_norm = 0.0
            for token, count in tf.items():
                tf_val = count / total
                idf_val = idf.get(token, 0)
                tfidf_val = tf_val * idf_val
                cand_norm += tfidf_val ** 2
                if token in jd_vec:
                    dot += tfidf_val * jd_vec[token]

            cand_norm = math.sqrt(cand_norm)
            if cand_norm > 0 and jd_norm > 0:
                scores[cid] = dot / (cand_norm * jd_norm)
            else:
                scores[cid] = 0.0

    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}

    return scores


def _candidate_text(c: dict) -> str:
    """Extract all text from a candidate profile for TF-IDF."""
    parts = []
    p = c.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(p.get("current_title", ""))
    parts.append(p.get("current_industry", ""))

    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))

    for s in c.get("skills", []):
        parts.append(s.get("name", ""))

    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " ".join(parts)


def compute_score(c: dict, tfidf_score: float = 0.5) -> float | None:
    """Compute final score. Returns None if honeypot."""
    if is_honeypot(c):
        return None

    skill_score, req_hits, bonus_hits = score_skills(c)
    career_score = score_career(c)
    behavioral = score_behavioral(c)
    trajectory = score_trajectory(c)
    assessment = score_assessments(c)

    profile_score = (
        skill_score * 0.35 +
        career_score * 0.30 +
        tfidf_score * 0.15 +
        trajectory * 0.10 +
        assessment * 0.10
    )

    final = profile_score * behavioral
    return min(final, 1.0)


def build_reasoning(c: dict, rank: int, score: float, req_hits: int) -> str:
    """Build honest 1-2 sentence reasoning per candidate."""
    profile = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    title = profile.get("current_title", "")
    yrs = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")
    rr = sig.get("recruiter_response_rate", 0)
    notice = sig.get("notice_period_days", 0)
    open_w = sig.get("open_to_work_flag", False)
    days_inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    github = sig.get("github_activity_score", -1)

    # Top skills matching JD
    matched = [s["name"] for s in skills if skill_hit(s.get("name", ""), REQUIRED_SKILLS)][:4]
    matched_str = ", ".join(matched) if matched else "limited AI core skills"

    # Concern flags
    concerns = []
    if days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days")
    if rr < 0.2:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if notice > 60:
        concerns.append(f"long notice period ({notice}d)")
    if not open_w:
        concerns.append("not marked open to work")

    loc_str = f"{location}, {country}" if location and country else (location or country or "unknown location")

    part1 = f"{title} with {yrs:.1f} yrs exp at {company} ({loc_str}); {req_hits} JD-relevant skills incl. {matched_str}."
    if concerns:
        part2 = f"Concerns: {'; '.join(concerns)}."
    elif rank <= 10:
        github_str = f" GitHub score {github:.0f}." if github > 0 else ""
        part2 = f"Strong engagement signals — response rate {rr:.0%}, notice {notice}d.{github_str}"
    else:
        part2 = f"Recruiter response rate {rr:.0%}; {'open to work' if open_w else 'not actively looking'}."

    return f"{part1} {part2}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading candidates from {CANDIDATES_FILE}...")
    scored = []
    honeypot_count = 0
    total = 0

    print("Computing TF-IDF semantic match scores...")
    tfidf_scores = compute_tfidf_scores()
    print(f"  TF-IDF done. Scoring {len(tfidf_scores):,} candidates.")

    with open(CANDIDATES_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            if total % 10000 == 0:
                print(f"  Processed {total:,} candidates...")

            tfidf = tfidf_scores.get(c.get("candidate_id", ""), 0.5)
            score = compute_score(c, tfidf_score=tfidf)
            if score is None:
                honeypot_count += 1
                continue

            _, req_hits, bonus_hits = score_skills(c)
            scored.append((score, req_hits, c))

    print(f"Total: {total:,} | Honeypots detected: {honeypot_count} | Scored: {len(scored):,}")

    # Sort by score desc, then candidate_id asc for ties
    scored.sort(key=lambda x: (-round(x[0], 4), x[2]["candidate_id"]))

    top100 = sorted(scored[:TOP_N], key=lambda x: (-round(x[0], 4), x[2]["candidate_id"]))

    # Write submission CSV
    print(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (score, req_hits, c) in enumerate(top100, start=1):
            cid = c["candidate_id"]
            reasoning = build_reasoning(c, rank, score, req_hits)
            # Clamp score to 4 decimal places, ensure monotone
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"Done! Top candidate: {top100[0][2]['candidate_id']} score={top100[0][0]:.4f}")
    print(f"Rank 10 score: {top100[9][0]:.4f}")
    print(f"Rank 100 score: {top100[99][0]:.4f}")

    # Quick sanity preview
    print("\n--- Top 10 ---")
    for rank, (score, req_hits, c) in enumerate(top100[:10], start=1):
        p = c["profile"]
        print(f"  #{rank:2d} {c['candidate_id']} | {p['current_title'][:30]:30s} | {p['years_of_experience']}yrs | req_hits={req_hits} | score={score:.4f}")


if __name__ == "__main__":
    main()
