import re
from typing import Dict, List


def classify_text(text: str) -> Dict:
    """
    Lightweight Bayesian-style prior classifier.
    This does not produce the final result by itself.
    It estimates a base narrative-risk prior from rhetoric,
    speculation, persuasion, and manipulation signals.
    """

    text_lower = text.lower()
    matched_terms: List[str] = []
    score = 0.12

    high_risk_terms = [
        "they don't want you to know",
        "the truth they do not want you to know",
        "deep state",
        "cover-up",
        "conspiracy",
        "propaganda machine",
        "rigged",
        "stolen election",
        "globalist",
        "traitor",
        "enemy within"
    ]

    medium_risk_terms = [
        "shocking",
        "explosive",
        "secret",
        "exposed",
        "banned",
        "urgent",
        "radical",
        "extreme",
        "weaponized",
        "sinister",
        "collapse",
        "meltdown",
        "panic",
        "outrageous",
        "humiliating"
    ]

    speculation_terms = [
        "reportedly",
        "claims",
        "unconfirmed",
        "anonymous sources",
        "must mean",
        "can only mean",
        "no doubt",
        "obviously",
        "clearly"
    ]

    ideological_terms = [
        "far-left",
        "far-right",
        "radical left",
        "radical right",
        "regime",
        "fascist",
        "socialist agenda"
    ]

    fear_terms = [
        "chaos",
        "catastrophe",
        "nightmare",
        "on the brink",
        "under siege",
        "coming for you",
        "terrifying",
        "devastating"
    ]

    attack_verbs = [
        "slammed",
        "blasted",
        "destroyed",
        "obliterated",
        "crushed",
        "humiliated",
        "rips into",
        "tears into"
    ]

    for term in high_risk_terms:
        if term in text_lower:
            matched_terms.append(term)
            score += 0.12

    for term in medium_risk_terms:
        if term in text_lower:
            matched_terms.append(term)
            score += 0.07

    for term in speculation_terms:
        if term in text_lower:
            matched_terms.append(term)
            score += 0.05

    for term in ideological_terms:
        if term in text_lower:
            matched_terms.append(term)
            score += 0.08

    for term in fear_terms:
        if term in text_lower:
            matched_terms.append(term)
            score += 0.06

    for term in attack_verbs:
        if term in text_lower:
            matched_terms.append(term)
            score += 0.07

    exclamation_count = text.count("!")
    question_count = text.count("?")
    all_caps_words = re.findall(r"\b[A-Z]{4,}\b", text)

    if exclamation_count >= 1:
        matched_terms.append("exclamatory_style")
        score += 0.04

    if exclamation_count >= 2:
        score += 0.03

    if question_count >= 2:
        matched_terms.append("dramatic_questioning")
        score += 0.03

    if "?!" in text or "!?" in text:
        matched_terms.append("mixed_punctuation_emphasis")
        score += 0.04

    if all_caps_words:
        matched_terms.append("all_caps_emphasis")
        score += min(len(all_caps_words) * 0.03, 0.10)

    loaded_pattern = re.search(
        r"\b(shocking|dangerous|radical|reckless|disastrous|humiliating|terrifying)\s+"
        r"(plan|policy|move|agenda|decision|attack|response|failure)\b",
        text_lower,
    )
    if loaded_pattern:
        matched_terms.append("loaded_descriptor_pattern")
        score += 0.06

    neutral_terms = [
        "said", "stated", "announced", "reported", "confirmed",
        "according to", "officials said", "statement"
    ]
    if any(term in text_lower for term in neutral_terms) and score > 0.18:
        score -= 0.04

    score = max(0.05, min(score, 0.95))

    if score >= 0.70:
        label = "high_risk"
    elif score >= 0.40:
        label = "medium_risk"
    else:
        label = "low_risk"

    return {
        "label": label,
        "risk_score": round(score, 4),
        "matched_terms": sorted(set(matched_terms)),
    }
