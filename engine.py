import math
import os
import random
import re
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv

from bayes import classify_text
from local_ai_model import LocalAIModel

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
USER_AGENT = os.getenv("USER_AGENT", "news-validator/1.0")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.35"))
SVD_K = int(os.getenv("SVD_K", "64"))
COORD_THRESHOLD = float(os.getenv("COORD_THRESHOLD", "0.90"))

_LOCAL_MODEL = None


def get_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        _LOCAL_MODEL = LocalAIModel()
    return _LOCAL_MODEL


def simple_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\b\w+\b", a.lower()))
    b_tokens = set(re.findall(r"\b\w+\b", b.lower()))

    if not a_tokens or not b_tokens:
        return 0.0

    overlap = len(a_tokens.intersection(b_tokens))
    denom = max(len(a_tokens), len(b_tokens))
    return round(overlap / denom, 4)


def detect_language(text: str) -> str:
    english_chars = re.findall(r"[A-Za-z]", text)
    if len(english_chars) >= 10:
        return "en"
    return "unknown"


def guess_categories(text: str) -> List[str]:
    text_lower = text.lower()
    categories = []

    rules = {
        "politics": [
            "election", "president", "government", "minister",
            "congress", "parliament", "cabinet", "vote", "coalition"
        ],
        "war_conflict": [
            "war", "missile", "invasion", "military", "troops",
            "attack", "strike", "drone", "conflict", "clash"
        ],
        "economy": [
            "inflation", "market", "stocks", "oil", "gas prices",
            "economy", "trade", "tariff", "gdp", "central bank"
        ],
        "health": [
            "virus", "hospital", "vaccine", "disease", "health", "outbreak"
        ],
        "technology": [
            "ai", "artificial intelligence", "openai", "google",
            "chip", "technology", "cyberattack", "cybersecurity"
        ],
        "energy": [
            "energy", "oil", "gas", "lng", "pipeline", "refinery", "shipping"
        ],
        "climate": [
            "climate", "wildfire", "emissions", "storm", "flood", "drought"
        ],
    }

    for category, keywords in rules.items():
        if any(word in text_lower for word in keywords):
            categories.append(category)

    return categories or ["general"]


def extract_domains_from_text(text: str) -> List[str]:
    urls = re.findall(r"https?://[^\s]+", text)
    domains = []

    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc:
            domains.append(parsed.netloc.lower())

    return list(sorted(set(domains)))


def infer_topic(text: str) -> str:
    text_lower = text.lower()

    topic_rules = {
        "geopolitics": [
            "war", "conflict", "military", "missile", "drone",
            "diplomacy", "sanction", "border", "troops"
        ],
        "energy": [
            "oil", "gas", "lng", "pipeline", "refinery", "opec", "fuel"
        ],
        "economy": [
            "inflation", "market", "stocks", "gdp", "trade",
            "tariff", "recession", "central bank", "interest rate"
        ],
        "elections": [
            "election", "vote", "campaign", "parliament", "coalition"
        ],
        "health": [
            "health", "virus", "disease", "vaccine", "outbreak"
        ],
        "technology": [
            "technology", "ai", "cyber", "semiconductor", "data breach"
        ],
        "climate": [
            "climate", "wildfire", "flood", "storm", "emissions", "drought"
        ],
    }

    for topic, keywords in topic_rules.items():
        if any(word in text_lower for word in keywords):
            return topic

    return "other"


def infer_event_type(text: str) -> str:
    text_lower = text.lower()

    if any(term in text_lower for term in ["missile", "attack", "strike", "drone", "troops", "war"]):
        return "military_escalation"
    if any(term in text_lower for term in ["talks", "negotiation", "mediation", "truce", "ceasefire"]):
        return "diplomatic_negotiation"
    if any(term in text_lower for term in ["sanction", "embargo", "export controls", "tariff"]):
        return "sanctions_trade_action"
    if any(term in text_lower for term in ["oil", "gas", "pipeline", "refinery", "shipping", "lng"]):
        return "energy_supply_event"
    if any(term in text_lower for term in ["inflation", "rates", "recession", "gdp", "market"]):
        return "macroeconomic_shift"
    if any(term in text_lower for term in ["election", "vote", "campaign", "coalition"]):
        return "election_instability"
    if any(term in text_lower for term in ["cyberattack", "ransomware", "network outage", "data breach"]):
        return "cyber_infrastructure_event"
    if any(term in text_lower for term in ["virus", "vaccine", "hospital", "outbreak"]):
        return "public_health_event"
    if any(term in text_lower for term in ["climate", "wildfire", "flood", "storm", "drought"]):
        return "climate_disruption"

    return "general_reporting"


def infer_primary_region(text: str) -> str:
    text_lower = text.lower()

    region_rules = {
        "Middle East": ["iran", "israel", "gaza", "tehran", "saudi", "uae", "qatar", "syria", "iraq"],
        "Europe": ["ukraine", "russia", "eu", "germany", "france", "britain", "nato"],
        "Asia-Pacific": ["china", "taiwan", "japan", "korea", "philippines", "india", "pakistan"],
        "Africa": ["sudan", "niger", "ethiopia", "sahel", "nigeria", "congo"],
        "Americas": ["united states", "u.s.", "us ", "canada", "mexico", "brazil"],
        "Global": ["opec", "g7", "g20", "world bank", "imf", "united nations"],
    }

    for region, keywords in region_rules.items():
        if any(word in text_lower for word in keywords):
            return region

    return "Unknown"


def extract_countries(text: str) -> List[str]:
    text_lower = text.lower()
    known = {
        "Iran": ["iran", "tehran"],
        "United States": ["united states", "u.s.", "us "],
        "Israel": ["israel"],
        "Russia": ["russia"],
        "Ukraine": ["ukraine"],
        "China": ["china"],
        "Taiwan": ["taiwan"],
        "India": ["india"],
        "Pakistan": ["pakistan"],
        "Saudi Arabia": ["saudi arabia", "saudi"],
        "Japan": ["japan"],
        "Germany": ["germany"],
        "France": ["france"],
        "United Kingdom": ["britain", "united kingdom", "uk "],
    }

    result = []
    for country, markers in known.items():
        if any(marker in text_lower for marker in markers):
            result.append(country)
    return result


def extract_actors(text: str) -> List[str]:
    text_lower = text.lower()
    actors = []

    actor_terms = [
        "government", "military", "army", "navy", "central bank",
        "president", "prime minister", "parliament", "opposition",
        "rebels", "officials", "state media", "defense ministry"
    ]

    for term in actor_terms:
        if term in text_lower:
            actors.append(term)

    return sorted(set(actors))


def compute_rhetoric_features(text: str) -> Dict:
    lower = text.lower()
    labels = []
    score = 0

    loaded_adjectives = [
        "shocking", "explosive", "outrageous", "disgraceful", "devastating",
        "terrifying", "radical", "extreme", "sinister", "weaponized",
        "dangerous", "humiliating", "catastrophic", "disastrous", "reckless"
    ]

    attack_verbs = [
        "slammed", "blasted", "crushed", "destroyed", "obliterated",
        "exposed", "humiliated", "attacked", "lashes out", "lashed out",
        "tears into", "rips into", "accuses", "denounces"
    ]

    fear_framing = [
        "chaos", "disaster", "meltdown", "panic", "on the brink",
        "under siege", "coming for you", "threatens", "fear grows",
        "collapse", "turmoil", "alarm", "nightmare"
    ]

    certainty_framing = [
        "proves", "clearly", "obviously", "no doubt",
        "unquestionably", "without question", "this changes everything"
    ]

    ideological_labels = [
        "far-left", "far right", "far-right", "regime", "globalist",
        "fascist", "socialist agenda", "deep state", "propaganda machine",
        "elite agenda", "radical left", "radical right"
    ]

    persuasion_phrases = [
        "the truth they do not want you to know",
        "the truth they don't want you to know",
        "americans should be worried",
        "complete disaster",
        "this changes everything",
        "what happens next will shock you",
        "they don't want you to know"
    ]

    moral_judgment = [
        "betrayal", "shameful", "corrupt", "failure", "blunder",
        "disgrace", "traitor", "enemy", "lies", "lying", "deception",
        "dishonest", "propaganda"
    ]

    speculation_as_fact = [
        "must mean", "can only mean", "proves that", "there is no doubt",
        "it is obvious that", "undeniably"
    ]

    neutral_reporting = [
        "said", "stated", "announced", "reported", "confirmed",
        "according to", "officials said", "statement", "talks", "meeting"
    ]

    if any(term in lower for term in loaded_adjectives):
        labels.append("loaded_adjectives")
        score += 16
    if any(term in lower for term in attack_verbs):
        labels.append("attack_adversarial_verbs")
        score += 18
    if any(term in lower for term in fear_framing):
        labels.append("fear_crisis_framing")
        score += 14
    if any(term in lower for term in certainty_framing):
        labels.append("certainty_without_evidence")
        score += 14
    if any(term in lower for term in ideological_labels):
        labels.append("ideological_labeling")
        score += 20
    if any(term in lower for term in persuasion_phrases):
        labels.append("direct_persuasion")
        score += 22
    if any(term in lower for term in moral_judgment):
        labels.append("moral_judgment_blame")
        score += 16
    if any(term in lower for term in speculation_as_fact):
        labels.append("speculation_presented_as_fact")
        score += 16

    exclamations = text.count("!")
    questions = text.count("?")

    if exclamations >= 1:
        labels.append("exclamatory_style")
        score += 6
    if exclamations >= 2:
        score += 4
    if questions >= 2:
        labels.append("dramatic_questioning")
        score += 5
    if "?!" in text or "!?" in text:
        score += 6

    all_caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if all_caps_words:
        labels.append("all_caps_emphasis")
        score += min(len(all_caps_words) * 4, 12)

    loaded_pattern = re.search(
        r"\b(shocking|dangerous|radical|reckless|disastrous|humiliating|terrifying)\s+"
        r"(plan|policy|move|agenda|decision|attack|response|failure)\b",
        lower,
    )
    if loaded_pattern:
        labels.append("loaded_descriptor_pattern")
        score += 10

    attack_pattern = re.search(
        r"\b(slammed|blasted|crushed|destroyed|exposed|humiliated)\b",
        lower,
    )
    if attack_pattern:
        labels.append("combat_style_framing")
        score += 10

    if any(term in lower for term in neutral_reporting) and score > 0:
        score -= 4

    if len(labels) >= 3:
        score += 6
    if len(labels) >= 5:
        score += 6

    score = max(0, min(score, 100))

    if score == 0:
        label = "low"
        analysis = "The headline appears highly restrained and mostly informational."
    elif score < 20:
        label = "low"
        analysis = "The headline is mostly neutral with only light rhetorical framing."
    elif score < 40:
        label = "moderate"
        analysis = "The headline contains noticeable persuasive or emotionally loaded wording."
    elif score < 65:
        label = "moderate"
        analysis = "The headline uses substantial rhetorical framing and adversarial language."
    else:
        label = "high"
        analysis = "The headline is strongly rhetorical, relying on loaded, ideological, or emotionally manipulative framing."

    return {
        "score": int(score),
        "label": label,
        "signals": sorted(set(labels)),
        "analysis": analysis,
    }


def bayesian_posterior(prior: float, likelihood_ratios: List[float]) -> float:
    safe_prior = min(max(prior, 0.001), 0.999)
    odds = safe_prior / (1.0 - safe_prior)

    for lr in likelihood_ratios:
        odds *= max(lr, 0.05)

    return max(0.0, min(odds / (1.0 + odds), 1.0))


def bounded_noise() -> float:
    a = random.uniform(-1.0, 1.0)
    b = random.uniform(-1.0, 1.0)
    return max(-1.0, min((a + b) / 2.0, 1.0))


def sample_std_dev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def compute_risk_drivers(
    text: str,
    credibility_score: float,
    coverage_percent: float,
    coordination_score: float,
    max_hist_sim: float,
    rhetoric_score: float,
) -> Dict:
    text_lower = text.lower()

    oil_terms = [
        "oil", "crude", "brent", "wti", "energy", "exports", "supply",
        "shipping", "pipeline", "refinery", "gas", "tanker", "opec",
        "lng", "diesel", "petrol", "fuel", "offshore", "production",
        "output", "barrel", "barrels", "natural gas", "energy market"
    ]
    chokepoint_terms = [
        "strait", "canal", "shipping lane", "sea lane", "port", "harbor",
        "transit route", "maritime route", "blockade", "bottleneck",
        "red sea", "black sea", "south china sea", "taiwan strait",
        "suez", "panama canal", "bab el-mandeb", "hormuz"
    ]
    escalation_terms = [
        "attack", "strike", "missile", "drone", "threat", "military",
        "invasion", "warning", "conflict", "war", "troops", "retaliation",
        "showdown", "airstrike", "shelling", "mobilization", "offensive",
        "defense ministry", "naval", "fighter jet", "incursion", "clash",
        "border tensions", "armed forces", "hostilities"
    ]
    diplomacy_terms = [
        "ceasefire", "truce", "talks", "negotiation", "mediation",
        "diplomacy", "peace plan", "summit", "dialogue", "agreement",
        "backchannel", "delegation", "envoy"
    ]
    volatility_terms = [
        "panic", "uncertainty", "shock", "volatility", "selloff",
        "surge", "plunge", "crash", "turmoil", "jitters", "risk-off",
        "market stress", "instability", "uncertain outlook"
    ]
    economy_terms = [
        "inflation", "economy", "market", "recession", "gdp", "trade",
        "stocks", "bond", "currency", "policy", "prices", "tariff",
        "interest rate", "rate cut", "rate hike", "central bank",
        "consumer prices", "manufacturing", "exports", "imports",
        "fiscal", "deficit", "debt", "employment", "unemployment"
    ]
    sanction_terms = [
        "sanction", "embargo", "export controls", "trade restriction",
        "tariffs", "blacklist", "secondary sanctions", "asset freeze"
    ]
    election_terms = [
        "election", "vote", "ballot", "campaign", "parliament",
        "president", "prime minister", "coalition", "opposition",
        "cabinet", "leadership contest", "referendum"
    ]
    infrastructure_terms = [
        "power grid", "blackout", "substation", "terminal", "port closure",
        "pipeline explosion", "refinery fire", "plant shutdown",
        "rail disruption", "airport closure", "factory shutdown"
    ]
    cyber_terms = [
        "cyberattack", "ransomware", "network outage", "data breach",
        "critical infrastructure", "satellite disruption", "communications outage"
    ]

    oil_count = sum(1 for term in oil_terms if term in text_lower)
    chokepoint_count = sum(1 for term in chokepoint_terms if term in text_lower)
    escalation_count = sum(1 for term in escalation_terms if term in text_lower)
    diplomacy_count = sum(1 for term in diplomacy_terms if term in text_lower)
    volatility_count = sum(1 for term in volatility_terms if term in text_lower)
    economy_count = sum(1 for term in economy_terms if term in text_lower)
    sanction_count = sum(1 for term in sanction_terms if term in text_lower)
    election_count = sum(1 for term in election_terms if term in text_lower)
    infrastructure_count = sum(1 for term in infrastructure_terms if term in text_lower)
    cyber_count = sum(1 for term in cyber_terms if term in text_lower)

    cred_factor = max(0.0, min(credibility_score / 100.0, 1.0))
    rhet_factor = max(0.0, min(rhetoric_score / 100.0, 1.0))
    coverage_factor = max(0.0, min(coverage_percent / 100.0, 1.0))
    hist_factor = max(0.0, min(max_hist_sim, 1.0))
    coordination_factor = max(0.0, min(coordination_score / 100.0, 1.0))

    oil_prior = 0.20
    escalation_prior = 0.24
    volatility_prior = 0.22

    oil_samples = []
    escalation_samples = []
    volatility_samples = []

    for _ in range(400):
        noise_oil = bounded_noise()
        noise_esc = bounded_noise()
        noise_vol = bounded_noise()

        oil_posterior = bayesian_posterior(
            oil_prior,
            [1.55 + noise_oil * 0.16] * oil_count
            + [1.42 + noise_oil * 0.12] * min(chokepoint_count, 3)
            + [1.24 + noise_oil * 0.08] * min(sanction_count, 3)
            + [1.30 + noise_oil * 0.10] * min(infrastructure_count, 2)
            + [1.10 + noise_oil * 0.05] * min(escalation_count, 3)
            + [1.08 + noise_oil * 0.05] * min(economy_count, 2)
            + ([1.10] if cred_factor > 0.60 else [])
            + ([1.08] if coverage_factor > 0.40 else [])
            + ([1.06] if hist_factor > 0.45 else [])
        )

        escalation_posterior = bayesian_posterior(
            escalation_prior,
            [1.75 + noise_esc * 0.18] * escalation_count
            + [1.12 + noise_esc * 0.05] * min(chokepoint_count, 2)
            + [1.18 + noise_esc * 0.07] * min(cyber_count, 2)
            + [1.10 + noise_esc * 0.05] * min(sanction_count, 2)
            + ([0.92] if diplomacy_count > 0 else [])
            + ([1.08] if rhet_factor > 0.45 else [])
            + ([1.07] if cred_factor > 0.60 else [])
            + ([1.05] if coordination_factor > 0.25 else [])
        )

        volatility_posterior = bayesian_posterior(
            volatility_prior,
            [1.55 + noise_vol * 0.16] * volatility_count
            + [1.24 + noise_vol * 0.08] * min(economy_count, 4)
            + [1.12 + noise_vol * 0.05] * min(election_count, 3)
            + [1.18 + noise_vol * 0.06] * min(cyber_count, 2)
            + [1.16 + noise_vol * 0.06] * min(infrastructure_count, 2)
            + [1.10 + noise_vol * 0.05] * min(sanction_count, 2)
            + ([1.06] if coverage_factor > 0.35 else [])
            + ([1.08] if rhet_factor > 0.50 else [])
        )

        coupled_oil = min(max(
            oil_posterior
            + 0.10 * escalation_posterior
            + 0.08 * (min(chokepoint_count, 2) / 2.0)
            + 0.05 * (min(sanction_count, 2) / 2.0)
            + 0.04 * hist_factor,
            0.0,
        ), 1.0)

        coupled_escalation = min(max(
            escalation_posterior
            + 0.05 * volatility_posterior
            + 0.04 * (min(cyber_count, 2) / 2.0),
            0.0,
        ), 1.0)

        coupled_volatility = min(max(
            volatility_posterior
            + 0.16 * escalation_posterior
            + 0.12 * oil_posterior
            + 0.06 * (min(election_count, 2) / 2.0)
            + 0.05 * (min(cyber_count, 2) / 2.0)
            + 0.05 * coverage_factor,
            0.0,
        ), 1.0)

        oil_samples.append(coupled_oil)
        escalation_samples.append(coupled_escalation)
        volatility_samples.append(coupled_volatility)

    oil_prob = sum(oil_samples) / len(oil_samples)
    escalation_prob = sum(escalation_samples) / len(escalation_samples)
    volatility_prob = sum(volatility_samples) / len(volatility_samples)

    return {
        "energySupplyDisruption": round(oil_prob, 4),
        "militaryEscalation": round(escalation_prob, 4),
        "marketUncertainty": round(volatility_prob, 4),
        "sanctionsTradeRestrictions": round(min(1.0, sanction_count * 0.22), 4),
        "cyberInfrastructureRisk": round(min(1.0, cyber_count * 0.30), 4),
        "electionPoliticalInstability": round(min(1.0, election_count * 0.22), 4),
        "oilRelevance": round(min(1.0, (oil_count + chokepoint_count + sanction_count) / 6.0), 4),
        "marketRelevance": round(min(1.0, (volatility_count + economy_count + election_count) / 6.0), 4),
        "variance": round(
            (
                ((oil_prob - ((oil_prob + escalation_prob + volatility_prob) / 3.0)) ** 2) +
                ((escalation_prob - ((oil_prob + escalation_prob + volatility_prob) / 3.0)) ** 2) +
                ((volatility_prob - ((oil_prob + escalation_prob + volatility_prob) / 3.0)) ** 2)
            ) / 3.0,
            6,
        ),
        "mcUncertainty": round(
            (sample_std_dev(oil_samples) + sample_std_dev(escalation_samples) + sample_std_dev(volatility_samples)) / 3.0,
            6,
        ),
    }


def compute_credibility_style(
    text: str,
    credibility_score: float,
    rhetoric_score: float,
    bayes_result: Dict,
) -> Dict:
    text_lower = text.lower()

    opinion_probability = 0.15
    speculation_level = 0.12
    evidence_strength = 0.55
    mode = "reporting"

    if any(term in text_lower for term in ["opinion", "analysis", "commentary", "editorial"]):
        mode = "analysis"
        opinion_probability += 0.30

    if any(term in text_lower for term in ["reportedly", "claims", "unconfirmed", "anonymous sources"]):
        speculation_level += 0.28
        evidence_strength -= 0.12

    if rhetoric_score >= 60:
        opinion_probability += 0.18
        evidence_strength -= 0.08

    if bayes_result.get("risk_score", 0) > 0.6:
        speculation_level += 0.10

    if credibility_score >= 70:
        evidence_strength += 0.18
    elif credibility_score < 40:
        evidence_strength -= 0.16

    opinion_probability = max(0.0, min(opinion_probability, 1.0))
    speculation_level = max(0.0, min(speculation_level, 1.0))
    evidence_strength = max(0.0, min(evidence_strength, 1.0))

    if opinion_probability >= 0.6:
        headline_type = "opinion"
    elif mode == "analysis":
        headline_type = "analysis"
    else:
        headline_type = "reporting"

    return {
        "headlineType": headline_type,
        "mode": mode,
        "speculationLevel": round(speculation_level, 4),
        "evidenceStrength": round(evidence_strength, 4),
        "opinionProbability": round(opinion_probability, 4),
    }


def compute_quick_summary(
    credibility_score: float,
    rhetoric_score: float,
    risk_drivers: Dict,
) -> str:
    if credibility_score >= 75 and risk_drivers["militaryEscalation"] >= 0.55:
        return "Higher-confidence geopolitical headline with meaningful escalation sensitivity."

    if credibility_score >= 60 and risk_drivers["energySupplyDisruption"] >= 0.45:
        return "Moderately credible headline with notable energy and supply-risk implications."

    if rhetoric_score >= 60:
        return "The headline uses strong rhetorical framing and should be interpreted cautiously."

    if credibility_score < 40:
        return "Low-confidence headline with limited verification support and elevated uncertainty."

    return "Mixed-confidence headline with moderate evidence and issue-specific risk signals."


def analyze(
    candidate_text: str,
    history_df: pd.DataFrame,
    newsapi_max_pages: int = 3,
    rss_max_per_feed: int = 30,
    svd_k: int = 64,
    sim_threshold: float = 0.35,
    coord_threshold: float = 0.90,
) -> Dict:
    text = candidate_text.strip()
    if not text:
        raise ValueError("candidate_text cannot be empty")

    lang = detect_language(text)
    categories = guess_categories(text)
    unknown_domains = extract_domains_from_text(text)
    topic = infer_topic(text)
    event_type = infer_event_type(text)
    primary_region = infer_primary_region(text)
    countries = extract_countries(text)
    actors = extract_actors(text)

    local_ai_output = None
    try:
        local_ai_output = get_local_model().infer(text)
    except Exception:
        local_ai_output = None

    bayes_result = classify_text(text)

    matched_articles = []
    max_hist_sim = 0.0

    if history_df is not None and not history_df.empty:
        for _, row in history_df.iterrows():
            hist_text = str(row.get("text", ""))
            similarity = simple_similarity(text, hist_text)

            if similarity > max_hist_sim:
                max_hist_sim = similarity

            if similarity >= sim_threshold:
                matched_articles.append(
                    {
                        "title": hist_text[:120],
                        "domain": "historical-source",
                        "category": "history_match",
                        "similarity": similarity,
                        "source": str(row.get("source", "")),
                        "url": "",
                    }
                )

    coverage_percent = min(len(matched_articles) * 20.0, 100.0)
    coordination_score = round(max_hist_sim * 100, 2)

    risk_penalty = bayes_result.get("risk_score", 0) * 35
    similarity_bonus = min(max_hist_sim * 40, 40)
    coverage_bonus = min(coverage_percent * 0.25, 25)

    credibility_score = 50 + similarity_bonus + coverage_bonus - risk_penalty
    credibility_score = max(0.0, min(round(credibility_score, 2), 100.0))

    rhetoric = compute_rhetoric_features(text)
    credibility_style = compute_credibility_style(
        text=text,
        credibility_score=credibility_score,
        rhetoric_score=rhetoric["score"],
        bayes_result=bayes_result,
    )
    risk_drivers = compute_risk_drivers(
        text=text,
        credibility_score=credibility_score,
        coverage_percent=coverage_percent,
        coordination_score=coordination_score,
        max_hist_sim=max_hist_sim,
        rhetoric_score=rhetoric["score"],
    )

    summary = compute_quick_summary(
        credibility_score=credibility_score,
        rhetoric_score=rhetoric["score"],
        risk_drivers=risk_drivers,
    )

    return {
        "headlineType": local_ai_output["headlineType"] if local_ai_output else credibility_style["headlineType"],
        "topic": local_ai_output["topic"] if local_ai_output else topic,
        "eventType": local_ai_output["eventType"] if local_ai_output else event_type,
        "primaryRegion": primary_region,
        "countries": countries,
        "actors": actors,
        "rhetoric": local_ai_output["rhetoric"] if local_ai_output else {
            "score": rhetoric["score"],
            "label": rhetoric["label"],
            "signals": rhetoric["signals"],
        },
        "credibilityStyle": local_ai_output["credibilityStyle"] if local_ai_output else {
            "mode": credibility_style["mode"],
            "speculationLevel": credibility_style["speculationLevel"],
            "evidenceStrength": credibility_style["evidenceStrength"],
            "opinionProbability": credibility_style["opinionProbability"],
        },
        "riskDrivers": local_ai_output["riskDrivers"] if local_ai_output else {
            "energySupplyDisruption": risk_drivers["energySupplyDisruption"],
            "militaryEscalation": risk_drivers["militaryEscalation"],
            "marketUncertainty": risk_drivers["marketUncertainty"],
            "sanctionsTradeRestrictions": risk_drivers["sanctionsTradeRestrictions"],
            "cyberInfrastructureRisk": risk_drivers["cyberInfrastructureRisk"],
            "electionPoliticalInstability": risk_drivers["electionPoliticalInstability"],
        },
        "marketRelevance": local_ai_output["marketRelevance"] if local_ai_output else risk_drivers["marketRelevance"],
        "oilRelevance": local_ai_output["oilRelevance"] if local_ai_output else risk_drivers["oilRelevance"],
        "volatilityRelevance": local_ai_output["volatilityRelevance"] if local_ai_output else risk_drivers["marketUncertainty"],
        "summary": local_ai_output["summary"] if local_ai_output else summary,
        "debug": {
            "credibilityScore": credibility_score,
            "coveragePercent": round(coverage_percent, 2),
            "coordinationScore": coordination_score,
            "maxHistSim": round(max_hist_sim, 4),
            "matchedCategories": categories,
            "unknownDomains": unknown_domains,
            "matchedArticles": matched_articles,
            "lang": lang,
            "rhetoricAnalysis": rhetoric["analysis"],
            "mcUncertainty": risk_drivers["mcUncertainty"],
            "variance": risk_drivers["variance"],
            "localModelUsed": local_ai_output is not None,
            "fetchMeta": {
                "newsapiEnabled": bool(NEWS_API_KEY),
                "newsapiPages": newsapi_max_pages,
                "rssMaxPerFeed": rss_max_per_feed,
                "svdK": svd_k,
                "simThreshold": sim_threshold,
                "coordThreshold": coord_threshold,
                "userAgent": USER_AGENT,
                "timeout": HTTP_TIMEOUT,
            },
        },
    }
