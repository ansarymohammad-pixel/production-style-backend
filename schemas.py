from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AnalyzeHeadlineRequest(BaseModel):
    headline: str


class AnalyzeTextRequest(BaseModel):
    text: str
    sim_threshold: float = 0.35
    svd_k: int = 64
    coord_threshold: float = 0.90
    newsapi_pages: int = 3
    rss_max: int = 30


class MatchedArticle(BaseModel):
    title: str = ""
    domain: str = ""
    category: str = ""
    similarity: float = 0.0
    source: str = ""
    url: str = ""


class RhetoricSchema(BaseModel):
    score: int = 0
    label: str = "low"
    signals: List[str] = Field(default_factory=list)


class CredibilityStyleSchema(BaseModel):
    mode: str = "reporting"
    speculationLevel: float = 0.0
    evidenceStrength: float = 0.5
    opinionProbability: float = 0.0


class RiskDriversSchema(BaseModel):
    energySupplyDisruption: float = 0.0
    militaryEscalation: float = 0.0
    marketUncertainty: float = 0.0
    sanctionsTradeRestrictions: float = 0.0
    cyberInfrastructureRisk: float = 0.0
    electionPoliticalInstability: float = 0.0


class DebugSchema(BaseModel):
    credibilityScore: float = 0.0
    coveragePercent: float = 0.0
    coordinationScore: float = 0.0
    maxHistSim: float = 0.0
    matchedCategories: List[str] = Field(default_factory=list)
    unknownDomains: List[str] = Field(default_factory=list)
    matchedArticles: List[MatchedArticle] = Field(default_factory=list)
    lang: str = "unknown"
    rhetoricAnalysis: str = ""
    mcUncertainty: float = 0.0
    variance: float = 0.0
    fetchMeta: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeHeadlineResponse(BaseModel):
    headlineType: str = "mixed"
    topic: str = "other"
    eventType: str = "general_reporting"
    primaryRegion: str = ""
    countries: List[str] = Field(default_factory=list)
    actors: List[str] = Field(default_factory=list)
    rhetoric: RhetoricSchema = Field(default_factory=RhetoricSchema)
    credibilityStyle: CredibilityStyleSchema = Field(default_factory=CredibilityStyleSchema)
    riskDrivers: RiskDriversSchema = Field(default_factory=RiskDriversSchema)
    marketRelevance: float = 0.0
    oilRelevance: float = 0.0
    volatilityRelevance: float = 0.0
    summary: str = ""
    debug: DebugSchema = Field(default_factory=DebugSchema)
