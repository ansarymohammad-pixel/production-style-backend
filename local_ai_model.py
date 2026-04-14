import os
import joblib

MODELS_DIR = "models"


def _load(name: str):
    path = os.path.join(MODELS_DIR, name)
    return joblib.load(path)


class LocalAIModel:
    def __init__(self):
        self.rhetoric_label = _load("rhetoric_label.pkl")
        self.headline_type = _load("headline_type.pkl")
        self.event_type = _load("event_type.pkl")
        self.topic = _load("topic.pkl")

        self.rhetoric_score = _load("rhetoric_score.pkl")
        self.energy_supply_disruption = _load("energy_supply_disruption.pkl")
        self.military_escalation = _load("military_escalation.pkl")
        self.market_uncertainty = _load("market_uncertainty.pkl")
        self.sanctions_trade_restrictions = _load("sanctions_trade_restrictions.pkl")
        self.cyber_infrastructure_risk = _load("cyber_infrastructure_risk.pkl")
        self.election_political_instability = _load("election_political_instability.pkl")

    def infer(self, headline: str) -> dict:
        rhetoric_score = int(round(self._clip(self.rhetoric_score.predict([headline])[0], 0, 100)))
        rhetoric_label = self.rhetoric_label.predict([headline])[0]
        headline_type = self.headline_type.predict([headline])[0]
        event_type = self.event_type.predict([headline])[0]
        topic = self.topic.predict([headline])[0]

        energy_supply_disruption = self._clip(self.energy_supply_disruption.predict([headline])[0], 0.0, 1.0)
        military_escalation = self._clip(self.military_escalation.predict([headline])[0], 0.0, 1.0)
        market_uncertainty = self._clip(self.market_uncertainty.predict([headline])[0], 0.0, 1.0)
        sanctions_trade_restrictions = self._clip(self.sanctions_trade_restrictions.predict([headline])[0], 0.0, 1.0)
        cyber_infrastructure_risk = self._clip(self.cyber_infrastructure_risk.predict([headline])[0], 0.0, 1.0)
        election_political_instability = self._clip(self.election_political_instability.predict([headline])[0], 0.0, 1.0)

        return {
            "headlineType": headline_type,
            "topic": topic,
            "eventType": event_type,
            "rhetoric": {
                "score": rhetoric_score,
                "label": rhetoric_label,
                "signals": []
            },
            "credibilityStyle": {
                "mode": "analysis" if headline_type == "analysis" else headline_type,
                "speculationLevel": 0.25 if headline_type != "reporting" else 0.10,
                "evidenceStrength": 0.50,
                "opinionProbability": 0.70 if headline_type == "opinion" else 0.20
            },
            "riskDrivers": {
                "energySupplyDisruption": energy_supply_disruption,
                "militaryEscalation": military_escalation,
                "marketUncertainty": market_uncertainty,
                "sanctionsTradeRestrictions": sanctions_trade_restrictions,
                "cyberInfrastructureRisk": cyber_infrastructure_risk,
                "electionPoliticalInstability": election_political_instability
            },
            "marketRelevance": market_uncertainty,
            "oilRelevance": energy_supply_disruption,
            "volatilityRelevance": market_uncertainty,
            "summary": f"Local model classified this as {event_type} in topic {topic}."
        }

    @staticmethod
    def _clip(value, low, high):
        return max(low, min(value, high))
