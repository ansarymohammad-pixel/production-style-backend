import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn

from engine import analyze

app = FastAPI(title="Narrative Risk Radar API")


@app.get("/health")
def health():
    return {"status": "ok"}


def get_demo_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "2026-02-02 09:00:00",
                "source_A",
                "Energy policy discussion and regional oil market reaction.",
            ),
            (
                "2026-02-02 10:00:00",
                "source_B",
                "Supreme Court hears arguments about tariff authority and trade powers.",
            ),
            (
                "2026-02-02 11:00:00",
                "source_C",
                "Iran warns against military escalation as diplomacy continues.",
            ),
            (
                "2026-02-02 12:00:00",
                "source_D",
                "Taiwan Strait tensions raise shipping and semiconductor concerns.",
            ),
            (
                "2026-02-02 13:00:00",
                "source_E",
                "Central bank signals new rate path as inflation outlook shifts.",
            ),
        ],
        columns=["time", "source", "text"],
    )


class AnalyzeHeadlineRequest(BaseModel):
    headline: str


class AnalyzeTextRequest(BaseModel):
    text: str
    newsapi_pages: int = 3
    rss_max: int = 30
    svd_k: int = 64
    sim_threshold: float = 0.35
    coord_threshold: float = 0.90


@app.post("/analyze-headline")
def analyze_headline(payload: AnalyzeHeadlineRequest):
    headline = payload.headline.strip()
    if not headline:
        raise HTTPException(status_code=400, detail="Headline is required")

    history_df = get_demo_history()

    try:
        out = analyze(
            candidate_text=headline,
            history_df=history_df,
            newsapi_max_pages=3,
            rss_max_per_feed=30,
            svd_k=64,
            sim_threshold=0.35,
            coord_threshold=0.90,
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-text")
def analyze_text(payload: AnalyzeTextRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    history_df = get_demo_history()

    try:
        out = analyze(
            candidate_text=text,
            history_df=history_df,
            newsapi_max_pages=payload.newsapi_pages,
            rss_max_per_feed=payload.rss_max,
            svd_k=payload.svd_k,
            sim_threshold=payload.sim_threshold,
            coord_threshold=payload.coord_threshold,
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
