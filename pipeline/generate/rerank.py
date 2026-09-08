"""Serving-side reranker.

Second stage of a retrieval cascade: BM25 + dense fusion narrows a paper's
chunks cheaply, then this reorders the survivors using a model trained on the
1,935-query benchmark. Held out on 34 unseen papers it is +0.160 nDCG@10 over
BM25 alone, CI [+0.134, +0.185], p<0.001.

Off by default (`NEUROPOD_RERANKER=on`). Two reasons to gate it rather than
always run it: the model file may be absent on a fresh clone, and a serving path
that silently changes behaviour depending on which files happen to exist is
worse than one that requires an explicit switch.

Feature extraction is imported from `pipeline.generate.features` — the same code
that produced the training matrix. Recomputing features a second way offline and
online is the classic route to a model that scores well in evaluation and
underperforms in production.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .features import build_features

logger = logging.getLogger("neuropod.rerank")

ROOT = Path(__file__).resolve().parents[2]
LGBM_PATH = ROOT / "eval" / "reranker_lgbm.txt"
LINEAR_PATH = ROOT / "eval" / "reranker.json"


class LinearReranker:
    """Logistic-regression scorer. Pure Python — no ML dependency at serve time.

    Kept as the fallback because it needs nothing installed, but note it is NOT
    an improvement on its own: on held-out papers it scored 0.320 against BM25's
    0.324 (p=0.598). It exists so the serving path is exercisable anywhere, not
    because it earns its place on quality.
    """

    name = "linear"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.coef = payload["coef"]
        self.intercept = payload["intercept"]
        self.mean = payload["scaler_mean"]
        self.scale = payload["scaler_scale"]

    def score(self, features: list[float]) -> float:
        total = self.intercept
        for value, mu, sd, w in zip(features, self.mean, self.scale, self.coef):
            total += ((value - mu) / (sd or 1.0)) * w
        return total

    def rerank(self, scored: list[dict], query: str) -> list[dict]:
        return _apply(self, scored, query)


class LGBMReranker:
    """LambdaMART. Optimizes NDCG over query groups rather than scoring
    candidates independently — worth +0.061 over the same-family pointwise
    GBDT on held-out papers."""

    name = "lambdamart"

    def __init__(self, booster) -> None:
        self.booster = booster

    def rerank(self, scored: list[dict], query: str) -> list[dict]:
        return _apply(self, scored, query, batch=True)

    def score_batch(self, matrix: list[list[float]]) -> list[float]:
        return list(self.booster.predict(matrix))


def _apply(model, scored: list[dict], query: str, *, batch: bool = False) -> list[dict]:
    chunks = [row["chunk"] for row in scored]
    dense = {
        row["chunk"]["id"]: row.get("dense_score") or 0.0 for row in scored
    }
    candidates = build_features(chunks, query, dense_scores=dense)
    by_id = {c.chunk_id: c for c in candidates}

    if batch:
        matrix = [by_id[row["chunk"]["id"]].features for row in scored]
        scores = model.score_batch(matrix)
    else:
        scores = [model.score(by_id[row["chunk"]["id"]].features) for row in scored]

    for row, s in zip(scored, scores):
        row["rerank_score"] = float(s)
    # Fused score kept as a stable, deterministic tie-break.
    return sorted(scored, key=lambda r: (-r["rerank_score"], -r["final_score"]))


def load_reranker():
    """Best available model, or None. Never raises into the request path."""
    if LGBM_PATH.exists():
        try:
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(LGBM_PATH))
            logger.info("reranker: lambdamart (%s)", LGBM_PATH.name)
            return LGBMReranker(booster)
        except ImportError:
            logger.warning("lightgbm not installed; falling back to the linear reranker")
        except OSError as exc:
            logger.warning("lightgbm present but its OpenMP runtime is missing (%s); "
                           "macOS: brew install libomp", exc.__class__.__name__)
        except Exception as exc:
            logger.warning("could not load %s: %s", LGBM_PATH.name, exc)

    if LINEAR_PATH.exists():
        try:
            logger.info("reranker: linear (%s)", LINEAR_PATH.name)
            return LinearReranker(json.loads(LINEAR_PATH.read_text()))
        except Exception as exc:
            logger.warning("could not load %s: %s", LINEAR_PATH.name, exc)

    logger.warning("NEUROPOD_RERANKER is on but no model file was found")
    return None
