"""
Bank Document Classifier — FastAPI Service
──────────────────────────────────────────
Endpoints:
  POST /classify        — classify a document (text or file)
  POST /train           — trigger retraining with new labelled samples
  POST /feedback        — submit a correction (feeds into next retrain)
  GET  /health          — liveness check
  GET  /classes         — list supported document classes
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Classifier import (adjust if running from repo root)
from classifier.model import BankDocumentClassifier, DOCUMENT_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Feedback store (in-memory; swap for DB in production) ────────────────────
FEEDBACK_STORE: List[dict] = []
FEEDBACK_PATH = Path(__file__).parent / "feedback_buffer.json"

def _load_feedback():
    if FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH) as f:
            return json.load(f)
    return []

def _save_feedback(store):
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(store, f, indent=2)


# ─── App lifecycle ────────────────────────────────────────────────────────────

classifier: Optional[BankDocumentClassifier] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, FEEDBACK_STORE
    logger.info("Loading classifier model...")
    classifier = BankDocumentClassifier()
    FEEDBACK_STORE = _load_feedback()
    logger.info(f"Classifier ready. {len(FEEDBACK_STORE)} feedback records loaded.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Bank Document Classifier API",
    version="1.0.0",
    description="ML-first document classification with silent LLM fallback.",
    lifespan=lifespan,
)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class ClassifyTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw document text to classify")

class ClassifyResponse(BaseModel):
    classification: str
    confidence: float
    source: str                    # "local_model" or "llm_fallback"
    extracted_fields: dict
    reasoning: str
    all_scores: Optional[dict] = None
    local_model_best_guess: Optional[str] = None
    local_model_confidence: Optional[float] = None

class TrainSample(BaseModel):
    text: str
    label: str = Field(..., description=f"One of: {DOCUMENT_CLASSES}")

class TrainRequest(BaseModel):
    samples: List[TrainSample]

class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    correct_label: str
    source: str = "user_correction"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/classes")
def list_classes():
    return {"classes": DOCUMENT_CLASSES, "confidence_threshold": classifier.threshold}


@app.post("/classify", response_model=ClassifyResponse)
def classify_text(req: ClassifyTextRequest):
    """Classify document from raw text."""
    try:
        result = classifier.classify(req.text)
        return ClassifyResponse(**result)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/file", response_model=ClassifyResponse)
async def classify_file(file: UploadFile = File(...)):
    """
    Classify document from uploaded file.
    Supports: .txt, .json (with 'text' key).
    For PDF/image, pre-extract text client-side or use /classify with text.
    """
    try:
        raw = await file.read()

        if file.filename.endswith(".json"):
            payload = json.loads(raw)
            text = payload.get("text", "")
        else:
            text = raw.decode("utf-8", errors="ignore")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file.")

        result = classifier.classify(text)
        return ClassifyResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def retrain(req: TrainRequest):
    """
    Trigger retraining with new labelled samples.
    New samples are merged with the base synthetic dataset.
    """
    for s in req.samples:
        if s.label not in DOCUMENT_CLASSES:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown label '{s.label}'. Allowed: {DOCUMENT_CLASSES}"
            )

    samples = [(s.text, s.label) for s in req.samples]
    try:
        classifier.retrain(samples)
        return {"status": "retrained", "new_samples": len(samples)}
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Submit a label correction. Corrections are buffered and used on
    the next /train call or scheduled nightly retraining.
    """
    if req.correct_label not in DOCUMENT_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown label '{req.correct_label}'. Allowed: {DOCUMENT_CLASSES}"
        )

    entry = req.model_dump()
    FEEDBACK_STORE.append(entry)
    _save_feedback(FEEDBACK_STORE)

    # Auto-retrain when buffer exceeds 20 corrections
    if len(FEEDBACK_STORE) >= 20:
        samples = [(e["text"], e["correct_label"]) for e in FEEDBACK_STORE]
        classifier.retrain(samples)
        FEEDBACK_STORE.clear()
        _save_feedback(FEEDBACK_STORE)
        return {"status": "feedback_saved_and_retrained", "buffer_size": 0}

    return {"status": "feedback_saved", "buffer_size": len(FEEDBACK_STORE)}
