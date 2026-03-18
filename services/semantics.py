import numpy as np
from core.config import settings
from models.schemas import SemanticDriftResult

_sbert_model_cache = {"model": None}

def get_sbert_model():
    if _sbert_model_cache["model"] is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model_cache["model"] = SentenceTransformer(settings.SBERT_MODEL)
    return _sbert_model_cache["model"]

def compute_semantic_drift(original_text: str, rewritten_text: str) -> SemanticDriftResult:
    model = get_sbert_model()
    embeddings = model.encode([original_text, rewritten_text])
    
    cos_sim = float(np.dot(embeddings[0], embeddings[1]) / 
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    
    similarity = round(cos_sim, 3)
    drift = round(1.0 - similarity, 3)
    drift_pct = round(drift * 100, 1)
    
    if similarity >= 0.9:
        verdict = "Excellent"
        detail = "Meaning fully preserved. Safe to use."
    elif similarity >= 0.75:
        verdict = "Good"
        detail = "Minor meaning shifts. Review recommended."
    elif similarity >= 0.6:
        verdict = "Caution"
        detail = "Significant meaning drift detected. The rewrite may have altered key concepts."
    else:
        verdict = "Danger"
        detail = "Major meaning loss. The rewrite has substantially changed the original content."
        
    return SemanticDriftResult(
        similarity=similarity,
        drift_pct=drift_pct,
        verdict=verdict,
        detail=detail,
        flags=["drift"] if similarity < 0.75 else []
    )
