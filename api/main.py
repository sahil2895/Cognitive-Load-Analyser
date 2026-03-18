from fastapi import FastAPI, HTTPException, Depends
from models.schemas import AnalyzeRequest, RewriteRequest, TutorRequest
from services.nlp import get_spacy
from services.ml import compute_ml_cli, compute_sentence_level_ml_cli
from services.rewriter import rewrite_basic_simplify, rewrite_difficult_sentences, rewrite_pedagogical
from services.ai_tutor import generate_tutor_feedback

app = FastAPI(
    title="Cognitive Load Analyzer API",
    description="API for analyzing cognitive load, rewriting text, and generating AI tutor suggestions.",
    version="1.0.0"
)

# Dependency to ensure NLP is loaded per request safely
def require_nlp():
    try:
        return get_spacy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load NLP model: {e}")

@app.get("/")
def read_root():
    return {"message": "Cognitive Load Analyzer API is running. Check /docs for endpoints."}

@app.post("/analyze")
def analyze_text(req: AnalyzeRequest, nlp=Depends(require_nlp)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")
    
    cli_data = compute_ml_cli(nlp, req.text)
    sentence_results = compute_sentence_level_ml_cli(nlp, req.text)
    
    return {
        "cli_data": cli_data,
        "sentence_results": sentence_results
    }

@app.post("/rewrite")
def rewrite_text(req: RewriteRequest, nlp=Depends(require_nlp)):
    if req.mode == "basic":
        return {"rewritten_text": rewrite_basic_simplify(req.text, req.target_level, req.model_name)}
    elif req.mode == "pedagogical":
        return {"rewritten_text": rewrite_pedagogical(req.text, req.target_level, req.model_name)}
    elif req.mode == "smart":
        sentence_results = compute_sentence_level_ml_cli(nlp, req.text)
        return {"rewritten_text": rewrite_difficult_sentences(nlp, req.text, sentence_results, req.target_level, req.model_name)}
    else:
        raise HTTPException(status_code=400, detail="Unknown mode. Use basic, smart, or pedagogical.")

@app.post("/tutor")
def tutor_tips(req: TutorRequest, nlp=Depends(require_nlp)):
    cli_data = compute_ml_cli(nlp, req.text)
    sentence_results = compute_sentence_level_ml_cli(nlp, req.text)
    tips = generate_tutor_feedback(cli_data, sentence_results, req.model_name)
    
    return {"tutor_suggestions": tips}
