import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from core.config import settings
from models.schemas import IntrinsicLoadResult, ExtraneousLoadResult, GermaneLoadResult, CLIScore, ProfileAdjustments, SentenceCLIScore, PercentileData
from services.scoring import compute_intrinsic_load, compute_extraneous_load, compute_germane_load

_ml_model_cache = {"model": None, "percentile_data": None, "loaded": False}

def get_ml_model():
    if _ml_model_cache["loaded"]:
        return _ml_model_cache["model"], _ml_model_cache["percentile_data"]
        
    model_path = os.path.join(settings.MODEL_DIR, "best_cli_model.joblib")
    percentile_path = os.path.join(settings.MODEL_DIR, "percentile_data.json")
    
    model = None
    percentile_data = None
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            
    if os.path.exists(percentile_path):
        try:
            with open(percentile_path) as f:
                percentile_data = json.load(f)
        except Exception as e:
            print(f"Failed to load percentile data: {e}")
            
    _ml_model_cache["model"] = model
    _ml_model_cache["percentile_data"] = percentile_data
    _ml_model_cache["loaded"] = True
    
    return model, percentile_data

def _extract_ml_features(intrinsic: IntrinsicLoadResult, extraneous: ExtraneousLoadResult, germane: GermaneLoadResult) -> Dict[str, float]:
    return {
        'intrinsic_avg_zipf': intrinsic.avg_zipf,
        'intrinsic_rare_ratio': intrinsic.rare_ratio,
        'intrinsic_term_ratio': intrinsic.term_ratio,
        'intrinsic_num_terms': intrinsic.num_terms,
        'intrinsic_score': intrinsic.intrinsic_score,
        'extraneous_avg_branching': extraneous.avg_branching,
        'extraneous_avg_sentence_length': extraneous.avg_sentence_length,
        'extraneous_avg_dependency_depth': extraneous.avg_dependency_depth,
        'extraneous_passive_count': extraneous.passive_count,
        'extraneous_nominalization_ratio': extraneous.nominalization_ratio,
        'extraneous_score': extraneous.extraneous_score,
        'germane_example_count': germane.example_count,
        'germane_summary_count': germane.summary_count,
        'germane_question_count': germane.question_count,
        'germane_scaffold_count': germane.scaffold_count,
        'germane_score': germane.germane_score,
    }

def compute_ml_cli(nlp, text: str) -> CLIScore:
    doc = nlp(text)
    intr = compute_intrinsic_load(text, doc)
    extr = compute_extraneous_load(text, doc)
    germ = compute_germane_load(text, doc)
    
    model, percentile_data = get_ml_model()
    
    if model is not None:
        features = _extract_ml_features(intr, extr, germ)
        feature_df = pd.DataFrame([features])
        
        col_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
        if os.path.exists(col_path):
            with open(col_path) as f:
                columns = json.load(f)
            feature_df = feature_df[columns]
            
        raw_prediction = float(model.predict(feature_df)[0])
        
        if percentile_data:
            pred_min = percentile_data["score_min"]
            pred_max = percentile_data["score_max"]
            if pred_max > pred_min:
                cli = round(max(min((raw_prediction - pred_min) / (pred_max - pred_min), 1.0), 0.0), 3)
            else:
                cli = 0.5
        else:
            cli = round(max(min((raw_prediction + 2.0) / 5.0, 1.0), 0.0), 3)
            
        scoring_method = "ml"
    else:
        raw_prediction = settings.W_INTR * intr.intrinsic_score + settings.W_EXTR * extr.extraneous_score - settings.W_GERM * germ.germane_score
        clipped = max(min(raw_prediction, 1.0), -1.0)
        cli = round((clipped + 1.0) / 2.0, 3)
        scoring_method = "rule_based"
        
    return CLIScore(
        intrinsic=intr,
        extraneous=extr,
        germane=germ,
        raw_score=round(raw_prediction, 3),
        cli=cli,
        label="Low" if cli < 0.33 else ("Medium" if cli < 0.66 else "High"),
        scoring_method=scoring_method
    )

def compute_ml_cli_with_profile(nlp, text: str, reading_level: str = "Intermediate", domain_familiarity: float = 0.5) -> CLIScore:
    base_score = compute_ml_cli(nlp, text)
    
    adjusted_intr_score = base_score.intrinsic.intrinsic_score * (1.0 - (domain_familiarity * 0.5))
    extr_mult = 1.2 if reading_level == "Beginner" else 0.8 if reading_level == "Advanced" else 1.0
    adjusted_extr_score = base_score.extraneous.extraneous_score * extr_mult
    
    intr_delta = (adjusted_intr_score - base_score.intrinsic.intrinsic_score) * 0.3
    extr_delta = (adjusted_extr_score - base_score.extraneous.extraneous_score) * 0.3
    
    profile_adjusted_cli = round(max(min(base_score.cli + intr_delta + extr_delta, 1.0), 0.0), 3)
    
    base_score.cli = profile_adjusted_cli
    base_score.label = "Low" if profile_adjusted_cli < 0.33 else ("Medium" if profile_adjusted_cli < 0.66 else "High")
    base_score.profile_adjustments = ProfileAdjustments(
        adjusted_intrinsic_score=round(adjusted_intr_score, 3),
        adjusted_extraneous_score=round(adjusted_extr_score, 3),
        base_cli_before_profile=base_score.cli
    )
    
    return base_score

def compute_sentence_level_ml_cli(nlp, text: str) -> List[SentenceCLIScore]:
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text: continue
        
        cli_result = compute_ml_cli(nlp, sentence_text)
        results.append(SentenceCLIScore(
            sentence=sentence_text,
            cli=cli_result.cli,
            label=cli_result.label,
            intrinsic=cli_result.intrinsic.intrinsic_score,
            extraneous=cli_result.extraneous.extraneous_score,
            germane=cli_result.germane.germane_score
        ))
    return results

def compute_percentile_rank(cli_score: float, raw_ml_score: Optional[float] = None) -> PercentileData:
    _, percentile_data = get_ml_model()
    
    if percentile_data is None:
        return PercentileData(available=False, message="Percentile data not available. Run train_ml.py first.")
        
    sorted_scores = percentile_data["sorted_scores"]
    dataset_size = percentile_data["dataset_size"]
    
    if raw_ml_score is not None:
        comparison_score = raw_ml_score
    else:
        pred_min = percentile_data["score_min"]
        pred_max = percentile_data["score_max"]
        comparison_score = cli_score * (pred_max - pred_min) + pred_min
        
    import bisect
    position = bisect.bisect_left(sorted_scores, comparison_score)
    percentile = round((position / dataset_size) * 100, 1)
    
    if percentile < 25:
        interp, tier = "Easier than most educational texts", "Easy"
    elif percentile < 50:
        interp, tier = "Easier than average educational text", "Below Average"
    elif percentile < 75:
        interp, tier = "Harder than average educational text", "Above Average"
    else:
        interp, tier = "Among the most difficult educational texts", "Difficult"
        
    return PercentileData(
        available=True,
        percentile=percentile,
        interpretation=interp,
        difficulty_tier=tier,
        corpus_name=percentile_data.get("dataset_name", "CLEAR Corpus"),
        corpus_size=dataset_size,
        corpus_mean_score=percentile_data.get("score_mean"),
        badge=f"Harder than {percentile}% of texts in the CLEAR Corpus ({dataset_size} samples)",
        message=""
    )
