import statistics
import textstat
from typing import List, Dict, Any

from models.schemas import DifficultyCliff, DifficultyRamp, SentenceCLIScore
from services.scoring import estimate_working_memory_slots
from services.ml import compute_ml_cli_with_profile, compute_sentence_level_ml_cli, compute_percentile_rank

def detect_difficulty_patterns(sentence_results: List[SentenceCLIScore]) -> DifficultyRamp:
    sentence_clis = [r.cli for r in sentence_results]
    n = len(sentence_clis)
    
    if n < 2:
        return DifficultyRamp(
            pattern="insufficient_data",
            overall_trend="flat",
            cliffs=[],
            cliff_count=0,
            recommendations=["Need at least 2 sentences for progression analysis."]
        )
        
    cliffs = []
    plateaus = []
    recommendations = []
    
    for i in range(1, n):
        jump = sentence_clis[i] - sentence_clis[i - 1]
        if jump > 0.25:
            cliffs.append(DifficultyCliff(
                jump=round(jump, 3),
                from_sentence=i,
                to_sentence=i+1,
                from_cli=round(sentence_clis[i-1], 3),
                to_cli=round(sentence_clis[i], 3)
            ))
            
    window_size = 3
    for i in range(n - window_size + 1):
        window = sentence_clis[i:i + window_size]
        if len(set(window)) > 1:
            window_std = statistics.stdev(window)
            if window_std < 0.03:
                plateaus.append(i)
                
    x_mean = (n - 1) / 2.0
    y_mean = sum(sentence_clis) / n
    numerator = sum((i - x_mean) * (sentence_clis[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator > 0 else 0.0
    
    if slope > 0.03: overall_trend = "increasing"
    elif slope < -0.03: overall_trend = "decreasing" 
    else: overall_trend = "flat"
    
    if len(cliffs) >= 2:
        pattern = "cliff_heavy"
        recommendations.append(f"Found {len(cliffs)} difficulty cliffs. Break complex sentences into progressive build-up.")
    elif len(plateaus) >= 2:
        pattern = "plateau_heavy"
        recommendations.append(f"Found {len(plateaus)} plateaus. Vary sentence complexity to maintain engagement.")
    elif len(cliffs) == 0 and overall_trend == "increasing":
        pattern = "ideal_ramp"
        recommendations.append("Good progressive difficulty increase. This follows learning science best practices.")
    elif overall_trend == "decreasing":
        pattern = "reverse_ramp"
        recommendations.append("Difficulty decreases over the passage. Consider starting simpler and building up.")
    else:
        pattern = "mixed"
        recommendations.append("Mixed difficulty pattern. Some variation is acceptable.")
        
    if cliffs:
        worst_cliff = max(cliffs, key=lambda c: c.jump)
        recommendations.append(
            f"Worst cliff: sentence {worst_cliff.from_sentence} → {worst_cliff.to_sentence} "
            f"(CLI jumps {worst_cliff.from_cli} → {worst_cliff.to_cli}). Add a bridging sentence."
        )
        
    return DifficultyRamp(
        pattern=pattern,
        overall_trend=overall_trend,
        cliffs=cliffs,
        cliff_count=len(cliffs),
        recommendations=recommendations
    )

def generate_verification_report(nlp, original_text: str, rewritten_text: str,
                                  target_cli: float = 0.4,
                                  reading_level: str = "Intermediate",
                                  domain_familiarity: float = 0.5) -> Dict[str, Any]:
    
    orig_data = compute_ml_cli_with_profile(nlp, original_text, reading_level, domain_familiarity)
    new_data = compute_ml_cli_with_profile(nlp, rewritten_text, reading_level, domain_familiarity)
    
    orig_cli, new_cli = orig_data.cli, new_data.cli
    intr_delta = round(orig_data.intrinsic.intrinsic_score - new_data.intrinsic.intrinsic_score, 3)
    extr_delta = round(orig_data.extraneous.extraneous_score - new_data.extraneous.extraneous_score, 3)
    germ_delta = round(new_data.germane.germane_score - orig_data.germane.germane_score, 3)
    
    cli_reduction = round(orig_cli - new_cli, 3)
    pct_reduction = round((cli_reduction / orig_cli) * 100, 1) if orig_cli > 0 else 0
    
    orig_wm = estimate_working_memory_slots(nlp, original_text)
    new_wm = estimate_working_memory_slots(nlp, rewritten_text)
    
    orig_fk = textstat.flesch_kincaid_grade(original_text)
    new_fk = textstat.flesch_kincaid_grade(rewritten_text)
    
    orig_percentile = compute_percentile_rank(orig_cli, orig_data.raw_score)
    new_percentile = compute_percentile_rank(new_cli, new_data.raw_score)
    
    orig_sents = compute_sentence_level_ml_cli(nlp, original_text)
    new_sents = compute_sentence_level_ml_cli(nlp, rewritten_text)
    
    target_reached = new_cli <= target_cli
    
    if target_reached and cli_reduction > 0:
        certification, cert_message = "CERTIFIED", f"✅ CLI Certified: Optimized for {reading_level} Learners (CLI {new_cli} ≤ {target_cli})"
    elif cli_reduction > 0:
        certification, cert_message = "IMPROVED", f"⚠️ Improved but target not reached (CLI {new_cli} > {target_cli}). Consider another iteration."
    else:
        certification, cert_message = "FAILED", f"❌ Rewrite did not reduce cognitive load (CLI {orig_cli} → {new_cli})"
        
    metrics_checked = [
        "ML-Calibrated CLI", "Intrinsic Load", "Extraneous Load", "Germane Load",
        "Flesch-Kincaid Grade", "Working Memory Slots", "Percentile Rank",
        "Sentence-Level Scores", "Vocabulary Difficulty", "Syntactic Complexity"
    ]
    
    return {
        "certification": certification,
        "cert_message": cert_message,
        "target_cli": target_cli,
        "target_reached": target_reached,
        "metrics_checked_count": len(metrics_checked),
        "metrics_list": metrics_checked,
        "scoring_method": new_data.scoring_method,
        "cli_comparison": {
            "original": orig_cli, "rewritten": new_cli,
            "reduction": cli_reduction, "reduction_pct": pct_reduction,
        },
        "component_deltas": {
            "intrinsic_reduction": intr_delta,
            "extraneous_reduction": extr_delta,
            "germane_improvement": germ_delta,
        },
        "flesch_kincaid": {
            "original_grade": round(orig_fk, 1),
            "rewritten_grade": round(new_fk, 1),
            "grade_reduction": round(orig_fk - new_fk, 1),
        },
        "working_memory": {
            "original_slots": orig_wm.slot_count,
            "rewritten_slots": new_wm.slot_count,
            "slots_freed": orig_wm.slot_count - new_wm.slot_count,
        },
        "percentile": {"original": orig_percentile.model_dump(), "rewritten": new_percentile.model_dump()},
        "radar_data": {
            "original": {"intrinsic": orig_data.intrinsic.intrinsic_score, "extraneous": orig_data.extraneous.extraneous_score, "germane": orig_data.germane.germane_score},
            "rewritten": {"intrinsic": new_data.intrinsic.intrinsic_score, "extraneous": new_data.extraneous.extraneous_score, "germane": new_data.germane.germane_score},
        },
        "sentence_tracking": {"original": [s.model_dump() for s in orig_sents], "rewritten": [s.model_dump() for s in new_sents]},
        "original_data": orig_data.model_dump(),
        "rewritten_data": new_data.model_dump(),
    }
