from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Concept(BaseModel):
    term: str
    zipf_frequency: float

class IntrinsicLoadResult(BaseModel):
    term_ratio: float
    num_terms: int
    avg_zipf: float
    rare_ratio: float
    intrinsic_score: float
    terms_sample: List[str] = []

class ExtraneousLoadResult(BaseModel):
    avg_dependency_depth: float
    avg_branching: float
    passive_count: int
    avg_sentence_length: float
    nominalization_ratio: float
    extraneous_score: float

class GermaneLoadResult(BaseModel):
    example_count: int
    summary_count: int
    question_count: int
    scaffold_count: int
    germane_score: float

class ProfileAdjustments(BaseModel):
    adjusted_intrinsic_score: float
    adjusted_extraneous_score: float
    base_cli_before_profile: float = 0.0

class CLIScore(BaseModel):
    intrinsic: IntrinsicLoadResult
    extraneous: ExtraneousLoadResult
    germane: GermaneLoadResult
    raw_score: float
    cli: float
    label: str
    scoring_method: str = "rule_based"
    profile_adjustments: Optional[ProfileAdjustments] = None

class SentenceCLIScore(BaseModel):
    sentence: str
    cli: float
    label: str
    intrinsic: float
    extraneous: float
    germane: float

class FeatureContribution(BaseModel):
    component: str
    reason: str
    severity: str

class ExplanationResult(BaseModel):
    human_readable: List[str]
    features: List[FeatureContribution]

class WorkingMemoryData(BaseModel):
    slot_count: int
    miller_capacity: int
    exceeds_capacity: bool
    severity: str
    color: str
    novel_concepts: List[Concept]
    utilization_pct: float
    recommendation: str

class PercentileData(BaseModel):
    available: bool
    percentile: float = 0.0
    interpretation: str = ""
    difficulty_tier: str = ""
    corpus_name: str = "CLEAR Corpus"
    corpus_size: int = 1000
    corpus_mean_score: Optional[float] = None
    badge: str = ""
    message: str = ""

class DifficultyCliff(BaseModel):
    jump: float
    from_sentence: int
    to_sentence: int
    from_cli: float
    to_cli: float

class DifficultyRamp(BaseModel):
    pattern: str
    overall_trend: str
    cliffs: List[DifficultyCliff]
    cliff_count: int
    recommendations: List[str]

class OptimizationHistory(BaseModel):
    iteration: int
    text: str
    cli: float
    issue_addressed: Optional[str] = None
    explanation: Optional[str] = None
    scores: Optional[Dict[str, float]] = None

class OptimizationResult(BaseModel):
    final_text: str
    target_reached: bool
    iterations_used: int
    history: List[OptimizationHistory]

class SemanticDriftResult(BaseModel):
    similarity: float
    drift_pct: float
    verdict: str
    detail: str
    flags: List[str] = []

class AnalyzeRequest(BaseModel):
    text: str
    target_level: str = "Intermediate"
    domain_familiarity: float = 0.5

class RewriteRequest(BaseModel):
    text: str
    target_level: str = "Beginner"
    mode: str = "basic"  # 'basic', 'smart', or 'pedagogical'
    model_name: Optional[str] = None

class TutorRequest(BaseModel):
    text: str
    model_name: Optional[str] = None
