import statistics
from typing import List

from core.config import settings
from models.schemas import (
    IntrinsicLoadResult, ExtraneousLoadResult, GermaneLoadResult,
    CLIScore, SentenceCLIScore, WorkingMemoryData, Concept,
    ExplanationResult, FeatureContribution
)
from services.nlp import (
    tokenize_words, average_word_zipf, rare_word_ratio, extract_candidate_terms,
    sentence_dependency_depths, average_branching_factor, count_passive_voice,
    textstat_avg_sentence_length_safe, presence_of_examples, presence_of_summaries,
    count_questions
)
from wordfreq import zipf_frequency
import math

def compute_intrinsic_load(text: str, doc) -> IntrinsicLoadResult:
    words = tokenize_words(text)
    total_words = max(1, len(words))
    terms = extract_candidate_terms(doc)
    term_ratio = len([t for t in terms if len(t.split()) >= 1]) / total_words
    avg_zipf = average_word_zipf(text)
    rare_ratio = rare_word_ratio(text)
    
    term_norm = min(term_ratio * 5.0, 1.0)
    zipf_norm = 1.0 - (min(max((avg_zipf - 1.0) / 6.0, 0.0), 1.0))
    intrinsic_score = round(0.5 * term_norm + 0.3 * rare_ratio + 0.2 * zipf_norm, 3)
    
    return IntrinsicLoadResult(
        term_ratio=round(term_ratio, 4),
        num_terms=len(terms),
        avg_zipf=round(avg_zipf, 3),
        rare_ratio=round(rare_ratio, 3),
        intrinsic_score=intrinsic_score,
        terms_sample=terms[:8]
    )


def compute_extraneous_load(text: str, doc) -> ExtraneousLoadResult:
    depths = sentence_dependency_depths(doc)
    avg_depth = sum(depths) / len(depths) if depths else 0.0
    branch = average_branching_factor(doc)
    passive = count_passive_voice(doc)
    avg_sent_len = textstat_avg_sentence_length_safe(text)
    
    words = tokenize_words(text)
    import re
    nominals = sum(1 for w in words if re.search(r"(tion|ment|ness|ity|ization)$", w.lower()))
    nom_ratio = nominals / max(1, len(words))
    
    depth_norm = min(avg_depth / 10.0, 1.0)
    branch_norm = min(branch / 3.0, 1.0)
    passive_norm = min(passive / max(1, len(list(doc.sents))), 1.0)
    sentlen_norm = min(avg_sent_len / 30.0, 1.0)
    nom_norm = min(nom_ratio * 10.0, 1.0)
    
    extraneous_score = round(0.3 * depth_norm + 0.25 * branch_norm + 0.15 * sentlen_norm + 0.15 * nom_norm + 0.15 * passive_norm, 3)
    
    return ExtraneousLoadResult(
        avg_dependency_depth=round(avg_depth, 3),
        avg_branching=round(branch, 3),
        passive_count=passive,
        avg_sentence_length=round(avg_sent_len, 3),
        nominalization_ratio=round(nom_ratio, 4),
        extraneous_score=extraneous_score
    )


def compute_germane_load(text: str, doc) -> GermaneLoadResult:
    ex_count = presence_of_examples(text)
    sum_count = presence_of_summaries(text)
    q_count = count_questions(text)
    import re
    scaffold_cues = len(re.findall(r"\b(first|second|third|then|next|step)\b", text.lower()))
    
    ex_norm = min(ex_count / 3.0, 1.0)
    sum_norm = min(sum_count / 2.0, 1.0)
    q_norm = min(q_count / 3.0, 1.0)
    scaffold_norm = min(scaffold_cues / 6.0, 1.0)
    
    germane_score = round(0.4 * ex_norm + 0.25 * sum_norm + 0.2 * q_norm + 0.15 * scaffold_norm, 3)
    
    return GermaneLoadResult(
        example_count=ex_count,
        summary_count=sum_count,
        question_count=q_count,
        scaffold_count=scaffold_cues,
        germane_score=germane_score
    )


def compute_cli(nlp, text: str) -> CLIScore:
    doc = nlp(text)
    intr = compute_intrinsic_load(text, doc)
    extr = compute_extraneous_load(text, doc)
    germ = compute_germane_load(text, doc)
    
    raw = settings.W_INTR * intr.intrinsic_score + settings.W_EXTR * extr.extraneous_score - settings.W_GERM * germ.germane_score
    clipped = max(min(raw, 1.0), -1.0)
    cli = round((clipped + 1.0) / 2.0, 3)
    
    return CLIScore(
        intrinsic=intr,
        extraneous=extr,
        germane=germ,
        raw_score=round(raw, 3),
        cli=cli,
        label="Low" if cli < 0.33 else ("Medium" if cli < 0.66 else "High"),
        scoring_method="rule_based"
    )

def compute_sentence_level_cli(nlp, text: str) -> List[SentenceCLIScore]:
    doc = nlp(text)
    results = []
    
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue
            
        cli_result = compute_cli(nlp, sentence_text)
        
        results.append(SentenceCLIScore(
            sentence=sentence_text,
            cli=cli_result.cli,
            label=cli_result.label,
            intrinsic=cli_result.intrinsic.intrinsic_score,
            extraneous=cli_result.extraneous.extraneous_score,
            germane=cli_result.germane.germane_score
        ))
        
    return results

def explain_cli(cli_obj: CLIScore) -> ExplanationResult:
    expl = []
    features = []
    
    intr = cli_obj.intrinsic.intrinsic_score
    extr = cli_obj.extraneous.extraneous_score
    germ = cli_obj.germane.germane_score
    
    if intr > 0.6:
        reason = "High intrinsic load: many domain terms / rare words detected."
        expl.append(reason)
        features.append(FeatureContribution(component="intrinsic", reason=reason, severity="high"))
    elif intr > 0.3:
        reason = "Moderate intrinsic load: some technical terms present."
        expl.append(reason)
        features.append(FeatureContribution(component="intrinsic", reason=reason, severity="medium"))
    else:
        expl.append("Low intrinsic load: general vocabulary.")
        
    if extr > 0.6:
        reason = "High extraneous load: complex sentence structures or many nominalizations."
        expl.append(reason)
        features.append(FeatureContribution(component="extraneous", reason=reason, severity="high"))
    elif extr > 0.3:
        reason = "Moderate extraneous load: some syntactic complexity."
        expl.append(reason)
        features.append(FeatureContribution(component="extraneous", reason=reason, severity="medium"))
    else:
        expl.append("Low extraneous load: simple presentation.")
        
    if germ > 0.5:
        reason = "High germane support: examples/summaries/questions present to aid learning."
        expl.append(reason)
        features.append(FeatureContribution(component="germane", reason=reason, severity="high"))
    elif germ > 0.2:
        reason = "Moderate germane support: some scaffolding cues present."
        expl.append(reason)
        features.append(FeatureContribution(component="germane", reason=reason, severity="medium"))
    else:
        reason = "Low germane support: lacks examples or retrieval prompts."
        expl.append(reason)
        features.append(FeatureContribution(component="germane", reason=reason, severity="low"))
        
    return ExplanationResult(human_readable=expl, features=features)

def estimate_working_memory_slots(nlp, text: str) -> WorkingMemoryData:
    doc = nlp(text)
    terms = extract_candidate_terms(doc)
    
    novel_concepts = []
    for term in terms:
        words = term.split()
        if not words: continue
        
        freqs = []
        for w in words:
            try:
                f = zipf_frequency(w.lower(), "en")
                if f is None or float('-inf') < f < float('inf') == False:
                    f = 1.0
            except Exception:
                f = 1.0
            freqs.append(f)
            
        avg_freq = sum(freqs) / len(freqs)
        if avg_freq < 4.0:
            novel_concepts.append(Concept(term=term, zipf_frequency=round(avg_freq, 2)))
            
    novel_concepts.sort(key=lambda x: x.zipf_frequency)
    slot_count = len(novel_concepts)
    miller_capacity = 7
    
    if slot_count <= 5: severity, color = "Low", "green"
    elif slot_count <= 9: severity, color = "Moderate", "orange"
    else: severity, color = "Overloaded", "red"
    
    if slot_count <= 5:
        rec = "Within comfortable working memory limits. Learners should be able to process this effectively."
    elif slot_count <= 9:
        rec = f"Near working memory capacity. Consider introducing {slot_count - 5} of these concepts earlier or providing definitions."
    else:
        top_hard = [c.term for c in novel_concepts[:3]]
        rec = f"Exceeds working memory capacity by ~{slot_count - 9} concepts. Consider pre-teaching: {', '.join(top_hard)}"
        
    return WorkingMemoryData(
        slot_count=slot_count,
        miller_capacity=miller_capacity,
        exceeds_capacity=slot_count > (miller_capacity + 2),
        severity=severity,
        color=color,
        novel_concepts=novel_concepts,
        utilization_pct=round(min(slot_count / miller_capacity * 100, 200), 1),
        recommendation=rec
    )

def compute_confidence(text: str, sentence_results: List[SentenceCLIScore]) -> float:
    words = tokenize_words(text)
    word_count = len(words)
    if word_count < 10: return 0.1
    
    length_confidence = min(word_count / 100.0, 1.0)
    scores = [r.cli for r in sentence_results]
    
    if len(scores) < 2:
        variance_confidence = 0.5
    else:
        var = statistics.variance(scores)
        variance_confidence = max(1.0 - (var * 3.0), 0.3)
        
    return round((length_confidence * 0.6) + (variance_confidence * 0.4), 2)

def identify_dominant_issue(cli_obj: CLIScore) -> str:
    intr = cli_obj.intrinsic.intrinsic_score
    extr = cli_obj.extraneous.extraneous_score
    germ_lack = 1.0 - cli_obj.germane.germane_score
    
    scores = {
        "intrinsic": intr * 1.0,
        "extraneous": extr * 1.2,
        "germane": germ_lack * 0.8
    }
    return max(scores, key=scores.get)
