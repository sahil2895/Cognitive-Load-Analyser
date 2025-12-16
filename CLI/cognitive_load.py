

import re
import math
from typing import Dict, Any, List
from wordfreq import zipf_frequency
import textstat

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text)

def average_word_zipf(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    freqs = []
    for w in words:
        try:
            f = zipf_frequency(w.lower(), "en")
            if math.isinf(f) or f is None:
                f = 1.0
        except Exception:
            f = 1.0
        freqs.append(f)
    return sum(freqs) / len(freqs)

def rare_word_ratio(text: str, threshold_zipf: float = 4.5) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    rare = [1 for w in words if zipf_frequency(w.lower(), "en") < threshold_zipf]
    return len(rare) / len(words)

def extract_candidate_terms(doc) -> List[str]:
    terms = set()
    for nc in doc.noun_chunks:
        cleaned = nc.text.strip().lower()
        if len(cleaned.split()) >= 1:
            terms.add(cleaned)
    prop = []
    for tok in doc:
        if tok.pos_ == "PROPN":
            prop.append(tok.text)
        else:
            if prop:
                terms.add(" ".join(prop).lower())
                prop = []
    if prop:
        terms.add(" ".join(prop).lower())
    return list(terms)

def sentence_dependency_depths(doc) -> List[int]:
    depths = []
    for sent in doc.sents:
        max_depth = 0
        for tok in sent:
            depth = 0
            node = tok
            while node.head != node:
                depth += 1
                node = node.head
                if depth > 200:
                    break
            if depth > max_depth:
                max_depth = depth
        depths.append(max_depth)
    return depths

def average_branching_factor(doc) -> float:
    counts = [len(list(tok.children)) for tok in doc]
    if not counts:
        return 0.0
    return sum(counts) / len(counts)

def count_passive_voice(doc) -> int:
    passive_count = 0
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if re.search(r"\b(was|were|is|are|been|be|being|has been|have been)\b", sent_text) and any(tok.tag_ == "VBN" for tok in sent):
            passive_count += 1
    return passive_count

def presence_of_examples(text: str) -> int:
    markers = [r"\bfor example\b", r"\be\.g\.", r"\bsuch as\b", r"\bfor instance\b", r"\bexample:\b"]
    count = 0
    t = text.lower()
    for m in markers:
        if re.search(m, t):
            count += 1
    count += len(re.findall(r"(^|\n)\s*example\b", t))
    return count

def presence_of_summaries(text: str) -> int:
    markers = [r"\bin summary\b", r"\bto conclude\b", r"\bin conclusion\b", r"\bto summariz", r"\bin brief\b"]
    t = text.lower()
    return sum(1 for m in markers if re.search(m, t))

def count_questions(text: str) -> int:
    return text.count("?")

def textstat_avg_sentence_length_safe(text: str) -> float:
    try:
        return textstat.avg_sentence_length(text)
    except Exception:
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        words = tokenize_words(text)
        return (len(words) / max(1, len(sentences))) if sentences else 0.0

def compute_intrinsic_load(text: str, doc) -> Dict[str, Any]:
    words = tokenize_words(text)
    total_words = max(1, len(words))
    terms = extract_candidate_terms(doc)
    term_ratio = len([t for t in terms if len(t.split()) >= 1]) / total_words
    avg_zipf = average_word_zipf(text)
    rare_ratio = rare_word_ratio(text)
    term_norm = min(term_ratio * 5.0, 1.0)
    zipf_norm = 1.0 - (min(max((avg_zipf - 1.0) / 6.0, 0.0), 1.0))
    intrinsic_score = round(0.5 * term_norm + 0.3 * rare_ratio + 0.2 * zipf_norm, 3)
    return {
        "term_ratio": round(term_ratio, 4),
        "num_terms": len(terms),
        "avg_zipf": round(avg_zipf, 3),
        "rare_ratio": round(rare_ratio, 3),
        "intrinsic_score": intrinsic_score,
        "terms_sample": terms[:8]
    }

def compute_extraneous_load(text: str, doc) -> Dict[str, Any]:
    depths = sentence_dependency_depths(doc)
    avg_depth = sum(depths) / len(depths) if depths else 0.0
    branch = average_branching_factor(doc)
    passive = count_passive_voice(doc)
    avg_sent_len = textstat_avg_sentence_length_safe(text)
    words = tokenize_words(text)
    nominals = sum(1 for w in words if re.search(r"(tion|ment|ness|ity|ization)$", w.lower()))
    nom_ratio = nominals / max(1, len(words))
    depth_norm = min(avg_depth / 10.0, 1.0)
    branch_norm = min(branch / 3.0, 1.0)
    passive_norm = min(passive / max(1, len(list(doc.sents))), 1.0)
    sentlen_norm = min(avg_sent_len / 30.0, 1.0)
    nom_norm = min(nom_ratio * 10.0, 1.0)
    extraneous_score = round(0.3 * depth_norm + 0.25 * branch_norm + 0.15 * sentlen_norm + 0.15 * nom_norm + 0.15 * passive_norm, 3)
    return {
        "avg_dependency_depth": round(avg_depth, 3),
        "avg_branching": round(branch, 3),
        "passive_count": passive,
        "avg_sentence_length": round(avg_sent_len, 3),
        "nominalization_ratio": round(nom_ratio, 4),
        "extraneous_score": extraneous_score
    }

def compute_germane_load(text: str, doc) -> Dict[str, Any]:
    ex_count = presence_of_examples(text)
    sum_count = presence_of_summaries(text)
    q_count = count_questions(text)
    scaffold_cues = len(re.findall(r"\b(first|second|third|then|next|step)\b", text.lower()))
    ex_norm = min(ex_count / 3.0, 1.0)
    sum_norm = min(sum_count / 2.0, 1.0)
    q_norm = min(q_count / 3.0, 1.0)
    scaffold_norm = min(scaffold_cues / 6.0, 1.0)
    germane_score = round(0.4 * ex_norm + 0.25 * sum_norm + 0.2 * q_norm + 0.15 * scaffold_norm, 3)
    return {
        "example_count": ex_count,
        "summary_count": sum_count,
        "question_count": q_count,
        "scaffold_count": scaffold_cues,
        "germane_score": germane_score
    }

def compute_cli(nlp, text: str) -> Dict[str, Any]:
    doc = nlp(text)
    intr = compute_intrinsic_load(text, doc)
    extr = compute_extraneous_load(text, doc)
    germ = compute_germane_load(text, doc)
    w_intr = 0.45
    w_extr = 0.45
    w_germ = 0.35
    raw = w_intr * intr["intrinsic_score"] + w_extr * extr["extraneous_score"] - w_germ * germ["germane_score"]
    clipped = max(min(raw, 1.0), -1.0)
    cli = round((clipped + 1.0) / 2.0, 3)
    return {
        "intrinsic": intr,
        "extraneous": extr,
        "germane": germ,
        "raw_score": round(raw, 3),
        "cli": cli,
        "label": "Low" if cli < 0.33 else ("Medium" if cli < 0.66 else "High")
    }

def explain_cli_result(cli_dict: Dict[str, Any]) -> List[str]:
    expl = []
    intr = cli_dict["intrinsic"]
    extr = cli_dict["extraneous"]
    germ = cli_dict["germane"]
    if intr["intrinsic_score"] > 0.6:
        expl.append("High intrinsic load: many domain terms / rare words detected.")
    elif intr["intrinsic_score"] > 0.3:
        expl.append("Moderate intrinsic load: some technical terms present.")
    else:
        expl.append("Low intrinsic load: general vocabulary.")
    if extr["extraneous_score"] > 0.6:
        expl.append("High extraneous load: complex sentence structures or many nominalizations.")
    elif extr["extraneous_score"] > 0.3:
        expl.append("Moderate extraneous load: some syntactic complexity.")
    else:
        expl.append("Low extraneous load: simple presentation.")
    if germ["germane_score"] > 0.5:
        expl.append("High germane support: examples/summaries/questions present to aid learning.")
    elif germ["germane_score"] > 0.2:
        expl.append("Moderate germane support: some scaffolding cues present.")
    else:
        expl.append("Low germane support: lacks examples or retrieval prompts.")
    return expl



