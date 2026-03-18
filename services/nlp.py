import re
import spacy
from typing import List, Optional
from wordfreq import zipf_frequency
import textstat

# Cache spacy model internally in module, protected by getter
_nlp_cache = None

def get_spacy():
    global _nlp_cache
    if _nlp_cache is None:
        try:
            _nlp_cache = spacy.load("en_core_web_sm", disable=["ner"])
            _nlp_cache.max_length = 2_000_000
        except Exception as e:
            print(f"Failed to load spacy model: {e}")
            raise
    return _nlp_cache

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
            if f is None or float('-inf') < f < float('inf') == False: # handle inf
                f = 1.0 # fallback
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
