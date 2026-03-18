import google.generativeai as genai
from typing import List, Optional, Dict
from core.config import settings
from models.schemas import SentenceCLIScore, OptimizationResult, OptimizationHistory
from services.scoring import identify_dominant_issue
from services.ml import compute_ml_cli_with_profile
from services.ai_tutor import _get_response_text

def get_genai_model(model_name: Optional[str] = None):
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(model_name or settings.DEFAULT_GEMINI_MODEL)

def rewrite_basic_simplify(paragraph: str, target_level: str = "Beginner", model_name: Optional[str] = None) -> str:
    model = get_genai_model(model_name)
    prompt = f"""
You are an expert educational editor. Rewrite the ORIGINAL paragraph to make it simpler for a {target_level} reader.
Constraints:
- Use short sentences (max ~15 words).
- Replace complex words with simple words where possible.
- Remove unnecessary nominalizations and simplify clause nesting.
- Preserve the main technical meaning and terms if essential.
- Add a one-sentence example or analogy if it helps clarity.

ORIGINAL:
\"\"\"{paragraph}\"\"\"

Produce the simplified paragraph only.
"""
    return _get_response_text(model.generate_content(prompt))

def rewrite_difficult_sentences(nlp, text: str, sentence_results: List[SentenceCLIScore], target_level: str = "Beginner", model_name: Optional[str] = None) -> str:
    model = get_genai_model(model_name)
    rewritten_sentences = []
    
    for r in sentence_results:
        if r.label == "High":
            prompt = f"""
Rewrite this sentence to reduce cognitive load for a {target_level} reader.

Sentence:
{r.sentence}

Rules:
- simplify wording
- shorten sentence
- keep original meaning
"""
            new_sentence = _get_response_text(model.generate_content(prompt))
            rewritten_sentences.append(new_sentence)
        else:
            rewritten_sentences.append(r.sentence)
            
    return " ".join(rewritten_sentences)

def rewrite_pedagogical(paragraph: str, target_level: str = "Beginner", model_name: Optional[str] = None) -> str:
    model = get_genai_model(model_name)
    prompt = f"""
You are an expert instructional designer. Rewrite the ORIGINAL paragraph to make it highly pedagogical for a {target_level} reader.
Constraints:
- Break down any highly technical definitions into simpler concepts.
- Provide a brief, relatable analogy for the main concept.
- Maintain a supportive, encouraging tone.
- Do not lose the core educational value.

ORIGINAL:
\"\"\"{paragraph}\"\"\"

Produce the pedagogically improved paragraph only.
"""
    return _get_response_text(model.generate_content(prompt))

def optimize_text(nlp, text: str, target_cli: float = 0.4, max_iterations: int = 3, 
                  reading_level: str = "Intermediate", domain_familiarity: float = 0.5,
                  objectives: Dict[str, str] = None, model_name: Optional[str] = None) -> OptimizationResult:
                  
    model = get_genai_model(model_name)
    if objectives is None:
        objectives = {"simplicity": "High", "technical_accuracy": "Medium", "pedagogy": "Medium"}
        
    current_text = text
    history = []
    
    cli_data = compute_ml_cli_with_profile(nlp, current_text, reading_level, domain_familiarity)
    current_cli = cli_data.cli
    history.append(OptimizationHistory(iteration=0, text=current_text, cli=current_cli, issue_addressed="baseline"))
    
    iteration = 1
    while current_cli > target_cli and iteration <= max_iterations:
        dominant_issue = identify_dominant_issue(cli_data)
        
        strategy_prompt = ""
        if dominant_issue == "intrinsic":
            strategy_prompt = "- The text has high intrinsic load (hard vocabulary). Please explain technical terms, reduce jargon, and use simpler synonyms."
        elif dominant_issue == "extraneous":
            strategy_prompt = "- The text has high extraneous load (complex grammar). Please break up long sentences, fix passive voice, and simplify the syntax."
        elif dominant_issue == "germane":
            strategy_prompt = "- The text lacks germane load (scaffolding). Please add a brief example, an analogy, or a summarizing sentence to help learning."
            
        obj_prompt = f"""
        Priorities:
        - Simplicity: {objectives.get('simplicity', 'High')}
        - Technical Accuracy: {objectives.get('technical_accuracy', 'High')}
        - Pedagogy: {objectives.get('pedagogy', 'Medium')}
        """
        
        prompt = f"""
You are an expert educational editor and learning engineer.
Your goal is to rewrite the ORIGINAL text to achieve a lower Cognitive Load Index (target < {target_cli}).
Current CLI is {current_cli}. The target reader is {reading_level}.

Feedback from the Cognitive Load Analyzer:
{strategy_prompt}
{obj_prompt}

ORIGINAL:
\"\"\"{current_text}\"\"\"

Produce ONLY the improved rewritten text without any conversational filler or formatting blocks.
"""
        new_text = _get_response_text(model.generate_content(prompt)).strip()
        if not new_text or new_text == current_text: break
            
        current_text = new_text
        cli_data = compute_ml_cli_with_profile(nlp, current_text, reading_level, domain_familiarity)
        current_cli = cli_data.cli
        
        history.append(OptimizationHistory(iteration=iteration, text=current_text, cli=current_cli, issue_addressed=dominant_issue))
        iteration += 1
        
    return OptimizationResult(
        final_text=current_text,
        target_reached=current_cli <= target_cli,
        iterations_used=len(history) - 1,
        history=history
    )
