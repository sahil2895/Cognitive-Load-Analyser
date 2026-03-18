import spacy
from services.ml import compute_ml_cli_with_profile
from services.scoring import explain_cli
from services.rewriter import optimize_text
from services.semantics import compute_semantic_drift

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000

text = "The utilization of cognitive load theory can be somewhat difficult. For example, intrinsic load deals with the complexity of the material itself. It has been shown by many studies that this is true."

print("Testing compute_ml_cli_with_profile...")
cli_data = compute_ml_cli_with_profile(nlp, text, reading_level="Beginner", domain_familiarity=0.2)
print("compute_ml_cli_with_profile OK. CLI:", cli_data.cli)

print("Testing explain_cli...")
explanation = explain_cli(cli_data)
print("explain_cli OK.", explanation.human_readable)

print("Testing optimize_text (1 iteration to save API)...")
try:
    opt_result = optimize_text(nlp, text, target_cli=0.3, max_iterations=1, reading_level="Beginner")
    print("optimize_text OK. Iterations:", opt_result.iterations_used)
    
    print("Testing compute_semantic_drift...")
    drift = compute_semantic_drift(text, opt_result.final_text)
    print("compute_semantic_drift OK. Meaning Preserved:", drift.similarity)
except Exception as e:
    print(f"API functions failed: {e}")

print("ALL TESTS COMPLETED")
