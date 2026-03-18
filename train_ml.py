import os
import json
import spacy
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from services.scoring import compute_cli
from core.config import settings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

os.makedirs(settings.MODEL_DIR, exist_ok=True)

print("Loading dataset: archive/CLEAR.csv...")
df = pd.read_csv("archive/CLEAR.csv")

# BT Easiness: Higher = easier (lower cognitive load).
# Invert so higher = harder (higher cognitive load).
df['target_cli'] = df['BT Easiness'] * -1

print(f"Dataset loaded with {len(df)} samples.")
print("Extracting NLP features using services/scoring.py (This will take a few minutes)...")

nlp = spacy.load('en_core_web_sm')

feature_list = []
rule_based_predictions = []

# Use first 1000 rows for training speed
df_sample = df.head(1000).copy()

for idx, text in enumerate(df_sample['Excerpt']):
    if idx > 0 and idx % 100 == 0:
        print(f"Processed {idx}/{len(df_sample)} texts...")
    
    cli_result = compute_cli(nlp, text)
    
    feats = {
        'intrinsic_avg_zipf': cli_result.intrinsic.avg_zipf,
        'intrinsic_rare_ratio': cli_result.intrinsic.rare_ratio,
        'intrinsic_term_ratio': cli_result.intrinsic.term_ratio,
        'intrinsic_num_terms': cli_result.intrinsic.num_terms,
        'intrinsic_score': cli_result.intrinsic.intrinsic_score,
        
        'extraneous_avg_branching': cli_result.extraneous.avg_branching,
        'extraneous_avg_sentence_length': cli_result.extraneous.avg_sentence_length,
        'extraneous_avg_dependency_depth': cli_result.extraneous.avg_dependency_depth,
        'extraneous_passive_count': cli_result.extraneous.passive_count,
        'extraneous_nominalization_ratio': cli_result.extraneous.nominalization_ratio,
        'extraneous_score': cli_result.extraneous.extraneous_score,
        
        'germane_example_count': cli_result.germane.example_count,
        'germane_summary_count': cli_result.germane.summary_count,
        'germane_question_count': cli_result.germane.question_count,
        'germane_scaffold_count': cli_result.germane.scaffold_count,
        'germane_score': cli_result.germane.germane_score,
    }

    feature_list.append(feats)
    rule_score = cli_result.cli
    rule_based_predictions.append(rule_score)

features_df = pd.DataFrame(feature_list)

X = features_df
y = df_sample['target_cli']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rule_preds_test = np.array(rule_based_predictions)[X_test.index]

print("\nTraining models with Evaluation Metrics...")

def evaluate_model(name, true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    ev = explained_variance_score(true_values, predicted_values)
    
    print(f"\n--- {name} Results ---")
    print(f"1. MSE:  {mse:.4f}")
    print(f"2. RMSE: {rmse:.4f}")
    print(f"3. MAE:  {mae:.4f}")
    print(f"4. R²:   {r2:.4f}")
    print(f"5. EV:   {ev:.4f}")
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "ev": ev}

# --- RANDOM FOREST ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_metrics = evaluate_model("Random Forest (ML)", y_test, rf_preds)

# --- XGBOOST ---
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_metrics = evaluate_model("XGBoost (ML)", y_test, xgb_preds)

# --- CURRENT RULE-BASED ---
rule_metrics = evaluate_model("Current Rule-Based System", y_test, rule_preds_test)

print("\n===== Saving Best Model =====")

if xgb_metrics["r2"] >= rf_metrics["r2"]:
    best_model = xgb_model
    best_name = "XGBoost"
    best_metrics = xgb_metrics
    best_preds_all = xgb_model.predict(X)
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_metrics = rf_metrics
    best_preds_all = rf_model.predict(X)

model_path = os.path.join(settings.MODEL_DIR, "best_cli_model.joblib")
joblib.dump(best_model, model_path)
print(f"Saved {best_name} model to {model_path}")

feature_columns_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
with open(feature_columns_path, "w") as f:
    json.dump(list(X.columns), f)

all_predictions = best_model.predict(X).tolist()
all_predictions_sorted = sorted(all_predictions)
pred_min, pred_max = min(all_predictions), max(all_predictions)

percentile_data = {
    "sorted_scores": all_predictions_sorted,
    "dataset_size": len(all_predictions),
    "dataset_name": "CLEAR Corpus (CommonLit Ease of Readability)",
    "score_min": pred_min,
    "score_max": pred_max,
    "score_mean": float(np.mean(all_predictions)),
    "score_std": float(np.std(all_predictions)),
    "percentile_25": float(np.percentile(all_predictions, 25)),
    "percentile_50": float(np.percentile(all_predictions, 50)),
    "percentile_75": float(np.percentile(all_predictions, 75)),
}

percentile_path = os.path.join(settings.MODEL_DIR, "percentile_data.json")
with open(percentile_path, "w") as f:
    json.dump(percentile_data, f, indent=2)

from datetime import datetime
metadata = {
    "model_name": best_name,
    "trained_on": datetime.now().isoformat(),
    "dataset": "CLEAR Corpus",
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "total_samples_used": len(X),
    "metrics": {k: round(v, 4) for k, v in best_metrics.items()},
    "rule_based_r2": round(rule_metrics["r2"], 4),
    "improvement_over_rules": round(best_metrics["r2"] - rule_metrics["r2"], 4),
    "normalization": {"min": round(pred_min, 4), "max": round(pred_max, 4)},
    "feature_columns": list(X.columns),
}

metadata_path = os.path.join(settings.MODEL_DIR, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ {best_name} saved as the production model.")
print(f"   R²: {best_metrics['r2']:.4f} (vs rule-based R²: {rule_metrics['r2']:.4f})")
print("\nComparison Complete. Model ready for use in the app.")
