"""
Minimal plotting script comparing GPT-3.5-Turbo vs GPT-4o anonymization results
Generates plots from existing test_real_profile and test_real_profile_gpt4o data
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

ROOT = Path(__file__).resolve().parent
RESULTS_GPT35 = ROOT / "anonymized_results" / "test_real_profile"
RESULTS_GPT4O = ROOT / "anonymized_results" / "test_real_profile_gpt4o"
OUTPUT_DIR = ROOT / "plots_minimal"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_jsonl(path: Path):
    """Load JSONL file"""
    data = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def extract_utility_scores(util_path: Path) -> pd.DataFrame:
    """Extract utility scores from utility JSONL file"""
    records = []
    utility_data = load_jsonl(util_path)
    
    for profile in utility_data:
        username = profile.get("username", "unknown")
        for i, comment_group in enumerate(profile.get("comments", [])):
            utility = comment_group.get("utility", {})
            if utility:
                for model, scores in utility.items():
                    record = {
                        "username": username,
                        "comment_group": i,
                        "model": model,
                        "readability": scores.get("readability", {}).get("score"),
                        "meaning": scores.get("meaning", {}).get("score"),
                        "hallucinations": scores.get("hallucinations", {}).get("score"),
                        "bleu": scores.get("bleu"),
                    }
                    records.append(record)
    
    return pd.DataFrame(records)

def extract_inference_accuracy(anon_path: Path, inf_path: Path) -> pd.DataFrame:
    """Extract inference success (comparing before/after anonymization)"""
    records = []
    anon_data = load_jsonl(anon_path)
    inf_data = load_jsonl(inf_path)
    
    if not inf_data:
        return pd.DataFrame()
    
    for profile in inf_data:
        username = profile.get("username", "unknown")
        for comment in profile.get("comments", []):
            predictions = comment.get("predictions", {})
            if predictions:
                for model, attrs in predictions.items():
                    # Count inferred attributes
                    num_inferred = len([a for a in attrs.keys() if a != "full_answer"])
                    record = {
                        "username": username,
                        "model": model,
                        "num_attributes_inferred": num_inferred,
                    }
                    records.append(record)
    
    return pd.DataFrame(records)

def plot_utility_comparison():
    """Plot utility scores for both models"""
    # Load utility data
    util35_0 = extract_utility_scores(RESULTS_GPT35 / "utility_0.jsonl")
    util4o_0 = extract_utility_scores(RESULTS_GPT4O / "utility_0.jsonl")
    
    if util35_0.empty and util4o_0.empty:
        print("No utility data found")
        return
    
    util35_0["model_type"] = "GPT-3.5-Turbo (Iter 0)"
    util4o_0["model_type"] = "GPT-4o (Iter 0)"
    
    combined = pd.concat([util35_0, util4o_0], ignore_index=True)
    
    # Plot readability, meaning, hallucinations
    metrics = ["readability", "meaning", "hallucinations"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        data_to_plot = combined[["model_type", metric]].dropna()
        sns.boxplot(data=data_to_plot, x="model_type", y=metric, ax=axes[idx], palette=["#EA985F", "#1D81A2"])
        axes[idx].set_title(f"{metric.capitalize()} Scores (Iteration 0)", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel("Score")
        axes[idx].set_xlabel("")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "utility_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: utility_comparison.png")
    plt.close()

def plot_bleu_scores():
    """Plot BLEU scores comparison"""
    util35_0 = extract_utility_scores(RESULTS_GPT35 / "utility_0.jsonl")
    util4o_0 = extract_utility_scores(RESULTS_GPT4O / "utility_0.jsonl")
    
    if util35_0.empty and util4o_0.empty:
        return
    
    util35_0["model_type"] = "GPT-3.5-Turbo"
    util4o_0["model_type"] = "GPT-4o"
    
    combined = pd.concat([util35_0, util4o_0], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = combined[["model_type", "bleu"]].dropna()
    sns.boxplot(data=data_to_plot, x="model_type", y="bleu", ax=ax, palette=["#EA985F", "#1D81A2"])
    ax.set_title("BLEU Score Comparison (Iteration 0)", fontsize=14, fontweight='bold')
    ax.set_ylabel("BLEU Score")
    ax.set_xlabel("")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bleu_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: bleu_comparison.png")
    plt.close()

def plot_inference_comparison():
    """Plot number of attributes inferred before/after anonymization"""
    inf0_35 = extract_inference_accuracy(RESULTS_GPT35 / "anonymized_0.jsonl", RESULTS_GPT35 / "inference_0.jsonl")
    inf1_35 = extract_inference_accuracy(RESULTS_GPT35 / "anonymized_0.jsonl", RESULTS_GPT35 / "inference_1.jsonl")
    inf2_35 = extract_inference_accuracy(RESULTS_GPT35 / "anonymized_0.jsonl", RESULTS_GPT35 / "inference_2.jsonl")
    
    inf0_4o = extract_inference_accuracy(RESULTS_GPT4O / "anonymized_0.jsonl", RESULTS_GPT4O / "inference_0.jsonl")
    inf1_4o = extract_inference_accuracy(RESULTS_GPT4O / "anonymized_0.jsonl", RESULTS_GPT4O / "inference_1.jsonl")
    inf2_4o = extract_inference_accuracy(RESULTS_GPT4O / "anonymized_0.jsonl", RESULTS_GPT4O / "inference_2.jsonl")
    
    if all([df.empty for df in [inf0_35, inf1_35, inf2_35, inf0_4o, inf1_4o, inf2_4o]]):
        return
    
    # Prepare data
    data = []
    
    for df, stage, model in [
        (inf0_35, "Original", "GPT-3.5-Turbo"),
        (inf1_35, "After Anon 1", "GPT-3.5-Turbo"),
        (inf2_35, "After Anon 2", "GPT-3.5-Turbo"),
        (inf0_4o, "Original", "GPT-4o"),
        (inf1_4o, "After Anon 1", "GPT-4o"),
        (inf2_4o, "After Anon 2", "GPT-4o"),
    ]:
        if not df.empty:
            avg_attrs = df["num_attributes_inferred"].mean()
            data.append({"Stage": stage, "Model": model, "Avg Attributes Inferred": avg_attrs})
    
    if not data:
        return
    
    df_plot = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot, x="Stage", y="Avg Attributes Inferred", hue="Model", ax=ax, palette=["#EA985F", "#1D81A2"])
    ax.set_title("Inference Success: Attributes Detected Across Anonymization Stages", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Number of Attributes Inferred")
    ax.set_xlabel("Stage")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "inference_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: inference_comparison.png")
    plt.close()

def plot_utility_by_profile():
    """Plot utility scores broken down by profile"""
    util35_0 = extract_utility_scores(RESULTS_GPT35 / "utility_0.jsonl")
    util4o_0 = extract_utility_scores(RESULTS_GPT4O / "utility_0.jsonl")
    
    if util35_0.empty and util4o_0.empty:
        return
    
    util35_0["model_type"] = "GPT-3.5-Turbo"
    util4o_0["model_type"] = "GPT-4o"
    
    combined = pd.concat([util35_0, util4o_0], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Average utility scores by profile and model
    avg_scores = combined.groupby(["username", "model_type"])[["readability", "meaning", "hallucinations"]].mean()
    avg_scores_reset = avg_scores.reset_index()
    
    # Reshape for plotting
    plot_data = []
    for _, row in avg_scores_reset.iterrows():
        for metric in ["readability", "meaning", "hallucinations"]:
            plot_data.append({
                "Profile": row["username"],
                "Model": row["model_type"],
                "Metric": metric,
                "Score": row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    sns.barplot(data=plot_df, x="Profile", y="Score", hue="Model", palette=["#EA985F", "#1D81A2"], ax=ax)
    ax.set_title("Average Utility Scores by Profile (Iteration 0)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Score")
    ax.set_xlabel("Profile")
    ax.set_ylim([0, 10])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "utility_by_profile.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: utility_by_profile.png")
    plt.close()

def main():
    print("\n📊 Generating minimal comparison plots...\n")
    
    plot_utility_comparison()
    plot_bleu_scores()
    plot_inference_comparison()
    plot_utility_by_profile()
    
    print(f"\n✅ All plots saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()
