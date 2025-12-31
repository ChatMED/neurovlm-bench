import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

def load_sectioned_csv(path):
    # dict to collect DataFrames per section
    sections = {}

    # keep section names in a separate list (DON'T overwrite the dict)
    section_headers = [
        "# Model-level Metrics",
        "# Overall per-Model (diagnosis_name)",
        "# Overall per-Model (diagnosis_detailed)",
        "# Overall per-Model (modality)",
        "# Overall per-Model (plane)",
        "# Per-Class Metrics (diagnosis_name)",
        "# Per-Subclass Metrics (diagnosis_detailed)",
        "# Per-Modality Metrics (modality)",
        "# Per-Plane Metrics (axial_plane vs plane)",
    ]

    current_name = None
    current_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            # only strip the newline; keep commas and internal spaces intact
            line = raw_line.rstrip("\n")
            line_stripped = line.strip()

            # start of a new section
            if line_stripped in section_headers:
                # flush previous section
                if current_name is not None and current_lines:
                    sections[current_name] = pd.read_csv(StringIO("\n".join(current_lines)))
                # set new section name (remove the leading '#' and surrounding spaces)
                current_name = line_stripped.lstrip("#").strip()
                current_lines = []
                continue

            # optional early stop if the file contains this marker
            if line_stripped in ["# confusion Matrix per Model (diagnosis_name)"]:
                break

            # collect non-empty lines as part of the current section
            if line_stripped != "":
                current_lines.append(line)

    # flush the last section after the loop
    if current_name is not None and current_lines:
        sections[current_name] = pd.read_csv(StringIO("\n".join(current_lines)))

    return sections


def plot_all(path):
    # Usage example:
    # NOTE: Your uploaded file is CSV (not Excel). Point to the CSV path.
    sections = load_sectioned_csv(path)
    print(list(sections.keys()))  # -> should list parsed section names


    model_metrics       = sections["Model-level Metrics"]
    overall_diagnosis_name      = sections["Overall per-Model (diagnosis_name)"]
    overall_diagnosis_detailed  = sections["Overall per-Model (diagnosis_detailed)"]
    overall_modality            = sections["Overall per-Model (modality)"]
    overall_plane               = sections["Overall per-Model (plane)"]
    per_diagnosis_name      = sections["Per-Class Metrics (diagnosis_name)"]
    per_diagnosis_detailed  = sections["Per-Subclass Metrics (diagnosis_detailed)"]
    per_modality            = sections["Per-Modality Metrics (modality)"]
    per_plane               = sections["Per-Plane Metrics (axial_plane vs plane)"]

    # -------------------
    # TABLE 1 – Core Performance
    # -------------------
    table1 = model_metrics[["model", "accuracy", "macro_f1"]]
    #table1.to_csv("results/table1_performance.csv", index=False)

    # -------------------
    # FIGURE 1 – Accuracy vs Macro-F1
    # -------------------
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=model_metrics, x="macro_f1", y="accuracy", hue="model", s=100, marker="o", edgecolor="black")
    plt.title("Accuracy vs Macro-F1")
    plt.xlabel("Macro-F1")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig("results/fig1_accuracy_vs_macroF1.png", dpi=300)
    plt.show()
    plt.close()


    # -------------------
    # TABLE 2 – Calibration
    # -------------------
    table2 = model_metrics[["model", "ece", "brier"]]
    #table2.to_csv("results/table2_calibration.csv", index=False)

    # -------------------
    # FIGURE 2 – Calibration (ECE) vs Accuracy
    # -------------------
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=model_metrics, x="accuracy", y="ece", hue="model", s=100, marker="o", edgecolor="black")
    plt.title("ECE vs Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("ECE (lower is better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig("results/fig2_ece_vs_accuracy.png", dpi=300)
    plt.show()
    plt.close()

    # -------------------
    # TABLE 3 – Efficiency & Latency
    # -------------------
    table3 = model_metrics[["model", "median_latency_ms", "avg_input_tokens", "avg_output_tokens", "avg_total_tokens"]]
    #table3.to_csv("results/table3_efficiency.csv", index=False)

    # -------------------
    # TABLE 4 – Cost
    # -------------------
    table4 = model_metrics[["model",  "avg_cost"]].copy()
    # optional: add derived columns

    table4["avg_cost"]        = model_metrics["avg_input_cost"]
    #table4.to_csv("results/table4_cost.csv", index=False)

    # -------------------
    # FIGURE 3 – Cost vs Accuracy
    # -------------------
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=model_metrics, x="avg_cost", y="accuracy", hue="model", s=100, marker="o", edgecolor="black")
    plt.title("Cost vs Accuracy")
    plt.xlabel("Average Cost (USD)")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig("results/fig3_cost_vs_accuracy.png", dpi=300)
    plt.show()
    plt.close()


    table8 = model_metrics[["model", "accuracy", "macro_f1", "ece", "avg_cost"]].copy()
    # Melt the dataframe to long format for seaborn
    table8_melted = table8.melt(id_vars="model", 
                            value_vars=["accuracy", "macro_f1", "ece", "avg_cost"],
                            var_name="metric", value_name="value")

    # Set figure size and style
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Create barplot
    ax = sns.barplot(data=table8_melted, x="model", y="value", hue="metric")

    # Formatting
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=7)
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.tight_layout()
    plt.show()

def main():
    path = "results/all_subset_1_temp_0_20250823_172529.csv"
    plot_all(path)


if __name__ == "__main__":
    main()
