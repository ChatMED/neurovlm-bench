import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_sectioned_csv(path):
    # dict to collect DataFrames per section
    sections = {}

    # keep section names in a separate list (DON'T overwrite the dict)
    section_headers = [
        "# Model-level Metrics",
        "# Overall per-Model (diagnosis_name)",
        "# Overall per-Model (diagnosis_detailed)",
        "# Overall per-Model (modality)",
        "# Overall per-Model (specialized_sequence)",
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
    #print(list(sections.keys()))  # -> should list parsed section names


    model_metrics       = sections["Model-level Metrics"]
    #overall_diagnosis_name      = sections["Overall per-Model (diagnosis_name)"]
    #overall_diagnosis_detailed  = sections["Overall per-Model (diagnosis_detailed)"]
    #overall_modality            = sections["Overall per-Model (modality)"]
    #overall_plane               = sections["Overall per-Model (plane)"]
    per_diagnosis_name      = sections["Per-Class Metrics (diagnosis_name)"]
    #per_diagnosis_detailed  = sections["Per-Subclass Metrics (diagnosis_detailed)"]
    #per_modality            = sections["Per-Modality Metrics (modality)"]
    #per_plane               = sections["Per-Plane Metrics (axial_plane vs plane)"]

    # -------------------
    # TABLE 1 – Core Performance
    # -------------------
    table1 = model_metrics[["model", "accuracy", "macro_f1"]]
    #table1.to_csv("results/table1_performance.csv", index=False)
    #print(table1)
    print(table1["model"])
    print(len(table1["model"]))

    # -------------------
    # FIGURE 1 – Accuracy vs Macro-F1
    # -------------------
    num_models = len(table1["model"])
    colors = plt.colormaps["tab20"](np.linspace(0, 1, num_models))
    color_map = dict(zip(table1["model"], colors))
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=model_metrics, x="macro_f1", y="accuracy", hue="model", s=150, marker="o", edgecolor="w", palette=color_map)
    plt.title("Accuracy vs Macro-F1")
    plt.xlabel("Macro-F1",fontsize=12)
    plt.ylabel("Accuracy",fontsize=12)
    plt.grid(True)

    # Add legend (model names on the right)
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[model], markersize=8)
                           for model in table1["model"]]
    labels = [model.split("/")[-1].replace(":", "") for model in table1["model"]]

    plt.legend(handles, labels, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig("results/plots/fig1_accuracy_vs_macroF1_"+path[12:-4]+".png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    # -------------------
    # TABLE 2 – Calibration
    # -------------------
    table2 = model_metrics[["model", "ece", "brier"]]
    #table2.to_csv("results/plots/table2_calibration.csv", index=False)

    # -------------------
    # FIGURE 2 – Calibration (ECE) vs Accuracy
    # -------------------
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=model_metrics, x="accuracy", y="ece", hue="model", s=150, marker="o", edgecolor="w", palette=color_map)
    plt.title("ECE vs Accuracy")
    plt.xlabel("Accuracy", fontsize=12)
    plt.ylabel("ECE (lower is better)", fontsize=12)
    
    plt.grid(True)

    # Add legend (model names on the right)
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[model], markersize=8)
                           for model in table1["model"]]
    labels = [model.split("/")[-1].replace(":", "") for model in table1["model"]]

    plt.legend(handles, labels, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig("results/plots/fig2_ece_vs_accuracy_"+path[12:-4]+".png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # -------------------
    # TABLE 3 – Efficiency & Latency
    # -------------------
    table3 = model_metrics[["model", "median_latency_ms", "avg_input_tokens", "avg_output_tokens", "avg_total_tokens"]]
    #table3.to_csv("results/plots/table3_efficiency.csv", index=False)

    # -------------------
    # TABLE 4 – Cost
    # -------------------
    table4 = model_metrics[["model",  "avg_cost"]].copy()
    # optional: add derived columns

    table4["avg_cost"]        = model_metrics["avg_input_cost"]
    #table4.to_csv("results/plots/table4_cost.csv", index=False)

    # -------------------
    # FIGURE 3 – Cost vs Accuracy
    # -------------------
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=model_metrics, x="avg_cost", y="accuracy", hue="model", s=150, marker="o", edgecolor="w", palette = color_map)
    plt.title("Cost vs Accuracy")
    plt.xlabel("Average Cost (USD)", fontsize=12)
    plt.ylabel("Accuracy",fontsize=12)
    plt.grid(True)

    # Add legend (model names on the right)
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[model], markersize=8)
                           for model in table1["model"]]
    labels = [model.split("/")[-1].replace(":", "") for model in table1["model"]]

    plt.legend(handles, labels, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig("results/plots/fig3_cost_vs_accuracy_"+path[12:-4]+".png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    table5 = model_metrics[["model", "accuracy", "macro_f1_abstention","weighted_f1_abstention", "ece"]].copy()
    # Melt the dataframe to long format for seaborn
    table5_melted = table5.melt(id_vars="model", 
                            value_vars=["accuracy", "macro_f1_abstention","weighted_f1_abstention","ece"],
                            var_name="metric", value_name="value")
    
    
    #print(table5_melted)
    print(type(table5_melted))
    #table5_sorted = table5_melted.sort_values(by='macro_f1_abstention', ascending=False)
    # Set figure size and style
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Create barplot
    ax = sns.barplot(data=table5_melted, x="model", y="value", hue="metric")

    # Formatting
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=7)
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.tight_layout()
    plt.show()
    
    
    
    # ---------- PLOT heatmap f1 models per class----------
    table6 = per_diagnosis_name[["model", "class","f1"]]
    models = table6["model"].drop_duplicates().tolist() 
    classes = per_diagnosis_name["class"].drop_duplicates().tolist()
    classes_plot = (per_diagnosis_name["class"].drop_duplicates().str.strip().str.title().tolist())
    # fix acronyms if needed
    acronyms = {"Ms": "MS"}
    classes_plot = [acronyms.get(c, c) for c in classes_plot]
    
    # index mappings
    model_to_idx = {m: i for i, m in enumerate(models)}
    class_to_idx = {c: j for j, c in enumerate(classes)}

    # initialize matrix with 0.0 (or np.nan if you prefer)
    data_list = np.zeros((len(models), len(classes)), dtype=float)

    # fill matrix using indices
    for _, row in per_diagnosis_name.iterrows():
        i = model_to_idx[row["model"]]
        j = class_to_idx[row["class"]]
        data_list[i, j] = row["f1"]
    
    # mask: keep columns that have at least one non-zero value
    keep_col_mask = np.any(data_list != 0.0, axis=0)

    # filter data and class labels
    data_list = data_list[:, keep_col_mask]
    classes = [c for c, keep in zip(classes, keep_col_mask) if keep]
    classes_plot = [c for c, keep in zip(classes_plot, keep_col_mask) if keep]
    
    plt.figure(figsize=(14, 10))
    im = plt.imshow(data_list, aspect='auto', cmap='Blues')
    # Tick labels
    plt.xticks(np.arange(len(classes)), classes_plot, rotation=45, ha="right")
    plt.yticks(np.arange(len(models)), models)

    # Annotate values with automatically chosen text color
    for i in range(data_list.shape[0]):
        for j in range(data_list.shape[1]):
            value = data_list[i, j]
            text_color = "white" if value > 0.5 else "black"
            #text_color = "white" 
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

    plt.colorbar(im, label="F1-score")
    plt.grid(False)
    #plt.title("Per-class F1 Heatmap (Improved Contrast)", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/plots/per_class_f1_heatmap_"+path[12:-4]+".png", dpi=300, bbox_inches="tight")
    plt.show()



def plot_heatmaps_zero_few_shot(path):
    # ----------------------------
    # 1) Phase 3 - zero shot
    # ----------------------------
    sections = load_sectioned_csv(path)
    print(list(sections.keys()))  # -> should list parsed section names

    overall_diagnosis_name      = sections["Overall per-Model (diagnosis_name)"]
    overall_diagnosis_detailed  = sections["Overall per-Model (diagnosis_detailed)"]
    overall_modality            = sections["Overall per-Model (modality)"]
    overall_modality_subtype    = sections["Overall per-Model (specialized_sequence)"]
    overall_plane               = sections["Overall per-Model (plane)"]
    
    table1 = overall_diagnosis_name[["model", "f1_macro_abstention"]].rename(columns={"f1_macro_abstention": "diagnosis_name"})
    table2 = overall_diagnosis_detailed[["model", "f1_macro_abstention"]].rename(columns={"f1_macro_abstention": "diagnosis_detailed"})
    table3 = overall_modality[["model", "f1_macro_abstention"]].rename(columns={"f1_macro_abstention": "modality"})
    table4 = overall_modality_subtype[["model", "f1_macro_abstention"]].rename(columns={"f1_macro_abstention": "specialized_sequence"})
    table5 = overall_plane[["model", "f1_macro_abstention"]].rename(columns={"f1_macro_abstention": "plane"})

    df = (table1.merge(table2, on="model", how="inner").merge(table3, on="model", how="inner").merge(table4, on="model", how="inner").merge(table5, on="model", how="inner").set_index("model"))
    
    # Coerce numbers like "1,00" -> 1.00
    #df = df.applymap(lambda x: float(str(x).replace(",", ".")) if not pd.isna(x) else np.nan)
    df = df.apply(
        lambda col: pd.to_numeric(col.astype(str).str.replace(",", "."), errors="coerce")
    )

    # ----------------------------
    # 2) IEEE JBHI heatmap column order for the 5 tasks:
    #    Modality → Plane → Specialized sequence → Diagnosis name → Diagnosis detailed
    # ----------------------------
    col_order = ["diagnosis_name", "diagnosis_detailed", "modality", "specialized_sequence", "plane" ]
    data = df[col_order].copy()

    pretty_cols = {
        "modality": "Modality",
        "plane": "Plane",
        "specialized_sequence": "MRI Sequence",
        "diagnosis_name": "Diagnosis",
        "diagnosis_detailed": "Diagnosis Detailed",
    }
    data.columns = [pretty_cols.get(c, c) for c in data.columns]

    # ----------------------------
    # 3) Column-wise normalization + gamma correction to avoid ceiling compression
    #    - Colors are normalized per column (task)
    #    - Values printed are absolute Macro-F1 scores
    # ----------------------------
    gamma = 0.5  # <1 emphasizes differences near the top end (e.g., 0.99–1.00)
    vals = data.values.astype(float)
    norm = np.zeros_like(vals)

    for j in range(vals.shape[1]):
        col = vals[:, j]
        cmin, cmax = np.nanmin(col), np.nanmax(col)
        if np.isclose(cmax, cmin):
            norm[:, j] = 0.5
        else:
            scaled = (col - cmin) / (cmax - cmin)  # [0,1]
            norm[:, j] = np.power(scaled, gamma)

    # ----------------------------
    # 4) Plot (matplotlib only)
    # ----------------------------
    fig_w = max(8.0, 1.2 * vals.shape[1])
    fig_h = max(3.5, 0.6 * vals.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm, aspect="auto", interpolation="nearest", cmap="Blues")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticklabels(data.index)

    # Subtle cell grid
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate absolute values
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "NA" if np.isnan(v) else f"{v:.3f}"
            text_color = "white" if norm[i, j] > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9)

    #ax.set_title(
    #    "Task-wise Macro-F1 (abstention-aware) for structured radiology prompting\n"
    #    "Colors normalized per task (column-wise) with gamma=0.5; numbers are absolute scores"
    #)

    # Because colors are column-normalized, the colorbar is “relative intensity” only.
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    cbar.set_ticks([])

    cbar.set_label("Relative performance\n(per-task normalized)", rotation=90)

    #cbar.set_label("Relative intensity (per-task normalized)")
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig('results/plots/heatmap_tasks_ph3_'+path[12:-4]+'.png', dpi=300, bbox_inches="tight")
    plt.show()

    

def plot_datasets(path):
    datasets = ['images-17','images-44c','figshare', 
                'stroke', 'sclerosis', 'aisd', 'Br35H']
    
    data_list1 = []
    for dataset in datasets:
        sections = load_sectioned_csv('results/'+dataset+'_'+path)
        print(list(sections.keys()))  # -> should list parsed section names

        #model_metrics       = sections["Model-level Metrics"]
        overall_diagnosis_name      = sections["Overall per-Model (diagnosis_name)"]
        #overall_diagnosis_detailed  = sections["Overall per-Model (diagnosis_detailed)"]
        #overall_modality            = sections["Overall per-Model (modality)"]
        #overall_plane               = sections["Overall per-Model (plane)"]
        per_diagnosis_name      = sections["Per-Class Metrics (diagnosis_name)"]
        #per_diagnosis_detailed  = sections["Per-Subclass Metrics (diagnosis_detailed)"]
        #per_modality            = sections["Per-Modality Metrics (modality)"]
        #per_plane               = sections["Per-Plane Metrics (axial_plane vs plane)"]
        
        table1 = overall_diagnosis_name[["model", "accuracy", "precision_macro","recall_macro"]].copy()
        
        
        table2 = per_diagnosis_name[["model", "class","accuracy", "precision","recall"]].copy()
        
        print(table1)
        print(table2)
        
        models = table1["model"].tolist() 
        print(models)
        classes = per_diagnosis_name["class"].drop_duplicates().tolist()
        classes_plot = (per_diagnosis_name["class"].drop_duplicates().str.strip().str.title().tolist())
        # fix acronyms if needed
        acronyms = {"Ms": "MS"}
        classes_plot = [acronyms.get(c, c) for c in classes_plot]
        
        #classes = list(set(table2["class"].tolist()))
        data_list1.append(table1["recall_macro"].tolist())
        #pivot_df = per_diagnosis_name.pivot(index="model",columns="class",values="recall")
        
        # index mappings
        model_to_idx = {m: i for i, m in enumerate(models)}
        class_to_idx = {c: j for j, c in enumerate(classes)}

        # initialize matrix with 0.0 (or np.nan if you prefer)
        data_list2 = np.zeros((len(models), len(classes)), dtype=float)

        # fill matrix using indices
        for _, row in per_diagnosis_name.iterrows():
            i = model_to_idx[row["model"]]
            j = class_to_idx[row["class"]] 
            data_list2[i, j] = row["recall"]
        
        # condition 1: at least one non-zero value in the column
        non_zero_mask = np.any(data_list2 != 0.0, axis=0)

        # condition 2: no NaN values in the column
        no_nan_mask = ~np.any(np.isnan(data_list2), axis=0)

        # combined mask
        keep_col_mask = non_zero_mask & no_nan_mask
        
        # mask: keep columns that have at least one non-zero value
        #keep_col_mask = np.any(data_list2 != 0.0, axis=0)
        #print(keep_col_mask)
        
        # filter data and class labels
        data_list2 = data_list2[:, keep_col_mask]
        classes = [c for c, keep in zip(classes, keep_col_mask) if keep]
        classes_plot = [c for c, keep in zip(classes_plot, keep_col_mask) if keep]
        
        # convert to list of lists
        #data_list2 = np.array(pivot_df.fillna(0.0).values.tolist())
        
        plt.figure(figsize=(len(classes)+4, len(models)/2))
        im = plt.imshow(data_list2, aspect='auto', cmap='Blues')

        # Tick labels
        plt.xticks(np.arange(len(classes)), classes_plot, rotation=45, ha="right")
        plt.yticks(np.arange(len(models)), models)

        # Annotate values with automatically chosen text color
        for i in range(data_list2.shape[0]):
            for j in range(data_list2.shape[1]):
                value = data_list2[i, j]
                text_color = "white" if value > 0.6 else "black"
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

        plt.colorbar(im, label="Recall")
        plt.grid(False)
        #plt.title(dataset + " recall heatmap", fontsize=16)
        plt.tight_layout()
        plt.savefig('results/plots/per_'+dataset+'_recall_heatmap_'+path[:-4]+'.png', dpi=300, bbox_inches="tight")
        plt.show()
        
        
    data=np.array([list(row) for row in zip(*data_list1)])
    # ---------- PLOT heatmap all datasets per model ----------
    plt.figure(figsize=(14, len(models)/2))
    im = plt.imshow(data, aspect='auto', cmap='Blues')

    # Tick labels
    plt.xticks(np.arange(len(datasets)), datasets, rotation=45, ha="right")
    plt.yticks(np.arange(len(models)), models)

    # Annotate values with automatically chosen text color
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if value > 0.6 else "black"
            #text_color = "white" 
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

    plt.colorbar(im, label="Recall")
    plt.grid(False)
    #plt.title("Per-dataset recall Heatmap (Improved Contrast)", fontsize=16)
    plt.tight_layout()
    plt.savefig('results/plots/per_dataset_recall_heatmap_'+path[:-4]+'.png', dpi=300, bbox_inches="tight")
    plt.show()
    

def main():
    
    csv_files = ['subset_1_temp_0_20250823_172529.csv',
                     
                   'subset_4+5+6_temp_0_20250909_122803.csv',
                   
                   'subset_0_25_7_temp_0_20250909_122709.csv',
                   
                   'subset_0_25_7_temp_0_20250909_134613_few_shot.csv']
    
    plot_all('results/all_' + csv_files[0])
    
    plot_heatmaps_zero_few_shot('results/all_' + csv_files[2])
    plot_heatmaps_zero_few_shot('results/all_' + csv_files[3])
    
    for path in csv_files:
        plot_datasets(path)


if __name__ == "__main__":
    main()
