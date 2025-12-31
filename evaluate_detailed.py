#!/usr/bin/env python3
"""
Neuroimaging JSON evaluator with HELM-style metrics
- Primary: macro-F1 and accuracy (class imbalance awareness)
- Per-class: accuracy/precision/recall/F1/support for diagnosis_name
- Per-subclass: accuracy/precision/recall/F1/support for diagnosis_detailed
- Per-modality: accuracy/precision/recall/F1/support (GT metadata.modality vs predicted modality)
- Per-plane: accuracy/precision/recall/F1/support (GT metadata.axial_plane vs predicted plane)
- Calibration: ECE (binning) + Brier score using diagnosis_confidence
- Format robustness: JSON validity rate (% of outputs that parse)
- Efficiency: median latency per image; $ cost per 1k images (from token usage)
- Safety/QA: flag hallucinated/mismatched ICD-10 and missing fields
- Field-wise accuracy for modality/plane/diagnosis_name/diagnosis_detailed
- Confusion matrix per model (diagnosis_name)
- Output: single CSV file (evaluation_results.csv) with header sections then detailed rows

Input JSONL expectations per item (minimal):
{
  "experiment_id": str,
  "input": {
    "image_path": str,
    "model_requested": str,
    "metadata": {
      "class": str,                # ground-truth diagnosis_name
      "subclass": str|null,        # ground-truth diagnosis_detailed
      "modality": str|null,        # ground-truth modality
      "axial_plane": str|null,      # ground-truth plane
      "modality_subtype": str|null # ground-truth modality_subtype
    }
  },
  "output": {
    "status": "success"|"error",
    "parsed_response": {
      "modality": str|null,
      "plane": str|null,
      "diagnosis_name": str|null,
      "diagnosis_detailed": str|null,
      "icd10_code": str|null,
      "severity_score": float|null,
      "diagnosis_confidence": float|null,
      "severity_confidence": float|null,
      "specialized_sequence": str|null
    }
  },
  "api_response": {
    "prompt_tokens": int,
    "completion_tokens": int,
    "latency_ms": float|int  # optional
  }
}
"""
import warnings

# Suppress *all* warnings
warnings.filterwarnings("ignore")

import json
import math
import argparse
import re
from collections import defaultdict, Counter
from typing import List, Callable, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


try:
    from sklearn.metrics import (
        f1_score,
        precision_score,
        accuracy_score,
        balanced_accuracy_score,
        recall_score,
        confusion_matrix,
        brier_score_loss,
        roc_auc_score,
        hamming_loss,
        ConfusionMatrixDisplay,
    )
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.utils import resample
except Exception as e:
    raise SystemExit("scikit-learn is required: pip install scikit-learn\n" + str(e))

from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar


# ----------------------------
# Config / Constants
# ----------------------------
DIAGNOSIS_LABELS = [
    "tumor",
    "stroke",
    "multiple sclerosis",
    "normal",
    "other abnormalities",
]

SUBCLASS_BY_CLASS = {
    "stroke": {"ischemic", "hemorrhagic"},
    "tumor": {
        "glioma",
        "meningioma",
        "pituitary tumor",
        "carcinoma",
        "germinoma",
        "granuloma",
        "medulloblastoma",
        "neurocytoma",
        "papilloma",
        "schwannoma",
        "tuberculoma",
    },
    "multiple sclerosis": {"null", None},
    "other abnormalities": {"null", None},
    "normal": {"null", None},
}

MANDATORY_FIELDS = ["modality", "plane", "diagnosis_name", "diagnosis_detailed", "specialized_sequence"]

# token pricing assumptions per 1M tokens (adjust as needed)
DEFAULT_PRICE_PER_1M_TOKENS_USD = 3.0


MODEL_PRICES = {
    "openai/gpt-5-mini": {"input_per_m": 0.25, "output_per_m": 2, "input_img_per_k": 0},
    "openai/gpt-5-chat": {"input_per_m": 1.25, "output_per_m": 10, "input_img_per_k": 0.0},
    "openai/gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60, "input_img_per_k": 0},
    "openai/gpt-4o": {"input_per_m": 2.50, "output_per_m": 10, "input_img_per_k": 0},
    "openai/gpt-4.1-2025-04-14": {"input_per_m": 5, "output_per_m": 15, "input_img_per_k": 0},

    "meta-llama/llama-4-maverick": {"input_per_m": 0.15, "output_per_m": 0.60, "input_img_per_k": 0.668},
    "meta-llama/llama-3.2-90b-vision-instruct": {"input_per_m": 0.35, "output_per_m": 0.40, "input_img_per_k": 0.506},
    "meta-llama/llama-3.2-11b-vision-instruct": {"input_per_m": 0.049, "output_per_m": 0.049, "input_img_per_k": 0.079},

    "medgemma_4b": {"input_per_m": 0.123, "output_per_m": 0.456, "input_img_per_k": 0.0},  # TODO: placeholder. Update these prices
    "medgemma_27b": {"input_per_m": 0.123, "output_per_m": 0.456, "input_img_per_k": 0.0},  # TODO: placeholder. Update these prices
    #"google/medgemma_4b_noquant": {"input_per_m": 0.123, "output_per_m": 0.456, "input_img_per_k": 0.0},

    "google/gemma-3-27b-it": {"input_per_m": 0.065, "output_per_m": 0.261, "input_img_per_k": 0.0},
    "google/gemini-2.5-pro": {"input_per_m": 1.25, "output_per_m": 10, "input_img_per_k": 5.16},
    "google/gemini-2.5-flash": {"input_per_m": 0.30, "output_per_m": 2.50, "input_img_per_k": 1.238},
    "google/gemini-2.0-flash-001": {"input_per_m": 0.10, "output_per_m": 0.40, "input_img_per_k": 0.026},

    "bedrock/us.amazon.nova-pro-v1:0": {"input_per_m": 0.80, "output_per_m": 3.20, "input_img_per_k": 1.20},  # TODO: placeholder. Update these prices
    "bedrock/amazon.nova-lite-v1:0": {"input_per_m": 0.06, "output_per_m": 0.24, "input_img_per_k": 0.09},  # TODO: placeholder. Update these prices

    "anthropic/claude-sonnet-4.5": {"input_per_m": 3, "output_per_m": 15, "input_img_per_k": 0.0},  # TODO: placeholder. Update these prices
    "x-ai/grok-4": {"input_per_m": 3, "output_per_m": 15, "input_img_per_k": 0.0},  # TODO: placeholder. Update these prices
    "qwen25vl_32b": {"input_per_m": 0.05, "output_per_m": 0.22, "input_img_per_k": 0.0},  # TODO: placeholder. Update these prices

}



# ----------------------------
# I/O
# ----------------------------
def load_jsonl(file_path: str, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                
            except json.JSONDecodeError:
                # Keep a placeholder for invalid lines
                item = {"output": {"status": "parse_error"}}
                
            if dataset is not None:
                metadata = item.get("input", {}).get("metadata", {})
                if metadata.get("dataset") != dataset:
                    continue  # skip if dataset doesn't match
 
            data.append(item)
    return data

# ----------------------------
# Helpers
# ----------------------------
def normalize_str(x: Any) -> str:
    return str(x).strip().lower().replace("_", " ") if x is not None else 'undetermined'

def normalize_modality_label(x: Any) -> str:
    """Normalize modality label for metric computation:
       - any value containing 'mri' -> 'mri'
       - else lowercased trimmed string
    """
    s = normalize_str(x)
    if "mr" in s:
        return "mri"
    # Whether to handle it like this???
    if s == 'null' or s == 'nan' or s is None or s == '' or s == 'None':
        return 'undetermined'
    
    return s

def normalize_modality_subtype_label(x: Any) -> str:
    """Normalize modality subtype label for metric computation:
       - any value containing 'mri' -> 'mri'
       - else lowercased trimmed string
    """
    s = normalize_str(x)
    
    if s == 't1+c':
        return 't1c+'
    if s == 't2-weighted':
        return 't2'
    if s == 'non-contrast' or s == 't1-weighted':
        return 't1'
    if 'post-contrast' in s or 'with contrast' in s:
        return 't1c+'
        
    # Whether to handle it like this???
    if s == 'null' or s == 'nan' or s is None or s == '' or s == 'None' or s == 'NaN' or s == 'n/a' or s == 'none' or s == 'not applicable' or s == 'brain':
        return 'undetermined'
    
    return s

def normalize_plane_label(x: Any) -> str:
    """Normalize plane label for metric computation:

    """
    s = normalize_str(x)
    
    if s == 's':
        return 'sagittal'
    
    # Whether to handle it like this???
    if s == 'null' or s == 'nan' or s is None or s == '' or s == 'None':
        return 'undetermined'   
    return s

def normalize_class_label(x: Any) -> str:
    """Normalize class label for metric computation:
       - any value containing 'tumor' -> 'tumor'
       - else lowercased trimmed string
    """
    s = normalize_str(x)
    
    if 'atrophy' in s:
        return 'other abnormalities'
    
    if any(tumor in s for tumor in SUBCLASS_BY_CLASS['tumor']):
        return 'tumor'
    if any(stroke in s for stroke in SUBCLASS_BY_CLASS['stroke']):
        return 'stroke'
    # Whether to handle it like this???
    if s == 'null' or s == 'nan' or s is None or s == '' or s == 'None':
        return 'undetermined'
    
    return s

def normalize_subclass_label(x: Any) -> str:
    """Normalize subclass label for metric computation

    """
    s = normalize_str(x)
    
    if s == 'papilloma':
        return 'papiloma'
    if 'neurocytoma' in s:
        return 'neurocitoma'
    if s == 'medulloblastoma':
        return 'meduloblastoma'
    if 'ischemic' in s:
        return 'ischemic'
    if s == 'glioblastoma':
        return 'glioma'
    if 'astrocytoma' in s:
        return 'glioma'
    if s == 'ependymoma':
        return 'glioma'
    
    if 'metastatic carcinoma' in s and 'meningioma' in s:
        return 'meningioma'
    if 'metastatic carcinoma' in s and'carcioma' in s:
        return 'carcioma'
    if 'metastatic carcinoma' in s and 'granuloma' in s:
        return 'granuloma'
    if 'metastatic carcinoma' in s and 'glioma' in s:
        return 'glioma'
    if 'metastatic tumor' in s and 'glioma' in s:
        return 'glioma'
    
    if 'abscess' in s:
        return 'undetermined'
    if 'craniopharyngioma' in s:
        return 'undetermined'
    

    if 'atrophy' in s:
        return 'undetermined'
    if s == 'calcified pineal gland':
        return 'undetermined'
    if s == 'central nervous system tumor':
        return 'undetermined'
    if s == 'gliomatosis cerebri':
        return 'glioma'
    if s == 'gliosis':
        return 'undetermined'
    if 'hemangioblastoma' in s:
        return 'undetermined'
    if s == 'other':
        return 'undetermined'
    if 'pituitary' in s:
        return 'pituitary tumor'
    
    if s == 'cavernoma':
        return 'undetermined'

    if 'metastases' in s or 'metastasis' in s or 'metastatic' in s:
        return 'undetermined'
    
    if s == 'normal' or s == 'multiple sclerosis' or s == 'colloid cyst':
        return 'undetermined'   # or 'other'
    
    # Whether to handle it like this???
    if s == 'null' or s == 'nan' or s is None or s == '' or s == 'None':
        return 'undetermined'
    
    return s



def evaluate_field(ground_truth: Any, predicted: Any, field_name: str) -> bool:
    if field_name == 'plane':
        gt = normalize_plane_label(ground_truth)
        pr = normalize_plane_label(predicted)
    
    if field_name == 'modality':
        gt = normalize_modality_label(ground_truth)
        pr = normalize_modality_label(predicted)
        
    if field_name == 'diagnosis_name':
        gt = normalize_class_label(ground_truth)
        pr = normalize_class_label(predicted)
    
    if field_name == 'diagnosis_detailed':
        gt = normalize_subclass_label(ground_truth)
        pr = normalize_subclass_label(predicted)
        
    if field_name == 'specialized_sequence':
        gt = normalize_modality_subtype_label(ground_truth)
        pr = normalize_modality_subtype_label(predicted)
    
    #if ground_truth is None or predicted is None:
    #    return False

    return gt == pr

def pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0

def ece_binned(y_true: List[str], y_pred: List[str], conf: List[float], bins: int = 10) -> float:
    if not y_true:
        return float("nan")
    correctness = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)])
    probs = np.clip(np.array([c if c is not None else 0.0 for c in conf], dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece_val = 0.0
    N = len(probs)
    for i in range(bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1]) if i > 0 else (probs >= edges[i]) & (probs <= edges[i + 1])
        if np.any(m):
            acc = float(np.mean(correctness[m]))
            conf_bin = float(np.mean(probs[m]))
            ece_val += (np.sum(m) / N) * abs(acc - conf_bin)
    return float(ece_val)

def per_label_accuracy(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[float]:
    """Per-label accuracy = (TP + TN) / N for each label, using one-vs-rest."""
    N = len(y_true)
    if N == 0:
        return []
    accs = []
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    for lab in labels:
        tp = np.sum((y_t == lab) & (y_p == lab))
        fn = np.sum((y_t == lab) & (y_p != lab))
        fp = np.sum((y_t != lab) & (y_p == lab))
        tn = N - (tp + fn + fp)
        
        #tn = np.sum((y_t != lab) & (y_p != lab))
        #if lab == 'other abnormalities':
            #print(tp)
            #print(tn)
            #print(N)
            #print(float((tp + tn) / N))
        accs.append(float((tp + tn) / N))
    return accs




# 95% z-value
Z = norm.ppf(0.975)

def accuracy_score_ci(y_true: List[str], y_pred: List[str], alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """Return accuracy and 95% CI using normal approximation."""
    acc = accuracy_score(y_true, y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0, (0.0, 0.0)
    se = np.sqrt(acc * (1 - acc) / n)
    ci = (max(0, acc - Z * se), min(1, acc + Z * se))
    return acc, ci

def per_label_accuracy_ci(y_true: List[str], y_pred: List[str], labels: List[str], alpha: float = 0.05) -> List[Tuple[float, Tuple[float, float]]]:
    """Per-label accuracy and 95% CI using one-vs-rest normal approximation."""
    N = len(y_true)
    if N == 0:
        return []
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    results = []
    for lab in labels:
        tp = np.sum((y_t == lab) & (y_p == lab))
        fn = np.sum((y_t == lab) & (y_p != lab))
        fp = np.sum((y_t != lab) & (y_p == lab))
        tn = N - (tp + fn + fp)
        
        acc = (tp + tn) / N
        se = np.sqrt(acc * (1 - acc) / N)
        ci = (max(0, acc - Z * se), min(1, acc + Z * se))
        results.append((float(acc), ci))
    return results

def balanced_accuracy_score_ci(
    y_true,
    y_pred,
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Balanced accuracy with bootstrap 95% CI (percentile method).

    Returns
    -------
    bacc : float
        Balanced accuracy on the full dataset.
    ci : (float, float)
        (lower, upper) confidence interval.
    """
    n = len(y_true)
    if n == 0:
        return 0.0, (0.0, 0.0)

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    base_bacc = balanced_accuracy_score(y_t, y_p)

    rng = np.random.RandomState(random_state)
    boot_scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = y_t[idx]
        b_y_p = y_p[idx]
        try:
            s = balanced_accuracy_score(b_y_t, b_y_p)
            boot_scores.append(s)
        except ValueError:
            # In case some pathological resample fails
            continue

    if not boot_scores:
        # Fallback: no valid bootstrap samples
        return float(base_bacc), (float(base_bacc), float(base_bacc))

    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return float(base_bacc), (lower, upper)


def per_label_balanced_accuracy(
    y_true: List[str], 
    y_pred: List[str], 
    labels: List[str]
) -> List[float]:
    """Per-label balanced accuracy = (Recall + Specificity) / 2 for each label, using one-vs-rest."""
    N = len(y_true)
    if N == 0:
        return []
    baccs = []
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    for lab in labels:
        tp = np.sum((y_t == lab) & (y_p == lab))
        fn = np.sum((y_t == lab) & (y_p != lab))
        fp = np.sum((y_t != lab) & (y_p == lab))
        tn = N - (tp + fn + fp)

        # recall (sensitivity)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        bacc = (recall + specificity) / 2.0
        baccs.append(float(bacc))
    return baccs

def per_label_balanced_accuracy_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None
) -> List[Tuple[float, Tuple[float, float]]]:
    """
    Per-label balanced accuracy with bootstrap 95% CI.

    Returns
    -------
    result : List[(float, (float, float))]
        For each label in `labels`, returns:
        (per_label_bacc, (lower_ci, upper_ci))
    """
    n = len(y_true)
    if n == 0:
        return []

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    # Base scores on full data
    base_baccs = per_label_balanced_accuracy(y_true, y_pred, labels)

    rng = np.random.RandomState(random_state)
    # One list of bootstrap scores per label
    boot_scores_per_label = [[] for _ in labels]

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = list(y_t[idx])
        b_y_p = list(y_p[idx])

        baccs = per_label_balanced_accuracy(b_y_t, b_y_p, labels)
        # Length is guaranteed == len(labels)
        for j, s in enumerate(baccs):
            boot_scores_per_label[j].append(s)

    results: List[Tuple[float, Tuple[float, float]]] = []
    for base_bacc, scores in zip(base_baccs, boot_scores_per_label):
        if scores:
            lower = float(np.percentile(scores, 100 * alpha / 2))
            upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        else:
            # Fallback if no bootstrap values for that label
            lower = upper = float(base_bacc)
        results.append((float(base_bacc), (lower, upper)))

    return results



def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate accuracy percentage."""
    return (correct / total * 100) if total > 0 else 0.0


def calculate_accuracy_ci(correct: int, total: int, alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """Accuracy % with 95% CI using normal approximation."""
    if total == 0:
        return 0.0, (0.0, 0.0)
    acc = correct / total
    se = np.sqrt(acc * (1 - acc) / total)
    ci = (max(0, acc - Z * se), min(1, acc + Z * se))
    return acc * 100, (ci[0] * 100, ci[1] * 100)

def f1_score_ci(y_true: List[str], y_pred: List[str], labels: List[str], average='macro', n_bootstraps=100, alpha: float = 0.05, random_state=None, zero_division=0) -> Tuple[float, Tuple[float, float]]:
    """Return F1 score and 95% CI using bootstrap resampling."""
    rng = np.random.RandomState(random_state)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average=average, zero_division=0)
    boot_scores = []
    n = len(y_true)
    if n == 0:
        return 0.0, (0.0, 0.0)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t, y_p = np.array(y_true)[idx], np.array(y_pred)[idx]
        try:
            boot_scores.append(f1_score(y_true=y_t, y_pred=y_p, labels=labels, average=average, zero_division=0))
        except ValueError:
            continue  # skip cases with missing classes
    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return f1, (lower, upper)

def per_label_f1_score_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    average: str = 'macro',          # kept for API symmetry; not used internally
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None,
    zero_division: int = 0
) -> List[Tuple[float, Tuple[float, float]]]:
    """
    Per-label F1 score with bootstrap 95% CI

    Returns
    -------
    result : List[(float, (float, float))]
        For each label in `labels`, returns:
        (per_label_f1, (lower_ci, upper_ci))
    """
    
    
    n = len(y_true)
    if n == 0:
        return []

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    # Base scores on full data
    base_baccs = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels = labels,
        zero_division=zero_division
    )
    
    rng = np.random.RandomState(random_state)
    # One list of bootstrap scores per label
    boot_scores_per_label = [[] for _ in labels]

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = list(y_t[idx])
        b_y_p = list(y_p[idx])

        #baccs = per_label_balanced_accuracy(b_y_t, b_y_p, labels)
        baccs = f1_score(
            y_true=b_y_t,
            y_pred=b_y_p,
            average=None,
            labels = labels,
            zero_division=zero_division
        )
        
        # Length is guaranteed == len(labels)
        for j, s in enumerate(baccs):
            boot_scores_per_label[j].append(s)

    results: List[Tuple[float, Tuple[float, float]]] = []
    for base_bacc, scores in zip(base_baccs, boot_scores_per_label):
        if scores:
            lower = float(np.percentile(scores, 100 * alpha / 2))
            upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        else:
            # Fallback if no bootstrap values for that label
            lower = upper = float(base_bacc)
        results.append((float(base_bacc), (lower, upper)))

    return results



def precision_score_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    average = None,
    zero_division: int = 0,
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Precision score with bootstrap 95% CI (percentile method).

    Returns
    -------
    precision : float
        Precision on the full dataset.
    ci : (float, float)
        (lower, upper) confidence interval.
    """
    n = len(y_true)
    if n == 0:
        return 0.0, (0.0, 0.0)

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    base_prec = precision_score(
        y_t, y_p, labels=labels, average=average, zero_division=zero_division
    )

    rng = np.random.RandomState(random_state)
    boot_scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = y_t[idx]
        b_y_p = y_p[idx]
        try:
            s = precision_score(
                b_y_t, b_y_p, labels=labels, average=average, zero_division=zero_division
            )
            boot_scores.append(s)
        except ValueError:
            # e.g., no positive predictions / labels in this resample
            continue

    if not boot_scores:
        # Fallback: no valid bootstrap samples
        return float(base_prec), (float(base_prec), float(base_prec))

    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return base_prec, (lower, upper)


def recall_score_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    average = None,
    zero_division: int = 0,
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Recall score with bootstrap 95% CI (percentile method).

    Returns
    -------
    recall : float
        Recall on the full dataset.
    ci : (float, float)
        (lower, upper) confidence interval.
    """
    n = len(y_true)
    if n == 0:
        return 0.0, (0.0, 0.0)

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    base_rec = recall_score(
        y_t, y_p, labels=labels, average=average, zero_division=zero_division
    )

    rng = np.random.RandomState(random_state)
    boot_scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = y_t[idx]
        b_y_p = y_p[idx]
        try:
            s = recall_score(
                b_y_t, b_y_p, labels=labels, average=average, zero_division=zero_division
            )
            boot_scores.append(s)
        except ValueError:
            # e.g., no samples for some label in this resample
            continue

    if not boot_scores:
        # Fallback: no valid bootstrap samples
        return base_rec, (base_rec, base_rec)

    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return base_rec, (lower, upper)

def per_label_precision_score_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    average: str = 'macro',          # kept for API symmetry; not used internally
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None,
    zero_division: int = 0
) -> List[Tuple[float, Tuple[float, float]]]:
    """
    Per-label precision score with bootstrap 95% CI.

    Returns
    -------
    result : List[(float, (float, float))]
        For each label in `labels`, returns:
        (per_label_precision, (lower_ci, upper_ci))
    """
    n = len(y_true)
    if n == 0:
        return []

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    # Base per-label precision scores on full data
    base_scores = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        zero_division=zero_division
    )

    rng = np.random.RandomState(random_state)
    # One list of bootstrap scores per label
    boot_scores_per_label = [[] for _ in labels]

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = list(y_t[idx])
        b_y_p = list(y_p[idx])

        scores = precision_score(
            y_true=b_y_t,
            y_pred=b_y_p,
            average=None,
            labels=labels,
            zero_division=zero_division
        )

        # Length is guaranteed == len(labels)
        for j, s in enumerate(scores):
            boot_scores_per_label[j].append(s)

    results: List[Tuple[float, Tuple[float, float]]] = []
    for base_score, scores in zip(base_scores, boot_scores_per_label):
        if scores:
            lower = float(np.percentile(scores, 100 * alpha / 2))
            upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        else:
            # Fallback if no bootstrap values for that label
            lower = upper = float(base_score)
        results.append((float(base_score), (lower, upper)))

    return results


def per_label_recall_score_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    average: str = 'macro',          # kept for API symmetry; not used internally
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None,
    zero_division: int = 0
) -> List[Tuple[float, Tuple[float, float]]]:
    """
    Per-label recall score with bootstrap 95% CI.

    Returns
    -------
    result : List[(float, (float, float))]
        For each label in `labels`, returns:
        (per_label_recall, (lower_ci, upper_ci))
    """
    n = len(y_true)
    if n == 0:
        return []

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    # Base per-label recall scores on full data
    base_scores = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=labels,
        zero_division=zero_division
    )

    rng = np.random.RandomState(random_state)
    # One list of bootstrap scores per label
    boot_scores_per_label = [[] for _ in labels]

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = list(y_t[idx])
        b_y_p = list(y_p[idx])

        scores = recall_score(
            y_true=b_y_t,
            y_pred=b_y_p,
            average=None,
            labels=labels,
            zero_division=zero_division
        )

        # Length is guaranteed == len(labels)
        for j, s in enumerate(scores):
            boot_scores_per_label[j].append(s)

    results: List[Tuple[float, Tuple[float, float]]] = []
    for base_score, scores in zip(base_scores, boot_scores_per_label):
        if scores:
            lower = float(np.percentile(scores, 100 * alpha / 2))
            upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        else:
            # Fallback if no bootstrap values for that label
            lower = upper = float(base_score)
        results.append((float(base_score), (lower, upper)))

    return results


def per_label_uncertainty(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    uncertain_values={"None", "none", None, "null", "nan", "undetermined", "", "NaN", "n/a"}
) -> Dict[str, List[float] | float]:
    """
    Compute per-label and global uncertainty percentages.
    
    - Per-label uncertainty = (# of uncertain predictions for that label) / (# of samples of that label).
    - Global uncertainty = (# of uncertain predictions overall) / (total samples).
    
    Args:
        y_true: list of ground truth labels
        y_pred: list of predicted labels
        labels: list of labels to evaluate
        uncertain_values: set of strings considered as "uncertain"/"refused"
    
    Returns:
        {
            "per_label": [uncertainty% per label],
            "global": overall_uncertainty%
        }
    """
    if len(y_true) == 0:
        return {"per_label": [], "global": 0.0}
    
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    per_label_results = []
    for lab in labels:
        mask = (y_t == lab)
        total = np.sum(mask)
        if total == 0:
            per_label_results.append(0.0)
            continue
        uncertain = np.sum([str(val).strip().lower() in uncertain_values for val in y_p[mask]])
        per_label_results.append(100.0 * uncertain / total)
    
    # Global uncertainty
    total_all = len(y_true)
    uncertain_all = np.sum([str(val).strip().lower() in uncertain_values for val in y_p])
    global_uncertainty = 100.0 * uncertain_all / total_all if total_all > 0 else 0.0

    return {"per_label": per_label_results, "global": global_uncertainty}



def f1_with_abstention(
    y_true: List[str], y_pred: List[str], labels: List[str], abstain_label: str = "undetermined", average = None, zero_division = 0) -> float:
    """
    F1 with abstention with 95% CI: each abstained instance (pred == abstain_label)
    is counted as a false negative for the correct class, but NOT as a
    false positive for any class.

    Parameters
    ----------
    y_true : List[str]
        True labels.
    y_pred : List[str]
        Predicted labels (may contain `abstain_label`).
    labels : List[str]
        List of *real* labels to evaluate F1 over. (Usually excludes `abstain_label`.)
    abstain_label : str, default="none"
        Label used for abstention in `y_pred`.
    average : {"macro", "weighted"}
        Averaging strategy for F1.

    Returns
    -------
    float
        F1 score with abstention handled as described.
    """
    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)

    # Real labels to be scored (ensure abstain_label is not in them)
    real_labels = [lab for lab in labels if lab != abstain_label]

    # Replace abstentions in predictions with dummy label:
    # - For the true class, this is a FN (because prediction != true label)
    # - Dummy is not in `labels`, so it does not generate FP for any class
    y_pred_cleaned = y_pred_arr.copy()
    y_pred_cleaned[y_pred_cleaned == abstain_label] = '__dummy__'

    # Compute F1; `labels=real_labels` ensures dummy is ignored as a class
    f1 = f1_score(y_true_arr, y_pred_cleaned, labels=real_labels, average=average, zero_division=0)
    return f1

def f1_with_abstention_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    abstain_label: str = "undetermined",
    average = None,
    zero_division = 0,
    n_bootstraps=100, 
    alpha: float = 0.05, 
    random_state=None
    ) -> Tuple[float, Tuple[float, float]]:

    """
    F1 with abstention: each abstained instance (pred == abstain_label)
    is counted as a false negative for the correct class, but NOT as a
    false positive for any class.

    Parameters
    ----------
    y_true : List[str]
        True labels.
    y_pred : List[str]
        Predicted labels (may contain `abstain_label`).
    labels : List[str]
        List of *real* labels to evaluate F1 over. (Usually excludes `abstain_label`.)
    abstain_label : str, default="none"
        Label used for abstention in `y_pred`.
    average : {"macro", "weighted"}
        Averaging strategy for F1.

    Returns
    -------
    float
        F1 score with abstention handled as described.
    """
    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)

    # Real labels to be scored (ensure abstain_label is not in them)
    real_labels = [lab for lab in labels if lab != abstain_label]

    # Replace abstentions in predictions with dummy label:
    # - For the true class, this is a FN (because prediction != true label)
    # - Dummy is not in `labels`, so it does not generate FP for any class
    y_pred_cleaned = y_pred_arr.copy()
    y_pred_cleaned[y_pred_cleaned == abstain_label] = '__dummy__'

    # Compute F1; `labels=real_labels` ensures dummy is ignored as a class
    f1 = f1_score(y_true_arr, y_pred_cleaned, labels=real_labels, average=average, zero_division=0)

    rng = np.random.RandomState(random_state)

    boot_scores = []
    n = len(y_true)
    if n == 0:
        return 0.0, (0.0, 0.0)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t, y_p = y_true_arr[idx], y_pred_cleaned[idx]
        try:
            boot_scores.append(f1_score(y_true=y_t, y_pred=y_p, labels=real_labels, average=average, zero_division=0))
        except ValueError:
            continue  # skip cases with missing classes
    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return f1, (lower, upper)


def per_label_f1_with_abstention_ci(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    average: str = 'macro',          # kept for API symmetry; not used internally
    abstain_label: str = "undetermined",
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    random_state: int = None,
    zero_division: int = 0
) -> List[Tuple[float, Tuple[float, float]]]:
    """
    Per-label F1 with abstention and bootstrap 95% CI.

    Each abstained instance (pred == abstain_label) is counted as a FN
    for the correct class but NOT as a FP for any class.

    Parameters
    ----------
    y_true : List[str]
        True labels.
    y_pred : List[str]
        Predicted labels (may contain `abstain_label`).
    labels : List[str]
        List of *real* labels to evaluate F1 over (usually excludes `abstain_label`).
    abstain_label : str
        Label used for abstention in `y_pred`.
    n_bootstraps : int
        Number of bootstrap resamples.
    alpha : float
        1 - confidence level (e.g. 0.05 for 95% CI).
    random_state : int or None
        Seed for RNG.
    zero_division : int
        Passed to sklearn.f1_score.

    Returns
    -------
    List[(float, (float, float))]
        For each label in `labels`, returns:
        (per_label_f1_with_abstention, (lower_ci, upper_ci))
    """
    n = len(y_true)
    if n == 0:
        return []

    y_t = np.array(y_true, dtype=object)
    y_p = np.array(y_pred, dtype=object)

    # Real labels (if abstain_label is not in `labels`, this is just `labels`)
    real_labels = [lab for lab in labels if lab != abstain_label]

    # ----- Base per-label F1 on full data -----
    y_pred_cleaned = y_p.copy()
    y_pred_cleaned[y_pred_cleaned == abstain_label] = '__dummy__'

    base_f1s = f1_score(
        y_true=y_t,
        y_pred=y_pred_cleaned,
        average=None,
        labels=real_labels,
        zero_division=zero_division
    )  # shape: (len(real_labels),)

    rng = np.random.RandomState(random_state)
    boot_scores_per_label = [[] for _ in real_labels]

    # ----- Bootstrap -----
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y_t = y_t[idx]
        b_y_p = y_p[idx]

        # Apply abstention logic in this bootstrap sample
        b_y_p_cleaned = b_y_p.copy()
        b_y_p_cleaned[b_y_p_cleaned == abstain_label] = '__dummy__'

        # Per-label F1 for this resample
        f1s = f1_score(
            y_true=b_y_t,
            y_pred=b_y_p_cleaned,
            average=None,
            labels=real_labels,
            zero_division=zero_division
        )

        for j, s in enumerate(f1s):
            boot_scores_per_label[j].append(s)

    # ----- Build CIs -----
    results: List[Tuple[float, Tuple[float, float]]] = []
    for base_f1, scores in zip(base_f1s, boot_scores_per_label):
        if scores:
            lower = float(np.percentile(scores, 100 * alpha / 2))
            upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        else:
            # Fallback if no bootstrap values for that label
            lower = upper = float(base_f1)
        results.append((float(base_f1), (lower, upper)))

    print(results)
    return results




# -----------------
# STEP 1: Collect aligned predictions
# -----------------

def collect_model_predictions_intersection(
    data: List[Dict[str, Any]], normalize_class_label=lambda x: str(x).strip()
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build y_true / y_pred lists for each model, restricted to the intersection
    of experiment_ids where ALL models have valid predictions.
    """
    per_model_valid: Dict[str, Set[str]] = defaultdict(set)
    expid_to_gt: Dict[str, str] = {}
    expid_model_pred: Dict[Tuple[str, str], str] = {}

    invalid_vals = {"", "null", "nan", None}

    for item in data:
        status = item.get("output", {}).get("status")
        if status != "success":
            continue

        model = item.get("input", {}).get("model_requested") or "UNKNOWN_MODEL"
        print(model)
        image_path = item.get("input", {}).get("image_path")
        
        
        parts = image_path.split('/')
        
        if 'medgemma' not in model:
            expid = "_".join(parts[4:]) if len(parts) > 3 else ""
        else:
            expid = image_path
        
        
        print(expid)
        
        metadata = item.get("input", {}).get("metadata", {})
        parsed = item.get("output", {}).get("parsed_response", {}) or {}

        gt_cls = metadata.get("class")
        pr_cls = parsed.get("diagnosis_name")
        
        if gt_cls in invalid_vals or str(gt_cls).strip().lower() in invalid_vals:
            continue
        if pr_cls in invalid_vals or str(pr_cls).strip().lower() in invalid_vals:
            continue

        expid_to_gt[expid] = normalize_class_label(gt_cls)
        expid_model_pred[(expid, model)] = normalize_class_label(pr_cls)
        per_model_valid[model].add(expid)

    if not per_model_valid:
        return {}

    # Common set of samples across all models
    common_experiments = set.intersection(*per_model_valid.values())
    #print(common_experiments)

    by_model_true: Dict[str, List[str]] = defaultdict(list)
    by_model_pred: Dict[str, List[str]] = defaultdict(list)

    for model in per_model_valid.keys():
        for expid in sorted(common_experiments):
            by_model_true[model].append(expid_to_gt[expid])
            by_model_pred[model].append(expid_model_pred[(expid, model)])

    results = {
        model: {"y_true": by_model_true[model], "y_pred": by_model_pred[model]}
        for model in by_model_true.keys()
    }
    return results


# -----------------
# STEP 2: Bootstrap CIs
# -----------------

def bootstrap_ci_metric(y_true, y_pred, metric_fn, n_bootstrap=2000, alpha=0.05):
    n = len(y_true)
    stats = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        stats.append(metric_fn([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(np.mean(stats)), (float(lower), float(upper))


# -----------------
# STEP 3: Pairwise comparisons
# -----------------

def paired_bootstrap_diff(y_true, y_pred1, y_pred2, metric_fn, n_bootstrap=2000, alpha=0.05):
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        m1 = metric_fn([y_true[i] for i in idx], [y_pred1[i] for i in idx])
        m2 = metric_fn([y_true[i] for i in idx], [y_pred2[i] for i in idx])
        diffs.append(m1 - m2)
    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), (float(lower), float(upper))


def mcnemar_test(y_true, y_pred1, y_pred2):
    b01 = sum((y_pred1[i] == y_true[i]) and (y_pred2[i] != y_true[i]) for i in range(len(y_true)))
    b10 = sum((y_pred1[i] != y_true[i]) and (y_pred2[i] == y_true[i]) for i in range(len(y_true)))
    table = [[0, b01], [b10, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue


# -----------------
# STEP 4: Main evaluation framework
# -----------------

def evaluate_models(data: List[Dict[str, Any]], metrics=("accuracy", "macro-f1")):
    model_preds = collect_model_predictions_intersection(data)
    model_names = list(model_preds.keys())
    print(model_names)
    # Per-model metrics with CIs
    per_model_results = {}
    for model, preds in model_preds.items():
        y_true, y_pred = preds["y_true"], preds["y_pred"]
        model_results = {}
        if "accuracy" in metrics:
            mean_acc, ci_acc = bootstrap_ci_metric(y_true, y_pred, balanced_accuracy_score)
            model_results["accuracy"] = {"mean": mean_acc, "ci": ci_acc}
        if "macro-f1" in metrics:
            mean_f1, ci_f1 = bootstrap_ci_metric(y_true, y_pred,
                                                 lambda yt, yp: f1_score(yt, yp, average="macro"))
            model_results["macro-f1"] = {"mean": mean_f1, "ci": ci_f1}
        per_model_results[model] = model_results

    # Pairwise comparisons
    pairwise_results = defaultdict(dict)
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                continue
            preds1 = model_preds[m1]
            preds2 = model_preds[m2]
            y_true = preds1["y_true"]
            assert y_true == preds2["y_true"], "Ground truth mismatch!"

            entry = {}
            if "accuracy" in metrics:
                diff, ci = paired_bootstrap_diff(y_true, preds1["y_pred"], preds2["y_pred"], balanced_accuracy_score)
                pval = mcnemar_test(y_true, preds1["y_pred"], preds2["y_pred"])
                entry["accuracy"] = {"diff": diff, "ci": ci, "pval": pval}
            if "macro-f1" in metrics:
                diff, ci = paired_bootstrap_diff(y_true, preds1["y_pred"], preds2["y_pred"],
                                                 lambda yt, yp: f1_score(yt, yp, average="macro"))
                entry["macro-f1"] = {"diff": diff, "ci": ci, "pval": None}  # no McNemar for F1
            pairwise_results[m1][m2] = entry

    return {"per_model": per_model_results, "pairwise": dict(pairwise_results)}

def summarize_pairwise_matrix(results, metric="accuracy", alpha=0.05, save_path=None) -> pd.DataFrame:
    """
    Build a matrix summary of pairwise comparisons for one metric.
    Each cell: Δ (m1 - m2), CI, p-value, winner.
    """
    pairwise = results["pairwise"]
    models = sorted(set(list(pairwise.keys()) + [m for d in pairwise.values() for m in d.keys()]))
    mat = pd.DataFrame(index=models, columns=models, dtype=object)

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if m1 == m2:
                mat.loc[m1, m2] = "—"
            elif j > i:  # only upper triangle filled
                comp = pairwise.get(m1, {}).get(m2, {})
                if metric not in comp:
                    mat.loc[m1, m2] = ""
                    continue
                entry = comp[metric]
                diff, ci, pval = entry["diff"], entry["ci"], entry.get("pval", None)

                # Winner logic
                if ci[0] > 0:  # CI strictly above 0
                    winner = m1
                elif ci[1] < 0:  # CI strictly below 0
                    winner = m2
                else:
                    winner = "Tie"

                mat.loc[m1, m2] = f"dif={diff:.3f}, CI=({ci[0]:.3f},{ci[1]:.3f}), " \
                                  f"p={pval if pval is not None else 'NA'}, better={winner}"
            elif j == i:
                mat.loc[m1, m2] = ""
            else:
                mat.loc[m1, m2] = ""  # lower triangle empty for clarity

    # Save to CSV if requested
    if save_path is not None:
        mat.to_csv(save_path, index=True)    


    return mat

# ----------------------------
# Core evaluation
# ----------------------------
def evaluate(data: List[Dict[str, Any]] , out_path: str) -> Dict[str, Any]:
    field_mappings = {
        "diagnosis_name": "class",
        "diagnosis_detailed": "subclass",
        "modality": "modality",
        "plane": "axial_plane",
        "specialized_sequence": "modality_subtype"
        
    }

    results: Dict[str, Any] = {
        "total_items": len(data),
        "detailed": [],
        "field_results": {},
        "overall_accuracy": 0.0,
        "per_model": {},
        "token_stats": {},
    }

    # counters
    field_counters = {k: {"correct": 0, "total": 0} for k in field_mappings}
    overall_all_fields_correct = 0

    # aggregation structures per model
    by_model_true: Dict[str, List[str]] = defaultdict(list)            # class
    by_model_pred: Dict[str, List[str]] = defaultdict(list)
    by_model_conf: Dict[str, List[float]] = defaultdict(list)
    
    by_model_true_unfiltered: Dict[str, List[str]] = defaultdict(list)            # class
    by_model_pred_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_unfiltered: Dict[str, List[float]] = defaultdict(list)
    
    by_model_latency_ms: Dict[str, List[float]] = defaultdict(list)
    by_model_json_valid: Counter = Counter()
    by_model_json_total: Counter = Counter()
    by_model_tokens: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    # subclass
    by_model_true_sub: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_sub: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_sub: Dict[str, List[float]] = defaultdict(list)
    
    by_model_true_sub_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_sub_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_sub_unfiltered: Dict[str, List[float]] = defaultdict(list)

    # modality
    by_model_true_mod: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_mod: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_mod: Dict[str, List[float]] = defaultdict(list)
    
    by_model_true_mod_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_mod_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_mod_unfiltered: Dict[str, List[float]] = defaultdict(list)
    
    # modality subtype
    by_model_true_mod_sub: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_mod_sub: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_mod_sub: Dict[str, List[float]] = defaultdict(list)
    
    by_model_true_mod_sub_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_mod_sub_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_mod_sub_unfiltered: Dict[str, List[float]] = defaultdict(list)

    # plane
    by_model_true_plane: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_plane: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_plane: Dict[str, List[float]] = defaultdict(list)

    by_model_true_plane_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_pred_plane_unfiltered: Dict[str, List[str]] = defaultdict(list)
    by_model_conf_plane_unfiltered: Dict[str, List[float]] = defaultdict(list)    

    safety_flags_rows: List[Dict[str, Any]] = []

    for item in data:
        status = item.get("output", {}).get("status")
        if status != "success":
            # still count format robustness per model if we can
            model_nm = item.get("input", {}).get("model_requested")
            if model_nm:
                by_model_json_total[model_nm] += 1
            continue

        model = item.get("input", {}).get("model_requested") or "UNKNOWN_MODEL"
        metadata = item.get("input", {}).get("metadata", {})
        parsed = item.get("output", {}).get("parsed_response", {}) or {}

        # format robustness
        by_model_json_total[model] += 1
        try:
            json.dumps(parsed)
            by_model_json_valid[model] += 1
            is_valid_json = True
        except Exception:
            is_valid_json = False

        # token & latency
        api = item.get("api_response", {})
        itok = int(api.get("prompt_tokens", 0) or 0)
        otok = int(api.get("completion_tokens", 0) or 0)
        by_model_tokens[model]["input"].append(itok)
        by_model_tokens[model]["output"].append(otok)
        by_model_tokens[model]["total"].append(itok + otok)
        lat_ms = api.get("latency_ms")
        if lat_ms is None:
            # some pipelines store seconds
            lat_s = api.get("latency_s")
            if lat_s is not None:
                try:
                    lat_ms = float(lat_s) * 1000.0
                except Exception:
                    lat_ms = None
        if isinstance(lat_ms, (int, float)):
            by_model_latency_ms[model].append(float(lat_ms))

        # per-sample field evaluation
        sample = {
            "experiment_id": item.get("experiment_id"),
            "image_path": item.get("input", {}).get("image_path"),
            "model": model,
            "input_tokens": itok,
            "output_tokens": otok,
            "total_tokens": itok + otok,
            "latency_ms": float(lat_ms) if isinstance(lat_ms, (int, float)) else None,
            "is_valid_json": is_valid_json,
            "field_scores": {},
        }

        correct_fields_this = 0
        for pred_field, gt_field in field_mappings.items():
            gt = metadata.get(gt_field)
            pr = parsed.get(pred_field)
            ok = evaluate_field(gt, pr, pred_field)
            field_counters[pred_field]["total"] += 1
            if ok:
                field_counters[pred_field]["correct"] += 1
                correct_fields_this += 1
            sample["field_scores"][pred_field] = {
                "ground_truth": gt,
                "predicted": pr,
                "correct": ok,
            }

        if correct_fields_this == len(field_mappings):
            overall_all_fields_correct += 1

        # classification series for metrics (class)
        gt_cls = metadata.get("class")
        pr_cls = parsed.get("diagnosis_name")
        conf = parsed.get("diagnosis_confidence")
        conf_f = None
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = None
        
        invalid_vals = {None, 'null', 'nan', ''}
        
        if gt_cls not in invalid_vals:
            by_model_true[model].append(normalize_class_label(gt_cls))
            by_model_pred[model].append(normalize_class_label(pr_cls))
            by_model_conf[model].append(conf_f if conf_f is not None else 0.0)
        
        by_model_true_unfiltered[model].append(normalize_class_label(gt_cls))
        by_model_pred_unfiltered[model].append(normalize_class_label(pr_cls))
        by_model_conf_unfiltered[model].append(conf_f if conf_f is not None else 0.0)

        # subclass series
        gt_sub = metadata.get("subclass")
        pr_sub = parsed.get("diagnosis_detailed")
        if gt_sub not in invalid_vals: # and pr_sub is not None:
            by_model_true_sub[model].append(normalize_subclass_label(gt_sub))
            by_model_pred_sub[model].append(normalize_subclass_label(pr_sub))
            by_model_conf_sub[model].append(conf_f if conf_f is not None else 0.0)

        by_model_true_sub_unfiltered[model].append(normalize_subclass_label(gt_sub))
        by_model_pred_sub_unfiltered[model].append(normalize_subclass_label(pr_sub))
        by_model_conf_sub_unfiltered[model].append(conf_f if conf_f is not None else 0.0)

        # modality series (normalize MRI)
        gt_mod = metadata.get("modality")
        pr_mod = parsed.get("modality")
        if gt_mod not in invalid_vals: # and pr_mod is not None:
            by_model_true_mod[model].append(normalize_modality_label(gt_mod))
            by_model_pred_mod[model].append(normalize_modality_label(pr_mod))
        
        by_model_true_mod_unfiltered[model].append(normalize_modality_label(gt_mod))
        by_model_pred_mod_unfiltered[model].append(normalize_modality_label(pr_mod))
            
        # modality subtype series
        gt_mod_sub = metadata.get("modality_subtype")
        pr_mod_sub = parsed.get("specialized_sequence")
        if gt_mod_sub not in invalid_vals: # and pr_mod is not None:
            by_model_true_mod_sub[model].append(normalize_modality_subtype_label(gt_mod_sub))
            by_model_pred_mod_sub[model].append(normalize_modality_subtype_label(pr_mod_sub))
        
        by_model_true_mod_sub_unfiltered[model].append(normalize_modality_subtype_label(gt_mod_sub))
        by_model_pred_mod_sub_unfiltered[model].append(normalize_modality_subtype_label(pr_mod_sub))

        # plane series
        gt_plane = metadata.get("axial_plane")
        pr_plane = parsed.get("plane")
        if gt_plane not in invalid_vals: # and pr_plane is not None:
            by_model_true_plane[model].append(normalize_plane_label(gt_plane))
            by_model_pred_plane[model].append(normalize_plane_label(pr_plane))
        
        by_model_true_plane_unfiltered[model].append(normalize_plane_label(gt_plane))
        by_model_pred_plane_unfiltered[model].append(normalize_plane_label(pr_plane))

        # safety checks
        missing = [f for f in MANDATORY_FIELDS if parsed.get(f, None) in (None, "")]
        diag = normalize_class_label(pr_cls)
        sub = normalize_subclass_label(parsed.get("diagnosis_detailed")) if parsed.get("diagnosis_detailed") is not None else None
        icd = parsed.get("icd10_code")

        subclass_ok = True
        if diag in SUBCLASS_BY_CLASS:
            allowed = SUBCLASS_BY_CLASS[diag]
            # treat "null" string and None equally
            sub_norm = None if sub in ("", "null") else sub
            if allowed is not None and sub_norm not in allowed:
                subclass_ok = False
        # ICD-10 with normal diagnosis is suspicious
        halluc_icd = bool(icd) and diag == "normal"

        safety_issue = bool(missing) or not subclass_ok or halluc_icd
        sample["safety_flag"] = int(safety_issue)
        if safety_issue:
            safety_flags_rows.append(
                {
                    "experiment_id": sample["experiment_id"],
                    "model": model,
                    "missing_fields": ",".join(missing) if missing else "",
                    "subclass_ok": subclass_ok,
                    "icd10_with_normal": halluc_icd,
                }
            )

        results["detailed"].append(sample)

    # aggregate field accuracies
    for f, c in field_counters.items():
        results["field_results"][f] = {
            "accuracy_pct": pct(c["correct"], c["total"]),
            "correct": c["correct"],
            "total": c["total"],
        }
    results["overall_accuracy"] = pct(overall_all_fields_correct, len(results["detailed"]))

    # per-model metrics
    all_models = sorted(
        set(
            list(by_model_true.keys())
            + list(by_model_true_unfiltered.keys())
            + list(by_model_json_total.keys())
            + list(by_model_true_sub.keys())
            + list(by_model_true_sub_unfiltered.keys())
            + list(by_model_true_mod.keys())
            + list(by_model_true_mod_unfiltered.keys())
            + list(by_model_true_mod_sub.keys())
            + list(by_model_true_mod_sub_unfiltered.keys())
            + list(by_model_true_plane.keys())
            + list(by_model_true_plane_unfiltered.keys())
        )
    )

    for model in all_models:
        # ----- CLASS (diagnosis_name) -----
        y_t = by_model_true.get(model, [])
        y_t_unfiltered = by_model_true_unfiltered.get(model, [])
        print(len(y_t))
        print(len(y_t_unfiltered))
        
        y_p = by_model_pred.get(model, [])
        y_p_unfiltered = by_model_pred_unfiltered.get(model, [])
        print(len(y_p_unfiltered))
        
        y_c = by_model_conf.get(model, [])
        
        
        labels_present_cls = sorted(set(y_t) | set(y_p)) if (y_t or y_p) else []
        
        print(model)
        print('class ' + str(len(labels_present_cls)) + ' ' + str(labels_present_cls))
        
        
        if y_t and y_p:
            macro_f1 = float(f1_score(y_t, y_p, average="macro"))
            macro_f1_ci = f1_score_ci(y_t, y_p, labels = labels_present_cls, average="macro")
            
            micro_f1 = float(f1_score(y_t, y_p, average="micro"))
            micro_f1_ci = f1_score_ci(y_t, y_p, labels = labels_present_cls, average="micro")
            
            weighted_f1 = float(f1_score(y_t, y_p, average="weighted"))
            weighted_f1_ci = f1_score_ci(y_t, y_p, labels = labels_present_cls, average="weighted")
            
            macro_f1_abstention = f1_with_abstention(y_t, y_p, labels = labels_present_cls, abstain_label = "undetermined", average = "macro")
            macro_f1_abstention_ci = f1_with_abstention_ci(y_t, y_p, labels = labels_present_cls, abstain_label = "undetermined", average = "macro")
            
            micro_f1_abstention = f1_with_abstention(y_t, y_p, labels = labels_present_cls, abstain_label = "undetermined", average = "micro")
            micro_f1_abstention_ci = f1_with_abstention_ci(y_t, y_p, labels = labels_present_cls, abstain_label = "undetermined", average = "micro")
            
            weighted_f1_abstention = f1_with_abstention(y_t, y_p, labels = labels_present_cls, abstain_label = "undetermined", average = "weighted")
            weighted_f1_abstention_ci = f1_with_abstention_ci(y_t, y_p, labels = labels_present_cls, abstain_label = "undetermined", average = "weighted")
            
            #acc = float(np.mean([t == p for t, p in zip(y_t, y_p)]))
            #acc = accuracy_score(y_t, y_p)
            acc = balanced_accuracy_score(y_t, y_p)
            acc_ci = balanced_accuracy_score_ci(y_t, y_p)
            
            uncertainty = per_label_uncertainty(y_t, y_p, labels=labels_present_cls)["global"]

            prec_c = precision_score(y_t, y_p, average=None, labels=labels_present_cls, zero_division=0)
            prec_c_ci = per_label_precision_score_ci(y_t, y_p, average=None, labels=labels_present_cls, zero_division=0)

            rec_c = recall_score(y_t, y_p, average=None, labels=labels_present_cls, zero_division=0)
            rec_c_ci = per_label_recall_score_ci(y_t, y_p, average=None, labels=labels_present_cls, zero_division=0)
            
            f1_c = f1_score(y_t, y_p, average=None, labels=labels_present_cls, zero_division=0)
            f1_c_ci = per_label_f1_score_ci(y_t, y_p, labels=labels_present_cls, average=None, zero_division=0)
            
            f1_c_abstention = f1_with_abstention(y_t, y_p, average=None, labels=labels_present_cls, zero_division=0, abstain_label = "undetermined")
            f1_c_abstention_ci = per_label_f1_with_abstention_ci(y_t, y_p, labels=labels_present_cls, average=None, zero_division=0, abstain_label = "undetermined")
            
            # why nan?
            #weighted_f1_c = f1_score(y_t, y_p, average="weighted", labels=labels_present_cls) #, zero_division=0)
            
            #acc_c = per_label_accuracy(y_t, y_p, labels_present_cls)
            acc_c = per_label_balanced_accuracy(y_t, y_p, labels = labels_present_cls)
            acc_c_ci = per_label_balanced_accuracy_ci(y_t, y_p, labels = labels_present_cls)
            

            support_c = [int(sum(1 for t in y_t if t == lab)) for lab in labels_present_cls]

            # overall (macro) metrics for diagnosis_name
            overall_prec_macro_cls = float(precision_score(y_t, y_p, average="macro", zero_division=0))
            overall_prec_macro_cls_ci = precision_score_ci(y_t, y_p, labels = labels_present_cls, average="macro", zero_division=0)
            
            overall_rec_macro_cls = float(recall_score(y_t, y_p, average="macro", zero_division=0))
            overall_rec_macro_cls_ci = recall_score_ci(y_t, y_p, labels = labels_present_cls, average="macro", zero_division=0)
            
            overall_f1_macro_cls = float(f1_score(y_t, y_p, average="macro", zero_division=0))
            overall_f1_macro_cls_ci = f1_score_ci(y_t, y_p, labels = labels_present_cls, average="macro", zero_division=0)
            
            overall_f1_macro_abstention_cls = float(f1_with_abstention(y_t, y_p, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_cls))
            overall_f1_macro_abstention_cls_ci = f1_with_abstention_ci(y_t, y_p, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_cls)
            
            overall_f1_weighted_cls = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
            overall_f1_weighted_cls_ci = f1_score_ci(y_t, y_p, labels = labels_present_cls, average="weighted", zero_division=0)
            
            overall_f1_weighted_abstention_cls = float(f1_with_abstention(y_t, y_p, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_cls))
            overall_f1_weighted_abstention_cls_ci = f1_with_abstention_ci(y_t, y_p, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_cls)
            
            
            overall_acc_cls = acc
            overall_acc_cls_ci = acc_ci

            # calibration
            ece_val = ece_binned(y_t, y_p, y_c, bins=10)
            try:
                y_bin = np.array([1 if t == p else 0 for t, p in zip(y_t, y_p)], dtype=float)
                conf_arr = np.clip(np.array([c if c is not None else 0.0 for c in y_c], dtype=float), 0, 1)
                brier = float(brier_score_loss(y_bin, conf_arr))
                auc_cls = float(roc_auc_score(y_bin, conf_arr)) if len(np.unique(y_bin)) == 2 else float("nan")
            except Exception:
                brier = float("nan")
                auc_cls = float("nan")
                ece_val = float("nan")

            # confusion matrix
            cm_to_plot = confusion_matrix(y_t, y_p, labels=labels_present_cls, normalize="all")#.tolist()
            cm = cm_to_plot.tolist()
            cm_rows_to_plot = confusion_matrix(y_t, y_p, labels=labels_present_cls, normalize="true")#.tolist()
            
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_cls)+2, len(labels_present_cls)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_to_plot, display_labels=labels_present_cls) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per diagnosis name')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normall_class_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_cls)+2, len(labels_present_cls)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_rows_to_plot, display_labels=labels_present_cls) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per diagnosis name')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normrows_class_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            
           
        else:
            macro_f1 = micro_f1 = acc = float("nan")
            prec_c = rec_c = f1_c = f1_c_abstention = np.array([])
            acc_c = []
            support_c = []
            overall_prec_macro_cls = overall_rec_macro_cls = overall_f1_macro_cls = overall_f1_macro_abstention_cls = overall_f1_weighted_cls = overall_f1_weighted_abstention_cls = overall_acc_cls = float("nan")
            ece_val = brier = auc_cls = float("nan")
            cm = []
            labels_present_cls = []
         
            prec_c_ci = rec_c_ci = f1_c_ci = f1_c_abstention_ci = (0.0, (0.0, 0.0))
            acc_c_ci = (0.0, (0.0, 0.0))
            overall_prec_macro_cls_ci = overall_rec_macro_cls_ci = overall_f1_macro_cls_ci = overall_f1_macro_abstention_cls_ci = overall_f1_weighted_cls_ci = overall_f1_weighted_abstention_cls_ci = overall_acc_cls_ci = (0.0, (0.0, 0.0))
 

        # ----- SUBCLASS -----
        y_t_sub = by_model_true_sub.get(model, [])
        y_p_sub = by_model_pred_sub.get(model, [])
        
        y_t_sub_unfiltered = by_model_true_sub_unfiltered.get(model, [])
        y_p_sub_unfiltered = by_model_pred_sub_unfiltered.get(model, [])
        
        y_c_sub = by_model_conf_sub.get(model, [])
        
        labels_present_sub = sorted(set(y_t_sub) | set(y_p_sub)) if (y_t_sub or y_p_sub) else []

        print('subclass ' + str(len(labels_present_sub)) + ' ' + str(labels_present_sub))
        
        if y_t_sub and y_p_sub:
            prec_s = precision_score(y_t_sub, y_p_sub, average=None, labels=labels_present_sub, zero_division=0)
            prec_s_ci = per_label_precision_score_ci(y_t_sub, y_p_sub, average=None, labels=labels_present_sub, zero_division=0)

            rec_s = recall_score(y_t_sub, y_p_sub, average=None, labels=labels_present_sub, zero_division=0)
            rec_s_ci = per_label_recall_score_ci(y_t_sub, y_p_sub, average=None, labels=labels_present_sub, zero_division=0)
            
            f1_s = f1_score(y_t_sub, y_p_sub, average=None, labels=labels_present_sub, zero_division=0)
            f1_s_ci = per_label_f1_score_ci(y_t_sub, y_p_sub, labels=labels_present_sub, average=None, zero_division=0)
            
            f1_s_abstention = f1_with_abstention(y_t_sub, y_p_sub, average=None, labels=labels_present_sub, zero_division=0, abstain_label = "undetermined")
            f1_s_abstention_ci = per_label_f1_with_abstention_ci(y_t_sub, y_p_sub, labels=labels_present_sub, average=None, zero_division=0, abstain_label = "undetermined")
            
            # why nan?
            #weighted_f1_s = f1_score(y_t_sub, y_p_sub, average="weighted", labels=labels_present_sub, zero_division=0)
            
            #acc_s = per_label_accuracy(y_t_sub, y_p_sub, labels=labels_present_sub)
            acc_s = per_label_balanced_accuracy(y_t_sub, y_p_sub, labels = labels_present_sub)
            acc_s_ci = per_label_balanced_accuracy_ci(y_t_sub, y_p_sub, labels = labels_present_sub)
            
            support_s = [int(sum(1 for t in y_t_sub if t == lab)) for lab in labels_present_sub]
            # overall (macro) metrics for subclass
            overall_prec_macro_sub = float(precision_score(y_t_sub, y_p_sub, average="macro", zero_division=0))
            overall_prec_macro_sub_ci = precision_score_ci(y_t_sub, y_p_sub, labels = labels_present_sub, average="macro", zero_division=0)
            
            overall_rec_macro_sub = float(recall_score(y_t_sub, y_p_sub, average="macro", zero_division=0))
            overall_rec_macro_sub_ci = recall_score_ci(y_t_sub, y_p_sub, labels = labels_present_sub, average="macro", zero_division=0)
            
            overall_f1_macro_sub = float(f1_score(y_t_sub, y_p_sub, average="macro", zero_division=0))
            overall_f1_macro_sub_ci = f1_score_ci(y_t_sub, y_p_sub, labels = labels_present_sub, average="macro", zero_division=0)
            
            overall_f1_macro_abstention_sub = float(f1_with_abstention(y_t_sub, y_p_sub, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_sub))
            overall_f1_macro_abstention_sub_ci = f1_with_abstention_ci(y_t_sub, y_p_sub, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_sub)
            
            overall_f1_weighted_sub = float(f1_score(y_t_sub, y_p_sub, average="weighted", zero_division=0))
            overall_f1_weighted_sub_ci = f1_score_ci(y_t_sub, y_p_sub, labels = labels_present_sub, average="weighted", zero_division=0)
            
            overall_f1_weighted_abstention_sub = float(f1_with_abstention(y_t_sub, y_p_sub, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_sub))
            overall_f1_weighted_abstention_sub_ci = f1_with_abstention_ci(y_t_sub, y_p_sub, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_sub)
            
            overall_acc_sub = balanced_accuracy_score(y_t_sub, y_p_sub)
            overall_acc_sub_ci = balanced_accuracy_score_ci(y_t_sub, y_p_sub)
            #overall_acc_sub = float(np.mean([t == p for t, p in zip(y_t_sub, y_p_sub)]))            
            
            cm_sub_to_plot = confusion_matrix(y_t_sub, y_p_sub, labels=labels_present_sub, normalize="all")#.tolist()
            cm_sub = cm_sub_to_plot.tolist()
            cm_sub_rows_to_plot = confusion_matrix(y_t_sub, y_p_sub, labels=labels_present_sub, normalize="true")#.tolist()
                       
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_sub)+2, len(labels_present_sub)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_sub_to_plot, display_labels=labels_present_sub) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per detailed diagnosis')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normall_subclass_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_sub)+2, len(labels_present_sub)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_sub_rows_to_plot, display_labels=labels_present_sub) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per detailed diagnosis')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normrows_subclass_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
                
            
            
            # calibration
            ece_sub = ece_binned(y_t_sub, y_p_sub, y_c_sub, bins=10)
            try:
                y_bin_sub = np.array([1 if t == p else 0 for t, p in zip(y_t_sub, y_p_sub)], dtype=float)
                conf_arr_sub = np.clip(np.array([c if c is not None else 0.0 for c in y_c_sub], dtype=float), 0, 1)
                brier_sub = float(brier_score_loss(y_bin_sub, conf_arr_sub))
                auc_sub = float(roc_auc_score(y_bin_sub, conf_arr_sub)) if len(np.unique(y_bin_sub)) == 2 else float("nan")
            except Exception:
                brier_sub = float("nan")
                auc_sub = float("nan")
                ece_sub = float("nan")
                
            # no confidences for subclass -> calibration placeholders
            #auc_sub = float("nan"); ece_sub = float("nan"); brier_sub = float("nan")
            
        else:
            prec_s = rec_s = f1_s = f1_s_abstention = np.array([])
            acc_s = []
            support_s = []
            labels_present_sub = []
            overall_prec_macro_sub = overall_rec_macro_sub = overall_f1_macro_sub = overall_f1_macro_abstention_sub = overall_f1_weighted_sub = overall_f1_weighted_abstention_sub = overall_acc_sub = float("nan")
            cm_sub = []
            auc_sub = ece_sub = brier_sub = float("nan")  
            
            prec_s_ci = rec_s_ci = f1_s_ci = f1_s_abstention_ci = (0.0, (0.0, 0.0))
            acc_s_ci = (0.0, (0.0, 0.0))
            overall_prec_macro_sub_ci = overall_rec_macro_sub_ci = overall_f1_macro_sub_ci = overall_f1_macro_abstention_sub_ci = overall_f1_weighted_sub_ci = overall_f1_weighted_abstention_sub_ci = overall_acc_sub_ci = (0.0, (0.0, 0.0))
  

        # ----- MODALITY -----
        y_t_mod = by_model_true_mod.get(model, [])
        y_p_mod = by_model_pred_mod.get(model, [])
        
        y_t_mod_unfiltered = by_model_true_mod_unfiltered.get(model, [])
        y_p_mod_unfiltered = by_model_pred_mod_unfiltered.get(model, [])
        
        y_c_mod = by_model_conf_mod.get(model, [])
        labels_present_mod = sorted(set(y_t_mod) | set(y_p_mod)) if (y_t_mod or y_p_mod) else []
        
        print('modality ' +  str(len(labels_present_mod)) + ' ' + str(labels_present_mod))
        
        if y_t_mod and y_p_mod:
            prec_m = precision_score(y_t_mod, y_p_mod, average=None, labels=labels_present_mod, zero_division=0)
            prec_m_ci = per_label_precision_score_ci(y_t_mod, y_p_mod, average=None, labels=labels_present_mod, zero_division=0)

            rec_m = recall_score(y_t_mod, y_p_mod, average=None, labels=labels_present_mod, zero_division=0)
            rec_m_ci = per_label_recall_score_ci(y_t_mod, y_p_mod, average=None, labels=labels_present_mod, zero_division=0)
            
            f1_m = f1_score(y_t_mod, y_p_mod, average=None, labels=labels_present_mod, zero_division=0)
            f1_m_ci = per_label_f1_score_ci(y_t_mod, y_p_mod, labels=labels_present_mod, average=None, zero_division=0)
            
            f1_m_abstention = f1_with_abstention(y_t_mod, y_p_mod, average=None, labels=labels_present_mod, zero_division=0, abstain_label = "undetermined")
            f1_m_abstention_ci = per_label_f1_with_abstention_ci(y_t_mod, y_p_mod, labels=labels_present_mod, average=None, zero_division=0, abstain_label = "undetermined")
            
            #weighted_f1_m = f1_score(y_t_mod, y_p_mod, average="weighted", labels=labels_present_mod, zero_division=0)
            
            #acc_m = per_label_accuracy(y_t_mod, y_p_mod, labels=labels_present_mod)
            acc_m = per_label_balanced_accuracy(y_t_mod, y_p_mod, labels=labels_present_mod)
            acc_m_ci = per_label_balanced_accuracy_ci(y_t_mod, y_p_mod, labels = labels_present_mod)
            
            support_m = [int(sum(1 for t in y_t_mod if t == lab)) for lab in labels_present_mod]
            
            # overall (macro) metrics for modality
            overall_prec_macro_mod = float(precision_score(y_t_mod, y_p_mod, average="macro", zero_division=0))
            overall_prec_macro_mod_ci = precision_score_ci(y_t_mod, y_p_mod, labels = labels_present_mod, average="macro", zero_division=0)
            
            overall_rec_macro_mod = float(recall_score(y_t_mod, y_p_mod, average="macro", zero_division=0))
            overall_rec_macro_mod_ci = recall_score_ci(y_t_mod, y_p_mod, labels = labels_present_mod, average="macro", zero_division=0)
            
            overall_f1_macro_mod = float(f1_score(y_t_mod, y_p_mod, average="macro", zero_division=0))
            overall_f1_macro_mod_ci = f1_score_ci(y_t_mod, y_p_mod, labels = labels_present_mod, average="macro", zero_division=0)
            
            overall_f1_macro_abstention_mod = float(f1_with_abstention(y_t_mod, y_p_mod, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod))
            overall_f1_macro_abstention_mod_ci = f1_with_abstention_ci(y_t_mod, y_p_mod, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod)
            
            overall_f1_weighted_mod = float(f1_score(y_t_mod, y_p_mod, average="weighted", zero_division=0))
            overall_f1_weighted_mod_ci = f1_score_ci(y_t_mod, y_p_mod, labels = labels_present_mod, average="weighted", zero_division=0)
            
            overall_f1_weighted_abstention_mod = float(f1_with_abstention(y_t_mod, y_p_mod, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod))
            overall_f1_weighted_abstention_mod_ci = f1_with_abstention_ci(y_t_mod, y_p_mod, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod)
            
            overall_acc_mod = balanced_accuracy_score(y_t_mod, y_p_mod)
            overall_acc_mod_ci = balanced_accuracy_score_ci(y_t_mod, y_p_mod)
            #overall_acc_mod = float(np.mean([t == p for t, p in zip(y_t_mod, y_p_mod)]))
            
            cm_mod_to_plot = confusion_matrix(y_t_mod, y_p_mod, labels=labels_present_mod, normalize="all")#.tolist()
            cm_mod = cm_mod_to_plot.tolist()
            cm_mod_rows_to_plot = confusion_matrix(y_t_mod, y_p_mod, labels=labels_present_mod, normalize="true")#.tolist()
            
                       
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_mod)+2, len(labels_present_mod)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_mod_to_plot, display_labels=labels_present_mod) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per modality')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normall_modality_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_mod)+2, len(labels_present_mod)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_mod_rows_to_plot, display_labels=labels_present_mod) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per modality')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normrows_modality_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            
            # calibration
            ece_mod = ece_binned(y_t_mod, y_p_mod, y_c_mod, bins=5)
            try:
                y_bin_mod = np.array([1 if t == p else 0 for t, p in zip(y_t_mod, y_p_mod)], dtype=float)
                conf_arr_mod = np.clip(np.array([c if c is not None else 0.0 for c in y_c_mod], dtype=float), 0, 1)
                brier_mod = float(brier_score_loss(y_bin_mod, conf_arr_mod))
                auc_mod = float(roc_auc_score(y_bin_mod, conf_arr_mod)) if len(np.unique(y_bin_mod)) == 2 else float("nan")
            except Exception:
                brier_mod = float("nan")
                auc_mod = float("nan")
                ece_mod = float("nan")
                
            # no confidences for modality -> calibration placeholders
            #auc_mod = float("nan"); ece_mod = float("nan"); brier_mod = float("nan")
        else:
            prec_m = rec_m = f1_m = f1_m_abstention = np.array([])
            acc_m = []
            support_m = []
            labels_present_mod = []
            overall_prec_macro_mod = overall_rec_macro_mod = overall_f1_macro_mod = overall_f1_macro_abstention_mod = overall_f1_weighted_mod = overall_f1_weighted_abstention_mod = overall_acc_mod = float("nan")
            cm_mod = []
            auc_mod = ece_mod = brier_mod = float("nan")
            
            prec_m_ci = rec_m_ci = f1_m_ci = f1_m_abstention_ci = (0.0, (0.0, 0.0))
            acc_m_ci = (0.0, (0.0, 0.0))
            overall_prec_macro_mod_ci = overall_rec_macro_mod_ci = overall_f1_macro_mod_ci = overall_f1_macro_abstention_mod_ci = overall_f1_weighted_mod_ci = overall_f1_weighted_abstention_mod_ci = overall_acc_mod_ci = (0.0, (0.0, 0.0))
            

        # ----- MODALITY SUBTYPE-----
        y_t_mod_sub = by_model_true_mod_sub.get(model, [])
        y_p_mod_sub = by_model_pred_mod_sub.get(model, [])
        
        y_t_mod_sub_unfiltered = by_model_true_mod_sub_unfiltered.get(model, [])
        y_p_mod_sub_unfiltered = by_model_pred_mod_sub_unfiltered.get(model, [])
        
        y_c_mod_sub = by_model_conf_mod_sub.get(model, [])
        labels_present_mod_sub = sorted(set(y_t_mod_sub) | set(y_p_mod_sub)) if (y_t_mod_sub or y_p_mod_sub) else []
        
        print('modality subtype ' +  str(len(labels_present_mod_sub)) + ' ' + str(labels_present_mod_sub))
        
        if y_t_mod_sub and y_p_mod_sub:
            prec_m_sub = precision_score(y_t_mod_sub, y_p_mod_sub, average=None, labels=labels_present_mod_sub, zero_division=0)
            prec_m_sub_ci = per_label_precision_score_ci(y_t_mod_sub, y_p_mod_sub, average=None, labels=labels_present_mod_sub, zero_division=0)

            rec_m_sub = recall_score(y_t_mod_sub, y_p_mod_sub, average=None, labels=labels_present_mod_sub, zero_division=0)
            rec_m_sub_ci = per_label_recall_score_ci(y_t_mod_sub, y_p_mod_sub, average=None, labels=labels_present_mod_sub, zero_division=0)
            
            f1_m_sub = f1_score(y_t_mod_sub, y_p_mod_sub, average=None, labels=labels_present_mod_sub, zero_division=0)
            f1_m_sub_ci = per_label_f1_score_ci(y_t_mod_sub, y_p_mod_sub, labels=labels_present_mod_sub, average=None, zero_division=0)
            
            f1_m_sub_abstention = f1_with_abstention(y_t_mod_sub, y_p_mod_sub, average=None, labels=labels_present_mod_sub, zero_division=0, abstain_label = "undetermined")
            f1_m_sub_abstention_ci = per_label_f1_with_abstention_ci(y_t_mod_sub, y_p_mod_sub, labels=labels_present_mod_sub, average=None, zero_division=0, abstain_label = "undetermined")
            
            #weighted_f1_m_sub = f1_score(y_t_mod_sub, y_p_mod_sub, average="weighted", labels=labels_present_mod_sub, zero_division=0)
            
            #acc_m_sub = per_label_accuracy(y_t_mod_sub, y_p_mod_sub, labels=labels_present_mod_sub)
            acc_m_sub = per_label_balanced_accuracy(y_t_mod_sub, y_p_mod_sub, labels=labels_present_mod_sub)
            acc_m_sub_ci = per_label_balanced_accuracy_ci(y_t_mod_sub, y_p_mod_sub, labels = labels_present_mod_sub)
            
            support_m_sub = [int(sum(1 for t in y_t_mod_sub if t == lab)) for lab in labels_present_mod_sub]
            
            # overall (macro) metrics for modality subtype
            overall_prec_macro_mod_sub = float(precision_score(y_t_mod_sub, y_p_mod_sub, average="macro", zero_division=0))
            overall_prec_macro_mod_sub_ci = precision_score_ci(y_t_mod_sub, y_p_mod_sub, labels = labels_present_mod_sub, average="macro", zero_division=0)
            
            overall_rec_macro_mod_sub = float(recall_score(y_t_mod_sub, y_p_mod_sub, average="macro", zero_division=0))
            overall_rec_macro_mod_sub_ci = recall_score_ci(y_t_mod_sub, y_p_mod_sub, labels = labels_present_mod_sub, average="macro", zero_division=0)
            
            overall_f1_macro_mod_sub = float(f1_score(y_t_mod_sub, y_p_mod_sub, average="macro", zero_division=0))
            overall_f1_macro_mod_sub_ci = f1_score_ci(y_t_mod_sub, y_p_mod_sub, labels = labels_present_mod_sub, average="macro", zero_division=0)
            
            overall_f1_macro_abstention_mod_sub = float(f1_with_abstention(y_t_mod_sub, y_p_mod_sub, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod_sub))
            overall_f1_macro_abstention_mod_sub_ci = f1_with_abstention_ci(y_t_mod_sub, y_p_mod_sub, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod_sub)
            
            overall_f1_weighted_mod_sub = float(f1_score(y_t_mod_sub, y_p_mod_sub, average="weighted", zero_division=0))
            overall_f1_weighted_mod_sub_ci = f1_score_ci(y_t_mod_sub, y_p_mod_sub, labels = labels_present_mod_sub, average="weighted", zero_division=0)
            
            overall_f1_weighted_abstention_mod_sub = float(f1_with_abstention(y_t_mod_sub, y_p_mod_sub, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod_sub))
            overall_f1_weighted_abstention_mod_sub_ci = f1_with_abstention_ci(y_t_mod_sub, y_p_mod_sub, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_mod_sub)
            
            overall_acc_mod_sub = balanced_accuracy_score(y_t_mod_sub, y_p_mod_sub)
            overall_acc_mod_sub_ci = balanced_accuracy_score_ci(y_t_mod_sub, y_p_mod_sub)
            
            #overall_acc_mod_sub = float(np.mean([t == p for t, p in zip(y_t_mod_sub, y_p_mod_sub)]))
            cm_mod_sub_to_plot = confusion_matrix(y_t_mod_sub, y_p_mod_sub, labels=labels_present_mod_sub, normalize="all")#.tolist()
            cm_mod_sub = cm_mod_sub_to_plot.tolist()
            cm_mod_sub_rows_to_plot = confusion_matrix(y_t_mod_sub, y_p_mod_sub, labels=labels_present_mod_sub, normalize="true")#.tolist()
            
                       
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_mod_sub)+2, len(labels_present_mod_sub)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_mod_sub_to_plot, display_labels=labels_present_mod_sub) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per specialized sequence')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normall_modality_sub_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_mod_sub)+2, len(labels_present_mod_sub)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_mod_sub_rows_to_plot, display_labels=labels_present_mod_sub) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per specialized sequence')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normrows_modality_sub_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            
            # calibration
            ece_mod_sub = ece_binned(y_t_mod_sub, y_p_mod_sub, y_c_mod_sub, bins=5)
            try:
                y_bin_mod_sub = np.array([1 if t == p else 0 for t, p in zip(y_t_mod_sub, y_p_mod_sub)], dtype=float)
                conf_arr_mod_sub = np.clip(np.array([c if c is not None else 0.0 for c in y_c_mod_sub], dtype=float), 0, 1)
                brier_mod_sub = float(brier_score_loss(y_bin_mod_sub, conf_arr_mod_sub))
                auc_mod_sub = float(roc_auc_score(y_bin_mod_sub, conf_arr_mod_sub)) if len(np.unique(y_bin_mod_sub)) == 2 else float("nan")
            except Exception:
                brier_mod_sub = float("nan")
                auc_mod_sub = float("nan")
                ece_mod_sub = float("nan")
                
            # no confidences for modality -> calibration placeholders
            #auc_mod_sub = float("nan"); ece_mod_sub = float("nan"); brier_mod_sub = float("nan")
        else:
            prec_m_sub = rec_m_sub = f1_m_sub = f1_m_sub_abstention = np.array([])
            acc_m_sub = []
            support_m_sub = []
            labels_present_mod_sub = []
            overall_prec_macro_mod_sub = overall_rec_macro_mod_sub = overall_f1_macro_mod_sub = overall_f1_macro_abstention_mod_sub = overall_f1_weighted_mod_sub = overall_f1_weighted_abstention_mod_sub= overall_acc_mod_sub = float("nan")
            cm_mod_sub = []
            auc_mod_sub = ece_mod_sub = brier_mod_sub = float("nan")
          
            prec_m_sub_ci = rec_m_sub_ci = f1_m_sub_ci = f1_m_sub_abstention_ci = (0.0, (0.0, 0.0))
            acc_m_sub_ci = (0.0, (0.0, 0.0))
            overall_prec_macro_mod_sub_ci = overall_rec_macro_mod_sub_ci = overall_f1_macro_mod_sub_ci = overall_f1_macro_abstention_mod_sub_ci = overall_f1_weighted_mod_sub_ci = overall_f1_weighted_abstention_mod_sub_ci = overall_acc_mod_sub_ci = (0.0, (0.0, 0.0))
    

        # ----- PLANE -----
        y_t_pl = by_model_true_plane.get(model, [])
        y_p_pl = by_model_pred_plane.get(model, [])
        
        y_t_pl_unfiltered = by_model_true_plane_unfiltered.get(model, [])
        y_p_pl_unfiltered = by_model_pred_plane_unfiltered.get(model, [])
        
        y_c_pl = by_model_conf_plane.get(model, [])
        labels_present_pl = sorted(set(y_t_pl) | set(y_p_pl)) if (y_t_pl or y_p_pl) else []
        
        print('plane ' +  str(len(labels_present_pl)) + ' ' + str(labels_present_pl))
        
        if y_t_pl and y_p_pl:
            prec_p = precision_score(y_t_pl, y_p_pl, average=None, labels=labels_present_pl, zero_division=0)
            prec_p_ci = per_label_precision_score_ci(y_t_pl, y_p_pl, average=None, labels=labels_present_pl, zero_division=0)

            rec_p = recall_score(y_t_pl, y_p_pl, average=None, labels=labels_present_pl, zero_division=0)
            rec_p_ci = per_label_recall_score_ci(y_t_pl, y_p_pl, average=None, labels=labels_present_pl, zero_division=0)
            
            f1_p = f1_score(y_t_pl, y_p_pl, average=None, labels=labels_present_pl, zero_division=0)
            f1_p_ci = per_label_f1_score_ci(y_t_pl, y_p_pl, labels=labels_present_pl, average=None, zero_division=0)
            
            f1_p_abstention = f1_with_abstention(y_t_pl, y_p_pl, average=None, labels=labels_present_pl, zero_division=0, abstain_label = "undetermined")
            f1_p_abstention_ci = per_label_f1_with_abstention_ci(y_t_pl, y_p_pl, labels=labels_present_pl, average=None, zero_division=0, abstain_label = "undetermined")
            
            #weighted_f1_p = f1_score(y_t_pl, y_p_pl, average="weighted", labels=labels_present_pl, zero_division=0)
            
            #acc_p = per_label_accuracy(y_t_pl, y_p_pl, labels=labels_present_pl)
            acc_p = per_label_balanced_accuracy(y_t_pl, y_p_pl, labels=labels_present_pl)
            acc_p_ci = per_label_balanced_accuracy_ci(y_t_pl, y_p_pl, labels = labels_present_pl)
            
            support_p = [int(sum(1 for t in y_t_pl if t == lab)) for lab in labels_present_pl]
            
            # overall (macro) metrics for plane
            overall_prec_macro_pl = float(precision_score(y_t_pl, y_p_pl, average="macro", zero_division=0))
            overall_prec_macro_pl_ci = precision_score_ci(y_t_pl, y_p_pl, labels = labels_present_pl, average="macro", zero_division=0)
            
            overall_rec_macro_pl = float(recall_score(y_t_pl, y_p_pl, average="macro", zero_division=0))
            overall_rec_macro_pl_ci = recall_score_ci(y_t_pl, y_p_pl, labels = labels_present_pl, average="macro", zero_division=0)
            
            overall_f1_macro_pl = float(f1_score(y_t_pl, y_p_pl, average="macro", zero_division=0))
            overall_f1_macro_pl_ci = f1_score_ci(y_t_pl, y_p_pl, labels = labels_present_pl, average="macro", zero_division=0)
            
            overall_f1_macro_abstention_pl = float(f1_with_abstention(y_t_pl, y_p_pl, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_pl))
            overall_f1_macro_abstention_pl_ci = f1_with_abstention_ci(y_t_pl, y_p_pl, average="macro", zero_division=0, abstain_label = "undetermined", labels = labels_present_pl)
            
            overall_f1_weighted_pl = float(f1_score(y_t_pl, y_p_pl, average="weighted", zero_division=0))
            overall_f1_weighted_pl_ci = f1_score_ci(y_t_pl, y_p_pl, labels = labels_present_pl, average="weighted", zero_division=0)
            
            overall_f1_weighted_abstention_pl = float(f1_with_abstention(y_t_pl, y_p_pl, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_pl))
            overall_f1_weighted_abstention_pl_ci = f1_with_abstention_ci(y_t_pl, y_p_pl, average="weighted", zero_division=0, abstain_label = "undetermined", labels = labels_present_pl)
            
            overall_acc_pl = balanced_accuracy_score(y_t_pl, y_p_pl)
            overall_acc_pl_ci = balanced_accuracy_score_ci(y_t_pl, y_p_pl)
            #overall_acc_pl = float(np.mean([t == p for t, p in zip(y_t_pl, y_p_pl)]))
            
            cm_pl_to_plot = confusion_matrix(y_t_pl, y_p_pl, labels=labels_present_pl, normalize="all")#.tolist()
            cm_pl = cm_pl_to_plot.tolist()
            cm_pl_rows_to_plot = confusion_matrix(y_t_pl, y_p_pl, labels=labels_present_pl, normalize="true")#.tolist()
                       
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_pl)+2, len(labels_present_pl)+2))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_pl_to_plot, display_labels=labels_present_pl) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per plane')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normall_plane_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
            
            #if not model == 'bedrock/amazon.nova-lite-v1:0':
            fig, ax = plt.subplots(figsize=(len(labels_present_pl)+1, len(labels_present_pl)+1))   # larger figure
            display = ConfusionMatrixDisplay(confusion_matrix=cm_pl_rows_to_plot, display_labels=labels_present_pl) 
            display.plot(cmap="Blues", #values_format=".1%", 
                         ax=ax, colorbar=False) 
            plt.title(model + ' per plane')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90, ha='right')          
            plt.tight_layout() # Adjust layout to prevent labels from overlapping         
            plt.savefig('results/confusion_matrix/cm_normrows_plane_' + re.sub(r"[\/.:]", "_", model) + '_' + out_path[8:] +'.pdf', format='pdf')
            plt.show() # Optional: display the plot
            plt.close()
             
            # calibration
            ece_pl = ece_binned(y_t_pl, y_p_pl, y_c_pl, bins=5)
            try:
                y_bin_pl = np.array([1 if t == p else 0 for t, p in zip(y_t_pl, y_p_pl)], dtype=float)
                conf_arr_pl = np.clip(np.array([c if c is not None else 0.0 for c in y_c_pl], dtype=float), 0, 1)
                brier_pl = float(brier_score_loss(y_bin_pl, conf_arr_pl))
                auc_pl = float(roc_auc_score(y_bin_pl, conf_arr_pl)) if len(np.unique(y_bin_pl)) == 2 else float("nan")
            except Exception:
                brier_pl = float("nan")
                auc_pl = float("nan")
                ece_pl = float("nan")
                
            # no confidences for plane -> calibration placeholders
            #auc_pl = float("nan"); ece_pl = float("nan"); brier_pl = float("nan")
        else:
            prec_p = rec_p = f1_p = f1_p_abstention = np.array([])
            acc_p = []
            support_p = []
            labels_present_pl = []
 
            overall_prec_macro_pl = overall_rec_macro_pl = overall_f1_macro_pl = overall_f1_macro_abstention_pl = overall_f1_weighted_pl = overall_f1_weighted_abstention_pl = overall_acc_pl = float("nan")
            cm_pl = []
            auc_pl = ece_pl = brier_pl = float("nan")
            
            prec_p_ci = rec_p_ci = f1_p_ci = f1_p_abstention_ci = (0.0, (0.0, 0.0))
            acc_p_ci = (0.0, (0.0, 0.0))
            overall_prec_macro_pl_ci = overall_rec_macro_pl_ci = overall_f1_macro_pl_ci = overall_f1_macro_abstention_pl_ci = overall_f1_weighted_pl_ci = overall_f1_weighted_abstention_pl_ci = overall_acc_pl_ci = (0.0, (0.0, 0.0))
        

        y_true = np.column_stack([y_t_unfiltered, y_t_sub_unfiltered, y_t_mod_unfiltered, y_t_mod_sub_unfiltered, y_t_pl_unfiltered])  
        y_pred = np.column_stack([y_p, y_p_sub_unfiltered, y_p_mod_unfiltered, y_p_mod_sub_unfiltered, y_p_pl_unfiltered])
  
        m = MultiLabelBinarizer().fit(y_true)
        
        macro_f1_global = f1_score(m.transform(y_true), m.transform(y_pred), average="macro")
        micro_f1_global = f1_score(m.transform(y_true), m.transform(y_pred), average="micro")
        weighted_f1_global = f1_score(m.transform(y_true), m.transform(y_pred), average="weighted")
        
        #acc = float(np.mean([t == p for t, p in zip(y_t, y_p)]))
        acc_global = accuracy_score(m.transform(y_true), m.transform(y_pred))
        
        hamming_loss_global = hamming_loss(m.transform(y_true), m.transform(y_pred))
        #Hamming loss: 0.4166666666666667

        # format robustness / latency / tokens / cost
        valid = by_model_json_valid.get(model, 0)
        total = by_model_json_total.get(model, 0)
        json_valid_pct = pct(valid, total)
        lat_list = by_model_latency_ms.get(model, [])
        median_latency_ms = float(np.median(lat_list)) if lat_list else float("nan")

        tok = by_model_tokens.get(model, {})
        avg_input_tok = float(np.mean(tok.get("input", [0]))) if tok.get("input") else 0.0
        avg_output_tok = float(np.mean(tok.get("output", [0]))) if tok.get("output") else 0.0
        avg_total_tok = float(np.mean(tok.get("total", [0]))) if tok.get("total") else 0.0
        
        tokens_per_1k_images = avg_total_tok * 1000.0
        
        print("-----------------------------")
        print(f"Model {model}")
        print("-----------------------------")
        usd_cost_per_1k = tokens_per_1k_images / 1_000_000.0 * DEFAULT_PRICE_PER_1M_TOKENS_USD

        # calc prices
        avg_input_cost = MODEL_PRICES[model].get("input_per_m", DEFAULT_PRICE_PER_1M_TOKENS_USD) * avg_input_tok / 1_000_000.0
        avg_output_cost = MODEL_PRICES[model].get("output_per_m", DEFAULT_PRICE_PER_1M_TOKENS_USD) * avg_output_tok / 1_000_000.0
        avg_cost = avg_input_cost + avg_output_cost

        results["per_model"][model] = {
            # global (class) summary
            "macro_f1": macro_f1,
            "macro_f1_ci": macro_f1_ci,
            "micro_f1": micro_f1,
            "micro_f1_ci": micro_f1_ci,
            "weighted_f1": weighted_f1,
            "weighted_f1_ci": weighted_f1_ci,
            "macro_f1_abstention": macro_f1_abstention,
            "macro_f1_abstention_ci": macro_f1_abstention_ci,
            "micro_f1_abstention": micro_f1_abstention,
            "micro_f1_abstention_ci": micro_f1_abstention_ci,
            "weighted_f1_abstention": weighted_f1_abstention,
            "weighted_f1_abstention_ci": weighted_f1_abstention_ci,
            "accuracy": acc,
            "accuracy_ci": acc_ci,
            "ece": ece_val,
            "brier": brier,
            "json_valid_pct": json_valid_pct,
            "median_latency_ms": median_latency_ms,
            "avg_input_tokens": avg_input_tok,
            "avg_output_tokens": avg_output_tok,
            "avg_total_tokens": avg_total_tok,
            "avg_input_cost": avg_input_cost,
            "avg_output_cost": avg_output_cost,
            "avg_cost": avg_cost,
            "tokens_per_1k_images": tokens_per_1k_images,
            "usd_cost_per_1k_images": usd_cost_per_1k,
            "uncertainty": uncertainty,
            "macro_f1_global": macro_f1_global,
            "micro_f1_global": micro_f1_global,
            "weighted_f1_global": weighted_f1_global,
            "acc_global": acc_global,
            "hamming_loss_global": hamming_loss_global,
            # per-class (diagnosis_name)
            "labels": labels_present_cls,
            "per_class_accuracy": acc_c,  # per-label accuracy
            "per_class_accuracy_ci": acc_c_ci,  # per-label accuracy
            "per_class_precision": prec_c.tolist() if hasattr(prec_c, "tolist") else [],
            "per_class_precision_ci": prec_c_ci,
            "per_class_recall": rec_c.tolist() if hasattr(rec_c, "tolist") else [],
            "per_class_recall_ci": rec_c_ci,
            "per_class_f1": f1_c.tolist() if hasattr(f1_c, "tolist") else [],
            "per_class_f1_ci": f1_c_ci,
            "per_class_f1_abstention": f1_c_abstention.tolist() if hasattr(f1_c_abstention, "tolist") else [],
            "per_class_f1_abstention_ci": f1_c_abstention_ci,
            "per_class_support": support_c,
            "confusion_matrix": cm,
            # per-subclass
            "subclass_labels": labels_present_sub,
            "per_subclass_accuracy": acc_s,
            "per_subclass_accuracy_ci": acc_s_ci,  # per-label accuracy
            "per_subclass_precision": prec_s.tolist() if hasattr(prec_s, "tolist") else [],
            "per_subclass_precision_ci": prec_s_ci,
            "per_subclass_recall": rec_s.tolist() if hasattr(rec_s, "tolist") else [],
            "per_subclass_recall_ci": rec_s_ci,
            "per_subclass_f1": f1_s.tolist() if hasattr(f1_s, "tolist") else [],
            "per_subclass_f1_ci": f1_s_ci,
            "per_subclass_f1_abstention": f1_s_abstention.tolist() if hasattr(f1_s_abstention, "tolist") else [],
            "per_subclass_f1_abstention_ci": f1_s_abstention_ci,
            "per_subclass_support": support_s,
            # per-modality
            "modality_labels": labels_present_mod,
            "per_modality_accuracy": acc_m,
            "per_modality_accuracy_ci": acc_m_ci,  # per-label accuracy
            "per_modality_precision": prec_m.tolist() if hasattr(prec_m, "tolist") else [],
            "per_modality_precision_ci": prec_m_ci,
            "per_modality_recall": rec_m.tolist() if hasattr(rec_m, "tolist") else [],
            "per_modality_recall_ci": rec_m_ci,
            "per_modality_f1": f1_m.tolist() if hasattr(f1_m, "tolist") else [],
            "per_modality_f1_ci": f1_m_ci,
            "per_modality_f1_abstention": f1_m_abstention.tolist() if hasattr(f1_m_abstention, "tolist") else [],
            "per_modality_f1_abstention_ci": f1_m_abstention_ci,
            "per_modality_support": support_m,
            # per-modality-subtype
            "modality_sub_labels": labels_present_mod_sub,
            "per_modality_sub_accuracy": acc_m_sub,
            "per_modality_sub_accuracy_ci": acc_m_sub_ci,  # per-label accuracy
            "per_modality_sub_precision": prec_m_sub.tolist() if hasattr(prec_m_sub, "tolist") else [],
            "per_modality_sub_precision_ci": prec_m_sub_ci,
            "per_modality_sub_recall": rec_m_sub.tolist() if hasattr(rec_m_sub, "tolist") else [],
            "per_modality_sub_recall_ci": rec_m_sub_ci,
            "per_modality_sub_f1": f1_m_sub.tolist() if hasattr(f1_m_sub, "tolist") else [],
            "per_modality_sub_f1_ci": f1_m_sub_ci,
            "per_modality_sub_f1_abstention": f1_m_sub_abstention.tolist() if hasattr(f1_m_sub_abstention, "tolist") else [],
            "per_modality_sub_f1_abstention_ci": f1_m_sub_abstention_ci,
            "per_modality_sub_support": support_m_sub,
            # per-plane
            "plane_labels": labels_present_pl,
            "per_plane_accuracy": acc_p,
            "per_plane_accuracy_ci": acc_p_ci,  # per-label accuracy
            "per_plane_precision": prec_p.tolist() if hasattr(prec_p, "tolist") else [],
            "per_plane_precision_ci": prec_p_ci,
            "per_plane_recall": rec_p.tolist() if hasattr(rec_p, "tolist") else [],
            "per_plane_recall_ci": rec_p_ci,
            "per_plane_f1": f1_p.tolist() if hasattr(f1_p, "tolist") else [],
            "per_plane_f1_ci": f1_p_ci,
            "per_plane_f1_abstention": f1_p_abstention.tolist() if hasattr(f1_p_abstention, "tolist") else [],
            "per_plane_f1_abstention_ci": f1_p_abstention_ci,
            "per_plane_support": support_p,
        }

        # add overall metrics and additional confusion matrices
        results["per_model"][model].update({
            # diagnosis_name overall
            "auc": auc_cls,
            "overall_class_accuracy": overall_acc_cls,
            "overall_class_accuracy_ci": overall_acc_cls_ci,
            "overall_class_precision": overall_prec_macro_cls,
            "overall_class_precision_ci": overall_prec_macro_cls_ci,
            "overall_class_recall": overall_rec_macro_cls,
            "overall_class_recall_ci": overall_rec_macro_cls_ci,
            "overall_class_f1": overall_f1_macro_cls,
            "overall_class_f1_ci": overall_f1_macro_cls_ci,
            "overall_class_f1_abstention": overall_f1_macro_abstention_cls,
            "overall_class_f1_abstention_ci": overall_f1_macro_abstention_cls_ci,
            "overall_class_weighted_f1": overall_f1_weighted_cls,
            "overall_class_weighted_f1_ci": overall_f1_weighted_cls_ci,
            "overall_class_weighted_f1_abstention": overall_f1_weighted_abstention_cls,
            "overall_class_weighted_f1_abstention_ci": overall_f1_weighted_abstention_cls_ci,
            "overall_class_auc": auc_cls,
            "overall_class_brier": brier,
            "overall_class_ece": ece_val,
            # subclass overall
            "overall_sub_accuracy": overall_acc_sub,
            "overall_sub_accuracy_ci": overall_acc_sub_ci,
            "overall_sub_precision": overall_prec_macro_sub,
            "overall_sub_precision_ci": overall_prec_macro_sub_ci,
            "overall_sub_recall": overall_rec_macro_sub,
            "overall_sub_recall_ci": overall_rec_macro_sub_ci,
            "overall_sub_f1": overall_f1_macro_sub,
            "overall_sub_f1_ci": overall_f1_macro_sub_ci,
            "overall_sub_f1_abstention": overall_f1_macro_abstention_sub,
            "overall_sub_f1_abstention_ci": overall_f1_macro_abstention_sub_ci,
            "overall_sub_weighted_f1": overall_f1_weighted_sub,
            "overall_sub_weighted_f1_ci": overall_f1_weighted_sub_ci,
            "overall_sub_weighted_f1_abstention": overall_f1_weighted_abstention_sub,
            "overall_sub_weighted_f1_abstention_ci": overall_f1_weighted_abstention_sub_ci,
            "overall_sub_auc": auc_sub,
            "overall_sub_brier": brier_sub,
            "overall_sub_ece": ece_sub,
            "cm_sub": cm_sub,
            # modality overall
            "overall_mod_accuracy": overall_acc_mod,
            "overall_mod_accuracy_ci": overall_acc_mod_ci,
            "overall_mod_precision": overall_prec_macro_mod,
            "overall_mod_precision_ci": overall_prec_macro_mod_ci,
            "overall_mod_recall": overall_rec_macro_mod,
            "overall_mod_recall_ci": overall_rec_macro_mod_ci,
            "overall_mod_f1": overall_f1_macro_mod,
            "overall_mod_f1_ci": overall_f1_macro_mod_ci,
            "overall_mod_f1_abstention": overall_f1_macro_abstention_mod,
            "overall_mod_f1_abstention_ci": overall_f1_macro_abstention_mod_ci,
            "overall_mod_weighted_f1": overall_f1_weighted_mod,
            "overall_mod_weighted_f1_ci": overall_f1_weighted_mod_ci,
            "overall_mod_weighted_f1_abstention": overall_f1_weighted_abstention_mod,
            "overall_mod_weighted_f1_abstention_ci": overall_f1_weighted_abstention_mod_ci,
            "overall_mod_auc": auc_mod,
            "overall_mod_brier": brier_mod,
            "overall_mod_ece": ece_mod,
            "cm_mod": cm_mod,
            # modality subtype overall
            "overall_mod_sub_accuracy": overall_acc_mod_sub,
            "overall_mod_sub_accuracy_ci": overall_acc_mod_sub_ci,
            "overall_mod_sub_precision": overall_prec_macro_mod_sub,
            "overall_mod_sub_precision_ci": overall_prec_macro_mod_sub_ci,
            "overall_mod_sub_recall": overall_rec_macro_mod_sub,
            "overall_mod_sub_recall_ci": overall_rec_macro_mod_sub_ci,
            "overall_mod_sub_f1": overall_f1_macro_mod_sub,
            "overall_mod_sub_f1_ci": overall_f1_macro_mod_sub_ci,
            "overall_mod_sub_f1_abstention": overall_f1_macro_abstention_mod_sub,
            "overall_mod_sub_f1_abstention_ci": overall_f1_macro_abstention_mod_sub_ci,
            "overall_mod_sub_weighted_f1": overall_f1_weighted_mod_sub,
            "overall_mod_sub_weighted_f1_ci": overall_f1_weighted_mod_sub_ci,
            "overall_mod_sub_weighted_f1_abstention": overall_f1_weighted_abstention_mod_sub,
            "overall_mod_sub_weighted_f1_abstention_ci": overall_f1_weighted_abstention_mod_sub_ci,
            "overall_mod_sub_auc": auc_mod_sub,
            "overall_mod_sub_brier": brier_mod_sub,
            "overall_mod_sub_ece": ece_mod_sub,
            "cm_sub_mod": cm_mod_sub,
            # plane overall
            "overall_pl_accuracy": overall_acc_pl,
            "overall_pl_accuracy_ci": overall_acc_pl_ci,
            "overall_pl_precision": overall_prec_macro_pl,
            "overall_pl_precision_ci": overall_prec_macro_pl_ci,
            "overall_pl_recall": overall_rec_macro_pl,
            "overall_pl_recall_ci": overall_rec_macro_pl_ci,
            "overall_pl_f1": overall_f1_macro_pl,
            "overall_pl_f1_ci": overall_f1_macro_pl_ci,
            "overall_pl_f1_abstention": overall_f1_macro_abstention_pl,
            "overall_pl_f1_abstention_ci": overall_f1_macro_abstention_pl_ci,
            "overall_pl_weighted_f1": overall_f1_weighted_pl,
            "overall_pl_weighted_f1_ci": overall_f1_weighted_pl_ci,
            "overall_pl_weighted_f1_abstention": overall_f1_weighted_abstention_pl,
            "overall_pl_weighted_f1_abstention_ci": overall_f1_weighted_abstention_pl_ci,
            "overall_pl_auc": auc_pl,
            "overall_pl_brier": brier_pl,
            "overall_pl_ece": ece_pl,
            "cm_pl": cm_pl,
        })

    return results

# ----------------------------
# CSV writer (single file with header sections)
# ----------------------------
def write_csv(results: Dict[str, Any], out_path: str) -> None:
    lines: List[str] = []

    # ---- Model-level summary ----
    lines.append("# Model-level Metrics")
    header = [
        "model",
        "macro_f1",
        "macro_f1_ci",
        "micro_f1",
        "micro_f1_ci",
        "weighted_f1",
        "weighted_f1_ci",
        "macro_f1_abstention",
        "macro_f1_abstention_ci",
        "micro_f1_abstention",
        "micro_f1_abstention_ci",
        "weighted_f1_abstention",
        "weighted_f1_abstention_ci",
        "accuracy",
        "accuracy_ci",
        "ece",
        "brier",
        "json_valid_pct",
        "median_latency_ms",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
        "tokens_per_1k_images",
        "usd_cost_per_1k_images",
        "avg_input_cost",
        "avg_output_cost",
        "avg_cost",
        "uncertainty",
        "macro_f1_global",
        "micro_f1_global",
        "weighted_f1_global",
        "accuracy_global",
        "hamming_loss_global",
    ]
    lines.append(",".join(header))
    for model, m in results["per_model"].items():
        row = [
            model,
            f"{m['macro_f1']:.6f}" if not math.isnan(m["macro_f1"]) else "",
            f"{m['macro_f1_ci'][0]:.6f} ({m['macro_f1_ci'][1][0]:.6f} - {m['macro_f1_ci'][1][1]:.6f})",
            f"{m['micro_f1']:.6f}" if not math.isnan(m["micro_f1"]) else "",
            f"{m['micro_f1_ci'][0]:.6f} ({m['micro_f1_ci'][1][0]:.6f} - {m['micro_f1_ci'][1][1]:.6f})",
            f"{m['weighted_f1']:.6f}" if not math.isnan(m["weighted_f1"]) else "",
            f"{m['weighted_f1_ci'][0]:.6f} ({m['weighted_f1_ci'][1][0]:.6f} - {m['weighted_f1_ci'][1][1]:.6f})",
            f"{m['macro_f1_abstention']:.6f}" if not math.isnan(m["macro_f1_abstention"]) else "",
            f"{m['macro_f1_abstention_ci'][0]:.6f} ({m['macro_f1_abstention_ci'][1][0]:.6f} - {m['macro_f1_abstention_ci'][1][1]:.6f})",
            f"{m['micro_f1_abstention']:.6f}" if not math.isnan(m["micro_f1_abstention"]) else "",
            f"{m['micro_f1_abstention_ci'][0]:.6f} ({m['micro_f1_abstention_ci'][1][0]:.6f} - {m['micro_f1_abstention_ci'][1][1]:.6f})",
            f"{m['weighted_f1_abstention']:.6f}" if not math.isnan(m["weighted_f1_abstention"]) else "",
            f"{m['weighted_f1_abstention_ci'][0]:.6f} ({m['weighted_f1_abstention_ci'][1][0]:.6f} - {m['weighted_f1_abstention_ci'][1][1]:.6f})", 
            f"{m['accuracy']:.6f}" if not math.isnan(m["accuracy"]) else "",
            f"{m['accuracy_ci'][0]:.6f} ({m['accuracy_ci'][1][0]:.6f} - {m['accuracy_ci'][1][1]:.6f})",
            f"{m['ece']:.6f}" if not (m["ece"] != m["ece"]) else "",
            f"{m['brier']:.6f}" if not (m["brier"] != m["brier"]) else "",
            f"{m['json_valid_pct']:.2f}",
            f"{m['median_latency_ms']:.2f}" if not (m["median_latency_ms"] != m["median_latency_ms"]) else "",
            f"{m['avg_input_tokens']:.2f}",
            f"{m['avg_output_tokens']:.2f}",
            f"{m['avg_total_tokens']:.2f}",
            f"{m['tokens_per_1k_images']:.2f}",
            f"{m['usd_cost_per_1k_images']:.4f}",
            f"{m['avg_input_cost']:.6f}",
            f"{m['avg_output_cost']:.6f}",
            f"{m['avg_cost']:.6f}",
            f"{m['uncertainty']}",
            f"{m['macro_f1_global']:.6f}" if not math.isnan(m["macro_f1_global"]) else "",
            f"{m['micro_f1_global']:.6f}" if not math.isnan(m["micro_f1_global"]) else "",
            f"{m['weighted_f1_global']:.6f}" if not math.isnan(m["weighted_f1_global"]) else "",
            f"{m['acc_global']:.6f}" if not math.isnan(m["acc_global"]) else "",
            f"{m['hamming_loss_global']:.6f}" if not math.isnan(m["hamming_loss_global"]) else "",
            
        ]
        lines.append(",".join(row))

    # ---- Overall per-Model sections ----
    def fmt(x, nd=6):
        try:
            if x != x:
                return ""
            return f"{x:.{nd}f}"
        except Exception:
            return ""

    # diagnosis_name overall
    lines.append("")
    lines.append("# Overall per-Model (diagnosis_name)")
    lines.append("model,accuracy,accuracy_ci,precision_macro,precision_macro_ci,recall_macro,recall_macro_ci,f1_macro,f1_macro_ci,f1_macro_abstention,f1_macro_abstention_ci,f1_weighted,f1_weighted_ci,f1_weighted_abstention,f1_weighted_abstention_ci,auc,brier,ece")
    for model, m in results["per_model"].items():
        lines.append(
            ",".join([
                model,
                fmt(m.get("overall_class_accuracy")),
                f'{m.get("overall_class_accuracy_ci")[0]:.6f} ({m.get("overall_class_accuracy_ci")[1][0]:.6f} - {m.get("overall_class_accuracy_ci")[1][1]:.6f})',
                fmt(m.get("overall_class_precision")),
                f'{m.get("overall_class_precision_ci")[0]:.6f} ({m.get("overall_class_precision_ci")[1][0]:.6f} - {m.get("overall_class_precision_ci")[1][1]:.6f})',
                fmt(m.get("overall_class_recall")),
                f'{m.get("overall_class_recall_ci")[0]:.6f} ({m.get("overall_class_recall_ci")[1][0]:.6f} - {m.get("overall_class_recall_ci")[1][1]:.6f})',
                fmt(m.get("overall_class_f1")),
                f'{m.get("overall_class_f1_ci")[0]:.6f} ({m.get("overall_class_f1_ci")[1][0]:.6f} - {m.get("overall_class_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_class_f1_abstention")),
                f'{m.get("overall_class_f1_abstention_ci")[0]:.6f} ({m.get("overall_class_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_class_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_class_weighted_f1")),
                f'{m.get("overall_class_weighted_f1_ci")[0]:.6f} ({m.get("overall_class_weighted_f1_ci")[1][0]:.6f} - {m.get("overall_class_weighted_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_class_weighted_f1_abstention")),
                f'{m.get("overall_class_weighted_f1_abstention_ci")[0]:.6f} ({m.get("overall_class_weighted_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_class_weighted_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_class_auc")),
                fmt(m.get("overall_class_brier")),
                fmt(m.get("overall_class_ece")),
            ])
        )

    # subclass overall
    lines.append("")
    lines.append("# Overall per-Model (diagnosis_detailed)")
    lines.append("model,accuracy,accuracy_ci,precision_macro,precision_macro_ci,recall_macro,recall_macro_ci,f1_macro,f1_macro_ci,f1_macro_abstention,f1_macro_abstention_ci,f1_weighted,f1_weighted_ci,f1_weighted_abstention,f1_weighted_abstention_ci,auc,brier,ece")
    for model, m in results["per_model"].items():
        lines.append(
            ",".join([
                model,
                fmt(m.get("overall_sub_accuracy")),
                f'{m.get("overall_sub_accuracy_ci")[0]:.6f} ({m.get("overall_sub_accuracy_ci")[1][0]:.6f} - {m.get("overall_sub_accuracy_ci")[1][1]:.6f})',
                fmt(m.get("overall_sub_precision")),
                f'{m.get("overall_sub_precision_ci")[0]:.6f} ({m.get("overall_sub_precision_ci")[1][0]:.6f} - {m.get("overall_sub_precision_ci")[1][1]:.6f})',
                fmt(m.get("overall_sub_recall")),
                f'{m.get("overall_sub_recall_ci")[0]:.6f} ({m.get("overall_sub_recall_ci")[1][0]:.6f} - {m.get("overall_sub_recall_ci")[1][1]:.6f})',
                fmt(m.get("overall_sub_f1")),
                f'{m.get("overall_sub_f1_ci")[0]:.6f} ({m.get("overall_sub_f1_ci")[1][0]:.6f} - {m.get("overall_sub_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_sub_f1_abstention")),
                f'{m.get("overall_sub_f1_abstention_ci")[0]:.6f} ({m.get("overall_sub_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_sub_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_sub_weighted_f1")),
                f'{m.get("overall_sub_weighted_f1_ci")[0]:.6f} ({m.get("overall_sub_weighted_f1_ci")[1][0]:.6f} - {m.get("overall_sub_weighted_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_sub_weighted_f1_abstention")),
                f'{m.get("overall_sub_weighted_f1_abstention_ci")[0]:.6f} ({m.get("overall_sub_weighted_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_sub_weighted_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_sub_auc")),
                fmt(m.get("overall_sub_brier")),
                fmt(m.get("overall_sub_ece")),
            ])
        )

    # modality overall
    lines.append("")
    lines.append("# Overall per-Model (modality)")
    lines.append("model,accuracy,accuracy_ci,precision_macro,precision_macro_ci,recall_macro,recall_macro_ci,f1_macro,f1_macro_ci,f1_macro_abstention,f1_macro_abstention_ci,f1_weighted,f1_weighted_ci,f1_weighted_abstention,f1_weighted_abstention_ci,auc,brier,ece")
    for model, m in results["per_model"].items():
        lines.append(
            ",".join([
                model,
                fmt(m.get("overall_mod_accuracy")),
                f'{m.get("overall_mod_accuracy_ci")[0]:.6f} ({m.get("overall_mod_accuracy_ci")[1][0]:.6f} - {m.get("overall_mod_accuracy_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_precision")),
                f'{m.get("overall_mod_precision_ci")[0]:.6f} ({m.get("overall_mod_precision_ci")[1][0]:.6f} - {m.get("overall_mod_precision_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_recall")),
                f'{m.get("overall_mod_recall_ci")[0]:.6f} ({m.get("overall_mod_recall_ci")[1][0]:.6f} - {m.get("overall_mod_recall_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_f1")),
                f'{m.get("overall_mod_f1_ci")[0]:.6f} ({m.get("overall_mod_f1_ci")[1][0]:.6f} - {m.get("overall_mod_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_f1_abstention")),
                f'{m.get("overall_mod_f1_abstention_ci")[0]:.6f} ({m.get("overall_mod_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_mod_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_weighted_f1")),
                f'{m.get("overall_mod_weighted_f1_ci")[0]:.6f} ({m.get("overall_mod_weighted_f1_ci")[1][0]:.6f} - {m.get("overall_mod_weighted_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_weighted_f1_abstention")),
                f'{m.get("overall_mod_weighted_f1_abstention_ci")[0]:.6f} ({m.get("overall_mod_weighted_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_mod_weighted_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_auc")),
                fmt(m.get("overall_mod_brier")),
                fmt(m.get("overall_mod_ece")),
            ])
        )
    
    # modality subtype overall
    lines.append("")
    lines.append("# Overall per-Model (specialized_sequence)")
    lines.append("model,accuracy,accuracy_ci,precision_macro,precision_macro_ci,recall_macro,recall_macro_ci,f1_macro,f1_macro_ci,f1_macro_abstention,f1_macro_abstention_ci,f1_weighted,f1_weighted_ci,f1_weighted_abstention,f1_weighted_abstention_ci,auc,brier,ece")
    for model, m in results["per_model"].items():
        lines.append(
            ",".join([
                model,
                fmt(m.get("overall_mod_sub_accuracy")),
                f'{m.get("overall_mod_sub_accuracy_ci")[0]:.6f} ({m.get("overall_mod_sub_accuracy_ci")[1][0]:.6f} - {m.get("overall_mod_sub_accuracy_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_sub_precision")),
                f'{m.get("overall_mod_sub_precision_ci")[0]:.6f} ({m.get("overall_mod_sub_precision_ci")[1][0]:.6f} - {m.get("overall_mod_sub_precision_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_sub_recall")),
                f'{m.get("overall_mod_sub_recall_ci")[0]:.6f} ({m.get("overall_mod_sub_recall_ci")[1][0]:.6f} - {m.get("overall_mod_sub_recall_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_sub_f1")),
                f'{m.get("overall_mod_sub_f1_ci")[0]:.6f} ({m.get("overall_mod_sub_f1_ci")[1][0]:.6f} - {m.get("overall_mod_sub_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_sub_f1_abstention")),
                f'{m.get("overall_mod_sub_f1_abstention_ci")[0]:.6f} ({m.get("overall_mod_sub_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_mod_sub_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_sub_weighted_f1")),
                f'{m.get("overall_mod_sub_weighted_f1_ci")[0]:.6f} ({m.get("overall_mod_sub_weighted_f1_ci")[1][0]:.6f} - {m.get("overall_mod_sub_weighted_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_sub_weighted_f1_abstention")),
                f'{m.get("overall_mod_sub_weighted_f1_abstention_ci")[0]:.6f} ({m.get("overall_mod_sub_weighted_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_mod_sub_weighted_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_mod_sub_auc")),
                fmt(m.get("overall_mod_sub_brier")),
                fmt(m.get("overall_mod_sub_ece")),
            ])
        )

    # plane overall
    lines.append("")
    lines.append("# Overall per-Model (plane)")
    lines.append("model,accuracy,accuracy_ci,precision_macro,precision_macro_ci,recall_macro,recall_macro_ci,f1_macro,f1_macro_ci,f1_macro_abstention,f1_macro_abstention_ci,f1_weighted,f1_weighted_ci,f1_weighted_abstention,f1_weighted_abstention_ci,auc,brier,ece")
    for model, m in results["per_model"].items():
        lines.append(
            ",".join([
                model,
                fmt(m.get("overall_pl_accuracy")),
                f'{m.get("overall_pl_accuracy_ci")[0]:.6f} ({m.get("overall_pl_accuracy_ci")[1][0]:.6f} - {m.get("overall_pl_accuracy_ci")[1][1]:.6f})',
                fmt(m.get("overall_pl_precision")),
                f'{m.get("overall_pl_precision_ci")[0]:.6f} ({m.get("overall_pl_precision_ci")[1][0]:.6f} - {m.get("overall_pl_precision_ci")[1][1]:.6f})',
                fmt(m.get("overall_pl_recall")),
                f'{m.get("overall_pl_recall_ci")[0]:.6f} ({m.get("overall_pl_recall_ci")[1][0]:.6f} - {m.get("overall_sub_recall_ci")[1][1]:.6f})',
                fmt(m.get("overall_pl_f1")),
                f'{m.get("overall_pl_f1_ci")[0]:.6f} ({m.get("overall_pl_f1_ci")[1][0]:.6f} - {m.get("overall_pl_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_pl_f1_abstention")),
                f'{m.get("overall_pl_f1_abstention_ci")[0]:.6f} ({m.get("overall_pl_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_pl_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_pl_weighted_f1")),
                f'{m.get("overall_pl_weighted_f1_ci")[0]:.6f} ({m.get("overall_pl_weighted_f1_ci")[1][0]:.6f} - {m.get("overall_pl_weighted_f1_ci")[1][1]:.6f})',
                fmt(m.get("overall_pl_weighted_f1_abstention")),
                f'{m.get("overall_pl_weighted_f1_abstention_ci")[0]:.6f} ({m.get("overall_pl_weighted_f1_abstention_ci")[1][0]:.6f} - {m.get("overall_pl_weighted_f1_abstention_ci")[1][1]:.6f})',
                fmt(m.get("overall_pl_auc")),
                fmt(m.get("overall_pl_brier")),
                fmt(m.get("overall_pl_ece")),
            ])
        )

    # ---- Per-class metrics ----
    #lines.append("")
    #lines.append("# Per-Class Metrics (diagnosis_name)")
    #lines.append("model,class,accuracy,precision,recall,f1,support")
    #for model, m in results["per_model"].items():  
        #labels = m.get("labels", [])
        #for i, lab in enumerate(labels):
            #accv = m["per_class_accuracy"][i] if i < len(m["per_class_accuracy"]) else ""
            #prec = m["per_class_precision"][i] if i < len(m["per_class_precision"]) else ""
            #rec = m["per_class_recall"][i] if i < len(m["per_class_recall"]) else ""
            #f1v = m["per_class_f1"][i] if i < len(m["per_class_f1"]) else ""
            #sup = m["per_class_support"][i] if i < len(m["per_class_support"]) else ""
            #lines.append(
                #f"{model},{lab},{accv:.6f},{prec:.6f},{rec:.6f},{f1v:.6f},{sup}"
            #)
            
    # ---- Per-class metrics ----
    lines.append("")
    lines.append("# Per-Class Metrics (diagnosis_name)")
    lines.append("model,class,accuracy,accuracy_ci,precision,precision_ci,recall,recall_ci,f1,f1_ci,f1_abstention,f1_abstention_ci,support")

    # collect global set of labels (union from all models)
    all_labels = set()
    for m in results["per_model"].values():
        all_labels.update(m.get("labels", []))
    all_labels = list(all_labels)

    # iterate class-first
    for lab in all_labels:
        for model, m in results["per_model"].items():
            labels = m.get("labels", [])
            if lab in labels:
                i = labels.index(lab)
                accv = m["per_class_accuracy"][i] if i < len(m["per_class_accuracy"]) else float("nan")
                accv_ci = m["per_class_accuracy_ci"][i]
                
                prec = m["per_class_precision"][i] if i < len(m["per_class_precision"]) else float("nan")
                prec_ci = m["per_class_precision_ci"][i] if i < len(m["per_class_precision_ci"]) else float("nan")
                
                rec  = m["per_class_recall"][i] if i < len(m["per_class_recall"]) else float("nan")
                rec_ci  = m["per_class_recall_ci"][i] if i < len(m["per_class_recall_ci"]) else float("nan")
                
                f1v  = m["per_class_f1"][i] if i < len(m["per_class_f1"]) else float("nan")
                f1v_ci  = m["per_class_f1_ci"][i] if i < len(m["per_class_f1_ci"]) else float("nan")
                 
                f1v_a  = m["per_class_f1_abstention"][i] if i < len(m["per_class_f1_abstention"]) else float("nan")
                
                f1v_a_ci  = m["per_class_f1_abstention_ci"][i] if i < len(m["per_class_f1_abstention_ci"]) else float("nan")
                
                sup  = m["per_class_support"][i] if i < len(m["per_class_support"]) else ""
                
                if i < len(m["per_class_f1_abstention_ci"]):
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},{f1v_a_ci[0]:.6f} ({f1v_a_ci[1][0]:.6f} - {f1v_a_ci[1][1]:.6f}),{sup}"
                    )
                else:
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},nan,{sup}"
                )

    # ---- Per-subclass metrics ----
    #lines.append("")
    #lines.append("# Per-Subclass Metrics (diagnosis_detailed)")
    #lines.append("model,subclass,accuracy,precision,recall,f1,support")
    #for model, m in results["per_model"].items():
        #labels = m.get("subclass_labels", [])
        #for i, lab in enumerate(labels):
            #accv = m["per_subclass_accuracy"][i] if i < len(m["per_subclass_accuracy"]) else ""
            #prec = m["per_subclass_precision"][i] if i < len(m["per_subclass_precision"]) else ""
            #rec = m["per_subclass_recall"][i] if i < len(m["per_subclass_recall"]) else ""
            #f1v = m["per_subclass_f1"][i] if i < len(m["per_subclass_f1"]) else ""
            #sup = m["per_subclass_support"][i] if i < len(m["per_subclass_support"]) else ""
            #lines.append(
                #f"{model},{lab},{accv:.6f},{prec:.6f},{rec:.6f},{f1v:.6f},{sup}"
            #)

    # ---- Per-Subclass Metrics ----
    lines.append("")
    lines.append("# Per-Subclass Metrics (diagnosis_detailed)")
    lines.append("model,subclass,accuracy,accuracy_ci,precision,precision_ci,recall,recall_ci,f1,f1_ci,f1_abstention,f1_abstention_ci,support")

    # collect global set of subclasses (union from all models)
    all_subclasses = set()
    for m in results["per_model"].values():
        all_subclasses.update(m.get("subclass_labels", []))
    all_subclasses = list(all_subclasses)

    # iterate subclass-first
    for lab in all_subclasses:
        for model, m in results["per_model"].items():
            labels = m.get("subclass_labels", [])
            if lab in labels:
                i = labels.index(lab)
                accv = m["per_subclass_accuracy"][i] if i < len(m["per_subclass_accuracy"]) else float("nan")
                accv_ci = m["per_subclass_accuracy_ci"][i]
                
                prec = m["per_subclass_precision"][i] if i < len(m["per_subclass_precision"]) else float("nan")
                prec_ci = m["per_subclass_precision_ci"][i] if i < len(m["per_subclass_precision_ci"]) else float("nan")
                
                rec  = m["per_subclass_recall"][i] if i < len(m["per_subclass_recall"]) else float("nan")
                rec_ci  = m["per_subclass_recall_ci"][i] if i < len(m["per_subclass_recall_ci"]) else float("nan")
                
                f1v  = m["per_subclass_f1"][i] if i < len(m["per_subclass_f1"]) else float("nan")
                f1v_ci  = m["per_subclass_f1_ci"][i] if i < len(m["per_subclass_f1_ci"]) else float("nan")
                 
                f1v_a  = m["per_subclass_f1_abstention"][i] if i < len(m["per_subclass_f1_abstention"]) else float("nan")
                
                f1v_a_ci  = m["per_subclass_f1_abstention_ci"][i] if i < len(m["per_subclass_f1_abstention_ci"]) else float("nan")
                
                sup  = m["per_subclass_support"][i] if i < len(m["per_subclass_support"]) else ""
                
                if i < len(m["per_subclass_f1_abstention_ci"]):
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},{f1v_a_ci[0]:.6f} ({f1v_a_ci[1][0]:.6f} - {f1v_a_ci[1][1]:.6f}),{sup}"
                    )
                else:
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},nan,{sup}"
                )

    
    # ---- Per-modality metrics ----
    #lines.append("")
    #lines.append("# Per-Modality Metrics (modality)")
    #lines.append("model,modality,accuracy,precision,recall,f1,support")
    #for model, m in results["per_model"].items():
        #labels = m.get("modality_labels", [])
        #for i, lab in enumerate(labels):
            #accv = m["per_modality_accuracy"][i] if i < len(m["per_modality_accuracy"]) else ""
            #prec = m["per_modality_precision"][i] if i < len(m["per_modality_precision"]) else ""
            #rec = m["per_modality_recall"][i] if i < len(m["per_modality_recall"]) else ""
            #f1v = m["per_modality_f1"][i] if i < len(m["per_modality_f1"]) else ""
            #sup = m["per_modality_support"][i] if i < len(m["per_modality_support"]) else ""
            #lines.append(
                #f"{model},{lab},{accv:.6f},{prec:.6f},{rec:.6f},{f1v:.6f},{sup}"
            #)
    
    # ---- Per-Modality Metrics ----
    lines.append("")
    lines.append("# Per-Modality Metrics (modality)")
    lines.append("model,modality,accuracy,accuracy_ci,precision,precision_ci,recall,recall_ci,f1,f1_ci,f1_abstention,f1_abstention_ci,support")

    # collect global set of modalities (union from all models)
    all_modalities = set()
    for m in results["per_model"].values():
        all_modalities.update(m.get("modality_labels", []))
        
    all_modalities = list(all_modalities)

    # iterate modality-first
    for lab in all_modalities:
        for model, m in results["per_model"].items():
            labels = m.get("modality_labels", [])
            if lab in labels:
                i = labels.index(lab)
                accv = m["per_modality_accuracy"][i] if i < len(m["per_modality_accuracy"]) else float("nan")
                accv_ci = m["per_modality_accuracy_ci"][i]
                
                prec = m["per_modality_precision"][i] if i < len(m["per_modality_precision"]) else float("nan")
                prec_ci = m["per_modality_precision_ci"][i] if i < len(m["per_modality_precision_ci"]) else float("nan")
                
                rec  = m["per_modality_recall"][i] if i < len(m["per_modality_recall"]) else float("nan")
                rec_ci  = m["per_modality_recall_ci"][i] if i < len(m["per_modality_recall_ci"]) else float("nan")
                
                f1v  = m["per_modality_f1"][i] if i < len(m["per_modality_f1"]) else float("nan")
                f1v_ci  = m["per_modality_f1_ci"][i] if i < len(m["per_modality_f1_ci"]) else float("nan")
                 
                f1v_a  = m["per_modality_f1_abstention"][i] if i < len(m["per_modality_f1_abstention"]) else float("nan")
                
                f1v_a_ci  = m["per_modality_f1_abstention_ci"][i] if i < len(m["per_modality_f1_abstention_ci"]) else float("nan")
                
                sup  = m["per_modality_support"][i] if i < len(m["per_modality_support"]) else ""
                
                if i < len(m["per_modality_f1_abstention_ci"]):
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},{f1v_a_ci[0]:.6f} ({f1v_a_ci[1][0]:.6f} - {f1v_a_ci[1][1]:.6f}),{sup}"
                    )
                else:
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},nan,{sup}"
                )
                
    
    # ---- Per-modality subtype metrics ----
    #lines.append("")
    #lines.append("# Per-Modality Subtype Metrics (specialized_sequence)")
    #lines.append("model,modality,accuracy,precision,recall,f1,support")
    #for model, m in results["per_model"].items():
        #labels = m.get("modality_sub_labels", [])
        #for i, lab in enumerate(labels):
            #accv = m["per_modality_sub_accuracy"][i] if i < len(m["per_modality_sub_accuracy"]) else ""
            #prec = m["per_modality_sub_precision"][i] if i < len(m["per_modality_sub_precision"]) else ""
            #rec = m["per_modality_sub_recall"][i] if i < len(m["per_modality_sub_recall"]) else ""
            #f1v = m["per_modality_sub_f1"][i] if i < len(m["per_modality_sub_f1"]) else ""
            #sup = m["per_modality_sub_support"][i] if i < len(m["per_modality_sub_support"]) else ""
            #lines.append(
                #f"{model},{lab},{accv:.6f},{prec:.6f},{rec:.6f},{f1v:.6f},{sup}"
            #)
    
    # ---- Per-Modality sybtype Metrics ----
    lines.append("")
    lines.append("# Per-Modality Subtype Metrics (specialized_sequence)")
    lines.append("model,modality_subtype,accuracy,accuracy_ci,precision,precision_ci,recall,recall_ci,f1,f1_ci,f1_abstention,f1_abstention_ci,support")

    # collect global set of modalities (union from all models)
    all_modalities_sub = set()
    for m in results["per_model"].values():
        all_modalities_sub.update(m.get("modality_sub_labels", []))
        
    all_modalities_sub = list(all_modalities_sub)

    # iterate modality-first
    for lab in all_modalities_sub:
        for model, m in results["per_model"].items():
            labels = m.get("modality_sub_labels", [])
            if lab in labels:
                i = labels.index(lab)
                accv = m["per_modality_sub_accuracy"][i] if i < len(m["per_modality_sub_accuracy"]) else float("nan")
                accv_ci = m["per_modality_sub_accuracy_ci"][i]
                
                prec = m["per_modality_sub_precision"][i] if i < len(m["per_modality_sub_precision"]) else float("nan")
                prec_ci = m["per_modality_sub_precision_ci"][i] if i < len(m["per_modality_sub_precision_ci"]) else float("nan")
                
                rec  = m["per_modality_sub_recall"][i] if i < len(m["per_modality_sub_recall"]) else float("nan")
                rec_ci  = m["per_modality_sub_recall_ci"][i] if i < len(m["per_modality_sub_recall_ci"]) else float("nan")
                
                f1v  = m["per_modality_sub_f1"][i] if i < len(m["per_modality_sub_f1"]) else float("nan")
                f1v_ci  = m["per_modality_sub_f1_ci"][i] if i < len(m["per_modality_sub_f1_ci"]) else float("nan")
                 
                f1v_a  = m["per_modality_sub_f1_abstention"][i] if i < len(m["per_modality_sub_f1_abstention"]) else float("nan")
                
                f1v_a_ci  = m["per_modality_sub_f1_abstention_ci"][i] if i < len(m["per_modality_sub_f1_abstention_ci"]) else float("nan")
                
                sup  = m["per_modality_sub_support"][i] if i < len(m["per_modality_sub_support"]) else ""
                
                if i < len(m["per_modality_sub_f1_abstention_ci"]):
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},{f1v_a_ci[0]:.6f} ({f1v_a_ci[1][0]:.6f} - {f1v_a_ci[1][1]:.6f}),{sup}"
                    )
                else:
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},nan,{sup}"
                )
 
    # ---- Per-plane metrics ----
    #lines.append("")
    #lines.append("# Per-Plane Metrics (axial_plane vs plane)")
    #lines.append("model,plane,accuracy,precision,recall,f1,support")
    #for model, m in results["per_model"].items():
        #labels = m.get("plane_labels", [])
        # NOTE: we stored precision for plane in 'per_plane_precision' mistakenly as rec earlier; fix robustly:
        #per_plane_precision = m.get("per_plane_precision")
        #if per_plane_precision and len(per_plane_precision) == len(labels):
            #plane_prec = per_plane_precision
        #else:
            #plane_prec = m.get("per_plane_recall", [])  # fallback (shouldn't happen)
        #for i, lab in enumerate(labels):
            #accv = m["per_plane_accuracy"][i] if i < len(m["per_plane_accuracy"]) else ""
            #prec = plane_prec[i] if i < len(plane_prec) else ""
            #rec = m["per_plane_recall"][i] if i < len(m["per_plane_recall"]) else ""
            #f1v = m["per_plane_f1"][i] if i < len(m["per_plane_f1"]) else ""
            #sup = m["per_plane_support"][i] if i < len(m["per_plane_support"]) else ""
            #lines.append(
                #f"{model},{lab},{accv:.6f},{prec:.6f},{rec:.6f},{f1v:.6f},{sup}"
            #)
    
    # ---- Per-Plane Metrics ----
    lines.append("")
    lines.append("# Per-Plane Metrics (axial_plane vs plane)")
    lines.append("model,lines,accuracy,accuracy_ci,precision,precision_ci,recall,recall_ci,f1,f1_ci,f1_abstention,f1_abstention_ci,support")

    # collect global set of planes (union from all models)
    all_planes = set()
    for m in results["per_model"].values():
        all_planes.update(m.get("plane_labels", []))
    all_planes = list(all_planes)

    # iterate plane-first
    for lab in all_planes:
        for model, m in results["per_model"].items():
            labels = m.get("plane_labels", [])
            if lab in labels:
                i = labels.index(lab)
                
                # get plane precision robustly
                per_plane_precision = m.get("per_plane_precision")
                if per_plane_precision and len(per_plane_precision) == len(labels):
                    plane_prec = per_plane_precision
                else:
                    plane_prec = m.get("per_plane_recall", [])  # fallback
                    
                accv = m["per_plane_accuracy"][i] if i < len(m["per_plane_accuracy"]) else float("nan")
                accv_ci = m["per_plane_accuracy_ci"][i]
                
                prec = m["per_plane_precision"][i] if i < len(m["per_plane_precision"]) else float("nan")
                prec_ci = m["per_plane_precision_ci"][i] if i < len(m["per_plane_precision_ci"]) else float("nan")
                
                rec  = m["per_plane_recall"][i] if i < len(m["per_plane_recall"]) else float("nan")
                rec_ci  = m["per_plane_recall_ci"][i] if i < len(m["per_plane_recall_ci"]) else float("nan")
                
                f1v  = m["per_plane_f1"][i] if i < len(m["per_plane_f1"]) else float("nan")
                f1v_ci  = m["per_plane_f1_ci"][i] if i < len(m["per_plane_f1_ci"]) else float("nan")
                 
                f1v_a  = m["per_plane_f1_abstention"][i] if i < len(m["per_plane_f1_abstention"]) else float("nan")
                
                f1v_a_ci  = m["per_plane_f1_abstention_ci"][i] if i < len(m["per_plane_f1_abstention_ci"]) else float("nan")
                
                sup  = m["per_plane_support"][i] if i < len(m["per_plane_support"]) else ""
                
                if i < len(m["per_plane_f1_abstention_ci"]):
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},{f1v_a_ci[0]:.6f} ({f1v_a_ci[1][0]:.6f} - {f1v_a_ci[1][1]:.6f}),{sup}"
                    )
                else:
                    lines.append(
                        f"{model},{lab},{accv:.6f},{accv_ci[0]:.6f} ({accv_ci[1][0]:.6f} - {accv_ci[1][1]:.6f}),{prec:.6f},{prec_ci[0]:.6f} ({prec_ci[1][0]:.6f} - {prec_ci[1][1]:.6f}),{rec:.6f},{rec_ci[0]:.6f} ({rec_ci[1][0]:.6f} - {rec_ci[1][1]:.6f}),{f1v:.6f},{f1v_ci[0]:.6f} ({f1v_ci[1][0]:.6f} - {f1v_ci[1][1]:.6f}),{f1v_a:.6f},nan,{sup}"
                )
       

    # ---- Confusion matrices ----
    lines.append("")
    lines.append("# confusion Matrix per Model (diagnosis_name)")
    for model, m in results["per_model"].items():
        labels = m.get("labels", [])
        cm = m.get("confusion_matrix", [])
        
        if not labels or not cm:
            continue
        lines.append("")
        lines.append(f"model,{model}")
        lines.append(",".join(["true\\pred"] + labels))
        for i, row in enumerate(cm):
            lines.append(",".join([labels[i]] + [str(v*100)+'%' for v in row]))

    # ---- Additional Confusion Matrices (subclass/modality/plane) ----
    # subclass
    lines.append("")
    lines.append("# Confusion Matrix per Model (diagnosis_detailed)")
    for model, m in results["per_model"].items():
        labels = m.get("subclass_labels", [])
        cm = m.get("cm_sub", [])#.tolist()
        if not labels or not cm:
            continue
        lines.append("")
        lines.append(f"model,{model}")
        lines.append(",".join(["true\\pred"] + labels))
        for i, row in enumerate(cm):
            lines.append(",".join([labels[i]] + [str(v*100)+'%' for v in row]))

    # modality
    lines.append("")
    lines.append("# Confusion Matrix per Model (modality)")
    for model, m in results["per_model"].items():
        labels = m.get("modality_labels", [])
        cm = m.get("cm_mod", [])#.tolist()
        if not labels or not cm:
            continue
        lines.append("")
        lines.append(f"model,{model}")
        lines.append(",".join(["true\\pred"] + labels))
        for i, row in enumerate(cm):
            lines.append(",".join([labels[i]] + [str(v*100)+'%' for v in row]))
    
    # modality subtype
    lines.append("")
    lines.append("# Confusion Matrix per Model (specialized_sequence)")
    for model, m in results["per_model"].items():
        labels = m.get("modality_sub_labels", [])
        cm = m.get("cm_mod_sub", [])#.tolist()
        if not labels or not cm:
            continue
        lines.append("")
        lines.append(f"model,{model}")
        lines.append(",".join(["true\\pred"] + labels))
        for i, row in enumerate(cm):
            lines.append(",".join([labels[i]] + [str(v*100)+'%' for v in row]))

    # plane
    lines.append("")
    lines.append("# Confusion Matrix per Model (plane)")
    for model, m in results["per_model"].items():
        labels = m.get("plane_labels", [])
        cm = m.get("cm_pl", [])#.tolist()
        if not labels or not cm:
            continue
        lines.append("")
        lines.append(f"model,{model}")
        lines.append(",".join(["true\\pred"] + labels))
        for i, row in enumerate(cm):
            lines.append(",".join([labels[i]] + [str(v*100)+'%' for v in row]))

    # ---- Field-wise accuracies (global) ----
    lines.append("")
    lines.append("# Field-wise Accuracy (global)")
    lines.append("field,accuracy_pct,correct,total")
    for f, m in results["field_results"].items():
        lines.append(f"{f},{m['accuracy_pct']:.2f},{m['correct']},{m['total']}")

    # ---- Detailed rows ----
    lines.append("")
    lines.append("# Detailed Results")
    detail_header = [
        "experiment_id",
        "model",
        "image_path",
        "latency_ms",
        "is_valid_json",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        # per-field gt/pred/correct
        "diagnosis_name_gt",
        "diagnosis_name_pred",
        "diagnosis_name_correct",
        "diagnosis_detailed_gt",
        "diagnosis_detailed_pred",
        "diagnosis_detailed_correct",
        "modality_gt",
        "modality_pred",
        "modality_correct",
        "modality_sub_gt",
        "modality_sub_pred",
        "modality_sub_correct",
        "plane_gt",
        "plane_pred",
        "plane_correct",
        "safety_flag",
    ]
    lines.append(",".join(detail_header))


    for s in results["detailed"]:
        fs = s["field_scores"]
        row = [
            str(s.get("experiment_id", "")),
            str(s.get("model", "")),
            str(s.get("image_path", "")),
            "" if s.get("latency_ms") is None else f"{s['latency_ms']:.2f}",
            str(bool(s.get("is_valid_json", False))),
            str(s.get("input_tokens", 0)),
            str(s.get("output_tokens", 0)),
            str(s.get("total_tokens", 0)),
            str(fs.get("diagnosis_name", {}).get("ground_truth", "")),
            str(fs.get("diagnosis_name", {}).get("predicted", "")),
            str(fs.get("diagnosis_name", {}).get("correct", "")),
            str(fs.get("diagnosis_detailed", {}).get("ground_truth", "")),
            str(fs.get("diagnosis_detailed", {}).get("predicted", "")),
            str(fs.get("diagnosis_detailed", {}).get("correct", "")),
            str(fs.get("modality", {}).get("ground_truth", "")),
            str(fs.get("modality", {}).get("predicted", "")),
            str(fs.get("modality", {}).get("correct", "")),
            str(fs.get("specialized_sequence", {}).get("ground_truth", "")),
            str(fs.get("specialized_sequence", {}).get("predicted", "")),
            str(fs.get("specialized_sequence", {}).get("correct", "")),
            str(fs.get("plane", {}).get("ground_truth", "")),
            str(fs.get("plane", {}).get("predicted", "")),
            str(fs.get("plane", {}).get("correct", "")),
            str(s.get("safety_flag", 0)),
        ]
        # escape CSV values that contain commas/newlines/quotes
        row_escaped = []
        for v in row:
            v = "" if v is None else str(v)
            if any(ch in v for ch in [",", "\n", '"']):
                row_escaped.append('"' + v.replace('"', '""') + '"')
            else:
                row_escaped.append(v)
        lines.append(",".join(row_escaped))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------
# CLI
# ----------------------------
def main(): 
    
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    results_dir = parent_dir / 'chatmed' / 'results' 
    
    jsonl_files = [#'subset_1_temp_0_20251121_144713_grok',
                   #'medgemma_4b_temp_0_subset_1_noquant',
                   
                   
                   'subset_1_temp_0_20250823_172529',
                     
                   'subset_4+5+6_temp_0_20250909_122803',
                   
                   'subset_0_25_7_temp_0_20250909_122709',
                   
                   'subset_0_25_7_temp_0_20250909_134613_few_shot',
                   
                   #'subset_2_temp_0_20250820_011422',
                   #'subset_3_temp_0_20250820_101317',
                   
                   #'subset_5_temp_0_20250909_161009',
                   
                   #'subset_4_temp_0_20250909_122803',
                   
                   #'25_subset_0_25_temp_0'
                   
                   
                   
                   
                   #'qwen25vl_32b_subset_1 temp0',
                   #'subset_0_25_7_temp_0_20250909_122709',
                   #'subset_0_25_7_temp_0_20250909_134613_few_shot'
                   
                   #, '25_subset_0_25_temp_0', 
                   
                   
                   #'few_shot_subset_sample_temp_0_20250827_124624',
                   #'medgemma_4b_temp_0_subset_1', 'medgemma_27b_temp_0_subset_1'
                   ]
    
    #jsonl_files = ['subset_1_temp_0.1_20250819_204238', 'subset_1_temp_0.2_20250819_222536', 
    #               'subset_1_temp_0_20250819_183734',#'subset_1_temp_1_20250820_001452',
    #               'subset_2_temp_0.1_20250820_035128', 'subset_2_temp_0.2_20250820_063250',
    #               'subset_2_temp_0_20250820_011422', #'subset_2_temp_1_20250820_085818',
    #               'subset_3_temp_0.1_20250820_121254','subset_3_temp_0.2_20250820_143813',
    #               'subset_3_temp_0_20250820_101317'#,'subset_3_temp_1_20250820_165956'
    #               ]
    
    for jsonl_file in jsonl_files:
        jsonl_file_name = str(results_dir) + '/' + jsonl_file + '.jsonl' #sys.argv[1]
        

        data = load_jsonl(jsonl_file_name)
        print(f"Loaded {len(data)} samples from {jsonl_file}")
          
        # Evaluate predictions and save results
        results = evaluate(data, 'results/all_' + jsonl_file)
        #write_csv(results, str(results_dir) + '/' + 'evaluation_results_all_' + jsonl_file + '.csv')
        write_csv(results, 'results/all_' + jsonl_file + '.csv')
        
        # Pairwise comparisons mcnemar test results
        #data = load_jsonl(jsonl_file_name)
        #results1 = evaluate_models(data)
        
        #for m1, row in results1["pairwise"].items():
        #    for m2, res in row.items():
        #        print(f"{m1} vs {m2}:", res)

        #acc_matrix = summarize_pairwise_matrix(results1, metric="accuracy", save_path = "acc.csv")
        #f1_matrix = summarize_pairwise_matrix(results1, metric="macro-f1", save_path = "f1.csv")
        #print("Accuracy comparisons:")
        #print(acc_matrix)
        #print("\nMacro-F1 comparisons:")
        #print(f1_matrix)
        
        # Evaluate predictions and save results per dataset
        
        #results_datasets = []
        
        datasets = ['images-17','images-44c', 'figshare', 'stroke', 'sclerosis', 'aisd', 'Br35H']
        for dataset in datasets:
            data_dataset = load_jsonl(jsonl_file_name, dataset)
            print(f"Loaded {len(data_dataset)} samples from {jsonl_file}")
            # Evaluate predictions and save results
            results_dataset = evaluate(data_dataset, 'results/' + dataset + '_' + jsonl_file)
            write_csv(results_dataset, 'results/' + dataset + '_' + jsonl_file + '.csv') 
        
if __name__ == "__main__":
    main()
