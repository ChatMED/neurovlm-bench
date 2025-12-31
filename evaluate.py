import json
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate_field(ground_truth: str, predicted: str, field_name: str) -> bool:
    """
    Evaluate a single field comparison.
    Returns True if they match, False otherwise.
    """
    if ground_truth is None or predicted is None:
        return False
    
    # Normalize strings for comparison
    gt_normalized = str(ground_truth).lower().strip()
    pred_normalized = str(predicted).lower().strip()
    
    # Handle special cases for modality field
    if field_name == "modality":
        # Extract base modality (MRI) from more specific descriptions
        if "mri" in pred_normalized and "mri" in gt_normalized:
            return True
    
    # Handle special cases for plane field  
    if field_name == "plane":
        # Map axial_plane to plane
        return gt_normalized == pred_normalized
    
    return gt_normalized == pred_normalized

def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate accuracy percentage."""
    return (correct / total * 100) if total > 0 else 0.0

def evaluate_predictions(data: List[Dict]) -> Dict:
    """
    Evaluate all predictions against ground truth.
    Returns detailed evaluation metrics.
    """
    # Field mappings
    field_mappings = {
        "diagnosis_name": "class",
        "diagnosis_detailed": "subclass", 
        "modality": "modality",
        "plane": "axial_plane"
    }
    
    results = {
        "total_samples": len(data),
        "field_results": {},
        "overall_accuracy": 0,
        "detailed_results": [],
        "token_stats": {}
    }
    
    # Initialize counters for each field
    field_counters = {field: {"correct": 0, "total": 0} for field in field_mappings.keys()}
    overall_correct = 0
    
    # Token tracking by model
    model_token_stats = defaultdict(lambda: {
        "input_tokens": [],
        "output_tokens": [],
        "total_tokens": [],
        "sample_count": 0
    })
    
    for item in data:
        # Skip items without successful output
        if item.get("output", {}).get("status") != "success":
            continue
            
        metadata = item.get("input", {}).get("metadata", {})
        parsed_response = item.get("output", {}).get("parsed_response", {})
        model = item.get("input", {}).get("model_requested")
        
        # Extract token information from usage stats
        api_response = item.get("api_response", {})
        input_tokens = api_response.get("prompt_tokens", 0)
        output_tokens = api_response.get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens
        
        # Track tokens by model
        if model:
            model_token_stats[model]["input_tokens"].append(input_tokens)
            model_token_stats[model]["output_tokens"].append(output_tokens)
            model_token_stats[model]["total_tokens"].append(total_tokens)
            model_token_stats[model]["sample_count"] += 1
        
        sample_result = {
            "experiment_id": item.get("experiment_id"),
            "image_path": item.get("input", {}).get("image_path"),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "field_scores": {}
        }
        
        sample_correct_count = 0
        
        # Evaluate each field
        for pred_field, gt_field in field_mappings.items():
            ground_truth = metadata.get(gt_field)
            predicted = parsed_response.get(pred_field)
            
            is_correct = evaluate_field(ground_truth, predicted, pred_field)
            
            field_counters[pred_field]["total"] += 1
            if is_correct:
                field_counters[pred_field]["correct"] += 1
                sample_correct_count += 1
            
            sample_result["field_scores"][pred_field] = {
                "correct": is_correct,
                "ground_truth": ground_truth,
                "predicted": predicted
            }
        
        # Check if all fields are correct for this sample
        if sample_correct_count == len(field_mappings):
            overall_correct += 1
            
        results["detailed_results"].append(sample_result)
    
    # Calculate accuracies
    for field, counter in field_counters.items():
        accuracy = calculate_accuracy(counter["correct"], counter["total"])
        results["field_results"][field] = {
            "accuracy": accuracy,
            "correct": counter["correct"],
            "total": counter["total"]
        }
    
    results["overall_accuracy"] = calculate_accuracy(overall_correct, len(results["detailed_results"]))
    
    # Calculate token statistics
    for model, stats in model_token_stats.items():
        if stats["sample_count"] > 0:
            avg_input = sum(stats["input_tokens"]) / stats["sample_count"]
            avg_output = sum(stats["output_tokens"]) / stats["sample_count"]
            avg_total = sum(stats["total_tokens"]) / stats["sample_count"]
            
            # Calculate tokens per 1k images (multiply by 1000)
            tokens_per_1k_input = avg_input * 1000
            tokens_per_1k_output = avg_output * 1000
            tokens_per_1k_total = avg_total * 1000
            
            results["token_stats"][model] = {
                "avg_input_tokens": avg_input,
                "avg_output_tokens": avg_output,
                "avg_total_tokens": avg_total,
                "tokens_per_1k_images_input": tokens_per_1k_input,
                "tokens_per_1k_images_output": tokens_per_1k_output,
                "tokens_per_1k_images_total": tokens_per_1k_total,
                "sample_count": stats["sample_count"]
            }
    
    return results

def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report."""
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total samples evaluated: {results['total_samples']}")
    print(f"Overall accuracy (all fields correct): {results['overall_accuracy']:.2f}%")
    print()
    
    print("Field-wise Accuracy:")
    print("-" * 40)
    for field, metrics in results["field_results"].items():
        print(f"{field:20}: {metrics['accuracy']:6.2f}% ({metrics['correct']}/{metrics['total']})")
    
    print()
    print("Sample Breakdown by Model:")
    print("-" * 40)
    
    # Group by model
    model_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for sample in results["detailed_results"]:
        model = sample["model"]
        model_stats[model]["total"] += 1
        if all(score["correct"] for score in sample["field_scores"].values()):
            model_stats[model]["correct"] += 1
    
    for model, stats in model_stats.items():
        accuracy = calculate_accuracy(stats["correct"], stats["total"])
        print(f"{model:30}: {accuracy:6.2f}% ({stats['correct']}/{stats['total']})")

    print()
    print("Token Usage Statistics by Model:")
    print("=" * 120)
    
    # Print token stats header
    header = f"{'Model':<30} {'Avg Input':<12} {'Avg Output':<12} {'Avg Total':<12} {'Input/1k':<12} {'Output/1k':<12} {'Total/1k':<12} {'Samples':<8}"
    print(header)
    print("-" * len(header))
    
    # Print token stats for each model
    for model in sorted(results["token_stats"].keys()):
        stats = results["token_stats"][model]
        row = (f"{model:<30} "
               f"{stats['avg_input_tokens']:<12.1f} "
               f"{stats['avg_output_tokens']:<12.1f} "
               f"{stats['avg_total_tokens']:<12.1f} "
               f"{stats['tokens_per_1k_images_input']:<12.0f} "
               f"{stats['tokens_per_1k_images_output']:<12.0f} "
               f"{stats['tokens_per_1k_images_total']:<12.0f} "
               f"{stats['sample_count']:<8}")
        print(row)

    print()
    print("Per-Field Accuracy by Model:")
    print("=" * 80)
    
    # Calculate per-field accuracy by model
    model_field_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    
    for sample in results["detailed_results"]:
        model = sample["model"]
        for field, score in sample["field_scores"].items():
            model_field_stats[model][field]["total"] += 1
            if score["correct"]:
                model_field_stats[model][field]["correct"] += 1
    
    # Print header
    fields = list(results["field_results"].keys())
    header = f"{'Model':<30} " + " ".join(f"{field:>15}" for field in fields)
    print(header)
    print("-" * len(header))
    
    # Print per-model field accuracies
    for model in sorted(model_field_stats.keys()):
        row = f"{model:<30}"
        for field in fields:
            field_stats = model_field_stats[model][field]
            if field_stats["total"] > 0:
                accuracy = calculate_accuracy(field_stats["correct"], field_stats["total"])
                row += f" {accuracy:>13.1f}%"
            else:
                row += f" {'N/A':>13}"
        print(row)
    
    print()
    print("Detailed Per-Field Model Performance:")
    print("-" * 80)
    
    for field in fields:
        print(f"\n{field.upper()}:")
        for model in sorted(model_field_stats.keys()):
            field_stats = model_field_stats[model][field]
            if field_stats["total"] > 0:
                accuracy = calculate_accuracy(field_stats["correct"], field_stats["total"])
                print(f"  {model:<30}: {accuracy:6.2f}% ({field_stats['correct']}/{field_stats['total']})")

def save_detailed_results(results: Dict, output_file: str):
    """Save detailed results to a CSV file."""
    rows = []
    for sample in results["detailed_results"]:
        row = {
            "experiment_id": sample["experiment_id"],
            "image_path": sample["image_path"],
            "model": sample["model"],
            "input_tokens": sample["input_tokens"],
            "output_tokens": sample["output_tokens"],
            "total_tokens": sample["total_tokens"]
        }
        
        for field, score in sample["field_scores"].items():
            row[f"{field}_correct"] = score["correct"]
            row[f"{field}_ground_truth"] = score["ground_truth"]
            row[f"{field}_predicted"] = score["predicted"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

def main():
    """Main evaluation function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <jsonl_file>")
        sys.exit(1)

    # Load data
    jsonl_file = sys.argv[1]
    
    try:
        data = load_jsonl_data(jsonl_file)
        print(f"Loaded {len(data)} samples from {jsonl_file}")
        
        # Evaluate predictions
        results = evaluate_predictions(data)
        
        # Print report
        print_evaluation_report(results)
        
        # Save detailed results
        save_detailed_results(results, "evaluation_results.csv")
        
    except FileNotFoundError:
        print(f"Error: File {jsonl_file} not found.")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


if __name__ == "__main__":
    main()