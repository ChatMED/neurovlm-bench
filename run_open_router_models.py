import base64
import csv
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from .constants import API_KEY, PARAMETERS, PROMPT, OUTPUT_CSV, OUTPUT_TEXT, IMAGE_MAP, MODELS, NEURORAD_PROMPT
from .client import send_image_and_text_to_openrouter, send_image_to_openrouter
from .utils import clean_response, encode_image_to_base64, get_image_paths_pathlib, load_prompt, read_imgs, parse_json_response


def main(image_paths: List[str], models: List[str]) -> None:
    """
    Main function proceessin only images without text input.
    """          

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(
            file,
            quoting=csv.QUOTE_MINIMAL,   # default, but explicit is nice
            lineterminator="\n"          # keeps the file POSIX-style; Excel is fine with it
        )
        writer.writerow(["image", "model", "response"])

        for image_path in image_paths:
            # for each model run the experiment
            for model in models:
                try:
                    print(f"Processing: {image_path} with {model}")
                    response = send_image_to_openrouter(image_path, model)
                    writer.writerow([image_path, model, clean_response(response)])
                except Exception as e:
                    print(f"Error with {image_path} + {model}: {e}")
                    writer.writerow([image_path, model, f"ERROR: {e}"])


def main_with_text(image_paths: List[str], models: List[str], text_input_csv: str) -> None:
    """Similar to main, but with text input from a CSV file containing chapter summaries.
    """
    # chapter summary map
    chapter_summary_map = {}

    # read the csv file with the chapter numbers and summary
    with open(text_input_csv, 'r') as f:
        reader = csv.reader(f)
        skip = next(reader)
        for chapter_tuple in reader:
            key = chapter_tuple[0].split('.')[0]  # Get the chapter number as the key
            summary = chapter_tuple[1] if len(chapter_tuple) > 1 else ""
            chapter_summary_map[key] = summary
                
    with open(OUTPUT_TEXT, mode="w", newline="", encoding="utf-8") as file:

        for image_path in image_paths:
            # Extract chapter number from the image path
            chapter_number = os.path.basename(image_path).split('_')[0]  # Assuming the image filename starts with the chapter number
            summary = chapter_summary_map.get(chapter_number, "No summary available for this chapter.")

            image_name = os.path.basename(image_path)

            # for each model run the experiment
            for model in models:
                try:
                    print(f"Processing: {image_path} with {model}")
                    response = send_image_and_text_to_openrouter(image_path, summary, model)
                    file.write(f"Image: {image_name}\nModel: {model}\n\nResponse:\n\n{response}\n\n\n---------------------------------------------------------------\n\n\n")
                except Exception as e:
                    print(f"Error with {image_path} + {model}: {e}")
                    file.write(f"Image: {image_name}\nModel: {model}\n\nError:\n\n{e}\n\n\n---------------------------------------------------------------\n\n\n")
            


def classifiction(image_csv: List[tuple], models: List[str]) -> None:
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(
            file,
            quoting=csv.QUOTE_MINIMAL,   # default, but explicit is nice
            lineterminator="\n"          # keeps the file POSIX-style; Excel is fine with it
        )
        writer.writerow(["image", "model", "true", "predicted"])

        # rows_written = 0                    # counter for batching

        with open(image_csv, 'r') as f:
            reader = csv.reader(f)
            skip = next(reader) # skip first row
        
            for image_tuple in reader:
                for model in models:
                    image_path = image_tuple[0]
                    true_label = image_tuple[1]
                    try:
                        print(f"Processing: {image_path} with {model}")
                        response = send_image_to_openrouter(f'./archive/AugmentedAlzheimerDataset/{image_path}', model)
                        writer.writerow([image_path, model, true_label, clean_response(response)])
                    except Exception as e:
                        print(f"Error with {image_path} + {model}: {e}")
                        writer.writerow([image_path, model, f"ERROR: {e}"])


def process_image_csv(image_mapping_csv: str, models: List[str], system_prompt: str, output_jsonl: str = None) -> None:
    """
    Read the CSV file with image paths and process each image with the specified models. 
    Save the results in a JSONL file.
    """
    if output_jsonl is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl = f"neurorad_experiment_{timestamp}.jsonl"

    with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        total_processed = 0

        with open(image_mapping_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip first row

            for row_idx, row in enumerate(reader):
                if len(row) < 8:  # Ensure row has enough columns
                    continue
                    
                image_path = f'../../ivan/dataset/{row[1]}'
                metadata = {
                    "class": row[2],
                    "subclass": row[4], 
                    "modality": row[5],
                    "axial_plane": row[7]
                }

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                for model in models:
                    try:
                        print(f"Processing {image_path} with {model} (Entry {total_processed + 1})")
                        response = send_image_to_openrouter(image_path, model, system_prompt)
                        parsed_response = parse_json_response(response["content"])
                        
                        result_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "experiment_id": f"neurorad_{total_processed:06d}",
                            
                            # API response metadata
                            "api_response": {
                                "id": response["id"],
                                "model": response["model"],
                                "prompt_tokens": response["prompt_tokens"],
                                "completion_tokens": response["completion_tokens"],
                                "total_tokens": response["total_tokens"]
                            },
                            
                            # Input data
                            "input": {
                                "image_path": image_path,
                                "model_requested": model,
                                "system_prompt_type": "neurorad_prompt_trim",
                                "metadata": metadata
                            },
                            
                            # Model output
                            "output": {
                                "parsed_response": parsed_response,
                                "status": "success",
                                "error": None
                            },
                            
                            # Processing metadata
                            "processing": {
                                "row_index": row_idx + 1,
                                "parameters": PARAMETERS
                            }
                        }
                        
                    except Exception as e:
                        print(f"Error processing {image_path} with {model}: {e}")
                        result_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "experiment_id": f"neurorad_{total_processed:06d}",
                            
                            "api_response": None,
                            
                            "input": {
                                "image_path": image_path,
                                "model_requested": model,
                                "system_prompt_type": "neurorad_prompt_trim",
                                "metadata": metadata
                            },
                            
                            "output": {
                                "parsed_response": None,
                                "status": "error",
                                "error": str(e)
                            },
                            
                            "processing": {
                                "row_index": row_idx + 1,
                                "parameters": PARAMETERS
                            }
                        }
                    
                    # Write to JSONL
                    jsonl_file.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
                    
                    total_processed += 1
                    
                    # Progress update
                    if total_processed % 100 == 0:
                        print(f"Progress: {total_processed} entries processed...")

    print(f"Experiment complete! Results saved to: {output_jsonl}")
    print(f"Total entries processed: {total_processed}")


def process_image_csv_dual_output(image_mapping_csv: str, models: List[str], system_prompt: str, 
                                 output_jsonl: str = None, output_csv: str = None) -> None:
    """
    Read the CSV file with image paths and process each image with the specified models. 
    Save the results in both JSONL and CSV formats simultaneously.
    """
    if output_jsonl is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl = f"experiments_{timestamp}.jsonl"
    
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"experiments_{timestamp}.csv"

    # Define CSV headers
    csv_headers = [
        'row_index', 'image_path', 'true_class', 'true_subclass', 'true_modality', 'true_axial_plane',
        'model_requested', 'model_actual', 'status', 'error',
        'api_id', 'prompt_tokens', 'completion_tokens', 'total_tokens',
        'predicted_modality', 'predicted_plane', 'predicted_diagnosis_name', 
        'predicted_diagnosis_detailed', 'predicted_icd10_code', 'predicted_anatomical_localisation',
        'predicted_rationale', 'predicted_severity_score', 'predicted_severity_description',
        'diagnosis_confidence', 'rationale_confidence', 'severity_confidence',
        'predicted_next_steps'  # Will be JSON string for array data
    ]

    with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file, \
         open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        
        # Initialize CSV writer
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        csv_writer.writerow(csv_headers)

        total_processed = 0

        with open(image_mapping_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip first row

            for row_idx, row in enumerate(reader):
                if len(row) < 8:  # Ensure row has enough columns
                    continue
                    
                image_path = f'dataset/{row[1]}'
                true_class = row[2]
                true_subclass = row[4]
                true_modality = row[5]
                true_axial_plane = row[7]

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                for model in models:
                    try:
                        print(f"Processing {image_path} with model {model} (Row {total_processed + 1})")
                        response = send_image_to_openrouter(image_path, model, system_prompt)
                        parsed_response = parse_json_response(response["content"])
                        
                        # Create JSONL entry (full structure)
                        jsonl_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "row_index": total_processed,
                            "input": {
                                "image_path": image_path,
                                "model_requested": model,
                                "ground_truth": {
                                    "class": true_class,
                                    "subclass": true_subclass,
                                    "modality": true_modality,
                                    "axial_plane": true_axial_plane
                                }
                            },
                            "api_response": {
                                "id": response["id"],
                                "model": response["model"],
                                "prompt_tokens": response["prompt_tokens"],
                                "completion_tokens": response["completion_tokens"],
                                "total_tokens": response["total_tokens"]
                            },
                            "output": {
                                "parsed_response": parsed_response,
                                "status": "success",
                                "error": None
                            },
                            "parameters": PARAMETERS
                        }
                        
                        # Create CSV row (flattened structure)
                        csv_row = [
                            total_processed,
                            image_path,
                            true_class,
                            true_subclass,
                            true_modality,
                            true_axial_plane,
                            model,
                            response["model"],
                            "success",
                            "",  # error
                            response["id"],
                            response["prompt_tokens"],
                            response["completion_tokens"],
                            response["total_tokens"],
                            # Predicted values from parsed response
                            parsed_response.get('modality', '') if 'error' not in parsed_response else '',
                            parsed_response.get('plane', '') if 'error' not in parsed_response else '',
                            parsed_response.get('diagnosis_name', '') if 'error' not in parsed_response else '',
                            parsed_response.get('diagnosis_detailed', '') if 'error' not in parsed_response else '',
                            parsed_response.get('icd10_code', '') if 'error' not in parsed_response else '',
                            parsed_response.get('anatomical_localisation', '') if 'error' not in parsed_response else '',
                            parsed_response.get('rationale', '') if 'error' not in parsed_response else '',
                            parsed_response.get('severity_score', '') if 'error' not in parsed_response else '',
                            parsed_response.get('severity_description', '') if 'error' not in parsed_response else '',
                            parsed_response.get('diagnosis_confidence', '') if 'error' not in parsed_response else '',
                            parsed_response.get('rationale_confidence', '') if 'error' not in parsed_response else '',
                            parsed_response.get('severity_confidence', '') if 'error' not in parsed_response else '',
                            json.dumps(parsed_response.get('next_steps', [])) if 'error' not in parsed_response else ''
                        ]
                        
                    except Exception as e:
                        print(f"Error processing {image_path} with model {model}: {e}")
                        
                        # JSONL error entry
                        jsonl_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "row_index": total_processed,
                            "input": {
                                "image_path": image_path,
                                "model_requested": model,
                                "ground_truth": {
                                    "class": true_class,
                                    "subclass": true_subclass,
                                    "modality": true_modality,
                                    "axial_plane": true_axial_plane
                                }
                            },
                            "api_response": None,
                            "output": {
                                "parsed_response": None,
                                "status": "error",
                                "error": str(e)
                            },
                            "parameters": PARAMETERS
                        }
                        
                        # CSV error row
                        csv_row = [
                            total_processed,
                            image_path,
                            true_class,
                            true_subclass,
                            true_modality,
                            true_axial_plane,
                            model,
                            "",  # model_actual
                            "error",
                            str(e),  # error
                            "",  # api_id
                            "",  # prompt_tokens
                            "",  # completion_tokens
                            "",  # total_tokens
                            # Empty predicted values for error case
                            "", "", "", "", "", "", "", "", "", "", "", "", ""
                        ]
                    
                    # Write to both files
                    jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
                    
                    csv_writer.writerow(csv_row)
                    csv_file.flush()
                    
                    total_processed += 1
                    
                    # Progress update every 100 entries
                    if total_processed % 100 == 0:
                        print(f"Progress: {total_processed} entries processed...")

    print(f"Experiment complete!")
    print(f"JSONL results saved to: {output_jsonl}")
    print(f"CSV results saved to: {output_csv}")
    print(f"Total entries processed: {total_processed}")


def jsonl_to_csv_summary(jsonl_file: str, output_csv: str):
    """Convert JSONL results to CSV summary for analysis"""
    import pandas as pd
    
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            
            # Flatten for CSV
            flat_entry = {
                'timestamp': entry['timestamp'],
                'image_path': entry['input']['image_path'],
                'model': entry['input']['model_requested'],
                'status': entry['output']['status'],
                'true_class': entry['input']['metadata']['class'],
                'true_modality': entry['input']['metadata']['modality'],
                'prompt_tokens': entry['api_response']['prompt_tokens'] if entry['api_response'] else None,
                'total_tokens': entry['api_response']['total_tokens'] if entry['api_response'] else None,
            }
            
            # Add diagnosis fields if successful
            if entry['output']['status'] == 'success' and entry['output']['parsed_response']:
                resp = entry['output']['parsed_response']
                flat_entry.update({
                    'predicted_diagnosis': resp.get('diagnosis_name'),
                    'predicted_modality': resp.get('modality'),
                    'diagnosis_confidence': resp.get('diagnosis_confidence'),
                    'severity_score': resp.get('severity_score')
                })
            
            data.append(flat_entry)
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    process_image_csv(IMAGE_MAP, MODELS, NEURORAD_PROMPT)
