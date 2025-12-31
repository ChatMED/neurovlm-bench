import base64
import json
import csv
from pathlib import Path
from typing import List


def get_image_paths_pathlib(folder_path: str) -> List[str]:
    """
    Get all image file paths using pathlib (more modern approach)
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    image_paths = []
    
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))
    
    return sorted(image_paths)


def load_prompt(prompt_name: str) -> str:
    """Load prompt from file in prompts directory"""
    script_dir = Path(__file__).parent
    prompt_file = script_dir / "prompts" / f"{prompt_name}.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()
    

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded


def read_imgs(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        skip = next(reader) # skip first row
        yield reader 



def clean_response(text: str) -> str:
    """
    Replace any CR/LF sequences with a single space so every record
    stays on one physical row when opened in Excel.
    """
    if text is None:
        return ""
    # Get rid of \r, \n, or \r\n
    return " ".join(text.splitlines()).strip()


def parse_json_response(response_text: str) -> dict:
    """
    Parse JSON response by finding first { and last } brackets
    """
    try:
        # Find the first opening brace and last closing brace
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        
        if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
            raise ValueError("No valid JSON object found in response")
        
        # Extract the JSON substring
        json_str = response_text[first_brace:last_brace + 1]
        
        # Parse the JSON
        parsed_data = json.loads(json_str)
        
        # Validate expected fields from your neurorad prompt
        expected_fields = [
            'modality', "specialized_sequence" , 'plane', 'diagnosis_name', 'diagnosis_detailed',
            'icd10_code', 'severity_score', 'diagnosis_confidence', 'severity_confidence'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in expected_fields if field not in parsed_data]
        if missing_fields:
            print(f"Warning: Missing fields: {missing_fields}")
        
        return parsed_data
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing failed: {str(e)}", "raw_response": response_text}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}", "raw_response": response_text}