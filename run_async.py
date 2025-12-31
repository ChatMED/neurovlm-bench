import asyncio
import aiohttp
import aiofiles

import csv
import json
import os
from datetime import datetime
from typing import List, Dict

from constants import API_KEY, PARAMETERS, PROMPT, OUTPUT_CSV, IMAGE_MAP, MODELS, NEURORAD_PROMPT
from client import send_image_to_openrouter_async, send_image_few_shot_to_openrouter_async
from utils import parse_json_response


async def process_image_csv_async(image_mapping_csv: str, models: List[str], system_prompt: str,
                                 output_jsonl: str = None, max_concurrent: int = 10) -> None:
    """
    Async version - much faster for I/O bound operations
    """
    if output_jsonl is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl = f"neurorad_experiment_{timestamp}.jsonl"

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_task(session, row_idx, row, model, total_count):
        async with semaphore:  # Limit concurrent requests
            if len(row) < 8:
                return None

            #TODO add here ../../home/ivan on the server
            image_path = f'../../ivan/dataset/{row[1]}'
            metadata = {
                "class": row[2],
                "subclass": row[4], 
                "modality": row[5],
                "modality_subtype": row[6],
                "axial_plane": row[7]
            }

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return None

            try:
                print(f"Processing {image_path} with {model} (Entry {total_count})")
                response = await send_image_to_openrouter_async(session, image_path, model, system_prompt)
                parsed_response = parse_json_response(response["content"])
                
                result_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": f"neurorad_{total_count:06d}",
                    "api_response": {
                        "id": response["id"],
                        "model": response["model"],
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                        "total_tokens": response["total_tokens"]
                    },
                    "input": {
                        "image_path": image_path,
                        "model_requested": model,
                        "system_prompt_type": "neurorad_prompt_trim",
                        "metadata": metadata
                    },
                    "output": {
                        "parsed_response": parsed_response,
                        "status": "success",
                        "error": None
                    },
                    "processing": {
                        "row_index": row_idx + 1,
                        "parameters": PARAMETERS
                    }
                }
                
            except Exception as e:
                print(f"Error processing {image_path} with {model}: {e}")
                result_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": f"neurorad_{total_count:06d}",
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
            
            return result_entry

    # Prepare all tasks
    tasks = []
    total_count = 0
    
    with open(image_mapping_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for row_idx, row in enumerate(rows):
            for model in models:
                task = process_single_task(session, row_idx, row, model, total_count)
                tasks.append(task)
                total_count += 1

        print(f"Total tasks to process: {len(tasks)}")
        
        # Process all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Write results to file
    successful_results = [r for r in results if isinstance(r, dict)]
    
    async with aiofiles.open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in successful_results:
            await f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Experiment complete! Results saved to: {output_jsonl}")
    print(f"Total entries processed: {len(successful_results)}")


# Streaming version
async def process_image_csv_async_streaming(image_mapping_csv: str, models: List[str], temperature: float, system_prompt: str, 
                                          output_jsonl: str = None, max_concurrent: int = 10) -> None:
    """
    Async version that writes results as they complete - memory efficient
    """
    if output_jsonl is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl = f"neurorad_experiment_{timestamp}.jsonl"

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Shared counter for progress tracking
    completed_count = 0
    total_expected = 0
    
    # Lock for thread-safe file writing
    file_lock = asyncio.Lock()
    
    async def write_result_to_file(result_entry):
        """Write a single result to file immediately"""
        nonlocal completed_count
        
        async with file_lock:
            async with aiofiles.open(output_jsonl, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            
            completed_count += 1
            
            # Progress update every 100 completions
            if completed_count % 100 == 0:
                print(f"Progress: {completed_count}/{total_expected} completed ({completed_count/total_expected*100:.1f}%)")
    
    async def process_single_task(session, row_idx, row, model, total_count):
        async with semaphore:  # Limit concurrent requests
            if len(row) < 8:
                return
                
            image_path = f'../../ivan/dataset/{row[1]}'
            metadata = {
                "dataset": row[0],
                "class": row[2],
                "subclass": row[4], 
                "modality": row[5],
                "modality_subtype": row[6],
                "axial_plane": row[7]
            }

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return

            try:
                print(f"Processing {image_path} with {model} (Entry {total_count})")
                response = await send_image_to_openrouter_async(session, image_path, model, temperature, system_prompt)
                parsed_response = parse_json_response(response["content"])
                
                result_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": f"neurorad_{total_count:06d}",
                    "api_response": {
                        "id": response["id"],
                        "model": response["model"],
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                        "total_tokens": response["total_tokens"]
                    },
                    "input": {
                        "image_path": image_path,
                        "model_requested": model,
                        "system_prompt_type": "neurorad_prompt_trim",
                        "metadata": metadata
                    },
                    "output": {
                        "parsed_response": parsed_response,
                        "status": "success",
                        "error": None
                    },
                    "processing": {
                        "row_index": row_idx + 1,
                        "parameters": {
                            "temperature": temperature,
                            "seed": PARAMETERS["seed"],

                        }
                    }
                }
                
            except Exception as e:
                print(f"Error processing {image_path} with {model}: {e}")
                result_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": f"neurorad_{total_count:06d}",
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
            
            # Write result immediately when task completes
            await write_result_to_file(result_entry)

    # Initialize empty output file
    async with aiofiles.open(output_jsonl, 'w', encoding='utf-8') as f:
        pass  # Create empty file

    # Prepare all tasks
    with open(image_mapping_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    total_count = 0
    for row in rows:
        if len(row) >= 8:
            total_expected += len(models)
    
    print(f"Total tasks to process: {total_expected}")
    
    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        total_count = 0
        
        for row_idx, row in enumerate(rows):
            for model in models:
                task = process_single_task(session, row_idx, row, model, total_count)
                tasks.append(task)
                total_count += 1

        # Process all tasks concurrently - results are written as they complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"Experiment complete! Results saved to: {output_jsonl}")
    print(f"Total entries processed: {completed_count}")


async def process_image_csv_async_streaming_few_shot(image_mapping_csv: str, models: List[str], temperature: float,
                                            system_prompt: str, examples_dict: Dict[str,str],
                                            output_jsonl: str = None, max_concurrent: int = 10) -> None:
    """
    Async version that writes results as they complete - memory efficient
    """
    if output_jsonl is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl = f"neurorad_experiment_{timestamp}.jsonl"

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Shared counter for progress tracking
    completed_count = 0
    total_expected = 0

    # Lock for thread-safe file writing
    file_lock = asyncio.Lock()

    async def write_result_to_file(result_entry):
        """Write a single result to file immediately"""
        nonlocal completed_count

        async with file_lock:
            async with aiofiles.open(output_jsonl, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

            completed_count += 1

            # Progress update every 100 completions
            if completed_count % 100 == 0:
                print(
                    f"Progress: {completed_count}/{total_expected} completed ({completed_count / total_expected * 100:.1f}%)")

    async def process_single_task(session, row_idx, row, model, total_count):
        async with semaphore:  # Limit concurrent requests
            if len(row) < 8:
                return

            image_path = f'../../ivan/dataset/{row[1]}'
            metadata = {
                "dataset": row[0],
                "class": row[2],
                "subclass": row[4],
                "modality": row[5],
                "axial_plane": row[7]
            }

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return

            try:
                print(f"Processing {image_path} with {model} (Entry {total_count})")
                response = await send_image_few_shot_to_openrouter_async(
                    session=session,
                    image_path=image_path,
                    model=model,
                    temperature=temperature,
                    system_prompt_text=system_prompt,
                    examples_dict=examples_dict
                )
                parsed_response = parse_json_response(response["content"])

                result_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": f"neurorad_{total_count:06d}",
                    "api_response": {
                        "id": response["id"],
                        "model": response["model"],
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                        "total_tokens": response["total_tokens"]
                    },
                    "input": {
                        "image_path": image_path,
                        "model_requested": model,
                        "system_prompt_type": "neurorad_prompt_trim",
                        "metadata": metadata
                    },
                    "output": {
                        "parsed_response": parsed_response,
                        "status": "success",
                        "error": None
                    },
                    "processing": {
                        "row_index": row_idx + 1,
                        "parameters": {
                            "temperature": temperature,
                            "seed": PARAMETERS["seed"],

                        }
                    }
                }

            except Exception as e:
                print(f"Error processing {image_path} with {model}: {e}")
                result_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": f"neurorad_{total_count:06d}",
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

            # Write result immediately when task completes
            await write_result_to_file(result_entry)

    # Initialize empty output file
    async with aiofiles.open(output_jsonl, 'w', encoding='utf-8') as f:
        pass  # Create empty file

    # Prepare all tasks
    with open(image_mapping_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    total_count = 0
    for row in rows:
        if len(row) >= 8:
            total_expected += len(models)

    print(f"Total tasks to process: {total_expected}")

    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        total_count = 0

        for row_idx, row in enumerate(rows):
            for model in models:
                task = process_single_task(session, row_idx, row, model, total_count)
                tasks.append(task)
                total_count += 1

        # Process all tasks concurrently - results are written as they complete
        await asyncio.gather(*tasks, return_exceptions=True)

    print(f"Experiment complete! Results saved to: {output_jsonl}")
    print(f"Total entries processed: {completed_count}")



# To run the async version:
def run_async_experiment(image_map, models, temperature, prompt, output_jsonl=None):
    asyncio.run(process_image_csv_async_streaming(image_map, models, temperature, prompt, output_jsonl=output_jsonl))

def run_async_experiment_few_shot(image_map, models, temperature, prompt, examples: Dict[str,str], output_jsonl=None):
    asyncio.run(process_image_csv_async_streaming_few_shot(image_map, models, temperature, prompt, examples_dict=examples, output_jsonl=output_jsonl))

if __name__ == "__main__":
    run_async_experiment(IMAGE_MAP, MODELS, PARAMETERS["temperature"], NEURORAD_PROMPT)
