from typing import Dict, List, Any

import requests
import aiohttp

from constants import API_KEY, PARAMETERS, PROMPT
from utils import encode_image_to_base64


def send_image_to_openrouter(image_path: str, model: str, system_prompt: str) -> str:
    
    encoded_image = encode_image_to_base64(image_path)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://finki.ukim.mk",
        "X-Title": "Image-to-LLM Script",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "temperature": PARAMETERS["temperature"], # Low for consistency
        "seed": PARAMETERS["seed"], # Fixed seed for reproducibility
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed with status {response.status_code}: {response.text}")
    
    json_response = response.json()
    
    return {
        "id": json_response["id"],
        "model": json_response["model"],
        "prompt_tokens": json_response["usage"]["prompt_tokens"],
        "completion_tokens": json_response["usage"]["completion_tokens"],
        "total_tokens": json_response["usage"]["total_tokens"],
        "content": json_response["choices"][0]["message"]["content"]
    }


def send_image_and_text_to_openrouter(image_path: str, summary: str, model: str) -> str:
    encoded_image = encode_image_to_base64(image_path)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "Image-to-LLM Script",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": f"# User input:{summary}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed with status {response.status_code}: {response.text}")
    
    print(response.text)  # For debugging, print the full response
    return response.json()["choices"][0]["message"]["content"]


async def send_image_to_openrouter_async(session: aiohttp.ClientSession, 
                                        image_path: str, model: str, temperature:float, system_prompt: str) -> dict:
    """Async version of send_image_to_openrouter"""
    encoded_image = encode_image_to_base64(image_path)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://finki.ukim.mk",
        "X-Title": "Image-to-LLM Script",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "seed": PARAMETERS["seed"],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }]
            }
        ]
    }

    async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                           headers=headers, json=payload) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"Failed with status {response.status}: {text}")
        
        json_response = await response.json()
        
        return {
            "id": json_response["id"],
            "model": json_response["model"],
            "prompt_tokens": json_response["usage"]["prompt_tokens"],
            "completion_tokens": json_response["usage"]["completion_tokens"],
            "total_tokens": json_response["usage"]["total_tokens"],
            "content": json_response["choices"][0]["message"]["content"]
        }


def build_examples_user_content(examples_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Build the content array for the FEW-SHOT examples as a *user* message:
    [ text-intro, (text+image) x N, text-outro ]
    """
    content: List[Dict[str, Any]] = []
    if not examples_dict:
        return content

    content.append({
        "type": "text",
        "text": (
            "### FEW-SHOT EXAMPLES\n"
            "Below are example images and ONLY their class label. "
            "Use them to calibrate class boundaries.\n"
            "Allowed classes: tumor | stroke | multiple sclerosis | normal | other abnormalities\n"
        )
    })

    for idx, (img_path, cls_raw) in enumerate(examples_dict.items(), start=1):
        cls_name = cls_raw
        header = f"Example {idx} — class: {cls_name}"
        content.append({"type": "text", "text": header})
        b64 = encode_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    content.append({
        "type": "text",
        "text": (
            "### END OF FEW-SHOT EXAMPLES\n"
            "Now analyze the next image and output ONLY the JSON as per the schema."
        )
    })
    return content


def build_fewshot_class_only_system_content(
    base_instructions_text: str,
    examples_dict: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Builds system content with your base instructions followed by a FEW-SHOT block.
    examples_dict: { image_path: class_name } (class_name is one of the 5 allowed)
    """
    content = [{"type": "text", "text": base_instructions_text.strip()}]

    if examples_dict:
        content.append({
            "type": "text",
            "text": (
                "### FEW-SHOT EXAMPLES\n"
                "Below are example images and ONLY their class label. "
                "Use them to calibrate your understanding of class boundaries.\n"
                "Allowed classes: tumor | stroke | multiple sclerosis | normal | other abnormalities\n"
            )
        })

        for idx, (img_path, cls_raw) in enumerate(examples_dict.items(), start=1):
            cls_name = cls_raw
            header = f"Example {idx} — class: {cls_name}"
            content.append({"type": "text", "text": header})
            b64 = encode_image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        content.append({
            "type": "text",
            "text": (
                "### END OF FEW-SHOT EXAMPLES\n"
                "Now analyze the next image and output ONLY the JSON as per the schema."
            )
        })

    return content


def build_query_user_content(query_image_path: str) -> List[Dict[str, Any]]:
    q_b64 = encode_image_to_base64(query_image_path)
    return [
        {"type": "text", "text": "Analyze this image and output ONLY the JSON per the schema."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}}
    ]


async def send_image_few_shot_to_openrouter_async(
    session: aiohttp.ClientSession,
    image_path: str,
    model: str,
    temperature: float,
    system_prompt_text: str,
    examples_dict: Dict[str, str],
) -> dict:
    """
    Few-shot version with examples placed in *user* messages (no images in system).
    - system: text-only instructions/schema
    - user #1: few-shot examples (class label + image)
    - user #2: the query image
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://finki.ukim.mk",
        "X-Title": "Image-to-LLM Script",
        "Content-Type": "application/json"
    }

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt_text.strip()}]
        }
    ]

    # Add few-shot examples as a user message (if any)
    examples_content = build_examples_user_content(examples_dict)
    if len(examples_content) > 0:
        messages.append({"role": "user", "content": examples_content})

    # Add the query image as a final user message
    messages.append({"role": "user", "content": build_query_user_content(image_path)})

    payload = {
        "model": model,
        "temperature": temperature,
        "seed": PARAMETERS.get("seed"),
        "messages": messages,
        # Some models accept this; if you see an error, remove this line.
        # "response_format": {"type": "json_object"}
    }

    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    ) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"Failed with status {response.status}: {text}")

        jr = await response.json()
        return {
            "id": jr["id"],
            "model": jr["model"],
            "prompt_tokens": jr["usage"]["prompt_tokens"],
            "completion_tokens": jr["usage"]["completion_tokens"],
            "total_tokens": jr["usage"]["total_tokens"],
            "content": jr["choices"][0]["message"]["content"]
        }