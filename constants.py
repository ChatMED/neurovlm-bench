import os

from dotenv import load_dotenv
from utils import load_prompt

# load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

# Read all image files from the specified folder
IMAGE_MAP = r'image_mapping.csv'  # CSV file with image paths and classes

PARAMETERS = {
    "temperature": 0.1,  # 0, 0.1, 0.2, 0.3
    "seed": 42
}

IMAGE_PATHS = [
    './kaggle.jpg']  # Alternatives ["./use_cases/image1.jpg", "./use_cases/image2.jpg", "./use_cases/image3.jpg", "./use_cases/image4.jpg", "./use_cases/image5.jpg"]

MODELS = [
    "openai/gpt-4o",
    # "openai/gpt-4.1-2025-04-14",
    "openai/gpt-5-chat",
    "google/gemini-2.5-pro",
    # "meta-llama/llama-3.2-90b-vision-instruct",
    # "meta-llama/llama-4-maverick",  # $0.15/M
    # "google/gemini-2.5-flash",  # $0.30/M input tokens, $2.50/M output token
    "meta-llama/llama-4-maverick",  # $0.15/M
    "google/gemini-2.5-flash",  # $0.30/M input tokens, $2.50/M output token
    # "google/gemini-2.0-flash-001",  # $0.10/M input token, $0.40/M output tokens
    # "google/gemma-3-27b-it",  # $0.067/M input tokens, $0.267/M output tokens
    # "openai/gpt-5-mini",
    # "openai/gpt-4o-mini",
    # "meta-llama/llama-3.2-11b-vision-instruct",
    "x-ai/grok-4",
    # "anthropic/claude-sonnet-4.5"
]  # Models on OpenRouter

MODELS_PRICE_LIST = [
    # model, input price per M tokens, output price per M tokens, price per K input imgs
    ("openai/gpt-5-mini",  0.25, 2, 0),
    ("openai/gpt-5-chat", 1.25, 10, 0.0),
    ("openai/gpt-4o-mini", 0.15, 0.60, 0),
    ("openai/gpt-4o", 2.50, 10, 0),
    ("openai/gpt-4.1-2025-04-14", 5, 15, 0),

    ("meta-llama/llama-4-maverick", 0.15, 0.60, 0.668),
    ("meta-llama/llama-3.2-90b-vision-instruct", 0.35, 0.40, 0.506),
    ("meta-llama/llama-3.2-11b-vision-instruct", 0.049, 0.049, 0.079),

    ("google/medgemma_4b", 0.123, 0.456, 0.0), # TODO: This is a placeholder. Update these prices
    ("google/medgemma_27b", 0.123, 0.456, 0.0), # TODO: This is a placeholder. Update these prices

    ("google/gemma-3-27b-it", 0.067, 0.267, 0.0),
    ("google/gemini-2.5-pro", 1.25, 10, 5.16),
    ("google/gemini-2.5-flash", 0.30, 2.50, 1.238),
    ("google/gemini-2.0-flash-001", 0.10, 0.40, 0.026),

    ("x-ai/grok-4", 3, 15, 0.0),
    ("anthropic/claude-sonnet-4.5", 3, 15, 0.0),

    ("bedrock/us.amazon.nova-pro-v1:0", 0.80, 3.20, 1.20), # TODO: This is a placeholder. Update these prices
    ("bedrock/amazon.nova-lite-v1:0", 0.06, 0.24, 0.09), # TODO: This is a placeholder. Update these prices
]  # Models on OpenRouter

OUTPUT_CSV = "openrouter_pubmed_only_image.csv"
OUTPUT_TEXT = "openrouter_pubmed_image_and_text.txt"

# Load prompts from files
PROMPT = load_prompt("medical_image_analysis")
PROMPT_WITH_TEXT = load_prompt("medical_image_and_text_analysis")
NEURORAD_PROMPT = load_prompt("neurorad_prompt_trim_5classes_refined")
