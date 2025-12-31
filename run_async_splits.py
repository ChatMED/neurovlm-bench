import os
from datetime import datetime
from run_async import run_async_experiment, run_async_experiment_few_shot
from constants import MODELS, NEURORAD_PROMPT
import pandas as pd

# TODO restore this later after initial test
SUBSETS = ['subsets/subset_0_25_7.csv']
# SUBSETS = ['subsets/subset_sample.csv']

FEW_SHOT_EXAMPLES_PATH = "few_shot_examples.csv"
IMAGES_ROOT = "../../ivan/dataset"


TEMPERATURES = [0, 1]

few_shot = False

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    df_examples = pd.read_csv(FEW_SHOT_EXAMPLES_PATH)

    if IMAGES_ROOT is not None:
        # append it as a prefix to the column file_path in df_examples
        df_examples['file_path'] = df_examples['file_path'].apply(lambda x: os.path.join(IMAGES_ROOT, x))

    examples_dict = dict(zip(df_examples["file_path"], df_examples["class"]))
    print (examples_dict)

    for subset in SUBSETS:
        for temperature in TEMPERATURES:
            print(f"Running experiment for {subset} with temperature {temperature}")
            subset_name = subset.replace('subsets', '').replace('.csv', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/{subset_name}_temp_{temperature}_{timestamp}.jsonl"
            print(f"Output will be saved to: {output_file}")

            models = []
            if temperature != 1:
                models = [m for m in MODELS if "openai/gpt-5" not in m]
            else:
                models = [m for m in MODELS if "openai/gpt-5" in m]

            if few_shot:

                run_async_experiment_few_shot(
                    image_map=subset,
                    models=models,
                    temperature=temperature,
                    prompt=NEURORAD_PROMPT,
                    examples=examples_dict,
                    output_jsonl=output_file
                )
            else:
                # Single-shot experiments
                run_async_experiment(subset, models, temperature, NEURORAD_PROMPT, output_jsonl=output_file)




            print(f"Completed: {subset} with temperature {temperature}")
            print("-" * 50)

    print("All experiments completed.")
