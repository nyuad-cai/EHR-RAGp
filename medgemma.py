import os
import re
import json
import pandas as pd
from vllm import LLM, SamplingParams

os.environ['HUGGINGFACE_HUB_TOKEN'] = "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"
os.environ['HF_TOKEN'] = "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"

CUSTOM_CACHE = "/scratch/sas10092/huggingface_cache"
# Set all relevant environment variables
os.environ['HF_HOME'] = CUSTOM_CACHE
os.environ['HF_HUB_CACHE'] = os.path.join(CUSTOM_CACHE)
os.environ['TRANSFORMERS_CACHE'] = CUSTOM_CACHE
os.environ['HUGGINGFACE_HUB_CACHE'] = CUSTOM_CACHE
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ['CC']="/share/apps/NYUAD5/gcc/9.2.0/bin/gcc"
os.environ['CXX']="/share/apps/NYUAD5/gcc/9.2.0/bin/g++"



llm = LLM(model="/scratch/sas10092/huggingface_cache/models--google--medgemma-27b-text-it/snapshots/6b08c481126ff65a9b8fa5ab4d691b152b8edb5d/",)
# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512000
)


def generate_prompt_strings() -> list[tuple[str, str]]:
    """
    Reads a one-to-many GEMs CSV and returns a list of (ICD-9 code, prompt) tuples.
    Each prompt is a single string combining system + user instructions.
    """
    df = pd.read_csv('/scratch/sas10092/ehr-foundation/notebooks/medgemma_p.csv',dtype=str)
    required = {'icd9', 'icd9_description', 'icd10', 'icd10_description'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required column(s): {missing}")

    system = (
            "You are a clinical coding assistant specializing in procedure codes. "
            "Your task is to **rank all** candidate ICD-10-PCS codes for a given ICD-9-CM Volume 3 procedure code "
            "using only the textual meaning (do NOT rely on extra clinical context). "
            "Evaluate true procedural semantics—root operation, approach, body part, device qualifiers, "
            "and other code-specific attributes—rather than surface word overlap. "
            "Return your answer **only** as a JSON array sorted best→worst, where each element has keys "
            "`rank`, `icd10_code`, and `reason`. Include every candidate in the output; do not filter any out. "
            "If none of the candidates are semantically acceptable (unlikely since all must be ranked), "
            "set their `reason` to \"no_good_match\"."
    )


    prompts = []
    for icd9_code, group in df.groupby("icd9", sort=False):
        # Build the user section
        parts = [
            f"ICD-9 code: {icd9_code}",
            f"ICD-9 description: \"{group.iloc[0]['icd9_description']}\"",
            "",
            "Candidate ICD-10 codes:"
        ]
        for idx, row in group.reset_index(drop=True).iterrows():
            parts.append(f"{idx+1}. {row['icd10']}: \"{row['icd10_description']}\"")
        parts.append("\nReturn only the JSON array—no extra text.")

        prompt = system + "\n".join(parts)
        prompts.append((icd9_code, prompt))

    return prompts



def run_mapping_pipeline() -> list[dict]:
    """
    Runs the generated prompts through the MedGemma model via VLLM
    and returns a list of dicts with ICD-9 codes and their JSON responses.
    """
    # Load prompts
    prompts = generate_prompt_strings()

    # Instantiate VLLM with MedGemma


    results = []
    for icd9_code, prompt in prompts:
        # Generate; receive a list of RequestOutput
        request_outputs = llm.generate([prompt], sampling_params=sampling_params)
        # Each RequestOutput has .outputs, a list of generated chunks
        if request_outputs and request_outputs[0].outputs:
            text = request_outputs[0].outputs[0].text.strip()
        else:
            text = ""
        results.append({"icd9_code": icd9_code, "response": text})

    return results


def parse_mapping_results(results: list[dict]) -> pd.DataFrame:
    """
    Parses LLM mapping results into a pandas DataFrame with columns:
    - icd9_code
    - icd10_code
    - rank
    - reason

    Args:
        results: List of dicts, each with 'icd9_code' and 'response' where response
                 is a string containing a JSON array (possibly fenced by ```json).

    Returns:
        pd.DataFrame: One row per mapping entry.
    """
    rows = []
    for item in results:
        icd9 = item.get('icd9_code')
        resp = item.get('response', '')

        # Remove triple backticks and optional "json" marker
        # and extract the JSON array
        # First try to strip fences
        resp_stripped = re.sub(r'```(?:json)?', '', resp).strip()

        # Then find the JSON array
        start = resp_stripped.find('[')
        end = resp_stripped.rfind(']') + 1
        if start == -1 or end == -1:
            continue  # skip if no JSON array found

        json_text = resp_stripped[start:end]

        try:
            mapping = json.loads(json_text)
        except json.JSONDecodeError:
            continue  # skip invalid JSON

        # Flatten entries
        for entry in mapping:
            rows.append({
                'icd9_code': icd9,
                'icd10_code': entry.get('icd10_code'),
                'rank': entry.get('rank'),
                'reason': entry.get('reason')
            })

    return pd.DataFrame(rows)



results = run_mapping_pipeline()

ranked_df = parse_mapping_results(results)
ranked_df.to_csv('/scratch/sas10092/ehr-foundation/notebooks/1-2-many_gems_p_ranked.csv', index=False)



