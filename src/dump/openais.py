import os
import re
import time
import json
import openai


import pandas as pd

from tqdm import tqdm



def generate_prompt_strings(path) -> list[tuple[str, str]]:
    """
    Reads a one-to-many GEMs CSV and returns a list of (ICD-9 code, prompt) tuples.
    Each prompt is a single string combining system + user instructions.
    """
    df = pd.read_csv(path,dtype=str)
    required = {'icd9', 'icd9_description', 'icd10', 'icd10_description'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required column(s): {missing}")

#     system = (
#         "You are a clinical coding assistant. Your task is to map an ICD-9 code to the single "
#         "best ICD-10 candidate using only the textual meaning (do not rely on extra clinical context). "
#         "Evaluate true clinical semantics—synonymy, laterality, specificity, body site, temporal detail—"
#         "rather than surface word overlap. Return your answer only as a JSON array sorted best->worst, "
#         "where each element has keys rank, icd10_code, and reason. If none are acceptable, return [\"no_good_match\"].\n\n"
#     )

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
#         parts.append("\nReturn only the JSON array—no extra text.")

#         prompt = system + "\n".join(parts)
        prompt = "\n".join(parts)
        prompts.append((icd9_code, prompt))

    return prompts




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


os.environ['OPENAI_API_KEY']="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"  # <-- Or set OPENAI_API_KEY in your environment
client = openai.OpenAI()

def run_openai_mapping_pipeline_batched(
    prompts: list[tuple[str, str]],
    model_name: str = 'gpt-4o-mini',
    batch_size: int = 100,
    sleep_sec: int = 3,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> list[dict]:
    """
    prompts: list of (icd9_code, prompt_string)
    returns: list of {icd9_code, response}
    """
    results = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} / {(len(prompts) + batch_size - 1) // batch_size}")

        for icd9_code, prompt_str in batch:
            messages = [
                {
                    "role": "system",
                    "content": (
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
                },
                {"role": "user", "content": prompt_str}
            ]

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=1
                )
                results.append({
                    "icd9_code": icd9_code,
                    "response": response.choices[0].message.content.strip()
                })

            except Exception as e:
                print(f"Error with ICD-9 code {icd9_code}: {e}")
                results.append({
                    "icd9_code": icd9_code,
                    "response": None,
                    "error": str(e)
                })

            time.sleep(sleep_sec)  # throttle per request

    return results

path = '/scratch/sas10092/ehr-foundation/notebooks/gpt_p.csv'
# 1-2-many_gems_p.csv
prompts = generate_prompt_strings(path=path)
results = run_openai_mapping_pipeline_batched(prompts)
df_results = parse_mapping_results(results)
df_results.to_csv('/scratch/sas10092/ehr-foundation/notebooks/1-2-many_gems_p_ranked_gpt.csv', index=False)




























# Diagnosis code
# "You are a clinical coding assistant specializing in diagnosis codes. "
# "Your task is to **rank all** candidate ICD-10-cm diagnosis codes for a given ICD-9-CM diagnosis code "
# "using only the textual meaning (do NOT rely on extra clinical context). "
# "Evaluate true clinical semantics—synonymy, laterality, specificity, body site, temporal detail rather than surface word overlap."
# "Return your answer **only** as a JSON array sorted best→worst, where each element has keys "
# "`rank`, `icd10_code`, and `reason`. Include every candidate in the output; do not filter any out. "
# "If none of the candidates are semantically acceptable (unlikely since all must be ranked), "
# "set their `reason` to \"no_good_match\"."


# Procedure codes
# "You are a clinical coding assistant specializing in procedure codes. "
# "Your task is to **rank all** candidate ICD-10-PCS codes for a given ICD-9-CM Volume 3 procedure code "
# "using only the textual meaning (do NOT rely on extra clinical context). "
# "Evaluate true procedural semantics—root operation, approach, body part, device qualifiers, "
# "and other code-specific attributes—rather than surface word overlap. "
# "Return your answer **only** as a JSON array sorted best→worst, where each element has keys "
# "`rank`, `icd10_code`, and `reason`. Include every candidate in the output; do not filter any out. "
# "If none of the candidates are semantically acceptable (unlikely since all must be ranked), "
# "set their `reason` to \"no_good_match\"."