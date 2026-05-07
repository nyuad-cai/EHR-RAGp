import gc
import os
import torch
import polars as pl
from src.models.baseline_models import load_hf_model
from  datasets import load_from_disk
from src.models.utils import predict_dataset, compute_metrics_with_ci_llm

import argparse

parser = argparse.ArgumentParser(description='LLM pretraining command line interface')
parser.add_argument('--model-name', type=str, required=True)
args = parser.parse_args()


os.environ['HUGGINGFACE_HUB_TOKEN'] = "add"
os.environ["HF_TOKEN"] = "add"

hf_token = os.getenv("HF_TOKEN")

CUSTOM_CACHE = "path/to/"
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["TRANSFORMERS_CACHE"] = CUSTOM_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CUSTOM_CACHE
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ['CC']="path/to/"
os.environ['CXX']="path/to/"

# Qwen/Qwen2.5-7B-Instruct
# google/medgemma-1.5-4b-it
model_name=args.model_name
print(model_name)
dataset_path = "path/to/descemb_dataset"
data_idx_path = './downstream_idx.parquet'
dataset = load_from_disk(dataset_path)



window = 'within_stay_descemb'
task = 'y_mort_1yr'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=data_idx_path, window=window, task_name=task, model_bundle=bundle) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)


window = 'within_stay_descemb'
task = 'y_icu_readmit_30'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=data_idx_path, window=window, task_name=task, model_bundle=bundle) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)


window = 'within48_descemb'
task = 'y_mort'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=data_idx_path, window=window, task_name=task, model_bundle=bundle) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)


window = 'within24_descemb'
task = 'y_los_7'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=data_idx_path, window=window, task_name=task, model_bundle=bundle) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)

print('All tasks completed!')

