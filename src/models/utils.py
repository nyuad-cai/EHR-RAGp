from transformers import CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, MODEL_MAPPING
import yaml
import os
import torch
import torch.nn as nn
from typing import Callable

import torch.distributed as dist

def get_config_and_model_cls(model_type: str, mode: str = 'mlm'):
    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"Unknown model_type: {model_type}")
    config_cls = CONFIG_MAPPING[model_type]   # e.g. BertConfig, ModernBertConfig

    if mode == 'mlm':  
        if config_cls not in MODEL_FOR_MASKED_LM_MAPPING:
            raise ValueError(f"No MaskedLM model registered for config: {config_cls}")
        model_cls = MODEL_FOR_MASKED_LM_MAPPING[config_cls]  # e.g. BertForMaskedLM
    else:
        if config_cls not in MODEL_MAPPING:
            raise ValueError(f"No base model registered for config: {config_cls}")
        model_cls = MODEL_MAPPING[config_cls]

    return config_cls, model_cls


def fix_roberta_longformer_max_pos(cfg):

    model_type = getattr(cfg, "model_type", "").lower()

    if model_type == "roberta":
        if cfg.max_position_embeddings == 512:
            cfg.max_position_embeddings = 513

    elif model_type == "longformer":
        if cfg.max_position_embeddings == 512:
            cfg.max_position_embeddings = 4097

    return cfg



def load_config_with_env(path):
    # read file
    with open(path, "r") as f:
        raw_text = f.read()
    expanded = os.path.expandvars(raw_text)
    
    return yaml.safe_load(expanded)


class Time2Vec(nn.Module):

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 16,
        periodic_activation: Callable = torch.sin,
    ):
        super().__init__()
        assert out_features >= 1, "out_features must be >= 1"

        self.in_features = in_features
        self.out_features = out_features
        self.periodic_activation = periodic_activation

        self.W = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))

        self.W0 = nn.Parameter(torch.randn(in_features))
        self.b0 = nn.Parameter(torch.randn(1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:

        v1 = self.periodic_activation(tau @ self.W + self.b)
        v2 = (tau @ self.W0).unsqueeze(-1) + self.b0

        return torch.cat([v2, v1], dim=-1)
    

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()