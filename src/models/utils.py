

import os
import yaml
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist

from typing import Callable
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from transformers import CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING


def get_config_and_model_cls(model_type: str, mode: str = "mlm"):
    assert mode in ["mlm", "eval", "causal"], "mode must be 'mlm', 'eval', or 'causal'"

    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"Unknown model_type: {model_type}")

    config_cls = CONFIG_MAPPING[model_type]  # e.g. BertConfig, MambaConfig

    if mode == "mlm":
        if config_cls not in MODEL_FOR_MASKED_LM_MAPPING:
            raise ValueError(f"No MaskedLM model registered for config: {config_cls}")
        model_cls = MODEL_FOR_MASKED_LM_MAPPING[config_cls]  # e.g. BertForMaskedLM

    elif mode == "eval":
        if config_cls not in MODEL_MAPPING:
            raise ValueError(f"No base model registered for config: {config_cls}")
        model_cls = MODEL_MAPPING[config_cls]  # e.g. BertModel, MambaModel

    elif mode == "causal":
        if config_cls not in MODEL_FOR_CAUSAL_LM_MAPPING:
            raise ValueError(f"No CausalLM model registered for config: {config_cls}")
        model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[config_cls]  # e.g. MambaForCausalLM

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




def get_bootstrap_ci(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    num_iter: int = 1000,
    alpha: float = 0.05,
    ndigits: int = 3,
):
    device = y_score.device

    y_true = y_true.detach().view(-1).to(device).long()
    y_score = y_score.detach().view(-1).to(device)

    auroc_point = BinaryAUROC().to(device)(y_score, y_true)
    auprc_point = BinaryAveragePrecision().to(device)(y_score, y_true)

    n = y_true.numel()
    auroc_samples = torch.empty(num_iter, device=device)
    auprc_samples = torch.empty(num_iter, device=device)

    for i in range(num_iter):
        idx = torch.randint(0, n, (n,), device=device)
        auroc_samples[i] = BinaryAUROC().to(device)(y_score[idx], y_true[idx])
        auprc_samples[i] = BinaryAveragePrecision().to(device)(y_score[idx], y_true[idx])

    # Percentile CI
    q_low = alpha / 2.0         # 2.5%
    q_high = 1.0 - alpha / 2.0  # 97.5%

    auroc_low = torch.quantile(auroc_samples, q_low)
    auroc_high = torch.quantile(auroc_samples, q_high)

    auprc_low = torch.quantile(auprc_samples, q_low)
    auprc_high = torch.quantile(auprc_samples, q_high)

    def _fmt(point, low, high):
        p = float(point.detach().cpu())
        l = float(low.detach().cpu())
        h = float(high.detach().cpu())
        return f"{round(p, ndigits)} ({round(l, ndigits)}, {round(h, ndigits)})"

    auroc_text = _fmt(auroc_point, auroc_low, auroc_high)
    auprc_text = _fmt(auprc_point, auprc_low, auprc_high)

    return auroc_text, auprc_text


def gather_1d_varlen_pl(module, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().view(-1)

    if not getattr(module, "trainer", None) or module.trainer.world_size == 1:
        return x

    device = x.device
    local_len = torch.tensor([x.numel()], device=device, dtype=torch.long)

    all_lens = module.all_gather(local_len).view(-1) 
    max_len = int(all_lens.max().item())

    if x.numel() < max_len:
        pad = torch.zeros(max_len - x.numel(), device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    x_gather = module.all_gather(x_pad)

    chunks = []
    for r in range(x_gather.shape[0]):
        chunks.append(x_gather[r, : int(all_lens[r].item())])
    return torch.cat(chunks, dim=0)


def log_bootstrap_ci_text_percentile(
    module,
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    prefix: str = "test",
    num_iter: int = 1000,
    alpha: float = 0.05,
    ndigits: int = 3,
):
    y_all = gather_1d_varlen_pl(module, y_true)
    s_all = gather_1d_varlen_pl(module, y_score)

    if not getattr(module, "trainer", None) or module.trainer.is_global_zero:
        auroc_ci_text, auprc_ci_text = get_bootstrap_ci(
            y_true=y_all,
            y_score=s_all,
            num_iter=num_iter,
            alpha=alpha,
            ndigits=ndigits,
        )

        # Use wandb.log directly for string-based CI values
        wandb.log({
            f"{prefix}_auroc_ci": auroc_ci_text,
            f"{prefix}_auprc_ci": auprc_ci_text,
        }, commit=False)