import faiss
import torch
import numpy as np
from collections import defaultdict

def build_index(hf_dataset):
    sids = hf_dataset["subject_id"]
    index = defaultdict(list)
    for i, sid in enumerate(sids):
        index[sid].append(i)
    return index

def to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, list):
        return v
    return v



def build_faiss_index(ch_embs, metric: str = "l2"):

    # Accept either:
    #  - list[Tensor] each (1,D) or (D,)
    #  - Tensor of shape (N,D) or (D,) or (1,D)
    if torch.is_tensor(ch_embs):
        v = ch_embs.detach().cpu()
        if v.ndim == 1:
            xb = v.unsqueeze(0).numpy()
        elif v.ndim == 2:
            xb = v.numpy()
        else:
            raise ValueError(f"ch_embs tensor must be (D,) or (N,D). Got {tuple(v.shape)}")
    else:
        xb_list = []
        for t in ch_embs:
            if not torch.is_tensor(t):
                raise TypeError(f"Expected torch.Tensor embeddings, got {type(t)}")
            v = t.detach().cpu()
            if v.ndim == 2 and v.shape[0] == 1:
                v = v[0]
            elif v.ndim != 1:
                raise ValueError(f"Each embedding must be (D,) or (1,D). Got {tuple(v.shape)}")
            xb_list.append(v.numpy())
        xb = np.stack(xb_list, axis=0)

    xb = xb.astype(np.float32, copy=False)
    xb = np.ascontiguousarray(xb)  # <-- FIX

    dim = xb.shape[1]
    metric = metric.lower()

    if metric in ("cosine", "ip", "inner_product", "dot"):
        faiss.normalize_L2(xb)      # now safe
        index = faiss.IndexFlatIP(dim)
    elif metric in ("l2", "euclidean"):
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unknown metric={metric}. Use 'l2' or 'cosine'.")

    index.add(xb)
    return index, xb





