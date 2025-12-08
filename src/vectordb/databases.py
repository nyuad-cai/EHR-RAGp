
import json
import torch
import numpy as np
import polars as pl

import torch.nn as nn
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
from transformers import RoFormerModel
from ..models.models import EHREmbeddings
from ..data.datasets import SequencesGenerator
from typing import List, Dict, Union, Optional
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


class RoformerEHREmbedder(nn.Module):

    def __init__(
        self,
        config,
        dropout: float = 0.1,
        ckpt_path: str = None,   
        pool: str = "cls",              
        attn_pool_dim: int = 128,       
        normalize: bool = False,         
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.config = config
        self.pool = pool
        self.normalize = normalize
        self.device_ = device
        self.dtype_ = dtype

        # backbone without classification head
        self.backbone = RoFormerModel(self.config)

        # plug in your custom input embedding layer
        self.backbone.embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.embedding_size,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout
        )

        # optional attentive pooling
#         if self.pool == "attn":
#             self.attn_pool = nn.Sequential(
#                 nn.Linear(self.config.hidden_size, attn_pool_dim, bias=True),
#                 nn.Tanh(),
#                 nn.Linear(attn_pool_dim, 1, bias=False)
#             )

        # load weights (either full HF state_dict or your Lightning ckpt)
        if ckpt_path:
            self.get_pretrained_weights(backbone= self.backbone, ckpt_path = ckpt_path)

        self.eval().to(self.device_, dtype=self.dtype_)

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        type_ids: torch.Tensor,
        visit_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Returns [B, D] embeddings.
        """
        # Build inputs_embeds via your EHREmbeddings.encode
        inputs_embeds = self.backbone.embeddings.encode(
            input_ids=input_ids.to(self.device_),
            type_ids=type_ids.to(self.device_),
            visit_ids=visit_ids.to(self.device_),
            stage_ids=stage_ids.to(self.device_)
        ).to(self.dtype_)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(self.device_)
        )
        # candidates: last_hidden_state, pooled_output (if available)
        last_hidden = outputs.last_hidden_state  # [B, L, H]

        if self.pool == "cls":
            vec = last_hidden[:, 0, :]  # [CLS]-style
        elif self.pool == "mean":
            # mask-aware mean pooling
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B,L,1]
            vec = (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
#         else:  # attn
#             scores = self.attn_pool(last_hidden).squeeze(-1)         # [B,L]
#             scores = scores.masked_fill(attention_mask == 0, -1e9)
#             w = torch.softmax(scores, dim=-1).unsqueeze(-1)          # [B,L,1]
#             vec = (last_hidden * w).sum(dim=1)                        # [B,H]

        if self.normalize:
            vec = nn.functional.normalize(vec, p=2, dim=-1)
        return vec  # [B, H]
    
    def get_pretrained_weights(self,
                               backbone: nn.Module,
                               ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        new_sd = {}
        for k, v in sd.items():
            # drop non-backbone heads
            if k.startswith(("cls.", "lm_head.", "classifier.", "score.")):
                continue
            # strip leading "backbone."
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            # map "roformer." -> "" (RoFormerModel params live at root)
            if k.startswith("roformer."):
                k = k[len("roformer."):]
            new_sd[k] = v

        # keep only keys that actually exist in target
        target_keys = set(backbone.state_dict().keys())
        filtered = {k: v for k, v in new_sd.items() if k in target_keys}

        print(f"keeping {len(filtered)}/{len(new_sd)} keys")
        missing, unexpected = backbone.load_state_dict(filtered, strict=False)
        print("Loaded with:", {"missing": missing, "unexpected": unexpected})


class EmbedCollator:
    def __init__(self) -> None:
        pass


    def __call__(self, batch: List[Union[Dict, List[Dict]]]) -> Dict[str, torch.Tensor]:
        chunks = self._flatten(batch)
        out = self._stack(chunks)
        
        return out


    def _flatten(self, batch) -> List[Dict]:
        out = []
        for item in batch:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                out.extend(item)
            else:
                raise TypeError(f"Unexpected item type: {type(item)}")
        if not out:
            raise ValueError("Empty batch after normalization.")
        return out

    def _stack(self, chunks: List[Dict]) -> Dict[str, torch.Tensor]:
        keys = list(chunks[0].keys())
        out = {}
        for k in keys:
            # Skip known non-numeric or variable-shaped fields
            if k in ("text_values",):  # add others you don't want to collate
                continue

            # Replace Nones with safe defaults
            seq_list = []
            for c in chunks:
                v = c[k]
                if isinstance(v, list):
                    v = [0 if x is None else x for x in v]  # 0 for ints/floats
                elif v is None:
                    # single value case (shouldn't happen for sequences, but guard anyway)
                    v = 0
                seq_list.append(torch.as_tensor(v))
            out[k] = torch.stack(seq_list, 0)
        return out
    

class ChromaEHREmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embedder, collate_fn, batch_size: Optional[int] = None):
        self.embedder = embedder
        self.collate_fn = collate_fn
        self.batch_size = batch_size  # None => all at once

    def _decode_docs(self, docs: Documents):
        """Accept raw dicts or JSON strings; preserve list-of-lists (patients→chunks)."""
        out: List[Union[dict, list]] = []
        for item in docs:
            if isinstance(item, str):
                out.append(json.loads(item))                     # JSON → dict
            elif isinstance(item, (list, tuple)):
                out.append([json.loads(x) if isinstance(x, str) else x for x in item])
            else:
                out.append(item)                                 # dict
        return out

    def _slice_batch(self, batch_tensors: Dict[str, torch.Tensor], sl: slice) -> Dict[str, torch.Tensor]:
        keys = ("input_ids", "attention_mask", "type_ids", "visit_ids", "stage_ids")
        return {k: batch_tensors[k][sl] for k in keys if k in batch_tensors}

    def __call__(self, docs: Documents) -> Embeddings:
        # 1) Handle both cases: dicts (upsert) or JSON strings (retrieval)
        docs = self._decode_docs(docs)

        # 2) Collate to tensors
        batch_inputs = self.collate_fn(docs)  # -> dict of tensors [B, L]

        # 3) Device/dtype
        device = getattr(self.embedder, "device_", "cpu")
        for k in ("input_ids", "attention_mask", "type_ids", "visit_ids", "stage_ids"):
            if k not in batch_inputs:
                raise KeyError(f"Missing required key '{k}' in collated batch.")
            batch_inputs[k] = batch_inputs[k].to(device=device, dtype=torch.long)

        # 4) Encode (optional mini-batching)
        B = batch_inputs["input_ids"].size(0)
        bs = self.batch_size or B

        embs_out: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, B, bs):
                end = min(start + bs, B)
                sub = self._slice_batch(batch_inputs, slice(start, end))
                v = self.embedder.encode(
                    input_ids=sub["input_ids"],
                    attention_mask=sub["attention_mask"],
                    type_ids=sub["type_ids"],
                    visit_ids=sub["visit_ids"],
                    stage_ids=sub["stage_ids"],
                )
                embs_out.append(v.detach().cpu())

        return torch.cat(embs_out, dim=0).tolist()
    







class VectorDBUploader:
    def __init__(self,
                 collection,
                 seq_gen: SequencesGenerator,
                 main_window: str,
                 seq_length: int,
                 limits: dict,
                 data_idx_path: str,
                 data_path: str,
                 needed_cols: list = ['subject_id', 'input_ids', 'attention_mask', 'visit_ids', 'stage_ids', 'type_ids']
                 ):
        # Initialize client and collection

        self.collection = collection
        # Initialize sequence generator
        self.seq_gen = seq_gen

        # Store parameters
        self.main_window = main_window
        self.seq_length = seq_length
        self.limits = limits
        self.needed_cols = needed_cols

        # Load index and dataset
        self.data_idx = pl.read_parquet(data_idx_path)
        sub_ids = set(self.data_idx.get_column("subject_id").to_list())
        hf_dataset = load_from_disk(data_path)

        # Filter dataset to match index subjects
        hf_dataset = hf_dataset.filter(
            lambda sids: [sid in sub_ids for sid in sids],
            batched=True,
            input_columns="subject_id",
        )

        # Format and subset dataset
        hf_dataset = (
            hf_dataset
            .flatten_indices()
            .select_columns(self.needed_cols)
            .with_format("numpy", columns=self.needed_cols, output_all_columns=False)
        )

        self.hf_dataset = hf_dataset
        self.index = self._build_index()

    def _build_index(self):
        """Build subject_id → list of indices mapping."""
        sids = self.hf_dataset["subject_id"]
        index = defaultdict(list)
        for i, sid in enumerate(sids):
            index[sid].append(i)
        return index

    def upsert_chunks(self):
        """Iterate over data_idx and upload encoded chunks to ChromaDB."""
        for i in tqdm(range(len(self.data_idx))):
            stay = self.data_idx[i]
            split = stay['split'][0]
            subject_id = stay['subject_id'][0]
            start = self.limits[self.main_window][self.seq_length][0]
            end = stay[self.limits[self.main_window][self.seq_length][1]][0]
            timeline_encoded = self.hf_dataset.select(self.index[subject_id])[0]

            # Slice history window
            history_window = {
                k: (v[start:end].tolist() if isinstance(v, (list, np.ndarray)) else v)
                for k, v in timeline_encoded.items()
            }

            # Generate chunks and metadata
            chunks = self.seq_gen.get_overlapped_chunks(history_window)
            docs = [json.dumps(ch, separators=(",", ":")) for ch in chunks]
            ids = [f"{subject_id}__{i:05d}" for i in range(len(chunks))]
            metas = [{"subject_id": subject_id, "chunk_idx": i, "split": split} for i in range(len(chunks))]

            # Upload to Chroma
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            del(chunks, docs, ids, metas, timeline_encoded,history_window)  # free memory