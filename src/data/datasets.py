
import os
import json
import torch
import faiss
import bisect
import random
import chromadb
import numpy as np
import polars as pl

from torch import nn
from .utils import pack_clmbr_chunks
from math import ceil
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, Iterable, List, Optional, Any, Union, Literal, Tuple, Callable

class Tokenizer:
    def __init__(
        self,
        codes_parquet_fp: str,
        special_tokens: Optional[Iterable[str]] = None,
        force_special_ids: bool = True,  # pin [PAD]=0 etc.
    ):
        if special_tokens is None:
            special_tokens = ["[PAD]", "[MASK]", "[CLS]", "[UNK]"]


        df_codes = pl.read_parquet(str(codes_parquet_fp), columns=["code"])
        base_codes = df_codes.get_column("code").to_list()
        seen = set()
        unique_codes = []
        for c in base_codes:
            if c not in seen:
                unique_codes.append(c)
                seen.add(c)

        vocab_list: List[str] = []
        special_tokens = list(special_tokens)

        if force_special_ids:
            for tok in ["[PAD]", "[MASK]", "[CLS]", "[UNK]"]:
                if tok in special_tokens and tok not in seen:
                    vocab_list.append(tok)
                    seen.add(tok)
                elif tok in special_tokens and tok in seen:

                    vocab_list.append(tok)
            for tok in special_tokens:
                if tok not in seen:
                    vocab_list.append(tok)
                    seen.add(tok)
        else:
            for tok in special_tokens:
                if tok not in seen:
                    vocab_list.append(tok)
                    seen.add(tok)


        vocab_list.extend(unique_codes)


        self.id2code: List[str] = vocab_list
        self.code2id: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab_list)}
        self.vocab_size: int = len(self.id2code)

        self.pad_token  = "[PAD]" if "[PAD]" in self.code2id else None
        self.mask_token = "[MASK]" if "[MASK]" in self.code2id else None
        self.cls_token  = "[CLS]" if "[CLS]" in self.code2id else None
        self.unk_token  = "[UNK]" if "[UNK]" in self.code2id else None

        self.pad_id  = self.code2id[self.pad_token]  if self.pad_token  else 0
        self.mask_id = self.code2id[self.mask_token] if self.mask_token else None
        self.cls_id  = self.code2id[self.cls_token]  if self.cls_token  else None
        self.unk_id  = self.code2id[self.unk_token]  if self.unk_token  else None


        type_set = set()
        for tok in self.id2code:
            prefix = tok.split("//", 1)[0]
            type_set.add(prefix)

        types_sorted = sorted(t for t in type_set if t not in ("[PAD]",))
        self.type2id: Dict[str, int] = {"[PAD]": 0}
        next_id = 1
        for sp in ["[MASK]", "[CLS]", "[UNK]"]:
            if sp in type_set:
                self.type2id[sp] = next_id; next_id += 1
        for t in types_sorted:
            if t not in self.type2id:
                self.type2id[t] = next_id
                next_id += 1

        self._code2id_df = pl.DataFrame({"code": self.id2code,
                                         "input_id": list(range(self.vocab_size))}) \
                               .with_columns(pl.col("code").cast(pl.Categorical))
        self._type2id_df = pl.DataFrame({"code_type": list(self.type2id.keys()),
                                         "type_id":   list(self.type2id.values())}) \
                               .with_columns(pl.col("code_type").cast(pl.Categorical))


    def encode(self, codes: Iterable[str]) -> List[int]:
        get = self.code2id.get
        if self.unk_id is not None:
            fallback = self.unk_id
        else:
            fallback = self.pad_id if self.pad_id is not None else 0
        return [get(c, fallback) for c in codes]

    def decode(self, ids: Iterable[int]) -> List[str]:
        out = []
        for i in ids:
            if 0 <= i < self.vocab_size:
                out.append(self.id2code[i])
            else:
                out.append(self.unk_token or "[UNK]")
        return out

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "id2code": self.id2code,
            "special_tokens": [t for t in ["[PAD]", "[MASK]", "[CLS]", "[UNK]"] if t in self.code2id],
            "type2id": self.type2id,
            "pad_id": self.pad_id,
            "mask_id": self.mask_id,
            "cls_id": self.cls_id,
            "unk_id": self.unk_id,
        }
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        path = Path(path)
        with open(path) as f:
            obj = json.load(f)

        tok = cls.__new__(cls) 

        tok.id2code = obj["id2code"]
        tok.code2id = {tok_: i for i, tok_ in enumerate(tok.id2code)}
        tok.vocab_size = len(tok.id2code)

        tok.special_tokens = obj.get("special_tokens", [])
        tok.type2id = obj.get("type2id", {})

        tok.pad_token  = "[PAD]" if "[PAD]" in tok.code2id else None
        tok.mask_token = "[MASK]" if "[MASK]" in tok.code2id else None
        tok.cls_token  = "[CLS]" if "[CLS]" in tok.code2id else None
        tok.unk_token  = "[UNK]" if "[UNK]" in tok.code2id else None

        tok.pad_id  = obj.get("pad_id", tok.code2id.get("[PAD]", 0))
        tok.mask_id = obj.get("mask_id", tok.code2id.get("[MASK]")) if "[MASK]" in tok.code2id else None
        tok.cls_id  = obj.get("cls_id", tok.code2id.get("[CLS]"))   if "[CLS]" in tok.code2id else None
        tok.unk_id  = obj.get("unk_id", tok.code2id.get("[UNK]"))   if "[UNK]" in tok.code2id else None

        # Rebuild Polars lookup frames
        tok._code2id_df = pl.DataFrame({"code": tok.id2code,
                                        "input_id": list(range(tok.vocab_size))}) \
                              .with_columns(pl.col("code").cast(pl.Categorical))
        tok._type2id_df = pl.DataFrame({"code_type": list(tok.type2id.keys()),
                                        "type_id":   list(tok.type2id.values())}) \
                              .with_columns(pl.col("code_type").cast(pl.Categorical))
        return tok

    @property
    def code2id_df(self) -> pl.DataFrame:
        return self._code2id_df

    @property
    def type2id_df(self) -> pl.DataFrame:
        return self._type2id_df
    



class SequencesGenerator:
    def __init__(
        self,
        tokenizer_path: str,
        chunk_length: int = 1024,
        overlap: int = 128,
        return_numeric: bool = False,
        return_text: bool = False,
        return_time: bool = False,
        return_ids: bool = False,
        dataset_name: str = "mimic",
    ):

        if tokenizer_path is not None:
            self.tokenizer = Tokenizer.load(tokenizer_path)
            
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.return_numeric = return_numeric
        self.return_text = return_text
        self.return_time = return_time
        self.return_ids = return_ids
        self.dataset_name = dataset_name
        
    def encode_sequence(
        self,
        timeline: pl.DataFrame,
        max_length: Optional[int] = None,
        pad_to_max: bool = False,
        truncation: Literal["head", "tail"] = "tail",
        add_cls: bool = False,
    ) -> Dict[str, Union[List[int], List[float], List[str]]]:
        """
        Vectorized build of:
          input_ids, attention_mask, visit_ids, stage_ids, type_ids
          + optional numeric/text streams (+ masks)
        """
        df = timeline
        
        if "seq_id" in df.columns:

            uniq = df.select(pl.col("seq_id")).unique(maintain_order=True)
            uniq = uniq.with_row_count(name="visit_ids_raw")  # 0..K-1
            df = df.join(uniq, on="seq_id", how="left").with_columns(
                (pl.col("visit_ids_raw") ).alias("visit_id").fill_null(0)
            ).drop("visit_ids_raw")
        else:
            df = df.with_columns(pl.lit(0).alias("visit_id"))

        stage_cols = ["out_id", "er_id", "hadm_id", "icustay_id"]
        present_stages = [c for c in stage_cols if c in df.columns]
        if present_stages:

            expr = pl.lit(0)
            for i, col in enumerate(present_stages, start=1):
                expr = pl.when(expr.eq(0) & pl.col(col).is_not_null()).then(i).otherwise(expr)
            df = df.with_columns(expr.alias("stage_id"))
        else:
            df = df.with_columns(pl.lit(0).alias("stage_id"))

        df = df.join(
            self.tokenizer.type2id_df,
            on=pl.col("code_type").cast(pl.Categorical),
            how="left",
        ).with_columns(pl.col("type_id").fill_null(0))

        df = df.join(
            self.tokenizer.code2id_df,
            on=pl.col("code").cast(pl.Categorical),
            how="left",
        )


        unk_id = self.tokenizer.unk_id if self.tokenizer.unk_id is not None else self.tokenizer.pad_id or 0
        df = df.with_columns(pl.col("input_id").fill_null(unk_id))

        df = df.with_columns(
            pl.when(pl.col("code").str.starts_with("TIME-GAP//"))
              .then(0)
              .otherwise(pl.col("visit_id"))
              .alias("visit_id"),
            pl.when(pl.col("code").str.starts_with("TIME-GAP//"))
              .then(0)
              .otherwise(pl.col("stage_id"))
              .alias("stage_id"),
        )


        if add_cls and (self.tokenizer.cls_token is not None):
            cls_row = {
                "code": self.tokenizer.cls_token,
                "code_type": "[CLS]",
                "visit_id": 0,
                "stage_id": 0,
                "type_id": self.tokenizer.type2id.get("[CLS]", 0),
                "input_id": self.tokenizer.cls_id,
            }

            if self.return_numeric:
                cls_row["numeric_value"] = None
            if self.return_text:
                cls_row["text_value"] = None

            df = pl.concat([pl.DataFrame([cls_row]), df], how="vertical_relaxed")


        input_ids = df.get_column("input_id").cast(pl.Int64).to_list()
        type_ids = df.get_column("type_id").cast(pl.Int64).to_list()
        visit_ids = df.get_column("visit_id").cast(pl.Int64).to_list()
        stage_ids = df.get_column("stage_id").cast(pl.Int64).to_list()
        attention_mask = [1] * len(input_ids)


        value_payload = self._build_value_streams(
            df=df,
            max_length=max_length,
            pad_to_max=pad_to_max,
            truncation=truncation,
        )

        # --- truncate/pad core streams in one go ---
        input_ids      = self._truncate(input_ids,      max_length, truncation)
        type_ids       = self._truncate(type_ids,       max_length, truncation)
        visit_ids      = self._truncate(visit_ids,      max_length, truncation)
        stage_ids      = self._truncate(stage_ids,      max_length, truncation)
        attention_mask = [1] * len(input_ids)

        if pad_to_max and max_length is not None and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            pad_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else 0
            input_ids      = input_ids + [pad_id] * pad_len
            type_ids       = type_ids + [0] * pad_len
            visit_ids      = visit_ids + [0] * pad_len
            stage_ids      = stage_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "visit_ids": visit_ids,
            "stage_ids": stage_ids,
            "type_ids": type_ids,
        }
        out.update(value_payload)
        return out

    def get_overlapped_chunks(
        self,
        timeline: Dict[str, Iterable],
        chunk_length: Optional[int] = None,
        overlap: Optional[int] = None,
        add_cls_per_chunk: bool = True,
    ) -> List[Dict[str, List[Any]]]:

        if chunk_length is None:
            chunk_length = self.chunk_length
        if overlap is None:
            overlap = self.overlap

        if getattr(self, "dataset_name", None) == "ehrshot":
            fields = [
                "tokens",
                "valid_tokens",
                "ages",
                "normalized_ages",
                "timestamps",
            ]

            n = len(timeline["tokens"])
            if n == 0:
                return []

            payload = chunk_length
            step = max(1, payload - overlap)

            num_chunks = 1 if n <= payload else ceil((n - payload) / step) + 1
            starts = [i * step for i in range(num_chunks)]

            chunks = []
            for start in starts:
                end = min(n, start + payload)

                sliced = {
                    k: list(timeline[k][start:end])
                    for k in fields
                    if k in timeline
                }

                chunk_len = len(sliced["tokens"])

                sliced["valid_tokens"] = [True] * chunk_len
                sliced["patient_lengths"] = [chunk_len]
                sliced["label_indices"] = []

                chunks.append(sliced)

            return chunks

        fields = ["input_ids", "attention_mask", "visit_ids", "stage_ids", "type_ids"]
        for extra in ("numeric_values", "numeric_mask", "text_values", "text_mask", "time_diff"):
            if extra in timeline and extra not in fields:
                fields.append(extra)

        n = len(timeline["input_ids"])
        payload = chunk_length - (1 if add_cls_per_chunk else 0)
        step = max(1, payload - overlap)

        num_chunks = 1 if n <= payload else ceil((n - payload) / step) + 1
        starts = [i * step for i in range(num_chunks)]

        cls_defaults = {
            "input_ids": self.tokenizer.cls_id if self.tokenizer.cls_id is not None else (self.tokenizer.pad_id or 0),
            "attention_mask": 1,
            "visit_ids": 0,
            "stage_ids": 0,
            "type_ids": self.tokenizer.type2id.get("[CLS]", 0),
            "numeric_values": 0.0,
            "numeric_mask": 0,
            "text_values": "",
            "text_mask": 0,
            "time_diff": 0.0,
        }

        chunks = []
        for start in starts:
            end = min(n, start + payload)
            sliced = {k: list(timeline[k][start:end]) for k in fields if k in timeline}

            if "attention_mask" in sliced:
                sliced["attention_mask"] = [1] * len(sliced["input_ids"])

            if add_cls_per_chunk:
                for k in list(sliced.keys()):
                    sliced[k] = [cls_defaults[k]] + sliced[k]

            cur_len = len(sliced["input_ids"])
            if cur_len < chunk_length:
                pad_len = chunk_length - cur_len
                for k in list(sliced.keys()):
                    sliced[k] = self._pad_list(sliced[k], pad_len, 0)

            chunks.append(sliced)

        return chunks


    def _build_value_streams(
        self,
        df: pl.DataFrame,
        max_length: Optional[int],
        pad_to_max: bool,
        truncation: Literal["head", "tail"],
    ) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {}
        # Numeric stream
        if self.return_numeric:
            if "numeric_value" in df.columns:
                vals = df.get_column("numeric_value").to_list()
            else:
                vals = [None] * df.height
            num_mask = [1 if (v is not None) else 0 for v in vals]
            vals = [0.0 if v is None else float(v) for v in vals]

            vals = self._truncate(vals, max_length, truncation)
            num_mask = self._truncate(num_mask, max_length, truncation)

            if pad_to_max and max_length is not None and len(vals) < max_length:
                pad_len = max_length - len(vals)
                vals += [0.0] * pad_len
                num_mask += [0] * pad_len

            out["numeric_values"] = vals
            out["numeric_mask"] = num_mask

        # Text stream
        if self.return_text:
            if "text_value" in df.columns:
                txt = df.get_column("text_value").to_list()
            else:
                txt = [None] * df.height
            txt = [("" if (t is None or str(t) == "___") else str(t)) for t in txt]
            txt_mask = [1 if (t != "") else 0 for t in txt]

            txt = self._truncate(txt, max_length, truncation)
            txt_mask = self._truncate(txt_mask, max_length, truncation)

            if pad_to_max and max_length is not None and len(txt) < max_length:
                pad_len = max_length - len(txt)
                txt += [""] * pad_len
                txt_mask += [0] * pad_len

            out["text_values"] = txt
            out["text_mask"] = txt_mask

            
        if self.return_time:
            if "time_diff" in df.columns:
                df = df.with_columns(pl.col(['time_diff'])).fill_null(0.0)
                time_diff = df.get_column("time_diff").to_list()
                time_diff = self._scale_time_deltas(time_diff)
                time_stamp = df.get_column("time").to_list()
            else:
                time_diff = [None] * df.height
                time_stamp = [None] * df.height


            if pad_to_max and max_length is not None and len(time_diff) < max_length:
                pad_len = max_length - len(time_diff)
                time_diff += [0] * pad_len
                time_stamp += [0] * pad_len


            out["time_diff"] = time_diff
            out["time_stamp"] = time_stamp
            
        if self.return_ids:
            if "seq_id" in df.columns:
                
                seq_id = df.get_column("seq_id").cast(pl.Int32).to_list()
                out_id = df.get_column("out_id").cast(pl.Int32).to_list()
                er_id =  df.get_column("er_id").cast(pl.Int32).to_list()
                hadm_id = df.get_column("hadm_id").cast(pl.Int32).to_list()
                icustay_id = df.get_column("hadm_id").cast(pl.Int32).to_list()
            else:
                seq_id = [None] * df.height
                out_id = [None] * df.height
                er_id =  [None] * df.height
                hadm_id = [None] * df.height
                icustay_id = [None] * df.height

            if pad_to_max and max_length is not None and len(time_diff) < max_length:
                pad_len = max_length - len(time_diff)
                seq_id += [0] * pad_len
                out_id += [0] * pad_len
                er_id += [0] * pad_len
                hadm_id += [0] * pad_len
                icustay_id += [0] * pad_len

            out["seq_id"] = seq_id
            out["out_id"] = out_id
            out["er_id"] = er_id
            out["hadm_id"] = hadm_id
            out["icustay_id"] = icustay_id

        return out
    
    def get_time_based_chunks(
        self,
        timeline: Dict[str, Iterable],
        window_hours: float,
        anchor_from_first_valid_time: bool = True,
        keep_prefix_tokens: bool = True,
    ) -> List[Dict[str, List[Any]]]:

        if getattr(self, "dataset_name", None) == "ehrshot":
            if "tokens" not in timeline:
                raise ValueError("timeline must contain 'tokens'")
            if "timestamps" not in timeline:
                raise ValueError("timeline must contain 'timestamps' for EHRShot time-based chunking")

            n = len(timeline["tokens"])
            if n == 0:
                return []

            fields = [
                "tokens",
                "valid_tokens",
                "ages",
                "normalized_ages",
                "timestamps",
            ]

            time_stamps = list(timeline["timestamps"])

            first_clinical_idx = None
            for i in range(1, n):
                if time_stamps[i] > time_stamps[i - 1]:
                    if i > 0 and (time_stamps[i] - time_stamps[i - 1]) > 365 * 24 * 3600:
                        first_clinical_idx = i
                        break

            if first_clinical_idx is None:
                # Fallback: no obvious prefix/clinical split.
                first_clinical_idx = 0

            prefix_idx = list(range(first_clinical_idx)) if keep_prefix_tokens else []

            t0 = time_stamps[first_clinical_idx]
            window_size = window_hours * 3600.0

            if window_size <= 0:
                raise ValueError("window_hours must be > 0")

            buckets: Dict[int, List[int]] = {}

            for i in range(first_clinical_idx, n):
                elapsed = time_stamps[i] - t0
                window_id = int(elapsed // window_size) if elapsed >= 0 else 0
                buckets.setdefault(window_id, []).append(i)

            raw_chunks: List[Dict[str, List[Any]]] = []

            for chunk_idx, window_id in enumerate(sorted(buckets.keys())):
                if chunk_idx == 0:
                    idxs = prefix_idx + buckets[window_id]
                else:
                    idxs = buckets[window_id]

                chunk = {
                    k: [timeline[k][j] for j in idxs]
                    for k in fields
                    if k in timeline
                }

                chunk_len = len(chunk["tokens"])
                chunk["valid_tokens"] = [True] * chunk_len
                chunk["patient_lengths"] = [chunk_len]
                chunk["label_indices"] = []

                raw_chunks.append(chunk)

            final_chunks: List[Dict[str, List[Any]]] = []

            for chunk in raw_chunks:
                if len(chunk["tokens"]) <= self.chunk_length:
                    final_chunks.append(chunk)
                else:
                    sub_chunks = self.get_overlapped_chunks(
                        timeline=chunk,
                        chunk_length=self.chunk_length,
                        overlap=0,
                        add_cls_per_chunk=False,
                    )
                    final_chunks.extend(sub_chunks)

            return final_chunks


        if "input_ids" not in timeline:
            raise ValueError("timeline must contain 'input_ids'")
        if "time_stamp" not in timeline:
            raise ValueError("timeline must contain 'time_stamp' for time-based chunking")

        n = len(timeline["input_ids"])
        if n == 0:
            return []

        fields = [k for k, v in timeline.items() if isinstance(v, (list, tuple))]
        time_stamps = list(timeline["time_stamp"])

        def _is_valid_time(x: Any) -> bool:
            return x is not None

        anchor_idx = None
        for i, ts in enumerate(time_stamps):
            if _is_valid_time(ts):
                anchor_idx = i
                break

        if anchor_idx is None:
            return [{k: list(v) for k, v in timeline.items() if isinstance(v, (list, tuple))}]

        t0 = time_stamps[anchor_idx]
        window_size = window_hours * 3600.0

        if window_size <= 0:
            raise ValueError("window_hours must be > 0")

        prefix_idx = list(range(anchor_idx)) if keep_prefix_tokens else []

        buckets: Dict[int, List[int]] = {}
        for i in range(anchor_idx, n):
            ts = time_stamps[i]

            if not _is_valid_time(ts):
                window_id = 0
            else:
                elapsed = (ts - t0).total_seconds()
                window_id = int(elapsed // window_size) if elapsed >= 0 else 0

            buckets.setdefault(window_id, []).append(i)

        chunks: List[Dict[str, List[Any]]] = []
        for chunk_idx, window_id in enumerate(sorted(buckets.keys())):
            idxs = (prefix_idx + buckets[window_id]) if chunk_idx == 0 else buckets[window_id]

            chunk = {}
            for k in fields:
                chunk[k] = [timeline[k][j] for j in idxs]

            chunks.append(chunk)

        final_chunks = []

        for chunk in chunks:
            sub_chunks = self.get_overlapped_chunks(
                timeline=chunk,
                chunk_length=self.chunk_length,
                overlap=0,
                add_cls_per_chunk=True,
            )
            final_chunks.extend(sub_chunks)

        return final_chunks
    
    def get_visit_level_chunks(
        self,
        timeline: Dict[str, Iterable],
        keep_prefix_tokens: bool = True,
    ) -> List[Dict[str, List[Any]]]:

        if "input_ids" not in timeline:
            raise ValueError("timeline must contain 'input_ids'")
        if "seq_id" not in timeline:
            raise ValueError("timeline must contain 'seq_id' for visit-level chunking")

        n = len(timeline["input_ids"])
        if n == 0:
            return []

        fields = [k for k, v in timeline.items() if isinstance(v, (list, tuple))]
        seq_ids = list(timeline["seq_id"])

        def _is_valid_visit(x: Any) -> bool:
            return x is not None and x != 0

        # find first actual visit event
        first_visit_idx = None
        for i, sid in enumerate(seq_ids):
            if _is_valid_visit(sid):
                first_visit_idx = i
                break

        # if no valid seq_id exists, return overlapped chunks on full sequence
        if first_visit_idx is None:
            return self.get_overlapped_chunks(
                timeline={k: list(v) for k, v in timeline.items() if isinstance(v, (list, tuple))},
                chunk_length=self.chunk_length,
                overlap=0,
                add_cls_per_chunk=True,
            )

        prefix_idx = list(range(first_visit_idx)) if keep_prefix_tokens else []

        buckets: Dict[Any, List[int]] = {}
        visit_order: List[Any] = []

        for i in range(first_visit_idx, n):
            sid = seq_ids[i]
            if not _is_valid_visit(sid):
                continue

            if sid not in buckets:
                buckets[sid] = []
                visit_order.append(sid)
            buckets[sid].append(i)

        raw_chunks: List[Dict[str, List[Any]]] = []
        for chunk_idx, sid in enumerate(visit_order):
            idxs = (prefix_idx + buckets[sid]) if chunk_idx == 0 else buckets[sid]

            chunk = {}
            for k in fields:
                chunk[k] = [timeline[k][j] for j in idxs]

            raw_chunks.append(chunk)

        final_chunks: List[Dict[str, List[Any]]] = []
        for chunk in raw_chunks:
            sub_chunks = self.get_overlapped_chunks(
                timeline=chunk,
                chunk_length=self.chunk_length,
                overlap=0,
                add_cls_per_chunk=True,
            )
            final_chunks.extend(sub_chunks)

        return final_chunks

    
    def get_care_stage_level_chunks(
        self,
        timeline: Dict[str, Iterable],
        keep_prefix_tokens: bool = True,
    ) -> List[Dict[str, List[Any]]]:

        if "input_ids" not in timeline:
            raise ValueError("timeline must contain 'input_ids'")
        if "seq_id" not in timeline:
            raise ValueError("timeline must contain 'seq_id' for care-stage chunking")

        n = len(timeline["input_ids"])
        if n == 0:
            return []

        fields = [k for k, v in timeline.items() if isinstance(v, (list, tuple))]

        seq_ids = list(timeline["seq_id"]) if "seq_id" in timeline else [None] * n
        er_ids = list(timeline["er_id"]) if "er_id" in timeline else [None] * n
        out_ids = list(timeline["out_id"]) if "out_id" in timeline else [None] * n
        hadm_ids = list(timeline["hadm_id"]) if "hadm_id" in timeline else [None] * n
        icu_ids = list(timeline["icustay_id"]) if "icustay_id" in timeline else [None] * n

        def _valid(x: Any) -> bool:
            return x is not None and x != 0

        # make hadm mutually exclusive with icu at the row level
        hadm_ids = [
            None if _valid(icu) else hadm
            for hadm, icu in zip(hadm_ids, icu_ids)
        ]

        # per-row stage priority
        def _stage_key(i: int):
            if _valid(icu_ids[i]):
                return ("icu", icu_ids[i])
            elif _valid(hadm_ids[i]):
                return ("hadm", hadm_ids[i])
            elif _valid(er_ids[i]):
                return ("er", er_ids[i])
            elif _valid(out_ids[i]):
                return ("out", out_ids[i])
            return None

        # first row that belongs to a visit
        first_visit_idx = None
        for i in range(n):
            if _valid(seq_ids[i]):
                first_visit_idx = i
                break

        if first_visit_idx is None:
            return self.get_overlapped_chunks(
                timeline={k: list(v) for k, v in timeline.items() if isinstance(v, (list, tuple))},
                chunk_length=self.chunk_length,
                overlap=0,
                add_cls_per_chunk=True,
            )

        prefix_idx = list(range(first_visit_idx)) if keep_prefix_tokens else []

        # group by (seq_id, stage_type, stage_id), preserving first-seen order
        buckets: Dict[Any, List[int]] = {}
        chunk_order: List[Any] = []

        for i in range(first_visit_idx, n):
            if not _valid(seq_ids[i]):
                continue

            stage = _stage_key(i)
            if stage is None:
                continue

            key = (seq_ids[i], stage[0], stage[1])

            if key not in buckets:
                buckets[key] = []
                chunk_order.append(key)
            buckets[key].append(i)

        final_chunks: List[Dict[str, List[Any]]] = []

        for chunk_idx, key in enumerate(chunk_order):
            idxs = (prefix_idx + buckets[key]) if chunk_idx == 0 else buckets[key]

            chunk = {}
            for k in fields:
                chunk[k] = [timeline[k][j] for j in idxs]

            sub_chunks = self.get_overlapped_chunks(
                timeline=chunk,
                chunk_length=self.chunk_length,
                overlap=0,
                add_cls_per_chunk=True,
            )

#             for sub_chunk in sub_chunks:
#                 sub_chunk["source_seq_id"] = key[0]
#                 sub_chunk["source_stage_type"] = key[1]
#                 sub_chunk["source_stage_id"] = key[2]

            final_chunks.extend(sub_chunks)

        return final_chunks
    
    def _scale_time_deltas(self, deltas_list):
        deltas = np.asarray(deltas_list, dtype=float)
        compressed = np.log1p(deltas)              
        scaled = compressed / np.log(5328.93125)         
        return scaled.tolist()

    @staticmethod
    def _truncate(seq: List[Any], max_length: Optional[int], truncation: str) -> List[Any]:
        if max_length is None or len(seq) <= max_length:
            return seq
        return seq[-max_length:] if truncation == "head" else seq[:max_length]

    @staticmethod
    def _pad_list(lst: List[Any], pad_len: int, pad_value: Any) -> List[Any]:
        if pad_len <= 0:
            return lst
        return lst + [pad_value] * pad_len






class EHRPretrainDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 data_idx_path: str,
                 seq_generator: SequencesGenerator,
                 needed_cols: list = ['subject_id', 'input_ids', 'attention_mask', 'visit_ids', 'stage_ids', 'type_ids'],
                 split: str = 'all') -> None:
        
        hf_dataset = load_from_disk(dataset_path)
        self.hf_dataset = hf_dataset.flatten_indices().select_columns(needed_cols) \
                                      .with_format("numpy", columns=needed_cols, output_all_columns=False)
        
        self.seq_generator = seq_generator
        
        sids = self.hf_dataset["subject_id"]        
        self.index = defaultdict(list)
        for i, sid in enumerate(sids):
            self.index[sid].append(i)
        
        data_idx =  pl.scan_parquet(data_idx_path).collect()
        splits = {'all': data_idx,
                  'train':data_idx.filter(pl.col('split') == 'train'),
                  'val':  data_idx.filter(pl.col('split') == 'val')}

        self.data_idx, self.cum, self.subj = self._get_chunks_count(data_idx=splits[split],
                                                                    chunk_length=self.seq_generator.chunk_length,
                                                                    overlap=self.seq_generator.overlap)
        
        
    def __len__(self) -> int:
        return self.cum[-1]


    
    def __getitem__(self,
                    idx: int):
        

        subject_id, chunk_id = self._get_chunk_at_idx(idx=idx,
                                                      cumm_sum=self.cum,
                                                      subjects=self.subj)
        

        timeline_encoded = self.hf_dataset.select(self.index[subject_id])[0]
        chunks = self.seq_generator.get_overlapped_chunks(timeline= timeline_encoded,
                                                          chunk_length= self.seq_generator.chunk_length,
                                                          overlap=self.seq_generator.overlap)
        

        return chunks[chunk_id]
    

    def _build_dataset_index(self,
                             data_path, 
                             subject_col="subject_id") -> pl.DataFrame:
        pieces = []
        for p in os.listdir(data_path):
            df = (pl.scan_parquet(os.path.join(data_path,p)).select(subject_col).collect()
                    .group_by(subject_col)
                    .len()
                    .rename({"len": "n_events"})
                 )
            df = df.with_columns(pl.lit(str(p)).alias("shard"))  # optional
            pieces.append(df)


        df = (pl.concat(pieces, how="vertical")
                  .group_by([subject_col, "shard"])
                  .agg(pl.col("n_events").sum())
                  .rename({subject_col:"subject_id"})).sort('subject_id')

        df = df.filter(pl.col('n_events') >3)

        return df
    
    
    def _read_timeline(self,
                       subject_id:int) -> pl.DataFrame:
        
        shard = self.data_idx.filter(pl.col('subject_id') == subject_id)['shard'][0]

        data = pl.scan_parquet(os.path.join(self.data_path,shard),parallel='auto').select(
                                            ['subject_id','seq_id','out_id','er_id','hadm_id', 
                                             'icustay_id','time','code','numeric_value','code_type',
                                             'text_value']).filter(
                                              pl.col('subject_id') == subject_id).collect()
        return data

    
    def _get_chunks_count(self,
                          data_idx: pl.DataFrame,
                          chunk_length: int,
                          overlap: int):
        payload  = chunk_length - 1
        step     = payload - overlap

        data_idx = data_idx.with_columns(
            pl.col("n_events")
              .map_elements(lambda n: 1 if n<=payload else ceil((n-payload)/step)+1,return_dtype=pl.Int32)
              .alias("n_chunks")
        )
        data_idx = data_idx.with_columns(
            pl.col("n_chunks").cum_sum().alias("cum_chunks")
        )

        cum  = data_idx["cum_chunks"]   
        subj = data_idx["subject_id"]
        shards = data_idx['shard']
        return data_idx, cum, subj

    def _get_chunk_at_idx(self,
                          cumm_sum: list,
                          subjects: list,
                          idx: int) -> Tuple[int,int,str]:
        i = bisect.bisect_right(cumm_sum, idx)
        left = cumm_sum[i-1] if i > 0 else 0
        return subjects[i], idx - left





    


class MLMDataCollator:
    def __init__(
        self,
        tokenizer,
        protected_tokens: List[str],
        mask_prob: float = 0.15,
        replace_prob: float = 0.80,
        random_prob: float = 0.10,
    ) -> None:

        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.protected_ids = self._build_protected_ids(protected_tokens)
        if tokenizer.mask_id is None:
            raise ValueError("Tokenizer must define a [MASK] token/id.")

    def __call__(self, batch: List[Union[Dict, List[Dict]]]) -> Dict[str, torch.Tensor]:
        chunks = self._flatten(batch)
        out = self._stack(chunks)
        masked_ids, labels = self._mask_batch(out["input_ids"], out["attention_mask"])
        out["input_ids"] = masked_ids
        out["labels"] = labels
        return out


    def _flatten(self, batch) -> List[Dict]:
        out: List[Dict] = []
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

    def _build_protected_ids(self, protected_tokens: List[str]) -> torch.BoolTensor:

        ids = set()
        for tok in protected_tokens:
            if tok in self.tokenizer.code2id:
                ids.add(self.tokenizer.code2id[tok])
        mask = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
        for i in ids:
            mask[i] = True
        return mask

    def _mask_batch(
        self,
        input_ids: torch.Tensor,      
        attention_mask: torch.Tensor  
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = input_ids.device
        prot = self.protected_ids.to(device)
        eligible = attention_mask.bool() & (~prot[input_ids])

        sample = torch.rand_like(input_ids.float()) < self.mask_prob
        to_mask = eligible & sample

        masked = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        labels[to_mask] = input_ids[to_mask]

        r = torch.rand_like(input_ids.float())
        to_mask80 = to_mask & (r < self.replace_prob)
        to_rand10 = to_mask & (r >= self.replace_prob) & (r < self.replace_prob + self.random_prob)

        # 80% -> [MASK]
        masked[to_mask80] = self.tokenizer.mask_id

        # 10% -> random allowed token
        allowed = (~prot).nonzero(as_tuple=False).squeeze(1).to(device)
        if allowed.numel() == 0:
            allowed = torch.arange(self.tokenizer.vocab_size, device=device)
        if to_rand10.any():
            rand_ids = allowed[torch.randint(0, allowed.numel(), (to_rand10.sum(),), device=device)]
            masked[to_rand10] = rand_ids
        return masked, labels
    

PROTECTED_TOKENS = [
    "[PAD]", "[CLS]", "[MASK]",
    "OUTPATIENT-START","OUTPATIENT-END",
    "EMERGENCY-START","EMERGENCY-END",
    "ADMISSION-AT-HOSPITAL","ADMISSION-AT-ICU",
    "DISCHARGE-FROM-HOSPITAL","DISCHARGE-FROM-ICU"]

limits = {
    'within24_query': {512:  ['w24_start_512',  'w24_end_512' ],
                       1024: ['w24_start_1024', 'w24_end_1024'],
                       1536: ['w24_start_1536', 'w24_end_1536'],
                       2048: ['w24_start_2048', 'w24_end_2048'],
                      },
    
    'within24_hist_icu': {512: ['wStay_min', 'w24_start_512' ],
                         1024: ['wStay_min', 'w24_start_1024'],
                         1536: ['wStay_min', 'w24_start_1536'],
                         2048: ['wStay_min', 'w24_start_2048'],
                      },
    
    'within24_hist_full': {512: [ 0, 'w24_start_512' ],
                          1024: [ 0, 'w24_start_1024'],
                          1536: [ 0, 'w24_start_1536'],
                          2048: [ 0, 'w24_start_2048'],
                          },

    
    'within48_query': {512:  ['w48_start_512',  'w48_end_512' ],
                       1024: ['w48_start_1024', 'w48_end_1024'],
                       1536: ['w48_start_1536', 'w48_end_1536'],
                       2048: ['w48_start_2048', 'w48_end_2048'],
                      },

    'within48_hist_icu': {512: ['wStay_min', 'w48_start_512' ],
                         1024: ['wStay_min', 'w48_start_1024'],
                         1536: ['wStay_min', 'w48_start_1536'],
                         2048: ['wStay_min', 'w48_start_2048'],
                      },
    
    'within48_hist_full': {512: [ 0, 'w48_start_512' ],
                          1024: [ 0, 'w48_start_1024'],
                          1536: [ 0, 'w48_start_1536'],
                          2048: [ 0, 'w48_start_2048'],
                          },
    
    'within_stay_query': {512:  ['wStay_start_512',  'wStay_end_512' ],
                          1024: ['wStay_start_1024', 'wStay_end_1024'],
                          1536: ['wStay_start_1536', 'wStay_end_1536'],
                          2048: ['wStay_start_2048', 'wStay_end_2048'],
                         },
    
    'within_stay_hist_icu': {512:  ['wStay_min', 'wStay_start_512' ],
                             1024: ['wStay_min', 'wStay_start_1024'],
                             1536: ['wStay_min', 'wStay_start_1536'],
                             2048: ['wStay_min', 'wStay_start_2048'],
                            },
    
    'within_stay_hist_full': {512:  [ 0, 'wStay_start_512' ],
                              1024: [ 0, 'wStay_start_1024'],
                              1536: [ 0, 'wStay_start_1536'],
                              2048: [ 0, 'wStay_start_2048'],
                             },
    }

class EvalDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 data_idx_path: str,
                 seq_gen: SequencesGenerator,
                 limits_dict: dict,
                 task: str = 'y_mort',
                 main_window: str = 'within48_query', 
                 seq_length: int = 512,
                 use_time: bool = True,
                 use_numeric: bool = False,
                 add_cls=True,
                 split: str = 'train',
                 use_long_context: bool = False) -> None:
        
        needed_cols = ['subject_id', 'input_ids', 'attention_mask', 
                       'visit_ids', 'stage_ids', 'type_ids']

        
        
        BOUNDARIES = {"within24_query": ["w24_min", "w24_max"],
                      "within48_query": ["w48_min", "w48_max"],
                      "within_stay_query": ["wStay_min", "wStay_max"]}
        if use_time:
            needed_cols.append('time_diff')
        if use_numeric:
            needed_cols.append('numeric_values')
            needed_cols.append('numeric_mask')
            
            
        self.main_window = main_window
        self.seq_length = seq_length
        self.use_long_context = use_long_context


        if use_long_context:
            if main_window not in BOUNDARIES:
                raise ValueError(f"Long-context boundaries not defined for: {main_window}")
            self.horizon_min_col = BOUNDARIES[main_window][0]
            self.horizon_max_col = BOUNDARIES[main_window][1]
            self.start_limit = None
            self.end_limit = None
        else:
            self.start_limit = limits_dict[main_window][seq_length][0]
            self.end_limit = limits_dict[main_window][seq_length][1]
            self.horizon_min_col = None
            self.horizon_max_col = None
        
        self.task = task
        self.add_cls = add_cls
        self.seq_gen = seq_gen
        
        
        self.data_idx =  pl.scan_parquet(data_idx_path).collect()
        self.data_idx =  self.data_idx.filter(pl.col('split') == split)
        
        sub_ids = set(self.data_idx.get_column("subject_id").to_list())
        hf_dataset = load_from_disk(dataset_path)

        
        hf_dataset = hf_dataset.filter(
            lambda sids: [sid in sub_ids for sid in sids],
            batched=True,
            input_columns="subject_id",
        )

        self.hf_dataset = (
            hf_dataset
            .flatten_indices()
            .select_columns(needed_cols)
            .with_format("numpy", columns=needed_cols, output_all_columns=False)
        )
        
        
        sids = self.hf_dataset["subject_id"]        
        self.index = defaultdict(list)
        for i, sid in enumerate(sids):
            self.index[sid].append(i)
        

        
    def __len__(self) -> int:
        return len(self.data_idx)


    
    def __getitem__(self,
                    idx: int):
        
        stay = self.data_idx[idx]
        subject_id = stay['subject_id'][0]
        label = stay[self.task][0]
       
        

        if self.use_long_context:
            horizon_min = stay[self.horizon_min_col][0]
            horizon_max = stay[self.horizon_max_col][0]
            payload_len = self.seq_length - 1 if self.add_cls else self.seq_length
            
            end = horizon_max
            start = max(horizon_min, end - payload_len)
        else:
            start = stay[self.start_limit][0]
            end = stay[self.end_limit][0]
        
        
        timeline_encoded = self.hf_dataset.select(self.index[subject_id])[0]
        prediction_window = {k: (v[start:end] if isinstance(v, (list, np.ndarray)) else v) for k, v in timeline_encoded.items()}
        prediction_window = self.seq_gen.get_overlapped_chunks(prediction_window, add_cls_per_chunk=self.add_cls)
        prediction_window[0]['label'] = label
        
        return prediction_window[0]


class EvalCollator:
    def __init__(
        self,
        tokenizer=None,
        protected_tokens: List[str] = None,
        use_mask_augmentation: bool = False,
        augment_prob: float = 0.3,
        mask_prob: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        self.use_mask_augmentation = use_mask_augmentation
        self.augment_prob = augment_prob
        self.mask_prob = mask_prob

        if protected_tokens is None:
            protected_tokens = [
                "[PAD]", "[CLS]", "[MASK]",
                "OUTPATIENT-START", "OUTPATIENT-END",
                "EMERGENCY-START", "EMERGENCY-END",
                "ADMISSION-AT-HOSPITAL", "ADMISSION-AT-ICU",
                "DISCHARGE-FROM-HOSPITAL", "DISCHARGE-FROM-ICU",
            ]

        self.protected_ids = None
        if self.use_mask_augmentation:
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when use_mask_augmentation=True")
            if tokenizer.mask_id is None:
                raise ValueError("Tokenizer must define a [MASK] token/id.")
            self.protected_ids = self._build_protected_ids(protected_tokens)

    def __call__(
        self,
        batch: List[Union[Dict, List[Dict]]],
        apply_mask_augmentation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        chunks = self._flatten(batch)
        out = self._stack(chunks)

        if self.use_mask_augmentation and apply_mask_augmentation:
            out["input_ids"] = self._mask_batch(
                input_ids=out["input_ids"],
                attention_mask=out["attention_mask"],
            )

        if "numeric_values" in out:
            vals = out["numeric_values"].float()
            finite_mask = torch.isfinite(vals)

            if "numeric_mask" in out:
                mask = out["numeric_mask"].bool() & finite_mask
            else:
                mask = finite_mask

            vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

            out["numeric_values"] = vals
            out["numeric_mask"] = mask

        if "time_diff" in out:
            t = out["time_diff"].float()
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            out["time_diff"] = t

        return out

    def _flatten(self, batch) -> List[Dict]:
        out: List[Dict] = []
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
            if k in ("text_values",):
                continue

            seq_list = []
            for c in chunks:
                v = c[k]
                if isinstance(v, list):
                    v = [0 if x is None else x for x in v]
                elif v is None:
                    v = 0
                seq_list.append(torch.as_tensor(v))
            out[k] = torch.stack(seq_list, 0)
        return out

    def _build_protected_ids(self, protected_tokens: List[str]) -> torch.BoolTensor:
        ids = set()
        for tok in protected_tokens:
            if tok in self.tokenizer.code2id:
                ids.add(self.tokenizer.code2id[tok])

        mask = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
        for i in ids:
            mask[i] = True
        return mask

    def _mask_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = input_ids.device
        masked = input_ids.clone()

        prot = self.protected_ids.to(device)
        eligible = attention_mask.bool() & (~prot[input_ids])

        # choose which samples in batch get augmentation
        apply_aug = (torch.rand(input_ids.size(0), device=device) < self.augment_prob).unsqueeze(1)

        # choose which eligible tokens to mask
        to_mask = eligible & apply_aug & (torch.rand_like(input_ids.float()) < self.mask_prob)

        masked[to_mask] = self.tokenizer.mask_id
        return masked


class RetrievalDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 data_idx_path: str,
                 vectordb_path:str,
                 tokenizer_path:str,
                 limits_dict: dict,
                 chunking_strategy:str = 'overlap',
                 task: str = 'y_mort',
                 query_window: str = 'within48_query',
                 history_window: str = 'within48_hist_full', 
                 top_k: int = 8,
                 seq_length_q: int = 512,
                 overlap_q: int= 0,
                 seq_length_h: int = 256,
                 overlap_h: int = 0,
                 use_time: bool = True,
                 use_numeric: bool = False,
                 add_cls=True,
                 window_hours: float = 6.0,
                 uniform_retrieval: bool = False,
                 split: str = 'train') -> None:
        
        assert chunking_strategy in ['overlap','time','visit','care_stage']
        
        self.chunking_strategy = chunking_strategy
        self.uniform_retrieval = uniform_retrieval

        if self.chunking_strategy == 'overlap':
            needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids']
        elif self.chunking_strategy == 'time':
            needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids','time_stamp']
        elif self.chunking_strategy == 'visit':
            needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids','seq_id']
        elif self.chunking_strategy == 'care_stage':
            needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids',
                           'seq_id', 'out_id', 'er_id', 'hadm_id', 'icustay_id']
        
        
        
        if use_time:
            needed_cols.append('time_diff')
        if use_numeric:
            needed_cols.append('numeric_values')
            needed_cols.append('numeric_mask')
            
            
        self.start_limit_q = limits_dict[query_window][seq_length_q][0]
        self.end_limit_q   = limits_dict[query_window][seq_length_q][1]
        
        self.start_limit_h = limits_dict[history_window][seq_length_q][0]
        self.end_limit_h   = limits_dict[history_window][seq_length_q][1]
        
        
        self.task = task
        self.add_cls = add_cls
        
        self.query_gen = SequencesGenerator(tokenizer_path=tokenizer_path,
                                            chunk_length=seq_length_q,
                                            overlap=overlap_q)
        self.history_gen = SequencesGenerator(tokenizer_path=tokenizer_path,
                                            chunk_length=seq_length_h,
                                            overlap=overlap_h)
        
        
        self.vectordb_path = vectordb_path
        self.top_k = top_k
        self.window_hours = window_hours
        
        self.data_idx =  pl.scan_parquet(data_idx_path).collect()
        self.data_idx =  self.data_idx.filter(pl.col('split') == split)
        
        sub_ids = set(self.data_idx.get_column("subject_id").to_list())
        hf_dataset = load_from_disk(dataset_path)

        
        hf_dataset = hf_dataset.filter(
            lambda sids: [sid in sub_ids for sid in sids],
            batched=True,
            input_columns="subject_id",)

        self.hf_dataset = (
            hf_dataset
            .flatten_indices()
            .select_columns(needed_cols)
            .with_format("numpy", columns=needed_cols, output_all_columns=False))
        
        
        sids = self.hf_dataset["subject_id"]        
        self.index = defaultdict(list)
        for i, sid in enumerate(sids):
            self.index[sid].append(i)
        

        
    def __len__(self) -> int:
        return len(self.data_idx)
    
    def _to_list(self, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list):
            return v
        return v
    def __getitem__(self,
                    idx: int):
        
        stay = self.data_idx[idx]
        subject_id = stay['subject_id'][0]
        stay_id = stay['icustay_id'][0]
        label = stay[self.task][0]
        
        timeline_encoded = self.hf_dataset.select(self.index[subject_id])[0]
        timeline_encoded = {k: self._to_list(v) for k, v in timeline_encoded.items()}
        
        start_q = stay[self.start_limit_q][0]
        end_q = stay[self.end_limit_q][0]
        start_h = stay[self.start_limit_h][0] if isinstance(self.start_limit_h, str) else int(self.start_limit_h)
        end_h = stay[self.end_limit_h][0] if isinstance(self.end_limit_h, str) else int(self.end_limit_h)


        query = {k: (v[start_q:end_q] if isinstance(v, (list, np.ndarray)) else v) for k, v in timeline_encoded.items()}
        query = self.query_gen.get_overlapped_chunks(query, add_cls_per_chunk=self.add_cls)[0]
        
        history = {k: (v[start_h:end_h] if isinstance(v, (list, np.ndarray)) else v) for k, v in timeline_encoded.items()}
        
        if self.chunking_strategy == 'overlap':
            history = self.history_gen.get_overlapped_chunks(history, 
                                                             add_cls_per_chunk=self.add_cls)
        elif self.chunking_strategy == 'time':
            history = self.history_gen.get_time_based_chunks(timeline=history,
                                                             window_hours=self.window_hours,
                                                             keep_prefix_tokens=True,
                                                             anchor_from_first_valid_time=True)
        elif self.chunking_strategy == 'visit':
            history = self.history_gen.get_visit_level_chunks(timeline=history,
                                                              keep_prefix_tokens=True)
        elif self.chunking_strategy == 'care_stage':
            history = self.history_gen.get_care_stage_level_chunks(timeline=history,
                                                                   keep_prefix_tokens=True)
        allowed_keys = ["input_ids", "attention_mask", "visit_ids", "stage_ids","type_ids",
                        "time_diff", "numeric_values", "numeric_mask"]
        history = [{k: chunk[k] for k in allowed_keys if k in chunk} for chunk in history]
        
        
        if self.uniform_retrieval:
            num_hist = len(history)
            candidate_ids = list(range(num_hist))

            k = min(self.top_k, len(candidate_ids))
            ids = random.sample(candidate_ids, k)

            history = [history[i] for i in ids]
        
        else:
            history_idx = faiss.read_index(os.path.join(self.vectordb_path,f'{stay_id}.faiss'))
            qid = history_idx.ntotal-1
            query_embed = history_idx.reconstruct(qid)
            
            sim, ids = self._query_faiss(index=history_idx,q_emb=query_embed, top_k=self.top_k, metric='cosine')

            ids = ids[0].tolist()
            ids = [i for i in ids if (i != -1) and (i != qid) and (0 <= i < len(history))]
            
            history = [history[i] for i in ids]
            
        
        out = {'query': query,
               'history': history,
               'label':label}
        return out
    

    def _query_faiss(self,index, q_emb, top_k: int = 10, metric: str = "cosine"):

        xq = q_emb
        if xq.ndim == 1:
            xq = xq.reshape(1, -1)
        elif xq.ndim == 2:
            pass
        else:
            raise ValueError(f"q_emb must be (D,) or (B,D). Got {xq.shape}")

        xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)

        metric = metric.lower()

        if metric in ("cosine", "ip", "inner_product", "dot"):
#             faiss.normalize_L2(xq)
            scores, ids = index.search(xq, top_k)
            return scores, ids

        elif metric in ("l2", "euclidean"):
            dists, ids = index.search(xq, top_k)
            return dists, ids

        else:
            raise ValueError(f"Unknown metric={metric}. Use 'l2' or 'cosine'.")


class RetrievalCollator:
    def __init__(self, chunk_collator, top_k: int):
        self.chunk_collator = chunk_collator
        self.top_k = top_k

    def __call__(self, batch):
        query_chunks = [b["query"] for b in batch]
        q = self.chunk_collator(query_chunks, apply_mask_augmentation=False)

        retrieved_lists = [b["history"] for b in batch]

        # NEW: track which history slots are real
        valid_masks = []

        for i in range(len(retrieved_lists)):
            n_real = len(retrieved_lists[i])

            # template for padding
            template = query_chunks[i] if n_real == 0 else retrieved_lists[i][0]

            if n_real < self.top_k:
                z = {
                    k: ([0] * len(template[k]) if isinstance(template[k], list) else 0)
                    for k in template.keys()
                }
                retrieved_lists[i] = retrieved_lists[i] + [z] * (self.top_k - n_real)
            elif n_real > self.top_k:
                retrieved_lists[i] = retrieved_lists[i][:self.top_k]
                n_real = self.top_k

            # NEW: 1 for real, 0 for padded
            valid = [1] * n_real + [0] * (self.top_k - n_real)
            valid_masks.append(valid)

        flat_retrieved = [ch for lst in retrieved_lists for ch in lst]
        r_flat = self.chunk_collator(flat_retrieved, apply_mask_augmentation=True)

        B = len(batch)
        K = self.top_k
        r = {}
        for k, v in r_flat.items():
            if v.dim() == 2:
                r[k] = v.view(B, K, v.size(-1))
            else:
                r[k] = v.view(B, K, *v.shape[1:])

        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        # NEW: tensor mask
        history_valid_mask = torch.tensor(valid_masks, dtype=torch.long)

        return {"query": q, "history": r, "history_valid_mask": history_valid_mask, "label": labels}




class CLMBRRetrievalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_idx_path: str,
        vectordb_path: str,
        task: str,
        split: str = "train",
        top_k: int = 8,
        query_length: int = 512,
        history_chunk_length: int = 256,
        history_overlap: int = 0,
        chunking_strategy: str = "overlap",
        window_hours: float = 24.0,
    ) -> None:

        assert chunking_strategy in ["overlap", "time"]

        self.vectordb_path = vectordb_path
        self.task = task
        self.top_k = top_k
        self.query_length = query_length
        self.chunking_strategy = chunking_strategy
        self.window_hours = window_hours

        self.arrow_ds = load_from_disk(dataset_path)
        
        splits = {"train":"train",
                  "val":"tuning",
                  "test":"held_out"}
        split = splits[split]
        
        self.data_idx = (
            pl.read_parquet(data_idx_path)
            .filter((pl.col("task") == task) & (pl.col("split") == split))
        )

        self.history_gen = SequencesGenerator(
            tokenizer_path=None,
            chunk_length=history_chunk_length,
            overlap=history_overlap,
            dataset_name="ehrshot",
        )

    def __len__(self) -> int:
        return self.data_idx.height

    def _to_list(self, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, list):
            return v
        return v

    def _get_label(self, row):
        for col in ["boolean_value", "integer_value", "float_value", "categorical_value"]:
            value = row[col][0]
            if value is not None:
                return value
        raise ValueError("No valid label value found.")

    def _slice_timeline(self, timeline, start: int, end: int):
        out = {
            k: v[start:end]
            for k, v in timeline.items()
            if isinstance(v, list)
        }

        if "tokens" in out:
            n = len(out["tokens"])
            out["valid_tokens"] = [True] * n
            out["patient_lengths"] = [n]
            out["label_indices"] = []

        return out

    def __getitem__(self, idx: int):
        row = self.data_idx[idx]

        subject_id = int(row["subject_id"][0])
        example_id = int(row["example_id"][0])
        arrow_row_idx = int(row["arrow_row_idx"][0])

        prediction_idx = int(row["prediction_idx"][0])
        history_min_idx = int(row["history_min_idx"][0])

        label = self._get_label(row)

        sample = self.arrow_ds[arrow_row_idx]
        tr = sample["transformer"]

        timeline = {
            "tokens": self._to_list(tr["tokens"]),
            "valid_tokens": self._to_list(tr["valid_tokens"]),
            "ages": self._to_list(tr["ages"]),
            "normalized_ages": self._to_list(tr["normalized_ages"]),
            "timestamps": self._to_list(tr["timestamps"]),
        }

        query_start = max(history_min_idx, prediction_idx - self.query_length)
        query_end = prediction_idx

        history_start = history_min_idx
        history_end = query_start

        query = self._slice_timeline(timeline, query_start, query_end)

        history_timeline = self._slice_timeline(timeline, history_start, history_end)

        if len(history_timeline["tokens"]) == 0:
            history_chunks = []
        elif self.chunking_strategy == "overlap":
            history_chunks = self.history_gen.get_overlapped_chunks(
                timeline=history_timeline,
                add_cls_per_chunk=False,
            )
        elif self.chunking_strategy == "time":
            history_chunks = self.history_gen.get_time_based_chunks(
                timeline=history_timeline,
                window_hours=self.window_hours,
            )

        faiss_path = os.path.join(
            self.vectordb_path,
            f"{subject_id}_{example_id}.faiss",
        )

        history_index = faiss.read_index(faiss_path)

        qid = history_index.ntotal - 1
        query_embed = history_index.reconstruct(qid)

        _, ids = self._query_faiss(
            index=history_index,
            q_emb=query_embed,
            top_k=self.top_k,
            metric="cosine",
        )

        ids = ids[0].tolist()

        ids = [
            i for i in ids
            if (i != -1) and (i != qid) and (0 <= i < len(history_chunks))
        ]

        retrieved_history = [history_chunks[i] for i in ids]

        return {
            "query": query,
            "history": retrieved_history,
            "label": label,
            "subject_id": subject_id,
            "example_id": example_id,
        }

    def _query_faiss(self, index, q_emb, top_k: int = 10, metric: str = "cosine"):
        xq = q_emb

        if xq.ndim == 1:
            xq = xq.reshape(1, -1)
        elif xq.ndim != 2:
            raise ValueError(f"q_emb must be (D,) or (B,D). Got {xq.shape}")

        xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)

        metric = metric.lower()

        if metric in ("cosine", "ip", "inner_product", "dot"):
            scores, ids = index.search(xq, top_k)
            return scores, ids

        if metric in ("l2", "euclidean"):
            dists, ids = index.search(xq, top_k)
            return dists, ids

        raise ValueError(f"Unknown metric={metric}. Use 'l2' or 'cosine'.")



class CLMBRRetrievalCollator:
    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        query_chunks = [b["query"] for b in batch]

        query_batch = pack_clmbr_chunks(
            query_chunks,
            patient_id=0,
        )

        retrieved_lists = [list(b["history"]) for b in batch]

        valid_masks = []
        padded_history_chunks = []

        for i, retrieved in enumerate(retrieved_lists):
            n_real = len(retrieved)

            if n_real > self.top_k:
                retrieved = retrieved[: self.top_k]
                n_real = self.top_k

            if n_real == 0:
                template = query_chunks[i]
            else:
                template = retrieved[0]

            dummy = self._make_dummy_chunk(template)

            if n_real < self.top_k:
                retrieved = retrieved + [dummy] * (self.top_k - n_real)

            valid_masks.append([1] * n_real + [0] * (self.top_k - n_real))
            padded_history_chunks.extend(retrieved)

        history_batch = pack_clmbr_chunks(
            padded_history_chunks,
            patient_id=0,
        )

        labels = torch.tensor(
            [self._label_to_float(b["label"]) for b in batch],
            dtype=torch.float32,
        )

        history_valid_mask = torch.tensor(
            valid_masks,
            dtype=torch.long,
        )

        return {
            "query": query_batch,
            "history": history_batch,
            "history_valid_mask": history_valid_mask,
            "label": labels,
        }

    def _make_dummy_chunk(self, template: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Create a minimal one-token dummy CLMBR chunk.

        It is not meant to carry information.
        It only preserves B*K packed sequence structure.
        Downstream modules must ignore it using history_valid_mask.
        """

        dummy = {
            "tokens": [0],
            "valid_tokens": [True],
            "ages": [0.0],
            "normalized_ages": [0.0],
            "timestamps": [0],
            "patient_lengths": [1],
            "label_indices": [],
        }

        return dummy

    @staticmethod
    def _label_to_float(label):
        if isinstance(label, bool):
            return float(label)

        if label is None:
            raise ValueError("Label is None.")

        return float(label)

