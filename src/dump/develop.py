import os
import json
import faiss
import torch
import numpy as np
import polars as pl

from torch import nn

from math import ceil
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, Iterable, List, Optional, Any, Union, Literal, Tuple, Callable


from lightning.pytorch.loggers import WandbLogger

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
    ):

        self.tokenizer = Tokenizer.load(tokenizer_path)
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.return_numeric = return_numeric
        self.return_text = return_text
        self.return_time = return_time
        self.return_ids = return_ids

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
        """
        Sliding-window chunking with optional [CLS] per chunk and padding.
        """
        if chunk_length is None or overlap is None:
            chunk_length = self.chunk_length
            overlap = self.overlap

        fields = ["input_ids", "attention_mask", "visit_ids", "stage_ids", "type_ids"]
        for extra in ("numeric_values", "numeric_mask", "text_values", "text_mask", 'time_diff'):
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
            "time_diff":0.0}

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
                add_cls_per_chunk=True
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

import random
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
                 split: str = 'train') -> None:
        
        assert chunking_strategy in ['overlap','time','visit','care_stage']
        
        self.chunking_strategy = chunking_strategy
        
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
        
        
        history_idx = faiss.read_index(os.path.join(self.vectordb_path,f'{stay_id}.faiss'))
        qid = history_idx.ntotal-1
        query_embed = history_idx.reconstruct(qid)
        
        sim, ids = self._query_faiss(index=history_idx,q_emb=query_embed, top_k=self.top_k, metric='cosine')

        ids = ids[0].tolist()
        ids = [i for i in ids if (i != -1) and (i != qid) and (0 <= i < len(history))]
        
        history = [history[i] for i in ids]
        
        

        # random sampling
        # exclude last chunk if it corresponds to query (your qid logic)
        # num_hist = len(history)
        # candidate_ids = list(range(num_hist))
        # # sample without replacement
        # k = min(self.top_k, len(candidate_ids))
        # ids = random.sample(candidate_ids, k)
        
        # keep
        # history = [history[i] for i in ids]
        
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
    




import os
import yaml
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist

from typing import Callable
from transformers import BertConfig
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from transformers import CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING




BERT_VARIANTS = {
    "bert": {},
    "medbert": dict(
        hidden_size=192,
        intermediate_size=64,
        num_attention_heads=6,
        num_hidden_layers=6,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ),

    "cehrbert": dict(
        hidden_size=128,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ),

    "behrt": dict(
        hidden_size=288,
        intermediate_size=512,
        num_attention_heads=12,
        num_hidden_layers=6,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ),

    "hibehrt": dict(
        hidden_size=150,
        intermediate_size=108,
        num_attention_heads=6,
        num_hidden_layers=4,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.3,
    ),
}



def get_config_and_model_cls(model_type: str, mode: str = "mlm", variant: str = None):
    assert mode in ["mlm", "eval", "causal"]

    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"Unknown model_type: {model_type}")

    config_cls = CONFIG_MAPPING[model_type]

    if mode == "mlm":
        model_cls = MODEL_FOR_MASKED_LM_MAPPING[config_cls]
    elif mode == "eval":
        model_cls = MODEL_MAPPING[config_cls]
    else:
        model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[config_cls]

    variant_kwargs = {}
    if variant is not None and issubclass(config_cls, BertConfig):
        variant_kwargs = BERT_VARIANTS.get(variant, {})
        if variant not in BERT_VARIANTS:
            raise ValueError(f"Unknown BERT variant: {variant}")

    def build_config(**kwargs):
        return config_cls(**variant_kwargs, **kwargs)

    return build_config, model_cls


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




import os
import torch


import lightning as lt
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

from torchmetrics.classification import Accuracy, BinaryAUROC, BinaryAveragePrecision

class EHREmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = 0,
        type_vocab_size: int = 28,
        visit_vocab_size: int = 102,
        stage_vocab_size: int = 5,
        dropout: float = 0.1,
        use_position_embeddings: bool = False,
        max_position_embeddings: int = 0,
        use_time: bool = False,
        time_in_features: int = 1,
        time_out_features: int = 16,
        use_numeric: bool = False,
        numeric_hidden_size: int = 16,   
    ):
        super().__init__()

        self.tok_emb   = nn.Embedding(vocab_size,       embedding_size, padding_idx=pad_token_id)
        self.type_emb  = nn.Embedding(type_vocab_size,  embedding_size, padding_idx=pad_token_id)
        self.visit_emb = nn.Embedding(visit_vocab_size, embedding_size, padding_idx=pad_token_id)
        self.stage_emb = nn.Embedding(stage_vocab_size, embedding_size, padding_idx=pad_token_id)

        
        self.use_position_embeddings = use_position_embeddings
        if use_position_embeddings:
            if max_position_embeddings <= 0:
                raise ValueError("max_position_embeddings must be > 0 when use_position_embeddings=True")
            self.pos_emb = nn.Embedding(max_position_embeddings, embedding_size)
        else:
            self.pos_emb = None

        
        self.use_time = use_time
        if use_time:
            self.time2vec = Time2Vec(
                in_features=time_in_features,
                out_features=time_out_features,
                periodic_activation=torch.sin,
            )
            self.time_proj = nn.Linear(time_out_features, embedding_size)
        else:
            self.time2vec = None
            self.time_proj = None

        
        self.use_numeric = use_numeric
        if use_numeric:
            self.numeric_hidden_size = numeric_hidden_size
            
            self.num_proj1 = nn.Linear(1, numeric_hidden_size)
            self.num_proj2 = nn.Linear(numeric_hidden_size, embedding_size)
            self.num_act = nn.GELU()

            
            self.null_numeric = nn.Parameter(torch.zeros(embedding_size))
            nn.init.normal_(self.null_numeric, mean=0.0, std=0.02)

            nn.init.xavier_uniform_(self.num_proj1.weight)
            nn.init.zeros_(self.num_proj1.bias)
            nn.init.xavier_uniform_(self.num_proj2.weight)
            nn.init.zeros_(self.num_proj2.bias)
        else:
            self.num_proj1 = None
            self.num_proj2 = None
            self.num_act = None
            self.null_numeric = None

        self.norm = nn.LayerNorm(embedding_size)
        self.drop = nn.Dropout(dropout)

    def encode(
        self,
        input_ids,
        type_ids,
        visit_ids,
        stage_ids,
        time_feats=None,          
        numeric_values=None,     
        numeric_mask=None,        
    ):
        
        x = self.tok_emb(input_ids.long())
        x = x + self.type_emb(type_ids.long())
        x = x + self.visit_emb(visit_ids.long())
        x = x + self.stage_emb(stage_ids.long())

        
        if self.pos_emb is not None:
            shape = input_ids.size()
            seqlen = shape[-1]

            position_ids = torch.arange(seqlen, device=input_ids.device)

            if input_ids.dim() == 2:        
                bsz = shape[0]
                position_ids = position_ids.unsqueeze(0).expand(bsz, seqlen)          
            elif input_ids.dim() == 3:        
                bsz, n = shape[0], shape[1]
                position_ids = position_ids.view(1, 1, seqlen).expand(bsz, n, seqlen) 
            else:
                raise ValueError(f"Unsupported input_ids.dim()={input_ids.dim()}")
            x = x + self.pos_emb(position_ids)

        
        if self.use_time:
            if time_feats is None:
                raise ValueError("time_feats must be provided when use_time=True")
            if time_feats.dim() == 2:
                time_feats = time_feats.unsqueeze(-1)
            elif time_feats.dim() != 3:
                raise ValueError(f"Unexpected time_feats.dim()={time_feats.dim()}, expected 2 or 3")
            t = self.time2vec(time_feats.float())   
            t = self.time_proj(t)                   
            x = x + t

        
        if self.use_numeric:
            if numeric_values is None or numeric_mask is None:
                raise ValueError("numeric_values and numeric_mask must be provided when use_numeric=True")

            
            v = numeric_values.float().unsqueeze(-1)       
            
            h = self.num_act(self.num_proj1(v))             
            num_emb = self.num_proj2(h)                     

            mask = numeric_mask.bool().unsqueeze(-1)        
            num_emb = torch.where(mask, num_emb, self.null_numeric.view(1, 1, -1))
            x = x + num_emb

        return self.drop(self.norm(x))

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            return inputs_embeds
        x = self.tok_emb(input_ids.long())
        return self.drop(self.norm(x))
    



class EHRRAPEncoders(nn.Module):
    def __init__(
        self,
        config,
        backbone,                 
        dropout: float = 0.1,
        pooling: str = "cls",       
        use_time: bool = False,
        use_numeric: bool = False,
        ckpt_path = None,
        return_token_level: bool = False,
    ):
        super().__init__()
        self.config = config
        self.pooling = pooling
        self.use_time = use_time
        self.use_numeric = use_numeric
        self.return_token_level = return_token_level
        self.backbone = backbone(config)

        rope_model_types = {"modernbert", "roformer","mamba"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types

        self.ehr_embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout,
            use_position_embeddings=not is_rope,
            max_position_embeddings=(getattr(config, "max_position_embeddings", 0) if not is_rope else 0),
            use_time=use_time,
            time_in_features=1,
            time_out_features=16,
            use_numeric=use_numeric,
        )

        
        if ckpt_path:
            self.get_pretrained_weights(ckpt_path=ckpt_path)


    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,L,1]
            summed = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            return summed/lengths

    def _encode(
        self,
        x: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=x["input_ids"],
            type_ids=x["type_ids"],
            visit_ids=x["visit_ids"],
            stage_ids=x["stage_ids"],
            time_feats=x.get("time_diff", None) if self.use_time else None,
            numeric_values=x.get("numeric_values", None) if self.use_numeric else None,
            numeric_mask=x.get("numeric_mask", None) if self.use_numeric else None)

        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=x["attention_mask"],
            output_hidden_states=False,
            return_dict=True)
        
        seq = out.last_hidden_state
        vec = self._pool(seq, x["attention_mask"])
        return {"seq": seq, "vec": vec}

    def _flatten_history(self, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, K, L = h["input_ids"].shape
        flat = {}
        for k, v in h.items():
            if v is None:
                continue
            if v.dim() == 3:
                flat[k] = v.reshape(B * K, L)
        return flat

    def _unflatten_history(
        self,
        seq_flat: torch.Tensor,
        vec_flat: torch.Tensor, 
        B: int,
        K: int,
    ) -> Dict[str, torch.Tensor]:
        
        L = seq_flat.shape[1]
        H = seq_flat.shape[2]
        seq = seq_flat.reshape(B, K, L, H)
        vec = vec_flat.reshape(B, K, H)
        return {"seq": seq, "vec": vec}



    def forward(
        self,
        batch: Dict[str, Any],
        query_key: str = "query",
        history_key: str = "history", 
    ) -> Dict[str, torch.Tensor]:
        q = batch[query_key]
        h = batch[history_key]

        q_out = self._encode(q)

        B, K, L = h["input_ids"].shape
        h_flat = self._flatten_history(h)
        h_out_flat = self._encode(h_flat)
        h_out = self._unflatten_history(h_out_flat["seq"], h_out_flat["vec"], B=B, K=K)

        out = {"query_vec": q_out["vec"], "hist_vec": h_out["vec"]}

        if self.return_token_level:
            out["query_seq"]  = q_out["seq"]
            out["hist_seq"]   = h_out["seq"]
            out["query_mask"] = q["attention_mask"]
            out["hist_mask"]  = h["attention_mask"]

        return out
    
    def get_pretrained_weights(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

        mt = getattr(self.backbone.config, "model_type", "").lower()
        prefix_map = {
            "bert":       "backbone.bert.",
            "roberta":    "backbone.roberta.",
            "longformer": "backbone.longformer.",
            "modernbert": "backbone.model.",
            "roformer":   "backbone.roformer.",
            "big_bird":   "backbone.bert.",
            "mamba":      "backbone.backbone.",
            "mamba2":     "backbone.backbone.",
        }
        backbone_prefix = prefix_map.get(mt, None)

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val.", "backbone.lm_head."]

        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                new_k = k
            elif backbone_prefix and k.startswith(backbone_prefix):
                new_k = "backbone." + k[len(backbone_prefix):]
            else:
                new_k = k

            remapped[new_k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print("weights loaded successfully!")
        print("missing keys:", missing)
        print("+" * 50)
        print("unexpected keys:", unexpected)





import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# KL version
class PrototypeRetrievalModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_prototypes: int = 256,
        query_temperature: float = 0.1,
        history_temperature: float = 0.025,
        normalize_prototypes: bool = True,
        softmax_threshold: float = 0.8,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.query_temperature = float(query_temperature)
        self.history_temperature = float(history_temperature)
        self.normalize_prototypes = normalize_prototypes
    
        self.softmax_threshold = float(softmax_threshold)
        self.softmax_temperature = float(softmax_temperature)

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)


    def _proto_probs(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        P = self.prototypes
        x = F.normalize(x, p=2, dim=-1)
        P = F.normalize(P, p=2, dim=-1)
        logits = x @ P.t()
        return F.softmax(logits / temperature, dim=-1)

    def _compute_neg_ce(self, q_probs: torch.Tensor, h_probs: torch.Tensor) -> torch.Tensor:
        return (q_probs.unsqueeze(1) * h_probs.clamp_min(1e-8).log()).sum(dim=-1)



    def _weights_and_mask_softmax(
        self,
        scores: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if valid_mask is not None:
            valid_bool = valid_mask.bool()
            scores = scores.masked_fill(~valid_bool, -1e9)
        else:
            valid_bool = torch.ones_like(scores, dtype=torch.bool)

        weights = F.softmax(scores / self.softmax_temperature, dim=-1)

        hard = (weights >= self.softmax_threshold).float()
        keep = hard + weights - weights.detach()

        keep = keep * valid_bool.float()

        return {
            "attn_weights": weights,                  
            "attn_mask": keep,    
        }

    def _entropy(self, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def forward(
        self,
        query_vec: torch.Tensor,
        hist_vec: torch.Tensor,
        hist_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, K, _ = hist_vec.shape

        if hist_valid_mask is not None:
            hist_valid_mask = hist_valid_mask.to(device=hist_vec.device, dtype=torch.long)
            valid_f = hist_valid_mask.to(dtype=hist_vec.dtype)
            n_valid = valid_f.sum().clamp_min(1.0)
        else:
            valid_f = None
            n_valid = None

        q_probs = self._proto_probs(query_vec,temperature= self.query_temperature )   # [B, P]
        h_probs = self._proto_probs(hist_vec, temperature= self.history_temperature)    # [B, K, P]

        q_ent_all = self._entropy(q_probs)
        h_ent_all = self._entropy(h_probs)

        q_max = q_probs.max(dim=-1).values.mean()
        h_max_all = h_probs.max(dim=-1).values

        if valid_f is None:
            q_ent = q_ent_all.mean()
            h_ent = h_ent_all.mean()
            h_max = h_max_all.mean()
        else:
            q_ent = q_ent_all.mean()
            h_ent = (h_ent_all * valid_f).sum() / n_valid
            h_max = (h_max_all * valid_f).sum() / n_valid

        mean_q_probs = q_probs.mean(dim=0)
        if valid_f is None:
            mean_h_probs = h_probs.mean(dim=(0, 1))
        else:
            valid = valid_f.unsqueeze(-1)
            mean_h_probs = (h_probs * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)

        mean_q_ent = self._entropy(mean_q_probs)
        mean_h_ent = self._entropy(mean_h_probs)

        neg_ce = self._compute_neg_ce(q_probs, h_probs)   # [B, K]

        wm = self._weights_and_mask_softmax(neg_ce, valid_mask=hist_valid_mask)

        out = {
            "neg_ce": neg_ce,
            "attn_mask": wm["attn_mask"],
            "attn_weights": wm["attn_weights"],
            "query_probs": q_probs,
            "hist_probs": h_probs,
            "diag_q_ent": q_ent,
            "diag_q_max": q_max,
            "diag_h_ent": h_ent,
            "diag_h_max": h_max,
            "diag_mean_q_ent": mean_q_ent,
            "diag_mean_h_ent": mean_h_ent,
        }
        return out




class FusionModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_weights_as_gating: bool = False,
        output_mode: str = "query",  
        return_seq: bool = False,
    ):
        super().__init__()
        self.use_weights_as_gating = use_weights_as_gating
        self.output_mode = output_mode
        self.return_seq = return_seq

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_mult * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True, 
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,)

        self.out_norm = nn.LayerNorm(hidden_size)
        # self.hist_score_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query_vec: torch.Tensor,
        hist_vec: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        hist_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B, K, H = hist_vec.shape
        assert query_vec.shape == (B, H)

        if self.use_weights_as_gating and attn_weights is not None:
            hist_vec = hist_vec * attn_weights.unsqueeze(-1).to(hist_vec.dtype)
            # hist_vec = self.hist_score_proj(hist_vec)

        # if self.use_weights_as_gating and attn_weights is not None:
        #     alpha = attn_weights.unsqueeze(-1).to(hist_vec.dtype)
        #     hist_vec = hist_vec * (1.0 + alpha)


        # if self.use_weights_as_gating and attn_weights is not None:
        #     mask = hist_valid_mask.float()                         # [B, K]
        #     num_valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]

        #     scaled_weights = attn_weights * num_valid              # rescale
        #     scaled_weights = scaled_weights * mask                 # zero out padding

        #     hist_vec = hist_vec * scaled_weights.unsqueeze(-1).to(hist_vec.dtype)

        elif attn_mask is not None:
            hist_vec = hist_vec * attn_mask.unsqueeze(-1).to(hist_vec.dtype)

        x = torch.cat([query_vec.unsqueeze(1), hist_vec], dim=1)  # [B, 1+K, H]

        if hist_valid_mask is None:
            src_key_padding_mask = None
            keep = torch.ones(B, 1 + K, device=x.device, dtype=x.dtype)
        else:
            query_valid = torch.ones(B, 1, device=x.device, dtype=torch.bool)
            keep_bool = torch.cat([query_valid, hist_valid_mask.bool()], dim=1)   # [B, 1+K]
            src_key_padding_mask = ~keep_bool
            keep = keep_bool.to(x.dtype)

        x_fused = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x_fused = self.out_norm(x_fused)

        if self.output_mode == "query":
            fused_vec = x_fused[:, 0, :]
        elif self.output_mode == "mean":
            w = keep.unsqueeze(-1)
            denom = w.sum(dim=1).clamp_min(1.0)
            fused_vec = (x_fused * w).sum(dim=1) / denom

        out = {"fused_vec": fused_vec}
        if self.return_seq:
            out["fused_seq"] = x_fused
            out["fused_keep_mask"] = keep
        return out



    
class EHRRAPEvalModel(lt.LightningModule):
    def __init__(
        self,
        config,
        backbone,
        ckpt_path: Optional[str] = None,
        lr: float = 2e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
        pooling: str = "cls",
        use_numeric: bool = False,
        use_time: bool = False,
        # --- prototype module ---
        num_prototypes: int = 256,
        query_temperature: float = 0.1,
        history_temperature: float = 0.025,
        normalize_prototypes: bool = True,
        softmax_threshold: float = 0.8,
        softmax_temperature: float = 1.0,
        sample_ent_lambda: float = 0.001,
        usage_ent_lambda: float = 0.001,
        use_prototypes: bool = False,
        # --- fusion module ---
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        fusion_ff_mult: int = 4,
        fusion_output_mode: str = "query",
        use_weights_as_gating: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.config = config
        self.sample_ent_lambda = float(sample_ent_lambda)
        self.usage_ent_lambda = float(usage_ent_lambda)
        self.use_prototypes = use_prototypes
        self.encoders = EHRRAPEncoders(
            config=config,
            backbone=backbone,
            dropout=dropout,
            pooling=pooling,
            use_time=use_time,
            use_numeric=use_numeric,
            ckpt_path=ckpt_path,
            return_token_level=False,
        )

        self.prototypes = PrototypeRetrievalModule(
            hidden_size=config.hidden_size,
            num_prototypes=num_prototypes,
            query_temperature=query_temperature,
            history_temperature=history_temperature,
            normalize_prototypes=normalize_prototypes,
            softmax_threshold=softmax_threshold,
            softmax_temperature=softmax_temperature,
        )

        self.fusion = FusionModule(
            hidden_size=config.hidden_size,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            ff_mult=fusion_ff_mult,
            dropout=dropout,
            use_weights_as_gating=use_weights_as_gating,
            output_mode=fusion_output_mode,
            return_seq=False,
        )

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

        self._train_preds, self._train_labels = [], []
        self._val_preds, self._val_labels = [], []
        self._test_preds, self._test_labels = [], []
        
#         if freeze:
#             for p in self.encoders.parameters():
#                 p.requires_grad = False
#             for p in self.prototypes.parameters():
#                 p.requires_grad = True
#             for p in self.fusion.parameters():
#                 p.requires_grad = True
#             for p in self.classifier.parameters():
#                 p.requires_grad = True

# with ordering
    # def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    #     enc_out = self.encoders(batch, query_key="query", history_key="history")

    #     proto_out = None
    #     attn_mask = None
    #     attn_weights = None
    #     hist_valid_mask = batch["history_valid_mask"]
    #     hist_vec = enc_out["hist_vec"]

    #     if self.use_prototypes:
    #         proto_out = self.prototypes(
    #             query_vec=enc_out["query_vec"],
    #             hist_vec=hist_vec,
    #             hist_valid_mask=hist_valid_mask,
    #         )

    #         attn_mask = proto_out["attn_mask"]
    #         attn_weights = proto_out["attn_weights"]

    #         # sort by prototype score, highest first
    #         sort_scores = proto_out["attn_weights"]  # or proto_out["neg_ce"]
    #         sort_idx = torch.argsort(sort_scores, dim=1, descending=True)

    #         hist_vec = torch.gather(
    #             hist_vec,
    #             dim=1,
    #             index=sort_idx.unsqueeze(-1).expand(-1, -1, hist_vec.size(-1)),
    #         )

    #         hist_valid_mask = torch.gather(hist_valid_mask, dim=1, index=sort_idx)

    #         if attn_mask is not None:
    #             attn_mask = torch.gather(attn_mask, dim=1, index=sort_idx)

    #         if attn_weights is not None:
    #             attn_weights = torch.gather(attn_weights, dim=1, index=sort_idx)

    #     fuse_out = self.fusion(
    #         query_vec=enc_out["query_vec"],
    #         hist_vec=hist_vec,
    #         attn_mask=attn_mask,
    #         attn_weights=attn_weights,
    #         hist_valid_mask=hist_valid_mask,
    #     )

    #     fused_vec = fuse_out["fused_vec"]
    #     logits = self.classifier(fused_vec).squeeze(-1)

    #     out = {
    #         "logits": logits,
    #         "fused_vec": fused_vec,
    #     }
    #     if proto_out is not None:
    #         out["proto"] = proto_out

    #     return out

# without ordering
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        enc_out = self.encoders(batch, query_key="query", history_key="history")

        proto_out = None
        attn_mask = None
        attn_weights = None
        hist_valid_mask = batch["history_valid_mask"]

        if self.use_prototypes:
            proto_out = self.prototypes(
                query_vec=enc_out["query_vec"],
                hist_vec=enc_out["hist_vec"],
                hist_valid_mask=hist_valid_mask,
            )
            attn_mask = proto_out["attn_mask"]          # soft/ST mask
            attn_weights = proto_out["attn_weights"]    # soft weights

        fuse_out = self.fusion(
            query_vec=enc_out["query_vec"],
            hist_vec=enc_out["hist_vec"],
            attn_mask=attn_mask,
            attn_weights=attn_weights,
            hist_valid_mask=hist_valid_mask,            # real padding mask only
        )

        fused_vec = fuse_out["fused_vec"]
        logits = self.classifier(fused_vec).squeeze(-1)

        out = {
            "logits": logits,
            "fused_vec": fused_vec,
        }
        if proto_out is not None:
            out["proto"] = proto_out

        return out

    def shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        out = self.forward(batch)

        proto = out.get("proto", None)

        # --- log proto diagnostics if present ---
        if proto is not None and "diag_q_ent" in proto:
            self.log(f"{stage}_sample_q_ent", proto["diag_q_ent"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_q_max", proto["diag_q_max"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_sample_h_ent", proto["diag_h_ent"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_h_max", proto["diag_h_max"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)

        if proto is not None and "diag_mean_q_ent" in proto:
            self.log(f"{stage}_batch_q_ent", proto["diag_mean_q_ent"], prog_bar=False, on_step=True if stage == "train" else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_batch_h_ent", proto["diag_mean_h_ent"], prog_bar=False, on_step=True if stage == "train" else False, on_epoch=True, logger=True, sync_dist=True)

        logits = out["logits"]
        y = batch["label"].float().view(-1)

        task_loss = self.criterion(logits, y)


        sample_ent_reg = task_loss.new_zeros(())
        usage_ent_reg = task_loss.new_zeros(())

        if proto is not None and "diag_q_ent" in proto and self.sample_ent_lambda > 0.0:
            sample_ent_reg = proto["diag_q_ent"] + proto["diag_h_ent"]
            # sample_ent_reg = proto["diag_h_ent"]
        if proto is not None and "diag_mean_q_ent" in proto and self.usage_ent_lambda > 0.0:
            usage_ent_reg = -(proto["diag_mean_q_ent"] + proto["diag_mean_h_ent"])
            # usage_ent_reg = -(proto["diag_mean_q_ent"])

            


        # total_loss = task_loss + self.ent_lambda * ent_reg
        total_loss = (
            task_loss
            + self.sample_ent_lambda * sample_ent_reg
            + self.usage_ent_lambda * usage_ent_reg
        )

        # --- logging losses ---
        self.log(f"{stage}_loss", task_loss, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{stage}_total_loss", total_loss, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)


        pos_score = torch.sigmoid(logits)

        if stage == "train":
            self._train_labels.append(y.detach())
            self._train_preds.append(pos_score.detach())
        elif stage == "val":
            self._val_labels.append(y.detach())
            self._val_preds.append(pos_score.detach())
        elif stage == "test":
            self._test_labels.append(y.detach())
            self._test_preds.append(pos_score.detach())

        if proto is not None and "attn_weights" in proto and proto["attn_weights"] is not None and not self.fusion.use_weights_as_gating:
            with torch.no_grad():
                weights = proto["attn_weights"].float()
                mask = batch["history_valid_mask"].float()

                hard_keep = (weights >= self.prototypes.softmax_threshold).float()
                keep_rate = (hard_keep * mask).sum() / mask.sum().clamp_min(1.0)

            self.log(f"{stage}_keep_rate",keep_rate,prog_bar=False, on_step=True, on_epoch=True,logger=True,sync_dist=True)

        if proto is not None and "attn_weights" in proto and proto["attn_weights"] is not None and self.fusion.use_weights_as_gating:
            with torch.no_grad():
                weights = proto["attn_weights"].float()
                mask = batch["history_valid_mask"].float()

                weights = weights * mask
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                weights_norm = weights / denom

                weight_entropy = -(weights_norm * (weights_norm.clamp_min(1e-8).log())).sum(dim=1)

                valid_counts = mask.sum(dim=1).clamp_min(1.0)
                max_entropy = valid_counts.log().clamp_min(1e-8)
                weight_entropy_norm = (weight_entropy / max_entropy).mean()

            self.log(f"{stage}_weight_entropy",weight_entropy.mean(), prog_bar=False, on_step=True, on_epoch=True,logger=True,sync_dist=True)
            self.log(f"{stage}_weight_entropy_norm", weight_entropy_norm, prog_bar=False, on_step=True, on_epoch=True, logger=True,sync_dist=True)


        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="test")


    # def on_train_start(self) -> None:
    #     if not self.use_prototypes:
    #         return
    #     self._proto_init = self.prototypes.prototypes.detach().clone()

    # def on_after_backward(self) -> None:
    #     if not self.use_prototypes or not hasattr(self, "_proto_init"):
    #         return

    #     current = self.prototypes.prototypes.detach()
    #     delta_norm = (current - self._proto_init).norm()
    #     current_norm = current.norm().clamp_min(1e-12)
    #     rel_delta = delta_norm / current_norm

        
    #     proto_param = self.prototypes.prototypes

    #     if proto_param.grad is None:
    #         grad_norm = 0.0
    #     else:
    #         grad_norm = proto_param.grad.norm()
            
    #     self.log(
    #         "train_proto_delta_norm",
    #         delta_norm,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #     )
    #     self.log(
    #         "train_proto_rel_delta",
    #         rel_delta,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #     )    

    #     self.log(
    #         "train_proto_grad_norm",
    #         grad_norm,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True)


    def on_train_epoch_end(self) -> None:
        if not self._train_preds:
            return
        y = torch.cat(self._train_labels)
        p = torch.cat(self._train_preds)
        self.log("train_auroc", self.train_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("train_auprc", self.train_auprc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self._train_labels.clear()
        self._train_preds.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        y = torch.cat(self._val_labels)
        p = torch.cat(self._val_preds)
        self.log("val_auroc", self.val_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("val_auprc", self.val_auprc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self._val_labels.clear()
        self._val_preds.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        y = torch.cat(self._test_labels)
        p = torch.cat(self._test_preds)
        self.log("test_auroc", self.test_auroc(p, y.long()), on_epoch=True, logger=True)
        self.log("test_auprc", self.test_auprc(p, y.long()), on_epoch=True, logger=True)

        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=y,
            y_score=p,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3,
            )

        self._test_labels.clear()
        self._test_preds.clear()

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoders.parameters()},
            {"params": self.fusion.parameters()},
            {"params": self.classifier.parameters()},
        ]

        if self.use_prototypes:
            param_groups.insert(1, {"params": self.prototypes.parameters()})

        optimizer = torch.optim.SGD(
            param_groups,
            lr=self.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=self.wd,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.max_epochs,
            eta_min=0.0,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

lt.seed_everything(24, workers=True)
x = 256
y = 32
chunking_startegy = 'overlap'
window ='w24'
top_k = 24
batch_size = 20
window_hours = 24.0
train_dataset = RetrievalDataset(data_idx_path='./downstream_idx.parquet',
                                 dataset_path= './data/meds_normalized_arrow/',
                                 vectordb_path=f'/faiss/1024/{chunking_startegy}/{x}/{window}/',
                                 tokenizer_path='./vocab.json',
                                 limits_dict=limits,
                                 chunking_strategy = chunking_startegy,
                                 task= 'y_icu_readmit_30',
                                 query_window= 'within_stay_query',
                                 history_window= 'within_stay_hist_full',
                                 top_k = top_k,
                                 seq_length_q = 1024,
                                 overlap_q= 0,
                                 seq_length_h= x,
                                 overlap_h= y,
                                 use_time= True,
                                 use_numeric= True,
                                 add_cls=True,
                                 window_hours= window_hours,
                                 split= 'train')

val_dataset =RetrievalDataset(data_idx_path='./downstream_idx.parquet',
                                 dataset_path= './data/meds_normalized_arrow/',
                                 vectordb_path=f'/faiss/1024/{chunking_startegy}/{x}/{window}/',
                                 tokenizer_path='./vocab.json',
                                 limits_dict=limits,
                                 chunking_strategy = chunking_startegy,
                                 task= 'y_icu_readmit_30',
                                 query_window= 'within_stay_query',
                                 history_window= 'within_stay_hist_full',
                                 top_k = top_k,
                                 seq_length_q = 1024,
                                 overlap_q= 0,
                                 seq_length_h= x,
                                 overlap_h= y,
                                 use_time= True,
                                 use_numeric= True,
                                 add_cls=True,
                                 window_hours= window_hours,
                                 split= 'val')

test_dataset = RetrievalDataset(data_idx_path='./downstream_idx.parquet',
                                 dataset_path= './data/meds_normalized_arrow/',
                                 vectordb_path=f'/faiss/1024/{chunking_startegy}/{x}/{window}/',
                                 tokenizer_path='./vocab.json',
                                 limits_dict=limits,
                                 chunking_strategy = chunking_startegy,
                                 task= 'y_icu_readmit_30',
                                 query_window= 'within_stay_query',
                                 history_window= 'within_stay_hist_full',
                                 top_k = top_k,
                                 seq_length_q = 1024,
                                 overlap_q= 0,
                                 seq_length_h= x,
                                 overlap_h= y,
                                 use_time= True,
                                 use_numeric= True,
                                 add_cls=True,
                                 window_hours= window_hours,
                                 split= 'test')


chunk_collator = EvalCollator(tokenizer=train_dataset.query_gen.tokenizer,
                              use_mask_augmentation=True,
                              augment_prob=0.25,
                              mask_prob=0.125,
                             )

retrieval_collator = RetrievalCollator(chunk_collator, top_k)


from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=retrieval_collator,
                              num_workers=4,
                              prefetch_factor=2,
                              persistent_workers=True,
                              pin_memory=True,
                              pin_memory_device='cuda',
                              drop_last=True,
                            ) 


val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=retrieval_collator,
                            num_workers=4,
                            prefetch_factor=2,
                            persistent_workers=True,
                            pin_memory=True,
                            pin_memory_device='cuda',
                            drop_last=True,
                            )

test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=retrieval_collator,
                            num_workers=4,
                            prefetch_factor=2,
                            persistent_workers=True,
                            pin_memory=True,
                            pin_memory_device='cuda',
                            drop_last=True,
                            )





ConfigClass, ModelClass = get_config_and_model_cls(model_type='roformer', mode='eval', variant=None)

cfg = ConfigClass(vocab_size=train_dataset.query_gen.tokenizer.vocab_size,
                  cls_token_id=train_dataset.query_gen.tokenizer.cls_id,
                  pad_token_id=train_dataset.query_gen.tokenizer.pad_id,
                  type_vocab_size=28,
                  visit_vocab_size=102,
                  stage_vocab_size=5,
                  refernece_compile=False)

cfg = fix_roberta_longformer_max_pos(cfg)




model = EHRRAPEvalModel(config=cfg,
                        backbone=ModelClass,
                        ckpt_path='./models/mlm/wandb/run-20251128_073215-roformer_13218339_1024_128_15_maskprob_12_5overlap/files/ckpt/epoch=65-step=665082.ckpt',
                        lr=0.00022646864633184765,
                        wd=0.0014790962846295604,
                        max_epochs=100,
                        dropout=0.1,
                        freeze=False,
                        pooling='mean',
                        use_numeric=True,
                        use_time=True,

                        # prototype module
                        num_prototypes=512,
                        query_temperature=0.02,
                        history_temperature=0.08,
                        normalize_prototypes=True,

                        softmax_threshold=0.04,             
                        softmax_temperature=0.25,             
                        sample_ent_lambda=0.000,
                        usage_ent_lambda=0.006, #0.006
                        use_prototypes=True,

                        # fusion
                        fusion_layers=2,
                        fusion_heads=4,
                        fusion_ff_mult=4,
                        fusion_output_mode='mean',
                        use_weights_as_gating=True)


wandb_logger = WandbLogger(project='MedEHR_Eval',
                           save_dir='./lightning_logs',)


torch.set_float32_matmul_precision('high')
trainer = lt.Trainer(accelerator='auto', 
                     devices='auto',
                     strategy='ddp_find_unused_parameters_true',
                     logger=wandb_logger, 
                     log_every_n_steps=1,
                     num_sanity_val_steps=0,
                     max_epochs=20,
                     precision='bf16-mixed',
#                     callbacks=[early_stop,lr_monitor,checkpoint_callback],
                    )


trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model=model, dataloaders=test_dataloader,ckpt_path='last', verbose=True)

















# Median version
# class PrototypeRetrievalModule(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int = 768,
#         num_prototypes: int = 256,
#         temperature: float = 0.1,
#         align_mode: str = "soft",
#         sim_mode: str = "cosine",
#         combine_mode: str = "add",
#         lambda_sim: float = 0.5,
#         normalize_prototypes: bool = True,
#         detach_hard_alignment: bool = True,
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_prototypes = num_prototypes
#         self.temperature = float(temperature)
#         self.align_mode = align_mode
#         self.sim_mode = sim_mode
#         self.combine_mode = combine_mode
#         self.lambda_sim = float(lambda_sim)

#         self.normalize_prototypes = normalize_prototypes
#         self.detach_hard_alignment = detach_hard_alignment

#         self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
#         nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

#     def _proto_probs(self, x: torch.Tensor) -> torch.Tensor:
#         P = self.prototypes
#         if self.normalize_prototypes:
#             P = F.normalize(P, p=2, dim=-1)

#         if self.sim_mode == "cosine":
#             x = F.normalize(x, p=2, dim=-1)
#             P = F.normalize(P, p=2, dim=-1)

#         logits = x @ P.t()
#         return F.softmax(logits / self.temperature, dim=-1)

#     def _alignment_scores(self, q_probs: torch.Tensor, h_probs: torch.Tensor) -> torch.Tensor:
#         # q_probs: [B, P], h_probs: [B, K, P] -> [B, K]
#         if self.align_mode == "soft":
#             return (q_probs.unsqueeze(1) * h_probs).sum(dim=-1)

#         elif self.align_mode == "kl":
#             # equivalent to negative cross-entropy up to a query-only constant
#             # higher is better (less negative = better match)
#             return (q_probs.unsqueeze(1) * h_probs.log()).sum(dim=-1)

#         elif self.align_mode == "hard":
#             q_id = q_probs.argmax(dim=-1)         # [B]
#             h_id = h_probs.argmax(dim=-1)         # [B, K]
#             align = (h_id == q_id.unsqueeze(1)).float()
#             return align.detach() if self.detach_hard_alignment else align

#         raise ValueError(f"Unknown align_mode={self.align_mode}")

#     def _similarity_scores(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
#         if self.sim_mode == "dot":
#             return (h * q.unsqueeze(1)).sum(dim=-1)

#         elif self.sim_mode == "cosine":
#             qn = F.normalize(q, p=2, dim=-1)
#             hn = F.normalize(h, p=2, dim=-1)
#             return (hn * qn.unsqueeze(1)).sum(dim=-1)

#         raise ValueError(f"Unknown sim_mode={self.sim_mode}")

#     def _combine_scores(self, sim: torch.Tensor, align: torch.Tensor) -> torch.Tensor:
#         if self.combine_mode == "mul":
#             return sim * align

#         elif self.combine_mode == "add":
#             lam = self.lambda_sim
#             return lam * sim + (1.0 - lam) * align

#         raise ValueError(f"Unknown combine_mode={self.combine_mode}")
    
#     def _robust_z(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, eps: float = 1e-3,) -> torch.Tensor:

#             B, K = x.shape
#             z = torch.zeros_like(x)

#             if valid_mask is None:
#                 med = x.median(dim=1, keepdim=True).values
#                 mad = (x - med).abs().median(dim=1, keepdim=True).values.clamp_min(eps)
#                 return (x - med) / mad

#             valid_mask = valid_mask.bool()

#             for b in range(B):
#                 vb = valid_mask[b]
#                 if vb.sum() == 0:
#                     continue

#                 xb = x[b, vb]  # valid scores only
#                 med = xb.median()
#                 mad = (xb - med).abs().median().clamp_min(eps)

#                 z[b, vb] = (xb - med) / mad

#             return z

#     def _weights_and_mask(
#         self,
#         scores: torch.Tensor,
#         valid_mask: Optional[torch.Tensor] = None,  # [B, K] (1=real,0=pad)
#     ) -> Dict[str, torch.Tensor]:
#         B, K = scores.shape

#         if valid_mask is not None:
#             valid_mask = valid_mask.to(device=scores.device, dtype=torch.long)
#             valid_bool = valid_mask.bool()
#         else:
#             valid_bool = torch.ones_like(scores, dtype=torch.bool)

#         scores_z = self._robust_z(scores, valid_mask=valid_mask)   # [B, K]

#         keep = (scores_z > 0) & valid_bool

#         # always keep best valid chunk
#         scores_for_argmax = scores_z.masked_fill(~valid_bool, float("-inf"))
#         best_idx = scores_for_argmax.argmax(dim=1)  # [B]

#         has_valid = valid_bool.any(dim=1)
#         if has_valid.any():
#             rows = torch.arange(B, device=scores.device)[has_valid]
#             keep[rows, best_idx[has_valid]] = True

#         attn_mask = keep.long()

#         return {
#             "attn_weights": None,     # no softmax weighting in this version
#             "attn_mask": attn_mask,
#             "selection_scores": scores_z,
#         }

#     def _entropy(self, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#         p = p.clamp_min(eps)
#         return -(p * p.log()).sum(dim=-1)

#     def forward(
#         self,
#         query_vec: torch.Tensor,                 # [B, H]
#         hist_vec: torch.Tensor,                  # [B, K, H]
#         hist_valid_mask: Optional[torch.Tensor] = None,  # [B, K] (1 real, 0 pad)
#     ) -> Dict[str, torch.Tensor]:

#         B, K, _ = hist_vec.shape

#         if hist_valid_mask is not None:
#             hist_valid_mask = hist_valid_mask.to(device=hist_vec.device, dtype=torch.long)
#             assert hist_valid_mask.shape == (B, K), f"hist_valid_mask must be [B,K], got {hist_valid_mask.shape}"
#             valid_f = hist_valid_mask.to(dtype=hist_vec.dtype)
#             n_valid = valid_f.sum().clamp_min(1.0)
#         else:
#             valid_f = None
#             n_valid = None

#         # ---- prototype assignments ----
#         q_probs = self._proto_probs(query_vec)     # [B, P]
#         h_probs = self._proto_probs(hist_vec)      # [B, K, P]


#         # # ---- diagnostics ----
#         # q_ent = self._entropy(q_probs).mean()
#         # q_max = q_probs.max(dim=-1).values.mean()

#         # h_ent_all = self._entropy(h_probs)         # [B, K]
#         # h_max_all = h_probs.max(dim=-1).values     # [B, K]

#         # if valid_f is None:
#         #     h_ent = h_ent_all.mean()
#         #     h_max = h_max_all.mean()
#         # else:
#         #     h_ent = (h_ent_all * valid_f).sum() / n_valid
#         #     h_max = (h_max_all * valid_f).sum() / n_valid

#         # ----- with batch entropy -----
#         q_ent_all = self._entropy(q_probs)               # [B]
#         h_ent_all = self._entropy(h_probs)               # [B, K]

#         q_max = q_probs.max(dim=-1).values.mean()
#         h_max_all = h_probs.max(dim=-1).values

#         if valid_f is None:
#             q_ent = q_ent_all.mean()
#             h_ent = h_ent_all.mean()
#             h_max = h_max_all.mean()
#         else:
#             h_ent = (h_ent_all * valid_f).sum() / n_valid
#             h_max = (h_max_all * valid_f).sum() / n_valid
#             q_ent = q_ent_all.mean()

#         # ---- batch-mean entropy (diversity / usage) ----
#         mean_q_probs = q_probs.mean(dim=0)   # [P]

#         if valid_f is None:
#             mean_h_probs = h_probs.mean(dim=(0, 1))   # [P]
#         else:
#             valid = valid_f.unsqueeze(-1)             # [B, K, 1]
#             mean_h_probs = (h_probs * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)

#         mean_q_ent = self._entropy(mean_q_probs)      # scalar
#         mean_h_ent = self._entropy(mean_h_probs)      # scalar

#         # ---- scores ----
#         align = self._alignment_scores(q_probs, h_probs)      # [B, K]
#         sim   = self._similarity_scores(query_vec, hist_vec)  # [B, K]

#         if valid_f is not None:
#             align = align * valid_f
#             sim   = sim * valid_f

#         final = self._combine_scores(sim, align)              # [B, K]

#         # do NOT set pads to -1e9 here; robust z must be computed over valid positions only
#         wm = self._weights_and_mask(final, valid_mask=hist_valid_mask)
#         attn_mask = wm["attn_mask"]

#         out = {
#             "attn_mask": attn_mask,
#             "final_scores": final,
#             "selection_scores": wm["selection_scores"],  # z-scored final scores
#             "sim": sim,
#             "align": align,
#             "query_probs": q_probs,
#             "hist_probs": h_probs,
#             "diag_q_ent": q_ent,
#             "diag_q_max": q_max,
#             "diag_h_ent": h_ent,
#             "diag_h_max": h_max,
#             # new diagnostics
#             "diag_mean_q_ent": mean_q_ent,    # entropy of batch-mean query usage
#             "diag_mean_h_ent": mean_h_ent,    # entropy of batch-mean history usage
#         }

#         return out













# class PrototypeRetrievalModule(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int = 768,
#         num_prototypes: int = 256,
#         temperature: float = 0.1,
#         align_mode: str = "soft",
#         sim_mode: str = "cosine",       
#         combine_mode: str = "mul",
#         lambda_sim: float = 0.5, 
#         normalize_prototypes: bool = True,
#         detach_hard_alignment: bool = True, 
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_prototypes = num_prototypes
#         self.temperature = float(temperature)
#         self.align_mode = align_mode
#         self.sim_mode = sim_mode
#         self.combine_mode = combine_mode
#         self.lambda_sim = float(lambda_sim)

#         self.normalize_prototypes = normalize_prototypes
#         self.detach_hard_alignment = detach_hard_alignment

#         self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
#         nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

#     def _proto_probs(self, x: torch.Tensor) -> torch.Tensor:
#         P = self.prototypes

#         if self.sim_mode == "cosine":
#             x = F.normalize(x, p=2, dim=-1)
#             P = F.normalize(P, p=2, dim=-1)

#         elif self.normalize_prototypes:
#             P = F.normalize(P, p=2, dim=-1)

#         logits = x @ P.t()
#         return F.softmax(logits / self.temperature, dim=-1)

#     def _alignment_scores(self, q_probs: torch.Tensor, h_probs: torch.Tensor) -> torch.Tensor:
        
#         if self.align_mode == "soft":
#             return (q_probs.unsqueeze(1) * h_probs).sum(dim=-1)

#         elif self.align_mode == "hard":
#             q_id = q_probs.argmax(dim=-1)           
#             h_id = h_probs.argmax(dim=-1)          
#             align = (h_id == q_id.unsqueeze(1)).float()
#             return align.detach() if self.detach_hard_alignment else align

#     def _similarity_scores(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        
#         if self.sim_mode == "dot":
#             return (h * q.unsqueeze(1)).sum(dim=-1)

#         elif self.sim_mode == "cosine":
#             qn = F.normalize(q, p=2, dim=-1)
#             hn = F.normalize(h, p=2, dim=-1)
#             return (hn * qn.unsqueeze(1)).sum(dim=-1)

#     def _combine_scores(self, sim: torch.Tensor, align: torch.Tensor) -> torch.Tensor:
#         if self.combine_mode == "mul":
#             return sim * align

#         elif self.combine_mode == "add":
#             lam = self.lambda_sim
#             return lam * sim + (1.0 - lam) * align

#     def _weights_and_mask(self, scores: torch.Tensor) -> Dict[str, torch.Tensor]:

#         B, K = scores.shape

#         scores_z = self._robust_z(scores)                                               


#         keep = (scores_z > 0.0)

#         best_idx = scores_z.argmax(dim=1)                                            
#         keep[torch.arange(B, device=scores.device), best_idx] = True

#         attn_mask = keep.long()

#         return {"attn_weights": None, "attn_mask": attn_mask}

#     def _robust_z(self, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
#         med = x.median(dim=1, keepdim=True).values
#         mad = (x - med).abs().median(dim=1, keepdim=True).values
#         mad = mad.clamp_min(eps)
#         return (x - med) / mad
    
#     def _entropy(self, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#         # p: [..., P]
#         p = p.clamp_min(eps)
#         return -(p * p.log()).sum(dim=-1)


#     def forward(
#         self,
#         query_vec: torch.Tensor,
#         hist_vec: torch.Tensor,
#         return_debug: Optional[bool] = None,
#     ) -> Dict[str, torch.Tensor]:

#         q_probs = self._proto_probs(query_vec)
#         h_probs = self._proto_probs(hist_vec)

#         # ---- diagnostics ----
#         q_ent = self._entropy(q_probs).mean()           # scalar
#         q_max = q_probs.max(dim=-1).values.mean()       # scalar

#         h_ent = self._entropy(h_probs).mean()           # scalar (mean over B,K)
#         h_max = h_probs.max(dim=-1).values.mean()       # scalar
#         # ---- diagnostics ----


#         align = self._alignment_scores(q_probs, h_probs)
#         # align_z = self._robust_z(align)

#         sim = self._similarity_scores(query_vec, hist_vec)
#         # sim_z = self._robust_z(sim)

#         final = self._combine_scores(sim, align)

#         wm = self._weights_and_mask(final)
# #         attn_weights = wm["attn_weights"]
#         attn_mask = wm["attn_mask"] 

#         out = {                      
# #             "attn_weights": attn_weights,               
#             "attn_mask": attn_mask,                     
#             "final_scores": final,
#             'sim':sim,
#             "align":align,
#             "query_probs": q_probs,          
#             "hist_probs": h_probs,
#             "diag_q_ent": q_ent,
#             "diag_q_max": q_max,
#             "diag_h_ent": h_ent,
#             "diag_h_max": h_max,
#         }
#         return out


# class EHRRAPEvalModel(lt.LightningModule):
#     def __init__(
#         self,
#         config,
#         backbone,                      
#         ckpt_path: Optional[str] = None,
#         lr: float = 2e-5,
#         wd: float = 0.001,
#         max_epochs: int = 100,
#         dropout: float = 0.1,
#         freeze: bool = False,
#         pooling: str = "cls",
#         use_numeric: bool = False,
#         use_time: bool = False,
#         optimizer: str = "sgd",
#         # --- prototype module ---
#         num_prototypes: int = 256,
#         proto_temperature: float = 0.1,
#         align_mode: str = "soft",          # "soft" | "hard"
#         sim_mode: str = "cosine",          # "cosine" | "dot"
#         combine_mode: str = "mul",         # "mul" | "add"
#         lambda_sim: float = 0.5,
#         attn_threshold: float = 0.0,
#         attn_temperature: float = 1.0,
#         renormalize_after_mask: bool = False,
#         lambda_me: float = 1.0,
#         normalize_prototypes: bool = True,
#         detach_hard_alignment: bool = True,
#         # --- fusion module ---
#         fusion_layers: int = 2,
#         fusion_heads: int = 4,
#         fusion_ff_mult: int = 4,
#         fusion_use_weights_as_gating: bool = True,
#         fusion_output_mode: str = "query",  # "query" | "mean"
#         # --- misc ---
#         return_debug: bool = False):
#         super().__init__()
#         self.save_hyperparameters(ignore=["backbone"])
#         self.config = config
#         self.optimizer_name = optimizer

#         rope_model_types = {"modernbert", "roformer", "mamba"}
#         model_type = getattr(config, "model_type", "").lower()
#         is_rope = model_type in rope_model_types


#         self.encoders = EHRRAPEncoders(
#             config=config,
#             backbone=backbone,
#             dropout=dropout,
#             pooling=pooling,
#             use_time=use_time,
#             use_numeric=use_numeric,
#             ckpt_path=ckpt_path,
#             return_token_level=False)

#         self.prototypes = PrototypeRetrievalModule(hidden_size=config.hidden_size,
#                                                    num_prototypes=num_prototypes,
#                                                    temperature=proto_temperature,
#                                                    align_mode=align_mode,
#                                                    sim_mode=sim_mode,
#                                                    combine_mode=combine_mode,
#                                                    lambda_sim=lambda_sim,
# #                                                    attn_threshold=attn_threshold,
# #                                                    attn_temperature=attn_temperature,
# #                                                    renormalize_after_mask=renormalize_after_mask,
# #                                                    return_debug=return_debug,
#                                                    normalize_prototypes=normalize_prototypes,
#                                                    detach_hard_alignment=detach_hard_alignment
#                                                   )

#         self.fusion = FusionModule(hidden_size=config.hidden_size,
#                                    num_layers=fusion_layers,
#                                    num_heads=fusion_heads,
#                                    ff_mult=fusion_ff_mult,
#                                    dropout=dropout,
#                                    use_weights_as_gating=fusion_use_weights_as_gating,
#                                    output_mode=fusion_output_mode,
#                                    return_seq=False)

#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.criterion = nn.BCEWithLogitsLoss()


# #         if freeze:
# #             for p in self.encoders.parameters():
# #                 p.requires_grad = False
# #             for p in self.prototypes.parameters():
# #                 p.requires_grad = True
# #             for p in self.fusion.parameters():
# #                 p.requires_grad = True
# #             for p in self.classifier.parameters():
# #                 p.requires_grad = True

#         self.lr = lr
#         self.wd = wd
#         self.max_epochs = max_epochs

#         self.train_auroc = BinaryAUROC()
#         self.train_auprc = BinaryAveragePrecision()
#         self.val_auroc = BinaryAUROC()
#         self.val_auprc = BinaryAveragePrecision()
#         self.test_auroc = BinaryAUROC()
#         self.test_auprc = BinaryAveragePrecision()

#         self._train_preds, self._train_labels = [], []
#         self._val_preds, self._val_labels = [], []
#         self._test_preds, self._test_labels = [], []
        

#         self.return_debug = return_debug

#     def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:

#         enc_out = self.encoders(batch, query_key="query", history_key="history")

        
#         proto_out = self.prototypes(
#             query_vec=enc_out["query_vec"],
#             hist_vec=enc_out["hist_vec"])

        # print('\n', 'sim', proto_out['sim'],'\n')
        # print('\n','align', proto_out['align'],'\n')
        # print('\n','final_scores=',proto_out['final_scores'],'\n')
        # # print('query_probs', proto_out['query_probs'])
        # # print('hist_probs', proto_out['hist_probs'])
        
        
#         # print('attn_weights=',proto_out['attn_weights'],'\n')
#         # print('attn_mask=',proto_out['attn_mask'],'\n\n\n\n')
        

#         fuse_out = self.fusion(
#             query_vec=enc_out["query_vec"],
#             hist_vec=enc_out["hist_vec"],
#             attn_mask=proto_out.get("attn_mask", None),
#             attn_weights=proto_out.get("attn_weights", None),)

#         fused_vec = fuse_out["fused_vec"]
# #         print('fused_vec=', fused_vec,)
#         logits = self.classifier(fused_vec).squeeze(-1) 

#         out = {"logits": logits, "fused_vec": fused_vec, "proto": proto_out}

#         if self.return_debug:
#             out["debug"] = {"attn_keep_rate": proto_out["attn_mask"].float().mean(dim=1),
#                             "attn_max_weight": proto_out["attn_weights"].max(dim=1).values,}
#         return out


#     def shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
#         out = self.forward(batch)

#         if "proto" in out:
#             proto = out["proto"]
#             if "diag_q_ent" in proto:
#                 self.log(f"{stage}_q_ent", proto["diag_q_ent"], prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
#                 self.log(f"{stage}_q_max", proto["diag_q_max"], prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
#                 self.log(f"{stage}_h_ent", proto["diag_h_ent"], prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
#                 self.log(f"{stage}_h_max", proto["diag_h_max"], prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)

#         logits = out["logits"]
#         y = batch["label"].float().view(-1)
#         loss = self.criterion(logits, y)

#         # ------------------- ME-MAX regularizer (entropy of mean assignment) -------------------
#         # Uses your existing prototype probabilities:
#         #   out["proto"]["query_probs"] and out["proto"]["hist_probs"] with shape [B, K] (probabilities over K prototypes)
#         if stage == "train" and "proto" in out and ("query_probs" in out["proto"]) and ("hist_probs" in out["proto"]):
#             q_probs = out["proto"]["query_probs"]   # [B, K]
#             h_probs = out["proto"]["hist_probs"]    # [B, K]
#             if q_probs.dim() > 2:
#                 q_probs = q_probs.reshape(-1, q_probs.shape[-1])
#             if h_probs.dim() > 2:
#                 h_probs = h_probs.reshape(-1, h_probs.shape[-1])
#             # mean over the two "views" (query + history) for this batch
#             p_bar_local = torch.cat([q_probs, h_probs], dim=0).mean(dim=0)  # [K]

#             # If distributed, average across devices to get global batch mean
#             p_bar_all = self.all_gather(p_bar_local)   # [world_size, K] (typical Lightning behavior)
#             p_bar = p_bar_all.mean(dim=0)              # [K]

#             eps = 1e-8
#             H_pbar = -(p_bar * (p_bar + eps).log()).sum()

#             lambda_me = getattr(self.hparams, "lambda_me", 0.0)  # set this in config
#             loss = loss - lambda_me * H_pbar

#             self.log(f"{stage}_me_max_entropy", H_pbar, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
#             self.log(f"{stage}_me_max_term", (-lambda_me * H_pbar), prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
#         # --------------------------------------------------------------------------------------

#         pos_score = torch.sigmoid(logits)
#         self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)

#         if stage == "train":
#             self._train_labels.append(y.detach())
#             self._train_preds.append(pos_score.detach())
#         elif stage == "val":
#             self._val_labels.append(y.detach())
#             self._val_preds.append(pos_score.detach())
#         elif stage == "test":
#             self._test_labels.append(y.detach())
#             self._test_preds.append(pos_score.detach())

#         if "proto" in out and "attn_mask" in out["proto"]:
#             keep_rate = out["proto"]["attn_mask"].float().mean()
#             self.log(f"{stage}_keep_rate", keep_rate, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

#         return loss

#     def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
#         return self.shared_step(batch, stage="train")

#     def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
#         return self.shared_step(batch, stage="val")

#     def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
#         return self.shared_step(batch, stage="test")


#     def on_train_epoch_end(self) -> None:
#         if not self._train_preds:
#             return
#         y = torch.cat(self._train_labels)
#         p = torch.cat(self._train_preds)
#         self.log("train_auroc", self.train_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
#         self.log("train_auprc", self.train_auprc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
#         self._train_labels.clear()
#         self._train_preds.clear()

#     def on_validation_epoch_end(self) -> None:
#         if not self._val_preds:
#             return
#         y = torch.cat(self._val_labels)
#         p = torch.cat(self._val_preds)
#         self.log("val_auroc", self.val_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
#         self.log("val_auprc", self.val_auprc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
#         self._val_labels.clear()
#         self._val_preds.clear()

#     def on_test_epoch_end(self) -> None:
#         if not self._test_preds:
#             return
#         y = torch.cat(self._test_labels)
#         p = torch.cat(self._test_preds)
#         self.log("test_auroc", self.test_auroc(p, y.long()), on_epoch=True, logger=True)
#         self.log("test_auprc", self.test_auprc(p, y.long()), on_epoch=True, logger=True)

#         log_bootstrap_ci_text_percentile(
#             module=self,
#             y_true=y,
#             y_score=p,
#             prefix="test",
#             num_iter=1000,
#             alpha=0.05,
#             ndigits=3,
#             )

#         self._test_labels.clear()
#         self._test_preds.clear()

#     def on_fit_start(self):
#         opt = self.optimizers()
#         proto_ids = {id(p) for p in self.prototypes.parameters()}
#         opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
#         assert proto_ids.issubset(opt_ids), "Prototypes params are NOT in optimizer param_groups"

#     def configure_optimizers(self):
#         # sanity: make sure prototypes are trainable
#         assert any(p.requires_grad for p in self.prototypes.parameters()), "Prototypes have requires_grad=False"

#         # (optional but recommended) sanity: ensure prototypes are registered + present
#         proto_ids = {id(p) for p in self.prototypes.parameters()}
#         all_ids = {id(p) for p in self.parameters()}
#         assert proto_ids.issubset(all_ids), "Prototypes params are not in model.parameters() (not registered?)"

#         optimizer = torch.optim.AdamW(
#             [
#                 {"params": self.encoders.parameters()},
#                 {"params": self.fusion.parameters()},
#                 {"params": self.classifier.parameters()},
#                 {"params": self.prototypes.parameters()},  # <-- explicit
#             ],
#             lr=self.hparams.lr,
#             weight_decay=self.hparams.wd,
#         )

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer=optimizer,
#             T_max=self.max_epochs,
#             eta_min=0.0,)
#         return {"optimizer": optimizer, "lr_scheduler": scheduler}