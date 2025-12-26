
import os
import json
import torch
import bisect

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
                      },
    
    'within24_hist_icu': {512: ['wStay_min', 'w24_start_512' ],
                         1024: ['wStay_min', 'w24_start_1024'],
                         1536: ['wStay_min', 'w24_start_1536'],
                      },
    
    'within24_hist_full': {512: [ 0, 'w24_start_512' ],
                          1024: [ 0, 'w24_start_1024'],
                          1536: [ 0, 'w24_start_1536'],
                          },

    
    'within48_query': {512:  ['w48_start_512',  'w48_end_512' ],
                       1024: ['w48_start_1024', 'w48_end_1024'],
                       1536: ['w48_start_1536', 'w48_end_1536'],
                      },

    'within48_hist_icu': {512: ['wStay_min', 'w48_start_512' ],
                         1024: ['wStay_min', 'w48_start_1024'],
                         1536: ['wStay_min', 'w48_start_1536']
                      },
    
    'within48_hist_full': {512: [ 0, 'w48_start_512' ],
                          1024: [ 0, 'w48_start_1024'],
                          1536: [ 0, 'w48_start_1536']
                          },
    
    'within_stay_query': {512:  ['wStay_start_512',  'wStay_end_512' ],
                          1024: ['wStay_start_1024', 'wStay_end_1024'],
                          1536: ['wStay_start_1536', 'wStay_end_1536']
                         },
    
    'within_stay_hist_icu': {512:  ['wStay_min', 'wStay_start_512' ],
                             1024: ['wStay_min', 'wStay_start_1024'],
                             1536: ['wStay_min', 'wStay_start_1536'],
                            },
    
    'within_stay_hist_full': {512:  [ 0, 'w48_start_512' ],
                              1024: [ 0, 'w48_start_1024'],
                              1536: [ 0, 'w48_start_1536'],
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
                 split: str = 'train') -> None:
        
        needed_cols = ['subject_id', 'input_ids', 'attention_mask', 
                       'visit_ids', 'stage_ids', 'type_ids']

        if use_time:
            needed_cols.append('time_diff')
        if use_numeric:
            needed_cols.append('numeric_values')
            needed_cols.append('numeric_mask')
        self.start_limit = limits_dict[main_window][seq_length][0]
        self.end_limit   = limits_dict[main_window][seq_length][1]
        self.task = task
        
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

        # (then continue)
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
       
        
        
        start = stay[self.start_limit][0]
        end = stay[self.end_limit][0]

        
        timeline_encoded = self.hf_dataset.select(self.index[subject_id])[0]
        prediction_window = {k: (v[start:end] if isinstance(v, (list, np.ndarray)) else v) for k, v in timeline_encoded.items()}
        prediction_window = self.seq_gen.get_overlapped_chunks(prediction_window)
        prediction_window[0]['label'] = label
        
        return prediction_window[0]


class EvalCollator:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: List[Union[Dict, List[Dict]]]) -> Dict[str, torch.Tensor]:
        chunks = self._flatten(batch)
        out = self._stack(chunks)

        # ---- CLEAN NUMERIC VALUES ----
        if "numeric_values" in out:
            vals = out["numeric_values"].float()          # [B, L]
            finite_mask = torch.isfinite(vals)            # True where not NaN/inf

            # if numeric_mask already exists, AND it with finite_mask
            if "numeric_mask" in out:
                mask = out["numeric_mask"].bool() & finite_mask
            else:
                mask = finite_mask

            # replace NaN/inf with 0.0 (or any neutral value)
            vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

            out["numeric_values"] = vals
            out["numeric_mask"] = mask

#         # ---- OPTIONAL: CLEAN TIME FEATURES TOO ----
#         if "time_diff" in out:
#             t = out["time_diff"].float()
#             t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
#             out["time_diff"] = t

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




