import os
import yaml
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import polars as pl


from tqdm import tqdm
from pathlib import Path
from datetime import time, timedelta



def read_gem_file(gems_path:str, icd9_list_path:str, icd10_list_path: str)-> pd.DataFrame: 
    colspecs = [(0, 5), (6, 13), (14, 19)]
    colnames = ['icd9', 'icd10', 'flags']
    
    gems = pd.read_fwf(gems_path, colspecs=colspecs, names=colnames, dtype=str)
    icd9_list = pd.read_csv(icd9_list_path,dtype=str)
    icd10_list = pd.read_csv(icd10_list_path,dtype=str)
    
    gems['icd9'] = gems['icd9'].str.strip()
    gems['icd10'] = gems['icd10'].str.strip()
    gems[['approximate', 'no_map', 'combination', 'scenario', 'choice_list']] = \
        (gems["flags"].apply(lambda x: pd.Series(list(x))).astype(int))
#     gems = gems[gems.no_map == 0]

    gems.drop(columns=['flags'], inplace=True)
    
    
    icd9_list = dict(zip(icd9_list.icd9, icd9_list.description))
    icd10_list = dict(zip(icd10_list.icd10, icd10_list.description))

    gems['icd9_description'] = gems.icd9.map(icd9_list)
    gems['icd10_description'] = gems.icd10.map(icd10_list)
    
    gems = gems[['icd9','icd9_description',
                 'icd10','icd10_description',
                 'approximate', 'combination', 
                 'scenario','choice_list','no_map']]
    
    return gems


def clean_race(race):
    splits = race.split('//')
    
    if splits[1].startswith('WHITE'):
        splits[1] =  'WHITE'
        
    elif splits[1].startswith('UNABLE'):
        splits[1] =  'UNKNOWN'

    elif splits[1].startswith('BLACK'):
        splits[1] =  'BLACK'

    elif splits[1].startswith('PATIENT'):
        splits[1] =  'UNKNOWN'
        
    elif splits[1].startswith('ASIAN'):
        splits[1] =  'ASIAN'

    elif splits[1].startswith('HISPANIC'):
        splits[1] =  'HISPANIC'

    elif splits[1].startswith('NATIVE'):
        splits[1] =  'NATIVE HAWAIIAN'

    elif splits[1].startswith('AMERICAN INDIAN'):
        splits[1] =  'AMERICAN INDIAN'
        
    return '//'.join(splits)



def clean_outpatient_measurements(
    df: pd.DataFrame,
    pre_post_buffer: pd.Timedelta = timedelta(days=1),
    max_gap_days: int = 30,
    lab_like=("LAB", "MICROBIOLOGY"),
    admission_code="ADMISSION-AT-HOSPITAL",
    discharge_code="DISCHARGE-FROM-HOSPITAL",
) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["code_type"] = out["code"].str.split("//").str[0]
    for c in ("out_id", "er_id", "disch_id"):
        out[c] = np.nan
    out["_order"] = np.arange(len(out))

    # Build visit windows
    admissions = (
        out[out["code"].str.startswith(admission_code, na=False)]
        [["subject_id", "hadm_id", "time"]]
        .rename(columns={"time": "admit_time"})
    )
    discharges = (
        out[out["code"].str.startswith(discharge_code, na=False)]
        [["subject_id", "hadm_id", "time"]]
        .rename(columns={"time": "disch_time"})
    )
    visits = pd.merge(admissions, discharges, on=["subject_id", "hadm_id"], how="inner")

    # 1) Original orphan hadm_id / out_id logic
    orphan = out[out["hadm_id"].isna() & out["code_type"].isin(lab_like)]
    drops = []
    for idx, row in orphan.iterrows():
        sid, t = row["subject_id"], row["time"]
        if pd.isna(t):
            drops.append(idx); continue
        pv = visits[visits["subject_id"] == sid].copy()
        if pv.empty:
            drops.append(idx); continue
        pv["delta"] = (pv["admit_time"] - t).abs()
        nearest = pv.loc[pv["delta"].idxmin()]
        hid, delta = nearest["hadm_id"], nearest["delta"]
        if delta <= pre_post_buffer:
            out.at[idx, "hadm_id"] = hid
        elif delta <= timedelta(days=max_gap_days):
            out.at[idx, "out_id"] = hid
        else:
            drops.append(idx)
    out = out.drop(index=drops)

    # 2) Tag ER and Discharge windows for *all* lab/micro events
    lab_events = out[out["code_type"].isin(lab_like)]
    for idx, row in lab_events.iterrows():
        sid, t = row["subject_id"], row["time"]
        if pd.isna(t):
            continue
        pv = visits[visits["subject_id"] == sid]
        for _, v in pv.iterrows():
            hid, a, d = v["hadm_id"], v["admit_time"], v["disch_time"]
            # ER: before admission
            if a - pre_post_buffer <= t < a:
                out.at[idx, "er_id"] = hid
                out.at[idx, "hadm_id"] = hid
            # Discharge: after discharge
            if d < t <= d + pre_post_buffer:
                out.at[idx, "disch_id"] = hid
                out.at[idx, "hadm_id"] = hid

    # 3) Restore order and return
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out



def push_procedure_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust PROCEDURE timestamps and properly order events per patient, visit-aware.
    
    Steps:
    - Push PROCEDURE timestamps at midnight to 23:59:59
    - Ensure RACE/GENDER always at the top of each patient's sequence
    - Sort events by:
        1. subject_id
        2. static (True first)
        3. earliest time per hadm_id (to respect admission order)
        4. event time within visits
    """
    df = df.copy()
    
    # Fix PROCEDURE timestamps at 00:00:00 → end of day
    mask_procedure = df["code"].str.startswith("PROCEDURE")
    mask_midnight = df["time"].dt.time == time(0, 0)
    df.loc[mask_procedure & mask_midnight, "time"] = (
        df.loc[mask_procedure & mask_midnight, "time"].dt.normalize() +
        pd.Timedelta(hours=23, minutes=59, seconds=59)
    )
    
    # Mark static tokens like RACE and GENDER
    df["is_static"] = df["code"].str.startswith(("RACE", "GENDER"))
    
    # Compute earliest time per hadm_id per patient to get admission order
    admission_order = (
        df[~df["hadm_id"].isna()]
        .groupby(["subject_id", "hadm_id"])["time"]
        .min()
        .reset_index()
        .rename(columns={"time": "hadm_order_time"})
    )
    
    df = df.merge(admission_order, on=["subject_id", "hadm_id"], how="left")

    # NaN hadm_id (outpatient) → set hadm_order_time = time (so it's positioned by actual time)
    df["hadm_order_time"] = df["hadm_order_time"].fillna(df["time"])

    # Final sort: static first, then visit order, then time
    df = df.sort_values(
        by=["subject_id", "is_static", "hadm_order_time", "time"],
        ascending=[True, False, True, True]
    ).drop(columns=["is_static", "hadm_order_time"]).reset_index(drop=True)

    return df



def make_mapping_decesion(decision1,decision2):
    if decision1 == decision2:
        return decision1
    elif decision1 != decision2:
        if type(decision2) == float:
            return decision1[:3]
        elif decision1[:3] == decision2[:3]:
            return decision1[:3]
        else:
            return decision2[:3]
        



def build_best_icd10_mapper(gems_df: pd.DataFrame) -> dict:
    """
    Build best ICD-9 → ICD-10 mapping from GEMs file, prioritizing:
    - no_map == 0 (valid map)
    - combination == 0 (single-code map)
    - approximate == 0 (exact match preferred)
    - lowest choice_list (preferred target)

    Assumes ICD-9 codes are already in the same format as the target dataset.
    """
    df = gems_df.copy()

    # Filter for valid, standalone, mappable entries
    df = df[(df["no_map"] == 0) & (df["combination"] == 0)]

    # Ranking mechanism: prioritize exact match, low choice_list, shorter code
    df["rank"] = (
        df["approximate"] * 100 +
        df["choice_list"] * 10 +
        df["icd10"].str.len()
    )

    # Choose best ICD-10 per ICD-9
    best = (
        df.sort_values(["icd9", "rank"])
        .drop_duplicates(subset="icd9", keep="first")
    )

    # No padding applied to ICD-9 codes
    return dict(zip(best["icd9"].astype(str), best["icd10"].astype(str)))


def create_pairs_from_list(input_list):
    if not input_list:
        return []

    # Check if the list has an odd number of elements
    if len(input_list) % 2 != 0:
        # If odd, the last element is handled separately
        pairs = list(zip(input_list[:-1:2], input_list[1::2]))
        pairs.append(input_list[-1])  # Add the last element as a single item
    else:
        # If even, create pairs from the entire list
        pairs = list(zip(input_list[::2], input_list[1::2]))
    return pairs



def add_emer_outp_boundries(subject_sequence: pd.DataFrame):
    
    hadm_ids = subject_sequence.hadm_id.unique()
    hadm_ids = hadm_ids[~np.isnan(hadm_ids)]
    
    patient_admissions = []
    demographics = subject_sequence[subject_sequence.code.str.startswith(('RACE','GENDER'))]
    death = subject_sequence[subject_sequence.code.str.startswith('MEDS_DEATH')]
    
    for idx in hadm_ids:
        patient_admissions.append(subject_sequence[subject_sequence.seq_id == idx])
    
    processed_admissions = []
    for admission in patient_admissions:
        
        seq_id = admission.seq_id.unique()[0]

        emer = admission[admission.er_id==seq_id]
        if emer.shape[0] != 0:
            start = emer.iloc[[0]]
            start['code'] = 'EMERGENCY-START'
            start.iloc[:,9:] =  np.nan
            start['code_type'] = 'EMERGENCY-START'

            end = emer.iloc[[-1]]
            end['code'] = 'EMERGENCY-END'
            end.iloc[:,9:] =  np.nan
            end['code_type'] = 'EMERGENCY-END'
            emer = pd.concat((start,emer,end))

        hadm = admission[admission.hadm_id==seq_id]
        out = admission[admission.out_id==seq_id]

        pre_out = out[out.time < hadm.time.iloc[0]]
        if pre_out.shape[0] != 0:
            start = pre_out.iloc[[0]]
            start['code'] = 'OUTPATIENT-START'
            start.iloc[:,9:] =  np.nan
            start['code_type'] = 'OUTPATIENT-START'
            
            end = pre_out.iloc[[-1]]
            end['code'] = 'OUTPATIENT-END'
            end.iloc[:,9:] =  np.nan
            end['code_type'] = 'OUTPATIENT-END'

            pre_out = pd.concat((start,pre_out,end))

        post_out = out[out.time > hadm.time.iloc[-1]]
        if post_out.shape[0] != 0:
            start = post_out.iloc[[0]]
            start['code'] = 'OUTPATIENT-START'
            start.iloc[:,9:] =  np.nan
            start['code_type'] = 'OUTPATIENT-START'

            end = post_out.iloc[[-1]]
            end['code'] = 'OUTPATIENT-END'
            end.iloc[:,9:] =  np.nan
            end['code_type'] = 'OUTPATIENT-END'

            post_out = pd.concat((start,post_out,end))
        
        processed = pd.concat((pre_out,emer,hadm,post_out))
        processed_admissions.append(processed)
    
    return pd.concat((demographics,*processed_admissions,death),ignore_index=True)


def within(duration: float):
    if duration >= 0 and duration <= 7:
        return '1-W'
    elif duration > 7 and duration <= 14:
        return '2-W'
    elif duration > 14 and duration <= 21:
        return '3-W'
    elif duration > 21 and duration <= 30:
        return '1-M'
    elif duration > 30 and duration <= 60:
        return '2-M'    
    elif duration > 60 and duration <= 90:
        return '3-M' 
    elif duration > 90 and duration <= 120:
        return '4-M' 
    elif duration > 120 and duration <= 150:
        return '5-M' 
    elif duration > 150 and duration <= 180:
        return '6-M' 
    elif duration > 180 and duration <= 210:
        return '7-M' 
    elif duration > 210 and duration <= 240:
        return '8-M'
    elif duration > 240 and duration <= 270:
        return '9-M' 
    elif duration > 270 and duration <= 300:
        return '10-M'
    elif duration > 300 and duration <= 330:
        return '11-M'    
    elif duration > 330 and duration <= 360:
        return '12-M'     
    elif duration > 360:
        return '1-Y+'
    elif duration < 0:
        return '1-W'

def add_time_tokens(subject_sequence: pd.DataFrame):
    
    subject_sequence['time_diff'] = subject_sequence.time.diff().dt.total_seconds()/(3600*24)
    hadm_ids = subject_sequence.hadm_id.unique()
    hadm_ids = hadm_ids[~np.isnan(hadm_ids)]
    
    patient_admissions = []
    demographics = subject_sequence[subject_sequence.code.str.startswith(('RACE','GENDER'))]
    death = subject_sequence[subject_sequence.code.str.startswith('MEDS_DEATH')]
    
    for idx in hadm_ids:
        patient_admissions.append(subject_sequence[subject_sequence.seq_id == idx])
    
    processed_admissions = []
    for admission in patient_admissions:
        
        seq_id = admission.seq_id.unique()[0]
        emer = admission[admission.er_id==seq_id]
        hadm = admission[admission.hadm_id==seq_id]
        out = admission[admission.out_id==seq_id]
        pre_out = out[out.time < hadm.time.iloc[0]]
        post_out = out[out.time > hadm.time.iloc[-1]]
        
        if pre_out.shape[0] != 0:
            if pd.isna(pre_out.iloc[0].time_diff):
                pass
            else:
                time_diff = pre_out.iloc[0].time_diff
                time_period = within(time_diff)
                
                time_token = pre_out.iloc[[0]]
                time_token['code'] = 'TIME-GAP//' + time_period
                time_token.iloc[:,9:-1] =  np.nan
                time_token['numeric_value'] = time_diff
                time_token['text_value'] = time_period
                time_token['code_type'] = 'TIME-GAP'
                
                pre_out = pd.concat((time_token,pre_out))
        
        if emer.shape[0] != 0:
            if pd.isna(emer.iloc[0].time_diff):
                pass
            else:
                time_diff = emer.iloc[0].time_diff
                time_period = within(time_diff)
                
                time_token = emer.iloc[[0]]
                time_token['code'] = 'TIME-GAP//' + time_period
                time_token.iloc[:,9:-1] =  np.nan
                time_token['numeric_value'] = time_diff
                time_token['text_value'] = time_period
                time_token['code_type'] = 'TIME-GAP'
                
                emer = pd.concat((time_token,emer))
        else:
            if pd.isna(hadm.iloc[0].time_diff):
                pass
            else:
                time_diff = hadm.iloc[0].time_diff
                time_period = within(time_diff)

                time_token = hadm.iloc[[0]]
                time_token['code'] = 'TIME-GAP//' + time_period
                time_token.iloc[:,9:-1] =  np.nan
                time_token['numeric_value'] = time_diff
                time_token['text_value'] = time_period
                time_token['code_type'] = 'TIME-GAP'

                hadm = pd.concat((time_token,hadm))


        if post_out.shape[0] != 0:
            if pd.isna(post_out.iloc[0].time_diff):
                pass
            else:
                time_diff = post_out.iloc[0].time_diff
                time_period = within(time_diff)
                
                time_token = post_out.iloc[[0]]
                time_token['code'] = 'TIME-GAP//' + time_period
                time_token.iloc[:,9:-1] =  np.nan
                time_token['numeric_value'] = time_diff
                time_token['text_value'] = time_period
                time_token['code_type'] = 'TIME-GAP'
                
                post_out = pd.concat((time_token,post_out))

        
        processed = pd.concat((pre_out,emer,hadm,post_out))
        processed_admissions.append(processed)
    
    if death.shape[0] != 0:
        if death.iloc[0].time_diff > 1:
            time_diff = death.iloc[0].time_diff
            time_period = within(time_diff)
                
            time_token = death.iloc[[0]]
            time_token['code'] = 'TIME-GAP//' + time_period
            time_token.iloc[:,9:-1] =  np.nan
            time_token['numeric_value'] = time_diff
            time_token['text_value'] = time_period
            time_token['code_type'] = 'TIME-GAP'
                
            death = pd.concat((time_token,death))
            
            
    
    return pd.concat((demographics,*processed_admissions,death),ignore_index=True)






def get_mortality_labels(data_path: str) -> pl.DataFrame:
    # TODO: In-ICU moratlity labels 
    # TODO: time-bound mortality label 24/48
    pieces = []
    for file in tqdm(os.listdir(data_path)):
        data = pl.scan_parquet(os.path.join(data_path,file)).collect()
        idx = (data.group_by(['subject_id','seq_id'])
                   .len()
                   .rename({'len':'n_events'})
                   .sort(['subject_id'])
                   .filter(pl.col('seq_id')
                   .is_nan() == False)
              )
        
        idx = idx.with_columns(pl.lit(file).alias('shard'))
        
        died_in_hosp = (data.filter(pl.col('died_in_hosp')
                            .is_nan() == False)['subject_id',
                                                'seq_id',
                                                'died_in_hosp'])
        one_year_mortality = (data.filter(pl.col('code').str.starts_with('MEDS'))
                                  .filter((pl.col("time").dt.hour() == 0) & 
                                          (pl.col("time").dt.minute() == 0) &
                                          (pl.col("time").dt.second() == 0)
                                         ).select(pl.col('subject_id'))
                                   .with_columns(pl.lit(1).alias("one_year_mortality"))
                    )
        
        
        idx = idx.join(died_in_hosp, on=['subject_id','seq_id'],how='left')
        idx = idx.join(one_year_mortality, on=['subject_id'],how='left').fill_null(0)
        idx = idx.with_columns(pl.col(['died_in_hosp','one_year_mortality']).cast(pl.Int64))
        pieces.append(idx)


    df = (pl.concat(pieces, how="vertical").sort('subject_id'))

    return df





def run_meds_transform_from_dict(config: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        config_fp = f.name

    try:
        subprocess.run(
            [
                "MEDS_transform-runner",
                f"pipeline_config_fp={config_fp}",
                "~parallelize",
            ],
            check=True,
        )
    finally:
        os.remove(config_fp)

# MEDS-tranform configs phase1: basic metadata aggregation and outlier occlusion
phase1_config = {
    "input_dir": None,
    "output_dir": None,
    "stages": [
        {
            "aggregate_code_metadata": {
                "aggregations": [
                    "values/n_occurrences",
                    "values/sum",
                    "values/sum_sqd",
                    "values/min",
                    "values/max",
                ]
            }
        },
        {
            "occlude_outliers": {
                "stddev_cutoff": 3,
            }
        },
    ],
}
# MEDS-transform configs phase2: advanced metadata, vocab indexing, normalization
phase2_config = {
    "input_dir": None,
    "output_dir": None,
    "stages": [
        {
            "aggregate_code_metadata": {
                "aggregations": [
                    "values/n_occurrences",
                    "values/sum",
                    "values/sum_sqd",
                    "code/n_occurrences",
                    "code/n_subjects",
                    "values/min",
                    "values/max",
                ]
            }
        },
        "fit_vocabulary_indices",
        {
            "fit_normalization": {
                "_base_stage": "aggregate_code_metadata",
                "aggregations": [
                    "values/n_occurrences",
                    "values/sum",
                    "values/sum_sqd",
                ],
            }
        },
        "normalization",
    ],
}



from datasets import Dataset, Features, Sequence, Value
import os
import polars as pl


def build_arrow_dataset(
    data_idx_fp,
    normalized_train_dir,
    output_dir,
    seq_gen,
    writer_batch_size=1000,
):
    data_idx = pl.read_parquet(data_idx_fp)

    features = Features({
        "subject_id": Value("int32"),
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "visit_ids": Sequence(Value("int16")),
        "stage_ids": Sequence(Value("int8")),
        "type_ids": Sequence(Value("int16")),
        "numeric_values": Sequence(Value("float32")),
        "numeric_mask": Sequence(Value("int8")),
        "text_values": Sequence(Value("string")),
        "text_mask": Sequence(Value("int8")),
        "time_diff": Sequence(Value("float32")),
        "time_stamp": Sequence(Value("timestamp[s]")),
        "seq_id": Sequence(Value("int32")),
        "out_id": Sequence(Value("int32")),
        "er_id": Sequence(Value("int32")),
        "hadm_id": Sequence(Value("int32")),
        "icustay_id": Sequence(Value("int32")),
    })

    def gen():
        shard_cache = {}

        for row in data_idx.iter_rows(named=True):
            subject_id = row["subject_id"]
            shard = row["shard"]

            if shard not in shard_cache:
                shard_cache[shard] = pl.read_parquet(
                    os.path.join(normalized_train_dir, shard)
                )

            file_df = shard_cache[shard]
            seq = file_df.filter(pl.col("subject_id") == subject_id)

            ex = seq_gen.encode_sequence(seq)

            yield {
                "subject_id": int(subject_id),
                "input_ids": ex["input_ids"],
                "attention_mask": ex["attention_mask"],
                "visit_ids": ex["visit_ids"],
                "stage_ids": ex["stage_ids"],
                "type_ids": ex["type_ids"],
                "numeric_values": ex["numeric_values"],
                "numeric_mask": ex["numeric_mask"],
                "text_values": ex["text_values"],
                "text_mask": ex["text_mask"],
                "time_diff": ex["time_diff"],
                "time_stamp": ex["time_stamp"],
                "seq_id": ex["seq_id"],
                "out_id": ex["out_id"],
                "er_id": ex["er_id"],
                "hadm_id": ex["hadm_id"],
                "icustay_id": ex["icustay_id"],
            }

    ds_arrow = Dataset.from_generator(
        gen,
        features=features,
        writer_batch_size=writer_batch_size,
    )

    ds_arrow.save_to_disk(output_dir)



def empty_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()