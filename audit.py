from collections import defaultdict
import os
import polars as pl
from tqdm.notebook import tqdm

data_path = '/scratch/sas10092/ehr-foundation/data/meds_normalized/data/train/'
downstream_idx = pl.read_parquet('/scratch/sas10092/ehr-foundation/resources/downstream_idx.parquet')


# delay_threshold_hours = 1.5

# audit = {
#     "within24": {
#         "samples": 0,
#         "samples_with_labs": 0,
#         "samples_with_potential_leakage": 0,
#         "total_labs": 0,
#         "potential_leaked_labs": 0,
#     },
#     "within48": {
#         "samples": 0,
#         "samples_with_labs": 0,
#         "samples_with_potential_leakage": 0,
#         "total_labs": 0,
#         "potential_leaked_labs": 0,
#     },
#     "within_stay": {
#         "samples": 0,
#         "samples_with_labs": 0,
#         "samples_with_potential_leakage": 0,
#         "total_labs": 0,
#         "potential_leaked_labs": 0,
#     },
# }


# for i in tqdm(range(len(downstream_idx))):

#     stay = downstream_idx[i]

#     subject_id = stay['subject_id'][0]
#     shard = stay['shard'][0]

#     q_start_24 = stay['w24_start_1024'][0]
#     q_end_24 = stay['w24_end_1024'][0]

#     q_start_48 = stay['w48_start_1024'][0]
#     q_end_48 = stay['w48_end_1024'][0]

#     q_start_st = stay['wStay_start_1024'][0]
#     q_end_st = stay['wStay_end_1024'][0]


#     shard_df = pl.read_parquet(
#         os.path.join(data_path, shard)
#     )

#     timeline = shard_df.filter(
#         pl.col('subject_id') == subject_id
#     )


#     windows = {
#         "within24": (q_start_24, q_end_24),
#         "within48": (q_start_48, q_end_48),
#         "within_stay": (q_start_st, q_end_st),
#     }


#     for name, (start, end) in windows.items():

#         audit[name]["samples"] += 1

#         query = timeline[start:end]

#         labs = query.filter(
#             pl.col("code").str.starts_with("LAB")
#         )

#         if labs.height == 0:
#             continue

#         audit[name]["samples_with_labs"] += 1
#         audit[name]["total_labs"] += labs.height


#         # last timestamp in query = prediction cutoff
#         cutoff = query["time"].max()


#         # labs occurring close to the cutoff
#         potential = labs.filter(
#             (
#                 cutoff - pl.col("time")
#             ).dt.total_hours()
#             <= delay_threshold_hours
#         )


#         n_potential = potential.height

#         audit[name]["potential_leaked_labs"] += n_potential

#         if n_potential > 0:
#             audit[name]["samples_with_potential_leakage"] += 1



# summary = []

# for window, values in audit.items():

#     summary.append({
#         "window": window,
#         "samples": values["samples"],
#         "samples_with_labs": values["samples_with_labs"],
#         "samples_with_potential_leakage": values["samples_with_potential_leakage"],
#         "total_labs": values["total_labs"],
#         "potential_leaked_labs": values["potential_leaked_labs"],
#         "sample_leakage_rate": (
#             values["samples_with_potential_leakage"]
#             / values["samples"]
#         ),
#         "lab_leakage_rate": (
#             values["potential_leaked_labs"]
#             / values["total_labs"]
#             if values["total_labs"] > 0 else 0
#         ),
#     })


# audit_summary = pl.DataFrame(summary)

# audit_summary.write_parquet('lab_audit.parquet')

# audit = {
#     "within24": {
#         "total": 0,
#         "with_diagnosis": 0,
#         "diagnosis_events": 0,
#     },
#     "within48": {
#         "total": 0,
#         "with_diagnosis": 0,
#         "diagnosis_events": 0,
#     },
#     "within_stay": {
#         "total": 0,
#         "with_diagnosis": 0,
#         "diagnosis_events": 0,
#     },
# }


# for i in tqdm(range(len(downstream_idx))):

#     stay = downstream_idx[i]

#     subject_id = stay['subject_id'][0]
#     shard = stay['shard'][0]

#     q_start_24 = stay['w24_start_1024'][0]
#     q_end_24 = stay['w24_end_1024'][0]

#     q_start_48 = stay['w48_start_1024'][0]
#     q_end_48 = stay['w48_end_1024'][0]

#     q_start_st = stay['wStay_start_1024'][0]
#     q_end_st = stay['wStay_end_1024'][0]


#     shard_df = pl.read_parquet(
#         os.path.join(data_path, shard)
#     )

#     timeline = shard_df.filter(
#         pl.col('subject_id') == subject_id
#     )


#     queries = {
#         "within24": timeline[q_start_24:q_end_24],
#         "within48": timeline[q_start_48:q_end_48],
#         "within_stay": timeline[q_start_st:q_end_st],
#     }


#     for name, query in queries.items():

#         diagnosis_events = query.filter(
#             pl.col('code').str.starts_with('DIAGNOSIS')
#         )

#         n_diag = diagnosis_events.height

#         audit[name]["total"] += 1

#         if n_diag > 0:
#             audit[name]["with_diagnosis"] += 1

#         audit[name]["diagnosis_events"] += n_diag



# summary = []

# for window, values in audit.items():

#     summary.append({
#         "window": window,
#         "total_queries": values["total"],
#         "queries_with_diagnosis": values["with_diagnosis"],
#         "diagnosis_event_count": values["diagnosis_events"],
#         "diagnosis_presence_rate": 
#             values["with_diagnosis"] / values["total"]
#     })


# audit_summary = pl.DataFrame(summary)

# audit_summary.write_parquet('audit_summary.parquet')







# leakage_candidates = []
# for i in tqdm(range(len(downstream_idx))):

#     stay = downstream_idx[i]

#     subject_id = stay["subject_id"][0]
#     stay_id = stay["icustay_id"][0]
#     shard = stay["shard"][0]

#     shard_df = pl.read_parquet(
#         os.path.join(data_path, shard)
#     )

#     timeline = shard_df.filter(
#         pl.col("subject_id") == subject_id
#     )

#     windows = {
#         "within24": (
#             stay["w24_start_1024"][0],
#             stay["w24_end_1024"][0],
#         ),
#         "within48": (
#             stay["w48_start_1024"][0],
#             stay["w48_end_1024"][0],
#         ),
#         "within_stay": (
#             stay["wStay_start_1024"][0],
#             stay["wStay_end_1024"][0],
#         ),
#     }
    
#     for name, (start, end) in windows.items():

#         query = timeline[start:end]

#         diag = query.filter(
#             pl.col("code").str.starts_with("DIAGNOSIS")
#         )
        
#         if diag.height > 0:
            
#             leakage_candidates.append({
#                 "icustay_id": stay_id,
#                 "subject_id": subject_id,
#                 "window": name,
#                 "query_start_idx": start,
#                 "query_end_idx": end,
#                 "diagnosis_codes": diag["code"].to_list(),
#                 "diagnosis_times": diag["time"].to_list(),
#             })

# candidates = pl.DataFrame(leakage_candidates)
# candidates.write_parquet('candidates.parquet')



def extract_query_features(query):

    features = {}

    # total events
    features["n_events"] = query.height

    # unique event types
    if "type" in query.columns:
        type_counts = (
            query
            .group_by("type")
            .len()
        )

        for row in type_counts.iter_rows(named=True):
            features[f"type_{row['type']}"] = row["len"]

    return features



import os
import polars as pl
from tqdm import tqdm


records_los = []
records_mort = []
records_icu = []
records_mort_1yr = []


for i in tqdm(range(len(downstream_idx))):

    stay = downstream_idx[i]

    subject_id = stay["subject_id"][0]
    icustay_id = stay["icustay_id"][0]
    split = stay["split"][0]
    shard = stay["shard"][0]

    shard_df = pl.read_parquet(
        os.path.join(data_path, shard)
    )

    timeline = shard_df.filter(
        pl.col("subject_id") == subject_id
    )


    def build_record(start_idx, end_idx, label_col):

        query = timeline[start_idx:end_idx]

        features = extract_query_features(query)

        # metadata first
        record = {
            "subject_id": subject_id,
            "icustay_id": icustay_id,
            "split": split,
        }

        # add extracted features
        record.update(features)

        # add label
        record["label"] = stay[label_col][0]

        return record


    # Long LOS 7d: within24
    records_los.append(
        build_record(
            stay["w24_start_1024"][0],
            stay["w24_end_1024"][0],
            "y_los_7"
        )
    )


    # In-hospital mortality: within48
    records_mort.append(
        build_record(
            stay["w48_start_1024"][0],
            stay["w48_end_1024"][0],
            "y_mort"
        )
    )


    # ICU readmission: withinStay
    records_icu.append(
        build_record(
            stay["wStay_start_1024"][0],
            stay["wStay_end_1024"][0],
            "y_icu_readmit_30"
        )
    )


    # 1 year mortality: withinStay
    records_mort_1yr.append(
        build_record(
            stay["wStay_start_1024"][0],
            stay["wStay_end_1024"][0],
            "y_mort_1yr"
        )
    )


# Convert to DataFrames
los_df = pl.DataFrame(records_los)
mort_df = pl.DataFrame(records_mort)
icu_df = pl.DataFrame(records_icu)
mort_1yr_df = pl.DataFrame(records_mort_1yr)


save_path = "./tabular_baselines"

os.makedirs(save_path, exist_ok=True)

los_df.write_parquet(
    os.path.join(save_path, "los_7.parquet")
)

mort_df.write_parquet(
    os.path.join(save_path, "mortality.parquet")
)

icu_df.write_parquet(
    os.path.join(save_path, "icu_readmit_30.parquet")
)

mort_1yr_df.write_parquet(
    os.path.join(save_path, "mortality_1yr.parquet")
)