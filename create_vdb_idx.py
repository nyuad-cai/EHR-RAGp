import os
from src.data.datasets import limits
from src.vectordb.databases import build_indices

data_idx_path = './resources/downstream_idx.parquet'
hf_dataset_path = './data/meds_normalized_arrow/'
tokenizer_path = './resources/vocab.json'
ckpt_path = './models/mlm/wandb/run-20251128_073215-roformer_13218339_1024_128_15_maskprob_12_5overlap/files/ckpt/epoch=65-step=665082.ckpt'
embedder_model = 'roformer'

seq_length_q = 1024
overlap_q = 0

windows = [
    ('within24_query', 'within24_hist_full', 'w24'),
    ('within48_query', 'within48_hist_full', 'w48'),
    ('within_stay_query', 'within_stay_hist_full', 'wstay'),
]

history_settings = [
    {'chunking_strategy': 'overlap', 'seq_length_h': 256,  'overlap_h': 32,  'window_hours': 6.0},
    {'chunking_strategy': 'overlap', 'seq_length_h': 512,  'overlap_h': 64,  'window_hours': 6.0},
    {'chunking_strategy': 'overlap', 'seq_length_h': 1024, 'overlap_h': 128, 'window_hours': 6.0},
    {'chunking_strategy': 'time',    'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 6.0},
    {'chunking_strategy': 'time',    'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 12.0},
    {'chunking_strategy': 'time',    'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 24.0},
    {'chunking_strategy': 'visit',   'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 6.0},
    {'chunking_strategy': 'care_stage', 'seq_length_h': 256, 'overlap_h': 0, 'window_hours': 6.0},
]

for cfg in history_settings:
    for main_window_q, main_window_h, window in windows:
        if cfg['chunking_strategy'] == 'time':
            span_dir = str(cfg['window_hours'])
        else:
            span_dir = str(cfg['seq_length_h'])

        save_path = os.path.join('/','faiss',str(seq_length_q), cfg['chunking_strategy'], str(span_dir), window)

        os.makedirs(save_path, exist_ok=True)

        print(f"Running: strategy={cfg['chunking_strategy']}, span={span_dir}, window={window}")

        build_indices(
            data_idx_path=data_idx_path,
            hf_dataset_path=hf_dataset_path,
            tokenizer_path=tokenizer_path,
            save_path=save_path,
            main_window_q=main_window_q,
            main_window_h=main_window_h,
            limits_dict=limits,
            ckpt_path=ckpt_path,
            embedder_model=embedder_model,
            chunking_strategy=cfg['chunking_strategy'],
            seq_length_q=seq_length_q,
            overlap_q=overlap_q,
            seq_length_h=cfg['seq_length_h'],
            overlap_h=cfg['overlap_h'],
            window_hours=cfg['window_hours'],
        )