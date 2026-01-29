import os
import yaml
import torch
import wandb
import optuna
import argparse

import multiprocessing as mp
import lightning.pytorch as lt

from torch.utils.data import DataLoader
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from src.models.models import EHRRAPEvalModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from src.data.datasets import limits, SequencesGenerator, RetrievalEvalDataset, EvalCollator, RetrievalCollator
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos, load_config_with_env
from src.vectordb.databases import EHREmbedder, ChromaEHREmbeddingFunction, EmbedCollator
from transformers import RoFormerConfig, RoFormerModel




    
def main():
    seq_gen = SequencesGenerator(tokenizer_path='./vocab.json',
                                chunk_length=512,
                                overlap=0)

    # IMPORTANT
    VOCAB_SIZE = 47377
    config = RoFormerConfig(vocab_size=VOCAB_SIZE,
                            hidden_size=768,
                            num_hidden_layers=12,
                            num_attention_heads=12,
                            intermediate_size=3072,
                            max_position_embeddings=1536,
                            pad_token_id=0,
                            type_vocab_size= 28,
                            visit_vocab_size= 102,
                            stage_vocab_size= 5)


    embedder = EHREmbedder(
        config=config,
        backbone=RoFormerModel,
        ckpt_path="/scratch/sas10092/ehr-foundation/models/mlm/wandb/run-20251128_073215-roformer_13218339_1024_128_15_maskprob_12_5overlap/files/ckpt/epoch=65-step=665082.ckpt",
        pooling="mean",                        
        normalize=True)

    collate_fn = EmbedCollator()

    top_k = 18
    train_dataset = RetrievalEvalDataset(dataset_path='./data/meds_normalized_arrow/',
                                data_idx_path='./downstream_idx.parquet',
                                seq_gen=seq_gen,
                                embedder = embedder,
                                collate_fn= collate_fn,
                                limits_dict=limits,
                                vectordb_path='/scratch/sas10092/ehr-foundation/data/vectordbs/roformer/mean/within48_hist_full_512_64',
                                task='y_mort',
                                main_window='within48_query',
                                seq_length=512,
                                top_k=top_k,
                                use_time=True,
                                use_numeric=True,
                                add_cls=True,
                                split='train')

    val_dataset = RetrievalEvalDataset(dataset_path='./data/meds_normalized_arrow/',
                                data_idx_path='./downstream_idx.parquet',
                                seq_gen=seq_gen,
                                embedder = embedder,
                                collate_fn= collate_fn,
                                limits_dict=limits,
                                vectordb_path='/scratch/sas10092/ehr-foundation/data/vectordbs/roformer/mean/within48_hist_full_512_64',
                                task='y_mort',
                                main_window='within48_query',
                                seq_length=512,
                                top_k=top_k,
                                use_time=True,
                                use_numeric=True,
                                add_cls=True,
                                split='val')
    

  
        
    chunk_collator = EvalCollator()
    retrieval_collator = RetrievalCollator(chunk_collator=chunk_collator, top_k=top_k)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=retrieval_collator,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True,
        pin_memory_device='cuda'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=retrieval_collator,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True,
        pin_memory_device='cuda'
    )
    
    ConfigClass, ModelClass = get_config_and_model_cls('roberta', mode='eval', variant=None)

    cfg = ConfigClass(
    vocab_size=seq_gen.tokenizer.vocab_size,
    cls_token_id=seq_gen.tokenizer.cls_id,
    pad_token_id=seq_gen.tokenizer.pad_id,
    type_vocab_size=28,
    visit_vocab_size=102,
    stage_vocab_size=5,
    refernece_compile=False)
    cfg = fix_roberta_longformer_max_pos(cfg)

    ckpt_path = '/scratch/sas10092/ehr-foundation/models/mlm/wandb/run-20251126_174933-roberta_13216221_512_64_15_maskprob_12_5overlap/files/ckpt/epoch=87-step=198352.ckpt'


    model = EHRRAPEvalModel(config=cfg,
                            backbone=ModelClass,
                            ckpt_path= ckpt_path,
                            lr= 1e-05,
                            wd= 0.001,
                            max_epochs= 75,
                            dropout= 0.1,
                            freeze= False,
                            pooling= 'mean',
                            use_numeric= True,
                            use_time= True,
                            optimizer= 'sgd',
                            num_prototypes= 1024,
                            proto_temperature= 0.07,
                            align_mode= 'soft',
                            sim_mode= 'cosine',
                            combine_mode= 'add',
                            lambda_sim= 0.1,
                            attn_threshold= 0.05,
                            attn_temperature= 0.07,
                            renormalize_after_mask= True,
                            normalize_prototypes= True,
                            detach_hard_alignment= False,
                            fusion_layers= 2,
                            fusion_heads= 4,
                            fusion_ff_mult= 4,
                            fusion_use_weights_as_gating= True,
                            fusion_output_mode= 'mean',
                            return_debug= False)
    torch.set_float32_matmul_precision('high')
    trainer = lt.Tratrainer = lt.Trainer(accelerator='auto', 
                        devices='auto',
                        strategy='auto',
    #                     logger=wandb_logger, 
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        max_epochs=75,
                        precision='16-mixed',
    #                     callbacks=[early_stop,lr_monitor,checkpoint_callback],
                        )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()