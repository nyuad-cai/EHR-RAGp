import os
import yaml
import torch
import wandb


import multiprocessing as mp
import lightning.pytorch as lt
import torch.distributed as dist

from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from src.models.models import EHRRAPEvalModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from src.data.datasets import limits, SequencesGenerator, RetrievalEvalDataset, EvalCollator, RetrievalCollator
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos, load_config_with_env
from src.vectordb.databases import EHREmbedder, ChromaEHREmbeddingFunction, EmbedCollator



def main():
    seq_gen = SequencesGenerator(tokenizer_path='./vocab.json',
                                    chunk_length=1024,
                                    overlap=0)

    ConfigClassRet, ModelClassRet = get_config_and_model_cls(model_type='roformer', mode='eval', variant=None)

    cfg_ret = ConfigClassRet(vocab_size=seq_gen.tokenizer.vocab_size,
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
                config=cfg_ret,
                backbone=ModelClassRet,
                ckpt_path='/scratch/sas10092/ehr-foundation/models/mlm/wandb/run-20251128_073215-roformer_13218339_1024_128_15_maskprob_12_5overlap/files/ckpt/epoch=65-step=665082.ckpt',
                pooling='mean',                        
                normalize='True')



    collate_fn_ret = EmbedCollator()


    # Settings:
    top_k = 1
    chroma_db_path = '/scratch/sas10092/ehr-foundation/data/vectordbs/roformer/mean/within24_hist_full_1024_128'
    task = 'y_los_7'
    main_window = 'within24_query'
    seq_length = 1024
    ckpt_path = '/scratch/sas10092/ehr-foundation/models/with_retrieval/wandb/run-20260120_183506-1_roformer_13809868_y_los_7_with_retrieval_0/files/ckpt/epoch=28-step=309401.ckpt'
    learning_rate = 0.0000121
    weight_decay= 0.003602
    pooling_enc= 'cls'
    num_prototypes= 1024
    lambda_sim=0
    fusion_gating= False
    pooling_fuse='query'
    test_dataset = RetrievalEvalDataset(dataset_path='/scratch/sas10092/ehr-foundation/data/meds_normalized_arrow',
                                        data_idx_path='./downstream_idx.parquet',
                                        seq_gen=seq_gen,
                                        embedder = embedder,
                                        collate_fn= collate_fn_ret,
                                        limits_dict=limits,
                                        vectordb_path=chroma_db_path,
                                        task=task,
                                        main_window=main_window,
                                        seq_length=seq_length,
                                        top_k=top_k,
                                        use_time=True,
                                        use_numeric=True,
                                        add_cls=True,
                                        split='test')


    chunk_collator = EvalCollator()
    retrieval_collator = RetrievalCollator(chunk_collator=chunk_collator, top_k=top_k)

    from torch.utils.data import DataLoader


    test_dataloader = DataLoader(dataset=test_dataset,
                                        batch_size=4,
                                        collate_fn=retrieval_collator,
                                        num_workers=4,
                                        prefetch_factor=2,
                                        persistent_workers=True,
                                        pin_memory=True,
                                        pin_memory_device='cuda',
                                        drop_last=True,
                                        )
    ConfigClass, ModelClass = get_config_and_model_cls(model_type='roformer', mode='eval', variant=None)


    cfg = ConfigClass(vocab_size=seq_gen.tokenizer.vocab_size,
                    cls_token_id=seq_gen.tokenizer.cls_id,
                    pad_token_id=seq_gen.tokenizer.pad_id,
                    type_vocab_size=28,
                    visit_vocab_size=102,
                    stage_vocab_size=5,
                    refernece_compile=False)

    cfg = fix_roberta_longformer_max_pos(cfg)


    model = EHRRAPEvalModel(config=cfg,
                            backbone=ModelClass,
                            ckpt_path= '/scratch/sas10092/ehr-foundation/models/mlm/wandb/run-20251128_073215-roformer_13218339_1024_128_15_maskprob_12_5overlap/files/ckpt/epoch=65-step=665082.ckpt',
                            lr= learning_rate,
                            wd= weight_decay,
                            max_epochs= 75,
                            dropout= 0.1,
                            freeze= False,
                            pooling=pooling_enc,
                            use_numeric= True,
                            use_time= True,
                            optimizer= 'sgd',
                            num_prototypes= num_prototypes,
                            proto_temperature= 0.07,
                            align_mode= 'soft',
                            sim_mode= 'cosine',
                            combine_mode= 'add',
                            lambda_sim= lambda_sim,
                            attn_threshold= 0.09,#1/top_k,
                            attn_temperature= 0.07,
                            renormalize_after_mask= True,
                            normalize_prototypes= True,
                            detach_hard_alignment= False,
                            fusion_layers= 2,
                            fusion_heads= 4,
                            fusion_ff_mult= 4,
                            fusion_use_weights_as_gating= fusion_gating,
                            fusion_output_mode= pooling_fuse,
                            return_debug= False)

    import lightning.pytorch as lt
    torch.set_float32_matmul_precision('high')
    trainer = lt.Trainer(accelerator='gpu', 
                        devices='auto',
                        strategy='auto',
    #                     logger=wandb_logger, 
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        max_epochs=75,
                        precision='16-mixed',
    #                     callbacks=[early_stop,lr_monitor,checkpoint_callback],
                        )

    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()