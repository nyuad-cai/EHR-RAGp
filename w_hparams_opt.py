import os
import yaml
import torch
import wandb
import optuna
import argparse

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



parser = argparse.ArgumentParser(description='MLM pretraining command line interface')
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--chroma-db-path', type=str, required=True, default=None)  

args = parser.parse_args()
config = load_config_with_env(args.config_path)


lt.seed_everything(24, workers=True)
def get_run_dir(wandb_logger):
    run = wandb_logger.experiment
    d = getattr(run, "dir", None)
    if callable(d):
        d = d()
    if not d:
        base = wandb_logger.save_dir or "."
        d = os.path.join(base, str(wandb_logger.version))
    return d  

@rank_zero_only
def make_dir(p):
    os.makedirs(p, exist_ok=True)


def objective(trial: optuna.trial.Trial) -> float:
    
    try:
        learning_rate =0.0000121 #trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        weight_decay =0.003602 #trial.suggest_float("weight_decay", 1e-3, 1e-2)
        pooling_enc ='cls'#trial.suggest_categorical("pooling_enc",['cls','mean'])
        pooling_fuse ='query' #trial.suggest_categorical("pooling_fuse",['query','mean'])
        fusion_gating = False #trial.suggest_categorical("fusion_gating", [True, False])
        num_prototypes =1024 #trial.suggest_categorical("num_prototypes",[128,256,512,1024])
        lambda_sim = 0#trial.suggest_float("lambda_sim", 0.0, 0.5, step=0.1)

        hparams={'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'pooling': pooling_enc,
                'pooling_fuse': pooling_fuse,
                'fusion_gating': fusion_gating,
                'num_prototypes': num_prototypes,
                'lambda_sim': lambda_sim}
        

        seq_gen = SequencesGenerator(tokenizer_path=config['seq_gen']['tokenizer_path'],
                                        chunk_length=config['seq_gen']['seq_length'],
                                        overlap=config['seq_gen']['overlap'])
        
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
            ckpt_path=config['retriever']['ckpt_path'],
            pooling=config['retriever']['pooling'],                        
            normalize=config['retriever']['normalize'],)
        
        collate_fn_ret = EmbedCollator()

        ehr_embedf = ChromaEHREmbeddingFunction(embedder=embedder, 
                                                collate_fn=collate_fn_ret)
        top_k = config['dataset']['top_k']

        train_dataset = RetrievalEvalDataset(dataset_path=config['dataset']['data_path'],
                                             data_idx_path=config['dataset']['data_idx_path'],
                                             seq_gen=seq_gen,
                                             embedder = embedder,
                                             collate_fn= collate_fn_ret,
                                             limits_dict=limits,
                                             vectordb_path=args.chroma_db_path,
                                             task=config['dataset']['task'],
                                             main_window=config['dataset']['main_window_query'],
                                             seq_length=config['seq_gen']['seq_length'],
                                             top_k=top_k,
                                             use_time=True,
                                             use_numeric=True,
                                             add_cls=True,
                                             split='train')

        
        val_dataset = RetrievalEvalDataset(dataset_path=config['dataset']['data_path'],
                                             data_idx_path=config['dataset']['data_idx_path'],
                                             seq_gen=seq_gen,
                                             embedder = embedder,
                                             collate_fn= collate_fn_ret,
                                             limits_dict=limits,
                                             vectordb_path=args.chroma_db_path,
                                             task=config['dataset']['task'],
                                             main_window=config['dataset']['main_window_query'],
                                             seq_length=config['seq_gen']['seq_length'],
                                             top_k=top_k,
                                             use_time=True,
                                             use_numeric=True,
                                             add_cls=True,
                                             split='val')

        test_dataset = RetrievalEvalDataset(dataset_path=config['dataset']['data_path'],
                                             data_idx_path=config['dataset']['data_idx_path'],
                                             seq_gen=seq_gen,
                                             embedder = embedder,
                                             collate_fn= collate_fn_ret,
                                             limits_dict=limits,
                                             vectordb_path=args.chroma_db_path,
                                             task=config['dataset']['task'],
                                             main_window=config['dataset']['main_window_query'],
                                             seq_length=config['seq_gen']['seq_length'],
                                             top_k=top_k,
                                             use_time=True,
                                             use_numeric=True,
                                             add_cls=True,
                                             split='test')
        chunk_collator = EvalCollator() 
        retrieval_collator = RetrievalCollator(chunk_collator=chunk_collator, top_k=top_k)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config['dataloader']['batch_size'],
                                      collate_fn=retrieval_collator,
                                      num_workers=4,
                                      prefetch_factor=4,
                                      persistent_workers=True,
                                      drop_last=True,
                                      pin_memory_device='cuda',
                                      pin_memory=True,
                                      )
        
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    collate_fn=retrieval_collator,
                                    num_workers=1,
                                    prefetch_factor=1,
                                    persistent_workers=False,
                                    pin_memory=False,
                                    pin_memory_device='cuda',
                                    drop_last=True,
                                    )
        test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    collate_fn=retrieval_collator,
                                    num_workers=1,
                                    prefetch_factor=1,
                                    persistent_workers=False,
                                    pin_memory=False,
                                    pin_memory_device='cuda',
                                    drop_last=True,
                                    )   

        ConfigClass, ModelClass = get_config_and_model_cls(model_type=config['backbone_name'], mode='eval', variant=None)

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
                                ckpt_path= config['model']['ckpt_path'],
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
                
        wandb.login(key=config['logger']['wandb_api_key'])
        

        wandb_logger = WandbLogger(project='MedEHR_Eval',
                                entity=config['logger']['entity'],
                                save_dir=config['logger']['log_dir'],
                                version=f"2_{config['backbone_name']}_{config['job_id']}_{config['dataset']['task']}_{config['version']}_{trial.number}",
                                name=f"2_{config['backbone_name']}_{config['job_id']}_{config['dataset']['task']}_{config['version']}_{trial.number}",
                                tags=[config['version'],config['dataset']['task'],config['backbone_name'],'hparams_opt']) 
        
        run_dir = get_run_dir(wandb_logger)         
        ckpt_dir = os.path.join(run_dir, "ckpt")
        make_dir(ckpt_dir)
        
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                monitor='val_loss',
                                                mode='min',
                                                every_n_epochs=1,
                                                save_top_k=1)

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   mode='min', 
                                   patience=5)

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        torch.set_float32_matmul_precision('high')

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            
            precision = "16-mixed" if major >= 8 else "16-mixed"
        else:
            precision = '32-true'
        trainer = lt.Trainer(accelerator='gpu', 
                            devices='auto',
                            strategy='ddp_find_unused_parameters_true',
                            logger=wandb_logger, 
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                            max_epochs=75,
                            precision=precision,
                            callbacks=[early_stop,lr_monitor,checkpoint_callback],
                            )
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')

    except optuna.exceptions.TrialPruned:
        wandb.finish()
        raise
    finally:
        wandb.finish()

    return early_stop.best_score.item()

def main():
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(study_name=config['backbone_name'],
                                direction="minimize", 
                                storage=f'/path/to',
                                pruner=pruner,
                                load_if_exists=True,
                                sampler=TPESampler(seed=24))

    study.optimize(objective, n_trials=1,show_progress_bar=True,gc_after_trial=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
