import os
import torch
import wandb
import shutil
import optuna
import argparse


import lightning.pytorch as lt

from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from src.models.models import EHRRAPEvalModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from src.data.datasets import limits, RetrievalDataset, EvalCollator, RetrievalCollator
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos, load_config_with_env






parser = argparse.ArgumentParser(description='MLM pretraining command line interface')
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--chunking-strategy', type=str, default='overlap', choices=['overlap','time','visit','care_stage'])
parser.add_argument('--span', type=str, default='6.0', choices=['6.0','12.0','24.0','256','512','1024'])
parser.add_argument('--use-prototypes', action='store_true')


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

def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    else:
        raise RuntimeError("No GPU available")

gpu_bs = {256:{'NVIDIA A100 80GB PCIe': 12 ,
               'NVIDIA A100-SXM4-80GB':12,
               'NVIDIA H100 NVL': 16,
               'NVIDIA H200':20
               },

          512:{'NVIDIA A100 80GB PCIe': 8,
               'NVIDIA A100-SXM4-80GB':8,
               'NVIDIA H100 NVL': 12,
               'NVIDIA H200': 16
                },
          1024:{'NVIDIA A100 80GB PCIe': 4,
                'NVIDIA A100-SXM4-80GB':4,
                'NVIDIA H100 NVL': 6,
                'NVIDIA H200': 8
                } 
           }
def objective(trial: optuna.trial.Trial) -> float:
    
    try:
        # training
        learning_rate =trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        weight_decay =trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True)
        # model
        pooling_enc =trial.suggest_categorical("pooling_enc",['cls','mean'])
        pooling_fuse =trial.suggest_categorical("pooling_fuse",['query','mean'])
        use_augmentation = trial.suggest_categorical("use_augmentation", [1, 0])

        if args.use_prototypes:
            usage_lambda = trial.suggest_categorical("ent_lambda", [0.004, 0.005, 0.006, 0.007])
            proto_temperature = trial.suggest_categorical("proto_temperature", [[0.025,0.1], [0.05,0.2],[0.02,0.08]])
            softmax_temperature = trial.suggest_categorical("softmax_temperature", [0.1,0.15,0.2,0.25])
            num_prototypes =trial.suggest_categorical("num_prototypes",[64, 128, 256, 512])   
            prot_status = "with_proto"
        else:
            usage_lambda = 0.0
            use_augmentation = False
            proto_temperature = 1.0
            softmax_temperature=0
            num_prototypes = 1
            prot_status = "without_proto"

        if args.chunking_strategy == "overlap":
            seq_length_h = 256 
            overlap_map = {256: 32, 512: 64, 1024: 128} if config['dataset']['seq_length_q'] == 1024 else {256: 32, 512: 64}
            overlap_h = overlap_map[seq_length_h]
            window_hours = None
            span_dir = int(args.span)

            if seq_length_h == 1024:
                top_k = 8
                softmax_threshold = (1/top_k) * 0.96
                gpu_name = get_gpu_name()
                batch_size = gpu_bs[seq_length_h][gpu_name]
            elif seq_length_h == 512:
                top_k = 12
                softmax_threshold = (1/top_k) * 0.96
                gpu_name = get_gpu_name()
                batch_size = gpu_bs[seq_length_h][gpu_name]
            elif seq_length_h == 256:
                top_k = 24
                softmax_threshold = (1/top_k) * 0.96
                gpu_name = get_gpu_name()
                batch_size = gpu_bs[seq_length_h][gpu_name]

            

        elif args.chunking_strategy == "time":
            window_hours = 6.0
            seq_length_h = 256
            overlap_h = 0
            span_dir = float(args.span)
            top_k = 24
            softmax_threshold = (1/top_k) * 0.96
            gpu_name = get_gpu_name()
            batch_size = gpu_bs[seq_length_h][gpu_name]

        elif args.chunking_strategy == "visit":
            seq_length_h = 256
            overlap_h = 0
            window_hours = None
            span_dir = int(args.span)
            top_k = 24
            softmax_threshold = (1/top_k) * 0.96
            gpu_name = get_gpu_name()
            batch_size = gpu_bs[seq_length_h][gpu_name]

        elif args.chunking_strategy == "care_stage":
            seq_length_h = 256
            overlap_h = 0
            window_hours = None
            span_dir = int(args.span)
            top_k = 24
            softmax_threshold = (1/top_k) * 0.96
            gpu_name = get_gpu_name()
            batch_size = gpu_bs[seq_length_h][gpu_name]

        
        
        if config['dataset']['task'] == 'y_los_7':
            window = 'w24'
        elif config['dataset']['task'] == 'y_mort':    
            window = 'w48'
        elif config['dataset']['task'] in ['y_mort_1yr','y_icu_readmit_30']:
            window = 'wstay'

        train_dataset = RetrievalDataset(data_idx_path=config['dataset']['data_idx_path'],
                                         dataset_path= config['dataset']['data_path'],
                                         vectordb_path=f"/faiss/{config['dataset']['seq_length_q']}/{args.chunking_strategy}/{span_dir}/{window}",
                                         tokenizer_path=config['dataset']['tokenizer_path'],
                                         limits_dict=limits,
                                         chunking_strategy= args.chunking_strategy,
                                         task= config['dataset']['task'],
                                         query_window=config['dataset']['main_window_query'],
                                         history_window=config['dataset']['main_window_history'],
                                         top_k=top_k,
                                         seq_length_q = config['dataset']['seq_length_q'],
                                         overlap_q= config['dataset']['overlap_q'],
                                         seq_length_h= seq_length_h,
                                         overlap_h=overlap_h,
                                         use_time= True,
                                         use_numeric= True,
                                         add_cls=True,
                                         window_hours= window_hours,
                                         split= 'train')
        
        val_dataset = RetrievalDataset(data_idx_path=config['dataset']['data_idx_path'],
                                         dataset_path= config['dataset']['data_path'],
                                         vectordb_path=f"/faiss/{config['dataset']['seq_length_q']}/{args.chunking_strategy}/{span_dir}/{window}",
                                         tokenizer_path=config['dataset']['tokenizer_path'],
                                         limits_dict=limits,
                                         chunking_strategy= args.chunking_strategy,
                                         task= config['dataset']['task'],
                                         query_window=config['dataset']['main_window_query'],
                                         history_window=config['dataset']['main_window_history'],
                                         top_k=top_k,
                                         seq_length_q = config['dataset']['seq_length_q'],
                                         overlap_q= config['dataset']['overlap_q'],
                                         seq_length_h= seq_length_h,
                                         overlap_h=overlap_h,
                                         use_time= True,
                                         use_numeric= True,
                                         add_cls=True,
                                         window_hours= window_hours,
                                         split= 'val')
        

        

        chunk_collator = EvalCollator(tokenizer=train_dataset.query_gen.tokenizer,
                                      use_mask_augmentation= True if use_augmentation == 1 else False,
                                      augment_prob=0.25,
                                      mask_prob=0.125,
                                      )
        retrieval_collator = RetrievalCollator(chunk_collator=chunk_collator, top_k=top_k)

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
 

        ConfigClass, ModelClass = get_config_and_model_cls(model_type=config['backbone_name'], mode='eval', variant=config.get('variant', None))

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
                                ckpt_path=config['model']['ckpt_path'],
                                lr=learning_rate,
                                wd=weight_decay,
                                max_epochs=100,
                                dropout=0.1,
                                freeze=False,
                                pooling=pooling_enc,
                                use_numeric=True,
                                use_time=True,

                                # prototype module
                                num_prototypes=num_prototypes,
                                query_temperature=proto_temperature[0],
                                history_temperature=proto_temperature[1],
                                normalize_prototypes=True,

                                softmax_threshold=softmax_threshold,             
                                softmax_temperature=softmax_temperature,             
                                sample_ent_lambda=0.0,
                                usage_ent_lambda=usage_lambda,
                                use_prototypes=True,

                                # fusion
                                fusion_layers=2,
                                fusion_heads=4,
                                fusion_ff_mult=4,
                                fusion_output_mode=pooling_fuse,
                                use_weights_as_gating=True)



                
        wandb.login(key=config['logger']['wandb_api_key'])
        
        if config.get('variant') is not None:
            version = f"{config['backbone_name']}_{config['variant']}_{config['job_id']}_{config['dataset']['task']}_{args.chunking_strategy}_{span_dir}_{prot_status}_{trial.number}"
            name = f"{config['backbone_name']}_{config['variant']}_{config['job_id']}_{config['dataset']['task']}_{args.chunking_strategy}_{span_dir}_{prot_status}_{trial.number}"
            tags = [config['version'],config['dataset']['task'],config['backbone_name'],'hparams_opt',args.chunking_strategy, config["variant"],'existing-fm']
        else:
            version = f"{config['backbone_name']}_{config['job_id']}_{config['dataset']['task']}_{args.chunking_strategy}_{span_dir}_{prot_status}_{trial.number}"
            name = f"{config['backbone_name']}_{config['job_id']}_{config['dataset']['task']}_{args.chunking_strategy}_{span_dir}_{prot_status}_{trial.number}"
            tags = [config['version'],config['dataset']['task'],config['backbone_name'],'hparams_opt',args.chunking_strategy]

        wandb_logger = WandbLogger(project='MedEHR_Eval',
                                entity=config['logger']['entity'],
                                save_dir=config['logger']['log_dir'],
                                version=version,
                                name=name,
                                tags=tags) 
        
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
                                   patience=6)

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        torch.set_float32_matmul_precision('high')

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            
            precision = "bf16-mixed" if major >= 8 else "16-mixed"
        else:
            precision = '32-true'
        trainer = lt.Trainer(accelerator='gpu', 
                            devices='auto',
                            strategy='ddp_find_unused_parameters_true',
                            logger=wandb_logger, 
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                            max_epochs=100,
                            precision=precision,
                            callbacks=[early_stop,lr_monitor,checkpoint_callback],
                            )

        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # shutil.rmtree(ckpt_dir, ignore_errors=True)

    except optuna.exceptions.TrialPruned:
        wandb.finish()
        raise
    finally:
        wandb.finish()

    return early_stop.best_score.item()

def main():
    pruner = optuna.pruners.NopPruner() 
    prot_status = "with_proto" if args.use_prototypes else "without_proto"
    
    if config.get('variant') is not None:
        db_path = f'sqlite:////scratch/sas10092/ehr-foundation/models/optuna_dbs/with_retrieval/{config["backbone_name"]}_{config["variant"]}_{prot_status}_{config["dataset"]["task"]}_{args.span}_{args.chunking_strategy}.db'
    else:
        varint = None
        db_path = f'sqlite:////scratch/sas10092/ehr-foundation/models/optuna_dbs/with_retrieval/{config["backbone_name"]}_{prot_status}_{config["dataset"]["task"]}_{args.span}_{args.chunking_strategy}.db'

    study = optuna.create_study(study_name=config['backbone_name'],
                                direction="minimize", 
                                storage=db_path,
                                pruner=pruner,
                                sampler=TPESampler(),
                                load_if_exists=True)

    study.optimize(objective, n_trials=25,show_progress_bar=True,gc_after_trial=True)


if __name__ == "__main__":
    main()