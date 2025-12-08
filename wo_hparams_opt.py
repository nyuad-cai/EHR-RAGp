import os
import yaml
import torch
import wandb
import optuna
import argparse

import lightning.pytorch as lt

from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from src.models.models import EvalModel
from lightning.pytorch.loggers import WandbLogger
from src.data.datasets import limits, SequencesGenerator, EvalDataset, EvalCollator
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos, load_config_with_env


# lt.seed_everything(24, workers=True)
parser = argparse.ArgumentParser(description='MLM pretraining command line interface')
parser.add_argument('--config-path', type=str, required=True)
args = parser.parse_args()





config = load_config_with_env(args.config_path)



def objective(trial: optuna.trial.Trial) -> float:
    ConfigClass, ModelClass = get_config_and_model_cls(config['backbone_name'], mode='eval')
    try:
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
        pooling = trial.suggest_categorical("pooling",['cls','mean'])
        use_numeric = trial.suggest_categorical("use_numeric", [True, False])




        seq_gen = SequencesGenerator(tokenizer_path=config['seq_gen']['tokenizer_path'],
                                        chunk_length=config['seq_gen']['seq_length'],
                                        overlap=config['seq_gen']['overlap'])
        
        hparams={
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'pooling': pooling,
            'use_numeric': use_numeric}

        collate_fn = EvalCollator()

        train_dataset = EvalDataset(dataset_path=config['dataset']['data_path'],
                                    data_idx_path=config['dataset']['data_idx_path'],
                                    seq_gen=seq_gen,
                                    seq_length=config['seq_gen']['seq_length'],
                                    limits_dict=limits,
                                    main_window=config['main_window'],
                                    task=config['task'],
                                    use_time=True,
                                    use_numeric=use_numeric,
                                    split='train')

        val_dataset = EvalDataset(dataset_path=config['dataset']['data_path'],
                                    data_idx_path=config['dataset']['data_idx_path'],
                                    seq_gen=seq_gen,
                                    seq_length=config['seq_gen']['seq_length'],
                                    limits_dict=limits,
                                    main_window=config['main_window'],
                                    task=config['task'],
                                    use_time=True,
                                    use_numeric=use_numeric,
                                    split='val')


        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    num_workers=8,
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=4)

        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    num_workers=8,
                                    shuffle=False,
                                    collate_fn=collate_fn,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=4)




        cfg = ConfigClass(
            vocab_size=seq_gen.tokenizer.vocab_size,
            cls_token_id=seq_gen.tokenizer.cls_id,
            pad_token_id=seq_gen.tokenizer.pad_id,
            type_vocab_size=28,
            visit_vocab_size=102,
            stage_vocab_size=5,
            refernece_compile=False,
            )


        cfg = fix_roberta_longformer_max_pos(cfg)


        model = EvalModel(config=cfg,
                          backnone=ModelClass,
                          ckpt_path=config['ckpt_path'],
                          lr=learning_rate,
                          wd=weight_decay,
                          max_epochs=75,
                          pooling=pooling,
                          use_numeric=use_numeric,
                          use_time=True,
                          freeze=False,
                          optimizer='sgd')




        wandb.login(key=config['logger']['wandb_api_key'])

        wandb_logger = WandbLogger(project='MedEHR_Eval',
                                entity=config['logger']['entity'],
                                save_dir=config['logger']['log_dir'],
                                version=f"{config['backbone_name']}_{config['seq_gen']['seq_length']}_{config['job_id']}_{config['task']}_{config['main_window']}_{config['version']}_{trial.number}",
                                name=f"{config['backbone_name']}_{config['seq_gen']['seq_length']}_{config['job_id']}_{config['task']}_{config['main_window']}_{config['version']}_{trial.number}",
                                tags=[config['version'],config['task'],config['backbone_name'],'hparams_opt']) 

        # ckpt_dir = os.path.join(wandb_logger.experiment.dir, 'ckpt')
        # os.makedirs(ckpt_dir, exist_ok=True)
        # checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
        #                                     monitor='val_loss', 
        #                                     mode='min',
        #                                     every_n_epochs=1,
        #                                     save_top_k=1,)

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   mode='min', 
                                  patience=5)

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        torch.set_float32_matmul_precision('high')
        trainer = lt.Trainer(accelerator='auto', 
                            devices='auto',
                            strategy='auto',
                            logger=wandb_logger, 
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                            max_epochs=75,
                            precision='16-mixed', 
                            callbacks=[early_stop,lr_monitor]#,checkpoint_callback]
                            )
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    except optuna.exceptions.TrialPruned:
        wandb.finish()
        raise
    finally:
        wandb.finish()

    return early_stop.best_score.item()

pruner = optuna.pruners.NopPruner()

study = optuna.create_study(study_name=config['backbone_name'],
                            direction="minimize", 
                            storage=f'sqlite:////scratch/sas10092/ehr-foundation/models/optuna_dbs/{config["backbone_name"]}_{config["task"]}.db',
                            pruner=pruner,
                            load_if_exists=True,
                            sampler=TPESampler())

study.optimize(objective, n_trials=50,show_progress_bar=True,gc_after_trial=True)