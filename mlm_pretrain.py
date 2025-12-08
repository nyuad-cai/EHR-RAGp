import os
import torch
import wandb
import argparse

import lightning.pytorch as lt



from torch.utils.data import DataLoader
from lightning.pytorch.tuner import Tuner
from src.models.models import MLMPretraining
from lightning.pytorch.loggers import WandbLogger
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.data.datasets import SequencesGenerator, EHRPretrainDataset, MLMDataCollator, PROTECTED_TOKENS

parser = argparse.ArgumentParser(description='MLM pretraining command line interface')

parser.add_argument('--learning-rate', type=float, default=0.00001)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-epochs', type=int, default=100)
parser.add_argument('--chunk-length', type=int, default=1536,)
parser.add_argument('--overlap', type=int, default=192,)

args = parser.parse_args()
tokenizer_path = os.getenv('TOKENIZER_PATH')
data_path = os.getenv('DATA_PATH')
data_idx_path = os.getenv('DATA_IDX_PATH')
wandb_api_key = os.getenv('WANDB_API_KEY')
backbone_name = os.getenv('BACKBONE')
log_dir = os.getenv('LOG_DIR')
version = os.getenv('VERSION')
job_id = os.getenv('SLURM_JOB_ID')

ConfigClass, ModelClass = get_config_and_model_cls(backbone_name)

def get_run_dir(wandb_logger):
    run = wandb_logger.experiment
    d = getattr(run, "dir", None)
    if callable(d):
        d = d()
    # If still None, fall back to logger save_dir/version
    if not d:
        base = wandb_logger.save_dir or "."
        d = os.path.join(base, str(wandb_logger.version))
    return d  # guaranteed string now

@rank_zero_only
def make_dir(p):
    os.makedirs(p, exist_ok=True)



seq_gen = SequencesGenerator(tokenizer_path= tokenizer_path,
                             chunk_length=args.chunk_length,
                             overlap=args.overlap,
                             return_numeric=False,
                             return_text=False)

train_dataset = EHRPretrainDataset(dataset_path=data_path,
                                   data_idx_path=data_idx_path,
                                   seq_generator=seq_gen,
                                   split='train')

val_dataset = EHRPretrainDataset(dataset_path=data_path,
                                 data_idx_path=data_idx_path,
                                 seq_generator=seq_gen,
                                 split='val')

collate_fn = MLMDataCollator(tokenizer=seq_gen.tokenizer,
                             protected_tokens=PROTECTED_TOKENS,
                             mask_prob=0.15,
                             replace_prob=0.8,
                             random_prob=0.1)

train_dataoader = DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                             num_workers=8,
                             shuffle=True,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=4)

val_dataoader = DataLoader(dataset=val_dataset,
                             batch_size=args.batch_size,
                             num_workers=8,
                             shuffle=True,
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


model = MLMPretraining(config=cfg,
                       backbone=ModelClass,
                       lr=args.learning_rate,
                       wd=args.weight_decay,
                       max_epochs=args.max_epochs)


wandb.login(key=wandb_api_key)

wandb_logger = WandbLogger(project='MedEHR_Pretraining',
                           entity='nyuad-cai',
                           save_dir=log_dir,
                           version=f'{backbone_name}_{job_id}_{args.chunk_length}_{args.overlap}_{version}',
                           name=f'{backbone_name}_{job_id}_{args.learning_rate}_{args.chunk_length}_{args.overlap}_{version}',
                           tags=[backbone_name,version])




run_dir = get_run_dir(wandb_logger)          # <- never None
ckpt_dir = os.path.join(run_dir, "ckpt")
make_dir(ckpt_dir)
checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                      monitor='val_loss', 
                                      mode='min',
                                      every_n_epochs=1,
                                      save_top_k=5,)

early_stop = EarlyStopping(monitor='val_loss', 
                        min_delta=0.001,
                        mode='min', 
                        patience=5)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

def main():
    torch.set_float32_matmul_precision('high')
    trainer = lt.Trainer(accelerator='gpu', 
                        devices='auto',
                        strategy='ddp_find_unused_parameters_true',
                        logger=wandb_logger, 
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        max_epochs=args.max_epochs,
                        precision='16-mixed', 
                        callbacks=[checkpoint_callback,early_stop,lr_monitor]
                        )
    tuner = Tuner(trainer)

    lr_finder = tuner.lr_find(model,
                              train_dataloaders=train_dataoader,
                              num_training=200, 
                              method='fit',
                              mode='exponential', 
                              update_attr=True)
    print(f"Suggested learning rate: {lr_finder.suggestion()}") 

    trainer.fit(model=model, train_dataloaders=train_dataoader, val_dataloaders=val_dataoader)

if __name__ == "__main__":
    main()









