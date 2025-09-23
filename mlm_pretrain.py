import os
import torch
import wandb
import argparse

import lightning.pytorch as lt

from transformers import RoFormerConfig
from torch.utils.data import DataLoader
from src.models.models import MLMPretraining
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.data.datasets import (Tokenizer, 
                               SequencesGenerator, 
                               EHRPretrainDataset, 
                               MLMDataCollator, 
                               PROTECTED_TOKENS)

parser = argparse.ArgumentParser(description='MLM pretraining command line interface')

# General arguments
parser.add_argument('--method', type=str, default='msn',
                    choices=['msn','simclr'])

# Data arguments
parser.add_argument('--modality', type=str, default='OCT-slice',
                    choices=['Infra-Red','OCT-slice','OCT-vol','Both'])
parser.add_argument('--normalization', type=str, default='OCT-slice',
                    choices=['Infra-Red','OCT-slice','OCT-vol','ImageNet','Both'])

# MSN arguments
parser.add_argument('--dim',type=int, default=384)
parser.add_argument('--num-prototypes', type=int, default=1024)
parser.add_argument('--learning-rate', type=float, default=0.00001)
parser.add_argument('--weight-decay', type=float, default=0.0001)
parser.add_argument('--mask-ratio', type=float, default=0.15)

# SimCLR arguments
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--pretrained', action='store_true')

# Trainer arguments
parser.add_argument('--max-epochs', type=int, default=100)

args = parser.parse_args()





seq_gen = SequencesGenerator(tokenizer_path=os.getenv('TOKENIZER_PATH'),
                             chunk_length=1024,
                             overlap=128,
                             return_numeric=False,
                             return_text=False)

dataset = EHRPretrainDataset(data_path=os.getenv('DATA_PATH'),
                             data_idx_path=os.getenv('DATA_IDX_PATH'),
                             seq_generator=seq_gen)

collate_fn = MLMDataCollator(tokenizer=seq_gen.tokenizer,
                             protected_tokens=PROTECTED_TOKENS,
                             mask_prob=0.15,
                             replace_prob=0.1)

dataoader = DataLoader(dataset=dataset,
                       batch_size=32,
                       shuffle=True,
                       collate_fn=collate_fn)



cfg = RoFormerConfig(
    vocab_size=seq_gen.tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=1024,
    pad_token_id=seq_gen.tokenizer.pad_id,
    type_vocab_size= 28,
    visit_vocab_size= 102,
    stage_vocab_size= 5)

model = MLMPretraining(config=cfg,
                       lr=1e-6,
                       wd=0.01,
                       max_epochs=100)


wandb.login(key=os.getenv('WANDB_API_KEY'))

wandb_logger = WandbLogger(project='OCT-Pretraining',
                        save_dir=os.getenv('LOG_DIR'),
                        # version=f'{version}{backbone_name}{job_id}',
                        # name=f'{run_name}{backbone_name}{job_id}_{learning_rate}',tags=['resnet18']
                        )

# ckpt_dir = os.path.join(wandb_logger.experiment.dir, 'ckpt')
# os.mkdir(ckpt_dir)
# checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
#                                     monitor='val_loss', 
#                                     mode='min',
#                                     every_n_epochs=2,
#                                     save_top_k=25,)

# early_stop = EarlyStopping(monitor='val_loss', 
#                         min_delta=0.000001,
#                         mode='min', 
#                         patience=10)

# lr_monitor = LearningRateMonitor(logging_interval='epoch')
torch.set_float32_matmul_precision('high')
trainer = lt.Trainer(accelerator='gpu', 
                    devices=torch.cuda.device_count(),
                    strategy='ddp',
                    # logger=wandb_logger, 
                    # log_every_n_steps=1,
                    max_epochs=args.max_epochs,
                    precision=16, 
                    # callbacks=[checkpoint_callback,early_stop,lr_monitor,pruning_callback]
                    )

trainer.fit(model=model, train_dataloaders=dataoader)








