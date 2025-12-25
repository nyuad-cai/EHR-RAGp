import os
import yaml
import torch
import wandb
import optuna
import argparse

import lightning.pytorch as lt

from types import SimpleNamespace
from torch.utils.data import DataLoader
from src.models.models import EvalModel
from lightning.pytorch.loggers import WandbLogger
from src.models.baseline_models import DescEmbEvalModel
from src.data.baseline_datasets import DescEmbDataset, DescEmbCollator
from src.models.baseline_models import GenHPFDownstreamModule, GenHPFEncoder
from src.models.baseline_models import REMedWithGenHPF, REMedLightningModule
from src.data.baseline_datasets import REMedGenHPFPoolDataset, REMedGenHPFCollator
from src.data.datasets import limits, SequencesGenerator, EvalDataset, EvalCollator
from src.data.baseline_datasets import HierarchicalGenHPFDataset, GenHPFEvalCollator
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos, load_config_with_env


# lt.seed_everything(24, workers=True)
parser = argparse.ArgumentParser(description='MLM pretraining command line interface')
parser.add_argument('--config-path', type=str, required=True)
args = parser.parse_args()





config = load_config_with_env(args.config_path)



ConfigClass, ModelClass = get_config_and_model_cls(config['backbone_name'], mode='eval')

learning_rate = False
weight_decay = False
pooling = False
use_numeric = True    
dropout = False



if config['backbone_name'] in ['roberta','longformer','big_bird','roformer','modernbert']:
    ConfigClass, ModelClass = get_config_and_model_cls(config['backbone_name'], mode='eval')

    seq_gen = SequencesGenerator(tokenizer_path=config['seq_gen']['tokenizer_path'],
                                    chunk_length=config['seq_gen']['seq_length'],
                                    overlap=config['seq_gen']['overlap'])
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
    test_dataset = EvalDataset(dataset_path=config['dataset']['data_path'],
                                data_idx_path=config['dataset']['data_idx_path'],
                                seq_gen=seq_gen,
                                seq_length=config['seq_gen']['seq_length'],
                                limits_dict=limits,
                                main_window=config['main_window'],
                                task=config['task'],
                                use_time=True,
                                use_numeric=use_numeric,
                                split='test')
    
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
    test_dataloader = DataLoader(dataset=test_dataset,
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



elif config['backbone_name'] == 'descemb':    
    train_dataset = DescEmbDataset(dataset_path=config['dataset']['data_path'],
                                data_idx_path=config['dataset']['data_idx_path'],
                                task=config['task'],
                                main_window=config['main_window'],
                                max_word_len=12,
                                max_events=511,
                                split='train') 
    val_dataset = DescEmbDataset(dataset_path=config['dataset']['data_path'],
                                data_idx_path=config['dataset']['data_idx_path'],
                                task=config['task'],
                                main_window=config['main_window'],
                                max_word_len=12,
                                max_events=511,
                                split='val')
    test_dataset = DescEmbDataset(dataset_path=config['dataset']['data_path'],
                            data_idx_path=config['dataset']['data_idx_path'],
                            task=config['task'],
                            main_window=config['main_window'],
                            max_word_len=12,
                            max_events=511,
                            split='test') 
    collate_fn = DescEmbCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
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
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=8,
                                shuffle=False,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)
    cfg = SimpleNamespace(bert_model_name="google/bert_uncased_L-2_H-128_A-2",
                            pred_embed_dim=128,
                            pred_hidden_dim=256,     
                            max_event_len=511,       
                            rnn_layer=1,
                            init_bert_random=False,  
                            task="binary")
    model = DescEmbEvalModel(config=cfg,
                            lr=learning_rate,
                            max_epochs=75,
                            dropout=dropout,
                            freeze=args.freeze)

elif config['backbone_name'] in ['genhpf']:
    train_dataset = HierarchicalGenHPFDataset(dataset_path=config['dataset']['data_path'],
                                                data_idx_path=config['dataset']['data_idx_path'],
                                                seq_field=config['main_window'],
                                                label_field=config['task'],
                                                split='train',
                                                tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                                max_events=511,
                                                max_tokens=64)
    val_dataset = HierarchicalGenHPFDataset(dataset_path=config['dataset']['data_path'],
                                            data_idx_path=config['dataset']['data_idx_path'],
                                            seq_field=config['main_window'],
                                            label_field=config['task'],
                                            split='val',
                                            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                            max_events=511,
                                            max_tokens=64)
    test_dataset = HierarchicalGenHPFDataset(dataset_path=config['dataset']['data_path'],
                                            data_idx_path=config['dataset']['data_idx_path'],
                                            seq_field=config['main_window'],
                                            label_field=config['task'],
                                            split='test',
                                            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                            max_events=511,
                                            max_tokens=64)
    collate_fn = GenHPFEvalCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
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
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=8,
                                shuffle=False,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)

    encoder = GenHPFEncoder(vocab_size=train_dataset.tokenizer.vocab_size,
                            pad_token_id=train_dataset.tokenizer.pad_token_id,
                            encoder_embed_dim=128,
                            encoder_layers=2,
                            encoder_ffn_embed_dim=512,
                            encoder_attention_heads=4,
                            agg_embed_dim=128,
                            agg_layers=4,
                            agg_ffn_embed_dim=512,
                            agg_attention_heads=4,
                            dropout=dropout,
                            max_token_len=64,   
                            max_events=511,
                            encoder_only=False,
                            ckpt_path=config['ckpt_path'])
    model = GenHPFDownstreamModule(encoder=encoder,
                                    lr=learning_rate,
                                    max_epochs=75,
                                    num_outputs=1,
                                    pos_weight=1.0)


    
elif config['backbone_name'] == 'remed':
    train_dataset = REMedGenHPFPoolDataset(hf_path=config['dataset']['data_path'],
                                           data_idx_path=config['dataset']['data_idx_path'],
                                           seq_field=config['main_window'],
                                           time_field=config['time_field'],
                                           time_diff_field=config['time_diff_field'],
                                           label_field=config['task'],
                                           split='train',
                                           tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                           seq_len=512,
                                           max_tokens=64)
    val_dataset = REMedGenHPFPoolDataset(hf_path=config['dataset']['data_path'],
                                         data_idx_path=config['dataset']['data_idx_path'],
                                         seq_field=config['main_window'],
                                         time_field=config['time_field'],
                                         time_diff_field=config['time_diff_field'],
                                         label_field=config['task'],
                                         split='val',
                                         tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                         seq_len=512,
                                         max_tokens=64)
    test_dataset = REMedGenHPFPoolDataset(hf_path=config['dataset']['data_path'],
                                          data_idx_path=config['dataset']['data_idx_path'],
                                          seq_field=config['main_window'],
                                          time_field=config['time_field'],
                                          time_diff_field=config['time_diff_field'],
                                          label_field=config['task'],
                                          split='test',
                                          tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                          seq_len=512,
                                          max_tokens=64)
    collate_fn = REMedGenHPFCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
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
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config['dataloader']['batch_size'],
                                 num_workers=8,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 pin_memory=True,
                                 persistent_workers=True,
                                 prefetch_factor=4)
    encoder = GenHPFEncoder(vocab_size=train_dataset.tokenizer.vocab_size,
                            pad_token_id=train_dataset.tokenizer.pad_token_id,
                            encoder_embed_dim=128,
                            encoder_layers=2,
                            encoder_ffn_embed_dim=512,
                            encoder_attention_heads=4,
                            agg_embed_dim=128,
                            agg_layers=4,
                            agg_ffn_embed_dim=512,
                            agg_attention_heads=4,
                            dropout=dropout,
                            max_token_len=64,   
                            max_events=512,
                            encoder_only=True,
                            ckpt_path=config['ckpt_path'])
    remed_genhpf = REMedWithGenHPF(genhpf_encoder=encoder,
                                    pred_dim=512,
                                    num_classes=1,
                                    pred_time=config['pred_time'],
                                    max_retrieve_len=128,
                                    n_heads=8,
                                    n_layers=2,
                                    dropout=dropout,
                                    freeze_encoder=True)
    model = REMedLightningModule(model=remed_genhpf,
                                    lr=learning_rate,
                                    max_epochs=75,
                                    pos_weight=1.0,
                                    freeze_encoder=True,
                                    use_warmup=True,
                                    warmup_steps=500,
                                    num_classes=1)   


wandb.login(key=config['logger']['wandb_api_key'])

wandb_logger = WandbLogger(project='MedEHR_Final',
                        entity=config['logger']['entity'],
                        save_dir=config['logger']['log_dir'],
                        version=f"{config['backbone_name']}_{config['seq_gen']['seq_length']}_{config['job_id']}_{config['task']}_{config['main_window']}_{config['version']}",
                        name=f"{config['backbone_name']}_{config['seq_gen']['seq_length']}_{config['job_id']}_{config['task']}_{config['main_window']}_{config['version']}",
                        tags=[config['version'],config['task'],config['backbone_name'],'hparams_opt']) 

ckpt_dir = os.path.join(wandb_logger.experiment.dir, 'ckpt')
os.makedirs(ckpt_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                    monitor='val_loss', 
                                    mode='min',
                                    every_n_epochs=1,
                                    save_top_k=1,)

early_stop = EarlyStopping(monitor='val_loss',
                            min_delta=0.001,
                            mode='min', 
                            patience=5)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    
    precision = "bf16-mixed" if major >= 8 else "16-mixed"
else:
    precision = '32-true'
trainer = lt.Trainer(accelerator='auto', 
                    devices='auto',
                    strategy='auto',
                    logger=wandb_logger, 
                    log_every_n_steps=1,
                    num_sanity_val_steps=0,
                    max_epochs=75,
                    precision=precision, 
                    callbacks=[early_stop,lr_monitor,checkpoint_callback]
                    )
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')


