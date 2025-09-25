
import os
import torch


import lightning as lt
import torch.nn as nn

from transformers import RoFormerForMaskedLM
from torchmetrics.classification import Accuracy




class EHREmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = 0,
        type_vocab_size: int = 28,
        visit_vocab_size: int = 102,
        stage_vocab_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tok_emb   = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.type_emb  = nn.Embedding(type_vocab_size,  embedding_size, padding_idx=pad_token_id)
        self.visit_emb = nn.Embedding(visit_vocab_size, embedding_size, padding_idx=pad_token_id)
        self.stage_emb = nn.Embedding(stage_vocab_size, embedding_size, padding_idx=pad_token_id)
        self.norm      = nn.LayerNorm(embedding_size)
        self.drop      = nn.Dropout(dropout)

    def encode(self, input_ids, type_ids, visit_ids, stage_ids):
        x = self.tok_emb(input_ids.long())
        x = x + self.type_emb(type_ids.long())
        x = x + self.visit_emb(visit_ids.long())
        x = x + self.stage_emb(stage_ids.long())
        return self.drop(self.norm(x))


    def forward(self, input_ids=None, 
                token_type_ids=None, 
                inputs_embeds=None,
                **kwargs):
        if inputs_embeds is not None:
            return inputs_embeds
        return self.drop(self.norm(self.tok_emb(input_ids.long())))
    



class MLMPretraining(lt.LightningModule):
    """
    One-class trainer:
      - builds RoFormerForMaskedLM
      - swaps in EHREmbeddings
      - ties decoder to token embedding
      - trains with MLM labels (ignore_index = -100)
    Batch contract (keys): input_ids, attention_mask, type_ids, visit_ids, stage_ids, labels
    """
    def __init__(
        self,
        config,
        lr: float = 2e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.top_1_train = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=1,ignore_index=-100)
        self.top_3_train = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=3,ignore_index=-100)
        self.top_5_train = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=5,ignore_index=-100)

        self.top_1_val = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=1,ignore_index=-100)
        self.top_3_val = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=3,ignore_index=-100)
        self.top_5_val = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=5,ignore_index=-100)

        self.backbone = RoFormerForMaskedLM(config)


        ehr_emb = EHREmbeddings(
            vocab_size=config.vocab_size, 
            embedding_size=config.embedding_size, 
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size, 
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size, 
            dropout=dropout)
        
        self.backbone.roformer.embeddings = ehr_emb
        self.backbone.cls.predictions.decoder.weight = self.backbone.roformer.embeddings.tok_emb.weight

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs



    def forward(self, 
                input_ids, 
                attention_mask, 
                type_ids, 
                visit_ids, 
                stage_ids, 
                labels=None):
        
        inputs_embeds = self.backbone.roformer.embeddings.encode(
        input_ids=input_ids,
        type_ids=type_ids,
        visit_ids=visit_ids,
        stage_ids=stage_ids)
        return self.backbone(inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask,
                             labels=labels)


    def training_step(self, batch, batch_idx):
        out = self.forward(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            type_ids=batch["type_ids"],
                            visit_ids=batch["visit_ids"],
                            stage_ids=batch["stage_ids"],
                            labels=batch["labels"])
        
        loss = out.loss
        preds  = out.logits.view(-1, out.logits.size(-1))
        target = batch["labels"].view(-1)

        top1 = self.top_1_train(preds, target)
        top3 = self.top_3_train(preds, target)
        top5 = self.top_5_train(preds, target)
        perplexity = torch.exp(loss)

        self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_top1", top1, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_top3", top3, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_top5", top5, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            labels=batch["labels"])
        
        loss = out.loss
        preds  = out.logits.view(-1, out.logits.size(-1))
        target = batch["labels"].view(-1)
        
        top1 = self.top_1_val(preds, target)
        top3 = self.top_3_val(preds, target)
        top5 = self.top_5_val(preds, target)
        perplexity = torch.exp(loss)

        self.log("val_perplexity", perplexity, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_top1", top1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_top3", top3, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_top5", top5, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):


        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=self.wd)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         eta_min=0,
                                                         T_max=self.max_epochs
                                                         )
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}