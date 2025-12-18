
import math
import torch
import torch.nn as nn
import lightning.pytorch as lt

from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig
from .utils import log_bootstrap_ci_text_percentile
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from transformers.models.roformer.modeling_roformer import RoFormerConfig, RoFormerEncoder


#########################################################
# DescEmb model
#########################################################

class BertEventEncoder(nn.Module):
    def __init__(
        self,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        pred_embed_dim: int = 128,
        init_bert_random: bool = False,
    ):
        super().__init__()

        if init_bert_random:
            config = AutoConfig.from_pretrained(bert_model_name)
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = AutoModel.from_pretrained(bert_model_name)

        hidden_size = self.bert.config.hidden_size
        self.post_encode_proj = nn.Linear(hidden_size, pred_embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, S, W = input_ids.shape

        flat_ids = input_ids.view(B * S, W)
        flat_mask = attention_mask.view(B * S, W)

        outputs = self.bert(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :] 

        event_emb = self.post_encode_proj(cls_emb)  
        event_emb = event_emb.view(B, S, -1)        
        return event_emb
    

class GRUEventHead(nn.Module):

    def __init__(self, pred_embed_dim: int=128, 
                 pred_hidden_dim: int=256, 
                 max_event_len: int =511,
                 n_layers: int = 1, 
                 dropout: float = 0.3, 
                 task: str = "binary"):
        super().__init__()
        self.pred_embed_dim = pred_embed_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.n_layers = n_layers
        self.max_event_len = max_event_len
        self.task = task

        self.model = nn.GRU(
            input_size=self.pred_embed_dim,
            hidden_size=self.pred_hidden_dim,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
            num_layers=self.n_layers,
        )

        out_dim = 18 if task == "diagnosis" else 1
        self.final_proj = nn.Linear(self.pred_hidden_dim, out_dim)

    def pack_pad_seq(self, x: torch.Tensor, lengths: torch.Tensor):
        lengths = lengths.view(-1).cpu()
        lengths[lengths > self.max_event_len] = self.max_event_len

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.model(packed)
        output_seq, output_len = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0
        )
        return output_seq, output_len

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
       
        self.model.flatten_parameters()

        output_seq, _ = self.pack_pad_seq(x, seq_len) 
        i = range(x.size(0))
        last_hidden = output_seq[i, -1, :]             

        logits = self.final_proj(last_hidden)          
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)                
        return logits
    

class DescEmbEvalModel(lt.LightningModule):
    def __init__(
        self,
        config,
        lr: float = 2e-5,
        wd: float = 0.0,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        bert_model_name = getattr(config, "bert_model_name", "google/bert_uncased_L-2_H-128_A-2")
        pred_embed_dim = getattr(config, "pred_embed_dim", 128)
        pred_hidden_dim = getattr(config, "pred_hidden_dim", 256)
        max_event_len = getattr(config, "max_event_len", 511)
        task = getattr(config, "task", "binary")

        self.encoder = BertEventEncoder(
            bert_model_name=bert_model_name,
            pred_embed_dim=pred_embed_dim,
            init_bert_random=getattr(config, "init_bert_random", False),
        )
        self.classifier = GRUEventHead(
            pred_embed_dim=pred_embed_dim,
            pred_hidden_dim=pred_hidden_dim,
            max_event_len=max_event_len,
            n_layers=getattr(config, "rnn_layer", 1),
            dropout=dropout,
            task=task,
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.classifier.parameters():
                p.requires_grad = True

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.criterion = nn.BCEWithLogitsLoss()

        self.train_step_preds = []
        self.train_step_label = []
        self.val_step_preds = []
        self.val_step_label = []
        self.test_step_preds = []
        self.test_step_label = []

        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

    def forward(self, input_ids, attention_mask, seq_len=None, labels=None):
        event_emb = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  

        if seq_len is None:
            event_mask = attention_mask.any(dim=-1)     
            seq_len = event_mask.sum(dim=-1)             
        logits = self.classifier(event_emb, seq_len)       
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            seq_len=batch["seq_len"],
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)

        pos_score = torch.sigmoid(logits)

        self.train_step_label.append(y)
        self.train_step_preds.append(pos_score)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_step_label) == 0:
            return
        y = torch.cat(self.train_step_label)
        pos_score = torch.cat(self.train_step_preds)

        auroc = self.train_auroc(pos_score, y.long())
        auprc = self.train_auprc(pos_score, y.long())

        self.log("train_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)

        self.train_step_label.clear()
        self.train_step_preds.clear()

    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            seq_len=batch["seq_len"],
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.val_step_label.append(y)
        self.val_step_preds.append(pos_score)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        if len(self.val_step_label) == 0:
            return
        y = torch.cat(self.val_step_label)
        pos_score = torch.cat(self.val_step_preds)

        auroc = self.val_auroc(pos_score, y.long())
        auprc = self.val_auprc(pos_score, y.long())

        self.log("val_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log("val_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)

        self.val_step_label.clear()
        self.val_step_preds.clear()

    def test_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            seq_len=batch["seq_len"],
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.test_step_label.append(y)
        self.test_step_preds.append(pos_score)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_test_epoch_end(self, *args, **kwargs) -> None:
        if len(self.test_step_label) == 0:
            return
        y = torch.cat(self.test_step_label)
        pos_score = torch.cat(self.test_step_preds)

        auroc = self.test_auroc(pos_score, y.long())
        auprc = self.test_auprc(pos_score, y.long())

        self.log("test_auroc", auroc, on_epoch=True, logger=True)
        self.log("test_auprc", auprc, on_epoch=True, logger=True)

        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=y,
            y_score=pos_score,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3,
        )
        self.test_step_label.clear()
        self.test_step_preds.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer,
        #     eta_min=0,
        #     T_max=self.max_epochs,
        # )
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler}
    

#########################################################
# GenHPF model
#########################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)                      
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)                              
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    
class GenHPFEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        encoder_embed_dim: int = 128,
        encoder_layers: int = 2,
        encoder_ffn_embed_dim: int = 512,
        encoder_attention_heads: int = 4,
        agg_embed_dim: int = 128,
        agg_layers: int = 4,
        agg_ffn_embed_dim: int = 512,
        agg_attention_heads: int = 4,
        dropout: float = 0.1,
        max_token_len: int = 128,
        max_events: int = 511,
        encoder_only: bool = False 
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_only = encoder_only
        self.pad_token_id = pad_token_id
        self.encoder_embed_dim = encoder_embed_dim
        self.agg_embed_dim = agg_embed_dim

        self.word_embeddings = nn.Embedding(vocab_size, 
                                            encoder_embed_dim, 
                                            padding_idx=pad_token_id)

        
        self.token_pos_encoding = PositionalEncoding(d_model=encoder_embed_dim, 
                                                     dropout=dropout, 
                                                     max_len=max_token_len)

        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_embed_dim,
            nhead=encoder_attention_heads,
            dim_feedforward=encoder_ffn_embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers
        )

        
        self.post_encode_proj = nn.Linear(encoder_embed_dim, agg_embed_dim)

        
        self.event_pos_encoding = PositionalEncoding(
            d_model=agg_embed_dim, dropout=dropout, max_len=max_events
        )

        agg_layer = nn.TransformerEncoderLayer(
            d_model=agg_embed_dim,
            nhead=agg_attention_heads,
            dim_feedforward=agg_ffn_embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.event_aggregator = nn.TransformerEncoder(
            agg_layer, num_layers=agg_layers
        )

        self.event_layer_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-12)
        self.agg_layer_norm = nn.LayerNorm(agg_embed_dim, eps=1e-12)

    def forward(
        self,
        input_ids: torch.Tensor,         
        padding_mask: Optional[torch.Tensor] = None,  
#         encoder_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, W = input_ids.shape

        flat_ids = input_ids.view(B * S, W)


        x_tok = self.word_embeddings(flat_ids)
        x_tok = self.token_pos_encoding(x_tok)
        x_tok = self.event_layer_norm(x_tok)

        token_pad_mask = flat_ids.eq(self.pad_token_id) 


        x_tok = self.event_encoder(
            x_tok, src_key_padding_mask=token_pad_mask
        ) 

        if token_pad_mask.any():
            x_tok = x_tok.masked_fill(token_pad_mask.unsqueeze(-1), 0.0)
            lengths = (~token_pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        else:
            lengths = torch.full(
                (B * S, 1), W, device=x_tok.device, dtype=torch.long
            )

        event_emb = x_tok.sum(dim=1) / lengths 

    
        event_emb = self.post_encode_proj(event_emb) 
        event_emb = event_emb.view(B, S, -1)        

        if padding_mask is None:
            padding_mask = input_ids.eq(self.pad_token_id).all(dim=2)  


        event_emb = self.event_pos_encoding(event_emb)
        event_emb = self.agg_layer_norm(event_emb)

        if self.encoder_only:
            return event_emb, padding_mask

        x = self.event_aggregator(
            event_emb, src_key_padding_mask=padding_mask
        ) 

        return x, padding_mask
    

class GenHPFSimCLRModel(nn.Module):
    def __init__(self, encoder: GenHPFEncoder, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder

        D = encoder.agg_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, proj_dim),
        )

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x, pad_mask = self.encoder(input_ids, padding_mask=padding_mask)

        if pad_mask is not None and pad_mask.any():
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        lengths = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / lengths

        z = self.proj(pooled)
        return z
    


class GenHPFClassifier(nn.Module):

    def __init__(
        self,
        encoder: GenHPFEncoder,
        num_outputs: int = 1,  
    ):
        super().__init__()
        self.encoder = encoder
        self.num_outputs = num_outputs

        self.classifier = nn.Linear(encoder.agg_embed_dim, num_outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        x, pad_mask = self.encoder(input_ids, padding_mask=padding_mask)


        if pad_mask is not None and pad_mask.any():
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)


        lengths = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / lengths

        logits = self.classifier(pooled)  
        if self.num_outputs == 1:
            logits = logits.squeeze(-1) 
        return logits
    

class GenHPFDownstreamModule(lt.LightningModule):
    def __init__(
        self,
        encoder: GenHPFEncoder,
        num_outputs: int = 1,
        lr: float = 2e-5,
        wd: float = 1e-3,
        max_epochs: int = 100,
        pos_weight: float = 1.0,  
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.model = GenHPFClassifier(
            encoder=encoder,
            num_outputs=num_outputs,
        )

        # loss
        if num_outputs == 1:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        if num_outputs == 1:
            self.train_auroc = BinaryAUROC()
            self.train_auprc = BinaryAveragePrecision()
            self.val_auroc = BinaryAUROC()
            self.val_auprc = BinaryAveragePrecision()
            self.test_auroc = BinaryAUROC()
            self.test_auprc = BinaryAveragePrecision()

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.train_step_preds = []
        self.train_step_label = []
        self.val_step_preds = []
        self.val_step_label = []
        self.test_step_preds = []
        self.test_step_label = []

    def forward(self, input_ids, padding_mask):
        logits = self.model(input_ids=input_ids, padding_mask=padding_mask)
        return logits


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]       
        padding_mask = batch["padding_mask"] 
        y = batch["label"].float().view(-1) 

        logits = self.forward(input_ids, padding_mask) 

        if self.hparams.num_outputs == 1:
            loss = self.criterion(logits, y)
            pos_score = torch.sigmoid(logits)
            self.train_step_label.append(y.detach())
            self.train_step_preds.append(pos_score.detach())
        else:
            y_long = y.long()
            loss = self.criterion(logits, y_long)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        if self.hparams.num_outputs != 1:
            return

        y = torch.cat(self.train_step_label)
        pos_score = torch.cat(self.train_step_preds)

        auroc = self.train_auroc(pos_score, y.long())
        auprc = self.train_auprc(pos_score, y.long())

        self.log("train_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)

        self.train_step_label.clear()
        self.train_step_preds.clear()

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]
        y = batch["label"].float().view(-1)

        logits = self.forward(input_ids, padding_mask)

        if self.hparams.num_outputs == 1:
            loss = self.criterion(logits, y)
            pos_score = torch.sigmoid(logits)
            self.val_step_label.append(y.detach())
            self.val_step_preds.append(pos_score.detach())
        else:
            y_long = y.long()
            loss = self.criterion(logits, y_long)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        if self.hparams.num_outputs != 1:
            return

        y = torch.cat(self.val_step_label)
        pos_score = torch.cat(self.val_step_preds)

        auroc = self.val_auroc(pos_score, y.long())
        auprc = self.val_auprc(pos_score, y.long())

        self.log("val_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log("val_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)

        self.val_step_label.clear()
        self.val_step_preds.clear()


    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]
        y = batch["label"].float().view(-1)

        logits = self.forward(input_ids, padding_mask)

        if self.hparams.num_outputs == 1:
            loss = self.criterion(logits, y)
            pos_score = torch.sigmoid(logits)
            self.test_step_label.append(y.detach())
            self.test_step_preds.append(pos_score.detach())
        else:
            y_long = y.long()
            loss = self.criterion(logits, y_long)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        if self.hparams.num_outputs != 1:
            return

        y = torch.cat(self.test_step_label)
        pos_score = torch.cat(self.test_step_preds)

        auroc = self.test_auroc(pos_score, y.long())
        auprc = self.test_auprc(pos_score, y.long())

        self.log("test_auroc", auroc, on_epoch=True, logger=True)
        self.log("test_auprc", auprc, on_epoch=True, logger=True)
        
        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=y,
            y_score=pos_score,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3,
        )

        self.test_step_label.clear()
        self.test_step_preds.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         scheduler = CosineAnnealingLR(
#             optimizer=optimizer,
#             eta_min=0.0,
#             T_max=self.max_epochs,
#         )
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler}
    
#########################################################
# REMed model
#########################################################


class Retriever(nn.Module):
    def __init__(self, pred_dim: int):
        super().__init__()
        in_dim = pred_dim + 1  # repr + time
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim // 8),
            nn.LayerNorm(in_dim // 8),
            nn.ReLU(),
            nn.Linear(in_dim // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, reprs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        times = times.unsqueeze(-1).type(reprs.dtype)
        return self.model(torch.cat([reprs, times], dim=-1)).squeeze(-1)  # (B, L)


class ReprTimeEnc(nn.Module):
    def __init__(self, pred_dim: int, dropout: float, pred_time: int):
        super().__init__()
        self.pred_time = pred_time
        div_term = torch.exp(torch.arange(0, pred_dim, 2) * (-math.log(10000.0) / pred_dim))
        self.register_buffer("div_term", div_term)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, times: torch.Tensor):

        times = self.pred_time * 60 - times
        src_pad_mask = x.eq(0).all(dim=-1)  # (B, L)

        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(times.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] = torch.cos(times.unsqueeze(-1) * self.div_term)
        x = self.dropout(x + pe)
        return x, src_pad_mask


class Predictor(nn.Module):
    def __init__(self, 
                 pred_dim: int=512, 
                 dropout: float=0.2, 
                 pred_time: int=48, 
                 n_layers: int=2, 
                 n_heads: int=8, 
                 max_len: int=128):
        super().__init__()
        self.time_enc = ReprTimeEnc(pred_dim, dropout, pred_time)
        config = RoFormerConfig(
            hidden_size=pred_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=pred_dim * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_len,
        )
        self.model = RoFormerEncoder(config)

    def forward(self, reprs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        x, src_pad_mask = self.time_enc(reprs, times)

        mask = src_pad_mask * torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
        mask = mask.unsqueeze(-1).unsqueeze(1)  
        return self.model(x, attention_mask=mask)["last_hidden_state"] 


class PredOutPutLayer(nn.Module):
    def __init__(self, pred_dim: int, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.final_proj = nn.Linear(pred_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.ne(0).any(dim=-1).unsqueeze(-1)  
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.final_proj(pooled)  
        return logits



class REMed(nn.Module):
    def __init__(
        self,
        pred_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.2,
        max_retrieve_len: int = 128,
        pred_time: int = 48,
        num_classes: int = 1,
    ):
        super().__init__()
        self.pred_dim = pred_dim
        self.max_retrieve_len = max_retrieve_len

        self.predictor = Predictor(pred_dim, dropout, pred_time, n_layers, n_heads, max_retrieve_len)
        self.emb2out_model = PredOutPutLayer(pred_dim, num_classes=num_classes)
        self.retriever = Retriever(pred_dim)

        self.register_buffer("random_token_emb", torch.randn(pred_dim))
        self.set_mode("retriever")

    def set_mode(self, mode: str):
        self.mode = mode
        if mode == "retriever":
            self.requires_grad_(False)
            self.retriever.requires_grad_(True)
        elif mode == "predictor":
            self.requires_grad_(True)
            self.retriever.requires_grad_(False)

    def forward(self, reprs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:

        reprs = nn.functional.pad(reprs, (0, 0, 0, self.max_retrieve_len))
        times = nn.functional.pad(times, (0, self.max_retrieve_len))


        times = torch.where(reprs.eq(0).all(dim=-1), torch.tensor(1e10, device=times.device, dtype=times.dtype), times)

        sim = self.retriever(reprs, times)  # (B, L+Kpad)

        _sim = torch.where(reprs.eq(0).all(dim=-1), torch.zeros_like(sim), sim)
        topk_values, topk_indices = torch.topk(_sim, self.max_retrieve_len, dim=1)

        topk = torch.gather(reprs, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.pred_dim))
        topk_times = torch.gather(times, 1, topk_indices)
        B, K, E = topk.shape


        topk_times, order = topk_times.sort(dim=1)
        topk = topk.gather(1, order.unsqueeze(-1).repeat(1, 1, E))
        topk_values = topk_values.gather(1, order)

        def _retriever_path():
            _topk_values = topk_values.reshape(B * K, 1)
            _topk = topk.reshape(B * K, 1, -1)
            _topk_times = topk_times.reshape(B * K, 1)

            zero_idcs = _topk.eq(0).all(dim=-1)
            _topk_times = torch.where(zero_idcs, torch.zeros_like(_topk_times), _topk_times)
            _topk_values = torch.where(zero_idcs, torch.zeros_like(_topk_values), _topk_values)


            _topk = torch.where(zero_idcs.unsqueeze(-1), self.random_token_emb.expand(B * K, 1, E), _topk)
            _topk_values = _topk_values + 1e-10
            _topk_values = (_topk_values.reshape(B, K) / _topk_values.reshape(B, K).sum(dim=1, keepdim=True)).reshape(B * K)

            enc = self.predictor(_topk, _topk_times)
            logits = self.emb2out_model(enc) 


            logits = torch.sum((_topk_values.unsqueeze(-1) * logits).reshape(B, K, -1), dim=1)
            return logits

        def _predictor_path():
            # ensure first token isn't all-zeros
            topk[:, 0, :] = torch.where(
                topk[:, 0, :].sum(dim=-1, keepdim=True) == 0,
                self.random_token_emb.expand(B, E),
                topk[:, 0, :],
            )
            enc = self.predictor(topk, topk_times)
            logits = self.emb2out_model(enc) 
            return logits

        if self.training:
            if self.training and self.mode == "retriever":
                logits = _retriever_path()
            else:
                logits = _predictor_path()
        else:
            logits = _predictor_path()

        return logits



class REMedWithGenHPF(nn.Module):
    def __init__(
        self,
        genhpf_encoder: nn.Module,
        pred_dim: int = 512,
        num_classes: int = 1,
        pred_time: int = 48,
        max_retrieve_len: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = genhpf_encoder
        self.freeze_encoder = freeze_encoder
        self.num_classes = num_classes

        # encoder output dim -> pred_dim (fixes agg_embed_dim=128 vs pred_dim=512 mismatch)
        enc_out_dim = getattr(genhpf_encoder, "agg_embed_dim", None)
        if enc_out_dim is None:
            raise ValueError("genhpf_encoder must expose agg_embed_dim (encoder output dim).")

        self.enc_to_pred = nn.Identity() if enc_out_dim == pred_dim else nn.Linear(enc_out_dim, pred_dim)

        self.remed = REMed(
            pred_dim=pred_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_retrieve_len=max_retrieve_len,
            pred_time=pred_time,
            num_classes=num_classes,
        )

        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()  # no dropout in encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        times: torch.Tensor,
        return_probs: bool = False,
    ):
        # keep encoder deterministic if frozen (Lightning may flip modules to train())
        if self.freeze_encoder:
            self.encoder.eval()

        reprs, pad_mask = self.encoder(input_ids, padding_mask=padding_mask)  # (B,S,enc_out_dim)
        reprs = self.enc_to_pred(reprs)  # (B,S,pred_dim)

        if pad_mask is not None and pad_mask.any():
            reprs = reprs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        logits = self.remed(reprs, times.float())  # (B,C)

        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # (B,)
            return (logits, torch.sigmoid(logits)) if return_probs else logits

        return (logits, torch.softmax(logits, dim=-1)) if return_probs else logits
    

class REMedLightningModule(lt.LightningModule):

    def __init__(
        self,
        model,  # REMedWithGenHPF, exposes .remed.set_mode(...)
        lr: float = 1e-5,
        max_epochs: int = 100,
        pos_weight: float = 1.0,
        freeze_encoder: bool = True,
        use_warmup: bool = False,
        warmup_steps: int = 500,
        num_classes: int = 1,   # <-- ADD BACK
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.automatic_optimization = False

        self.num_classes = num_classes

        if num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
            )
            self.train_auroc = BinaryAUROC()
            self.train_auprc = BinaryAveragePrecision()
            self.val_auroc = BinaryAUROC()
            self.val_auprc = BinaryAveragePrecision()
            self.test_auroc = BinaryAUROC()
            self.test_auprc = BinaryAveragePrecision()
        else:
            self.criterion = nn.CrossEntropyLoss()
            # keep metrics out for multiclass unless you explicitly want them

        self.train_step_preds, self.train_step_label = [], []
        self.val_step_preds, self.val_step_label = [], []
        self.test_step_preds, self.test_step_label = [], []

        self._freeze_encoder = freeze_encoder

        # IMPORTANT: don't leave the model stuck in "predictor" mode from debug prints
        self.model.remed.set_mode("retriever")

    def on_train_batch_start(self, batch, batch_idx):
        if self._freeze_encoder and hasattr(self.model, "encoder"):
            self.model.encoder.eval()

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            padding_mask=batch["padding_mask"],
            times=batch["times"],
        )

    def _step_once(self, batch, y, mode: str):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        self.model.remed.set_mode(mode)

        logits = self.forward(batch)

        if self.num_classes == 1:
            logits = logits.view(-1)
            loss = self.criterion(logits, y)
        else:
            # y expected as class indices shape (B,)
            logits = logits.view(y.size(0), -1)
            loss = self.criterion(logits, y.long())

        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt.step()
        if sch is not None:
            sch.step()

        return loss, logits

    def training_step(self, batch, batch_idx):
        if self.num_classes == 1:
            y = batch["label"].float().view(-1)
        else:
            y = batch["label"].view(-1)

        loss1, _ = self._step_once(batch, y, mode="retriever")
        loss2, logits2 = self._step_once(batch, y, mode="predictor")

        loss = 0.5 * (loss1 + loss2)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.num_classes == 1:
            probs = torch.sigmoid(logits2.detach().view(-1))
            self.train_step_label.append(y.detach())
            self.train_step_preds.append(probs)

        return loss

    def on_train_epoch_end(self):
        if self.num_classes != 1:
            return
        y = torch.cat(self.train_step_label).long()
        p = torch.cat(self.train_step_preds)
        self.log("train_auroc", self.train_auroc(p, y), prog_bar=True)
        self.log("train_auprc", self.train_auprc(p, y), prog_bar=True)
        self.train_step_label.clear()
        self.train_step_preds.clear()

    def validation_step(self, batch, batch_idx):
        self.model.remed.set_mode("predictor")

        if self.num_classes == 1:
            y = batch["label"].float().view(-1)
            logits = self.forward(batch).view(-1)
            loss = self.criterion(logits, y)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            probs = torch.sigmoid(logits.detach())
            self.val_step_label.append(y.detach())
            self.val_step_preds.append(probs)
            return loss
        else:
            y = batch["label"].view(-1).long()
            logits = self.forward(batch).view(y.size(0), -1)
            loss = self.criterion(logits, y)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            return loss

    def on_validation_epoch_end(self):
        if self.num_classes != 1:
            return
        y = torch.cat(self.val_step_label).long()
        p = torch.cat(self.val_step_preds)
        self.log("val_auroc", self.val_auroc(p, y), prog_bar=True)
        self.log("val_auprc", self.val_auprc(p, y), prog_bar=True)
        self.val_step_label.clear()
        self.val_step_preds.clear()

    def test_step(self, batch, batch_idx):
        self.model.remed.set_mode("predictor")

        if self.num_classes == 1:
            y = batch["label"].float().view(-1)
            logits = self.forward(batch).view(-1)
            loss = self.criterion(logits, y)
            self.log("test_loss", loss, prog_bar=True, on_epoch=True)
            probs = torch.sigmoid(logits.detach())
            self.test_step_label.append(y.detach())
            self.test_step_preds.append(probs)
            return loss
        else:
            y = batch["label"].view(-1).long()
            logits = self.forward(batch).view(y.size(0), -1)
            loss = self.criterion(logits, y)
            self.log("test_loss", loss, prog_bar=True, on_epoch=True)
            return loss

    def on_test_epoch_end(self):
        if self.num_classes != 1:
            return
        y = torch.cat(self.test_step_label).long()
        p = torch.cat(self.test_step_preds)
        self.log("test_auroc", self.test_auroc(p, y))
        self.log("test_auprc", self.test_auprc(p, y))
        self.test_step_label.clear()
        self.test_step_preds.clear()

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

        if self.hparams.use_warmup:
            sch = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1 / 100, end_factor=1.0, total_iters=self.hparams.warmup_steps
            )
        else:
            sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1.0, total_iters=1)
        return {"optimizer": opt, "lr_scheduler": sch}
    

