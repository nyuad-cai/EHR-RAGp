
import os
import torch


import lightning as lt
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from .utils import Time2Vec, log_bootstrap_ci_text_percentile
from torchmetrics.classification import Accuracy, BinaryAUROC, BinaryAveragePrecision




class EHREmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = 0,
        type_vocab_size: int = 28,
        visit_vocab_size: int = 102,
        stage_vocab_size: int = 5,
        dropout: float = 0.1,
        use_position_embeddings: bool = False,
        max_position_embeddings: int = 0,
        use_time: bool = False,
        time_in_features: int = 1,
        time_out_features: int = 16,
        use_numeric: bool = False,
        numeric_hidden_size: int = 16,   
    ):
        super().__init__()

        self.tok_emb   = nn.Embedding(vocab_size,       embedding_size, padding_idx=pad_token_id)
        self.type_emb  = nn.Embedding(type_vocab_size,  embedding_size, padding_idx=pad_token_id)
        self.visit_emb = nn.Embedding(visit_vocab_size, embedding_size, padding_idx=pad_token_id)
        self.stage_emb = nn.Embedding(stage_vocab_size, embedding_size, padding_idx=pad_token_id)

        
        self.use_position_embeddings = use_position_embeddings
        if use_position_embeddings:
            if max_position_embeddings <= 0:
                raise ValueError("max_position_embeddings must be > 0 when use_position_embeddings=True")
            self.pos_emb = nn.Embedding(max_position_embeddings, embedding_size)
        else:
            self.pos_emb = None

        
        self.use_time = use_time
        if use_time:
            self.time2vec = Time2Vec(
                in_features=time_in_features,
                out_features=time_out_features,
                periodic_activation=torch.sin,
            )
            self.time_proj = nn.Linear(time_out_features, embedding_size)
        else:
            self.time2vec = None
            self.time_proj = None

        
        self.use_numeric = use_numeric
        if use_numeric:
            self.numeric_hidden_size = numeric_hidden_size
            
            self.num_proj1 = nn.Linear(1, numeric_hidden_size)
            self.num_proj2 = nn.Linear(numeric_hidden_size, embedding_size)
            self.num_act = nn.GELU()

            
            self.null_numeric = nn.Parameter(torch.zeros(embedding_size))
            nn.init.normal_(self.null_numeric, mean=0.0, std=0.02)

            nn.init.xavier_uniform_(self.num_proj1.weight)
            nn.init.zeros_(self.num_proj1.bias)
            nn.init.xavier_uniform_(self.num_proj2.weight)
            nn.init.zeros_(self.num_proj2.bias)
        else:
            self.num_proj1 = None
            self.num_proj2 = None
            self.num_act = None
            self.null_numeric = None

        self.norm = nn.LayerNorm(embedding_size)
        self.drop = nn.Dropout(dropout)

    def encode(
        self,
        input_ids,
        type_ids,
        visit_ids,
        stage_ids,
        time_feats=None,          
        numeric_values=None,     
        numeric_mask=None,        
    ):
        
        x = self.tok_emb(input_ids.long())
        x = x + self.type_emb(type_ids.long())
        x = x + self.visit_emb(visit_ids.long())
        x = x + self.stage_emb(stage_ids.long())

        
        if self.pos_emb is not None:
            shape = input_ids.size()
            seqlen = shape[-1]

            position_ids = torch.arange(seqlen, device=input_ids.device)

            if input_ids.dim() == 2:        
                bsz = shape[0]
                position_ids = position_ids.unsqueeze(0).expand(bsz, seqlen)          
            elif input_ids.dim() == 3:        
                bsz, n = shape[0], shape[1]
                position_ids = position_ids.view(1, 1, seqlen).expand(bsz, n, seqlen) 
            else:
                raise ValueError(f"Unsupported input_ids.dim()={input_ids.dim()}")
            x = x + self.pos_emb(position_ids)

        
        if self.use_time:
            if time_feats is None:
                raise ValueError("time_feats must be provided when use_time=True")
            if time_feats.dim() == 2:
                time_feats = time_feats.unsqueeze(-1)
            elif time_feats.dim() != 3:
                raise ValueError(f"Unexpected time_feats.dim()={time_feats.dim()}, expected 2 or 3")
            t = self.time2vec(time_feats.float())   
            t = self.time_proj(t)                   
            x = x + t

        
        if self.use_numeric:
            if numeric_values is None or numeric_mask is None:
                raise ValueError("numeric_values and numeric_mask must be provided when use_numeric=True")

            
            v = numeric_values.float().unsqueeze(-1)       
            
            h = self.num_act(self.num_proj1(v))             
            num_emb = self.num_proj2(h)                     

            mask = numeric_mask.bool().unsqueeze(-1)        
            num_emb = torch.where(mask, num_emb, self.null_numeric.view(1, 1, -1))
            x = x + num_emb

        return self.drop(self.norm(x))

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            return inputs_embeds
        x = self.tok_emb(input_ids.long())
        return self.drop(self.norm(x))
    


class MLMPretraining(lt.LightningModule):
    def __init__(
        self,
        config,
        backbone,                      # e.g. ModernBertForMaskedLM, BertForMaskedLM, RoFormerForMaskedLM
        lr: float = 1e-6,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        self.top_1_train = Accuracy(
            task="multiclass",
            num_classes=config.vocab_size,
            top_k=1,
            ignore_index=-100,
        )
        self.top_1_val = Accuracy(
            task="multiclass",
            num_classes=config.vocab_size,
            top_k=1,
            ignore_index=-100,
        )

        # instantiate backbone
        self.backbone = backbone(config)

        # ---- RoPE vs non-RoPE auto-detection ----
        rope_model_types = {"modernbert", "roformer"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types

        # EHREmbeddings: no pos-emb for RoPE, local pos-emb for others
        self.ehr_embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout,
            use_position_embeddings=not is_rope,
            max_position_embeddings=(
                getattr(config, "max_position_embeddings", 0)
                if not is_rope
                else 0
            ),
        )

        # tie LM head to token embeddings
        lm_head = self.backbone.get_output_embeddings()
        lm_head.weight = self.ehr_embeddings.tok_emb.weight

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

    def forward(
        self,
        input_ids,
        attention_mask,
        type_ids,
        visit_ids,
        stage_ids,
        labels=None,
    ):
        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=input_ids,
            type_ids=type_ids,
            visit_ids=visit_ids,
            stage_ids=stage_ids,
        )

        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        out = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            labels=batch["labels"],
        )

        loss = out.loss
        preds  = out.logits.view(-1, out.logits.size(-1))
        target = batch["labels"].view(-1)

        top1 = self.top_1_train(preds, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_top1", top1,  prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            labels=batch["labels"],
        )

        loss = out.loss
        preds  = out.logits.view(-1, out.logits.size(-1))
        target = batch["labels"].view(-1)

        top1 = self.top_1_val(preds, target)

        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_top1", top1,  prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0,
            T_max=self.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    

class EvalModel(lt.LightningModule):
    def __init__(
        self,
        config,
        backnone,
        ckpt_path: str = None,
        lr: float = 2e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
        pooling: str = 'cls',
        use_numeric: bool = True,
        use_time: bool = True,
        optimizer: str = 'sgd', 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pooling = pooling
        self.backbone = backnone(config)
        self.optimizer = optimizer
        rope_model_types = {"modernbert", "roformer", "mamba"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types

        self.train_step_preds = []
        self.train_step_label = []
        self.val_step_preds = []
        self.val_step_label = []
        self.test_step_preds = []
        self.test_step_label = []

        self.ehr_embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout,
            use_position_embeddings=not is_rope,
            max_position_embeddings=(
                getattr(config, "max_position_embeddings", 0)
                if not is_rope
                else 0
            ),
            use_time=use_time,
            time_in_features=1,
            time_out_features=16,
            use_numeric=use_numeric,  
        )

    

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()

        if ckpt_path:
            self.get_pretrained_weights(ckpt_path=ckpt_path)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        # metrics (unchanged)
        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

    def forward(
        self,
        input_ids,
        attention_mask,
        type_ids,
        visit_ids,
        stage_ids,
        time_feats=None,
        numeric_values=None,  
        numeric_mask=None,    
        labels=None,
    ):
        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=input_ids,
            type_ids=type_ids,
            visit_ids=visit_ids,
            stage_ids=stage_ids,
            time_feats=time_feats,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,      
        )

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  

        # pooling
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  
            summed = (last_hidden * mask).sum(dim=1)                 
            lengths = mask.sum(dim=1).clamp(min=1.0)                  
            pooled = summed / lengths                               
        elif self.pooling == 'cls':
            pooled = last_hidden[:, 0, :]

        logits = self.classifier(pooled).squeeze(-1)              
        return logits

    def training_step(self, batch, batch_idx):

        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            time_feats=batch.get("time_diff", None),
            numeric_values=batch.get("numeric_values", None),  
            numeric_mask=batch.get("numeric_mask", None),    
            labels=None,
        )

        y = batch["label"].float().view(-1)    
        loss = self.criterion(logits, y)

        pos_score = torch.sigmoid(logits)       

        self.train_step_label.append(y)
        self.train_step_preds.append(pos_score)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        y = torch.cat(self.train_step_label)
        pos_score = torch.cat(self.train_step_preds)

        auroc = self.train_auroc(pos_score, y.long())
        auprc = self.train_auprc(pos_score, y.long())

        self.log('train_auroc', auroc, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log('train_auprc', auprc, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

        self.train_step_label.clear()
        self.train_step_preds.clear()

    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            time_feats=batch.get("time_diff", None),
            numeric_values=batch.get("numeric_values", None),  
            numeric_mask=batch.get("numeric_mask", None),    
            labels=None,
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.val_step_label.append(y)
        self.val_step_preds.append(pos_score)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self,*arg, **kwargs) -> None:
        y = torch.cat(self.val_step_label)
        pos_score = torch.cat(self.val_step_preds)

        auroc = self.val_auroc(pos_score, y.long())
        auprc = self.val_auprc(pos_score, y.long())

        self.log('val_auroc', auroc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_auprc', auprc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.val_step_label.clear()
        self.val_step_preds.clear()

    def test_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            time_feats=batch.get("time_diff", None),
            numeric_values=batch.get("numeric_values", None), 
            numeric_mask=batch.get("numeric_mask", None),     
            labels=None,
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.test_step_label.append(y)
        self.test_step_preds.append(pos_score)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss    

    def on_test_epoch_end(self,*arg, **kwargs) -> None:
        y = torch.cat(self.test_step_label)
        pos_score = torch.cat(self.test_step_preds)

        auroc = self.test_auroc(pos_score, y.long())
        auprc = self.test_auprc(pos_score, y.long())

        self.log('test_auroc', auroc, on_epoch=True, logger=True)
        self.log('test_auprc', auprc, on_epoch=True, logger=True)

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

        if self.optimizer == 'adamw':
            decay, no_decay = [], []
            for name, p in self.named_parameters():
                # biases + LayerNorm weights should NOT get weight decay
                if "bias" in name or "LayerNorm" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)

            optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": self.wd},
                                        {"params": no_decay, "weight_decay": 0.0}],
                                        lr=self.lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-8)
            
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                        lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=0.9,
                                        nesterov=True,
                                        weight_decay=self.wd,)


        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.max_epochs,
            eta_min=0)

        return {"optimizer": optimizer,"lr_scheduler": scheduler,}
    


    def get_pretrained_weights(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

        mt = getattr(self.backbone.config, "model_type", "").lower()
        prefix_map = {
            "bert":       "backbone.bert.",
            "roberta":    "backbone.roberta.",
            "longformer": "backbone.longformer.",
            "modernbert": "backbone.model.",
            "roformer":   "backbone.roformer.",
            "big_bird":   "backbone.bert.",
            "mamba":      "backbone.backbone.",
            "mamba2":     "backbone.backbone.",
        }
        backbone_prefix = prefix_map.get(mt, None)

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val.", "backbone.lm_head."]

        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                new_k = k
            elif backbone_prefix and k.startswith(backbone_prefix):
                new_k = "backbone." + k[len(backbone_prefix):]
            else:
                new_k = k

            remapped[new_k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print("weights loaded successfully!")
        print("missing keys:", missing)
        print("+" * 50)
        print("unexpected keys:", unexpected)




###############################
# Ours
###############################
class EHRRAPEncoders(nn.Module):
    def __init__(
        self,
        config,
        backbone,                 
        dropout: float = 0.1,
        pooling: str = "cls",       
        use_time: bool = False,
        use_numeric: bool = False,
        ckpt_path = None,
        return_token_level: bool = False,
    ):
        super().__init__()
        self.config = config
        self.pooling = pooling
        self.use_time = use_time
        self.use_numeric = use_numeric
        self.return_token_level = return_token_level
        self.backbone = backbone(config)

        rope_model_types = {"modernbert", "roformer","mamba"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types

        self.ehr_embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout,
            use_position_embeddings=not is_rope,
            max_position_embeddings=(getattr(config, "max_position_embeddings", 0) if not is_rope else 0),
            use_time=use_time,
            time_in_features=1,
            time_out_features=16,
            use_numeric=use_numeric,
        )
        if ckpt_path:
            self.get_pretrained_weights(ckpt_path=ckpt_path)

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,L,1]
            summed = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            return summed/lengths

    def _encode(
        self,
        x: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=x["input_ids"],
            type_ids=x["type_ids"],
            visit_ids=x["visit_ids"],
            stage_ids=x["stage_ids"],
            time_feats=x.get("time_diff", None) if self.use_time else None,
            numeric_values=x.get("numeric_values", None) if self.use_numeric else None,
            numeric_mask=x.get("numeric_mask", None) if self.use_numeric else None)

        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=x["attention_mask"],
            output_hidden_states=False,
            return_dict=True)
        
        seq = out.last_hidden_state
        vec = self._pool(seq, x["attention_mask"])
        return {"seq": seq, "vec": vec}

    def _flatten_history(self, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, K, L = h["input_ids"].shape
        flat = {}
        for k, v in h.items():
            if v is None:
                continue
            if v.dim() == 3:
                flat[k] = v.reshape(B * K, L)
        return flat

    def _unflatten_history(
        self,
        seq_flat: torch.Tensor,
        vec_flat: torch.Tensor, 
        B: int,
        K: int,
    ) -> Dict[str, torch.Tensor]:
        
        L = seq_flat.shape[1]
        H = seq_flat.shape[2]
        seq = seq_flat.reshape(B, K, L, H)
        vec = vec_flat.reshape(B, K, H)
        return {"seq": seq, "vec": vec}

    def forward(
        self,
        batch: Dict[str, Any],
        query_key: str = "query",
        history_key: str = "history", 
    ) -> Dict[str, torch.Tensor]:
        q = batch[query_key]
        h = batch[history_key]

        q_out = self._encode(q)

        B, K, L = h["input_ids"].shape
        h_flat = self._flatten_history(h)
        h_out_flat = self._encode(h_flat)
        h_out = self._unflatten_history(h_out_flat["seq"], h_out_flat["vec"], B=B, K=K)

        out = {"query_vec": q_out["vec"], "hist_vec": h_out["vec"]}

        if self.return_token_level:
            out["query_seq"]  = q_out["seq"]
            out["hist_seq"]   = h_out["seq"]
            out["query_mask"] = q["attention_mask"]
            out["hist_mask"]  = h["attention_mask"]

        return out
    
    def get_pretrained_weights(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

        mt = getattr(self.backbone.config, "model_type", "").lower()
        prefix_map = {
            "bert":       "backbone.bert.",
            "roberta":    "backbone.roberta.",
            "longformer": "backbone.longformer.",
            "modernbert": "backbone.model.",
            "roformer":   "backbone.roformer.",
            "big_bird":   "backbone.bert.",
            "mamba":      "backbone.backbone.",
            "mamba2":     "backbone.backbone.",
        }
        backbone_prefix = prefix_map.get(mt, None)

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val.", "backbone.lm_head."]

        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                new_k = k
            elif backbone_prefix and k.startswith(backbone_prefix):
                new_k = "backbone." + k[len(backbone_prefix):]
            else:
                new_k = k

            remapped[new_k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print("weights loaded successfully!")
        print("missing keys:", missing)
        print("+" * 50)
        print("unexpected keys:", unexpected)





class PrototypeRetrievalModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_prototypes: int = 256,
        temperature: float = 0.1,
        align_mode: str = "soft",
        sim_mode: str = "cosine",       
        combine_mode: str = "mul",
        lambda_sim: float = 0.5, 
        attn_threshold: float = 0.5,
        attn_temperature: float = 0.1,
        renormalize_after_mask: bool = True,
        return_debug: bool = False,
        normalize_prototypes: bool = True,
        detach_hard_alignment: bool = True, 
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.temperature = float(temperature)
        self.align_mode = align_mode
        self.sim_mode = sim_mode
        self.combine_mode = combine_mode
        self.lambda_sim = float(lambda_sim)

        self.attn_threshold = float(attn_threshold)
        self.attn_temperature = attn_temperature
        self.renormalize_after_mask = bool(renormalize_after_mask)

        self.return_debug = return_debug
        self.normalize_prototypes = normalize_prototypes
        self.detach_hard_alignment = detach_hard_alignment

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def _proto_probs(self, x: torch.Tensor) -> torch.Tensor:
        P = self.prototypes
        if self.normalize_prototypes:
            P = F.normalize(P, p=2, dim=-1)

        if self.sim_mode == "cosine":
            x = F.normalize(x, p=2, dim=-1)

        logits = x @ P.t()
        return F.softmax(logits / self.temperature, dim=-1)

    def _alignment_scores(self, q_probs: torch.Tensor, h_probs: torch.Tensor) -> torch.Tensor:
        
        if self.align_mode == "soft":
            return (q_probs.unsqueeze(1) * h_probs).sum(dim=-1)

        elif self.align_mode == "hard":
            q_id = q_probs.argmax(dim=-1)           
            h_id = h_probs.argmax(dim=-1)          
            align = (h_id == q_id.unsqueeze(1)).float()
            return align.detach() if self.detach_hard_alignment else align

    def _similarity_scores(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        
        if self.sim_mode == "dot":
            return (h * q.unsqueeze(1)).sum(dim=-1)

        elif self.sim_mode == "cosine":
            qn = F.normalize(q, p=2, dim=-1)
            hn = F.normalize(h, p=2, dim=-1)
            return (hn * qn.unsqueeze(1)).sum(dim=-1)

    def _combine_scores(self, sim: torch.Tensor, align: torch.Tensor) -> torch.Tensor:
        if self.combine_mode == "mul":
            return sim * align

        elif self.combine_mode == "add":
            lam = self.lambda_sim
            return lam * sim + (1.0 - lam) * align

    def _weights_and_mask(self, scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        w = F.softmax(scores/self.attn_temperature, dim=-1)

        if self.attn_threshold <= 0.0:
            mask = torch.ones_like(w, dtype=torch.long) 
            return {"attn_weights": w, "attn_mask": mask}

        mask = (w >= self.attn_threshold).long()

        if self.renormalize_after_mask:
            w_masked = w * mask.to(w.dtype)
            denom = w_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            w = w_masked / denom
        else:
            w = w * mask.to(w.dtype)

        return {"attn_weights": w, "attn_mask": mask}

    def forward(
        self,
        query_vec: torch.Tensor,
        hist_vec: torch.Tensor,
        return_debug: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:

        if return_debug is None:
            return_debug = self.return_debug

        q_probs = self._proto_probs(query_vec)
        h_probs = self._proto_probs(hist_vec)

        align = self._alignment_scores(q_probs, h_probs)
        sim = self._similarity_scores(query_vec, hist_vec)
        
        final = self._combine_scores(sim, align)

        wm = self._weights_and_mask(final)
        attn_weights = wm["attn_weights"]
        attn_mask = wm["attn_mask"]        

        out = {                      
            "attn_weights": attn_weights,               
            "attn_mask": attn_mask,                     
            "final_scores": final,
            'sim':sim,
            "align":align,
            "query_probs": q_probs,          
            "hist_probs": h_probs,  
        }

        if return_debug:
            out.update({
                "query_proto_probs": q_probs,          
                "hist_proto_probs": h_probs,           
                "align_scores": align,                  
                "sim_scores": sim,                      
                "softmax_scores": F.softmax(final, dim=-1), 
            })

        return out
    



class FusionModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_weights_as_gating: bool = False,
        output_mode: str = "query",  
        return_seq: bool = False,
    ):
        super().__init__()
        self.use_weights_as_gating = use_weights_as_gating
        self.output_mode = output_mode
        self.return_seq = return_seq

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_mult * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True, 
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query_vec: torch.Tensor,                
        hist_vec: torch.Tensor,                  
        attn_mask: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,  
    ) -> Dict[str, torch.Tensor]:

        B, K, H = hist_vec.shape
        assert query_vec.shape == (B, H)
        if self.use_weights_as_gating and attn_weights is not None:
            hist_vec = hist_vec * attn_weights.unsqueeze(-1).to(hist_vec.dtype) 
        x = torch.cat([query_vec.unsqueeze(1), hist_vec], dim=1)  
        if attn_mask is None:
            keep = torch.ones(B, 1 + K, device=x.device, dtype=torch.long)
        else:
            keep = torch.cat([torch.ones(B, 1, device=x.device, dtype=attn_mask.dtype), attn_mask],dim=1)  
        src_key_padding_mask = (keep == 0)
        x_fused = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x_fused = self.out_norm(x_fused)

        if self.output_mode == "query":
            fused_vec = x_fused[:, 0, :] 
        elif self.output_mode == "mean":
            w = keep.to(x_fused.dtype).unsqueeze(-1)  
            denom = w.sum(dim=1).clamp_min(1.0)
            fused_vec = (x_fused * w).sum(dim=1) / denom
        out = {"fused_vec": fused_vec}
        if self.return_seq:
            out["fused_seq"] = x_fused
            out["fused_keep_mask"] = keep
        return out
    


class EHRRAPEvalModel(lt.LightningModule):
    def __init__(
        self,
        config,
        backbone,                      
        ckpt_path: Optional[str] = None,
        lr: float = 2e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
        pooling: str = "cls",
        use_numeric: bool = False,
        use_time: bool = False,
        optimizer: str = "sgd",
        # --- prototype module ---
        num_prototypes: int = 256,
        proto_temperature: float = 0.1,
        align_mode: str = "soft",          # "soft" | "hard"
        sim_mode: str = "cosine",          # "cosine" | "dot"
        combine_mode: str = "mul",         # "mul" | "add"
        lambda_sim: float = 0.5,
        attn_threshold: float = 0.0,
        attn_temperature: float = 1.0,
        renormalize_after_mask: bool = False,
        normalize_prototypes: bool = True,
        detach_hard_alignment: bool = True,
        # --- fusion module ---
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        fusion_ff_mult: int = 4,
        fusion_use_weights_as_gating: bool = True,
        fusion_output_mode: str = "query",  # "query" | "mean"
        # --- misc ---
        return_debug: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.config = config
        self.optimizer_name = optimizer

        rope_model_types = {"modernbert", "roformer", "mamba"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types


        self.encoders = EHRRAPEncoders(
            config=config,
            backbone=backbone,
            dropout=dropout,
            pooling=pooling,
            use_time=use_time,
            use_numeric=use_numeric,
            ckpt_path=ckpt_path,
            return_token_level=False)

        self.prototypes = PrototypeRetrievalModule(hidden_size=config.hidden_size,
                                                   num_prototypes=num_prototypes,
                                                   temperature=proto_temperature,
                                                   align_mode=align_mode,
                                                   sim_mode=sim_mode,
                                                   combine_mode=combine_mode,
                                                   lambda_sim=lambda_sim,
                                                   attn_threshold=attn_threshold,
                                                   attn_temperature=attn_temperature,
                                                   renormalize_after_mask=renormalize_after_mask,
                                                   return_debug=return_debug,
                                                   normalize_prototypes=normalize_prototypes,
                                                   detach_hard_alignment=detach_hard_alignment)

        self.fusion = FusionModule(hidden_size=config.hidden_size,
                                   num_layers=fusion_layers,
                                   num_heads=fusion_heads,
                                   ff_mult=fusion_ff_mult,
                                   dropout=dropout,
                                   use_weights_as_gating=fusion_use_weights_as_gating,
                                   output_mode=fusion_output_mode,
                                   return_seq=False)

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()


#         if freeze:
#             for p in self.encoders.parameters():
#                 p.requires_grad = False
#             for p in self.prototypes.parameters():
#                 p.requires_grad = True
#             for p in self.fusion.parameters():
#                 p.requires_grad = True
#             for p in self.classifier.parameters():
#                 p.requires_grad = True

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

        self._train_preds, self._train_labels = [], []
        self._val_preds, self._val_labels = [], []
        self._test_preds, self._test_labels = [], []
        

        self.return_debug = return_debug

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:

        enc_out = self.encoders(batch, query_key="query", history_key="history")

        
        proto_out = self.prototypes(
            query_vec=enc_out["query_vec"],
            hist_vec=enc_out["hist_vec"])
#         print('sim', proto_out['sim'],'\n')
#         print('align', proto_out['align'],'\n')
# #         print('query_probs', proto_out['query_probs'])
# #         print('hist_probs', proto_out['hist_probs'])
        
#         print('final_scores=',proto_out['final_scores'],'\n')
#         print('attn_weights=',proto_out['attn_weights'],'\n')
#         print('attn_mask=',proto_out['attn_mask'],'\n\n\n\n')
        

        fuse_out = self.fusion(
            query_vec=enc_out["query_vec"],
            hist_vec=enc_out["hist_vec"],
            attn_mask=proto_out.get("attn_mask", None),
            attn_weights=proto_out.get("attn_weights", None),)

        fused_vec = fuse_out["fused_vec"]
#         print('fused_vec=', fused_vec,)
        logits = self.classifier(fused_vec).squeeze(-1) 

        out = {"logits": logits, "fused_vec": fused_vec, "proto": proto_out}

        if self.return_debug:
            out["debug"] = {"attn_keep_rate": proto_out["attn_mask"].float().mean(dim=1),
                            "attn_max_weight": proto_out["attn_weights"].max(dim=1).values,}
        return out

    def shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        out = self.forward(batch)
        logits = out["logits"]

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)

        pos_score = torch.sigmoid(logits)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        if stage == "train":
            self._train_labels.append(y.detach())
            self._train_preds.append(pos_score.detach())
        elif stage == "val":
            self._val_labels.append(y.detach())
            self._val_preds.append(pos_score.detach())
        elif stage == "test":
            self._test_labels.append(y.detach())
            self._test_preds.append(pos_score.detach())

        if "proto" in out and "attn_mask" in out["proto"]:
            keep_rate = out["proto"]["attn_mask"].float().mean()
            self.log(f"{stage}_keep_rate", keep_rate, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        if "proto" in out and "attn_weights" in out["proto"]:
            max_w = out["proto"]["attn_weights"].max(dim=1).values.mean()
            self.log(f"{stage}_attn_maxw", max_w, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="test")


    def on_train_epoch_end(self) -> None:
        if not self._train_preds:
            return
        y = torch.cat(self._train_labels)
        p = torch.cat(self._train_preds)
        self.log("train_auroc", self.train_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("train_auprc", self.train_auprc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self._train_labels.clear()
        self._train_preds.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        y = torch.cat(self._val_labels)
        p = torch.cat(self._val_preds)
        self.log("val_auroc", self.val_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("val_auprc", self.val_auprc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self._val_labels.clear()
        self._val_preds.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        y = torch.cat(self._test_labels)
        p = torch.cat(self._test_preds)
        self.log("test_auroc", self.test_auroc(p, y.long()), on_epoch=True, logger=True)
        self.log("test_auprc", self.test_auprc(p, y.long()), on_epoch=True, logger=True)

        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=y,
            y_score=p,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3,
            )

        self._test_labels.clear()
        self._test_preds.clear()

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            decay, no_decay = [], []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if "bias" in name or "LayerNorm" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)

            optimizer = torch.optim.AdamW(
                [{"params": decay, "weight_decay": self.wd},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-8,)

        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad],
                lr=self.lr,)

        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                [p for p in self.parameters() if p.requires_grad],
                lr=self.lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.wd,)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.max_epochs,
            eta_min=0.0,)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
