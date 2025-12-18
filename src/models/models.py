
import os
import torch


import lightning as lt
import torch.nn as nn
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
        use_time: bool = True,
        time_in_features: int = 1,
        time_out_features: int = 16,
        use_numeric: bool = True,
        numeric_hidden_size: int = 16,   # <-- small bottleneck for numeric
    ):
        super().__init__()

        self.tok_emb   = nn.Embedding(vocab_size,       embedding_size, padding_idx=pad_token_id)
        self.type_emb  = nn.Embedding(type_vocab_size,  embedding_size, padding_idx=pad_token_id)
        self.visit_emb = nn.Embedding(visit_vocab_size, embedding_size, padding_idx=pad_token_id)
        self.stage_emb = nn.Embedding(stage_vocab_size, embedding_size, padding_idx=pad_token_id)

        # ---- positional (local) ----
        self.use_position_embeddings = use_position_embeddings
        if use_position_embeddings:
            if max_position_embeddings <= 0:
                raise ValueError("max_position_embeddings must be > 0 when use_position_embeddings=True")
            self.pos_emb = nn.Embedding(max_position_embeddings, embedding_size)
        else:
            self.pos_emb = None

        # ---- time (Time2Vec) ----
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

        # ---- numeric values ----
        self.use_numeric = use_numeric
        if use_numeric:
            self.numeric_hidden_size = numeric_hidden_size
            # 1 scalar -> small hidden -> embedding_size
            self.num_proj1 = nn.Linear(1, numeric_hidden_size)
            self.num_proj2 = nn.Linear(numeric_hidden_size, embedding_size)
            self.num_act = nn.GELU()

            # learned embedding for "no numeric value"
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
        time_feats=None,          # (B, L) or (B, L, time_in_features)
        numeric_values=None,      # (B, L) normalized in [-3, 3]
        numeric_mask=None,        # (B, L) bool/int: True if numeric is present
    ):
        # base token + type + visit + stage
        x = self.tok_emb(input_ids.long())
        x = x + self.type_emb(type_ids.long())
        x = x + self.visit_emb(visit_ids.long())
        x = x + self.stage_emb(stage_ids.long())

        # positional (local) embeddings
        if self.pos_emb is not None:
            bsz, seqlen = input_ids.size()
            position_ids = torch.arange(
                seqlen, device=input_ids.device
            ).unsqueeze(0).expand(bsz, seqlen)
            x = x + self.pos_emb(position_ids)

        # time (Time2Vec)
        if self.use_time:
            if time_feats is None:
                raise ValueError("time_feats must be provided when use_time=True")
            if time_feats.dim() == 2:
                time_feats = time_feats.unsqueeze(-1)
            elif time_feats.dim() != 3:
                raise ValueError(f"Unexpected time_feats.dim()={time_feats.dim()}, expected 2 or 3")
            t = self.time2vec(time_feats.float())   # (B, L, time_out_features)
            t = self.time_proj(t)                   # (B, L, embedding_size)
            x = x + t

        # numeric values
        if self.use_numeric:
            if numeric_values is None or numeric_mask is None:
                raise ValueError("numeric_values and numeric_mask must be provided when use_numeric=True")

            # (optional safety) clamp extreme values
            v = numeric_values.float().unsqueeze(-1)        # (B, L, 1)
            # small bottleneck then project to emb size
            h = self.num_act(self.num_proj1(v))             # (B, L, H_num)
            num_emb = self.num_proj2(h)                     # (B, L, D)

            mask = numeric_mask.bool().unsqueeze(-1)        # (B, L, 1)
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
        rope_model_types = {"modernbert", "roformer"}
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

        # self.backbone.embeddings = self.ehr_embeddings

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

        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=0.9,
                                        nesterov=True,
                                        weight_decay=self.wd)


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
        }
        backbone_prefix = prefix_map.get(mt, None)

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val."]

        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                new_k = k                      # matches self.ehr_embeddings.*
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


