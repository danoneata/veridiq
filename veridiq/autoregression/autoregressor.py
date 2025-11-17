import joblib
import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from util_blocks import WindowedTransformerBlock, build_windowed_causal_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape becomes [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MyTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, wsa):
        super().__init__()

        self.window_size = wsa
        self.layers = nn.ModuleList([
            WindowedTransformerBlock(d_model, nhead, dim_feedforward, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask, mask=None, is_causal=None):  # mask and is_casual are not used by this block
        _, T, _ = src.shape
        device = src.device

        attn_mask = build_windowed_causal_mask(
            T=T,
            W=self.window_size,
            device=device,
            key_padding_mask=src_key_padding_mask
        )

        for layer in self.layers:
            src = layer(src, attn_mask=attn_mask)

        return self.final_ln(src)


class AutoregressorHead(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config["model_hparams"]
        self.epochs = config["epochs"]
        self.save_hyperparameters()

        self.pe = PositionalEncoding(d_model=self.config["d_model"])

        self.project_layer = nn.Linear(self.config["feats_dim"], self.config["d_model"])

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            dim_feedforward=self.config["d_ff"],
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.config["num_layers"])

        self.head = nn.Linear(self.config["d_model"], self.config["feats_dim"])

    def forward(self, x, pad_mask, predict_last=False):
        x = self.project_layer(x)
        x = self.pe(x)

        seq_len = x.size(1)

        decoder_out = self.transformer(
            src=x,
            mask=torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device),
            src_key_padding_mask=~pad_mask.bool(),
            is_causal=True
        )

        if predict_last:
            output = self.head(decoder_out[:, -1, :])
        else:
            output = self.head(decoder_out)

        return output

    def predict_scores(self, features, masks):

        masks = masks[:, 1:]

        output = self.forward(features[:, :-1, :], masks)
        mse_loss = torch.nn.functional.mse_loss(output, features[:, 1:, :], reduction='none')

        mse_loss, _ = torch.max(mse_loss.mean(dim=-1), dim=-1)  # TODO: try multiple versions

        return mse_loss

    def training_step(self, batch, batch_idx):
        features, masks, _, _ = batch
        masks = masks[:, 1:]

        output = self.forward(features[:, :-1, :], masks)
        loss = torch.nn.functional.mse_loss(output, features[:, 1:, :], reduction='none')

        loss = (loss * masks.unsqueeze(2).float()).sum()
        non_zero_elements = (masks.sum() * features.shape[2])
        loss = loss / non_zero_elements

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        features, masks, _, _ = batch
        masks = masks[:, 1:]

        output = self.forward(features[:, :-1, :], masks)
        loss = torch.nn.functional.mse_loss(output, features[:, 1:, :], reduction='none')

        loss = (loss * masks.unsqueeze(2).float()).sum()
        non_zero_elements = (masks.sum() * features.shape[2])
        loss = loss / non_zero_elements

        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
