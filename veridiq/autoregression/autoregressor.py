import joblib
import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn


class PCA_Wrapper():
    def __init__(self, config):
        if "apply_pca" in config.keys():
            pca = joblib.load(config["apply_pca"]["pca_model_path"])
            self.W = torch.tensor(pca.components_, dtype=torch.float32).cuda()  # Principal components
            self.mean = torch.tensor(pca.mean_, dtype=torch.float32).cuda()  # Mean for centering
            self.n_components = config["apply_pca"]["pca_n_components"]
        else:
            self.W = None
            self.mean = None

    def __call__(self, x):
        if self.mean is not None:
            x = (x - self.mean) @ self.W.T
            x = x[:, :, :self.n_components]
        return x


class Quantizer():
    def __init__(self, config):
        if "apply_quantization" in config.keys():
            self.quant_method = config["apply_quantization"]["quant_method"]
            self.quant_buckets = config["apply_quantization"]["quant_buckets"]
            self.quant_scaler = config["apply_quantization"]["quant_scaler"]

            if self.quant_method == "static_values":
                self.min = torch.from_numpy(np.load(config["apply_quantization"]["quant_min"])).cuda()
                self.max = torch.from_numpy(np.load(config["apply_quantization"]["quant_max"])).cuda()
            else:
                self.min = None
                self.max = None
        else:
            self.quant_method = None

    def _update_min_max(self, x):
        if self.min is None:
            self.min = x.min(dim=0)[0].min(dim=0)[0]
            self.max = x.max(dim=0)[0].max(dim=0)[0]
        else:
            self.min = 0.99 * self.min + 0.01 * x.min(dim=0)[0].min(dim=0)[0]
            self.max = 0.99 * self.max + 0.01 * x.max(dim=0)[0].max(dim=0)[0]

    def _transform(self, x):
        scale = (self.max - self.min) / (self.quant_buckets - 1)
        quantized = torch.round((x - self.min) / scale)
        quantized = torch.clip(quantized, 0, self.quant_buckets - 1)
        quantized /= self.quant_scaler

        return quantized

    def revert(self, x):
        scale = (self.max - self.min) / (self.quant_buckets - 1)
        return x * scale + self.min

    def __call__(self, x):
        if self.quant_method:
            if self.quant_method == "moving_avg":
                self._update_min_max(x)
                x = self._transform(x)
            elif self.quant_method == "static_values":
                x = self._transform(x)
            else:
                raise ValueError("Choose a correct quantization method!")
        return x


class PositionalEncoding(nn.Module):

    # def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    #     super().__init__()
    #     self.dropout = nn.Dropout(p=dropout)

    #     position = torch.arange(max_len).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    #     pe = torch.zeros(max_len, 1, d_model)
    #     pe[:, 0, 0::2] = torch.sin(position * div_term)
    #     pe[:, 0, 1::2] = torch.cos(position * div_term)
    #     self.register_buffer('pe', pe)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Arguments:
    #         x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    #     """
    #     x = x + self.pe[:x.size(0)]
    #     return self.dropout(x)

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

        # self.pca = PCA_Wrapper(config)
        # self.quant = Quantizer(config)

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

    def predict_scores(self, video_feats, audio_feats, masks, labels, test_level):

        features = torch.cat((video_feats, audio_feats), dim=-1)
        masks = masks[:, 1:]

        # features = self.pca(features)
        # features = self.quant(features)

        output = self.forward(features[:, :-1, :], masks)
        mse_loss = torch.nn.functional.mse_loss(output, features[:, 1:, :], reduction='none')

        if test_level == "video_level":
            mse_loss, _ = torch.max(mse_loss.mean(dim=-1), dim=-1)
        elif test_level == "frame_level":
            mse_loss = mse_loss.mean(dim=-1).view(-1)
            labels = labels[:, 1:].view(-1)
        else:
            raise ValueError("Wrong test level. Expected: video_level/frame_level; Got: " + test_level)

        return mse_loss, labels

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, masks, _, _ = batch
        features = torch.cat((video_feats, audio_feats), dim=-1)
        masks = masks[:, 1:]

        # features = self.pca(features)
        # features = self.quant(features)

        output = self.forward(features[:, :-1, :], masks)
        loss = torch.nn.functional.mse_loss(output, features[:, 1:, :], reduction='none')

        loss = (loss * masks.unsqueeze(2).float()).sum()
        non_zero_elements = (masks.sum() * features.shape[2])
        loss = loss / non_zero_elements

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        video_feats, audio_feats, masks, _, _ = batch
        features = torch.cat((video_feats, audio_feats), dim=-1)
        masks = masks[:, 1:]

        # features = self.pca(features)
        # features = self.quant(features)

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
