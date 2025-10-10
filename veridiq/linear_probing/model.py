import lightning as L
import torch
import torch.nn as nn


class ConvTorchModel(nn.Module):
    def __init__(self, feats_dim, num_layers, hidden_dim, kernel_size, **kwargs):
        super(ConvTorchModel, self).__init__()
        layers = []
        layers.append(self.make_layer(feats_dim, hidden_dim, kernel_size))
        for _ in range(num_layers - 1):
            layers.append(self.make_layer(hidden_dim, hidden_dim, kernel_size))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def make_layer(self, dim1, dim2, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(
                dim1,
                dim2,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.conv(x)  # (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, output_dim)
        x = self.fc(x)  # (batch_size, seq_len, 1)
        return x


class LinearTorchModel(nn.Module):
    def __init__(self, feats_dim, **kwargs):
        super(LinearTorchModel, self).__init__()
        self.fc = nn.Linear(feats_dim, 1)

    def forward(self, x):
        return self.fc(x)


MODELS = {
    "conv": ConvTorchModel,
    "linear": LinearTorchModel,
}


class MyModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config["model_hparams"]
        self.feats_dim = self.config["feats_dim"]
        self.input_type = self.config["input_type"]

        self.save_hyperparameters()

        model_type = config.get("model_type", "linear")
        self.model = MODELS[model_type](**config["model_hparams"])

    def forward(self, input_feats, per_frame=False):
        video_feats, audio_feats = input_feats[0], input_feats[1]

        if self.input_type == "both":
            fused_features = torch.cat((video_feats, audio_feats), dim=-1)
        elif self.input_type == "audio":
            fused_features = audio_feats
        elif (
            self.input_type == "video" or self.input_type == "multimodal"
        ):  # for multimodal, video_feats and audio_feats are equal
            fused_features = video_feats
        else:
            raise ValueError(f"Error! Input type: {self.input_type}")

        output = self.model(fused_features)
        output = output[:, :, 0]

        if per_frame:
            return torch.logsumexp(output, dim=-1), output
        else:
            return torch.logsumexp(output, dim=-1)

    def predict_scores(self, video_feats, audio_feats):
        scores = self.forward((video_feats, audio_feats))
        return scores

    def predict_scores_per_frame(self, video_feats, audio_feats):
        scores = self.forward((video_feats, audio_feats), per_frame=True)
        return scores

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch

        output = self.forward((video_feats, audio_feats))
        score = output.unsqueeze(1)
        score = torch.cat((-score, score), 1)

        loss = torch.nn.functional.cross_entropy(score, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch

        output = self.forward((video_feats, audio_feats))
        score = output.unsqueeze(1)
        score = torch.cat((-score, score), 1)

        loss = torch.nn.functional.cross_entropy(score, labels)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer], []
