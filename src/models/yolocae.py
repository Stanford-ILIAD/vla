"""
yolocae.py

PyTorch-Lightning Module Definition for the YOLO-CAE Latent Actions Model.
"""
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class YOLOCAE(pl.LightningModule):
    def __init__(self, hparams, detector):
        super(YOLOCAE, self).__init__()

        # Save Hyper-Parameters and Detector
        self.hparams, self.detector = hparams, detector

        # Lock Detector from Gradient Updates
        for param in self.detector.parameters():
            param.requires_grad = False

        # Class Variables + Helpers
        self.feat_dim, self.latent_dim = self.detector.n_classes + 2, self.hparams.latent_dim
        self.img_size = (self.hparams.img_size, self.hparams.img_size)
        self.stored = [None]

        # Build Model
        self.build_model()

    def build_model(self):
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(self.feat_dim + self.hparams.state_dim + self.hparams.state_dim, self.hparams.hidden),
            nn.Tanh(),
            nn.Linear(self.hparams.hidden, self.hparams.hidden),
            nn.Tanh(),
            nn.Linear(self.hparams.hidden, self.latent_dim)
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(self.feat_dim + self.hparams.state_dim + self.latent_dim, self.hparams.hidden),
            nn.Tanh(),
            nn.Linear(self.hparams.hidden, self.hparams.hidden),
            nn.Tanh(),
            nn.Linear(self.hparams.hidden, self.hparams.state_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size,
                                              gamma=self.hparams.lr_gamma)

        return [optimizer], [scheduler]

    ## Helper Functions
    def store_visual(self, im, keep_waiting=False):
        obj_feats = [None]
        with torch.no_grad():
            if keep_waiting:
                while None in obj_feats:
                    obj_feats = self.detector.get_object_features(im)
            else:
                obj_feats = self.detector.get_object_features(im)

        if None not in obj_feats:
            self.stored = obj_feats

    def decoder(self, context, use_stored=False):
        if use_stored:
            obj_feats = self.stored
        else:
            with torch.no_grad():
                obj_feats = self.features.get_object_features(context[0])

        # Action when no detection
        if None in obj_feats:
            return torch.zeros(context[1].shape[0], 7)

        obj_feats = torch.stack(obj_feats, 0)
        x = torch.cat((obj_feats, context[1], context[2]), 1)

        return self.dec(x)

    ## Default Training (Lightning) Code
    def forward(self, obj_feats, s, a):
        # Don't run on missed object detections!
        if torch.sum(obj_feats) < 0.1:
            raise ValueError("Object Features are Weird!")

        # Create Input to Encoder --> Objects, State, Action
        x = torch.cat((obj_feats, s, a), 1)
        z = self.enc(x)

        # Create Input to Decoder --> Objects, State, Latent Action
        x = torch.cat((obj_feats, s, z), 1)

        # Return Predicted Action
        return self.dec(x)

    def training_step(self, batch, batch_idx):
        # Extract Batch
        obj_feats, state, action = batch

        # Get Predicted Action
        predicted_action = self.forward(obj_feats, state, action)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action)

        # Log Loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extract Batch
        obj_feats, state, action = batch

        # Get Predicted Action
        predicted_action = self.forward(obj_feats, state, action)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_action, action)

        # Log Loss
        self.log('val_loss', loss, prog_bar=True)
