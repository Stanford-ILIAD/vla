"""
train.py

Train a YOLO-v5 Latent Actions Model (Conditional AutoEncoder; CAE) on Real-Robot Kinesthetic Demonstrations. Packages
code for loading fine-tuned YOLO-v5 Model, defining PyTorch-Lightning Modules for Latent Action Models, and for
performing data augmentation & training.

Additionally saves model checkpoints and logs statistics.
"""
from argparse import Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from tap import Tap
from torch.utils.data import DataLoader
from typing import List

from src.logging import MetricLogger
from src.models import YOLOCAE
from src.preprocessing import get_dataset
from yolo import YOLODetector

import numpy as np
import os
import pytorch_lightning as pl
import torch


class ArgumentParser(Tap):
    # fmt: off
    # Demonstration and Checkpoint Parameters
    demonstrations: str                                         # Path to demonstrations collected with robot
    checkpoint: str = "checkpoints/"                            # Path to Checkpoint Directory

    # Configuration Parameters
    n_classes: int                                              # Number of Total Object Classes
    classes: List[str]                                          # Names of each Object Class (len == n_classes)

    # YOLO-v5 Parameters
    yolo_model: str                                             # Path to in-domain pretrained YOLO-v5 Detector

    # Model Parameters
    latent_dim: int = 1                                         # Dimensionality of Latent Space (match input)
    state_dim: int = 7                                          # Dimensionality of Robot State (7-DoF Joint Angles)

    # YOLOCAE Model Parameters
    hidden: int = 30                                            # Size of AutoEncoder Hidden Layer (Dylan Magic)

    # End2End CAE Model Parameters
    feat_dim: int = 32                                          # Size to Compress CNN Representation (Embedding)

    # Preprocessing Parameters
    img_size: int = 640                                         # Resized Image Size (default: 640 x 640)

    # GPUs
    gpus: int = 0                                               # Number of GPUs to run with (defaults to cpu)

    # Training Parameters
    epochs: int = 1000                                          # Number of training epochs to run
    bsz: int = 6000                                             # Batch Size for training
    lr: float = 0.01                                            # Learning Rate for training
    lr_step_size: int = 400                                     # How many epochs to run before LR decay
    lr_gamma: float = 0.1                                       # Learning Rate Gamma (decay rate)

    # Train-Val/Augmentation/Noise Parameters
    val_split: float = 0.1                                      # Percentage of Data to use as Validation
    noise_std: float = 0.01                                     # Standard Deviation for Gaussian to draw noise from
    window: int = 10                                            # Window-Shift Augmentation to apply to demo states

    # Random Seed
    seed: int = 21                                              # Random Seed (for Reproducibility)
    # fmt: on


def train():
    # Parse Arguments --> Convert from Namespace --> Dict --> Namespace because of weird WandB Bug
    print("[*] Starting up...")
    args = Namespace(**ArgumentParser().parse_args().as_dict())
    print('\t[*] "Does the walker choose the path, or the path the walker?" (Garth Nix - Sabriel)\n')

    # Create Run Name
    run_name = f"vla-data={os.path.basename(args.demonstrations)}-z={args.latent_dim}-w={args.window}" \
               f"-n={args.noise_std:.2f}-h={args.hidden}-ep={args.epochs}-x{args.seed}"

    # Set Randomness + Device
    print(f"[*] Setting Random Seed to {args.seed}!\n")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLO-v5 Detector
    print("[*] Loading Pre-Trained YOLO-v5 Detector...\n")
    detector = YOLODetector(args.yolo_model, device)

    # Create Dataset
    print("[*] Creating YOLO Training and Validation Datasets from Demonstrations...\n")
    train_dataset, val_dataset = get_dataset(args.demonstrations, detector, args.classes, val_split=args.val_split,
                                             img_size=(args.img_size, args.img_size), window=args.window,
                                             noise_std=args.noise_std, device=device, seed=args.seed)

    # Initialize DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bsz, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bsz, shuffle=False)

    # Create Model
    print("[*] Initializing Latent Actions Model...\n")
    nn = YOLOCAE(args, detector)

    # Create Trainer
    print("[*] Training...\n")
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.checkpoint, "runs", run_name, run_name + "+" + "{train_loss:.2f}-{val_loss:.2f}"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    logger = MetricLogger(name=run_name, save_dir=args.checkpoint)
    trainer = pl.Trainer(default_root_dir=args.checkpoint, max_epochs=args.epochs, gpus=args.gpus, logger=logger,
                         checkpoint_callback=checkpoint_callback)

    # Fit
    trainer.fit(nn, train_loader, val_loader)


if __name__ == "__main__":
    train()
