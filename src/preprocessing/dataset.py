"""
dataset.py

Core class definition for datasets, image and data preprocessing, and other necessary startup steps to run before
training the YOLO CAE Latent Action Model.
"""
from torch.utils.data import Dataset, ConcatDataset

from .util import letterbox

import os
import pickle
import torch


class SingleDemo(Dataset):
    def __init__(self, data, detector, is_train=True, img_size=(640, 640), window=10, noise_std=0.01,
                 device=torch.device("cpu"), seed=21):
        # Full Reproducibility!
        torch.manual_seed(seed)

        # Split Demonstration into Image Observation (vis) and Joint States (demo)
        self.vis, self.demo = data
        self.img_size = img_size
        self.is_train, self.window, self.noise_std = is_train, window, noise_std
        self.noise = torch.normal(torch.zeros(len(self.demo), 7), std=noise_std)

        # Calculate All YOLO Detections Once and Store Them
        with torch.no_grad():
            detector.eval()

            # First, Preprocess Image for YOLO
            padded = letterbox(self.vis, 640)[:, :, ::-1].copy()
            img = torch.FloatTensor(padded).permute(2, 0, 1).unsqueeze(0) / 256.0
            self.stored_feats = detector.get_object_features(img.to(device))[0]

    def __len__(self):
        return len(self.demo) - self.window

    def __getitem__(self, idx):
        item = self.demo[idx]
        image = self.stored_feats

        # State is the first 7 Features (7-DoF) of the demonstration iteration [we don't care about velocity]
        state = torch.FloatTensor(item[:7])

        # Augmentation/Replacement --> Add Noise to Initial State
        if self.is_train:
            # If Training --> Noise the Input (Randomly) each Epoch -- NOTE: DON'T DO THIS!
            # state += torch.normal(torch.zeros(state.shape), std=self.noise_std)

            # Fixed Noise!
            state += self.noise[idx]
        else:
            # Otherwise --> Apply fixed noise mask
            state += self.noise[idx]

        # Compute Next State with Fixed Window
        next_state = self.demo[idx + self.window][:7]

        # Compute action as the difference between the two states (qdot -> poor man's derivative = velocity)
        action = (torch.from_numpy(next_state) - state).float()

        return image, state, action


def get_dataset(demonstrations, yolo_detector, classes, val_split=0.1, img_size=(640, 640), window=10, noise_std=0.01,
                device=torch.device("cpu"), seed=21):
    print("\t[*] Preparing Stored Detections...")
    train, val = [], []
    for object_demo in [x for x in os.listdir(demonstrations) if ".pkl" in x]:
        if object_demo.split("-")[0] in classes:
            # Retrieve Data
            with open(os.path.join(demonstrations, object_demo), "rb") as f:
                demos = pickle.load(f)

            # Iterate through Demos, allocating (1 - val_split) fraction to train, and the rest to val
            n_train = int(len(demos) * (1 - val_split))
            for d in demos[:n_train]:
                # Iterate through Window
                for w in range(1, window + 1):
                    ds = SingleDemo(d, yolo_detector, is_train=True, img_size=img_size, window=w, noise_std=noise_std,
                                    device=device, seed=seed)

                    # Error Handling
                    if ds.stored_feats is None:
                        continue

                    train.append(ds)

            # Allocate the Remainder to Validation
            for d in demos[-max(1, len(demos) - n_train) :]:
                # Iterate through Window
                for w in range(1, window + 1):
                    ds = SingleDemo(d, yolo_detector, is_train=False, img_size=img_size, window=w, noise_std=noise_std,
                                    device=device,  seed=seed)

                    # Error Handling
                    if ds.stored_feats is None:
                        continue

                    val.append(ds)

    # Create Datasets
    return ConcatDataset(train), ConcatDataset(val)
