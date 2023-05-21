from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import DLOSegmentationDataset
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    transform = et.ExtCompose([
                et.ExtResize(513),
                et.ExtCenterCrop(513),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

    train_dst = DLOSegmentationDataset(
        root="../dlo_dataset", base_dir="train/train", n_imgs=None, transform=transform)


    train_loader = data.DataLoader(
            train_dst, batch_size=1, shuffle=False, num_workers=2,
            drop_last=True)


    for (images, labels) in train_loader:

        img = images[0].detach().cpu().numpy()
        target = labels[0].cpu().numpy()

        print(target)

        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
        target = train_dst.decode_target(target).astype(np.uint8)

        fig = plt.figure()
        plt.imshow(img)

        fig = plt.figure()
        plt.imshow(target)

        plt.show()