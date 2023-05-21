import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


class DLOSegmentationDataset(data.Dataset):
    """
        background:    id: 0; color: [0, 0, 0];
        dlo:           id: 1; color: [255, 255, 255]; (r, g, b)
    """
    train_id_to_color = [[0, 0, 0], [255, 255, 255]] 
    train_id_to_color = np.array(train_id_to_color)

    def __init__(self,
                 root,
                 base_dir,
                 transform=None):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        
        data_dir = os.path.join(self.root, base_dir)

        imgs_dir = os.path.join(data_dir, 'imgs')
        masks_dir = os.path.join(data_dir, 'masks')

        self.images = []
        self.masks = []

        for root, dirs, files in os.walk(imgs_dir):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext == '.png' or ext == '.jpg':
                    self.images.append(os.path.join(imgs_dir, file))
                    if(os.path.exists(os.path.join(masks_dir, file))):
                        self.masks.append(os.path.join(masks_dir, file))
                    else:
                        print("Warning: the mask ", file, " doesn't exist.")
        # else:
        #     self.images = [os.path.join(imgs_dir, str(x) + ".png") for x in range(n_imgs)]
        #     self.masks = [os.path.join(masks_dir, str(x) + ".png") for x in range(n_imgs)]


    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, mask):
        if np.max(mask) > 1:
            mask = mask / 255.0
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return mask.astype(np.int32)
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.train_id_to_color[mask]
    
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert("L") # gray 0~255
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return np.array(img), self.encode_target(np.array(mask))
