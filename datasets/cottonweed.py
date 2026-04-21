import os

import numpy as np
import torch.utils.data as data
from PIL import Image


class CottonWeedSegmentation(data.Dataset):
    """CottonWeed segmentation dataset.

    Expected folder layout:
        root/
          train/
            images/*.jpg|*.png
            masks_trainid/*.png
          val/
            images/*.jpg|*.png
            masks_trainid/*.png
          test/
            images/*.jpg|*.png
            masks_trainid/*.png
    """

    # Default 8-class mapping for cottonweedV4-style RGB masks.
    # class ids:
    # 0 background
    # 1 cotton
    # 2 abuth
    # 3 canger
    # 4 machixian
    # 5 longkui
    # 6 tianxuanhua
    # 7 niujincao
    cmap = np.array(
        [
            [0, 0, 0],
            [22, 244, 22],
            [121, 234, 249],
            [243, 170, 161],
            [248, 169, 227],
            [127, 107, 114],
            [18, 155, 187],
            [61, 61, 245],
        ],
        dtype=np.uint8,
    )

    def __init__(self, root, split="train", transform=None, mask_dir="masks_trainid"):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform

        if split not in ["train", "val", "test"]:
            raise ValueError('Invalid split. Please use split="train", "val" or "test".')

        image_dir = os.path.join(self.root, split, "images")
        mask_dir = os.path.join(self.root, split, mask_dir)
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise RuntimeError(
                "Dataset not found or incomplete. "
                "Expected directories: {}/images and {}/<mask_dir>".format(split, split)
            )

        image_names = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        image_names.sort()

        self.images = []
        self.masks = []
        for image_name in image_names:
            stem = os.path.splitext(image_name)[0]
            mask_path = os.path.join(mask_dir, stem + ".png")
            if not os.path.isfile(mask_path):
                continue
            self.images.append(os.path.join(image_dir, image_name))
            self.masks.append(mask_path)

        if not self.images:
            raise RuntimeError("No valid image-mask pairs found in {}".format(self.root))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transform is not None:
            image, target = self.transform(image, target)

        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, target):
        target = np.asarray(target)

        # If mask is already train-id format (HxW), keep it.
        if target.ndim == 2:
            encoded = target
        # RGB mask format (HxWx3): map color to train id.
        elif target.ndim == 3 and target.shape[-1] == 3:
            encoded = np.zeros(target.shape[:2], dtype=np.uint8)
            encoded[(target[:, :, 0] == 22) & (target[:, :, 1] == 244) & (target[:, :, 2] == 22)] = 1
            encoded[(target[:, :, 0] == 121) & (target[:, :, 1] == 234) & (target[:, :, 2] == 249)] = 2
            encoded[(target[:, :, 0] == 243) & (target[:, :, 1] == 170) & (target[:, :, 2] == 161)] = 3
            encoded[(target[:, :, 0] == 248) & (target[:, :, 1] == 169) & (target[:, :, 2] == 227)] = 4
            encoded[(target[:, :, 0] == 127) & (target[:, :, 1] == 107) & (target[:, :, 2] == 114)] = 5
            encoded[(target[:, :, 0] == 18) & (target[:, :, 1] == 155) & (target[:, :, 2] == 187)] = 6
            encoded[(target[:, :, 0] == 61) & (target[:, :, 1] == 61) & (target[:, :, 2] == 245)] = 7
        else:
            raise ValueError("Unsupported mask shape for cottonweed: {}".format(target.shape))

        encoded = np.array(encoded, copy=True, dtype=np.uint8)
        encoded[encoded == 255] = 0
        return encoded

    @classmethod
    def decode_target(cls, mask):
        mask = np.array(mask, copy=True)
        mask[mask == 255] = 0
        mask = np.clip(mask, 0, cls.cmap.shape[0] - 1)
        return cls.cmap[mask]
