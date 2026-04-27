#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path('/root/autodl-tmp/dataset/cottonweedV4_train1k')
SPLITS = ['train', 'val', 'test']

# RGB -> trainId(4-class): 0 background, 1 cotton, 2 abuth, 3 others
COLOR_TO_ID = {
    (0, 0, 0): 0,
    (22, 244, 22): 1,      # cotton
    (121, 234, 249): 2,    # abuth
    (243, 170, 161): 3,
    (248, 169, 227): 3,
    (127, 107, 114): 3,
    (18, 155, 187): 3,
    (61, 61, 245): 3,
}


def encode_rgb_mask_to_4class(mask_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)

    # Unknown colors default to background(0)
    for color, cls_id in COLOR_TO_ID.items():
        m = (
            (mask_rgb[:, :, 0] == color[0])
            & (mask_rgb[:, :, 1] == color[1])
            & (mask_rgb[:, :, 2] == color[2])
        )
        out[m] = cls_id
    return out


def main():
    for split in SPLITS:
        src_dir = ROOT / split / 'masks'
        dst_dir = ROOT / split / 'masks_trainid4'
        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for p in sorted(src_dir.glob('*.png')):
            arr = np.array(Image.open(p).convert('RGB'))
            out = encode_rgb_mask_to_4class(arr)
            Image.fromarray(out, mode='L').save(dst_dir / p.name)
            count += 1
        print(f'{split}: converted {count} masks -> {dst_dir}')


if __name__ == '__main__':
    main()
