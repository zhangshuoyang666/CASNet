#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from PIL import Image

SRC_ROOT = Path('/root/autodl-tmp/dataset/cottonweedV4_train1k')
DST_ROOT = Path('/root/autodl-tmp/dataset/cottonweedV4_train1k_4class')

# RGB colors in original masks
COLOR_BG = np.array([0, 0, 0], dtype=np.uint8)
COLOR_COTTON = np.array([22, 244, 22], dtype=np.uint8)
COLOR_ABUTH = np.array([121, 234, 249], dtype=np.uint8)
COLOR_OTHERS = np.array([243, 170, 161], dtype=np.uint8)  # required unified others color

SPLITS = ['train', 'val', 'test']


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def map_mask_to_4class(mask_rgb: np.ndarray):
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    bg = np.all(mask_rgb == COLOR_BG, axis=-1)
    cotton = np.all(mask_rgb == COLOR_COTTON, axis=-1)
    abuth = np.all(mask_rgb == COLOR_ABUTH, axis=-1)

    out[bg] = COLOR_BG
    out[cotton] = COLOR_COTTON
    out[abuth] = COLOR_ABUTH

    others = ~(bg | cotton | abuth)
    out[others] = COLOR_OTHERS
    return out


def process_split(split: str):
    src_img_dir = SRC_ROOT / split / 'images'
    src_mask_dir = SRC_ROOT / split / 'masks'

    dst_img_dir = DST_ROOT / split / 'images'
    dst_mask_dir = DST_ROOT / split / 'masks_4class'
    ensure_dir(dst_img_dir)
    ensure_dir(dst_mask_dir)

    image_files = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    mapped = 0

    for img_path in image_files:
        stem = img_path.stem
        mask_path = src_mask_dir / f'{stem}.png'
        if not mask_path.exists():
            continue

        # hard-link/copy image
        dst_img = dst_img_dir / img_path.name
        if not dst_img.exists():
            try:
                os.link(img_path, dst_img)
            except OSError:
                dst_img.write_bytes(img_path.read_bytes())

        mask = np.array(Image.open(mask_path).convert('RGB'))
        mapped_mask = map_mask_to_4class(mask)
        Image.fromarray(mapped_mask).save(dst_mask_dir / f'{stem}.png')
        mapped += 1

    return len(image_files), mapped


def write_class_names():
    class_file = DST_ROOT / 'class_names_4class.txt'
    class_file.write_text('background\ncotton\nabuth\nothers\n', encoding='utf-8')


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f'Source dataset not found: {SRC_ROOT}')

    ensure_dir(DST_ROOT)
    summary = []
    for split in SPLITS:
        total, mapped = process_split(split)
        summary.append((split, total, mapped))

    write_class_names()

    summary_path = DST_ROOT / 'prepare_4class_summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write('source_root=' + str(SRC_ROOT) + '\n')
        f.write('target_root=' + str(DST_ROOT) + '\n')
        f.write('others_rgb=243,170,161\n')
        for split, total, mapped in summary:
            f.write(f'{split}: images={total}, mapped_pairs={mapped}\n')

    print('Done.')
    for split, total, mapped in summary:
        print(f'{split}: images={total}, mapped_pairs={mapped}')
    print('Summary:', summary_path)


if __name__ == '__main__':
    main()
