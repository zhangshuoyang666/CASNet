import argparse
import json
import os
import random
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SoyCotton COCO annotations for DeepLab cottonweed loader")
    parser.add_argument("--coco_json", type=str, required=True, help="path to COCO json")
    parser.add_argument("--images_dir", type=str, required=True, help="directory containing source images")
    parser.add_argument("--output_root", type=str, required=True, help="output dataset root")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed for split")
    parser.add_argument("--copy_images", action="store_true", default=False, help="copy images instead of symlink")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def link_or_copy(src, dst, copy_images):
    if os.path.lexists(dst):
        os.remove(dst)
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def ann_to_mask(ann, height, width):
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((height, width), dtype=bool)

    # Polygon format
    if isinstance(seg, list):
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        for poly in seg:
            if not poly or len(poly) < 6:
                continue
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(xy, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8).astype(bool)

    # RLE format
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required for RLE segmentation. Please install pycocotools.") from exc

    rle = seg
    if isinstance(rle.get("counts", None), list):
        rle = mask_utils.frPyObjects([rle], height, width)[0]
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def write_classes_txt(categories, output_root):
    class_names = ["background"] + [c["name"] for c in categories]
    class_path = os.path.join(output_root, "classes.txt")
    with open(class_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    return class_names


def main():
    args = parse_args()

    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = sorted(coco.get("categories", []), key=lambda x: x["id"])
    if not images or not categories:
        raise RuntimeError("Invalid COCO json: missing images or categories.")

    ensure_dir(args.output_root)
    class_names = write_classes_txt(categories, args.output_root)

    cat_id_to_train_id = {cat["id"]: i + 1 for i, cat in enumerate(categories)}

    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["image_id"]].append(ann)

    missing = []
    for img in images:
        img_path = os.path.join(args.images_dir, img["file_name"])
        if not os.path.isfile(img_path):
            missing.append(img["file_name"])
    if missing:
        raise RuntimeError(
            "Missing {} images in images_dir: {}\nExpected images under: {}".format(
                len(missing), ", ".join(missing[:5]), args.images_dir
            )
        )

    image_ids = [img["id"] for img in images]
    random.Random(args.seed).shuffle(image_ids)
    val_count = int(len(image_ids) * args.val_ratio)
    if args.val_ratio > 0 and val_count == 0 and len(image_ids) > 1:
        val_count = 1
    val_ids = set(image_ids[:val_count])

    split_map = {}
    for image_id in image_ids:
        split_map[image_id] = "val" if image_id in val_ids else "train"

    for split in ["train", "val"]:
        ensure_dir(os.path.join(args.output_root, split, "images"))
        ensure_dir(os.path.join(args.output_root, split, "masks_trainid"))

    for img in images:
        image_id = img["id"]
        split = split_map[image_id]
        width = int(img["width"])
        height = int(img["height"])
        file_name = img["file_name"]
        stem = os.path.splitext(os.path.basename(file_name))[0]

        src_img = os.path.join(args.images_dir, file_name)
        dst_img = os.path.join(args.output_root, split, "images", os.path.basename(file_name))
        link_or_copy(src_img, dst_img, args.copy_images)

        mask = np.zeros((height, width), dtype=np.uint8)
        anns = ann_by_image.get(image_id, [])
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id not in cat_id_to_train_id:
                continue
            train_id = cat_id_to_train_id[cat_id]
            ann_mask = ann_to_mask(ann, height, width)
            mask[ann_mask] = train_id

        mask_path = os.path.join(args.output_root, split, "masks_trainid", stem + ".png")
        Image.fromarray(mask, mode="L").save(mask_path)

    print("Prepared dataset at: {}".format(args.output_root))
    print("Class names: {}".format(class_names))
    print("Train images: {}".format(len(image_ids) - len(val_ids)))
    print("Val images: {}".format(len(val_ids)))


if __name__ == "__main__":
    main()
