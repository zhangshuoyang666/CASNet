#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import network


def main():
    model_name = 'deeplabv3plus_mobilenet'
    num_classes = 4
    output_stride = 16
    attention_type = 'spatial_cbam'
    use_saff = True
    enable_boundary_aux = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.modeling.__dict__[model_name](
        num_classes=num_classes,
        output_stride=output_stride,
        attention_type=attention_type,
        use_saff=use_saff,
        enable_boundary_aux=enable_boundary_aux,
    ).to(device)
    model.eval()

    # Params
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs/GFLOPs (optional)
    gflops = None
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 513, 513, device=device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = float(macs) * 2.0 / 1e9
    except Exception:
        gflops = None

    # FPS benchmark
    dummy = torch.randn(1, 3, 513, 513, device=device)
    warmup = 20
    iters = 120
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
    fps = iters / max(t1 - t0, 1e-12)

    out = {
        'model': model_name,
        'num_classes': num_classes,
        'output_stride': output_stride,
        'attention_type': attention_type,
        'use_saff': use_saff,
        'enable_boundary_aux': enable_boundary_aux,
        'params': int(params),
        'trainable_params': int(trainable_params),
        'GFLOPs': gflops,
        'FPS': float(fps),
        'input_size': [1, 3, 513, 513],
        'device': str(device),
    }

    exp_root = '/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-masterPLUS/5cottonweedV4（cotton-abuth-others）'
    os.makedirs(exp_root, exist_ok=True)
    out_json = os.path.join(exp_root, 'model_stats.json')
    out_txt = os.path.join(exp_root, 'model_stats.txt')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f"model: {out['model']}\n")
        f.write(f"params: {out['params']}\n")
        f.write(f"trainable_params: {out['trainable_params']}\n")
        f.write(f"GFLOPs: {out['GFLOPs']}\n")
        f.write(f"FPS: {out['FPS']:.4f}\n")
        f.write(f"device: {out['device']}\n")
        f.write(f"input_size: {out['input_size']}\n")

    print('Saved:', out_json)
    print('Saved:', out_txt)


if __name__ == '__main__':
    main()
