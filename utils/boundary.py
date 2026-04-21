import torch
import torch.nn.functional as F


def _morph_boundary(binary_mask, radius):
    if radius <= 0:
        return torch.zeros_like(binary_mask)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(binary_mask, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = -F.max_pool2d(-binary_mask, kernel_size=kernel_size, stride=1, padding=radius)
    return (dilated - eroded > 0).float()


def generate_boundary_target(mask, num_classes, boundary_width=3, ignore_index=255):
    """
    Generate class-aware boundary band from semantic mask.

    Args:
        mask: [B, H, W] or [H, W], int class ids.
        num_classes: class count (include background).
        boundary_width: boundary radius in pixels.
        ignore_index: ignored label id.

    Returns:
        boundary map with shape [B, 1, H, W], float in {0, 1}.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError("mask must have shape [B,H,W] or [H,W], got {}".format(tuple(mask.shape)))

    b, h, w = mask.shape
    device = mask.device
    boundary = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32)
    valid = (mask != ignore_index)

    for class_id in range(num_classes):
        class_region = ((mask == class_id) & valid).float().unsqueeze(1)
        class_boundary = _morph_boundary(class_region, boundary_width)
        boundary = torch.maximum(boundary, class_boundary)

    boundary = boundary * valid.unsqueeze(1).float()
    return boundary
