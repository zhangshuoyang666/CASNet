from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]


class WarmupPolyLR(_LRScheduler):
    """PolyLR with linear warmup: ramps from start_factor*base_lr to base_lr
    over warmup_iters, then follows Poly decay for the remaining iters."""
    def __init__(self, optimizer, max_iters, warmup_iters, power=0.9,
                 start_factor=1e-3, last_epoch=-1, min_lr=1e-6):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.power = power
        self.start_factor = start_factor
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / max(self.warmup_iters, 1)
            factor = self.start_factor + (1.0 - self.start_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            t = self.last_epoch - self.warmup_iters
            T = max(self.max_iters - self.warmup_iters, 1)
            return [max(base_lr * (1 - t / T) ** self.power, self.min_lr)
                    for base_lr in self.base_lrs]