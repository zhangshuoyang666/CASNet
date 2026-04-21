# Next Actions

## Current Status
- Starting serial optimization from baseline (mIoU=94.66%)
- Step 1: Learning rate screening with bs=32

## Decision Criteria
1. **update_best**: mIoU > current best by >0.15%
2. **marginal_gain**: 0% < gain <= 0.15%
3. **reject**: gain <= 0% or curves unstable
4. **fail_fast**: clearly worse in first 30% of training
5. **promote_to_confirm**: short screen looks promising, run full 1600 iters

## Pending Steps
- [ ] Step 1: LR screening (4 candidates)
- [ ] Step 1c: LR confirmation
- [ ] Step 2: Warmup
- [ ] Step 3.1: Vertical flip
- [ ] Step 3.2: Rotation ±30°
- [ ] Step 4: λ_edge
- [ ] Step 5: weight_decay
- [ ] Step 6: ASPP rates
