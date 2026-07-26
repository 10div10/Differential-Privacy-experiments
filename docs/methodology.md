# Methodology Notes

## What DP-SGD actually changes vs. plain SGD

Standard SGD: compute the gradient of the loss over a batch, average it,
step the optimizer.

DP-SGD (Abadi et al., 2016) does three additional things per step:

1. **Per-sample gradient computation.** Instead of one averaged gradient
   for the batch, compute a separate gradient for every individual
   example. This is what breaks BatchNorm (its statistics depend on the
   whole batch, not a single sample) and is also why DP-SGD is more
   compute- and memory-hungry than plain SGD — this is the main source of
   the training-time overhead this project measures.
2. **Per-sample gradient clipping.** Clip each per-sample gradient to a
   maximum L2 norm `C` (`--max-grad-norm` in `train.py`). This bounds how
   much any single example can influence the aggregated gradient,
   regardless of outliers.
3. **Noise addition.** Add Gaussian noise (scaled by a "noise multiplier"
   σ, calibrated to `C`) to the sum of clipped per-sample gradients before
   averaging and stepping the optimizer.

The noise is what provides the formal privacy guarantee; the clipping is
what makes that guarantee possible to state at all (without a bound on
per-example influence, you can't bound how much noise is "enough").

## What ε and δ mean here

(ε, δ)-differential privacy bounds how much the output distribution of the
training algorithm can differ between two datasets that differ in exactly
one example. Informally:

- **ε (epsilon)** is the privacy budget — smaller means a stronger
  guarantee (harder to tell whether any specific example was in the
  training set). ε is on a log-ish scale in practice: the difference
  between ε=1 and ε=8 is much larger than it looks numerically.
- **δ (delta)** is the probability the guarantee fails entirely (e.g., a
  catastrophic case where privacy is violated with certainty). It's
  conventionally set very small (here, 1e-5) and roughly should be less
  than 1/(dataset size).

## Why "target epsilon" and "realized epsilon" can differ slightly

`train.py` uses Opacus's `make_private_with_epsilon`, which picks a noise
multiplier σ such that, after the full planned number of steps
(epochs × batches/epoch), the RDP accountant reports a cumulative ε at or
near the target. The realized ε logged at the end of training is read
directly from the accountant and should match the target closely — small
deviations come from the accountant's search resolution over noise
multipliers.

## Accounting method

Opacus's default accountant here is **RDP (Rényi Differential Privacy)**
accounting, which tracks privacy loss across many composed steps more
tightly than naive composition would. An alternative is the **PRV
(Privacy Random Variable)** accountant, which can give a tighter bound in
some regimes at higher computational cost. If reporting these numbers in
a context where the exact accountant matters (e.g., a paper or a rigorous
audit), name the accountant explicitly — "ε=3 under RDP accounting" is a
meaningfully different (and non-comparable) claim from "ε=3 under PRV
accounting."

## Reference

Abadi, M. et al. "Deep Learning with Differential Privacy." CCS 2016.
(The paper that introduced DP-SGD and the moments accountant that Opacus's
RDP accountant descends from.)
