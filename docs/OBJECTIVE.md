# Training objective

The classification term is a class-weighted focal loss with exponent 2 and label
smoothing 0.1. Class weights are inverse training-class frequencies, normalised to
mean one. The smoothed target is (1-epsilon) times the one-hot label plus epsilon/7.

For a positive auxiliary weight, add half the sum of image-to-prototype weighted
cross entropy and prototype-to-image multi-positive divergence, multiplied by
lambda. The reverse term averages log probabilities over all same-class images
and subtracts log of their count. Classes absent from the batch do not contribute.
See `src/skinclip/losses.py` for the exact equations as executable operations.

Selected F11 and F10 use lambda=1. F01, F00, MetaBlock, the linear probe and the
metadata-only comparator use lambda=0. Adam uses coupled weight decay, excluding
the learned logit-scale parameter. Gradients are clipped at norm 1.
