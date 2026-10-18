import math

import torch
import torch.nn.functional as F

from skinclip.losses import class_weights, focal_loss, i2t_loss, t2i_loss, total_loss


def naive_t2i(a, y):
    B, K = a.shape
    vals = []
    for c in range(K):
        pos = [j for j in range(B) if y[j] == c]
        if not pos:
            continue
        logrho = a[:, c] - torch.logsumexp(a[:, c], 0)
        vals.append(-sum(logrho[j] for j in pos) / len(pos) - math.log(len(pos)))
    return torch.stack(vals).mean()


def test_t2i_matches_naive_loop():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        a = torch.randn(17, 7, generator=g) * 5
        y = torch.randint(0, 7, (17,), generator=g)
        assert torch.allclose(t2i_loss(a, y), naive_t2i(a, y), atol=1e-6)


def test_t2i_permutation_invariant_and_nonnegative():
    a = torch.randn(32, 7) * 3
    y = torch.randint(0, 7, (32,))
    p = torch.randperm(32)
    assert torch.allclose(t2i_loss(a, y), t2i_loss(a[p], y[p]), atol=1e-6)
    onehot = F.one_hot(y, 7).float()
    n = onehot.sum(0)
    logrho = a - torch.logsumexp(a, 0, keepdim=True)
    Lc = -(onehot * logrho).sum(0)[n > 0] / n[n > 0] - torch.log(n[n > 0])
    assert (Lc >= -1e-6).all()


def test_t2i_zero_under_perfect_separation():
    y = torch.tensor([0, 0, 1, 2, 2, 2])
    a = F.one_hot(y, 3).float() * 200.0
    assert t2i_loss(a, y).item() < 1e-5


def test_single_positive_and_single_class_batches_are_finite():
    a = torch.randn(8, 7, requires_grad=True)
    for y in (torch.tensor([3, 0, 0, 0, 0, 0, 0, 0]), torch.zeros(8, dtype=torch.long)):
        L = t2i_loss(a, y)
        L.backward()
        assert torch.isfinite(L) and torch.isfinite(a.grad).all()
        a.grad = None


def test_i2t_uniform_weights_equals_cross_entropy():
    a, y = torch.randn(10, 7), torch.randint(0, 7, (10,))
    assert torch.allclose(i2t_loss(a, y, torch.ones(7)), F.cross_entropy(a, y), atol=1e-6)


def test_class_weights_mean_one_and_inverse_frequency():
    w = class_weights([0] * 70 + [1] * 20 + [2] * 10, 3)
    assert abs(w.mean().item() - 1) < 1e-6 and w[2] > w[1] > w[0]


def test_focal_reduces_to_smoothed_ce_when_gamma_zero():
    a, y = torch.randn(12, 7), torch.randint(0, 7, (12,))
    ref = F.cross_entropy(a, y, label_smoothing=0.1)
    assert torch.allclose(focal_loss(a, y, torch.ones(7), gamma=0.0, smoothing=0.1), ref, atol=1e-6)


def test_lambda_zero_has_no_contrastive_term():
    a, y = torch.randn(12, 7), torch.randint(0, 7, (12,))
    L0, parts = total_loss(a, y, torch.ones(7), 0.0, 2.0, 1.0, 0.1)
    assert math.isnan(parts["loss_clip"]) and torch.allclose(L0, focal_loss(a, y, torch.ones(7)))
