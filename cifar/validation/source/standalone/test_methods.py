"""Mathematical identities and old-implementation regression checks."""
import json
import hashlib
import math
import torch
from torch.nn import functional as F
import reference_train as old
from methods import angular_psi, circle_training_logits, create_model, train_logits, inference_logits, METHODS
from grid import ROOT, SOURCE, settings, pipeline, common_id
from train import learning_rate


def main():
    torch.set_num_threads(2)
    torch.manual_seed(314)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.06)
    checks = []
    # Angular polynomial equals the piecewise angular formula, with correct gradients.
    theta = torch.linspace(.011, math.pi - .011, 999, dtype=torch.float64)
    for m in (1, 2, 3, 4):
        k = (m * theta / math.pi).floor()
        expected = (-1.)**k * torch.cos(m * theta) - 2 * k
        torch.testing.assert_close(angular_psi(theta.cos(), m), expected)
        values = angular_psi(theta.cos(), m)
        assert torch.all(values[:-1] >= values[1:])
        c = torch.tensor([-.87, -.31, .19, .78], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: angular_psi(t, m), (c,))
    checks.append('L/Sphere angular formula, monotonicity and finite-difference gradients (m=1..4)')
    # Circle's CE formulation equals the paper's single-positive pair expression.
    c = torch.tensor([[.4, -.2, .1], [-.1, .5, .25]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 1]); m = .25; scale = 64.
    transformed = circle_training_logits(c, y, m, scale)
    rows = torch.arange(2)
    mask = F.one_hot(y, 3).bool()
    neg = transformed[~mask].reshape(2, 2)
    pos = transformed[rows, y]
    expected = F.softplus(torch.logsumexp(neg, 1) - pos).mean()
    actual = F.cross_entropy(transformed, y)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual, c)[0]
    alpha = (c.detach() + m).clamp_min(0)
    alpha[rows, y] = (1 + m - c.detach()[rows, y]).clamp_min(0)
    expected_grad = (transformed.detach().softmax(1) - F.one_hot(y, 3)) * scale * alpha / 2
    torch.testing.assert_close(grad, expected_grad)
    checks.append('Circle equation and detached adaptive-weight derivatives')
    # Existing five model initializations, inference, train losses and gradients.
    x = torch.randn(2, 3, 32, 32, device=device); labels = torch.tensor([7, 23], device=device)
    rx, ry = x.cpu(), labels.cpu()
    for method in METHODS[:5]:
        cfg = settings(method)[4] | dict(method=method)
        torch.manual_seed(101); before = old.create_model(method).cpu().eval()
        torch.manual_seed(101); after = create_model(method, 20).cpu().eval()
        assert before.state_dict().keys() == after.state_dict().keys()
        for name, value in before.state_dict().items():
            torch.testing.assert_close(value, after.state_dict()[name], rtol=0, atol=0)
        a = old.inference_logits(before, rx, cfg)
        b = inference_logits(after, rx, cfg)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if method in {'arcface', 'cosface'}:
            old_logits = old.margin_training_logits(before.cosine_logits(rx), ry, method, cfg['margin'], cfg['scale'])
        else:
            old_logits = a
        new_logits, _ = train_logits(after, rx, ry, cfg, 0)
        torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
        F.cross_entropy(old_logits, ry).backward(); F.cross_entropy(new_logits, ry).backward()
        for pa, pb in zip(before.parameters(), after.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
        del before, after
    checks.append('Existing five methods: exact CPU initialization/logits/loss-gradient regression, avoiding nondeterministic CUDA reductions')
    # Zero-margin endpoint identities; multiplicative m=1 includes both norms.
    for method in ['msoftmax', 'lsoftmax', 'sphereface', 'arcface', 'cosface']:
        cfg = settings(method)[3] | dict(method=method, margin=1 if method in {'lsoftmax', 'sphereface'} else 0)
        model = create_model(method, 20).to(device).eval()
        a, _ = train_logits(model, x, labels, cfg, 100000)
        b = inference_logits(model, x, cfg)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        del model
    checks.append('Zero additive / unit multiplicative margin endpoint identities')
    # Exercise deepest model gradients for every method without loading test data.
    for method in METHODS:
        cfg = settings(method)[common_id(method)] | dict(method=method)
        model = create_model(method, 110).to(device).train()
        for step in [0, 60000]:
            adjusted, _ = train_logits(model, x, labels, cfg, step)
            loss = F.cross_entropy(adjusted, labels)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.zero_grad(set_to_none=True)
        del model
    checks.append('ResNet-110 all 10 methods: finite forward/backward, initial and late margin annealing')
    for d in [20, 32, 56, 110]:
        cfg = pipeline(d) | dict(lr=.1)
        assert abs(learning_rate(cfg, cfg['milestones'][0], 0, 351) - .01) < 1e-12
        if d >= 56:
            assert .01 < learning_rate(cfg, 1, 0, 351) < .011
            assert abs(learning_rate(cfg, 5, 350, 351) - .1) < 1e-12
    checks.append('Per-depth warmup and learning-rate milestone endpoints')
    result = dict(passed=True, device=device, checks=checks,
                  source_sha256={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in
                                 ['grid.py','methods.py','train.py','resnet_cifar.py','reference_train.py']})
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'unit_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
