"""Full-grid periodic statistics. No voxel downsampling or class metadata."""
import math
import torch
from torch import nn

CDF = (.01, .05, .25, .5, .75, .95, .99)
SCALES = ('h', '2h', 'L16', 'L8')
MOMENTS = ('s1', 's2', 's4', 'maximum', 'active', 'tail025')
NAMES = ['mean', 'variance'] + [f'cdf_{t:g}' for t in CDF]
NAMES += [f'{axis}_{scale}_{moment}' for axis in 'zyx' for scale in SCALES for moment in MOMENTS]
NAMES += [f'{axis}_{kind}' for axis in 'zyx' for kind in ('beta', 'jump_ratio')]
NAMES += ['gradient_zy', 'gradient_zx', 'gradient_yx', 'log_contrast', 'h_over_L']
assert len(NAMES) == 92


def extract(rho):
    ell = rho.float().log()
    lo, hi = torch.aminmax(ell)
    span = hi - lo
    q = (ell - lo) / span.clamp_min(1e-12)
    n = q.shape[0]
    assert q.shape == (n, n, n) and n >= 32 and n % 16 == 0
    mean = q.mean()
    values = [mean, (q - mean).square().mean()]
    values += [(q <= t).float().mean() for t in CDF]
    pairs, gradients = [], []
    for axis in range(3):
        first = []
        for shift in (1, 2, max(1, n // 16), max(1, n // 8)):
            diff = torch.roll(q, -shift, axis) - q
            d = diff.abs()
            s1, s2 = d.mean(), d.square().mean()
            values.extend((s1, s2, d.square().square().mean(), d.amax(),
                           (d > 1e-6).float().mean(), (d > .25).float().mean()))
            first.append((s1, s2))
            if shift == 1:
                gradients.append(diff)
        s1, s2 = first[0]
        beta = torch.where(s2 > 1e-20, (first[1][1] / s2.clamp_min(1e-20)).clamp_min(1e-20).log2(), 0.)
        pairs.extend((beta, s2 / s1.clamp_min(1e-20)))
    values.extend(pairs)
    norm = sum(g.square().mean() for g in gradients).clamp_min(1e-20)
    values.append((gradients[0] * gradients[1]).mean() / norm)
    values.extend(((gradients[0] * gradients[2]).mean() / norm,
                   (gradients[1] * gradients[2]).mean() / norm,
                   span / math.log(10000.), q.new_tensor(1. / n)))
    return torch.stack(values).reshape(1, -1)


def formula_inputs(x):
    """Ten named physical summaries; quadratic ridge is an explicit formula."""
    i = {name: k for k, name in enumerate(NAMES)}
    get = lambda name: x[..., i[name]]
    beta = torch.stack([get(f'{a}_beta') for a in 'zyx'], -1).mean(-1)
    jump = torch.stack([get(f'{a}_jump_ratio') for a in 'zyx'], -1).mean(-1)
    active = torch.stack([get(f'{a}_h_active') for a in 'zyx'], -1).mean(-1)
    directional = torch.stack([get(f'{a}_h_s2') for a in 'zyx'], -1)
    directions = directional / directional.sum(-1, keepdim=True).clamp_min(1e-20)
    return torch.stack((get('mean'), get('variance'), get('cdf_0.95') - get('cdf_0.05'),
                        beta, jump, active, directions[..., 0], directions[..., 1],
                        get('gradient_yx'), get('log_contrast')), -1)


FORMULA_NAMES = ['q_mean', 'q_variance', 'fraction_005_095', 'beta_mean', 'jump_mean',
                 'active_mean', 's2_z_fraction', 's2_y_fraction', 'gradient_yx', 'log_contrast_normalized']


def polynomial(x, degree):
    columns = [torch.ones_like(x[..., 0])] + [x[..., j] for j in range(x.shape[-1])]
    if degree == 2:
        columns += [x[..., i] * x[..., j] for i in range(x.shape[-1]) for j in range(i, x.shape[-1])]
    return torch.stack(columns, -1)


class StatsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(92))
        self.register_buffer('std', torch.ones(92))
        self.net = nn.Sequential(nn.Linear(92, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
                                 nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class ExplicitFormula(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.degree = config['degree']
        for name in ('mean', 'std', 'coefficients'):
            self.register_buffer(name, torch.tensor(config[name], dtype=torch.float64))

    def forward(self, x):
        z = (formula_inputs(x).double() - self.mean) / self.std
        return (polynomial(z, self.degree) @ self.coefficients).clamp(0, 1).reshape(-1, 1).float()
