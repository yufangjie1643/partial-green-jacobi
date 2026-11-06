"""Minimal loader for the frozen kernels, without missing historical drivers.

Definitions are compiled verbatim from the archived Python AST. The only
precision adapter is the same FP64 substitution used by precision_adapter.py.
No file in solver/ is edited. New stopping/initial-residual handling is shared
by all methods and its true residual is recorded independently.
"""
from __future__ import annotations
import ast
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import math
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / 'solver'
ARCHIVE = ROOT / 'exp/asg_independent_20261030/results'
os.environ.setdefault('MAX_JOBS', '2')
os.environ['PATH'] = str(Path(sys.executable).parent) + ':/usr/local/cuda/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, str(SOLVER))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def definitions(relative, names, extra=None):
    path = SOLVER / relative
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    scope = dict(np=np, torch=torch, nn=nn, Path=Path, sys=sys, os=os,
                 lru_cache=lru_cache, MethodType=MethodType, SimpleNamespace=SimpleNamespace,
                 HERE=path.parent, __file__=str(path))
    scope.update(extra or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class Backend:
    def __init__(self):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.C0 = np.zeros((6, 6)); self.C0[:3, :3] = 4/3
        self.C0[np.arange(3), np.arange(3)] += 2
        self.C0[np.arange(3, 6), np.arange(3, 6)] = 1
        self.offsets = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                                 [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        from itertools import product
        self.geometry = definitions('src/ladecky/ladecky_operator.py', ['gradients'],
                                    dict(OFFSETS=self.offsets, product=product))
        self.hybrid = definitions('src/materials/hybrid_adapter.py', ['extension', 'HybridFIOperator'])
        self.formula = definitions('src/reference_medium/formula_adapter.py',
                                  ['extension','geometry','tables','enable_fi_reference'],
                                  dict(PIPE=SOLVER/'src/assembly_pipeline'))
        spectral = SOLVER/'src/spectral'; sys.path.insert(0, str(spectral))
        self.native = module('native_adapter', spectral/'native_adapter.py')
        self.precision = module('topopt_precision', spectral/'precision_adapter.py')
        contest = self.native.c
        contest.extension = lambda: self.precision.extension('kernels', 'fp64')
        self.native.extension = lambda: self.precision.extension('native', 'fp64')
        source = inspect.getsource(contest.attach)
        assert source.count('op._fft_symbol_inv.dtype != torch.float32') == 1
        source = source.replace('op._fft_symbol_inv.dtype != torch.float32',
                                'op._fft_symbol_inv.dtype != torch.float64')
        scope = dict(vars(contest)); exec(compile(source, '<frozen-fp64-adapter>', 'exec'), scope)
        contest.attach = scope['attach']
        self.pcg = module('topopt_pcg', SOLVER/'src/runtime/fi/source/femlib/batched/pcg.py')
        ed = definitions('experiments/smooth_contrast/study.py', ['element_diagonal'],
                         dict(gradients=self.geometry.gradients, C0=self.C0))
        s = SimpleNamespace(C0=self.C0, OFFSETS=self.offsets, element_diagonal=ed.element_diagonal,
                            HybridFIOperator=self.hybrid.HybridFIOperator,
                            comp=SimpleNamespace(cs=SimpleNamespace(
                                s=SimpleNamespace(f=SimpleNamespace(formula=self.formula)),
                                a=SimpleNamespace(native=self.native))))
        self.frozen = definitions('experiments/adaptive_green_scaling/study.py',
                                  ['build','diagonal','set_alpha'], dict(s=s))
        self.features = module('topopt_features', SOLVER/'experiments/global_alpha_stats/features.py')
        self.extract = torch.compile(self.features.extract, fullgraph=True)
        models = definitions('experiments/asg_independent/train.py', ['Model','load_model'])
        freeze = json.loads((ARCHIVE/'MODELS_FROZEN.json').read_text())
        selected = freeze['selected'][freeze['primary']]
        self.model_file = ARCHIVE/'models'/selected['name']/'model.pt'
        assert hashlib.sha256(self.model_file.read_bytes()).hexdigest() == selected['model_sha256']
        self.net = models.load_model(ARCHIVE, selected['name'])
        self.threshold = selected['threshold']
        self.packed0 = torch.tensor([self.C0[i,j] for i in range(6) for j in range(i,6)], device='cuda')
        self.loads = np.eye(6)[3:]

    def build(self, a):
        return self.frozen.build(np.asarray(a, dtype=np.float64))

    def update(self, op, a):
        field = torch.as_tensor(a, dtype=torch.float64, device='cuda')
        op._material_field.copy_(field)
        op._hybrid_coeff.copy_(field.reshape(-1,1)*self.packed0)
        op._diagonal = op._log_diagonal = None

    def select(self, op, method):
        if method == 'green': alpha = 0.
        elif method == 'gj': alpha = 1.
        elif method == 'asg':
            pred = self.net(self.extract(op._material_field)).reshape(())
            alpha = float(torch.where(pred < self.threshold, torch.zeros_like(pred), pred))
        else: alpha = float(method)
        self.frozen.set_alpha(op, alpha)
        return alpha

    def solve(self, op, rhs, x0=None, rtol=1e-8, maxiter=4000):
        norm = float(torch.linalg.vector_norm(rhs))
        # Solid shear stress * nodal face area * sqrt(number of nodes).
        atol = 1e-13 / math.sqrt(op.mesh.Nx)
        target = max(atol, rtol*norm)
        u = torch.zeros_like(rhs) if x0 is None else op.project_zero_mean(x0)
        absolute = float(torch.linalg.vector_norm(rhs-op.apply(u)))
        iterations = restarts = 0
        if absolute > target:
            while iterations < maxiter:
                u, it, _ = self.pcg.solve_pcg_batched(
                    op, rhs, x0=u, tol=target/max(norm, 1e-30), max_iter=maxiter-iterations,
                    update_mode='compiled', convergence_check_stride=4)
                iterations += it
                u = op.project_zero_mean(u)
                absolute = float(torch.linalg.vector_norm(rhs-op.apply(u)))
                if absolute <= target or not np.isfinite(absolute) or restarts >= 2: break
                restarts += 1
        info = dict(iterations=iterations, restarts=restarts, rhs_norm=norm,
                    absolute_residual=absolute, relative_residual=absolute/max(norm, 1e-300),
                    threshold=target, converged=bool(np.isfinite(absolute) and absolute <= target))
        return u, info

    @lru_cache(None)
    def energy_extension(self):
        from torch.utils.cpp_extension import load
        return load(name='asg_topopt_energy_20261103', sources=[str(Path(__file__).with_name('energy.cu'))],
                    extra_include_paths=[str(SOLVER/'src/materials')],
                    extra_cuda_cflags=['-O3'], extra_cflags=['-O3'], verbose=False)

    @lru_cache(None)
    def affine(self, n):
        offsets = np.array([[a&1,(a>>1)&1,(a>>2)&1] for a in range(8)])/n
        tensors = []
        for macro in self.loads:
            e = np.array([[macro[0],macro[5]/2,macro[4]/2],
                          [macro[5]/2,macro[1],macro[3]/2],
                          [macro[4]/2,macro[3]/2,macro[2]]])
            tensors.append((offsets@e.T).reshape(-1))
        return torch.tensor(np.array(tensors),device='cuda',dtype=torch.float64)

    def energies(self, op, solutions):
        n = op.mesh.Nx
        return self.energy_extension().evaluate(torch.cat(solutions,dim=0).contiguous(),
                                               self.packed0,self.affine(n),n)

    def provenance(self):
        paths = [p for p in SOLVER.rglob('*') if p.suffix in ('.py','.cu','.cuh') and '.cache' not in p.parts]
        return dict(model_sha256=hashlib.sha256(self.model_file.read_bytes()).hexdigest(),
                    threshold=self.threshold,
                    frozen_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    adapter='Verbatim AST definitions; FP64 contest substitution matches frozen precision_adapter; missing historical driver imports bypassed.')
