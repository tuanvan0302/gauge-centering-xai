"""Pair (difference-of-convex) arithmetic, the three methods of notebook nb02, and their cost accounting.

Methods (gauge-centering-xai/paper-idea.md, Sec. 5, group A):

A  naive recursive NDC pair (Paper1, Lemma 1): constant weights split as W = W+ - W-, ReLU lifted to
   (max(P, Q), Q), square rule at products of two variable factors, no stabilisation; input pair
   (relu(x), relu(-x)). Gradients by autograd through the pair.
B  Hidden Monotonicity as its official code runs it (DC-Decomposition-for-Explanability, commit 31be19b,
   config of configs/nb01/methods/splitcam_a40_sc.json): the input node 'x' outputs (x/2, -x/2), but the
   layers that read the input receive (x, 0) (hybrid_network.py, branch input_node_name == 'x'); after
   every chunk both streams are multiplied by theta when max|.| > Theta (threshold_scale_both_
   simultaneously), then moved so that P - Q equals the cached activation of the original network;
   products by the value-level rule (HM Eq. 19). Backward: four sensitivities started at (e/2, 0, 0, e/2),
   ReLU and max-pool patterns of the original network, after every chunk the alpha shift
   (adaptive_shift_independent) and then a correction to the cached original gradient. The official
   Conv2d backprop returns the two middle sensitivities swapped (direct_conv2d_split.py vs
   direct_linear_split.py); conv_swap=True reproduces it, conv_swap=False is the true derivative.
C  gauge-centred pair (Paper1, Prop. 17): P = f + Q, Q = C/2 ||x - x*||^2, evaluated at x* by a plain
   forward pass and autograd.

Two engines:
SeqNet  networks given as a list of chunks (ReLU MLP, VGG16); has the HM backward that B needs.
Ops     networks written once as a function of an arithmetic (gated net, attention); the same function
        evaluates the original network (torch or mpmath), the pair of A and the pair of B.
"""
import contextlib
import time
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input

U = 2.0 ** -53                       # unit roundoff of IEEE binary64
DT = torch.float64


# ------------------------------------------------------------------------------------- cost accounting

class Ledger:
    """Bytes of the distinct tensor storages a method holds: weights, cache and tensors saved by autograd.

    Storages are counted once. Every registered tensor stays referenced until the ledger is dropped, so a
    freed block cannot be counted twice under a reused address."""
    KINDS = ("weights", "cache", "autograd")

    def __init__(self):
        self._seen, self._keep = set(), []
        self.bytes = dict.fromkeys(self.KINDS, 0)

    def add(self, kind, *objs):
        for t in objs:
            if isinstance(t, dict):
                self.add(kind, *t.values())
            elif isinstance(t, (list, tuple)):
                self.add(kind, *t)
            elif torch.is_tensor(t):
                s = t.untyped_storage()
                key = (s.data_ptr(), s.nbytes())
                if key[1] == 0 or key in self._seen:
                    continue
                self._seen.add(key)
                self._keep.append(t)
                self.bytes[kind] += s.nbytes()

    @contextlib.contextmanager
    def autograd(self):
        """Count every tensor autograd saves for the backward pass while the block runs."""
        def pack(t):
            self.add("autograd", t)
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            yield

    def row(self):
        r = {f"mem_{k}": self.bytes[k] for k in self.KINDS}
        r["mem_total"] = sum(self.bytes.values())
        return r


def timed(fn, repeats):
    """Run fn() `repeats` times; return the last result and the median wall time in seconds."""
    ts, out = [], None
    for _ in range(max(1, repeats)):
        out = None                       # release the previous run (its ledger keeps tensors alive)
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


# --------------------------------------------------------------------------------------------- metrics

def rel_err(v, t):
    """|v - t| / |t| for a float64 result v and a truth t (float or mpmath); NaN if v is not finite."""
    v = float(v)
    if not np.isfinite(v):
        return float("nan")
    if isinstance(t, mp.mpf):
        return float(abs(mp.mpf(v) - t) / abs(t))
    return abs(v - float(t)) / abs(float(t))


def rel_err_vec(g, t):
    """||g - t||_2 / ||t||_2 for a float64 vector g and a truth t (float array or mpmath object array)."""
    g = np.asarray(g, dtype=float).ravel()
    if not np.all(np.isfinite(g)):
        return float("nan")
    t = np.asarray(t).ravel()
    if t.dtype == object:
        num = mp.fsum((mp.mpf(a) - b) ** 2 for a, b in zip(g, t))
        return float(mp.sqrt(num / mp.fsum(b * b for b in t)))
    return float(np.linalg.norm(g - t) / np.linalg.norm(t))


def chi_ratio(num, den):
    """chi = num / |den| as a float; inf if den == 0, NaN if num is not finite."""
    num, den = float(num), abs(float(den))
    if not np.isfinite(num):
        return float("nan")
    return num / den if den > 0 else float("inf")


def certified_depth(depths, errs, delta):
    """Largest depth such that the error is below delta at every depth up to it (0 if the first fails)."""
    best = 0
    for d, e in sorted(zip(depths, errs)):
        if not (np.isfinite(e) and e < delta):
            break
        best = d
    return best


def state_stats(p, m, a):
    """Norm-wise chi, invariant error and magnitude of a split pair (p, m) of the activation a."""
    with torch.no_grad():
        an = a.norm().item()
        return {"chi": chi_ratio((p.abs() + m.abs()).norm().item(), an),
                "err": chi_ratio(((p - m) - a).norm().item(), an),
                "maxabs": max(p.abs().max().item(), m.abs().max().item())}


# ------------------------------------------------------------------------------------ SeqNet: chunks

def _rep(t, n):
    """Repeat a per-image tensor n times along the batch (rows of g and rows of h are stacked)."""
    return t if n == 1 else t.repeat(n, *([1] * (t.dim() - 1)))


class Chunk:
    weighted = False

    def __init__(self, name):
        self.name = name

    def weights(self):
        return []

    def split_weights(self):
        return []


class Identity(Chunk):
    """The input placeholder 'x' and Dropout in eval mode."""

    def orig(self, a, pat):
        return a

    def split(self, p, m, pat):
        return p, m

    def back(self, dp, dm, pat):
        return dp, dm


class Affine(Chunk):
    """Conv2d or Linear with constant weights, split W = W+ - W-, b = b+ - b-."""
    weighted = True

    def __init__(self, name, mod=None, W=None, b=None, device="cpu"):
        super().__init__(name)
        if mod is not None:
            W, b = mod.weight, mod.bias
            self.conv = isinstance(mod, nn.Conv2d)
            if self.conv:
                self.kw = dict(stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups)
        else:
            self.conv = False
        self.W = W.detach().to(device, DT)
        self.b = None if b is None else b.detach().to(device, DT)
        self.Wp = self.W.clamp(min=0)
        self.Wm = self.Wp - self.W
        self.bp = None if self.b is None else self.b.clamp(min=0)
        self.bm = None if self.b is None else self.bp - self.b

    def weights(self):
        return [self.W] + ([] if self.b is None else [self.b])

    def split_weights(self):
        return [self.Wp, self.Wm] + ([] if self.b is None else [self.bp, self.bm])

    def _op(self, x, W):
        return F.conv2d(x, W, **self.kw) if self.conv else F.linear(x, W)

    def _opT(self, d, W, shape):
        return conv2d_input(shape, W, d, **self.kw) if self.conv else d @ W

    def _bv(self, b):
        return b.view(1, -1, 1, 1) if self.conv else b

    def orig(self, a, pat):
        z = self._op(a, self.W)
        return z if self.b is None else z + self._bv(self.b)

    def split(self, p, m, pat):
        n = p.shape[0]
        x = torch.cat([p, m])
        yp, ym = self._op(x, self.Wp), self._op(x, self.Wm)
        zp, zm = yp[:n] + ym[n:], ym[:n] + yp[n:]
        if self.b is not None:
            zp, zm = zp + self._bv(self.bp), zm + self._bv(self.bm)
        return zp, zm

    def back(self, dp, dm, pat):
        """Transpose of the pair map without bias: (W+^T dp + W-^T dm, W-^T dp + W+^T dm)."""
        n = dp.shape[0]
        shape = (2 * n,) + tuple(pat["in_shape"][1:])
        d = torch.cat([dp, dm])
        tp, tm = self._opT(d, self.Wp, shape), self._opT(d, self.Wm, shape)
        return tp[:n] + tm[n:], tm[:n] + tp[n:]


class ReLU(Chunk):
    """ReLU; on a pair it is the Maxout (max(z+, z-), z-)."""

    def orig(self, z, pat):
        pat["mask"] = z > 0              # same convention as autograd: relu'(0) = 0
        return F.relu(z)

    def split(self, p, m, pat):
        return torch.maximum(p, m), m

    def back(self, dp, dm, pat):
        mask = _rep(pat["mask"], dp.shape[0] // pat["mask"].shape[0]).to(dp.dtype)
        return dp * mask, dp * (1 - mask) + dm


class MaxPool(Chunk):
    """Max pooling in the convex form max_j(a+_j - a-_j) = max_j{a+_j + sum_{i!=j} a-_i} - sum_i a-_i
    (HM Sec. 3.4, use_monotone_maxpool=True)."""

    def __init__(self, name, mod):
        super().__init__(name)
        self.kw = dict(kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding, ceil_mode=mod.ceil_mode)

    def _sum(self, x):
        return F.avg_pool2d(x, divisor_override=1, **self.kw)

    def orig(self, a, pat):
        out, pat["idx"] = F.max_pool2d(a, return_indices=True, **self.kw)
        return out

    def split(self, p, m, pat):
        s = self._sum(m)
        return F.max_pool2d(p - m, **self.kw) + s, s

    def back(self, dp, dm, pat):
        n = dp.shape[0]
        idx = _rep(pat["idx"], n // pat["idx"].shape[0])
        shape = (n,) + tuple(pat["in_shape"][1:])
        sp = dp.new_zeros(shape[0], shape[1], shape[2] * shape[3]).scatter_add_(2, idx.flatten(2), dp.flatten(2))
        sp = sp.view(shape)
        _, sum_T = torch.func.vjp(self._sum, dp.new_zeros(shape))
        return sp, sum_T(dp + dm)[0] - sp


class AvgPool(Chunk):
    """Adaptive average pooling: a positive linear map applied to each stream."""

    def __init__(self, name, mod):
        super().__init__(name)
        self.size = mod.output_size

    def _f(self, x):
        return F.adaptive_avg_pool2d(x, self.size)

    def orig(self, a, pat):
        return self._f(a)

    def split(self, p, m, pat):
        return self._f(p), self._f(m)

    def back(self, dp, dm, pat):
        _, f_T = torch.func.vjp(self._f, dp.new_zeros((dp.shape[0],) + tuple(pat["in_shape"][1:])))
        return f_T(dp)[0], f_T(dm)[0]


class Flatten(Chunk):
    def orig(self, a, pat):
        return a.flatten(1)

    def split(self, p, m, pat):
        return p.flatten(1), m.flatten(1)

    def back(self, dp, dm, pat):
        shape = (-1,) + tuple(pat["in_shape"][1:])
        return dp.reshape(shape), dm.reshape(shape)


# ------------------------------------------------------------------------------------ SeqNet: network

class SeqNet:
    """A network as a list of chunks; chunk k maps state k to state k + 1 (state 0 is the input).
    Chunk names follow the torch.fx node names of the official HM code ('x', 'features_0', ...)."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.names = [c.name for c in chunks]

    @classmethod
    def from_vgg(cls, model, device="cpu"):
        chunks = [Identity("x")]
        for i, mod in enumerate(model.features):
            chunks.append(_chunk(f"features_{i}", mod, device))
        chunks += [AvgPool("avgpool", model.avgpool), Flatten("flatten")]
        for i, mod in enumerate(model.classifier):
            chunks.append(_chunk(f"classifier_{i}", mod, device))
        return cls(chunks)

    @classmethod
    def mlp(cls, Ws, w_out):
        """ReLU MLP without bias: x -> [lin_l, relu_l] for every W in Ws -> scalar readout w_out."""
        chunks = [Identity("x")]
        for l, W in enumerate(Ws):
            chunks += [Affine(f"lin{l + 1}", W=W), ReLU(f"relu{l + 1}")]
        chunks.append(Affine("out", W=w_out.reshape(1, -1)))
        return cls(chunks)

    def weights(self):
        return [t for c in self.chunks for t in c.weights()]

    def split_weights(self):
        return [t for c in self.chunks for t in c.split_weights()]

    # -- the original network (method C, and the cache of method B)
    def original(self, x, target=None, all_grads=True):
        """Forward and backward pass in float64. Returns states a[k], patterns pat[k], the target index,
        f = output[target] and grad[k] = df/da[k] for every state (all_grads) or only the input."""
        x = x.detach().to(DT).requires_grad_(True)
        states, pats, a = [x], [], x
        with torch.enable_grad():
            for c in self.chunks:
                pat = {"in_shape": tuple(a.shape)}
                a = c.orig(a, pat)
                states.append(a)
                pats.append(pat)
            out = a
            target = int(out[0].argmax()) if target is None else int(target)
            f = out[0, target]
            grads = torch.autograd.grad(f, states if all_grads else [x])
        return SimpleNamespace(a=[s.detach() for s in states], pat=pats, target=target, f=f.detach(),
                               grad=[g.detach() for g in grads], out=out.detach())

    # -- method A
    def forward_A(self, x, target, ref_states=None):
        """Naive pair from (relu(x), relu(-x)) with autograd. Returns P, Q of output[target], their input
        gradients and, if reference states are given, per-state statistics (measurement only)."""
        xv = x.detach().to(DT).requires_grad_(True)
        p, m = F.relu(xv), F.relu(-xv)
        stats = []
        with torch.enable_grad():
            for k, c in enumerate(self.chunks):
                p, m = c.split(p, m, None)
                if ref_states is not None:
                    stats.append({"state": k + 1, "name": c.name, **state_stats(p, m, ref_states[k + 1])})
            P, Q = p[0, target], m[0, target]
            gP, = torch.autograd.grad(P, xv, retain_graph=True)
            gQ, = torch.autograd.grad(Q, xv)
        return SimpleNamespace(P=P.detach(), Q=Q.detach(), gP=gP, gQ=gQ, stats=stats)

    # -- method B
    def forward_B(self, cache, theta, Theta, keep_pairs=False, measure=True):
        """HM forward: chunk 'x' turns the input into (x/2, -x/2), the next chunk reads (x, 0) as in the
        official code; after every chunk the threshold scaling and the correction to the cached activation.
        Returns the output pair, per-chunk statistics (norms only if `measure`) and optionally every pair."""
        x = cache.a[0]
        p, m = x / 2, -x / 2
        stats, pairs = [], {}
        with torch.no_grad():
            for k, c in enumerate(self.chunks):
                if k > 0 and self.chunks[k - 1].name == "x":
                    p, m = x, torch.zeros_like(x)
                p, m = c.split(p, m, cache.pat[k])
                a = cache.a[k + 1]
                mx = max(p.abs().max().item(), m.abs().max().item())
                pre = state_stats(p, m, a) if measure else None
                scaled = mx > Theta
                if scaled:
                    p, m = theta * p, theta * m
                e = a - (p - m)
                p, m = p + e / 2, m - e / 2
                s = {"state": k + 1, "name": c.name, "scaled": scaled, "maxabs_pre": mx,
                     "maxabs": max(p.abs().max().item(), m.abs().max().item())}
                if measure:
                    post = state_stats(p, m, a)
                    s.update(chi_pre=pre["chi"], err_pre=pre["err"], chi=post["chi"], err=post["err"])
                stats.append(s)
                if keep_pairs:
                    pairs[c.name] = (p.clone(), m.clone())
        return SimpleNamespace(p=p, m=m, stats=stats, pairs=pairs)

    def backward_B(self, cache, alpha, keep=(), conv_swap=True, correct=True):
        """HM backward. Sensitivities (pp, np, pn, nn) = (dg/dz+, dg/dz-, dh/dz+, dh/dz-) start at
        (e/2, 0, 0, e/2) on the output; for every chunk, last to first: map them to the chunk input with
        the original patterns, shift by alpha, correct to the cached df/d(input) (if `correct`).
        conv_swap=True swaps np and pn after every Conv2d, as the official code does. Returns the four
        input sensitivities and those of the chunks named in `keep`."""
        e = torch.zeros_like(cache.out)
        e[0, cache.target] = 1.0
        pp, np_, pn, nn_ = 0.5 * e, 0 * e, 0 * e, 0.5 * e
        kept = {}
        with torch.no_grad():
            for k in reversed(range(len(self.chunks))):
                c = self.chunks[k]
                dp, dm = c.back(torch.cat([pp, pn]), torch.cat([np_, nn_]), cache.pat[k])
                pp, pn, np_, nn_ = dp[:1], dp[1:], dm[:1], dm[1:]
                if conv_swap and isinstance(c, Affine) and c.conv:
                    np_, pn = pn, np_
                if alpha:
                    sp, sn = alpha * (pp + np_), alpha * (pn + nn_)
                    pp, np_, pn, nn_ = pp - sp, np_ - sp, pn - sn, nn_ - sn
                if correct:
                    cur = (pp - np_) - (pn - nn_)
                    cc = (cache.grad[k] - cur) / 4
                    pp, np_, pn, nn_ = pp + cc, np_ - cc, pn - cc, nn_ + cc
                if c.name in keep:
                    kept[c.name] = (pp.clone(), np_.clone(), pn.clone(), nn_.clone())
        return SimpleNamespace(pp=pp, np=np_, pn=pn, nn=nn_, kept=kept)


def _chunk(name, mod, device="cpu"):
    if isinstance(mod, (nn.Conv2d, nn.Linear)):
        return Affine(name, mod=mod, device=device)
    if isinstance(mod, nn.ReLU):
        return ReLU(name)
    if isinstance(mod, nn.MaxPool2d):
        return MaxPool(name, mod)
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return AvgPool(name, mod)
    if isinstance(mod, nn.Flatten):
        return Flatten(name)
    if isinstance(mod, nn.Dropout):
        return Identity(name)
    raise TypeError(f"unsupported module {type(mod).__name__}")


chunk_from_module = _chunk


def chunk_kind(c):
    """Short type name of a chunk: conv, linear, relu, maxpool, avgpool, flatten, identity."""
    if isinstance(c, Affine):
        return "conv" if c.conv else "linear"
    return {Identity: "identity", ReLU: "relu", MaxPool: "maxpool", AvgPool: "avgpool", Flatten: "flatten"}[type(c)]


def _flat(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.detach().flatten().numpy()
    return np.asarray(v, dtype=float).ravel()


def result_row(network, depth, method, r, f_true, g_true, seconds):
    """One row of a results table: errors against the truth, chi = (|P| + |Q|)/|f| of the value and
    || |dP| + |dQ| || / ||grad f|| of the gradient (true values in the denominators), memory and time."""
    fT = float(f_true)
    finite = bool(np.isfinite(r.P) and np.isfinite(r.Q))
    row = dict(network=network, depth=depth, method=method, f_true=fT,
               err_val=rel_err(r.value, f_true) if finite else float("nan"), err_grad=float("nan"),
               chi_val=chi_ratio(abs(r.P) + abs(r.Q), fT) if finite else float("nan"), chi_grad=float("nan"),
               overflow=not finite)
    if r.grad is not None:
        gP, gQ = _flat(r.gP), _flat(r.gQ)
        gt = np.asarray([float(v) for v in np.asarray(g_true).ravel()])
        if finite and np.all(np.isfinite(gP)) and np.all(np.isfinite(gQ)):
            row["err_grad"] = rel_err_vec(r.grad, g_true)
            row["chi_grad"] = chi_ratio(np.linalg.norm(np.abs(gP) + np.abs(gQ)), np.linalg.norm(gt))
    row.update(r.ledger.row())
    row["time_s"] = seconds
    trace = getattr(r, "stats", None) if method == "B" else None
    trace = trace if trace else (getattr(r, "trace", None) if method == "B" else None)
    if trace:
        row["n_scaled"] = int(sum(bool(s["scaled"]) for s in trace))
        row["maxabs_max"] = float(max(s["maxabs"] for s in trace))
    return row


# ------------------------------------------------------------------ SeqNet: the three methods, timed

def seq_method_C(net, x, target, C=1.0):
    """Gauge-centred pair at x* = x: P = f + Q, Q = C/2 ||x - x*||^2 (= 0), from forward + autograd."""
    led = Ledger()
    led.add("weights", net.weights())
    xs = x.detach().to(DT)
    with led.autograd():
        cache = net.original(xs, target, all_grads=False)
    gf = cache.grad[0]
    dx = xs - xs                                      # x - x*, exactly zero at the centre
    Qv, gQ = 0.5 * C * (dx * dx).sum(), C * dx
    P, gP = cache.f + Qv, gf + gQ
    return SimpleNamespace(value=(P - Qv).item(), grad=(gP - gQ).flatten().numpy(), P=P.item(), Q=Qv.item(),
                           gP=gP, gQ=gQ, ledger=led)


def seq_method_A(net, x, target, ref_states=None):
    led = Ledger()
    led.add("weights", net.split_weights())
    with led.autograd():
        r = net.forward_A(x, target, ref_states)
    return SimpleNamespace(value=(r.P - r.Q).item(), grad=(r.gP - r.gQ).flatten().numpy(), P=r.P.item(),
                           Q=r.Q.item(), gP=r.gP, gQ=r.gQ, stats=r.stats, ledger=led)


def seq_method_B(net, x, target, theta, Theta, alpha, keep_pairs=False, keep_sens=(), conv_swap=True,
                 measure=True):
    led = Ledger()
    led.add("weights", net.weights(), net.split_weights())
    with led.autograd():
        cache = net.original(x, target, all_grads=True)
    led.add("cache", cache.a, cache.grad, [list(p.values()) for p in cache.pat])
    fw = net.forward_B(cache, theta, Theta, keep_pairs, measure)
    bw = net.backward_B(cache, alpha, keep_sens, conv_swap=conv_swap)
    P, Q = fw.p[0, cache.target].item(), fw.m[0, cache.target].item()
    # HM's delta^g and delta^h of the input; their difference is the invariant the correction enforces
    gP, gQ = bw.pp - bw.np, bw.pn - bw.nn
    return SimpleNamespace(value=P - Q, grad=(gP - gQ).flatten().numpy(), P=P, Q=Q, gP=gP, gQ=gQ,
                           stats=fw.stats, pairs=fw.pairs, sens=bw.kept, cache=cache, ledger=led)


# ------------------------------------------------------------------------------- Ops: generic networks

def _is_t(a):
    return torch.is_tensor(a)


def relu(a):
    if _is_t(a):
        return torch.relu(a)
    if isinstance(a, np.ndarray):
        return np.maximum(a, 0 * a)
    return a if a > 0 else 0 * a


def vmax(a, b):
    if _is_t(a):
        return torch.maximum(a, b)
    if isinstance(a, np.ndarray):
        return np.maximum(a, b)
    return a if a >= b else b


def tr(a):
    return a.transpose(-1, -2) if _is_t(a) else a.swapaxes(-1, -2)


def cat(xs):
    return torch.cat(xs, -1) if _is_t(xs[0]) else np.concatenate(xs, -1)


def to_mp(a):
    """float64 tensor or array -> numpy object array of mpmath numbers with exactly the same values."""
    a = a.detach().cpu().numpy() if _is_t(a) else np.asarray(a, dtype=float)
    return np.vectorize(lambda v: mp.mpf(float(v)), otypes=[object])(a)


class PlainOps:
    """The original network. Works on torch tensors and on mpmath scalars / object arrays.
    With `record`, node values are stored in that dict (the cache of method B)."""

    def __init__(self, record=None):
        self.record = record

    def inp(self, x):
        return x

    def const(self, c):
        return c

    def affine(self, x, w, b):
        return w * x + b

    def lin(self, X, W):
        return X @ W

    def relu(self, z):
        return relu(z)

    def mul(self, u, v):
        return u * v

    def matmul(self, A, B):
        return A @ B

    def T(self, A):
        return tr(A)

    def scale(self, u, c):
        return u * c

    def add(self, u, v):
        return u + v

    def cat(self, xs):
        return cat(xs)

    def row(self, X, i):
        return X[..., i:i + 1, :] if i != -1 else X[..., -1:, :]

    def item(self, v):
        return v[..., 0, 0] if (_is_t(v) or isinstance(v, np.ndarray)) and getattr(v, "ndim", 0) >= 2 else v

    def softmax(self, S):
        if _is_t(S):
            return torch.softmax(S, -1)
        E = np.vectorize(mp.exp, otypes=[object])(S - S.max(axis=-1, keepdims=True))
        return E * (1 / E.sum(-1)[..., None])

    def node(self, v, name):
        if self.record is not None:
            self.record[name] = v.detach().clone() if _is_t(v) else v
        return v


class PairOps(PlainOps):
    """Method A: the naive NDC pair. Constant weights are split once (kept for the memory count);
    `rule` is 'square' (Paper1 Lemma 1, convex components) or 'value' (HM Eq. 19)."""

    def __init__(self, rule="square"):
        super().__init__(None)
        self.rule, self._split = rule, {}

    def split_of(self, W):
        """(W+, W-) of a constant weight. Torch weights are split once and kept (the model holds them, so
        id() stays valid); mpmath arrays are split on every use (they are not timed or counted)."""
        if not _is_t(W):
            return np.maximum(W, 0 * W), np.maximum(-W, 0 * W)
        key = id(W)
        if key not in self._split:
            Wp = W.clamp(min=0)
            self._split[key] = (W, Wp, Wp - W)
        return self._split[key][1:]

    def presplit(self, weights):
        for W in weights:
            if _is_t(W):
                self.split_of(W)
        return self.split_weights()

    def split_weights(self):
        return [t for (_, Wp, Wm) in self._split.values() for t in (Wp, Wm)]

    def inp(self, x):
        return relu(x), relu(-x)

    def const(self, c):
        return (c, 0 * c) if c >= 0 else (0 * c, -c)

    def affine(self, x, w, b):
        P, Q = x
        wp, wm = max(w, 0.0), max(-w, 0.0)
        bp, bm = max(b, 0.0), max(-b, 0.0)
        return wp * P + wm * Q + bp, wm * P + wp * Q + bm

    def lin(self, X, W):
        Wp, Wm = self.split_of(W)
        return X[0] @ Wp + X[1] @ Wm, X[0] @ Wm + X[1] @ Wp

    def relu(self, z):
        return vmax(z[0], z[1]), z[1]

    def mul(self, u, v):
        P, Q = u
        R, S = v
        if self.rule == "square":
            return 0.5 * ((P + R) * (P + R) + (Q + S) * (Q + S)), 0.5 * ((P + S) * (P + S) + (Q + R) * (Q + R))
        return P * R + Q * S, P * S + Q * R

    def matmul(self, A, B):
        PA, QA = A
        PB, QB = B
        if self.rule == "value":
            return PA @ PB + QA @ QB, PA @ QB + QA @ PB
        # square rule on every product A_ik B_kj, summed over k (Paper1 Example 11)
        PA, QA = PA[..., :, :, None], QA[..., :, :, None]
        PB, QB = PB[..., None, :, :], QB[..., None, :, :]
        P = ((PA + PB) * (PA + PB) + (QA + QB) * (QA + QB)).sum(-2) * 0.5
        Q = ((PA + QB) * (PA + QB) + (QA + PB) * (QA + PB)).sum(-2) * 0.5
        return P, Q

    def T(self, A):
        return tr(A[0]), tr(A[1])

    def scale(self, u, c):
        assert c > 0
        return u[0] * c, u[1] * c

    def add(self, u, v):
        return u[0] + v[0], u[1] + v[1]

    def cat(self, xs):
        return cat([x[0] for x in xs]), cat([x[1] for x in xs])

    def row(self, X, i):
        return PlainOps.row(self, X[0], i), PlainOps.row(self, X[1], i)

    def item(self, v):
        return PlainOps.item(self, v[0]), PlainOps.item(self, v[1])

    def softmax(self, S):
        raise NotImplementedError("no recursive DC lift of softmax (Paper1 Sec. 2.4; HM code has a placeholder)")

    def node(self, v, name):
        return v


class HMOps(PairOps):
    """Method B (forward only): the input enters as (x, 0) like in the official code, products by the
    value-level rule (HM Eq. 19), and after every node the threshold scaling and the correction to the
    cached original value."""

    def __init__(self, cache, theta, Theta):
        super().__init__("value")
        self.cache, self.theta, self.Theta, self.trace = cache, theta, Theta, []

    def inp(self, x):
        return x, 0 * x

    def node(self, v, name):
        P, Q = v
        a = self.cache[name]
        mx = max(P.abs().max().item(), Q.abs().max().item())
        scaled = mx > self.Theta
        if scaled:
            P, Q = self.theta * P, self.theta * Q
        e = a - (P - Q)
        P, Q = P + e / 2, Q - e / 2
        self.trace.append({"name": name, "maxabs_pre": mx, "scaled": scaled,
                           "maxabs": max(P.abs().max().item(), Q.abs().max().item())})
        return P, Q


# ---------------------------------------------------------------------------------- the gated network

def _one_like(x):
    t = x[0] if isinstance(x, tuple) else x
    return torch.ones((), dtype=DT) if _is_t(t) else mp.mpf(1)


def gate_net(ops, x, w, b):
    """Paper1 eq. (11): h_0 = 1, g_l = ReLU(w_l x + b_l), h_l = ReLU(g_l h_{l-1}), f = h_L."""
    x = ops.node(ops.inp(x), "x")
    h = ops.const(_one_like(x))
    for l, (wl, bl) in enumerate(zip(w, b)):
        z = ops.node(ops.affine(x, float(wl), float(bl)), f"z{l + 1}")
        g = ops.node(ops.relu(z), f"g{l + 1}")
        u = ops.node(ops.mul(g, h), f"u{l + 1}")
        h = ops.node(ops.relu(u), f"h{l + 1}")
    return h


def gate_make(L_max, seed, x_star):
    """TN1 construction: slopes +-U(0.3, 0.7) with random signs, pre-activations at x* in U(0.8, 1.2)."""
    rng = np.random.default_rng(seed)
    w = rng.choice([-1.0, 1.0], L_max) * rng.uniform(0.3, 0.7, L_max)
    a_star = rng.uniform(0.8, 1.2, L_max)
    return w, a_star - w * x_star


def gate_truth(w, b, x):
    """f, f', f'' at x in mpmath (valid while every ReLU is active): f = prod_l (w_l x + b_l)."""
    x = mp.mpf(x)
    a = [mp.mpf(wl) * x + mp.mpf(bl) for wl, bl in zip(w, b)]
    r = [mp.mpf(wl) / al for wl, al in zip(w, a)]
    f = mp.fprod(a)
    s1, s2 = mp.fsum(r), mp.fsum(ri * ri for ri in r)
    return f, f * s1, f * (s1 * s1 - s2)


def gate_pair_mp(w, b, x):
    """Exact naive pair (square rule) with forward-mode derivatives (P, Q, P', Q') at x (TN1)."""
    zero, one = mp.mpf(0), mp.mpf(1)
    X = (mp.mpf(x), zero, one, zero)
    H = (one, zero, zero, zero)

    def scale(c, P, Q, dP, dQ):
        return (c * P, c * Q, c * dP, c * dQ) if c >= 0 else (-c * Q, -c * P, -c * dQ, -c * dP)

    def addc(c, P, Q, dP, dQ):
        return (P + c, Q, dP, dQ) if c >= 0 else (P, Q - c, dP, dQ)

    def lift(P, Q, dP, dQ):
        return (P, Q, dP, dQ) if P >= Q else (Q, Q, dQ, dQ)

    def prod(P, Q, dP, dQ, R, S, dR, dS):
        return (((P + R) ** 2 + (Q + S) ** 2) / 2, ((P + S) ** 2 + (Q + R) ** 2) / 2,
                (P + R) * (dP + dR) + (Q + S) * (dQ + dS), (P + S) * (dP + dS) + (Q + R) * (dQ + dR))

    for wl, bl in zip(w, b):
        G = lift(*addc(mp.mpf(bl), *scale(mp.mpf(wl), *X)))
        H = lift(*prod(*H, *G))
    return H


def gate_curvature_bound(w, b, x_star, rho):
    """Rigorous C >= sup_{|x - x*| <= rho} |f''| in closed form (TN1), rounded up to float64."""
    lo, hi = mp.mpf(x_star) - mp.mpf(rho), mp.mpf(x_star) + mp.mpf(rho)
    A = [max(abs(mp.mpf(wl) * lo + mp.mpf(bl)), abs(mp.mpf(wl) * hi + mp.mpf(bl))) for wl, bl in zip(w, b)]
    r = [abs(mp.mpf(wl)) / Am for wl, Am in zip(w, A)]
    C = mp.fprod(A) * (mp.fsum(r) ** 2 - mp.fsum(ri * ri for ri in r))
    return float(np.nextafter(float(C), np.inf))


# ------------------------------------------------------------------------------------- attention (TN3)

N_TOK, X_DIM = 6, 3


def mha_sample(n, g):
    """In-context linear regression prompts Z (n, 6, 4) and query labels (n,) (TN3)."""
    x = torch.randn(n, N_TOK, X_DIM, generator=g, dtype=DT)
    w = torch.randn(n, X_DIM, 1, generator=g, dtype=DT) / X_DIM ** 0.5
    y = (x @ w)[..., 0]
    Z = torch.cat([x, y[..., None]], -1)
    Z[:, -1, -1] = 0.0                   # the query label is hidden
    return Z, y[:, -1]


def att_scale(dh, attn):
    """Score scale, a float64 constant of the model (TN3): 1/(sqrt(dh) n) for linear, 1/sqrt(dh) for softmax."""
    return 1.0 / (dh ** 0.5 * N_TOK) if attn == "linear" else 1.0 / dh ** 0.5


def mha_net(ops, Z, M, attn):
    """TN3 Transformer: embedding, N blocks [multi-head attention + residual, ReLU MLP + residual],
    readout of the query token. Every operation is a node (a chunk of the HM code)."""
    X = ops.node(ops.lin(ops.node(ops.inp(Z), "x"), M["E"]), "E")
    for bi, blk in enumerate(M["blocks"]):
        c = att_scale(blk["WQ"][0].shape[-1], attn)
        heads = []
        for h in range(len(blk["WQ"])):
            t = f"b{bi + 1}h{h + 1}"
            Q = ops.node(ops.lin(X, blk["WQ"][h]), f"{t}_Q")
            K = ops.node(ops.lin(X, blk["WK"][h]), f"{t}_K")
            V = ops.node(ops.lin(X, blk["WV"][h]), f"{t}_V")
            S = ops.node(ops.matmul(Q, ops.node(ops.T(K), f"{t}_KT")), f"{t}_S")
            S = ops.node(ops.scale(S, c), f"{t}_Sc")
            if attn == "softmax":
                S = ops.node(ops.softmax(S), f"{t}_A")
            heads.append(ops.node(ops.matmul(S, V), f"{t}_O"))
        O = ops.node(ops.lin(ops.node(ops.cat(heads), f"b{bi + 1}_cat"), blk["WO"]), f"b{bi + 1}_WO")
        X = ops.node(ops.add(X, O), f"b{bi + 1}_res1")
        Hd = ops.node(ops.lin(X, blk["W1"]), f"b{bi + 1}_W1")
        Hd = ops.node(ops.relu(Hd), f"b{bi + 1}_relu")
        Hd = ops.node(ops.lin(Hd, blk["W2"]), f"b{bi + 1}_W2")
        X = ops.node(ops.add(X, Hd), f"b{bi + 1}_res2")
    R = ops.node(ops.lin(ops.node(ops.row(X, -1), "query"), M["head"]), "out")
    return ops.item(R)


def mha_init(N, d, H, seed, init_std):
    """TN3 initialisation (seed = N in TN3)."""
    g = torch.Generator().manual_seed(seed)
    dh = d // H
    rn = lambda *s, sd=init_std: (torch.randn(*s, generator=g, dtype=DT) * sd).requires_grad_()
    return dict(E=rn(X_DIM + 1, d, sd=(X_DIM + 1) ** -0.5), head=rn(d, 1, sd=d ** -0.5),
                blocks=[dict(WQ=rn(H, d, dh), WK=rn(H, d, dh), WV=rn(H, d, dh),
                             WO=rn(d, d), W1=rn(d, 2 * d), W2=rn(2 * d, d)) for _ in range(N)])


def mha_params(M):
    return [M["E"], M["head"]] + [v for b in M["blocks"] for v in b.values()]


def mha_random(N, d, H, seed, init_std):
    """An untrained model in the list form of mha_net (used by the engine checks)."""
    M = mha_init(N, d, H, seed, init_std)
    M = dict(E=M["E"].detach(), head=M["head"].detach(),
             blocks=[{k: v.detach() for k, v in b.items()} for b in M["blocks"]])
    return mha_heads_to_lists(M)


def mha_forward_batch(Z, M, attn):
    """Batched float64 forward (training only); same graph as mha_net with PlainOps."""
    X = Z @ M["E"]
    for blk in M["blocks"]:
        c = att_scale(blk["WQ"].shape[-1], attn)
        heads = []
        for h in range(blk["WQ"].shape[0]):
            Q, K, V = X @ blk["WQ"][h], X @ blk["WK"][h], X @ blk["WV"][h]
            S = (Q @ K.transpose(-1, -2)) * c
            if attn == "softmax":
                S = torch.softmax(S, -1)
            heads.append(S @ V)
        X = X + torch.cat(heads, -1) @ blk["WO"]
        X = X + torch.relu(X @ blk["W1"]) @ blk["W2"]
    return (X[..., -1, :] @ M["head"])[..., 0]


def mha_train(N, d, H, attn, steps, lr, batch, clip, init_std, z_test, y_test):
    """TN3 training: Adam, MSE, gradient clipping; non-finite steps are skipped and counted."""
    M = mha_init(N, d, H, seed=N, init_std=init_std)
    opt = torch.optim.Adam(mha_params(M), lr=lr)
    g = torch.Generator().manual_seed(100 + N)
    skips = 0
    for _ in range(steps):
        Z, y = mha_sample(batch, g)
        loss = ((mha_forward_batch(Z, M, attn) - y) ** 2).mean()
        opt.zero_grad()
        if not torch.isfinite(loss):
            skips += 1
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(mha_params(M), clip)):
            skips += 1
            continue
        opt.step()
    with torch.no_grad():
        mse = ((mha_forward_batch(z_test, M, attn) - y_test) ** 2).mean().item()
    detached = dict(E=M["E"].detach(), head=M["head"].detach(),
                    blocks=[{k: v.detach() for k, v in b.items()} for b in M["blocks"]])
    return detached, mse, skips


def mha_heads_to_lists(M):
    """Trained model -> per-head weights as separate contiguous tensors (persistent objects, so that the
    weight split of PairOps can be cached), the form mha_net expects."""
    out = dict(E=M["E"], head=M["head"], blocks=[])
    for b in M["blocks"]:
        nb = {k: v for k, v in b.items()}
        for k in ("WQ", "WK", "WV"):
            nb[k] = [b[k][h].contiguous() for h in range(b[k].shape[0])]
        out["blocks"].append(nb)
    return out


def mha_weights(M):
    """Every weight tensor of a list-form model."""
    ws = [M["E"], M["head"]]
    for b in M["blocks"]:
        for k in ("WQ", "WK", "WV"):
            ws += b[k]
        ws += [b["WO"], b["W1"], b["W2"]]
    return ws


def mha_to_mp(M):
    conv = lambda v: [to_mp(t) for t in v] if isinstance(v, list) else to_mp(v)
    return dict(E=to_mp(M["E"]), head=to_mp(M["head"]),
                blocks=[{k: conv(v) for k, v in b.items()} for b in M["blocks"]])


def mha_truth(M_mp, Z_mp, attn, mask, fd_h):
    """f and its gradient on the masked cells, in mpmath; gradient by central differences (TN3)."""
    ops = PlainOps()
    f = mha_net(ops, Z_mp, M_mp, attn)
    g = []
    for i, j in zip(*np.nonzero(mask)):
        Zp, Zm = Z_mp.copy(), Z_mp.copy()
        Zp[i, j] += fd_h
        Zm[i, j] -= fd_h
        g.append((mha_net(ops, Zp, M_mp, attn) - mha_net(ops, Zm, M_mp, attn)) / (2 * fd_h))
    return f, np.array(g, dtype=object)


# ------------------------------------------------------------------------ Ops: the three methods, timed

def ops_method_C(net_fn, x, weights, C=1.0, mask=None):
    """Gauge-centred pair at x* = x by a plain float64 forward pass and autograd."""
    led = Ledger()
    led.add("weights", weights)
    xv = x.detach().to(DT).clone().requires_grad_(True)
    with led.autograd(), torch.enable_grad():
        f = net_fn(PlainOps(), xv)
        gf, = torch.autograd.grad(f, xv)
    dx = (xv - xv).detach()                        # x - x*, exactly zero at the centre
    Qv, gQ = 0.5 * C * (dx * dx).sum(), C * dx
    P, gP = f.detach() + Qv, gf + gQ
    sel = (lambda t: t[mask]) if mask is not None else (lambda t: t.flatten())
    return SimpleNamespace(value=(P - Qv).item(), grad=sel(gP - gQ).numpy(), P=P.item(), Q=Qv.item(),
                           gP=sel(gP).numpy(), gQ=sel(gQ).numpy(), ledger=led)


def ops_method_A(net_fn, x, weights, rule="square", mask=None):
    """Naive pair; gradients of P and Q by autograd. The weights are split before the pass, so that
    W+ and W- are counted as weights and not as tensors saved by autograd."""
    led = Ledger()
    ops = PairOps(rule)
    led.add("weights", ops.presplit(weights))
    xv = x.detach().to(DT).clone().requires_grad_(True)
    with led.autograd(), torch.enable_grad():
        P, Q = net_fn(ops, xv)
        gP, = torch.autograd.grad(P, xv, retain_graph=True)
        gQ, = torch.autograd.grad(Q, xv)
    sel = (lambda t: t[mask]) if mask is not None else (lambda t: t.flatten())
    return SimpleNamespace(value=(P - Q).item(), grad=sel(gP - gQ).numpy(), P=P.item(), Q=Q.item(),
                           gP=sel(gP).numpy(), gQ=sel(gQ).numpy(), ledger=led)


def ops_method_B(net_fn, x, weights, theta, Theta):
    """HM forward with the cache of the original network; value only (HM defines no backward for product
    nodes, and its official code has an identity placeholder there)."""
    led = Ledger()
    led.add("weights", weights)
    cache = {}
    with torch.no_grad():
        net_fn(PlainOps(cache), x.detach().to(DT))
        ops = HMOps(cache, theta, Theta)
        led.add("weights", ops.presplit(weights))
        P, Q = net_fn(ops, x.detach().to(DT))
    led.add("cache", list(cache.values()))
    return SimpleNamespace(value=(P - Q).item(), grad=None, P=P.item(), Q=Q.item(), trace=ops.trace, ledger=led)


# ------------------------------------------------------------------------------------- ReLU MLP (mpmath)

def mlp_make(d_in, width, depth, seed):
    """Bias-free ReLU MLP, He-initialised, then every layer multiplied by the positive factor that keeps
    ||a_l|| = ||x*|| at the point x* (data-dependent normalisation as in LSUV); all-positive readout
    1/sqrt(width). Returns the weights, the readout and x* (shape (1, d_in))."""
    g = torch.Generator().manual_seed(seed)
    dims = [d_in] + [width] * depth
    Ws = [torch.randn(dims[l + 1], dims[l], generator=g, dtype=DT) * (2.0 / dims[l]) ** 0.5 for l in range(depth)]
    x = torch.randn(1, d_in, generator=g, dtype=DT)
    a = x[0]
    for W in Ws:
        W *= a.norm() / torch.relu(W @ a).norm()
        a = torch.relu(W @ a)
    w_out = torch.full((width,), width ** -0.5, dtype=DT)
    return Ws, w_out, x


def mlp_truth_and_chi_mp(Ws, w_out, x):
    """For every depth l: f_l = w_out . a_l and grad f_l in mpmath, and the exact naive pair (A) with
    forward-mode derivatives, giving chi_val = (P + Q)/|f| and chi_grad = || |dP| + |dQ| || / ||grad f||."""
    xm = to_mp(x[0])
    d = len(xm)
    Wm_ = [to_mp(W) for W in Ws]
    wo = to_mp(w_out)
    one, zero = mp.mpf(1), mp.mpf(0)
    # original network: activations and Jacobians d a / d x
    a = xm.copy()
    J = np.array([[one if i == j else zero for j in range(d)] for i in range(d)], dtype=object)
    # naive pair from (relu(x), relu(-x)) with forward-mode derivatives
    p = np.array([v if v > 0 else zero for v in xm], dtype=object)
    m = np.array([-v if v < 0 else zero for v in xm], dtype=object)
    dp = np.array([[one if (i == j and xm[i] > 0) else zero for j in range(d)] for i in range(d)], dtype=object)
    dm = np.array([[-one if (i == j and xm[i] < 0) else zero for j in range(d)] for i in range(d)], dtype=object)
    rows = []
    for l, W in enumerate(Wm_):
        Wp, Wn = np.maximum(W, 0 * W), np.maximum(-W, 0 * W)
        z, Jz = W @ a, W @ J
        act = np.array([v > 0 for v in z])
        a = np.array([v if k else zero for v, k in zip(z, act)], dtype=object)
        J = np.array([Jz[i] if act[i] else Jz[i] * 0 for i in range(len(z))], dtype=object)
        zp, zm = Wp @ p + Wn @ m, Wn @ p + Wp @ m
        dzp, dzm = Wp @ dp + Wn @ dm, Wn @ dp + Wp @ dm
        take = np.array([u >= v for u, v in zip(zp, zm)])
        p = np.array([u if t else v for u, v, t in zip(zp, zm, take)], dtype=object)
        dp = np.array([dzp[i] if take[i] else dzm[i] for i in range(len(zp))], dtype=object)
        m, dm = zm, dzm
        f = mp.fsum(wo * a)
        gf = wo @ J
        P, Q = mp.fsum(wo * p), mp.fsum(wo * m)            # readout weights are positive
        gP, gQ = wo @ dp, wo @ dm
        nf = mp.sqrt(mp.fsum(v * v for v in gf))
        rows.append(dict(depth=l + 1, f=f, grad=gf, P=P, Q=Q,
                         chi_val=(P + Q) / abs(f),
                         chi_grad=mp.sqrt(mp.fsum((abs(u) + abs(v)) ** 2 for u, v in zip(gP, gQ))) / nf))
    return rows
