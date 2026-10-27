"""Engine of notebook nb03: the split explanations of Hidden Monotonicity (HM) computed in difference-sum
coordinates, with HM's tuned stabilisers (theta, Theta, alpha) or with the balanced gauge rule (Gauge-X).

HM runs two streams (p, m) through the network. In the coordinates d = p - m, sigma = p + m its official code
(DC-Decomposition-for-Explanability, commit 31be19b) reduces to the following; nb03 §3 derives it and §4 checks
it against the code.

forward   d_k = a_k, the activation of the original network (HM overrides d by the cached activation).
          sigma_k follows a recursion of non-negative terms:
              Conv2d, Linear    sigma' = |W| sigma + |b|
              ReLU              sigma' = sigma + relu(-z)
              MaxPool2d         convex form: sigma' = maxpool(a) + window_sum(sigma - a)
                                winner-takes-all (wta): sigma at the argmax of a
              AdaptiveAvgPool   sigma' = avgpool(sigma)
          The layer that reads the input gets the pair (x, 0), so sigma = x there (official code).
          Stabiliser after every chunk:  'hm'        sigma *= theta when max(|p|, |m|) > Theta
                                         'balanced'  sigma *= ||a|| / ||sigma||               (Gauge-X)
                                         'zero'      sigma = 0                                (G0)
backward  In HM's units (start (pp, np, pn, nn) = (e/2, 0, 0, e/2)) the four sensitivities at a chunk input are
              pp = (D + Gamma + S + Delta) / 4,     np = (-D - Gamma + S + Delta) / 4,
              pn = (-D + Gamma + S - Delta) / 4,    nn = (D - Gamma + S - Delta) / 4,
          with D = df/da (exact, since HM corrects the sensitivities to the cached gradient) and (Gamma, S) the
          gauge recursion seeded with (0, e) at the output. For every chunk, last to first:
              Gamma <- B_dd Gamma + B_ds S,   S <- B_ss S,   then S <- t S   (t = 1 - 2 alpha: HM's shift)
          B_dd is the backprop of the original network; B_ss = |W|^T (affine), 1 (ReLU), window_sum^T (convex
          pool), argmax scatter (wta pool); the injections are B_ds = -(1 - mask) (ReLU), scatter - window_sum^T
          (convex pool), 0 otherwise. Delta = 0 for the true derivative. The official Conv2d backprop swaps np and
          pn; in these coordinates Gamma and Delta trade places after every Conv2d, and Delta follows B_ss and the
          shift like S.
maps      at the input of chunk l (HM's convention), nearest-upsampled to 224 x 224:
              cam_g   sum_c (pp - np)(p - m) = 1/2 sum_c (D + Gamma) a
              grad    mean_c (pp - np)       = 1/2 mean_c (D + Gamma)            (HM's grad_only)
              cam_pp  sum_c pp p             = 1/8 sum_c (D + Gamma + S + Delta)(sigma + a)
              lrp     mean_c (R+ - R-), HM's code rules on the streams ((sigma + a)/2, (sigma - a)/2)
balanced  t*_l is the smallest t with ||Gamma_l(t)|| = ||D_l|| (Gamma_l(0) = 0): first bracket on a grid of t, then
          regula falsi on log ratio (the ratio grows roughly exponentially in t); alpha*_l = (1 - t*_l) / 2.
          If the ratio stays below 1, t* = 1 (alpha* = 0).

Families of one explanation row: 'hm' (official code: alpha = 0.4, theta/Theta, conv swap), 'hm_true' (the same
with the true derivative), 'gauge' (true derivative, t*_l, balanced sigma) and 'g0' (t = 0, sigma = 0: the
classical map). SplitLRP does not use alpha or the swap, so its 'hm_true' equals 'hm' and is not computed.
"""
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input

THETA, BIG_THETA, ALPHA_HM = 0.1, 10.0, 0.4      # configs/nb01/methods/*.json
LRP_FLOOR = 1e-5                                 # HM's floor of the normalised streams (lrp_direct_layers.py)
AVG_EPS = 1e-9                                   # HM's stabiliser of the average-pool z-rule
SIZE = (224, 224)                                # HM upsamples every map to the input size
FAMILIES = ("hm", "hm_true", "gauge", "g0")
KINDS = ("cam_g", "cam_pp", "grad", "lrp")


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def _bview(t, like):
    """Per-row scalars t (T,) shaped to broadcast against a batch `like` of shape (T, ...)."""
    return t.view(-1, *[1] * (like.dim() - 1))


class Chunk:
    """One HM chunk: a single layer of the torchvision model, named as HM's torch.fx node."""

    def __init__(self, name, mod):
        self.name = name
        if mod is None or isinstance(mod, (nn.Dropout, nn.Identity)):
            self.kind = "identity"
        elif isinstance(mod, (nn.Conv2d, nn.Linear)):
            self.kind = "conv" if isinstance(mod, nn.Conv2d) else "linear"
            self.W = mod.weight.detach()
            self.b = None if mod.bias is None else mod.bias.detach()
            self.Wabs = self.W.abs()
            self.babs = None if self.b is None else self.b.abs()
            if self.kind == "conv":
                self.kw = dict(stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups)
        elif isinstance(mod, nn.ReLU):
            self.kind = "relu"
        elif isinstance(mod, nn.MaxPool2d):
            k, s = _pair(mod.kernel_size), _pair(mod.stride or mod.kernel_size)
            assert k == s and _pair(mod.padding) == (0, 0) and _pair(mod.dilation) == (1, 1) and not mod.ceil_mode, \
                "only non-overlapping max pools (VGG)"
            self.kind, self.k = "maxpool", k
        elif isinstance(mod, nn.AdaptiveAvgPool2d):
            self.kind, self.size = "avgpool", _pair(mod.output_size)
        elif isinstance(mod, nn.Flatten):
            self.kind = "flatten"
        else:
            raise TypeError(f"unsupported module {type(mod).__name__}")

    def signed_parts(self):
        """HM's weight_pos and weight_neg (SplitLRP only); built on use, so the engine keeps one extra copy of the
        weights (|W|) on the GPU instead of three."""
        return self.W.clamp(min=0), (-self.W).clamp(min=0)

    # -- affine maps
    def op(self, x, W, b=None):
        if self.kind == "conv":
            y = F.conv2d(x, W, None, **self.kw)
            return y if b is None else y + b.view(1, -1, 1, 1)
        return F.linear(x, W, b)

    def op_t(self, g, W, in_shape):
        """Transpose of x -> op(x, W) applied to a batch g of output sensitivities."""
        if self.kind == "conv":
            return conv2d_input((g.shape[0],) + tuple(in_shape[1:]), W, g, **self.kw)
        return g @ W

    # -- pooling helpers
    def window_sum(self, x):
        return F.avg_pool2d(x, self.k, self.k, divisor_override=1)

    def window_sum_t(self, g, in_shape):
        """Transpose of window_sum: every input pixel lies in at most one window."""
        h, w = g.shape[-2] * self.k[0], g.shape[-1] * self.k[1]
        up = g.repeat_interleave(self.k[0], dim=-2).repeat_interleave(self.k[1], dim=-1)
        if (h, w) == tuple(in_shape[-2:]):
            return up
        out = g.new_zeros((g.shape[0],) + tuple(in_shape[1:]))
        out[..., :h, :w] = up
        return out

    def avg_t(self, g, in_shape):
        """Transpose of the adaptive average pool."""
        if tuple(in_shape[-2:]) == self.size:
            return g
        x = g.new_zeros((g.shape[0],) + tuple(in_shape[1:])).requires_grad_(True)
        with torch.enable_grad():
            return torch.autograd.grad(F.adaptive_avg_pool2d(x, self.size), x, g)[0]


def scatter_to(g, idx, in_shape):
    """Route a batch g (T, C, h, w) to the input positions idx (1, C, h, w) of a max pool."""
    T, C = g.shape[:2]
    out = g.new_zeros(T, C, in_shape[2] * in_shape[3])
    out.scatter_add_(2, idx.flatten(2).expand(T, -1, -1), g.flatten(2))
    return out.view(T, C, in_shape[2], in_shape[3])


def gather_at(s, idx):
    return torch.gather(s.flatten(2), 2, idx.flatten(2).expand(s.shape[0], -1, -1)).view(
        s.shape[0], s.shape[1], *idx.shape[2:])


class SplitNet:
    """A torchvision VGG cut into HM's chunks 'x', features_i, avgpool, flatten, classifier_i.
    State k is the input of chunk k; state K = len(chunks) is the vector of logits."""

    def __init__(self, model):
        p = next(model.parameters())
        self.dtype, self.device = p.dtype, p.device
        mods = [("x", None)] + [(f"features_{i}", m) for i, m in enumerate(model.features)]
        mods += [("avgpool", model.avgpool), ("flatten", nn.Flatten())]
        mods += [(f"classifier_{i}", m) for i, m in enumerate(model.classifier)]
        self.chunks = [Chunk(n, m) for n, m in mods]
        self.names = [c.name for c in self.chunks]
        self.index = {n: k for k, n in enumerate(self.names)}

    # ------------------------------------------------------------------ original network
    def original(self, x, target=None):
        """Forward and backward pass of the original network: activations a[k], ReLU masks and max-pool argmax
        patterns, D[k] = df/da[k] with f the logit of `target` (the predicted class if None)."""
        x = x.detach().to(self.device, self.dtype).requires_grad_(True)
        a, pat, h = [x], [], x
        with torch.enable_grad():
            for c in self.chunks:
                pc = {"in_shape": tuple(h.shape)}
                h = self._forward(c, h, pc)
                a.append(h)
                pat.append(pc)
            target = int(h[0].argmax()) if target is None else int(target)
            D = torch.autograd.grad(h[0, target], a)
        return SimpleNamespace(a=[t.detach() for t in a], pat=pat, D=[g.detach() for g in D], target=target,
                               logits=h.detach())

    @staticmethod
    def _forward(c, h, pc):
        if c.kind == "identity":
            return h.view_as(h)
        if c.kind in ("conv", "linear"):
            return c.op(h, c.W, c.b)
        if c.kind == "relu":
            pc["mask"] = h > 0                     # autograd's convention: relu'(0) = 0
            return F.relu(h)
        if c.kind == "maxpool":
            out, pc["idx"] = F.max_pool2d(h, c.k, c.k, return_indices=True)
            return out
        if c.kind == "avgpool":
            return F.adaptive_avg_pool2d(h, c.size)
        return h.flatten(1)

    # ------------------------------------------------------------------ gauge stream (forward)
    def sigma(self, cache, mode, pool="convex", theta=THETA, Theta=BIG_THETA):
        """Gauge stream sigma[k] at every state under the stabiliser `mode` ('hm', 'balanced' or 'zero');
        also returns the states after which HM's threshold scaling fired."""
        a = cache.a
        if mode == "zero":
            return [torch.zeros_like(t) for t in a], []
        sig, scaled = [torch.zeros_like(a[0])] + [None] * len(self.chunks), []
        for k, c in enumerate(self.chunks):
            if c.name == "x":                       # official code: the layer that reads the input gets (x, 0)
                sig[k + 1] = a[k + 1].clone()
                continue
            s = self._sigma_step(c, sig[k], a[k], cache.pat[k], pool)
            if mode == "hm":
                if float(((s.abs() + a[k + 1].abs()) / 2).max()) > Theta:    # max(|p|, |m|) = (|sigma| + |a|) / 2
                    s = theta * s
                    scaled.append(k + 1)
            elif mode == "balanced":
                ns = s.norm()
                if ns > 0:
                    s = s * (a[k + 1].norm() / ns)
            else:
                raise ValueError(mode)
            sig[k + 1] = s
        return sig, scaled

    @staticmethod
    def _sigma_step(c, s, a_in, pc, pool):
        if c.kind == "identity":
            return s
        if c.kind in ("conv", "linear"):
            return c.op(s, c.Wabs, c.babs)
        if c.kind == "relu":
            return s + F.relu(-a_in)
        if c.kind == "maxpool":
            if pool == "convex":
                return F.max_pool2d(a_in, c.k, c.k) + c.window_sum(s - a_in)
            return gather_at(s, pc["idx"])
        if c.kind == "avgpool":
            return F.adaptive_avg_pool2d(s, c.size)
        return s.flatten(1)

    # ------------------------------------------------------------------ gauge sensitivities (backward)
    def gauge(self, cache, ts, layers, pool="convex", swap=False):
        """(Gamma, S, Delta) at the input of every chunk named in `layers`, one batch row per value of t in ts.
        swap=True reproduces the official Conv2d backprop (Delta is None otherwise)."""
        ts = torch.as_tensor(np.asarray(ts, dtype=np.float64), dtype=self.dtype, device=self.device).reshape(-1)
        e = cache.logits.new_zeros(ts.numel(), cache.logits.shape[1])
        e[:, cache.target] = 1.0
        G, S, Dl = torch.zeros_like(e), e, (torch.zeros_like(e) if swap else None)
        stop, out = min(self.index[n] for n in layers), {}
        for k in range(len(self.chunks) - 1, stop - 1, -1):
            c = self.chunks[k]
            G, S, Dl = self._gauge_step(c, G, S, Dl, cache.pat[k], pool)
            if swap and c.kind == "conv":
                G, Dl = Dl, G
            S = S * _bview(ts, S)
            if Dl is not None:
                Dl = Dl * _bview(ts, Dl)
            if c.name in layers:
                out[c.name] = (G, S, Dl)
        return out

    @staticmethod
    def _gauge_step(c, G, S, Dl, pc, pool):
        """One chunk of the gauge recursion: Gamma <- B_dd Gamma + B_ds S, S <- B_ss S, Delta <- B_ss Delta."""
        if c.kind == "identity":
            return G, S, Dl
        shape = pc["in_shape"]
        if c.kind in ("conv", "linear"):
            G = c.op_t(G, c.W, shape)
            if Dl is None:
                return G, c.op_t(S, c.Wabs, shape), None
            SD = c.op_t(torch.cat([S, Dl]), c.Wabs, shape)
            return G, SD[:S.shape[0]], SD[S.shape[0]:]
        if c.kind == "relu":
            return torch.where(pc["mask"], G, -S), S, Dl
        if c.kind == "maxpool":
            if pool == "convex":
                sT = c.window_sum_t(S, shape)
                return scatter_to(G + S, pc["idx"], shape) - sT, sT, \
                    None if Dl is None else c.window_sum_t(Dl, shape)
            return scatter_to(G, pc["idx"], shape), scatter_to(S, pc["idx"], shape), \
                None if Dl is None else scatter_to(Dl, pc["idx"], shape)
        if c.kind == "avgpool":
            return c.avg_t(G, shape), c.avg_t(S, shape), None if Dl is None else c.avg_t(Dl, shape)
        return G.reshape((-1,) + tuple(shape[1:])), S.reshape((-1,) + tuple(shape[1:])), \
            None if Dl is None else Dl.reshape((-1,) + tuple(shape[1:]))

    # ------------------------------------------------------------------ SplitLRP (HM's code rules)
    def lrp(self, cache, sig, layers):
        """HM's SplitLRP relevances (R+, R-) at the input of every chunk named in `layers`, on the streams
        p = (sigma + a)/2, m = (sigma - a)/2; relevance starts at (p, m) of the target logit."""
        a, K = cache.a, len(self.chunks)
        p, m = (sig[K] + a[K]) / 2, (sig[K] - a[K]) / 2
        Rp, Rm = torch.zeros_like(p), torch.zeros_like(m)
        Rp[0, cache.target], Rm[0, cache.target] = p[0, cache.target], m[0, cache.target]
        stop, out = min(self.index[n] for n in layers), {}
        for k in range(K - 1, stop - 1, -1):
            c = self.chunks[k]
            p, m = (sig[k] + a[k]) / 2, (sig[k] - a[k]) / 2
            Rp, Rm = self._lrp_step(c, Rp, Rm, p, m)
            if c.name in layers:
                out[c.name] = (Rp, Rm)
        return out

    @staticmethod
    def _lrp_step(c, Rp, Rm, p, m):
        if c.kind in ("identity", "relu"):
            return Rp, Rm
        if c.kind in ("conv", "linear"):
            return SplitNet._lrp_affine(c, Rp, Rm, p, m)
        if c.kind == "maxpool":                     # winner-takes-all on each stream's own input
            ip = F.max_pool2d(p, c.k, c.k, return_indices=True)[1]
            im = F.max_pool2d(m, c.k, c.k, return_indices=True)[1]
            return scatter_to(Rp, ip, p.shape), scatter_to(Rm, im, m.shape)
        if c.kind == "avgpool":
            return SplitNet._lrp_avg(Rp, p), SplitNet._lrp_avg(Rm, m)
        return Rp.view(p.shape), Rm.view(m.shape)

    @staticmethod
    def _lrp_affine(c, Rp, Rm, p, m):
        # DirectConv2dSplitLRP / DirectLinearSplitLRP: streams normalised by their joint L1 norm, floored at 1e-5
        # (a float32 scalar, as in HM), no bias, non-finite ratios set to 0
        norm = p.abs().sum() + m.abs().sum()
        floor = torch.tensor(LRP_FLOOR)
        pn, mn = torch.maximum(p / norm, floor), torch.maximum(m / norm, floor)
        if c.kind == "conv":
            st, pd, dl, gr = c.kw["stride"], c.kw["padding"], c.kw["dilation"], c.kw["groups"]
            st, pd = _pair(st), _pair(pd)
            opad = (0, 0)
            if any(s > 1 for s in st):
                kh, kw = c.W.shape[-2:]
                opad = (max(0, p.shape[2] - ((Rp.shape[2] - 1) * st[0] - 2 * pd[0] + kh)),
                        max(0, p.shape[3] - ((Rp.shape[3] - 1) * st[1] - 2 * pd[1] + kw)))
            fwd = lambda x, W: F.conv2d(x, W, None, st, pd, dl, gr)
            bwd = lambda s, W: F.conv_transpose2d(s, W, None, st, pd, opad, gr, dl)
        else:
            fwd = lambda x, W: F.linear(x, W, None)
            bwd = lambda s, W: F.linear(s, W.t())
        Wp, Wn = c.signed_parts()
        zp = fwd(pn, Wp) + fwd(mn, Wn)
        zm = fwd(pn, Wn) + fwd(mn, Wp)
        sp, sm = Rp / zp, Rm / zm
        sp = torch.where(torch.isfinite(sp), sp, torch.zeros_like(sp))
        sm = torch.where(torch.isfinite(sm), sm, torch.zeros_like(sm))
        return pn * (bwd(sp, Wp) + bwd(sm, Wn)), mn * (bwd(sp, Wn) + bwd(sm, Wp))

    @staticmethod
    def _lrp_avg(R, x):
        # DirectAvgPool2dSplitLRP, adaptive case: z-rule with uniform weights
        (oh, ow), (ih, iw) = R.shape[-2:], x.shape[-2:]
        sh, sw = ih // oh, iw // ow
        kh, kw = ih - (oh - 1) * sh, iw - (ow - 1) * sw
        z = F.adaptive_avg_pool2d(x, (oh, ow)) + AVG_EPS
        C = x.shape[1]
        kern = torch.ones((C, 1, kh, kw), device=x.device, dtype=x.dtype) / (kh * kw)
        opad = (max(0, ih - ((oh - 1) * sh + kh)), max(0, iw - ((ow - 1) * sw + kw)))
        spread = F.conv_transpose2d(R / z, kern, None, (sh, sw), (0, 0), opad, groups=C)[:, :, :ih, :iw]
        return x * spread


# ---------------------------------------------------------------------------------------------- maps

def upsample(m):
    """(T, h, w) -> (T, 224, 224), nearest neighbour as HM."""
    return F.interpolate(m[:, None], size=SIZE, mode="nearest")[:, 0]


def cam_g(D, G, a):
    return 0.5 * ((D + G) * a).sum(1)


def grad_map(D, G):
    return 0.5 * (D + G).mean(1)


def cam_pp(D, G, S, Dl, sig, a):
    pp = 0.25 * (D + G + S + (0 if Dl is None else Dl))
    return (pp * (0.5 * (sig + a))).sum(1)


def lrp_map(Rp, Rm):
    return (Rp - Rm).mean(1)


def ratio_curve(G, D):
    """||Gamma_l(t)|| / ||D_l|| for every row t of the batch G."""
    return (G.flatten(1).norm(dim=1) / D.norm()).double().cpu().numpy()


def bracket(ratio, ts):
    """First grid interval [t0, t1] where the ratio reaches 1, as (t0, r0, t1, r1); None if it stays below 1."""
    r, ts = np.asarray(ratio, dtype=float), np.asarray(ts, dtype=float)
    hit = np.nonzero(r >= 1.0)[0]
    if len(hit) == 0:
        return None
    j = max(int(hit[0]), 1)                   # ratio(0) = 0, so j >= 1 whenever ts[0] = 0
    return [float(ts[j - 1]), float(r[j - 1]), float(ts[j]), float(r[j])]


def _guess(b):
    """Root of log ratio on the bracket (t0, r0, t1, r1): log-linear interpolation, bisection if r0 = 0."""
    t0, r0, t1, r1 = b
    if r0 <= 0.0:
        return 0.5 * (t0 + t1)
    return t0 + (t1 - t0) * np.log(1.0 / r0) / np.log(r1 / r0)


def balance(net, cache, ts, layers, pool, swap=False, iters=4):
    """Balanced gauge weight at every layer: t*_l = smallest t with ||Gamma_l(t)|| = ||D_l||.

    A grid pass over ts gives the ratio curves and the first bracket of each layer; the ratio grows roughly
    exponentially in t (gauge recursion of depth ~ number of chunks above l), so the root is refined by regula
    falsi on log ratio, each iteration one backward pass with one row per layer. Returns the grid gauge tensors
    and per layer: t_star, alpha_star, balanced, ratio (grid curve), ratio_star (achieved) and (G, S, Delta) at t*."""
    grid = net.gauge(cache, ts, layers, pool, swap=swap)
    info, br = {}, {}
    for l in layers:
        rc = ratio_curve(grid[l][0], cache.D[net.index[l]])
        info[l] = dict(ratio=rc)
        b = bracket(rc, ts)
        if b is None:                          # the gauge never reaches the size of the gradient: alpha* = 0
            G, S, Dl = (None if t is None else t[-1:] for t in grid[l])
            info[l].update(t_star=float(ts[-1]), balanced=False, ratio_star=float(rc[-1]), at_star=(G, S, Dl))
        else:
            br[l] = b
    live = [l for l in layers if l in br]
    for _ in range(iters if live else 0):
        guess = [_guess(br[l]) for l in live]
        g = net.gauge(cache, guess, live, pool, swap=swap)
        for i, l in enumerate(live):
            G, S, Dl = (None if t is None else t[i:i + 1] for t in g[l])
            r = float(G.norm() / cache.D[net.index[l]].norm())
            t0, r0, t1, r1 = br[l]
            if r >= 1.0:
                br[l] = [t0, r0, guess[i], r]
            else:
                br[l] = [guess[i], r, t1, r1]
            info[l].update(t_star=float(guess[i]), balanced=True, ratio_star=r, at_star=(G, S, Dl))
    for l in layers:
        info[l]["alpha_star"] = alpha_of_t(info[l]["t_star"])
    return grid, info


def t_of_alpha(alpha):
    return 1.0 - 2.0 * alpha


def alpha_of_t(t):
    return (1.0 - t) / 2.0


def explain(net, x, rows, families=FAMILIES, ts=None, target=None, cache=None, official_grid=False, keep=False):
    """Raw maps (signed, native resolution of the layer) of every row and family at the input x.

    rows: dicts with keys id, layer (chunk name), kind ('cam_g', 'cam_pp', 'grad', 'lrp'), pool ('convex' or 'wta').
    ts:   grid of t for the balanced rule; starts at t = 0 and contains t = 1 - 2*0.4 = 0.2, the value 'hm' and
          'hm_true' read from the official grid and from the grid of the balanced rule (nb03: linspace(0, 1, 41)).
    Returns SimpleNamespace(maps={(row id, family): tensor (h, w)}, diag=[dict per (pool, derivative, layer)],
    target, cache). official_grid=True also runs the official recursion on the whole grid (for the alpha* curves);
    keep=True keeps the gauge tensors in .gauge (checks of nb03 §4)."""
    ts = np.linspace(0.0, 1.0, 41) if ts is None else np.asarray(ts, dtype=float)
    t_hm = t_of_alpha(ALPHA_HM)
    j_hm = int(np.argmin(abs(ts - t_hm)))
    assert ts[0] == 0.0 and abs(ts[j_hm] - t_hm) < 1e-12, "the grid must start at t = 0 and contain t = 0.2"
    cache = net.original(x, target) if cache is None else cache
    a, D = cache.a, cache.D
    out, diag, kept = {}, [], {}
    grad_rows = [r for r in rows if r["kind"] in ("cam_g", "cam_pp", "grad")]
    sig_cache = {}

    def sig_of(mode, pool):
        if (mode, pool) not in sig_cache:
            sig_cache[(mode, pool)] = net.sigma(cache, mode, pool)[0]
        return sig_cache[(mode, pool)]

    def record(pool, derivative, info):
        for l, d in info.items():
            diag.append(dict(pool=pool, derivative=derivative, layer=l, t_star=d["t_star"],
                             alpha_star=d["alpha_star"], balanced=d["balanced"], ratio_star=d["ratio_star"],
                             ratio=d["ratio"]))

    for pool in sorted({r["pool"] for r in grad_rows}):
        prow = [r for r in grad_rows if r["pool"] == pool]
        layers = sorted({r["layer"] for r in prow}, key=net.index.get)
        gr = go = info_t = info_o = None
        if "gauge" in families:
            gr, info_t = balance(net, cache, ts, layers, pool, swap=False)
            record(pool, "true", info_t)
            j_true = j_hm
        elif "hm_true" in families:
            gr, j_true = net.gauge(cache, [t_hm], layers, pool, swap=False), 0
        if official_grid:
            go, info_o = balance(net, cache, ts, layers, pool, swap=True)
            record(pool, "official", info_o)
            jo = j_hm
        elif "hm" in families:
            go, jo = net.gauge(cache, [t_hm], layers, pool, swap=True), 0
        if keep:
            kept[pool] = dict(true=gr, official=go, info_true=info_t, info_official=info_o)
        for r in prow:
            l, k = r["layer"], net.index[r["layer"]]
            for fam in families:
                if fam == "g0":                    # t = 0: Gamma = 0 and S = 0 (nb03 §4 checks it on the grid)
                    G, S, Dl = torch.zeros_like(a[k]), torch.zeros_like(a[k]), None
                elif fam == "hm_true":
                    G, S, Dl = (t[j_true:j_true + 1] if t is not None else None for t in gr[l])
                elif fam == "gauge":
                    G, S, Dl = info_t[l]["at_star"]
                else:
                    G, S, Dl = (t[jo:jo + 1] if t is not None else None for t in go[l])
                if r["kind"] == "cam_g":
                    m = cam_g(D[k], G, a[k])
                elif r["kind"] == "grad":
                    m = grad_map(D[k], G)
                else:
                    smode = {"hm": "hm", "hm_true": "hm", "gauge": "balanced", "g0": "zero"}[fam]
                    m = cam_pp(D[k], G, S, Dl, sig_of(smode, pool)[k], a[k])
                out[(r["id"], fam)] = m[0]

    lrp_rows = [r for r in rows if r["kind"] == "lrp"]
    for pool in sorted({r["pool"] for r in lrp_rows}):
        prow = [r for r in lrp_rows if r["pool"] == pool]
        for fam in families:
            if fam == "hm_true":                   # SplitLRP uses neither alpha nor the conv backprop
                continue
            smode = {"hm": "hm", "gauge": "balanced", "g0": "zero"}[fam]
            R = net.lrp(cache, sig_of(smode, pool), [r["layer"] for r in prow])
            for r in prow:
                out[(r["id"], fam)] = lrp_map(*R[r["layer"]])[0]
    res = SimpleNamespace(maps=out, diag=diag, target=cache.target, cache=cache)
    if keep:
        res.gauge, res.sigma = kept, sig_cache
    return res


def post(m, op):
    """HM's post-processing of a formula: 'none' keeps the sign, 'abs' takes the absolute value."""
    return m.abs() if op == "abs" else m


def map224(m, op):
    """Native-resolution raw map -> HM's evaluated map (224 x 224, float32 numpy)."""
    return post(upsample(m[None].float()), op)[0].cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------------------------- map similarity

def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(np.ravel(a), np.ravel(b)).statistic)


def ssim(a, b):
    """SSIM of two maps after each is scaled to [0, 1] (min-max), as in randomisation sanity checks."""
    from skimage.metrics import structural_similarity
    na = (a - a.min()) / max(float(a.max() - a.min()), 1e-30)
    nb = (b - b.min()) / max(float(b.max() - b.min()), 1e-30)
    return float(structural_similarity(na, nb, data_range=1.0))


def top_overlap(a, b, q=0.1):
    """Share of the top-q pixels of a that are also in the top-q pixels of b."""
    k = max(1, int(round(q * a.size)))
    ia = np.argpartition(-np.ravel(a), k - 1)[:k]
    ib = np.argpartition(-np.ravel(b), k - 1)[:k]
    return len(np.intersect1d(ia, ib)) / k


def rel_diff(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).norm() / b.norm().clamp_min(1e-300))
