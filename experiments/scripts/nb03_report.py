"""Tables and figures of notebook nb03, built only from the files a run saved in RUN_DIR/data.

Nothing here touches a model, so display names can be changed (pass a modified copy of default_labels())
and everything redrawn in seconds. Data files use fixed English keys (method, row, family, pool, derivative,
metric, comparison, ...); display names come from the labels. Figures go to RUN_DIR/figures, tables to
RUN_DIR/tables; their file names start with NB_ID.

Methods: 'nb01_r1' ... 'nb01_r9' are the nine Table-1 rows that nb01 scored (rows 1-6 HM's Split rows with the
official code, 7-8 LayerCAM, 9 Random); 'hm_true_rK' and 'gauge_rK' are the rows nb03 scored (K = 1..6).
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NB_ID = "nb03"
MAP_FAMILIES = ["hm", "hm_true", "gauge"]          # columns of the map figure
SCORE_FAMILIES = ["hm", "hm_true", "gauge"]        # 'hm' = nb01's scores of the official code
SPLIT_ROWS = ["r1", "r2", "r3", "r4", "r5", "r6"]
OTHER_METHODS = ["nb01_r7", "nb01_r8", "nb01_r9"]   # LayerCAM (14), LayerCAM (2), Random
COMPARISONS = ["gauge-hm", "gauge-hm_true", "hm_true-hm"]
# validated categorical set (dataviz validate_palette.js, light surface, all pairs): violet, yellow, aqua
FAMILY_COLORS = {"hm": "#6250d6", "hm_true": "#eda100", "gauge": "#1baf7a"}
FAMILY_MARKERS = {"hm": "o", "hm_true": "s", "gauge": "D"}
DERIV_COLORS = {"official": "#6250d6", "true": "#1baf7a"}       # the colours of 'hm' and of 'gauge'
PAIR_COLORS = ["#2a78d6", "#eb6834"]                            # blue, orange (validated pair), no family colour
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
REF_STYLES = {"nb01_r7": dict(color=INK2, ls="--"), "nb01_r8": dict(color=INK2, ls=":"),
              "nb01_r9": dict(color=MUTED, ls="-.")}
# diverging blue (better) <-> red (worse) with a neutral grey midpoint, equal steps per arm
DIVERGING = LinearSegmentedColormap.from_list(
    "worse_better", ["#a42c2b", "#e34948", "#f3aeae", "#f0efec", "#86b6ef", "#2a78d6", "#1c5cab"])
METRIC_KEYS = ["selectivity", "attribution_localisation", "pointing_game", "pf_auc5", "pf_auc20", "max_sensitivity"]
RESULT_COLUMNS = {"selectivity": "selectivity_pos", "attribution_localisation": "attribution_localisation",
                  "pointing_game": "pointing_game", "pf_auc5": "pixel_flipping_pos_auc@5",
                  "pf_auc20": "pixel_flipping_pos_auc@20", "max_sensitivity": "max_sensitivity"}
DIRECTION = {"selectivity": -1, "attribution_localisation": 1, "pointing_game": 1, "pf_auc5": 1, "pf_auc20": 1,
             "max_sensitivity": -1}
HM_COMBOS = [("convex", "features_12"), ("convex", "features_14"), ("wta", "features_12"), ("wta", "features_14"),
             ("wta", "features_26")]

_LABELS = {
    "families": {"hm": "HM (code chính thức, nb01)", "hm_true": "HM (đạo hàm đúng)", "gauge": "Gauge-X"},
    "rows": {"r1": "SplitCAM (12, g)", "r2": "SplitCAM (14, g)", "r3": "SplitCAM (26, +, g)",
             "r4": "SplitGrad (12, +, g)", "r5": "SplitGrad (14, +, g)", "r6": "SplitLRP (26)"},
    "others": {"nb01_r7": "LayerCAM (14)", "nb01_r8": "LayerCAM (2)", "nb01_r9": "Random"},
    "comparisons": {"gauge-hm": "Gauge-X − HM (chính thức)", "gauge-hm_true": "Gauge-X − HM (đạo hàm đúng)",
                    "hm_true-hm": "HM (đạo hàm đúng) − HM (chính thức)"},
    "posts": {"none": "giữ dấu", "abs": "|·|"},
    "pools": {"convex": "maxpool lồi", "wta": "maxpool wta"},
    "derivatives": {"true": "đạo hàm đúng", "official": "code chính thức (đổi chỗ ở conv)"},
    "pairs": {"hm~hm_true": "HM chính thức ~ HM đạo hàm đúng", "gauge~hm": "Gauge-X ~ HM chính thức",
              "gauge~hm_true": "Gauge-X ~ HM đạo hàm đúng"},
    "stages": {"load": "nạp ảnh", "maps": "bản đồ (3 họ, 6 hàng)", "compare": "cầu nối + so bản đồ",
               "metrics4": "4 thước đo tất định", "maxsens": "Max Sens", "total": "cả ảnh"},
    "metrics": {"selectivity": "Select. ↓", "attribution_localisation": "Attr. Loc. ↑", "pointing_game": "Point. ↑",
                "pf_auc5": "PF AUC@5 ↑", "pf_auc20": "PF AUC@20 ↑", "max_sensitivity": "Max Sens. ↓",
                "spearman": "Spearman", "ssim": "SSIM", "top10": "trùng top-10%"},
    "columns": {
        "param": "tham số", "value": "giá trị", "reason": "lý do", "check": "kiểm tra", "passed": "đạt",
        "detail": "chi tiết", "row": "hàng", "family": "họ", "pool": "maxpool", "derivative": "đạo hàm",
        "layer": "tầng", "kind": "loại bản đồ", "post": "hậu xử lý", "hm_post": "hậu xử lý của HM",
        "dtype": "kiểu số", "rel_diff": "sai lệch tương đối", "tol": "ngưỡng", "alpha_star": "α*", "t_star": "t*",
        "mean": "trung bình", "sd": "độ lệch chuẩn", "min": "nhỏ nhất", "max": "lớn nhất", "n": "số ảnh",
        "balanced": "đạt cân bằng", "ratio_star": "tỉ số tại t*", "in_band": "trong dải H1", "hm_uses": "HM dùng",
        "pair": "cặp", "method": "phương pháp", "seconds": "giây/ảnh", "peak_gib": "đỉnh bộ nhớ (GiB)",
        "image": "ảnh", "image_name": "tên ảnh", "target": "lớp được giải thích", "confidence": "độ tin cậy",
        "quantity": "đại lượng", "note": "ghi chú", "hypothesis": "giả thuyết", "verdict": "kết luận",
        "metric": "thước đo", "nb01": "nb01", "ours": "nb03", "stage": "giai đoạn", "comparison": "so sánh",
        "se": "SE", "z": "z", "mark": "kết quả", "spearman": "Spearman", "off_pixels": "tỉ lệ pixel lệch",
        "n_pass": "số ảnh đạt", "worst_rel_diff": "sai lệch lớn nhất", "worst_spearman": "Spearman nhỏ nhất",
        "worst_off_pixels": "tỉ lệ pixel lệch lớn nhất", "gpu": "GPU", "worker": "worker", "images": "số ảnh",
        "median": "trung vị", "leader": "dẫn đầu", "best_gauge": "Gauge-X tốt nhất", "best_hm": "HM tốt nhất",
        "best_layercam": "LayerCAM tốt nhất", "vs_hm": "Gauge-X − HM", "vs_layercam": "Gauge-X − LayerCAM",
        "n_undefined": "số ảnh không xác định",
    },
    "titles": {
        "alpha": "α* theo quy tắc cân bằng, từng ảnh test (vạch đen: trung bình; vạch đứt: α = 0.4 của HM; dải xám: dải H1)",
        "ratio": "‖Γ(t)‖ / ‖∂f/∂a‖ theo t = 1 − 2α (trung vị và khoảng tứ phân vị qua các ảnh)",
        "swap": "Độ tương đồng giữa các bản đồ (Spearman, trung bình ± SE qua ảnh)",
        "maps": "Bản đồ của ba họ trên ảnh test {image} (lớp {target}); hậu xử lý như HM",
        "scores": "6 thước đo trên ảnh test (trung bình ± SE theo ảnh), hậu xử lý như HM",
        "paired": "Hiệu ghép cặp theo ảnh, z = hiệu / SE theo hướng tốt (xanh: tốt hơn, đỏ: kém hơn); "
                  "ô ghi kết quả (+ / = / −, ngưỡng 2·SE) và z",
    },
    "axes": {"alpha": "α*", "t": "t = 1 − 2α", "ratio": "‖Γ‖ / ‖∂f/∂a‖ (thang log)", "spearman": "Spearman",
             "hm_alpha": "α của HM = 0.4", "band": "dải H1", "original": "ảnh gốc + mask",
             "z": "z (dương: tốt hơn)", "same_as_hm": "= HM"},
}


def default_labels():
    """A fresh, editable copy of the default labels."""
    return copy.deepcopy(_LABELS)


def _dirs(run_dir):
    run_dir = Path(run_dir)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    return run_dir / "data", run_dir / "figures", run_dir / "tables"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def _ref_labels(ax, items, min_gap=0.075):
    """Direct labels [(y, text)] of horizontal reference lines at the right edge of `ax`, nudged apart (in axes
    fraction) so that close lines do not print on top of each other. Call after the data limits are final."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    pos = sorted(((y - lo) / (hi - lo), text) for y, text in items)
    ys = [p for p, _ in pos]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    if ys[-1] > 1.0:
        ys[-1] = 1.0
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
    for y, (_, text) in zip(ys, pos):
        ax.annotate(text, (1.0, y), xycoords="axes fraction", xytext=(3, 0), textcoords="offset points",
                    fontsize=6.5, color=INK2, va="center", ha="left")


def method_label(method, labels):
    """'gauge_r3' -> 'SplitCAM (26, +, g) · Gauge-X'; 'nb01_r2' -> '... · HM (code chính thức, nb01)'."""
    if method in labels["others"]:
        return labels["others"][method]
    fam, _, row = method.rpartition("_")
    if row not in labels["rows"]:                  # not a method key (e.g. the 'method' column of rows.csv)
        return method
    fam = "hm" if fam == "nb01" else fam
    return f"{labels['rows'].get(row, row)} · {labels['families'].get(fam, fam)}"


def _display(t, labels):
    """Replace key values (row, family, method, ...) by display names and rename the columns."""
    t = t.copy()
    for col, key in (("row", "rows"), ("family", "families"), ("pool", "pools"), ("derivative", "derivatives"),
                     ("post", "posts"), ("hm_post", "posts"), ("pair", "pairs"), ("metric", "metrics"),
                     ("comparison", "comparisons"), ("stage", "stages")):
        if col in t.columns:
            t[col] = t[col].map(lambda v, k=key: labels[k].get(v, v) if isinstance(v, str) else v)
    for col in ("method", "leader", "best_gauge", "best_hm", "best_layercam"):
        if col in t.columns:
            t[col] = t[col].map(lambda v: method_label(v, labels) if isinstance(v, str) else v)
    return t.rename(columns=labels["columns"])


def show_csv(run_dir, name, labels=None, **read_kw):
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    return _display(pd.read_csv(data / name, **read_kw), labels)


def show_env(run_dir, labels=None):
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    env = json.loads((data / "env.json").read_text(encoding="utf-8"))
    env = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in env.items()}
    return pd.Series(env, name=labels["columns"]["value"]).to_frame()


def _se(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def _pm(m, s, nd=3):
    if not np.isfinite(m):
        return "–"
    return f"{m:.{nd}f}" if not np.isfinite(s) else f"{m:.{nd}f} ± {s:.{nd}f}"


def mark(mean, se, direction):
    """'+' better by more than 2 SE, '−' worse by more than 2 SE, '=' within 2 SE, '?' without an SE."""
    if not np.isfinite(se) or se == 0:
        return "?"
    if abs(mean) <= 2 * se:
        return "="
    return "+" if direction * mean > 0 else "−"


# ------------------------------------------------------------------------------------------ alpha*

def read_alpha(run_dir):
    """alpha_star.csv with 'derivative' kept as text: pandas reads a column holding only 'true' as booleans."""
    data, _, _ = _dirs(run_dir)
    return pd.read_csv(data / "alpha_star.csv", dtype={"derivative": str})


def alpha_table(run_dir, labels=None, save=True, band=(0.33, 0.42)):
    """alpha* per (pool, derivative, layer): mean, sd, range over images; which combinations HM's rows use."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    a = read_alpha(run_dir)
    g = a.groupby(["pool", "derivative", "layer"])
    t = g["alpha_star"].agg(["mean", "std", "min", "max", "count"]).rename(columns={"std": "sd", "count": "n"})
    t["balanced"] = g["balanced"].mean()
    t["ratio_star"] = g["ratio_star"].median()
    t = t.reset_index()
    t["hm_uses"] = [(p, l) in HM_COMBOS for p, l in zip(t["pool"], t["layer"])]
    t["in_band"] = (t["mean"] >= band[0]) & (t["mean"] <= band[1])
    if save:
        t.to_csv(tables / f"{NB_ID}_alpha_star.csv", index=False)
    return _display(t.round(4), labels)


def plot_alpha(run_dir, labels=None, save=True, band=(0.33, 0.42), alpha_hm=0.4):
    labels = labels or _LABELS
    data, figs, _ = _dirs(run_dir)
    a = read_alpha(run_dir)
    combos = [c for c in HM_COMBOS if ((a["pool"] == c[0]) & (a["layer"] == c[1])).any()]
    ders = [d for d in ("true", "official") if (a["derivative"] == d).any()]
    fig, ax = plt.subplots(figsize=(1.6 * len(combos) + 2.4, 3.6))
    _style(ax)
    ax.axhspan(*band, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(alpha_hm, color=INK2, lw=1.0, ls="--", zorder=1)
    rng = np.random.default_rng(0)
    for i, (pool, layer) in enumerate(combos):
        for j, der in enumerate(ders):
            v = a[(a["pool"] == pool) & (a["layer"] == layer) & (a["derivative"] == der)]["alpha_star"].to_numpy()
            if not len(v):
                continue
            x0 = i + (j - (len(ders) - 1) / 2) * 0.34
            ax.scatter(x0 + rng.uniform(-0.1, 0.1, len(v)), v, s=10, color=DERIV_COLORS[der], alpha=0.5, lw=0,
                       zorder=3)
            ax.plot([x0 - 0.14, x0 + 0.14], [np.mean(v)] * 2, color=INK, lw=1.8, zorder=4)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"{labels['pools'][p]}\n{l.replace('features_', 'tầng ')}" for p, l in combos], fontsize=8)
    ax.set_ylim(min(band[0], a["alpha_star"].min()) - 0.02, max(band[1], alpha_hm, a["alpha_star"].max()) + 0.02)
    ax.set_ylabel(labels["axes"]["alpha"], color=INK2)
    ax.set_title(labels["titles"]["alpha"], fontsize=9, color=INK, loc="left")
    handles = [Line2D([], [], marker="o", ls="", color=DERIV_COLORS[d], label=labels["derivatives"][d]) for d in ders]
    handles += [Line2D([], [], color=INK2, ls="--", label=labels["axes"]["hm_alpha"]),
                Patch(color=GRID, label=f"{labels['axes']['band']} [{band[0]}, {band[1]}]")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    if save:
        fig.savefig(figs / f"{NB_ID}_alpha_star.png", dpi=150, bbox_inches="tight")
    return fig


def plot_ratio_curves(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, figs, _ = _dirs(run_dir)
    z = np.load(data / "ratio_curves.npz", allow_pickle=False)
    ts, curves, keys = z["ts"], z["curves"], [str(k) for k in z["keys"]]   # curves: (image, key, t)
    combos = [c for c in HM_COMBOS if any(k.startswith(f"{c[0]}|") and k.endswith(f"|{c[1]}") for k in keys)]
    fig, axes = plt.subplots(1, len(combos), figsize=(2.7 * len(combos), 3.0), sharey=True, squeeze=False)
    for ax, (pool, layer) in zip(axes[0], combos):
        _style(ax)
        for der in ("true", "official"):
            key = f"{pool}|{der}|{layer}"
            if key not in keys:
                continue
            c = np.clip(curves[:, keys.index(key), :], 1e-6, None)
            med, lo, hi = np.median(c, 0), np.quantile(c, 0.25, 0), np.quantile(c, 0.75, 0)
            ax.fill_between(ts, lo, hi, color=DERIV_COLORS[der], alpha=0.18, lw=0)
            ax.plot(ts, med, color=DERIV_COLORS[der], lw=2, label=labels["derivatives"][der])
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.axvline(0.2, color=INK2, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e4)
        ax.set_xlim(0, 0.6)
        ax.set_title(f"{labels['pools'][pool]}, {layer.replace('features_', 'tầng ')}", fontsize=8.5, color=INK)
        ax.set_xlabel(labels["axes"]["t"], color=INK2, fontsize=8)
    axes[0][0].set_ylabel(labels["axes"]["ratio"], color=INK2, fontsize=8)
    axes[0][-1].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(labels["titles"]["ratio"], fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    if save:
        fig.savefig(figs / f"{NB_ID}_ratio_curves.png", dpi=150, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------------------- bridge, swap effect

def bridge_table(run_dir, labels=None, save=True):
    """Per row: how closely the engine's 'hm' maps reproduce the maps nb01 saved, over all images."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    b = pd.read_csv(data / "bridge.csv")
    t = b.groupby("row", sort=False).agg(n=("image", "nunique"), n_pass=("passed", "sum"),
                                         worst_rel_diff=("rel_diff", "max"), worst_spearman=("spearman", "min"),
                                         worst_off_pixels=("off_pixels", "max")).reset_index()
    if save:
        t.to_csv(tables / f"{NB_ID}_bridge.csv", index=False)
    return _display(t, labels)


def swap_table(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    s = pd.read_csv(data / "swap_effect.csv")
    rows = []
    for (row, pair), g in s.groupby(["row", "pair"], sort=False):
        r = {"row": row, "pair": pair}
        for m in ("spearman", "ssim", "top10"):
            r[m] = _pm(g[m].mean(), _se(g[m]), 2)
        rows.append(r)
    t = pd.DataFrame(rows)
    if save:
        t.to_csv(tables / f"{NB_ID}_swap_effect.csv", index=False)
    t = _display(t, labels)
    return t.rename(columns={m: labels["metrics"][m] for m in ("spearman", "ssim", "top10")})


def plot_swap(run_dir, labels=None, save=True, pairs=("hm~hm_true", "gauge~hm")):
    labels = labels or _LABELS
    data, figs, _ = _dirs(run_dir)
    s = pd.read_csv(data / "swap_effect.csv")
    rows = list(dict.fromkeys(s["row"]))
    pairs = [p for p in pairs if (s["pair"] == p).any()]
    fig, ax = plt.subplots(figsize=(1.3 * len(rows) + 2.8, 3.2))
    _style(ax)
    w = 0.8 / len(pairs)
    low = -0.2
    for j, pair in enumerate(pairs):
        g = s[s["pair"] == pair].groupby("row")["spearman"]
        m, e = g.mean().reindex(rows), g.apply(_se).reindex(rows)
        x = np.arange(len(rows)) + (j - (len(pairs) - 1) / 2) * w
        ax.bar(x, m, width=w - 0.03, color=PAIR_COLORS[j], label=labels["pairs"][pair], zorder=3)
        ax.errorbar(x, m, yerr=e, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=4)
        low = min(low, float(np.nanmin(m - e.fillna(0))) - 0.05)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([labels["rows"].get(r, r) for r in rows], fontsize=8, rotation=15, ha="right")
    ax.set_ylim(max(low, -1.05), 1.05)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel(labels["axes"]["spearman"], color=INK2)
    ax.set_title(labels["titles"]["swap"], fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    if save:
        fig.savefig(figs / f"{NB_ID}_swap_effect.png", dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------------------------- maps

def plot_maps(run_dir, labels=None, image=0, save=True):
    """Maps of every row (lines) and family (columns) on one test image, post-processed as HM's row."""
    import torch
    import torch.nn.functional as F
    labels = labels or _LABELS
    data, figs, _ = _dirs(run_dir)
    rows = pd.read_csv(data / "rows.csv")
    imgs = pd.read_csv(data / "images.csv")
    ex = np.load(data / "fig_images.npz")
    if f"image_{image}" not in ex.files:
        return None
    z = np.load(data / "maps" / f"image_{image:03d}.npz")
    target = int(imgs.loc[imgs["image"] == image, "target"].iloc[0])
    fig, axes = plt.subplots(len(rows), 1 + len(MAP_FAMILIES), figsize=(1.6 * (1 + len(MAP_FAMILIES)), 1.55 * len(rows)),
                             squeeze=False)
    for i, r in enumerate(rows.itertuples()):
        ax = axes[i][0]
        ax.imshow(ex[f"image_{image}"])
        msk = ex[f"mask_{image}"]
        ax.contour(msk[..., 0] if msk.ndim == 3 else msk, levels=[0.5], colors=["#ffffff"], linewidths=0.8)
        ax.set_ylabel(labels["rows"].get(r.id, r.id), fontsize=7, color=INK, rotation=0, ha="right", va="center")
        for j, fam in enumerate(MAP_FAMILIES):
            ax = axes[i][j + 1]
            key = f"{r.id}|{fam}"
            if key in z.files:
                m = torch.from_numpy(z[key])[None, None].float()
                m = F.interpolate(m, size=(224, 224), mode="nearest")[0, 0].numpy()
                if r.hm_post == "abs":
                    m = np.abs(m)
                    ax.imshow(m / max(m.max(), 1e-30), cmap="Blues", vmin=0, vmax=1)
                else:
                    v = max(np.abs(m).max(), 1e-30)
                    ax.imshow(m / v, cmap="RdBu_r", vmin=-1, vmax=1)
            else:
                ax.text(0.5, 0.5, labels["axes"]["same_as_hm"], ha="center", va="center", fontsize=8, color=MUTED,
                        transform=ax.transAxes)
            if i == 0:
                ax.set_title(labels["families"][fam], fontsize=7, color=INK)
    if len(rows):
        axes[0][0].set_title(labels["axes"]["original"], fontsize=7, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(labels["titles"]["maps"].format(image=image, target=target), fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    if save:
        fig.savefig(figs / f"{NB_ID}_maps_image{image:03d}.png", dpi=150, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------------------------------ scores

def table1(run_dir, labels=None, save=True):
    """Table 1 extended: mean ± SE over images of every method (nb01's nine rows and nb03's rows) and metric."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    s = pd.read_csv(data / "summary.csv")
    order = [f"{f}_{r}" for r in SPLIT_ROWS for f in ("nb01", "hm_true", "gauge")] + OTHER_METHODS
    s = s.assign(cell=[_pm(m, e) for m, e in zip(s["mean"], s["se"])])
    t = s.pivot_table(index="method", columns="metric", values="cell", aggfunc="first")
    t = t.reindex(index=[m for m in order if m in t.index], columns=[k for k in METRIC_KEYS if k in t.columns])
    t = t.reset_index()
    if save:
        t.to_csv(tables / f"{NB_ID}_table1.csv", index=False)
    t = _display(t, labels)
    return t.rename(columns={k: labels["metrics"][k] for k in METRIC_KEYS})


def paired_table(run_dir, labels=None, comparison="gauge-hm", save=True):
    """Per Split row and metric: mean over images of the paired difference ± SE and the mark (+ / = / −)."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    p = pd.read_csv(data / "paired.csv")
    p = p[p["comparison"] == comparison]
    p = p.assign(cell=[f"{_pm(m, e)} {k}" for m, e, k in zip(p["mean"], p["se"], p["mark"])])
    t = p.pivot_table(index="row", columns="metric", values="cell", aggfunc="first")
    t = t.reindex(index=[r for r in SPLIT_ROWS if r in t.index], columns=[k for k in METRIC_KEYS if k in t.columns])
    t = t.reset_index()
    if save:
        t.to_csv(tables / f"{NB_ID}_paired_{comparison}.csv", index=False)
    t = _display(t, labels)
    return t.rename(columns={k: labels["metrics"][k] for k in METRIC_KEYS})


def plot_scores(run_dir, labels=None, save=True):
    """One panel per metric: the Split rows on x, the three families as markers with SE bars; LayerCAM and Random
    as reference lines."""
    labels = labels or _LABELS
    data, figs, _ = _dirs(run_dir)
    s = pd.read_csv(data / "summary.csv").set_index(["method", "metric"])
    keys = [k for k in METRIC_KEYS if k in s.index.get_level_values("metric")]
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        _style(ax)
        for j, fam in enumerate(SCORE_FAMILIES):
            src = "nb01" if fam == "hm" else fam
            x, m, e = [], [], []
            for i, r in enumerate(SPLIT_ROWS):
                if (f"{src}_{r}", k) in s.index:
                    x.append(i + (j - 1) * 0.22)
                    m.append(s.loc[(f"{src}_{r}", k), "mean"])
                    e.append(s.loc[(f"{src}_{r}", k), "se"])
            ax.errorbar(x, m, yerr=e, fmt=FAMILY_MARKERS[fam], ms=6, color=FAMILY_COLORS[fam], mec="#fcfcfb", mew=0.8,
                        ecolor=FAMILY_COLORS[fam], elinewidth=1.2, capsize=0, lw=0, zorder=3)
        refs = []
        for meth in OTHER_METHODS:
            if (meth, k) in s.index and np.isfinite(s.loc[(meth, k), "mean"]):
                y = s.loc[(meth, k), "mean"]
                ax.axhline(y, lw=1.1, zorder=2, **REF_STYLES[meth])
                refs.append((y, labels["others"][meth]))
        _ref_labels(ax, refs)
        ax.set_xticks(range(len(SPLIT_ROWS)))
        ax.set_xticklabels([labels["rows"][r] for r in SPLIT_ROWS], fontsize=7, rotation=30, ha="right")
        ax.set_title(labels["metrics"][k], fontsize=9, color=INK, loc="left")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    handles = [Line2D([], [], marker=FAMILY_MARKERS[f], ls="", color=FAMILY_COLORS[f], label=labels["families"][f])
               for f in SCORE_FAMILIES]
    handles += [Line2D([], [], lw=1.1, label=labels["others"][m], **REF_STYLES[m]) for m in OTHER_METHODS]
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(labels["titles"]["scores"], fontsize=9, color=INK, x=0.01, ha="left", y=1.08)
    fig.tight_layout(w_pad=4.0)
    if save:
        fig.savefig(figs / f"{NB_ID}_scores.png", dpi=150, bbox_inches="tight")
    return fig


def plot_paired(run_dir, labels=None, save=True, zmax=6.0):
    """Heat map of the paired comparisons: rows x metrics, colour = z in the direction of 'better'."""
    labels = labels or _LABELS
    data, figs, _ = _dirs(run_dir)
    p = pd.read_csv(data / "paired.csv")
    comps = [c for c in COMPARISONS if (p["comparison"] == c).any()]
    keys = [k for k in METRIC_KEYS if (p["metric"] == k).any()]
    fig, axes = plt.subplots(1, len(comps), figsize=(3.6 * len(comps) + 2.2, 3.3), squeeze=False, sharey=True,
                             gridspec_kw={"wspace": 0.06})
    im = None
    for c, (ax, comp) in enumerate(zip(axes[0], comps)):
        g = p[p["comparison"] == comp].set_index(["row", "metric"])
        Z = np.full((len(SPLIT_ROWS), len(keys)), np.nan)
        for i, r in enumerate(SPLIT_ROWS):
            for j, k in enumerate(keys):
                if (r, k) not in g.index:
                    continue
                m, e = g.loc[(r, k), "mean"], g.loc[(r, k), "se"]
                z = DIRECTION[k] * m / e if np.isfinite(e) and e > 0 else 0.0
                Z[i, j] = np.clip(z, -zmax, zmax)
                ink = "#ffffff" if abs(Z[i, j]) > 0.6 * zmax else INK
                txt = f"{g.loc[(r, k), 'mark']} {z:+.1f}" if np.isfinite(e) and e > 0 else g.loc[(r, k), "mark"]
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=ink)
        im = ax.imshow(np.ma.masked_invalid(Z), cmap=DIVERGING, vmin=-zmax, vmax=zmax, aspect="auto")
        for i, r in enumerate(SPLIT_ROWS):
            if np.isnan(Z[i]).all():
                ax.text((len(keys) - 1) / 2, i, labels["axes"]["same_as_hm"], ha="center", va="center", fontsize=7,
                        color=MUTED)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([labels["metrics"][k] for k in keys], fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(len(SPLIT_ROWS)))
        if c == 0:
            ax.set_yticklabels([labels["rows"][r] for r in SPLIT_ROWS], fontsize=7)
        else:
            ax.tick_params(left=False, labelleft=False)
        ax.set_title(labels["comparisons"][comp], fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(np.arange(len(keys) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(SPLIT_ROWS) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#fcfcfb", lw=2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(colors=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    cb.set_label(labels["axes"]["z"], fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(labels["titles"]["paired"], fontsize=9, color=INK, x=0.01, ha="left", y=1.02)
    if save:
        fig.savefig(figs / f"{NB_ID}_paired.png", dpi=150, bbox_inches="tight")
    return fig


def best_table(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "best_rows.csv")
    for c in ("vs_hm", "vs_layercam"):
        t[c] = [f"{_pm(m, e)} {k}" for m, e, k in zip(t[f"{c}_mean"], t[f"{c}_se"], t[f"{c}_mark"])]
    t = t[["metric", "leader", "best_gauge", "best_hm", "vs_hm", "best_layercam", "vs_layercam"]]
    if save:
        t.to_csv(tables / f"{NB_ID}_best_rows.csv", index=False)
    return _display(t, labels)


# ---------------------------------------------------------------------------------- timing, verdict

def timing_table(run_dir, labels=None, save=True):
    """Median and mean seconds per image of every stage of the worker pipeline."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "timing.csv")
    stages = [s for s in ("load", "maps", "compare", "metrics4", "maxsens", "total") if s in t.columns]
    out = pd.DataFrame([dict(stage=s, median=t[s].median(), mean=t[s].mean(), max=t[s].max()) for s in stages])
    if save:
        out.to_csv(tables / f"{NB_ID}_timing.csv", index=False)
    return _display(out.round(2), labels)


def gpu_table(run_dir, labels=None, save=True):
    """Images, mean seconds per image and peak memory per (GPU, worker)."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "timing.csv", dtype={"gpu": str})
    g = t.groupby(["gpu", "worker"]).agg(images=("image", "count"), seconds=("total", "mean"),
                                         peak_gib=("peak_gib", "max")).reset_index()
    if save:
        g.to_csv(tables / f"{NB_ID}_workers.csv", index=False)
    return _display(g.round(2), labels)


def verdict_table(run_dir, labels=None):
    labels = labels or _LABELS
    v = json.loads((Path(run_dir) / f"{NB_ID}_verdict.json").read_text(encoding="utf-8"))
    return _display(pd.DataFrame(v["rows"]), labels)


def redraw_all(run_dir, labels=None):
    """Redraw every table and figure whose data exist; returns {name: object}."""
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    out = {}
    jobs = [("alpha_star.csv", "alpha_table", alpha_table), ("alpha_star.csv", "alpha_plot", plot_alpha),
            ("ratio_curves.npz", "ratio_curves", plot_ratio_curves), ("bridge.csv", "bridge", bridge_table),
            ("swap_effect.csv", "swap_table", swap_table), ("swap_effect.csv", "swap_plot", plot_swap),
            ("summary.csv", "table1", table1), ("summary.csv", "scores_plot", plot_scores),
            ("paired.csv", "paired_plot", plot_paired), ("best_rows.csv", "best_rows", best_table),
            ("timing.csv", "timing", timing_table), ("timing.csv", "workers", gpu_table)]
    for f, name, fn in jobs:
        if (data / f).exists():
            out[name] = fn(run_dir, labels)
            if hasattr(out[name], "savefig"):
                plt.close(out[name])
    if (data / "paired.csv").exists():
        for comp in COMPARISONS:
            out[f"paired_{comp}"] = paired_table(run_dir, labels, comp)
    if (data / "fig_images.npz").exists():
        for k in [int(n.split("_")[1]) for n in np.load(data / "fig_images.npz").files if n.startswith("image_")]:
            fig = plot_maps(run_dir, labels, image=k)
            if fig is not None:
                out[f"maps_{k}"] = fig
                plt.close(fig)
    return out
