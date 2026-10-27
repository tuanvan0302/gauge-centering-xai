"""Tables and figures of notebook nb02, built only from the files a run saved in RUN_DIR/data.

Nothing here runs a network, so titles, column names, method and network names can be changed (pass a
modified copy of default_labels()) and everything redrawn in seconds. Data files use fixed English keys
(depth, method, err_val, ...); display names come from the labels. Figures go to RUN_DIR/figures and
tables to RUN_DIR/tables; their file names start with NB_ID.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NB_ID = "nb02"
U = 2.0 ** -53
METHODS = ["A", "B", "C"]
COLORS = {"A": "#eb6834", "B": "#6250d6", "C": "#2a78d6"}
MARKERS = {"A": "o", "B": "D", "C": "s"}
MUTED, INK = "#8a8984", "#0b0b0b"
FLOOR, CEIL = 1e-18, 1e3

_LABELS = {
    "methods": {"A": "A · DC ngây thơ", "B": "B · HM (sc + cache)", "C": "C · gauge centring"},
    "networks": {"mlp": "ReLU MLP", "gate": "Mạng cổng", "vgg16": "VGG16", "mha_linear": "MHA linear",
                 "mha_softmax": "MHA softmax"},
    "depth_axis": {"mlp": "độ sâu $L$ (số tầng)", "gate": "độ sâu $L$ (số tầng)", "vgg16": "tầng",
                   "mha_linear": "số khối $N$", "mha_softmax": "số khối $N$"},
    "columns": {
        "param": "tham số", "value": "giá trị", "reason": "lý do", "key": "mục",
        "check": "kiểm tra", "network": "mạng", "tol": "ngưỡng", "expected": "kỳ vọng", "passed": "đạt",
        "detail": "chi tiết", "depth": "độ sâu", "method": "phương pháp", "f_true": "f (đúng)",
        "err_val": "sai số giá trị", "err_grad": "sai số gradient", "chi_val": "χ giá trị",
        "chi_grad": "χ gradient", "overflow": "tràn", "mem_weights": "bộ nhớ trọng số",
        "mem_cache": "bộ nhớ cache", "mem_autograd": "bộ nhớ autograd", "mem_total": "bộ nhớ tổng",
        "time_s": "thời gian (s)", "n_scaled": "số lần co", "maxabs_max": "max|cặp|",
        "image": "ảnh", "wnid": "lớp", "file": "file", "target": "lớp dự đoán", "prob": "xác suất",
        "f_ref": "f (tham chiếu)", "node": "chunk", "quantity": "đại lượng",
        "rel_diff_official": "lệch so với HM (B như code)", "rel_diff_true": "lệch so với HM (đạo hàm thật)",
        "scale": "‖HM‖", "attn": "attention", "steps": "số bước", "test_mse": "MSE kiểm tra",
        "baseline_mse": "MSE dự đoán 0", "skipped": "bước bỏ qua", "seconds": "giây", "source": "nguồn",
        "max_depth": "độ sâu lớn nhất", "certified_val": "L* giá trị", "certified_grad": "L* gradient",
        "max_err_val": "sai số giá trị lớn nhất", "max_err_grad": "sai số gradient lớn nhất",
        "mem_at_max": "bộ nhớ ở độ sâu lớn nhất", "time_at_max": "thời gian ở độ sâu lớn nhất (s)",
        "needs_cache": "cần cache mạng gốc", "hyperparams": "siêu tham số", "id": "#", "claim": "dự đoán",
        "rule": "quy tắc", "measured": "đo được", "status": "kết luận",
    },
    "panels": {"err_val": "Sai số tương đối của giá trị", "err_grad": "Sai số tương đối của gradient",
               "mem": "Bộ nhớ phải giữ", "chi": "$\\log_{10}\\chi$ của cặp theo tầng",
               "layer_err": "Sai số $\\|(a^+-a^-)-a\\|/\\|a\\|$ theo tầng", "time": "Thời gian (giây)",
               "chunks": "Độ lớn cặp của B sau từng chunk", "summary": "Độ chính xác và bộ nhớ ở độ sâu lớn nhất"},
    "axes": {"err": "sai số tương đối (float64)", "mem": "byte", "chi": "$\\log_{10}\\chi$",
             "maxabs": "max |cặp| sau ổn định", "chunk": "chunk (theo thứ tự)", "time": "giây"},
    "notes": {"pred": "dự đoán $u\\,\\chi_A$ (Bổ đề 3)", "overflow": "A tràn float64", "delta": "$\\delta$",
              "theta": "$\\Theta$", "pre": "B trước khi sửa", "na": "không định nghĩa",
              "mem_kinds": {"mem_weights": "trọng số", "mem_cache": "cache", "mem_autograd": "autograd"}},
    "titles": {"mlp": "ReLU MLP (trọng số cố định), float64", "gate": "Mạng cổng vô hướng, float64",
               "vgg16": "VGG16, float64", "mha_linear": "MHA linear (đã huấn luyện), float64",
               "mha_softmax": "MHA softmax (đã huấn luyện), float64"},
}


def default_labels():
    """A fresh, editable copy of the default labels."""
    return copy.deepcopy(_LABELS)


def _dirs(run_dir):
    run_dir = Path(run_dir)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    return run_dir / "data", run_dir / "figures", run_dir / "tables"


def _relabel(t, labels):
    t = t.copy()
    if "method" in t.columns:
        t["method"] = t["method"].map(lambda m: labels["methods"].get(m, m))
    if "network" in t.columns:
        t["network"] = t["network"].map(lambda m: labels["networks"].get(m, m))
    return t.rename(columns=labels["columns"])


def show_csv(run_dir, name, labels=None, **read_kw):
    """data/<name> with method, network and column keys replaced by display names."""
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    return _relabel(pd.read_csv(data / name, **read_kw), labels)


def show_env(run_dir, labels=None):
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    env = json.loads((data / "env.json").read_text(encoding="utf-8"))
    return pd.DataFrame({"key": list(env), "value": [str(v) for v in env.values()]}).rename(columns=labels["columns"])


def fmt_bytes(b):
    if not np.isfinite(b):
        return "—"
    for unit, k in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if b >= k:
            return f"{b / k:.2f} {unit}"
    return f"{int(b)} B"


def _fmt_err(e):
    return "tràn" if not np.isfinite(e) else ("0" if e == 0 else f"{e:.1e}")


def _fmt_worst(e):
    """Largest error of a summary row: NaN = not defined, inf = some depth overflowed."""
    if np.isnan(e):
        return "—"
    return "tràn" if np.isinf(e) else ("0" if e == 0 else f"{e:.1e}")


def _clip(e):
    e = np.asarray(e, dtype=float)
    return np.clip(np.where(e == 0, FLOOR, e), FLOOR, CEIL)


def _style(ax):
    ax.grid(alpha=0.25, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------------- per-depth networks

def depth_table(run_dir, network, labels=None, depths=None, save=True):
    """Depth x method table of the value error, gradient error, chi and memory."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / f"{network}.csv")
    if depths is not None:
        t = t[t["depth"].isin(depths)]
    rows = []
    for d, g in t.groupby("depth"):
        r = {labels["columns"]["depth"]: d}
        for _, x in g.sort_values("method").iterrows():
            m = labels["methods"].get(x["method"], x["method"])
            r[f"{labels['columns']['err_val']} · {m}"] = _fmt_err(x["err_val"]) if not x["overflow"] else "tràn"
            r[f"{labels['columns']['err_grad']} · {m}"] = (_fmt_err(x["err_grad"]) if np.isfinite(x["err_grad"])
                                                         or x["overflow"] else labels["notes"]["na"])
            r[f"{labels['columns']['chi_val']} · {m}"] = f"{x['chi_val']:.2e}" if np.isfinite(x["chi_val"]) else "—"
            r[f"{labels['columns']['mem_total']} · {m}"] = fmt_bytes(x["mem_total"])
        rows.append(r)
    out = pd.DataFrame(rows)
    if save:
        out.to_csv(tables / f"{NB_ID}_{network}.csv", index=False)
    return out


def plot_depth(run_dir, network, labels=None, save=True):
    """Three panels against depth: value error (with the prediction u*chi_A), gradient error, memory."""
    labels = labels or _LABELS
    data, figures, _ = _dirs(run_dir)
    t = pd.read_csv(data / f"{network}.csv")
    exact = data / f"{network}_exact.csv"
    ex = pd.read_csv(exact) if exact.exists() else None
    depths = np.sort(t["depth"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, key in zip(axes[:2], ("err_val", "err_grad")):
        a = t[t["method"] == "A"].sort_values("depth")
        if len(a) and a["overflow"].astype(bool).any():
            d0 = a.loc[a["overflow"].astype(bool), "depth"].min()
            ax.axvspan(d0 - 0.5, depths.max() + 0.5, color=MUTED, alpha=0.12, lw=0)
            ax.text(d0 + 0.3, 3 * FLOOR, labels["notes"]["overflow"], color=MUTED, fontsize=8)
        ax.axhline(1e-6, color=MUTED, lw=1, ls="--")
        ax.axhline(U, color=MUTED, lw=1, ls=":")
        col = "log10_chi_val_A" if key == "err_val" else "log10_chi_grad_A"
        if ex is not None and col in ex and ex[col].notna().any():
            with np.errstate(over="ignore"):         # chi far beyond float64 range is drawn off the axis
                pred = U * 10.0 ** ex[col].to_numpy(dtype=float)
            ok = pred < CEIL
            ax.semilogy(ex["depth"][ok], pred[ok], color=COLORS["A"], lw=1, ls="--", alpha=0.7,
                        label=labels["notes"]["pred"])
        for m in METHODS:
            g = t[(t["method"] == m)].sort_values("depth")
            e = g[key].to_numpy(dtype=float)
            fin = np.isfinite(e)
            if fin.any():
                ax.semilogy(g["depth"][fin], _clip(e[fin]), marker=MARKERS[m], color=COLORS[m], lw=1.8, ms=4,
                            label=labels["methods"][m])
        ax.set_ylim(FLOOR / 3, CEIL * 3)
        ax.set_title(labels["panels"][key], fontsize=11, color=INK)
        ax.set_xlabel(labels["depth_axis"][network])
        _style(ax)
    axes[0].set_ylabel(labels["axes"]["err"])
    axes[0].legend(fontsize=8, loc="center right")
    ax = axes[2]
    for m in METHODS:
        g = t[t["method"] == m].sort_values("depth")
        if g["mem_total"].notna().any():
            ax.semilogy(g["depth"], g["mem_total"], marker=MARKERS[m], color=COLORS[m], lw=1.8, ms=4,
                        label=labels["methods"][m])
    ax.set_title(labels["panels"]["mem"], fontsize=11, color=INK)
    ax.set_xlabel(labels["depth_axis"][network])
    ax.set_ylabel(labels["axes"]["mem"])
    _style(ax)
    fig.suptitle(labels["titles"][network], fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(figures / f"{NB_ID}_{network}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_mha_chunks(run_dir, labels=None, theta_big=10.0, save=True):
    """Magnitude of the B pair after every chunk of the linear-attention models, one line per depth."""
    labels = labels or _LABELS
    data, figures, _ = _dirs(run_dir)
    t = pd.read_csv(data / "mha_linear_chunks.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    depths = sorted(t["depth"].unique())
    cmap = plt.get_cmap("viridis", max(2, len(depths)))
    for i, d in enumerate(depths):
        g = t[t["depth"] == d].sort_values("idx")
        ax.semilogy(g["idx"], g["maxabs"], color=cmap(i), lw=1.2, label=f"N = {d}")
    ax.axhline(theta_big, color=MUTED, ls="--", lw=1)
    ax.text(0, theta_big * 1.15, labels["notes"]["theta"], color=MUTED, fontsize=8)
    ax.set_xlabel(labels["axes"]["chunk"])
    ax.set_ylabel(labels["axes"]["maxabs"])
    ax.set_title(labels["panels"]["chunks"], fontsize=11, color=INK)
    ax.legend(fontsize=8, ncol=4)
    _style(ax)
    fig.tight_layout()
    if save:
        fig.savefig(figures / f"{NB_ID}_mha_linear_chunks.png", dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------------------------- VGG16

def vgg_table(run_dir, labels=None, save=True):
    """Median over images of the final-logit metrics, memory and time per method."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "vgg_final.csv")
    agg = t.groupby("method").agg(err_val=("err_val", "median"), err_grad=("err_grad", "median"),
                                  chi_val=("chi_val", "median"), chi_grad=("chi_grad", "median"),
                                  mem_weights=("mem_weights", "median"), mem_cache=("mem_cache", "median"),
                                  mem_autograd=("mem_autograd", "median"), mem_total=("mem_total", "median"),
                                  time_s=("time_s", "median"), image=("image", "count")).reset_index()
    out = agg.copy()
    for c in ("err_val", "err_grad"):
        out[c] = out[c].map(_fmt_err)
    for c in ("chi_val", "chi_grad"):
        out[c] = out[c].map(lambda v: f"{v:.2e}")
    for c in ("mem_weights", "mem_cache", "mem_autograd", "mem_total"):
        out[c] = out[c].map(fmt_bytes)
    out["time_s"] = out["time_s"].map(lambda v: f"{v:.2f}")
    out = _relabel(out, labels)
    if save:
        out.to_csv(tables / f"{NB_ID}_vgg16.csv", index=False)
    return out


def plot_vgg(run_dir, labels=None, save=True):
    """Per-layer chi and invariant error of A and B (median and range over images), memory by kind, time."""
    labels = labels or _LABELS
    data, figures, _ = _dirs(run_dir)
    L = pd.read_csv(data / "vgg_layers.csv")
    F = pd.read_csv(data / "vgg_final.csv")
    names = L[L["method"] == "A"].drop_duplicates("state").sort_values("state")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.4))
    for ax, key, title in ((axes[0, 0], "chi", "chi"), (axes[0, 1], "err", "layer_err")):
        for m in ("A", "B"):
            g = L[L["method"] == m].groupby("state")[key]
            med = g.median()
            y = np.log10(med.clip(lower=1e-300)) if key == "chi" else _clip(med)
            if key == "chi":
                ax.plot(med.index, y, marker=MARKERS[m], color=COLORS[m], lw=1.6, ms=3, label=labels["methods"][m])
                ax.fill_between(med.index, np.log10(g.min().clip(lower=1e-300)), np.log10(g.max().clip(lower=1e-300)),
                                color=COLORS[m], alpha=0.15, lw=0)
            else:
                ax.semilogy(med.index, y, marker=MARKERS[m], color=COLORS[m], lw=1.6, ms=3, label=labels["methods"][m])
        if key == "chi":
            ax.axhline(np.log10(1 / U), color=MUTED, ls=":", lw=1)
            ax.text(names["state"].min(), np.log10(1 / U) + 0.3, "$1/u$", color=MUTED, fontsize=8)
        else:
            gp = L[L["method"] == "B"].groupby("state")["err_pre"].median()
            ax.semilogy(gp.index, _clip(gp), color=COLORS["B"], ls="--", lw=1, label=labels["notes"]["pre"])
            ax.set_ylim(FLOOR / 3, CEIL * 3)
        ticks = names[names["name"].str.contains("features_|classifier_", regex=True)
                      & names["kind"].isin(["conv", "linear"])]
        ax.set_xticks(ticks["state"])
        ax.set_xticklabels(ticks["name"].str.replace("features_", "f").str.replace("classifier_", "c"),
                           rotation=90, fontsize=7)
        ax.set_title(labels["panels"][title], fontsize=11, color=INK)
        ax.set_ylabel(labels["axes"]["chi"] if key == "chi" else labels["axes"]["err"])
        ax.legend(fontsize=8)
        _style(ax)
    ax = axes[1, 0]
    kinds = ["mem_weights", "mem_cache", "mem_autograd"]
    med = F.groupby("method")[kinds].median().reindex(METHODS)
    bottom = np.zeros(len(med))
    shades = {"mem_weights": 0.95, "mem_cache": 0.6, "mem_autograd": 0.3}
    for k in kinds:
        vals = med[k].to_numpy(dtype=float) / 2 ** 30
        ax.bar([labels["methods"][m] for m in med.index], vals, bottom=bottom, color=[COLORS[m] for m in med.index],
               alpha=shades[k], edgecolor="white")
        bottom += np.nan_to_num(vals)
    ax.set_ylabel("GiB")
    ax.set_title(labels["panels"]["mem"], fontsize=11, color=INK)
    ax.legend(handles=[Patch(facecolor=MUTED, alpha=shades[k], label=labels["notes"]["mem_kinds"][k]) for k in kinds],
              fontsize=8)
    _style(ax)
    ax = axes[1, 1]
    tm = F.groupby("method")["time_s"].median().reindex(METHODS)
    ax.bar([labels["methods"][m] for m in tm.index], tm.to_numpy(dtype=float), color=[COLORS[m] for m in tm.index])
    ax.set_ylabel(labels["axes"]["time"])
    ax.set_title(labels["panels"]["time"], fontsize=11, color=INK)
    _style(ax)
    fig.suptitle(labels["titles"]["vgg16"], fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(figures / f"{NB_ID}_vgg16.png", dpi=150, bbox_inches="tight")
    return fig


def fidelity_table(run_dir, labels=None, save=True):
    """Worst relative difference to the official HM code per quantity, for B as the code runs it and for
    B with the true conv derivative."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "hm_fidelity.csv")
    agg = t.groupby("quantity").agg(rel_diff_official=("rel_diff_official", "max"),
                                    rel_diff_true=("rel_diff_true", "max"), passed=("passed", "all")).reset_index()
    first = t[t["rel_diff_true"] > 1e-10]
    if save:
        agg.to_csv(tables / f"{NB_ID}_hm_fidelity.csv", index=False)
    out = agg.copy()
    for c in ("rel_diff_official", "rel_diff_true"):
        out[c] = out[c].map(lambda v: f"{v:.1e}")
    return out.rename(columns=labels["columns"]), (first["node"].iloc[-1] if len(first) else None)


# ----------------------------------------------------------------------------------------- summaries

def summary_table(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "summary.csv")
    out = t.copy()
    for c in ("max_err_val", "max_err_grad"):
        out[c] = out[c].map(_fmt_worst)
    for c in ("certified_val", "certified_grad"):
        out[c] = out[c].map(lambda v: "—" if np.isnan(v) else f"{int(v)}")
    out["mem_at_max"] = out["mem_at_max"].map(fmt_bytes)
    out["time_at_max"] = out["time_at_max"].map(lambda v: f"{v:.3g}" if np.isfinite(v) else "—")
    out = _relabel(out, labels)
    if save:
        out.to_csv(tables / f"{NB_ID}_summary.csv", index=False)
    return out


def predictions_table(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "predictions.csv")
    out = _relabel(t, labels)
    if save:
        out.to_csv(tables / f"{NB_ID}_predictions.csv", index=False)
    return out


def plot_summary(run_dir, labels=None, save=True):
    """One panel per network: memory at the largest depth (x) against the largest value error (y)."""
    labels = labels or _LABELS
    data, figures, _ = _dirs(run_dir)
    t = pd.read_csv(data / "summary.csv")
    nets = [n for n in _LABELS["networks"] if n in set(t["network"])]
    fig, axes = plt.subplots(1, len(nets), figsize=(3.3 * len(nets), 3.6), squeeze=False)
    for ax, n in zip(axes[0], nets):
        g = t[(t["network"] == n) & np.isfinite(t["mem_at_max"])]     # skip methods not defined for this network
        unit, k = next(((u, k) for u, k in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10))
                        if g["mem_at_max"].max() >= k), ("B", 1))
        for _, r in g.iterrows():
            y = r["max_err_val"]
            y = CEIL if not np.isfinite(y) else min(max(y, FLOOR), CEIL)     # errors above CEIL sit on the top edge
            ax.semilogy(r["mem_at_max"] / k, y, MARKERS[r["method"]], color=COLORS[r["method"]], ms=9)
        ax.axhline(1e-6, color=MUTED, ls="--", lw=1)
        ax.set_ylim(FLOOR / 3, CEIL * 3)
        ax.set_xlim(0, 1.15 * g["mem_at_max"].max() / k)
        ax.set_title(labels["networks"][n], fontsize=10, color=INK)
        ax.set_xlabel(f"{labels['panels']['mem'].lower()} ({unit})")
        _style(ax)
    axes[0][0].set_ylabel(labels["axes"]["err"])
    fig.legend(handles=[Line2D([], [], ls="", marker=MARKERS[m], color=COLORS[m], ms=8, label=labels["methods"][m])
                        for m in METHODS], loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(labels["panels"]["summary"], fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    if save:
        fig.savefig(figures / f"{NB_ID}_summary.png", dpi=150, bbox_inches="tight")
    return fig


def redraw_all(run_dir, labels=None):
    """Every table and figure of the notebook from data/, with the given labels."""
    data, _, _ = _dirs(run_dir)
    out = {}
    for n in ("mlp", "gate", "mha_linear", "mha_softmax"):
        if (data / f"{n}.csv").exists():
            out[f"table_{n}"] = depth_table(run_dir, n, labels)
            plot_depth(run_dir, n, labels)
            plt.close("all")
    if (data / "mha_linear_chunks.csv").exists():
        plot_mha_chunks(run_dir, labels)
        plt.close("all")
    if (data / "vgg_final.csv").exists():
        out["table_vgg16"] = vgg_table(run_dir, labels)
        plot_vgg(run_dir, labels)
        plt.close("all")
    if (data / "hm_fidelity.csv").exists():
        out["table_fidelity"] = fidelity_table(run_dir, labels)[0]
    if (data / "summary.csv").exists():
        out["summary"] = summary_table(run_dir, labels)
        plot_summary(run_dir, labels)
        plt.close("all")
    if (data / "predictions.csv").exists():
        out["predictions"] = predictions_table(run_dir, labels)
    return out
