"""Tables and figures of notebook nb01, built only from the files a run saved in RUN_DIR/data.

Nothing here computes a metric or touches a model, so titles, column names and row names can be
changed (pass a modified copy of default_labels()) and everything redrawn in seconds.
Data files use fixed English keys (row, metric, passed, ...); display names come from the labels.
Figures go to RUN_DIR/figures, Table-1 style tables to RUN_DIR/tables; their file names start with NB_ID.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NB_ID = "nb01"
METRIC_KEYS = ["selectivity", "attribution_localisation", "pointing_game", "pf_auc5", "pf_auc20", "max_sensitivity"]

_LABELS = {
    # values of any "metric" column, and the metric columns of Table-1 style tables
    "metrics": {
        "selectivity": "Select. ↓",
        "attribution_localisation": "Attr. Loc. ↑",
        "pointing_game": "Point. ↑",
        "pf_auc5": "PF AUC@5 ↑",
        "pf_auc20": "PF AUC@20 ↑",
        "max_sensitivity": "Max Sens. ↓",
    },
    # column key in data/*.csv -> display name
    "columns": {
        "param": "tham số", "value": "giá trị", "reason": "lý do",
        "check": "kiểm tra", "passed": "đạt", "detail": "chi tiết", "note": "ghi chú",
        "row": "hàng", "table1": "Bảng 1", "abs_paper": "abs (bài)", "group": "nhóm", "config_file": "file config",
        "formula": "công thức", "op": "op", "found": "tìm thấy", "source": "nguồn",
        "metric": "thước đo", "metric_name": "tên thước đo", "quantus_class": "lớp Quantus", "params": "tham số",
        "result_column": "cột kết quả HM", "direction": "hướng", "result_method": "tên phương pháp trong kết quả HM",
        "label": "phương pháp", "layer": "tầng", "paper": "HM", "mean": "ta", "delta": "Δ", "se": "SE",
        "s_seed": "s_seed", "tol": "ngưỡng", "z": "z", "n_values": "số ô có giá trị",
        "phase": "giai đoạn", "method": "phương pháp HM", "seconds": "giây", "per_image_s": "giây/ảnh",
        "cache": "cache", "quantity": "đại lượng", "image": "ảnh",
        "peak_gib": "đỉnh GPU (GiB)", "wall_s": "giây đồng hồ", "n_workers": "số worker", "fresh": "tính mới",
    },
    "method_column": "phương pháp",
    "rows": {},                      # row number -> display name; empty = "<label> <layer>" from row_to_method.csv
    "table_values": {"mean": "Ta (trung bình theo ảnh)", "paper": "Bảng 1 của HM", "delta": "Δ = ta − HM",
                     "se": "SE", "se_eff": "SE hiệu dụng"},
    "seed_table": "Max Sens: trung bình theo seed",
    "seed_column": "seed {seed}",
    "fig_example": {"title": "Ảnh test đầu tiên và mask", "image": "{image}\n{class_id}",
                    "mask": "mask (resize + crop như HM)"},
    "fig_maps": {"title": "Bản đồ giải thích, chuẩn hoá theo max|·|", "original": "ảnh gốc",
                 "column": "{row}. {name}"},
}


def default_labels():
    """A fresh, editable copy of the default labels."""
    return copy.deepcopy(_LABELS)


def _dirs(run_dir):
    run_dir = Path(run_dir)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    return run_dir / "data", run_dir / "figures", run_dir / "tables"


def show_csv(run_dir, name, labels=None, **read_kw):
    """data/<name> with metric keys and column keys replaced by their display names."""
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    t = pd.read_csv(data / name, **read_kw)
    if "metric" in t.columns:
        t["metric"] = t["metric"].map(lambda m: labels["metrics"].get(m, m))
    return t.rename(columns=labels["columns"])


def show_env(run_dir, labels=None):
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    env = json.loads((data / "env.json").read_text())
    return pd.Series(env, name=labels["columns"]["value"]).to_frame()


def row_names(run_dir, labels=None):
    labels = labels or _LABELS
    data, _, _ = _dirs(run_dir)
    ref = pd.read_csv(data / "row_to_method.csv")
    names = ref["label"] + " " + ref["layer"].astype(str)
    return pd.Series([labels["rows"].get(r, n) for r, n in zip(ref["row"], names)], index=ref["row"], name="name")


def table1(run_dir, value="mean", labels=None, save=True):
    """Table-1 layout (rows x metrics) of one column of data/summary_stats.csv: mean, paper, delta, se, ..."""
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    stats = pd.read_csv(data / "summary_stats.csv")
    t = stats.pivot(index="row", columns="metric", values=value).reindex(columns=METRIC_KEYS)
    t.columns = [labels["metrics"][c] for c in t.columns]
    t.insert(0, labels["method_column"], row_names(run_dir, labels))
    t.index.name = labels["columns"]["row"]
    if save:
        t.to_csv(tables / f"{NB_ID}_table1_{value}.csv")
    return t


def maxsens_seed_table(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, _, tables = _dirs(run_dir)
    t = pd.read_csv(data / "maxsens_seed_means.csv", index_col="row")
    t.columns = [labels["seed_column"].format(seed=c) for c in t.columns]
    t.insert(0, labels["method_column"], row_names(run_dir, labels).reindex(t.index))
    t.index.name = labels["columns"]["row"]
    if save:
        t.to_csv(tables / f"{NB_ID}_maxsens_seed_means.csv")
    return t


def plot_example_mask(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, figures, _ = _dirs(run_dir)
    d = np.load(data / "fig_example_mask.npz")
    lab = labels["fig_example"]
    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
    ax[0].imshow(d["image"])
    ax[0].set_title(lab["image"].format(image=str(d["image_name"]), class_id=str(d["class_id"])), fontsize=8)
    ax[1].imshow(d["image"])
    ax[1].imshow(np.ma.masked_where(~d["mask"], d["mask"]), alpha=0.5, cmap="autumn")
    ax[1].set_title(lab["mask"], fontsize=8)
    for a in ax:
        a.axis("off")
    fig.suptitle(lab["title"], fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(figures / f"{NB_ID}_example_mask.png", dpi=120)
    return fig


def plot_maps_grid(run_dir, labels=None, save=True):
    labels = labels or _LABELS
    data, figures, _ = _dirs(run_dir)
    d = np.load(data / "fig_maps_grid.npz")
    maps, originals, rows = d["maps"], d["originals"], d["rows"]
    names = row_names(run_dir, labels)
    lab = labels["fig_maps"]
    n_img, n_rows = maps.shape[:2]
    fig, axes = plt.subplots(n_img, n_rows + 1, figsize=(1.9 * (n_rows + 1), 2.0 * n_img + 0.4), squeeze=False)
    for i in range(n_img):
        axes[i, 0].imshow(originals[i])
        axes[i, 0].set_title(f"{lab['original']}\n{str(d['image_names'][i]).split('_val_')[-1][:8]}", fontsize=7)
        for j in range(n_rows):
            axes[i, j + 1].imshow(maps[i, j], cmap="bwr", vmin=-1, vmax=1)
            if i == 0:
                axes[i, j + 1].set_title(lab["column"].format(row=int(rows[j]), name=names[int(rows[j])]), fontsize=6)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(lab["title"], fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(figures / f"{NB_ID}_maps_grid.png", dpi=120)
    return fig
