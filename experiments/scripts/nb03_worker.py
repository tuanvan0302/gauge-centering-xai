"""Worker process of notebook nb03 §6: the whole pipeline of one test image at a time, on one GPU (or the CPU).

Started by the notebook as
    python scripts/nb03_worker.py RUN_DIR WORKER
with CUDA_VISIBLE_DEVICES set to one GPU (empty: CPU) and the OpenMP/MKL thread count set to its share of the CPUs.
Reads RUN_DIR/run_config.json (written by the notebook). Images are shared out dynamically: a worker takes the next
image k that has no result yet by creating RUN_DIR/workers/claims/image_KKK atomically, so a faster GPU takes more
images, and a rerun resumes where the last one stopped. For every image k the worker
  1. reads the input tensor and the mask that nb01 cached for image k, and the class nb01 explained, so both sides
     explain the same tensor and the same class;
  2. computes the raw maps of the six Table-1 rows with gaugexai.explain for the families hm (official code: only
     for the bridge to nb01 and for §11), hm_true and gauge, with alpha* and the ratio curves of the balanced rule;
  3. compares the hm maps with the maps nb01 saved (bridge), and the families with each other (§11);
  4. scores the variants (hm_true and gauge, HM's post-processing) with HM's QuantusEvaluator, one call per
     metric as nb01: Selectivity, Attribution Localisation, Pointing Game, Pixel Flipping;
  5. scores Max Sensitivity: every variant restarts np.random at seed 42·10⁴ + k, so all variants of the image see
     the same noisy inputs; the engine computes all rows and both families of a noisy input at once (cached).
Writes data/maps/image_KKK.npz (raw maps and ratio curves), data/per_image/image_KKK.json (meta, alpha*, bridge,
similarities, timing) and, last, data/scores/image_KKK.csv, whose existence marks the image as done.
The notebook follows the progress by counting data/scores/*.csv. stdout/stderr go to workers/wW.log, a failed image
to workers/errors/image_KKK.txt (the worker moves on; it stops after MAX_FAILURES failed images in a row).

The helpers available_cpus, visible_gpus, gpu_inventory, nb01_mask and attrloc_undefined import nothing heavy at module
level; the notebook uses them too.
"""
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

MAX_FAILURES = 3


def available_cpus():
    """CPUs this process may use: the cgroup quota of a container if there is one, else the affinity mask."""
    for path in ("/sys/fs/cgroup/cpu.max",):                                 # cgroup v2: "<quota> <period>"
        try:
            quota, period = Path(path).read_text().split()[:2]
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError):
            pass
    try:                                                                     # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def gpu_inventory():
    """[{index, name, memory_total_mib, memory_used_mib, utilization_pct}] from nvidia-smi ([] without a GPU)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        i, name, total, used, util = [s.strip() for s in line.split(",")]
        gpus.append({"index": int(i), "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
                     "utilization_pct": int(util)})
    return gpus


def visible_gpus():
    return [g["index"] for g in gpu_inventory()]


def attrloc_undefined(a, s):
    """True when Quantus 0.6 AttributionLocalisation returns NaN by design: the attribution inside the mask exceeds
    the total (inside / total > 1), which only a signed map can do. `a` is a map as HM's evaluator gets it (channels
    averaged, then normalised by max |a| as Quantus does); `s` the mask (first channel if it has three)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    a = a.mean(0) if a.ndim == 3 else a
    a = a / max(float(np.abs(a).max()), 1e-30)
    s = np.asarray(s, dtype=np.float64)
    s = s[..., 0] if s.ndim == 3 else s
    return bool((a * s).sum() / (a.sum() + 1e-9) > 1)


def nb01_mask(m):
    """The segmentation mask nb01's evaluator used, from HM's cache file baseline_binary_mask.npz (array `m`):
    HM stacks the (224, 224, 3) mask three times and its evaluate_image_folder keeps the first slice."""
    if m.shape[0] == 3:
        return m[0].astype(bool)
    if m.shape[-1] == 3:
        return m[:, :, 0].astype(bool)
    return m.astype(bool)


def main():
    run_dir, w = Path(sys.argv[1]), int(sys.argv[2])
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, cfg["hm_repo"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    import torchvision.transforms as T
    logging.disable(logging.CRITICAL)             # HM logs every tensor at DEBUG level
    from xai.evaluation.dataset_utils import _load_dataset
    from xai.evaluation.quantus_evaluator import QuantusEvaluator
    import gaugexai as gx
    import quantus_batched

    cuda = torch.cuda.device_count() > 0          # is_available() stays True on Windows with no visible GPU
    device = torch.device("cuda" if cuda else "cpu")
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = cfg["tf32"]
    torch.set_num_threads(cfg["cpu_threads"])
    if cuda:
        quantus_batched.install(cfg["quantus_chunk"])
    log = logging.getLogger("nb03_worker")

    dataset, _ = _load_dataset(True, cfg["imagenet"], cfg["imagenet_s"], "ImageNetS50", -1, cfg["split_seed"],
                               False, log)
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   # HM's load_image
    model = torchvision.models.vgg16(weights="IMAGENET1K_V1").to(device).eval().float()
    net = gx.SplitNet(model)
    rows = cfg["rows"]                            # dicts: id, layer, kind, pool, hm_post
    row_of = {r["id"]: r for r in rows}
    variants = [tuple(v) for v in cfg["variants"]]  # (row id, family, post)
    ms_families = tuple(dict.fromkeys(f for _, f, _ in variants))
    metric_cfgs = [json.loads(Path(p).read_text()) for p in cfg["metric_files"]]
    det_cfgs = [c for c in metric_cfgs if c["metric_class"] != "max_sensitivity"]
    ms_cfgs = [c for c in metric_cfgs if c["metric_class"] == "max_sensitivity"]
    ts = np.asarray(cfg["t_grid"], dtype=float)

    data = Path(cfg["data_dir"])
    dirs = {name: data / name for name in ("maps", "per_image", "scores")}
    claims, errors = run_dir / "workers" / "claims", run_dir / "workers" / "errors"
    for d in [*dirs.values(), claims, errors]:
        d.mkdir(parents=True, exist_ok=True)
    n_images = cfg["n_images"]
    gc.collect()
    gc.freeze()                                   # HM calls gc.collect() after every map: skip the long-lived objects

    def claim(k):
        try:
            os.close(os.open(claims / f"image_{k:03d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def process(k):
        img = cfg["images"][k]                    # test position k: dataset index, image name, nb01 files
        tim, t0 = {}, time.perf_counter()
        raw, mask_ds, class_id, _, image_name = dataset[img["dataset_index"]]
        assert image_name == img["image_name"], (image_name, img["image_name"])
        x_ours = tf(raw).unsqueeze(0)
        if img.get("nb01_input"):                 # the tensor and the mask nb01 evaluated (HM's cache files)
            x = torch.from_numpy(np.load(img["nb01_input"])["explanation"]).unsqueeze(0).float()
            mask = nb01_mask(np.load(img["nb01_mask"])["explanation"])
        else:
            x = x_ours
            mask = np.asarray(mask_ds, dtype=bool)
            mask = mask[..., 0] if mask.ndim == 3 else mask
        input_diff = float((x - x_ours).abs().max())
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
        own_pred = int(logits.argmax())
        target = own_pred if img.get("target") is None else int(img["target"])
        confidence = float(torch.softmax(logits[0].double(), 0)[target])
        sync()
        tim["load"] = time.perf_counter() - t0

        # 1. maps of hm (bridge, §11), hm_true and gauge; alpha* and ratio curves
        t0 = time.perf_counter()
        ex = gx.explain(net, x, rows, families=tuple(cfg["map_families"]), ts=ts, target=target)
        sync()
        tim["maps"] = time.perf_counter() - t0
        raw_maps = {f"{rid}|{fam}": m.float().cpu().numpy() for (rid, fam), m in ex.maps.items()}
        curves = {f"ratio|{d['pool']}|{d['derivative']}|{d['layer']}": np.asarray(d["ratio"]) for d in ex.diag}
        np.savez_compressed(dirs["maps"] / f"image_{k:03d}.npz", **raw_maps, **curves)
        alpha = [dict(pool=d["pool"], derivative=d["derivative"], layer=d["layer"], t_star=d["t_star"],
                      alpha_star=d["alpha_star"], balanced=d["balanced"], ratio_star=d["ratio_star"]) for d in ex.diag]

        # 2. bridge to nb01 and similarity of the families
        t0 = time.perf_counter()
        bridge = []
        for r in rows:
            f = img.get("nb01_maps", {}).get(r["id"])
            if not f or (r["id"], "hm") not in ex.maps:
                continue
            ours, theirs = gx.map224(ex.maps[(r["id"], "hm")], r["hm_post"]), np.load(f)["explanation"]
            theirs = theirs.mean(0) if theirs.ndim == 3 else theirs
            bridge.append(dict(row=r["id"], kind=r["kind"], rel_diff=gx.rel_diff(ours, theirs),
                               spearman=gx.spearman(ours, theirs),
                               off_pixels=float((np.abs(ours - theirs) > 0.01 * np.abs(theirs).max()).mean())))
        sim = []
        for r in rows:
            get = lambda fam: (np.abs(raw_maps[f"{r['id']}|{fam}"]) if r["hm_post"] == "abs"
                               else raw_maps[f"{r['id']}|{fam}"])
            for pair in cfg["pairs"]:
                a, b = pair.split("~")
                if f"{r['id']}|{a}" in raw_maps and f"{r['id']}|{b}" in raw_maps:
                    A, B = get(a), get(b)
                    sim.append(dict(row=r["id"], pair=pair, spearman=gx.spearman(A, B), ssim=gx.ssim(A, B),
                                    top10=gx.top_overlap(A, B)))
        tim["compare"] = time.perf_counter() - t0

        # 3. Selectivity, Attribution Localisation, Pointing Game, Pixel Flipping: all variants in one call per metric
        t0 = time.perf_counter()
        maps = {f"{rid}|{fam}|{post}": gx.map224(ex.maps[(rid, fam)], post) for rid, fam, post in variants}
        results = {name: {} for name in maps}
        for mc in det_cfgs:
            ev = QuantusEvaluator(model=model, device=str(device), metric_configs=[dict(mc)], softmax=False)
            res = ev.evaluate_single_image(image=x, explanations=maps, target_class=target, segmentation_mask=mask,
                                           explain_func=None, logger=log)
            for name, vals in res.items():
                results[name].update(vals)
            del ev
        sync()
        tim["metrics4"] = time.perf_counter() - t0

        # 4. Max Sensitivity: one engine call per noisy input for every row and family
        t0 = time.perf_counter()
        store, n_engine = {}, [0]

        def explain_for(rid, fam, post):
            def explain_func(model, inputs, targets, **kwargs):
                batch = []
                for i in range(inputs.shape[0]):
                    xi = np.ascontiguousarray(inputs[i:i + 1], dtype=np.float32)
                    tgt = int(np.asarray(targets).reshape(-1)[i]) if targets is not None else target
                    key = (hashlib.sha1(xi.tobytes()).hexdigest(), tgt)
                    if key not in store:
                        r = gx.explain(net, torch.from_numpy(xi), rows, families=ms_families, ts=ts, target=tgt)
                        store[key] = r.maps
                        n_engine[0] += 1
                    batch.append(gx.map224(store[key][(rid, fam)], post))
                return np.stack(batch)
            return explain_func

        for mc in ms_cfgs:
            ev = QuantusEvaluator(model=model, device=str(device), metric_configs=[dict(mc)], softmax=False)
            for rid, fam, post in variants:
                name = f"{rid}|{fam}|{post}"
                np.random.seed(cfg["split_seed"] * 10_000 + k)
                res = ev.evaluate_single_image(image=x, explanations={name: maps[name]}, target_class=target,
                                               segmentation_mask=mask, explain_func={name: explain_for(rid, fam, post)},
                                               logger=log)
                results[name].update(res[name])
            del ev
        store.clear()
        sync()
        tim["maxsens"] = time.perf_counter() - t0
        tim["total"] = sum(tim.values())

        meta = dict(image=k, dataset_index=img["dataset_index"], image_name=image_name, class_id=class_id,
                    target=target, own_pred=own_pred, nb01_pred=img.get("target"), confidence=confidence,
                    input_max_abs_diff=input_diff, worker=w, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    engine_calls_maxsens=n_engine[0],
                    peak_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None)
        (dirs["per_image"] / f"image_{k:03d}.json").write_text(json.dumps(
            dict(meta=meta, alpha=alpha, bridge=bridge, similarity=sim, timing=tim), indent=1), encoding="utf-8")
        recs = [dict(image=k, image_name=image_name, row=rid, family=fam, post=post, **results[f"{rid}|{fam}|{post}"])
                for rid, fam, post in variants]
        out = dirs["scores"] / f"image_{k:03d}.csv"
        tmp = out.with_suffix(".tmp")
        pd.DataFrame(recs).to_csv(tmp, index=False)
        tmp.replace(out)                          # written last: marks image k as done

    failures = 0
    for k in range(n_images):
        if (dirs["scores"] / f"image_{k:03d}.csv").exists() or not claim(k):
            continue
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            process(k)
            failures = 0
        except Exception:                         # keep the GPU busy with the next image; the notebook reports it
            (errors / f"image_{k:03d}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"image {k} failed:\n{traceback.format_exc()}", flush=True)
            failures += 1
            if failures >= MAX_FAILURES:
                raise RuntimeError(f"worker {w}: {MAX_FAILURES} images in a row failed, see {errors}")
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
