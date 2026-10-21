"""Worker process of notebook nb01: runs one heavy phase for its share of the images.

Started by the notebook (§6 maps, §7 metrics4, §8 maxsens) as
    python scripts/nb01_worker.py RUN_DIR PHASE WORKER N_WORKERS
with CUDA_VISIBLE_DEVICES set to the chosen GPU. Reads RUN_DIR/run_config.json (written by the notebook),
handles images k with k % N_WORKERS == WORKER, one HM call per image (HM's own split_id/num_splits option)
with seed = seed·10⁴ + k. Pixel Flipping and Selectivity go through scripts/quantus_batched.py (quantus_chunk
perturbation steps per forward pass); tf32 in run_config.json sets PyTorch's TF32 flags. Writes into RUN_DIR/workers/:
    progress_{PHASE}_w{WORKER}.json   progress, read by the notebook's bar
    timing_{PHASE}_w{WORKER}.json     seconds, images, peak GPU memory, whether anything new was computed
Its stdout/stderr (HM's own prints) go to RUN_DIR/workers/{PHASE}_w{WORKER}.log.
"""
import gc
import json
import logging
import os
import shutil
import sys
import time
import warnings
from pathlib import Path


def utf8_file_handlers():
    # HM opens its log files with the locale encoding; on Windows its emoji log lines then fail
    init = logging.FileHandler.__init__

    def patched(self, filename, mode="a", encoding=None, *args, **kwargs):
        init(self, filename, mode, encoding or "utf-8", *args, **kwargs)
    logging.FileHandler.__init__ = patched


def filled(base):
    # number of non-empty metric cells in the per-image results; a cached pass adds none
    import pandas as pd
    return sum(int(pd.read_csv(p).notna().sum().sum()) for p in Path(base).glob("image_*/results.csv"))


def main():
    run_dir, phase, w, n_w = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, cfg["hm_repo"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    warnings.filterwarnings("ignore")
    utf8_file_handlers()

    import torch
    from xai.evaluation import MethodFirstEvaluator
    from xai.evaluation.model_utils import _load_model
    from xai.evaluation.dataset_utils import _load_dataset
    from xai.evaluation.unified_utils import _set_random_seeds
    from nb_progress import Progress
    import quantus_batched
    for h in logging.getLogger().handlers:
        h.setLevel(logging.ERROR)
    log = logging.getLogger("hm_repro")
    # on Ampere and newer, PyTorch runs float32 convolutions as TF32 unless told otherwise
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = cfg.get("tf32", True)
    quantus_batched.install(cfg.get("quantus_chunk", 1))

    n_images, main_dir = cfg["n_images"], Path(cfg["main_dir"])
    images = [k for k in range(n_images) if k % n_w == w]
    dataset, test_idx = _load_dataset(True, cfg["imagenet"], cfg["imagenet_s"], "ImageNetS50", -1,
                                      cfg["split_seed"], False, log)
    test_idx = [int(i) for i in test_idx]

    seeds = cfg["maxsens_seeds"] if phase == "maxsens" else [cfg["split_seed"]]
    metrics_dir = cfg["metrics4_dir"] if phase == "metrics4" else cfg["maxsens_dir"]
    units = sum(len(files) for _, files in cfg["groups"].values()) if phase == "maps" \
        else len(seeds) * len(cfg["groups"])
    bar = Progress(run_dir / "workers", f"{phase} w{w}", len(images) * units, name=f"progress_{phase}_w{w}.json",
                   disable=True)
    timing = []
    for seed in seeds:
        base = main_dir if seed == cfg["split_seed"] else run_dir / f"maxsens_seed{seed}"
        for group, (dtype, _) in cfg["groups"].items():
            if base != main_dir:   # extra Max Sens seed: link this worker's maps into its own folder
                for src in (main_dir / group).glob("image_*"):
                    if int(src.name.split("_")[1]) in images:
                        dst = base / group / src.name
                        dst.mkdir(parents=True, exist_ok=True)
                        for f in [*src.glob("*.npz"), src / "metadata.json"]:
                            if not (dst / f.name).exists():
                                try:
                                    os.link(f, dst / f.name)
                                except OSError:
                                    shutil.copy2(f, dst / f.name)
            model = _load_model("vgg16", torch.device("cuda"), dtype, log)
            ev = MethodFirstEvaluator(
                model_name="vgg16", dtype=dtype, num_images=-1, random_seed=seed, innerloop_n=cfg["innerloop_n"],
                output_dir=base / group, imagenet_data_dir=cfg["imagenet"], imagenet_s_data_dir=cfg["imagenet_s"],
                methods_config_dir=str(Path(cfg["methods_dir"]) / group), metrics_config_dir=metrics_dir)
            # hand over the model and the checked test list; each call below then handles image k only
            ev.model, ev.dataset, ev.stratified_indices, ev.num_images = model, dataset, test_idx, n_images
            methods = list(ev.methods_config) if phase == "maps" else [None]
            # HM calls gc.collect() after every explanation (twice per Max Sens sample); freezing the
            # long-lived objects (torch, model, dataset, ...) makes each call skip them: ~0.13 s -> ~0 s
            gc.collect()
            gc.freeze()
            for m in methods:
                if m is not None:
                    ev.methods_config = [m]
                before = len(list((base / group).glob("image_*/*.npz"))) if phase == "maps" else filled(base / group)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for k in images:
                    ev.split_id, ev.num_splits = k, n_images
                    _set_random_seeds(seed * 10_000 + k)
                    (ev.run_map_generation_phase if phase == "maps" else ev.run_evaluation_phase)()
                    bar.update()
                after = len(list((base / group).glob("image_*/*.npz"))) if phase == "maps" else filled(base / group)
                timing.append({"phase": phase if phase != "maxsens" else f"maxsens_seed{seed}",
                               "method": m["name"] if m is not None else group,
                               "seconds": time.perf_counter() - t0, "n_images": len(images),
                               "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "fresh": after > before})
            ev.logger.removeHandler(ev.main_file_handler)
            ev.main_file_handler.close()
            gc.unfreeze()
            del ev, model
            gc.collect()
            torch.cuda.empty_cache()
    bar.close()
    (run_dir / "workers" / f"timing_{phase}_w{w}.json").write_text(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
