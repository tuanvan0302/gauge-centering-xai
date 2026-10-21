"""Run a notebook of this folder headless and show its progress in the terminal.

Naming convention: notebook nbNN_<slug>.ipynb writes to results/nbNN/n{N_IMAGES}/ (results/nbNN/ if it has
no N_IMAGES), reads configs/nbNN/ and reference/nbNN/, and prefixes its figures and tables with nbNN_.

Usage, from gauge-centering-xai/experiments:
    uv run python scripts/run_notebook.py nb01                                     # settings as in the notebook
    uv run python scripts/run_notebook.py nb01 --n-images 566 --gpu 1 --workers 3  # override for this run only

--n-images rewrites the notebook's "N_IMAGES = ..." line in memory; --gpu and --workers are passed to the kernel
as NB_GPU and NB_WORKERS (nb01 starts that many worker processes on that GPU for its heavy phases).
The outer bar counts code cells and names the current section; the inner bar follows the long phases through
RUN_DIR/progress.json, written by the notebook (scripts/nb_progress.py). Each cell's printed lines are echoed.
The notebook file itself is not changed; the executed copy with all outputs is saved as
RUN_DIR/nbNN_<slug>.executed.ipynb.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from tqdm import tqdm

EXP = Path(__file__).resolve().parents[1]
N_IMAGES_LINE = re.compile(r"^N_IMAGES = (\d+)", re.M)


class PhaseBar(threading.Thread):
    """Polls progress.json and draws it as the inner bar."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop, self.bar, self.phase = path, threading.Event(), None, None

    def run(self):
        while not self.stop.wait(1.0):
            try:
                p = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            # a new phase (or the same phase restarted) gets a fresh bar
            if self.bar is None or p["phase"] != self.phase or p["done"] < self.bar.n:
                if self.bar is not None:
                    self.bar.close()
                self.phase = p["phase"]
                self.bar = tqdm(total=p["total"], desc=p["phase"], unit=p["unit"], position=1, leave=False,
                                dynamic_ncols=True)
            self.bar.update(p["done"] - self.bar.n)

    def close(self):
        self.stop.set()
        self.join(timeout=2)
        if self.bar is not None:
            self.bar.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nb_id", help="notebook code, e.g. nb01")
    ap.add_argument("--n-images", type=int, default=None, help="override N_IMAGES (nb01: 1 = smoke test, 566 = full run)")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index for the kernel and its workers (NB_GPU)")
    ap.add_argument("--workers", type=int, default=None, help="number of worker processes (NB_WORKERS)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):   # Vietnamese text also when the output is piped to a log file
        stream.reconfigure(encoding="utf-8")
    if sys.platform == "win32":   # zmq needs a selector event loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if args.gpu is not None:      # inherited by the kernel process
        os.environ["NB_GPU"] = str(args.gpu)
    if args.workers is not None:
        os.environ["NB_WORKERS"] = str(args.workers)

    found = sorted(EXP.glob(f"{args.nb_id}_*.ipynb"))
    if len(found) != 1:
        sys.exit(f"cần đúng một file {args.nb_id}_*.ipynb trong {EXP}, tìm thấy {[p.name for p in found]}")
    notebook = found[0]
    nb = nbf.read(notebook, as_version=4)
    cfg = next((c for c in nb.cells if c.cell_type == "code" and N_IMAGES_LINE.search(c.source)), None)
    if args.n_images is not None:
        if cfg is None:
            sys.exit(f"{notebook.name} không có dòng N_IMAGES = ..., không đổi được --n-images")
        cfg.source = N_IMAGES_LINE.sub(f"N_IMAGES = {args.n_images}", cfg.source, count=1)
    n_images = int(N_IMAGES_LINE.search(cfg.source).group(1)) if cfg is not None else None
    run_dir = EXP / "results" / args.nb_id / (f"n{n_images}" if n_images is not None else "")
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_file = run_dir / "progress.json"
    progress_file.unlink(missing_ok=True)
    out = run_dir / f"{notebook.stem}.executed.ipynb"

    # section title of every cell: the last "## ..." heading above it
    titles, title = {}, "mở đầu"
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown":
            m = re.search(r"^##\s+(.+)$", c.source, re.M)
            title = m.group(1).strip() if m else title
        titles[i] = title
    n_code = sum(c.cell_type == "code" for c in nb.cells)

    print(f"notebook : {notebook.name}\nN_IMAGES : {n_images}\nGPU      : {os.environ.get('NB_GPU', 'như notebook')}"
          f"\nworker   : {os.environ.get('NB_WORKERS', 'như notebook')}\nkết quả  : {run_dir}\n")
    cells_bar = tqdm(total=n_code, desc="cell", unit="cell", position=0, dynamic_ncols=True)
    phase_bar = PhaseBar(progress_file)
    phase_bar.start()

    def on_start(cell, cell_index):
        kind = "hiển thị" if cell.source.startswith("# ▶") else "tính"
        cells_bar.set_description(f"§ {titles[cell_index][:40]} ({kind})")

    def on_executed(cell, cell_index, execute_reply):
        # called once the reply and all outputs of the cell have arrived
        cells_bar.update(1)
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream" and o.get("name") == "stdout":
                for line in o.get("text", "").rstrip().splitlines():
                    tqdm.write(f"  {line}")

    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(EXP)}},
                            on_cell_start=on_start, on_cell_executed=on_executed)
    t0, ok = time.time(), False
    try:
        client.execute()
        ok = True
    except CellExecutionError as e:
        tqdm.write(f"\nLỖI trong cell:\n{str(e)[-3000:]}")
    except KeyboardInterrupt:
        tqdm.write("\nĐã dừng (Ctrl+C). Kết quả đã tính vẫn nằm trong cache; chạy lại sẽ tiếp tục từ đó.")
    finally:
        phase_bar.close()
        cells_bar.close()
        nbf.write(nb, out)
    print(f"\n{'XONG' if ok else 'CHƯA XONG'} sau {(time.time() - t0) / 60:.1f} phút. Notebook đã chạy: {out}")
    verdict = run_dir / f"{args.nb_id}_verdict.json"
    if ok and verdict.exists():
        print("Kết luận:", json.loads(verdict.read_text(encoding="utf-8"))["label"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
