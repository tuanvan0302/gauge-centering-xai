"""Progress bar for the long phases of a notebook in this folder.

Shows a tqdm bar in the notebook and mirrors its state to RUN_DIR/progress.json, which
scripts/run_notebook.py reads to draw the same progress in the terminal.
"""
import json
import os
import time
from pathlib import Path

from tqdm.auto import tqdm


class Progress:
    def __init__(self, run_dir, phase, total, unit="ảnh", name="progress.json", disable=False):
        self.path = Path(run_dir) / name          # workers write progress_{phase}_w{i}.json with disable=True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.phase, self.total, self.unit, self.done = phase, total, unit, 0
        self.bar = tqdm(total=total, desc=phase, unit=unit, disable=disable)
        self._write("running")

    def update(self, k=1):
        if k <= 0:
            return
        self.done += k
        self.bar.update(k)
        self._write("running")

    def close(self):
        self.bar.close()
        self._write("done")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _write(self, state):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"phase": self.phase, "done": self.done, "total": self.total, "unit": self.unit,
                                   "state": state, "time": time.time()}, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(tmp, self.path)
        except PermissionError:   # the runner is reading the file right now (Windows); the next update rewrites it
            pass
