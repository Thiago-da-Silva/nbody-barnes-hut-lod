"""Instrumentacao: medias moveis para o HUD e gravacao de metricas por quadro em CSV."""
import csv
import os
import time

import psutil


class Profiler:
    FIELDS = ["frame", "wall_s", "scenario", "n", "method", "theta", "lod", "cull", "pix_thresh",
              "sort_ms", "build_ms", "force_ms", "integrate_ms", "lod_ms", "upload_ms", "gpu_ms",
              "frame_ms", "fps", "interactions", "nodes", "drawn_points", "aggregates", "culled",
              "rss_mb", "vram_mb", "energy_err", "momentum", "sim_time", "cam_dist"]

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.avg = {}
        self.proc = psutil.Process(os.getpid())
        self.rss_mb = self.proc.memory_info().rss / 2**20
        self._last_rss = 0.0
        self.recording = False
        self._file = None
        self._writer = None
        self.frame = 0
        self.t0 = time.perf_counter()

    def update(self, values):
        for k, v in values.items():
            if isinstance(v, (int, float)):
                prev = self.avg.get(k)
                self.avg[k] = v if prev is None else prev + self.alpha * (v - prev)
        now = time.perf_counter()
        if now - self._last_rss > 0.5:
            self.rss_mb = self.proc.memory_info().rss / 2**20
            self._last_rss = now
        self.frame += 1
        if self.recording:
            row = {k: values.get(k, "") for k in self.FIELDS}
            row["frame"] = self.frame
            row["wall_s"] = round(now - self.t0, 4)
            row["rss_mb"] = round(self.rss_mb, 2)
            self._writer.writerow(row)

    def toggle_recording(self, directory="results/logs"):
        if self.recording:
            self._file.close()
            self.recording = False
            return None
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, time.strftime("session_%Y%m%d_%H%M%S.csv"))
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self.recording = True
        return path

    def close(self):
        if self.recording:
            self._file.close()
            self.recording = False
