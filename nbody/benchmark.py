"""
Experimentos da Etapa 4 (execucao sem janela). Gera CSVs em results/data/.

  python benchmark.py physics   # tempo de octree/forca x N x metodo (theta)
  python benchmark.py leaf      # tamanho maximo da folha da octree
  python benchmark.py accuracy  # erro de forca x theta
  python benchmark.py energy    # conservacao de energia/momento ao longo do tempo
  python benchmark.py render    # LoD: FPS/GPU/qualidade de imagem x distancia
  python benchmark.py all       # todos (opcao --quick reduz tamanhos)
"""
import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import psutil

from src.forces import accel_brute, accel_direct_subset
from src.simulation import Simulation

OUT = os.path.join("results", "data")
THETAS = [0.3, 0.5, 0.7, 1.0]


def _writer(name, fields):
    os.makedirs(OUT, exist_ok=True)
    f = open(os.path.join(OUT, name), "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------- fisica
def bench_physics(sizes, brute_max, reps, scenarios):
    f, w = _writer("physics.csv", ["scenario", "n", "method", "theta", "rep", "build_ms",
                                   "force_ms", "step_ms", "interactions", "nodes",
                                   "tree_mb", "state_mb", "rss_mb"])
    for scen in scenarios:
        for n in sizes:
            sim = Simulation(scen, n, theta=0.7)
            for _ in range(3):  # aquece e sai do estado inicial
                sim.step()
            configs = [("bh", th) for th in THETAS]
            if n <= brute_max:
                configs.insert(0, ("brute", 0.0))
            for method, th in configs:
                sim.use_bh = method == "bh"
                sim.theta = th
                sim.compute_forces()  # aquecimento da configuracao
                r = max(1, reps // 3) if method == "brute" and n > 20000 else reps
                for rep in range(r):
                    t0 = time.perf_counter()
                    sim.compute_forces()
                    step = (time.perf_counter() - t0) * 1e3
                    w.writerow(dict(scenario=scen, n=n, method=method, theta=th, rep=rep,
                                    build_ms=sim.timings["build"], force_ms=sim.timings["force"],
                                    step_ms=step, interactions=sim.interactions,
                                    nodes=sim.tree.count, tree_mb=sim.tree.nbytes() / 2**20,
                                    state_mb=(sim.memory_bytes() - sim.tree.nbytes()) / 2**20,
                                    rss_mb=rss_mb()))
                f.flush()
                log(f"[physics] {scen:8s} N={n:7d} {method:5s} theta={th:.1f} "
                    f"build={sim.timings['build']:.2f}ms force={sim.timings['force']:.2f}ms")
            del sim
            gc.collect()
    f.close()


def bench_leaf(n, leaves, reps):
    f, w = _writer("leaf.csv", ["n", "leaf", "theta", "nodes", "build_ms", "force_ms",
                                "inter_per_particle", "tree_mb"])
    for leaf in leaves:
        sim = Simulation("plummer", n, theta=0.7, leaf_size=leaf)
        sim.compute_forces()
        b, fr = [], []
        for _ in range(reps):
            sim.compute_forces()
            b.append(sim.timings["build"])
            fr.append(sim.timings["force"])
        w.writerow(dict(n=n, leaf=leaf, theta=0.7, nodes=sim.tree.count,
                        build_ms=np.median(b), force_ms=np.median(fr),
                        inter_per_particle=sim.interactions / n, tree_mb=sim.tree.nbytes() / 2**20))
        log(f"[leaf] N={n} leaf={leaf} build={np.median(b):.2f} force={np.median(fr):.2f}")
    f.close()


# --------------------------------------------------------------------- precisao
def bench_accuracy(sizes, samples, scenarios):
    f, w = _writer("accuracy.csv", ["scenario", "n", "theta", "err_median", "err_mean",
                                    "err_p90", "err_p99", "err_max", "inter_per_particle"])
    rng = np.random.default_rng(1)
    for scen in scenarios:
        for n in sizes:
            sim = Simulation(scen, n, theta=0.5)
            idx = np.sort(rng.choice(n, size=min(samples, n), replace=False)).astype(np.int64)
            ref = np.empty((len(idx), 3))
            accel_direct_subset(sim.pos, sim.mass, idx, sim.eps, sim.G, ref)
            refn = np.linalg.norm(ref, axis=1)
            for th in [0.0, 0.1, 0.2] + THETAS + [1.2]:
                sim.theta = th
                sim.compute_forces()  # mesma ordem de particulas (posicoes inalteradas)
                err = np.linalg.norm(sim.acc[idx] - ref, axis=1) / refn
                w.writerow(dict(scenario=scen, n=n, theta=th, err_median=np.median(err),
                                err_mean=err.mean(), err_p90=np.percentile(err, 90),
                                err_p99=np.percentile(err, 99), err_max=err.max(),
                                inter_per_particle=sim.interactions / n))
                log(f"[accuracy] {scen} N={n} theta={th:.1f} median={np.median(err):.2e} "
                    f"p99={np.percentile(err, 99):.2e}")
            f.flush()
    f.close()


# ---------------------------------------------------------------------- energia
def bench_energy(n, t_end, dt, every):
    f, w = _writer("energy.csv", ["config", "integrator", "method", "theta", "dt", "step",
                                  "time", "E", "dE_rel", "P", "L_rel", "wall_ms"])
    configs = [("leapfrog", "brute", 0.0)] + [("leapfrog", "bh", th) for th in THETAS] + \
              [("euler", "bh", 0.5), ("euler", "brute", 0.0)]
    steps = int(round(t_end / dt))
    for integ, method, th in configs:
        sim = Simulation("plummer", n, seed=3, theta=th, dt=dt)
        sim.use_bh = method == "bh"
        sim.integrator = integ
        sim.compute_forces()
        E0 = sim.kinetic_energy() + sim.exact_potential_energy()
        L0 = np.linalg.norm(sim.angular_momentum()) + 1e-30
        name = f"{integ}-{method}" + (f"-{th:.1f}" if method == "bh" else "")
        wall = 0.0
        for st in range(steps + 1):
            if st % every == 0:
                E = sim.kinetic_energy() + sim.exact_potential_energy()
                w.writerow(dict(config=name, integrator=integ, method=method, theta=th, dt=dt,
                                step=st, time=sim.time, E=E, dE_rel=(E - E0) / abs(E0),
                                P=np.linalg.norm(sim.momentum()),
                                L_rel=np.linalg.norm(sim.angular_momentum() - sim.L0) / L0,
                                wall_ms=wall))
            if st < steps:
                t0 = time.perf_counter()
                sim.step()
                wall += (time.perf_counter() - t0) * 1e3
        f.flush()
        log(f"[energy] {name:22s} dE/E0 final = {(E - E0) / abs(E0):+.3e}  "
            f"tempo medio/passo = {wall / steps:.2f} ms")
    f.close()


# --------------------------------------------------------------------- render
def bench_render(sizes, distances, frames, width, height, thresholds):
    import moderngl
    from src.camera import Camera
    from src.lod import LodSelector
    from src.renderer import Renderer

    ctx = moderngl.create_standalone_context(require=330)
    tex = ctx.texture((width, height), 3)
    fbo = ctx.framebuffer([tex])
    rend = Renderer(ctx)
    lod = LodSelector()
    f, w = _writer("render.csv", ["n", "distance", "mode", "pix_thresh", "lod_ms", "upload_ms",
                                  "gpu_ms", "cpu_render_ms", "frame_ms", "drawn", "aggregates",
                                  "culled", "rmse", "psnr", "flux_ratio"])
    os.makedirs(os.path.join("results", "images"), exist_ok=True)

    def read():
        return np.frombuffer(fbo.read(components=3), np.uint8).reshape(height, width, 3)[::-1]

    for n in sizes:
        sim = Simulation("disk", n, theta=0.7)
        for _ in range(2):
            sim.step()
        for dist in distances:
            cam = Camera(distance=dist)
            vp = cam.projection(width / height) @ cam.view()
            ps = cam.proj_scale(height)
            modes = [("all", 0.0), ("cull", 0.0)] + [("lod", t) for t in thresholds]
            ref_img = None
            for mode, thr in modes:
                rec = {k: [] for k in ("lod_ms", "upload_ms", "gpu_ms", "cpu_render_ms", "frame_ms")}
                for fr in range(frames + 3):
                    t_start = time.perf_counter()
                    ni = na = nc = 0
                    if mode == "all":
                        ni = lod.pack_all(sim)
                    else:
                        ni, na, nc = lod.select(sim, cam.eye(), vp, ps, thr, mode == "lod", True)
                    count = ni + na
                    lms = (time.perf_counter() - t_start) * 1e3
                    t0 = time.perf_counter()
                    rend.upload_vertices(lod.vert[:count])
                    up = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    gpu = rend.render(fbo, (width, height), vp, ps, count)
                    cpu_r = (time.perf_counter() - t0) * 1e3
                    if fr >= 3:
                        rec["lod_ms"].append(lms)
                        rec["upload_ms"].append(up * 1e3)
                        rec["gpu_ms"].append(gpu)
                        rec["cpu_render_ms"].append(cpu_r)
                        rec["frame_ms"].append((time.perf_counter() - t_start) * 1e3)
                img = read().astype(np.float64)
                if ref_img is None:
                    ref_img = img
                mse = np.mean((img - ref_img) ** 2)
                psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
                flux = img.sum() / max(ref_img.sum(), 1.0)
                if n == sizes[-1] or n == 50000:
                    from PIL import Image
                    Image.fromarray(img.astype(np.uint8)).save(
                        os.path.join("results", "images", f"render_n{n}_d{dist:g}_{mode}{thr:g}.png"))
                drawn = ni + na
                w.writerow(dict(n=n, distance=dist, mode=mode, pix_thresh=thr,
                                **{k: float(np.median(v)) for k, v in rec.items()},
                                drawn=drawn, aggregates=na, culled=nc, rmse=np.sqrt(mse),
                                psnr=psnr, flux_ratio=flux))
                log(f"[render] N={n:8d} d={dist:6.1f} {mode:4s} thr={thr:3.1f} "
                    f"drawn={drawn:8d} gpu={np.median(rec['gpu_ms']):6.3f}ms "
                    f"lod={np.median(rec['lod_ms']):6.3f}ms frame={np.median(rec['frame_ms']):6.2f}ms "
                    f"psnr={psnr:5.1f}dB")
            f.flush()
        del sim
        gc.collect()
    f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["physics", "leaf", "accuracy", "energy", "render", "all"])
    p.add_argument("--quick", action="store_true", help="versao reduzida para teste rapido")
    a = p.parse_args()
    q = a.quick
    t0 = time.time()
    if a.what in ("physics", "all"):
        bench_physics(sizes=[1000, 5000, 10000] if q else [1000, 5000, 10000, 20000, 50000, 100000],
                      brute_max=10000 if q else 50000, reps=2 if q else 7,
                      scenarios=["plummer"] if q else ["plummer", "disk"])
    if a.what in ("leaf", "all"):
        bench_leaf(n=10000 if q else 50000, leaves=[1, 2, 4, 8, 16, 32, 64], reps=3 if q else 7)
    if a.what in ("accuracy", "all"):
        bench_accuracy(sizes=[5000] if q else [10000, 50000], samples=1000 if q else 4000,
                       scenarios=["plummer", "disk"])
    if a.what in ("energy", "all"):
        bench_energy(n=1000 if q else 4000, t_end=2.0 if q else 20.0, dt=0.01,
                     every=10)
    if a.what in ("render", "all"):
        bench_render(sizes=[10000] if q else [10000, 50000, 200000, 1000000],
                     distances=[5, 20] if q else [3, 8, 15, 30, 60, 120, 250],
                     frames=5 if q else 30, width=1920, height=1080,
                     thresholds=[2.0] if q else [1.0, 2.0, 4.0])
    log(f"concluido em {time.time() - t0:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
