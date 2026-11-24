"""
Gera as figuras (results/figures/*.pdf|png) e tabelas LaTeX (results/tables/*.tex)
a partir dos CSVs de results/data/ produzidos por benchmark.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = os.path.join("results", "data")
FIG = os.path.join("results", "figures")
TAB = os.path.join("results", "tables")

# paleta categorica validada (ordem fixa; nunca reciclada)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#2b2b2b"
MUTED = "#8a8a85"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e2",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})
COL1 = (3.45, 2.5)   # largura de uma coluna IEEE (polegadas)
COL2 = (7.1, 2.6)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print("figura:", name)


def method_label(m, th):
    return "Força bruta" if m == "brute" else f"BH θ={th:.1f}"


def load(name):
    p = os.path.join(DATA, name)
    return pd.read_csv(p) if os.path.exists(p) else None


# ------------------------------------------------------------------ fisica
def fig_physics(df):
    g = df.groupby(["scenario", "n", "method", "theta"], as_index=False).median(numeric_only=True)
    g["total_ms"] = g["build_ms"] + g["force_ms"]
    g.loc[g.method == "brute", "total_ms"] = g.loc[g.method == "brute", "force_ms"]
    fig, axes = plt.subplots(1, 2, figsize=COL2, sharey=True)
    for ax, scen in zip(axes, ["plummer", "disk"]):
        d = g[g.scenario == scen]
        keys = [("brute", 0.0)] + [("bh", t) for t in sorted(d[d.method == "bh"].theta.unique())]
        for i, (m, th) in enumerate(keys):
            s = d[(d.method == m) & (np.isclose(d.theta, th))].sort_values("n")
            if s.empty:
                continue
            ax.plot(s.n, s.total_ms, marker=MARKERS[i], color=C[i], label=method_label(m, th),
                    markeredgecolor="white", markeredgewidth=0.6)
        # referencias de inclinacao
        n_ref = np.array([1e3, 1e5])
        base = d[(d.method == "brute") & (d.n == d.n.min())].total_ms
        if len(base):
            ref = base.iloc[0] * (n_ref / d.n.min()) ** 2
            ax.plot(n_ref, ref, ls=":", color=MUTED, lw=1)
            ax.text(n_ref[1] * 0.9, ref[1], "∝ N²", color=MUTED, fontsize=7, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Número de partículas N")
        ax.set_title("Esfera de Plummer" if scen == "plummer" else "Galáxia de disco")
        ax.axhline(1000 / 30, color=MUTED, lw=0.8, ls="--")
        ax.text(d.n.min(), 1000 / 30 * 1.15, "30 FPS", color=MUTED, fontsize=6.5)
    axes[0].set_ylabel("Tempo por passo (ms)")
    axes[1].legend(loc="upper left", ncol=1)
    save(fig, "fig_tempo_forca")

    # speedup sobre a forca bruta
    fig, ax = plt.subplots(figsize=COL1)
    d = g[g.scenario == "plummer"]
    br = d[d.method == "brute"].set_index("n").total_ms
    for i, th in enumerate(sorted(d[d.method == "bh"].theta.unique())):
        s = d[(d.method == "bh") & np.isclose(d.theta, th)].set_index("n").total_ms
        sp = (br / s).dropna()
        ax.plot(sp.index, sp.values, marker=MARKERS[i + 1], color=C[i + 1], label=f"θ={th:.1f}",
                markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Número de partículas N")
    ax.set_ylabel("Speedup sobre força bruta (×)")
    ax.legend(loc="upper left")
    save(fig, "fig_speedup")
    return g


def table_physics(g):
    d = g[g.scenario == "plummer"]
    ns = sorted(d.n.unique())
    keys = [("brute", 0.0)] + [("bh", t) for t in sorted(d[d.method == "bh"].theta.unique())]
    lines = [r"\begin{tabular}{l" + "r" * len(ns) + "}", r"\toprule",
             "Método & " + " & ".join(f"{n // 1000}k" for n in ns) + r" \\", r"\midrule"]
    for m, th in keys:
        row = []
        for n in ns:
            s = d[(d.method == m) & np.isclose(d.theta, th) & (d.n == n)]
            row.append(f"{s.total_ms.iloc[0]:.1f}" if len(s) else "--")
        name = "Força bruta" if m == "brute" else rf"BH $\theta={th:.1f}$"
        lines.append(name + " & " + " & ".join(row) + r" \\")
    b = d[(d.method == "bh") & np.isclose(d.theta, 0.7)].sort_values("n")
    lines.append(r"\midrule")
    lines.append(r"Octree (constr.) & " + " & ".join(f"{x:.1f}" for x in b.build_ms) + r" \\")
    lines.append(r"Memória octree (MB) & " + " & ".join(f"{x:.1f}" for x in b.tree_mb) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_desempenho.tex", lines)


def write_table(name, lines):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("tabela:", name)


# ---------------------------------------------------------------- precisao
def fig_accuracy(df):
    d = df[df.theta > 0]
    fig, axes = plt.subplots(1, 2, figsize=COL2)
    ax = axes[0]
    i = 0
    for scen, name in [("plummer", "Plummer"), ("disk", "Disco")]:
        s = d[(d.scenario == scen) & (d.n == d.n.max())].sort_values("theta")
        ax.plot(s.theta, s.err_median, marker=MARKERS[i], color=C[i], label=f"{name} (mediana)")
        ax.plot(s.theta, s.err_p99, marker=MARKERS[i], color=C[i], ls="--", label=f"{name} (P99)",
                markerfacecolor="white")
        i += 1
    ax.set_yscale("log")
    ax.set_xlabel("Critério de abertura θ")
    ax.set_ylabel("Erro relativo da força |Δa|/|a|")
    ax.legend(loc="lower right")
    ax.set_title(f"Erro × θ (N = {d.n.max() // 1000}k)")

    ax = axes[1]
    i = 0
    for scen, name in [("plummer", "Plummer"), ("disk", "Disco")]:
        s = d[(d.scenario == scen) & (d.n == d.n.max())].sort_values("theta")
        ax.plot(s.theta, s.inter_per_particle, marker=MARKERS[i], color=C[i], label=name)
        i += 1
    ax.axhline(d.n.max() - 1, color=MUTED, ls=":", lw=1)
    ax.text(0.15, (d.n.max() - 1) * 0.6, "força bruta (N−1)", color=MUTED, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xlabel("Critério de abertura θ")
    ax.set_ylabel("Interações por partícula")
    ax.legend(loc="upper right")
    ax.set_title("Custo × θ")
    save(fig, "fig_precisao")


def fig_tradeoff(acc, phys):
    """Fronteira precisao x desempenho para N fixo."""
    fig, ax = plt.subplots(figsize=COL1)
    n = acc.n.max()
    for i, (scen, name) in enumerate([("plummer", "Plummer"), ("disk", "Disco")]):
        a = acc[(acc.scenario == scen) & (acc.n == n) & (acc.theta > 0)]
        p = phys[(phys.scenario == scen) & (phys.n == n) & (phys.method == "bh")]
        m = a.merge(p[["theta", "total_ms"]], on="theta")
        if m.empty:
            continue
        ax.plot(m.total_ms, m.err_median, marker=MARKERS[i], color=C[i], label=name)
        for _, r in m.iterrows():
            ax.annotate(f"θ={r.theta:.1f}", (r.total_ms, r.err_median), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color=MUTED)
        br = phys[(phys.scenario == scen) & (phys.n == n) & (phys.method == "brute")]
        if len(br):
            ax.axvline(br.total_ms.iloc[0], color=C[i], lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tempo por passo (ms)")
    ax.set_ylabel("Erro mediano da força")
    ax.set_title(f"Trade-off precisão × desempenho (N = {n // 1000}k)")
    ax.legend(loc="upper right")
    save(fig, "fig_tradeoff")


def table_accuracy(df):
    n = df.n.max()
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$\theta$ & Mediana & P99 & Máx. & Inter./part. \\", r"\midrule"]
    for _, r in df[(df.scenario == "plummer") & (df.n == n)].sort_values("theta").iterrows():
        lines.append(f"{r.theta:.1f} & {r.err_median:.1e} & {r.err_p99:.1e} & {r.err_max:.1e} & "
                     f"{r.inter_per_particle:,.0f}".replace(",", ".") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_precisao.tex", lines)


# ----------------------------------------------------------------- energia
def fig_energy(df):
    fig, ax = plt.subplots(figsize=(3.45, 3.1))
    order = [c for c in df.config.unique()]
    for i, cfg in enumerate(order):
        s = df[df.config == cfg]
        lab = cfg.replace("leapfrog", "Leapfrog").replace("euler", "Euler") \
                 .replace("-brute", " força bruta").replace("-bh-", " BH θ=")
        ls = (0, (4, 2)) if cfg.startswith("euler") else "-"
        lw = 2.2 if cfg == "euler-bh-0.5" else 1.2
        ax.plot(s.time, np.abs(s.dE_rel) + 1e-12, color=C[i % len(C)], ls=ls, lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Tempo simulado (unidades N-corpos)")
    ax.set_ylabel("|ΔE/E₀|")
    ax.set_title("Conservação de energia (Plummer, N = 4k, dt = 0,01)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5)
    ax.set_ylim(1e-9, 1)
    save(fig, "fig_energia")


def table_energy(df):
    order = {c: i for i, c in enumerate(df.config.unique())}
    last = df.sort_values("step").groupby("config").tail(1)
    last = last.sort_values("config", key=lambda c: c.map(order))
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Configuração & $|\Delta E/E_0|$ final & $|\mathbf{P}|$ final & ms/passo \\",
             r"\midrule"]
    for _, r in last.iterrows():
        lab = r.config.replace("leapfrog", "Leapfrog").replace("euler", "Euler") \
                      .replace("-brute", " / bruta").replace("-bh-", r" / BH $\theta$=")
        lines.append(f"{lab} & {abs(r.dE_rel):.1e} & {r.P:.1e} & {r.wall_ms / max(r.step, 1):.2f}"
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_energia.tex", lines)


# ------------------------------------------------------------------ render
def fig_render(df):
    n = df.n.max()
    d = df[df.n == n]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    series = [("all", 0.0, "Sem LoD"), ("cull", 0.0, "Só culling")] + \
             [("lod", t, f"LoD {t:g} px") for t in sorted(d[d["mode"] == "lod"].pix_thresh.unique())]
    for i, (m, t, lab) in enumerate(series):
        s = d[(d["mode"] == m) & np.isclose(d.pix_thresh, t)].sort_values("distance")
        kw = dict(marker=MARKERS[i], color=C[i], label=lab, markeredgecolor="white",
                  markeredgewidth=0.5)
        axes[0].plot(s.distance, s.drawn, **kw)
        axes[1].plot(s.distance, s.frame_ms, **kw)
        if m != "all":
            axes[2].plot(s.distance, np.minimum(s.psnr, 80), **kw)
    axes[0].set_ylabel("Primitivas desenhadas")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Tempo de renderização\npor quadro (ms)")
    axes[2].set_ylabel("PSNR vs. sem LoD (dB)")
    axes[2].text(d.distance.min(), 79, "80 = idêntico", fontsize=6, color=MUTED, va="top")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Distância da câmera")
    axes[0].legend(loc="lower left", fontsize=6)
    n_txt = f"{n:,}".replace(",", ".")
    fig.suptitle(f"Nível de Detalhe na renderização (galáxia de disco, N = {n_txt}, 1920×1080)",
                 fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_lod")

    # escalabilidade com N em distancia fixa intermediaria
    fig, ax = plt.subplots(figsize=COL1)
    dist = sorted(df.distance.unique())[len(df.distance.unique()) // 2 + 1]
    for i, (m, t, lab) in enumerate([("all", 0.0, "Sem LoD"), ("lod", 2.0, "LoD 2 px")]):
        s = df[(df.distance == dist) & (df["mode"] == m) & np.isclose(df.pix_thresh, t)].sort_values("n")
        ax.plot(s.n, s.gpu_ms, marker=MARKERS[i], color=C[i], label=f"{lab} — GPU")
        ax.plot(s.n, s.frame_ms, marker=MARKERS[i], color=C[i], ls="--",
                markerfacecolor="white", label=f"{lab} — total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Número de partículas N")
    ax.set_ylabel("Tempo (ms)")
    ax.set_title(f"Custo de renderização (distância {dist:g})")
    ax.legend(loc="upper left", fontsize=6)
    save(fig, "fig_lod_escala")


def table_render(df):
    n = df.n.max()
    d = df[df.n == n]
    lines = [r"\begin{tabular}{rlrrrrr}", r"\toprule",
             r"Dist. & Modo & Primitivas & CPU (ms) & GPU (ms) & Quadro (ms) & PSNR (dB) \\", r"\midrule"]
    for dist in sorted(d.distance.unique()):
        for m, t, lab in [("all", 0.0, "sem LoD"), ("cull", 0.0, "culling"), ("lod", 2.0, "LoD 2px")]:
            s = d[(d.distance == dist) & (d["mode"] == m) & np.isclose(d.pix_thresh, t)]
            if s.empty:
                continue
            r = s.iloc[0]
            psnr = r"$\infty$" if not np.isfinite(r.psnr) else f"{r.psnr:.1f}"
            tot = r.frame_ms
            lines.append(f"{dist:g} & {lab} & {int(r.drawn):,} & {r.lod_ms:.1f} & {r.gpu_ms:.2f} & {tot:.1f} & {psnr}"
                         .replace(",", ".") + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write_table("tab_lod.tex", lines)


def fig_leaf(df):
    fig, ax = plt.subplots(figsize=COL1)
    ax.plot(df.leaf, df.build_ms + df.force_ms, marker="o", color=C[0], label="Total")
    ax.plot(df.leaf, df.force_ms, marker="s", color=C[1], label="Força")
    ax.plot(df.leaf, df.build_ms, marker="^", color=C[2], label="Construção da octree")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Partículas máximas por folha")
    ax.set_ylabel("Tempo por passo (ms)")
    ax.set_title(f"Tamanho da folha (Plummer, N = {df.n.iloc[0] // 1000}k, θ = 0,7)")
    ax.legend(loc="upper left")
    save(fig, "fig_folha")


def main():
    phys = load("physics.csv")
    acc = load("accuracy.csv")
    g = None
    if phys is not None:
        g = fig_physics(phys)
        table_physics(g)
    if acc is not None:
        fig_accuracy(acc)
        table_accuracy(acc)
        if g is not None:
            fig_tradeoff(acc, g)
    en = load("energy.csv")
    if en is not None:
        fig_energy(en)
        table_energy(en)
    rd = load("render.csv")
    if rd is not None:
        fig_render(rd)
        table_render(rd)
    lf = load("leaf.csv")
    if lf is not None:
        fig_leaf(lf)


if __name__ == "__main__":
    main()
