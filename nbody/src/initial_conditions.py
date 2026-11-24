"""
Condicoes iniciais sinteticas (unidades N-corpos: G = 1, massa total = 1).

  * plummer:   esfera de Plummer em equilibrio (aglomerado globular),
               amostrada pelo metodo de Aarseth, Henon & Wielen (1974).
  * disk:      galaxia de disco = bojo de Plummer + disco exponencial em
               rotacao, com velocidade circular obtida das forcas reais.
  * collision: duas galaxias de disco em rota de colisao.

Todas retornam (pos[N,3] f64, vel[N,3] f64, mass[N] f64, color[N,3] f32).
"""
import numpy as np

from .forces import accel_barnes_hut
from .octree import Octree


def _random_directions(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _star_colors(rng, n, kind):
    """Paleta aproximada de temperatura estelar (vermelho -> azul)."""
    if kind == "old":      # populacao velha (bojo / aglomerado): amarelo-alaranjado
        base = np.array([1.0, 0.78, 0.52])
        jitter = 0.12
    elif kind == "young":  # disco: branco-azulado
        base = np.array([0.62, 0.74, 1.0])
        jitter = 0.18
    else:
        base = np.array([0.9, 0.85, 0.8])
        jitter = 0.2
    t = rng.uniform(-1, 1, size=(n, 1))
    warm = np.array([1.0, 0.65, 0.4])
    cool = np.array([0.55, 0.7, 1.0])
    col = base + jitter * np.where(t > 0, t * (cool - base), -t * (warm - base))
    col *= rng.uniform(0.75, 1.15, size=(n, 1))
    return np.clip(col, 0.0, 1.5).astype(np.float32)


def plummer(n, rng, mass_total=1.0, a=1.0, rmax=10.0):
    # raio: inversao da massa acumulada M(r) = r^3 / (1 + r^2)^(3/2)
    r = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        x = rng.uniform(1e-10, 1.0, size=todo.size)
        rr = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
        ok = rr < rmax
        r[todo[ok]] = rr[ok]
        todo = todo[~ok]
    pos = _random_directions(rng, n) * r[:, None]

    # velocidade: amostragem por rejeicao de g(q) = q^2 (1 - q^2)^(7/2)
    q = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        x4 = rng.uniform(0, 1, size=todo.size)
        x5 = rng.uniform(0, 0.1, size=todo.size)
        ok = x5 < x4 ** 2 * (1 - x4 ** 2) ** 3.5
        q[todo[ok]] = x4[ok]
        todo = todo[~ok]
    ve = np.sqrt(2.0) * (1.0 + r ** 2) ** -0.25
    vel = _random_directions(rng, n) * (q * ve)[:, None]

    # escala para massa total e raio a: v ~ sqrt(G M / a)
    pos *= a
    vel *= np.sqrt(mass_total / a)
    mass = np.full(n, mass_total / n)
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
    return pos, vel, mass


def _tree_accel(pos, mass, eps, theta=0.5):
    tree = Octree(leaf_size=8)
    perm = tree.sort_order(pos)
    ps, ms = pos[perm].copy(), mass[perm].copy()
    tree.build(ps, ms, np.zeros((len(ps), 3), np.float32))
    acc_s = np.empty_like(ps)
    phi = np.empty(len(ps))
    inter = np.empty(len(ps), np.int64)
    accel_barnes_hut(ps, ms, tree.count, tree.start, tree.end, tree.next, tree.is_leaf,
                     tree.mass, tree.com, tree.size, theta, eps, 1.0, acc_s, phi, inter)
    acc = np.empty_like(acc_s)
    acc[perm] = acc_s
    return acc


def disk_galaxy(n, rng, eps=0.05, mass_total=1.0, bulge_frac_mass=0.25,
                bulge_frac_n=0.2, rd=1.0, zd=0.06, rmax=6.0, sigma_frac=0.12):
    nb = max(int(n * bulge_frac_n), 1)
    nd = n - nb
    mb = mass_total * bulge_frac_mass
    md = mass_total - mb

    # bojo: Plummer compacto
    pb, vb, _ = plummer(nb, rng, mass_total=mb, a=0.25 * rd, rmax=3.0)
    # disco exponencial: Sigma(R) ~ exp(-R/Rd) -> R ~ Gamma(2, Rd)
    R = np.empty(nd)
    todo = np.arange(nd)
    while todo.size:
        rr = rng.gamma(2.0, rd, size=todo.size)
        ok = rr < rmax
        R[todo[ok]] = rr[ok]
        todo = todo[~ok]
    ph = rng.uniform(0, 2 * np.pi, nd)
    z = rng.normal(0, zd, nd)
    pd = np.stack([R * np.cos(ph), R * np.sin(ph), z], axis=1)

    pos = np.concatenate([pb, pd])
    mass = np.concatenate([np.full(nb, mb / nb), np.full(nd, md / nd)])

    # velocidade circular a partir da forca radial real (Barnes-Hut)
    acc = _tree_accel(pos, mass, eps)
    ad = acc[nb:]
    a_r = (ad[:, 0] * pd[:, 0] + ad[:, 1] * pd[:, 1]) / np.maximum(R, 1e-9)
    vc = np.sqrt(np.maximum(-a_r * R, 0.0))
    sig = sigma_frac * vc
    vR = rng.normal(0, 1, nd) * sig
    vphi = vc + rng.normal(0, 1, nd) * sig * 0.5
    vz = rng.normal(0, 1, nd) * sig * 0.5
    vd = np.stack([vR * np.cos(ph) - vphi * np.sin(ph),
                   vR * np.sin(ph) + vphi * np.cos(ph), vz], axis=1)
    # bojo sente tambem o potencial do disco: aumenta levemente a dispersao
    vb *= np.sqrt(1.0 + md / mb * 0.25)
    vel = np.concatenate([vb, vd])

    color = np.concatenate([_star_colors(rng, nb, "old"), _star_colors(rng, nd, "young")])
    pos -= np.average(pos, axis=0, weights=mass)
    vel -= np.average(vel, axis=0, weights=mass)
    return pos, vel, mass, color


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def make_scenario(name, n, seed=0, eps=0.05):
    rng = np.random.default_rng(seed)
    if name == "plummer":
        pos, vel, mass = plummer(n, rng)
        color = _star_colors(rng, n, "mixed")
    elif name == "disk":
        pos, vel, mass, color = disk_galaxy(n, rng, eps=eps)
    elif name == "collision":
        n1 = n // 2
        # cada galaxia com massa 0.5: as velocidades internas escalam com sqrt(M)
        p1, v1, m1, c1 = disk_galaxy(n1, rng, eps=eps)
        p2, v2, m2, c2 = disk_galaxy(n - n1, rng, eps=eps)
        v1 *= np.sqrt(0.5)
        v2 *= np.sqrt(0.5)
        rot = _rotation_x(np.radians(60.0))
        p2 = p2 @ rot.T
        v2 = v2 @ rot.T
        c2 = (c2 * np.array([1.0, 0.8, 0.75], np.float32)).astype(np.float32)  # tom distinto
        p1 += np.array([-5.0, -1.2, 0.0])
        p2 += np.array([5.0, 1.2, 0.0])
        v1 += np.array([0.16, 0.0, 0.0])
        v2 += np.array([-0.16, 0.0, 0.0])
        pos = np.concatenate([p1, p2])
        vel = np.concatenate([v1, v2])
        mass = np.concatenate([m1, m2]) * 0.5
        color = np.concatenate([c1, c2]).astype(np.float32)
        pos -= np.average(pos, axis=0, weights=mass)
        vel -= np.average(vel, axis=0, weights=mass)
    else:
        raise ValueError(f"cenario desconhecido: {name}")
    if name != "plummer":
        # discos gerados no plano XY; a camera usa Y como "cima" -> disco no plano XZ
        pos = pos[:, [0, 2, 1]]
        vel = vel[:, [0, 2, 1]]
    return (np.ascontiguousarray(pos, np.float64), np.ascontiguousarray(vel, np.float64),
            np.ascontiguousarray(mass, np.float64), np.ascontiguousarray(color, np.float32))
