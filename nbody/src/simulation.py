"""
Estado da simulacao (Structure of Arrays) e integrador Leapfrog (KDK).

    v(t + dt/2) = v(t) + a(t) dt/2          (kick)
    x(t + dt)   = x(t) + v(t + dt/2) dt     (drift)
    a(t + dt)   = F(x(t + dt)) / m          (octree + Barnes-Hut ou forca bruta)
    v(t + dt)   = v(t + dt/2) + a(t + dt) dt/2   (kick)

A cada passo as particulas sao reordenadas pela curva de Morton, o que melhora
a localidade de memoria e faz cada no da octree corresponder a um intervalo
contiguo do buffer de vertices usado na renderizacao.
"""
import time
import numpy as np

from .forces import accel_barnes_hut, accel_brute
from .initial_conditions import make_scenario
from .octree import Octree


class Simulation:
    def __init__(self, scenario="disk", n=10000, seed=0, theta=0.7, eps=0.05, dt=0.005,
                 leaf_size=4, G=1.0):
        self.scenario = scenario
        self.seed = seed
        self.theta = theta
        self.eps = eps
        self.dt = dt
        self.G = G
        self.use_bh = True
        self.integrator = "leapfrog"   # ou "euler" (comparacao de estabilidade)
        self.tree = Octree(leaf_size)
        self.timings = {"sort": 0.0, "build": 0.0, "force": 0.0, "integrate": 0.0}
        self.reset(scenario, n)

    # ------------------------------------------------------------------ estado
    def reset(self, scenario=None, n=None, seed=None):
        if scenario is not None:
            self.scenario = scenario
        if seed is not None:
            self.seed = seed
        n = n or self.n
        self.pos, self.vel, self.mass, self.color = make_scenario(
            self.scenario, n, self.seed, self.eps)
        self.n = len(self.mass)
        self.acc = np.zeros_like(self.pos)
        self.phi = np.zeros(self.n)
        self.inter = np.zeros(self.n, np.int64)
        self.interactions = 0
        self.time = 0.0
        self.steps = 0
        self.compute_forces()
        self.E0 = self.total_energy()
        self.P0 = self.momentum()
        self.L0 = self.angular_momentum()

    def _permute(self, perm):
        self.pos = self.pos[perm]
        self.vel = self.vel[perm]
        self.mass = self.mass[perm]
        self.color = self.color[perm]

    # ------------------------------------------------------------------ fisica
    def build_tree(self):
        t0 = time.perf_counter()
        perm = self.tree.sort_order(self.pos)
        self._permute(perm)
        t1 = time.perf_counter()
        self.tree.build(self.pos, self.mass, self.color)
        t2 = time.perf_counter()
        self.timings["sort"] = (t1 - t0) * 1e3
        self.timings["build"] = (t2 - t0) * 1e3  # inclui codigos de Morton + ordenacao

    def compute_forces(self):
        # a octree e sempre reconstruida: e usada tambem pelo LoD da renderizacao
        self.build_tree()
        t0 = time.perf_counter()
        t = self.tree
        if self.use_bh:
            self.interactions = accel_barnes_hut(
                self.pos, self.mass, t.count, t.start, t.end, t.next, t.is_leaf,
                t.mass, t.com, t.size, self.theta, self.eps, self.G,
                self.acc, self.phi, self.inter)
        else:
            self.interactions = accel_brute(self.pos, self.mass, self.eps, self.G,
                                            self.acc, self.phi)
        self.timings["force"] = (time.perf_counter() - t0) * 1e3

    def step(self, dt=None):
        dt = self.dt if dt is None else dt
        if self.integrator == "euler":
            return self._step_euler(dt)
        t0 = time.perf_counter()
        self.vel += 0.5 * dt * self.acc
        self.pos += dt * self.vel
        t_int = time.perf_counter() - t0
        self.compute_forces()
        t0 = time.perf_counter()
        self.vel += 0.5 * dt * self.acc
        t_int += time.perf_counter() - t0
        self.timings["integrate"] = t_int * 1e3
        self.time += dt
        self.steps += 1

    def _step_euler(self, dt):
        """Euler explicito (x e v com a aceleracao do inicio do passo)."""
        t0 = time.perf_counter()
        self.pos += dt * self.vel
        self.vel += dt * self.acc
        t_int = time.perf_counter() - t0
        self.compute_forces()
        self.timings["integrate"] = t_int * 1e3
        self.time += dt
        self.steps += 1

    # ----------------------------------------------------------- diagnosticos
    def kinetic_energy(self):
        return 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", self.vel, self.vel)))

    def potential_energy(self):
        """Usa o potencial calculado junto com a forca (exato na forca bruta,
        aproximado pelo mesmo theta no Barnes-Hut)."""
        return 0.5 * float(np.sum(self.mass * self.phi))

    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy()

    def exact_potential_energy(self):
        acc = np.empty_like(self.pos)
        phi = np.empty(self.n)
        accel_brute(self.pos, self.mass, self.eps, self.G, acc, phi)
        return 0.5 * float(np.sum(self.mass * phi))

    def momentum(self):
        return (self.mass[:, None] * self.vel).sum(axis=0)

    def angular_momentum(self):
        return (self.mass[:, None] * np.cross(self.pos, self.vel)).sum(axis=0)

    def rebase_energy(self):
        """Redefine E0 apos trocar o estimador do potencial (metodo ou theta)."""
        self.E0 = self.total_energy()

    def energy_error(self):
        return (self.total_energy() - self.E0) / abs(self.E0)

    def memory_bytes(self):
        arrs = (self.pos, self.vel, self.acc, self.mass, self.color, self.phi, self.inter)
        return sum(a.nbytes for a in arrs) + self.tree.nbytes()
