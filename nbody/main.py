"""
Simulacao gravitacional de N-corpos com Barnes-Hut e Nivel de Detalhe adaptativo.

Uso:  python main.py --scenario disk --n 20000

Controles
  Mouse esq. (arrastar)  orbitar camera          Roda      zoom
  W/A/S/D  Q/E           mover alvo da camera    Espaco    pausar
  B   Barnes-Hut <-> forca bruta                 [ / ]     theta -/+
  I   integrador Leapfrog <-> Euler
  L   liga/desliga LoD                           , / .     limiar LoD (px) -/+
  F   liga/desliga frustum culling               - / =     velocidade do tempo
  O   octree: off / nivel / corte do LoD         9 / 0     nivel exibido -/+
  1/2/3  cenario plummer / disk / collision      N / M     N /2  x2 (reinicia)
  Backspace reinicia   R grava CSV   P captura tela   H HUD   Z/X exposicao   Esc sai
"""
import argparse
import os
import time

import glfw
import moderngl
import numpy as np

from src.camera import Camera
from src.lod import LodSelector, nodes_at_level
from src.profiler import Profiler
from src.renderer import Renderer, TextOverlay
from src.simulation import Simulation

SCENARIOS = {glfw.KEY_1: "plummer", glfw.KEY_2: "disk", glfw.KEY_3: "collision"}
OCTREE_MODES = ["off", "nivel", "corte LoD"]


class App:
    def __init__(self, args):
        if not glfw.init():
            raise RuntimeError("falha ao iniciar GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        self.win = glfw.create_window(args.width, args.height, "N-corpos Barnes-Hut + LoD", None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("falha ao criar janela OpenGL 3.3")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1 if args.vsync else 0)
        self.ctx = moderngl.create_context()
        self.renderer = Renderer(self.ctx)
        self.text = TextOverlay(self.ctx)
        self.lod = LodSelector()
        self.cam = Camera(distance=args.distance)
        self.prof = Profiler()

        print("Gerando condicoes iniciais e compilando kernels (Numba)...")
        self.sim = Simulation(args.scenario, args.n, seed=args.seed, theta=args.theta,
                              eps=args.eps, dt=args.dt, leaf_size=args.leaf)
        self.sim.use_bh = not args.brute
        if args.brute:
            self.sim.compute_forces()
            self.sim.rebase_energy()
        self.time_scale = 1.0
        self.paused = False
        self.use_lod = not args.no_lod
        self.use_cull = not args.no_lod
        self.pix_thresh = args.pix
        self.octree_mode = 0
        self.octree_level = 3
        self.show_hud = True
        self.dragging = False
        self.last_mouse = (0.0, 0.0)
        self.message = ""
        self.message_t = 0.0
        self.record_path = None
        self.max_frames = args.frames
        self.shot_at_end = args.shot
        if args.record:
            self.record_path = self.prof.toggle_recording()

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_mouse_button_callback(self.win, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self.on_cursor)
        glfw.set_scroll_callback(self.win, self.on_scroll)

    # ---------------------------------------------------------------- entrada
    def notify(self, msg):
        self.message = msg
        self.message_t = time.perf_counter()
        print(msg)

    def on_key(self, win, key, sc, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        s = self.sim
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_B:
            s.use_bh = not s.use_bh
            s.compute_forces()
            s.rebase_energy()
            self.notify("Metodo: " + ("Barnes-Hut" if s.use_bh else "Forca bruta O(N^2)"))
        elif key == glfw.KEY_I:
            s.integrator = "euler" if s.integrator == "leapfrog" else "leapfrog"
            self.notify(f"Integrador: {s.integrator}")
        elif key == glfw.KEY_L:
            self.use_lod = not self.use_lod
        elif key == glfw.KEY_F:
            self.use_cull = not self.use_cull
        elif key == glfw.KEY_LEFT_BRACKET:
            s.theta = round(max(0.0, s.theta - 0.05), 2)
        elif key == glfw.KEY_RIGHT_BRACKET:
            s.theta = round(min(1.5, s.theta + 0.05), 2)
        elif key == glfw.KEY_COMMA:
            self.pix_thresh = max(0.25, self.pix_thresh / 1.25)
        elif key == glfw.KEY_PERIOD:
            self.pix_thresh = min(256.0, self.pix_thresh * 1.25)
        elif key == glfw.KEY_MINUS:
            self.time_scale = max(1 / 16, self.time_scale / 2)
        elif key == glfw.KEY_EQUAL:
            self.time_scale = min(8.0, self.time_scale * 2)
        elif key in SCENARIOS:
            s.reset(SCENARIOS[key])
            self.notify(f"Cenario: {s.scenario}  N={s.n}")
        elif key == glfw.KEY_N:
            s.reset(n=max(500, s.n // 2))
            self.notify(f"N = {s.n}")
        elif key == glfw.KEY_M:
            s.reset(n=min(2_000_000, s.n * 2))
            self.notify(f"N = {s.n}")
        elif key == glfw.KEY_BACKSPACE:
            s.reset()
        elif key == glfw.KEY_O:
            self.octree_mode = (self.octree_mode + 1) % len(OCTREE_MODES)
        elif key == glfw.KEY_9:
            self.octree_level = max(0, self.octree_level - 1)
        elif key == glfw.KEY_0:
            self.octree_level = min(21, self.octree_level + 1)
        elif key == glfw.KEY_Z:
            self.renderer.exposure /= 1.25
        elif key == glfw.KEY_X:
            self.renderer.exposure *= 1.25
        elif key == glfw.KEY_H:
            self.show_hud = not self.show_hud
        elif key == glfw.KEY_R:
            path = self.prof.toggle_recording()
            self.notify(f"Gravando em {path}" if path else "Gravacao encerrada")
        elif key == glfw.KEY_P:
            self.screenshot()

    def on_mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS
            self.last_mouse = glfw.get_cursor_pos(win)

    def on_cursor(self, win, x, y):
        if self.dragging:
            lx, ly = self.last_mouse
            self.cam.rotate(x - lx, y - ly)
        self.last_mouse = (x, y)

    def on_scroll(self, win, dx, dy):
        self.cam.zoom(dy)

    def poll_movement(self, dt):
        k = lambda key: 1.0 if glfw.get_key(self.win, key) == glfw.PRESS else 0.0
        fwd = k(glfw.KEY_W) - k(glfw.KEY_S)
        right = k(glfw.KEY_D) - k(glfw.KEY_A)
        up = k(glfw.KEY_E) - k(glfw.KEY_Q)
        if fwd or right or up:
            self.cam.move(fwd, right, up, dt)

    def screenshot(self):
        from PIL import Image
        w, h = glfw.get_framebuffer_size(self.win)
        data = self.ctx.screen.read(viewport=(0, 0, w, h), components=3)
        img = Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)
        os.makedirs("results/screenshots", exist_ok=True)
        path = time.strftime("results/screenshots/shot_%Y%m%d_%H%M%S.png")
        img.save(path)
        self.notify(f"Captura salva: {path}")

    # ------------------------------------------------------------------- laco
    def run(self):
        last = time.perf_counter()
        while not glfw.window_should_close(self.win):
            t_frame0 = time.perf_counter()
            glfw.poll_events()
            self.poll_movement(t_frame0 - last)
            last = t_frame0
            s = self.sim
            stepped = False
            if not self.paused:
                s.step(s.dt * self.time_scale)
                stepped = True

            w, h = glfw.get_framebuffer_size(self.win)
            if w == 0 or h == 0:
                glfw.wait_events()
                continue
            view = self.cam.view()
            vp = self.cam.projection(w / h) @ view
            proj_scale = self.cam.proj_scale(h)

            # selecao de LoD/culling sobre a mesma octree da fisica; o resultado ja e
            # o buffer de vertices compacto (a ordem de Morton muda a cada passo)
            n_ind = n_agg = n_cull = 0
            t0 = time.perf_counter()
            if self.use_lod or self.use_cull:
                n_ind, n_agg, n_cull = self.lod.select(s, self.cam.eye(), vp, proj_scale,
                                                       self.pix_thresh, self.use_lod, self.use_cull)
            else:
                n_ind = self.lod.pack_all(s)
            drawn = n_ind + n_agg
            lod_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            self.renderer.upload_vertices(self.lod.vert[:drawn])
            t_upload = time.perf_counter() - t0

            n_boxes = 0
            if self.octree_mode == 1:
                t = s.tree
                boxes = np.empty((t.count, 4), np.float32)
                n_boxes = nodes_at_level(t.count, t.level, t.center, t.size, self.octree_level, boxes)
                self.renderer.upload_boxes(boxes[:n_boxes])
            elif self.octree_mode == 2 and self.use_lod:
                n_boxes = n_agg
                self.renderer.upload_boxes(self.lod.box[:n_agg])

            gpu_ms = self.renderer.render(self.ctx.screen, (w, h), vp, proj_scale, drawn, n_boxes)

            frame_ms = (time.perf_counter() - t_frame0) * 1e3
            values = {
                "scenario": s.scenario, "n": s.n, "method": "bh" if s.use_bh else "brute",
                "theta": s.theta, "lod": int(self.use_lod), "cull": int(self.use_cull),
                "pix_thresh": round(self.pix_thresh, 3),
                "sort_ms": s.timings["sort"] if stepped else 0.0,
                "build_ms": s.timings["build"] if stepped else 0.0,
                "force_ms": s.timings["force"] if stepped else 0.0,
                "integrate_ms": s.timings["integrate"] if stepped else 0.0,
                "lod_ms": lod_ms, "upload_ms": t_upload * 1e3, "gpu_ms": gpu_ms,
                "frame_ms": frame_ms, "fps": 1000.0 / max(frame_ms, 1e-6),
                "interactions": s.interactions, "nodes": s.tree.count, "drawn_points": drawn,
                "aggregates": n_agg, "culled": n_cull,
                "vram_mb": self.renderer.vram_bytes() / 2**20,
                "energy_err": s.energy_error(), "momentum": float(np.linalg.norm(s.momentum())),
                "sim_time": s.time, "cam_dist": self.cam.distance,
            }
            self.prof.update(values)
            if self.show_hud:
                self.text.draw(self.hud_lines(values), (w, h))
            glfw.swap_buffers(self.win)
            if self.max_frames and self.prof.frame >= self.max_frames:
                if self.shot_at_end:
                    self.screenshot()
                glfw.set_window_should_close(self.win, True)
            if self.prof.frame % 15 == 0:
                glfw.set_window_title(self.win, f"N-corpos  N={s.n}  {self.prof.avg['fps']:.1f} FPS")
        self.prof.close()
        glfw.terminate()

    def hud_lines(self, v):
        a = self.prof.avg
        s = self.sim
        rec = "  [REC]" if self.prof.recording else ""
        lines = [
            f"Cenario: {s.scenario:<10} N = {s.n:,}{rec}".replace(",", "."),
            f"Metodo : {'Barnes-Hut' if s.use_bh else 'Forca bruta'}   theta = {s.theta:.2f}"
            f"   eps = {s.eps}",
            f"{s.integrator}  dt = {s.dt * self.time_scale:.4g} (x{self.time_scale:g})   t = {s.time:.2f}"
            f"{'   PAUSADO' if self.paused else ''}",
            f"FPS {a['fps']:6.1f}   quadro {a['frame_ms']:7.2f} ms",
            f"  octree  {a['build_ms']:7.2f} ms   ({s.tree.count:,} nos)".replace(",", "."),
            f"  forca   {a['force_ms']:7.2f} ms   ({v['interactions'] / s.n:,.0f} interacoes/part.)"
            .replace(",", "."),
            f"  integr. {a['integrate_ms']:7.2f} ms",
            f"  selecao {a['lod_ms']:7.2f} ms   upload {a['upload_ms']:5.2f} ms",
            f"  GPU     {a['gpu_ms']:7.2f} ms",
            f"LoD {'ON ' if self.use_lod else 'OFF'} (limiar {self.pix_thresh:.2f} px)   "
            f"culling {'ON' if self.use_cull else 'OFF'}",
            f"  desenhados {v['drawn_points']:,}  impostores {v['aggregates']:,}  "
            f"descartados {v['culled']:,}".replace(",", "."),
            f"Octree: {OCTREE_MODES[self.octree_mode]}"
            + (f" {self.octree_level}" if self.octree_mode == 1 else ""),
            f"dE/E0 = {v['energy_err']:+.3e}   |P| = {v['momentum']:.2e}",
            f"RAM {self.prof.rss_mb:7.1f} MB   VRAM (buffers) {v['vram_mb']:6.1f} MB",
        ]
        if self.message and time.perf_counter() - self.message_t < 3.0:
            lines.append(self.message)
        return lines


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="disk", choices=["plummer", "disk", "collision"])
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--theta", type=float, default=0.7)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--leaf", type=int, default=4, help="particulas maximas por folha")
    p.add_argument("--pix", type=float, default=2.0, help="limiar do LoD em pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--distance", type=float, default=14.0)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--vsync", action="store_true")
    p.add_argument("--record", action="store_true", help="grava metricas desde o inicio")
    p.add_argument("--brute", action="store_true", help="inicia com forca bruta")
    p.add_argument("--no-lod", action="store_true", help="inicia sem LoD e sem culling")
    p.add_argument("--frames", type=int, default=0, help="encerra apos K quadros (0 = nunca)")
    p.add_argument("--shot", action="store_true", help="salva captura de tela ao encerrar")
    return p.parse_args()


if __name__ == "__main__":
    App(parse_args()).run()
