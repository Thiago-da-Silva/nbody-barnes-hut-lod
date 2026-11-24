"""Gera as capturas usadas no artigo (renderizacao offscreen, sem janela)."""
import os

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.camera import Camera
from src.lod import LodSelector, nodes_at_level
from src.renderer import Renderer
from src.simulation import Simulation

W, H = 1200, 800
OUT = os.path.join("results", "images")


def render(ctx, fbo, rend, sim, cam, lod_thr=None, boxes=None, box_color=(0.05, 0.12, 0.08)):
    vp = cam.projection(W / H) @ cam.view()
    ps = cam.proj_scale(H)
    lod = LodSelector()
    if lod_thr is None:
        count = lod.pack_all(sim)
    else:
        ni, na, _ = lod.select(sim, cam.eye(), vp, ps, lod_thr, True, True)
        count = ni + na
    rend.upload_vertices(lod.vert[:count])
    nb = 0
    if boxes is not None:
        rend.upload_boxes(boxes)
        nb = len(boxes)
    rend.render(fbo, (W, H), vp, ps, count, nb, box_color)
    a = np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1]
    return Image.fromarray(a.copy())


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.framebuffer([ctx.texture((W, H), 3)])
    rend = Renderer(ctx)
    rend.gain = 0.45

    # (a) galaxia de disco apos ~1,5 rotacoes do disco interno
    disk = Simulation("disk", 60000, theta=0.7, dt=0.01)
    for _ in range(600):
        disk.step()
    cam = Camera(distance=11.0, yaw=0.4, pitch=0.75)
    img_a = render(ctx, fbo, rend, disk, cam)

    # (b) colisao de galaxias
    col = Simulation("collision", 60000, theta=0.7, dt=0.01)
    for _ in range(1800):
        col.step()
    cam_c = Camera(distance=22.0, yaw=1.2, pitch=0.9)
    cam_c.target = np.average(col.pos, axis=0, weights=col.mass)
    img_b = render(ctx, fbo, rend, col, cam_c)

    # (c) nos da octree no nivel 4
    t = disk.tree
    boxes = np.empty((t.count, 4), np.float32)
    nb = nodes_at_level(t.count, t.level, t.center, t.size, 4, boxes)
    img_c = render(ctx, fbo, rend, disk, cam, boxes=boxes[:nb], box_color=(0.05, 0.16, 0.10))

    # (d) corte do LoD (nos agregados) visto de longe
    cam_far = Camera(distance=40.0, yaw=0.4, pitch=0.75)
    vp = cam_far.projection(W / H) @ cam_far.view()
    lod = LodSelector()
    ni, na, _ = lod.select(disk, cam_far.eye(), vp, cam_far.proj_scale(H), 3.0, True, True)
    img_d = render(ctx, fbo, rend, disk, Camera(distance=11.0, yaw=0.4, pitch=0.75),
                   boxes=lod.box[:na].copy(), box_color=(0.18, 0.07, 0.02))
    print(f"LoD (d=40, 3 px): {ni} individuais + {na} impostores de {disk.n}")

    names = ["(a)", "(b)", "(c)", "(d)"]
    imgs = [img_a, img_b, img_c, img_d]
    for nm, im in zip("abcd", imgs):
        im.save(os.path.join(OUT, f"captura_{nm}.png"))
    grid = Image.new("RGB", (W * 2, H * 2))
    font = ImageFont.load_default(size=44)
    for i, (im, lab) in enumerate(zip(imgs, names)):
        d = ImageDraw.Draw(im)
        d.text((24, 18), lab, fill=(230, 230, 230), font=font)
        grid.paste(im, ((i % 2) * W, (i // 2) * H))
    grid = grid.resize((W, H))
    grid.save(os.path.join("results", "figures", "fig_capturas.png"))
    print("salvo results/figures/fig_capturas.png")


if __name__ == "__main__":
    main()
