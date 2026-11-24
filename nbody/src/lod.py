"""
Nivel de Detalhe (LoD) reaproveitando a octree da fisica.

Percorre a octree (sem pilha, via ponteiros next) e, para cada no:
  * descarta-o se a esfera envolvente estiver fora do frustum (opcional);
  * se o tamanho projetado na tela  s * f / d  for menor que `pix_thresh`
    pixels, desenha o no inteiro como UM impostor (sprite agregado) com
    posicao = centro de massa, cor = cor media ponderada pela massa e
    fluxo luminoso = soma dos fluxos das estrelas contidas;
  * se for folha "grande", emite suas estrelas individualmente;
  * caso contrario, desce para os filhos.
A travessia ja escreve o buffer de vertices compacto (estrelas individuais e
impostores no mesmo formato), de modo que o custo de upload para a GPU passa a
ser proporcional ao que e efetivamente desenhado, e nao a N.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def lod_select(n_nodes, start, end, nxt, is_leaf, n_mass, n_com, n_col, n_size, n_center,
               cam, planes, proj_scale, pix_thresh, use_lod, use_cull, pos, color, mass,
               mass_unit, out_vert, out_box):
    """Retorna (n_individuais, n_agregados, n_culled).
    out_vert[v] = (x, y, z, r, g, b, fluxo, tamanho_mundo); tamanho 0 = estrela
    out_box[m]  = (cx, cy, cz, meia_aresta) dos nos agregados (visualizacao)."""
    n_ind = 0
    n_agg = 0
    n_cull = 0
    k = 0
    sqrt3 = math.sqrt(3.0)
    while k < n_nodes:
        s = n_size[k]
        if use_cull:
            r = 0.5 * s * sqrt3
            cx = n_center[k, 0]
            cy = n_center[k, 1]
            cz = n_center[k, 2]
            outside = False
            for p in range(6):
                if planes[p, 0] * cx + planes[p, 1] * cy + planes[p, 2] * cz + planes[p, 3] < -r:
                    outside = True
                    break
            if outside:
                n_cull += end[k] - start[k]
                k = nxt[k]
                continue
        dx = n_com[k, 0] - cam[0]
        dy = n_com[k, 1] - cam[1]
        dz = n_com[k, 2] - cam[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
        proj = s * proj_scale / d
        cnt = end[k] - start[k]
        if use_lod and proj < pix_thresh and cnt > 1:
            v = n_ind + n_agg
            out_vert[v, 0] = n_com[k, 0]
            out_vert[v, 1] = n_com[k, 1]
            out_vert[v, 2] = n_com[k, 2]
            out_vert[v, 3] = n_col[k, 0]
            out_vert[v, 4] = n_col[k, 1]
            out_vert[v, 5] = n_col[k, 2]
            out_vert[v, 6] = n_mass[k] / mass_unit
            out_vert[v, 7] = s
            out_box[n_agg, 0] = n_center[k, 0]
            out_box[n_agg, 1] = n_center[k, 1]
            out_box[n_agg, 2] = n_center[k, 2]
            out_box[n_agg, 3] = 0.5 * s
            n_agg += 1
            k = nxt[k]
        elif is_leaf[k]:
            for j in range(start[k], end[k]):
                v = n_ind + n_agg
                out_vert[v, 0] = pos[j, 0]
                out_vert[v, 1] = pos[j, 1]
                out_vert[v, 2] = pos[j, 2]
                out_vert[v, 3] = color[j, 0]
                out_vert[v, 4] = color[j, 1]
                out_vert[v, 5] = color[j, 2]
                out_vert[v, 6] = mass[j] / mass_unit
                out_vert[v, 7] = 0.0
                n_ind += 1
            k = nxt[k]
        else:
            k += 1
    return n_ind, n_agg, n_cull


@njit(parallel=True, cache=True)
def pack_all(pos, color, mass, mass_unit, out_vert):
    """Buffer de vertices com todas as particulas (modo sem LoD)."""
    for j in prange(pos.shape[0]):
        out_vert[j, 0] = pos[j, 0]
        out_vert[j, 1] = pos[j, 1]
        out_vert[j, 2] = pos[j, 2]
        out_vert[j, 3] = color[j, 0]
        out_vert[j, 4] = color[j, 1]
        out_vert[j, 5] = color[j, 2]
        out_vert[j, 6] = mass[j] / mass_unit
        out_vert[j, 7] = 0.0


@njit(cache=True)
def nodes_at_level(n_nodes, level, n_center, n_size, target, out_box):
    """Caixas dos nos em um nivel da octree (ou folhas mais rasas)."""
    m = 0
    for k in range(n_nodes):
        if level[k] == target:
            out_box[m, 0] = n_center[k, 0]
            out_box[m, 1] = n_center[k, 1]
            out_box[m, 2] = n_center[k, 2]
            out_box[m, 3] = 0.5 * n_size[k]
            m += 1
    return m


def frustum_planes(vp):
    """Planos do frustum (Gribb & Hartmann) a partir da matriz view-projection
    (convencao de vetor coluna). Cada linha: (a, b, c, d) normalizado."""
    r0, r1, r2, r3 = vp[0], vp[1], vp[2], vp[3]
    planes = np.array([r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2])
    planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    return planes


class LodSelector:
    """Produz o buffer de vertices a desenhar (com ou sem LoD/culling)."""

    def __init__(self):
        self.vert = np.empty((0, 8), np.float32)
        self.box = np.empty((0, 4), np.float32)

    def _ensure(self, n, nodes):
        if self.vert.shape[0] < n + nodes:
            self.vert = np.empty((n + nodes, 8), np.float32)
        if self.box.shape[0] < nodes:
            self.box = np.empty((nodes, 4), np.float32)

    def select(self, sim, cam_pos, vp, proj_scale, pix_thresh, use_lod, use_cull):
        t = sim.tree
        self._ensure(sim.n, t.count)
        planes = frustum_planes(vp)
        n_ind, n_agg, n_cull = lod_select(
            t.count, t.start, t.end, t.next, t.is_leaf, t.mass, t.com, t.color, t.size,
            t.center, np.asarray(cam_pos, np.float64), planes, float(proj_scale),
            float(pix_thresh), use_lod, use_cull, sim.pos, sim.color, sim.mass,
            float(sim.mass.mean()), self.vert, self.box)
        return n_ind, n_agg, n_cull

    def pack_all(self, sim):
        self._ensure(sim.n, 0)
        pack_all(sim.pos, sim.color, sim.mass, float(sim.mass.mean()), self.vert)
        return sim.n
