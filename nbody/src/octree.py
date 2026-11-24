"""
Octree linear baseada em curva de Morton (Z-order), compartilhada entre a
fisica (Barnes-Hut) e a renderizacao (Nivel de Detalhe).

Construcao:
  1. As posicoes sao quantizadas em uma grade 2^21 x 2^21 x 2^21 dentro do
     cubo envolvente e codificadas em um codigo de Morton de 63 bits.
  2. As particulas sao ordenadas pelo codigo; assim, todo no da octree
     corresponde a um intervalo contiguo [start, end) do vetor ordenado.
  3. A arvore e gerada em pre-ordem (DFS). Com isso a subarvore do no k ocupa
     os indices [k, next[k]) e a travessia pode ser feita sem pilha:
     "abrir" um no = k + 1, "aceitar/pular" um no = next[k].
  4. Massa, centro de massa e cor media de cada no sao obtidos em O(1) por
     somas de prefixo sobre o vetor ordenado.
"""
import numpy as np
from numba import njit, prange

MORTON_BITS = 21
GRID = 1 << MORTON_BITS


@njit(inline="always")
def _expand_bits(v):
    v = v & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


@njit(parallel=True, cache=True)
def morton_codes(pos, bmin, inv_cell):
    n = pos.shape[0]
    codes = np.empty(n, dtype=np.uint64)
    gmax = GRID - 1
    for i in prange(n):
        qx = min(max(int((pos[i, 0] - bmin[0]) * inv_cell), 0), gmax)
        qy = min(max(int((pos[i, 1] - bmin[1]) * inv_cell), 0), gmax)
        qz = min(max(int((pos[i, 2] - bmin[2]) * inv_cell), 0), gmax)
        codes[i] = (_expand_bits(np.uint64(qx)) << np.uint64(2)) | \
                   (_expand_bits(np.uint64(qy)) << np.uint64(1)) | \
                   _expand_bits(np.uint64(qz))
    return codes


@njit(cache=True)
def build_nodes(codes, leaf_size, start, end, level, parent, nxt, is_leaf):
    """Gera os nos em pre-ordem. Retorna o numero de nos ou -1 se faltou espaco."""
    n = codes.shape[0]
    max_nodes = start.shape[0]
    stack = np.empty((512, 4), dtype=np.int64)  # (s, e, nivel, pai)
    cs = np.empty(8, dtype=np.int64)
    ce = np.empty(8, dtype=np.int64)
    sp = 0
    stack[0, 0] = 0
    stack[0, 1] = n
    stack[0, 2] = 0
    stack[0, 3] = -1
    sp = 1
    cnt = 0
    while sp > 0:
        sp -= 1
        s = stack[sp, 0]
        e = stack[sp, 1]
        lv = stack[sp, 2]
        p = stack[sp, 3]
        if cnt >= max_nodes:
            return -1
        k = cnt
        cnt += 1
        start[k] = s
        end[k] = e
        level[k] = lv
        parent[k] = p
        if e - s <= leaf_size or lv >= MORTON_BITS:
            is_leaf[k] = 1
            continue
        is_leaf[k] = 0
        shift = np.uint64(3 * (MORTON_BITS - 1 - lv))
        nc = 0
        i = s
        while i < e:
            o = (codes[i] >> shift) & np.uint64(7)
            j = i + 1
            while j < e and ((codes[j] >> shift) & np.uint64(7)) == o:
                j += 1
            cs[nc] = i
            ce[nc] = j
            nc += 1
            i = j
        # empilha em ordem reversa para que o primeiro filho seja visitado primeiro
        for c in range(nc - 1, -1, -1):
            stack[sp, 0] = cs[c]
            stack[sp, 1] = ce[c]
            stack[sp, 2] = lv + 1
            stack[sp, 3] = k
            sp += 1
    # tamanho das subarvores -> ponteiro "next" (skip)
    sub = np.ones(cnt, dtype=np.int64)
    for k in range(cnt - 1, 0, -1):
        sub[parent[k]] += sub[k]
    for k in range(cnt):
        nxt[k] = k + sub[k]
    return cnt


@njit(parallel=True, cache=True)
def node_properties(cnt, start, end, level, pos, mass, color, bmin, root_size,
                    n_mass, n_com, n_col, n_size, n_center):
    n = pos.shape[0]
    # somas de prefixo (sequenciais, O(N))
    cm = np.zeros(n + 1)
    cx = np.zeros((n + 1, 3))
    cc = np.zeros((n + 1, 3))
    for i in range(n):
        m = mass[i]
        cm[i + 1] = cm[i] + m
        for a in range(3):
            cx[i + 1, a] = cx[i, a] + m * pos[i, a]
            cc[i + 1, a] = cc[i, a] + m * color[i, a]
    cell = root_size / GRID
    for k in prange(cnt):
        s = start[k]
        e = end[k]
        M = cm[e] - cm[s]
        n_mass[k] = M
        inv = 1.0 / M if M > 0 else 0.0
        for a in range(3):
            n_com[k, a] = (cx[e, a] - cx[s, a]) * inv
            n_col[k, a] = (cc[e, a] - cc[s, a]) * inv
        lv = level[k]
        n_size[k] = root_size / (1 << lv)
        # centro geometrico do cubo, a partir da celula quantizada da 1a particula
        span = 1 << (MORTON_BITS - lv)
        for a in range(3):
            q = int((pos[s, a] - bmin[a]) / cell)
            q = min(max(q, 0), GRID - 1)
            q0 = (q // span) * span
            n_center[k, a] = bmin[a] + (q0 + 0.5 * span) * cell


class Octree:
    """Armazena a octree em Structure of Arrays (vetores contiguos)."""

    def __init__(self, leaf_size=8):
        self.leaf_size = leaf_size
        self.count = 0
        self._alloc(1024)

    def _alloc(self, m):
        self.capacity = m
        self.start = np.empty(m, np.int64)
        self.end = np.empty(m, np.int64)
        self.level = np.empty(m, np.int64)
        self.parent = np.empty(m, np.int64)
        self.next = np.empty(m, np.int64)
        self.is_leaf = np.empty(m, np.uint8)
        self.mass = np.empty(m)
        self.com = np.empty((m, 3))
        self.color = np.empty((m, 3))
        self.size = np.empty(m)
        self.center = np.empty((m, 3))

    def nbytes(self):
        arrs = (self.start, self.end, self.level, self.parent, self.next, self.is_leaf,
                self.mass, self.com, self.color, self.size, self.center)
        return sum(a.nbytes for a in arrs)

    def sort_order(self, pos):
        """Calcula o cubo envolvente e a permutacao de Morton das particulas."""
        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        side = float((hi - lo).max()) * (1.0 + 1e-6) + 1e-9
        center = 0.5 * (lo + hi)
        self.bmin = center - 0.5 * side
        self.root_size = side
        codes = morton_codes(pos, self.bmin, GRID / side)
        perm = np.argsort(codes, kind="stable")
        self._codes = codes[perm]
        return perm

    def build(self, pos_sorted, mass_sorted, color_sorted):
        """Constroi os nos; deve ser chamado apos aplicar a permutacao de sort_order."""
        n = pos_sorted.shape[0]
        if self.capacity < 2 * n // self.leaf_size + 64:
            self._alloc(4 * n // self.leaf_size + 1024)
        while True:
            cnt = build_nodes(self._codes, self.leaf_size, self.start, self.end,
                              self.level, self.parent, self.next, self.is_leaf)
            if cnt > 0:
                break
            self._alloc(self.capacity * 2)
        self.count = cnt
        node_properties(cnt, self.start, self.end, self.level, pos_sorted, mass_sorted,
                        color_sorted, self.bmin, self.root_size, self.mass, self.com,
                        self.color, self.size, self.center)
        return cnt
