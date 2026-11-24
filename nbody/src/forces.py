"""
Calculo das aceleracoes gravitacionais.

  * accel_brute:      soma direta par-a-par, O(N^2) (referencia exata)
  * accel_barnes_hut: travessia sem pilha da octree, O(N log N)

Ambos usam amortecimento de Plummer (softening):
    a_i = G * sum_j m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2)
    phi_i = -G * sum_j m_j / sqrt(|x_j - x_i|^2 + eps^2)
"""
import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def accel_brute(pos, mass, eps, G, acc, phi):
    n = pos.shape[0]
    eps2 = eps * eps
    for i in prange(n):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        ax = 0.0
        ay = 0.0
        az = 0.0
        p = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            r2 = dx * dx + dy * dy + dz * dz + eps2
            inv = 1.0 / math.sqrt(r2)
            mi = mass[j] * inv
            mi3 = mi * inv * inv
            ax += mi3 * dx
            ay += mi3 * dy
            az += mi3 * dz
            p -= mi
        acc[i, 0] = G * ax
        acc[i, 1] = G * ay
        acc[i, 2] = G * az
        phi[i] = G * p
    return n * (n - 1)


@njit(parallel=True, fastmath=True, cache=True)
def accel_barnes_hut(pos, mass, n_nodes, start, end, nxt, is_leaf, n_mass, n_com,
                     n_size, theta, eps, G, acc, phi, inter):
    """Criterio de abertura de Barnes & Hut: o no e aceito como massa pontual
    se s/d < theta  <=>  s^2 < theta^2 d^2. Com theta = 0 nenhum no e aceito e
    o resultado coincide com a forca bruta (teste de corretude)."""
    n = pos.shape[0]
    eps2 = eps * eps
    th2 = theta * theta
    for i in prange(n):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        ax = 0.0
        ay = 0.0
        az = 0.0
        p = 0.0
        cnt = 0
        k = 0
        while k < n_nodes:
            if is_leaf[k]:
                for j in range(start[k], end[k]):
                    if j == i:
                        continue
                    dx = pos[j, 0] - xi
                    dy = pos[j, 1] - yi
                    dz = pos[j, 2] - zi
                    r2 = dx * dx + dy * dy + dz * dz + eps2
                    inv = 1.0 / math.sqrt(r2)
                    mi = mass[j] * inv
                    mi3 = mi * inv * inv
                    ax += mi3 * dx
                    ay += mi3 * dy
                    az += mi3 * dz
                    p -= mi
                cnt += end[k] - start[k]
                k = nxt[k]
            else:
                dx = n_com[k, 0] - xi
                dy = n_com[k, 1] - yi
                dz = n_com[k, 2] - zi
                d2 = dx * dx + dy * dy + dz * dz
                s = n_size[k]
                if s * s < th2 * d2:
                    inv = 1.0 / math.sqrt(d2 + eps2)
                    mi = n_mass[k] * inv
                    mi3 = mi * inv * inv
                    ax += mi3 * dx
                    ay += mi3 * dy
                    az += mi3 * dz
                    p -= mi
                    cnt += 1
                    k = nxt[k]
                else:
                    k += 1
        acc[i, 0] = G * ax
        acc[i, 1] = G * ay
        acc[i, 2] = G * az
        phi[i] = G * p
        inter[i] = cnt
    return inter.sum()


@njit(parallel=True, fastmath=True, cache=True)
def accel_direct_subset(pos, mass, idx, eps, G, acc_out):
    """Aceleracao exata apenas para um subconjunto de particulas (amostragem
    do erro de forca em populacoes grandes)."""
    n = pos.shape[0]
    eps2 = eps * eps
    for t in prange(idx.shape[0]):
        i = idx[t]
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            inv = 1.0 / math.sqrt(r2)
            mi3 = mass[j] * inv * inv * inv
            ax += mi3 * dx
            ay += mi3 * dy
            az += mi3 * dz
        acc_out[t, 0] = G * ax
        acc_out[t, 1] = G * ay
        acc_out[t, 2] = G * az
