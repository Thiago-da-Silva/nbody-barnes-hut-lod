# Simulação Gravitacional de N-Corpos — Barnes-Hut + Nível de Detalhe

Projeto de Computação Gráfica (Eixo 4 — Dinâmica Molecular Clássica, adaptado
para mecânica celeste). Simulação interativa de milhares de estrelas com:

- **Octree linear por curva de Morton** (`src/octree.py`): partículas ordenadas
  em Z-order, nós em pré-ordem com ponteiro *skip* (travessia sem pilha),
  massa / centro de massa / cor média por somas de prefixo.
- **Barnes-Hut** O(N log N) com critério `s/d < θ` e *softening* de Plummer
  (`src/forces.py`); θ = 0 reproduz exatamente a força bruta (teste de corretude).
- **Força bruta** O(N²) paralela para comparação.
- **Integrador Leapfrog KDK** (simplético) e Euler explícito para comparação
  (`src/simulation.py`).
- **LoD reaproveitando a mesma octree** (`src/lod.py`): nós cujo tamanho
  projetado é menor que um limiar em pixels viram um único impostor com fluxo
  luminoso conservado; *frustum culling* por nó.
- **Renderização ModernGL** (`src/renderer.py`): point sprites gaussianos,
  blending aditivo em framebuffer HDR (RGBA16F) e *tone mapping*.
- **Instrumentação** (`src/profiler.py`): tempos de octree/força/LoD/upload,
  tempo de GPU (query `GL_TIME_ELAPSED`), FPS, RAM, VRAM de buffers, erro de
  energia — gravados em CSV por quadro.

Física em NumPy + Numba (JIT paralelo), estruturas em *Structure of Arrays*.

## Instalação

```bash
pip install -r requirements.txt
```

Requer GPU com OpenGL 3.3. A primeira execução compila os kernels Numba
(~10–20 s); as seguintes usam o cache.

## Execução interativa

```bash
python main.py --scenario disk --n 20000
```

Cenários: `plummer` (aglomerado globular), `disk` (galáxia espiral),
`collision` (duas galáxias colidindo).

| Tecla | Ação | Tecla | Ação |
|---|---|---|---|
| Arrastar mouse | orbitar câmera | Roda | zoom |
| W A S D / Q E | mover câmera | Espaço | pausar |
| B | Barnes-Hut ↔ força bruta | `[` / `]` | θ − / + |
| L | liga/desliga LoD | `,` / `.` | limiar LoD (px) − / + |
| F | frustum culling | `-` / `=` | velocidade do tempo |
| I | Leapfrog ↔ Euler | O | octree: off / nível / corte do LoD |
| 9 / 0 | nível exibido da octree | 1 / 2 / 3 | cenário |
| N / M | N ÷2 / ×2 | Backspace | reiniciar |
| R | gravar métricas em CSV | P | captura de tela |
| H | HUD | Z / X | exposição |

## Experimentos (Etapa 4)

```bash
python benchmark.py all     # ~15-25 min; --quick para um teste rápido
python plots.py             # figuras em results/figures, tabelas em results/tables
```

| Experimento | Arquivo | Conteúdo |
|---|---|---|
| `physics` | `results/data/physics.csv` | tempo de octree e força × N (1k–100k) × método (bruta, θ = 0,3…1,0), memória |
| `leaf` | `results/data/leaf.csv` | efeito do tamanho máximo da folha |
| `accuracy` | `results/data/accuracy.csv` | erro relativo da força × θ |
| `energy` | `results/data/energy.csv` | ΔE/E₀ e momento ao longo do tempo, Leapfrog vs Euler |
| `render` | `results/data/render.csv` | primitivas, tempo GPU/CPU e PSNR do LoD × distância × N (até 1M) |


## Estrutura

```
main.py            aplicação interativa (GLFW + ModernGL)
benchmark.py       experimentos sem janela -> CSV
plots.py           gráficos (Matplotlib) e tabelas LaTeX
src/octree.py      octree de Morton
src/forces.py      Barnes-Hut e força bruta
src/simulation.py  estado SoA + Leapfrog
src/initial_conditions.py  Plummer, disco, colisão
src/lod.py         seleção de LoD e frustum culling
src/renderer.py    shaders, HDR, caixas da octree, HUD
src/camera.py      câmera e matrizes
src/profiler.py    métricas e CSV
```
