"""
Pipeline de renderizacao (ModernGL / OpenGL 3.3):

  1. Estrelas e impostores (mesmo buffer) como point sprites gaussianos, com blending aditivo
     em um framebuffer HDR (RGBA16F). O tamanho do sprite de um impostor e o
     tamanho projetado do no; sua intensidade e normalizada pela area para que
     o fluxo total seja igual a soma das estrelas que ele representa.
  2. (Opcional) caixas da octree desenhadas com instancing.
  3. Tone mapping exponencial + correcao gama para o framebuffer de destino.
"""
import os
import numpy as np
import moderngl

STAR_VS = """
#version 330
uniform mat4 u_vp;
uniform float u_point_px;
uniform float u_proj_scale;
uniform float u_max_px;
in vec3 in_pos;
in vec3 in_col;
in float in_flux;
in float in_wsize;
out vec3 v_col;
out float v_int;
void main() {
    gl_Position = u_vp * vec4(in_pos, 1.0);
    float w = max(gl_Position.w, 1e-4);
    float size_px = clamp(max(u_point_px, in_wsize * u_proj_scale / w), 1.0, u_max_px);
    gl_PointSize = size_px;
    // fluxo conservado: intensidade * area = constante
    v_int = in_flux * (u_point_px * u_point_px) / (size_px * size_px);
    v_col = in_col;
}
"""

STAR_FS = """
#version 330
uniform float u_gain;
in vec3 v_col;
in float v_int;
out vec4 frag;
void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float g = exp(-4.0 * r2);
    frag = vec4(v_col * (v_int * g * u_gain), 1.0);
}
"""

BOX_VS = """
#version 330
uniform mat4 u_vp;
in vec3 in_vert;
in vec4 in_inst;
void main() {
    gl_Position = u_vp * vec4(in_inst.xyz + in_vert * in_inst.w, 1.0);
}
"""

BOX_FS = """
#version 330
uniform vec3 u_color;
out vec4 frag;
void main() { frag = vec4(u_color, 1.0); }
"""

QUAD_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

TONEMAP_FS = """
#version 330
uniform sampler2D u_hdr;
uniform float u_exposure;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec3 hdr = texture(u_hdr, v_uv).rgb;
    vec3 bg = mix(vec3(0.004, 0.005, 0.012), vec3(0.012, 0.010, 0.025), v_uv.y);
    vec3 c = 1.0 - exp(-hdr * u_exposure);
    c = pow(c, vec3(1.0 / 2.2)) + bg;
    frag = vec4(c, 1.0);
}
"""

TEXT_VS = """
#version 330
uniform vec2 u_screen;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    vec2 p = in_pos / u_screen * 2.0 - 1.0;
    gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
}
"""

TEXT_FS = """
#version 330
uniform sampler2D u_atlas;
uniform vec4 u_color;
in vec2 v_uv;
out vec4 frag;
void main() { frag = vec4(u_color.rgb, u_color.a * texture(u_atlas, v_uv).r); }
"""

_CUBE_EDGES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, -1, -1], [1, 1, -1], [1, 1, -1], [-1, 1, -1], [-1, 1, -1], [-1, -1, -1],
    [-1, -1, 1], [1, -1, 1], [1, -1, 1], [1, 1, 1], [1, 1, 1], [-1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    [-1, -1, -1], [-1, -1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1], [-1, 1, -1], [-1, 1, 1],
], dtype="f4")


def _mat_bytes(m):
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


class GrowBuffer:
    """Buffer de GPU que cresce sob demanda (orphaning a cada escrita)."""

    def __init__(self, ctx, nbytes=1024):
        self.ctx = ctx
        self.capacity = nbytes
        self.buf = ctx.buffer(reserve=nbytes, dynamic=True)
        self.version = 0

    def write(self, data):
        data = np.ascontiguousarray(data)
        nb = data.nbytes
        if nb > self.capacity:
            self.buf.release()
            self.capacity = int(nb * 1.5)
            self.buf = self.ctx.buffer(reserve=self.capacity, dynamic=True)
            self.version += 1
        else:
            self.buf.orphan(self.capacity)
        if nb:
            self.buf.write(data.tobytes())


class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.star_prog = ctx.program(vertex_shader=STAR_VS, fragment_shader=STAR_FS)
        self.box_prog = ctx.program(vertex_shader=BOX_VS, fragment_shader=BOX_FS)
        self.tone_prog = ctx.program(vertex_shader=QUAD_VS, fragment_shader=TONEMAP_FS)
        self.vertices = GrowBuffer(ctx)
        self.boxes = GrowBuffer(ctx)
        self.cube = ctx.buffer(_CUBE_EDGES.tobytes())
        self.empty_vao = ctx.vertex_array(self.tone_prog, [])
        self._vaos = {}
        self.hdr_size = (0, 0)
        self.hdr_tex = None
        self.hdr_fbo = None
        self.query = ctx.query(time=True)
        self.gpu_ms = 0.0
        self.point_px = 2.5
        self.gain = 0.35
        self.exposure = 1.0

    # ---------------------------------------------------------------- buffers
    def _vao(self, key, builder, *versions):
        cached = self._vaos.get(key)
        if cached is None or cached[0] != versions:
            if cached is not None:
                cached[1].release()
            self._vaos[key] = (versions, builder())
        return self._vaos[key][1]

    def upload_vertices(self, vert):
        """vert[:, 8] = (x, y, z, r, g, b, fluxo, tamanho_mundo) em float32."""
        self.vertices.write(vert)
        return vert.nbytes

    def upload_boxes(self, boxes):
        self.boxes.write(boxes.astype("f4", copy=False))
        self.n_boxes = boxes.shape[0]

    def _ensure_hdr(self, w, h):
        if self.hdr_size == (w, h):
            return
        if self.hdr_fbo is not None:
            self.hdr_fbo.release()
            self.hdr_tex.release()
        self.hdr_tex = self.ctx.texture((w, h), 4, dtype="f2")
        self.hdr_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.hdr_fbo = self.ctx.framebuffer(color_attachments=[self.hdr_tex])
        self.hdr_size = (w, h)

    def vram_bytes(self):
        hdr = self.hdr_size[0] * self.hdr_size[1] * 8
        return self.vertices.capacity + self.boxes.capacity + hdr

    # ------------------------------------------------------------------ desenho
    def render(self, target, size, vp, proj_scale, count, n_boxes=0,
               box_color=(0.05, 0.12, 0.08)):
        """Desenha `count` vertices do buffer (estrelas e impostores) e as
        caixas da octree, com tone mapping para `target`."""
        w, h = size
        ctx = self.ctx
        self._ensure_hdr(w, h)
        sp = self.star_prog
        sp["u_vp"].write(_mat_bytes(vp))
        sp["u_point_px"].value = self.point_px
        sp["u_proj_scale"].value = proj_scale
        sp["u_max_px"].value = 256.0
        sp["u_gain"].value = self.gain

        with self.query:
            self.hdr_fbo.use()
            ctx.viewport = (0, 0, w, h)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            if count:
                vao = self._vao("stars", lambda: ctx.vertex_array(
                    sp, [(self.vertices.buf, "3f 3f 1f 1f", "in_pos", "in_col", "in_flux",
                          "in_wsize")]), self.vertices.version)
                vao.render(moderngl.POINTS, vertices=count)
            if n_boxes:
                self.box_prog["u_vp"].write(_mat_bytes(vp))
                self.box_prog["u_color"].value = box_color
                vao = self._vao("box", lambda: ctx.vertex_array(
                    self.box_prog, [(self.cube, "3f", "in_vert"),
                                    (self.boxes.buf, "4f/i", "in_inst")]), self.boxes.version)
                vao.render(moderngl.LINES, instances=n_boxes)

            target.use()
            ctx.viewport = (0, 0, w, h)
            ctx.disable(moderngl.BLEND)
            self.hdr_tex.use(0)
            self.tone_prog["u_hdr"].value = 0
            self.tone_prog["u_exposure"].value = self.exposure
            self.empty_vao.render(moderngl.TRIANGLES, vertices=3)
        self.gpu_ms = self.query.elapsed / 1e6
        return self.gpu_ms


class TextOverlay:
    """HUD em texto usando um atlas de fonte gerado com Pillow."""

    def __init__(self, ctx, size=15):
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib
        font_path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts",
                                 "ttf", "DejaVuSansMono.ttf")
        font = ImageFont.truetype(font_path, size)
        asc, desc = font.getmetrics()
        self.cw = int(round(font.getlength("M")))
        self.ch = asc + desc
        cols, rows = 16, 7
        img = Image.new("L", (cols * self.cw, rows * self.ch), 0)
        draw = ImageDraw.Draw(img)
        for c in range(32, 127):
            i = c - 32
            draw.text(((i % cols) * self.cw, (i // cols) * self.ch), chr(c), fill=255, font=font)
        # glifo 95 = bloco solido (fundo do painel)
        i = 95
        draw.rectangle([(i % cols) * self.cw, (i // cols) * self.ch,
                        (i % cols + 1) * self.cw - 1, (i // cols + 1) * self.ch - 1], fill=255)
        self.cols = cols
        self.aw, self.ah = img.size
        self.ctx = ctx
        self.tex = ctx.texture(img.size, 1, img.tobytes())
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.prog = ctx.program(vertex_shader=TEXT_VS, fragment_shader=TEXT_FS)
        self.vbo = GrowBuffer(ctx)
        self.vao = None
        self.vao_version = -1

    def _quad(self, out, x, y, w, h, gi):
        u0 = (gi % self.cols) * self.cw / self.aw
        v0 = (gi // self.cols) * self.ch / self.ah
        u1 = u0 + self.cw / self.aw
        v1 = v0 + self.ch / self.ah
        out += [x, y, u0, v0, x + w, y, u1, v0, x + w, y + h, u1, v1,
                x, y, u0, v0, x + w, y + h, u1, v1, x, y + h, u0, v1]

    def draw(self, lines, screen_size, x0=10, y0=10, panel=True):
        verts = []
        if panel:
            width = max(len(s) for s in lines) * self.cw + 16
            height = len(lines) * self.ch + 12
            self._quad(verts, x0 - 8, y0 - 6, width, height, 95)
        n_panel = len(verts) // 4
        for li, line in enumerate(lines):
            for ci, ch in enumerate(line):
                c = ord(ch)
                if 33 <= c <= 126:
                    self._quad(verts, x0 + ci * self.cw, y0 + li * self.ch, self.cw, self.ch, c - 32)
        data = np.array(verts, "f4")
        self.vbo.write(data)
        if self.vao is None or self.vao_version != self.vbo.version:
            if self.vao is not None:
                self.vao.release()
            self.vao = self.ctx.vertex_array(self.prog, [(self.vbo.buf, "2f 2f", "in_pos", "in_uv")])
            self.vao_version = self.vbo.version
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.tex.use(0)
        self.prog["u_atlas"].value = 0
        self.prog["u_screen"].value = screen_size
        n_total = len(verts) // 4
        if panel:
            self.prog["u_color"].value = (0.0, 0.0, 0.0, 0.55)
            self.vao.render(moderngl.TRIANGLES, vertices=n_panel, first=0)
        self.prog["u_color"].value = (0.85, 0.92, 1.0, 1.0)
        self.vao.render(moderngl.TRIANGLES, vertices=n_total - n_panel, first=n_panel)
        ctx.disable(moderngl.BLEND)
