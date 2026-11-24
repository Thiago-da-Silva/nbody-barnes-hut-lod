"""Camera orbital/livre e matrizes de projecao (convencao de vetor coluna)."""
import math
import numpy as np


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class Camera:
    """Orbita em torno de `target` (arrasto do mouse), zoom pela roda e
    translacao livre do alvo com WASD/QE no referencial da camera."""

    def __init__(self, distance=14.0, yaw=0.6, pitch=0.55, fovy=50.0):
        self.target = np.zeros(3)
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.fovy = fovy
        self.near = 0.01
        self.far = 2000.0

    def eye(self):
        cp = math.cos(self.pitch)
        d = np.array([cp * math.cos(self.yaw), math.sin(self.pitch), cp * math.sin(self.yaw)])
        return self.target + self.distance * d

    def basis(self):
        f = self.target - self.eye()
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 1.0, 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u

    def rotate(self, dx, dy):
        self.yaw += dx * 0.005
        self.pitch = float(np.clip(self.pitch + dy * 0.005, -1.55, 1.55))

    def zoom(self, steps):
        self.distance = float(np.clip(self.distance * (0.9 ** steps), 0.05, 1500.0))

    def move(self, fwd, right, up, dt):
        f, r, u = self.basis()
        speed = self.distance * 0.8 * dt
        self.target += speed * (fwd * f + right * r + up * u)

    def view(self):
        return look_at(self.eye(), self.target, np.array([0.0, 1.0, 0.0]))

    def projection(self, aspect):
        return perspective(self.fovy, aspect, self.near, self.far)

    def proj_scale(self, height):
        """Pixels por unidade de mundo a distancia 1."""
        return height / (2.0 * math.tan(math.radians(self.fovy) / 2.0))
