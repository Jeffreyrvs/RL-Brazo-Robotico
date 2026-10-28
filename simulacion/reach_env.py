"""
reach_env.py

Entorno de Gymnasium: el agente controla los 6 ángulos del brazo y debe
llevar la punta (efector final) a un punto objetivo (x, y, z).

No requiere visión: el objetivo es simplemente un punto con coordenadas
conocidas. En simulación se sortea un punto alcanzable; en el brazo real
lo escribiría el operador.

Requiere: numpy, gymnasium, y brazo_cinematica.py en la misma carpeta.

Unidades: las de la tesis del brazo (probablemente cm). Ángulos internos en radianes.

Modelo del brazo: Tabla 9 de la tesis (ver brazo_cinematica.py). Las longitudes
L3 y L4 de ese archivo son PROVISIONALES hasta medirlas en el brazo real.

Límites de giro: Tabla 15 de la tesis (pruebas con límites mecánicos).
Nota: la articulación 6 es un servo MG995 sin encoder; en el simulador se
trata igual que las demás.

Velocidades: el control manual del Arduino (Codigo_final.ino) mueve cada articulación a
una velocidad fija y baja (J1 7.5, J2 4.5, J3 7.2, J4 5.6, J5 16.4 grados/s; J6 es un servo,
~66.7 grados/s). En el simulador cada paso dura `control_period_s` segundos, así que el giro
máximo por paso de cada articulación es velocidad x periodo. Una acción continua a en [-1, 1]
equivale a mover la articulación a su velocidad fija durante la fracción |a| del periodo.

Dificultad (currículo): con start_spread_deg = 30 (por defecto) el brazo
arranca a ~30° por articulación de una pose que lleva la punta al objetivo,
lo que hace la tarea aprendible. Cuando el agente domine ese caso, se puede
subir start_spread_deg (60, 90...) o usar None (inicio totalmente aleatorio,
la versión más difícil).
"""

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from brazo_cinematica import end_effector_position, NUM_JOINTS

# --- Límites de giro de cada articulación (grados), Tabla 15 de la tesis --
JOINT_LIMITS_DEG = np.array([
    [-270.0, 90.0],    # articulación 1 (cintura)
    [-60.0, 60.0],     # articulación 2 (hombro)
    [-135.0, 135.0],   # articulación 3 (codo)
    [-180.0, 180.0],   # articulación 4
    [-70.0, 70.0],     # articulación 5
    [-90.0, 90.0],     # articulación 6 (servo)
])

# Velocidad máxima de cada articulación (grados/s) según el modo manual de Codigo_final.ino
# (pulsos de 625 Hz o 500 Hz con 800 pulsos/vuelta y las reducciones de la tesis).
# J6 es un servo que avanza 2 grados cada 30 ms en manual.
VELOCIDAD_MAX_DEG_S = np.array([7.5, 4.5, 7.2, 5.6, 16.4, 66.7])

POS_SCALE = 100.0  # para normalizar posiciones (alcance del brazo ~100)


def nombre_modelo(algoritmo: str, radio: float, delta, periodo: float) -> str:
    """Nombre por defecto de los modelos guardados (lo usan train_ppo, diagnostico y comparar)."""
    if delta is not None:
        return f"{algoritmo}_reach_radio{radio:g}_delta{delta:g}"
    return f"{algoritmo}_reach_radio{radio:g}_periodo{periodo:g}"


class Reach6DOFEnv(gym.Env):
    """Alcanzar un punto objetivo con la punta del brazo de 6 GDL."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_period_s: float = 0.5,         # duración de cada paso, en segundos
        max_delta_deg: Optional[float] = None, # si se indica, giro máximo uniforme por paso (grados),
                                               # en lugar de velocidad real x periodo
        success_radius: float = 1.5,   # distancia para dar el objetivo por alcanzado
        max_steps: int = 200,          # pasos máximos por episodio
        action_penalty: float = 0.01,  # castigo pequeño a movimientos bruscos
        success_bonus: float = 10.0,
        start_spread_deg: Optional[float] = 30.0,  # None = inicio totalmente aleatorio
    ):
        super().__init__()
        self.lo = np.deg2rad(JOINT_LIMITS_DEG[:, 0])
        self.hi = np.deg2rad(JOINT_LIMITS_DEG[:, 1])
        self.centro = 0.5 * (self.lo + self.hi)
        self.semirrango = 0.5 * (self.hi - self.lo)
        if max_delta_deg is None:
            self.max_delta = np.deg2rad(VELOCIDAD_MAX_DEG_S * control_period_s)
        else:
            self.max_delta = np.full(NUM_JOINTS, np.deg2rad(max_delta_deg))
        self.success_radius = success_radius
        self.max_steps = max_steps
        self.action_penalty = action_penalty
        self.success_bonus = success_bonus
        self.start_spread = None if start_spread_deg is None else np.deg2rad(start_spread_deg)

        # Acción: 6 valores en [-1, 1]; cada uno es una fracción del giro
        # máximo por paso de esa articulación.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(NUM_JOINTS,), dtype=np.float32)

        # Observación (15 valores): 6 ángulos normalizados a [-1, 1],
        # posición de la punta, posición del objetivo y vector punta->objetivo
        # (las posiciones divididas entre POS_SCALE).
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(NUM_JOINTS + 9,), dtype=np.float32
        )

        self.q = self.centro.copy()
        self.target = np.zeros(3)
        self.steps = 0

    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        q_norm = (self.q - self.centro) / self.semirrango
        ee = end_effector_position(self.q)
        obs = np.concatenate([
            q_norm,
            ee / POS_SCALE,
            self.target / POS_SCALE,
            (self.target - ee) / POS_SCALE,
        ])
        return obs.astype(np.float32)

    def _distance(self) -> float:
        return float(np.linalg.norm(end_effector_position(self.q) - self.target))

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # El objetivo es la posición de la punta en una pose aleatoria
        # (usando 80% del rango de cada articulación, alrededor de su centro,
        # para no pedir poses extremas), así que siempre es alcanzable.
        # La pose inicial depende de start_spread: cerca de la pose objetivo
        # (currículo) o totalmente aleatoria. Si la punta queda demasiado
        # cerca del objetivo, se vuelve a sortear.
        for _ in range(20):
            q_goal = self.centro + rng.uniform(-0.8, 0.8, NUM_JOINTS) * self.semirrango
            self.target = end_effector_position(q_goal)
            if self.start_spread is None:
                self.q = rng.uniform(self.lo, self.hi)
            else:
                self.q = np.clip(
                    q_goal + rng.uniform(-self.start_spread, self.start_spread, NUM_JOINTS),
                    self.lo, self.hi,
                )
            if self._distance() > 3.0 * self.success_radius:
                break

        self.steps = 0
        return self._observation(), {"distance": self._distance()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.q = np.clip(self.q + action * self.max_delta, self.lo, self.hi)
        self.steps += 1

        dist = self._distance()
        success = dist < self.success_radius

        reward = -dist / POS_SCALE - self.action_penalty * float(np.mean(action ** 2))
        if success:
            reward += self.success_bonus

        terminated = bool(success)
        truncated = bool(self.steps >= self.max_steps and not terminated)

        info = {"distance": dist, "is_success": success}
        return self._observation(), float(reward), terminated, truncated, info
