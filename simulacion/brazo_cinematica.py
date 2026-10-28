"""
brazo_cinematica.py

Cinemática directa del brazo de 6 GDL según la Tabla 9 (Denavit-Hartenberg)
de la tesis del brazo.  Reemplaza a dh_6gdl.py, cuyos parámetros (tomados del
código MATLAB del Anexo IV) NO coinciden con la tesis.

Convención: A_i = Rz(theta_i + offset_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)

    Eslabón   theta_i      d_i   a_i   alpha_i
       1      theta1       l1    0     -90°
       2      theta2       l2    l3    180°
       3      theta3-90°   0     0     -90°
       4      theta4       l4    0     -90°
       5      theta5       0     0     +90°
       6      theta6       l5    0     0°

Con todos los ángulos en cero, esta tabla reproduce EXACTAMENTE lo que reporta
la tesis (sección 7.8): posición (103.3, 9.5, 60.5) y orientación
R06 = [[0,0,1],[0,1,0],[-1,0,0]]. De ahí salen tres datos:
    l1 = 60.5,   l2 = 9.5,   l3 + l4 + l5 = 103.3
(unidades de la tesis, probablemente cm).

PENDIENTE: medir en el brazo real cómo se reparte 103.3 entre l3, l4 y l5.
Los valores L3 y L4 de abajo son PROVISIONALES (L5 se calcula para conservar
la suma). Reemplázalos por las medidas reales.

Ángulos de entrada q: RADIANES. Se asume que son los mismos ángulos de
articulación que usa la tesis en sus límites (Tabla 15); confírmalo en el brazo.

Requiere: numpy
"""

import numpy as np

NUM_JOINTS = 6

# --- Dimensiones de los eslabones ----------------------------------------
L1 = 60.5            # de la tesis (z en home)
L2 = 9.5             # de la tesis (y en home)
SUMA_L3_L4_L5 = 103.3  # de la tesis (x en home)

L3 = 45.0            # PROVISIONAL: medir en el brazo real
L4 = 40.0            # PROVISIONAL: medir en el brazo real
L5 = SUMA_L3_L4_L5 - L3 - L4   # si mides L5 directamente, escribe aquí su valor

# --- Tabla DH (Tabla 9 de la tesis) --------------------------------------
DH_D = np.array([L1, L2, 0.0, L4, 0.0, L5])
DH_A = np.array([0.0, L3, 0.0, 0.0, 0.0, 0.0])
DH_ALPHA = np.deg2rad([-90.0, 180.0, -90.0, -90.0, 90.0, 0.0])
DH_THETA_OFFSET = np.deg2rad([0.0, 0.0, -90.0, 0.0, 0.0, 0.0])


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Matriz de transformación homogénea 4x4 de un eslabón (DH estándar)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def _validar(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (NUM_JOINTS,):
        raise ValueError(f"q debe tener {NUM_JOINTS} elementos, recibió {q.shape}")
    return q


def forward_kinematics(q) -> np.ndarray:
    """Pose del efector final respecto a la base: matriz homogénea 4x4 (T06)."""
    q = _validar(q)
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        T = T @ dh_matrix(q[i] + DH_THETA_OFFSET[i], DH_D[i], DH_A[i], DH_ALPHA[i])
    return T


def end_effector_position(q) -> np.ndarray:
    """Posición (x, y, z) del efector final."""
    return forward_kinematics(q)[:3, 3]


def joint_positions(q) -> np.ndarray:
    """Origen de cada marco, de la base al efector final: arreglo (7, 3)."""
    q = _validar(q)
    puntos = [np.zeros(3)]
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        T = T @ dh_matrix(q[i] + DH_THETA_OFFSET[i], DH_D[i], DH_A[i], DH_ALPHA[i])
        puntos.append(T[:3, 3].copy())
    return np.array(puntos)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    T_home = forward_kinematics(np.zeros(NUM_JOINTS))
    print("Home (todos los ángulos en 0)")
    print("  Posición calculada:", T_home[:3, 3], "  (tesis: [103.3, 9.5, 60.5])")
    print("  R06 calculada:\n", T_home[:3, :3])
    print("  R06 de la tesis:\n", np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

    assert np.allclose(T_home[:3, 3], [103.3, 9.5, 60.5], atol=1e-6), "no coincide la posición home"
    assert np.allclose(T_home[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-9), "no coincide R06"
    print("\nOK: coincide con la posición y orientación home de la tesis.")

    q_prueba = np.deg2rad([30, -20, 45, 10, 60, -15])
    print("\nq = [30, -20, 45, 10, 60, -15] grados")
    print("  Posición del efector final:", end_effector_position(q_prueba))
