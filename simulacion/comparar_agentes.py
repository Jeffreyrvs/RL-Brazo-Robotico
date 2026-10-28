"""
comparar_agentes.py

Compara, sobre los MISMOS episodios, tres formas de llevar la punta al objetivo:
  1. Agente aleatorio             (línea base mínima)
  2. Controlador clásico          (método matemático con el Jacobiano; no aprende)
  3. Agente entrenado con RL      (tu modelo guardado, si se indica)

Imprime una tabla y guarda cada episodio en un CSV, lista para el informe.

Uso (con los mismos parámetros con los que entrenaste el modelo):
    python comparar_agentes.py --modelo ppo_reach_radio1.5_periodo0.5 --radio 1.5
    python comparar_agentes.py --sin-modelo                 # solo aleatorio y clásico
    python comparar_agentes.py --modelo sac_reach_radio5_periodo0.5 --algoritmo sac --radio 5

Requiere: numpy, gymnasium; y stable-baselines3 solo si se usa --modelo.
Archivos en la misma carpeta: brazo_cinematica.py, reach_env.py
"""

import argparse
import csv

import numpy as np

from brazo_cinematica import end_effector_position, NUM_JOINTS
from reach_env import Reach6DOFEnv, nombre_modelo


def jacobiano(q, eps=1e-5):
    """Jacobiano de posición (3x6) por diferencias finitas."""
    J = np.zeros((3, NUM_JOINTS))
    p0 = end_effector_position(q)
    for i in range(NUM_JOINTS):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (end_effector_position(dq) - p0) / eps
    return J


def controlador_clasico(env):
    """Mueve las articulaciones con la pseudo-inversa amortiguada del Jacobiano."""
    J = jacobiano(env.q)
    error = env.target - end_effector_position(env.q)
    dq = J.T @ np.linalg.solve(J @ J.T + 1.0 * np.eye(3), error)
    accion = dq / env.max_delta
    return np.clip(accion / max(1.0, np.abs(accion).max()), -1.0, 1.0)


def evaluar(nombre, accion_fn, env, episodios, semilla):
    """Corre los episodios y devuelve una lista de filas por episodio."""
    filas = []
    for ep in range(episodios):
        obs, info = env.reset(seed=semilla + ep)
        d_inicial = info["distance"]
        terminado = truncado = False
        pasos = 0
        while not (terminado or truncado):
            obs, _, terminado, truncado, info = env.step(accion_fn(obs))
            pasos += 1
        filas.append([nombre, ep, d_inicial, info["distance"], pasos, int(info["is_success"])])
    return filas


def resumen(nombre, filas):
    exito = np.array([f[5] for f in filas], dtype=float)
    finales = np.array([f[3] for f in filas])
    pasos_ok = [f[4] for f in filas if f[5]]
    return (nombre, exito.mean(), finales.mean(),
            np.mean(pasos_ok) if pasos_ok else float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, default=None, help="modelo guardado (sin .zip)")
    parser.add_argument("--sin-modelo", action="store_true", help="solo aleatorio y clásico")
    parser.add_argument("--algoritmo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--radio", type=float, default=1.5)
    parser.add_argument("--dispersion", type=float, default=30.0)
    parser.add_argument("--inicio-aleatorio", action="store_true")
    parser.add_argument("--delta", type=float, default=None,
                        help="giro máximo uniforme por paso (grados); si no se indica, velocidades reales x --periodo")
    parser.add_argument("--periodo", type=float, default=0.5,
                        help="duración de cada paso del simulador, en segundos")
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--semilla", type=int, default=1000)
    parser.add_argument("--salida", type=str, default="comparacion_agentes.csv")
    args = parser.parse_args()

    spread = None if args.inicio_aleatorio else args.dispersion
    env = Reach6DOFEnv(success_radius=args.radio, start_spread_deg=spread,
                       max_delta_deg=args.delta, control_period_s=args.periodo)

    todas, resumenes = [], []

    filas = evaluar("aleatorio", lambda obs: env.action_space.sample(), env,
                    args.episodios, args.semilla)
    todas += filas
    resumenes.append(resumen("Agente aleatorio", filas))

    filas = evaluar("clasico", lambda obs: controlador_clasico(env), env,
                    args.episodios, args.semilla)
    todas += filas
    resumenes.append(resumen("Controlador clásico", filas))

    if not args.sin_modelo:
        ruta_modelo = args.modelo or nombre_modelo(args.algoritmo, args.radio, args.delta, args.periodo)
        from stable_baselines3 import PPO, SAC  # solo se necesita con un modelo
        modelo = (SAC if args.algoritmo == "sac" else PPO).load(ruta_modelo)
        filas = evaluar("rl", lambda obs: modelo.predict(obs, deterministic=True)[0], env,
                        args.episodios, args.semilla)
        todas += filas
        resumenes.append(resumen(f"Agente RL ({args.algoritmo.upper()})", filas))

    print(f"\nEpisodios: {args.episodios}   radio de éxito: {args.radio:g}   "
          f"inicio: {'aleatorio' if spread is None else f'dispersión {spread:g}°'}   "
          f"{'paso uniforme ' + format(args.delta, 'g') + '°' if args.delta is not None else 'velocidades reales, periodo ' + format(args.periodo, 'g') + ' s'}\n")
    print(f"{'Método':<24}{'Éxito':>8}{'Dist. final media':>20}{'Pasos (solo éxitos)':>22}")
    for nombre, exito, dist, pasos in resumenes:
        pasos_txt = "-" if np.isnan(pasos) else f"{pasos:.0f}"
        print(f"{nombre:<24}{exito:>8.0%}{dist:>20.1f}{pasos_txt:>22}")

    with open(args.salida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "episodio", "distancia_inicial", "distancia_final", "pasos", "exito"])
        for fila in todas:
            w.writerow([fila[0], fila[1], round(fila[2], 3), round(fila[3], 3), fila[4], fila[5]])
    print(f"\nDetalle por episodio guardado en {args.salida}")


if __name__ == "__main__":
    main()
