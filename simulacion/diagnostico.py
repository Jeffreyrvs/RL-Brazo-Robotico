"""
diagnostico.py

Analiza un modelo ya entrenado para entender POR QUÉ no llega al objetivo:
¿pasa cerca y se pasa de largo (oscila), o se queda lejos?

Usa los MISMOS parámetros con los que entrenaste ese modelo (radio, periodo o delta, dispersión).

Uso:
    python diagnostico.py --radio 5                   # busca ppo_reach_radio5_periodo0.5.zip
    python diagnostico.py --modelo ppo_reach --radio 5    # un modelo con otro nombre
    python diagnostico.py --algoritmo sac --radio 1.5

Requiere: pip install numpy gymnasium stable-baselines3
Archivos en la misma carpeta: dh_6gdl.py, reach_env.py y el modelo .zip
"""

import argparse

import numpy as np
from stable_baselines3 import PPO, SAC

from reach_env import Reach6DOFEnv, nombre_modelo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, default=None,
                        help="nombre del modelo (sin .zip); por defecto <algoritmo>_reach_radio<R>_periodo<P>")
    parser.add_argument("--algoritmo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--radio", type=float, default=1.5)
    parser.add_argument("--dispersion", type=float, default=30.0)
    parser.add_argument("--inicio-aleatorio", action="store_true")
    parser.add_argument("--delta", type=float, default=None,
                        help="giro máximo uniforme por paso (grados); si no se indica, velocidades reales x --periodo")
    parser.add_argument("--periodo", type=float, default=0.5,
                        help="duración de cada paso del simulador, en segundos")
    parser.add_argument("--episodios", type=int, default=100)
    args = parser.parse_args()

    nombre = args.modelo or nombre_modelo(args.algoritmo, args.radio, args.delta, args.periodo)
    algoritmo = SAC if args.algoritmo == "sac" else PPO
    model = algoritmo.load(nombre)
    spread = None if args.inicio_aleatorio else args.dispersion
    env = Reach6DOFEnv(success_radius=args.radio, start_spread_deg=spread,
                       max_delta_deg=args.delta, control_period_s=args.periodo)

    finales, minimas, iniciales, trayectorias = [], [], [], []
    for ep in range(args.episodios):
        obs, info = env.reset(seed=1000 + ep)
        iniciales.append(info["distance"])
        dists = [info["distance"]]
        terminado = truncado = False
        while not (terminado or truncado):
            accion, _ = model.predict(obs, deterministic=True)
            obs, _, terminado, truncado, info = env.step(accion)
            dists.append(info["distance"])
        finales.append(dists[-1])
        minimas.append(min(dists))
        trayectorias.append(dists)

    finales, minimas, iniciales = map(np.array, (finales, minimas, iniciales))
    exito = np.mean(minimas < args.radio)

    print(f"\nModelo: {nombre}.zip   Episodios: {args.episodios}")
    print(f"Éxito (llegó a menos de {args.radio:g} cm):  {exito:.0%}")
    print(f"Distancia inicial media:                 {iniciales.mean():6.1f} cm")
    print(f"Distancia final media:                   {finales.mean():6.1f} cm")
    print(f"Distancia MÍNIMA alcanzada (media):      {minimas.mean():6.1f} cm  "
          f"<- si es mucho menor que la final, pasa cerca pero no se queda")
    fallos = finales[minimas >= args.radio]
    if len(fallos):
        print(f"Distancia final media SOLO en los fallos: {fallos.mean():5.1f} cm")
    print("\nPorcentaje de episodios cuya punta llegó, en algún momento, a menos de:")
    for limite in sorted({args.radio, 5, 10, 20}):
        print(f"  {limite:>5.1f} cm: {np.mean(minimas < limite):4.0%}")

    print("\nDistancia al objetivo cada 20 pasos (3 episodios):")
    for k in range(3):
        d = trayectorias[k]
        print("  " + " ".join(f"{x:5.1f}" for x in d[::20]))


if __name__ == "__main__":
    main()
