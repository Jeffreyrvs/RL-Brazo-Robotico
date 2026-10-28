"""
train_ppo.py

Entrena un agente (PPO o SAC) en Reach6DOFEnv, lo evalúa y lo guarda.

Uso:
    python train_ppo.py                              # PPO, 500 mil pasos, dispersión de inicio 30°
    python train_ppo.py --steps 1500000              # PPO con entrenamiento más largo
    python train_ppo.py --algoritmo sac --steps 200000   # SAC (suele aprender con menos pasos)
    python train_ppo.py --dispersion 60              # tarea más difícil
    python train_ppo.py --inicio-aleatorio           # versión más difícil: inicio totalmente aleatorio
    python train_ppo.py --radio 5                    # aceptar el objetivo a menos de 5 cm (más fácil)
    python train_ppo.py --delta 1                    # pasos uniformes de 1° (ignora las velocidades reales)
    python train_ppo.py --periodo 1                  # cada paso dura 1 s (el brazo puede girar el doble por paso)

Currículo de tolerancia (empezar fácil y apretar poco a poco):
    python train_ppo.py --radio 5
    python train_ppo.py --radio 3   --continuar ppo_reach_radio5_periodo0.5
    python train_ppo.py --radio 1.5 --continuar ppo_reach_radio3_periodo0.5

Cada corrida guarda su modelo con un nombre que incluye el algoritmo, el radio y el periodo (o el delta),
para que una corrida no sobrescriba a otra. Usa --salida para poner otro nombre.

Requiere: pip install numpy gymnasium stable-baselines3
Archivos en la misma carpeta: dh_6gdl.py, reach_env.py
"""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from reach_env import Reach6DOFEnv, nombre_modelo


def evaluar(politica, env, episodios=100, seed=1000):
    """Devuelve (tasa_de_exito, distancia_final_media, pasos_medios).

    politica: función que recibe la observación y devuelve una acción.
    """
    exitos, distancias, pasos = [], [], []
    for ep in range(episodios):
        obs, info = env.reset(seed=seed + ep)
        terminado = truncado = False
        n = 0
        while not (terminado or truncado):
            obs, _, terminado, truncado, info = env.step(politica(obs))
            n += 1
        exitos.append(info["is_success"])
        distancias.append(info["distance"])
        pasos.append(n)
    return float(np.mean(exitos)), float(np.mean(distancias)), float(np.mean(pasos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algoritmo", choices=["ppo", "sac"], default="ppo",
                        help="algoritmo de RL a usar")
    parser.add_argument("--steps", type=int, default=500_000, help="pasos de entrenamiento")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--salida", type=str, default=None,
                        help="nombre del modelo guardado (por defecto <algoritmo>_reach_radio<R>_periodo<P>)")
    parser.add_argument("--continuar", type=str, default=None,
                        help="nombre (sin .zip) de un modelo guardado desde el cual seguir entrenando")
    parser.add_argument("--radio", type=float, default=1.5,
                        help="distancia (cm) para dar el objetivo por alcanzado")
    parser.add_argument("--dispersion", type=float, default=30.0,
                        help="grados por articulación que el inicio se aleja de una pose que alcanza el objetivo")
    parser.add_argument("--inicio-aleatorio", action="store_true",
                        help="inicio totalmente aleatorio (ignora --dispersion)")
    parser.add_argument("--delta", type=float, default=None,
                        help="giro máximo uniforme por paso (grados); si no se indica, se usan las "
                             "velocidades reales de cada articulación x --periodo")
    parser.add_argument("--periodo", type=float, default=0.5,
                        help="duración de cada paso del simulador, en segundos")
    args = parser.parse_args()

    salida = args.salida or nombre_modelo(args.algoritmo, args.radio, args.delta, args.periodo)
    spread = None if args.inicio_aleatorio else args.dispersion
    env = Monitor(Reach6DOFEnv(success_radius=args.radio, start_spread_deg=spread,
                               max_delta_deg=args.delta, control_period_s=args.periodo))
    eval_env = Reach6DOFEnv(success_radius=args.radio, start_spread_deg=spread,
                            max_delta_deg=args.delta, control_period_s=args.periodo)

    # Línea base: un agente que actúa al azar. El agente entrenado debe
    # superarlo claramente; si no, algo está mal.
    azar = lambda obs: eval_env.action_space.sample()
    exito, dist, pasos = evaluar(azar, eval_env)
    print(f"[Agente aleatorio]  éxito={exito:.0%}  distancia final={dist:.1f} cm  pasos={pasos:.0f}")

    algoritmo = SAC if args.algoritmo == "sac" else PPO
    if args.continuar:
        model = algoritmo.load(args.continuar, env=env)
        # Evaluación de partida: cómo le va al modelo cargado con las nuevas reglas
        punto_de_partida = lambda obs: model.predict(obs, deterministic=True)[0]
        exito, dist, pasos = evaluar(punto_de_partida, eval_env)
        print(f"[Modelo cargado: {args.continuar}, antes de seguir]  éxito={exito:.0%}  "
              f"distancia final={dist:.1f} cm  pasos={pasos:.0f}")
    else:
        model = algoritmo("MlpPolicy", env, seed=args.seed, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=not args.continuar)
    print(f"Entrenamiento terminado en {(time.time() - t0) / 60:.1f} min")

    entrenado = lambda obs: model.predict(obs, deterministic=True)[0]
    exito, dist, pasos = evaluar(entrenado, eval_env)
    print(f"[Agente entrenado ({args.algoritmo.upper()})]  éxito={exito:.0%}  "
          f"distancia final={dist:.1f} cm  pasos={pasos:.0f}")

    model.save(salida)
    print(f"Modelo guardado en {salida}.zip")


if __name__ == "__main__":
    main()
