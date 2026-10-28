"""
lector_encoders.py

Lee las cuentas de encoder que manda el Arduino monitor (monitor_encoders_brazo.ino),
las convierte a ángulos de articulación y, con el modelo del brazo
(brazo_cinematica.py), calcula dónde debería estar la punta.

Sirve para VALIDAR el modelo contra el brazo real mientras lo mueves con la
pantalla (HMI), sin mandarle ningún comando desde Python.

Uso:
    python lector_encoders.py --puerto COM4                 # Mega: articulaciones 1 a 5
    python lector_encoders.py --puerto COM4 --activas 4,5   # Mega con solo J4 y J5 conectadas
    python lector_encoders.py --puerto COM3 --articulaciones 4,5   # Uno con 2 encoders
    python lector_encoders.py --simular                     # prueba sin hardware

Teclas (escribe y presiona Enter):
    Enter vacío   registra la pose actual y muestra la posición de la punta que
                  predice el modelo, y su distancia a la pose registrada antes
    z             pone en cero los contadores (hazlo con el brazo en la pose home)
    q             salir

Validación sin sistema de coordenadas: registra dos poses, marca con lápiz sobre
una hoja la posición de la punta en cada una, mide con regla la distancia entre
las dos marcas y compárala con la distancia que imprime el modelo.

Requiere: pip install numpy pyserial   (pyserial no hace falta con --simular)
Archivo en la misma carpeta: brazo_cinematica.py
"""

import argparse
import csv
import math
import queue
import sys
import threading
import time

import numpy as np

from brazo_cinematica import end_effector_position, NUM_JOINTS

# --- Datos del brazo ------------------------------------------------------
# Pulsos por vuelta de cada encoder (por canal). Las articulaciones 4 y 5 (NEMA 17)
# son 1000 PPR según la hoja de datos de la tesis. Para 1 a 3 (NEMA 23) es un
# SUPUESTO: confírmalo en la etiqueta u hoja de datos del motor.
PPR = {1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000}

# Reducción total motor -> articulación (tesis, Tabla 11).
REDUCCION = {
    1: 25 * 1.5,        # 1:25 con piñones 16 -> 24
    2: 50.0,            # 1:50
    3: 25 * 22 / 14,    # 1:25 con piñones 14 -> 22
    4: 50.0,            # 1:50
    5: 13.7,            # 1:13.7
}

# Si al mover una articulación en sentido positivo el ángulo baja, cambia su signo a -1.
SIGNO = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

# El Arduino cuenta en ambos flancos del canal A (decodificación x2).
FACTOR_DECODIFICACION = 2


def cuentas_a_grados(cuentas, articulaciones):
    """Convierte cuentas del encoder (del motor) en grados de cada articulación.

    cuentas: lista con una cuenta por canal. articulaciones: número de articulación
    conectada a cada canal. Devuelve {articulacion: grados}.
    """
    grados = {}
    for cuenta, j in zip(cuentas, articulaciones):
        vueltas_motor = cuenta / (FACTOR_DECODIFICACION * PPR[j])
        grados[j] = SIGNO[j] * vueltas_motor * 360.0 / REDUCCION[j]
    return grados


def posicion_punta(grados):
    """Posición (x, y, z) de la punta según el modelo. Las articulaciones que no
    se leen se consideran en 0 (la 6, un servo sin encoder, no cambia la posición
    de la punta, solo su orientación)."""
    q = np.zeros(NUM_JOINTS)
    for j, g in grados.items():
        q[j - 1] = math.radians(g)
    return end_effector_position(q)


def parsear_linea(linea, n_canales):
    """'E,<ms>,<c1>,...' -> (ms, [cuentas]) o None si la línea no es válida."""
    partes = linea.strip().split(",")
    if len(partes) != n_canales + 2 or partes[0] != "E":
        return None
    try:
        return int(partes[1]), [int(p) for p in partes[2:]]
    except ValueError:
        return None


class SerialSimulado:
    """Imita al Arduino monitor para probar este script sin hardware."""

    def __init__(self, n_canales):
        self.n = n_canales
        self.t0 = time.time()
        self.offset = [0] * n_canales

    def _cuentas_reales(self):
        t = time.time() - self.t0
        return [int(3000 * math.sin(0.8 * t + k)) for k in range(self.n)]

    def readline(self):
        time.sleep(0.02)
        ms = int((time.time() - self.t0) * 1000)
        cuentas = [c - o for c, o in zip(self._cuentas_reales(), self.offset)]
        return ("E,%d,%s\n" % (ms, ",".join(map(str, cuentas)))).encode()

    def write(self, datos):
        if b"Z" in datos:
            self.offset = self._cuentas_reales()

    def close(self):
        pass


def hilo_teclado(cola):
    """Lee lo que escribes y lo manda a la cola de eventos."""
    for linea in sys.stdin:
        cola.put(linea.strip().lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--puerto", type=str, default=None, help="por ejemplo COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--articulaciones", type=str, default="1,2,3,4,5",
                        help="articulación conectada a cada canal, en orden")
    parser.add_argument("--activas", type=str, default=None,
                        help="articulaciones realmente conectadas (el resto se ignora); "
                             "por defecto, todas las de --articulaciones")
    parser.add_argument("--poses", type=str, default="poses_validacion.csv",
                        help="archivo donde se guardan las poses registradas")
    parser.add_argument("--registro", type=str, default=None,
                        help="si se indica, guarda todas las lecturas en este CSV")
    parser.add_argument("--cero", action="store_true",
                        help="poner los contadores en cero al conectar")
    parser.add_argument("--simular", action="store_true", help="probar sin hardware")
    parser.add_argument("--duracion", type=float, default=0.0,
                        help="segundos antes de salir (0 = hasta que escribas q)")
    args = parser.parse_args()

    articulaciones = [int(a) for a in args.articulaciones.split(",")]
    for j in articulaciones:
        if j not in PPR:
            sys.exit(f"Articulación {j} no válida: solo 1 a 5 tienen encoder.")
    n = len(articulaciones)
    activas = [int(a) for a in args.activas.split(",")] if args.activas else list(articulaciones)
    for j in activas:
        if j not in articulaciones:
            sys.exit(f"La articulación {j} de --activas no está en --articulaciones.")

    if args.simular:
        conexion = SerialSimulado(n)
    else:
        if not args.puerto:
            sys.exit("Indica el puerto con --puerto (por ejemplo --puerto COM4) o usa --simular.")
        import serial  # solo se necesita con hardware real
        conexion = serial.Serial(args.puerto, args.baud, timeout=1)
        time.sleep(2)  # el Arduino se reinicia al abrir el puerto
        conexion.reset_input_buffer()
    if args.cero:
        conexion.write(b"Z")

    eventos = queue.Queue()
    threading.Thread(target=hilo_teclado, args=(eventos,), daemon=True).start()

    archivo_registro = None
    escritor_registro = None
    if args.registro:
        archivo_registro = open(args.registro, "w", newline="")
        escritor_registro = csv.writer(archivo_registro)
        escritor_registro.writerow(
            ["ms"] + [f"cuenta_J{j}" for j in articulaciones]
            + [f"grados_J{j}" for j in activas] + ["x", "y", "z"])

    poses = []
    ultima = None  # última lectura válida: (ms, cuentas, grados, punta)
    ultima_impresion = 0.0
    inicio = time.time()
    print("Enter = registrar pose | z = poner en cero | q = salir")

    try:
        while True:
            if args.duracion and time.time() - inicio > args.duracion:
                break

            # Teclas
            while not eventos.empty():
                tecla = eventos.get()
                if tecla == "q":
                    raise KeyboardInterrupt
                if tecla == "z":
                    conexion.write(b"Z")
                    print("\nContadores en cero.")
                elif tecla == "" and ultima is not None:
                    ms, cuentas, grados, punta = ultima
                    fila = [len(poses) + 1, ms] + [round(grados[j], 3) for j in activas] \
                        + [round(float(v), 3) for v in punta]
                    poses.append(punta)
                    texto = "  ".join(f"J{j}={grados[j]:7.2f}°" for j in activas)
                    print(f"\nPose {len(poses)}: {texto}")
                    print(f"  Punta según el modelo: x={punta[0]:.1f}  y={punta[1]:.1f}  z={punta[2]:.1f}")
                    if len(poses) >= 2:
                        d = float(np.linalg.norm(poses[-1] - poses[-2]))
                        print(f"  Distancia a la pose anterior según el modelo: {d:.1f}")
                        print("  -> compárala con la distancia medida con regla entre las marcas")
                    nuevo = not _existe(args.poses)
                    with open(args.poses, "a", newline="") as f:
                        w = csv.writer(f)
                        if nuevo:
                            w.writerow(["pose", "ms"] + [f"J{j}_grados" for j in activas]
                                       + ["x", "y", "z"])
                        w.writerow(fila)

            # Lectura del Arduino
            crudo = conexion.readline().decode("utf-8", errors="ignore")
            datos = parsear_linea(crudo, n)
            if datos is None:
                continue
            ms, cuentas = datos
            grados = cuentas_a_grados(cuentas, articulaciones)
            grados = {j: g for j, g in grados.items() if j in activas}  # ignora canales sin conectar
            punta = posicion_punta(grados)
            ultima = (ms, cuentas, grados, punta)

            if escritor_registro:
                escritor_registro.writerow([ms] + cuentas
                                           + [round(grados[j], 3) for j in activas]
                                           + [round(float(v), 3) for v in punta])

            if time.time() - ultima_impresion > 0.2:
                ultima_impresion = time.time()
                texto = "  ".join(f"J{j}={grados[j]:7.2f}°" for j in activas)
                print(f"\r{texto}  | punta=({punta[0]:6.1f},{punta[1]:6.1f},{punta[2]:6.1f})", end="")
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        conexion.close()
        if archivo_registro:
            archivo_registro.close()
        print(f"\nTerminado. Poses registradas: {len(poses)} (archivo {args.poses})")


def _existe(ruta):
    import os
    return os.path.exists(ruta)


if __name__ == "__main__":
    main()
