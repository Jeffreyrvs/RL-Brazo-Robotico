"""
serial_to_csv_encoder.py
-------------------------
Adaptación de tu script de captura para el Arduino monitor de encoders que
usamos (monitor_encoders_brazo.ino). Lee su stream y lo guarda en un .csv con
el MISMO formato largo que tenías (una fila por articulación y por muestra),
más el ángulo de articulación y su velocidad.

Qué cambia respecto a tu versión original:
  - El sketch manda una línea por muestra (cada 20 ms) con las cuentas de todos
    los encoders:   E,<milisegundos>,<cuenta1>,<cuenta2>,...
    (no eventos de pulso con dirección). Por eso la dirección y la velocidad se
    CALCULAN aquí, con la diferencia entre muestras consecutivas.
  - Solo existe la fuente 'E' (posición real del encoder). La fuente 'C'
    (comandada) no está disponible con este sketch, porque no lee las líneas
    PUL/DIR del brazo.
  - driver_id es el número de articulación (1 a 5).
  - Columnas nuevas al final: angle_deg (ángulo de la articulación) y
    velocity_dps (grados por segundo).

Columnas del CSV:
    timestamp_us   tiempo del Arduino en microsegundos (resolución real: 1 ms)
    driver_id      articulación (1 a 5)
    source         siempre 'E'
    direction      -1, 0 o +1 según cómo cambió la cuenta desde la muestra anterior
    position       cuenta del encoder (relativa al último cero)
    dt_us          tiempo desde la muestra anterior de esa articulación
    velocity_pps   cuentas por segundo, con signo (cuentas = x2 de los pulsos del encoder)
    angle_deg      ángulo de la articulación (usa PPR, REDUCCION y SIGNO de lector_encoders.py)
    velocity_dps   grados por segundo de la articulación

Requisitos:
    pip install pyserial          (no hace falta con --simular)
    Archivos en la misma carpeta: lector_encoders.py y brazo_cinematica.py

Uso:
    python serial_to_csv_encoder.py --port COM5 --cero
    python serial_to_csv_encoder.py --port COM5 --activas 4,5
    python serial_to_csv_encoder.py --simular --duracion 5      # prueba sin hardware
"""

import argparse
import csv
import datetime
import sys
import time

from lector_encoders import SerialSimulado, parsear_linea, cuentas_a_grados, PPR

ENCABEZADO = ["timestamp_us", "driver_id", "source", "direction", "position",
              "dt_us", "velocity_pps", "angle_deg", "velocity_dps"]


def main():
    parser = argparse.ArgumentParser(
        description="Captura CSV de los encoders del brazo desde el Arduino monitor")
    parser.add_argument("--port", default=None, help="Puerto serie, ej: COM5 o /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (debe coincidir con el sketch: 115200)")
    parser.add_argument("--output", default=None, help="Nombre del archivo .csv de salida (opcional)")
    parser.add_argument("--articulaciones", default="1,2,3,4,5",
                        help="articulación conectada a cada canal del sketch, en orden "
                             "(Mega: 1,2,3,4,5; Uno: por ejemplo 4,5)")
    parser.add_argument("--activas", default=None,
                        help="articulaciones realmente conectadas (el resto no se guarda); "
                             "por defecto, todas las de --articulaciones")
    parser.add_argument("--cero", action="store_true",
                        help="poner los contadores en cero al conectar (con el brazo en home)")
    parser.add_argument("--solo-cambios", action="store_true",
                        help="guardar solo las filas en que la cuenta cambió (más liviano)")
    parser.add_argument("--simular", action="store_true", help="probar sin hardware")
    parser.add_argument("--duracion", type=float, default=0.0,
                        help="segundos antes de detenerse (0 = hasta Ctrl+C)")
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

    if args.output:
        out_path = args.output
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"pulsos_encoder_{stamp}.csv"

    if args.simular:
        print("Modo simulado (sin hardware).")
        ser = SerialSimulado(n)
    else:
        if not args.port:
            sys.exit("Indica el puerto con --port (por ejemplo --port COM5) o usa --simular.")
        import serial  # solo se necesita con hardware real
        print(f"Conectando a {args.port} @ {args.baud} baudios...")
        try:
            ser = serial.Serial(args.port, args.baud, timeout=1)
        except serial.SerialException as e:
            print(f"No se pudo abrir el puerto: {e}")
            sys.exit(1)
        time.sleep(2)  # el Arduino se reinicia al abrir el puerto
        ser.reset_input_buffer()

    if args.cero:
        ser.write(b"Z")
        print("Contadores puestos en cero.")

    print(f"Guardando en: {out_path}")
    print("Ctrl+C para detener.\n")

    # última muestra por articulación: (t_us, cuenta, grados)
    ultima = {}
    row_count = 0
    inicio = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ENCABEZADO)

        try:
            while True:
                if args.duracion and time.time() - inicio > args.duracion:
                    break

                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue

                if raw.startswith("#"):
                    print(raw)
                    continue

                datos = parsear_linea(raw, n)
                if datos is None:
                    continue
                ms, cuentas = datos
                t_us = ms * 1000
                grados = cuentas_a_grados(cuentas, articulaciones)

                for j, cuenta in zip(articulaciones, cuentas):
                    if j not in activas:
                        continue
                    g = grados[j]
                    previa = ultima.get(j)
                    if previa is not None and t_us < previa[0]:
                        previa = None  # el Arduino se reinició: se pierde la referencia

                    dt_us = vel_pps = vel_dps = ""
                    direction = 0
                    if previa is not None:
                        dt = t_us - previa[0]
                        if dt > 0:
                            delta = cuenta - previa[1]
                            direction = (delta > 0) - (delta < 0)
                            dt_us = dt
                            vel_pps = round(delta * 1_000_000 / dt, 3)
                            vel_dps = round((g - previa[2]) * 1_000_000 / dt, 4)
                    ultima[j] = (t_us, cuenta, g)

                    if args.solo_cambios and previa is not None and direction == 0:
                        continue

                    writer.writerow([t_us, j, "E", direction, cuenta, dt_us,
                                     vel_pps, round(g, 4), vel_dps])
                    row_count += 1

                if row_count and row_count % 200 == 0:
                    f.flush()
                    print(f"\r{row_count} filas guardadas...", end="", flush=True)

        except KeyboardInterrupt:
            print(f"\n\nDetenido por el usuario.")
        finally:
            f.flush()
            ser.close()

    print(f"\nTotal de filas guardadas: {row_count}  ({out_path})")
    print("\nTip: en pandas, df[df.driver_id == 4] separa una articulación; con")
    print("df.pivot(index='timestamp_us', columns='driver_id', values='angle_deg')")
    print("obtienes los ángulos de todas las articulaciones en una misma base de tiempo.")


if __name__ == "__main__":
    main()
