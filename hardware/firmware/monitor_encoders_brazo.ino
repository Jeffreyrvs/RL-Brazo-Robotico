/*
  monitor_encoders_brazo.ino

  Segundo Arduino SOLO DE LECTURA: escucha en paralelo las señales de encoder
  (canales A y B) de los motores del brazo y cuenta pulsos con decodificación
  de cuadratura. No genera ninguna señal hacia el brazo. NO se carga en el
  Arduino Mega original del brazo.

  Cableado (por cada encoder; con el sistema APAGADO y SOLO si medí <= 5 V):
    EA+ del driver --[resistencia 1 kΩ]--> pin de canal A de este Arduino
    EB+ del driver --[resistencia 1 kΩ]--> pin de canal B de este Arduino
    EGND del driver ---------------------> GND de este Arduino
  NO conectar VCC del encoder al Arduino. Este Arduino se alimenta por USB.

  Pines:
    Arduino Mega 2560 (5 encoders, articulaciones 1 a 5):
        canal A: 2, 3, 18, 19, 20   (pines con interrupción)
        canal B: 22, 23, 24, 25, 26
    Arduino Uno/Nano (solo 2 encoders a la vez):
        canal A: 2, 3
        canal B: 4, 5

  Resolución: se cuenta en ambos flancos del canal A (x2), es decir
  2 x PPR cuentas por vuelta del MOTOR.

  Protocolo serial (115200 baudios):
    Arduino -> PC, cada 20 ms:  "E,<ms>,<cuenta1>,<cuenta2>,...\n"
    PC -> Arduino:              'Z'  pone en cero todos los contadores
                                (hacerlo con el brazo en la pose home)
*/

#if defined(__AVR_ATmega2560__)
  #define NUM_ENC 5
  const uint8_t pinA[NUM_ENC] = {2, 3, 18, 19, 20};
  const uint8_t pinB[NUM_ENC] = {22, 23, 24, 25, 26};
#else
  #define NUM_ENC 2
  const uint8_t pinA[NUM_ENC] = {2, 3};
  const uint8_t pinB[NUM_ENC] = {4, 5};
#endif

void leerComandos();
void reportarEstado();

volatile long cuenta[NUM_ENC];

// Registros y máscaras precalculados: leer el pin dentro de la interrupción
// con estos punteros es mucho más rápido que digitalRead().
volatile uint8_t* regA[NUM_ENC];
volatile uint8_t* regB[NUM_ENC];
uint8_t maskA[NUM_ENC];
uint8_t maskB[NUM_ENC];

unsigned long ultimoReporte = 0;
const unsigned long INTERVALO_REPORTE_MS = 20;

// Se ejecuta en cada cambio del canal A del encoder i.
// Si A y B tienen el mismo nivel, el giro es en un sentido; si no, en el otro.
static inline void decodificar(uint8_t i) {
  bool a = (*regA[i] & maskA[i]) != 0;
  bool b = (*regB[i] & maskB[i]) != 0;
  if (a == b) {
    cuenta[i]++;
  } else {
    cuenta[i]--;
  }
}

// Una función de interrupción por encoder (el hardware no permite pasarle el índice).
void isr0() { decodificar(0); }
void isr1() { decodificar(1); }
#if NUM_ENC >= 5
void isr2() { decodificar(2); }
void isr3() { decodificar(3); }
void isr4() { decodificar(4); }
#endif

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_ENC; i++) {
    pinMode(pinA[i], INPUT_PULLUP);   // evita lecturas ruidosas en canales sin conectar
    pinMode(pinB[i], INPUT_PULLUP);
    regA[i] = portInputRegister(digitalPinToPort(pinA[i]));
    maskA[i] = digitalPinToBitMask(pinA[i]);
    regB[i] = portInputRegister(digitalPinToPort(pinB[i]));
    maskB[i] = digitalPinToBitMask(pinB[i]);
    cuenta[i] = 0;
  }

  attachInterrupt(digitalPinToInterrupt(pinA[0]), isr0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pinA[1]), isr1, CHANGE);
#if NUM_ENC >= 5
  attachInterrupt(digitalPinToInterrupt(pinA[2]), isr2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pinA[3]), isr3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pinA[4]), isr4, CHANGE);
#endif
}

void loop() {
  leerComandos();
  reportarEstado();
}

void leerComandos() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'Z') {
      noInterrupts();
      for (uint8_t i = 0; i < NUM_ENC; i++) cuenta[i] = 0;
      interrupts();
    }
  }
}

void reportarEstado() {
  unsigned long ahora = millis();
  if (ahora - ultimoReporte < INTERVALO_REPORTE_MS) return;
  ultimoReporte = ahora;

  // Copia de los contadores con interrupciones apagadas un instante,
  // para no leer un valor a medio actualizar.
  long copia[NUM_ENC];
  noInterrupts();
  for (uint8_t i = 0; i < NUM_ENC; i++) copia[i] = cuenta[i];
  interrupts();

  Serial.print("E,");
  Serial.print(ahora);
  for (uint8_t i = 0; i < NUM_ENC; i++) {
    Serial.print(',');
    Serial.print(copia[i]);
  }
  Serial.println();
}
