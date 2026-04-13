"""
UPDRS Parte 3 — Analisis Cuantitativo de Movimientos
=====================================================
Fusion de ESP32-CAM (vision con MediaPipe) + MPU6050 (inercial via I2C)

5 Variables cuantitativas:
  1. Frecuencia    FFT de magnitud de aceleracion (Hz)
  2. Amplitud      Rango de distancia pulgar-indice desde vision (normalizado)
  3. Vel. angular  RMS del giroscopio (deg/s)
  4. Regularidad   Coeficiente de variacion de intervalos entre taps (%)
  5. Jerk          Derivada discreta de aceleracion: |a[i+1]-a[i]|/dt (g/s)

Correspondencia cualitativa UPDRS:
  Frecuencia baja   -> bradicinesia
  Amplitud baja     -> hipocinesia
  Omega baja        -> lentitud
  CV alto           -> irregularidad
  Jerk alto         -> falta de control

Normalizacion -> ponderacion -> indice compuesto -> mapeo UPDRS 0-4

Layout de ventana: video (800x600) + panel lateral (300 px)
"""

import cv2
import numpy as np
import requests
import mediapipe as mp
import time
import threading
from collections import deque


# =====================================================================
#  CONFIGURACION
# =====================================================================
ESP32_IP   = "192.168.4.1"
FRAME_URL  = f"http://{ESP32_IP}/frame"
SENSOR_URL = f"http://{ESP32_IP}/sensor"

VIDEO_W, VIDEO_H = 800, 600       # area de video
PANEL_W          = 300             # ancho panel lateral
CANVAS_W         = VIDEO_W + PANEL_W   # 1100 px total


# =====================================================================
#  MEDIAPIPE
# =====================================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =====================================================================
#  CONSTANTES VISUALES
# =====================================================================
RIGHT_COLORS = {
    'THUMB': (0, 0, 255),     'INDEX': (0, 255, 255),
    'MIDDLE': (0, 255, 0),    'RING': (255, 0, 0),
    'PINKY': (255, 0, 255)
}
LEFT_COLORS = {
    'THUMB': (0, 128, 255),   'INDEX': (128, 255, 255),
    'MIDDLE': (128, 255, 128),'RING': (255, 128, 128),
    'PINKY': (255, 128, 255)
}
FINGER_LANDMARKS = {
    'THUMB': [1,2,3,4],  'INDEX': [5,6,7,8],  'MIDDLE': [9,10,11,12],
    'RING': [13,14,15,16], 'PINKY': [17,18,19,20]
}
FINGER_CONNECTIONS = {
    'THUMB': [(1,2),(2,3),(3,4)],     'INDEX': [(5,6),(6,7),(7,8)],
    'MIDDLE': [(9,10),(10,11),(11,12)],'RING': [(13,14),(14,15),(15,16)],
    'PINKY': [(17,18),(18,19),(19,20)]
}
PALM_CONNECTIONS = [(0,1),(1,5),(5,9),(9,13),(13,17),(0,17)]


# =====================================================================
#  SIGNAL PROCESSOR  —  Buffers + calculo de las 5 variables
# =====================================================================
class SignalProcessor:
    """
    Almacena muestras del IMU y de vision, y calcula
    las 5 variables cuantitativas para el scoring UPDRS.
    """

    def __init__(self, max_samples=512):
        self._max = max_samples
        # --- Buffers IMU ---
        self.ax    = deque(maxlen=max_samples)
        self.ay    = deque(maxlen=max_samples)
        self.az    = deque(maxlen=max_samples)
        self.gx    = deque(maxlen=max_samples)
        self.gy    = deque(maxlen=max_samples)
        self.gz    = deque(maxlen=max_samples)
        self.imu_t = deque(maxlen=max_samples)   # ms del ESP32

        # --- Buffers Vision ---
        self.thumb_idx_dist = deque(maxlen=max_samples)
        self.tap_intervals  = deque(maxlen=100)

    # ----- Ingesta de datos -----

    def add_imu_samples(self, samples):
        """
        Agrega muestras del IMU al buffer.
        Cada muestra: [ax, ay, az, gx, gy, gz, ms]
        """
        for s in samples:
            if len(s) < 7:
                continue
            self.ax.append(s[0]);  self.ay.append(s[1]);  self.az.append(s[2])
            self.gx.append(s[3]);  self.gy.append(s[4]);  self.gz.append(s[5])
            self.imu_t.append(s[6])

    def add_vision_sample(self, thumb_index_normalized):
        """Agrega distancia pulgar-indice normalizada por tamano de mano."""
        self.thumb_idx_dist.append(thumb_index_normalized)

    def add_tap_interval(self, interval_sec):
        """Agrega intervalo entre taps consecutivos (segundos)."""
        self.tap_intervals.append(interval_sec)

    # ----- Variable 1: FRECUENCIA (FFT) -----

    def compute_frequency(self):
        """
        Frecuencia dominante del movimiento via FFT.

        Procedimiento:
          1. Calcular magnitud de aceleracion |a| = sqrt(ax^2+ay^2+az^2)
          2. Remover componente DC (media)
          3. Aplicar ventana Hanning para reducir leakage espectral
          4. FFT y buscar pico en banda 0.5-8 Hz
             (bradicinesia se manifiesta en frecuencias bajas,
              tremor de reposo ~4-6 Hz, movimiento voluntario ~1-5 Hz)
          5. Estimar fs desde timestamps del ESP32

        Requiere minimo 64 muestras (~1.3 s a 50 Hz).
        Referencia: sano ~4-5 Hz, severo ~1 Hz
        """
        n = len(self.ax)
        if n < 64:
            return 0.0

        mag = np.sqrt(np.array(self.ax)**2 +
                      np.array(self.ay)**2 +
                      np.array(self.az)**2)
        mag = mag - np.mean(mag)  # remover DC

        # Frecuencia de muestreo estimada
        t_sec = np.array(self.imu_t) / 1000.0
        dt = np.diff(t_sec)
        dt = dt[dt > 0.001]
        if len(dt) == 0:
            return 0.0
        fs = 1.0 / np.median(dt)

        # FFT con ventana Hanning
        win = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(mag * win))
        freqs    = np.fft.rfftfreq(n, d=1.0 / fs)

        # Buscar pico en 0.5 — 8 Hz
        mask = (freqs >= 0.5) & (freqs <= 8.0)
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][np.argmax(spectrum[mask])])

    # ----- Variable 2: AMPLITUD (Vision) -----

    def compute_amplitude(self):
        """
        Amplitud del movimiento de golpeteo desde vision.

        Usa el rango intercuartil P95-P5 de la distancia pulgar-indice
        normalizada por el tamano de la mano.  Esto mide cuanto se abre
        y cierra la mano durante el ejercicio.

        Referencia: sano ~0.6-0.8, severo ~0.05-0.15
        """
        if len(self.thumb_idx_dist) < 20:
            return 0.0
        arr = np.array(self.thumb_idx_dist)
        return float(np.percentile(arr, 95) - np.percentile(arr, 5))

    # ----- Variable 3: VELOCIDAD ANGULAR (Giroscopio) -----

    def compute_angular_velocity(self):
        """
        RMS de la velocidad angular total.

        omega_rms = sqrt( mean( gx^2 + gy^2 + gz^2 ) )

        Mide la rapidez de rotacion del antebrazo/mano.
        Correlacion con severidad: r = -0.914 (literatura).

        Referencia: sano ~150-250 deg/s, severo ~20-40 deg/s
        """
        if len(self.gx) < 10:
            return 0.0
        g = np.array(self.gx)**2 + np.array(self.gy)**2 + np.array(self.gz)**2
        return float(np.sqrt(np.mean(g)))

    # ----- Variable 4: REGULARIDAD (CV) -----

    def compute_cv(self):
        """
        Coeficiente de variacion de intervalos entre taps.

        CV = (desviacion_estandar / media) * 100  [%]

        Mide la irregularidad del ritmo de golpeteo.
        Correlacion con UPDRS-III: R = 0.66 (literatura).

        Referencia: sano ~5-10%, severo ~40-60%
        """
        if len(self.tap_intervals) < 3:
            return 0.0
        arr = np.array(self.tap_intervals)
        mu = np.mean(arr)
        if mu < 0.01:
            return 0.0
        return float((np.std(arr) / mu) * 100.0)

    # ----- Variable 5: JERK (Suavidad) -----

    def compute_jerk(self):
        """
        Jerk medio absoluto: suavidad del movimiento.

        jerk[i] = |a_mag[i+1] - a_mag[i]| / dt[i]

        donde a_mag es la magnitud de aceleracion.
        Jerk alto = movimiento brusco, entrecortado.

        Referencia: sano ~2-5 g/s, severo ~25-40 g/s
        """
        n = len(self.ax)
        if n < 10:
            return 0.0

        mag = np.sqrt(np.array(self.ax)**2 +
                      np.array(self.ay)**2 +
                      np.array(self.az)**2)
        t_sec = np.array(self.imu_t) / 1000.0

        dt = np.diff(t_sec)
        da = np.abs(np.diff(mag))
        valid = dt > 0.001
        if not np.any(valid):
            return 0.0
        return float(np.mean(da[valid] / dt[valid]))

    # ----- Resumen -----

    def compute_all(self):
        """Retorna diccionario con las 5 variables calculadas."""
        return {
            'frequency':   self.compute_frequency(),
            'amplitude':   self.compute_amplitude(),
            'angular_vel': self.compute_angular_velocity(),
            'cv':          self.compute_cv(),
            'jerk':        self.compute_jerk(),
        }

    def reset(self):
        self.__init__(max_samples=self._max)


# =====================================================================
#  UPDRS SCORER — Normalizacion + Ponderacion + Mapeo 0-4
# =====================================================================
class UPDRSScorer:
    """
    Normaliza las 5 variables a [0,1] y combina con pesos
    para producir un indice compuesto mapeado a UPDRS 0-4.

    Normalizacion:
      Variables donde MENOR = PEOR (freq, amp, omega):
        score = clamp( (normal - valor) / (normal - severo), 0, 1 )
      Variables donde MAYOR = PEOR (CV, jerk):
        score = clamp( (valor - normal) / (severo - normal), 0, 1 )

      score = 0 -> normal,  score = 1 -> severo

    Indice compuesto = sum( peso_i * score_i )
    UPDRS = round( indice * 4 )  ->  0, 1, 2, 3, 4
    """

    # Rangos de referencia (basados en literatura, calibrar con datos propios)
    RANGES = {
        'frequency':   {'normal': 4.5,  'severe': 1.0},    # Hz
        'amplitude':   {'normal': 0.65, 'severe': 0.10},   # normalizado
        'angular_vel': {'normal': 180., 'severe': 30.},     # deg/s
        'cv':          {'normal': 8.,   'severe': 50.},     # %
        'jerk':        {'normal': 3.,   'severe': 35.},     # g/s
    }

    # Pesos segun importancia clinica (suman 1.0)
    WEIGHTS = {
        'frequency':   0.25,    # bradicinesia (clave)
        'amplitude':   0.20,    # hipocinesia
        'angular_vel': 0.15,    # lentitud
        'cv':          0.25,    # irregularidad (alta correlacion R=0.66)
        'jerk':        0.15,    # falta de control / suavidad
    }

    # Etiquetas y colores UPDRS
    LABELS = {0: "Normal", 1: "Leve", 2: "Moderado leve", 3: "Moderado", 4: "Severo"}
    COLORS = {
        0: (0, 200, 0),       # verde
        1: (0, 210, 170),     # verde-cyan
        2: (0, 210, 210),     # amarillo
        3: (0, 140, 255),     # naranja
        4: (0, 50, 255),      # rojo
    }

    def normalize_var(self, name, value):
        """Normaliza una variable a [0, 1].  0=normal, 1=severo."""
        r = self.RANGES[name]
        if name in ('cv', 'jerk'):    # mayor = peor
            score = (value - r['normal']) / (r['severe'] - r['normal'])
        else:                          # menor = peor
            score = (r['normal'] - value) / (r['normal'] - r['severe'])
        return max(0.0, min(1.0, score))

    def compute(self, metrics):
        """
        Calcula puntaje UPDRS a partir del dict de metricas.

        Retorna:
          individual : dict con score normalizado [0,1] por variable
          composite  : indice compuesto [0,1]
          updrs      : entero 0-4
          label      : texto descriptivo
        """
        individual = {}
        for name in self.WEIGHTS:
            individual[name] = self.normalize_var(name, metrics.get(name, 0.0))

        composite = sum(self.WEIGHTS[k] * individual[k] for k in self.WEIGHTS)
        composite = max(0.0, min(1.0, composite))

        updrs = int(round(composite * 4))
        updrs = max(0, min(4, updrs))

        return individual, composite, updrs, self.LABELS[updrs]


# =====================================================================
#  HAND TRACKER — Deteccion de ejercicios por mano (vision)
# =====================================================================
class HandTracker:
    """Tracking de golpeteo, apertura/cierre, y pronacion por mano."""

    def __init__(self):
        self.tap_count     = 0
        self.last_tap      = False
        self.last_tap_time = 0.0
        self.pron_count    = 0
        self.last_pron     = ""
        self.oc_count      = 0
        self.last_oc       = ""

    def update_tap(self, tapping, sig_proc):
        now = time.time()
        if tapping and not self.last_tap:
            self.tap_count += 1
            if self.last_tap_time > 0:
                interval = now - self.last_tap_time
                if 0.05 < interval < 3.0:   # filtrar intervalos absurdos
                    sig_proc.add_tap_interval(interval)
            self.last_tap_time = now
        self.last_tap = tapping

    def update_pron(self, state):
        if state != "Neutro" and state != self.last_pron and self.last_pron:
            self.pron_count += 1
        self.last_pron = state

    def update_oc(self, state):
        if state != self.last_oc and self.last_oc:
            self.oc_count += 1
        self.last_oc = state


# =====================================================================
#  SENSOR READER — Hilo dedicado para leer IMU sin bloquear video
# =====================================================================
class SensorReader(threading.Thread):
    """
    Hilo daemon que consulta /sensor del ESP32 en paralelo
    al loop de video, para no reducir FPS.
    """

    def __init__(self, url, interval=0.08):
        super().__init__(daemon=True)
        self.url      = url
        self.interval = interval
        self._lock    = threading.Lock()
        self._samples = []
        self.running  = True
        self.connected = False

    def run(self):
        while self.running:
            try:
                r = requests.get(self.url, timeout=1)
                if r.status_code == 200:
                    data = r.json().get('s', [])
                    with self._lock:
                        self._samples = data
                    self.connected = True
                else:
                    self.connected = False
            except Exception:
                self.connected = False
            time.sleep(self.interval)

    def get_samples(self):
        """Retorna y vacia las muestras acumuladas (thread-safe)."""
        with self._lock:
            s = self._samples
            self._samples = []
            return s

    def stop(self):
        self.running = False


# =====================================================================
#  INSTANCIAS GLOBALES
# =====================================================================
hand_trackers = {}
sig_proc      = SignalProcessor(max_samples=512)
scorer        = UPDRSScorer()
sensor_reader = SensorReader(SENSOR_URL, interval=0.08)


# =====================================================================
#  HELPERS
# =====================================================================
def get_frame(url):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def hand_scale(lms):
    """Distancia muneca -> MCP medio (referencia de tamano de mano)."""
    w, m = lms.landmark[0], lms.landmark[9]
    return np.hypot(w.x - m.x, w.y - m.y)


# =====================================================================
#  DETECCION DE EJERCICIOS
# =====================================================================
def detect_tapping(lms):
    """
    UPDRS 3.4  Golpeteo: contacto pulgar-indice.
    Umbral normalizado por tamano de mano.
    """
    thumb, index = lms.landmark[4], lms.landmark[8]
    dist = np.hypot(thumb.x - index.x, thumb.y - index.y)
    s = hand_scale(lms)
    return (dist / s) < 0.40 if s > 0.01 else False


def thumb_index_normalized(lms):
    """Distancia pulgar-indice / tamano mano  (para amplitud)."""
    thumb, index = lms.landmark[4], lms.landmark[8]
    dist = np.hypot(thumb.x - index.x, thumb.y - index.y)
    s = hand_scale(lms)
    return dist / s if s > 0.01 else 0.0


def detect_open_close(lms):
    """
    UPDRS 3.5  Mano abierta/cerrada.
    Cuenta 4 dedos (sin pulgar): tip.y < PIP.y = extendido.
    """
    tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
    ext = sum(1 for t, p in tips_pips if lms.landmark[t].y < lms.landmark[p].y)
    if ext >= 3:
        return "Abierta"
    if ext <= 1:
        return "Cerrada"
    return "Parcial"


def detect_pronation(lms, label):
    """
    UPDRS 3.6  Pronacion / Supinacion.

    Vector normal al plano de la palma via producto vectorial 3D:
      v1 = muneca -> MCP indice (landmark 5)
      v2 = muneca -> MCP menique (landmark 17)
      normal = v1 x v2

    El signo de normal.z indica hacia donde apunta la palma.
    Se invierte para mano izquierda.

    Movimiento del paciente:
      Extender brazo al frente, rotar antebrazo alternando:
      - Supinacion: palma hacia arriba / camara (gesto de "pedir monedas")
      - Pronacion:  palma hacia abajo (gesto de "echar sal")
    """
    wrist   = lms.landmark[0]
    idx_mcp = lms.landmark[5]
    pnk_mcp = lms.landmark[17]

    v1 = np.array([idx_mcp.x - wrist.x, idx_mcp.y - wrist.y, idx_mcp.z - wrist.z])
    v2 = np.array([pnk_mcp.x - wrist.x, pnk_mcp.y - wrist.y, pnk_mcp.z - wrist.z])
    nz = np.cross(v1, v2)[2]

    th = 0.006
    if label == "Right":
        if nz < -th:
            return "Supinacion"
        if nz > th:
            return "Pronacion"
    else:
        if nz > th:
            return "Supinacion"
        if nz < -th:
            return "Pronacion"
    return "Neutro"


# =====================================================================
#  FUNCIONES DE DIBUJO
# =====================================================================
def draw_landmarks(canvas, lms, label):
    """Dibuja esqueleto de la mano sobre el area de video."""
    colors = RIGHT_COLORS if label == "Right" else LEFT_COLORS

    for s_i, e_i in PALM_CONNECTIONS:
        s, e = lms.landmark[s_i], lms.landmark[e_i]
        cv2.line(canvas,
                 (int(s.x * VIDEO_W), int(s.y * VIDEO_H)),
                 (int(e.x * VIDEO_W), int(e.y * VIDEO_H)),
                 (180, 180, 180), 2)

    for fname, color in colors.items():
        for s_i, e_i in FINGER_CONNECTIONS[fname]:
            s, e = lms.landmark[s_i], lms.landmark[e_i]
            cv2.line(canvas,
                     (int(s.x * VIDEO_W), int(s.y * VIDEO_H)),
                     (int(e.x * VIDEO_W), int(e.y * VIDEO_H)),
                     color, 2)
        for li in FINGER_LANDMARKS[fname]:
            lm = lms.landmark[li]
            cv2.circle(canvas, (int(lm.x * VIDEO_W), int(lm.y * VIDEO_H)), 4, color, -1)

    wr = lms.landmark[0]
    cv2.circle(canvas, (int(wr.x * VIDEO_W), int(wr.y * VIDEO_H)), 6, (220, 220, 220), -1)
    cv2.putText(canvas, label,
                (max(int(wr.x * VIDEO_W) - 15, 2), max(int(wr.y * VIDEO_H) - 12, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bar(canvas, x, y, w, h, value, max_val, color):
    """Barra de progreso horizontal con fondo gris."""
    fill = int(w * min(value / max_val, 1.0)) if max_val > 0 else 0
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 55), -1)
    if fill > 0:
        cv2.rectangle(canvas, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 95), 1)


def draw_panel(canvas):
    """
    Panel lateral derecho con:
      - Estado por mano (golpeteo, apertura, rotacion)
      - 5 variables cuantitativas con barras
      - Score UPDRS compuesto con barra de color
    """
    x0   = VIDEO_W
    font = cv2.FONT_HERSHEY_SIMPLEX
    px   = x0 + 12
    bw   = PANEL_W - 40     # ancho de barras

    # Fondo del panel
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, VIDEO_H), (25, 25, 30), -1)
    cv2.line(canvas, (x0, 0), (x0, VIDEO_H), (70, 70, 80), 2)

    y = 24

    # ========== TITULO ==========
    cv2.putText(canvas, "UPDRS Parte 3", (px, y), font, 0.55,
                (255, 255, 120), 2, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, "Analisis Cuantitativo", (px, y), font, 0.32,
                (140, 140, 150), 1, cv2.LINE_AA)
    y += 10
    cv2.line(canvas, (px, y), (x0 + PANEL_W - 12, y), (50, 50, 60), 1)
    y += 14

    # ========== INFO POR MANO ==========
    for hlabel in ["Right", "Left"]:
        tk = hand_trackers.get(hlabel)
        if not tk:
            continue

        name = "Mano Der." if hlabel == "Right" else "Mano Izq."
        hdr  = (200, 200, 255) if hlabel == "Right" else (255, 200, 255)

        cv2.putText(canvas, name, (px, y), font, 0.43, hdr, 1, cv2.LINE_AA)
        y += 16

        # Golpeteo
        tc = (0, 255, 100) if tk.last_tap else (110, 110, 110)
        cv2.putText(canvas, f"Golpeteo: {'SI' if tk.last_tap else 'NO'}  Taps:{tk.tap_count}",
                    (px + 6, y), font, 0.31, tc, 1, cv2.LINE_AA)
        y += 14

        # Abrir/cerrar
        oc = tk.last_oc or "---"
        oc_c = {"Abierta": (0, 255, 160), "Cerrada": (50, 120, 255),
                "Parcial": (255, 210, 0)}.get(oc, (110, 110, 110))
        cv2.putText(canvas, f"Mano: {oc}  Cambios:{tk.oc_count}",
                    (px + 6, y), font, 0.31, oc_c, 1, cv2.LINE_AA)
        y += 14

        # Pronacion
        pr = tk.last_pron or "---"
        pr_c = {"Supinacion": (0, 220, 255), "Pronacion": (255, 160, 0),
                "Neutro": (110, 110, 110)}.get(pr, (110, 110, 110))
        cv2.putText(canvas, f"Rot: {pr}  Cambios:{tk.pron_count}",
                    (px + 6, y), font, 0.31, pr_c, 1, cv2.LINE_AA)
        y += 18

    cv2.line(canvas, (px, y), (x0 + PANEL_W - 12, y), (50, 50, 60), 1)
    y += 12

    # ========== 5 VARIABLES CUANTITATIVAS ==========
    metrics = sig_proc.compute_all()
    individual, composite, updrs, ulabel = scorer.compute(metrics)

    cv2.putText(canvas, "Variables Sensor+Vision", (px, y), font, 0.38,
                (200, 200, 120), 1, cv2.LINE_AA)
    y += 16

    # Descripcion de cada variable y su correspondencia UPDRS
    var_rows = [
        ("Frecuencia",  f"{metrics['frequency']:.1f} Hz",    'frequency',   "bradicinesia"),
        ("Amplitud",    f"{metrics['amplitude']:.2f}",       'amplitude',   "hipocinesia"),
        ("Vel.Angular", f"{metrics['angular_vel']:.1f} d/s", 'angular_vel', "lentitud"),
        ("CV",          f"{metrics['cv']:.1f} %",            'cv',          "irregularidad"),
        ("Jerk",        f"{metrics['jerk']:.1f} g/s",        'jerk',        "falta control"),
    ]

    for vname, vstr, vkey, vcorr in var_rows:
        vscore = individual.get(vkey, 0)
        # Color gradiente verde->rojo segun severidad
        r = int(vscore * 255)
        g = int((1 - vscore) * 190)
        bar_col = (0, g, r)

        cv2.putText(canvas, f"{vname}: {vstr}", (px + 4, y), font, 0.30,
                    (170, 170, 170), 1, cv2.LINE_AA)
        # Mostrar correspondencia UPDRS en gris
        tw = cv2.getTextSize(f"{vname}: {vstr}", font, 0.30, 1)[0][0]
        cv2.putText(canvas, vcorr, (px + 6 + tw + 4, y), font, 0.24,
                    (100, 100, 110), 1, cv2.LINE_AA)
        y += 10
        draw_bar(canvas, px + 4, y, bw, 7, vscore, 1.0, bar_col)
        y += 14

    y += 2
    cv2.line(canvas, (px, y), (x0 + PANEL_W - 12, y), (50, 50, 60), 1)
    y += 14

    # ========== SCORE UPDRS COMPUESTO ==========
    updrs_color = scorer.COLORS.get(updrs, (200, 200, 200))

    cv2.putText(canvas, "Indice UPDRS", (px, y), font, 0.42,
                (255, 255, 200), 1, cv2.LINE_AA)
    y += 20

    cv2.putText(canvas, f"{updrs} - {ulabel}", (px + 4, y), font, 0.50,
                updrs_color, 2, cv2.LINE_AA)
    y += 16

    draw_bar(canvas, px + 4, y, bw, 12, composite, 1.0, updrs_color)
    cv2.putText(canvas, f"{composite * 100:.0f}%", (px + bw + 8, y + 10),
                font, 0.30, (150, 150, 150), 1, cv2.LINE_AA)
    y += 22

    # Desglose de pesos
    cv2.putText(canvas, "Pesos:", (px + 4, y), font, 0.26,
                (110, 110, 120), 1, cv2.LINE_AA)
    y += 12
    for k, w in scorer.WEIGHTS.items():
        short = {'frequency': 'Freq', 'amplitude': 'Ampl', 'angular_vel': 'Omega',
                 'cv': 'CV', 'jerk': 'Jerk'}[k]
        cv2.putText(canvas, f"{short}:{w:.0%}", (px + 6, y), font, 0.24,
                    (90, 90, 100), 1, cv2.LINE_AA)
        y += 11

    # ========== CONTROLES ==========
    y_bot = VIDEO_H - 38
    cv2.line(canvas, (px, y_bot), (x0 + PANEL_W - 12, y_bot), (50, 50, 60), 1)
    y_bot += 14
    imu_status = "OK" if sensor_reader.connected else "sin datos"
    cv2.putText(canvas, f"IMU: {len(sig_proc.ax)} muestras ({imu_status})",
                (px, y_bot), font, 0.28, (100, 100, 110), 1, cv2.LINE_AA)
    y_bot += 13
    cv2.putText(canvas, "[R] Reset   [Q] Salir", (px, y_bot), font, 0.30,
                (130, 130, 130), 1, cv2.LINE_AA)


# =====================================================================
#  LOOP PRINCIPAL
# =====================================================================
print("=" * 50)
print("  UPDRS Parte 3 - Analisis Cuantitativo")
print("  ESP32-CAM + MPU6050")
print(f"  Frame:  {FRAME_URL}")
print(f"  Sensor: {SENSOR_URL}")
print("  Q = salir   R = reset contadores")
print("=" * 50)

# Iniciar hilo de lectura IMU
sensor_reader.start()

while True:
    # 1. Obtener frame de la camara
    frame_data = get_frame(FRAME_URL)
    if not frame_data:
        time.sleep(0.2)
        continue

    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # 2. Obtener muestras IMU del hilo paralelo
    imu_samples = sensor_reader.get_samples()
    if imu_samples:
        sig_proc.add_imu_samples(imu_samples)

    # 3. Procesar frame con MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # 4. Construir canvas: video (800x600) + panel lateral (300px)
    video  = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    canvas = np.zeros((VIDEO_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:, :VIDEO_W] = video

    # 5. Procesar manos detectadas
    if results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            hlabel = results.multi_handedness[idx].classification[0].label

            if hlabel not in hand_trackers:
                hand_trackers[hlabel] = HandTracker()
            tk = hand_trackers[hlabel]

            # Amplitud para vision
            tid = thumb_index_normalized(hand_lm)
            sig_proc.add_vision_sample(tid)

            # Actualizar tracker
            tk.update_tap(detect_tapping(hand_lm), sig_proc)
            tk.update_oc(detect_open_close(hand_lm))
            tk.update_pron(detect_pronation(hand_lm, hlabel))

            # Dibujar landmarks sobre area de video
            draw_landmarks(canvas, hand_lm, hlabel)

    # 6. Dibujar panel lateral con metricas y score
    draw_panel(canvas)

    # 7. Mostrar
    cv2.imshow("UPDRS - Analisis Cuantitativo", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        hand_trackers.clear()
        sig_proc.reset()
        print("Contadores y buffers reiniciados")

    time.sleep(0.02)

# Cleanup
sensor_reader.stop()
cv2.destroyAllWindows()
hands.close()
