"""
UPDRS Parte 3 - Analisis Cuantitativo de Movimientos
=====================================================
Fusion ESP32-CAM (vision + MediaPipe) + MPU6050 (IMU via I2C)

5 Variables cuantitativas:
  1. Frecuencia    FFT de aceleracion sin gravedad  (Hz)
  2. Amplitud      Rango P95-P5 pulgar-indice (normalizado por tamano mano)
  3. Vel. angular  RMS de giroscopio (deg/s)
  4. Regularidad   CV de intervalos entre taps (%)
  5. Jerk          Magnitud vector derivada de aceleracion (g/s)

Correspondencia UPDRS:
  freq baja -> bradicinesia     amp baja -> hipocinesia
  omega baja -> lentitud        CV alto -> irregularidad
  jerk alto -> falta de control

Layout: 1100x840
  Top   (600): video 800x600  |  panel lateral 300x600
  Bottom(240): 3 graficas tiempo real (accel / gyro / dist pulgar-indice)
"""

import os
# Silencia warnings de JPEG corrupto de OpenCV (vienen del decoder libjpeg)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

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

VIDEO_W, VIDEO_H = 800, 600
PANEL_W          = 300
PLOT_H           = 240              # altura area de graficas
CANVAS_W         = VIDEO_W + PANEL_W   # 1100
CANVAS_H         = VIDEO_H + PLOT_H    # 840

# Debug: imprime en terminal cada N muestras recibidas
DEBUG_PRINT_EVERY = 20


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
    'THUMB': (0, 0, 255),   'INDEX': (0, 255, 255),
    'MIDDLE': (0, 255, 0),  'RING': (255, 0, 0),
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
#  SCROLLING PLOT - Grafico deslizante con OpenCV
# =====================================================================
class ScrollingPlot:
    """
    Grafica tipo osciloscopio: mantiene los ultimos N valores
    y los dibuja como linea continua escalada automaticamente.
    """

    def __init__(self, width, height, title, unit, color,
                 n_points=250, y_range=None):
        self.width    = width
        self.height   = height
        self.title    = title
        self.unit     = unit
        self.color    = color
        self.n_points = n_points
        self.data     = deque(maxlen=n_points)
        self.y_range  = y_range    # None = auto-scale

    def add(self, value):
        if value is None or not np.isfinite(value):
            return
        self.data.append(float(value))

    def draw(self, canvas, x, y):
        # Fondo
        cv2.rectangle(canvas, (x, y), (x + self.width, y + self.height),
                      (18, 18, 25), -1)
        cv2.rectangle(canvas, (x, y), (x + self.width, y + self.height),
                      (70, 70, 85), 1)

        # Titulo + unidad
        cv2.putText(canvas, self.title, (x + 8, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, self.unit, (x + self.width - 55, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (150, 150, 160), 1, cv2.LINE_AA)

        if len(self.data) < 2:
            cv2.putText(canvas, "(esperando datos...)",
                        (x + 8, y + self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (120, 120, 130), 1, cv2.LINE_AA)
            return

        arr = np.asarray(self.data, dtype=np.float32)

        # Rango Y
        if self.y_range:
            y_min, y_max = self.y_range
        else:
            y_min = float(np.min(arr))
            y_max = float(np.max(arr))
        if y_max - y_min < 1e-6:
            y_max = y_min + 1e-6

        # Area grafica (deja margen para titulo y ejes)
        plot_top    = y + 24
        plot_bottom = y + self.height - 6
        plot_left   = x + 6
        plot_right  = x + self.width - 60
        plot_h      = plot_bottom - plot_top
        plot_w      = plot_right - plot_left

        # Linea cero si el rango cruza cero
        if y_min < 0 < y_max:
            zero_norm = 1.0 - ((0 - y_min) / (y_max - y_min))
            zy = plot_top + int(zero_norm * plot_h)
            cv2.line(canvas, (plot_left, zy), (plot_right, zy),
                     (55, 55, 70), 1, cv2.LINE_AA)

        # Polyline de los datos
        n = len(arr)
        pts = np.empty((n, 2), dtype=np.int32)
        for i in range(n):
            px = plot_left + int(i * plot_w / max(n - 1, 1))
            norm = (arr[i] - y_min) / (y_max - y_min)
            py = plot_bottom - int(norm * plot_h)
            pts[i] = (px, py)
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=self.color, thickness=1, lineType=cv2.LINE_AA)

        # Etiquetas de rango y valor actual
        cv2.putText(canvas, f"{y_max:+.2f}", (plot_left + 2, plot_top + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (130, 130, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{y_min:+.2f}", (plot_left + 2, plot_bottom - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (130, 130, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{arr[-1]:+.2f}",
                    (plot_right + 6, plot_top + plot_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    self.color, 1, cv2.LINE_AA)


# =====================================================================
#  SIGNAL PROCESSOR - buffers + calculo de las 5 variables
# =====================================================================
class SignalProcessor:
    """
    Procesa senales IMU y vision para las 5 variables UPDRS.

    Mejoras sobre version anterior:
      - Remocion de gravedad via media movil antes de FFT
      - FFT sobre el eje con mayor varianza (mas sensible al movimiento)
      - Jerk como magnitud del vector 3D (no solo magnitud escalar)
    """

    def __init__(self, max_samples=1024):
        self._max = max_samples
        # IMU buffers
        self.ax    = deque(maxlen=max_samples)
        self.ay    = deque(maxlen=max_samples)
        self.az    = deque(maxlen=max_samples)
        self.gx    = deque(maxlen=max_samples)
        self.gy    = deque(maxlen=max_samples)
        self.gz    = deque(maxlen=max_samples)
        self.imu_t = deque(maxlen=max_samples)

        # Vision buffers
        self.thumb_idx_dist = deque(maxlen=max_samples)
        self.tap_intervals  = deque(maxlen=100)

    # ---- Ingesta ----

    def add_imu_samples(self, samples):
        for s in samples:
            if len(s) < 7:
                continue
            self.ax.append(s[0]);  self.ay.append(s[1]);  self.az.append(s[2])
            self.gx.append(s[3]);  self.gy.append(s[4]);  self.gz.append(s[5])
            self.imu_t.append(s[6])

    def add_vision_sample(self, thumb_index_normalized):
        self.thumb_idx_dist.append(thumb_index_normalized)

    def add_tap_interval(self, interval_sec):
        self.tap_intervals.append(interval_sec)

    def effective_fs(self):
        """Frecuencia de muestreo efectiva en Hz."""
        if len(self.imu_t) < 2:
            return 0.0
        t = np.asarray(self.imu_t, dtype=np.float64) / 1000.0
        dt = np.diff(t)
        dt = dt[dt > 0.001]
        if len(dt) == 0:
            return 0.0
        return float(1.0 / np.median(dt))

    # ---- 1. FRECUENCIA ----

    def compute_frequency(self):
        """
        FFT del eje con mayor varianza dinamica (sin gravedad).

        La gravedad aparece como componente DC en un eje (~1g en reposo).
        Se elimina restando la media. El eje con mayor varianza restante
        es el que captura mejor el movimiento periodico.

        Ventana Hanning para reducir leakage, busqueda en 0.5-8 Hz.
        """
        n = len(self.ax)
        if n < 64:
            return 0.0

        ax = np.asarray(self.ax, dtype=np.float64)
        ay = np.asarray(self.ay, dtype=np.float64)
        az = np.asarray(self.az, dtype=np.float64)

        # Remover gravedad (DC)
        ax -= np.mean(ax); ay -= np.mean(ay); az -= np.mean(az)

        # Elegir eje con mayor varianza
        variances = [np.var(ax), np.var(ay), np.var(az)]
        signal = [ax, ay, az][int(np.argmax(variances))]

        # Si la varianza es ridicula, no hay movimiento real
        if np.max(variances) < 1e-4:
            return 0.0

        fs = self.effective_fs()
        if fs <= 0:
            return 0.0

        # FFT con ventana Hanning
        win = np.hanning(n)
        spec = np.abs(np.fft.rfft(signal * win))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        mask = (freqs >= 0.5) & (freqs <= 8.0)
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][np.argmax(spec[mask])])

    # ---- 2. AMPLITUD ----

    def compute_amplitude(self):
        if len(self.thumb_idx_dist) < 20:
            return 0.0
        arr = np.asarray(self.thumb_idx_dist)
        return float(np.percentile(arr, 95) - np.percentile(arr, 5))

    # ---- 3. VELOCIDAD ANGULAR ----

    def compute_angular_velocity(self):
        """RMS del vector giroscopio."""
        if len(self.gx) < 10:
            return 0.0
        g = (np.asarray(self.gx) ** 2 +
             np.asarray(self.gy) ** 2 +
             np.asarray(self.gz) ** 2)
        return float(np.sqrt(np.mean(g)))

    # ---- 4. REGULARIDAD ----

    def compute_cv(self):
        if len(self.tap_intervals) < 3:
            return 0.0
        arr = np.asarray(self.tap_intervals)
        mu = np.mean(arr)
        if mu < 0.01:
            return 0.0
        return float((np.std(arr) / mu) * 100.0)

    # ---- 5. JERK ----

    def compute_jerk(self):
        """
        Magnitud del vector jerk: sqrt( (da_x/dt)^2 + (da_y/dt)^2 + (da_z/dt)^2 )

        Mas sensible que la derivada de la magnitud, porque captura
        cambios en cualquier direccion (no solo del modulo).
        """
        n = len(self.ax)
        if n < 10:
            return 0.0

        ax = np.asarray(self.ax, dtype=np.float64)
        ay = np.asarray(self.ay, dtype=np.float64)
        az = np.asarray(self.az, dtype=np.float64)
        t  = np.asarray(self.imu_t, dtype=np.float64) / 1000.0

        dt = np.diff(t)
        valid = dt > 0.001
        if not np.any(valid):
            return 0.0

        jx = np.diff(ax)[valid] / dt[valid]
        jy = np.diff(ay)[valid] / dt[valid]
        jz = np.diff(az)[valid] / dt[valid]
        jmag = np.sqrt(jx * jx + jy * jy + jz * jz)
        return float(np.mean(jmag))

    # ---- Resumen ----

    def compute_all(self):
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
#  UPDRS SCORER
# =====================================================================
class UPDRSScorer:
    RANGES = {
        'frequency':   {'normal': 4.5,  'severe': 1.0},
        'amplitude':   {'normal': 0.65, 'severe': 0.10},
        'angular_vel': {'normal': 180., 'severe': 30.},
        'cv':          {'normal': 8.,   'severe': 50.},
        'jerk':        {'normal': 3.,   'severe': 35.},
    }
    WEIGHTS = {
        'frequency':   0.25, 'amplitude':   0.20, 'angular_vel': 0.15,
        'cv':          0.25, 'jerk':        0.15,
    }
    LABELS = {0: "Normal", 1: "Leve", 2: "Moderado leve",
              3: "Moderado", 4: "Severo"}
    COLORS = {0: (0, 200, 0),   1: (0, 210, 170), 2: (0, 210, 210),
              3: (0, 140, 255), 4: (0, 50, 255)}

    def normalize_var(self, name, value):
        r = self.RANGES[name]
        if name in ('cv', 'jerk'):
            score = (value - r['normal']) / (r['severe'] - r['normal'])
        else:
            score = (r['normal'] - value) / (r['normal'] - r['severe'])
        return max(0.0, min(1.0, score))

    def compute(self, metrics):
        individual = {k: self.normalize_var(k, metrics.get(k, 0.0))
                      for k in self.WEIGHTS}
        composite = sum(self.WEIGHTS[k] * individual[k] for k in self.WEIGHTS)
        composite = max(0.0, min(1.0, composite))
        updrs = int(round(composite * 4))
        updrs = max(0, min(4, updrs))
        return individual, composite, updrs, self.LABELS[updrs]


# =====================================================================
#  HAND TRACKER
# =====================================================================
class HandTracker:
    def __init__(self):
        self.tap_count     = 0
        self.last_tap      = False
        self.last_tap_time = 0.0
        self.pron_count    = 0
        self.last_pron     = ""
        self.oc_count      = 0
        self.last_oc       = ""

    def update_tap(self, tapping, sig):
        now = time.time()
        if tapping and not self.last_tap:
            self.tap_count += 1
            if self.last_tap_time > 0:
                interval = now - self.last_tap_time
                if 0.05 < interval < 3.0:
                    sig.add_tap_interval(interval)
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
#  SENSOR READER - hilo con logging
# =====================================================================
class SensorReader(threading.Thread):
    """
    Hilo daemon que consulta /sensor del ESP32.
    Imprime a terminal cada DEBUG_PRINT_EVERY muestras.
    """

    def __init__(self, url, interval=0.08):
        super().__init__(daemon=True)
        self.url      = url
        self.interval = interval
        self._lock    = threading.Lock()
        self._samples = []
        self.running  = True
        self.connected      = False
        self.total_samples  = 0
        self.last_error     = ""
        self.last_recv_time = 0.0
        self.last_print_n   = 0

    def run(self):
        while self.running:
            try:
                r = requests.get(self.url, timeout=2)
                if r.status_code == 200:
                    data = r.json().get('s', [])
                    if data:
                        with self._lock:
                            self._samples.extend(data)
                        self.total_samples += len(data)
                        self.last_recv_time = time.time()
                        self.connected = True
                        self.last_error = ""

                        # Log periodico
                        if self.total_samples - self.last_print_n >= DEBUG_PRINT_EVERY:
                            self.last_print_n = self.total_samples
                            last = data[-1]
                            print(
                                "[IMU] n=%-5d  a=(%+.2f,%+.2f,%+.2f)g  "
                                "g=(%+6.1f,%+6.1f,%+6.1f)d/s  (batch=%d)"
                                % (self.total_samples,
                                   last[0], last[1], last[2],
                                   last[3], last[4], last[5], len(data))
                            )
                    else:
                        # Respuesta vacia -> MPU no esta produciendo datos
                        self.connected = False
                        self.last_error = "buffer vacio (revisar MPU en ESP32)"
                else:
                    self.connected = False
                    self.last_error = f"HTTP {r.status_code}"
            except requests.exceptions.ConnectionError:
                self.connected = False
                self.last_error = "sin conexion"
            except requests.exceptions.Timeout:
                self.connected = False
                self.last_error = "timeout"
            except Exception as e:
                self.connected = False
                self.last_error = type(e).__name__

            time.sleep(self.interval)

    def get_samples(self):
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
sig_proc      = SignalProcessor(max_samples=1024)
scorer        = UPDRSScorer()
sensor_reader = SensorReader(SENSOR_URL, interval=0.08)

# Graficas: buffers con los ultimos valores
plot_accel    = ScrollingPlot(366, PLOT_H, "Aceleracion |a| sin gravedad",
                              "g",     (120, 220, 120), n_points=250)
plot_gyro     = ScrollingPlot(366, PLOT_H, "Velocidad angular |w|",
                              "deg/s", (120, 220, 255), n_points=250)
plot_vision   = ScrollingPlot(368, PLOT_H, "Distancia pulgar-indice (vision)",
                              "norm",  (255, 220, 120), n_points=250)

# Buffer corto de gravedad para high-pass online (media movil)
_grav_buf = deque(maxlen=60)   # ~0.6s a 100Hz


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
    w, m = lms.landmark[0], lms.landmark[9]
    return np.hypot(w.x - m.x, w.y - m.y)


# =====================================================================
#  DETECCION
# =====================================================================
def detect_tapping(lms):
    thumb, index = lms.landmark[4], lms.landmark[8]
    dist = np.hypot(thumb.x - index.x, thumb.y - index.y)
    s = hand_scale(lms)
    return (dist / s) < 0.40 if s > 0.01 else False


def thumb_index_normalized(lms):
    thumb, index = lms.landmark[4], lms.landmark[8]
    dist = np.hypot(thumb.x - index.x, thumb.y - index.y)
    s = hand_scale(lms)
    return dist / s if s > 0.01 else 0.0


def detect_open_close(lms):
    tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
    ext = sum(1 for t, p in tips_pips if lms.landmark[t].y < lms.landmark[p].y)
    if ext >= 3: return "Abierta"
    if ext <= 1: return "Cerrada"
    return "Parcial"


def detect_pronation(lms, label):
    wrist   = lms.landmark[0]
    idx_mcp = lms.landmark[5]
    pnk_mcp = lms.landmark[17]
    v1 = np.array([idx_mcp.x - wrist.x, idx_mcp.y - wrist.y, idx_mcp.z - wrist.z])
    v2 = np.array([pnk_mcp.x - wrist.x, pnk_mcp.y - wrist.y, pnk_mcp.z - wrist.z])
    nz = np.cross(v1, v2)[2]
    th = 0.006
    if label == "Right":
        if nz < -th: return "Supinacion"
        if nz > th:  return "Pronacion"
    else:
        if nz > th:  return "Supinacion"
        if nz < -th: return "Pronacion"
    return "Neutro"


# =====================================================================
#  DIBUJO
# =====================================================================
def draw_landmarks(canvas, lms, label):
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
            cv2.circle(canvas, (int(lm.x * VIDEO_W), int(lm.y * VIDEO_H)),
                       4, color, -1)

    wr = lms.landmark[0]
    cv2.circle(canvas, (int(wr.x * VIDEO_W), int(wr.y * VIDEO_H)),
               6, (220, 220, 220), -1)
    cv2.putText(canvas, label,
                (max(int(wr.x * VIDEO_W) - 15, 2),
                 max(int(wr.y * VIDEO_H) - 12, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bar(canvas, x, y, w, h, value, max_val, color):
    fill = int(w * min(value / max_val, 1.0)) if max_val > 0 else 0
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 55), -1)
    if fill > 0:
        cv2.rectangle(canvas, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 95), 1)


def draw_sensor_badge(canvas, x, y):
    """Indicador grande de estado del sensor IMU."""
    connected = sensor_reader.connected
    color = (0, 200, 0) if connected else (0, 50, 255)
    text  = "IMU OK" if connected else "IMU OFFLINE"

    # Punto LED
    cv2.circle(canvas, (x + 10, y + 10), 7, color, -1)
    cv2.circle(canvas, (x + 10, y + 10), 7, (220, 220, 220), 1)

    # Texto
    cv2.putText(canvas, text, (x + 24, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Estadisticas
    fs = sig_proc.effective_fs()
    info = "N=%d  fs=%.1f Hz" % (sensor_reader.total_samples, fs)
    cv2.putText(canvas, info, (x + 24, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 180), 1, cv2.LINE_AA)

    if not connected and sensor_reader.last_error:
        cv2.putText(canvas, "err: " + sensor_reader.last_error[:28],
                    (x + 24, y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 255), 1, cv2.LINE_AA)


def draw_panel(canvas):
    x0   = VIDEO_W
    font = cv2.FONT_HERSHEY_SIMPLEX
    px   = x0 + 12
    bw   = PANEL_W - 40

    # Fondo
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, VIDEO_H), (25, 25, 30), -1)
    cv2.line(canvas, (x0, 0), (x0, VIDEO_H), (70, 70, 80), 2)

    y = 22

    # ===== TITULO =====
    cv2.putText(canvas, "UPDRS Parte 3", (px, y), font, 0.55,
                (255, 255, 120), 2, cv2.LINE_AA)
    y += 16
    cv2.putText(canvas, "Analisis Cuantitativo", (px, y), font, 0.32,
                (140, 140, 150), 1, cv2.LINE_AA)
    y += 8

    # ===== SENSOR BADGE =====
    cv2.rectangle(canvas, (px - 4, y), (x0 + PANEL_W - 8, y + 52),
                  (15, 15, 20), -1)
    cv2.rectangle(canvas, (px - 4, y), (x0 + PANEL_W - 8, y + 52),
                  (60, 60, 70), 1)
    draw_sensor_badge(canvas, px, y + 2)
    y += 60

    # ===== INFO POR MANO =====
    for hlabel in ["Right", "Left"]:
        tk = hand_trackers.get(hlabel)
        if not tk:
            continue
        name = "Mano Der." if hlabel == "Right" else "Mano Izq."
        hdr  = (200, 200, 255) if hlabel == "Right" else (255, 200, 255)

        cv2.putText(canvas, name, (px, y), font, 0.43, hdr, 1, cv2.LINE_AA)
        y += 15

        tc = (0, 255, 100) if tk.last_tap else (110, 110, 110)
        cv2.putText(canvas, f"Golpeteo: {'SI' if tk.last_tap else 'NO'}  Taps:{tk.tap_count}",
                    (px + 6, y), font, 0.31, tc, 1, cv2.LINE_AA)
        y += 13

        oc = tk.last_oc or "---"
        oc_c = {"Abierta": (0, 255, 160), "Cerrada": (50, 120, 255),
                "Parcial": (255, 210, 0)}.get(oc, (110, 110, 110))
        cv2.putText(canvas, f"Mano: {oc}  Cambios:{tk.oc_count}",
                    (px + 6, y), font, 0.31, oc_c, 1, cv2.LINE_AA)
        y += 13

        pr = tk.last_pron or "---"
        pr_c = {"Supinacion": (0, 220, 255), "Pronacion": (255, 160, 0),
                "Neutro": (110, 110, 110)}.get(pr, (110, 110, 110))
        cv2.putText(canvas, f"Rot: {pr}  Cambios:{tk.pron_count}",
                    (px + 6, y), font, 0.31, pr_c, 1, cv2.LINE_AA)
        y += 16

    cv2.line(canvas, (px, y), (x0 + PANEL_W - 12, y), (50, 50, 60), 1)
    y += 10

    # ===== VARIABLES =====
    metrics = sig_proc.compute_all()
    individual, composite, updrs, ulabel = scorer.compute(metrics)

    cv2.putText(canvas, "Variables Sensor+Vision", (px, y), font, 0.38,
                (200, 200, 120), 1, cv2.LINE_AA)
    y += 14

    var_rows = [
        ("Frecuencia",  f"{metrics['frequency']:.2f} Hz",    'frequency',   "bradicinesia"),
        ("Amplitud",    f"{metrics['amplitude']:.3f}",       'amplitude',   "hipocinesia"),
        ("Vel.Angular", f"{metrics['angular_vel']:.1f} d/s", 'angular_vel', "lentitud"),
        ("CV",          f"{metrics['cv']:.1f} %",            'cv',          "irregularidad"),
        ("Jerk",        f"{metrics['jerk']:.2f} g/s",        'jerk',        "falta control"),
    ]
    for vname, vstr, vkey, vcorr in var_rows:
        vscore = individual.get(vkey, 0)
        r = int(vscore * 255)
        g = int((1 - vscore) * 190)
        bar_col = (0, g, r)

        cv2.putText(canvas, f"{vname}: {vstr}", (px + 4, y), font, 0.30,
                    (180, 180, 180), 1, cv2.LINE_AA)
        tw = cv2.getTextSize(f"{vname}: {vstr}", font, 0.30, 1)[0][0]
        cv2.putText(canvas, vcorr, (px + 6 + tw + 4, y), font, 0.24,
                    (100, 100, 110), 1, cv2.LINE_AA)
        y += 9
        draw_bar(canvas, px + 4, y, bw, 6, vscore, 1.0, bar_col)
        y += 12

    y += 2
    cv2.line(canvas, (px, y), (x0 + PANEL_W - 12, y), (50, 50, 60), 1)
    y += 12

    # ===== SCORE UPDRS =====
    updrs_color = scorer.COLORS.get(updrs, (200, 200, 200))
    cv2.putText(canvas, "Indice UPDRS", (px, y), font, 0.42,
                (255, 255, 200), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, f"{updrs} - {ulabel}", (px + 4, y), font, 0.50,
                updrs_color, 2, cv2.LINE_AA)
    y += 14
    draw_bar(canvas, px + 4, y, bw, 11, composite, 1.0, updrs_color)
    cv2.putText(canvas, f"{composite * 100:.0f}%",
                (px + bw + 8, y + 9), font, 0.30,
                (150, 150, 150), 1, cv2.LINE_AA)
    y += 20

    # Pesos
    weight_txt = "pesos: " + " ".join(
        f"{k[:4]}:{int(v*100)}" for k, v in scorer.WEIGHTS.items())
    cv2.putText(canvas, weight_txt, (px + 4, y), font, 0.25,
                (110, 110, 120), 1, cv2.LINE_AA)

    # ===== CONTROLES =====
    y_bot = VIDEO_H - 32
    cv2.line(canvas, (px, y_bot), (x0 + PANEL_W - 12, y_bot), (50, 50, 60), 1)
    y_bot += 14
    cv2.putText(canvas, "[R] Reset   [Q] Salir   [P] Imprime metricas",
                (px, y_bot), font, 0.28, (130, 130, 130), 1, cv2.LINE_AA)


def draw_plots(canvas):
    """Dibuja las 3 graficas en el area inferior (y=600..840)."""
    y0 = VIDEO_H
    plot_accel.draw(canvas, 0, y0)
    plot_gyro.draw (canvas, 366, y0)
    plot_vision.draw(canvas, 732, y0)


# =====================================================================
#  LOOP PRINCIPAL
# =====================================================================
print("=" * 60)
print("  UPDRS Parte 3 - Analisis Cuantitativo")
print("  ESP32-CAM + MPU6050")
print("  Frame : " + FRAME_URL)
print("  Sensor: " + SENSOR_URL)
print("  Teclas: Q=salir  R=reset  P=imprime metricas")
print("=" * 60)

# Iniciar lectura IMU en paralelo
sensor_reader.start()

# FPS counter
fps_ts   = time.time()
fps_n    = 0
fps_val  = 0.0

# Para log de metricas en terminal
last_metrics_log = time.time()

while True:
    # ---- Frame ----
    frame_data = get_frame(FRAME_URL)
    if not frame_data:
        time.sleep(0.2)
        continue

    try:
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        frame = None
    if frame is None:
        continue

    # ---- IMU ----
    imu_samples = sensor_reader.get_samples()
    if imu_samples:
        sig_proc.add_imu_samples(imu_samples)

        # Actualizar graficas: magnitud de aceleracion sin gravedad + |gyro|
        for s in imu_samples:
            ax, ay, az = s[0], s[1], s[2]
            gx, gy, gz = s[3], s[4], s[5]
            mag_a = float(np.sqrt(ax*ax + ay*ay + az*az))
            _grav_buf.append(mag_a)
            grav = float(np.mean(_grav_buf)) if len(_grav_buf) > 5 else 1.0
            plot_accel.add(mag_a - grav)        # remueve ~1g
            plot_gyro.add(float(np.sqrt(gx*gx + gy*gy + gz*gz)))

    # ---- Vision ----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # ---- Canvas ----
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    video  = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    canvas[:VIDEO_H, :VIDEO_W] = video

    # ---- Procesar manos ----
    if results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            hlabel = results.multi_handedness[idx].classification[0].label
            if hlabel not in hand_trackers:
                hand_trackers[hlabel] = HandTracker()
            tk = hand_trackers[hlabel]

            tid = thumb_index_normalized(hand_lm)
            sig_proc.add_vision_sample(tid)
            plot_vision.add(tid)

            tk.update_tap(detect_tapping(hand_lm), sig_proc)
            tk.update_oc(detect_open_close(hand_lm))
            tk.update_pron(detect_pronation(hand_lm, hlabel))

            draw_landmarks(canvas, hand_lm, hlabel)

    # ---- Panel + graficas ----
    draw_panel(canvas)
    draw_plots(canvas)

    # ---- FPS ----
    fps_n += 1
    now = time.time()
    if now - fps_ts >= 1.0:
        fps_val = fps_n / (now - fps_ts)
        fps_ts = now
        fps_n = 0
    cv2.putText(canvas, f"FPS: {fps_val:.1f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    # ---- Log periodico de metricas a terminal ----
    if now - last_metrics_log >= 3.0:
        last_metrics_log = now
        m = sig_proc.compute_all()
        fs = sig_proc.effective_fs()
        print("[MET] fs=%.1fHz  freq=%.2fHz  amp=%.3f  omega=%.1fd/s  CV=%.1f%%  jerk=%.2fg/s"
              % (fs, m['frequency'], m['amplitude'],
                 m['angular_vel'], m['cv'], m['jerk']))

    # ---- Display ----
    cv2.imshow("UPDRS - Analisis Cuantitativo", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        hand_trackers.clear()
        sig_proc.reset()
        _grav_buf.clear()
        plot_accel.data.clear()
        plot_gyro.data.clear()
        plot_vision.data.clear()
        print("[RESET] Contadores, buffers y graficas reiniciados")
    elif key == ord('p'):
        m = sig_proc.compute_all()
        individual, composite, updrs, ulabel = scorer.compute(m)
        print("")
        print("=" * 50)
        print("  METRICAS ACTUALES")
        print("=" * 50)
        print(f"  fs efectiva     = {sig_proc.effective_fs():.2f} Hz")
        print(f"  Muestras IMU    = {sensor_reader.total_samples}")
        for k, v in m.items():
            print(f"  {k:12s}    = {v:.3f}  -> score {individual[k]:.2f}")
        print(f"  Indice compuesto = {composite:.3f}")
        print(f"  UPDRS            = {updrs} ({ulabel})")
        print("=" * 50)

    time.sleep(0.01)

# Cleanup
sensor_reader.stop()
cv2.destroyAllWindows()
hands.close()
