"""
UPDRS Parte 3 - Analisis Cuantitativo de Movimientos
=====================================================
Fusion ESP32-CAM (vision + MediaPipe) + MPU6050 (IMU via I2C).

5 Variables cuantitativas:
  1. Frecuencia    FFT de aceleracion sin gravedad             (Hz)
  2. Amplitud      Rango P95-P5 pulgar-indice (norm. tamano)   (-)
  3. Vel. angular  RMS giroscopio                              (deg/s)
  4. Regularidad   CV intervalos entre taps                    (%)
  5. Jerk          Magnitud derivada de a_dynamic              (g/s)

Pipeline IMU (NUEVO):
  raw -> median(3) -> IIR LP 15Hz -> gravity comp.filter -> a_dynamic
  Antes de FFT: resample a grid uniforme + detrend + Hanning + rfft

Layout 1100x840:
  Top   (600): video 800x600 | panel lateral 300x600
  Bottom(240): 3 graficas tiempo real (a_dyn, |w|, dist pulgar-indice)
"""

import os
# Silencia warnings de JPEG corrupto (libjpeg) - debe ir antes de importar cv2
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import math
import numpy as np
import requests
import mediapipe as mp
import time
import threading
from collections import deque


# =====================================================================
#  CONFIGURACION + CONSTANTES
# =====================================================================
ESP32_IP   = "192.168.4.1"
FRAME_URL  = f"http://{ESP32_IP}/frame"
SENSOR_URL = f"http://{ESP32_IP}/sensor"
STATUS_URL = f"http://{ESP32_IP}/status"

# Schema key del payload /sensor; cross-reference: esp32_stream.py:IMU_PAYLOAD_KEY.
IMU_PAYLOAD_KEY = "s"

VIDEO_W, VIDEO_H = 800, 600
PANEL_W          = 300
PLOT_H           = 240
CANVAS_W         = VIDEO_W + PANEL_W
CANVAS_H         = VIDEO_H + PLOT_H

# Hand-label string constants (MediaPipe convention).
HAND_RIGHT = "Right"
HAND_LEFT  = "Left"

# Open/close states.
OC_OPEN    = "Abierta"
OC_CLOSED  = "Cerrada"
OC_PARTIAL = "Parcial"

# Pronation/supination states.
ROT_PRON   = "Pronacion"
ROT_SUP    = "Supinacion"
ROT_NEUTRO = "Neutro"

# Sensor target sample rate (matches firmware SAMPLE_INTERVAL_MS=10ms).
TARGET_FS_HZ = 100.0

# Si pasan mas de OFFLINE_AFTER_S sin recibir muestras, marca IMU OFFLINE.
# Sin esto, la badge parpadea cada poll vacio normal (~ cada 80 ms).
OFFLINE_AFTER_S = 1.5

# Cache de metricas: recompute mas alla de este intervalo.
METRICS_CACHE_S = 0.25   # 4 Hz - mas que suficiente para UI a 30 FPS.

# Log a terminal cada N muestras IMU recibidas.
DEBUG_PRINT_EVERY = 50


# =====================================================================
#  MEDIAPIPE
# =====================================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =====================================================================
#  CONSTANTES VISUALES
# =====================================================================
RIGHT_COLORS = {
    'THUMB': (0, 0, 255),   'INDEX': (0, 255, 255),
    'MIDDLE': (0, 255, 0),  'RING': (255, 0, 0),
    'PINKY': (255, 0, 255),
}
LEFT_COLORS = {
    'THUMB': (0, 128, 255),    'INDEX': (128, 255, 255),
    'MIDDLE': (128, 255, 128), 'RING': (255, 128, 128),
    'PINKY': (255, 128, 255),
}
FINGER_LANDMARKS = {
    'THUMB': [1, 2, 3, 4],   'INDEX': [5, 6, 7, 8],
    'MIDDLE': [9, 10, 11, 12], 'RING': [13, 14, 15, 16],
    'PINKY': [17, 18, 19, 20],
}
FINGER_CONNECTIONS = {
    'THUMB': [(1, 2), (2, 3), (3, 4)],
    'INDEX': [(5, 6), (6, 7), (7, 8)],
    'MIDDLE': [(9, 10), (10, 11), (11, 12)],
    'RING': [(13, 14), (14, 15), (15, 16)],
    'PINKY': [(17, 18), (18, 19), (19, 20)],
}
PALM_CONNECTIONS = [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


# =====================================================================
#  HELPERS
# =====================================================================
def lm_dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def lm_to_pixel(lm):
    return int(lm.x * VIDEO_W), int(lm.y * VIDEO_H)


# =====================================================================
#  SCROLLING PLOT (polyline vectorizado)
# =====================================================================
class ScrollingPlot:
    def __init__(self, width, height, title, unit, color,
                 n_points=250, y_range=None):
        self.width    = width
        self.height   = height
        self.title    = title
        self.unit     = unit
        self.color    = color
        self.data     = deque(maxlen=n_points)
        self.y_range  = y_range

    def add(self, value):
        if value is None or not math.isfinite(value):
            return
        self.data.append(float(value))

    def draw(self, canvas, x, y):
        cv2.rectangle(canvas, (x, y), (x + self.width, y + self.height),
                      (18, 18, 25), -1)
        cv2.rectangle(canvas, (x, y), (x + self.width, y + self.height),
                      (70, 70, 85), 1)
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

        arr = np.fromiter(self.data, dtype=np.float32, count=len(self.data))

        if self.y_range:
            y_min, y_max = self.y_range
        else:
            y_min = float(arr.min())
            y_max = float(arr.max())
        if y_max - y_min < 1e-6:
            y_max = y_min + 1e-6

        plot_top    = y + 24
        plot_bottom = y + self.height - 6
        plot_left   = x + 6
        plot_right  = x + self.width - 60
        plot_h      = plot_bottom - plot_top
        plot_w      = plot_right - plot_left

        if y_min < 0 < y_max:
            zy = plot_top + int((1.0 - (-y_min) / (y_max - y_min)) * plot_h)
            cv2.line(canvas, (plot_left, zy), (plot_right, zy),
                     (55, 55, 70), 1, cv2.LINE_AA)

        # Vectorized polyline: O(n) numpy ops instead of Python for-loop.
        n = len(arr)
        xs = plot_left + (np.arange(n) * plot_w / max(n - 1, 1)).astype(np.int32)
        ys = plot_bottom - ((arr - y_min) / (y_max - y_min) * plot_h).astype(np.int32)
        pts = np.stack([xs, ys], axis=1)
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=self.color, thickness=1, lineType=cv2.LINE_AA)

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
#  IMU PREPROCESSOR  (NUEVO)
# =====================================================================
class IMUPreprocessor:
    """
    Pipeline online por muestra:
        raw -> median(3) -> IIR LP -> gravity comp.filter -> a_dynamic
        gyro_offset (residual) -> IIR LP

    Saca (a_dyn_xyz, gyro_xyz, t_ms). El firmware ya hace su propia
    cal de gyro al boot; esto compensa offset residual y fija la
    referencia de gravedad.

    Coeficientes (fs=100Hz):
      LP_ALPHA   = 0.5    -> cutoff ~16 Hz (sobre la banda de tremor)
      GRAV_ALPHA = 0.97   -> cutoff ~0.5 Hz (deja pasar movimiento dinamico)
    """

    def __init__(self, lp_alpha=0.5, grav_alpha=0.97, gyro_offset=None):
        self.LP_ALPHA   = lp_alpha
        self.GRAV_ALPHA = grav_alpha
        self.gyro_offset = (np.zeros(3) if gyro_offset is None
                            else np.asarray(gyro_offset, dtype=np.float64))

        # IIR low-pass state (per axis).
        self._lp_a = np.zeros(3)
        self._lp_g = np.zeros(3)

        # 3-tap median ring buffer per axis (3 axes x 3 samples).
        self._med_a = np.zeros((3, 3))
        self._med_g = np.zeros((3, 3))
        self._med_idx = 0
        self._n_seen = 0       # warmup counter for median

        # Gravity tracked via complementary filter; bootstrap on first sample.
        self._gravity = None

    def reset(self):
        self.__init__(self.LP_ALPHA, self.GRAV_ALPHA, self.gyro_offset)

    def process(self, ax, ay, az, gx, gy, gz, t_ms):
        a = np.array((ax, ay, az), dtype=np.float64)
        g = np.array((gx, gy, gz), dtype=np.float64) - self.gyro_offset

        # Update median ring then take per-axis median.
        self._med_a[:, self._med_idx] = a
        self._med_g[:, self._med_idx] = g
        self._med_idx = (self._med_idx + 1) % 3
        self._n_seen = min(self._n_seen + 1, 3)
        if self._n_seen >= 3:
            a_m = np.median(self._med_a, axis=1)
            g_m = np.median(self._med_g, axis=1)
        else:
            a_m, g_m = a, g

        # IIR low-pass.
        self._lp_a = self.LP_ALPHA * a_m + (1.0 - self.LP_ALPHA) * self._lp_a
        self._lp_g = self.LP_ALPHA * g_m + (1.0 - self.LP_ALPHA) * self._lp_g

        # Gravity bootstrap then complementary tracking.
        if self._gravity is None:
            self._gravity = self._lp_a.copy()
        else:
            self._gravity = (self.GRAV_ALPHA * self._gravity
                             + (1.0 - self.GRAV_ALPHA) * self._lp_a)

        a_dyn = self._lp_a - self._gravity
        return (a_dyn[0], a_dyn[1], a_dyn[2],
                self._lp_g[0], self._lp_g[1], self._lp_g[2],
                t_ms)


# =====================================================================
#  SIGNAL PROCESSOR
# =====================================================================
class SignalProcessor:
    """
    Almacena muestras YA preprocesadas (de IMUPreprocessor) y vision.
    compute_all() esta cacheado a METRICS_CACHE_S (4 Hz por defecto).

    Mejora clave para FFT: resamplea a grid uniforme + detrend + Hanning.
    np.asarray(deque) se hace UNA vez en compute_all() y se comparte entre
    los sub-calculos (antes se rebuilea ~5 veces por compute).
    """

    def __init__(self, max_samples=1024):
        self._max = max_samples
        self.ax = deque(maxlen=max_samples)
        self.ay = deque(maxlen=max_samples)
        self.az = deque(maxlen=max_samples)
        self.gx = deque(maxlen=max_samples)
        self.gy = deque(maxlen=max_samples)
        self.gz = deque(maxlen=max_samples)
        self.t  = deque(maxlen=max_samples)

        self.thumb_idx_dist = deque(maxlen=max_samples)
        self.tap_intervals  = deque(maxlen=100)

        self._cache = None
        self._cache_t = 0.0

    def add_processed_sample(self, sample7):
        ax, ay, az, gx, gy, gz, t_ms = sample7
        self.ax.append(ax); self.ay.append(ay); self.az.append(az)
        self.gx.append(gx); self.gy.append(gy); self.gz.append(gz)
        self.t.append(t_ms)

    def add_vision_sample(self, normalized):
        self.thumb_idx_dist.append(normalized)

    def add_tap_interval(self, interval_sec):
        self.tap_intervals.append(interval_sec)

    def reset(self):
        self.__init__(max_samples=self._max)

    # ---- helpers ----

    def _effective_fs(self, t_arr):
        if len(t_arr) < 2:
            return 0.0
        dt = np.diff(t_arr) / 1000.0
        dt = dt[dt > 0.001]
        return float(1.0 / np.median(dt)) if len(dt) else 0.0

    def effective_fs(self):
        if len(self.t) < 2:
            return 0.0
        return self._effective_fs(np.asarray(self.t, dtype=np.float64))

    @staticmethod
    def _resample_uniform(signal, t_ms, fs_target):
        """Linear-interpolate signal to a uniform time grid at fs_target Hz.
        Devuelve (signal_unif, n) o (None, 0) si no hay datos suficientes."""
        if len(signal) < 8:
            return None, 0
        t_s = t_ms / 1000.0
        duration = t_s[-1] - t_s[0]
        if duration < 0.5:           # menos de medio segundo: FFT no tiene resolucion
            return None, 0
        n = int(duration * fs_target)
        if n < 64:
            return None, 0
        t_unif = np.linspace(t_s[0], t_s[-1], n)
        return np.interp(t_unif, t_s, signal), n

    @staticmethod
    def _detrend(signal):
        """Resta la mejor recta en minimos cuadrados (quita drift lineal)."""
        n = len(signal)
        x = np.arange(n)
        slope, intercept = np.polyfit(x, signal, 1)
        return signal - (slope * x + intercept)

    # ---- 1. FRECUENCIA ----

    def _compute_frequency(self, ax, ay, az, t_ms):
        if len(ax) < 64:
            return 0.0
        # Eje con mayor varianza dinamica (a_dyn ya viene sin gravedad).
        variances = (np.var(ax), np.var(ay), np.var(az))
        if max(variances) < 1e-5:        # casi DC -> sin movimiento
            return 0.0
        signal = (ax, ay, az)[int(np.argmax(variances))]

        # Resamplear a grid uniforme antes de FFT.
        unif, n = self._resample_uniform(signal, t_ms, TARGET_FS_HZ)
        if unif is None:
            return 0.0

        # Detrend + ventana Hanning.
        unif = self._detrend(unif)
        spec = np.abs(np.fft.rfft(unif * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / TARGET_FS_HZ)

        mask = (freqs >= 0.5) & (freqs <= 8.0)
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][np.argmax(spec[mask])])

    # ---- 2. AMPLITUD ----

    def _compute_amplitude(self):
        if len(self.thumb_idx_dist) < 20:
            return 0.0
        arr = np.fromiter(self.thumb_idx_dist, dtype=np.float64,
                          count=len(self.thumb_idx_dist))
        return float(np.percentile(arr, 95) - np.percentile(arr, 5))

    # ---- 3. VEL ANGULAR ----

    @staticmethod
    def _compute_angular_velocity(gx, gy, gz):
        if len(gx) < 10:
            return 0.0
        return float(np.sqrt(np.mean(gx * gx + gy * gy + gz * gz)))

    # ---- 4. CV ----

    def _compute_cv(self):
        if len(self.tap_intervals) < 3:
            return 0.0
        arr = np.fromiter(self.tap_intervals, dtype=np.float64,
                          count=len(self.tap_intervals))
        mu = float(np.mean(arr))
        if mu < 0.01:
            return 0.0
        return float((np.std(arr) / mu) * 100.0)

    # ---- 5. JERK ----

    @staticmethod
    def _compute_jerk(ax, ay, az, t_ms):
        if len(ax) < 10:
            return 0.0
        t_s = t_ms / 1000.0
        dt = np.diff(t_s)
        valid = dt > 0.001
        if not np.any(valid):
            return 0.0
        jx = np.diff(ax)[valid] / dt[valid]
        jy = np.diff(ay)[valid] / dt[valid]
        jz = np.diff(az)[valid] / dt[valid]
        return float(np.mean(np.sqrt(jx * jx + jy * jy + jz * jz)))

    # ---- compute_all con cache ----

    def compute_all(self, force=False):
        now = time.time()
        if not force and self._cache and (now - self._cache_t) < METRICS_CACHE_S:
            return self._cache

        # Convertir deques a arrays UNA vez; los sub-calculos los comparten.
        if len(self.ax):
            ax = np.fromiter(self.ax, dtype=np.float64, count=len(self.ax))
            ay = np.fromiter(self.ay, dtype=np.float64, count=len(self.ay))
            az = np.fromiter(self.az, dtype=np.float64, count=len(self.az))
            gx = np.fromiter(self.gx, dtype=np.float64, count=len(self.gx))
            gy = np.fromiter(self.gy, dtype=np.float64, count=len(self.gy))
            gz = np.fromiter(self.gz, dtype=np.float64, count=len(self.gz))
            t_ms = np.fromiter(self.t, dtype=np.float64, count=len(self.t))
            fs = self._effective_fs(t_ms)
        else:
            ax = ay = az = gx = gy = gz = t_ms = np.empty(0)
            fs = 0.0

        result = {
            'frequency':   self._compute_frequency(ax, ay, az, t_ms),
            'amplitude':   self._compute_amplitude(),
            'angular_vel': self._compute_angular_velocity(gx, gy, gz),
            'cv':          self._compute_cv(),
            'jerk':        self._compute_jerk(ax, ay, az, t_ms),
            'fs':          fs,
        }
        self._cache = result
        self._cache_t = now
        return result


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
        updrs = max(0, min(4, int(round(composite * 4))))
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
        # Guard clauses (was nested conditional with hardcoded "Neutro").
        if not self.last_pron:
            self.last_pron = state
            return
        if state == ROT_NEUTRO or state == self.last_pron:
            self.last_pron = state
            return
        self.pron_count += 1
        self.last_pron = state

    def update_oc(self, state):
        if state != self.last_oc and self.last_oc:
            self.oc_count += 1
        self.last_oc = state


# =====================================================================
#  SENSOR READER
# =====================================================================
class SensorReader(threading.Thread):
    """
    Hilo daemon que consulta /sensor del ESP32 con keep-alive (Session)
    y timeouts cortos. Marca OFFLINE solo tras OFFLINE_AFTER_S sin datos
    (antes parpadeaba en cada poll vacio normal).
    """

    def __init__(self, url, interval=0.05):
        super().__init__(daemon=True)
        self.url      = url
        self.interval = interval
        self._lock    = threading.Lock()
        self._samples = []
        self.running  = True
        self.total_samples  = 0
        self.last_recv_time = 0.0
        self.last_error     = ""
        self._last_log_bucket = 0

        self._session = requests.Session()
        self._session.headers.update({"Connection": "keep-alive"})

    @property
    def connected(self):
        # Single source of truth; deriva del ultimo recv en lugar de un flag
        # que parpadeaba en cada poll vacio.
        return (time.time() - self.last_recv_time) < OFFLINE_AFTER_S

    def run(self):
        # connect_timeout corto -> recovery rapido tras Wi-Fi drop.
        # read_timeout corto -> hilo no se queda 2s bloqueado.
        timeouts = (0.3, 0.5)
        while self.running:
            try:
                r = self._session.get(self.url, timeout=timeouts)
                if r.status_code == 200:
                    data = r.json().get(IMU_PAYLOAD_KEY, [])
                    if data:
                        with self._lock:
                            self._samples.extend(data)
                        self.total_samples += len(data)
                        self.last_recv_time = time.time()
                        self.last_error = ""

                        bucket = self.total_samples // DEBUG_PRINT_EVERY
                        if bucket > self._last_log_bucket:
                            self._last_log_bucket = bucket
                            last = data[-1]
                            print("[IMU] n=%-5d  a=(%+.2f,%+.2f,%+.2f)g  "
                                  "g=(%+6.1f,%+6.1f,%+6.1f)d/s  (batch=%d)"
                                  % (self.total_samples,
                                     last[0], last[1], last[2],
                                     last[3], last[4], last[5], len(data)))
                else:
                    self.last_error = f"HTTP {r.status_code}"
            except requests.exceptions.ConnectionError:
                self.last_error = "sin conexion"
            except requests.exceptions.Timeout:
                self.last_error = "timeout"
            except Exception as e:
                self.last_error = type(e).__name__

            time.sleep(self.interval)

    def get_samples(self):
        with self._lock:
            out = self._samples
            self._samples = []
        return out

    def stop(self):
        self.running = False


# =====================================================================
#  INSTANCIAS GLOBALES
# =====================================================================
hand_trackers = {}
imu_pre       = IMUPreprocessor()
sig_proc      = SignalProcessor(max_samples=1024)
scorer        = UPDRSScorer()
sensor_reader = SensorReader(SENSOR_URL, interval=0.05)

plot_accel  = ScrollingPlot(366, PLOT_H,
                            "Aceleracion |a_dyn| (sin gravedad)",
                            "g",     (120, 220, 120), n_points=300)
plot_gyro   = ScrollingPlot(366, PLOT_H,
                            "Velocidad angular |w|",
                            "deg/s", (120, 220, 255), n_points=300)
plot_vision = ScrollingPlot(368, PLOT_H,
                            "Distancia pulgar-indice (vision)",
                            "norm",  (255, 220, 120), n_points=300)


# =====================================================================
#  HTTP HELPERS
# =====================================================================
def get_frame(url, session):
    try:
        r = session.get(url, timeout=(0.5, 1.5))
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


# =====================================================================
#  DETECCION
# =====================================================================
def hand_scale(lms):
    return lm_dist2d(lms.landmark[0], lms.landmark[9])


def thumb_index_normalized(lms):
    s = hand_scale(lms)
    if s < 0.01:
        return 0.0
    return lm_dist2d(lms.landmark[4], lms.landmark[8]) / s


def detect_tapping(lms):
    s = hand_scale(lms)
    if s < 0.01:
        return False
    return (lm_dist2d(lms.landmark[4], lms.landmark[8]) / s) < 0.40


def detect_open_close(lms):
    tips_pips = ((8, 6), (12, 10), (16, 14), (20, 18))
    ext = sum(1 for tip, pip in tips_pips
              if lms.landmark[tip].y < lms.landmark[pip].y)
    if ext >= 3:
        return OC_OPEN
    if ext <= 1:
        return OC_CLOSED
    return OC_PARTIAL


def detect_pronation(lms, label):
    """Sign of palm normal's z-component, flipped per hand."""
    wrist   = lms.landmark[0]
    idx_mcp = lms.landmark[5]
    pnk_mcp = lms.landmark[17]
    v1 = np.array((idx_mcp.x - wrist.x, idx_mcp.y - wrist.y, idx_mcp.z - wrist.z))
    v2 = np.array((pnk_mcp.x - wrist.x, pnk_mcp.y - wrist.y, pnk_mcp.z - wrist.z))
    nz = float(np.cross(v1, v2)[2])
    sign = -1.0 if label == HAND_RIGHT else 1.0
    nz_signed = sign * nz
    th = 0.006
    if nz_signed >  th: return ROT_SUP
    if nz_signed < -th: return ROT_PRON
    return ROT_NEUTRO


# =====================================================================
#  DIBUJO
# =====================================================================
def _draw_segments(canvas, lms, segments, color, thickness=2):
    for s_i, e_i in segments:
        s = lm_to_pixel(lms.landmark[s_i])
        e = lm_to_pixel(lms.landmark[e_i])
        cv2.line(canvas, s, e, color, thickness)


def draw_landmarks(canvas, lms, label):
    colors = RIGHT_COLORS if label == HAND_RIGHT else LEFT_COLORS

    _draw_segments(canvas, lms, PALM_CONNECTIONS, (180, 180, 180), 2)

    for fname, color in colors.items():
        _draw_segments(canvas, lms, FINGER_CONNECTIONS[fname], color, 2)
        for li in FINGER_LANDMARKS[fname]:
            cv2.circle(canvas, lm_to_pixel(lms.landmark[li]), 4, color, -1)

    wx, wy = lm_to_pixel(lms.landmark[0])
    cv2.circle(canvas, (wx, wy), 6, (220, 220, 220), -1)
    cv2.putText(canvas, label, (max(wx - 15, 2), max(wy - 12, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bar(canvas, x, y, w, h, value, max_val, color):
    fill = int(w * min(value / max_val, 1.0)) if max_val > 0 else 0
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 55), -1)
    if fill > 0:
        cv2.rectangle(canvas, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 95), 1)


def draw_sensor_badge(canvas, x, y, fs):
    connected = sensor_reader.connected
    color = (0, 200, 0) if connected else (0, 50, 255)
    text  = "IMU OK" if connected else "IMU OFFLINE"

    cv2.circle(canvas, (x + 10, y + 10), 7, color, -1)
    cv2.circle(canvas, (x + 10, y + 10), 7, (220, 220, 220), 1)
    cv2.putText(canvas, text, (x + 24, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(canvas,
                "N=%d  fs=%.1f Hz" % (sensor_reader.total_samples, fs),
                (x + 24, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 180), 1, cv2.LINE_AA)
    if not connected and sensor_reader.last_error:
        cv2.putText(canvas, "err: " + sensor_reader.last_error[:28],
                    (x + 24, y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 255), 1, cv2.LINE_AA)


def _draw_section_header(canvas, x, y, text, color):
    cv2.putText(canvas, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_hand_section(canvas, px, y, hlabel, tk):
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = "Mano Der." if hlabel == HAND_RIGHT else "Mano Izq."
    hdr  = (200, 200, 255) if hlabel == HAND_RIGHT else (255, 200, 255)

    cv2.putText(canvas, name, (px, y), font, 0.43, hdr, 1, cv2.LINE_AA)
    y += 15

    tc = (0, 255, 100) if tk.last_tap else (110, 110, 110)
    cv2.putText(canvas,
                f"Golpeteo: {'SI' if tk.last_tap else 'NO'}  Taps:{tk.tap_count}",
                (px + 6, y), font, 0.31, tc, 1, cv2.LINE_AA)
    y += 13

    oc = tk.last_oc or "---"
    oc_c = {OC_OPEN: (0, 255, 160), OC_CLOSED: (50, 120, 255),
            OC_PARTIAL: (255, 210, 0)}.get(oc, (110, 110, 110))
    cv2.putText(canvas, f"Mano: {oc}  Cambios:{tk.oc_count}",
                (px + 6, y), font, 0.31, oc_c, 1, cv2.LINE_AA)
    y += 13

    pr = tk.last_pron or "---"
    pr_c = {ROT_SUP: (0, 220, 255), ROT_PRON: (255, 160, 0),
            ROT_NEUTRO: (110, 110, 110)}.get(pr, (110, 110, 110))
    cv2.putText(canvas, f"Rot: {pr}  Cambios:{tk.pron_count}",
                (px + 6, y), font, 0.31, pr_c, 1, cv2.LINE_AA)
    return y + 16


def _draw_metrics_section(canvas, px, y, bw, metrics, individual):
    font = cv2.FONT_HERSHEY_SIMPLEX
    _draw_section_header(canvas, px, y, "Variables Sensor+Vision", (200, 200, 120))
    y += 14

    var_rows = (
        ("Frecuencia",  f"{metrics['frequency']:.2f} Hz",    'frequency',   "bradicinesia"),
        ("Amplitud",    f"{metrics['amplitude']:.3f}",       'amplitude',   "hipocinesia"),
        ("Vel.Angular", f"{metrics['angular_vel']:.1f} d/s", 'angular_vel', "lentitud"),
        ("CV",          f"{metrics['cv']:.1f} %",            'cv',          "irregularidad"),
        ("Jerk",        f"{metrics['jerk']:.2f} g/s",        'jerk',        "falta control"),
    )
    for vname, vstr, vkey, vcorr in var_rows:
        score = individual.get(vkey, 0.0)
        bar_col = (0, int((1 - score) * 190), int(score * 255))
        cv2.putText(canvas, f"{vname}: {vstr}", (px + 4, y), font, 0.30,
                    (180, 180, 180), 1, cv2.LINE_AA)
        tw = cv2.getTextSize(f"{vname}: {vstr}", font, 0.30, 1)[0][0]
        cv2.putText(canvas, vcorr, (px + 6 + tw + 4, y), font, 0.24,
                    (100, 100, 110), 1, cv2.LINE_AA)
        y += 9
        draw_bar(canvas, px + 4, y, bw, 6, score, 1.0, bar_col)
        y += 12
    return y


def _draw_score_section(canvas, px, y, bw, composite, updrs, ulabel):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = scorer.COLORS.get(updrs, (200, 200, 200))
    _draw_section_header(canvas, px, y, "Indice UPDRS", (255, 255, 200))
    y += 18
    cv2.putText(canvas, f"{updrs} - {ulabel}", (px + 4, y), font, 0.50,
                color, 2, cv2.LINE_AA)
    y += 14
    draw_bar(canvas, px + 4, y, bw, 11, composite, 1.0, color)
    cv2.putText(canvas, f"{composite * 100:.0f}%",
                (px + bw + 8, y + 9), font, 0.30,
                (150, 150, 150), 1, cv2.LINE_AA)
    return y + 20


def draw_panel(canvas, metrics, individual, composite, updrs, ulabel):
    x0   = VIDEO_W
    font = cv2.FONT_HERSHEY_SIMPLEX
    px   = x0 + 12
    bw   = PANEL_W - 40

    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, VIDEO_H), (25, 25, 30), -1)
    cv2.line(canvas, (x0, 0), (x0, VIDEO_H), (70, 70, 80), 2)

    y = 22
    cv2.putText(canvas, "UPDRS Parte 3", (px, y), font, 0.55,
                (255, 255, 120), 2, cv2.LINE_AA)
    y += 16
    cv2.putText(canvas, "Analisis Cuantitativo", (px, y), font, 0.32,
                (140, 140, 150), 1, cv2.LINE_AA)
    y += 8

    cv2.rectangle(canvas, (px - 4, y), (x0 + PANEL_W - 8, y + 52),
                  (15, 15, 20), -1)
    cv2.rectangle(canvas, (px - 4, y), (x0 + PANEL_W - 8, y + 52),
                  (60, 60, 70), 1)
    draw_sensor_badge(canvas, px, y + 2, metrics.get('fs', 0.0))
    y += 60

    for hlabel in (HAND_RIGHT, HAND_LEFT):
        tk = hand_trackers.get(hlabel)
        if tk:
            y = _draw_hand_section(canvas, px, y, hlabel, tk)

    cv2.line(canvas, (px, y), (x0 + PANEL_W - 12, y), (50, 50, 60), 1)
    y += 10

    y = _draw_metrics_section(canvas, px, y, bw, metrics, individual)
    y += 2
    cv2.line(canvas, (px, y), (x0 + PANEL_W - 12, y), (50, 50, 60), 1)
    y += 12

    y = _draw_score_section(canvas, px, y, bw, composite, updrs, ulabel)

    weight_txt = "pesos: " + " ".join(
        f"{k[:4]}:{int(v*100)}" for k, v in scorer.WEIGHTS.items())
    cv2.putText(canvas, weight_txt, (px + 4, y), font, 0.25,
                (110, 110, 120), 1, cv2.LINE_AA)

    y_bot = VIDEO_H - 32
    cv2.line(canvas, (px, y_bot), (x0 + PANEL_W - 12, y_bot), (50, 50, 60), 1)
    y_bot += 14
    cv2.putText(canvas, "[R] Reset   [Q] Salir   [P] Imprime metricas",
                (px, y_bot), font, 0.28, (130, 130, 130), 1, cv2.LINE_AA)


def draw_plots(canvas):
    plot_accel.draw(canvas, 0,   VIDEO_H)
    plot_gyro.draw (canvas, 366, VIDEO_H)
    plot_vision.draw(canvas, 732, VIDEO_H)


# =====================================================================
#  MAIN LOOP
# =====================================================================
print("=" * 60)
print("  UPDRS Parte 3 - Analisis Cuantitativo")
print("  ESP32-CAM + MPU6050")
print("  Frame : " + FRAME_URL)
print("  Sensor: " + SENSOR_URL)
print("  Teclas: Q=salir  R=reset  P=imprime metricas")
print("=" * 60)

sensor_reader.start()

# Session reusable para /frame (keep-alive => ahorra TCP setup en cada frame).
frame_session = requests.Session()
frame_session.headers.update({"Connection": "keep-alive"})

fps_ts = time.time(); fps_n = 0; fps_val = 0.0
last_metrics_log = time.time()

while True:
    frame_data = get_frame(FRAME_URL, frame_session)
    if not frame_data:
        time.sleep(0.2)
        continue

    try:
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        frame = None
    if frame is None:
        continue

    # IMU: pasa cada raw por el preprocesador antes de almacenar.
    imu_samples = sensor_reader.get_samples()
    if imu_samples:
        for s in imu_samples:
            if len(s) < 7:
                continue
            processed = imu_pre.process(s[0], s[1], s[2], s[3], s[4], s[5], s[6])
            sig_proc.add_processed_sample(processed)
            ax, ay, az, gx, gy, gz, _ = processed
            plot_accel.add(math.sqrt(ax * ax + ay * ay + az * az))
            plot_gyro.add (math.sqrt(gx * gx + gy * gy + gz * gz))

    # Vision
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:VIDEO_H, :VIDEO_W] = cv2.resize(frame, (VIDEO_W, VIDEO_H))

    if results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            hlabel = results.multi_handedness[idx].classification[0].label
            tk = hand_trackers.setdefault(hlabel, HandTracker())

            tid = thumb_index_normalized(hand_lm)
            sig_proc.add_vision_sample(tid)
            plot_vision.add(tid)

            tk.update_tap(detect_tapping(hand_lm), sig_proc)
            tk.update_oc(detect_open_close(hand_lm))
            tk.update_pron(detect_pronation(hand_lm, hlabel))

            draw_landmarks(canvas, hand_lm, hlabel)

    # Una sola compute_all() por frame (cacheada a 4Hz).
    metrics = sig_proc.compute_all()
    individual, composite, updrs, ulabel = scorer.compute(metrics)

    draw_panel(canvas, metrics, individual, composite, updrs, ulabel)
    draw_plots(canvas)

    # FPS
    fps_n += 1
    now = time.time()
    if now - fps_ts >= 1.0:
        fps_val = fps_n / (now - fps_ts)
        fps_ts, fps_n = now, 0
    cv2.putText(canvas, f"FPS: {fps_val:.1f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    if now - last_metrics_log >= 3.0:
        last_metrics_log = now
        print("[MET] fs=%.1fHz  freq=%.2fHz  amp=%.3f  omega=%.1fd/s  CV=%.1f%%  jerk=%.2fg/s"
              % (metrics['fs'], metrics['frequency'], metrics['amplitude'],
                 metrics['angular_vel'], metrics['cv'], metrics['jerk']))

    cv2.imshow("UPDRS - Analisis Cuantitativo", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        hand_trackers.clear()
        sig_proc.reset()
        imu_pre.reset()
        plot_accel.data.clear()
        plot_gyro.data.clear()
        plot_vision.data.clear()
        print("[RESET] Contadores, buffers, preprocesador y graficas reiniciados")
    elif key == ord('p'):
        m = sig_proc.compute_all(force=True)
        ind, comp, upd, lab = scorer.compute(m)
        print("\n" + "=" * 50)
        print("  METRICAS ACTUALES")
        print("=" * 50)
        print(f"  fs efectiva     = {m['fs']:.2f} Hz")
        print(f"  Muestras IMU    = {sensor_reader.total_samples}")
        for k in ('frequency', 'amplitude', 'angular_vel', 'cv', 'jerk'):
            print(f"  {k:12s}    = {m[k]:.3f}  -> score {ind[k]:.2f}")
        print(f"  Indice compuesto = {comp:.3f}")
        print(f"  UPDRS            = {upd} ({lab})")
        print("=" * 50)

# Cleanup
sensor_reader.stop()
cv2.destroyAllWindows()
hands.close()
