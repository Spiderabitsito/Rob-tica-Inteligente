"""
UPDRS Parte 3 - Analisis Cuantitativo de Movimientos
=====================================================
Fusion ESP32-S3 CAM (vision + MediaPipe) + MPU6050 (IMU via I2C),
sensor montado en la falange distal del dedo indice.

Metricas:
  Primarias (clinicas):
    UPDRS grade        0..4    indice compuesto, escala log
    In_a (acelerom.)   x veces baseline saludable (Sousa Paixao 2019)
    In_g (giroscopio)  x veces baseline saludable

  Secundarias (diagnostico):
    Frecuencia (Hz)    pico FFT en 3-7 Hz (banda del tremor parkinsoniano)
    Amplitud vision    rango P95-P5 distancia pulgar-indice
    RMS-a (g)          RMS de aceleracion dinamica en 5s
    CV (%)             coef. variacion intervalos entre taps
    Jerk (g/s)         derivada de aceleracion dinamica

Pipeline IMU:
  raw -> median(3) -> IIR LP 15Hz -> comp.filter HP 1Hz gravedad -> a_dynamic
  RMS-a en ventana 5s, |a_dyn| con gravedad ya removida
  FFT con resample uniforme + detrend + Hanning, banda 0.5-8 Hz

Calibracion (OBLIGATORIA primera vez):
  Tecla [C] graba 10s con sensor inmovil en mano sana -> baseline MPS/STDPS
  Persistido en ~/.updrs_baseline.json con metadatos del setup

Atajos UI:
  Q  salir             G  toggle plots crudos (default oculto)
  R  reset contadores  F  toggle FFT spectrum (debug)
  P  imprime metricas  C  calibrar baseline saludable
"""

import os
# Silencia warnings de JPEG corrupto (libjpeg) - debe ir antes de importar cv2
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import datetime
import json
import math
import numpy as np
import requests
import mediapipe as mp
import time
import threading
from collections import deque
from PIL import Image, ImageDraw, ImageFont


# =====================================================================
#  CONFIGURACION + CONSTANTES
# =====================================================================
ESP32_IP   = "192.168.4.1"
FRAME_URL  = f"http://{ESP32_IP}/frame"
SENSOR_URL = f"http://{ESP32_IP}/sensor"
STATUS_URL = f"http://{ESP32_IP}/status"

# Schema key del payload /sensor; cross-reference: esp32_stream.py:IMU_PAYLOAD_KEY.
IMU_PAYLOAD_KEY = "s"

# Layout: video grande + panel lateral ancho. Plots crudos se muestran
# bajo el video solo si el usuario presiona G (default oculto).
VIDEO_W, VIDEO_H = 960, 720
PANEL_W          = 380
PLOT_H           = 200            # altura de la franja de plots cuando esta visible
CANVAS_W         = VIDEO_W + PANEL_W            # 1340
CANVAS_H_PLOTS   = VIDEO_H + PLOT_H              # con plots: 920
CANVAS_H_NOPLOTS = VIDEO_H                       # sin plots: 720

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

# Bandpass del pipeline IMU (Brainstorm de paper review: 1-15 Hz para
# falange distal evita ruido HF y drift de baja frec.).
BP_LOW_HZ  = 1.0
BP_HIGH_HZ = 15.0

# Ventana de RMS para In_a / In_g (Sousa Paixao 2019, 5 s).
RMS_WINDOW_S = 5.0

# Si pasan mas de OFFLINE_AFTER_S sin recibir muestras, marca IMU OFFLINE.
# Sin esto, la badge parpadea cada poll vacio normal (~ cada 80 ms).
OFFLINE_AFTER_S = 1.5

# Cache de metricas: recompute mas alla de este intervalo.
METRICS_CACHE_S = 0.25   # 4 Hz - mas que suficiente para UI a 30 FPS.

# Log a terminal cada N muestras IMU recibidas.
DEBUG_PRINT_EVERY = 50

# Calibracion: 10 s minimo (paper review recomendo >= 10 s para stdev estable).
CALIBRATION_DURATION_S = 10.0
BASELINE_PATH = os.path.expanduser("~/.updrs_baseline.json")


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
#  TEXT RENDERER (Pillow + TTF -> texto antialiased nitido)
# =====================================================================
class TextRenderer:
    """
    cv2.putText con FONT_HERSHEY_* dibuja con strokes lineales sin
    antialias verdadero -> el texto se ve "144p" como reporto el usuario.
    Esta clase usa Pillow + TrueType para renderizado nitido.

    Patron: encolar varios draws por frame y aplicarlos en UNA pasada
    (cada conversion ndarray<->PIL es costosa, ~3-5 ms en 1340x920).
    """

    # Buscar fuentes en orden: Windows -> macOS -> Linux. Cae a default si nada.
    FONT_CANDIDATES = (
        "C:/Windows/Fonts/segoeui.ttf",          # Segoe UI Regular
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    FONT_BOLD_CANDIDATES = (
        "C:/Windows/Fonts/segoeuib.ttf",         # Segoe UI Bold
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )

    def __init__(self):
        self._regular_path = self._find_font(self.FONT_CANDIDATES)
        self._bold_path    = self._find_font(self.FONT_BOLD_CANDIDATES)
        self._cache        = {}                  # (size, bold) -> ImageFont
        self._queue        = []                  # list[(x, y, text, size, color, bold)]

    @staticmethod
    def _find_font(candidates):
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def font(self, size, bold=False):
        key = (size, bold)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._bold_path if bold else self._regular_path
        try:
            font = (ImageFont.truetype(path, size) if path
                    else ImageFont.load_default())
        except Exception:
            font = ImageFont.load_default()
        self._cache[key] = font
        return font

    def measure(self, text, size, bold=False):
        font = self.font(size, bold)
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def queue(self, x, y, text, size=14, color=(220, 220, 220), bold=False):
        # color es BGR (convencion cv2). Lo guardamos asi y convertimos al flush.
        self._queue.append((x, y, text, size, color, bold))

    def flush(self, canvas):
        """Aplica todos los draws encolados en una sola pasada PIL."""
        if not self._queue:
            return
        img_pil = Image.fromarray(canvas[:, :, ::-1])    # BGR -> RGB
        draw = ImageDraw.Draw(img_pil)
        for x, y, text, size, color, bold in self._queue:
            # color BGR -> RGB para PIL
            rgb = (int(color[2]), int(color[1]), int(color[0]))
            draw.text((x, y), text, fill=rgb, font=self.font(size, bold))
        canvas[:] = np.asarray(img_pil)[:, :, ::-1]      # RGB -> BGR
        self._queue.clear()


# Singleton (instanciado tras la primera funcion que lo necesite, ver mas abajo).
text = TextRenderer()


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
        # Marco
        cv2.rectangle(canvas, (x, y), (x + self.width, y + self.height),
                      (18, 18, 25), -1)
        cv2.rectangle(canvas, (x, y), (x + self.width, y + self.height),
                      (70, 70, 85), 1)
        # Titulo y unidad - se encolan al TextRenderer global y se aplican
        # con el flush del frame (consistencia tipografica con el panel).
        text.queue(x + 10, y + 6,  self.title, size=12,
                   color=(220, 220, 230), bold=True)
        text.queue(x + self.width - 60, y + 8, self.unit, size=10,
                   color=(150, 150, 160))

        if len(self.data) < 2:
            text.queue(x + 10, y + self.height // 2 - 6,
                       "(esperando datos...)", size=11,
                       color=(120, 120, 130))
            return

        arr = np.fromiter(self.data, dtype=np.float32, count=len(self.data))

        if self.y_range:
            y_min, y_max = self.y_range
        else:
            y_min = float(arr.min())
            y_max = float(arr.max())
        if y_max - y_min < 1e-6:
            y_max = y_min + 1e-6

        plot_top    = y + 26
        plot_bottom = y + self.height - 6
        plot_left   = x + 8
        plot_right  = x + self.width - 70
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

        text.queue(plot_left + 2, plot_top - 2,  f"{y_max:+.2f}",
                   size=10, color=(130, 130, 140))
        text.queue(plot_left + 2, plot_bottom - 12, f"{y_min:+.2f}",
                   size=10, color=(130, 130, 140))
        text.queue(plot_right + 6, plot_top + plot_h // 2 - 6,
                   f"{arr[-1]:+.2f}", size=14, color=self.color, bold=True)


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

    Coeficientes (fs=100Hz, banda 1-15 Hz - brainstorm-validated):
      LP_ALPHA   = 0.485  -> cutoff ~14.6 Hz (formula correcta: arccos(...)
                              sobre IIR de un polo; arriba de 15 Hz en
                              falange distal solo hay ruido HF)
      GRAV_ALPHA = 0.94   -> cutoff ~1.0 Hz (mata drift lento que sesgaba
                              el RMS hacia arriba; antes era 0.5 Hz)

    NOTA cientifica: la implementacion son DOS IIR de UN polo (6 dB/oct
    cada uno), NO un Butterworth de orden 4 (24 dB/oct) como en Sousa
    Paixao 2019 / Keba 2025. La transicion es mas blanda. Razones:
      - menor overhead computacional online (un MAC por sample)
      - sin fase de "warmup" para filtfilt
      - la banda de tremor (3-7 Hz) queda bien aislada igual al estar
        lejos de los cutoffs (1 y 15 Hz)
    Si se quiere fidelidad estricta al paper, cambiar a scipy.signal
    butter+filtfilt (requiere acumular ventana antes de filtrar).
    """

    def __init__(self, lp_alpha=0.485, grav_alpha=0.94, gyro_offset=None):
        self.LP_ALPHA   = lp_alpha
        self.GRAV_ALPHA = grav_alpha
        self.gyro_offset = (np.zeros(3) if gyro_offset is None
                            else np.asarray(gyro_offset, dtype=np.float64))
        self._init_state()

    def _init_state(self):
        # IIR low-pass state (per axis).
        self._lp_a = np.zeros(3)
        self._lp_g = np.zeros(3)
        # 3-tap median ring buffer per axis (3 axes x 3 samples).
        self._med_a = np.zeros((3, 3))
        self._med_g = np.zeros((3, 3))
        self._med_idx = 0
        self._n_seen = 0
        # Gravity tracked via complementary filter; bootstrap on first sample.
        self._gravity = None

    def reset(self):
        self._init_state()

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
#  INDEX CALCULATOR  (Sousa Paixao 2019 Tremor Normality Index)
# =====================================================================
class IndexCalculator:
    """
    In_j = (rms_j - MPS) / STDPS
      rms_j = RMS del modulo de aceleracion dinamica (o giroscopio)
              en ventana de 5 s para sujeto j
      MPS   = media   poblacional sana (calibrada por usuario)
      STDPS = desv. estandar poblacional sana

    Interpretacion: In = 1 -> tremor a 1 sigma del baseline saludable;
    In = 8 -> 8x sigma (paciente Parkinson tipico segun Sousa Paixao 2019).

    Calibracion obligatoria primera vez (paper review):
      - >= 10 s con sensor inmovil en mano sana (>=500 muestras a 100Hz)
      - guarda site/fs/banda en JSON para auto-invalidar si cambia setup

    Reference values en paper son para wrist; nuestro setup es falange
    distal del indice (~1.5-2.5x mas amplitud por palanca). Valores
    default son placeholders para no crashear; refuse_score=True hasta
    que el usuario presione [C].

    DESVIACIONES CONOCIDAS vs Sousa Paixao 2019 (declaradas para no
    enganar al lector cientifico):

    1) STDPS practical vs paper:
       El paper computa MPS, STDPS sobre la distribucion de RMS POR
       SUJETO en una poblacion sana de N personas. Aca calibramos con
       UN solo sujeto (1 mano sana), tomando MPS = RMS de la traza
       continua y STDPS = stdev de las muestras |a_dyn| dentro de la
       traza. Bajo hipotesis de ergodicidad (mano realmente quieta) MPS
       converge bien, pero STDPS difiere por ~sqrt(N) y mide stdev
       intra-sujeto en vez de entre-sujetos. Por eso los thresholds
       (1.5, 5, 12, 25) en UPDRSScorer son empiricos y NO portados
       directos del paper. Para uso clinico cuantitativo, calibrar con
       una cohorte multi-sujeto y reemplazar el contenido de baseline.

    2) max(0, In):
       El paper permite In negativo cuando un sujeto es "mas quieto"
       que la media sana. Aqui clip en 0 por simplicidad de UI (el
       UPDRS ya tiene 0 como suelo). Para reportes cientificos quitar
       el clip en los logs/CSV.

    3) Bandpass:
       Paper usa Butterworth orden 4 (24 dB/oct) en 1-50 Hz. Nosotros
       usamos un par IIR de 1 polo (6 dB/oct) en 1-15 Hz, mas blando
       en transicion pero mejor adaptado a la falange distal donde
       arriba de 15 Hz solo hay ruido HF. Ver IMUPreprocessor.
    """

    SCHEMA_VERSION = 1

    DEFAULT_BASELINE = {
        "schema_version": SCHEMA_VERSION,
        "site": "distal_index",
        "fs_hz": TARGET_FS_HZ,
        "bp_low_hz": BP_LOW_HZ,
        "bp_high_hz": BP_HIGH_HZ,
        # Placeholders (no calibrado todavia)
        "mps_a_g":     0.012,
        "stdps_a_g":   0.005,
        "mps_g_dps":   1.20,
        "stdps_g_dps": 0.50,
        "n_samples":   0,
        "duration_s":  0.0,
        "timestamp":   "",
    }

    def __init__(self, path=BASELINE_PATH):
        self.path = path
        self.baseline = self._load()
        self.in_a = 0.0
        self.in_g = 0.0
        self.rms_a = 0.0
        self.rms_g = 0.0
        # Modo calibracion (set por main): mientras este True, acumula muestras.
        self.cal_active = False
        self.cal_started_at = 0.0
        self.cal_buf_a = []           # list[float] |a_dyn|
        self.cal_buf_g = []           # list[float] |w|

    # ---- persistence ----

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Sanity check: solo acepta si la version y el setup coinciden.
            if (data.get("schema_version") == self.SCHEMA_VERSION
                    and data.get("site") == self.DEFAULT_BASELINE["site"]
                    and abs(data.get("fs_hz", 0) - TARGET_FS_HZ) < 1
                    and abs(data.get("bp_low_hz", 0)  - BP_LOW_HZ)  < 0.1
                    and abs(data.get("bp_high_hz", 0) - BP_HIGH_HZ) < 0.1):
                return {**self.DEFAULT_BASELINE, **data}
            print("[CAL] Baseline existente con setup distinto - ignorado.")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[CAL] No se pudo leer {self.path}: {e}")
        return dict(self.DEFAULT_BASELINE)

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.baseline, f, indent=2)
            print(f"[CAL] Baseline guardado en {self.path}")
        except Exception as e:
            print(f"[CAL] No se pudo guardar baseline: {e}")

    @property
    def is_calibrated(self):
        return self.baseline.get("n_samples", 0) >= 500   # ~5s @ 100Hz minimo

    # ---- compute on every metrics tick ----

    def update(self, a_arr_x, a_arr_y, a_arr_z, g_arr_x, g_arr_y, g_arr_z, t_ms):
        """Recompute In_a, In_g sobre la ventana mas reciente RMS_WINDOW_S."""
        if len(a_arr_x) < 50:
            self.in_a = 0.0; self.in_g = 0.0
            self.rms_a = 0.0; self.rms_g = 0.0
            return

        # Slice ventana de RMS_WINDOW_S desde el final.
        t_s = t_ms / 1000.0
        cutoff = t_s[-1] - RMS_WINDOW_S
        mask = t_s >= cutoff
        if mask.sum() < 50:
            return

        a_mag = np.sqrt(a_arr_x[mask] ** 2 + a_arr_y[mask] ** 2 + a_arr_z[mask] ** 2)
        g_mag = np.sqrt(g_arr_x[mask] ** 2 + g_arr_y[mask] ** 2 + g_arr_z[mask] ** 2)

        self.rms_a = float(np.sqrt(np.mean(a_mag ** 2)))
        self.rms_g = float(np.sqrt(np.mean(g_mag ** 2)))

        b = self.baseline
        # In = (rms - MPS) / STDPS, clamp en 0 (un sujeto puede tener menos
        # tremor que la media saludable -> reporta 0, no negativo).
        self.in_a = max(0.0, (self.rms_a - b["mps_a_g"])   / max(b["stdps_a_g"],   1e-6))
        self.in_g = max(0.0, (self.rms_g - b["mps_g_dps"]) / max(b["stdps_g_dps"], 1e-6))

    # ---- calibration mode ----

    def start_calibration(self):
        self.cal_active = True
        self.cal_started_at = time.time()
        self.cal_buf_a = []
        self.cal_buf_g = []
        print(f"[CAL] Iniciada. Mantener sensor INMOVIL en mano SANA "
              f"durante {CALIBRATION_DURATION_S:.0f} s ...")

    def feed_calibration(self, a_arr_x, a_arr_y, a_arr_z,
                         g_arr_x, g_arr_y, g_arr_z):
        """Acumula |a_dyn| y |w| del ultimo batch durante la calibracion."""
        if not self.cal_active:
            return
        a_mag = np.sqrt(a_arr_x ** 2 + a_arr_y ** 2 + a_arr_z ** 2)
        g_mag = np.sqrt(g_arr_x ** 2 + g_arr_y ** 2 + g_arr_z ** 2)
        self.cal_buf_a.extend(a_mag.tolist())
        self.cal_buf_g.extend(g_mag.tolist())
        if time.time() - self.cal_started_at >= CALIBRATION_DURATION_S:
            self.finish_calibration()

    def finish_calibration(self):
        self.cal_active = False
        # Mismo umbral que is_calibrated: si guardamos un baseline con menos
        # de 500 muestras, is_calibrated lo rechazaria al cargarlo despues.
        if len(self.cal_buf_a) < 500:
            print(f"[CAL] FALLIDA: solo {len(self.cal_buf_a)} muestras "
                  f"(minimo 500 = ~5 s). Verifica que el sensor transmita.")
            return
        a = np.asarray(self.cal_buf_a, dtype=np.float64)
        g = np.asarray(self.cal_buf_g, dtype=np.float64)

        # MPS: RMS de la senal en reposo (= ruido de fondo + micro-tremor sano)
        # STDPS: stdev de la magnitud (variabilidad esperada en sano)
        mps_a   = float(np.sqrt(np.mean(a * a)))
        stdps_a = float(np.std(a))
        mps_g   = float(np.sqrt(np.mean(g * g)))
        stdps_g = float(np.std(g))

        self.baseline.update({
            "schema_version": self.SCHEMA_VERSION,
            "site":          "distal_index",
            "fs_hz":         TARGET_FS_HZ,
            "bp_low_hz":     BP_LOW_HZ,
            "bp_high_hz":    BP_HIGH_HZ,
            "mps_a_g":       mps_a,
            "stdps_a_g":     max(stdps_a, 1e-4),
            "mps_g_dps":     mps_g,
            "stdps_g_dps":   max(stdps_g, 1e-2),
            "n_samples":     len(a),
            "duration_s":    float(len(a) / TARGET_FS_HZ),
            "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
        })
        self._save()
        print(f"[CAL] Baseline OK: MPS_a={mps_a:.4f}g  STDPS_a={stdps_a:.4f}g"
              f"  MPS_g={mps_g:.2f}d/s  STDPS_g={stdps_g:.2f}d/s"
              f"  ({len(a)} muestras)")

    def cal_progress(self):
        if not self.cal_active:
            return 0.0
        return min(1.0, (time.time() - self.cal_started_at) / CALIBRATION_DURATION_S)


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
        self._init_buffers()

    def _init_buffers(self):
        m = self._max
        self.ax = deque(maxlen=m)
        self.ay = deque(maxlen=m)
        self.az = deque(maxlen=m)
        self.gx = deque(maxlen=m)
        self.gy = deque(maxlen=m)
        self.gz = deque(maxlen=m)
        self.t  = deque(maxlen=m)
        self.thumb_idx_dist = deque(maxlen=m)
        self.tap_intervals  = deque(maxlen=100)
        self._cache = None
        self._cache_t = 0.0

    def add_processed_sample(self, sample7):
        sx, sy, sz, rx, ry, rz, ts = sample7
        self.ax.append(sx); self.ay.append(sy); self.az.append(sz)
        self.gx.append(rx); self.gy.append(ry); self.gz.append(rz)
        self.t.append(ts)

    def add_vision_sample(self, normalized):
        self.thumb_idx_dist.append(normalized)

    def add_tap_interval(self, interval_sec):
        self.tap_intervals.append(interval_sec)

    def reset(self):
        self._init_buffers()

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

    def compute_all(self, index_calc=None, force=False):
        """
        index_calc: instancia de IndexCalculator. Si se pasa, se actualiza
        In_a/In_g y, si esta en modo calibracion, recibe las muestras del
        ultimo batch para acumularlas.
        """
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

        # Actualizar In_a / In_g (Sousa Paixao 2019) y alimentar calibracion.
        if index_calc is not None and len(ax):
            index_calc.update(ax, ay, az, gx, gy, gz, t_ms)
            if index_calc.cal_active:
                # En modo cal solo nos importan las muestras de los ultimos
                # ~METRICS_CACHE_S; las anteriores ya se acumularon.
                if len(t_ms) >= 1:
                    cutoff = t_ms[-1] - METRICS_CACHE_S * 1000.0
                    m = t_ms >= cutoff
                    if m.sum() > 0:
                        index_calc.feed_calibration(ax[m], ay[m], az[m],
                                                    gx[m], gy[m], gz[m])

        result = {
            'frequency':   self._compute_frequency(ax, ay, az, t_ms),
            'amplitude':   self._compute_amplitude(),
            'angular_vel': self._compute_angular_velocity(gx, gy, gz),
            'cv':          self._compute_cv(),
            'jerk':        self._compute_jerk(ax, ay, az, t_ms),
            'fs':          fs,
            # Indices clinicos (paper 3); 0 si index_calc=None.
            'in_a':        index_calc.in_a if index_calc else 0.0,
            'in_g':        index_calc.in_g if index_calc else 0.0,
            'rms_a':       index_calc.rms_a if index_calc else 0.0,
            'rms_g':       index_calc.rms_g if index_calc else 0.0,
        }
        self._cache = result
        self._cache_t = now
        return result


# =====================================================================
#  UPDRS SCORER (refactor a partir del brainstorm con paper review)
# =====================================================================
class UPDRSScorer:
    """
    Cambios clave vs version anterior (que se quedaba pegada en 2-3):

      1. **In_a (Sousa Paixao) es el driver primario** (peso 50%).
         Es la unica metrica calibrada contra una poblacion de referencia,
         asi que evita el sesgo de min/max de sesion.

      2. **Composite leaner**: In_a + freq + cv + jerk. Se quita amplitud
         vision del composite (solo display) porque depende de si el
         usuario esta haciendo finger-tap o no - no es severidad.

      3. **Escala log para In_a** (Weber-Fechner, Keba 2025):
         log(rms) ~ UPDRS_grade. Se mapea directamente:
            In_a < 1.5  -> grade 0  (within 1.5 sigma del baseline sano)
            1.5 <= 5    -> grade 1  (slight)
            5 <= 12     -> grade 2  (mild)
            12 <= 25    -> grade 3  (moderate)
            > 25        -> grade 4  (severe)

      4. **Refuse_score** mientras IndexCalculator no este calibrado:
         devolvemos UPDRS=None y que la UI lo muestre como "Calibrar [C]".

      5. **SNR gate**: si rms_a < 1.5 * MPS_a, reportar "Reposo" en vez
         de UPDRS=0 (no es lo mismo "no hay tremor" que "no hay senal").
    """

    # Thresholds aplicados a In_a directamente.
    # Sousa Paixao 2019 reporta healthy mean ~1, PD sin carga ~8 (accel).
    #   1.5 (limite 0/1): defensible (>1.5 sigma del baseline sano)
    #   5.0 (1/2):        un poco bajo del PD-mean del paper, agresivo
    #   12 y 25 (2/3, 3/4): EXTRAPOLADOS por nosotros, no estan en el
    #                       paper. Refinar contra cohorte propia cuando
    #                       haya pacientes graduados por neurologo.
    IN_A_GRADE_THRESHOLDS = (1.5, 5.0, 12.0, 25.0)

    # Pesos del composite (suman 1.0). In_a domina porque es la unica
    # metrica calibrada contra poblacion sana.
    WEIGHTS = {
        'in_a':      0.50,
        'frequency': 0.15,
        'cv':        0.25,
        'jerk':      0.10,
    }

    # Rangos para metricas secundarias (escala log para freq y jerk).
    # frequency: normal 0 (sin pico en 3-7 Hz), severe 6 (pico fuerte en banda PD)
    # cv:        normal 5%, severe 50%
    # jerk:      normal 0.5 g/s, severe 30 g/s (escala log)
    RANGES = {
        'frequency': {'normal': 0.5, 'severe': 6.0},   # Hz
        'cv':        {'normal': 5.0, 'severe': 50.0},  # %
        'jerk':      {'normal': 0.5, 'severe': 30.0},  # g/s
    }

    LABELS = {0: "Normal", 1: "Leve", 2: "Moderado leve",
              3: "Moderado", 4: "Severo"}
    COLORS = {0: (0, 200, 0),   1: (0, 210, 170), 2: (0, 210, 210),
              3: (0, 140, 255), 4: (0, 50, 255)}

    @staticmethod
    def _normalize_in_a(in_a):
        """Mapea In_a -> [0,1] usando thresholds clinicos discretos."""
        th = UPDRSScorer.IN_A_GRADE_THRESHOLDS
        if in_a < th[0]:               return 0.0
        if in_a < th[1]:               return 0.25
        if in_a < th[2]:               return 0.50
        if in_a < th[3]:               return 0.75
        return 1.0

    @staticmethod
    def _normalize_log(value, normal, severe):
        """log10 normalize entre normal y severe; clamp en [0,1].

        Weber-Fechner: la severidad de tremor es logaritmica con la
        magnitud de la senal. Esto evita que un solo outlier rail-pegue
        la metrica a 1 (problema del normalize lineal anterior)."""
        if value <= normal:
            return 0.0
        if value >= severe:
            return 1.0
        # log space mapping
        log_v = math.log10(max(value, 1e-9))
        log_n = math.log10(max(normal, 1e-9))
        log_s = math.log10(max(severe, 1e-9))
        return max(0.0, min(1.0, (log_v - log_n) / (log_s - log_n)))

    def compute(self, metrics, index_calibrated=True):
        """
        index_calibrated=False -> devuelve UPDRS=None y label="Calibrar [C]";
        no se publica un score sin baseline porque seria meaningless.
        """
        individual = {
            'in_a':      self._normalize_in_a(metrics.get('in_a', 0.0)),
            'frequency': self._normalize_log(metrics.get('frequency', 0.0),
                                             self.RANGES['frequency']['normal'],
                                             self.RANGES['frequency']['severe']),
            'cv':        self._normalize_log(metrics.get('cv', 0.0),
                                             self.RANGES['cv']['normal'],
                                             self.RANGES['cv']['severe']),
            'jerk':      self._normalize_log(metrics.get('jerk', 0.0),
                                             self.RANGES['jerk']['normal'],
                                             self.RANGES['jerk']['severe']),
        }

        if not index_calibrated:
            return individual, 0.0, None, "Calibrar [C]"

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
index_calc    = IndexCalculator()
sensor_reader = SensorReader(SENSOR_URL, interval=0.05)

# Plots crudos (default ocultos; toggle con [G]).
# Width: 3 plots iguales sumando CANVAS_W (1340 / 3 ~= 446 c/u).
_PLOT_W = CANVAS_W // 3
plot_accel  = ScrollingPlot(_PLOT_W, PLOT_H,
                            "Aceleracion |a_dyn| (sin gravedad)",
                            "g",     (120, 220, 120), n_points=300)
plot_gyro   = ScrollingPlot(_PLOT_W, PLOT_H,
                            "Velocidad angular |w|",
                            "deg/s", (120, 220, 255), n_points=300)
plot_vision = ScrollingPlot(CANVAS_W - 2 * _PLOT_W, PLOT_H,
                            "Distancia pulgar-indice (vision)",
                            "norm",  (255, 220, 120), n_points=300)

# Estado de UI (toggleable por teclas).
ui_state = {
    "show_plots": False,    # [G]
    "show_fft":   False,    # [F] - placeholder, FFT debug pendiente
}


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
    for start_idx, end_idx in segments:
        pa = lm_to_pixel(lms.landmark[start_idx])
        pb = lm_to_pixel(lms.landmark[end_idx])
        cv2.line(canvas, pa, pb, color, thickness)


def draw_landmarks(canvas, lms, label):
    colors = RIGHT_COLORS if label == HAND_RIGHT else LEFT_COLORS

    _draw_segments(canvas, lms, PALM_CONNECTIONS, (180, 180, 180), 2)

    for fname, color in colors.items():
        _draw_segments(canvas, lms, FINGER_CONNECTIONS[fname], color, 2)
        for li in FINGER_LANDMARKS[fname]:
            cv2.circle(canvas, lm_to_pixel(lms.landmark[li]), 4, color, -1)

    wx, wy = lm_to_pixel(lms.landmark[0])
    cv2.circle(canvas, (wx, wy), 6, (220, 220, 220), -1)
    text.queue(max(wx - 18, 2), max(wy - 24, 4), label,
               size=12, color=(255, 255, 255), bold=True)


def draw_bar(canvas, x, y, w, h, value, max_val, color):
    fill = int(w * min(value / max_val, 1.0)) if max_val > 0 else 0
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 55), -1)
    if fill > 0:
        cv2.rectangle(canvas, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 95), 1)


def draw_sensor_badge(canvas, x, y, fs):
    """Pildora con LED + etiqueta IMU OK/OFFLINE + N muestras."""
    connected = sensor_reader.connected
    led = (0, 200, 0) if connected else (0, 50, 255)
    label = "IMU OK" if connected else "IMU OFFLINE"

    # Fondo de la pildora
    cv2.rectangle(canvas, (x, y), (x + PANEL_W - 24, y + 40),
                  (15, 15, 20), -1)
    cv2.rectangle(canvas, (x, y), (x + PANEL_W - 24, y + 40),
                  (60, 60, 70), 1)
    # LED
    cv2.circle(canvas, (x + 14, y + 20), 8, led, -1)
    cv2.circle(canvas, (x + 14, y + 20), 8, (220, 220, 220), 1)
    # Texto via Pillow (nitido)
    text.queue(x + 32, y + 6,  label, size=18, color=led, bold=True)
    text.queue(x + 32, y + 24, f"N={sensor_reader.total_samples}  fs={fs:.1f} Hz",
               size=12, color=(170, 170, 180))
    if not connected and sensor_reader.last_error:
        text.queue(x + 32, y + 24,
                   "err: " + sensor_reader.last_error[:24],
                   size=11, color=(120, 120, 255))


def _draw_section_header(x, y, label, color):
    text.queue(x, y, label, size=14, color=color, bold=True)


def _draw_hand_section(canvas, px, y, hlabel, tk):
    """Card por mano: detected + tap/oc/pron en una linea limpia."""
    name = "Mano Derecha" if hlabel == HAND_RIGHT else "Mano Izquierda"
    hdr  = (200, 200, 255) if hlabel == HAND_RIGHT else (255, 200, 255)
    text.queue(px, y, name, size=14, color=hdr, bold=True)
    y += 22

    tc = (0, 255, 100) if tk.last_tap else (140, 140, 140)
    text.queue(px + 8, y,
               f"Tap {'ON' if tk.last_tap else 'off'}    {tk.tap_count} toques",
               size=12, color=tc)
    y += 18

    oc = tk.last_oc or "---"
    oc_c = {OC_OPEN: (0, 255, 160), OC_CLOSED: (50, 120, 255),
            OC_PARTIAL: (255, 210, 0)}.get(oc, (140, 140, 140))
    text.queue(px + 8, y, f"Mano: {oc}    {tk.oc_count} cambios",
               size=12, color=oc_c)
    y += 18

    pr = tk.last_pron or "---"
    pr_c = {ROT_SUP: (0, 220, 255), ROT_PRON: (255, 160, 0),
            ROT_NEUTRO: (140, 140, 140)}.get(pr, (140, 140, 140))
    text.queue(px + 8, y, f"Rot: {pr}    {tk.pron_count} cambios",
               size=12, color=pr_c)
    return y + 22


def _draw_in_indices(canvas, px, y, bw, metrics, calibrated):
    """Tarjeta destacada del Tremor Normality Index (Sousa Paixao 2019).

    Es la metrica clinica primaria; va GRANDE.
    """
    # Marco
    h = 100
    cv2.rectangle(canvas, (px - 4, y), (px + bw + 4, y + h),
                  (18, 22, 28), -1)
    cv2.rectangle(canvas, (px - 4, y), (px + bw + 4, y + h),
                  (60, 80, 100), 1)
    text.queue(px + 4, y + 4, "Tremor Normality Index", size=12,
               color=(150, 200, 255), bold=True)

    in_a = metrics.get('in_a', 0.0)
    in_g = metrics.get('in_g', 0.0)

    if not calibrated:
        text.queue(px + 4, y + 28,
                   "Sin calibrar", size=18,
                   color=(255, 180, 100), bold=True)
        text.queue(px + 4, y + 56,
                   "Pulsa  C  con sensor inmovil",
                   size=12, color=(180, 180, 200))
        text.queue(px + 4, y + 74,
                   "en mano sana ~10 s",
                   size=12, color=(180, 180, 200))
        return y + h + 10

    # Color escala: verde (~1) -> amarillo (~5) -> naranja (~12) -> rojo (>20)
    def color_for(v):
        if v < 1.5:  return (0, 220, 0)
        if v < 5.0:  return (0, 220, 220)
        if v < 12.0: return (0, 160, 255)
        return (40, 50, 255)

    # In_a (acelerometro) - mas grande
    text.queue(px + 4, y + 22, "Acel.", size=11, color=(160, 160, 170))
    text.queue(px + 50, y + 22,
               f"{in_a:.1f}x", size=24,
               color=color_for(in_a), bold=True)
    text.queue(px + 4 + bw - 60, y + 30,
               f"RMS {metrics.get('rms_a', 0.0)*1000:.0f} mg",
               size=10, color=(140, 140, 150))

    # In_g (giroscopio) - secundario
    text.queue(px + 4, y + 60, "Giro.", size=11, color=(160, 160, 170))
    text.queue(px + 50, y + 56,
               f"{in_g:.1f}x", size=20,
               color=color_for(in_g), bold=True)
    text.queue(px + 4 + bw - 60, y + 64,
               f"RMS {metrics.get('rms_g', 0.0):.1f} d/s",
               size=10, color=(140, 140, 150))

    return y + h + 10


def _draw_secondary_metrics(canvas, px, y, bw, metrics, individual):
    """Bloque pequeño de metricas auxiliares con sus barras."""
    text.queue(px, y, "Variables auxiliares", size=12,
               color=(200, 200, 120), bold=True)
    y += 18

    rows = (
        ("Frecuencia",   f"{metrics['frequency']:.1f} Hz",    'frequency',
            "pico tremor"),
        ("CV inter-tap", f"{metrics['cv']:.0f} %",            'cv',
            "regularidad"),
        ("Jerk",         f"{metrics['jerk']:.1f} g/s",        'jerk',
            "suavidad"),
        ("Amp. vision",  f"{metrics['amplitude']:.2f}",        None,
            "solo display"),
    )
    for label, val_str, key, hint in rows:
        score = individual.get(key, 0.0) if key else 0.0
        bar_color = (0, int((1 - score) * 200), int(score * 255)) if key \
                    else (90, 90, 110)
        text.queue(px, y, label, size=11, color=(200, 200, 200))
        text.queue(px + 110, y, val_str, size=11, color=(255, 255, 255), bold=True)
        text.queue(px + 220, y, hint, size=10, color=(120, 120, 130))
        y += 14
        if key:
            draw_bar(canvas, px, y, bw, 5, score, 1.0, bar_color)
        y += 10
    return y


def _draw_score_section(canvas, px, y, bw, composite, updrs, ulabel,
                         calibrated):
    """UPDRS grade GRANDE con barra y % - la lectura primaria del paciente."""
    color = scorer.COLORS.get(updrs, (200, 200, 200)) if updrs is not None \
            else (160, 160, 170)
    text.queue(px, y, "UPDRS Parte 3", size=12,
               color=(255, 255, 200), bold=True)
    y += 22

    if updrs is None:
        # No calibrado: mostramos la instruccion en lugar de un score falso.
        text.queue(px, y, ulabel, size=24, color=color, bold=True)
        y += 32
        text.queue(px, y,
                   "El score se publica solo despues",
                   size=11, color=(160, 160, 170))
        y += 16
        text.queue(px, y,
                   "de la calibracion saludable.",
                   size=11, color=(160, 160, 170))
        return y + 18

    # Numero gigante + label
    text.queue(px, y, f"{updrs}", size=64, color=color, bold=True)
    text.queue(px + 70, y + 14, ulabel, size=20, color=color, bold=True)
    text.queue(px + 70, y + 44,
               f"compuesto {composite*100:.0f} %",
               size=11, color=(180, 180, 190))
    y += 76
    draw_bar(canvas, px, y, bw, 12, composite, 1.0, color)
    return y + 20


def _draw_calibration_overlay(canvas):
    """Overlay grande mientras [C] esta corriendo (solo sobre el video)."""
    if not index_calc.cal_active:
        return
    progress = index_calc.cal_progress()

    # Dim solo la region del video; el panel queda intacto.
    video_region = canvas[:VIDEO_H, :VIDEO_W]
    video_region[:] = (video_region * 0.35).astype(np.uint8)

    text.queue(VIDEO_W // 2 - 200, VIDEO_H // 2 - 60,
               "CALIBRANDO BASELINE",
               size=32, color=(255, 220, 100), bold=True)
    text.queue(VIDEO_W // 2 - 230, VIDEO_H // 2 - 16,
               "Mantener sensor INMOVIL en mano SANA",
               size=18, color=(220, 220, 230))
    text.queue(VIDEO_W // 2 - 60, VIDEO_H // 2 + 30,
               f"{progress*100:.0f} %",
               size=40, color=(0, 220, 220), bold=True)

    bar_x = VIDEO_W // 2 - 200
    bar_y = VIDEO_H // 2 + 90
    draw_bar(canvas, bar_x, bar_y, 400, 14, progress, 1.0, (0, 200, 220))


def draw_panel(canvas, metrics, individual, composite, updrs, ulabel):
    x0 = VIDEO_W
    px = x0 + 16
    bw = PANEL_W - 32

    # Fondo + separador
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, VIDEO_H), (22, 22, 28), -1)
    cv2.line(canvas, (x0, 0), (x0, VIDEO_H), (70, 70, 80), 2)

    y = 14
    text.queue(px, y, "UPDRS Parte 3", size=22,
               color=(255, 240, 130), bold=True)
    y += 30
    text.queue(px, y, "Analisis cuantitativo del tremor",
               size=11, color=(150, 150, 165))
    y += 22

    # Sensor badge
    draw_sensor_badge(canvas, px, y, metrics.get('fs', 0.0))
    y += 50

    # ===== UPDRS GRADE (grande, el headline) =====
    calibrated = index_calc.is_calibrated
    y = _draw_score_section(canvas, px, y, bw, composite, updrs, ulabel,
                            calibrated)
    y += 6

    # ===== Tremor Normality Index =====
    y = _draw_in_indices(canvas, px, y, bw, metrics, calibrated)

    # Linea separadora
    cv2.line(canvas, (px, y), (px + bw, y), (60, 60, 75), 1)
    y += 10

    # ===== Manos =====
    for hlabel in (HAND_RIGHT, HAND_LEFT):
        tk = hand_trackers.get(hlabel)
        if tk:
            y = _draw_hand_section(canvas, px, y, hlabel, tk)

    cv2.line(canvas, (px, y), (px + bw, y), (60, 60, 75), 1)
    y += 10

    # ===== Variables auxiliares =====
    y = _draw_secondary_metrics(canvas, px, y, bw, metrics, individual)

    # ===== Hotkeys footer =====
    y_bot = VIDEO_H - 28
    cv2.line(canvas, (px, y_bot - 6), (px + bw, y_bot - 6), (60, 60, 75), 1)
    text.queue(px, y_bot,
               "Q salir  R reset  C calibrar  G plots  F FFT  P print",
               size=10, color=(140, 140, 150))


def draw_plots(canvas):
    """Render de los 3 plots crudos en la franja inferior."""
    x = 0
    plot_accel.draw(canvas, x, VIDEO_H);                 x += plot_accel.width
    plot_gyro.draw (canvas, x, VIDEO_H);                 x += plot_gyro.width
    plot_vision.draw(canvas, x, VIDEO_H)


# =====================================================================
#  PER-FRAME PROCESSING (separated for clarity & to avoid name shadowing
#  warnings between the main loop and class methods)
# =====================================================================
def process_imu_batch(samples):
    """Push one batch of raw IMU samples through preprocessor + plots."""
    for sample in samples:
        if len(sample) < 7:
            continue
        proc = imu_pre.process(sample[0], sample[1], sample[2],
                               sample[3], sample[4], sample[5], sample[6])
        sig_proc.add_processed_sample(proc)
        a_dx, a_dy, a_dz, g_x, g_y, g_z, _ = proc
        plot_accel.add(math.sqrt(a_dx * a_dx + a_dy * a_dy + a_dz * a_dz))
        plot_gyro.add (math.sqrt(g_x * g_x + g_y * g_y + g_z * g_z))


def process_hands(canvas_img, results):
    """Update vision-derived signals and draw landmarks for each detected hand."""
    if not results.multi_hand_landmarks:
        return
    for idx, hand_lm in enumerate(results.multi_hand_landmarks):
        hlabel = results.multi_handedness[idx].classification[0].label
        tracker = hand_trackers.setdefault(hlabel, HandTracker())

        tid = thumb_index_normalized(hand_lm)
        sig_proc.add_vision_sample(tid)
        plot_vision.add(tid)

        tracker.update_tap(detect_tapping(hand_lm), sig_proc)
        tracker.update_oc(detect_open_close(hand_lm))
        tracker.update_pron(detect_pronation(hand_lm, hlabel))

        draw_landmarks(canvas_img, hand_lm, hlabel)


def print_metrics_dump(metrics_dict, individual_scores, composite_score,
                       updrs_grade, label_text):
    print("\n" + "=" * 60)
    print("  METRICAS ACTUALES")
    print("=" * 60)
    print(f"  fs efectiva       = {metrics_dict['fs']:.2f} Hz")
    print(f"  Muestras IMU      = {sensor_reader.total_samples}")
    print(f"  Calibrado         = {'SI' if index_calc.is_calibrated else 'NO'}")
    if index_calc.is_calibrated:
        b = index_calc.baseline
        print(f"    baseline        MPS_a={b['mps_a_g']:.4f}g"
              f"  STDPS_a={b['stdps_a_g']:.4f}g")
        print(f"                    MPS_g={b['mps_g_dps']:.2f}d/s"
              f"  STDPS_g={b['stdps_g_dps']:.2f}d/s")
        print(f"                    n={b['n_samples']}  ts={b['timestamp']}")
    print(f"  In_a (acelerom.)  = {metrics_dict.get('in_a', 0.0):.2f} x baseline")
    print(f"  In_g (giroscop.)  = {metrics_dict.get('in_g', 0.0):.2f} x baseline")
    print(f"  RMS-a (5s)        = {metrics_dict.get('rms_a', 0.0)*1000:.2f} mg")
    print(f"  RMS-g (5s)        = {metrics_dict.get('rms_g', 0.0):.2f} d/s")
    print("  --- secundarias ---")
    aux = ('frequency', 'amplitude', 'angular_vel', 'cv', 'jerk')
    for k in aux:
        score = individual_scores.get(k, 0.0)
        print(f"  {k:14s}  = {metrics_dict[k]:.3f}  -> score {score:.2f}")
    print("  --- UPDRS ---")
    if updrs_grade is None:
        print(f"  Indice compuesto  = {label_text} (sin calibracion)")
    else:
        print(f"  Indice compuesto  = {composite_score:.3f}")
        print(f"  UPDRS             = {updrs_grade} ({label_text})")
    print("=" * 60)


# =====================================================================
#  MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("  UPDRS Parte 3 - Analisis Cuantitativo")
    print("  ESP32-CAM + MPU6050  (sensor en falange distal del indice)")
    print("  Frame : " + FRAME_URL)
    print("  Sensor: " + SENSOR_URL)
    print("  Teclas: Q salir  R reset  C calibrar  G plots  F FFT  P print")
    print("=" * 60)

    if not index_calc.is_calibrated:
        print("")
        print("  >>> NO HAY BASELINE CALIBRADO <<<")
        print("  El indice UPDRS quedara oculto hasta que se calibre.")
        print("  Pulsa  C  con sensor inmovil en mano sana ~10 s.")
        print("")

    sensor_reader.start()

    # Session reusable para /frame (keep-alive ahorra setup TCP por frame).
    frame_session = requests.Session()
    frame_session.headers.update({"Connection": "keep-alive"})

    fps_ts = time.time()
    fps_n = 0
    fps_val = 0.0
    last_metrics_log = time.time()

    try:
        while True:
            frame_data = get_frame(FRAME_URL, frame_session)
            if not frame_data:
                time.sleep(0.2)
                continue

            try:
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8),
                                     cv2.IMREAD_COLOR)
            except Exception:
                frame = None
            if frame is None:
                continue

            imu_samples = sensor_reader.get_samples()
            if imu_samples:
                process_imu_batch(imu_samples)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Canvas size depende de si los plots estan visibles.
            ch = CANVAS_H_PLOTS if ui_state["show_plots"] else CANVAS_H_NOPLOTS
            canvas = np.zeros((ch, CANVAS_W, 3), dtype=np.uint8)
            canvas[:VIDEO_H, :VIDEO_W] = cv2.resize(frame, (VIDEO_W, VIDEO_H))

            process_hands(canvas, results)

            metrics = sig_proc.compute_all(index_calc=index_calc)
            individual, composite, updrs, ulabel = scorer.compute(
                metrics, index_calibrated=index_calc.is_calibrated)

            draw_panel(canvas, metrics, individual, composite, updrs, ulabel)
            if ui_state["show_plots"]:
                draw_plots(canvas)

            fps_n += 1
            now = time.time()
            if now - fps_ts >= 1.0:
                fps_val = fps_n / (now - fps_ts)
                fps_ts, fps_n = now, 0
            text.queue(10, 4, f"FPS {fps_val:.1f}", size=12,
                       color=(255, 255, 255), bold=True)

            # Overlay grande si estamos calibrando.
            _draw_calibration_overlay(canvas)

            # Flush UNICO de todo el texto Pillow del frame.
            text.flush(canvas)

            if now - last_metrics_log >= 3.0:
                last_metrics_log = now
                print("[MET] fs=%.1fHz  In_a=%.2fx  In_g=%.2fx  freq=%.2fHz"
                      "  CV=%.1f%%  jerk=%.2fg/s  cal=%s"
                      % (metrics['fs'], metrics.get('in_a', 0.0),
                         metrics.get('in_g', 0.0), metrics['frequency'],
                         metrics['cv'], metrics['jerk'],
                         "OK" if index_calc.is_calibrated else "PENDIENTE"))

            cv2.imshow("UPDRS - Analisis Cuantitativo", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                hand_trackers.clear()
                sig_proc.reset()
                imu_pre.reset()
                plot_accel.data.clear()
                plot_gyro.data.clear()
                plot_vision.data.clear()
                print("[RESET] Contadores, buffers, preprocesador y graficas reiniciados")
            elif key == ord('c'):
                if index_calc.cal_active:
                    print("[CAL] Ya hay calibracion en curso.")
                else:
                    index_calc.start_calibration()
            elif key == ord('g'):
                ui_state["show_plots"] = not ui_state["show_plots"]
                print(f"[UI] Plots crudos: {'ON' if ui_state['show_plots'] else 'OFF'}")
            elif key == ord('f'):
                ui_state["show_fft"] = not ui_state["show_fft"]
                print(f"[UI] FFT debug: {'ON' if ui_state['show_fft'] else 'OFF'} (no implementado todavia)")
            elif key == ord('p'):
                m = sig_proc.compute_all(index_calc=index_calc, force=True)
                ind, comp, upd, lab = scorer.compute(
                    m, index_calibrated=index_calc.is_calibrated)
                print_metrics_dump(m, ind, comp, upd, lab)
    finally:
        sensor_reader.stop()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
