"""
UPDRS Parte 3 - Analisis de movimientos por vision artificial
Ejercicios: 3.4 Golpeteo de dedos, 3.5 Movimientos de manos, 3.6 Pronacion/Supinacion
Fuente de video: ESP32-CAM via WiFi AP

Layout: video 800x600 a la izquierda + panel info 280px a la derecha
"""

import cv2
import numpy as np
import requests
import mediapipe as mp
import time
from collections import deque

# -------- CONFIGURACION ----------
ESP32_IP = "192.168.4.1"
FRAME_URL = f"http://{ESP32_IP}/frame"

# Dimensiones fijas de visualizacion
VIDEO_W, VIDEO_H = 800, 600
PANEL_W = 280
CANVAS_W = VIDEO_W + PANEL_W   # 1080 total

# -------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------- COLORES POR DEDO ----------
RIGHT_COLORS = {
    'THUMB':  (0,   0,   255),
    'INDEX':  (0,   255, 255),
    'MIDDLE': (0,   255, 0),
    'RING':   (255, 0,   0),
    'PINKY':  (255, 0,   255)
}
LEFT_COLORS = {
    'THUMB':  (0,   128, 255),
    'INDEX':  (128, 255, 255),
    'MIDDLE': (128, 255, 128),
    'RING':   (255, 128, 128),
    'PINKY':  (255, 128, 255)
}
FINGER_LANDMARKS = {
    'THUMB':  [1, 2, 3, 4],
    'INDEX':  [5, 6, 7, 8],
    'MIDDLE': [9, 10, 11, 12],
    'RING':   [13, 14, 15, 16],
    'PINKY':  [17, 18, 19, 20]
}
FINGER_CONNECTIONS = {
    'THUMB':  [(1,2),(2,3),(3,4)],
    'INDEX':  [(5,6),(6,7),(7,8)],
    'MIDDLE': [(9,10),(10,11),(11,12)],
    'RING':   [(13,14),(14,15),(15,16)],
    'PINKY':  [(17,18),(18,19),(19,20)]
}
PALM_CONNECTIONS = [(0,1),(1,5),(5,9),(9,13),(13,17),(0,17)]


# -------- TRACKER UPDRS ----------
class UPDRSTracker:
    def __init__(self):
        self.tap_count      = 0
        self.last_tap       = False
        self.tap_times      = deque(maxlen=30)
        self.pron_count     = 0
        self.last_pron      = ""
        self.oc_count       = 0
        self.last_oc        = ""

    def update_tap(self, tapping):
        if tapping and not self.last_tap:
            self.tap_count += 1
            self.tap_times.append(time.time())
        self.last_tap = tapping

    def tap_freq(self):
        now = time.time()
        recent = [t for t in self.tap_times if now - t <= 5.0]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / (recent[-1] - recent[0])

    def update_pron(self, state):
        if state != "Neutro" and state != self.last_pron and self.last_pron:
            self.pron_count += 1
        self.last_pron = state

    def update_oc(self, state):
        if state != self.last_oc and self.last_oc:
            self.oc_count += 1
        self.last_oc = state


trackers = {}


# -------- HELPERS ----------
def get_frame(url):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.content
    except requests.RequestException:
        pass
    return None


def hand_scale(lms):
    """Distancia muneca -> MCP dedo medio como referencia de tamano."""
    w, m = lms.landmark[0], lms.landmark[9]
    return np.hypot(w.x - m.x, w.y - m.y)


# -------- DETECCION DE EJERCICIOS ----------

def detect_tapping(lms):
    """
    UPDRS 3.4: Golpeteo indice-pulgar.
    Umbral normalizado por tamano de mano (robusto a distancia de camara).
    """
    thumb, index = lms.landmark[4], lms.landmark[8]
    dist = np.hypot(thumb.x - index.x, thumb.y - index.y)
    s = hand_scale(lms)
    return (dist / s) < 0.40 if s > 0.01 else False


def detect_open_close(lms):
    """
    UPDRS 3.5: Mano abierta/cerrada.
    Cuenta dedos extendidos comparando tip.y < PIP.y (4 dedos sin pulgar).
    """
    tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
    ext = sum(1 for t, p in tips_pips if lms.landmark[t].y < lms.landmark[p].y)
    if ext >= 3:
        return "Abierta"
    if ext <= 1:
        return "Cerrada"
    return "Parcial"


def detect_pronation(lms, hand_label):
    """
    UPDRS 3.6: Pronacion / Supinacion.

    Calcula el vector normal al plano de la palma usando producto vectorial 3D
    de los vectores muneca->MCP_indice y muneca->MCP_menique.

    El componente Z de la normal indica hacia donde apunta la palma:
      - Right hand: normal.z < 0 -> Supinacion (palma hacia camara)
                    normal.z > 0 -> Pronacion  (dorso hacia camara)
      - Left hand:  signo invertido

    Movimiento del paciente:
      Extender el brazo al frente, rotar el antebrazo alternando rapidamente
      entre palma arriba (supinacion) y palma abajo (pronacion).
    """
    wrist     = lms.landmark[0]
    index_mcp = lms.landmark[5]
    pinky_mcp = lms.landmark[17]

    v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    nz = np.cross(v1, v2)[2]

    th = 0.006
    if hand_label == "Right":
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


# -------- DIBUJO ----------

def draw_landmarks(canvas, lms, hand_label):
    """Dibuja los landmarks de la mano sobre el area de video del canvas."""
    colors = RIGHT_COLORS if hand_label == "Right" else LEFT_COLORS

    for s_i, e_i in PALM_CONNECTIONS:
        s, e = lms.landmark[s_i], lms.landmark[e_i]
        p1 = (int(s.x * VIDEO_W), int(s.y * VIDEO_H))
        p2 = (int(e.x * VIDEO_W), int(e.y * VIDEO_H))
        cv2.line(canvas, p1, p2, (180, 180, 180), 2)

    for fname, color in colors.items():
        for s_i, e_i in FINGER_CONNECTIONS[fname]:
            s, e = lms.landmark[s_i], lms.landmark[e_i]
            p1 = (int(s.x * VIDEO_W), int(s.y * VIDEO_H))
            p2 = (int(e.x * VIDEO_W), int(e.y * VIDEO_H))
            cv2.line(canvas, p1, p2, color, 2)
        for li in FINGER_LANDMARKS[fname]:
            lm = lms.landmark[li]
            cv2.circle(canvas, (int(lm.x * VIDEO_W), int(lm.y * VIDEO_H)), 4, color, -1)

    wr = lms.landmark[0]
    cv2.circle(canvas, (int(wr.x * VIDEO_W), int(wr.y * VIDEO_H)), 6, (220, 220, 220), -1)

    # Etiqueta pequena junto a la muneca
    wx = int(wr.x * VIDEO_W) - 15
    wy = int(wr.y * VIDEO_H) - 12
    cv2.putText(canvas, hand_label, (max(wx, 2), max(wy, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_panel(canvas):
    """Dibuja el panel lateral derecho con toda la informacion UPDRS."""
    x0 = VIDEO_W
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Fondo del panel
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, VIDEO_H), (25, 25, 30), -1)
    # Linea separadora
    cv2.line(canvas, (x0, 0), (x0, VIDEO_H), (80, 80, 90), 2)

    px = x0 + 15   # margen izquierdo del texto
    y = 30

    # --- Titulo ---
    cv2.putText(canvas, "UPDRS - Parte 3", (px, y), font, 0.6, (255, 255, 120), 2, cv2.LINE_AA)
    y += 22
    cv2.putText(canvas, "Evaluacion Motora", (px, y), font, 0.38, (160, 160, 170), 1, cv2.LINE_AA)
    y += 8
    cv2.line(canvas, (px, y), (x0 + PANEL_W - 15, y), (55, 55, 65), 1)
    y += 20

    # --- Lista de ejercicios ---
    exercises = [
        ("3.4", "Golpeteo de dedos"),
        ("3.5", "Movimientos de manos"),
        ("3.6", "Pronacion / Supinacion"),
    ]
    for code, name in exercises:
        cv2.putText(canvas, f"{code} {name}", (px, y), font, 0.35, (140, 140, 150), 1, cv2.LINE_AA)
        y += 18

    y += 4
    cv2.line(canvas, (px, y), (x0 + PANEL_W - 15, y), (55, 55, 65), 1)
    y += 20

    # --- Info por mano ---
    for hand_label in ["Right", "Left"]:
        tk = trackers.get(hand_label)
        if not tk:
            continue

        name = "Mano Derecha" if hand_label == "Right" else "Mano Izquierda"
        hdr_color = (200, 200, 255) if hand_label == "Right" else (255, 200, 255)

        cv2.putText(canvas, name, (px, y), font, 0.52, hdr_color, 1, cv2.LINE_AA)
        y += 24

        # 3.4 Golpeteo
        tap_active = tk.last_tap
        tap_col = (0, 255, 100) if tap_active else (120, 120, 120)
        tap_txt = "SI" if tap_active else "NO"
        cv2.putText(canvas, f"Golpeteo: {tap_txt}", (px + 8, y), font, 0.40, tap_col, 1, cv2.LINE_AA)
        y += 18
        cv2.putText(canvas, f"Taps: {tk.tap_count}  ({tk.tap_freq():.1f} Hz)", (px + 8, y),
                    font, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        y += 22

        # 3.5 Abrir/Cerrar
        oc = tk.last_oc or "---"
        oc_cols = {"Abierta": (0, 255, 160), "Cerrada": (50, 120, 255), "Parcial": (255, 210, 0)}
        cv2.putText(canvas, f"Mano: {oc}", (px + 8, y), font, 0.40,
                    oc_cols.get(oc, (140, 140, 140)), 1, cv2.LINE_AA)
        y += 18
        cv2.putText(canvas, f"Cambios: {tk.oc_count}", (px + 8, y),
                    font, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        y += 22

        # 3.6 Pronacion/Supinacion
        pr = tk.last_pron or "---"
        pr_cols = {"Supinacion": (0, 220, 255), "Pronacion": (255, 160, 0), "Neutro": (140, 140, 140)}
        cv2.putText(canvas, f"Rotacion: {pr}", (px + 8, y), font, 0.40,
                    pr_cols.get(pr, (140, 140, 140)), 1, cv2.LINE_AA)
        y += 18
        cv2.putText(canvas, f"Cambios: {tk.pron_count}", (px + 8, y),
                    font, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        y += 30

    # --- Controles al fondo ---
    y_bottom = VIDEO_H - 45
    cv2.line(canvas, (px, y_bottom), (x0 + PANEL_W - 15, y_bottom), (55, 55, 65), 1)
    y_bottom += 20
    cv2.putText(canvas, "[R] Reset contadores", (px, y_bottom), font, 0.36, (140, 190, 140), 1, cv2.LINE_AA)
    y_bottom += 18
    cv2.putText(canvas, "[Q] Salir", (px, y_bottom), font, 0.36, (190, 140, 140), 1, cv2.LINE_AA)


# -------- LOOP PRINCIPAL ----------
print("Conectando a ESP32-CAM...")
print(f"URL: {FRAME_URL}")
print("Presiona Q para salir, R para reiniciar contadores")

while True:
    frame_data = get_frame(FRAME_URL)
    if not frame_data:
        print("Sin frame, reintentando...")
        time.sleep(0.2)
        continue

    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # Procesar con MediaPipe a resolucion original
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Crear canvas: video (800x600) + panel lateral (280px)
    video = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    canvas = np.zeros((VIDEO_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:, :VIDEO_W] = video

    # Dibujar landmarks y evaluar ejercicios
    if results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            if label not in trackers:
                trackers[label] = UPDRSTracker()
            tk = trackers[label]

            # Evaluar
            tk.update_tap(detect_tapping(hand_lm))
            tk.update_oc(detect_open_close(hand_lm))
            tk.update_pron(detect_pronation(hand_lm, label))

            # Dibujar landmarks sobre el area de video
            draw_landmarks(canvas, hand_lm, label)

    # Dibujar panel lateral con info
    draw_panel(canvas)

    cv2.imshow("UPDRS Vision Artificial - ESP32", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        trackers.clear()
        print("Contadores reiniciados")

    time.sleep(0.02)

cv2.destroyAllWindows()
hands.close()
