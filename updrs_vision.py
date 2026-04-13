"""
UPDRS Parte 3 - Análisis de movimientos por visión artificial
Ejercicios: 3.4 Golpeteo de dedos, 3.5 Movimientos de manos, 3.6 Pronación/Supinación
Fuente de video: ESP32-CAM vía WiFi AP
"""

import cv2
import numpy as np
import requests
import mediapipe as mp
import time
from collections import deque

# -------- CONFIG ----------
ESP32_IP = "192.168.4.1"
FRAME_URL = f"http://{ESP32_IP}/frame"

# Factor de escala para visualización (1.5 = 960x720 con VGA, 2.0 con QVGA)
DISPLAY_SCALE = 1.5

# -------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------- COLORES POR DEDO ----------
RIGHT_HAND_COLORS = {
    'THUMB':  (0,   0,   255),
    'INDEX':  (0,   255, 255),
    'MIDDLE': (0,   255, 0),
    'RING':   (255, 0,   0),
    'PINKY':  (255, 0,   255)
}
LEFT_HAND_COLORS = {
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
    """Acumula métricas por mano para evaluación UPDRS."""

    def __init__(self):
        self.tap_count        = 0
        self.last_tap_state   = False
        self.tap_times        = deque(maxlen=30)

        self.pronation_count  = 0
        self.last_pron_state  = ""

        self.open_close_count = 0
        self.last_hand_state  = ""

    # --- Golpeteo ---
    def update_tap(self, is_tapping: bool):
        if is_tapping and not self.last_tap_state:
            self.tap_count += 1
            self.tap_times.append(time.time())
        self.last_tap_state = is_tapping

    def tap_frequency(self) -> float:
        """Hz promedio en los últimos 5 segundos."""
        now = time.time()
        recent = [t for t in self.tap_times if now - t <= 5.0]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / (recent[-1] - recent[0])

    # --- Pronación/Supinación ---
    def update_pronation(self, state: str):
        if state != "neutro" and state != self.last_pron_state and self.last_pron_state != "":
            self.pronation_count += 1
        self.last_pron_state = state

    # --- Abrir/Cerrar ---
    def update_hand_state(self, state: str):
        if state != self.last_hand_state and self.last_hand_state != "":
            self.open_close_count += 1
        self.last_hand_state = state


trackers: dict[str, UPDRSTracker] = {}


# -------- HELPERS ----------
def get_frame(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.content
    except requests.RequestException:
        pass
    return None


def hand_scale(hand_landmarks) -> float:
    """Distancia muñeca → MCP del dedo medio como referencia de tamaño."""
    w = hand_landmarks.landmark[0]
    m = hand_landmarks.landmark[9]
    return np.hypot(w.x - m.x, w.y - m.y)


# -------- DETECCIÓN DE EJERCICIOS ----------

def finger_tapping(hand_landmarks) -> bool:
    """
    UPDRS 3.4: Golpeteo índice-pulgar.
    Umbral normalizado por tamaño de mano para ser robusto a distancia de la cámara.
    """
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    dist = np.hypot(thumb.x - index.x, thumb.y - index.y)
    scale = hand_scale(hand_landmarks)
    if scale < 0.01:
        return False
    return (dist / scale) < 0.40


def hand_open_close(hand_landmarks) -> str:
    """
    UPDRS 3.5: Cuenta dedos extendidos comparando tip vs PIP.
    Más robusto que comparar solo el dedo medio con la muñeca.
    """
    tips_pips = [(8,6), (12,10), (16,14), (20,18)]  # (tip, PIP) para índice..meñique
    extended = sum(
        1 for tip_i, pip_i in tips_pips
        if hand_landmarks.landmark[tip_i].y < hand_landmarks.landmark[pip_i].y
    )
    if extended >= 3:
        return "abierta"
    if extended <= 1:
        return "cerrada"
    return "parcial"


def pronation_supination(hand_landmarks, hand_label: str) -> str:
    """
    UPDRS 3.6: Pronación / Supinación.

    Usa el vector normal al plano de la palma (producto vectorial 3D).
    MediaPipe z: negativo = más cerca de la cámara que la muñeca.

    Convención:
      - Right hand: normal.z < 0 → supinación (palma hacia cámara/arriba)
                    normal.z > 0 → pronación  (dorso hacia cámara/arriba)
      - Left hand:  signo invertido
    """
    wrist     = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]   # base índice
    pinky_mcp = hand_landmarks.landmark[17]  # base meñique

    v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    normal_z = np.cross(v1, v2)[2]

    threshold = 0.008
    if hand_label == "Right":
        if normal_z < -threshold:
            return "supinacion"
        elif normal_z > threshold:
            return "pronacion"
    else:
        if normal_z > threshold:
            return "supinacion"
        elif normal_z < -threshold:
            return "pronacion"
    return "neutro"


# -------- DIBUJO ----------

def draw_text_box(image, lines, pos, font_scale=0.5, padding=8,
                  bg_color=(30, 30, 30), alpha=0.72):
    """
    Dibuja un cuadro de texto semi-transparente con múltiples líneas.
    Cada elemento de `lines` es (texto, color_BGR).
    Retorna (x2, y2) esquina inferior-derecha del cuadro.
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    h_img, w_img = image.shape[:2]

    # Medir cada línea
    sizes = [cv2.getTextSize(text, font, font_scale, thickness) for text, _ in lines]
    max_w  = max(s[0][0] for s in sizes)
    line_h = max(s[0][1] for s in sizes)
    base   = max(s[1]     for s in sizes)

    row_h  = line_h + base + padding
    box_w  = max_w + padding * 2
    box_h  = row_h * len(lines) + padding

    x, y   = pos
    x      = max(0, min(x, w_img - box_w - 2))
    y      = max(0, min(y, h_img - box_h - 2))
    x2, y2 = min(x + box_w, w_img - 1), min(y + box_h, h_img - 1)

    # Fondo semitransparente
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    # Borde
    cv2.rectangle(image, (x, y), (x2, y2), (110, 110, 110), 1)

    # Texto
    y_cursor = y + padding + line_h
    for (text, color), _ in zip(lines, sizes):
        cv2.putText(image, text, (x + padding, y_cursor),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        y_cursor += row_h

    return (x2, y2)


def draw_colored_landmarks(image, hand_landmarks, hand_label: str):
    h, w, _ = image.shape
    colors = RIGHT_HAND_COLORS if hand_label == "Right" else LEFT_HAND_COLORS

    for s_i, e_i in PALM_CONNECTIONS:
        s = hand_landmarks.landmark[s_i]
        e = hand_landmarks.landmark[e_i]
        cv2.line(image, (int(s.x*w), int(s.y*h)),
                         (int(e.x*w), int(e.y*h)), (180,180,180), 2)

    for fname, color in colors.items():
        for s_i, e_i in FINGER_CONNECTIONS[fname]:
            s = hand_landmarks.landmark[s_i]
            e = hand_landmarks.landmark[e_i]
            cv2.line(image, (int(s.x*w), int(s.y*h)),
                             (int(e.x*w), int(e.y*h)), color, 2)
        for lm_i in FINGER_LANDMARKS[fname]:
            lm = hand_landmarks.landmark[lm_i]
            cv2.circle(image, (int(lm.x*w), int(lm.y*h)), 5, color, -1)

    wr = hand_landmarks.landmark[0]
    cv2.circle(image, (int(wr.x*w), int(wr.y*h)), 7, (220,220,220), -1)


def draw_hand_info(image, hand_landmarks, label: str, tracker: UPDRSTracker):
    h, w, _ = image.shape

    # --- Evaluar ejercicios ---
    is_tapping = finger_tapping(hand_landmarks)
    tracker.update_tap(is_tapping)
    freq = tracker.tap_frequency()

    estado = hand_open_close(hand_landmarks)
    tracker.update_hand_state(estado)

    rotacion = pronation_supination(hand_landmarks, label)
    tracker.update_pronation(rotacion)

    # --- Colores por estado ---
    header_col = (200, 200, 255) if label == "Right" else (255, 200, 255)
    tap_col    = (0, 255, 100)   if is_tapping else (160, 160, 160)

    rot_col = {
        "supinacion": (0, 220, 255),
        "pronacion":  (255, 160, 0),
        "neutro":     (180, 180, 180)
    }.get(rotacion, (180,180,180))

    hand_col = {
        "abierta": (0, 255, 160),
        "cerrada": (50, 120, 255),
        "parcial": (255, 210, 0)
    }.get(estado, (180,180,180))

    # --- Construir líneas ---
    lines = [
        (f"Mano {label}",                              header_col),
        (f"3.4 Golpeteo: {'SI' if is_tapping else 'NO'}  {freq:.1f}Hz", tap_col),
        (f"    Taps acum.: {tracker.tap_count}",       (160, 160, 160)),
        (f"3.5 Mano {estado}",                         hand_col),
        (f"3.6 {rotacion.capitalize()}",               rot_col),
        (f"    Cambios: {tracker.pronation_count}",    (160, 160, 160)),
    ]

    # Posición: encima del bounding box de la mano
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    box_x = max(int(min(xs)*w) - 10, 5)
    box_y = max(int(min(ys)*h) - 10, 5)

    bg = (50, 35, 35) if label == "Right" else (35, 35, 55)
    draw_text_box(image, lines, (box_x, box_y),
                  font_scale=0.48, padding=7, bg_color=bg)


def draw_global_panel(image):
    """Panel informativo fijo en esquina superior derecha."""
    h, w = image.shape[:2]
    lines = [
        ("UPDRS - Parte 3",        (255, 255, 120)),
        ("3.4 Golpeteo dedos",     (200, 200, 200)),
        ("3.5 Mov. de manos",      (200, 200, 200)),
        ("3.6 Pron. / Supin.",     (200, 200, 200)),
        ("",                       (0,0,0)),
        ("[R] Reset contadores",   (180, 220, 180)),
        ("[Q] Salir",              (180, 180, 180)),
    ]
    draw_text_box(image, lines, (w - 240, 8),
                  font_scale=0.44, padding=6, bg_color=(15, 15, 40), alpha=0.80)


# -------- LOOP PRINCIPAL ----------
print("Conectando a ESP32...")
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

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            if label not in trackers:
                trackers[label] = UPDRSTracker()
            draw_colored_landmarks(frame, hand_lm, label)
            draw_hand_info(frame, hand_lm, label, trackers[label])

    draw_global_panel(frame)

    # Escalar para visualización (el procesamiento usa resolución original)
    display = cv2.resize(frame, (0, 0),
                         fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                         interpolation=cv2.INTER_LINEAR)
    cv2.imshow("UPDRS Vision Artificial - ESP32", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        trackers.clear()
        print("Contadores reiniciados")

    time.sleep(0.02)

cv2.destroyAllWindows()
hands.close()
