"""
ESP32-S3 CAM + MPU6050 — Servidor HTTP
=======================================
Endpoints:
  /frame   -> JPEG de la camara (VGA 640x480)
  /sensor  -> JSON con muestras acumuladas del MPU6050
  /        -> Pagina HTML de estado

Conexiones MPU6050:
  SDA -> GPIO 8
  SCL -> GPIO 9
  VCC -> 3.3V
  GND -> GND

El servidor usa select.poll() para ser no-bloqueante:
mientras espera clientes HTTP, lee el MPU6050 cada ~10 ms
y acumula muestras en un buffer circular.
"""

import network
import socket
import select
import struct
import time
from machine import I2C, Pin
from camera import Camera, FrameSize, PixelFormat


# =====================================================================
#  WIFI ACCESS POINT
# =====================================================================
ap = network.WLAN(network.AP_IF)
ap.active(True)
ap.config(
    essid="ESP32-S312",
    password="12345678",
    authmode=network.AUTH_WPA2_PSK,
    max_clients=4,
    channel=1,
    hidden=False
)
print("AP IP:", ap.ifconfig())


# =====================================================================
#  MPU6050 — DRIVER I2C
# =====================================================================
class MPU6050:
    """
    Driver minimo para MPU6050 via I2C.

    Registros clave:
      0x6B  PWR_MGMT_1   Despertar (escribir 0x00)
      0x19  SMPLRT_DIV   Tasa muestreo = 1 kHz / (1 + valor)
      0x1A  CONFIG       Filtro pasa-bajo digital (DLPF)
      0x1B  GYRO_CONFIG  Rango giroscopio
      0x1C  ACCEL_CONFIG Rango acelerometro
      0x3B  ACCEL_XOUT_H Inicio de 14 bytes de datos

    Rangos disponibles:
      Acelerometro: 0x00=+-2g  0x08=+-4g  0x10=+-8g  0x18=+-16g
      Giroscopio:   0x00=+-250 0x08=+-500 0x10=+-1000 0x18=+-2000 deg/s
    """

    ADDR = 0x68

    # Factores de escala segun rango configurado
    _ASCALE = {0x00: 16384.0, 0x08: 8192.0, 0x10: 4096.0, 0x18: 2048.0}
    _GSCALE = {0x00: 131.0,   0x08: 65.5,   0x10: 32.8,   0x18: 16.4}

    def __init__(self, i2c, accel_range=0x08, gyro_range=0x08):
        self.i2c = i2c
        self.a_div = self._ASCALE.get(accel_range, 8192.0)
        self.g_div = self._GSCALE.get(gyro_range, 65.5)

        # 1. Despertar sensor
        self.i2c.writeto_mem(self.ADDR, 0x6B, b'\x00')
        time.sleep_ms(100)

        # 2. Sample rate: 1 kHz / (1 + 19) = 50 Hz
        self.i2c.writeto_mem(self.ADDR, 0x19, b'\x13')

        # 3. DLPF: BW ~44 Hz  (captura tremor 3-12 Hz, filtra ruido alto)
        self.i2c.writeto_mem(self.ADDR, 0x1A, b'\x03')

        # 4. Rangos
        self.i2c.writeto_mem(self.ADDR, 0x1C, bytes([accel_range]))  # +-4 g
        self.i2c.writeto_mem(self.ADDR, 0x1B, bytes([gyro_range]))   # +-500 deg/s

    def read(self):
        """
        Lee acelerometro + giroscopio en una sola transaccion I2C.
        Retorna: (ax, ay, az, gx, gy, gz)
                  ax/ay/az en g,  gx/gy/gz en deg/s
        """
        raw = self.i2c.readfrom_mem(self.ADDR, 0x3B, 14)
        ax = struct.unpack('>h', raw[0:2])[0]   / self.a_div
        ay = struct.unpack('>h', raw[2:4])[0]   / self.a_div
        az = struct.unpack('>h', raw[4:6])[0]   / self.a_div
        # raw[6:8] = temperatura (no se usa)
        gx = struct.unpack('>h', raw[8:10])[0]  / self.g_div
        gy = struct.unpack('>h', raw[10:12])[0] / self.g_div
        gz = struct.unpack('>h', raw[12:14])[0] / self.g_div
        return ax, ay, az, gx, gy, gz


# --- Inicializar I2C + MPU6050 ---
mpu = None
try:
    i2c = I2C(0, sda=Pin(8), scl=Pin(9), freq=400000)
    devs = i2c.scan()
    if 0x68 in devs:
        mpu = MPU6050(i2c, accel_range=0x08, gyro_range=0x08)
        print("MPU6050 OK  (+-4 g, +-500 deg/s, 50 Hz, DLPF 44 Hz)")
    else:
        print("MPU6050 NO detectado. Dispositivos I2C:", devs)
except Exception as e:
    print("Error I2C:", e)

# Buffer circular de muestras  [ax, ay, az, gx, gy, gz, ticks_ms]
sensor_buf = []
MAX_BUF = 64


# =====================================================================
#  CAMERA
# =====================================================================
cam = Camera(frame_size=FrameSize.VGA, pixel_format=PixelFormat.JPEG)
cam.init()
print("Camera VGA (640x480) lista")


# =====================================================================
#  HTTP SERVER  (no-bloqueante con select.poll)
# =====================================================================
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('', 80))
srv.listen(5)
srv.setblocking(False)

poller = select.poll()
poller.register(srv, select.POLLIN)

print("Servidor HTTP activo en puerto 80")
print("  /frame  -> JPEG")
print("  /sensor -> JSON IMU")


def build_sensor_json(buf):
    """
    Construye JSON compacto con las muestras acumuladas.
    Formato: {"s":[ [ax,ay,az,gx,gy,gz,ms], ... ]}
    Se construye manualmente para evitar dependencia de ujson
    y minimizar uso de memoria.
    """
    if not buf:
        return '{"s":[]}'
    parts = []
    for ax, ay, az, gx, gy, gz, t in buf:
        parts.append('[%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%d]'
                     % (ax, ay, az, gx, gy, gz, t))
    return '{"s":[' + ','.join(parts) + ']}'


# =====================================================================
#  LOOP PRINCIPAL
# =====================================================================
while True:
    # ---- 1. Leer sensor (cada iteracion del loop ~10 ms) ----
    if mpu:
        try:
            ax, ay, az, gx, gy, gz = mpu.read()
            sensor_buf.append((ax, ay, az, gx, gy, gz, time.ticks_ms()))
            if len(sensor_buf) > MAX_BUF:
                sensor_buf = sensor_buf[-MAX_BUF:]
        except Exception:
            pass

    # ---- 2. Esperar cliente HTTP (timeout 10 ms) ----
    events = poller.poll(10)
    if not events:
        continue

    # ---- 3. Atender peticion HTTP ----
    client = None
    try:
        client, addr = srv.accept()
        client.settimeout(2)
        request = client.recv(1024).decode()
        path = request.split(' ')[1] if ' ' in request else '/'

        if path == "/frame":
            # --- Enviar frame JPEG de la camara ---
            frame = cam.capture()
            if frame:
                client.send(
                    b'HTTP/1.1 200 OK\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Cache-Control: no-cache\r\n\r\n'
                )
                client.sendall(frame)
            else:
                client.send(b'HTTP/1.1 500 Error\r\n\r\n')

        elif path == "/sensor":
            # --- Enviar buffer de muestras IMU y vaciar ---
            payload = build_sensor_json(sensor_buf)
            sensor_buf = []   # vaciar despues de enviar
            client.send(
                b'HTTP/1.1 200 OK\r\n'
                b'Content-Type: application/json\r\n'
                b'Cache-Control: no-cache\r\n\r\n'
            )
            client.sendall(payload.encode())

        else:
            # --- Pagina de estado ---
            imu_status = "Conectado" if mpu else "No detectado"
            html = (
                '<html><head><title>ESP32 UPDRS</title></head><body>'
                '<h1>ESP32-CAM + MPU6050 &mdash; UPDRS</h1>'
                '<p>Camera: VGA 640x480</p>'
                '<p>MPU6050: %s</p>'
                '<img src="/frame" width="640">'
                '<p><a href="/sensor">Ver datos sensor</a></p>'
                '</body></html>' % imu_status
            )
            client.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n')
            client.sendall(html.encode())

    except Exception:
        pass
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass
