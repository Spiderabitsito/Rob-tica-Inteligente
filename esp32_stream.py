"""
ESP32-S3 CAM + MPU6050 — Servidor HTTP
=======================================
Endpoints:
  /frame   -> JPEG de la camara (VGA 640x480)
  /sensor  -> JSON con muestras acumuladas del MPU6050
  /status  -> JSON con diagnostico rapido
  /        -> Pagina HTML de estado

Conexiones MPU6050:
  SDA -> GPIO 8
  SCL -> GPIO 9
  VCC -> 3.3V
  GND -> GND
  AD0 -> GND (direccion 0x68) o 3.3V (direccion 0x69)

Diagnostico en consola Thonny:
  - Al iniciar: escaneo I2C con lista de direcciones
  - Cada ~2 segundos: muestra ultima lectura del MPU
  - Cada peticion /sensor: cuenta de muestras enviadas
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
print("")
print("=" * 50)
print("  ESP32 UPDRS - AP IP:", ap.ifconfig()[0])
print("=" * 50)


# =====================================================================
#  MPU6050 — DRIVER I2C
# =====================================================================
class MPU6050:
    """Driver minimo MPU6050."""

    _ASCALE = {0x00: 16384.0, 0x08: 8192.0, 0x10: 4096.0, 0x18: 2048.0}
    _GSCALE = {0x00: 131.0,   0x08: 65.5,   0x10: 32.8,   0x18: 16.4}

    def __init__(self, i2c, addr=0x68, accel_range=0x08, gyro_range=0x08):
        self.i2c   = i2c
        self.addr  = addr
        self.a_div = self._ASCALE.get(accel_range, 8192.0)
        self.g_div = self._GSCALE.get(gyro_range, 65.5)

        # Wake up
        self.i2c.writeto_mem(self.addr, 0x6B, b'\x00')
        time.sleep_ms(100)
        # Sample rate 1kHz / (1+9) = 100 Hz
        self.i2c.writeto_mem(self.addr, 0x19, b'\x09')
        # DLPF: 44 Hz bandwidth
        self.i2c.writeto_mem(self.addr, 0x1A, b'\x03')
        # Ranges
        self.i2c.writeto_mem(self.addr, 0x1C, bytes([accel_range]))
        self.i2c.writeto_mem(self.addr, 0x1B, bytes([gyro_range]))

    def read(self):
        raw = self.i2c.readfrom_mem(self.addr, 0x3B, 14)
        ax = struct.unpack('>h', raw[0:2])[0]   / self.a_div
        ay = struct.unpack('>h', raw[2:4])[0]   / self.a_div
        az = struct.unpack('>h', raw[4:6])[0]   / self.a_div
        gx = struct.unpack('>h', raw[8:10])[0]  / self.g_div
        gy = struct.unpack('>h', raw[10:12])[0] / self.g_div
        gz = struct.unpack('>h', raw[12:14])[0] / self.g_div
        return ax, ay, az, gx, gy, gz


# --- Inicializar I2C + MPU6050 con escaneo de ambas direcciones ---
mpu = None
i2c = None
mpu_addr = None

print("[I2C] Inicializando en SDA=GPIO8, SCL=GPIO9, 400kHz...")
try:
    i2c = I2C(0, sda=Pin(8), scl=Pin(9), freq=400000)
    devs = i2c.scan()
    print("[I2C] Dispositivos detectados:", [hex(d) for d in devs])

    for candidate in (0x68, 0x69):
        if candidate in devs:
            try:
                mpu = MPU6050(i2c, addr=candidate, accel_range=0x08, gyro_range=0x08)
                mpu_addr = candidate
                print("[MPU] OK en direccion %s (+-4g, +-500 deg/s, 100Hz, DLPF 44Hz)"
                      % hex(candidate))
                break
            except Exception as e:
                print("[MPU] Fallo init en", hex(candidate), ":", e)

    if not mpu:
        print("[MPU] NO detectado. Revisa cableado SDA=8, SCL=9, VCC=3.3V, GND")
except Exception as e:
    print("[I2C] Error:", e)


# Buffer circular: [ax, ay, az, gx, gy, gz, ms]
sensor_buf   = []
MAX_BUF      = 128     # ~1.3s a 100Hz
read_count   = 0        # total de lecturas
last_sample  = None     # ultima lectura para log


# =====================================================================
#  CAMERA
# =====================================================================
cam = Camera(frame_size=FrameSize.VGA, pixel_format=PixelFormat.JPEG)
cam.init()
print("[CAM] VGA 640x480 lista")


# =====================================================================
#  HTTP SERVER (no-bloqueante via select.poll)
# =====================================================================
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('', 80))
srv.listen(5)
srv.setblocking(False)

poller = select.poll()
poller.register(srv, select.POLLIN)

print("[HTTP] Servidor activo (puerto 80)")
print("         /frame   -> JPEG camara")
print("         /sensor  -> JSON MPU")
print("         /status  -> JSON diagnostico")
print("=" * 50)


def build_sensor_json(buf):
    """JSON compacto: {"s":[[ax,ay,az,gx,gy,gz,ms],...]}"""
    if not buf:
        return '{"s":[]}'
    parts = []
    for ax, ay, az, gx, gy, gz, t in buf:
        parts.append('[%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%d]'
                     % (ax, ay, az, gx, gy, gz, t))
    return '{"s":[' + ','.join(parts) + ']}'


def build_status_json():
    mpu_str = '"%s"' % (hex(mpu_addr) if mpu_addr else 'null')
    return ('{"mpu":%s,"reads":%d,"buf":%d}'
            % (mpu_str, read_count, len(sensor_buf)))


# =====================================================================
#  LOOP PRINCIPAL
# =====================================================================
next_log_ms = time.ticks_ms() + 2000   # log cada 2s

while True:
    # ---- 1. Leer sensor cada iteracion ----
    if mpu:
        try:
            s = mpu.read()
            sensor_buf.append((s[0], s[1], s[2], s[3], s[4], s[5], time.ticks_ms()))
            last_sample = s
            read_count += 1
            if len(sensor_buf) > MAX_BUF:
                sensor_buf = sensor_buf[-MAX_BUF:]
        except Exception as e:
            # no imprimir en cada error para no saturar
            pass

    # ---- Log periodico del MPU ----
    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, next_log_ms) >= 0 and last_sample:
        ax, ay, az, gx, gy, gz = last_sample
        print("[MPU] #%d  a=(%+.2f,%+.2f,%+.2f)g  g=(%+6.1f,%+6.1f,%+6.1f)d/s  buf=%d"
              % (read_count, ax, ay, az, gx, gy, gz, len(sensor_buf)))
        next_log_ms = time.ticks_add(now_ms, 2000)

    # ---- 2. Esperar cliente (timeout corto) ----
    events = poller.poll(5)
    if not events:
        continue

    # ---- 3. Atender peticion ----
    client = None
    try:
        client, addr = srv.accept()
        client.settimeout(3)
        request = client.recv(1024).decode()
        path = request.split(' ')[1] if ' ' in request else '/'

        if path == "/frame":
            frame = cam.capture()
            if frame:
                # Content-Length evita truncado/corrupcion JPEG
                hdr = (b'HTTP/1.1 200 OK\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame)).encode() + b'\r\n'
                       b'Cache-Control: no-cache\r\n'
                       b'Connection: close\r\n\r\n')
                client.send(hdr)
                client.sendall(frame)
            else:
                client.send(b'HTTP/1.1 500 Error\r\n\r\n')

        elif path == "/sensor":
            n_samples = len(sensor_buf)
            payload = build_sensor_json(sensor_buf).encode()
            sensor_buf = []
            hdr = (b'HTTP/1.1 200 OK\r\n'
                   b'Content-Type: application/json\r\n'
                   b'Content-Length: ' + str(len(payload)).encode() + b'\r\n'
                   b'Cache-Control: no-cache\r\n\r\n')
            client.send(hdr)
            client.sendall(payload)

        elif path == "/status":
            payload = build_status_json().encode()
            hdr = (b'HTTP/1.1 200 OK\r\n'
                   b'Content-Type: application/json\r\n'
                   b'Content-Length: ' + str(len(payload)).encode() + b'\r\n\r\n')
            client.send(hdr)
            client.sendall(payload)

        else:
            imu_str = hex(mpu_addr) if mpu_addr else "NO DETECTADO"
            html = (
                '<html><head><title>ESP32 UPDRS</title></head><body>'
                '<h1>ESP32-CAM + MPU6050</h1>'
                '<p>Camera: VGA 640x480</p>'
                '<p>MPU6050: %s  Lecturas: %d</p>'
                '<img src="/frame" width="640">'
                '<p><a href="/sensor">/sensor</a> | '
                '<a href="/status">/status</a></p>'
                '</body></html>' % (imu_str, read_count)
            )
            payload = html.encode()
            hdr = (b'HTTP/1.1 200 OK\r\n'
                   b'Content-Type: text/html\r\n'
                   b'Content-Length: ' + str(len(payload)).encode() + b'\r\n\r\n')
            client.send(hdr)
            client.sendall(payload)

    except Exception:
        pass
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass
