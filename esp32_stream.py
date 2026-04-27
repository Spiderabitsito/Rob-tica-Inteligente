"""
ESP32-S3 CAM + MPU6050 - Servidor HTTP
=======================================
Endpoints:
  /frame   -> JPEG VGA 640x480
  /sensor  -> JSON {"s":[[ax,ay,az,gx,gy,gz,ms], ...]}
  /status  -> JSON diagnostico (mpu addr, n_reads, n_errors, calibrado)
  /        -> Pagina HTML

Conexiones MPU6050:
  SDA -> GPIO 8     SCL -> GPIO 9
  VCC -> 3.3V       GND -> GND
  AD0 -> GND (0x68) o 3.3V (0x69)

Cambios respecto a la version anterior:
  - Buffer circular real con deque(maxlen=128) (antes lista + slice)
  - Auto-calibracion de gyro en boot (~1s de muestras estaticas)
  - Recovery I2C: reinicia el bus tras N errores consecutivos
  - Rate gate de muestreo a 100 Hz (evita oversampling)
  - send_response() helper - elimina copia-pega en handlers
  - /status reporta cal, errors, addr, fs efectiva
"""

import network
import socket
import select
import struct
import time
from collections import deque
from machine import I2C, Pin
from camera import Camera, FrameSize, PixelFormat


# Cross-reference contract: updrs_vision.py:IMU_PAYLOAD_KEY must match.
IMU_PAYLOAD_KEY = "s"

# Sample rate target. The MPU runs at 100 Hz internally; we gate reads
# so we never sample faster than this even if the loop runs faster.
SAMPLE_INTERVAL_MS = 10            # 100 Hz
BUF_LEN            = 200           # ~2 s @ 100 Hz - PC poll = 80 ms
GYRO_CAL_SAMPLES   = 100           # ~1 s static at boot
I2C_REINIT_AFTER   = 20            # consecutive read errors before bus reset


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
    hidden=False,
)
print("")
print("=" * 50)
print("  ESP32 UPDRS - AP IP:", ap.ifconfig()[0])
print("=" * 50)


# =====================================================================
#  MPU6050 DRIVER
# =====================================================================
class MPU6050:
    _ASCALE = {0x00: 16384.0, 0x08: 8192.0, 0x10: 4096.0, 0x18: 2048.0}
    _GSCALE = {0x00: 131.0,   0x08: 65.5,   0x10: 32.8,   0x18: 16.4}

    def __init__(self, i2c, addr=0x68, accel_range=0x08, gyro_range=0x08):
        self.i2c   = i2c
        self.addr  = addr
        self.a_div = self._ASCALE.get(accel_range, 8192.0)
        self.g_div = self._GSCALE.get(gyro_range, 65.5)
        # Boot-calibrated gyro offsets (deg/s); subtracted from every read.
        self.gx_off = 0.0
        self.gy_off = 0.0
        self.gz_off = 0.0
        self._configure()

    def _configure(self):
        self.i2c.writeto_mem(self.addr, 0x6B, b'\x00')   # wake
        time.sleep_ms(100)
        self.i2c.writeto_mem(self.addr, 0x19, b'\x09')   # 1 kHz / 10 = 100 Hz
        self.i2c.writeto_mem(self.addr, 0x1A, b'\x03')   # DLPF 44 Hz
        self.i2c.writeto_mem(self.addr, 0x1C, bytes([0x08]))  # +-4g
        self.i2c.writeto_mem(self.addr, 0x1B, bytes([0x08]))  # +-500 deg/s

    def read_raw(self):
        raw = self.i2c.readfrom_mem(self.addr, 0x3B, 14)
        ax = struct.unpack('>h', raw[0:2])[0]   / self.a_div
        ay = struct.unpack('>h', raw[2:4])[0]   / self.a_div
        az = struct.unpack('>h', raw[4:6])[0]   / self.a_div
        gx = struct.unpack('>h', raw[8:10])[0]  / self.g_div
        gy = struct.unpack('>h', raw[10:12])[0] / self.g_div
        gz = struct.unpack('>h', raw[12:14])[0] / self.g_div
        return ax, ay, az, gx, gy, gz

    def read(self):
        ax, ay, az, gx, gy, gz = self.read_raw()
        return ax, ay, az, gx - self.gx_off, gy - self.gy_off, gz - self.gz_off

    def calibrate_gyro(self, n_samples=GYRO_CAL_SAMPLES):
        """Average n_samples of the gyro at rest. Caller must keep sensor still."""
        sx = sy = sz = 0.0
        ok = 0
        for _ in range(n_samples):
            try:
                _, _, _, gx, gy, gz = self.read_raw()
                sx += gx; sy += gy; sz += gz
                ok += 1
            except Exception:
                pass
            time.sleep_ms(SAMPLE_INTERVAL_MS)
        if ok > 0:
            self.gx_off = sx / ok
            self.gy_off = sy / ok
            self.gz_off = sz / ok
        return ok


# =====================================================================
#  SENSOR STATE
# =====================================================================
class SensorState:
    def __init__(self):
        self.mpu          = None
        self.mpu_addr     = None
        self.buf          = deque((), BUF_LEN)
        self.read_count   = 0
        self.error_count  = 0
        self.consec_err   = 0
        self.last_sample  = None
        self.calibrated   = False
        self.last_read_ms = 0

    def append(self, sample, ts_ms):
        self.buf.append((sample[0], sample[1], sample[2],
                         sample[3], sample[4], sample[5], ts_ms))
        self.last_sample = sample
        self.read_count += 1

    def drain(self):
        out = list(self.buf)
        self.buf = deque((), BUF_LEN)
        return out


sensor = SensorState()


def init_i2c():
    """Returns (i2c, mpu) or (None, None) on failure. Tries both addresses."""
    print("[I2C] Init SDA=GPIO8, SCL=GPIO9, 400kHz...")
    try:
        i2c = I2C(0, sda=Pin(8), scl=Pin(9), freq=400000)
        devs = i2c.scan()
        print("[I2C] Devices:", [hex(d) for d in devs])
        for addr in (0x68, 0x69):
            if addr in devs:
                try:
                    return i2c, MPU6050(i2c, addr=addr), addr
                except Exception as e:
                    print("[MPU] init failed at", hex(addr), ":", e)
        print("[MPU] NOT detected. Check wiring SDA=8, SCL=9, VCC=3.3V, GND")
    except Exception as e:
        print("[I2C] Bus init error:", e)
    return None, None, None


# Boot: init + calibrate
i2c_bus, sensor.mpu, sensor.mpu_addr = init_i2c()
if sensor.mpu:
    print("[MPU] OK at %s (+-4g, +-500 d/s, 100Hz, DLPF 44Hz)" % hex(sensor.mpu_addr))
    print("[CAL] Calibrating gyro - keep sensor still for ~1s ...")
    n_ok = sensor.mpu.calibrate_gyro()
    if n_ok > 0:
        sensor.calibrated = True
        print("[CAL] Done (%d samples)  offsets g=(%.2f,%.2f,%.2f) d/s"
              % (n_ok, sensor.mpu.gx_off, sensor.mpu.gy_off, sensor.mpu.gz_off))
    else:
        print("[CAL] FAILED (no samples). Continuing without offset.")


# =====================================================================
#  CAMERA
# =====================================================================
cam = Camera(frame_size=FrameSize.VGA, pixel_format=PixelFormat.JPEG)
cam.init()
print("[CAM] VGA 640x480 ready")


# =====================================================================
#  HTTP SERVER
# =====================================================================
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('', 80))
srv.listen(5)
srv.setblocking(False)

poller = select.poll()
poller.register(srv, select.POLLIN)

print("[HTTP] Active port 80   /frame /sensor /status")
print("=" * 50)


def build_sensor_json(buf):
    if not buf:
        return '{"%s":[]}' % IMU_PAYLOAD_KEY
    parts = []
    for ax, ay, az, gx, gy, gz, t in buf:
        parts.append('[%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%d]'
                     % (ax, ay, az, gx, gy, gz, t))
    return '{"%s":[' % IMU_PAYLOAD_KEY + ','.join(parts) + ']}'


def build_status_json():
    addr = '"%s"' % hex(sensor.mpu_addr) if sensor.mpu_addr else 'null'
    return ('{"mpu":%s,"reads":%d,"errors":%d,"buf":%d,"cal":%s,"interval_ms":%d}'
            % (addr, sensor.read_count, sensor.error_count,
               len(sensor.buf), 'true' if sensor.calibrated else 'false',
               SAMPLE_INTERVAL_MS))


def send_response(client, content_type, payload, extra=b''):
    """Single source of truth for HTTP responses (replaces 4x copy-paste)."""
    hdr = (b'HTTP/1.1 200 OK\r\n'
           b'Content-Type: ' + content_type + b'\r\n'
           b'Content-Length: ' + str(len(payload)).encode() + b'\r\n'
           b'Cache-Control: no-cache\r\n'
           b'Connection: close\r\n'
           + extra +
           b'\r\n')
    client.send(hdr)
    client.sendall(payload)


def reinit_i2c():
    """Try to recover from runaway I2C errors by re-creating the bus + driver."""
    global i2c_bus
    print("[I2C] Reinit after %d consecutive errors" % sensor.consec_err)
    try:
        i2c_bus, sensor.mpu, sensor.mpu_addr = init_i2c()
        sensor.consec_err = 0
        if sensor.mpu:
            print("[I2C] Recovered at %s" % hex(sensor.mpu_addr))
    except Exception as e:
        print("[I2C] Reinit failed:", e)


# =====================================================================
#  MAIN LOOP
# =====================================================================
next_log_ms = time.ticks_ms() + 2000

while True:
    now_ms = time.ticks_ms()

    # 1. Sample MPU at most every SAMPLE_INTERVAL_MS
    if sensor.mpu and time.ticks_diff(now_ms, sensor.last_read_ms) >= SAMPLE_INTERVAL_MS:
        try:
            s = sensor.mpu.read()
            sensor.append(s, now_ms)
            sensor.last_read_ms = now_ms
            sensor.consec_err = 0
        except Exception:
            sensor.error_count += 1
            sensor.consec_err += 1
            if sensor.consec_err >= I2C_REINIT_AFTER:
                reinit_i2c()

    # Periodic diagnostic log
    if time.ticks_diff(now_ms, next_log_ms) >= 0 and sensor.last_sample:
        ax, ay, az, gx, gy, gz = sensor.last_sample
        print("[MPU] #%d  a=(%+.2f,%+.2f,%+.2f)g  g=(%+6.1f,%+6.1f,%+6.1f)d/s  buf=%d  err=%d"
              % (sensor.read_count, ax, ay, az, gx, gy, gz,
                 len(sensor.buf), sensor.error_count))
        next_log_ms = time.ticks_add(now_ms, 2000)

    # 2. Service HTTP (5 ms timeout)
    events = poller.poll(5)
    if not events:
        continue

    client = None
    try:
        client, addr = srv.accept()
        client.settimeout(3)
        request = client.recv(1024).decode()
        path = request.split(' ')[1] if ' ' in request else '/'

        if path == "/frame":
            frame = cam.capture()
            if frame:
                send_response(client, b'image/jpeg', frame)
            else:
                client.send(b'HTTP/1.1 500 Error\r\n\r\n')

        elif path == "/sensor":
            samples = sensor.drain()
            send_response(client, b'application/json',
                          build_sensor_json(samples).encode())

        elif path == "/status":
            send_response(client, b'application/json',
                          build_status_json().encode())

        else:
            imu_str = hex(sensor.mpu_addr) if sensor.mpu_addr else "NO DETECTADO"
            cal_str = "OK" if sensor.calibrated else "PENDIENTE"
            html = (
                '<html><head><title>ESP32 UPDRS</title></head><body>'
                '<h1>ESP32-CAM + MPU6050</h1>'
                '<p>Camera: VGA 640x480</p>'
                '<p>MPU6050: %s &nbsp; Cal: %s &nbsp; Reads: %d &nbsp; Errors: %d</p>'
                '<img src="/frame" width="640">'
                '<p><a href="/sensor">/sensor</a> | '
                '<a href="/status">/status</a></p>'
                '</body></html>' % (imu_str, cal_str,
                                    sensor.read_count, sensor.error_count)
            )
            send_response(client, b'text/html', html.encode())

    except Exception:
        # Specific OSError / ValueError would be tighter, but MicroPython's
        # request decoder raises a grab-bag; broad catch keeps the server alive.
        pass
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass
