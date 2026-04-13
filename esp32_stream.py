"""
ESP32-CAM Stream Server
Servidor HTTP que envía frames JPEG vía WiFi AP
Resolución: VGA (640x480)
"""

import network
import socket
from camera import Camera, FrameSize, PixelFormat

# -------- WIFI AP ----------
ap_if = network.WLAN(network.AP_IF)
ap_if.active(True)

ap_if.config(
    essid="ESP32-S312",
    password="12345678",
    authmode=network.AUTH_WPA2_PSK,
    max_clients=4,
    channel=1,
    hidden=False
)

print("\n--- Punt d'accés actiu ---")
print("IP:", ap_if.ifconfig())

# -------- CAMERA ----------
# Cambiado de QVGA (320x240) a VGA (640x480) para mejor resolución
cam = Camera(frame_size=FrameSize.VGA, pixel_format=PixelFormat.JPEG)
cam.init()

# -------- WEB ----------
def web_page():
    html = """<html>
    <head>
        <title>ESP32-CAM UPDRS</title>
    </head>
    <body>
        <h1>Streaming ESP32-CAM - UPDRS</h1>
        <img src="/frame" width="640">
    </body>
    </html>"""
    return html

# -------- SERVER ----------
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('', 80))
s.listen(5)

print("Servidor actiu en puerto 80")

# -------- LOOP ----------
while True:
    try:
        client, addr = s.accept()
        request = client.recv(1024).decode()
        path = request.split(' ')[1]

        if path == "/frame":
            frame = cam.capture()
            if frame:
                headers = (
                    b'HTTP/1.1 200 OK\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Access-Control-Allow-Origin: *\r\n'
                    b'Cache-Control: no-cache\r\n'
                    b'\r\n'
                )
                client.send(headers)
                client.sendall(frame)
            else:
                client.send(b'HTTP/1.1 500 Internal Server Error\r\n\r\n')

        else:
            response = web_page().encode()
            client.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n')
            client.sendall(response)

    except Exception as e:
        print("Error:", e)

    finally:
        client.close()
