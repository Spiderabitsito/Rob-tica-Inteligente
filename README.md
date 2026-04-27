# UPDRS Parte 3 — Análisis cuantitativo de movimientos

Sistema de evaluación cuantitativa de los ejercicios UPDRS Parte 3
(golpeteo de dedos, abrir/cerrar mano, prono-supinación) usando una
ESP32-S3 con cámara + IMU MPU6050 y procesamiento en PC con
MediaPipe + OpenCV.

## Hardware

| Componente | Modelo verificado |
|---|---|
| Placa | Freenove ESP32-S3 WROOM **CAM** (FNK0085) |
| Cámara | OV2640 / OV3660 (incluida en la placa) |
| IMU | MPU6050 GY-521 (3.3 V o 5 V; trae regulador) |

## Cableado MPU6050 ↔ Freenove FNK0085

| MPU6050 (GY-521) | ESP32-S3 (Freenove) |
|---|---|
| **VCC** | 3.3 V (también acepta 5 V) |
| **GND** | GND |
| **SDA** | **GPIO 41** |
| **SCL** | **GPIO 42** |
| **AD0** | GND → dirección `0x68` (o 3.3 V → `0x69`) |
| INT, XCL, XDA | sin conectar |

> **¿Por qué no GPIO 8 / 9?**
> En la Freenove FNK0085, GPIO 8 = `Y4` y GPIO 9 = `Y3` del bus paralelo
> de la cámara. Además, MicroPython los usa como default de `I2C(0)`
> cuando no le pasás argumentos — trampa clásica que hace que el `scan()`
> parezca dar señal de vida pero choque con la cámara.
> Pines verificados libres en esta placa: 1, 2, 14, 21, 41, 42, 47.

Si por algún motivo 41/42 están ocupados en tu hardware, el firmware
prueba automáticamente como fallback el par **GPIO 1 / GPIO 2**.

## Setup ESP32 (Thonny)

1. Flashea el firmware Freenove ESP32-S3 CAM con MicroPython
   (incluye los módulos `camera`, `network`, `socket`).
2. Sube `esp32_stream.py` al chip y renómbralo `main.py` para que
   arranque en cada boot.
3. **No se requiere instalar ningún paquete** vía `mip` ni copiar
   archivos extra. Todo lo que usa el firmware (`network`, `socket`,
   `select`, `struct`, `time`, `collections`, `machine`, `camera`)
   ya viene en el firmware Freenove.
4. Conecta el MPU6050 según la tabla de arriba ANTES de encender la
   placa, y mantén el sensor inmóvil durante el primer segundo
   (auto-cal del giroscopio).

Al boot, en la consola Thonny deberías ver algo como:

```
[CAM] VGA 640x480 ready
[I2C] Probing buses (Freenove FNK0085 free pins: 1,2,14,21,41,42,47)
[I2C] bus1 sda=41 scl=42 freq=400000  devs=['0x68']
[MPU] OK at 0x68 on bus1 sda=41 scl=42 freq=400000 (+-4g, +-500 d/s, 100Hz)
[CAL] Calibrating gyro - keep sensor still for ~1s ...
[CAL] Done (100 samples)  offsets g=(0.42,-0.18,0.05) d/s
[HTTP] Active port 80   /frame /sensor /status
```

Si en lugar de `0x68` aparece `[]`, revisa cableado y reinicia.

## Setup PC

```bash
pip install opencv-python mediapipe==0.10.7 numpy requests
```

> `mediapipe==0.10.7` está pinneado a propósito: a partir de 0.10.14
> eliminaron la API `mp.solutions.hands` que usamos.

## Uso

1. Conecta la PC al Wi-Fi `ESP32-S312` (clave `12345678`).
2. Verifica que el ESP32 responde abriendo `http://192.168.4.1/` en el navegador.
3. Corre `python updrs_vision.py`.

### Atajos en la ventana

| Tecla | Acción |
|---|---|
| `Q` | Salir |
| `R` | Reiniciar contadores, buffers y preprocesador |
| `P` | Imprimir métricas detalladas en terminal |

## Variables que mide

| # | Variable | Origen | Correspondencia clínica |
|---|---|---|---|
| 1 | Frecuencia (Hz) | FFT de aceleración (sin gravedad) | bradicinesia |
| 2 | Amplitud | rango P95–P5 distancia pulgar–índice (vision) | hipocinesia |
| 3 | Velocidad angular (°/s) | RMS giroscopio | lentitud |
| 4 | CV intervalos tap | desv/media de tiempos entre toques | irregularidad |
| 5 | Jerk (g/s) | derivada de aceleración dinámica | falta de control |

Cada variable se normaliza a [0, 1] (Normal → Severo) y se combina con
pesos (25/20/15/25/15) en un índice compuesto que se mapea al UPDRS 0–4.

## Pipeline IMU (preprocesamiento)

```
raw → median(3) → IIR low-pass @ ~16 Hz → comp. filter gravedad → a_dynamic
                                                                    ↓
                                              FFT: resample uniforme + detrend + Hanning
```

- **Mediana de 3 taps**: elimina picos de un solo sample.
- **IIR pasa-bajos** (α = 0.5, ~16 Hz cutoff): por encima de la banda de
  tremor (0.5–8 Hz) pero suficiente para ruido alto.
- **Filtro complementario** (α = 0.97): sigue la gravedad para aislar
  aceleración dinámica.
- **Resample + detrend antes de FFT**: elimina drift lineal y muestreo
  irregular para que la FFT tenga resolución correcta.

## Diagnóstico

`http://192.168.4.1/status` devuelve JSON con los pines reales que el
firmware terminó usando, errores I2C, calibración y ratio de muestras:

```json
{"mpu":"0x68","sda":41,"scl":42,"freq":400000,
 "reads":12450,"errors":0,"buf":7,"cal":true,"interval_ms":10}
```

Y la consola Thonny imprime cada 2 s la última lectura del MPU:

```
[MPU] #12450  a=(+0.02,-0.04,+0.99)g  g=(+0.3,-0.1,+0.2)d/s  buf=7  err=0
```
