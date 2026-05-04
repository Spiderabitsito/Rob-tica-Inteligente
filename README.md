# UPDRS Parte 3 — Análisis cuantitativo de movimientos

Sistema de evaluación cuantitativa de los ejercicios de la **Movement
Disorder Society – UPDRS Parte 3** (golpeteo de dedos, abrir/cerrar mano,
prono-supinación, tremor de reposo) usando una **ESP32-S3 con cámara +
IMU MPU6050** sobre la falange distal del dedo índice, y procesamiento
en PC con **MediaPipe + OpenCV + Pillow**.

---

## Hardware

| Componente | Modelo verificado |
|---|---|
| Placa | Freenove ESP32-S3 WROOM **CAM** (FNK0085) |
| Cámara | OV2640 / OV3660 (incluida en la placa) |
| IMU | MPU6050 GY-521 (3.3 V o 5 V; trae regulador) |
| Sensor mounting | Falange distal del **dedo índice** (la más cercana a la uña) |

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
> cuando no le pasás argumentos — trampa clásica.
> Pines verificados libres en esta placa: 1, 2, 14, 21, 41, 42, 47.

Si por algún motivo 41/42 están ocupados, el firmware prueba como
fallback **GPIO 1 / GPIO 2** automáticamente.

---

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

---

## Setup PC

### Paquetes Python obligatorios

```bash
pip install opencv-python==4.10.0.84 mediapipe==0.10.7 numpy>=1.24 requests Pillow>=9.0
```

| Paquete | Versión recomendada | Por qué |
|---|---|---|
| `opencv-python` | 4.10.x / 4.11.x | Render del canvas, captura de frames, decode JPEG |
| `mediapipe` | **`0.10.7`** | Pinneado: a partir de 0.10.14 eliminaron `mp.solutions.hands` |
| `numpy` | ≥ 1.24 | FFT, filtros, polyfit, manipulación de buffers |
| `requests` | ≥ 2.28 | HTTP polling al ESP32 (`/frame`, `/sensor`) |
| `Pillow` | ≥ 9.0 | Texto antialiased nítido (TTF) — reemplaza `cv2.putText` que se ve "144p" |

> Pillow viene transitivamente con MediaPipe pero lo declaramos
> explícitamente: si en el futuro MediaPipe usa builds slim sin Pillow,
> el código no se rompe.

### Ejecución

```bash
python updrs_vision.py
```

### Atajos en la ventana

| Tecla | Acción |
|---|---|
| `Q` | Salir |
| `R` | Reiniciar contadores, buffers y preprocesador |
| `C` | **Calibrar baseline saludable** (10 s con sensor inmóvil) |
| `G` | Toggle plots crudos (default OCULTO para no sobrecargar) |
| `F` | Toggle FFT spectrum debug (placeholder) |
| `P` | Imprimir métricas detalladas en terminal |

> **Primera vez:** el UPDRS aparece como "Calibrar [C]" hasta que
> presiones C y mantengas el sensor inmóvil durante 10 segundos en
> una **mano sana** (paciente o referencia). El baseline se persiste
> en `~/.updrs_baseline.json` y se reusa en arranques posteriores.

---

## Variables que mide

### Primarias (clínicas)

| Variable | Origen | Significado |
|---|---|---|
| **UPDRS Grade 0-4** | composite score | Lectura clínica: Normal / Leve / Moderado leve / Moderado / Severo |
| **In_a (×)** | RMS de \|a_dyn\| en 5 s, normalizado por baseline | Cuántas σ por encima del baseline saludable está el tremor (Sousa Paixão 2019) |
| **In_g (×)** | RMS de \|ω\| en 5 s, normalizado por baseline | Mismo concepto para velocidad angular |

### Secundarias (diagnóstico)

| Variable | Origen | Correspondencia clínica |
|---|---|---|
| Frecuencia (Hz) | FFT de \|a_dyn\| (banda 0.5–8 Hz) | banda parkinsoniana 3–7 Hz |
| Amplitud visión | rango P95–P5 distancia pulgar–índice | hipocinesia (sólo display, no entra al composite) |
| RMS-a (mg) | RMS aceleración dinámica 5 s | severidad del tremor |
| CV (%) | desv/media intervalos entre toques | irregularidad |
| Jerk (g/s) | derivada de aceleración dinámica | falta de control / suavidad |

### Composite UPDRS (refactorizado)

```
score = 0.50 · In_a   +    ← métrica primaria, calibrada vs población sana
        0.15 · freq   +    ← presencia de pico tremor
        0.25 · CV     +    ← regularidad
        0.10 · jerk        ← suavidad

UPDRS = round(score × 4)
```

In_g se muestra prominentemente pero **no entra en el composite**
porque el giroscopio en la falange distal capta artefactos de cambio
de orientación (sobre todo al hacer finger-tap) que inflan el valor
independientemente de la severidad del tremor.

---

## Pipeline IMU (preprocesamiento)

```
raw → median(3) → IIR LP @ 15 Hz → comp. filter HP @ 1 Hz → a_dynamic
                                                              ↓
                              FFT: resample uniforme + detrend + Hanning
                                                              ↓
                                  RMS en ventana 5 s → In_a, In_g
```

- **Mediana de 3 taps**: elimina picos de un solo sample.
- **IIR pasa-bajos** (α = 0.485, ~15 Hz cutoff): por encima de 15 Hz
  en la falange distal sólo hay ruido HF y artefactos de mount.
- **Filtro complementario** (α = 0.94, HP ~1 Hz): mata el drift lento
  que sesgaba el RMS hacia arriba.
- **Resample + detrend antes de FFT**: elimina drift lineal y muestreo
  irregular; FFT con resolución correcta sobre grid de 100 Hz.
- **Ventana RMS de 5 s**: estándar de Sousa Paixão 2019 para tremor index.

---

## Calibración del baseline saludable

`~/.updrs_baseline.json` ejemplo (datos sintéticos, no de paciente real):

```json
{
  "schema_version": 1,
  "site": "distal_index",
  "fs_hz": 100.0,
  "bp_low_hz": 1.0,
  "bp_high_hz": 15.0,
  "mps_a_g":     0.0125,
  "stdps_a_g":   0.0042,
  "mps_g_dps":   1.18,
  "stdps_g_dps": 0.51,
  "n_samples":   1000,
  "duration_s":  10.0,
  "timestamp":   "2026-05-04T17:42:13"
}
```

El archivo es **autodescriptivo**: si cambia el sample rate, el
bandpass o el sitio del sensor, el código rechaza el baseline y pide
recalibrar. Esto evita scoring contra un baseline obsoleto.

---

## Diagnóstico

`http://192.168.4.1/status` devuelve JSON con los pines reales que el
firmware terminó usando:

```json
{"mpu":"0x68","sda":41,"scl":42,"freq":400000,
 "reads":12450,"errors":0,"buf":7,"cal":true,"interval_ms":10}
```

Y la consola PC imprime cada 3 s un resumen de métricas:

```
[MET] fs=99.8Hz  In_a=8.40x  In_g=12.10x  freq=4.85Hz  CV=18.2%  jerk=4.10g/s  cal=OK
```

---

## Referencias bibliográficas

| Paper | Aporte al sistema |
|---|---|
| **Sousa Paixão, Peres & Andrade (2019)** — *Parameter Estimate from Accelerometer and Gyroscope for Characterization of Wrist Tremor in Individuals with Parkinson's Disease*, en Costa-Felix et al. (eds.), XXVI Brazilian Congress on Biomedical Engineering, IFMBE Proceedings 70/1, p. 513. ISBN 978-981-13-2119-1. | Fórmula del **Tremor Normality Index** `In_j = (rms_j − MPS) / STDPS`, ventana RMS de 5 s, pre-bandpass 1–50 Hz. |
| **Keba, Bachmann, Lass & Rätsep (2025)** — *Assessing Parkinson's Rest Tremor from the Wrist with Accelerometry and Gyroscope Signals in Patients with Deep Brain Stimulation: An Observational Study*, J. Clin. Med. 14, 2073. https://doi.org/10.3390/jcm14062073 | Justificación para usar **escala log** (Weber-Fechner: `log(rms) ∝ UPDRS`), bandpass 1–40 Hz, prioridad de accel sobre gyro en muñeca/falange. |
| **Ito et al. (2020)** — *The relationships between three-axis accelerometer measures of physical activity and motor symptoms in patients with Parkinson's disease*, BMC Neurology 20:340. https://doi.org/10.1186/s12883-020-01896-w | No incorporado en este sprint (orientado a monitoreo diario waist-worn, no a tareas cortas). Idea pendiente: **percentil 95–92.5** como estadística robusta para amplitud visión. |

---

## Agentes / skills usados durante el desarrollo

Este proyecto se construyó con asistencia de **Claude Code** y los
siguientes agentes/skills explícitos:

| Agente / skill | Plugin | Cuándo se usó |
|---|---|---|
| `general-purpose` (research mode) | core | Investigación de la pinout Freenove FNK0085, lectura de papers, brainstorming de integración |
| `simplify` skill | core | Code review en 3 ejes (reuse / quality / efficiency) tras cada commit grande |
| `superpowers:code-reviewer` | superpowers plugin | Code review estructurado contra plan + estándares |
| `everything-claude-code:code-reviewer` | ECC plugin | Code review focused multi-archivo |
| `everything-claude-code:planner` | ECC plugin | Planificación de features grandes (refactor de UPDRSScorer, pipeline IMU) |

> Nota: los agentes los invoca el modelo automáticamente cuando hace
> falta. No requieren configuración por parte del usuario más allá
> de tener los plugins instalados (`/plugin install everything-claude-code`
> y `/plugin install superpowers`).

---

## Estructura del repo

```
.
├── esp32_stream.py        ← Firmware MicroPython (sube a la ESP32 como main.py)
├── updrs_vision.py        ← App PC: vision + IMU + UI clínica
├── README.md              ← este archivo
└── ~/.updrs_baseline.json ← baseline saludable persistido (generado por [C])
```
