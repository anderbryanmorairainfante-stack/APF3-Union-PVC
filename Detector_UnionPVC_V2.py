import os
import threading
import time
import cv2
import numpy as np

# ---------------------------------------------------------------
# DETECTOR DE UNIÓN DE PVC DE 1/2" - V2 (sin barras, parámetros ya fijados)
# Modelo entrenado con muestras de 32 x 32 (proporción 1 : 1).
#
# Base fluida: hilo de cámara + hilo de detección, SIN barras de ajuste:
# todos los parámetros están en la sección PARÁMETROS de abajo.
#
# Teclas:  R = cambiar giro | F = pantalla completa / ventana
#          + / - = subir / bajar minNeighbors (ajuste fino sin barras) | ESC = salir
# ---------------------------------------------------------------

# ---------------- PARÁMETROS ----------------
IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

ARCHIVO_MODELO = "cascade.xml"
ETIQUETA = "Union PVC 1/2"

SCALE_FACTOR = 1.1        # pasos de escala finos (valores altos se saltan el objeto)
MIN_NEIGHBORS = 4         # sube con '+' si salen falsos positivos, baja con '-' si no detecta
PESO_MIN = 0.0            # sin filtro por peso; súbelo si quieres descartar detecciones débiles
MAX_CUADROS = 2           # máximo de detecciones mostradas
MARGEN = 0.10             # el recuadro se agranda 10 % alrededor de la detección

ANCHO_PROCESO = 640       # ancho (px) de la imagen usada SOLO para detectar; mayor = detecta objetos pequeños, pero más lento
MIN_SIZE_REAL = (40, 40)  # tamaño mínimo de la unión (ancho, alto) en la imagen real (1 : 1)

PANTALLA_COMPLETA = True
GIROS = [
    ("sin girar", None),
    ("90 grados derecha", cv2.ROTATE_90_CLOCKWISE),
    ("90 grados izquierda", cv2.ROTATE_90_COUNTERCLOCKWISE),
]
# --------------------------------------------


def tamano_pantalla():
    """Tamaño de la pantalla de Windows; si falla, 1920 x 1080."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


SW, SH = tamano_pantalla()

# ---------------------------------------------------------------
# MODELO Y CÁMARA
# ---------------------------------------------------------------
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def buscar_modelo(nombre):
    """Busca el modelo junto al script, en 'classifier/', en la carpeta de trabajo
    actual y, por último, en cualquier subcarpeta del script."""
    candidatos = [
        os.path.join(CARPETA_SCRIPT, nombre),
        os.path.join(CARPETA_SCRIPT, "classifier", nombre),
        os.path.join(os.getcwd(), nombre),
    ]
    for ruta in candidatos:
        if os.path.isfile(ruta):
            return ruta
    for raiz, _, archivos in os.walk(CARPETA_SCRIPT):
        if nombre in archivos:
            return os.path.join(raiz, nombre)
    return None


ruta_modelo = buscar_modelo(ARCHIVO_MODELO)
if ruta_modelo is None:
    print(f"ERROR: no se encontró '{ARCHIVO_MODELO}'.")
    print("Se buscó en:", CARPETA_SCRIPT)
    raise SystemExit
print("Modelo cargado desde:", ruta_modelo)

clasificador = cv2.CascadeClassifier(ruta_modelo)
if clasificador.empty():
    print(f"ERROR: se encontró '{ruta_modelo}' pero OpenCV no pudo leerlo.")
    raise SystemExit

cap = None
for url in URLS_PRUEBA:
    print("Probando:", url)
    prueba = cv2.VideoCapture(url)
    if prueba.isOpened():
        ok, _ = prueba.read()
        if ok:
            print("CONECTADO con:", url)
            cap = prueba
            break
    prueba.release()

if cap is None:
    print("No se pudo conectar con DroidCam. Probando la webcam de la PC...")
    prueba = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if prueba.isOpened():
        cap = prueba

if cap is None:
    print("ERROR: no se pudo abrir ninguna cámara.")
    raise SystemExit

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class LectorCamara:
    """Lee la cámara en un hilo aparte y guarda solo el cuadro más reciente."""

    def __init__(self, captura):
        self.captura = captura
        self.frame = None
        self.n = 0
        self.ok = True
        self.lock = threading.Lock()
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self):
        while self.ok:
            ret, f = self.captura.read()
            if not ret:
                self.ok = False
                break
            with self.lock:
                self.frame = f
                self.n += 1

    def ultimo(self):
        with self.lock:
            return self.frame, self.n


lector = LectorCamara(cap)

# Estado compartido entre el hilo principal y el de detección
estado = {
    "salir": False,
    "giro": 0,
    "min_neighbors": MIN_NEIGHBORS,
    "resultado": (0, []),     # (giro con el que se calculó, [(x, y, w, h, peso)])
}


def detectar(gris, min_neighbors, min_size):
    rects, _, pesos = clasificador.detectMultiScale3(
        gris,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=min_neighbors,
        minSize=min_size,
        outputRejectLevels=True,
    )
    pesos = np.array(pesos).flatten()
    return [(int(x), int(y), int(w), int(h), float(p))
            for (x, y, w, h), p in zip(rects, pesos)]


def trabajador_deteccion():
    ultimo_proc = -1
    while not estado["salir"]:
        frame, n = lector.ultimo()
        if frame is None or n == ultimo_proc:
            time.sleep(0.005)
            continue
        ultimo_proc = n

        giro_id = estado["giro"]
        min_neighbors = estado["min_neighbors"]

        codigo = GIROS[giro_id][1]
        rot = cv2.rotate(frame, codigo) if codigo is not None else frame
        h, w = rot.shape[:2]
        escala = ANCHO_PROCESO / w
        pequeno = cv2.resize(rot, (ANCHO_PROCESO, int(h * escala)))
        gris = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
        min_size = (int(MIN_SIZE_REAL[0] * escala), int(MIN_SIZE_REAL[1] * escala))

        finales = []
        for (x, y, dw, dh, p) in detectar(gris, min_neighbors, min_size):
            if p >= PESO_MIN:
                finales.append((int(x / escala), int(y / escala),
                                int(dw / escala), int(dh / escala), p))
        finales.sort(key=lambda d: d[4], reverse=True)
        estado["resultado"] = (giro_id, finales[:MAX_CUADROS])


threading.Thread(target=trabajador_deteccion, daemon=True).start()

# ---------------------------------------------------------------
# VENTANA (sin barras)
# ---------------------------------------------------------------
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
if PANTALLA_COMPLETA:
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

giro_actual = 0
completa = PANTALLA_COMPLETA
ultimo_n = -1
lienzo = None
dims_previas = None
fps = 0.0
t_anterior = time.time()

while True:
    frame, n = lector.ultimo()

    if frame is not None and n != ultimo_n:
        ultimo_n = n
        estado["giro"] = giro_actual

        # Girar y reducir la imagen al tamaño de pantalla ANTES de dibujar
        nombre_giro, codigo = GIROS[giro_actual]
        rot = cv2.rotate(frame, codigo) if codigo is not None else frame
        h, w = rot.shape[:2]
        esc = min(SW / w, SH / h) if completa else min(1.0, (SW * 0.9) / w, (SH * 0.8) / h)
        vista = cv2.resize(rot, (max(int(w * esc), 1), max(int(h * esc), 1)))
        vh, vw = vista.shape[:2]

        # Dibujar las detecciones calculadas por el otro hilo
        giro_res, lista = estado["resultado"]
        if giro_res == giro_actual:
            for (x, y, dw, dh, p) in lista:
                x, y, dw, dh = int(x * esc), int(y * esc), int(dw * esc), int(dh * esc)
                mx, my = int(dw * MARGEN), int(dh * MARGEN)
                x1, y1 = max(x - mx, 0), max(y - my, 0)
                x2, y2 = min(x + dw + mx, vw - 1), min(y + dh + my, vh - 1)
                cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vista, f"{ETIQUETA} {p:.1f}", (x1, max(y1 - 12, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            n_cuadros = len(lista)
        else:
            n_cuadros = 0

        ahora = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(ahora - t_anterior, 1e-6))
        t_anterior = ahora
        cv2.putText(vista, f"FPS: {fps:.0f}  Detecciones: {n_cuadros}  minNeighbors: {estado['min_neighbors']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vista, "R: girar | F: pantalla completa | +/-: sensibilidad | ESC: salir",
                    (10, vh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if completa:
            # Lienzo reutilizado (no se crea uno nuevo en cada cuadro)
            if lienzo is None or lienzo.shape[:2] != (SH, SW):
                lienzo = np.zeros((SH, SW, 3), dtype=np.uint8)
            if dims_previas != (vh, vw):
                lienzo[:] = 0
                dims_previas = (vh, vw)
            y0, x0 = (SH - vh) // 2, (SW - vw) // 2
            lienzo[y0:y0 + vh, x0:x0 + vw] = vista
            cv2.imshow("frame", lienzo)
        else:
            cv2.imshow("frame", vista)
    else:
        if not lector.ok:
            print("Se perdió la conexión con la cámara.")
            break
        time.sleep(0.003)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k in (ord('r'), ord('R')):
        giro_actual = (giro_actual + 1) % len(GIROS)
    elif k in (ord('f'), ord('F')):
        completa = not completa
        dims_previas = None
        cv2.setWindowProperty(
            "frame", cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if completa else cv2.WINDOW_NORMAL)
    elif k in (ord('+'), ord('=')):
        estado["min_neighbors"] = min(estado["min_neighbors"] + 1, 100)
    elif k in (ord('-'), ord('_')):
        estado["min_neighbors"] = max(estado["min_neighbors"] - 1, 1)

estado["salir"] = True
lector.ok = False
time.sleep(0.1)
cap.release()
cv2.destroyAllWindows()
