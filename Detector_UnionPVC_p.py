import os
import cv2

# ---------------------------------------------------------------
# CAPTURA DE POSITIVAS (imágenes CON la unión de PVC de 1/2")
# La unión se encuadra con un recuadro CUADRADO (1 : 1) y cada positiva
# se guarda de 32 x 32 px (ancho x alto).
#
# IMPORTANTE: este tamaño (32 x 32) debe ser el MISMO que pongas como
# "Sample width / Sample height" en Cascade Trainer GUI.
#
# Controles (con la ventana de video seleccionada):
#   S o ESPACIO = guardar la muestra
#   A / D       = mover el recuadro a la izquierda / derecha
#   W / X       = mover el recuadro hacia arriba / abajo
#   + / -       = agrandar / achicar el recuadro
#   M           = activar / desactivar guardar también la copia en espejo
#   ESC         = salir
#
# Consejo: la unión es un objeto PEQUEÑO, acércala a la cámara y encuádrala
# SOLO a ella (que llene casi todo el recuadro, con poco fondo alrededor).
# Varía: ángulo (de lado, de frente viendo el hueco, inclinada), distancia,
# iluminación, fondo, y la posición: en la mano, sobre una mesa, sobre una
# tubería, en una caja de obra, etc.
# Si tienes uniones de distintos colores (blanca, celeste, gris), inclúyelas.
# ---------------------------------------------------------------
Datos = "p"

IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

RELACION = 1.0            # ancho / alto del recuadro (1 : 1, como la muestra de 32 x 32)
ANCHO_INICIAL = 0.25      # fracción del ancho de la imagen que ocupa el recuadro al iniciar
TAM_MUESTRA = (32, 32)    # (ancho, alto) con el que se guarda cada positiva
GUARDAR_ESPEJO = False    # True = guarda también la copia volteada (la unión es casi simétrica)

# ---------------------------------------------------------------
carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), Datos)
os.makedirs(carpeta, exist_ok=True)
print("Las imágenes se guardarán en:", carpeta)

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
    print("Revisa: app DroidCam abierta, misma red WiFi e IP correcta.")
    raise SystemExit

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ok, frame = cap.read()
if not ok:
    print("No se recibe imagen de la cámara.")
    raise SystemExit
alto_f, ancho_f = frame.shape[:2]

bw = int(ancho_f * ANCHO_INICIAL)   # ancho del recuadro
cx = ancho_f // 2                   # centro horizontal del recuadro
cy = alto_f // 2                    # centro vertical del recuadro

count = len([f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")])
print("Controles: [S/ESPACIO] guardar | [A]/[D] izq-der | [W]/[X] arriba-abajo | [+]/[-] tamaño | [M] espejo | [ESC] salir")


def guardar(img):
    global count
    ruta = os.path.join(carpeta, f"objeto_{count}.jpg")
    if cv2.imwrite(ruta, img):
        print("Imagen guardada:", ruta)
        count += 1
    else:
        print("ERROR: no se pudo guardar en", ruta)


while True:
    ret, frame = cap.read()
    if not ret:
        print("No se recibe imagen de la cámara. Fin.")
        break

    imAux = frame.copy()

    # Recuadro 1 : 1, sin salirse de la imagen
    bw = max(min(bw, ancho_f, int(alto_f * RELACION)), 40)
    bh = int(bw / RELACION)
    x1 = max(0, min(cx - bw // 2, ancho_f - bw))
    y1 = max(0, min(cy - bh // 2, alto_f - bh))
    x2, y2 = x1 + bw, y1 + bh

    # Oscurecer todo lo que queda FUERA del recuadro: lo claro es lo que se guarda
    sombra = (frame * 0.35).astype("uint8")
    sombra[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    frame = sombra

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
    cv2.putText(frame, f"Carpeta {Datos}: {count} imagenes", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Union PVC DENTRO del recuadro | S guardar | M espejo: "
                + ("SI" if GUARDAR_ESPEJO else "NO"), (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Lo que se guardará (32 x 32), ampliado solo para verlo
    objeto = cv2.resize(imAux[y1:y2, x1:x2], TAM_MUESTRA, interpolation=cv2.INTER_AREA)
    vista = cv2.resize(objeto, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("frame", frame)
    cv2.imshow("objeto", vista)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord('s'), ord('S'), 32):
        guardar(objeto)
        if GUARDAR_ESPEJO:
            guardar(cv2.flip(objeto, 1))
    elif k in (ord('a'), ord('A')):
        cx -= 10
    elif k in (ord('d'), ord('D')):
        cx += 10
    elif k in (ord('w'), ord('W')):
        cy -= 10
    elif k in (ord('x'), ord('X')):
        cy += 10
    elif k in (ord('+'), ord('=')):
        bw += 20
    elif k == ord('-'):
        bw -= 20
    elif k in (ord('m'), ord('M')):
        GUARDAR_ESPEJO = not GUARDAR_ESPEJO
        print("Guardar copia en espejo:", "SI" if GUARDAR_ESPEJO else "NO")
    elif k == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Total de positivas en la carpeta:", count)
