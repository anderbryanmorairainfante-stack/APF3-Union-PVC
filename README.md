# APF3 – Detector de Unión de PVC de 1/2" con Haar Cascade

Proyecto APF3 de *Visión Computacional aplicada a Ingeniería Civil* (UTP).
Autor: **Bryan Ander Moraira Infante**

Detecta en tiempo real una **unión universal de PVC de 1/2"** con la cámara de un celular (DroidCam), usando un clasificador Haar Cascade entrenado con un dataset propio (Cascade Trainer GUI + OpenCV).

- Repositorio: https://github.com/anderbryanmorairainfante-stack/APF3-Union-PVC
- Video demostrativo: https://drive.google.com/file/d/1lbOUbQztRxf_Cm95uZp3N5_SRISSLpaA/view?usp=sharing

## Contenido
| Archivo / carpeta | Descripción |
|---|---|
| `APF3_BRYAN_ANDER_MORAIRA_INFANTE.py` | Detector en tiempo real |
| `cascade.xml` | Modelo entrenado (ventana 32×32, 20 etapas) |
| `p/` | 114 imágenes positivas (32×32 px) |
| `n/` | 178 imágenes negativas (cuadro completo 320×180 px) |
| `pos.lst`, `neg.lst`, `pos_samples.vec` | Archivos usados por Cascade Trainer GUI |
| `Detector_UnionPVC_p.py`, `Detector_UnionPVC_n.py` | Programas de captura del dataset |
| `capturas/` | Capturas de resultados |
| `APF3_Informe_Tecnico_MORAIRA_INFANTE.pdf` | Informe técnico |

## Entrenamiento (Cascade Trainer GUI)
Ventana 32×32 · 20 etapas · minHitRate 0.995 · maxFalseAlarmRate 0.5 · weightTrimRate 0.95 · maxDepth 1 · maxWeakCount 100 · Gentle AdaBoost · HAAR/BASIC.

## Uso
```bash
pip install opencv-python
python APF3_BRYAN_ANDER_MORAIRA_INFANTE.py          # DroidCam (editar IP_CELULAR en el código)
python APF3_BRYAN_ANDER_MORAIRA_INFANTE.py 0        # webcam de la PC
python APF3_BRYAN_ANDER_MORAIRA_INFANTE.py foto.jpg # imagen
```
Teclas: `ESC` salir · `C` correcta · `F` falso positivo · `N` no detectado · `G` guardar captura.
Barras: `minNeighbors` (más alto = menos falsos positivos) y `scale x10`.

## Resultados y limitaciones
Sobre el propio dataset: 113 de 114 positivas aceptadas; con minNeighbors 4, 1 de 178 negativas con falsa detección. En vivo: 26–29 FPS. No se midió con un conjunto de prueba independiente; el dataset usa una sola unión y las negativas no incluyen otras piezas de PVC. Ver el informe, secciones 6 y 7.
