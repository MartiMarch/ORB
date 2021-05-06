"""
    Algoritmo ORB.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Se abre la cámara
camara = cv2.VideoCapture(2)

# Se crea un objeto ORB
orb = cv2.ORB_create()

while True:
    # Se consume una imagen
    _,imagen = camara.read()

    # Se crea un punto clave
    puntosClave = orb.detect(imagen, None)

    # Se obtienen los descriptores
    puntosClave, descriptores = orb.compute(imagen, puntosClave)

    # Se dibujan los puntos clave
    imagenConPuntos = cv2.drawKeypoints(imagen, puntosClave, outImage=None, color=(255, 0, 0))

    # Se muestran los puntos clave
    cv2.imshow("ORB", imagenConPuntos)

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break