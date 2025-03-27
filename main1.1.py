import cv2
import numpy as np
import pytesseract
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph

# Configuración de la variable de entorno
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ajustar_brillo_contraste(imagen, alpha=2.0, beta=20):
    return cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

def procesar_imagen_para_ocr(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    ajustada = ajustar_brillo_contraste(suavizada, alpha=2.0, beta=30)
    umbral = cv2.adaptiveThreshold(ajustada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return umbral

def obtener_zona_de_texto(imagen, original):
    contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 500:
            continue

        epsilon = 0.02 * cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, epsilon, True)
        cv2.drawContours(original, [approx], -1, (0, 255, 0), 2)

        if len(approx) == 4:
            return approx
    return None

def transformar_perspectiva(imagen, contorno):
    puntos = contorno.reshape(4, 2)
    puntos = sorted(puntos, key=lambda x: x[1])

    top = sorted(puntos[:2], key=lambda x: x[0])
    bottom = sorted(puntos[2:], key=lambda x: x[0])

    puntos_ordenados = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    ancho = max(np.linalg.norm(puntos_ordenados[0] - puntos_ordenados[1]),
                np.linalg.norm(puntos_ordenados[2] - puntos_ordenados[3]))
    alto = max(np.linalg.norm(puntos_ordenados[0] - puntos_ordenados[3]),
               np.linalg.norm(puntos_ordenados[1] - puntos_ordenados[2]))

    destino = np.array([[0, 0], [ancho, 0], [ancho, alto], [0, alto]], dtype="float32")
    matriz = cv2.getPerspectiveTransform(puntos_ordenados, destino)
    warp = cv2.warpPerspective(imagen, matriz, (int(ancho), int(alto)))
    return warp


def guardar_texto_en_pdf(texto):
    # Crear un nuevo PDF y escribir el texto extraído
    pdf_filename = "texto_extraido.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

    # Estilos personalizados
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        name='CustomStyle',
        fontName='Helvetica',
        fontSize=12,
        textColor=colors.black,
        spaceAfter=12,
        spaceBefore=12,
        alignment=0  # 0 = left, 1 = center, 2 = right
    )

    # Dividir el texto en párrafos para mejor formato
    paragraphs = []
    for line in texto.splitlines():
        if line.strip():  # vacías
            paragraphs.append(Paragraph(line, custom_style))

    # con párrafos
    doc.build(paragraphs)


# Captura desde la cámara
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    cv2.imshow('Captura de texto - Presiona "s" para capturar', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        image = frame.copy()
        preprocesada = procesar_imagen_para_ocr(image)
        cv2.imshow('Imagen preprocesada', preprocesada)

        contorno = obtener_zona_de_texto(preprocesada, image)

        if contorno is not None:
            corregida = transformar_perspectiva(image, contorno)
            cv2.imshow('Imagen corregida', corregida)

            config = '--oem 3 --psm 6 -l spa'
            texto = pytesseract.image_to_string(corregida, config=config).strip()

            print("Texto procesado:", texto)
            if texto:
                print('Texto extraído:', texto)
                guardar_texto_en_pdf(texto) 
            else:
                print("No se detectó texto legible.")
        else:
            print("No se encontró texto en la imagen.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

