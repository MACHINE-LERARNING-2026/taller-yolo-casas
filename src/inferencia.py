from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np

from src.utils import validar_imagen, imagen_a_bytes

app = FastAPI(
    title="API YOLO - Detección de Casas",
    description="Detector de casas entrenado con YOLOv8",
    version="1.0.0"
)

# ==========================
# Cargar modelo entrenado
# ==========================
model = YOLO("models/house-yolo.pt")


@app.get("/")
def raiz():
    return {
        "proyecto": "Taller YOLO - Detección de Casas",
        "modelo": "YOLOv8 fine-tuned",
        "endpoint": "POST /detectar"
    }


@app.post("/detectar")
async def detectar_casas(archivo: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la imagen con bounding boxes dibujadas.
    """

    # 1️⃣ Validar imagen
    imagen = validar_imagen(archivo)

    # 2️⃣ Ejecutar inferencia
    results = model(imagen)

    # 3️⃣ Obtener imagen con bounding boxes
    imagen_resultado = results[0].plot()  # numpy array

    # 4️⃣ Convertir numpy → PIL
    imagen_pil = Image.fromarray(imagen_resultado)

    # 5️⃣ Convertir a bytes
    buffer = imagen_a_bytes(imagen_pil)

    return StreamingResponse(
        buffer,
        media_type="image/jpeg"
    )