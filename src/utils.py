from fastapi import UploadFile, HTTPException
from PIL import Image
import io

FORMATOS_PERMITIDOS = {"image/jpeg", "image/png", "image/webp"}


def validar_imagen(archivo: UploadFile) -> Image.Image:
    """
    Valida y abre una imagen subida.
    """
    if archivo.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no permitido. Usa: {FORMATOS_PERMITIDOS}"
        )

    contenido = archivo.file.read()

    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
        return imagen
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No se pudo procesar la imagen"
        )


def imagen_a_bytes(imagen: Image.Image, formato: str = "JPEG") -> io.BytesIO:
    """
    Convierte una imagen PIL a un buffer de bytes.
    """
    buffer = io.BytesIO()
    imagen.save(buffer, format=formato)
    buffer.seek(0)
    return buffer