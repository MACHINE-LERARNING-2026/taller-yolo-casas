--------------------------------Detector de Casas (YOLO)------------------------------

Contenido:

- Descripción
- Estructura del Proyecto
- Requisitos
	- 3.1 Instalación de PyTorch / CUDA
	- 3.2 Instalación de Librerías
- Descarga del repositorio
	- 4.1 Clonar/Descargar el repositorio
	- 4.2 Descargar como .zip
- Ejecución
- Uso del Sistema
- Preprocesamiento Implementado
- Consideraciones
- Descripción del dataset y origen de imágenes
- Instrucciones para reproducir el entrenamiento y la inferencia
- Resultados (métricas) y ejemplos — (plantilla para completar)
- Limitaciones y pasos futuros recomendados

Descripción
Este proyecto implementa un pipeline para detección de casas usando YOLO (Ultralytics YOLOv8) con arquitectura modular en Python.

El sistema permite:
- Entrenar un detector sobre un dataset personalizado (formato YOLO). 
- Realizar inferencia local mediante un servicio HTTP (FastAPI) o por línea de comandos.
- Guardar pesos finales en `models/house-yolo.pt`.

Formatos de imagen soportados:
- .png, .jpg, .jpeg, .tiff, .bmp

Los comandos en este README se muestran con `python` / `pip`. Si en tu entorno el intérprete es `python3` / `pip3`, sustitúyelos según corresponda.

Estructura del Proyecto
```
taller-yolo-casas/
├── data.yaml               # Descriptor del dataset (rutas train/val/test, nc, names)
├── src/
│   ├── train_yolo.py       # Script de entrenamiento
│   ├── inferencia.py       # Servicio FastAPI para inferencia
│   └── utils.py            # Funciones utilitarias (IO, dibujo, parseo de resultados)
├── models/
│   └── detect/             # Carpeta que genera Ultralytics durante el training
│       └── weights/
│           └── best.pt
├── train/                  # Imágenes y etiquetas de entrenamiento (YOLO txt)
├── valid/                  # Validación
├── requirements.txt
└── README.md
```

Requisitos
Para el correcto funcionamiento se recomienda:
- Python 3.9 o superior
- `pip` actualizado
- GPU opcional (para entrenar más rápido): CUDA y versión de `torch` compatible

Verifica la versión de Python:

```bash
python --version
```

3.1. Instalación de PyTorch / CUDA
Visita https://pytorch.org/ y elige la instalación adecuada según tu sistema operativo y versión de CUDA. Ejemplo (Windows/Linux, CPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Para GPU, sigue las instrucciones de la web oficial para instalar la build de `torch` compatible con tu versión de CUDA.

3.2. Instalación de Librerías
Instala las dependencias listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

3.3. Notas sobre certificados (macOS)
Si usas macOS y encuentras problemas SSL con algunas librerías, instala los certificados del sistema Python si aplica:

```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

Descarga del repositorio
4.1 Clonar/Descargar el repositorio

```bash
git clone <repo_url>
cd taller-yolo-casas
```

4.2 Descargar como .zip

Desde la interfaz web del repositorio descarga el ZIP, descomprímelo y entra en la carpeta:

```bash
cd taller-yolo-casas
```

Ejecución

1) Crear y activar un entorno virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

2) Entrenamiento (modo sencillo)

```bash
# Ejecuta el script de entrenamiento
python src/train_yolo.py
```

El script guarda los resultados en `models/detect/weights/best.pt` y copia `best.pt` a `models/house-yolo.pt`.

3) Inferencia con FastAPI

```bash
uvicorn src.inferencia:app --reload
```

Abre `http://127.0.0.1:8000/docs` para la documentación interactiva si el servidor está configurado.

Uso del Sistema
Al ejecutar la API o la inferencia CLI puedes:
- Enviar una imagen al endpoint `POST /detectar_casas` (campo `archivo`) y descargar la imagen anotada.
- Ejecutar inferencia directa con Ultralytics:

```bash
python -c "from ultralytics import YOLO; YOLO('models/house-yolo.pt').predict(source='ruta/imagen.jpg', save=True, imgsz=640)"
```

Preprocesamiento Implementado
Dependiendo de la implementación en `src/utils.py`, el pipeline puede aplicar:
- Redimensionamiento / normalización
- Data augmentation (durante training): flips, rotaciones, cambios de brillo/contraste
- Filtrado / limpieza de anotaciones

Consideraciones
- En la primera ejecución, Ultralytics puede descargar pesos preentrenados (p. ej. `yolov8n.pt`) en tu directorio de trabajo. Para evitar tener ese archivo en la raíz, coloca tus pesos base en `models/weights/` y ajusta `MODEL_BASE` en `src/train_yolo.py`.
- Si `uvicorn` o instalación fallan revisa que el `venv` esté activado.

Descripción del dataset y origen de imágenes
- Archivo descriptor: `data.yaml` (contiene rutas a `train`, `val`, `test` y el número de clases `nc`).
- Clases: `['Casa']` (1 clase) — confirma en `data.yaml`.
- Origen: dataset exportado desde Roboflow (ver `data.yaml` para detalles del workspace/proyecto/versión).
- Estructura esperada de carpetas (relativa a la raíz del repo):
	- `train/images`, `train/labels`
	- `valid/images`, `valid/labels`
	- `test/images`, `test/labels`

Instrucciones para reproducir el entrenamiento y la inferencia
1) Asegura que `data.yaml` apunte a las rutas correctas.
2) Ajusta parámetros en `src/train_yolo.py` o expón argumentos CLI (opcional): `MODEL_BASE`, `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`.
3) Ejecuta:

```bash
python src/train_yolo.py
```

4) Para validar y obtener métricas con Ultralytics:

```python
from ultralytics import YOLO
model = YOLO('models/house-yolo.pt')
metrics = model.val(data='data.yaml')
print(metrics)
```

Resultados (métricas) y ejemplos de detección
-- Esta sección la puedes completar con tus métricas y ejemplos.

- Dataset: número total de imágenes, splits (train/val/test), resolución media — __
- mAP@0.5: __
- Precision: __
- Recall: __

Ejemplos (añade rutas o enlaces a imágenes anotadas):
- True positives: `examples/correct/` — __
- Falsos positivos: `examples/errors/false_positives/` — __
- Falsos negativos: `examples/errors/false_negatives/` — __

Instrucciones rápidas para generar imágenes anotadas:

```python
from ultralytics import YOLO
model = YOLO('models/house-yolo.pt')
model.predict(source='examples/raw/imagen.jpg', imgsz=640, conf=0.25, save=True)
# Resultado en runs/predict/<timestamp>/imagen.jpg
```

Limitaciones y pasos futuros recomendados
- Si el dataset es pequeño existe riesgo de sobreajuste. Recomendaciones:
	- Aumentar datos (augmentations): rotaciones, flips, variaciones de brillo/contraste.
	- Recolectar más imágenes en distintas condiciones (iluminación, ángulos, entornos).
	- Realizar validación cruzada o usar técnicas de regularización.
	- Evaluar mAP en múltiples umbrales (mAP@[.5:.95]).

- Despliegue:
	- En producción, ejecutar Uvicorn sin `--reload` y orquestar con systemd/Docker/Gunicorn.
	- Añadir tests automáticos y CI que verifique endpoints y una inferencia mínima.

Cómo contribuir
- Si quieres, puedo añadir:
	- Scripts para generar reportes de métricas automáticamente.
	- Notebooks con visualización de curvas de entrenamiento.
	- Ejemplos curl/PowerShell más detallados.

---
Fecha de actualización: marzo 2026


