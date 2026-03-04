# Detector de Casas con YOLO

Este repositorio contiene un flujo mínimo para entrenar e inferir un modelo YOLO que detecta casas en imágenes. Está orientado al taller de detección de casas.

## Estructura del proyecto

```
taller-yolo-casas/
├── inferencia.py       # API con FastAPI para realizar inferencias
├── utils.py            # utilidades (guardar subidas, extraer detecciones)
├── models/             # pesos guardados (.pt)
│   └── house-yolo.pt
├── outputs/            # archivos temporales generados por la API
├── requirements.txt    # dependencias Python
└── README.md
```

## Instalación

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix
source venv/bin/activate
pip install -r requirements.txt
```

## Levantar la API

```bash
uvicorn inferencia:app --reload
```

La API correrá en `http://127.0.0.1:8000`.

### Endpoints

- `GET /` : formulario HTML para subir imagen y ver resultado.
- `POST /predict` : sube una imagen y devuelve una página con la imagen anotada.
- `POST /predict_json` : recibe imagen y devuelve JSON con las detecciones.
- `POST /predict_url` : recibe `url` y devuelve JSON con las detecciones.
- `GET /outputs/{file_name}` : sirve archivos anotados generados.
- `GET /health` : healthcheck y estado de carga del modelo.

## Notas

- Guarda tus pesos entrenados en `models/house-yolo.pt`.
- Para entrenar, crea `train_yolo.py` o usa la API de Ultralytics: `model.train(data='data.yaml', epochs=50)`.

