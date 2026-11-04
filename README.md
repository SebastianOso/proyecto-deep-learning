# Reconocimiento de dibujos QuickDraw

CNN para diez categorías: gato, pez, bicicleta, avión, casa, árbol, reloj, paraguas, helado y estrella.

El paper del proyecto está disponible en [quickdraw.pdf](quickdraw.pdf), ubicado en la raíz del repositorio. Describe la preparación de datos, los modelos y sus resultados.

## Instalación

Crear y activar el entorno desde la raíz del repo, con Python 3.12.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS y Linux (bash o zsh):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias para entrenar:

```sh
python -m pip install -r requirements.txt
```

Para abrir el EDA, agregar:

```sh
python -m pip install -r requirements-eda.txt
```

En cada terminal nueva hay que volver a activar el entorno. Después, todos los comandos siguientes funcionan con `python`. Para salir: `deactivate`.

## 1. Preparar los datos

```sh
python etl.py
```

El flujo es: **cargar .npy → seleccionar la muestra → limpiar → dividir → guardar .npz**.

Para volver a generar las particiones o cambiar el tamaño:

```sh
python etl.py --preparar
python etl.py --imagenes-por-clase 5000 --semilla 42
```

## 2. Explorar

```sh
python -m notebook eda.ipynb
```

El notebook muestra dimensiones, conteos de limpieza, balance de clases, intensidades y ejemplos. Las estadísticas de píxeles y los ejemplos utilizan entrenamiento. No carga TensorFlow ni entrena.

## 3. Entrenar las cuatro variantes

```sh
python entrenar_todos.py
```

Cada carpeta contiene:

- `mejor.keras`: mejor checkpoint por pérdida de validación.
- `modelo.h5`: el mismo modelo para inferencia.
- `historial.csv` y `curvas_entrenamiento.png`: registro completo y curvas.
- `resultados.json`: parámetros del experimento y métricas de validación.

Se elige por **menor pérdida de validación**. El modelo ganador y su historial se copian a `modelos/final/`. Solo después de guardar la selección se evalúa al ganador en test. Allí quedan `seleccion.json`, `test.json` y `predicciones_test.npz`.

Para entrenar una variante por separado, sin evaluar test ni actualizar el final:

```sh
python experimentos/base.py
python experimentos/dropout.py
python experimentos/aumento.py
python experimentos/combinado.py
```

Prueba operativa de dos épocas y 100 imágenes por clase, con salidas separadas:

```sh
python entrenar_todos.py --prueba-corta
```

Estas salidas no representan los resultados del paper.

## 4. Clasificar un dibujo

```sh
python modelo_final.py --imagen dibujo.png
```

Se carga el ganador de `modelos/final/modelo.h5`. No se entrena ni se descarga el dataset.

## Demo Flask

```sh
python -m pip install -r requirements-web.txt
python app.py
```

Abrir [http://127.0.0.1:5000](http://127.0.0.1:5000). La página conserva su interfaz y carga el modelo actual de `modelos/final/modelo.h5` al iniciar. Reiniciar Flask después de reemplazar ese archivo. Para detenerlo: `Ctrl+C`.