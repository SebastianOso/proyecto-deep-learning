# Clasificación de dibujos QuickDraw

Cuatro experimentos (base, dropout, aumento y combinado) y un modelo final combinado para diez categorías de dibujos de 28 × 28 píxeles.

## Organización

- `etl.py`: descarga de fuentes, validación, limpieza, muestreo y particiones cacheadas.
- `eda.ipynb`: exploración básica de calidad, balance, intensidades y ejemplos de entrenamiento.
- `quickdraw_config.py`: constantes y utilidades compartidas, sin TensorFlow.
- `quickdraw_comun.py`: entrenamiento, checkpoints, evaluación, exportación e inferencia.
- `experimentos/` y `modelo_final.py`: arquitecturas y puntos de entrada.
- `app.py`, `templates/` y `static/`: demo web local con Flask para dibujar y predecir.

## Instalación

Con Python 3.12 instalado, crear y activar el entorno virtual desde la raíz del repo.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS y Linux (bash o zsh)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Dependencias (todos los sistemas)

Con el entorno activo, instalar las dependencias de entrenamiento:

```sh
python -m pip install -r requirements.txt
```

Para trabajar solo con ETL y EDA, instalar estas dependencias en lugar de las de entrenamiento. También se pueden agregar al entorno de entrenamiento para abrir el notebook:

```sh
python -m pip install -r requirements-eda.txt
```

Al abrir una terminal nueva, volver a activar el entorno:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- macOS y Linux (bash o zsh): `source .venv/bin/activate`

Los siguientes comandos funcionan en los tres sistemas con el entorno activo. Para salir del entorno, ejecutar `deactivate`.

## Preparar datos

```sh
python etl.py
python etl.py --imagenes-por-clase 5000 --semilla 42
python etl.py --prueba-corta
```

El valor predeterminado es 5000 imágenes por clase y semilla 42. La prueba corta usa 100 por clase y un caché separado. La primera ejecución descarga los archivos completos de las diez categorías incluso en prueba corta; requiere conexión y espacio local.

Se eliminan imágenes vacías, se deduplican las fuentes globalmente y se excluyen patrones con etiquetas contradictorias antes del muestreo. Se divide por clase en 70 % entrenamiento, 15 % validación y el remanente para test, con redondeo entero. Los datos uint8 y el manifiesto con hashes se guardan en `datos/quickdraw/`, excluido de Git. Las particiones existentes se reutilizan y las dañadas o incompatibles se rechazan.

## Explorar datos

Desde la raíz del repo:

```sh
python -m notebook eda.ipynb
```

Ejecutar todas las celdas en orden. La primera celda de código permite configurar tamaño de muestra, semilla y prueba corta. El notebook reutiliza el ETL y puede preparar los datos si faltan. Muestra resúmenes y gráficas dentro del notebook; no genera un reporte escrito. Los ejemplos y estadísticas de píxeles usan solo entrenamiento; validación y test se usan para conteos y comprobaciones de separación. No importa TensorFlow ni entrena modelos. Los notebooks de entrenamiento anteriores no forman parte de este repo.

## Entrenamiento e inferencia

```sh
python experimentos/base.py --help
python experimentos/dropout.py --modo entrenar
python experimentos/aumento.py --modo entrenar
python experimentos/combinado.py --modo entrenar
python modelo_final.py --modo entrenar
python modelo_final.py --modo reanudar
python modelo_final.py --modo demostrar --imagen dibujo.png
```

Entrenar prepara los datos automáticamente si faltan. Normalización y aumento permanecen dentro del modelo. El entrenamiento guarda checkpoints cada cinco épocas y evalúa validación y test al terminar. Si la carpeta de salida ya contiene archivos, el comando de entrenamiento se detiene; reanudar requiere un checkpoint compatible. Los modelos H5 existentes sin checkpoint pueden usarse para inferencia, pero no para reanudar.

## Demo web: dibujar y predecir

Con el entorno virtual activo, instalar las dependencias y arrancar desde la raíz del repo (Windows, macOS o Linux):

```sh
python -m pip install -r requirements-web.txt
python app.py
```

Abrir [http://127.0.0.1:5000](http://127.0.0.1:5000) en el navegador. Dibujar con el ratón, lápiz o dedo en el cuadro blanco y pulsar **Predecir**. La página muestra las tres categorías con mayor puntuación; **Limpiar** permite empezar otro dibujo.

Se carga una sola vez `modelos/final/modelo.h5`, con normalización dentro del modelo. La demo reutiliza el preprocesamiento de inferencia: inversión del fondo blanco, recorte, centrado y redimensionado a 28 × 28. No descarga el dataset, no entrena y no guarda los dibujos. Si el modelo falta o tiene dimensiones incompatibles, el servidor informa del error al iniciar.

El servidor se ejecuta solo en el equipo local, en el puerto 5000 y sin modo debug. Para detenerlo, pulsar `Ctrl+C` en la terminal.
