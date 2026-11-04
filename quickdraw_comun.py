"""Funciones sencillas compartidas por los experimentos y la inferencia."""
import argparse
import csv
from pathlib import Path
import shutil
import sys

import numpy as np
from quickdraw_config import REPO, CLASES_ES, PREPROCESAMIENTO, guardar_json
from etl import cargar_datos


def configurar_tensorflow(semilla=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(semilla)
    tf.keras.mixed_precision.set_global_policy('float32')
    # Limitar los hilos evita sobrecargar el equipo durante la comparación en CPU.
    if not getattr(configurar_tensorflow, 'listo', False):
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        configurar_tensorflow.listo = True
    return tf


def compilar_modelo(modelo):
    import tensorflow as tf
    modelo.compile(optimizer=tf.keras.optimizers.Adam(.001),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo


def limpiar_salida(carpeta):
    """Reemplazar solo una carpeta de resultados conocida dentro del repo."""
    carpeta = Path(carpeta)
    raiz = (REPO / 'modelos').resolve()
    if carpeta.is_symlink() or carpeta.is_junction():
        raise ValueError('La salida no puede ser un enlace.')
    destino = carpeta.resolve()
    if not destino.is_relative_to(raiz) or destino == raiz:
        raise ValueError('La salida debe estar dentro de modelos/.')
    if carpeta.exists(): shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)


def medir(modelo, x, y, batch=128):
    import tensorflow as tf
    from sklearn.metrics import precision_recall_fscore_support
    salidas, perdidas = [], []
    for i in range(0, len(x), batch):
        salida = modelo(x[i:i+batch], training=False)
        # Calcular antes de numpy conserva los logits y la estabilidad de Keras.
        perdidas.append(tf.keras.losses.sparse_categorical_crossentropy(
            y[i:i+batch], salida).numpy())
        salidas.append(salida.numpy())
    probs = np.concatenate(salidas)
    if not np.isfinite(probs).all(): raise ValueError('Predicciones no finitas.')
    pred = probs.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=np.arange(10), average='macro', zero_division=0)
    return dict(n=len(y), loss=float(np.concatenate(perdidas).mean()),
                accuracy=float(np.mean(pred == y)), precision_macro=float(precision),
                recall_macro=float(recall), f1_macro=float(f1), aciertos=int(np.sum(pred == y))), probs


def leer_historial(ruta):
    with Path(ruta).open(encoding='utf-8', newline='') as archivo:
        return [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(archivo)]


def entrenar(nombre, crear_modelo, datos, epocas=30, batch=128, semilla=42,
             prueba_corta=False):
    tf = configurar_tensorflow(semilla)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(semilla)
    carpeta = REPO / 'modelos'
    if prueba_corta: carpeta /= 'prueba_corta'
    carpeta /= nombre
    modelo = crear_modelo(semilla)
    limpiar_salida(carpeta)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(carpeta / 'mejor.keras', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.CSVLogger(carpeta / 'historial.csv'),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    modelo.fit(datos['x_train'], datos['y_train'],
               validation_data=(datos['x_val'], datos['y_val']),
               epochs=epocas, batch_size=batch, shuffle=True, callbacks=callbacks, verbose=2)
    modelo = tf.keras.models.load_model(carpeta / 'mejor.keras', compile=False)
    modelo.save(carpeta / 'modelo.h5', include_optimizer=False)
    historial = leer_historial(carpeta / 'historial.csv')
    guardar_curvas(historial, carpeta / 'curvas_entrenamiento.png', nombre)
    metricas, _ = medir(modelo, datos['x_val'], datos['y_val'], batch)
    registro = dict(modelo=nombre, semilla=semilla, batch=batch, max_epocas=epocas,
                    paciencia=5, learning_rate=.001, epocas_realizadas=len(historial),
                    mejor_epoca=int(np.argmin([e['val_loss'] for e in historial]))+1,
                    val=metricas, parametros=int(modelo.count_params()),
                    tensorflow=tf.__version__, keras=tf.keras.__version__)
    guardar_json(carpeta / 'resultados.json', registro)
    print('VALIDACIÓN', nombre, metricas, flush=True)
    return registro


def guardar_curvas(historial, ruta, titulo):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epocas = range(1, len(historial) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for eje, claves in zip(ejes, [('loss', 'val_loss'), ('accuracy', 'val_accuracy')]):
        for clave in claves:
            eje.plot(epocas, [r[clave] for r in historial], label=clave)
        eje.set_xlabel('Época')
        eje.set_ylabel('Pérdida' if claves[0] == 'loss' else 'Accuracy')
        eje.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def preparar_dibujo(origen, fondo='auto'):
    """PNG/JPG -> uint8 (28,28,1), manteniendo el preprocesamiento original."""
    import numpy as np
    from PIL import Image

    if fondo not in ('auto', 'blanco', 'negro'):
        raise ValueError('Fondo no válido.')
    with Image.open(origen) as imagen:
        rgba = imagen.convert('RGBA')
    blanco = Image.new('RGBA', rgba.size, 'white')
    gris = np.asarray(Image.alpha_composite(blanco, rgba).convert('L'))
    if gris.size == 0 or min(gris.shape) < 2:
        raise ValueError('Imagen demasiado pequeña.')
    bordes = np.concatenate([gris[0], gris[-1], gris[:, 0], gris[:, -1]])
    claro = np.median(bordes) >= 128 if fondo == 'auto' else fondo == 'blanco'
    tinta = 255 - gris if claro else gris.copy()
    mascara = tinta > 35
    if int(mascara.sum()) < 3 or float(np.ptp(gris)) < 15:
        raise ValueError('El dibujo está vacío o no tiene contraste suficiente.')
    ys, xs = np.where(mascara)
    recorte = Image.fromarray(tinta).crop((int(xs.min()), int(ys.min()),
                                         int(xs.max()) + 1, int(ys.max()) + 1))
    ancho, alto = recorte.size
    escala = PREPROCESAMIENTO['contenido'] / max(ancho, alto)
    nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
    recorte = recorte.resize(nuevo, Image.Resampling.LANCZOS)
    salida = Image.new('L', (28, 28), 0)
    salida.paste(recorte, ((28 - nuevo[0]) // 2, (28 - nuevo[1]) // 2))
    return np.asarray(salida, dtype=np.uint8)[..., None]


def demostrar(carpeta, imagen, fondo):
    import tensorflow as tf

    ruta = Path(carpeta) / 'modelo.h5'
    if not ruta.exists():
        raise FileNotFoundError(f'Falta {ruta}. Primero entrena este modelo.')
    modelo = tf.keras.models.load_model(ruta, compile=False)
    if tuple(modelo.input_shape[1:]) != (28, 28, 1) or modelo.output_shape[-1] != 10:
        raise ValueError('Dimensiones del modelo incompatibles.')
    entrada = preparar_dibujo(imagen, fondo)[None, ...]
    probs = modelo(entrada, training=False).numpy()[0]
    for i in probs.argsort()[::-1][:3]:
        print(f'{CLASES_ES[i]}: {float(probs[i]):.2%}')


def argumentos(nombre, argv=None):
    parser = argparse.ArgumentParser(description=f'QuickDraw: {nombre}. Entrenar reemplaza los resultados de esa ejecución.')
    parser.add_argument('--modo', choices=('entrenar', 'demostrar'), default='entrenar')
    parser.add_argument('--epocas', type=int, default=30)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--imagenes-por-clase', type=int, default=5000)
    parser.add_argument('--semilla', type=int, default=42)
    parser.add_argument('--prueba-corta', action='store_true')
    parser.add_argument('--imagen', type=Path)
    parser.add_argument('--fondo', choices=('auto', 'blanco', 'negro'), default='auto')
    args = parser.parse_args(argv)
    if args.prueba_corta: args.epocas, args.imagenes_por_clase = 2, 100
    if args.epocas < 1 or args.batch < 1 or args.imagenes_por_clase < 7: parser.error('Tamaños no válidos.')
    if not 0 <= args.semilla <= 2**32 - 2: parser.error('Semilla fuera de rango.')
    if args.modo == 'demostrar' and not args.imagen: parser.error('Falta --imagen.')
    return args


def main(nombre, crear_modelo, argv=None):
    args = argumentos(nombre, argv)
    if args.modo == 'demostrar':
        carpeta = REPO / 'modelos'
        if args.prueba_corta: carpeta /= 'prueba_corta'
        demostrar(carpeta / nombre, args.imagen, args.fondo)
        return
    datos = cargar_datos(args.imagenes_por_clase, args.semilla, args.prueba_corta)
    entrenar(nombre, crear_modelo, datos, args.epocas, args.batch, args.semilla, args.prueba_corta)
