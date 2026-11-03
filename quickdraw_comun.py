"""Datos y entrenamiento local compartidos por los cinco modelos de QuickDraw.

Instalación (desde la raíz del repositorio):
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

Los imports numéricos son diferidos: importar este módulo o pedir --help no
descarga datos, no crea archivos y no exige cargar TensorFlow.
"""

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


REPO = Path(__file__).resolve().parent
CLASES_EN = ['cat', 'fish', 'bicycle', 'airplane', 'house', 'tree',
             'clock', 'umbrella', 'ice cream', 'star']
CLASES_ES = ['gato', 'pez', 'bicicleta', 'avión', 'casa', 'árbol',
             'reloj', 'paraguas', 'helado', 'estrella']
PREPROCESAMIENTO = dict(version=1, lado=28, contenido=24,
                       entrada='uint8_0_255', fondo_modelo='negro',
                       trazo_modelo='claro', centrado='bounding_box')
METRICAS = ('loss', 'val_loss', 'accuracy', 'val_accuracy')


def firma_json(valor):
    return hashlib.sha256(json.dumps(valor, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding='utf-8'))


def guardar_json(ruta, valor):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + '.tmp')
    temporal.write_text(json.dumps(valor, ensure_ascii=False, indent=2,
                                  allow_nan=False), encoding='utf-8')
    temporal.replace(ruta)


def sha_archivo(ruta):
    h = hashlib.sha256()
    with Path(ruta).open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def validar_npy(ruta):
    import numpy as np

    fuente = np.load(ruta, mmap_mode='r', allow_pickle=False)
    if fuente.ndim != 2 or fuente.shape[1] != 784 or fuente.dtype != np.uint8:
        raise ValueError(f'{ruta}: se esperaba uint8 con forma (N, 784).')
    return fuente


def descargar_fuentes(carpeta):
    """Una copia por categoría; solo publicar descargas completas y válidas."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    fuentes = []
    for nombre in CLASES_EN:
        ruta = carpeta / (nombre + '.npy')
        url = ('https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
               + quote(nombre, safe='') + '.npy')
        if not ruta.exists():
            temporal = ruta.with_suffix('.descarga.npy')
            print('Descargando:', nombre, flush=True)
            try:
                with urlopen(url, timeout=60) as respuesta, temporal.open('wb') as salida:
                    shutil.copyfileobj(respuesta, salida)
                fuente = validar_npy(temporal)
                del fuente
                temporal.replace(ruta)
            finally:
                temporal.unlink(missing_ok=True)
        else:
            fuente = validar_npy(ruta)
            del fuente
        fuentes.append(dict(clase=nombre, url=url, sha256=sha_archivo(ruta)))
    return fuentes


def preparar_particiones(rutas, por_clase=5000, semilla=42):
    """Deduplicar todos los originales antes de muestrear, como el notebook."""
    import numpy as np

    fuentes = [validar_npy(p) for p in rutas]
    vistos, contradictorios = {}, set()
    vacias = repetidas = 0
    for clase, fuente in enumerate(fuentes):
        for indice, fila in enumerate(fuente):
            if not np.any(fila):
                vacias += 1
                continue
            clave = hashlib.sha256(fila.tobytes()).digest()
            if clave in vistos:
                repetidas += 1
                if vistos[clave][0] != clase:
                    contradictorios.add(clave)
            else:
                vistos[clave] = (clase, indice)
        print('Revisada categoría', clase + 1, '/', len(fuentes), flush=True)
    candidatos = [[] for _ in fuentes]
    for clave, (clase, indice) in vistos.items():
        if clave not in contradictorios:
            candidatos[clase].append(indice)
    rng = np.random.default_rng(semilla)
    partes = {p: [] for p in ('train', 'val', 'test')}
    n_train, n_val = int(por_clase * .70), int(por_clase * .15)
    if min(n_train, n_val, por_clase - n_train - n_val) < 1:
        raise ValueError('Se necesitan al menos 7 imágenes por clase.')
    for clase, indices in enumerate(candidatos):
        if len(indices) < por_clase:
            raise ValueError(f'Clase {clase}: solo {len(indices)} imágenes válidas.')
        seleccion = rng.permutation(indices)[:por_clase]
        for nombre, filas in zip(partes, np.split(seleccion, [n_train, n_train + n_val])):
            partes[nombre].extend((clase, int(i)) for i in filas)
    resultado = {}
    for nombre, pares in partes.items():
        pares = np.asarray(pares, dtype=np.int64)
        pares = pares[rng.permutation(len(pares))]
        resultado['indices_' + nombre] = pares
        resultado['x_' + nombre] = np.stack(
            [fuentes[c][i].reshape(28, 28, 1) for c, i in pares])
        resultado['y_' + nombre] = pares[:, 0].astype(np.int32)
    comprobar_sin_fugas(resultado)
    informe = dict(vacias=vacias, repeticiones=repetidas,
                   patrones_con_etiquetas_contradictorias=len(contradictorios),
                   candidatos_por_clase=[len(c) for c in candidatos])
    return resultado, informe


def comprobar_sin_fugas(datos):
    conjuntos = []
    for parte in ('train', 'val', 'test'):
        imagenes = datos['x_' + parte]
        hashes = {hashlib.sha256(x.tobytes()).digest() for x in imagenes}
        if len(hashes) != len(imagenes):
            raise ValueError('Duplicados dentro de ' + parte)
        conjuntos.append(hashes)
    if any(conjuntos[i] & conjuntos[j] for i in range(3) for j in range(i)):
        raise ValueError('Fuga de imágenes idénticas entre particiones.')


def cargar_datos(por_clase, semilla, prueba_corta=False, exigir_existentes=False,
                 raiz=None):
    import numpy as np

    raiz = REPO if raiz is None else Path(raiz)
    carpeta = raiz / 'datos' / 'quickdraw'
    nombre = f'{"prueba_" if prueba_corta else ""}n{por_clase}_s{semilla}'
    particiones = carpeta / 'particiones' / nombre
    manifiesto_ruta = particiones / 'manifiesto.json'
    archivo_datos = particiones / 'particiones.npz'
    config = dict(version=1, clases=CLASES_EN, por_clase=por_clase,
                  semilla=semilla, fracciones=[.70, .15, .15],
                  deduplicacion='sha256_global', formato='numpy_bitmap')
    if manifiesto_ruta.exists():
        manifiesto = leer_json(manifiesto_ruta)
        if manifiesto['config'] != config:
            raise ValueError('Configuración de particiones incompatible.')
        if not archivo_datos.exists() or sha_archivo(archivo_datos) != manifiesto['sha256_npz']:
            raise ValueError('Particiones dañadas; no se regenerarán silenciosamente.')
    else:
        if exigir_existentes:
            raise FileNotFoundError('Faltan las particiones del entrenamiento a reanudar.')
        fuentes = descargar_fuentes(carpeta / 'originales')
        rutas = [carpeta / 'originales' / (c + '.npy') for c in CLASES_EN]
        arrays, limpieza = preparar_particiones(rutas, por_clase, semilla)
        particiones.mkdir(parents=True, exist_ok=True)
        temporal = particiones / 'particiones.tmp.npz'
        np.savez_compressed(temporal, **arrays)
        temporal.replace(archivo_datos)
        manifiesto = dict(config=config, fuentes=fuentes, limpieza=limpieza,
                          sha256_npz=sha_archivo(archivo_datos))
        guardar_json(manifiesto_ruta, manifiesto)
    with np.load(archivo_datos, allow_pickle=False) as archivo:
        datos = {k: archivo[k] for k in archivo.files}
    for parte in ('train', 'val', 'test'):
        x, y = datos['x_' + parte], datos['y_' + parte]
        if (x.dtype != np.uint8 or x.shape != (len(y), 28, 28, 1)
                or y.ndim != 1 or not np.issubdtype(y.dtype, np.integer)
                or np.any((y < 0) | (y >= len(CLASES_EN)))):
            raise ValueError('Formato de partición incompatible: ' + parte)
        print(parte, x.shape, flush=True)
    comprobar_sin_fugas(datos)
    return datos, firma_json(manifiesto)


def compilar_modelo(modelo):
    import tensorflow as tf

    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo


def configurar_tensorflow(semilla):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(semilla)
    tf.keras.mixed_precision.set_global_policy('float32')
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Dispositivo:', 'GPU' if gpus else 'CPU', '| TensorFlow:', tf.__version__)
    return tf


def comprobar_dependencias():
    """Fallar antes de descargar o entrenar si falta una dependencia."""
    faltantes = []
    for modulo, paquete in [('tensorflow', 'tensorflow'), ('numpy', 'numpy'),
                            ('matplotlib', 'matplotlib'), ('PIL', 'Pillow'),
                            ('sklearn', 'scikit-learn'), ('h5py', 'h5py')]:
        try:
            importlib.import_module(modulo)
        except ImportError:
            faltantes.append(paquete)
    if faltantes:
        raise ImportError('Faltan dependencias: ' + ', '.join(faltantes))


def estado_historial(historial):
    """Estado de EarlyStopping: menor val_loss, min_delta=0, paciencia=5."""
    mejor, espera, epoca_mejor = math.inf, 0, 0
    for epoca, registro in enumerate(historial, 1):
        if set(registro) != set(METRICAS) or not all(
                math.isfinite(float(v)) for v in registro.values()):
            raise ValueError('Historial incompleto o con métricas no finitas.')
        valor = registro['val_loss']
        if valor < mejor:
            mejor, espera, epoca_mejor = valor, 0, epoca
        else:
            espera += 1
    return dict(best=mejor, wait=espera, best_epoch=epoca_mejor)


def guardar_checkpoint(modelo, carpeta, config, historial, mejores_pesos):
    """Publicar un solo conjunto coherente; la copia anterior es transitoria."""
    carpeta = Path(carpeta)
    destino = carpeta / 'checkpoints'
    anterior = carpeta / '.checkpoints_anterior'
    with tempfile.TemporaryDirectory(prefix='.checkpoint_', dir=carpeta) as temporal:
        temporal = Path(temporal)
        modelo.save(temporal / 'checkpoint.keras')
        actuales = modelo.get_weights()
        try:
            modelo.set_weights(mejores_pesos)
            modelo.save_weights(temporal / 'mejor.weights.h5')
        finally:
            modelo.set_weights(actuales)
        hashes = {n: sha_archivo(temporal / n)
                  for n in ('checkpoint.keras', 'mejor.weights.h5')}
        guardar_json(temporal / 'estado.json', dict(config=config,
            epoca=len(historial), historial=historial, sha256=hashes))
        if anterior.exists():
            shutil.rmtree(anterior)
        if destino.exists():
            destino.replace(anterior)
        temporal.replace(destino)
    if anterior.exists():
        shutil.rmtree(anterior)
    print(f'Checkpoint actualizado: época {len(historial)}', flush=True)


def recuperar_checkpoint(carpeta, config):
    import tensorflow as tf

    carpeta = Path(carpeta)
    candidatos = [carpeta / 'checkpoints', carpeta / '.checkpoints_anterior']
    errores = []
    for ruta in candidatos:
        if not ruta.exists():
            continue
        try:
            estado = leer_json(ruta / 'estado.json')
            if estado['config'] != config:
                raise ValueError('La configuración difiere de la del checkpoint.')
            if estado['epoca'] != len(estado['historial']) or estado['epoca'] % 5:
                raise ValueError('Época de checkpoint incompatible.')
            estado_historial(estado['historial'])
            for nombre in ('checkpoint.keras', 'mejor.weights.h5'):
                if sha_archivo(ruta / nombre) != estado['sha256'][nombre]:
                    raise ValueError('Checkpoint incompleto o dañado: ' + nombre)
            modelo = tf.keras.models.load_model(ruta / 'checkpoint.keras')
            actuales = modelo.get_weights()
            modelo.load_weights(ruta / 'mejor.weights.h5')
            mejores = modelo.get_weights()
            modelo.set_weights(actuales)
            if modelo.optimizer is None:
                raise ValueError('Falta el optimizador en el checkpoint.')
            return modelo, estado['historial'], mejores
        except (ValueError, OSError, KeyError, TypeError) as error:
            errores.append(str(error))
    detalle = '; '.join(errores) or 'Todavía no se completaron 5 épocas.'
    raise RuntimeError('No hay checkpoint recuperable. ' + detalle)


def crear_control_entrenamiento(carpeta, config, historial=None, mejores_pesos=None):
    import tensorflow as tf

    class ControlEntrenamiento(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.historial = list(historial or [])
            self.mejores_pesos = mejores_pesos
            self.estado = estado_historial(self.historial)

        def on_epoch_end(self, epoch, logs=None):
            registro = {k: float((logs or {})[k]) for k in METRICAS}
            if not all(math.isfinite(v) for v in registro.values()):
                raise ValueError('Métricas no finitas; se conserva el checkpoint anterior.')
            if epoch != len(self.historial):
                raise ValueError('Época e historial no coinciden.')
            if registro['val_loss'] < self.estado['best']:
                self.mejores_pesos = self.model.get_weights()
            self.historial.append(registro)
            self.estado = estado_historial(self.historial)
            if len(self.historial) % 5 == 0:
                guardar_checkpoint(self.model, carpeta, config,
                                   self.historial, self.mejores_pesos)
            if self.estado['wait'] >= 5:
                print('Parada temprana; mejor época:', self.estado['best_epoch'])
                self.model.stop_training = True

    return ControlEntrenamiento()


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


def exportar_modelo(modelo, carpeta, ejemplos):
    import numpy as np
    import tensorflow as tf

    carpeta = Path(carpeta)
    # Validar ambos formatos antes de publicarlos con los nombres definitivos.
    with tempfile.TemporaryDirectory(prefix='.exportacion_', dir=carpeta) as temporal:
        temporal = Path(temporal)
        ruta = temporal / 'modelo.h5'
        modelo.save(ruta)
        modelo.save_weights(temporal / 'modelo.weights.h5')
        esperado = modelo(ejemplos, training=False).numpy()
        recargado = tf.keras.models.load_model(ruta, compile=False)
        np.testing.assert_allclose(esperado, recargado(ejemplos, training=False).numpy(),
                                   rtol=1e-5, atol=1e-6)
        recargado.load_weights(temporal / 'modelo.weights.h5')
        np.testing.assert_allclose(esperado, recargado(ejemplos, training=False).numpy(),
                                   rtol=1e-5, atol=1e-6)
        ruta.replace(carpeta / 'modelo.h5')
        (temporal / 'modelo.weights.h5').replace(carpeta / 'modelo.weights.h5')


def evaluar(modelo, x, y, batch, nombre):
    import numpy as np
    from sklearn.metrics import classification_report, f1_score

    resultados = modelo.evaluate(x, y, batch_size=batch, verbose=0, return_dict=True)
    pred = modelo.predict(x, batch_size=batch, verbose=0).argmax(axis=1)
    f1 = f1_score(y, pred, labels=np.arange(10), average='macro', zero_division=0)
    print(f'\n{nombre}: loss={resultados["loss"]:.6f} '
          f'accuracy={resultados["accuracy"]:.4%} F1 macro={f1:.6f}')
    print(classification_report(y, pred, labels=np.arange(10),
                                target_names=CLASES_ES, zero_division=0, digits=4))
    return dict(resultados, f1_macro=float(f1))


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


def entero_positivo(valor):
    numero = int(valor)
    if numero < 1:
        raise argparse.ArgumentTypeError('Debe ser un entero positivo.')
    return numero


def argumentos(nombre, argv=None):
    parser = argparse.ArgumentParser(description=f'QuickDraw local: {nombre}. '
        'Entrenar y reanudar evalúan test automáticamente al terminar.')
    parser.add_argument('--modo', choices=('entrenar', 'reanudar', 'demostrar'),
                        default='entrenar')
    parser.add_argument('--prueba-corta', action='store_true',
                        help='100 imágenes/clase y 2 épocas; salidas separadas.')
    parser.add_argument('--epocas', type=entero_positivo, default=30)
    parser.add_argument('--batch', type=entero_positivo, default=128)
    parser.add_argument('--imagenes-por-clase', type=entero_positivo, default=5000)
    parser.add_argument('--semilla', type=int, default=42)
    parser.add_argument('--imagen', type=Path, help='PNG/JPG para modo demostrar.')
    parser.add_argument('--fondo', choices=('auto', 'blanco', 'negro'), default='auto')
    args = parser.parse_args(argv)
    if args.semilla < 0 or args.semilla > 2**32 - 2:
        parser.error('--semilla debe estar entre 0 y 4294967294.')
    if args.prueba_corta:
        args.imagenes_por_clase, args.epocas = 100, 2
    if args.imagenes_por_clase < 7:
        parser.error('--imagenes-por-clase debe ser al menos 7.')
    if args.modo == 'demostrar' and args.imagen is None:
        parser.error('--modo demostrar requiere --imagen.')
    if args.modo != 'demostrar' and args.imagen is not None:
        parser.error('--imagen solo se usa con --modo demostrar.')
    return args


def ejecutar(nombre, crear_modelo, argv=None):
    args = argumentos(nombre, argv)
    carpeta = REPO / 'modelos'
    if args.prueba_corta:
        carpeta /= 'prueba_corta'
        print('PRUEBA OPERATIVA: estas métricas no son resultados finales.')
    carpeta /= nombre
    if args.modo == 'entrenar' and carpeta.exists() and any(carpeta.iterdir()):
        raise RuntimeError(f'Ya existe {carpeta}. Usa --modo reanudar. '
                           'Para empezar de cero, mueve la carpeta existente.')
    if args.modo == 'reanudar' and not any(
            (carpeta / n).exists() for n in ('checkpoints', '.checkpoints_anterior')):
        raise FileNotFoundError('No hay checkpoint; se guarda uno cada 5 épocas.')
    if args.modo != 'demostrar':
        comprobar_dependencias()
    tf = configurar_tensorflow(args.semilla)
    if args.modo == 'demostrar':
        demostrar(carpeta, args.imagen, args.fondo)
        return
    datos, firma = cargar_datos(args.imagenes_por_clase, args.semilla,
        args.prueba_corta, exigir_existentes=args.modo == 'reanudar')
    config = dict(version=1, modelo=nombre, firma_datos=firma,
        clases=CLASES_EN, preprocesamiento=PREPROCESAMIENTO, semilla=args.semilla,
        batch=args.batch, epocas=args.epocas, paciencia=5, learning_rate=.001,
        prueba_corta=args.prueba_corta, tensorflow=tf.__version__,
        keras=tf.keras.version() if hasattr(tf.keras, 'version') else 'integrado')
    carpeta.mkdir(parents=True, exist_ok=True)
    if args.modo == 'reanudar':
        modelo, historial, mejores = recuperar_checkpoint(carpeta, config)
        print('Reanudando desde época', len(historial), '| pasos de Adam:',
              int(modelo.optimizer.iterations.numpy()))
    else:
        modelo, historial, mejores = crear_modelo(args.semilla), [], None
    modelo.summary()
    control = crear_control_entrenamiento(carpeta, config, historial, mejores)
    if len(historial) < args.epocas and control.estado['wait'] < 5:
        modelo.fit(datos['x_train'], datos['y_train'],
            validation_data=(datos['x_val'], datos['y_val']), batch_size=args.batch,
            epochs=args.epocas, initial_epoch=len(historial), shuffle=True,
            callbacks=[control], verbose=2)
    modelo.set_weights(control.mejores_pesos)
    print('Mejor época de validación:', control.estado['best_epoch'])
    exportar_modelo(modelo, carpeta, datos['x_val'][:10])
    guardar_curvas(control.historial, carpeta / 'curvas_entrenamiento.png', nombre)
    evaluar(modelo, datos['x_val'], datos['y_val'], args.batch, 'VALIDACIÓN')
    evaluar(modelo, datos['x_test'], datos['y_test'], args.batch, 'TEST')
    print('Modelo, pesos y gráfica guardados en:', carpeta)


def main(nombre, crear_modelo, argv=None):
    try:
        ejecutar(nombre, crear_modelo, argv)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'Error: {error}', file=sys.stderr)
        if isinstance(error, ImportError):
            print('Instala las dependencias con: '
                  '.venv/bin/python -m pip install -r requirements.txt', file=sys.stderr)
        raise SystemExit(1) from error
