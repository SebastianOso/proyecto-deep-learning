"""QuickDraw: cargar .npy, limpiar la muestra y guardar particiones en un .npz."""
import argparse
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

import numpy as np
from quickdraw_config import REPO, CLASES_EN


def ruta_datos(por_clase=5000, semilla=42, prueba_corta=False, raiz=None):
    raiz = REPO if raiz is None else Path(raiz)
    prefijo = 'prueba_' if prueba_corta else ''
    return raiz / 'datos' / 'quickdraw' / f'{prefijo}n{por_clase}_s{semilla}.npz'


def limpiar_muestra(x, y, indices):
    """Quitar vacíos, copias exactas y patrones presentes en distintas clases."""
    no_vacias = np.any(x != 0, axis=(1, 2, 3))
    vacias = int((~no_vacias).sum())
    x, y, indices = x[no_vacias], y[no_vacias], indices[no_vacias]
    _, primeras, grupos, cantidades = np.unique(
        x.reshape(len(x), 784), axis=0, return_index=True,
        return_inverse=True, return_counts=True)
    minima = np.full(len(primeras), 10)
    maxima = np.full(len(primeras), -1)
    np.minimum.at(minima, grupos, y)
    np.maximum.at(maxima, grupos, y)
    contradictorios = minima != maxima
    conservar = np.sort(primeras[~contradictorios])
    limpieza = np.array([vacias, int((cantidades - 1).sum()),
                         int(contradictorios.sum())], dtype=np.int64)
    return x[conservar], y[conservar], indices[conservar], limpieza


def preparar_datos(por_clase=5000, semilla=42, prueba_corta=False, raiz=None):
    raiz = REPO if raiz is None else Path(raiz)
    originales = raiz / 'datos' / 'quickdraw' / 'originales'
    originales.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    imagenes, etiquetas, indices = [], [], []
    for clase, nombre in enumerate(CLASES_EN):
        ruta = originales / (nombre + '.npy')
        if not ruta.exists():
            print('Descargando:', nombre, flush=True)
            url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
            urlretrieve(url + quote(nombre, safe='') + '.npy', ruta)
        fuente = np.load(ruta, mmap_mode='r', allow_pickle=False)
        if fuente.ndim != 2 or fuente.shape[1] != 784 or fuente.dtype != np.uint8:
            raise ValueError(f'{ruta}: se esperaba uint8 con forma (N, 784).')
        filas = rng.permutation(len(fuente))[:por_clase]
        imagenes.append(fuente[filas].reshape(-1, 28, 28, 1))
        etiquetas.append(np.full(len(filas), clase, dtype=np.int32))
        indices.append(np.column_stack((np.full(len(filas), clase), filas)))
    x, y, indices, limpieza = limpiar_muestra(
        np.concatenate(imagenes), np.concatenate(etiquetas), np.concatenate(indices))
    n = int(np.bincount(y, minlength=10).min())
    if n < 7:
        raise ValueError('Se necesitan al menos siete dibujos válidos por clase.')
    partes = {p: [] for p in ('train', 'val', 'test')}
    for clase in range(10):
        seleccion = rng.permutation(np.flatnonzero(y == clase))[:n]
        a, b = int(n * .70), int(n * .15)
        for parte, filas in zip(partes, np.split(seleccion, [a, a + b])):
            partes[parte].extend(filas)
    datos = dict(limpieza=limpieza, por_clase=np.array(n), semilla=np.array(semilla),
                 seleccionadas=np.array(sum(len(a) for a in imagenes)))
    for parte, filas in partes.items():
        filas = rng.permutation(filas)
        datos['x_' + parte] = x[filas]
        datos['y_' + parte] = y[filas]
        datos['indices_' + parte] = indices[filas]
    salida = ruta_datos(por_clase, semilla, prueba_corta, raiz)
    np.savez_compressed(salida, **datos)
    print('Guardado:', salida, '| por clase:', n, '| limpieza:', limpieza.tolist(), flush=True)
    return datos


def cargar_datos(por_clase=5000, semilla=42, prueba_corta=False, raiz=None):
    ruta = ruta_datos(por_clase, semilla, prueba_corta, raiz)
    if not ruta.exists():
        return preparar_datos(por_clase, semilla, prueba_corta, raiz)
    with np.load(ruta, allow_pickle=False) as archivo:
        datos = {nombre: archivo[nombre] for nombre in archivo.files}
    for parte in ('train', 'val', 'test'):
        x, y = datos['x_' + parte], datos['y_' + parte]
        if (x.dtype != np.uint8 or x.shape != (len(y), 28, 28, 1)
                or y.ndim != 1 or not np.issubdtype(y.dtype, np.integer)
                or np.any((y < 0) | (y >= 10))):
            raise ValueError('Formato de partición incorrecto: ' + parte)
    return datos


def argumentos():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagenes-por-clase', type=int, default=5000)
    parser.add_argument('--semilla', type=int, default=42)
    parser.add_argument('--prueba-corta', action='store_true')
    parser.add_argument('--preparar', action='store_true', help='Volver a preparar las particiones.')
    args = parser.parse_args()
    if args.prueba_corta: args.imagenes_por_clase = 100
    if args.imagenes_por_clase < 7: parser.error('Se necesitan al menos 7 imágenes por clase.')
    if not 0 <= args.semilla <= 2**32 - 2: parser.error('Semilla fuera de rango.')
    return args


if __name__ == '__main__':
    args = argumentos()
    funcion = preparar_datos if args.preparar else cargar_datos
    datos = funcion(args.imagenes_por_clase, args.semilla, args.prueba_corta)
    for parte in ('train', 'val', 'test'):
        print(parte, datos['x_' + parte].shape)
