"""ETL de QuickDraw: descarga, limpieza, particiones y caché, sin TensorFlow."""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from quickdraw_config import (REPO, CLASES_EN, entero_positivo, firma_json,
                              leer_json, guardar_json, sha_archivo)


def ruta_particiones(por_clase, semilla, prueba_corta=False, raiz=None):
    raiz = REPO if raiz is None else Path(raiz)
    nombre = f'{"prueba_" if prueba_corta else ""}n{por_clase}_s{semilla}'
    return raiz / 'datos' / 'quickdraw' / 'particiones' / nombre


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
    particiones = ruta_particiones(por_clase, semilla, prueba_corta, raiz)
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


def argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagenes-por-clase', type=entero_positivo, default=5000)
    parser.add_argument('--semilla', type=int, default=42)
    parser.add_argument('--prueba-corta', action='store_true',
                        help='100 imágenes por clase; caché separado (descarga fuentes completas).')
    args = parser.parse_args(argv)
    if not 0 <= args.semilla <= 2**32 - 2:
        parser.error('--semilla debe estar entre 0 y 4294967294.')
    if args.prueba_corta:
        args.imagenes_por_clase = 100
    if args.imagenes_por_clase < 7:
        parser.error('--imagenes-por-clase debe ser al menos 7.')
    return args


def main(argv=None):
    args = argumentos(argv)
    try:
        _, firma = cargar_datos(args.imagenes_por_clase, args.semilla, args.prueba_corta)
        print('Particiones:', ruta_particiones(args.imagenes_por_clase,
                                              args.semilla, args.prueba_corta))
        print('Firma del manifiesto:', firma)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
