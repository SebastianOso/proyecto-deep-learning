"""Constantes y utilidades compartidas, sin dependencias numéricas."""

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent
CLASES_EN = ['cat', 'fish', 'bicycle', 'airplane', 'house', 'tree',
             'clock', 'umbrella', 'ice cream', 'star']
CLASES_ES = ['gato', 'pez', 'bicicleta', 'avión', 'casa', 'árbol',
             'reloj', 'paraguas', 'helado', 'estrella']
PREPROCESAMIENTO = dict(version=1, lado=28, contenido=24,
                       entrada='uint8_0_255', fondo_modelo='negro',
                       trazo_modelo='claro', centrado='bounding_box')


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding='utf-8'))


def guardar_json(ruta, valor):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(valor, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def sha_archivo(ruta):
    h = hashlib.sha256()
    with Path(ruta).open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


