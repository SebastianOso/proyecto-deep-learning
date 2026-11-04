"""Exportar el H5 existente a TensorFlow.js sin cambiar el modelo original.

Uso: python exportar_web.py
Exportador acotado a la CNN Sequential de este proyecto. No necesita instalar
tensorflowjs ni modificar las dependencias de entrenamiento.
"""

import argparse
import json
from pathlib import Path

from quickdraw_config import CLASES_ES, REPO, sha_archivo


def exportar(origen, destino):
    import numpy as np
    import tensorflow as tf

    modelo = tf.keras.models.load_model(origen, compile=False)
    if (tuple(modelo.input_shape) != (None, 28, 28, 1)
            or tuple(modelo.output_shape) != (None, 10)
            or not isinstance(modelo, tf.keras.Sequential)):
        raise ValueError('Se requiere la CNN Sequential de diez clases y entrada 28×28×1.')

    campos = {
        'Conv2D': ('filters', 'kernel_size', 'strides', 'padding', 'data_format',
                   'dilation_rate', 'activation', 'use_bias'),
        'MaxPooling2D': ('pool_size', 'strides', 'padding', 'data_format'),
        'Flatten': ('data_format',),
        'Dense': ('units', 'activation', 'use_bias'),
    }
    capas = [{'class_name': 'InputLayer', 'config': {
        'batch_input_shape': [None, 28, 28, 1], 'dtype': 'float32',
        'sparse': False, 'name': 'entrada_web',
    }}]
    portatil = tf.keras.Sequential([tf.keras.Input(shape=(28, 28, 1))])
    pesos, bloques, omitidas = [], [], []
    escala_encontrada = False
    for capa in modelo.layers:
        tipo = type(capa).__name__
        config = capa.get_config()
        if tipo == 'Rescaling':
            if (escala_encontrada or len(capas) != 1
                    or config['scale'] != 1. / 255 or config['offset'] != 0):
                raise ValueError('Se esperaba una única normalización inicial 1/255.')
            escala_encontrada = True
            continue
        if tipo in ('RandomRotation', 'RandomTranslation', 'Dropout'):
            # Son identidad en la inferencia del modelo original (training=False).
            omitidas.append(tipo)
            continue
        if tipo not in campos:
            raise ValueError(f'Capa no soportada por este exportador: {tipo}')
        if config.get('groups', 1) != 1:
            raise ValueError('No se soportan convoluciones agrupadas.')
        especificacion = {k: config[k] for k in campos[tipo]}
        especificacion.update(name=capa.name, trainable=False, dtype='float32')
        if tipo in ('Conv2D', 'Dense'):
            # Los inicializadores no se usan: todos los pesos se cargan del H5.
            especificacion.update(
                kernel_initializer={'class_name': 'Zeros', 'config': {}},
                bias_initializer={'class_name': 'Zeros', 'config': {}},
            )
        capas.append(dict(class_name=tipo, config=especificacion))
        clon = type(capa).from_config(config)
        portatil.add(clon)
        clon.set_weights(capa.get_weights())
        for variable in capa.weights:
            valor = np.asarray(variable.numpy(), dtype='<f4')
            nombre = variable.name.rsplit('/', 1)[-1].removesuffix(':0')
            pesos.append(dict(name=f'{capa.name}/{nombre}', shape=list(valor.shape), dtype='float32'))
            bloques.append(valor.tobytes(order='C'))
    if not escala_encontrada:
        raise ValueError('Falta la normalización 1/255 esperada en el modelo original.')

    # Verificar la copia de inferencia antes de escribir los archivos públicos.
    muestras = np.random.default_rng(42).integers(0, 256, (8, 28, 28, 1), dtype=np.uint8)
    esperado = modelo(muestras, training=False).numpy()
    obtenido = portatil(muestras.astype('float32') / 255, training=False).numpy()
    np.testing.assert_allclose(esperado, obtenido, rtol=1e-5, atol=1e-6)

    destino.mkdir(parents=True, exist_ok=True)
    datos = b''.join(bloques)
    (destino / 'pesos.bin').write_bytes(datos)
    artefacto = dict(
        format='layers-model', generatedBy=f'QuickDraw inference export / Keras {tf.keras.__version__}',
        convertedBy='exportar_web.py',
        modelTopology=dict(class_name='Sequential', config=dict(name='quickdraw_web', layers=capas),
                           backend='tensorflow', keras_version=tf.keras.__version__),
        weightsManifest=[dict(paths=['pesos.bin'], weights=pesos)],
        userDefinedMetadata=dict(clases=CLASES_ES, entrada='float32_0_1',
            origen='modelos/final/modelo.h5', sha256_origen=sha_archivo(origen),
            sha256_pesos=sha_archivo(destino / 'pesos.bin'),
            capas_solo_entrenamiento_omitidas=omitidas),
    )
    (destino / 'model.json').write_text(json.dumps(artefacto, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Exportado: {destino} ({len(datos):,} bytes de pesos, sin cuantización).')
    print('Copia de inferencia validada contra el H5 original; no se reentrenó.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelo', type=Path, default=REPO / 'modelos' / 'final' / 'modelo.h5')
    parser.add_argument('--salida', type=Path, default=REPO / 'docs' / 'modelo')
    args = parser.parse_args()
    exportar(args.modelo, args.salida)
