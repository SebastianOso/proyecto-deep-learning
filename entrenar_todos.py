"""Entrenar las cuatro variantes, elegir por validación y evaluar al ganador en test."""
from pathlib import Path
import shutil
import numpy as np
from etl import cargar_datos
from quickdraw_config import REPO, guardar_json
from quickdraw_comun import argumentos, entrenar, limpiar_salida, medir
from experimentos import base, dropout, aumento, combinado


def main(argv=None):
    args = argumentos('comparación de cuatro variantes', argv)
    if args.modo != 'entrenar': raise ValueError('Usa modelo_final.py para demostrar.')
    datos = cargar_datos(args.imagenes_por_clase, args.semilla, args.prueba_corta)
    raiz = REPO / 'modelos'
    if args.prueba_corta: raiz /= 'prueba_corta'
    resultados = []
    for nombre, modulo in [('base',base), ('dropout',dropout), ('aumento',aumento), ('combinado',combinado)]:
        print('\nENTRENANDO:', nombre, flush=True)
        resultados.append(entrenar(nombre, modulo.crear_modelo, datos, args.epocas,
                                    args.batch, args.semilla, args.prueba_corta))
    ganador = min(resultados, key=lambda r:r['val']['loss'])
    # Se cierra la selección antes de calcular cualquier resultado de prueba.
    final = raiz / 'final'
    limpiar_salida(final)
    for nombre in ('modelo.h5','mejor.keras','historial.csv','curvas_entrenamiento.png'):
        shutil.copy2(raiz / ganador['modelo'] / nombre, final / nombre)
    seleccion = dict(ganador=ganador['modelo'], criterio='menor pérdida de validación',
                     variantes=resultados, semilla=args.semilla, batch=args.batch,
                     por_clase=int(datos['por_clase']), prueba_corta=args.prueba_corta)
    guardar_json(final / 'seleccion.json', seleccion)
    import tensorflow as tf
    modelo = tf.keras.models.load_model(final / 'modelo.h5', compile=False)
    metricas, probabilidades = medir(modelo, datos['x_test'], datos['y_test'], args.batch)
    guardar_json(final / 'test.json', metricas)
    np.savez_compressed(final / 'predicciones_test.npz', probabilidades=probabilidades,
                        reales=datos['y_test'], indices=datos['indices_test'])
    print('\nGANADOR:', ganador['modelo'], '| TEST:', metricas, flush=True)


if __name__ == '__main__':
    main()
