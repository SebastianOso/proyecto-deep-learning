"""Modelo de QuickDraw_Keras_Combinado_Fundamentado.ipynb para uso local.

Instalar dependencias dentro de .venv:
    .venv/bin/python -m pip install -r requirements.txt

Entrenar (evalúa test al terminar y guarda H5, pesos y gráfica):
    .venv/bin/python modelo_final.py --modo entrenar
Reanudar desde el último checkpoint de cada 5 épocas:
    .venv/bin/python modelo_final.py --modo reanudar
Clasificar una imagen sin cargar el dataset:
    .venv/bin/python modelo_final.py --modo demostrar --imagen dibujo.png
"""

from quickdraw_comun import compilar_modelo, main


def crear_modelo(semilla=42):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (Input, Rescaling, Conv2D, MaxPooling2D,
                                         Flatten, Dense, Dropout, RandomRotation,
                                         RandomTranslation)

    tf.keras.utils.set_random_seed(semilla)
    modelo = Sequential([
        Input(shape=(28, 28, 1)),
        Rescaling(1. / 255),
        RandomRotation(10 / 360, fill_mode='constant', fill_value=0, seed=semilla),
        RandomTranslation(.1, .1, fill_mode='constant', fill_value=0, seed=semilla + 1),
        Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(2),
        Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(2),
        Flatten(), Dense(64, activation='relu'), Dropout(.3, seed=semilla),
        Dense(10, activation='softmax'),
    ], name='quickdraw_combinado')
    return compilar_modelo(modelo)


if __name__ == '__main__':
    main('final', crear_modelo)
