"""CNN del experimento que combina dropout y aumento de Keras.

    .venv/bin/python experimentos/combinado.py --modo entrenar
"""

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quickdraw_comun import compilar_modelo, main


def crear_modelo(semilla=42):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (Input, Rescaling, Conv2D, MaxPooling2D,
                                         Flatten, Dense, Dropout, RandomRotation,
                                         RandomTranslation)

    tf.keras.utils.set_random_seed(semilla)
    modelo = Sequential([
        Input(shape=(28, 28, 1)), Rescaling(1. / 255),
        RandomRotation(10 / 360, fill_mode='constant', fill_value=0, seed=semilla),
        RandomTranslation(.1, .1, fill_mode='constant', fill_value=0, seed=semilla + 1),
        Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(2),
        Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(2),
        Flatten(), Dense(64, activation='relu'), Dropout(.3, seed=semilla),
        Dense(10, activation='softmax'),
    ], name='quickdraw_combinado')
    return compilar_modelo(modelo)


if __name__ == '__main__':
    main('combinado', crear_modelo)
