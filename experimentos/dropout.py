"""CNN del experimento de regularización con dropout de 0.3.

    python experimentos/dropout.py --modo entrenar
"""

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quickdraw_comun import compilar_modelo, main


def crear_modelo(semilla=42):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (Input, Rescaling, Conv2D,
                                         MaxPooling2D, Flatten, Dense, Dropout)

    tf.keras.utils.set_random_seed(semilla)
    modelo = Sequential([
        Input(shape=(28, 28, 1)), Rescaling(1. / 255),
        Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(2),
        Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(2),
        Flatten(), Dense(64, activation='relu'), Dropout(.3, seed=semilla),
        Dense(10, activation='softmax'),
    ], name='quickdraw_dropout')
    return compilar_modelo(modelo)


if __name__ == '__main__':
    main('dropout', crear_modelo)
