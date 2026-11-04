"""Demo local: python app.py (http://127.0.0.1:5000)."""

import base64
import binascii
from io import BytesIO
from pathlib import Path
import sys
from threading import Lock

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image, UnidentifiedImageError

from quickdraw_comun import preparar_dibujo
from quickdraw_config import CLASES_ES


RUTA_MODELO = Path(__file__).resolve().parent / 'modelos' / 'final' / 'modelo.h5'


def crear_app(ruta_modelo=RUTA_MODELO):
    """Cargar el modelo una vez por proceso y crear el servidor local."""
    ruta_modelo = Path(ruta_modelo)
    if not ruta_modelo.is_file():
        raise RuntimeError(f'No se encontró el modelo: {ruta_modelo}')

    import tensorflow as tf

    try:
        modelo = tf.keras.models.load_model(ruta_modelo, compile=False)
    except Exception as error:
        raise RuntimeError(f'No se pudo cargar el modelo {ruta_modelo}: {error}') from error
    if (tuple(modelo.input_shape) != (None, 28, 28, 1)
            or tuple(modelo.output_shape) != (None, len(CLASES_ES))):
        raise RuntimeError('Modelo incompatible: se esperaba entrada (28, 28, 1) '
                           'y salida de diez clases.')

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
    app.extensions['modelo_quickdraw'] = modelo
    bloqueo = Lock()

    @app.get('/')
    def inicio():
        return render_template('index.html', clases=CLASES_ES)

    @app.errorhandler(413)
    def demasiado_grande(_error):
        return jsonify(error='El dibujo supera el límite de 1 MB.'), 413

    @app.post('/predecir')
    def predecir():
        if not request.is_json:
            return jsonify(error='Envía el dibujo en formato JSON.'), 400
        contenido = request.get_json(silent=True)
        imagen = contenido.get('imagen') if isinstance(contenido, dict) else None
        prefijo = 'data:image/png;base64,'
        if not isinstance(imagen, str) or not imagen.startswith(prefijo):
            return jsonify(error='Envía una imagen PNG válida del lienzo.'), 400
        try:
            archivo = BytesIO(base64.b64decode(imagen[len(prefijo):], validate=True))
            with Image.open(archivo) as png:
                if png.format != 'PNG' or not (2 <= png.width <= 1024
                                              and 2 <= png.height <= 1024):
                    return jsonify(error='Usa un PNG de entre 2 y 1024 píxeles por lado.'), 400
                png.verify()
            archivo.seek(0)
            entrada = preparar_dibujo(archivo, fondo='blanco')[None, ...]
        except (binascii.Error, UnidentifiedImageError, OSError, SyntaxError,
                Image.DecompressionBombError):
            return jsonify(error='No se pudo leer el dibujo. Limpia el lienzo e inténtalo otra vez.'), 400
        except ValueError as error:
            return jsonify(error=str(error)), 400

        try:
            with bloqueo:
                probabilidades = modelo(entrada, training=False).numpy()[0]
            if (probabilidades.shape != (len(CLASES_ES),)
                    or not np.all(np.isfinite(probabilidades))
                    or np.any((probabilidades < 0) | (probabilidades > 1))):
                raise ValueError('Salida del modelo inválida.')
        except Exception:
            app.logger.exception('No se pudo predecir el dibujo')
            return jsonify(error='No se pudo calcular la predicción. Inténtalo otra vez.'), 500
        indices = np.argsort(probabilidades)[::-1][:3]
        return jsonify(predicciones=[
            dict(clase=CLASES_ES[i], probabilidad=float(probabilidades[i]))
            for i in indices
        ])

    return app


if __name__ == '__main__':
    try:
        aplicacion = crear_app()
    except RuntimeError as error:
        print(f'Error al iniciar: {error}', file=sys.stderr)
        raise SystemExit(1) from error
    aplicacion.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
