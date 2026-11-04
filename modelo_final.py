"""Clasificar con el ganador seleccionado por entrenar_todos.py.

Entrenar y seleccionar: python entrenar_todos.py
Demostrar: python modelo_final.py --imagen dibujo.png
"""
import argparse
from pathlib import Path
from quickdraw_config import REPO
from quickdraw_comun import demostrar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagen', type=Path, required=True)
    parser.add_argument('--fondo', choices=('auto','blanco','negro'), default='auto')
    args = parser.parse_args()
    demostrar(REPO / 'modelos' / 'final', args.imagen, args.fondo)
