"""Demostración ejecutable del sistema de recomendación de películas.

Uso: python main.py --metodo todos --usuario ana
"""

import argparse

from src.datos import USUARIOS, cargar_catalogo, generar_interacciones
from src.metodos import METODOS


NOMBRES = {
    "supervisado": "a) APRENDIZAJE SUPERVISADO",
    "no_supervisado": "b) APRENDIZAJE NO SUPERVISADO",
    "semisupervisado": "c) APRENDIZAJE SEMISUPERVISADO",
    "autosupervisado": "d) APRENDIZAJE AUTOSUPERVISADO",
    "refuerzo": "e) APRENDIZAJE POR REFUERZO",
}


def main():
    parser = argparse.ArgumentParser(
        description="Ejemplo de cinco enfoques de ML para recomendar películas"
    )
    parser.add_argument(
        "--metodo", choices=["todos", *METODOS], default="todos",
        help="Método que quieres ejecutar (por defecto: todos)",
    )
    parser.add_argument(
        "--usuario", choices=list(USUARIOS), default="ana",
        help="Usuario ficticio al que se le recomendarán películas",
    )
    argumentos = parser.parse_args()

    catalogo = cargar_catalogo()
    interacciones = generar_interacciones(catalogo)
    elegidos = list(METODOS) if argumentos.metodo == "todos" else [argumentos.metodo]
    print("=" * 68)
    print("PLATAFORMA DE STREAMING - RECOMENDACIÓN PERSONALIZADA CON ML")
    print(f"Películas ficticias: {len(catalogo)} | Usuarios simulados: {len(USUARIOS)}")
    print(f"Usuario de ejemplo: {argumentos.usuario.capitalize()}")
    print("=" * 68)

    for clave in elegidos:
        resultado = METODOS[clave](catalogo, interacciones, argumentos.usuario)
        print(f"\n{NOMBRES[clave]}")
        print("Algoritmo:", resultado["algoritmo"])
        print("Solución:", resultado["explicacion"])
        print("Resultado:", resultado["metrica"])
        print("Recomendaciones:")
        for titulo, valor in resultado["recomendaciones"]:
            sufijo = "" if clave == "refuerzo" else f" (puntaje: {valor:.3f})"
            print(f"  - {titulo}{sufijo}")
    print("\nAviso: todos los perfiles, opiniones y clics son simulados para clase.")


if __name__ == "__main__":
    main()
