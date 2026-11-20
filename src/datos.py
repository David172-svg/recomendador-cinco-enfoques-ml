"""Catálogo ficticio e interacciones simuladas reproducibles.

No se recopilan datos de usuarios reales ni se consulta Netflix.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


RAIZ = Path(__file__).resolve().parents[1]

USUARIOS = {
    "ana": ("Ciencia ficción", "Acción"),
    "bruno": ("Drama", "Romance"),
    "carla": ("Animación", "Comedia"),
    "diego": ("Documental", "Suspenso"),
    "elena": ("Comedia", "Romance"),
    "felipe": ("Acción", "Suspenso"),
    "gabriela": ("Documental", "Drama"),
    "hugo": ("Ciencia ficción", "Documental"),
    "ines": ("Animación", "Romance"),
    "jorge": ("Suspenso", "Ciencia ficción"),
    "karla": ("Drama", "Comedia"),
    "luis": ("Acción", "Animación"),
}


def cargar_catalogo() -> list[dict]:
    with (RAIZ / "data" / "catalogo_peliculas.csv").open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    return [
        {"id": int(fila["id"]), "titulo": fila["titulo"],
         "generos": tuple(fila["generos"].split("|"))}
        for fila in filas
    ]


def lista_generos(catalogo: list[dict]) -> list[str]:
    return sorted({genero for pelicula in catalogo for genero in pelicula["generos"]})


def generar_interacciones(catalogo: list[dict], semilla: int = 2026) -> list[dict]:
    """Genera 17 opiniones por usuario para permitir entrenamiento y recomendaciones.

    La preferencia oculta sólo crea la simulación; los modelos aprenden de las
    observaciones (película, usuario, resultado de me gusta/no me gusta).
    """
    rng = np.random.default_rng(semilla)
    registros = []
    for usuario, gustos in USUARIOS.items():
        indices = rng.choice(len(catalogo), size=17, replace=False)
        for idx in indices:
            pelicula = catalogo[int(idx)]
            coincide = bool(set(gustos).intersection(pelicula["generos"]))
            probabilidad = 0.80 if coincide else 0.16
            registros.append({
                "usuario": usuario,
                "pelicula_id": pelicula["id"],
                "me_gusta": int(rng.random() < probabilidad),
            })
    return registros


def peliculas_no_vistas(catalogo: list[dict], registros: list[dict], usuario: str) -> list[dict]:
    vistas = {r["pelicula_id"] for r in registros if r["usuario"] == usuario}
    return [p for p in catalogo if p["id"] not in vistas]


def caracteristicas(usuario: str, pelicula: dict, generos: list[str]) -> dict:
    variables = {"usuario": usuario}
    for genero in generos:
        presente = int(genero in pelicula["generos"])
        variables[f"genero_{genero}"] = presente
        # Esta interacción permite aprender preferencias distintas por usuario.
        variables[f"usuario_{usuario}_genero_{genero}"] = presente
    return variables


def matriz_generos(catalogo: list[dict], generos: list[str]) -> np.ndarray:
    return np.array(
        [[int(g in p["generos"]) for g in generos] for p in catalogo],
        dtype=float,
    )
