"""Cinco maneras de recomendar películas usando aprendizaje automático.

Las salidas son demostraciones didácticas con un conjunto pequeño y simulado.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.semi_supervised import SelfTrainingClassifier

from src.datos import (
    USUARIOS, caracteristicas, lista_generos, matriz_generos,
    peliculas_no_vistas,
)


def _preparar(catalogo: list[dict], interacciones: list[dict]):
    indice = {p["id"]: p for p in catalogo}
    generos = lista_generos(catalogo)
    X = [caracteristicas(i["usuario"], indice[i["pelicula_id"]], generos)
         for i in interacciones]
    y = np.array([i["me_gusta"] for i in interacciones], dtype=int)
    return indice, generos, X, y


def _resultado(ranking: list[tuple[dict, float]], limite: int = 3) -> list[tuple[str, float]]:
    return [(p["titulo"], round(float(valor), 3)) for p, valor in ranking[:limite]]


def supervisado(catalogo: list[dict], interacciones: list[dict], usuario: str) -> dict:
    """Clasifica películas según la etiqueta histórica me gusta (1/0)."""
    _, generos, X, y = _preparar(catalogo, interacciones)
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    modelo = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1500))
    modelo.fit(X_ent, y_ent)
    exactitud = accuracy_score(y_prueba, modelo.predict(X_prueba))
    candidatas = peliculas_no_vistas(catalogo, interacciones, usuario)
    entradas = [caracteristicas(usuario, p, generos) for p in candidatas]
    probabilidades = modelo.predict_proba(entradas)[:, 1]
    orden = sorted(zip(candidatas, probabilidades), key=lambda x: -x[1])
    return {
        "algoritmo": "Regresión logística",
        "explicacion": "Aprende con etiquetas 1 = me gusta y 0 = no me gusta.",
        "metrica": f"Exactitud en datos reservados: {exactitud:.2%}",
        "recomendaciones": _resultado(orden),
    }


def no_supervisado(catalogo: list[dict], interacciones: list[dict], usuario: str) -> dict:
    """Forma grupos de espectadores sin etiquetas de tipo de usuario."""
    indice = {p["id"]: p for p in catalogo}
    generos = lista_generos(catalogo)
    perfiles = []
    for nombre in USUARIOS:
        vistos_positivos = [indice[i["pelicula_id"]] for i in interacciones
                             if i["usuario"] == nombre and i["me_gusta"] == 1]
        if vistos_positivos:
            perfiles.append(matriz_generos(vistos_positivos, generos).mean(axis=0))
        else:
            perfiles.append(np.zeros(len(generos)))
    perfiles = np.array(perfiles)
    nombres = list(USUARIOS)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    etiquetas = kmeans.fit_predict(perfiles)
    grupo = int(etiquetas[nombres.index(usuario)])
    integrantes = [nombre for nombre, g in zip(nombres, etiquetas) if g == grupo]
    votos = Counter(i["pelicula_id"] for i in interacciones
                    if i["usuario"] in integrantes and i["me_gusta"] == 1)
    perfil = perfiles[nombres.index(usuario)]
    candidatas = peliculas_no_vistas(catalogo, interacciones, usuario)
    ranking = []
    for pelicula in candidatas:
        afinidad = float(np.dot(perfil, matriz_generos([pelicula], generos)[0]))
        ranking.append((pelicula, votos[pelicula["id"]] + 0.25 * afinidad))
    ranking.sort(key=lambda x: -x[1])
    return {
        "algoritmo": "K-Means (3 grupos)",
        "explicacion": "Agrupa usuarios por sus géneros favoritos, sin clases predefinidas.",
        "metrica": "Usuarios del mismo grupo: " + ", ".join(integrantes),
        "recomendaciones": _resultado(ranking),
    }


def semisupervisado(catalogo: list[dict], interacciones: list[dict], usuario: str) -> dict:
    """Entrena inicialmente con pocas etiquetas y añade pseudoetiquetas."""
    _, generos, X_dict, y_real = _preparar(catalogo, interacciones)
    vectorizador = DictVectorizer(sparse=False)
    X = vectorizador.fit_transform(X_dict)
    rng = np.random.default_rng(123)
    conocidos = rng.choice(len(y_real), size=int(len(y_real) * 0.25), replace=False)
    y_parcial = np.full(len(y_real), -1, dtype=int)
    y_parcial[conocidos] = y_real[conocidos]
    if set(y_parcial[conocidos]) != {0, 1}:
        raise RuntimeError("La muestra etiquetada necesita ambas clases")
    modelo = SelfTrainingClassifier(
        estimator=LogisticRegression(max_iter=1500),
        criterion="k_best", k_best=15, max_iter=8,
    )
    modelo.fit(X, y_parcial)
    candidatas = peliculas_no_vistas(catalogo, interacciones, usuario)
    nuevas = vectorizador.transform([caracteristicas(usuario, p, generos)
                                     for p in candidatas])
    probas = modelo.predict_proba(nuevas)[:, 1]
    nuevas_etiquetas = int(np.sum((y_parcial == -1) & (modelo.transduction_ != -1)))
    ranking = sorted(zip(candidatas, probas), key=lambda x: -x[1])
    return {
        "algoritmo": "Autoentrenamiento (SelfTrainingClassifier + Regresión logística)",
        "explicacion": "Parte con 25% de opiniones etiquetadas y genera pseudoetiquetas para el resto.",
        "metrica": f"Etiquetas iniciales: {len(conocidos)}; pseudoetiquetas añadidas: {nuevas_etiquetas}",
        "recomendaciones": _resultado(ranking),
    }


def autosupervisado(catalogo: list[dict], interacciones: list[dict], usuario: str) -> dict:
    """Aprende a reconstruir géneros ocultados: las metas vienen de los propios datos."""
    generos = lista_generos(catalogo)
    matriz = matriz_generos(catalogo, generos)
    rng = np.random.default_rng(321)
    entradas, objetivos = [], []
    for fila in matriz:
        for _ in range(18):
            mascara = rng.random(len(fila)) > 0.35
            entradas.append(fila * mascara)
            objetivos.append(fila)
    X, Y = np.array(entradas), np.array(objetivos)
    modelo = Ridge(alpha=1.0).fit(X, Y)
    reconstruccion = np.clip(modelo.predict(X), 0, 1)
    error = mean_squared_error(Y, reconstruccion)
    embebidos = np.clip(modelo.predict(matriz), 0, 1)
    vistos_gustados = [i["pelicula_id"] for i in interacciones
                       if i["usuario"] == usuario and i["me_gusta"] == 1]
    indice = {pelicula["id"]: pos for pos, pelicula in enumerate(catalogo)}
    perfil = embebidos[[indice[i] for i in vistos_gustados]].mean(axis=0).reshape(1, -1)
    candidatas = peliculas_no_vistas(catalogo, interacciones, usuario)
    posiciones = [indice[p["id"]] for p in candidatas]
    similaridades = cosine_similarity(perfil, embebidos[posiciones])[0]
    ranking = sorted(zip(candidatas, similaridades), key=lambda x: -x[1])
    return {
        "algoritmo": "Reconstrucción de datos enmascarados + Ridge",
        "explicacion": "Oculta géneros y aprende a recuperarlos usando la versión completa del mismo catálogo.",
        "metrica": f"Error cuadrático de reconstrucción: {error:.4f} (en entrenamiento)",
        "recomendaciones": _resultado(ranking),
    }


def por_refuerzo(catalogo: list[dict], interacciones: list[dict], usuario: str) -> dict:
    """Bandido epsilon-greedy: aprende qué género obtiene más clics simulados."""
    generos = lista_generos(catalogo)
    rng = np.random.default_rng(789)
    intentos = np.zeros(len(generos), dtype=int)
    promedios = np.zeros(len(generos), dtype=float)
    recompensas = 0
    # Estas preferencias se usan ÚNICAMENTE por el simulador del usuario.
    # El algoritmo de recomendación observa solo premios 0/1 de cada ronda.
    gustos_ocultos = set(USUARIOS[usuario])
    for _ in range(200):
        explorar = rng.random() < 0.20
        if explorar:
            accion = int(rng.integers(len(generos)))
        else:
            mejores = np.flatnonzero(promedios == promedios.max())
            accion = int(rng.choice(mejores))
        probabilidad_clic = 0.80 if generos[accion] in gustos_ocultos else 0.15
        premio = int(rng.random() < probabilidad_clic)
        intentos[accion] += 1
        promedios[accion] += (premio - promedios[accion]) / intentos[accion]
        recompensas += premio
    mejor_genero = generos[int(np.argmax(promedios))]
    candidatas = [p for p in peliculas_no_vistas(catalogo, interacciones, usuario)
                  if mejor_genero in p["generos"]]
    if not candidatas:
        candidatas = peliculas_no_vistas(catalogo, interacciones, usuario)
    return {
        "algoritmo": "Bandido de varios brazos con epsilon-greedy (ε = 0.20)",
        "explicacion": "Prueba géneros, recibe clics/no clics y mejora la elección con la experiencia.",
        "metrica": (f"Tras 200 rondas simuladas: {recompensas} clics; "
                    f"mejor género aprendido: {mejor_genero}"),
        "recomendaciones": _resultado([(p, 1.0) for p in candidatas]),
    }


METODOS = {
    "supervisado": supervisado,
    "no_supervisado": no_supervisado,
    "semisupervisado": semisupervisado,
    "autosupervisado": autosupervisado,
    "refuerzo": por_refuerzo,
}
