# Sistema de recomendación de películas con cinco enfoques de Machine Learning

Proyecto de práctica para **Cómputo Distribuido y Paralelo – UPAO**.

## 1. Problema

Imaginemos una plataforma de streaming, parecida a Netflix, con muchos títulos y usuarios con gustos diferentes. La plataforma necesita decidir **qué películas recomendar a cada persona** para que encuentre contenido que le interese.

En este proyecto se resuelve el mismo problema de cinco maneras. Cada enfoque de Machine Learning aporta una solución distinta. Los nombres de películas, usuarios, gustos y clics son **ficticios o simulados**; no se usan datos de Netflix ni de personas reales.

## 2. Solución según cada categoría

| Categoría | Situación de ejemplo | Algoritmo | Solución que entrega |
| --- | --- | --- | --- |
| **a) Supervisado** | La plataforma sabe qué películas recibieron «me gusta» y «no me gusta». | Regresión logística | Predice la probabilidad de que una persona disfrute una película nueva. |
| **b) No supervisado** | La plataforma no conoce los tipos de espectadores. | K-Means | Agrupa a los usuarios por gustos parecidos y recomienda lo popular en su grupo. |
| **c) Semisupervisado** | Sólo una parte de las opiniones está etiquetada. | Autoentrenamiento con regresión logística | Aprovecha el 25% de opiniones conocidas y genera pseudoetiquetas para aprender de las demás. |
| **d) Autosupervisado** | Se quiere aprender de los datos de las películas sin anotaciones nuevas. | Reconstrucción de géneros ocultos con Ridge | Oculta parte de los géneros, aprende a reconstruirlos y utiliza sus representaciones para recomendar títulos similares. |
| **e) Por refuerzo** | Las recomendaciones pueden mejorar después de cada clic del espectador. | Bandido de varios brazos, epsilon-greedy | Prueba géneros, observa recompensas y adapta la elección con el tiempo. |

### Ejemplo concreto

Ana suele disfrutar películas de **acción** y **ciencia ficción**. La plataforma conoce 17 opiniones previas de Ana y dispone de títulos que ella todavía no vio.

- **Supervisado:** estima cuáles películas nuevas recibirían un «me gusta».
- **No supervisado:** identifica espectadores parecidos a Ana y recomienda títulos apreciados por ese grupo.
- **Semisupervisado:** aprende con unas pocas opiniones calificadas y amplía su entrenamiento con predicciones provisionales.
- **Autosupervisado:** aprende relaciones entre géneros sin pedir etiquetas nuevas, reconstruyendo información ocultada deliberadamente.
- **Por refuerzo:** muestra géneros, observa clics simulados y aumenta la frecuencia de los que dan mejores resultados.

**Resultado esperado:** cada método imprime tres títulos recomendados, su algoritmo y una medida o resumen del aprendizaje. Los títulos concretos pueden diferir entre métodos, porque cada uno aprende de forma distinta.

## 3. Requisitos

- Python 3.10 o más reciente.
- `numpy` y `scikit-learn`.

## 4. Cómo ejecutarlo

Abre una terminal dentro de esta carpeta y ejecuta:

```bash
python -m pip install -r requirements.txt
python main.py --metodo todos --usuario ana
```

Para ejecutar sólo un método:

```bash
python main.py --metodo supervisado --usuario ana
python main.py --metodo no_supervisado --usuario ana
python main.py --metodo semisupervisado --usuario ana
python main.py --metodo autosupervisado --usuario ana
python main.py --metodo refuerzo --usuario ana
```

Puedes sustituir `ana` por `bruno`, `carla`, `diego`, `elena`, `felipe`, `gabriela`, `hugo`, `ines`, `jorge`, `karla` o `luis`.

Para comprobar el proyecto:

```bash
python -m unittest discover -s tests -v
```

## 5. Archivos del repositorio

```text
recomendador_ml_github/
├── data/
│   └── catalogo_peliculas.csv      # 32 películas ficticias
├── src/
│   ├── __init__.py
│   ├── datos.py                     # Carga de datos y opiniones simuladas
│   └── metodos.py                   # Los cinco algoritmos
├── tests/
│   └── test_recomendador.py         # Pruebas automáticas
├── GUIA_EXPLICACION.md             # Explicación para el informe o exposición
├── main.py                         # Ejecuta las cinco soluciones
├── requirements.txt
└── README.md
```

## 6. Cómo subirlo a GitHub

1. Entra a GitHub y crea un repositorio vacío llamado `recomendador-cinco-enfoques-ml`.
2. Extrae el ZIP del proyecto y abre una terminal en su carpeta.
3. Ejecuta estos comandos, cambiando `TU_USUARIO` por tu nombre de usuario de GitHub:

```bash
git init
git add .
git commit -m "Proyecto de recomendación con cinco enfoques de ML"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/recomendador-cinco-enfoques-ml.git
git push -u origin main
```

Si GitHub solicita autenticación, inicia sesión de manera normal en GitHub o configura Git Credential Manager. No incluyas contraseñas ni tokens en el código.

## 7. Limitaciones

- Es una **demostración académica**, no un sistema productivo de Netflix.
- Las interacciones están simuladas; las métricas no describen rendimiento sobre usuarios reales.
- El autosupervisado reconstruye información enmascarada de los mismos registros, en lugar de usar etiquetas suministradas para esta tarea.
- El ejemplo de refuerzo es un **bandido de varios brazos**: aprende con recompensas, pero no modela estados complejos como un agente de aprendizaje por refuerzo completo.

Esta separación es importante porque muestra **qué aprende cada técnica y de dónde obtiene la información necesaria**.
