# Explicación del ejemplo para clase

**Tema:** Sistema de recomendación y personalización de películas en una plataforma de streaming.

**Problema:** Los usuarios tienen gustos diferentes y una plataforma ofrece muchas películas. Si recomienda lo mismo a todos, algunas personas recibirán contenido que no les interesa. Por eso proponemos cinco formas de aprender qué recomendar.

## a) Aprendizaje supervisado

**Ejemplo:** Ana marcó algunas películas con «me gusta» y otras con «no me gusta». Queremos saber si disfrutará una película nueva.

**Algoritmo:** Regresión logística.

**Solución:** Registramos el usuario, los géneros de cada película y su valoración (1 para «me gusta», 0 para «no me gusta»). El algoritmo aprende de estos ejemplos y calcula la probabilidad de aceptación de títulos todavía no vistos. Se recomiendan los de mayor probabilidad.

## b) Aprendizaje no supervisado

**Ejemplo:** No sabemos qué tipos de espectadores tenemos, pero sí observamos cuáles géneros disfruta cada usuario.

**Algoritmo:** K-Means.

**Solución:** Convertimos sus preferencias en números, por ejemplo cuánto suele gustarle acción, romance o documental. K-Means reúne a usuarios con perfiles semejantes. Después recomendamos a Ana películas que han gustado dentro de su grupo y que ella todavía no vio.

## c) Aprendizaje semisupervisado

**Ejemplo:** Sólo tenemos un 25% de opiniones calificadas. El resto de los registros no tiene una etiqueta confiable.

**Algoritmo:** Autoentrenamiento (SelfTrainingClassifier) con regresión logística.

**Solución:** Entrenamos primero con la pequeña parte conocida. Luego el modelo propone etiquetas para más registros, las utiliza como ejemplos adicionales y repite el proceso de manera controlada. Finalmente recomienda películas. Las nuevas etiquetas son aproximaciones, no opiniones reales.

## d) Aprendizaje autosupervisado

**Ejemplo:** Tenemos los géneros de las películas, pero queremos aprender relaciones entre ellos sin pedir a alguien que anote nuevos resultados.

**Algoritmo:** Reconstrucción de características ocultas con regresión Ridge.

**Solución:** Ocultamos aleatoriamente algunos géneros del catálogo y utilizamos la versión original como respuesta de entrenamiento. El modelo aprende a reconstruir la información faltante. Esa representación sirve para comparar películas y elegir las que se parecen a otras que Ana disfrutó. La respuesta de entrenamiento se obtiene automáticamente de los datos mismos.

## e) Aprendizaje por refuerzo

**Ejemplo:** La plataforma recomienda un género y observa si el espectador hace clic.

**Algoritmo:** Epsilon-greedy (bandido de varios brazos).

**Solución:** El agente elige géneros. Algunas veces explora opciones nuevas; otras veces muestra el género que ha recibido más clics. Con cada respuesta actualiza su estimación de recompensa. Tras varias rondas puede concentrarse en los géneros que más interesan al usuario. Los clics de este ejemplo se simulan para que funcione sin usuarios reales.

## Conclusión

Los cinco enfoques buscan mejorar las recomendaciones, pero **no aprenden de la misma manera**. El supervisado necesita respuestas conocidas; el no supervisado identifica grupos; el semisupervisado combina pocos datos etiquetados con muchos sin etiquetar; el autosupervisado crea una tarea usando los propios datos; y el aprendizaje por refuerzo mejora a partir de recompensas por sus decisiones.

**Nota de honestidad académica:** El ejemplo emplea 32 títulos inventados y 12 usuarios simulados. Es adecuado para demostrar los algoritmos, no para afirmar resultados reales de una empresa de streaming.
