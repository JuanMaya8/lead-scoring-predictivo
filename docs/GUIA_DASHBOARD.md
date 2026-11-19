# Guía del dashboard de Lead Scoring

Responsable: Jack. Esta guía explica qué muestra cada pestaña y cómo se lee, para el equipo comercial
y para la exposición del proyecto.

## Conceptos clave

| Concepto | Qué es |
|---|---|
| **Lead Score** | Probabilidad de conversión × 100 (de 0 a 100). |
| **Ranking** | Posición del lead ordenando de mayor a menor score. El #1 es el más probable. |
| **Prioridad** | Nivel A/B/C/D según el score, con una acción comercial y un tiempo de respuesta (SLA). |
| **Score out-of-fold** | Cada lead histórico fue puntuado por un modelo que **no lo vio** al entrenar, así que los resultados del dashboard son honestos. |

## Sistema de priorización

| Prioridad | Score | Acción | SLA |
|---|---|---|---|
| A - Caliente | ≥ 75 | Llamada inmediata de un asesor senior | < 1 hora |
| B - Tibio | 50 – 74 | Llamada + correo personalizado | < 24 horas |
| C - Templado | 25 – 49 | Nurturing: correo/WhatsApp con contenido del curso | Semanal |
| D - Frío | < 25 | Campañas automáticas de bajo costo | Mensual |

Los colores de los niveles van de azul oscuro (A) a azul claro (D): una sola escala ordenada,
legible para personas con daltonismo.

## Pestañas

1. **📊 Resumen.** Indicadores clave, distribución del score y conversión real por rango de score.
   Si las barras de conversión suben de izquierda a derecha, el modelo ordena bien a los leads.
2. **🏆 Ranking.** Tabla de leads ordenada por score, con búsqueda por número de lead y descarga en CSV.
   Es la lista que se entrega al equipo de ventas.
3. **🚦 Priorización.** Tarjetas por nivel, esfuerzo vs. resultado, curva de ganancia y el
   **planificador de capacidad**: se indica cuántos leads puede contactar el equipo y el dashboard
   estima cuántas conversiones se obtienen siguiendo el ranking frente a llamar al azar.
4. **🤖 Modelos.** Comparación de los modelos, curvas ROC (resaltado el modelo en producción),
   matriz de confusión con umbral ajustable y lift por decil.
5. **🔍 Factores clave.** Importancia de variables por permutación y conversión real por categoría.
6. **🧮 Puntuar leads.** Simulador para un lead nuevo (con medidor de score) y carga masiva de un CSV,
   con plantilla descargable.
7. **📘 Metodología.** Resumen CRISP-DM, fuga de información y métricas finales del modelo.

## Filtros

La barra lateral filtra por prioridad, rango de score, origen, fuente, ocupación y última actividad.
Los filtros afectan las pestañas Resumen, Ranking, Priorización y Factores clave. La pestaña Modelos
siempre muestra el conjunto de prueba completo para que las métricas sean comparables.

## Guion sugerido para la exposición (3 minutos)

1. **Resumen:** "De 9.240 leads, solo el 38,5 % convierte. El score separa bien a quienes sí convierten."
2. **Priorización:** "Los leads A y B son el 37 % de la base y concentran el 76 % de las conversiones,
   con una tasa de 80 %, la meta del caso."
3. **Planificador:** mover el control a 1.000 llamadas y mostrar la mejora frente al azar.
4. **Puntuar leads:** crear un lead "Working Professional" con SMS enviado y mucho tiempo en la web
   y mostrar que sale prioridad A.
