## 1) Plan de tiempo (para no morir por perfeccionismo)

**Si el examen mezcla práctica + teoría:**

* 5 min: leer todo y marcar qué pide *mínimos* (random_state, accuracy objetivo, gráficos, etc.).
* 60–70% del tiempo: práctica (porque es “mecánica” y da puntos seguros).
* 30–40%: teoría (porque con estructura puntúa aunque no sea perfecta).
* Últimos 5–10 min: repaso de “requisitos obligatorios” (semillas, matrices, métricas, etc.).

Regla de oro: **si una parte se atasca 5–7 min, pivota** (modelo alternativo / hiperparámetro estándar).

---

## 2) Checklist de requisitos “mata-notas” (lo que suele penalizar)

Añade al principio de tus apuntes una cajita con esto (para mirarlo al final):

* ✅ `random_state = X` EXACTO del enunciado
* ✅ `test_size = X` EXACTO
* ✅ `stratify=y` en clasificación
* ✅ pipeline fit **solo en train**
* ✅ gráficas “tras limpiar” (si lo piden)
* ✅ accuracy objetivo (80/84/90…) + **matriz de confusión**
* ✅ prueba con **>1 capa oculta** si lo piden (aunque no sea tu modelo final)
* ✅ comparación final en texto (2–5 líneas) si lo piden

Esto solo ya evita perder puntos “tontos”.

---

## 3) Kit de “si algo va mal” (para no entrar en pánico)

Ten una mini tabla mental:

### Si **MLP/KNN rinden fatal**

* Revisa **normalización** (StandardScaler) y **one-hot**.
* Revisa que convertiste strings (`strip/lower/upper`) antes de one-hot.
* Revisa NaNs / valores imposibles.
* Asegúrate de `stratify=y`.

### Si **train alto / test bajo** (overfitting)

* Baja neuronas/capas.
* Sube regularización L2.
* Early stopping (sklearn).
* Menos epochs / dropout (si entra en teoría).

### Si **train bajo y test bajo** (underfitting)

* Más neuronas/capas.
* Más epochs.
* Ajusta lr (si tu MLP propio).

### Si necesitas “salvar” una accuracy objetivo

Ten **un comodín**:

* Para tabular pequeño: **RandomForest / DecisionTree** suelen salvarte rápido (si el examen lo permite).
* Si solo permiten KNN/MLP: KNN con `weights="distance"` y probar k’s típicos suele levantar.

---

## 4) Lo único que yo añadiría a tus apuntes (extra óptimo)

### A) Una hoja de “frases que puntúan” (para teoría y justificación final)

Tres frases por tema:

* **Limpieza:** “Evito fuga eliminando IDs/constantes e imputo mediana/moda para no perder datos.”
* **Evaluación:** “Además de accuracy, miro matriz de confusión para entender errores; en médico priorizo reducir FN.”
* **Generalización:** “Si train>>val es sobreajuste → regularización/early stopping; si ambos bajos → más capacidad/entrenamiento.”
* **No supervisado:** “Sin etiquetas uso clustering/PCA para estructura; si luego quiero clasificar, semi-supervisado o etiquetado parcial.”
* **DL:** “CNN para imágenes por invariancias locales; AE para compresión/representación/anomalías.”
* **RL:** “Defino estado/acción/recompensa; problema típico reward sparsity → exploración/intrinsic/curriculum.”

Esto te permite contestar aunque estés “en blanco”.

### B) Un “micro-template” de respuesta teórica (10 líneas)

Para copiar siempre:

1. “Es un problema de ___ porque ___”
2. “Concepto clave: ___ (definición breve)”
3. “Aplicaría ___, ___, ___”
4. “Riesgo: ___”
5. “Mitigación: ___”
6. “Conclusión: elegiría ___”

---

## 5) Consejos “de guerra” el día del examen

* Lleva agua + algo de azúcar/cafeína (si te deja) y **boli de sobra**.
* Haz una pasada rápida para “cazar” palabras clave del enunciado: **semilla, split, métricas, thresholds**.
* No te enamores de tu primer modelo: busca **cumplir el umbral**, no optimizar.
* Si te obligan a justificar: escribe **3 líneas limpias** mejor que un párrafo largo.
* Si una parte te sale “regular” pero cumple el objetivo, **NO la toques**.

---