## Chuleta paso a paso (la receta que casi siempre funciona)

### 1) Carga + inspección (30 segundos que salvan puntos)

**Haz SIEMPRE estas 4 líneas/acciones:**

* `df.head()` → ver columnas.
* `df.info()` → tipos (num vs cat) y nulos.
* `df.isna().sum()` → cuántos nulos y dónde.
* `df.nunique()` → detectar **constantes** y **IDs**.

**Frase de examen:**

> “Primero inspecciono tipos y valores faltantes para decidir una limpieza justificable (evitar borrar filas al azar).”

---

### 2) Limpieza con criterio (lo que más penaliza si lo haces “a ojo”)

Aplica esta checklist en orden:

#### A) Quitar columnas que NO aportan (y justificar)

* **IDs** (`id`, `Subject ID`, `MRI ID`, etc.) → *fuente de fuga de info / memorizar muestras*.
* Columnas **constantes** (`nunique()==1`) → *no aportan señal*.
* Columnas “metadata” no relacionada (si el enunciado sugiere que no es clínica / no es feature real).

**Frase de examen:**

> “Elimino identificadores y constantes para evitar fuga de información y ruido.”

#### B) Normalizar texto en categóricas (esto da puntos fáciles)

Antes de one-hot:

* `strip()` espacios
* uniformar mayúsculas/minúsculas (`upper()` / `lower()`)
* unificar etiquetas raras (`DrugY` vs `drugY`, `M/F`, etc.)

**Frase de examen:**

> “Normalizo categorías para que el modelo no interprete como distintas etiquetas que solo cambian por mayúsculas/espacios.”

#### C) Valores imposibles (clave en heart/dementia típicamente)

Regla: **si un 0 o valor raro no tiene sentido físico**, conviértelo a `NaN` y luego imputas.

* Ej.: colesterol = 0, presión = 0, etc.

**Frase de examen:**

> “Convierto valores imposibles a NaN para tratarlos como faltantes y no introducir sesgo.”

#### D) Nulos: imputar vs borrar (cómo decidir sin liarla)

* Si hay **pocos NaN**: imputación sigue siendo defendible y evita perder datos.
* Si hay **muchos NaN** en una columna: puedes plantearte eliminar columna (si no es clave) o imputar con cuidado.

Regla estándar:

* **Numéricas** → `median` (robusta)
* **Categóricas** → `most_frequent`

**Frase de examen:**

> “Imputo mediana/moda para mantener tamaño de muestra y evitar borrar filas sin justificación.”

---

### 3) Separación X/y (antes de tocar nada “de modelo”)

* `y = df[target]`
* `X = df.drop(columns=[target])`
* Identifica listas:

  * `cat_cols = [...]`
  * `num_cols = [...]`

**Tip:** si dudas si algo es categórico: mira `df[col].dtype` y `df[col].nunique()`.

---

### 4) Split EXACTO del enunciado

Casi siempre te obligan a:

* `random_state = ...`
* `test_size = ...`
* usar `stratify=y` (si es clasificación) para mantener proporciones.

**Frase de examen:**

> “Uso split estratificado con la semilla pedida para reproducibilidad.”

---

### 5) Pipeline de preprocesado (la pieza que te “blinda”)

Esto es lo que más te conviene memorizar: **un pipeline reutilizable para todos los modelos**.

* Numéricas: `SimpleImputer(median)` + `StandardScaler()`
* Categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`

**Por qué esto puntúa:** porque evitas leakage y lo aplicas igual a MLP/KNN/árboles.

**Frase de examen:**

> “Encapsulo preprocesado en un ColumnTransformer para aplicarlo consistentemente y evitar fuga (fit solo en train).”

---

### 6) Gráficas mínimas que suelen pedir (elige 2–4 según el tiempo)

* **Distribución de clases** (bar) → siempre.
* **Histograma** de una numérica relevante por clase (Age, etc.).
* **Boxplot** de una variable potente por clase (MMSE, Na_to_K, etc.).
* (Opcional) correlación solo de numéricas.

**Frase de examen:**

> “Visualizo distribución de clases y separabilidad de variables clave.”

---

## Modelos: guion “multiclase” y “binario”

### A) Si el target es multiclase (Drug / Dementia con 3 clases)

**Tu MLP propio (NumPy):**

* Salida **softmax**
* Loss: **cross-entropy**
* Requisito típico: probar **>1 capa oculta** (aunque luego elijas otra)

**Evaluación:**

* `accuracy`
* `confusion_matrix`

**Frase de examen:**

> “Uso softmax+cross-entropy para multiclase, y reporto matriz de confusión para ver en qué clases falla.”

**Sklearn MLPClassifier:**

* Suele funcionar muy bien con `solver="lbfgs"` en datasets pequeños tabulares.
* O `adam + early_stopping=True` si te dejan.

**KNN:**

* Siempre con escalado (por eso el pipeline).
* Prueba varios `k` y te quedas con el mejor en test (en examen: 5–10 valores típicos).

---

### B) Si el target es binario (HeartDisease / softDrug vs hardDrug / Converted→Demented)

**Conversión de etiqueta (muy típico en examen):**

* `y_bin = (y in clases_positivas).astype(int)`

**MLP propio (NumPy):**

* Salida **sigmoid (1 neurona)**
* Loss: **binary cross-entropy**
* Métrica: accuracy + confusion matrix.

**Frase de examen:**

> “Para binario uso 1 neurona con sigmoid y BCE; la matriz de confusión me permite controlar falsos negativos.”

---

## Comparación y “texto de justificación” (lo que escribes al final)

### Qué mirar (en este orden)

1. **Accuracy**
2. **Confusion matrix** (¿en qué se equivoca?)
3. Si es médico: fíjate en **FN** (falsos negativos) / **recall** de clase enferma.

### Texto plantilla (cópialo tal cual y ajusta una frase)

> “Comparo modelos por accuracy y por matriz de confusión. Me quedo con ____ porque obtiene el mejor equilibrio entre rendimiento y tipo de error. En un contexto médico priorizo reducir falsos negativos (no dejar enfermos sin detectar), por lo que el recall de la clase positiva y los FN son especialmente relevantes.”

---

## Micro-chuleta de “qué hacer si algo va mal”

* **Train alto, test bajo** → sobreajuste: menos capas/neuronas, más regularización (L2), early stopping.
* **Train bajo y test bajo** → underfitting: más capacidad, más epochs, ajustar lr.
* **Accuracy por debajo del baseline** → revisa:

  * split estratificado,
  * encoding correcto de y,
  * preprocesado fit SOLO en train,
  * categorías normalizadas (mayúsculas/espacios),
  * valores imposibles tratados.

---

## Plantilla de código mínima (para copiar en examen)

*(En pseudocódigo estilo sklearn, para que te encaje con cualquier dataset)*

```python
df = pd.read_csv(...)
df.info(); df.isna().sum()

# 1) drop IDs / constantes
# 2) normalizar strings (strip/lower/upper)
# 3) valores imposibles -> NaN

X = df.drop(columns=[target])
y = df[target]

cat_cols = [...]
num_cols = [...]

pre = ColumnTransformer([
  ("num", Pipeline([("imp", SimpleImputer("median")), ("sc", StandardScaler())]), num_cols),
  ("cat", Pipeline([("imp", SimpleImputer("most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=..., random_state=..., stratify=y)

# Modelo (ejemplo)
pipe = Pipeline([("pre", pre), ("model", ...)])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
print(accuracy_score(yte, pred))
ConfusionMatrixDisplay.from_predictions(yte, pred)
```