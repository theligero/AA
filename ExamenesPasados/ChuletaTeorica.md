## Técnica rápida para contestar teoría (en 30–45 min)

### 1) Clasifica la pregunta en 10 segundos

Casi todas encajan en una de estas cajas (y si sabes la caja, sabes dónde buscar):

* **Evaluación / métricas** (matriz de confusión, precision/recall/F1, ROC/PR, AUC)
* **Generalización** (train vs val vs baseline, bias/varianza, overfitting) 
* **Pocos datos / mejorar resultados** (data augmentation, transfer learning, regularización, dropout)
* **No supervisado / sin etiquetas** (PCA, clustering, k-means, silhouette) 
* **Deep Learning “qué red es y para qué sirve”** (autoencoder, CNN)
* **Aprendizaje por refuerzo / agente** (MDP, Q-learning, política, recompensa)

Esto casa 1:1 con preguntas reales: autoencoder en junio , RL del agente al cofre , train/val/baseline , confusión con matrices , etc.

---

## 2) Usa una plantilla de respuesta que siempre “suena a examen”

Para casi cualquier pregunta teórica, responde en este orden (5–8 líneas, sin enrollarte):

1. **Qué problema es** (clasificación/regresión/no supervisado/RL, balanceado o no, etc.).
2. **Concepto clave + definición corta** (1 frase).
3. **2–3 argumentos técnicos** (qué harías / por qué funciona).
4. **Riesgo o limitación** (overfitting, clases desbalanceadas, fuga de info…).
5. **Cierre**: “por eso elegiría X frente a Y”.

Ejemplo mini-plantilla (para tenerla impresa):

* “Esto es **X** porque …”
* “La métrica/idea clave es **Y**: …”
* “Haría **A, B, C** porque …”
* “Ojo con **D**; lo mitigaría con **E**.”

---

## 3) “Mapa Ctrl+F”: qué palabra buscar en qué tema

Si te pierdes, **no navegues**, **busca** con estas keywords (funciona muy bien con PDFs):

| Si la pregunta menciona…                        | Ctrl+F                                           | Tema/sección útil             |
| ----------------------------------------------- | ------------------------------------------------ | ----------------------------- |
| matriz de confusión, TP/FP, precision/recall/F1 | `Confussion Matrix`, `Precision`, `Recall`, `F1` | Tema 04 (evaluación)          |
| ROC, AUC, umbral                                | `ROC`, `AUC`, `umbral`                           | Tema 04 (ROC/PR)              |
| clases desbalanceadas                           | `desbalance` / `PR`                              | Diferencias ROC vs PR         |
| train vs val, baseline, mejorar resultados      | `baseline`, `Bias`, `varianza`                   | Tema 04 (bias/var, baseline)  |
| pocos datos                                     | `data aumentation`, `Transfer Learning`          | Tema 04 (4.5–4.6)             |
| regularización                                  | `Regularization`, `L2`                           | Tema 03 (regularización)      |
| dropout                                         | `Dropout`                                        | Tema 07 (dropout)             |
| normalizar/escala                               | `Normalización`, `MinMax`, `z-Score`             | Tema 03 (normalización)       |
| one-hot / categóricas / imputación              | `One-hot`, `Imputer`                             | Tema 04 (preprocesado)        |
| datos sin etiqueta                              | `no supervisado`, `clustering`, `PCA`            | Tema 06                       |
| autoencoder / comprimir sprites                 | `autoencoder`, `encoder`, `decoder`              | Tema 07 (AE)                  |
| imágenes grandes / demasiados píxeles           | `Convolutional` / `convolucional`                | Tema 07 (CNN)                 |
| agente, cofre, política, recompensa             | `Markov`, `Q-Learning`, `política`, `recompensa` | Tema 08 (RL)                  |

---

## Cómo “blindarte” de verdad: prepara tus diapos para responder solas

Como puedes llevar apuntes, lo más rentable es:

* **Imprimir 1 hoja** con el “Mapa Ctrl+F” de arriba (literal).
* Poner **post-its/índice por pestañas** en el PDF (o en papel) con 6 pestañas:
  **Evaluación**, **Bias/Var**, **Pocos datos**, **No supervisado**, **Deep Learning**, **Refuerzo**.
* En cada pestaña, añade 3 bullets “de respuesta” (no teoría):

  * *Qué es* / *cuándo usar* / *cómo justificar*.

Con eso, aunque estés perdido, puedes contestar casi todo porque las preguntas recientes han ido siempre por ahí: imágenes+MLP , autoencoders , RL del agente , métricas y matrices , etc.

---