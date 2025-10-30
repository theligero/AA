import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights,one_hot_encoding, accuracy
from MLP import MLP
from public_test import compute_cost_test, predict_test

# Carga de datos y pesos
x,y = load_data('data/ex3data1.mat')
theta1, theta2 = load_weights('data/ex3weights.mat')

# Construcción del MLP y forward
mlp = MLP(theta1, theta2)
a1, a2, a3, z2, z3 = mlp.feedforward(x)

# Predicción y métricas básicas
p = mlp.predict(a3)
acc = accuracy(p, y)
print(f"Accuracy: {acc:.4f}")

# Coste (entropía cruzada)
y_one_hot = one_hot_encoding(y)
J = mlp.compute_cost(a3, y_one_hot)
print(f"Cost: {J:.12f}")

# Tests públicos
predict_test(p, y, accuracy)
compute_cost_test(mlp, a3, y_one_hot)

# Matriz de confusión y métricas para la clase 0
pos = 0
y_flat = y.reshape(-1)

TP = np.sum((p == pos) & (y_flat == pos))
FP = np.sum((p == pos) & (y_flat != pos))
TN = np.sum((p != pos) & (y_flat == pos))
FN = np.sum((p != pos) & (y_flat == pos))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print("\nConfusion matrix for class 0 (positive):")
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")