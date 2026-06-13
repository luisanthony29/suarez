import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
 
# ── DATASET SIMULADO (array bidimensional en el código) ──
# Cada fila = un dígito representado en 8x8 píxeles (64 valores, escala 0-16)
 
X = np.array([
    # Dígito 0
    [ 0, 0, 5,12,13, 3, 0, 0,  0, 4,15, 6, 6,14, 3, 0,
      0, 9,12, 0, 0,10, 7, 0,  0,10, 8, 0, 0,10, 8, 0,
      0,10, 8, 0, 0,10, 8, 0,  0, 9,12, 0, 0,12, 6, 0,
      0, 3,14, 7, 5,15, 3, 0,  0, 0, 5,13,13, 4, 0, 0],
    [ 0, 0, 6,11,14, 4, 0, 0,  0, 3,14, 7, 5,13, 2, 0,
      0, 8,13, 0, 0,11, 6, 0,  0,11, 7, 0, 0,11, 7, 0,
      0,11, 7, 0, 0,11, 7, 0,  0, 8,13, 0, 0,13, 5, 0,
      0, 2,13, 6, 6,14, 2, 0,  0, 0, 6,12,14, 3, 0, 0],
 
    # Dígito 1
    [ 0, 0, 0, 5,10, 0, 0, 0,  0, 0, 3,13,11, 0, 0, 0,
      0, 0, 8,14, 9, 0, 0, 0,  0, 0, 5,16, 8, 0, 0, 0,
      0, 0, 0,16, 6, 0, 0, 0,  0, 0, 0,16, 5, 0, 0, 0,
      0, 0, 0,16, 8, 0, 0, 0,  0, 0, 6,16,16, 5, 0, 0],
    [ 0, 0, 0, 4,11, 0, 0, 0,  0, 0, 2,14,10, 0, 0, 0,
      0, 0, 7,15, 8, 0, 0, 0,  0, 0, 4,16, 7, 0, 0, 0,
      0, 0, 0,16, 5, 0, 0, 0,  0, 0, 0,16, 4, 0, 0, 0,
      0, 0, 0,16, 7, 0, 0, 0,  0, 0, 5,16,15, 4, 0, 0],
 
    # Dígito 2
    [ 0, 0, 4,12,13, 5, 0, 0,  0, 3,14, 4, 5,14, 2, 0,
      0, 0, 0, 0, 6,14, 0, 0,  0, 0, 0, 4,14, 5, 0, 0,
      0, 0, 3,14, 6, 0, 0, 0,  0, 3,14, 5, 0, 0, 0, 0,
      0, 9,15, 8,10,12, 5, 0,  0, 1, 4, 4, 4, 3, 0, 0],
    [ 0, 0, 3,13,12, 4, 0, 0,  0, 2,13, 5, 4,13, 1, 0,
      0, 0, 0, 0, 5,13, 0, 0,  0, 0, 0, 3,13, 4, 0, 0,
      0, 0, 2,13, 5, 0, 0, 0,  0, 2,13, 4, 0, 0, 0, 0,
      0, 8,14, 7, 9,11, 4, 0,  0, 1, 3, 3, 3, 2, 0, 0],
 
    # Dígito 3
    [ 0, 0, 5,12,11, 2, 0, 0,  0, 2,13, 3, 9,11, 0, 0,
      0, 0, 0, 0, 8,12, 0, 0,  0, 0, 3,10,15, 5, 0, 0,
      0, 0, 0, 2,10,13, 0, 0,  0, 0, 0, 0, 5,14, 0, 0,
      0, 3,11, 3, 8,13, 0, 0,  0, 0, 5,13,11, 3, 0, 0],
    [ 0, 0, 4,11,10, 1, 0, 0,  0, 1,12, 2, 8,10, 0, 0,
      0, 0, 0, 0, 7,11, 0, 0,  0, 0, 2, 9,14, 4, 0, 0,
      0, 0, 0, 1, 9,12, 0, 0,  0, 0, 0, 0, 4,13, 0, 0,
      0, 2,10, 2, 7,12, 0, 0,  0, 0, 4,12,10, 2, 0, 0],
 
    # Dígito 4
    [ 0, 0, 0, 3,11, 8, 0, 0,  0, 0, 3,13,16, 8, 0, 0,
      0, 3,12, 4,16, 7, 0, 0,  0, 9,12, 0,16, 7, 0, 0,
      0,12,16,16,16,14, 3, 0,  0, 1, 3, 2,16, 7, 0, 0,
      0, 0, 0, 3,16, 7, 0, 0,  0, 0, 0, 2,13, 5, 0, 0],
    [ 0, 0, 0, 2,10, 7, 0, 0,  0, 0, 2,12,15, 7, 0, 0,
      0, 2,11, 3,15, 6, 0, 0,  0, 8,11, 0,15, 6, 0, 0,
      0,11,15,15,15,13, 2, 0,  0, 0, 2, 1,15, 6, 0, 0,
      0, 0, 0, 2,15, 6, 0, 0,  0, 0, 0, 1,12, 4, 0, 0],
 
    # Dígito 5
    [ 0, 2,12,16,14, 4, 0, 0,  0, 5,13, 4, 1, 0, 0, 0,
      0, 8,14, 2, 0, 0, 0, 0,  0, 5,16,13, 5, 0, 0, 0,
      0, 0, 3, 7,14, 6, 0, 0,  0, 0, 0, 0, 8,12, 0, 0,
      0, 4, 9, 4, 8,13, 0, 0,  0, 1,10,16,14, 4, 0, 0],
    [ 0, 1,11,15,13, 3, 0, 0,  0, 4,12, 3, 0, 0, 0, 0,
      0, 7,13, 1, 0, 0, 0, 0,  0, 4,15,12, 4, 0, 0, 0,
      0, 0, 2, 6,13, 5, 0, 0,  0, 0, 0, 0, 7,11, 0, 0,
      0, 3, 8, 3, 7,12, 0, 0,  0, 0, 9,15,13, 3, 0, 0],
 
    # Dígito 6
    [ 0, 0, 4,12,14, 5, 0, 0,  0, 3,14, 8, 3, 0, 0, 0,
      0, 8,13, 0, 0, 0, 0, 0,  0,12,16,14, 7, 0, 0, 0,
      0,12,10, 3,13, 9, 0, 0,  0,10, 8, 0, 4,14, 0, 0,
      0, 6,14, 5, 8,12, 0, 0,  0, 0, 5,14,14, 4, 0, 0],
    [ 0, 0, 3,11,13, 4, 0, 0,  0, 2,13, 7, 2, 0, 0, 0,
      0, 7,12, 0, 0, 0, 0, 0,  0,11,15,13, 6, 0, 0, 0,
      0,11, 9, 2,12, 8, 0, 0,  0, 9, 7, 0, 3,13, 0, 0,
      0, 5,13, 4, 7,11, 0, 0,  0, 0, 4,13,13, 3, 0, 0],
 
    # Dígito 7
    [ 0, 4,16,16,16,14, 3, 0,  0, 1, 4, 3, 8,15, 2, 0,
      0, 0, 0, 0, 9,13, 0, 0,  0, 0, 0, 4,15, 5, 0, 0,
      0, 0, 0,11,12, 0, 0, 0,  0, 0, 3,15, 5, 0, 0, 0,
      0, 0, 8,14, 1, 0, 0, 0,  0, 0,11,11, 0, 0, 0, 0],
    [ 0, 3,15,15,15,13, 2, 0,  0, 0, 3, 2, 7,14, 1, 0,
      0, 0, 0, 0, 8,12, 0, 0,  0, 0, 0, 3,14, 4, 0, 0,
      0, 0, 0,10,11, 0, 0, 0,  0, 0, 2,14, 4, 0, 0, 0,
      0, 0, 7,13, 0, 0, 0, 0,  0, 0,10,10, 0, 0, 0, 0],
 
    # Dígito 8
    [ 0, 0, 5,12,12, 5, 0, 0,  0, 4,14, 4, 4,13, 3, 0,
      0, 7,12, 0, 0,11, 7, 0,  0, 3,13,11, 9,14, 2, 0,
      0, 3,13, 7, 5,15, 3, 0,  0, 7,11, 0, 0,11, 8, 0,
      0, 3,14, 5, 5,14, 4, 0,  0, 0, 5,12,12, 5, 0, 0],
    [ 0, 0, 4,11,11, 4, 0, 0,  0, 3,13, 3, 3,12, 2, 0,
      0, 6,11, 0, 0,10, 6, 0,  0, 2,12,10, 8,13, 1, 0,
      0, 2,12, 6, 4,14, 2, 0,  0, 6,10, 0, 0,10, 7, 0,
      0, 2,13, 4, 4,13, 3, 0,  0, 0, 4,11,11, 4, 0, 0],
 
    # Dígito 9
    [ 0, 0, 5,13,13, 4, 0, 0,  0, 4,15, 5, 6,14, 3, 0,
      0, 9,11, 0, 0,12, 7, 0,  0, 9,12, 4, 8,16, 4, 0,
      0, 1, 6,12,14,16, 3, 0,  0, 0, 0, 0, 4,15, 1, 0,
      0, 2, 8, 6, 9,13, 0, 0,  0, 0, 5,14,13, 4, 0, 0],
    [ 0, 0, 4,12,12, 3, 0, 0,  0, 3,14, 4, 5,13, 2, 0,
      0, 8,10, 0, 0,11, 6, 0,  0, 8,11, 3, 7,15, 3, 0,
      0, 0, 5,11,13,15, 2, 0,  0, 0, 0, 0, 3,14, 0, 0,
      0, 1, 7, 5, 8,12, 0, 0,  0, 0, 4,13,12, 3, 0, 0],
])
 
# Etiquetas: 2 muestras por cada dígito (0-9)
y = np.array([0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9])
 
print("=== DATASET SIMULADO ===")
print(f"Shape X: {X.shape}  →  {X.shape[0]} imágenes de {X.shape[1]} píxeles (8x8)")
print(f"Shape y: {y.shape}  →  Clases: {np.unique(y).tolist()}")
 
# ── ENTRENAMIENTO Y PREDICCIÓN ──
# Train: primeras muestras | Test: segundas muestras
X_train, y_train = X[0::2], y[0::2]   # índices pares
X_test,  y_test  = X[1::2], y[1::2]   # índices impares
 
modelo = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
 
print("\n=== RESULTADOS ===")
for real, pred in zip(y_test, y_pred):
    estado = "✓ CORRECTO" if real == pred else "✗ ERROR"
    print(f"  Real: {real}  →  Predicho: {pred}  {estado}")
 
print(f"\nPrecisión: {accuracy_score(y_test, y_pred):.0%}")
 
# ── GRÁFICA 1: Dataset ──
fig, axes = plt.subplots(2, 10, figsize=(14, 3))
fig.suptitle("Dataset Simulado — Dígitos 0 al 9 (2 muestras c/u)", fontweight='bold')
for i in range(10):
    axes[0, i].imshow(X_train[i].reshape(8,8), cmap='gray_r', vmin=0, vmax=16)
    axes[0, i].set_title(f'{i}', fontsize=10)
    axes[0, i].axis('off')
    axes[1, i].imshow(X_test[i].reshape(8,8), cmap='gray_r', vmin=0, vmax=16)
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()
 
# ── GRÁFICA 2: Resultados de predicción ──
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle(f"Resultados del Clasificador KNN — Precisión: {accuracy_score(y_test,y_pred):.0%}",
             fontsize=13, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8,8), cmap='hot', vmin=0, vmax=16)
    ok = y_test[i] == y_pred[i]
    ax.set_title(f"Real:{y_test[i]}  Pred:{y_pred[i]}  {'✓' if ok else '✗'}",
                 color='green' if ok else 'red', fontsize=9, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.show()
 
# ── GRÁFICA 3: Matriz de confusión ──
cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=11,
                color='white' if cm[i,j] > 0 else 'lightgray', fontweight='bold')
ax.set_title("Matriz de Confusión", fontsize=13, fontweight='bold')
ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
plt.tight_layout()
plt.show()
