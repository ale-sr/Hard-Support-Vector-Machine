# -*- coding: utf-8 -*-
"""

# Multiplicadores de Lagrange 
$\frac{\partial f(x)}{ \partial x} = λ \frac{g( \partial x)}{ \partial x}$

Hallar los valores de $λ_i$ para cada elemento de entrenamiento $X_i$. 

El código ***GetLambda***  debe retornar un vector al cual denominaremos lambda, de modo que
  $lambda[i]$ será $0$, si el elemento $X[i]$ no tiene intercesión con ninguna de las rectas
  $XW^t + b >=1$ o $XW^t + b >=0$

- **Nota: Puede buscar en internet la forma de como hallar lambda.**
"""

import numpy as np 
import math
import pandas as pd
import random
import matplotlib.pyplot as plt 
import cvxopt
import seaborn as sns
import itertools

from sklearn.metrics import confusion_matrix

zeta = 0
  gamma = 1
  Q = 1
  C = 0.1

def compare(y1, y2):
  return sum([y1[i]==y2[i] for i in range(len(y1))])

def polynomial_kernel(x1, x2):
    return (zeta + gamma * np.dot(x1, x2)) ** Q

def kernel_matrix(X):
  K = (zeta + gamma * np.dot(X, X.T)) ** Q
  K = np.zeros((n, n))
  for i in range(n):
      for j in range(n):
        K[i, j] = polynomial_kernel(X[i], X[j])
  return K

def GetLambda(X, Y):
  K = kernel_matrix(X)
  Y = Y.astype(np.double)
  P = cvxopt.matrix(np.outer(Y, Y) * K)
  q = cvxopt.matrix(np.ones(n) * -1)
  constrain1 = np.diag(np.ones(n) * -1)
  constrain2 = np.identity(n)
  G = cvxopt.matrix(np.vstack((constrain1, constrain2)))
  constrain1 = np.zeros(n)
  constrain2 = np.ones(n) * C
  h = cvxopt.matrix(np.hstack((constrain1, constrain2)))
  A = cvxopt.matrix(Y, (1, n))
  b = cvxopt.matrix(0.0)
  cvxopt.solvers.options['show_progress'] = False
  solution = cvxopt.solvers.qp(P, q, G, h, A, b)
  return np.ravel(solution['x'])

"""## 2 Cálculo de los pesos W
$W_j = \sum_{i=0}^n \lambda_iy_ix_{ij}$  

Donde: λ_i es el i-esimo multiplicador de lagrange, W_j es el W-esimo peso y x_{ij} es el valor de la característica $j$ del objeto de entrenamiento $i-esimo$ y $y_i$ es la salida esperada (1 o - 1) del objeto $i$.

Recuerde la sumatoria solo recorre todos los elementos para los cuales el valor del multiplicador de lagrange $λ_i$ es diferente de 0.
"""

def Get_W(X,Y, lambda_arr):
  # write your code here
  W = []
  for j in range(k):
    W.append(sum([lambda_arr[i]*Y[i]*X[i][j] for i in range(n)]))
  return np.array(W)

"""## Cálculo de b

XW^t + b = 0 

$b = - ∑_{i=0}^n X_iW^t$

Donde $X_i$ es un vector $k$ dimensional y representa el objeto $i-esimo$ de entrenamiento y $k$ el número de características del objeto.
"""

def Get_b(X,W):
  # write your code here
  return -sum([np.dot(X[i], W.transpose()) for i in range(n)])/n

# training
def training(X, Y):
  # write your code here
  lambda_arr = GetLambda(X, Y)
  W = Get_W(X, Y, lambda_arr)
  b = Get_b(X, W)
  return np.array([W, b])

def loss_function(x, y, w, c, b):
  return w**2/2 + c*sum([max(0, 1-y[i]*(np.dot(x[i], w.transpose()) + b)) for i in range(y.size)])

def change_parameters(w, b, db, dw, alpha):
  for j in range(k):
    w[j] = w[j] - alpha*dw[j]
  b = b - alpha*db
  return w, b

def training_soft(x, y, c, alpha, epochs, xt = None, yt = None):
  w = np.array([np.random.rand() for i in range(k)])
  b = np.random.rand()
  errtr = []
  errts = []
  cont = 0
  while cont < epochs:
    dw = np.array([])
    for i in range(n):
      if y[i]*(np.dot(x[i], w.transpose()) + b) < 1:
        for j in range(k):
          dw = np.append(dw, -y[i]*x[i][j] + w[j])
        db = -y[i]
      else:
        dw = w
        db = 0
      w, b = change_parameters(w, b, db, dw, alpha)
    L = loss_function(x, y, w, c, b)
    errtr.append(L)
    if xt.any() and yt.any():
      errts.append(loss_function(xt, yt, w, c, b))
    cont += 1
  return w, b, errtr, errts

"""## Etapa de Testing

Para esta estapa solo se debe calcular

$f(X_j) = X_jW^t + b$

Pero dado que ya hemos calculado el valor de los parámetros $W$ y $b$, entonces remplazando tenemos

$f(X_j) = \sum_{i=0}^n \lambda_iy_i<X_{i},X_{j}> + b$

Donde: $X_i$ i-esimo  es el vector de entrenamiento y $X_j$ es el nuevo vector que pasa por el modelo para su predicciòn predecir la clase (1 o -1).

Finalmente para saber a que clase pertenece el nuevo vector $X_j$ vasta con verificar el signo de f(X_j). 

  - **If $f(X_j) >=0$ then $Y_j$ = 1 else $Y_j = -1$**
"""

def testing(X,W,b):
  Y_result = []
  # write your code here
  for j in range(m):
    f_xj = np.dot(X[j], W.transpose())+b
    if f_xj >= 0:
      Y_result.append(1)
    else:
      Y_result.append(-1)
  return np.array(Y_result)

def testing_2(X,Y,W,b):
  Y_result = []
  # write your code here
  for j in range(m):
    f_xj = Y[j]*(np.dot(X[j], W.transpose())+b)
    if f_xj >= 0:
      Y_result.append(1)
    else:
      Y_result.append(-1)
  return np.array(Y_result)

"""Base de Datos para Las pruebas:
[Download](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwihhZv-9JH3AhWUI7kGHZWDBV4QFnoECEMQAQ&url=http%3A%2F%2Fwww.saedsayad.com%2Fdatasets%2FIris.xls&usg=AOvVaw3HOrA0X468Juw2u4WM-YvO)

En esta base de datos existen 3 clases, solo utilize dos clases para hacer las pruebas.

- Separe el dataset en 70% para entrenar y 30% para hacer las pruebas

- Añada un valor 1 para la primera clase  y  -1 para la segunda clase.

- En la etapa de test, encuentre el número de elementos correctamente clasificados y el número de elementos incorrectamente clasificados para cada clase.

- Cree una matriz de confusión el cual nos mostrará la eficiencia del método.
"""

def normalizacion(aux_x, aux_y):
  n = aux_y.size
  k = len(aux_x.columns)

  x = np.array([[i*1.0 for i in range(k)] for j in range(n)])

  for i in range(n):
    lista = []
    for j in range(k):
      lista.append(aux_x.iloc[i][aux_x.columns[j]])
    lista = np.array(lista)
    x[i] = lista

  mat = [[i*1.0 for i in range(n)] for j in range(k)]
  for i in range(k):
    col = x[:,i]
    max_x = max(col)
    min_x = min(col)
    lista = []
    for e in col:
      lista.append( (e-min_x) / (max_x - min_x) )
    lista = np.array(lista)
    mat[i] = lista

  x_norm = np.vstack((mat)).T
  return x_norm

"""# Hard SVM todas las combinatorias Iris"""

df = pd.read_csv('iris.csv')

clases = df['variety'].unique()

for comb in itertools.combinations(clases, 2):
  print("1:", comb[0])
  print("-1:", comb[1])

  training_data = df.sample(frac=0.7, random_state=25)
  testing_data = df.drop(training_data.index)

  training_data['variety'] = training_data['variety'].replace(comb[0], 1)
  training_data['variety'] = training_data['variety'].replace(comb[1], -1)

  testing_data['variety'] = testing_data['variety'].replace(comb[0], 1)
  testing_data['variety'] = testing_data['variety'].replace(comb[1], -1)

  # 2 variedades
  variedades_nombre = [comb[0], comb[1]]
  variedades = [1, -1]

  # No se normaliza "y" porque es un string
  training_data = training_data.loc[training_data['variety'].isin(variedades)]
  train_x = training_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
  train_y = np.vstack(training_data[['variety']].to_numpy()).T[0]

  testing_data = testing_data.loc[testing_data['variety'].isin(variedades)]
  test_x = testing_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
  test_y = np.vstack(testing_data[['variety']].to_numpy()).T[0]

  n = train_y.size
  k = len(train_x.columns)

  train_x_norm = normalizacion(train_x, train_y)
  test_x_norm = normalizacion(test_x, test_y)

  W, b = training(train_x_norm, train_y)

  m = test_y.size
  y_pred = testing(test_x_norm, W, b)

  plt.plot(y_pred, 'purple', linewidth=2)
  plt.plot(test_y, "bo")
  plt.legend(["Pred", "Real"])
  plt.show()

  correct = compare(test_y, y_pred)
  print("Clasificados correctamente:", correct)
  print("Clasificados incorrectamente:", len(test_y)-correct)
  print("% de efectividad", round(100*correct/len(test_y), 2))

  matrix = confusion_matrix(test_y.tolist(), y_pred.tolist())
  df2 = pd.DataFrame(matrix, index=variedades_nombre, columns=variedades_nombre)
  sns.heatmap(df2, annot=True, cbar=None, cmap="Greens")
  plt.title("Confusion Matrix"), plt.tight_layout()
  plt.xlabel("Pred")
  plt.ylabel("Real")
  plt.show()