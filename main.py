
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

n_samples = 100
X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=240) #40, 240, 540, 620

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Визуализация данных с прямой, разделяющей классы
plt.figure(figsize=(8, 6))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, marker='o', label='Train Data')
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', label='Test Data')

# Построение прямой разделяющей данные
coef = model.coef_[0]
intercept = model.intercept_

x_vals = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
y_vals = -(intercept + coef[0]*x_vals) / coef[1]
plt.plot(x_vals, y_vals, color='black', linestyle='-', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Train Data with Decision Boundary')
plt.legend()

plt.show()
