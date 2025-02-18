import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_train import X_train, X_test, y_train, y_test
import matplotlib.pyplot as plt
import seaborn as sns


# Criar o modelo de Regressão Linear
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# Avaliar o modelo

print("Modelo Identificado: Regressão\n")

# Métricas de Regressão
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f}")
print(f"Erro Absoluto Médio (MAE): {mae:.4f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Gráfico: Valores Reais vs Previstos
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Previsões')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal')
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Regressão - Valores Reais vs Previstos")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico: Resíduos
residuos = y_test - y_pred
plt.figure(figsize=(7, 4))
sns.histplot(residuos, kde=True, bins=25, color='purple')
plt.xlabel("Resíduos (y_real - y_previsto)")
plt.title("Distribuição dos Resíduos")
plt.grid(True)
plt.show()