from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from data_train import X_train, X_test, y_train, y_test

# 1. Aplicar o SMOTE para balancear os dados
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 2. Definir o modelo
modelo = RandomForestClassifier(random_state=42)

# 3. Melhorar as faixas de parâmetros para RandomizedSearchCV
param_dist = {
    "n_estimators": [50, 100, 200, 300, 400, 500, 750, 1000, 1200],  # Número de árvores na floresta
    "max_depth": [None, 10, 20, 30, 40, 50, 70, 100],  # Profundidade máxima das árvores
    "min_samples_split": [2, 5, 10, 15, 20],  # Número mínimo de amostras para dividir um nó
    "min_samples_leaf": [1, 2, 4, 8, 16],  # Número mínimo de amostras em cada nó folha
    "max_features": ["sqrt", "log2", None],  # Número máximo de features a considerar
    "bootstrap": [True, False],  # Amostragem com reposição ou sem
    "class_weight": [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 4}]  # Ajuste de peso para dados desbalanceados
}

# 4. Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    modelo,
    param_distributions=param_dist,
    n_iter=50,  # Número de combinações aleatórias a serem testadas
    cv=5,  # Validação cruzada com 5 dobras
    scoring="accuracy",  # Métrica para avaliar cada modelo
    verbose=1,  # Nível de detalhamento no console
    random_state=42,
    n_jobs=-1  # Paralelismo para acelerar a busca
)

# 5. Realizar a busca pelos melhores hiperparâmetros
print("Otimizando os hiperparâmetros...")
random_search.fit(X_resampled, y_resampled)

# 6. Melhor modelo encontrado
print(f"\nMelhor combinação de parâmetros: {random_search.best_params_}")
best_model = random_search.best_estimator_

# Avaliar o melhor modelo nos dados de teste
y_pred = best_model.predict(X_test)

# 7. Avaliação do modelo
print("\n=== Avaliação do Modelo ===")
print(f"Acurácia no conjunto de teste: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()