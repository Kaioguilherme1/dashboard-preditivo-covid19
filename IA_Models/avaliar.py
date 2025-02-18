import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, classification_report
)
import numpy as np
from data_train import X_test, y_test


def avaliar_modelo(modelo, y_pred):
    """
    Avaliação do modelo para problemas de classificação e regressão.
    Fornece métricas, relatório de classificação e gráficos de avaliação.

    Parâmetros:
    - modelo: o modelo treinado (ex.: LinearRegression, RandomForestClassifier, etc.)
    - y_pred: valores previstos pelo modelo
    """
    print("======= Avaliação do Modelo =======\n")

    # Identificar o tipo de problema (classificação ou regressão)
    if len(np.unique(y_test)) <= 10 and y_test.dtype in [int, np.int64, np.int32]:
        # SUPÕE-SE CLASSIFICAÇÃO SE HOUVER MENOS DE 10 CLASSES DISTINTAS
        print("Modelo Identificado: Classificação\n")

        # Acurácia
        acuracia = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {acuracia:.4f}")

        # Relatório de Classificação
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))

        # Matriz de Confusão
        plt.figure(figsize=(6, 5))
        matriz_confusao = confusion_matrix(y_test, y_pred)
        sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.show()

        # Detalhes do Modelo (coeficientes e scores para interpretabilidade):
        try:
            if hasattr(modelo, "coef_"):
                print("\nCoeficientes do Modelo:")
                print(modelo.coef_)
            if hasattr(modelo, "feature_importances_"):
                print("\nImportância das Features:")
                print(modelo.feature_importances_)
        except Exception as e:
            print("Não foi possível exibir coeficientes ou importâncias das variáveis.")

        print("\n======= Avaliação Finalizada =======")
