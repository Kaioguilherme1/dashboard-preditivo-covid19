import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
data = pd.read_csv('../datasets/Brasil-2022-processado.csv')
print(f'Dataset Bruto shape: {data.shape}')

# Colunas que não serão usadas
colunas_drop = [
    'sintomas', 'profissionalSaude', 'racaCor', 'outrosSintomas', 'municipioNotificacao',
    'classificacaoFinal', 'dataNotificacao', 'dataInicioSintomas', 'dataEncerramento',
    'outrasCondicoes', 'profissionalSeguranca', 'condicoes', 'estado', 'municipio', 'estadoNotificacao'
]

# Remover colunas desnecessárias
dados = data.drop(columns=colunas_drop)

# Garantir tipos de dados corretos
dados['tomouPrimeiraDose'] = dados['tomouPrimeiraDose'].astype(int, errors='ignore')
dados['tomouSegundaDose'] = dados['tomouSegundaDose'].astype(int, errors='ignore')
dados['idade'] = dados['idade'].astype(int, errors='ignore')

# Remover duplicatas e valores ausentes para garantir consistência
dados = dados.dropna()
print(f'Dataset após remoção de NaN: {dados.shape}')

# - Dividindo o DataFrame entre colunas antes de "diagnosticoCOVID" e após
try:
    idx_diagnostico = list(dados.columns).index('diagnosticoCOVID')
except ValueError:
    raise ValueError("A coluna 'diagnosticoCOVID' não foi encontrada no DataFrame.")

# Colunas antes e incluindo "diagnosticoCOVID"
colunas_antes = dados.columns[:idx_diagnostico + 1]

# Colunas após "diagnosticoCOVID"
colunas_apos_diagnostico = dados.columns[idx_diagnostico + 1:]

# Filtrar apenas as colunas posteriores a "diagnosticoCOVID" que são binárias (com 2 valores únicos)
colunas_binarias = colunas_apos_diagnostico[dados[colunas_apos_diagnostico].nunique() == 2]

# Garantir que todas as colunas binárias contenham apenas valores numéricos
dados[colunas_binarias] = dados[colunas_binarias].apply(pd.to_numeric, errors='coerce')

# Calcular a soma (frequência de 1s) para cada coluna binária
frequencias = dados[colunas_binarias].sum().to_numpy()

# Garantir que frequencias contenham valores válidos e numéricos
frequencias = np.array(frequencias, dtype=float)

# Ordenar as colunas pela frequência em ordem decrescente
indices_ordenados = np.argsort(-frequencias)  # "-" para inversão (ordem decrescente)
colunas_ordenadas = colunas_binarias[indices_ordenados]  # Aplicar os índices nas colunas

# Selecionar os 20% mais frequentes
top_20_porcento = max(1, int(len(colunas_ordenadas) * 0.20))  # Garantir ao menos 1 coluna
colunas_selecionadas = colunas_ordenadas[:top_20_porcento]
# Filtrar o dataframe com as colunas selecionadas
dados_filtrados = dados[colunas_selecionadas]

# Combine o DataFrame final: antes da coluna "diagnosticoCOVID" + colunas selecionadas
dados_final = pd.concat([dados[colunas_antes], dados_filtrados], axis=1)

print(f"Dataset Final shape: {dados_final.shape}")
print(f'Numero de colunas selecionadas: {len(colunas_selecionadas)}')
print(dados_final.head())

# Codificar "sexo" para valores numéricos com LabelEncoder
if 'sexo' in dados.columns:
    label_encoder = LabelEncoder()
    dados_final['sexo'] = label_encoder.fit_transform(dados_final['sexo'])

# Separar variáveis independentes (Features) e a variável alvo (Label)
Features = dados_final.drop(columns=['diagnosticoCOVID'])  # Remove a variável alvo das Features
Label = dados_final['diagnosticoCOVID']

# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(Features, Label, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# tabela de porcentagem / acuracia = dataset 2021
# 10# / 0.7512
# 20% / 0.7761
# 30% / 0.7711
# 40% / 0.7745
# 50% / 0.7778
# 60% / 0.7761
# 70% / 0.7662
# 80% / 0.7761
# 90% / 0.7728
# 100% / 0.7794