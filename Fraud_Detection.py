from os import path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Função para localizar o creditcard.csv no .exe ou .py
def resource_path(relative_path):
    try:
        # Quando for executável (PyInstaller)
        base_path = sys._MEIPASS
    except AttributeError:
        # Quando for .py: usa a pasta onde o arquivo .py está
        base_path = path.dirname(path.abspath(__file__))

    return path.join(base_path, relative_path)

# Caminho correto para o csv
csv_path = resource_path('creditcard.csv')

# Carregar o dataset
df = pd.read_csv(csv_path)

# Exibir informações e distribuição das classes
print(df.info())
print(df['Class'].value_counts())

# Normalização da variável 'Amount'
scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

# Definir variáveis de entrada (X) e saída (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Aplicando SMOTE para balanceamento das classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Treinar o modelo RandomForest
model = RandomForestClassifier(n_estimators=20, random_state=42, verbose=1, n_jobs=-1)
model.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = model.predict(X_test)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print("AUC-ROC Score:")
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Gerar a Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Pausa para não fechar o terminal imediatamente
input("Pressione Enter para sair...")
