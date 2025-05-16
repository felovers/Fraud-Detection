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
from joblib import load, dump

# Função para localizar arquivos no .exe ou .py
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = path.dirname(path.abspath(__file__))
    return path.join(base_path, relative_path)

# Caminho do dataset e modelo
csv_path = resource_path('creditcard.csv')
model_path = resource_path('RF_Fraud_Model.pkl')

# Carregar o dataset
df = pd.read_csv(csv_path)

# Normalizar Amount
scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

# Separar X e y
X = df.drop('Class', axis=1)
y = df['Class']

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verifica se o modelo já foi salvo
if path.exists(model_path):
    print("\nModelo já existente encontrado. Carregando...")
    model = load(model_path)
else:
    print("\nNenhum modelo salvo encontrado. Treinando um novo modelo...")
    
    # Aplica SMOTE apenas no treino
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Treina modelo
    model = RandomForestClassifier(n_estimators=500, random_state=42, verbose=1, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)

    # Salva modelo
    dump(model, model_path)
    print(f"\nModelo salvo em: {model_path}\n")

# Previsões
y_pred = model.predict(X_test)

# Avaliação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("AUC-ROC Score:")
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

input("Pressione Enter para sair...")
