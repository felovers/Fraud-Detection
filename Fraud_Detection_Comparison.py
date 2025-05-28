#region Imports
from os import path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from xgboost import XGBClassifier
#endregion Imports

# Função para localizar arquivos no .exe ou .py
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = path.dirname(path.abspath(__file__))
    return path.join(base_path, relative_path)

# Caminho do dataset e modelo
csv_path = resource_path('creditcard.csv')
rf_model_path = resource_path('RF_Fraud_Model.pkl')
xgb_model_path = resource_path('XGB_Fraud_Model.pkl')

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

#region --- Modelo Random Forest ---
print('----- Modelo 1 - Random Forest -----')

# Verifica se o modelo já foi salvo
if path.exists(rf_model_path):
    print("\nModelo já existente encontrado. Carregando...")
    rf_model = load(rf_model_path)
else:
    print("\nNenhum modelo salvo encontrado. Treinando um novo modelo...")
    
    # Aplica SMOTE apenas no treino
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Treina modelo
    rf_model = RandomForestClassifier(n_estimators=500, random_state=42, verbose=1, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Salva modelo
    dump(rf_model, rf_model_path)
    print(f"\nModelo salvo em: {rf_model_path}\n")
#endregion --- Modelo Random Forest ---

#region --- Modelo XGBoost ---
print('\n----- Modelo 2 - XGBoost -----')

# Verirfica se o modelo já foi salvo
if path.exists(xgb_model_path):
    print("\nModelo XGBoost já existente encontrado. Carregando...")
    xgb_model = load(xgb_model_path)
else:
    print("\nNenhum modelo XGBoost encontrado. Treinando novo modelo...")

    # Aplica SMOTE apenas no treino
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Treina XGBoost
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        verbosity=1,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Salva modelo
    dump(xgb_model, xgb_model_path)
    print(f"\nModelo XGBoost salvo em: {xgb_model_path}")
#endregion --- Modelo XGBoost ---

#region Previsões e Avaliações dos Modelos
# Random Forest
print("\n=== Avaliação do Random Forest ===")
y_pred_rf = rf_model.predict(X_test)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC Score:")
print(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# XGBoost
print("\n=== Avaliação do XGBoost ===")
y_pred_xgb = xgb_model.predict(X_test)
print("\nRelatório de Classificação - XGBoost:")
print(classification_report(y_test, y_pred_xgb))
print("AUC-ROC Score - XGBoost:")
print(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))
#endregion Previsões e Avaliações dos Modelos

#region Matriz de Confusão de ambos os modelos
# Criar figura com 2 subplots (um ao lado do outro)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[0])
axs[0].set_title('Matriz de Confusão - RandomForest')
axs[0].set_xlabel('Previsto')
axs[0].set_ylabel('Real')

# XGBoost
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Greens', ax=axs[1])
axs[1].set_title('Matriz de Confusão - XGBoost')
axs[1].set_xlabel('Previsto')
axs[1].set_ylabel('Real')

# Mostra as duas matrizes de uma só vez
plt.tight_layout()
plt.show()
#endregion Matriz de Confusão de ambos os modelos

# Exigência de input para finalizar código, não permitindo o terminal fechar sozinho
input("Pressione Enter para sair...")
