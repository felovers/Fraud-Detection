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

#region Funções
# Função para localizar arquivos no .exe ou .py
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = path.dirname(path.abspath(__file__))
    return path.join(base_path, relative_path)

# Função para aplicar o SMOTE
def smote_resample(X_train, y_train):
    global smote_applied, X_train_resampled, y_train_resampled
    if not smote_applied:
        print('Aplicando SMOTE...')
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        smote_applied = True
    else:
        print('SMOTE já aplicado anteriormente.')
    return X_train_resampled, y_train_resampled
#endregion Funções

#region Definição de variáveis globais
smote_applied = False
X_train_resampled = None
y_train_resampled = None
#endregion Definição de variáveis globais

# Caminho do dataset e modelos
dataset_csv = resource_path('PS_20174392719_1491204439457_log.csv')
rf_model_path = resource_path('RF_PaySim_Model.pkl')
xgb_model_path = resource_path('XGB_PaySim_Model.pkl')

# Carregar o dataset
df = pd.read_csv(dataset_csv)

# Remover colunas não numéricas irrelevantes
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Transformar a coluna categórica 'type' em variáveis dummies
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Separar X e y
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)  # 'isFlaggedFraud' opcional
y = df['isFraud']

# Normalizar colunas numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#region --- Modelo Random Forest ---
print('----- Modelo 1 - Random Forest -----')

# Verifica se o modelo já foi salvo
if path.exists(rf_model_path):
    print("\nModelo RandomForest já existente encontrado. Carregando...")
    rf_model = load(rf_model_path)
else:
    print("\nNenhum modelo RandomForest encontrado. Treinando novo modelo...")

    # Aplica SMOTE apenas no treino
    X_train_resampled, y_train_resampled = smote_resample(X_train, y_train)

    # Treina modelo
    rf_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        verbose=1,
        n_jobs=-1
        )
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Salva modelo
    dump(rf_model, rf_model_path)
    print(f"\nModelo RandomForest salvo em: {rf_model_path}\n")

# Previsões e Avaliação - Random Forest
print("\n=== Avaliação do RandomForest ===")
y_pred_rf = rf_model.predict(X_test)
print("\nRelatório de Classificação - RandomForest:")
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC Score - RandomForest:")
print(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - RandomForest')
plt.show()
#endregion --- Modelo Random Forest ---

input('\nPressione Enter para iniciar o próximo modelo (XGBoost)...')

#region --- Modelo XGBoost ---
print('\n----- Modelo 2 - XGBoost -----')

# Verifica se o modelo já foi salvo
if path.exists(xgb_model_path):
    print("\nModelo XGBoost já existente encontrado. Carregando...")
    xgb_model = load(xgb_model_path)
else:
    print("\nNenhum modelo XGBoost encontrado. Treinando novo modelo...")

    # Aplica SMOTE apenas no treino
    X_train_resampled, y_train_resampled = smote_resample(X_train, y_train)

    # Treina XGBoost
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        verbosity=2,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Salva modelo
    dump(xgb_model, xgb_model_path)
    print(f"\nModelo XGBoost salvo em: {xgb_model_path}")

# Previsões e avaliação - XGBoost
print("\n=== Avaliação do XGBoost ===")
y_pred_xgb = xgb_model.predict(X_test)
print("\nRelatório de Classificação - XGBoost:")
print(classification_report(y_test, y_pred_xgb))
print("AUC-ROC Score - XGBoost:")
print(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - XGBoost')
plt.show()
#endregion --- Modelo XGBoost ---

input("Pressione Enter para sair...")
