import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
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
import io
import contextlib
from PIL import Image, ImageTk

# Recurso compatível com PyInstaller
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = path.dirname(path.abspath(__file__))
    return path.join(base_path, relative_path)

# Captura prints do processamento
def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()

# Função de processamento
def process(csv_path, model_path):
    df = pd.read_csv(csv_path)

    scaler = StandardScaler()
    df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if path.exists(model_path):
        print("\nModelo já existente encontrado. Carregando...")
        model = load(model_path)
    else:
        print("\nNenhum modelo salvo encontrado. Treinando um novo modelo...")

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=500, random_state=42, verbose=1, n_jobs=-1)
        model.fit(X_train_resampled, y_train_resampled)

        dump(model, model_path)
        print(f"\nModelo salvo em: {model_path}\n")

    y_pred = model.predict(X_test)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    print("AUC-ROC Score:")
    print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig('conf_matrix.png')
    plt.close()

# Redimensionamento proporcional
def resize_image_keep_aspect(image, max_width, max_height):
    w, h = image.size
    ratio = min(max_width / w, max_height / h)
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def show_image_small(path_img, label, max_width=700, max_height=400):
    img = Image.open(path_img)
    img_resized = resize_image_keep_aspect(img, max_width, max_height)
    img_tk = ImageTk.PhotoImage(img_resized)
    label.config(image=img_tk)
    label.image = img_tk

def open_image_window(path_img):
    top = tk.Toplevel()
    top.title("Imagem Ampliada")
    screen_w = top.winfo_screenwidth()
    screen_h = top.winfo_screenheight()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)
    img = Image.open(path_img)
    img_resized = resize_image_keep_aspect(img, max_w, max_h)
    img_tk = ImageTk.PhotoImage(img_resized)
    lbl_full = tk.Label(top, image=img_tk)
    lbl_full.image = img_tk
    lbl_full.pack(expand=True)
    lbl_full.bind("<Button-1>", lambda e: top.destroy())

def select_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        entry_csv.delete(0, tk.END)
        entry_csv.insert(0, file_path)

def select_model():
    file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("PKL files", "*.pkl")])
    if file_path:
        entry_modelo.delete(0, tk.END)
        entry_modelo.insert(0, file_path)

def execute():
    csv_path = entry_csv.get()
    model_path = entry_modelo.get()
    if not csv_path or not path.exists(csv_path):
        messagebox.showerror("Erro", "Arquivo CSV não encontrado.")
        return
    try:
        output = capture_output(process, csv_path, model_path)
        txt_output.config(state='normal')
        txt_output.delete("1.0", tk.END)
        txt_output.insert(tk.END, output)
        txt_output.config(state='disabled')

        show_image_small("conf_matrix.png", lbl_img)
        lbl_img.config(cursor="hand2")
        lbl_img.bind("<Enter>", lambda e: lbl_img.config(cursor="hand2"))
        lbl_img.bind("<Leave>", lambda e: lbl_img.config(cursor=""))
        lbl_img.bind("<Button-1>", lambda e: open_image_window("conf_matrix.png"))
    except Exception as e:
        messagebox.showerror("Erro durante o processamento", str(e))

# Interface
root = tk.Tk()
root.title("Detecção de Fraudes com Machine Learning")

# Labels e entradas
tk.Label(root, text="Selecione o CSV:").grid(row=0, column=0, sticky='w', padx=10, pady=(10, 0))
entry_csv = tk.Entry(root, width=60)
entry_csv.grid(row=1, column=0, padx=10)
tk.Button(root, text="Procurar", command=select_csv).grid(row=1, column=1, padx=5)

tk.Label(root, text="Arquivo do Modelo (.pkl):").grid(row=2, column=0, sticky='w', padx=10, pady=(10, 0))
entry_modelo = tk.Entry(root, width=60)
entry_modelo.insert(0, "RF_Fraud_Model.pkl")
entry_modelo.grid(row=3, column=0, padx=10)
tk.Button(root, text="Salvar Como", command=select_model).grid(row=3, column=1, padx=5)

tk.Button(root, text="Executar", command=execute).grid(row=4, column=0, columnspan=2, pady=10)

tk.Label(root, text="Saída:").grid(row=5, column=0, sticky='w', padx=10)
txt_output = ScrolledText(root, width=100, height=20, state='disabled')
txt_output.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

lbl_img = tk.Label(root)
lbl_img.grid(row=7, column=0, columnspan=2, pady=10)

tk.Button(root, text="Sair", command=root.destroy).grid(row=8, column=0, columnspan=2, pady=10)

root.mainloop()
