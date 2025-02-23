# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def procesar_datos(datos: pd.DataFrame):
    datos = datos.copy()
    datos = datos.rename(columns={"default payment next month": "incumplimiento"})
    datos = datos.drop(columns=["ID"]).dropna()
    datos = datos[(datos["EDUCATION"] != 0) & (datos["MARRIAGE"] != 0)]
    datos['EDUCATION'] = datos['EDUCATION'].apply(lambda nivel: 4 if nivel >= 4 else nivel).astype('category')
    atributos, objetivo = datos.drop(columns=['incumplimiento']), datos['incumplimiento']
    return datos, atributos, objetivo

def crear_pipeline() -> Pipeline:
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocesador = ColumnTransformer(
        transformers=[('codificar', OneHotEncoder(), columnas_categoricas)],
        remainder='passthrough'
    )
    modelo = Pipeline(steps=[
        ('preprocesador', preprocesador),
        ('clasificador', RandomForestClassifier(random_state=42))
    ])
    return modelo

def optimizar_modelo(modelo, atributos, objetivo):
    parametros = {
        'clasificador__n_estimators': [200],
        'clasificador__max_depth': [None],
        'clasificador__min_samples_split': [10],
        'clasificador__min_samples_leaf': [1, 2],
    }
    busqueda = GridSearchCV(
        modelo, parametros, cv=10, scoring='balanced_accuracy', n_jobs=-1, verbose=2, refit=True
    )
    return busqueda.fit(atributos, objetivo)

def guardar_modelo(modelo):
    os.makedirs('archivos/modelos', exist_ok=True)
    with gzip.open('archivos/modelos/modelo.pkl.gz', 'wb') as archivo:
        pickle.dump(modelo, archivo)

def calcular_metricas(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    predicciones_entrenamiento = modelo.predict(x_entrenamiento)
    predicciones_prueba = modelo.predict(x_prueba)
    
    metricas_entrenamiento = {
        "tipo": "metricas",
        "conjunto": "entrenamiento",
        "precision": float(precision_score(y_entrenamiento, predicciones_entrenamiento)),
        "exactitud_balanceada": float(balanced_accuracy_score(y_entrenamiento, predicciones_entrenamiento)),
        "recuperacion": float(recall_score(y_entrenamiento, predicciones_entrenamiento)),
        "puntaje_f1": float(f1_score(y_entrenamiento, predicciones_entrenamiento))
    }
    
    metricas_prueba = {
        "tipo": "metricas",
        "conjunto": "prueba",
        "precision": float(precision_score(y_prueba, predicciones_prueba)),
        "exactitud_balanceada": float(balanced_accuracy_score(y_prueba, predicciones_prueba)),
        "recuperacion": float(recall_score(y_prueba, predicciones_prueba)),
        "puntaje_f1": float(f1_score(y_prueba, predicciones_prueba))
    }
    
    return metricas_entrenamiento, metricas_prueba

def generar_matriz_confusion(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    matriz_entrenamiento = confusion_matrix(y_entrenamiento, modelo.predict(x_entrenamiento))
    matriz_prueba = confusion_matrix(y_prueba, modelo.predict(x_prueba))
    
    matriz_confusion_entrenamiento = {
        "tipo": "matriz_confusion",
        "conjunto": "entrenamiento",
        "real_0": {"predicho_0": int(matriz_entrenamiento[0, 0]), "predicho_1": int(matriz_entrenamiento[0, 1])},
        "real_1": {"predicho_0": int(matriz_entrenamiento[1, 0]), "predicho_1": int(matriz_entrenamiento[1, 1])}
    }
    
    matriz_confusion_prueba = {
        "tipo": "matriz_confusion",
        "conjunto": "prueba",
        "real_0": {"predicho_0": int(matriz_prueba[0, 0]), "predicho_1": int(matriz_prueba[0, 1])},
        "real_1": {"predicho_0": int(matriz_prueba[1, 0]), "predicho_1": int(matriz_prueba[1, 1])}
    }
    
    return matriz_confusion_entrenamiento, matriz_confusion_prueba

def guardar_metricas(metrics_entrenamiento, metrics_prueba, matriz_conf_entrenamiento, matriz_conf_prueba, archivo="archivos/resultados/metricas.json"):
    os.makedirs(os.path.dirname(archivo), exist_ok=True)
    resultados = [metrics_entrenamiento, metrics_prueba, matriz_conf_entrenamiento, matriz_conf_prueba]
    with open(archivo, "w") as f:
        for resultado in resultados:
            f.write(json.dumps(resultado) + "\n")

datos_prueba = pd.read_csv("archivos/entrada/datos_prueba.csv.zip", compression="zip")
datos_entrenamiento = pd.read_csv("archivos/entrada/datos_entrenamiento.csv.zip", compression="zip")

datos_prueba, x_prueba, y_prueba = procesar_datos(datos_prueba)
datos_entrenamiento, x_entrenamiento, y_entrenamiento = procesar_datos(datos_entrenamiento)

modelo_crediticio = crear_pipeline()
modelo_crediticio = optimizar_modelo(modelo_crediticio, x_entrenamiento, y_entrenamiento)

guardar_modelo(modelo_crediticio)

metricas_entrenamiento, metricas_prueba = calcular_metricas(modelo_crediticio, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)
matriz_conf_entrenamiento, matriz_conf_prueba = generar_matriz_confusion(modelo_crediticio, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

guardar_metricas(metricas_entrenamiento, metricas_prueba, matriz_conf_entrenamiento, matriz_conf_prueba)

