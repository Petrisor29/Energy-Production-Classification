import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score

# Importuri Modele
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1. Încărcare și Pregătire
df = pd.read_csv('Energy Production Dataset.csv')

# Discretizare Target (Low/Medium/High)
df['Production_Class'] = pd.qcut(df['Production'], q=3, labels=['Low', 'Medium', 'High'])

# Procesare Features
X_raw = df.drop(columns=['Production', 'Production_Class', 'Date'])
y = df['Production_Class']
X = pd.get_dummies(X_raw, columns=['Source', 'Day_Name', 'Month_Name', 'Season'], drop_first=True)

# Split 70/30 Stratificat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scalare (pentru kNN/Logistic/LDA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2. Funcție Evaluare Complexă
def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    try:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # Calculăm toate metricile
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

        return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "Prediction": y_pred}
    except Exception as e:
        return {"Model": name, "Accuracy": 0, "Error": str(e)}


# 3. Rularea Tuturor Modelelor
results_list = []

# Modele Scalate
models_scaled = [
    (LinearDiscriminantAnalysis(), "LDA"),
    (GaussianNB(), "GaussianNB"),
    (LogisticRegression(max_iter=1000), "Logistic Regression"),
    (KNeighborsClassifier(n_neighbors=5), "k-NN (k=5)"),
    (KNeighborsClassifier(n_neighbors=7), "k-NN (k=7)")
]

for model, name in models_scaled:
    results_list.append(evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, name))

# Modele Nescalate (Arbori)
trees = [
    (DecisionTreeClassifier(max_depth=4, random_state=42), "Decision Tree (depth=4)"),
    (DecisionTreeClassifier(max_depth=None, random_state=42), "Decision Tree (Full)")
]

for model, name in trees:
    res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    results_list.append(res)
    # Salvăm modelul full pentru analiză ulterioară
    if name == "Decision Tree (Full)":
        best_tree_model = model

# 4. Afișare Tabel
results_df = pd.DataFrame(results_list)
print("\n=== CLASAMENT FINAL DETALIAT ===")
print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].sort_values(by="Accuracy", ascending=False))

# 5. Analiză Detaliată Câștigător
print("\n=== DETALII ARBORE DE DECIZIE (FULL) ===")
# Luăm predicțiile din dataframe
best_preds = results_df.loc[results_df['Model'] == "Decision Tree (Full)", 'Prediction'].values[0]

print("\n1. Raport de Clasificare:")
print(classification_report(y_test, best_preds))

print("\n2. Matrice de Confuzie:")
cm = confusion_matrix(y_test, best_preds, labels=['Low', 'Medium', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title("Matrice de Confuzie - Decision Tree")
plt.xlabel("Predicție")
plt.ylabel("Realitate")
plt.show()

print("\n3. Importanța Factorilor:")
importances = pd.DataFrame({'Feature': X.columns, 'Importance': best_tree_model.feature_importances_})
print(importances.sort_values(by='Importance', ascending=False).head(10))