import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def load_prepare():
    """
    Convenience wrapper for setup
    """
    # drop doppelte index spalte
    df = pd.read_csv('data/balanced_bank.csv').iloc[:, 1:]

    num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = df.drop(num_cols, axis=1).columns.drop('y')

    for c in df[cat_cols].columns:
        df[c] = df[c].apply(str)
        df[c] = LabelEncoder().fit_transform(df[c])
        df[c] = pd.to_numeric(df[c]).astype(np.float)
    return df, num_cols, cat_cols

warnings.filterwarnings('ignore')
bank_marketing, num_cols, cat_cols = load_prepare()

# 1) Train / Test Split
    # a) Implementiere eine 10 Fold Cross Validation
    # b) Implementiere eine MonteCarlo Cross Validation
    # c) Implementiere Scaling für eine der beiden CV

# 2) Modelling (Implementierung muss nicht auf 1) aufbauen, ein einfacher Train/Test Split reicht)
    # a) Implementiere einen Random Forest und wähle die einflussreichsten Features aus
    # b) Implementiere eine Logistische Regression (mit den features aus a)
    # c) Berechne gängige Evaluationsmetriken für die Logistische Regression

# 3) Wrap-Up (Bonus)
    # Implementiere eine 5 Fold Cross Validation mit Gridsearch, die eine Support Vector Machine mit einem
    # Random Forest vergleicht. Als parameter für die SVM sollen kernel (rbf, polynomial) sowie d = (3,4) und
    # C (0.1, 1, 10) probiert werden. Für den RandomForest soll max features von √featureanzahl , 1/4 Features,
    # 1/2 Features und 3/4 der Features sowie die Anzahl der Bäume zwischen (200, 300, 400) variiert werden.
    # Bestimme die optimalen parametersets für beide Algorithmen.