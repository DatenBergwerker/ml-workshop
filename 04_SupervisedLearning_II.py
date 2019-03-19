import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


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

# preprocessing
ohc = OneHotEncoder(sparse=False)
scaler = StandardScaler()

X_ohc = ohc.fit_transform(X=bank_marketing[cat_cols].values)
X_ohc_df = pd.DataFrame(X_ohc,
                        columns=ohc.get_feature_names(bank_marketing[cat_cols].columns))

# Model matrix
X = pd.concat([bank_marketing[num_cols], X_ohc_df], axis=1)
y = bank_marketing['y']

# Model specification
params = {
    'RandomForest': {
        'n_estimators': 150,
        'max_features': 6
    },
    'SVC': {
        'C': 1,
        'probability': True,
        'kernel': 'rbf'
    }
}

results = {model: {'metrics': {'acc': [],
                               'precision': [],
                               'recall': []},
                   'preds': [], 'probs': []}
           for model in params.keys()}

# Train test split control
splits = 5
kfold = KFold(n_splits=splits)
for i, (train, test) in enumerate(kfold.split(X=X, y=y)):
    print(f'Currently in Run {i + 1}')
    X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]

    X_train.loc[:, num_cols] = scaler.fit_transform(X=X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X=X_test[num_cols])


    # Random Forest
    rf = RandomForestClassifier(**params['RandomForest'])
    rf.fit(X=X_train, y=y_train)
    preds = rf.predict(X=X_test)
    results['RandomForest']['metrics']['acc'].append(accuracy_score(y_true=y_test, y_pred=preds))
    results['RandomForest']['probs'].append(rf.predict_proba(X=X_test))
    results['RandomForest']['preds'].append(preds)
    print(f'RandomForest Accuracy: {results["RandomForest"]["metrics"]["acc"][-1]}')

print(f'Overall Accuracy: {sum(results["RandomForest"]["metrics"]["acc"]) / splits}')

# Implementiere eine Support Vector Machine und probiere verschiedene Parameter aus
# und evaluiere die Performance

# Unterscheidet sich die Feature Importance zwischen RandomForest und DecisionTree

# Wie kann man eine MonteCarlo Crossvalidation mit 15 runs implementieren?