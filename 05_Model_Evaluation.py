import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    results['RandomForest']['metrics']['precision'].append(precision_score(y_true=y_test, y_pred=preds, pos_label='yes'))
    results['RandomForest']['metrics']['recall'].append(recall_score(y_true=y_test, y_pred=preds, pos_label='yes'))
    results['RandomForest']['probs'].append(rf.predict_proba(X=X_test))
    results['RandomForest']['preds'].append(preds)


# Erweitere die Analyse um zwei Modelle deiner Wahl

# Lass dir zusätzlich zur Accuracy noch Precision und Recall ausgeben.
# Welches Modell detektiert die positive Klasse am besten?
# Welches Modell hat die geringste Wahrscheinlichkeit für false positives?