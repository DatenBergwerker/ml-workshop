import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


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
models = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [150, 300, 500],
            'max_features': [6]
        }
    }
}

results = {model: {'metrics': {'acc': [],
                               'precision': [],
                               'recall': []},
                   'preds': [], 'probs': [], 'parameter_rank': []}
           for model in models.keys()}

# Train test split control
splits = 5
kfold = KFold(n_splits=splits)
for i, (train, test) in enumerate(kfold.split(X=X, y=y)):
    print(f'Currently in Run {i + 1}')
    X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]

    for model in models:
        grid_cv = GridSearchCV(estimator=models[model]['model'], param_grid=models[model]['params'],
                               n_jobs=-1, verbose=True)
        grid_cv.fit(X=X_train, y=y_train)

        preds = grid_cv.predict(X=X_test)
        results[model]['metrics']['acc'].append(accuracy_score(y_true=y_test, y_pred=preds))
        results[model]['metrics']['precision'].append(precision_score(y_true=y_test, y_pred=preds, pos_label='yes'))
        results[model]['metrics']['recall'].append(recall_score(y_true=y_test, y_pred=preds, pos_label='yes'))
        results[model]['probs'].append(grid_cv.predict_proba(X=X_test))
        results[model]['preds'].append(preds)
        results[model]['parameter_rank'].append(grid_cv.cv_results_['rank_test_score'])

# bestes parameterset über alle runs auswählen (nach niedrigstem kumulierten rang)

grid_cv.cv_results_['params'][np.argmin(sum(results[model]['parameter_rank']))]