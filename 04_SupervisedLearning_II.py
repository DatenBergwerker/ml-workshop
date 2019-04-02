import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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


def plot_feature_importance(X, importances, top: int = None, min_imp: float = None):
    """
    Plot the feature importance as a bar chart. Works only with
    model objects that have feature_importance_ attributes (trees).
    """
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances}
                                       ).sort_values('Importance', ascending=True)
    if top:
        feature_importances = feature_importances.tail(top)
    if min_imp:
        feature_importances = feature_importances.loc[feature_importances.Importance >= min_imp]
    return feature_importances.plot(kind='barh', x='Feature', y='Importance')


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
    'DecisionTree': {
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

feature_importances = {'RandomForest': [], 'DecisionTree': []}
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
    feature_importances['RandomForest'].append(rf.feature_importances_)
    print(f'RandomForest Accuracy: {results["RandomForest"]["metrics"]["acc"][-1]}')
    
    # Support Vector Machine
    svc = SVC(**params['SVC'])
    svc.fit(X=X_train, y=y_train)
    preds = svc.predict(X=X_test)
    results['SVC']['metrics']['acc'].append(accuracy_score(y_true=y_test, y_pred=preds))
    results['SVC']['probs'].append(svc.predict_proba(X=X_test))
    results['SVC']['preds'].append(preds)
    print(f'SVC Accuracy: {results["SVC"]["metrics"]["acc"][-1]}')
    
    dt = DecisionTreeClassifier(**params['DecisionTree'])
    dt.fit(X=X_train, y=y_train)
    preds = dt.predict(X=X_test)
    results['DecisionTree']['metrics']['acc'].append(accuracy_score(y_true=y_test, y_pred=preds))
    results['DecisionTree']['probs'].append(dt.predict_proba(X=X_test))
    results['DecisionTree']['preds'].append(preds)
    feature_importances['DecisionTree'].append(dt.feature_importances_)
    print(f'DecisionTree Accuracy: {results["DecisionTree"]["metrics"]["acc"][-1]}')

print(f'Overall Accuracy RandomForest: {sum(results["RandomForest"]["metrics"]["acc"]) / splits}')
print(f'Overall Accuracy SVC: {sum(results["SVC"]["metrics"]["acc"]) / splits}')

# Implementiere eine Support Vector Machine und probiere verschiedene Parameter aus
# und evaluiere die Performance
# s. Line 86 ff.

# Unterscheidet sich die Feature Importance zwischen RandomForest und DecisionTree
rf_macro_feature_importance = np.array(feature_importances['RandomForest'])
rf_mean_feature_importance = np.mean(rf_macro_feature_importance, axis=0)
plot_feature_importance(X=X, importances=rf_mean_feature_importance, top=10)
plt.show()

dt_macro_feature_importance = np.array(feature_importances['DecisionTree'])
dt_mean_feature_importance = np.mean(dt_macro_feature_importance, axis=0)
plot_feature_importance(X=X, importances=dt_mean_feature_importance, top=10)
plt.show()

# Leichte unterschiede aber die top 4 sind gleich

# Wie kann man eine MonteCarlo Crossvalidation mit 15 runs implementieren?
# Monte Carlo CV am Beispiel vom RandomForest
results = {'RandomForest': {
    'metrics': {
        'acc': []
        }, 'probs': [], 'preds': []
    }
}
num_runs = 15
for i in range(num_runs):
    print(f'MonteCarlo CV Run: {i + 1}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    rf = RandomForestClassifier(**params['RandomForest'])
    rf.fit(X=X_train, y=y_train)
    preds = rf.predict(X=X_test)
    results['RandomForest']['metrics']['acc'].append(accuracy_score(y_true=y_test, y_pred=preds))
    results['RandomForest']['probs'].append(rf.predict_proba(X=X_test))
    results['RandomForest']['preds'].append(preds)
    feature_importances['RandomForest'].append(rf.feature_importances_)
    print(f'RandomForest Accuracy: {results["RandomForest"]["metrics"]["acc"][-1]}')
