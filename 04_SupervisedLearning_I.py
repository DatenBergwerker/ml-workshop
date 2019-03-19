import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# drop doppelte index spalte
bank_marketing = pd.read_csv('data/balanced_bank.csv').iloc[:, 1:]

num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
cat_cols = bank_marketing.drop(num_cols, axis=1).columns.drop('y')

for c in bank_marketing[cat_cols].columns:
    bank_marketing[c] = bank_marketing[c].apply(str)
    bank_marketing[c] = LabelEncoder().fit_transform(bank_marketing[c])
    bank_marketing[c] = pd.to_numeric(bank_marketing[c]).astype(np.float)

# preprocessing
ohc = OneHotEncoder(sparse=False)
scaler = StandardScaler()

X_ohc = ohc.fit_transform(X=bank_marketing[cat_cols].values)
X_ohc_df = pd.DataFrame(X_ohc,
                        columns=ohc.get_feature_names(bank_marketing[cat_cols].columns))

# train test split vorziehen
X = pd.concat([bank_marketing[num_cols], X_ohc_df], axis=1)
y = bank_marketing['y']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7)
# diese beiden zeilen hinter train_test_split setzen
X_train.loc[:, num_cols] = scaler.fit_transform(X=X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X=X_test[num_cols])

results = {}

# Logistic Regression
lr = LogisticRegression()
lr.fit(X=X_train, y=y_train)

probs = lr.predict_proba(X=X_test)
preds = lr.predict(X=X_test)
results.update({'lr': {'acc': accuracy_score(y_true=y_test, y_pred=preds),
                       'preds': preds,
                       'probs': probs}})

# Prediction Confidence for Class 1
pd.DataFrame(probs[:, 1], columns=['LR - Probability for Class 1']).hist()
plt.show()

# War alles richtig?
# Scaler verschoben

# Implementiere einen KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X=X_train, y=y_train)
probs = knn.predict_proba(X=X_test)
preds = knn.predict(X=X_test)
results.update({'knn': {'acc': accuracy_score(y_true=y_test, y_pred=preds),
                        'preds': preds,
                        'probs': probs}})
pd.DataFrame(probs[:, 1], columns=['KNN - Probability for Class 1']).hist()
plt.show()

# Implementiere einen Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X=X_train, y=y_train)
probs = tree.predict_proba(X=X_test)
preds = tree.predict(X=X_test)
results.update({'tree': {'acc': accuracy_score(y_true=y_test, y_pred=preds),
                         'preds': preds,
                         'probs': probs}})
pd.DataFrame(probs[:, 1], columns=['TREE - Probability for Class 1']).hist()
plt.show()


# Welche Features reduzieren den Gini Koeffzient am meisten?
def plot_feature_importance(X, model, top: int = None, min_imp: float = None):
    """
    Plot the feature importance as a bar chart. Works only with
    model objects that have feature_importance_ attributes (trees).
    """
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}
                                       ).sort_values('Importance', ascending=True)
    if top:
        feature_importances = feature_importances.tail(top)
    if min_imp:
        feature_importances = feature_importances.loc[feature_importances.Importance >= min_imp]
    return feature_importances.plot(kind='barh', x='Feature', y='Importance')


plot_feature_importance(X=X, model=tree)
plt.show()

# Unterscheidet sich die Performance in der Accuracy?
print([(model, results[model]['acc']) for model in results.keys()])


# Wie kann man die Ergebnisse auf Robustheit prüfen?
# z.B. mehr Folds (k-Fold CV), durch eine MonteCarlo Cross Validation (Bagging)