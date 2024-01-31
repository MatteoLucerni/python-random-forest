import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

titanic = pd.read_csv("http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
titanic.info()

titanic = titanic.drop('Name', axis=1)

titanic = pd.get_dummies(titanic)
titanic.head()

X = titanic.drop('Survived', axis=1).values
Y = titanic['Survived'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# single decision tree
dt = DecisionTreeClassifier(criterion='gini', max_depth=6)
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)
Y_pred_train = dt.predict(X_train)

acc = accuracy_score(Y_test, Y_pred)
acc_train = accuracy_score(Y_train, Y_pred_train)
print(f'Accuracy S - test: {acc} / train: {acc_train}')

dotfile = open('tree.dot', 'w')
export_graphviz(dt, out_file= dotfile, feature_names= titanic.columns.drop('Survived'))
dotfile.close()

print('=' * 50)

# random forest
rf = RandomForestClassifier(random_state=False, max_depth=8, n_estimators=30)
rf.fit(X_train, Y_train)
Y_pred= rf.predict(X_test)
Y_pred_train = rf.predict(X_train)

acc = accuracy_score(Y_test, Y_pred)
acc_train = accuracy_score(Y_train, Y_pred_train)

print(f'Accuracy RF - test: {acc} / train: {acc_train}')

print('=' * 50)

# iter to find best parameters
best_acc = 0
best_params = {'Depth': None, 'Trees': None}
best_diff = float('inf')

for depth in range(3,15):
    for trees in range(10, 110, 10):
        print(f'Depth: {depth} / Trees: {trees}')

        rf = RandomForestClassifier(random_state=False, max_depth=depth, n_estimators=trees)
        rf.fit(X_train, Y_train)
        Y_pred= rf.predict(X_test)
        Y_pred_train = rf.predict(X_train)

        acc = accuracy_score(Y_test, Y_pred)
        acc_train = accuracy_score(Y_train, Y_pred_train)

        diff = abs(acc - acc_train)

        # max diff to 0.08 to avoid high overfitting (adjustable)
        if acc > best_acc and diff <= 0.08:
            best_acc = acc
            best_params = {'Depth': depth, 'Trees': trees}
            best_diff = diff

print(f'Miglior accuracy sul test set: {best_acc}')
print(f'Migliori parametri: {best_params}')
print(f'Differenza tra test e training set: {best_diff}')