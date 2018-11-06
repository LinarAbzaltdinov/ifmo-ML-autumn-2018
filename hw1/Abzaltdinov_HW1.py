import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def replaceLettersByNumbers(dataframe):
    return dataframe.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], [1, 2, 3, 4, 5, 6, 7])

def replaceNumbersByLetters(dataframe):
    return dataframe.replace([1, 2, 3, 4, 5, 6, 7], ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

def writeToFile(class_predict, filename):
    result = pd.DataFrame()
    result['class'] = class_predict
    result = replaceNumbersByLetters(result)
    result.insert(0, 'id', testing_data.loc[:, 'id'])
    print(result.head())
    result.to_csv(filename + '.csv', index=False)


names = ["Nearest Neighbors",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Gaussian Process",
         "Linear SVM",
         "RBF SVM"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
]

training_data = pd.read_csv('data/train.csv')
testing_data = pd.read_csv('data/test.csv')
training_data = replaceLettersByNumbers(training_data)
testing_data = replaceLettersByNumbers(testing_data)

y = training_data['class'].values
X = training_data.drop('class', axis=1).values

k_fold = KFold(n_splits=10, shuffle=True).get_n_splits(X)

best_score = 0

for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1)
    avg_score = np.average(scores)
    print(name, ':', scores)
    print('avg:', avg_score)
    if avg_score > best_score:
        best_score = avg_score
        best_clf = clf

best_clf.fit(X, y)
class_predict = best_clf.predict(testing_data.values)
writeToFile(class_predict.T, 'predicted')

print('decisionTreeClassifiers:')

for depth in range(3, 20):
    clf = DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1)
    avg_score = np.average(scores)
    print(str(depth), ':', scores)
    print('avg:', avg_score)
    if avg_score > best_score:
        best_score = avg_score
        best_clf = clf

best_clf.fit(X, y)
class_predict = best_clf.predict(testing_data.values)
writeToFile(class_predict.T, 'decisionTreeClassifier')
