from MedicalDataClassification.helpers import *
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score


def build_tree(X, y, criterion, max_depth, class_weight, ccp_alpha):
    clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, class_weight=class_weight, ccp_alpha=ccp_alpha)
    clf = clf.fit(X, y)
    return clf


def build_random_forest(X, y, n_estimators, criterion, max_depth, bootstrap, class_weight, ccp_alpha):
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, bootstrap=bootstrap, class_weight=class_weight, ccp_alpha=ccp_alpha)
    clf = clf.fit(X, y)
    return clf


def find_best_tree(X, y):
    n = X.shape[1]
    parameters = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 2, 3, 4, 5, n],
                  'class_weight': (None, 'balanced'), 'ccp_alpha': [0.0, 0.5, 1.0, 2.0]}
    lo = LeaveOneOut()
    t = tree.DecisionTreeClassifier(criterion='gini')
    clf = GridSearchCV(t, parameters, cv=lo, scoring=['accuracy', 'f1_weighted'], refit='accuracy')
    clf.fit(X, y)
    return clf.best_estimator_, clf.best_params_, clf.best_score_, clf.cv_results_


def find_best_forest(X, y):
    n = X.shape[1]
    parameters = {'criterion': ('gini', 'entropy'), 'max_depth': [None, 2, 3, 4, 5, n],
                  'class_weight': (None, 'balanced', 'balanced_subsample'), 'ccp_alpha': [0.0, 0.5, 1.0, 2.0],
                  'n_estimators': [2, 3, 4, 5, 10], 'bootstrap': [True, False]}
    lo = LeaveOneOut()
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, cv=lo, scoring=['accuracy', 'f1_weighted'], refit='accuracy')
    clf.fit(X, y)
    return clf.best_estimator_, clf.best_params_, clf.best_score_, clf.cv_results_


def prepare_report(dataset, data, is_tree, name, num_attributes, X, y):
    #+ to co daje w best score, wyniki cross-valid
    #rysunki:
    # dla drzew:
    # dla lasów:
    with pd.ExcelWriter('results\dec_trees\File'+name+'.xlsx', engine='xlsxwriter') as writer:
        workbook = writer.book
        for s in ['Zbiór', 'Dane uruchomienia', 'Wyniki najlepszego',  'Walidacja krzyżowa']:
            worksheet = workbook.add_worksheet(s)
            writer.sheets[s] = worksheet
        df = pd.DataFrame(dataset['attributes'])
        df.to_excel(writer, sheet_name='Zbiór', index=False)
        df1 = pd.DataFrame(dataset['objects'])
        df1.to_excel(writer, sheet_name='Zbiór', startrow=num_attributes + 1, startcol=0, index=False)
        param = list(data[1].keys())
        vals = list(data[1].values())
        df2 = pd.DataFrame({'parametr':param, 'wartość':vals})
        df2.to_excel(writer, sheet_name='Dane uruchomienia', index=False)
        worksheet = writer.sheets['Wyniki najlepszego']
        worksheet.write(0, 0, 'trafność')
        worksheet.write(0, 1, data[2])
        clf = data[0]
        res = []
        for index, row in X.iterrows():
            res.append(clf.predict(row))
        df3 = pd.DataFrame({"Przewidziana klasa": y, "Prawdziwa klasa": res})
        df3.to_excel(writer, sheet_name='Wyniki najlepszego', startrow=2, startcol=0, index=False)
        good = []
        bad = []
        bad_classified = []
        for e_r, r in zip(y, res):
            if e_r == r:
                good.append(1)
            else:
                bad.append(1)
                if r is not None:
                    bad_classified.append(1)
        accuracy = float(sum(good)) / float(sum(good) + sum(bad))
        correct = float(sum(good)) / float(sum(good) + sum(bad_classified))
        d = (accuracy, correct, f1_score(y, res, average='weighted'))
        cv = {(k, v) for k, v in zip(['accuracy', 'correct', 'f1_score'], d)}
        df4 = pd.DataFrame(cv)
        df4.to_excel(writer, sheet_name='Wyniki najlepszego', startrow=4, startcol=0, index=False, header=False)
        



#'jose-medical-2017 (2).isf')jose-medical (1).isf
a, p, o = read_dataset('data\jose-medical-2017 (2).isf')
dataset = prepare_dataset(a, p, o)
ds = dataset_to_trees(dataset)
cols = list(ds.columns)
X = ds[cols[:-1]]
print(type(X))
y = ds[cols[-1]].values
#print(find_best_forest(X, y))
#print(find_best_tree(X, y))
