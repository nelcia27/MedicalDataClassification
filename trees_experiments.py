from MedicalDataClassification.helpers import *
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score
import numpy as np
import PIL
from PIL import Image
from MedicalDataClassification.DRSA import find_all_possible_decision_classes


def build_tree(X, y, criterion, max_depth, class_weight, ccp_alpha):#, max_depth, class_weight, ccp_alpha
    clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, class_weight=class_weight, ccp_alpha=ccp_alpha)#
    clf = clf.fit(X, y)
    return clf.feature_importances_


def build_random_forest(X, y, n_estimators, criterion, max_depth, bootstrap, class_weight, ccp_alpha):#, max_depth, bootstrap, class_weight, ccp_alpha
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, bootstrap=bootstrap, class_weight=class_weight, ccp_alpha=ccp_alpha)#, max_depth=max_depth, bootstrap=bootstrap, class_weight=class_weight, ccp_alpha=ccp_alpha
    clf = clf.fit(X, y)
    return clf.feature_importances_


def find_best_tree(X, y):
    n = X.shape[1]
    parameters = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 2, 3, 4, 5, n],
                  'class_weight': (None, 'balanced')}
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


def prepare_report_latex(dataset, data, is_tree, name, class_):
    attributes = [o['name'] for o in dataset['attributes']]
    clf = data
    if is_tree:
        plt.figure(figsize=(18, 18))
        tree.plot_tree(clf, fontsize=10, feature_names=attributes[:-1], class_names=class_, filled=True, impurity=True, rounded=True)
        plt.savefig('results\dec_trees_mgr\ImageTree'+name+'.jpg', bbox_inches='tight')
    else:
        n_estimators = len(clf.estimators_)
        images = []
        for i in range(n_estimators):
            plt.figure(figsize=(18, 24))
            tree.plot_tree(clf.estimators_[i], feature_names=attributes[:-1], class_names=class_, filled=True, impurity=True, rounded=True, fontsize=10)
            plt.savefig('results\dec_trees_mgr\ImageRF' + name + '_estimator_' + str(i) + '.jpg', bbox_inches='tight')
            images.append('results\dec_trees_mgr\ImageRF' + name + '_estimator_' + str(i) + '.jpg')
        imgs = [Image.open(i) for i in images]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack([np.asarray(i.resize(min_shape)) for i in imgs])
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save('results\dec_trees_mgr\ImageRF' + name + '_combined.jpg')
    return 0


def prepare_report(dataset, data, is_tree, name, X, y, class_):
    attributes = [o['name'] for o in dataset['attributes']]
    num_attributes = len(attributes[:-1])
    with pd.ExcelWriter('results\dec_trees\File'+name+'.xlsx', engine='xlsxwriter') as writer:
        workbook = writer.book
        for s in ['Zbiór', 'Dane uruchomienia', 'Wyniki najlepszego',  'Walidacja krzyżowa - trafność', 'Walidacja krzyżowa - f1']:
            worksheet = workbook.add_worksheet(s)
            writer.sheets[s] = worksheet
        df = pd.DataFrame(dataset['attributes'])
        df.to_excel(writer, sheet_name='Zbiór', index=False)
        df1 = pd.DataFrame(dataset['objects'])
        df1.to_excel(writer, sheet_name='Zbiór', startrow=num_attributes + 1, startcol=0, index=False)
        param = list(data[1].keys())
        vals = list(data[1].values())
        df2 = pd.DataFrame({'parametr': param, 'wartość': vals})
        df2.to_excel(writer, sheet_name='Dane uruchomienia', index=False)
        worksheet = writer.sheets['Wyniki najlepszego']
        worksheet.write(0, 0, 'trafność - cv')
        worksheet.write(0, 1, data[2])
        clf = data[0]
        res = []
        for index, row in X.iterrows():
            res.append(clf.predict([row]))
        worksheet.write(6, 0, 'Wyniki testu najlepszego klasyfikatora na pełnym zbiorze danych')
        df3 = pd.DataFrame({"Przewidziana klasa": y, "Prawdziwa klasa": res})
        df3.to_excel(writer, sheet_name='Wyniki najlepszego', startrow=7, startcol=0, index=False)
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
        df4.to_excel(writer, sheet_name='Wyniki najlepszego', startrow=2, startcol=0, index=False, header=False)
        df5 = pd.concat([pd.DataFrame(data[3]["params"]), pd.DataFrame(data[3]["mean_test_accuracy"], columns=["Accuracy"])], axis=1)
        df5.to_excel(writer, sheet_name='Walidacja krzyżowa - trafność', index=False)
        df6 = pd.concat([pd.DataFrame(data[3]["params"]), pd.DataFrame(data[3]["mean_test_f1_weighted"], columns=["F1_weighted"])], axis=1)
        df6.to_excel(writer, sheet_name='Walidacja krzyżowa - f1', index=False)
    clf = data[0]
    if is_tree:
        plt.figure(figsize=(18, 24))
        tree.plot_tree(clf, fontsize=10, feature_names=attributes[:-1], class_names=class_, filled=True, impurity=True, rounded=True)
        plt.savefig('results\dec_trees\ImageTree'+name+'.jpg', bbox_inches='tight')
    else:
        n_estimators = len(clf.estimators_)
        images = []
        for i in range(n_estimators):
            plt.figure(figsize=(18, 24))
            tree.plot_tree(clf.estimators_[i], feature_names=attributes[:-1], class_names=class_, filled=True, impurity=True, rounded=True, fontsize=10)
            plt.savefig('results\dec_trees\ImageRF' + name + '_estimator_' + str(i) + '.jpg', bbox_inches='tight')
            images.append('results\dec_trees\ImageRF' + name + '_estimator_' + str(i) + '.jpg')
        imgs = [Image.open(i) for i in images]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack([np.asarray(i.resize(min_shape)) for i in imgs])
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save('results\dec_trees\ImageRF' + name + '_combined.jpg')
    return 0


def run_experiment_single(dataset_name, type, name):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    ds = dataset_to_trees(dataset)
    cols = list(ds.columns)
    X = ds[cols[:-1]]
    y = ds[cols[-1]].values
    class_ = [str(i) for i in find_all_possible_decision_classes(dataset)]
    if type == 'tree':
        prepare_report(dataset, find_best_tree(X, y), True, name, X, y, class_)
    elif type == 'rf':
        prepare_report(dataset, find_best_forest(X, y), False, name, X, y, class_)


def run_experiment_single_latex(dataset_name, type, name, to_test):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]},
              'test': o} for o in dataset['objects']]
    dataset = folds[to_test]["train"]
    ds = dataset_to_trees(dataset)
    cols = list(ds.columns)
    X = ds[cols[:-1]]
    y = ds[cols[-1]].values
    class_ = [str(i) for i in find_all_possible_decision_classes(dataset)]
    if type == 'tree':
        prepare_report_latex(dataset, build_tree(X, y, 'gini'), True, name, class_)
    elif type == 'rf':
        prepare_report_latex(dataset, build_random_forest(X, y, 3, 'gini'), False, name, class_)


def tree_importance(dataset, criterion, max_depth, class_weight):
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]},
              'test': o} for o in dataset['objects']]
    res = np.zeros((7,), dtype=float)
    for f in folds:
        ds = dataset_to_trees(f['train'])
        cols = list(ds.columns)
        X = ds[cols[:-1]]
        y = ds[cols[-1]].values
        class_ = [str(i) for i in find_all_possible_decision_classes(dataset)]
        res = np.add(build_tree(X, y, criterion, max_depth=max_depth, class_weight=class_weight, ccp_alpha=0.0), res)
    return res/len(folds)


def forest_importance(dataset, n_estimators, criterion, max_depth, bootstrap, class_weight):
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]},
              'test': o} for o in dataset['objects']]
    res = np.zeros((7,), dtype=float)
    for f in folds:
        ds = dataset_to_trees(f['train'])
        cols = list(ds.columns)
        X = ds[cols[:-1]]
        y = ds[cols[-1]].values
        res = np.add(build_random_forest(X, y, n_estimators, criterion, max_depth, bootstrap, class_weight, 0.0), res)
    return res/len(folds)


def run_experiment_single_importance(dataset_name, type, name):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    if type == 'tree':
        print("tree")
        print("gini, niestosowana")
        print(tree_importance(dataset, 'gini', None, None))
        print("entropy, niestosowana")
        print(tree_importance(dataset, 'entropy', None, None))
        print('gini, stosowana')
        print(tree_importance(dataset, 'gini', None, 'balanced'))
        print('entropy, stosowana')
        print(tree_importance(dataset, 'entropy', None, 'balanced'))
    elif type == 'rf':
        print("rf")
        print("gini, niestosowana, 2")
        print(forest_importance(dataset, 2, 'gini', None, False, None))
        print("entropy, niestosowana, 2")
        print(forest_importance(dataset, 2, 'entropy', None, False, None))
        print('gini, stosowana, 2')
        print(forest_importance(dataset, 2, 'gini', None, False, 'balanced'))
        print('entropy, stosowana, 2')
        print(forest_importance(dataset, 2, 'entropy', None, False, 'balanced'))
        print('gini, subbalanced, 2')
        print(forest_importance(dataset, 2, 'gini', None, False, 'balanced_subsample'))
        print('entropy, subbalanced, 2')
        print(forest_importance(dataset, 2, 'entropy', None, False, 'balanced_subsample'))

        print("gini, niestosowana, 3")
        print(forest_importance(dataset, 3, 'gini', None, False, None))
        print("entropy, niestosowana, 3")
        print(forest_importance(dataset, 3, 'entropy', None, False, None))
        print('gini, stosowana, 3')
        print(forest_importance(dataset, 3, 'gini', None, False, 'balanced'))
        print('entropy, stosowana, 3')
        print(forest_importance(dataset, 3, 'entropy', None, False, 'balanced'))
        print('gini, subbalanced, 3')
        print(forest_importance(dataset, 3, 'gini', None, False, 'balanced_subsample'))
        print('entropy, subbalanced, 3')
        print(forest_importance(dataset, 3, 'entropy', None, False, 'balanced_subsample'))


def run_experiment_full():
    files = ['data\jose-medical-2017 (2).isf', 'data\jose-medical (1).isf']
    for i, file in enumerate(files):
        print("Working on file: " + file)
        print('Tree')
        run_experiment_single(file, 'tree', 'tree' + str(i))
        print('Random Forest')
        run_experiment_single(file, 'rf', 'random_forest'+str(i))
        print("Done")


def run_experiment_full_latex(to_test):
    files = ['data\jose-medical (1).isf']#'data\jose-medical-2017 (2).isf']
    for i, file in enumerate(files):
        print("Working on file: " + file)
        print('Tree')
        run_experiment_single_latex(file, 'tree', 'tree' + str(i), to_test)
        print('Random Forest')
        run_experiment_single_latex(file, 'rf', 'random_forest'+str(i), to_test)
        print("Done")


def run_experiment_full_importance():
    files = ['data\jose-medical-2017 (2).isf', 'data\jose-medical (1).isf']
    for i, file in enumerate(files):
        print("Working on file: " + file)
        print('Tree')
        run_experiment_single_importance(file, 'tree', 'tree' + str(i))
        print('Random Forest')
        run_experiment_single_importance(file, 'rf', 'random_forest'+str(i))
        print("Done")
#run_experiment_full_latex(2)
#run_experiment_full()
run_experiment_full_importance()
#'jose-medical-2017 (2).isf')jose-medical (1).isf
#print(find_best_forest(X, y))
#print(find_best_tree(X, y))
