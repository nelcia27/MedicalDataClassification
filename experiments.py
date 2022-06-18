from MedicalDataClassification.DRSA import *
from MedicalDataClassification.rules_induction import *
from MedicalDataClassification.classification import *
from MedicalDataClassification.helpers import *
from sklearn.metrics import f1_score


def run_DRSA(dataset, algorithm):
    downward_union = find_downward_union_of_classes(dataset)
    upward_union = find_upward_union_of_classes(dataset)
    dominating_set = find_dominating_set(dataset)
    dominated_set = find_dominated_set(dataset)
    lower_approx_downward_union = find_lower_approximation_of_decision_class(downward_union, dominated_set)
    upper_approx_downward_union = find_upper_approximation_of_decision_class(downward_union, dominated_set)
    boundaries_downward_union = find_boundaries(upper_approx_downward_union, lower_approx_downward_union, True)
    lower_approx_upward_union = find_lower_approximation_of_decision_class(upward_union, dominating_set)
    upper_approx_upward_union = find_upper_approximation_of_decision_class(upward_union, dominating_set)
    boundaries_upward_union = find_boundaries(upper_approx_upward_union, lower_approx_upward_union, False)
    approx_DRSA = {"lower_approx_downward_union": lower_approx_downward_union,
                   "upper_approx_downward_union": upper_approx_downward_union,
                   "boundaries_downward_union": boundaries_downward_union,
                   "lower_approx_upward_union": lower_approx_upward_union,
                   "upper_approx_upward_union": upper_approx_upward_union,
                   "boundaries_upward_union": boundaries_upward_union}
    stats = {'jakość klasyfikacji': calculate_quality_of_approximation_of_classification(dataset, boundaries_upward_union),
             'trafność dolne przybliżenie': calculate_accuracy_of_approximation_per_union(lower_approx_downward_union, upper_approx_downward_union),
             'trafność górne przybliżenie': calculate_accuracy_of_approximation_per_union(lower_approx_upward_union, upper_approx_upward_union)}
    data = {
        'unie': [downward_union, upward_union],
        'dominacja': [dominating_set, dominated_set],
        'approx': approx_DRSA,
        'stats': stats
    }
    if algorithm == 'DOMLEM':
        return DOMLEM_DRSA(approx_DRSA, dataset), data
    elif algorithm == 'DOMApriori':
        return DOMApriori_DRSA(), data

def run_VC_DRSA(dataset, algorithm, l):
    downward_union = find_downward_union_of_classes(dataset)
    upward_union = find_upward_union_of_classes(dataset)
    dominating_set = find_dominating_set(dataset)
    dominated_set = find_dominated_set(dataset)
    lower_approx_VC_downward = find_lower_approximation_of_decision_class_VC(downward_union, dominated_set, dataset,l)
    upper_approx_VC_upward = find_upper_approximation_of_decision_class_VC(dataset, upward_union, lower_approx_VC_downward)
    lower_approx_VC_upward = find_lower_approximation_of_decision_class_VC(upward_union, dominating_set, dataset, l)
    upper_approx_VC_downward = find_upper_approximation_of_decision_class_VC(dataset, downward_union, lower_approx_VC_upward)
    boundaries_downward_union_VC = find_boundaries(upper_approx_VC_downward, lower_approx_VC_downward, True)
    boundaries_upward_union_VC = find_boundaries(upper_approx_VC_upward, lower_approx_VC_upward, False)
    approx_VC_DRSA = {"lower_approx_downward_union": lower_approx_VC_downward,
                      "upper_approx_downward_union": upper_approx_VC_downward,
                      "boundaries_downward_union": boundaries_downward_union_VC,
                      "lower_approx_upward_union": lower_approx_VC_upward,
                      "upper_approx_upward_union": upper_approx_VC_upward,
                      "boundaries_upward_union": boundaries_upward_union_VC}
    stats = {'jakość klasyfikacji': calculate_quality_of_approximation_of_classification(dataset, boundaries_upward_union_VC),
             'trafność dolne przybliżenie': calculate_accuracy_of_approximation_per_union(lower_approx_VC_downward, upper_approx_VC_downward),
             'trafność górne przybliżenie': calculate_accuracy_of_approximation_per_union(lower_approx_VC_upward, upper_approx_VC_upward)}
    data = {
        'unie': [downward_union, upward_union],
        'dominacja': [dominating_set, dominated_set],
        'approx': approx_VC_DRSA,
        'stats': stats
    }
    if algorithm == 'DOMLEM':
        return DOMLEM_VC_DRSA(approx_VC_DRSA, dataset, l), data
    elif algorithm == 'DOMApriori':
        return DOMApriori_VC_DRSA(), data


def find_interesting_labels(labels_candidates, expected_result):
    tmp = {}
    cnt = []
    for lc in labels_candidates:
        tmp[lc] = expected_result.count(lc)
        cnt.append(tmp[lc])
    if max(cnt) / len(expected_result) > 0.6:
        return list(set(labels_candidates)-set([list(tmp.keys())[list(tmp.values()).index(max(cnt))]]))
    else:
        return labels_candidates


def leave_one_out_simple_classification(dataset_name, type, algorithm, l):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]}, 'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            (r, r_r, r_s), _ = run_VC_DRSA(f['train'], algorithm, l)
            result.append(classify_simple(f['test'], r['rule type 1/3'], classes)[0])
    else:
        for f in folds:
            (r, r_r, r_s), _ = run_DRSA(f['train'], algorithm)
            result.append(classify_simple(f['test'], r['rule type 1/3'], classes)[0])
    expected_result = [f['test'][-1] for f in folds]
    good = []
    bad = []
    bad_classified = []
    nones = []
    for e_r, r in zip(expected_result, result):
        if e_r == r:
            good.append(1)
        else:
            bad.append(1)
            if r is not None:
                bad_classified.append(1)
            else:
                nones.append(1)
    accuracy = float(sum(good)) / float(sum(good) + sum(bad))
    not_classified = len(nones) / len(result)
    correct = float(sum(good)) / float(sum(good) + sum(bad_classified))
    y_pred = []
    for r in result:
        if r is None:
            y_pred.append(0)
        else:
            y_pred.append(r)
    labels_candidates = list(set(expected_result))
    labels = find_interesting_labels(labels_candidates, expected_result)
    return accuracy, not_classified, correct, f1_score(expected_result, y_pred, labels=labels, average='weighted')


def leave_one_out_new_scheme_classification(dataset_name, type, algorithm, l):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]}, 'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            (r, r_r, r_s), _ = run_VC_DRSA(f['train'], algorithm, l)
            result.append(classify_new_scheme(f['test'], r['rule type 1/3'], classes, f['train'], downward_union, upward_union)[0])
    else:
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            (r, r_r, r_s), _ = run_DRSA(f['train'], algorithm)
            result.append(classify_new_scheme(f['test'], r['rule type 1/3'], classes, f['train'], downward_union, upward_union)[0])
    expected_result = [f['test'][-1] for f in folds]
    good = []
    bad = []
    bad_classified = []
    nones = []
    for e_r, r in zip(expected_result, result):
        if e_r == r:
            good.append(1)
        else:
            bad.append(1)
            if r is not None:
                bad_classified.append(1)
            else:
                nones.append(1)
    accuracy = float(sum(good))/float(sum(good)+sum(bad))
    not_classified = len(nones)/len(result)
    correct = float(sum(good))/float(sum(good)+sum(bad_classified))
    y_pred = []
    for r in result:
        if r is None:
            y_pred.append(0)
        else:
            y_pred.append(r)
    labels_candidates = list(set(expected_result))
    labels = find_interesting_labels(labels_candidates, expected_result)
    return accuracy, not_classified, correct, f1_score(expected_result, y_pred, labels=labels, average='weighted')


def find_best_model(dataset_name, range_, new_scheme, algorithm):
    best = []
    best_scores = (0.0, 0.0, 0.0, 0.0)
    error_range = []
    if new_scheme:
        for l in range_:
            try:
                accuracy, not_classified, correct, f1_score = leave_one_out_new_scheme_classification(dataset_name, 'VC-DRSA', algorithm, l)
            except:
                error_range.append(l)
                continue
            if f1_score > best_scores[3]:
                best_scores = (accuracy, not_classified, correct, f1_score)
                best = []
                best.append(l)
            elif f1_score == best_scores[3] and (accuracy, not_classified, correct, f1_score) == best_scores:
                best.append(l)
            elif f1_score == best_scores[3]:
                if accuracy > best_scores[0]:
                    best = []
                    best.append(l)
                    best_scores = (accuracy, not_classified, correct, f1_score)
                elif accuracy == best_scores[0]:
                    if correct > best_scores[2]:
                        best = []
                        best.append(l)
                        best_scores = (accuracy, not_classified, correct, f1_score)
                    elif correct == best_scores[2]:
                        if not_classified < best_scores[1]:
                            best = []
                            best.append(l)
                            best_scores = (accuracy, not_classified, correct, f1_score)
    else:
        for l in range_:
            try:
                accuracy, not_classified, correct, f1_score = leave_one_out_simple_classification(dataset_name, 'VC-DRSA', algorithm, l)
            except:
                error_range.append(l)
                continue
            if f1_score > best_scores[3]:
                best_scores = (accuracy, not_classified, correct, f1_score)
                best = []
                best.append(l)
            elif f1_score == best_scores[3] and (accuracy, not_classified, correct, f1_score) == best_scores:
                best.append(l)
            elif f1_score == best_scores[3]:
                if accuracy > best_scores[0]:
                    best = []
                    best.append(l)
                    best_scores = (accuracy, not_classified, correct, f1_score)
                elif accuracy == best_scores[0]:
                    if correct > best_scores[2]:
                        best = []
                        best.append(l)
                        best_scores = (accuracy, not_classified, correct, f1_score)
                    elif correct == best_scores[2]:
                        if not_classified < best_scores[1]:
                            best = []
                            best.append(l)
                            best_scores = (accuracy, not_classified, correct, f1_score)
    return best, best_scores


def prepare_report(data):
    return 0


def run_experiment_single(dataset_name, type, induction_algorithm, classification_algorithm):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    if type == 'DRSA':
        if classification_algorithm == 'old':
            (accuracy, not_classified, correct, f1_score) = leave_one_out_simple_classification(dataset_name, type, induction_algorithm, 1.0)
        elif classification_algorithm == 'new':
            (accuracy, not_classified, correct, f1_score) = leave_one_out_new_scheme_classification(dataset_name, type, induction_algorithm, 1.0)
        (r, r_r, r_s), d = run_DRSA(dataset, induction_algorithm)
    elif type == 'VC-DRSA':
        range_ = [0.05 * i for i in range(1, 20)]
        if classification_algorithm == 'old':
            b, _ = find_best_model(dataset_name, range_, False, induction_algorithm)
            if len(b) > 1:
                l = b[len(b)//2]
            else:
                l = b[0]
            (accuracy, not_classified, correct, f1_score) = leave_one_out_simple_classification(dataset_name, type, induction_algorithm, l)
        elif classification_algorithm == 'new':
            b, _ = find_best_model(dataset_name, range_, True, induction_algorithm)
            if len(b) > 1:
                l = b[len(b)//2]
            else:
                l = b[0]
            (accuracy, not_classified, correct, f1_score) = leave_one_out_new_scheme_classification(dataset_name, type, induction_algorithm, l)
        (r, r_r, r_s), d = run_VC_DRSA(dataset, induction_algorithm, l)
    data = {
        'zbiór': dataset,
        'uruchomienie': {'wersja algorytmu': type, 'algorytm indukcji reguł': induction_algorithm, 'algorytm klasyfikacji': classification_algorithm},
        'unie': d['unie'],
        'dominacja': d['dominacja'],
        'przybliżenia': d['approx'],
        'statystyki_przybliżeń': d['stats'],
        'reguły': r_r,
        'statystyki_reguł': [],
        'walidacja_krzyżowa': (accuracy, not_classified, correct, f1_score),
        'walidacja_krzyżowa_szczegóły': []
    }
    prepare_report(data)


def run_experiment_full():
    files = ['data\jose-medical-2017 (2).isf', 'data\jose-medical (1).isf']
    for file in files:
        print("Working on file: " + file)
        print('DRSA, DOMLEM, OLD')
        run_experiment_single(file, 'DRSA', 'DOMLEM', 'old')
        print('DRSA, DOMApriori, OLD')
        #run_experiment_single(file, 'DRSA', 'DOMApriori', 'old')
        print('DRSA, DOMLEM, NEW')
        run_experiment_single(file, 'DRSA', 'DOMLEM', 'new')
        print('DRSA, DOMApriori, NEW')
        #run_experiment_single(file, 'DRSA', 'DOMApriori', 'new')
        print('VC-DRSA, DOMLEM, OLD')
        run_experiment_single(file, 'VC-DRSA', 'DOMLEM', 'old')
        print('VC-DRSA, DOMApriori, OLD')
        #run_experiment_single(file, 'VC-DRSA', 'DOMApriori', 'old')
        print('VC-DRSA, DOMLEM, NEW')
        run_experiment_single(file, 'VC-DRSA', 'DOMLEM', 'new')
        print('VC-DRSA, DOMApriori, NEW')
        #run_experiment_single(file, 'VC-DRSA', 'DOMApriori', 'new')
        print("Done")


run_experiment_full()

# run_DRSA('data\exampleStef.isf')#'')#'jose-medical-2017 (2).isf')jose-medical (1).isf
# run_VC_DRSA('data\exampleStef.isf', 0.8)

#print("new")
#print(leave_one_out_new_scheme_classification('data\jose-medical-2017 (2).isf', 'DRSA', 1.0))
#print("simple")
#print(leave_one_out_simple_classification('data\jose-medical-2017 (2).isf', 'DRSA', 1.0))


