from DRSA import *
from rules_induction import *
from classification import *
from helpers import *
from sklearn.metrics import f1_score

# a, p, o = read_dataset(dataset_name)
# dataset = prepare_dataset(a, p, o)


def run_DRSA(dataset):
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
    calculate_quality_of_approximation_of_classification(dataset, boundaries_upward_union)
    calculate_quality_of_approximation_of_classification(dataset, boundaries_downward_union)
    calculate_accuracy_of_approximation_per_union(lower_approx_downward_union, upper_approx_downward_union)
    calculate_accuracy_of_approximation_per_union(lower_approx_upward_union, upper_approx_upward_union)
    return DOMLEM_DRSA(approx_DRSA, dataset) # rules, rules_readable, rules_to_stats =


def run_VC_DRSA(dataset, l):
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
    calculate_quality_of_approximation_of_classification(dataset, boundaries_upward_union_VC)
    calculate_quality_of_approximation_of_classification(dataset, boundaries_downward_union_VC)
    calculate_accuracy_of_approximation_per_union(lower_approx_VC_downward, upper_approx_VC_downward)
    calculate_accuracy_of_approximation_per_union(lower_approx_VC_upward, upper_approx_VC_upward)
    return DOMLEM_VC_DRSA(approx_VC_DRSA, dataset, l)


def find_interesting_labels(labels_candidates, expected_result):
    tmp = {}
    cnt = []
    for lc in labels_candidates:
        tmp[lc] = expected_result.count(lc)
        cnt.append(tmp[lc])
    if max(cnt)/ len(expected_result) > 0.6:
        return list(set(labels_candidates)-set(list(tmp.values()).index(max(cnt))))
    else:
        return labels_candidates

def leave_one_out_simple_classification(dataset_name, type, l):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]}, 'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            r, r_r, r_s = run_VC_DRSA(f['train'], l)
            result.append(classify_simple(f['test'], r['rule type 1/3'], classes)[0])
    else:
        for f in folds:
            r, r_r, r_s = run_DRSA(f['train'])
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


def leave_one_out_new_scheme_classification(dataset_name, type, l):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]}, 'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            r, r_r, r_s = run_VC_DRSA(f['train'], l)
            result.append(classify_new_scheme(f['test'], r['rule type 1/3'], classes, f['train'], downward_union, upward_union)[0])
    else:
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            r, r_r, r_s = run_DRSA(f['train'])
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


def find_best_model(dataset_name, range, new_scheme):
    best = None
    best_scores = (0.0, 0.0, 0.0, 0.0)
    if new_scheme:
        for l in range:
            accuracy, not_classified, correct, f1_score = leave_one_out_new_scheme_classification(dataset_name, 'VC-DRSA', l)

    else:
        for l in range:
            accuracy, not_classified, correct, f1_score = leave_one_out_simple_classification(dataset_name, 'VC-DRSA', l)

    return best, best_scores


def prepare_report(dataset_name):
    #drsa clasyf1
    #drsa clasyf2
    #vc drsa naj clasyf1
    #vcdrsa naj clasyf2
    # drsa clasyf1 domlem
    # drsa clasyf2 domlem
    # vc drsa naj clasyf1 domlem
    # vcdrsa naj clasyf2 domlem
    return 0


def run_experiment():
    files = ['data\jose-medical-2017 (2).isf', 'data\jose-medical (1).isf']
    for file in files:
        print("Working on file: " + file)
        prepare_report(file)
        print("Done")




# run_DRSA('data\exampleStef.isf')#'')#'jose-medical-2017 (2).isf')jose-medical (1).isf
# run_VC_DRSA('data\exampleStef.isf', 0.8)

#print("new")
#print(leave_one_out_new_scheme_classification('data\jose-medical-2017 (2).isf', 'DRSA', 1.0))
#print("simple")
#print(leave_one_out_simple_classification('data\jose-medical-2017 (2).isf', 'DRSA', 1.0))


