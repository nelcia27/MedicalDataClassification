from MedicalDataClassification.DRSA import *
from MedicalDataClassification.rules_induction import *
from MedicalDataClassification.classification import *
from MedicalDataClassification.helpers import *
from sklearn.metrics import f1_score
import pandas as pd


def run_DRSA(dataset, algorithm, max_length, min_support):
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
        return DOMApriori_DRSA(approx_DRSA, dataset, max_length, min_support), data


def run_VC_DRSA(dataset, algorithm, l, max_length, min_support):
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
        return DOMApriori_VC_DRSA(approx_VC_DRSA, dataset, l, max_length, min_support), data


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


def leave_one_out_simple_classification(dataset_name, type, algorithm, l, max_length, min_support):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]}, 'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            (r, r_r, r_s), _ = run_VC_DRSA(f['train'], algorithm, l, max_length, min_support)
            result.append(classify_simple(f['test'], r['rule type 1/3'], classes)[0])
    else:
        for f in folds:
            (r, r_r, r_s), _ = run_DRSA(f['train'], algorithm, max_length, min_support)
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


def leave_one_out_new_scheme_classification(dataset_name, type, algorithm, l, max_length, min_support):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]}, 'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            (r, r_r, r_s), _ = run_VC_DRSA(f['train'], algorithm, l, max_length, min_support)
            result.append(classify_new_scheme(f['test'], r['rule type 1/3'], classes, f['train'], downward_union, upward_union)[0])
    else:
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            (r, r_r, r_s), _ = run_DRSA(f['train'], algorithm, max_length, min_support)
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


def find_best_model(dataset_name, range_, new_scheme, algorithm, max_length, min_support):
    best = []
    best_scores = (0.0, 0.0, 0.0, 0.0)
    error_range = []
    if new_scheme:
        for l in range_:
            try:
                accuracy, not_classified, correct, f1_score = leave_one_out_new_scheme_classification(dataset_name, 'VC-DRSA', algorithm, l, max_length, min_support)
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
                accuracy, not_classified, correct, f1_score = leave_one_out_simple_classification(dataset_name, 'VC-DRSA', algorithm, l, max_length, min_support)
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
    print("ERRORS: ", error_range)
    return best, best_scores


def leave_one_out_old_full(dataset_name, type, algorithm, l, max_length, min_support):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]},
              'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            (r, r_r, r_s), _ = run_VC_DRSA(f['train'], algorithm, l, max_length, min_support)
            result.append(classify_simple(f['test'], r['rule type 1/3'], classes)[0])
    else:
        for f in folds:
            (r, r_r, r_s), _ = run_DRSA(f['train'], algorithm, max_length, min_support)
            result.append(classify_simple(f['test'], r['rule type 1/3'], classes)[0])
    expected_result = [o[-1] for o in dataset['objects']]
    return result, expected_result


def leave_one_out_new_full(dataset_name, type, algorithm, l, max_length, min_support):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    classes = find_all_possible_decision_classes(dataset)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]},
              'test': o} for o in dataset['objects']]
    result = []
    if type == "VC-DRSA":
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            (r, r_r, r_s), _ = run_VC_DRSA(f['train'], algorithm, l, max_length, min_support)
            result.append(classify_new_scheme(f['test'], r['rule type 1/3'], classes, f['train'], downward_union, upward_union)[0])
    else:
        for f in folds:
            downward_union = find_downward_union_of_classes(dataset)
            upward_union = find_upward_union_of_classes(dataset)
            (r, r_r, r_s), _ = run_DRSA(f['train'], algorithm, max_length, min_support)
            result.append(classify_new_scheme(f['test'], r['rule type 1/3'], classes, f['train'], downward_union, upward_union)[0])
    expected_result = [o[-1] for o in dataset['objects']]
    return result, expected_result


def calculate_rules_stats(rules, approx, size):
    stats = {}
    cnt = 1
    for rule in rules:
        if rule['class'][6] == '<':
            val = len(approx["lower_approx_downward_union"][int(rule['class'][-1])-1]['objects'])
        elif rule['class'][6] == '>':
            val = len(approx["lower_approx_upward_union"][int(rule['class'][-1])-2]['objects'])
        if val != 0:
            stats[cnt] = {'support': len(rule['covered']),
                          'strength': len(rule['covered'])/size,
                          'coverage': len(rule['covered'])/val}
        else:
            stats[cnt] = {'support': len(rule['covered']),
                          'strength': len(rule['covered']) / size,
                          'coverage': 0}
        cnt += 1
    return stats


def prepare_report(data, name, num_class, num_attributes):
    with pd.ExcelWriter('results\DRSA\File'+name+'.xlsx', engine='xlsxwriter') as writer:
        workbook = writer.book
        for s in ['Zbiór', 'Dane uruchomienia', 'Unie w dół', 'Unie w górę', 'Dominujące', 'Zdominowane', 'Przybliżenia unii', 'Reguły', 'Statystyki reguł', 'Walidacja krzyżowa']:
            worksheet = workbook.add_worksheet(s)
            writer.sheets[s] = worksheet
        df = pd.DataFrame(data['zbiór']['attributes'])
        df.to_excel(writer, sheet_name='Zbiór', index=False)
        df1 = pd.DataFrame(data['zbiór']['objects'])
        df1.to_excel(writer, sheet_name='Zbiór', startrow=num_attributes + 1, startcol=0, index=False)
        param = list(data['uruchomienie'].keys())
        vals = list(data['uruchomienie'].values())
        df2 = pd.DataFrame({'parametr':param, 'wartość':vals})
        df2.to_excel(writer, sheet_name='Dane uruchomienia', index=False)
        dw_un = [(list(d.keys())[0]) for d in data['unie'][0]]
        dw_un_ = [d[list(d.keys())[0]][1] for d in data['unie'][0]]
        up_un = [(list(d.keys())[0]) for d in data['unie'][1]]
        up_un_ = [d[list(d.keys())[0]][1] for d in data['unie'][1]]
        df3 = pd.DataFrame({"Unie": dw_un, "Obiekty": dw_un_})
        df3.to_excel(writer, sheet_name='Unie w dół', index=False)
        df4 = pd.DataFrame({"Unie": up_un, "Obiekty": up_un_})
        df4.to_excel(writer, sheet_name='Unie w górę', index=False)
        df5 = pd.DataFrame(data['dominacja'][0])
        df5 = df5.drop('objects', 1)
        df5.to_excel(writer, sheet_name='Dominujące', index=False)
        df6 = pd.DataFrame(data['dominacja'][1])
        df6 = df6.drop('objects', 1)
        df6.to_excel(writer, sheet_name='Zdominowane', index=False)
        k = list(data['przybliżenia'].keys())
        for i, k_ in enumerate(k):
            df_ = pd.DataFrame(data['przybliżenia'][k_])
            worksheet = writer.sheets['Przybliżenia unii']
            worksheet.write(i*num_class, 0, k_)
            df_.to_excel(writer, sheet_name='Przybliżenia unii', startrow=i*num_class + 1, startcol=0, index=False)
        df7 = pd.DataFrame(data['statystyki_przybliżeń'])
        worksheet = writer.sheets['Przybliżenia unii']
        worksheet.write(len(k) * num_class + 1, 0, 'Statystyki')
        df7.to_excel(writer, sheet_name='Przybliżenia unii', startrow=len(k)*num_class + 2, startcol=0, index=False)
        r = ['Reguła ' + str(i+1) for i in range(len(data['reguły']['rule type 1/3']))]
        df8 = pd.DataFrame({'Oznaczenie reguły': r, 'Reguła': data['reguły']['rule type 1/3']})
        df8.to_excel(writer, sheet_name='Reguły', index=False)
        df9 = pd.DataFrame(data['statystyki_reguł']).transpose()
        df9.to_excel(writer, sheet_name='Statystyki reguł', index=False)
        cv = {(k,v) for k,v in zip(['accuracy', 'not_classified', 'correct', 'f1_score'], data['walidacja_krzyżowa'])}
        df9 = pd.DataFrame(cv)
        df9.to_excel(writer, sheet_name='Walidacja krzyżowa', index=False, header=False)
        obj = ['a' + str(o) for o in range(len(data['walidacja_krzyżowa_szczegóły'][0]))]
        df10 = pd.DataFrame({'Obiekt': obj, 'Przewidziana klasa': data['walidacja_krzyżowa_szczegóły'][0],
                           'Prawdziwa klasa': data['walidacja_krzyżowa_szczegóły'][1]})
        df10.to_excel(writer, sheet_name='Walidacja krzyżowa szczegóły', index=False)


def prepare_report_latex(data):
    with pd.option_context("max_colwidth", 1000):
        dw_un = [(list(d.keys())[0]) for d in data['unie'][0]]
        dw_un_ = [d[list(d.keys())[0]][1] for d in data['unie'][0]]
        up_un = [(list(d.keys())[0]) for d in data['unie'][1]]
        up_un_ = [d[list(d.keys())[0]][1] for d in data['unie'][1]]
        df3 = pd.DataFrame({"Unie": dw_un, "Obiekty": dw_un_})
        print("unie w dół")
        print(df3.to_latex( index=False))
        df4 = pd.DataFrame({"Unie": up_un, "Obiekty": up_un_})
        print("unie w górę")
        print(df4.to_latex(index=False))
        df5 = pd.DataFrame(data['dominacja'][0])
        df5 = df5.drop('objects', 1)
        print("domunujące")
        print(df5.to_latex(index=False))
        print("zdominowane")
        df6 = pd.DataFrame(data['dominacja'][1])
        df6 = df6.drop('objects', 1)
        print(df6.to_latex(index=False))
        k = list(data['przybliżenia'].keys())
        print("przybliżenia unii")
        for i, k_ in enumerate(k):
            print(k)
            df_ = pd.DataFrame(data['przybliżenia'][k_])
            print(df_.to_latex(index=False))
        print("statystyki przybliżeń")
        df7 = pd.DataFrame(data['statystyki_przybliżeń'])
        print(df7.to_latex(index=False))
        print("reguły")
        r = ['Reguła ' + str(i+1) for i in range(len(data['reguły']['rule type 1/3']))]
        df8 = pd.DataFrame({'Oznaczenie reguły': r, 'Reguła': data['reguły']['rule type 1/3']})
        print(df8.to_latex(index=False))
        print("statystyki reguł")
        df9 = pd.DataFrame(data['statystyki_reguł']).transpose()
        print(df9.to_latex(index=False))


def run_experiment_single(dataset_name, type, induction_algorithm, classification_algorithm, name, max_length, min_support):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    if type == 'DRSA':
        l = 1.0
        if classification_algorithm == 'old':
            (accuracy, not_classified, correct, f1_score) = leave_one_out_simple_classification(dataset_name, type, induction_algorithm, 1.0, max_length, min_support)
            details = leave_one_out_old_full(dataset_name, type, induction_algorithm, 1.0, max_length, min_support)
        elif classification_algorithm == 'new':
            (accuracy, not_classified, correct, f1_score) = leave_one_out_new_scheme_classification(dataset_name, type, induction_algorithm, 1.0, max_length, min_support)
            details = leave_one_out_new_full(dataset_name, type, induction_algorithm, 1.0, max_length, min_support)
        (r, r_r, r_s), d = run_DRSA(dataset, induction_algorithm, max_length, min_support)
    elif type == 'VC-DRSA':
        range_ = [0.05 * i for i in range(1, 20)]
        if classification_algorithm == 'old':
            b, _ = find_best_model(dataset_name, range_, False, induction_algorithm, max_length, min_support)
            if len(b) > 1:
                l = b[len(b)//2]
            else:
                l = b[0]
            (accuracy, not_classified, correct, f1_score) = leave_one_out_simple_classification(dataset_name, type, induction_algorithm, l, max_length, min_support)
            details = leave_one_out_old_full(dataset_name, type, induction_algorithm, l, max_length, min_support)
        elif classification_algorithm == 'new':
            b, _ = find_best_model(dataset_name, range_, True, induction_algorithm, max_length, min_support)
            if len(b) > 1:
                l = b[len(b)//2]
            else:
                l = b[0]
            (accuracy, not_classified, correct, f1_score) = leave_one_out_new_scheme_classification(dataset_name, type, induction_algorithm, l, max_length, min_support)
            details = leave_one_out_new_full(dataset_name, type, induction_algorithm, l, max_length, min_support)
        (r, r_r, r_s), d = run_VC_DRSA(dataset, induction_algorithm, l, max_length, min_support)
    data = {
        'zbiór': dataset,
        'uruchomienie': {'wersja algorytmu': type, 'algorytm indukcji reguł': induction_algorithm, 'algorytm klasyfikacji': classification_algorithm, 'poziom spójności': l},
        'unie': d['unie'],
        'dominacja': d['dominacja'],
        'przybliżenia': d['approx'],
        'statystyki_przybliżeń': d['stats'],
        'reguły': r_r,
        'statystyki_reguł': calculate_rules_stats(r_s['rule type 1/3'], d['approx'], len(dataset['objects'])),
        'walidacja_krzyżowa': (accuracy, not_classified, correct, f1_score),
        'walidacja_krzyżowa_szczegóły': details
    }
    prepare_report(data, name, len(find_all_possible_decision_classes(dataset))+1, len(data['zbiór']['attributes']))


def run_experiment_single_latex(dataset_name, type, induction_algorithm, classification_algorithm, max_length, min_support, l, to_test):
    a, p, o = read_dataset(dataset_name)
    dataset = prepare_dataset(a, p, o)
    folds = [{'train': {"attributes": dataset['attributes'], "objects": [o1 for o1 in dataset['objects'] if o1 != o]},
              'test': o} for o in dataset['objects']]
    dataset = folds[to_test]["train"]
    classes = find_all_possible_decision_classes(dataset)
    test = folds[to_test]["test"]
    if type == 'DRSA':
        downward_union = find_downward_union_of_classes(dataset)
        upward_union = find_upward_union_of_classes(dataset)
        (r, r_r, r_s), d = run_DRSA(dataset, induction_algorithm, max_length, min_support)
        classify_new_scheme(test, r['rule type 1/3'], classes, dataset, downward_union, upward_union)
    elif type == 'VC-DRSA':
        downward_union = find_downward_union_of_classes(dataset)
        upward_union = find_upward_union_of_classes(dataset)
        (r, r_r, r_s), d = run_VC_DRSA(dataset, induction_algorithm, l, max_length, min_support)
        classify_new_scheme(test, r['rule type 1/3'], classes, dataset, downward_union, upward_union)
    data = {
        'zbiór': dataset,
        'uruchomienie': {'wersja algorytmu': type, 'algorytm indukcji reguł': induction_algorithm, 'algorytm klasyfikacji': classification_algorithm, 'poziom spójności': l},
        'unie': d['unie'],
        'dominacja': d['dominacja'],
        'przybliżenia': d['approx'],
        'statystyki_przybliżeń': d['stats'],
        'reguły': r_r,
        'statystyki_reguł': calculate_rules_stats(r_s['rule type 1/3'], d['approx'], len(dataset['objects'])),
    }
    #prepare_report_latex(data)


def run_experiment_full(max_length, min_support):
    files = ['data\jose-medical-2017 (2).isf', 'data\jose-medical (1).isf']
    for i, file in enumerate(files):
        print("Working on file: " + file)
        print('DRSA, DOMLEM, OLD')
        #run_experiment_single(file, 'DRSA', 'DOMLEM', 'old', 'DRSA_DOMLEM_OLD'+str(i), max_length, min_support)
        print('DRSA, DOMApriori, OLD')
        run_experiment_single(file, 'DRSA', 'DOMApriori', 'old', 'DRSA_DOMApriori_OLD'+str(i), max_length, min_support)
        print('DRSA, DOMLEM, NEW')
        #run_experiment_single(file, 'DRSA', 'DOMLEM', 'new', 'DRSA_DOMLEM_NEW'+str(i), max_length, min_support)
        print('DRSA, DOMApriori, NEW')
        run_experiment_single(file, 'DRSA', 'DOMApriori', 'new', 'DRSA_DOMApriori_NEW'+str(i), max_length, min_support)
        print('VC-DRSA, DOMLEM, OLD')
        #run_experiment_single(file, 'VC-DRSA', 'DOMLEM', 'old', 'VC_DRSA_DOMLEM_OLD'+str(i), max_length, min_support)
        print('VC-DRSA, DOMApriori, OLD')
        run_experiment_single(file, 'VC-DRSA', 'DOMApriori', 'old', 'VC_DRSA_DOMApriori_OLD'+str(i), max_length, min_support)
        print('VC-DRSA, DOMLEM, NEW')
        #run_experiment_single(file, 'VC-DRSA', 'DOMLEM', 'new', 'VC_DRSA_DOMLEM_NEW'+str(i), max_length, min_support)
        print('VC-DRSA, DOMApriori, NEW')
        run_experiment_single(file, 'VC-DRSA', 'DOMApriori', 'new', 'VC_DRSA_DOMApriori_NEW'+str(i), max_length, min_support)
        print("Done")


def run_experiment_full_latex(max_length, min_support, l, to_test):
    files = ['data\jose-medical (1).isf']#'data\jose-medical-2017 (2).isf',
    for i, file in enumerate(files):
        print("Working on file: " + file)
        print('DRSA, DOMLEM, OLD')
        run_experiment_single_latex(file, 'VC-DRSA', 'DOMApriori', 'new', max_length, min_support, l, to_test)
        print('DRSA, DOMApriori, OLD')
        #run_experiment_single(file, 'DRSA', 'DOMApriori', 'old', 'DRSA_DOMApriori_OLD'+str(i), max_length, min_support)
        print('DRSA, DOMLEM, NEW')
        #run_experiment_single(file, 'DRSA', 'DOMLEM', 'new', 'DRSA_DOMLEM_NEW'+str(i), max_length, min_support)
        print('DRSA, DOMApriori, NEW')
        #run_experiment_single(file, 'DRSA', 'DOMApriori', 'new', 'DRSA_DOMApriori_NEW'+str(i), max_length, min_support)
        print('VC-DRSA, DOMLEM, OLD')
        #run_experiment_single(file, 'VC-DRSA', 'DOMLEM', 'old', 'VC_DRSA_DOMLEM_OLD'+str(i), max_length, min_support)
        print('VC-DRSA, DOMApriori, OLD')
        #run_experiment_single(file, 'VC-DRSA', 'DOMApriori', 'old', 'VC_DRSA_DOMApriori_OLD'+str(i), max_length, min_support)
        print('VC-DRSA, DOMLEM, NEW')
        #run_experiment_single(file, 'VC-DRSA', 'DOMLEM', 'new', 'VC_DRSA_DOMLEM_NEW'+str(i), max_length, min_support)
        print('VC-DRSA, DOMApriori, NEW')
        #run_experiment_single(file, 'VC-DRSA', 'DOMApriori', 'new', 'VC_DRSA_DOMApriori_NEW'+str(i), max_length, min_support)
        print("Done")


#run_experiment_full(3, 1)
run_experiment_full_latex(3, 1, 0.8, 2)
# run_DRSA('data\exampleStef.isf')#'')#'jose-medical-2017 (2).isf')jose-medical (1).isf

#a, p, o = read_dataset('data\jose-medical-2017 (2).isf')
#dataset = prepare_dataset(a, p, o)
#(_, r, r_s), d = run_DRSA(dataset, "DOMApriori", 3, 2)
#print(r_s)
#print(r['rule type 1/3'])
#print(r['rule type 2/4'])
#print(_)
#print(len(r['rule type 1/3']))
#print(len(r['rule type 2/4']))

#print("VC")
#(_, r, r_s), d = run_VC_DRSA(dataset, "DOMApriori", 0.8, 3, 2)
#print(r_s)
#print(r['rule type 1/3'])
#print(_)
#print(len(r['rule type 1/3']))



