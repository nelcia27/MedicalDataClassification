import itertools


def find_all_possible_decision_classes(dataset):
    decision_classes = []
    for obj in dataset["objects"]:
        decision_classes.append(obj[-1])
    return list(set(decision_classes))


def find_upward_union_of_classes(dataset):
    all_possible_decision_classes = find_all_possible_decision_classes(dataset)
    downward_union_of_classes = []
    for class_ in all_possible_decision_classes[1:]:
        name = "at least class " + str(int(class_))
        tmp = [obj for obj in dataset["objects"] if obj[-1] >= class_]
        tmp_ = [t[0] for t in tmp]
        downward_union_of_classes.append({name: [tmp, tmp_]})
    return downward_union_of_classes


def find_downward_union_of_classes(dataset):
    all_possible_decision_classes = find_all_possible_decision_classes(dataset)
    upward_union_of_classes = []
    for class_ in all_possible_decision_classes[:-1]:
        name = "at most class " + str(int(class_))
        tmp = [obj for obj in dataset["objects"] if obj[-1] <= class_]
        tmp_ = [t[0] for t in tmp]
        upward_union_of_classes.append({name: [tmp, tmp_]})
    return upward_union_of_classes


def is_better(obj1, obj2, preference_type):
    return all(((x >= y and pt == 'gain') or (x <= y and pt == 'cost')) for x, y, pt in zip(obj1, obj2, preference_type))


def is_worst(obj1, obj2, preference_type):
    return all(((x <= y and pt == 'gain') or (x >= y and pt == 'cost')) for x, y, pt in zip(obj1, obj2, preference_type))


def get_preferences(dataset):
    return [crit['preference'] for crit in dataset['attributes']]


def find_dominating_set(dataset):
    # Zbiór wariantów dominujących (lepszych niż nasz) +
    dominating_set = []
    preferences = get_preferences(dataset)
    for obj in dataset['objects']:
        objects = [o for o in dataset['objects'] if is_better(o[1:-1], obj[1:-1], preferences[:-1])]
        dominating_set.append({'object': obj[0], 'dominance': [o[0] for o in objects], 'objects': objects})
    return dominating_set


def find_dominated_set(dataset):
    # Zbiór wariantów zdominowanych (gorszych od naszego) -
    dominated_set = []
    preferences = get_preferences(dataset)
    for obj in dataset['objects']:
        objects = [o for o in dataset['objects'] if is_worst(o[1:-1], obj[1:-1], preferences[:-1])]
        dominated_set.append({'object': obj[0], 'dominance': [o[0] for o in objects], 'objects': objects})
    return dominated_set


# standard DRSA
def find_lower_approximation_of_decision_class(unions, d_set):
    lower_approximation = []
    for union in unions:
        un = list(union.values())
        un_class = set([u for u in un[0][1]])
        tmp = []
        for d in d_set:
            if un_class.issuperset(set(d['dominance'])):
                tmp.append(d['dominance'])
        tmp_ = list(set(list(itertools.chain(*tmp))))
        tmp_prim = sorted([int(elem[1:]) for elem in tmp_])
        tmp_prim_ = ["a"+str(i) for i in tmp_prim]
        lower_approximation.append({'union': list(union.keys())[0], 'objects': tmp_prim_})
    return lower_approximation


def find_upper_approximation_of_decision_class(unions, d_set):
    upper_approximation = []
    for union in unions:
        un = list(union.values())
        un_class = set([u for u in un[0][1]])
        tmp = []
        for d in d_set:
            if d['object'] in un_class:
                tmp.append(d['dominance'])
        tmp_ = list(set(list(itertools.chain(*tmp))))
        tmp_prim = sorted([int(elem[1:]) for elem in tmp_])
        tmp_prim_ = ["a"+str(i) for i in tmp_prim]
        upper_approximation.append({'union': list(union.keys())[0], 'objects': tmp_prim_})
    return upper_approximation


def find_boundaries(upper_approx, lower_approx, is_downward):
    boundaries = []
    for i in range(len(upper_approx)):
        tmp = list(set(upper_approx[i]['objects'])-set(lower_approx[i]['objects']))
        tmp_prim = sorted([int(elem[1:]) for elem in tmp])
        tmp_prim_ = ["a"+str(i) for i in tmp_prim]
        if is_downward:
            boundaries.append({'class': i+1, 'objects': tmp_prim_})
        else:
            boundaries.append({'class': i+2, 'objects': tmp_prim_})
    return boundaries


# VC-DRSA
# find_boundaries same as in DRSA
def calculate_consistency(dominance_cone, union):
    d_p = set(dominance_cone['dominance'])
    intersection_cnt = len(list(d_p.intersection(union)))
    all_cnt = len(dominance_cone['dominance'])
    return float(intersection_cnt) / float(all_cnt)


def is_consistency_threshold_reached(dominance_cone, union, threshold):
    consistency = calculate_consistency(dominance_cone, union)
    if consistency >= threshold and dominance_cone['object'] in union:
        return True
    else:
        return False


# THRESHOLD (0,1]
def find_lower_approximation_of_decision_class_VC(unions, d_set, dataset, threshold):
    lower_approximation = []
    preference_type = get_preferences(dataset)
    for union in unions:
        un = list(union.values())
        un_class = set([u for u in un[0][1]])
        tmp = []
        for d in d_set:
            if is_consistency_threshold_reached(d, un_class, threshold):
                tmp.append(d['object'])
        tmp_prim = sorted([int(elem[1:]) for elem in tmp])
        tmp_prim_ = ["a" + str(i) for i in tmp_prim]
        lower_approximation.append({'union': list(union.keys())[0], 'objects': tmp_prim_})
    return lower_approximation


def find_upper_approximation_of_decision_class_VC(dataset, unions, lower_approx):
    upper_approximation = []
    for union, l_a in zip(unions, lower_approx):
        U = set([u[0] for u in dataset['objects']])
        tmp = U.difference(set(l_a['objects']))
        tmp_prim = sorted([int(elem[1:]) for elem in tmp])
        tmp_prim_ = ["a" + str(i) for i in tmp_prim]
        upper_approximation.append({'union': list(union.keys())[0], 'objects': tmp_prim_})
    return upper_approximation





