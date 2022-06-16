from collections import ChainMap


def classify_simple(object, rules, classes):
    matched = []
    object_ = object[1:-1]
    for rule in rules:
        match = True
        if rule is not None:
            for part in rule[:-2]:
                if part['sign'][0] is '>':
                    if object_[part['criterion']] < part['condition']:
                        match = False
                        break
                else:
                    if object_[part['criterion']] > part['condition']:
                        match = False
                        break
            if match:
                matched.append(rule[-3]['class'])
    assigned_class = []
    if len(matched) == 0:
        assigned_class.append(None)
    elif len(matched) == 1:
        assigned_class.append(int(matched[0][-1]))
    elif len(matched) > 1:
        at_least = list(set([int(a[-1]) for a in matched if a[:-1] == 'at least class ']))
        at_most = list(set([int(a[-1]) for a in matched if a[:-1] == 'at most class ']))
        if len(at_least) > 0 and len(at_most) == 0:
            assigned_class.append(min(at_least))
        elif len(at_least) == 0 and len(at_most) > 0:
            assigned_class.append(max(at_most))
        elif len(at_least) > 0 and len(at_most) > 0:
            a_m = max(at_most)
            a_l = min(at_least)
            if a_m == a_l:
                assigned_class.append(a_l)
            elif a_l < a_m:
                #while a_l <= a_m:
                assigned_class.append(a_m)
                #a_l += 1
            else:
                #while a_m <= a_l:
                assigned_class.append(None)
                #a_m += 1
    return assigned_class


def find_cond_p(dataset, rule):
    matched = []
    for o in dataset['objects']:
        match = True
        for part in rule:
            if part['sign'][0] is '>':
                if part['condition'] > o[part['criterion'] + 1]:
                    match = False
            else:
                if part['condition'] < o[part['criterion'] + 1]:
                    match = False
        if match:
            matched.append(o[0])
    return matched


def find_cl(dataset, clas):
    matched = []
    for o in dataset['objects']:
        if o[-1] == clas:
            matched.append(o[0])
    return matched


def find_interest_con_cordant(classes, c):
    res = []
    for cl in classes[1:]:
        if cl <= c:
            res.append('at least class ' + str(int(cl)))
    for cl in classes[:-1]:
        if cl >= c:
            res.append('at most class ' + str(int(cl)))
    return res


def find_interest_dis_cordant(classes, c):
    res = []
    for cl in classes[1:]:
        if cl > c:
            res.append('at least class ' + str(int(cl)))
    for cl in classes[:-1]:
        if cl < c:
            res.append('at most class ' + str(int(cl)))
    return res


def find_dis_con_cordant_rules(c, rules, classes):
    concordant_r = []
    discordant_r = []
    interest_con = find_interest_con_cordant(classes, c)
    interest_dis = find_interest_dis_cordant(classes, c)
    for r in rules:
        if r[0]['class'] in interest_con:
            concordant_r.append(r)
        elif r[0]['class'] in interest_dis:
            discordant_r.append(r)
    return concordant_r, discordant_r


def calculate_score_plus(dataset, c, rules, classes):
    concordant_r, _ = find_dis_con_cordant_rules(c, rules, classes)
    cl = find_cl(dataset, c)
    cond_ = []
    cond_cl = []
    for r in concordant_r:
        cond_r = find_cond_p(dataset, r)
        p = list(set(cond_r) & set(cl))
        for el in p:
            if el not in cond_cl:
                cond_cl.append(el)
        for el in cond_r:
            if el not in cond_:
                cond_.append(el)
    part = len(cond_cl)
    if float(len(cond_) * len(cl)) != 0.0:
        return float(part * part)/float(len(cond_) * len(cl))
    else:
        return 0.0


def calculate_score_minus(dataset, c, rules, classes, downward_unions, upward_unions):
    _, disconcordant_r = find_dis_con_cordant_rules(c, rules, classes)
    cond_ = []
    cond_cl = []
    cl_ = []
    for r in disconcordant_r:
        if r[0]['class'][:-1] == 'at least class ':
            cl = upward_unions[r[0]['class']][1]
        else:
            cl = downward_unions[r[0]['class']][1]
        cond_r = find_cond_p(dataset, r)
        p = list(set(cond_r) & set(cl))
        for el in p:
            if el not in cond_cl:
                cond_cl.append(el)
        for el in cond_r:
            if el not in cond_:
                cond_.append(el)
        for el in cl:
            if el not in cl_:
                cl_.append(el)
    part = len(cond_cl)
    if float(len(cond_) * len(cl_)) != 0.0:
        return float(part * part) / float(len(cond_) * len(cl_))
    else:
        return 0.0


def classify_new_scheme(object, rules, classes, dataset, downward_unions, upward_unions):
    matched = []
    rules_matched = []
    object_ = object[1:-1]
    downward_unions = dict(ChainMap(*downward_unions))
    upward_unions = dict(ChainMap(*upward_unions))
    for rule in rules:
        match = True
        if rule is not None:
            for part in rule[:-2]:
                if part['sign'][0] is '>':
                    if object_[part['criterion']] < part['condition']:
                        match = False
                        break
                else:
                    if object_[part['criterion']] > part['condition']:
                        match = False
                        break
            if match:
                matched.append(rule[-3]['class'])
                rules_matched.append(rule[:-2])
    assigned_class = []
    if len(matched) == 0:
        assigned_class.append(None)
    elif len(matched) == 1:
        cond_p = find_cond_p(dataset, rules_matched[0])
        max_score = 0
        best_class = []
        for c in classes:
            cl = find_cl(dataset, c)
            cond_p_cl = len(list(set(cond_p) & set(cl)))
            score = float(cond_p_cl * cond_p_cl)/float(len(cl) * len(cond_p))
            if score > max_score or len(best_class) == 0:
                max_score = score
                best_class = []
                best_class.append(c)
            elif score == max_score:
                best_class.append(c)
        if len(best_class) > 1:
            assigned_class.append(None)
        else:
            assigned_class.append(best_class[0])
    else:
        max_score = 0
        best_class = []
        for c in classes:
            score = calculate_score_plus(dataset, c, rules_matched, classes) - calculate_score_minus(dataset, c, rules_matched, classes, downward_unions, upward_unions)
            if score > max_score or len(best_class) == 0:
                max_score = score
                best_class = []
                best_class.append(c)
            elif score == max_score:
                best_class.append(c)
        if len(best_class) > 1:
            assigned_class.append(None)
        else:
            assigned_class.append(best_class[0])
    return assigned_class


