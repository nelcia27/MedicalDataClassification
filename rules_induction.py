import itertools


def check_acceptance_element(candidate, dataset):
    if candidate['criterion'] is not None:
        if candidate['rule_type'] == 1 or candidate['rule_type'] == 2:
            if candidate['preference'] == 'gain':
                return list(o[0] for o in dataset if o[candidate['criterion'] + 1] >= candidate['condition'])
            else:
                return list(o[0] for o in dataset if o[candidate['criterion'] + 1] <= candidate['condition'])
        elif candidate['rule_type'] == 3 or candidate['rule_type'] == 4:
            if candidate['preference'] == 'gain':
                return list(o[0] for o in dataset if o[candidate['criterion'] + 1] <= candidate['condition'])
            else:
                return list(o[0] for o in dataset if o[candidate['criterion'] + 1] >= candidate['condition'])
        else:
            return []
    else:
        return []


def check_acceptance(candidates, class_approx, dataset):
    covered = [check_acceptance_element(candidate, dataset) for candidate in candidates]
    if len(covered) > 0:
        result = set(covered[0])
        for s in covered[1:]:
            result = result.intersection(set(s))
        if (set(result)).issubset(set(class_approx)) is True and len(result) > 0:
            return False
        else:
            return True
    else:
        return True


def check_rules_cover(candidates, dataset):
    covered = [check_acceptance_element(candidate, dataset) for candidate in candidates]
    if len(covered) > 0:
        result = set(covered[0])
        for s in covered[1:]:
            result = result.intersection(set(s))
        return list(result)
    else:
        return []


def rules_cover(candidates, dataset):
    covered = [check_rules_cover(rule, dataset) for rule in candidates]
    if len(covered) > 0:
        result = covered[0]
        for s in covered[1:]:
            for e in s:
                if e not in result:
                    result.append(e)
        return result
    else:
        return []


def evaluate(check, best, candidates, orig_class_approx, dataset, not_covered):
    to_not_count = list(set(orig_class_approx) - set(not_covered))
    if len(candidates) > 0:
        candidates_new = candidates.copy() + [check]
        candidates_best = candidates.copy() + [best]
    else:
        candidates_new = [check]
        candidates_best = [best]
    covered_new = check_rules_cover(candidates_new, dataset)
    covered_new = list(set(covered_new) - set(to_not_count))
    covered_new_in_approx = set(covered_new).intersection(set(not_covered))
    covered_best = check_rules_cover(candidates_best, dataset)
    covered_best = list(set(covered_best) - set(to_not_count))
    covered_best_in_approx = set(covered_best).intersection(set(not_covered))
    if len(covered_new) > 0:
        wsp_new = len(covered_new_in_approx) / len(covered_new)
    else:
        wsp_new = 0
    if len(covered_best) > 0:
        wsp_best = len(covered_best_in_approx) / len(covered_best)
    else:
        wsp_best = 0
    check['covered'] = check_acceptance_element(check, dataset)
    if best['criterion'] is None or wsp_new > wsp_best:
        return check
    elif wsp_new == wsp_best:
        if len(covered_new_in_approx) > len(covered_best_in_approx):
            return check
        else:
            return best
    else:
        return best


def build_tmp_rules(covered):
    rules = []
    for candidates in covered:
        r_ = []
        cov = []
        for candidate in candidates:
            if candidate['rule_type'] == 1 or candidate['rule_type'] == 2:
                if candidate['preference'] == 'gain':
                    sign = ">= "
                else:
                    sign = "<= "
            elif candidate['rule_type'] == 3 or candidate['rule_type'] == 4:
                if candidate['preference'] == 'gain':
                    sign = "<= "
                else:
                    sign = ">= "
            r = candidate
            rule_type = candidate['rule_type']
            cov.append(candidate['covered'])
            r['sign'] = sign
            r_.append(r)
        result = set(cov[0])
        for s in cov[1:]:
            result.intersection_update(s)
        r_.append(list(result))
        r_.append(rule_type)
        rules.append(r_)
    return rules


def build_rules(rules, criteria):
    rules_ = []
    for candidate in rules:
        r = "("
        if candidate is not None:
            for part in candidate[:-2]:
                if len(r) > 1:
                    r += "& ("
                r += criteria[part['criterion']]
                r += " "
                r += part['sign']
                r += " "
                r += str(part['condition'])
                r += ") "
            r += "=> (class "
            if candidate[-1] == 1 or candidate[-1] == 2:
                r += '>='  # part['sign']
            elif candidate[-1] == 3 or candidate[-1] == 4:
                r += '<='
            r += " "
            r += part["class"][-1]
            r += ") "
            r += str(candidate[-2])  # czy printować covered
            rules_.append(r)
    return rules_


def build_rules_to_calculation(rules, criteria):
    rules_ = []
    for candidate in rules:
        r_ = {'conditions': None, 'class': None, 'covered': None}
        cond = []
        if candidate is not None:
            for part in candidate[:-2]:
                r = "("
                r += criteria[part['criterion']]
                r += " "
                r += part['sign']
                r += " "
                r += str(part['condition'])
                r += ")"
                cond.append(r)
            r_['conditions'] = cond
            c = "class "
            c += part['sign']
            c += " "
            c += part["class"][-1]
            r_['class'] = c
            r_['covered'] = candidate[-2]
            rules_.append(r_)
    return rules_


def find_rules(class_approx, dataset, rule_type, class_approx_name):
    not_covered = class_approx.copy()  # G
    rules_base = []  # P pogrubione
    preference = [c['preference'] for c in dataset['attributes']]
    criteria = [c['name'] for c in dataset['attributes']]
    objects = dataset['objects']
    while len(not_covered) > 0:
        candidates = []  # P
        tmp_covered = not_covered.copy()  # S
        while len(candidates) == 0 or check_acceptance(candidates, class_approx, objects):
            best = {'criterion': None, 'condition': None, 'class': None,
                    'rule_type': rule_type, 'preference': None, 'covered': None}  # w
            for i, c in enumerate(criteria[:-1]):
                cond = []
                for o in dataset['objects']:
                    if o[0] in tmp_covered:
                        cond.append((o[i + 1], i))
                for o in cond:
                    check = {'criterion': o[1], 'condition': o[0], 'class': class_approx_name,
                             'rule_type': rule_type, 'preference': preference[i]}
                    best = evaluate(check, best, candidates, class_approx, objects, tmp_covered)
            candidates.append(best)
            covered_by_actual = best['covered']
            tmp_covered = list(set(tmp_covered) & (set(covered_by_actual)))
        full = check_rules_cover(candidates, objects)
        for candidate in candidates:
            reduced = [c for c in candidates if
                       c['criterion'] != candidate['criterion'] or c['condition'] != candidate['condition']]
            not_full = check_rules_cover(reduced, objects)
            if full == not_full:
                candidates = reduced
        rules_base.append(candidates)
        not_covered = list((set(class_approx)) - (set(rules_cover(rules_base, objects))))
    return build_tmp_rules(rules_base)


def minimal_rule_set(rules, class_approx):
    minimal = []
    max_covered = (0, [], None)
    for rule in rules:
        if len(rule[-2]) > max_covered[0]:
            max_covered = (len(rule[-2]), rule[-2], rule)
    minimal.append(max_covered[-1])
    to_be_covered = list(set(class_approx) - set(max_covered[1]))
    while len(to_be_covered) > 0:
        candidates = [(rule, list(set(rule[-2]) & set(to_be_covered)), len(list(set(rule[-2]) & set(to_be_covered))))
                      for rule in rules]
        candidates = sorted(candidates, key=lambda x: x[2])
        minimal.append(candidates[-1][0])
        to_be_covered = list(set(to_be_covered) - set(candidates[-1][1]))
    return minimal


def DOMLEM_DRSA(approx, dataset):
    rules = {'rule type 1/3': [], 'rule type 2/4': []}
    rules_tmp = [[], []]

    # RULE TYPE 3
    for a in approx['lower_approx_downward_union']:
        rules_tmp[0] += minimal_rule_set(find_rules(a['objects'], dataset, 3, a['union']), a['objects'])

    # RULE TYPE 1
    for a in reversed(approx['lower_approx_upward_union']):
        rules_tmp[0] += minimal_rule_set(find_rules(a['objects'], dataset, 1, a['union']), a['objects'])
    rules['rule type 1/3'] = rules_tmp[0]

    # RULE TYPE 2
    for a in reversed(approx['upper_approx_upward_union']):
        rules_tmp[1] += minimal_rule_set(find_rules(a['objects'], dataset, 2, a['union']), a['objects'])

    # RULE TYPE 4
    for a in approx['upper_approx_downward_union']:
        rules_tmp[1] += minimal_rule_set(find_rules(a['objects'], dataset, 4, a['union']), a['objects'])
    rules['rule type 2/4'] = rules_tmp[1]

    criteria = [c['name'] for c in dataset['attributes']]
    rules_readable = {'rule type 1/3': build_rules(rules['rule type 1/3'], criteria),
                      'rule type 2/4': build_rules(rules['rule type 2/4'], criteria)}
    rules_to_stats = {'rule type 1/3': build_rules_to_calculation(rules['rule type 1/3'], criteria),
                      'rule type 2/4': build_rules_to_calculation(rules['rule type 2/4'], criteria)}
    return rules, rules_readable, rules_to_stats


def check_acceptance_VC(candidates, class_approx, dataset, l):
    covered = [check_acceptance_element(candidate, dataset) for candidate in candidates]
    if len(covered) > 0:
        result = set(covered[0])
        for s in covered[1:]:
            result = result.intersection(set(s))
        if float(len(set(result) & (set(class_approx))))/float(len(set(result))) >= l and len(result) > 0:
            return False
        else:
            return True
    else:
        return True


def find_rules_VC(class_approx, dataset, rule_type, class_approx_name, l):
    not_covered = class_approx.copy()  # G
    rules_base = []  # P pogrubione
    preference = [c['preference'] for c in dataset['attributes']]
    criteria = [c['name'] for c in dataset['attributes']]
    objects = dataset['objects']
    while len(not_covered) > 0:
        candidates = []  # P
        tmp_covered = not_covered.copy()  # S
        used = []
        while len(candidates) == 0 or check_acceptance_VC(candidates, class_approx, objects, l):
            best = {'criterion': None, 'condition': None, 'class': None,
                    'rule_type': rule_type, 'preference': None, 'covered': None}  # w
            for i, c in enumerate(criteria[:-1]):
                cond = []
                for o in dataset['objects']:
                    if o[0] in tmp_covered and (o[i + 1], i) not in used:
                        cond.append((o[i + 1], i))
                for o in cond:
                    check = {'criterion': o[1], 'condition': o[0], 'class': class_approx_name,
                             'rule_type': rule_type, 'preference': preference[i]}
                    best = evaluate(check, best, candidates, class_approx, objects, tmp_covered)
            candidates.append(best)
            used.append((best['condition'], best['criterion']))
            covered_by_actual = best['covered']
            tmp_covered = list(set(tmp_covered) & (set(covered_by_actual)))
        full = check_rules_cover(candidates, objects)
        for candidate in candidates:
            reduced = [c for c in candidates if
                       c['criterion'] != candidate['criterion'] or c['condition'] != candidate['condition']]
            not_full = check_rules_cover(reduced, objects)
            if full == not_full:
                candidates = reduced
        rules_base.append(candidates)
        not_covered = list((set(class_approx)) - (set(rules_cover(rules_base, objects))))
    return build_tmp_rules(rules_base)


def DOMLEM_VC_DRSA(approx, dataset, l):
    rules = {'rule type 1/3': []}
    rules_tmp = []

    # RULE TYPE 3
    for a in approx['lower_approx_downward_union']:
        rules_tmp += minimal_rule_set(find_rules_VC(a['objects'], dataset, 3, a['union'], l), a['objects'])

    # RULE TYPE 1
    for a in reversed(approx['lower_approx_upward_union']):
        rules_tmp += minimal_rule_set(find_rules_VC(a['objects'], dataset, 1, a['union'], l), a['objects'])
    rules['rule type 1/3'] = rules_tmp

    criteria = [c['name'] for c in dataset['attributes']]
    rules_readable = {'rule type 1/3': build_rules(rules['rule type 1/3'], criteria)}
    rules_to_stats = {'rule type 1/3': build_rules_to_calculation(rules['rule type 1/3'], criteria)}
    return rules, rules_readable, rules_to_stats


def update_support(ck, x, approx):
    for c_ in ck:
        goal = len(c_['w'])
        reach_goal = 0
        for c in c_['w']:
            if c['sign'] == '>=':
                if x[c['criterion']+1] >= c['condition']:
                    reach_goal += 1
            else:
                if x[c['criterion']+1] <= c['condition']:
                    reach_goal += 1
        if goal == reach_goal:
            if x[0] in approx:
                c_['positive_support'] += 1
            else:
                c_['negative_support'] += 1


def create_conditions(approx, crits, rule_type, prefs, dataset, class_approx_name, min_support):
    c1_ = []
    for x in approx:
        for a in crits:
            if rule_type == 1 or rule_type == 2:
                if prefs[a] == 'gain':
                    w_ = {'criterion': a, 'condition': dataset[int(x[1:])-1][a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'covered': None, 'sign': '>='}
                    w = {'w':[w_], 'positive_support':0, 'negative_support': 0}
                else:
                    w_ = {'criterion': a, 'condition': dataset[int(x[1:])-1][a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'covered': None, 'sign': '<='}
                    w = {'w': [w_], 'positive_support': 0, 'negative_support': 0}
            elif rule_type == 3 or rule_type == 4:
                if prefs[a] == 'gain':
                    w_ = {'criterion': a, 'condition': dataset[int(x[1:])-1][a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'covered': None, 'sign': '<='}
                    w = {'w': [w_], 'positive_support': 0, 'negative_support': 0}
                else:
                    w_ = {'criterion': a, 'condition': dataset[int(x[1:])-1][a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'covered': None, 'sign': '>='}
                    w = {'w': [w_], 'positive_support': 0, 'negative_support': 0}
            if w not in c1_:
                c1_.append(w)
    for x in dataset:
        update_support(c1_, x, approx)
    l1 = [c for c in c1_ if c['positive_support'] >= min_support]
    return l1, c1_


def build_candidates(strong_sets, k):
    l = [i for i in range(len(strong_sets))]
    tmp = list(itertools.combinations(set(l), k))
    return [(strong_sets[n[0]], strong_sets[n[1]]) for n in tmp]


def apriori2_gen(strong_sets, k):
    ck_ = []
    cand = build_candidates(strong_sets, 2)
    print("str_set: ", len(strong_sets))
    for c in cand:
        p = c[0]
        q = c[1]
        if p['negative_support'] > 0 and q['negative_support'] > 0:
            ok = True
            for i in range(k-1):
                if p['w'][i] != q['w'][i]:
                    ok = False
            if ok and p['w'][k-1]['criterion'] != q['w'][k-1]['criterion'] and p['w'][k-1]['condition'] < q['w'][k-1]['criterion']:
                c = {'w': p['w'][:-1] + list(q['w'][-1]), 'positive_support': 0, 'negative_support': 0}
                ck_.append(c)
    ck_new = []
    for s in ck_:
        if s in strong_sets:
            ck_new.append(s)
    return ck_new


def check_generality(c_, c):
    if c['w'] in c_['w'] and c['positive_support'] < c_['positive_support']:
        return True
    else:
        return False


def find_covered(c, dataset):
    covered = []
    for x in dataset:
        if c['sign'] == '>=':
            if x[c['criterion'] + 1] >= c['condition']:
                covered.append(x[0])
        else:
            if x[c['criterion'] + 1] <= c['condition']:
                covered.append(x[0])
    return covered


def build_tmp_rules_DOMA_priori(ls, dataset):
    rules = []
    for candidates in ls:
        r_ = []
        cov = []
        for candidate in candidates:
            for c in candidate['w']:
                if c['rule_type'] == 1 or c['rule_type'] == 2:
                    if c['preference'] == 'gain':
                        sign = ">= "
                    else:
                        sign = "<= "
                elif c['rule_type'] == 3 or c['rule_type'] == 4:
                    if c['preference'] == 'gain':
                        sign = "<= "
                    else:
                        sign = ">= "
                r = c
                rule_type = c['rule_type']
                cov.append(find_covered(c, dataset))
                r['sign'] = sign
                r_.append(r)
        if len(cov) == 0:
            result = []
        else:
            result = set(cov[0])
        if len(cov) > 1:
            for s in cov[1:]:
                result.intersection_update(s)
        r_.append(list(result))
        r_.append(rule_type)
        rules.append(r_)
    return rules


def apriori_dom_rules(class_approx, dataset, rule_type, class_approx_name, max_length, min_support):
    criteria = [c['preference'] for c in dataset['attributes']]
    crits = [i for i in range(len(criteria)-1)]
    l1, c1 = create_conditions(class_approx, crits,  rule_type, criteria[:-1], dataset['objects'], class_approx_name, min_support)
    l_k_1 = l1
    print('l_k_1', l_k_1)
    ls = [l1]
    k = 2
    while len(l_k_1) != 0 and k < max_length:
        ck = apriori2_gen(l_k_1, k-1)
        for x in dataset['objects']:
            update_support(ck, x, class_approx)
        lk = [c for c in ck if c['positive_support'] >= min_support]
        l_k_1 = lk
        ls.append(lk)
        k += 1
    for i,lk in enumerate(ls):
        lk_new = []
        for c in lk:
            if c['negative_support'] <= 0:
                lk_new.append(c)
        ls[i] = lk_new
        for n,lk in enumerate(ls[1:]):
            lk_new = []
            for c in lk:
                for lj in reversed(ls[1:n+1]):
                    for c_ in lj:
                        if not check_generality(c_, c):
                            lk_new.append(c)
            ls[n+1] = lk_new
    return build_tmp_rules_DOMA_priori(ls, dataset['objects'])


def minimal_rule_set_DOMA_priori():
    return 0


def DOMApriori_DRSA(approx, dataset, max_length, min_support):
    rules = {'rule type 1/3': [], 'rule type 2/4': []}
    rules_tmp = [[], []]

    # RULE TYPE 3
    for a in approx['lower_approx_downward_union']:
        rules_tmp[0] += minimal_rule_set(find_rules(a['objects'], dataset, 3, a['union']), a['objects'])

    # RULE TYPE 1
    for a in reversed(approx['lower_approx_upward_union']):
        rules_tmp[0] += apriori_dom_rules(a['objects'], dataset, 1, a['union'], max_length, min_support)#minimal_rule_set(, a['objects'])
    rules['rule type 1/3'] = rules_tmp[0]

    # RULE TYPE 2
    #for a in reversed(approx['upper_approx_upward_union']):
        #rules_tmp[1] += minimal_rule_set(find_rules(a['objects'], dataset, 2, a['union']), a['objects'])

    # RULE TYPE 4
    #for a in approx['upper_approx_downward_union']:
        #rules_tmp[1] += minimal_rule_set(find_rules(a['objects'], dataset, 4, a['union']), a['objects'])
    #rules['rule type 2/4'] = rules_tmp[1]

    criteria = [c['name'] for c in dataset['attributes']]
    rules_readable = {'rule type 1/3': build_rules(rules['rule type 1/3'], criteria)}#,
                      #'rule type 2/4': build_rules(rules['rule type 2/4'], criteria)}
    rules_to_stats = {'rule type 1/3': build_rules_to_calculation(rules['rule type 1/3'], criteria)}#,
                      #'rule type 2/4': build_rules_to_calculation(rules['rule type 2/4'], criteria)}
    return rules, rules_readable, rules_to_stats


def DOMApriori_VC_DRSA(approx, dataset, l, max_length, min_support):
    rules = {'rule type 1/3': []}
    rules_tmp = []

    # RULE TYPE 3
    for a in approx['lower_approx_downward_union']:
        rules_tmp += minimal_rule_set(find_rules_VC(a['objects'], dataset, 3, a['union'], l), a['objects'])

    # RULE TYPE 1
    for a in reversed(approx['lower_approx_upward_union']):
        rules_tmp += minimal_rule_set(find_rules_VC(a['objects'], dataset, 1, a['union'], l), a['objects'])
    rules['rule type 1/3'] = rules_tmp

    criteria = [c['name'] for c in dataset['attributes']]
    rules_readable = {'rule type 1/3': build_rules(rules['rule type 1/3'], criteria)}
    rules_to_stats = {'rule type 1/3': build_rules_to_calculation(rules['rule type 1/3'], criteria)}
    return rules, rules_readable, rules_to_stats
