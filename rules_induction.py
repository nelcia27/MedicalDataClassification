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
                class_ = part['class']
                if len(r) > 1:
                    r += "& ("
                r += criteria[part['criterion']]
                r += " "
                r += part['sign']
                r += " "
                r += str(part['condition'])
                r += ") "
            r += "=> (class "
            if "most" in class_:
                r += '<='
            else:
                r += '>='
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
                class_ = part['class']
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
            if "most" in class_:
                c += '<='
            else:
                c += '>='
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
        x_ = [o for o in dataset if o[0] == x][0]
        for a in crits:
            if rule_type == 1 or rule_type == 2:
                if prefs[a] == 'gain':
                    w_ = {'criterion': a, 'condition': x_[a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'sign': '>='}
                    w = {'w':[w_], 'positive_support': 0, 'negative_support': 0}
                else:
                    w_ = {'criterion': a, 'condition': x_[a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'sign': '<='}
                    w = {'w': [w_], 'positive_support': 0, 'negative_support': 0}
            elif rule_type == 3 or rule_type == 4:
                if prefs[a] == 'gain':
                    w_ = {'criterion': a, 'condition': x_[a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'sign': '<='}
                    w = {'w': [w_], 'positive_support': 0, 'negative_support': 0}
                else:
                    w_ = {'criterion': a, 'condition': x_[a+1], 'class': class_approx_name,
                          'rule_type': rule_type, 'preference': prefs[a], 'sign': '>='}
                    w = {'w': [w_], 'positive_support': 0, 'negative_support': 0}
            if w not in c1_:
                c1_.append(w)
    for x in dataset:
        update_support(c1_, x, approx)
    l1 = [c for c in c1_ if c['positive_support'] >= min_support]
    return l1, c1_


def build_candidates(strong_sets):
    interesting = []
    for s in strong_sets:
        if s['negative_support'] > 0:
            interesting.append(s)
    interesting_dict = {}
    for i in range(len(interesting)):
        interesting_dict[i] = interesting[i]
    crit_pairs = []
    for i in interesting:
        k = [k['criterion'] for k in i['w']]
        if k not in crit_pairs:
            crit_pairs.append(k)
    crit_pairs_dict = {}
    for i in range(len(crit_pairs)):
        crit_pairs_dict[i] = crit_pairs[i]
    possible_crit_pairs = []
    for p in crit_pairs:
        for q in crit_pairs:
            if p != q and p[:-1] == q[:-1]:
                if p + q == sorted(p + q) and p + q not in possible_crit_pairs:
                    possible_crit_pairs.append([p,q])
    groups = {}
    for ind, i in enumerate(crit_pairs):
        for elem in interesting_dict.items():
            k = [k['criterion'] for k in elem[1]['w']]
            if k == i:
                if ind in list(groups.keys()):
                    groups[ind].append([interesting_dict[elem[0]]])
                else:
                    groups[ind] = [[interesting_dict[elem[0]]]]
    candidates = []
    for c in possible_crit_pairs:
        p = c[0]
        q = c[1]
        for i in crit_pairs_dict.items():
            if i[1] == p:
                cand1 = groups[i[0]]
            elif i[1] == q:
                cand2 = groups[i[0]]
        for c1 in cand1:
            for c2 in cand2:
                candidates.append((c1, c2))
    return candidates


def apriori2_gen(strong_sets, k):
    ck_ = []
    cand = build_candidates(strong_sets)
    l = k-1
    for c in cand:
        p = c[0][0]
        q = c[1][0]
        ok1 = True
        for i in range(k-1):
            if p['w'][i]['criterion'] != q['w'][i]['criterion'] or p['w'][i]['condition'] != q['w'][i]['condition']:
                ok1 = False
        if ok1 and p['w'][l]['criterion'] != q['w'][l]['criterion'] and p['w'][l]['condition'] < q['w'][l]['condition']:
            tmp = p['w'].copy()
            tmp.append(q['w'][-1])
            c_ = {'w': tmp, 'positive_support': 0, 'negative_support': 0}
            ck_.append(c_)
    return ck_


def check_generality(c_, c):
    c_acc = c['positive_support']
    c__acc = c_['positive_support']
    if c_acc <= c__acc:
        ok = 0
        for c1 in c['w']:
            for c2 in c_['w']:
                if c1['criterion'] == c2['criterion']:
                    if c1['preference'] == 'gain':
                        if c2['condition'] >= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] <= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
                    else:
                        if c2['condition'] <= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] >= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
        if ok == len(c_['w']):
            return c
        else:
            return None
    else:
        return None


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
    rule = []
    cov = []
    for w in ls['w']:
        w_ = {'criterion': w['criterion'], 'condition': w['condition'], 'class': w['class'],
              'rule_type': w['rule_type'], 'preference': w['preference'], 'covered': find_covered(w, dataset), 'sign': w['sign']+" "}
        rule_type = w_['rule_type']
        cov.append(find_covered(w, dataset))
        rule.append(w_)
    result = set(cov[0])
    if len(cov) > 1:
        for s in cov[1:]:
            result.intersection_update(s)
    rule.append(list(result))
    rule.append(rule_type)
    return rule


def apriori_dom_rules(class_approx, dataset, rule_type, class_approx_name, max_length, min_support):
    criteria = [c['preference'] for c in dataset['attributes']]
    crits = [i for i in range(len(criteria)-1)]
    if max_length > len(crits):
        max_length = len(crits)
    l1, c1 = create_conditions(class_approx, crits,  rule_type, criteria[:-1], dataset['objects'], class_approx_name, min_support)
    l_k_1 = l1
    ls = [l1]
    k = 2
    while len(l_k_1) != 0 and k <= max_length:
        ck = apriori2_gen(l_k_1, k-1)
        for x in dataset['objects']:
            update_support(ck, x, class_approx)
        lk = [c for c in ck if c['positive_support'] >= min_support]
        l_k_1 = lk
        if len(l_k_1) > 0:
            ls.append(lk)
        k += 1
    for i, lk in enumerate(ls):
        lk_new = []
        for c in lk:
            if c['negative_support'] <= 0:
                lk_new.append(c)
        ls[i] = lk_new
    for i, lk in enumerate(ls):
        lk_delete = []
        for l1 in lk:
            for l2 in lk:
                if l1 != l2:
                    if len(l1['w']) < len(l2['w']):
                        d = check_generality(l1, l2)
                    elif len(l1['w']) > len(l2['w']):
                        d = check_generality(l2, l1)
                    elif len(l1['w']) == len(l2['w']):
                        d = check_generality(l2, l1)
                        if d is None:
                            d = check_generality(l1, l2)
                        else:
                            lk_delete.append(d)
                            d = check_generality(l1, l2)
                            lk_delete.append(d)
                    if d is not None:
                        lk_delete.append(d)
        ls[i] = [e for e in lk if e not in lk_delete]
    ls_ = []
    for l in ls:
        if len(l) != 0:
            ls_.append(l)
    r = []
    for l in ls_:
        for c in l:
            r.append(build_tmp_rules_DOMA_priori(c, dataset['objects']))
    return r


def check_generality_v2(c_, c):
    c_acc = len(c[-2])
    c__acc = len(c_[-2])
    if c_acc <= c__acc:
        ok = 0
        for c1 in c[:-2]:
            for c2 in c_[:-2]:
                if c1['criterion'] == c2['criterion']:
                    if c1['preference'] == 'gain':
                        if c2['condition'] >= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] <= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
                    else:
                        if c2['condition'] <= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] >= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
        if ok == len(c_[:-2]):
            return c
        else:
            return None
    else:
        return None


def minimal_rule_set_DOMA_priori(rules_prev, rules_new):
    rules_to_delete = []
    for r1 in rules_prev:
        for r2 in rules_new:
            if len(r1) <= len(r2):
                d = check_generality_v2(r1, r2)
                if d is not None:
                    rules_to_delete.append(d)
    return [e for e in rules_new if e not in rules_to_delete]


def check_reduce(r2, r1):
    ok = 0
    for c1 in r1[:-2]:
        for c2 in r2[:-2]:
            if c1['criterion'] == c2['criterion']:
                if c1['preference'] == 'gain':
                    if c2['condition'] >= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                        ok += 1
                        break
                    elif c2['condition'] <= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                        ok += 1
                        break
                else:
                    if c2['condition'] <= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                        ok += 1
                        break
                    elif c2['condition'] >= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                        ok += 1
                        break
    if ok == len(r2[:-2]):
        return r1


def minimal_check(rules):
    rules_to_delete = []
    for r1 in rules:
        for r2 in rules:
            if r1 != r2:
                if len(r1[-2]) == len(r2[-2]):
                    d = check_reduce(r1, r2)
                    if d is None:
                        d = check_reduce(r2, r1)
                    if d is not None:
                        rules_to_delete.append(d)
                        d = check_reduce(r2, r1)
                        if d is not None:
                            rules_to_delete.append(d)
    return [e for e in rules if e not in rules_to_delete]


def DOMApriori_DRSA(approx, dataset, max_length, min_support):
    rules = {'rule type 1/3': [], 'rule type 2/4': []}
    rules_tmp = [[], [], [], []]

    # RULE TYPE 3
    for a in approx['lower_approx_downward_union']:
        new_rules = apriori_dom_rules(a['objects'], dataset, 3, a['union'], max_length, min_support)
        n_r = minimal_check(minimal_rule_set_DOMA_priori(rules_tmp[0], new_rules))
        rules_tmp[0] += n_r

    # RULE TYPE 1
    for a in reversed(approx['lower_approx_upward_union']):
        new_rules = apriori_dom_rules(a['objects'], dataset, 1, a['union'], max_length, min_support)
        n_r = minimal_check(minimal_rule_set_DOMA_priori(rules_tmp[1], new_rules))
        rules_tmp[1] += n_r
    rules['rule type 1/3'] = rules_tmp[0] + rules_tmp[1]

    # RULE TYPE 4
    for a in approx['upper_approx_downward_union']:
        new_rules = apriori_dom_rules(a['objects'], dataset, 4, a['union'], max_length, min_support)
        n_r = minimal_check(minimal_rule_set_DOMA_priori(rules_tmp[2], new_rules))
        rules_tmp[2] += n_r

    # RULE TYPE 2
    for a in reversed(approx['upper_approx_upward_union']):
        new_rules = apriori_dom_rules(a['objects'], dataset, 1, a['union'], max_length, min_support)
        n_r = minimal_check(minimal_rule_set_DOMA_priori(rules_tmp[3], new_rules))
        rules_tmp[3] += n_r
    rules['rule type 2/4'] = rules_tmp[2] + rules_tmp[3]

    criteria = [c['name'] for c in dataset['attributes']]
    rules_readable = {'rule type 1/3': build_rules(rules['rule type 1/3'], criteria),
                      'rule type 2/4': build_rules(rules['rule type 2/4'], criteria)}
    rules_to_stats = {'rule type 1/3': build_rules_to_calculation(rules['rule type 1/3'], criteria),
                      'rule type 2/4': build_rules_to_calculation(rules['rule type 2/4'], criteria)}
    return rules, rules_readable, rules_to_stats


def check_generality_VC(c_, c):
    c_acc = float(c['positive_support'])/float(c['positive_support'] + c['negative_support'])
    c__acc = float(c_['positive_support'])/float(c_['positive_support'] + c_['negative_support'])
    if c_acc <= c__acc:
        ok = 0
        for c1 in c['w']:
            for c2 in c_['w']:
                if c1['criterion'] == c2['criterion']:
                    if c1['preference'] == 'gain':
                        if c2['condition'] >= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] <= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
                    else:
                        if c2['condition'] <= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] >= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
        if ok == len(c_['w']):
            return c
        else:
            return None
    else:
        return None


def apriori_dom_rules_VC(class_approx, dataset, rule_type, class_approx_name, max_length, min_support, l):
    criteria = [c['preference'] for c in dataset['attributes']]
    crits = [i for i in range(len(criteria) - 1)]
    if max_length > len(crits):
        max_length = len(crits)
    l1, c1 = create_conditions(class_approx, crits, rule_type, criteria[:-1], dataset['objects'], class_approx_name,
                               min_support)
    l_k_1 = l1
    ls = [l1]
    k = 2
    while len(l_k_1) != 0 and k <= max_length:
        ck = apriori2_gen(l_k_1, k - 1)
        for x in dataset['objects']:
            update_support(ck, x, class_approx)
        lk = [c for c in ck if c['positive_support'] >= min_support]
        l_k_1 = lk
        if len(l_k_1) > 0:
            ls.append(lk)
        k += 1
    for i, lk in enumerate(ls):
        lk_new = []
        for c in lk:
            if float(c['positive_support'])/float(c['positive_support'] + c['negative_support']) >= l:
                lk_new.append(c)
        ls[i] = lk_new
    for i, lk in enumerate(ls):
        lk_delete = []
        for l1 in lk:
            for l2 in lk:
                if l1 != l2:
                    if len(l1['w']) < len(l2['w']):
                        d = check_generality_VC(l1, l2)
                    elif len(l1['w']) > len(l2['w']):
                        d = check_generality_VC(l2, l1)
                    elif len(l1['w']) == len(l2['w']):
                        d = check_generality_VC(l2, l1)
                        if d is None:
                            d = check_generality_VC(l1, l2)
                        else:
                            lk_delete.append(d)
                            d = check_generality_VC(l1, l2)
                            if d is not None:
                                lk_delete.append(d)
                    if d is not None:
                        lk_delete.append(d)
        ls[i] = [e for e in lk if e not in lk_delete]
    ls_ = []
    for l in ls:
        if len(l) != 0:
            ls_.append(l)
    r = []
    for l in ls_:
        for c in l:
            r.append(build_tmp_rules_DOMA_priori(c, dataset['objects']))
    return r


def check_generality_VC_v2(c_, c, approx):
    c1_p = len([e for e in c[-2] if e in approx])
    c2_p = len([e for e in c_[-2] if e in approx])
    c_acc = float(c1_p)/float(len(c[-2]))
    c__acc = float(c2_p)/float(len(c_[-2]))
    if c_acc <= c__acc:
        ok = 0
        for c1 in c[:-2]:
            for c2 in c_[:-2]:
                if c1['criterion'] == c2['criterion']:
                    if c1['preference'] == 'gain':
                        if c2['condition'] >= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] <= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
                    else:
                        if c2['condition'] <= c1['condition'] and (c1['rule_type'] == 3 or c1['rule_type'] == 4):
                            ok += 1
                            break
                        elif c2['condition'] >= c1['condition'] and (c1['rule_type'] == 1 or c1['rule_type'] == 2):
                            ok += 1
                            break
        if ok == len(c_[:-2]):
            return c
        else:
            return None
    else:
        return None


def minimal_rule_set_DOMA_priori_VC(rules_prev, rules_new, approx):
    rules_to_delete = []
    for r1 in rules_prev:
        for r2 in rules_new:
            if len(r1) <= len(r2):
                d = check_generality_VC_v2(r1, r2, approx)
                if d is not None:
                    rules_to_delete.append(d)
    return [e for e in rules_new if e not in rules_to_delete]


def minimal_check_VC(rules, approx):
    rules_to_delete = []
    for r1 in rules:
        for r2 in rules:
            if r1 != r2:
                c1_p = len([e for e in r1[-2] if e in approx])
                c2_p = len([e for e in r2[-2] if e in approx])
                c_acc = float(c1_p) / float(len(r1[-2]))
                c__acc = float(c2_p) / float(len(r2[-2]))
                if c_acc == c__acc:
                    d = check_reduce(r1, r2)
                    if d is None:
                        d = check_reduce(r2, r1)
                    if d is not None:
                        rules_to_delete.append(d)
                        d = check_reduce(r2, r1)
                        if d is not None:
                            rules_to_delete.append(d)
    return [e for e in rules if e not in rules_to_delete]


def delete_doubled(rules_prev, rules_new):
    rules_to_delete = []
    for r1 in rules_prev:
        for r2 in rules_new:
            if [(r['condition'], r['criterion']) for r in r1[:-2]] == [(r['condition'], r['criterion']) for r in r2[:-2]]:
                rules_to_delete.append(r2)
    return [e for e in rules_new if e not in rules_to_delete]


def DOMApriori_VC_DRSA(approx, dataset, l, max_length, min_support):
    rules = {'rule type 1/3': []}
    rules_tmp = [[], []]

    # RULE TYPE 3
    for a in approx['lower_approx_downward_union']:
        new_rules = apriori_dom_rules_VC(a['objects'], dataset, 3, a['union'], max_length, min_support, l)
        n_r = delete_doubled(rules_tmp[0], minimal_check_VC(minimal_rule_set_DOMA_priori_VC(rules_tmp[0], new_rules, a['objects']), a['objects']))
        rules_tmp[0] += n_r

    # RULE TYPE 1
    for a in reversed(approx['lower_approx_upward_union']):
        new_rules = apriori_dom_rules_VC(a['objects'], dataset, 1, a['union'], max_length, min_support, l)
        n_r = delete_doubled(rules_tmp[1], minimal_check_VC(minimal_rule_set_DOMA_priori_VC(rules_tmp[1], new_rules, a['objects']), a['objects']))
        rules_tmp[1] += n_r
    rules['rule type 1/3'] = rules_tmp[0] + rules_tmp[1]

    criteria = [c['name'] for c in dataset['attributes']]
    rules_readable = {'rule type 1/3': build_rules(rules['rule type 1/3'], criteria)}
    rules_to_stats = {'rule type 1/3': build_rules_to_calculation(rules['rule type 1/3'], criteria)}
    return rules, rules_readable, rules_to_stats
