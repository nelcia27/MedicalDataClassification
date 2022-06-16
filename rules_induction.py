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

