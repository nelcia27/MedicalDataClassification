import itertools


def read_dataset(file):
    with open(file) as f:
        content = f.readlines()
        attributes = []
        preferences = []
        objects = []
        for ind, line in enumerate(content):
            if line[0] == '+':
                att = []
                for s in line:
                    if s != ":" and s != "+" and s != " ":
                        att.append(s)
                    elif s == ":":
                        break
                attributes.append(''.join(att))
            elif line == '**PREFERENCES\n':
                for i in range(len(attributes)):
                    tmp = []
                    for j, e in enumerate(content[ind + 2 + i]):
                        if e == ":":
                            preferences.append(content[ind + 2 + i][j + 2:-1])
            elif line == '**EXAMPLES\n':
                i = ind
                while i < len(content):
                    if content[i][0] == 'a':
                        tmp = str(content[i]).split(' ')
                        objects.append(tmp[1:])
                    i += 1
    return attributes, preferences, objects


def prepare_dataset(attributes, preferences, objects):
    obj_matrix = []
    for i, o in enumerate(objects):
        tmp = ["a" + str(i + 1)]
        for e in range(len(o) - 1):
            tmp.append(float(o[e]))
        tmp.append(float(o[-1][:-1]))
        obj_matrix.append(tmp)

    attr = []
    for a, p in zip(attributes, preferences):
        attr.append({"name": a, "preference": p})
    return {"attributes": attr, "objects": obj_matrix}


def calculate_quality_of_approximation_of_classification(dataset, boundaries):
    num_objects = len(dataset['objects'])
    tmp = []
    for b in boundaries:
        tmp.append(b['objects'])
    tmp_ = list(set(list(itertools.chain(*tmp))))
    bound_cnt = len(tmp_)
    return float(num_objects - bound_cnt) / float(num_objects)


def calculate_quality_of_approximation_per_union(unions, lower_approx):
    quality = {}
    for u, l in zip(unions, lower_approx):
        U = len(list(u.values())[0][1])
        B = len(l['objects'])
        if float(U - B) / float(U) >= 0:
            quality[list(u.keys())[0]] = float(U - B) / float(U)
        else:
            quality[list(u.keys())[0]] = 0.0
    return quality


def calculate_accuracy_of_approximation_per_union(lower_approx, upper_approx):
    accuracy = {}
    for l, u in zip(lower_approx, upper_approx):
        if len(u['objects']) > 0:
            accuracy[l['union']] = len(l['objects']) / len(u['objects'])
        else:
            accuracy[l['union']] = 0.0
    return accuracy

