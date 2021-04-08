with open('dataset_2/classification_data/params_sorted.csv', 'r') as inp:
    lr = True
    lr_pos = []
    lr_neg = []
    svm_pos = []
    svm_neg = []
    for line in inp:
        line = line.strip()
        if 'SVM params' in line:
            lr = False
        if lr:
            if '-' in line:
                lr_neg.append(line)
            else:
                lr_pos.append(line)
        else:
            if '-' in line:
                svm_neg.append(line)
            else:
                svm_pos.append(line)

with open('out.csv', 'w') as out:
    for x in lr_pos:
        print(x, file=out)
    for x in lr_neg:
        print(x, file=out)
    print(file=out)
    for x in svm_pos:
        print(x, file=out)
    for x in svm_neg:
        print(x, file=out)
    