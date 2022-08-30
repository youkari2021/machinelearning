from demo import *


def aggregate_clf(train_feature, train_label, clf_type, iter_num=5):
    x_trains, x_tests, y_trains, y_tests = [], [], [], []
    # clf_types = ['svm', 'tree', 'MLPClassifier', 'BernoulliNB', 'LogisticRegression']
    clfs = []
    weights = []
    for i in range(iter_num):
        print('第{}个分类器训练开始'.format(i))
        x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2)
        x_trains.append(x_train)
        x_tests.append(x_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        if clf_type == 'svm':
            clf = svm.SVC(decision_function_shape='ovo').fit(x_train, y_train)
        elif clf_type == 'tree':
            clf = tree.DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
        elif clf_type == 'MLPClassifier':
            clf = MLPClassifier().fit(x_train, y_train)
        elif clf_type == 'BernoulliNB':
            clf = BernoulliNB().fit(x_train, y_train)
        elif clf_type == 'LogisticRegression':
            clf = LogisticRegression(max_iter=1000).fit(x_train, y_train)
        clfs.append(clf)
        x_test_pred = clf.predict(x_test)
        train_acc = accuracy_score(y_test, x_test_pred)
        train_f1 = f1_score(y_test, x_test_pred, average='macro')
        print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')
        # weights.append((train_acc+train_f1)/2)
        weights.append(train_f1)
        print('第{}个分类器训练结束'.format(i))
    return clfs, weights, x_tests, y_tests

