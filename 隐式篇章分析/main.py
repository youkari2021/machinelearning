import aggregate
from demo import *


if __name__ == '__main__':
    train_feature, train_label, test_feature = data_init()
    x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2)
    clfs, weights, x_tests, y_tests = aggregate.aggregate_clf(x_train, y_train, clf_type='svm')
    # clf_types = ['svm', 'tree', 'MLPClassifier', 'BernoulliNB', 'LogisticRegression']
    test_pred = aggrerate(clfs, test_feature, weights)
    for i in range(len(x_tests)):
        re_x_test_pred = aggrerate(clfs, x_tests[i], weights)
        train_acc = accuracy_score(y_tests[i], re_x_test_pred)
        train_f1 = f1_score(y_tests[i], re_x_test_pred, average='macro')
        print(f'a: Acc={train_acc:.4f}, F1={train_f1:.4f}')
    with open('test_pred.txt', 'w') as f:
        for label in test_pred:
            f.write(str(label) + '\n')
    f.close()
    re_x_test_pred = aggrerate(clfs, x_test, weights)
    re_x_train_pred = aggrerate(clfs, x_train, weights)
    train_acc = accuracy_score(y_train, re_x_train_pred)
    train_f1 = f1_score(y_train, re_x_train_pred, average='macro')
    print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')
    train_acc = accuracy_score(y_test, re_x_test_pred)
    train_f1 = f1_score(y_test, re_x_test_pred, average='macro')
    print(f'Test Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')
