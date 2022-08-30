from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from demo import *
from sklearn.model_selection import GridSearchCV
import pickle


def clf(train_feature, train_label):
    parameters = {'C': np.linspace(1, 2, 10)}
    clf = svm.SVC(decision_function_shape='ovo')
    clf = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
    # x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2)
    # clf.fit(x_train, y_train)
    clf.fit(train_feature, train_label)
    f = open('model2', 'wb')
    pickle.dump(clf, f)
    f.close()
    # x_test_pred = clf.predict(x_test)
    x_test_pred = clf.predict(train_feature)
    # train_acc = accuracy_score(y_test, x_test_pred)
    train_acc = accuracy_score(train_label, x_test_pred)
    # train_f1 = f1_score(y_test, x_test_pred, average='macro')
    train_f1 = f1_score(train_label, x_test_pred, average='macro')
    print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')
    return clf


if __name__ == '__main__':
    train_feature, train_label, test_feature = data_init()
    # x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2)
    # f = open('xtest', 'wb')
    # pickle.dump(x_test, f)
    # f.close()
    # f = open('ytest', 'wb')
    # pickle.dump(y_test, f)
    # f.close()
    # clf = clf(x_train, y_train)
    clf = clf(train_feature, train_label)
    # f = open('xtest', 'rb')
    # x_test = pickle.load(f)
    # f.close()
    # f = open('ytest', 'rb')
    # y_test = pickle.load(f)
    # f.close()
    # f = open('model', 'rb')
    # clf = pickle.load(f)
    # f.close()
    # test_pred = clf.predict(x_test)
    # train_acc = accuracy_score(y_test, test_pred)
    # train_f1 = f1_score(y_test, test_pred, average='macro')
    # print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')
    test_pred = clf.predict(test_feature)
    with open('test_pred.txt', 'w') as f:
        for label in test_pred:
            f.write(str(label) + '\n')
    f.close()

