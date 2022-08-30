from demo import *
from sklearn.model_selection import GridSearchCV
import pickle

parameters = {'C': np.linspace(1, 2, 10)}
clf = svm.SVC(decision_function_shape='ovo')
clf = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
train_feature, train_label, test_feature = data_init()
clf.fit(train_feature, train_label)
f = open('model2', 'wb')
pickle.dump(clf, f)
f.close()
test_pred = clf.predict(test_feature)
with open('test_pred.txt', 'w') as f:
    for label in test_pred:
        f.write(str(label) + '\n')
f.close()
pred = clf.predict(train_feature)
train_acc = accuracy_score(train_label, pred)
train_f1 = f1_score(train_label, pred, average='macro')
print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')



