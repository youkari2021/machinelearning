from SVM_Functions import *
from sklearn.model_selection import train_test_split
import pickle


if __name__ == '__main__':
    train_features, train_label = loadData('task3_train.mat')
    test_features = loadmat('task3_test.mat')['X']
    x_train, x_test, y_train, y_test = train_test_split(train_features, train_label, test_size=0.2)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    linear_model = svmTrain_SMO(x_train, y_train, C=1, max_iter=20)
    start = time.perf_counter()
    print('pridicting ...', end='')
    pred = svmPredict(linear_model, x_test)
    print('Done', end='')
    end = time.perf_counter()
    print('( ' + str(end - start) + 's )')
    print()
    #
    # f = open('model', 'wb')
    # pickle.dump(linear_model, f)
    # f.close()
    # f = open('model', 'rb')
    # linear_model = pickle.load(f)
    # f.close()

    l = len(y_test)
    accuracy = np.sum(y_test == pred)/l
    print('模型准确率', accuracy)
    # predict = svmPredict(linear_model, test_features)
    # predict = predict.tolist()
    #
    # f = open('test_labels.txt', 'w')
    # for i in predict:
    #     s = str(int(i[0])) + '\n'
    #     f.write(s)
    # f.close()
    # pass
