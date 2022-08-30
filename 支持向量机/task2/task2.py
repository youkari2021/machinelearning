from SVM_Functions import *
import pickle


if __name__ == '__main__':
    data_x, data_y = loadData('task2.mat')
    # plotData(data_x, data_y)
    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # print(data_y.shape)
    # kernal = gaussianKernel(data_x, 1)
    # model = svmTrain_SMO(data_x, data_y, C=1, kernelFunction='gaussian', K_matrix=kernal)
    # pred = svmPredict(model, data_x, 1)
    # print(pred.shape)
    kernals = []
    models = []
    for sigma in sigmas:
        kernal = gaussianKernel(data_x, sigma)
        kernals.append(kernal)
    for c in C:
        c_model = []
        for kernal in kernals:
            model = svmTrain_SMO(data_x, data_y, C=c, kernelFunction='gaussian', K_matrix=kernal)
            c_model.append(model)
        models.append(c_model)
    preds = []
    for c_model in models:
        c_pred = []
        for i in range(len(c_model)):
            start = time.perf_counter()
            print('pridicting ...', end='')
            pred = svmPredict(c_model[i], data_x, sigmas[i])
            c_pred.append(pred)
            print('Done', end='')
            end = time.perf_counter()
            print('( ' + str(end - start) + 's )')
            print()
        preds.append(c_pred)

    accuracys = []
    l = len(preds[0][0])
    i = 0
    for c_pred in preds:
        c_accuracys = []
        for pred in c_pred:
            print(i)
            i += 1
            # error = 0
            # for i in range(l):
            #     if data_y[i] != pred[i]: error += 1
            accuracy = np.sum(pred == data_y)/l
            c_accuracys.append(accuracy)
        accuracys.append(c_accuracys)
    accuracys = np.mat(accuracys)
    f = open('accuracys', 'wb')
    pickle.dump(accuracys, f)
    f.close()
    print(accuracys)
    pass
