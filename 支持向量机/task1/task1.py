from SVM_Functions import *


if __name__ == '__main__':
    linear_data_x, linear_data_y = loadData('task1_linear.mat')
    gaussian_data_x, gaussian_data_y = loadData('task1_gaussian.mat')
    plotData(linear_data_x, linear_data_y)
    plotData(gaussian_data_x, gaussian_data_y)
    linear_model = svmTrain_SMO(linear_data_x, linear_data_y, C=1, max_iter=20)
    gaussian_kernal = gaussianKernel(gaussian_data_x, sigma=0.1)
    gaussian_model = svmTrain_SMO(gaussian_data_x, gaussian_data_y, C=1, kernelFunction='gaussian', K_matrix=gaussian_kernal)
    visualizeBoundaryLinear(linear_data_x, linear_data_y, linear_model)
    visualizeBoundaryGaussian(gaussian_data_x, gaussian_data_y, gaussian_model, 0.1)
    pass
