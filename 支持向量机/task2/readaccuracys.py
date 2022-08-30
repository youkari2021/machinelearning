import pickle
import numpy as np
import pandas as pd

f = open('accuracys', 'rb')
accuracys = pickle.load(f)
accuracys = np.around(accuracys, decimals=3)
accuracys_pd = pd.DataFrame(accuracys, index=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], columns=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
print(accuracys_pd)
accuracys_pd.to_csv('accuracys.csv', encoding='utf-8')

