import numpy as np
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10)

X_train = np.load('./data/input/full_xtrain.npy')
Y_train = np.load('./data/input/full_ytrain.npy')

X_test = np.load('./data/input/xtest.npy')
Y_test = np.load('./data/input/ytest.npy')

X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)

i = 1

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    np.save(f'./data/input/xtrain_{i}.npy', X_train)
    np.save(f'./data/input/xtest_{i}.npy', X_test)
    np.save(f'./data/input/ytrain_{i}.npy', Y_train)
    np.save(f'./data/input/ytest_{i}.npy', Y_test)
    i += 1