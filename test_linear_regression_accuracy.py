import slearn.linear_model as sl
import sklearn.preprocessing as pre
import sklearn.linear_model as sk
import sklearn.datasets as ds
import numpy as np
import pandas as pd
import pytest as pt


def test_linear_model_squared_loss_with_data_from_load_boston():
    X, y = ds.load_boston(True)
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    sl_model = sl.LinearRegression()
    sl_model.gaussian_init(13)

    sl_model.fit(X, y)
    
    # The cost would be 4.776976115429404e-20
    # The anwer is from the closed solution
    assert abs(np.square(np.sum(sl_model.predict(X) - y)) - 4.777e-20) < 1e-22


def test_linear_model_squared_loss_with_instance_over_10000():
    X = np.array([ [np.sqrt(i), np.random.normal(loc=20,scale=4), np.random.normal(loc=-1000, scale=400)] for i in range(200000)   ])
    y = np.array([ X[i][0] * 0.1 + X[i][1] * 0.5 + X[i][2]*0.3 for i in range(200000)])
    sl_model = sl.LinearRegression(False)
    sl_model.gaussian_init(3)
    sl_model.normal_equation(X, y)

    # Cost is around 2.66e-15 to 6.976e-14
    # The model is from the closed solution
    # Normal equation still has high performance with such big data set.
    print(np.square(np.sum(sl_model.predict(X) - y)))
    assert abs(np.square(np.sum(sl_model.predict(X) - y))) < 1e-12






test_linear_model_squared_loss_with_instance_over_10000()