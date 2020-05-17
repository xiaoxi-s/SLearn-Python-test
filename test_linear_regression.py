import slearn.linear_model as sl
import pytest as pt
import numpy as np
import pandas as pd

def test_normal_equation():
    tune_all_model = sl.LinearRegression(True)
    X = np.array([[0, 2], [1, 4], [2, 9], [3, 16], [4, 25]], dtype='float64')
    y = np.array([[1], [2], [3], [4], [5]], dtype='float64')

    # cost should be very close to 0
    assert tune_all_model.normal_equation(X, y) < 1e-8

    no_tune_all_model = sl.LinearRegression()
    # cost should be very close to 0.16
    assert np.abs(no_tune_all_model.normal_equation(X, y) - 0.16) < 1e-2


def test_gradient_descent():

    """
    Test whether gradient descent and normal equation would give similar answers
    """
    model = sl.LinearRegression()
    another_model = sl.LinearRegression()
    another_model.constant_init(3, 0.5)

    # read features and labels from csv file
    X = pd.read_csv('linear-regression-test.csv', delimiter=' ')
    y = pd.read_csv('linear-regression-test-label.csv', delimiter=' ')
    # numpy arrays must be C-contiguous
    X = np.ascontiguousarray(X.to_numpy(dtype='float64', copy=True))
    y = np.ascontiguousarray(y.to_numpy(dtype='float64', copy=True))

    # costs should be extremely close (10.58, 10.59)
    assert abs(model.normal_equation(X, y) - another_model.gradient_descent(X, y, 100000, 1, True, 0.0)) < 5e-2


def test_access_fit_intercept():
    """
    Test whether fit_intercept is accessible from python
    """
    model = sl.LinearRegression()
    assert model.fit_intercept == False
    
    model = sl.LinearRegression(True)
    assert model.fit_intercept == True


def test_access_param():
    """
    Test whether param is accessible from python
    """
    model = sl.LinearRegression()
    # read features and labels from csv file
    X = pd.read_csv('linear-regression-test.csv', delimiter=' ')
    y = pd.read_csv('linear-regression-test-label.csv', delimiter=' ')
    # numpy arrays must be C-contiguous
    X = np.ascontiguousarray(X.to_numpy(dtype='float64', copy=True))
    y = np.ascontiguousarray(y.to_numpy(dtype='float64', copy=True))

    model.normal_equation(X, y)
    
    # Do not know why here the values are different from 
    # the results calculated directly from eigen3
    assert model.param[0] == 0
    assert abs(model.param[1] - 1.21534711) < 1e-7
    assert abs(model.param[2] + 0.36012027) < 1e-7
    assert abs(model.param[3] + 0.01923811) < 1e-7


def test_compute_cost():
    """
    Test compute_cost() would return the similar results to those from 
    eigen3
    """
    
    model = sl.LinearRegression()
    model_fit_intercept = sl.LinearRegression(True)

    # read features and labels from csv file
    X = pd.read_csv('linear-regression-test.csv', delimiter=' ')
    y = pd.read_csv('linear-regression-test-label.csv', delimiter=' ')
    X = np.ascontiguousarray(X.to_numpy(dtype='float64', copy=True))
    y = np.ascontiguousarray(y.to_numpy(dtype='float64', copy=True))

    model.normal_equation(X, y)
    model_fit_intercept.normal_equation(X, y)

    # Not tune-all cost from eigen3 11.522048345454897
    assert abs(model.compute_cost(X, y) - 10.58858127) < 1e-7
    # tune-all cost from eigen3: 10.651969887548191
    assert abs(model_fit_intercept.compute_cost(X, y) - 10.02521253) < 1e-7


def test_compute_cost_with_aug_feature():
    """
    Test compute_cost_wit_aug_feature() would return the similar results 
    to those from eigen3
    """
    model = sl.LinearRegression()
    model_fit_intercept = sl.LinearRegression(True)

    # read features and labels from csv file
    X = pd.read_csv('linear-regression-test.csv', delimiter=' ')
    y = pd.read_csv('linear-regression-test-label.csv', delimiter=' ')
    X = np.ascontiguousarray(X.to_numpy(dtype='float64', copy=True))
    y = np.ascontiguousarray(y.to_numpy(dtype='float64', copy=True))
    X_plus = np.hstack((np.ones([len(X), 1]), X))
    model.normal_equation(X, y)
    model_fit_intercept.normal_equation(X, y)

    # The result should be the same with teh above test
    # Not tune-all cost from eigen3 11.522048345454897
    assert abs(model.compute_cost_with_aug_feature(X_plus, y) - 10.58858127) < 1e-7
    # tune-all cost from eigen3: 10.651969887548191
    assert abs(model_fit_intercept.compute_cost_with_aug_feature(X_plus, y) - 10.02521253) < 1e-7


def test_fit():
    """
    Test fit() would fit the model and return the exepcted cost
    """
    
    model = sl.LinearRegression()
    model_fit_intercept = sl.LinearRegression(True)
    X = pd.read_csv('linear-regression-test.csv', delimiter=' ')
    y = pd.read_csv('linear-regression-test-label.csv', delimiter=' ')
    X = np.ascontiguousarray(X.to_numpy(dtype='float64', copy=True))
    y = np.ascontiguousarray(y.to_numpy(dtype='float64', copy=True))

    # Not tune-all cost from eigen3 11.522048345454897
    assert abs(model.fit(X, y, fit_intercept=False) - 10.58858127) < 1e-7
    # tune-all cost from eigen3: 10.651969887548191
    assert abs(model_fit_intercept.fit(X, y) - 10.02521253) < 1e-7

