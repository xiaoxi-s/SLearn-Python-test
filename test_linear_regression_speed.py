import slearn.linear_model as sl
import sklearn.linear_model as sk
import sklearn.datasets as ds
import timeit as tm
import numpy as np

TEST_SLEARN = '''
X, y = ds.load_boston(True)
sl_model = sl.LinearRegression()
sl_model.gaussian_init(13)
sl_model.fit(X, y) '''


TEST_SKLEARN = '''
X, y = ds.load_boston(True)
sk_model = sk.LinearRegression()
sk_model.fit(X, y)'''


SETUP_CODE = '''
import slearn.linear_model as sl
import sklearn.linear_model as sk
import sklearn.datasets as ds
'''


def test_speed():
    print('SLearn time used: ' + str(tm.timeit(stmt=TEST_SLEARN,setup=SETUP_CODE, number=100)))
    print('Sklearn time used: ' + str(tm.timeit(stmt=TEST_SKLEARN, setup=SETUP_CODE, number=100)))


def benchmark():
    X, y = ds.load_boston(True)
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    sl_model = sl.LinearRegression()
    sl_model.gaussian_init(13)

    sl_model.fit(X, y)
    print("slearn model loss " + str(np.square(np.sum(sl_model.predict(X) - y))))
    #print("slearn model loss " + str(sl_model.gradient_descent(X, y, 10000000, 0.0000001)))

    sk_model = sk.LinearRegression()
    sk_model.fit(X, y)
    print("sklearn model loss " + str(np.square(np.sum (sk_model.predict(X) - y))))


test_speed()

benchmark()