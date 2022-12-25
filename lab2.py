from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Принимает на вход X, y и вычисляет веса по данной выборке
        
        n, k = X.shape
        
        X_train = X
        if self.fit_intercept:
            # Добавляем доп столбец единиц для включения свободного члена в матричный вид
            X_train = np.hstack((X, np.ones((n, 1))))

        self.w = (1/(X_train.T @ X_train)) @ X_train.T @ y # сам написал
        
        return self
        
    def predict(self, X):
        # Принимает на вход X и возвращает ответы модели

        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = X_train @ self.w

        return y_pred
    
    def get_weights(self):
        return self.w

def mean_squared_error(y_real: np.ndarray, y_predict: np.ndarray) -> float:
    return 1/len(y_predict) * np.sum((y_predict-y_real)*(y_predict-y_real))

linear_func = lambda x, w_0, w_1: w_0 + w_1 * x

w_0, w_1 = 0.1, 0.2

# по признакам сгенерируем значения таргетов с некоторым шумом
objects_num = 100
X = np.linspace(-10, 10, objects_num)
y = linear_func(X, w_0, w_1) + np.random.randn(objects_num) * 2.5


# выделим 30% объектов на тест
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
# plt.figure(figsize=(13, 7))
# plt.plot(X, linear_func(X, w_0, w_1), label='real', c='r')
# plt.scatter(X_train, y_train, label='train', c='b')
# plt.scatter(X_test, y_test, label='test', c='g')

# plt.title("Generated dataset")
# plt.grid(alpha=0.7)
# plt.legend()
# plt.show()

my_regr = MyLinearRegression()

my_regr.fit(X_train[:, np.newaxis], y_train)

predictions = my_regr.predict(X_test[:, np.newaxis])
w = my_regr.get_weights()

lib_regr = LinearRegression()
lib_regr.fit(X_train[:, np.newaxis], y_train)

# plt.figure(figsize=(13, 7))
# plt.plot(X, linear_func(X, w_0, w_1), label='real', c='r', alpha=0.5)

# plt.scatter(X_train, y_train, label='train')
# plt.scatter(X_test, y_test, label='test')
# plt.plot(X, my_regr.predict(X[:, np.newaxis]), label='ours', c='orange', linestyle='--')
# plt.plot(X, lib_regr.predict(X[:, np.newaxis]), label='sklearn', c='b', linestyle='dotted')

# plt.title("Different Prediction")
# plt.ylabel('target')
# plt.xlabel('feature')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

my_train_pred = my_regr.predict(X_train[:, np.newaxis])
my_test_pred = my_regr.predict(X_test[:, np.newaxis])

print(f'My model train MSE: {mean_squared_error(y_train, my_train_pred):.2f}')
print(f'My model test MSE: {mean_squared_error(y_test, my_test_pred):.2f}')

w0 = 0
w1 = np.linspace(start=-10, stop=10, num=objects_num)

# реальные значения
# real = y
real = linear_func(X, w_0, w_1)

# для каждого значения w1 нужно расчитать значения модели
mean_squared_errors = mean_squared_error(real, linear_func(X, w0, w1))

# для каждого значения predict`а нужно расчитать ошибку от реального значения
# for w1instance in w1:
#     mean_squared_errors.append(mean_squared_error(real, linear_func(X, w0, w1instance)))

opt_result = minimize_scalar(lambda x : mean_squared_error(real, linear_func(X, w0, x)))

w_opt = opt_result.x

# print(w_opt)

# plt.scatter(X, y)
# plt.plot(X, linear_func(X, 0, w_opt), color='red')
# plt.show()

# plt.plot(w1, mean_squared_errors)
# plt.show()

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d') # get current axis

w0 = np.linspace(start=-5, stop=5, num=objects_num)
w1 = np.linspace(start=-5, stop=5, num=objects_num)
mean_squared_errors = []

for w0instance, w1instance in zip(w0, w1):
    mean_squared_errors.append(mean_squared_error(real, linear_func(X, w0instance, w1instance)))



opt_result = minimize(lambda x : mean_squared_error(real, linear_func(X, x[0], x[1])), x0=[0,0])

print(opt_result)

# surf = ax.plot_surface(w0, w1, np.array(mean_squared_errors)[:, np.newaxis])
# ax.set_xlabel('w0')
# ax.set_ylabel('w1')
# ax.set_zlabel('Error')
# plt.show()