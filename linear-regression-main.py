# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def mean_squared_error_my_impl():
    Y_true = [1, 1, 2, 2, 4]  # Y_true = Y (original values)

    # Calculated values
    Y_pred = [0.6, 1.29, 1.99, 2.69, 3.4]  # Y_pred = Y'

    # Mean Squared Error
    MSE = np.square(np.subtract(Y_true, Y_pred)).mean()
    print(MSE)


def r_squared():
    x_values = [1, 2, 3]
    y_values = [1, 5, 25]

    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2

    print(r_squared)


def linear_model():
    boston = datasets.load_boston()
    bos = pd.DataFrame(boston.data, columns=boston.feature_names)
    bos['PRICE'] = boston.target
    X_rooms = bos.RM
    y_price = bos.PRICE

    X_rooms = np.array(X_rooms).reshape(-1, 1)
    y_price = np.array(y_price).reshape(-1, 1)
    X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_rooms, y_price, test_size=0.2, random_state=5)

    reg_1 = LinearRegression()
    reg_1.fit(X_train_1, Y_train_1)

    x_new = np.linspace(0, 30, 100)
    y_train_predict_1 = reg_1.predict(x_new[:, np.newaxis])
    rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))
    r2 = round(reg_1.score(X_train_1, Y_train_1), 2)


# plot the results
    plt.figure(figsize=(4, 3))
    ax = plt.axes()
    ax.scatter(X_train_1, Y_train_1)
    ax.plot(x_new, y_train_predict_1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.axis('tight')

    plt.show()

    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

if __name__ == '__main__':
    mean_squared_error_my_impl()
    r_squared()
    linear_model()
