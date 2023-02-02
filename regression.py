import data_processing as dp
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from scikeras.wrappers import KerasRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

def print_scores(scores):
    print("MSE mean: %.2f STD: (%.2f)" % (
        -scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
    print("RMSE mean: %.2f STD: (%.2f)" % (
        -scores['test_neg_root_mean_squared_error'].mean(), scores['test_neg_root_mean_squared_error'].std()))
    print("MAE mean: %.2f STD: (%.2f)" % (
        -scores['test_neg_mean_absolute_error'].mean(), scores['test_neg_mean_absolute_error'].std()))

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(63, input_shape=(63,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def neural_network(df, target, index):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
    # Repeated K Fold
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(estimator, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                             'neg_root_mean_squared_error'])
    print(f"---- NEURAL NETWORK RESULTS FOR {target} ----")
    print_scores(scores)


def decision_tree(df, target, index):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    dtr = DecisionTreeRegressor(random_state = 1)
    # Repeated K Fold

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(dtr, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                             'neg_root_mean_squared_error'])
    print(f"---- DECISION TREE REGRESSION FOR {target} ----")
    print_scores(scores)


def lasso(df, target, index, testing_df):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    working_testing_df = testing_df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)
    working_testing_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)
    normalized_testing_df = dp.min_max(working_testing_df, working_testing_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    lasso = Lasso(alpha=1.0, random_state=1)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    for train_index, test_index in cv.split(X):
        train = normalized_data.loc[train_index, :]
        x_train = train.drop(target, axis=1)
        y_train = train[target]
        lasso.fit(x_train, y_train)

    scores = cross_validate(lasso, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                       'neg_root_mean_squared_error'])
    print(f"---- LASSO REGRESSION FOR {target} ----")
    print_scores(scores)

    y_test_final = normalized_testing_df[target]
    x_test_final = normalized_testing_df.drop(target, axis=1)
    prediction = lasso.predict(x_test_final)
    pred_mae = mean_absolute_error(y_test_final, prediction)
    pred_mse = mean_squared_error(y_test_final, prediction)

    print(f"Prediction results of {target} MAE: {pred_mae} MSE: {pred_mse} ")
    print()

def random_forest(df, target, index):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    rfr = RandomForestRegressor(n_estimators=10, random_state=1)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(rfr, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                         'neg_root_mean_squared_error'])
    print(f"---- RANDOM FOREST REGRESSION FOR {target} ----")
    print_scores(scores)

def knn(df, target, index):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    knn_model = KNeighborsRegressor(n_neighbors=5)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(knn_model, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                         'neg_root_mean_squared_error'])
    print(f"---- KNN MODEL REGRESSION FOR {target} ----")
    print_scores(scores)

def support_vector_regression(df, target, index):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(svr_rbf, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                             'neg_root_mean_squared_error'])
    print(f"---- SUPPORT VECTOR REGRESSION FOR {target} ----")
    print_scores(scores)

def gaussian_process_regressor(df, target, index):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    scores = cross_validate(gpr, X, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                           'neg_root_mean_squared_error'])
    print(f"---- GAUSSIAN REGRESSION FOR {target} ----")
    print_scores(scores)

def polynomial_regression(df, target, index):
    # drop extra column
    working_df = df.drop(columns=df.columns[-index],  axis=1,  inplace=False)
    # drop rows with missing values
    working_df.dropna(axis=0, inplace=True)

    normalized_data = dp.min_max(working_df, working_df.columns, 0.0, 1.0)

    y = normalized_data[target]
    X = normalized_data.drop(target, axis=1)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X)
    poly_reg_model = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(poly_reg_model, poly_features, y, cv=cv, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                       'neg_root_mean_squared_error'])
    print(f"---- POLYNOMIAL REGRESSION FOR {target} ----")
    print_scores(scores)

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    training_df = df.iloc[:63,:]
    testing_df = df.iloc[63:,:]

    neural_network(training_df, 'TIMEsurvival_Years', 1)
    neural_network(training_df, 'TIMErecurrence_Years', 2)
    print()

    decision_tree(training_df, 'TIMEsurvival_Years', 1)
    decision_tree(training_df, 'TIMErecurrence_Years', 2)
    print()

    lasso(training_df, 'TIMEsurvival_Years', 1, testing_df)
    lasso(training_df, 'TIMErecurrence_Years', 2, testing_df)
    print()

    random_forest(training_df, 'TIMEsurvival_Years', 1)
    random_forest(training_df, 'TIMErecurrence_Years', 2)
    print()

    knn(training_df, 'TIMEsurvival_Years', 1)
    knn(training_df, 'TIMErecurrence_Years', 2)
    print()

    support_vector_regression(training_df, 'TIMEsurvival_Years', 1)
    support_vector_regression(training_df, 'TIMErecurrence_Years', 2)
    print()

    gaussian_process_regressor(training_df, 'TIMEsurvival_Years', 1)
    gaussian_process_regressor(training_df, 'TIMErecurrence_Years', 2)
    print()

    polynomial_regression(training_df, 'TIMEsurvival_Years', 1)
    polynomial_regression(training_df, 'TIMErecurrence_Years', 2)
    print()
