# assumes a linear relationship between independent and dependent variables

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_dataset(input_file):
    df = pd.read_csv(input_file) # sorted by discovery date
    df = df.dropna(how='any')
    X = df[['air_temperature', 'relative_humidity', 'wind_speed', 'precipitation']]
    y = df[['FIRE_SIZE']]
    return X, y


def train_test_split(X, y, split=0.8):
    split_size = int(X.shape[0] * split)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    return X_train, y_train, X_test, y_test


def MLR_train(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def SVR_train(X, y):
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    # svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    # svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
    svr_rbf.fit(X, y)
    return svr_rbf

def ENR_train(X, y):
    model = ElasticNetCV()
    model.fit(X, y)
    return model


def predict(model, X):
    return model.predict(X)


def calculate_mae(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)


if __name__ == "__main__":
    input_file = sys.argv[1]
    X, y = load_dataset(input_file)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # Multiple Linear Regression
    MLR_model = MLR_train(X_train, y_train)
    MLR_y_pred = predict(MLR_model, X_test)
    print("MLR: \n\tR2 Score: {}\n\tMAE Error: {}".format(
        r2_score(y_test, MLR_y_pred),
        calculate_mae(y_test, MLR_y_pred)
    ))

    # Elastic Net regression
    ENR_model = ENR_train(X_train, y_train)
    ENR_y_pred = predict(ENR_model, X_test)
    print("ENR: \n\tR2 Score: {}\n\tMAE Error: {}".format(
        r2_score(y_test, ENR_y_pred),
        calculate_mae(y_test, ENR_y_pred)
    ))

    # TODO: Optimal Subset Regression
    # gets best fitting model that contain one or more predictors from a set of variables

    
    # TODO: Stepwise Regression
    
    
    # TODO: Random Forest Regression
    # ensemble method



    # Support Vector Regression
    # assumes a linear relationship between independent and dependent variables
    # https://scikit-learn.org/stable/modules/svm.html#regression
    SVR_model = SVR_train(X_train, y_train)
    SVR_y_pred = predict(SVR_model, X_test)
    print("SVR: \n\tR2 Score: {}\n\tMAE Error: {}".format(
        r2_score(y_test, SVR_y_pred),
        calculate_mae(y_test, SVR_y_pred)
    ))
