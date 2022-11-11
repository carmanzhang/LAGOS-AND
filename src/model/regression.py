import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor

from model.available_model import ModelName

warnings.filterwarnings('ignore')


def use_regression(X_train, Y_train, X_test, model_switch: str):
    if model_switch == ModelName.linear:
        pred_y, feature_importance = linear_regressor(X_train, Y_train, X_test)
    elif model_switch == ModelName.logistic:
        pred_y, coefs, _ = logistic_regressor(X_train, Y_train, X_test)
        feature_importance = np.array(coefs[0])
    elif model_switch == ModelName.dt:
        pred_y, feature_importance = dt_regressor(X_train, Y_train, X_test)
    elif model_switch == ModelName.randomforest:
        pred_y, feature_importance = randomforest_regressor(X_train, Y_train, X_test)
    else:
        pass
    return pred_y, feature_importance


def linear_regressor(X_train, Y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict(X_test)
    print('used time: ', time.time() - s)
    return y_pred, model.coef_


def logistic_regressor(X_train, Y_train, X_test):
    # model = LogisticRegression(max_iter=1000, solver='newton-cg', tol=1e-5)
    model = LogisticRegression(max_iter=1000, tol=1e-4, class_weight='balanced', C=2)
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict_proba(X_test)
    print('used time: ', time.time() - s)
    # metrics = calc_metrics(Y_train, [p1 for (p0, p1) in y_pred])
    # pprint(metrics, pctg=True)
    y_pred = [p1 for (p0, p1) in y_pred]
    print(model.coef_, model.intercept_)
    return y_pred, model.coef_, model.intercept_


def dt_regressor(X_train, Y_train, X_test):
    model = DecisionTreeRegressor(max_depth=6)  # max_depth=,
    model.fit(X_train, Y_train)
    # depth = model.get_depth()
    # for i in range(depth):
    #     print(model.get_params(i+1))
    s = time.time()
    y_pred = model.predict(X_test)
    print('used time: ', time.time() - s)
    return y_pred, model.feature_importances_


def randomforest_regressor(X_train, Y_train, X_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_train)
    # s = time.time()
    y_pred = model.predict(X_test)
    # print('used time: ', time.time() - s)
    return y_pred, model.feature_importances_
