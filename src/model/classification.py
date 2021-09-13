import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from model.available_model import ModelName

warnings.filterwarnings('ignore')


def use_classifier(X_train, Y_train, X_test, model_switch: str):
    if model_switch == ModelName.linear:
        pred_y, feature_importance = linear_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.logistic:
        pred_y, coefs, _ = logistic_classifier(X_train, Y_train, X_test)
        feature_importance = np.array(coefs[0])
    elif model_switch == ModelName.dt:
        pred_y, feature_importance = dt_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.svm:
        pred_y, feature_importance = svm_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.xgboost:
        pred_y, feature_importance = xgboost_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.randomforest:
        pred_y, feature_importance = randomforest_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.gb:
        pred_y, feature_importance = gb_classifier(X_train, Y_train, X_test)
    elif model_switch == ModelName.mlp:
        pred_y, feature_importance = mlp_classifier(X_train, Y_train, X_test)
    else:
        pass
    return pred_y, feature_importance


def linear_classifier(X_train, Y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    return y_pred, model.coef_


def logistic_classifier(X_train, Y_train, X_test):
    model = LogisticRegression(max_iter=1000, tol=1e-4, class_weight='balanced', C=2)
    model.fit(X_train, Y_train)
    y_pred = model.predict_proba(X_test)
    y_pred = [p1 for (p0, p1) in y_pred]
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    return y_pred, model.coef_, model.intercept_


def dt_classifier(X_train, Y_train, X_test):
    model = DecisionTreeClassifier(ccp_alpha=0,
                                   criterion='gini',
                                   max_depth=5,
                                   max_features=None)

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred, model.feature_importances_


def svm_classifier(X_train, Y_train, X_test):
    model = SVC()  # max_depth=,
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred, []


def xgboost_classifier(X_train, Y_train, X_test):
    params = {'colsample_bytree': 0.9, 'reg_alpha': 3, 'reg_lambda': 1, 'random_state': int(time.time()),
              'learning_rate': 0.01,
              'n_estimators': 100,
              'max_depth': 8,
              'min_child_weight': 2,
              'gamma': 0.1,
              'subsample': 0.7,
              }
    model = XGBClassifier(**params)
    # return model
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred, model.feature_importances_


def randomforest_classifier(X_train, Y_train, X_test):
    model = RandomForestClassifier(n_estimators=100,
                                   criterion="gini",
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   max_features='auto',  # "auto" class_weight='balanced'
                                   )
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred, model


def gb_classifier(X_train, Y_train, X_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred, model.feature_importances_


def mlp_classifier(X_train, Y_train, X_test):
    model = MLPClassifier(hidden_layer_sizes=(50, 25, 6), activation='logistic', solver='adam', alpha=0.0001,
                          batch_size=32, learning_rate_init=0.0005, max_iter=40,
                          beta_1=0.6, beta_2=0.75, epsilon=1e-8, n_iter_no_change=3,
                          shuffle=True, verbose=False)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return y_pred, []
