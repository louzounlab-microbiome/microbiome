from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def fit_SVR(samples_data, samples_values, **kwargs):
    clf = SVR(**kwargs)
    return clf.fit(samples_data, samples_values)


def fit_random_forest(samples_data, samples_values, **kwargs):
    clf = RandomForestRegressor(**kwargs)
    return clf.fit(samples_data, samples_values)