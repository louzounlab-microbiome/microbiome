from sklearn import svm
from xgboost import XGBClassifier

if __name__ == "__main__":
    selected_parameters = {'kernel': 'linear', 'C': 0.01, 'gamma': 'auto'}
    clf = svm.SVC(kernel=selected_parameters['kernel'], C=selected_parameters['C'],
                          gamma=selected_parameters['gamma'], class_weight='balanced')
    clf = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000, objective='binary:logistic',
                        gamma=6, min_child_weight=5, sample_weight='balanced', booster='gblinear')
    print(clf)