from preprocess_and_distance_PSC_2classes import *
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from plot_confusion_matrix import *
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from xgboost import XGBClassifier

df, mapping_file, bact_dict = psc(perform_distance=True,level =4)
cols = [col for col in df.columns if len(df[col].unique()) != 1]
dist_mat = pd.DataFrame(columns = cols, index = cols)
df = df[cols]
def pca_and_conf_matrix_per_group(df):
    pca = PCA(n_components=min(round(df.shape[1] / 2) + 1, df.shape[0]))
    pca.fit(df)
    sum = 0
    num_comp = 0
    for (i, component) in enumerate(pca.explained_variance_ratio_):
        if sum <= 0.5:
            sum += component
        else:
            num_comp = i
            break
    if num_comp == 0:
        num_comp += 1

    otu_after_pca0, _ = apply_pca(df, n_components=num_comp, print_data=False)
    merged_data0 = otu_after_pca0.join(mapping_file)
    X = merged_data0.drop(['DiagnosisGroup'], axis=1)
    y = merged_data0['DiagnosisGroup']
    loo = LeaveOneOut()
    accuracy = []
    y_pred_list = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=4, n_estimators=100, learning_rate=10 / 100,
                              objective='binary:logistic', reg_lambda=300,
                              scale_pos_weight=(np.sum(y_train == 0) / np.sum(y_train == 1)))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
        y_pred_train =  model.predict(X_train)
    print('Precision train: ' + str(round(precision_score(y_train, y_pred_train), 2)))
    print('Recall train: ' + str(round(recall_score(y_train, y_pred_train), 2)))
    print('Precision: ' + str(round(precision_score(y, y_pred_list), 2)))
    print('Recall: ' + str(round(recall_score(y, y_pred_list), 2)))
    cnf_matrix = metrics.confusion_matrix(y_train, y_pred_train)
    class_names = ['Control', 'Sick']

    # cnf_matrix = metrics.confusion_matrix(y, y_pred_list)
    # class_names = ['Control', 'Sick']
    # # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

pca_and_conf_matrix_per_group(df)