from allergies import *
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from plot_confusion_matrix import *
from sklearn.metrics import recall_score, precision_score
from sklearn import mixture

df, mapping_file = allergies(perform_distance=False,level =5)
cols = [col for col in df.columns if len(df[col].unique()) != 1]
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
    X = merged_data0.drop(['AllergyType'], axis=1)
    y = merged_data0['AllergyType']
    loo = LeaveOneOut()
    accuracy = []
    y_pred_list = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=15 / 100,
                              objective='binary:logistic', reg_lambda=300)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)

    print('Precision: ' + str(round(precision_score(y, y_pred_list), 2)))
    print('Recall: ' + str(round(recall_score(y, y_pred_list), 2)))

    cnf_matrix = metrics.confusion_matrix(y, y_pred_list)
    class_names = ['Milk', 'Peanuts']
    # # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
otu_after_pca, _ = apply_pca(df.T, n_components=5)
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(otu_after_pca)
bact_label = {0:[],1:[]}
gmm_predicted = gmm.predict_proba(otu_after_pca)
for i in range(0,len(gmm_predicted)):
    for j in gmm_predicted[i]:
        if j>0.5:
            bact_label[int(list(gmm_predicted[i]).index(j))].append(otu_after_pca.index.values[i])

for key, value in bact_label.items():
    print (key, len(value))

df0 = df[bact_label[0]]
df1 = df[bact_label[1]]



pca_and_conf_matrix_per_group(df0)
pca_and_conf_matrix_per_group(df1)



print('done')
