from preprocess_and_distance_GVHD import *
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from plot_confusion_matrix import *
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from xgboost import XGBClassifier

df, mapping_file = gvhd(perform_distance=False,level =5)
cols = [col for col in df.columns if len(df[col].unique()) != 1]
dist_mat = pd.DataFrame(columns = cols, index = cols)
df = df[cols]
def allergies_distance_matrix(distance = 'spearman', clustering='spectral'):
    for i in range(0,df.shape[1]):
        for j in range(0,df.shape[1]):
            #Spearman correlation
            if distance == 'spearman':
                dist_mat.at[df.columns[i],df.columns[j]] = abs(round(scipy.stats.spearmanr(np.array(df.iloc[:,i]).astype(float),np.array(df.iloc[:,j]).astype(float))[0],4))
            #Euclidean distance
            else:
                dist_mat.at[df.columns[i],df.columns[j]]  =np.linalg.norm(np.array(df.iloc[:,i]).astype(float) - np.array(df.iloc[:,j]).astype(float))
    if clustering=='spectral':
        clustering = SpectralClustering(n_clusters=2,   affinity= 'precomputed', assign_labels ='discretize', random_state=0)
    else:
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='average')
    clustering.fit(dist_mat.values)
    bact_label1 =[]
    bact_label0 =[]
    for i in range(0,df.shape[1]):
        if clustering.labels_[i]==1:
            bact_label1.append(df.columns[i])
        else:
            bact_label0.append(df.columns[i])

    df1 = df[bact_label1]
    df0 = df[bact_label0]
    pca = PCA(n_components=min(round(df0.shape[1] / 2) + 1, df0.shape[0]))
    pca.fit(df0)
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

    otu_after_pca0, _ = apply_pca(df0, n_components=num_comp, print_data=False)
    merged_data0 = otu_after_pca0.join(mapping_file)
    X = merged_data0.drop(['disease'], axis =1)
    y = merged_data0['disease']
    loo = LeaveOneOut()
    accuracy = []
    y_pred_list = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=15 / 100,
                              objective='binary:logistic',
                              scale_pos_weight=(np.sum(y_train == 0) / np.sum(y_train == 1)),
                              reg_lambda=450)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)

    print( 'Precision: ' + str(round(precision_score(y,y_pred_list),2)))
    print( 'Recall: ' + str(round(recall_score(y, y_pred_list),2)))

    cnf_matrix = metrics.confusion_matrix(y,y_pred_list)
    class_names = ['Healthy', 'GVHD']
    # # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                             title='Normalized confusion matrix')
    plt.show()
    #
    pca = PCA(n_components=min(round(df1.shape[1] / 2) + 1, df1.shape[0]))
    pca.fit(df1)
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

    otu_after_pca1, _ = apply_pca(df1, n_components=num_comp,print_data=False)
    merged_data1 = otu_after_pca1.join(mapping_file)
    X = merged_data1.drop(['disease'], axis =1)
    y = merged_data1['disease']
    loo = LeaveOneOut()
    accuracy = []
    y_pred_list = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=15 / 100,
                              objective='binary:logistic',
                              scale_pos_weight=(np.sum(y_train == 0) / np.sum(y_train == 1)),
                              reg_lambda=450)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
    print( 'Precision: ' + str(round(precision_score(y,y_pred_list),2)))
    print( 'Recall: ' + str(round(recall_score(y, y_pred_list),2)))
    cnf_matrix = metrics.confusion_matrix(y,y_pred_list)
    class_names = ['Healthy', 'GVHD']
    # # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                             title='Normalized confusion matrix')
    plt.show()

for distance in ['spearman', 'euclidean']:
#distance = 'spearman'
    for clustering in ['spectral', 'hierarchical']:
        print ( '####' + str(distance) + '#####' + str(clustering))
        allergies_distance_matrix(distance=distance, clustering=clustering)


