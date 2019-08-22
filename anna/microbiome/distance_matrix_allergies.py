from allergies import *
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from plot_confusion_matrix import *
from sklearn.metrics import recall_score, precision_score

df, mapping_file = allergies(perform_distance=False,level =5)
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
        clustering = SpectralClustering(n_clusters=6,   affinity= 'precomputed', assign_labels ='discretize', random_state=0)
    else:
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='average')
    clustering.fit(dist_mat.values)
    bact_label1 =[]
    bact_label0 =[]
    bact_label2 = []
    bact_label3 = []
    bact_label4 = []
    bact_label5 = []

    for i in range(0,df.shape[1]):
        if clustering.labels_[i]==1:
            bact_label1.append(df.columns[i])
        elif clustering.labels_[i]==0:
            bact_label0.append(df.columns[i])
        elif clustering.labels_[i]==2:
            bact_label2.append(df.columns[i])
        elif clustering.labels_[i]==3:
            bact_label3.append(df.columns[i])
        elif clustering.labels_[i] == 4:
            bact_label4.append(df.columns[i])
        else:
            bact_label5.append(df.columns[i])

    df1 = df[bact_label1]
    df0 = df[bact_label0]
    df2 = df[bact_label2]
    df3 = df[bact_label3]
    df4 = df[bact_label4]
    df5 = df[bact_label5]

    pca_and_conf_matrix_per_group(df0)
    pca_and_conf_matrix_per_group(df1)
    pca_and_conf_matrix_per_group(df2)
    pca_and_conf_matrix_per_group(df3)
    pca_and_conf_matrix_per_group(df4)
    pca_and_conf_matrix_per_group(df5)
    print('done')

#for distance in ['spearman', 'euclidean']:
distance = 'euclidean'
clustering = 'spectral'
#for clustering in ['spectral', 'hierarchical']:
print ( '####' + str(distance) + '#####' + str(clustering))
allergies_distance_matrix(distance=distance, clustering=clustering)


