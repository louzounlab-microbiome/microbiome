from allergies import *
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from plot_confusion_matrix import *
from sklearn.metrics import recall_score, precision_score
level = 5
df, mapping_file, dict_bact, taxonomy  = allergies(perform_distance=True,level =level)
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
    y_pred_train = model.predict(X_train)
    print('Train Precision: ' + str(round(precision_score(y_train, y_pred_train), 2)))
    print('Train Recall: ' + str(round(recall_score(y_train, y_pred_train), 2)))

    cnf_matrix = metrics.confusion_matrix(y_train, y_pred_train)
    class_names = ['Milk', 'Peanuts']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                          title='Normalized confusion matrix(train)')
    plt.show()
    print('Precision: ' + str(round(precision_score(y, y_pred_list), 2)))
    print('Recall: ' + str(round(recall_score(y, y_pred_list), 2)))

    cnf_matrix = metrics.confusion_matrix(y, y_pred_list)
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
        clustering = SpectralClustering(n_clusters=2,   affinity= 'precomputed', assign_labels ='discretize', random_state=0)
    else:
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='average')
    clustering.fit(dist_mat.values)
    bact_label = {0: [], 1: []}

    for i in range(0, df.shape[1]):
        bact_label[clustering.labels_[i]].append(df.columns[i])

    bact_label_name = {0: [], 1: []}
    bact_label_tmp = {0: [], 1: []}
    bact_level = level - 1
    for k in [0, 1]:
        for i in bact_label[k]:
            for key, value in dict_bact.items():
                for j in value:
                    if i == j:
                        bact_label_tmp[k].append(key)
        bact_label_tmp[k] = set(bact_label_tmp[k])
        for i in bact_label_tmp[k]:
            if i != 'else':
                for j in taxonomy:
                    try:
                        if j.split(';')[bact_level] == i:
                            bact_label_name[k].append(','.join(j.split(';')[0:bact_level + 1]))
                            break
                    except:
                        continue
            else:
                bact_label_name[k].append('else')
        bact_label_name[k] = set(bact_label_name[k])
    df0 = df[bact_label[0]]
    df1 = df[bact_label[1]]
    # df2 = df[bact_label[2]]
    # df3 = df[bact_label[3]]
    # df4 = df[bact_label[4]]

    print(len(bact_label_name[0]))
    pca_and_conf_matrix_per_group(df0)
    print(len(bact_label_name[1]))
    pca_and_conf_matrix_per_group(df1)
    # print(len(bact_label_name[2]))
    # pca_and_conf_matrix_per_group(df2)
    # print(len(bact_label_name[3]))
    # pca_and_conf_matrix_per_group(df3)
    # print(len(bact_label_name[4]))
    # pca_and_conf_matrix_per_group(df4)

    print('done')

distance = 'euclidean'
# for clustering in ['spectral', 'hierarchical']:
#    for distance in ['spearman', 'euclidean']:
clustering = 'hierarchical'
print ( '####' + str(distance) + '#####' + str(clustering))
allergies_distance_matrix(distance=distance, clustering=clustering)


