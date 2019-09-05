from allergies_different_labels import *
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from plot_confusion_matrix import *
from sklearn.metrics import recall_score, precision_score

df, mapping_file = allergies()
df = df.join(mapping_file, how='inner')

df = df.loc[(df['AllergyType'] == 1) | (df['AllergyType'] == 0)]
df = df.drop(['AllergyType', 'SuccessDescription'], axis =1)
mapping_file = mapping_file.loc[(mapping_file['AllergyType']  == 1) | (mapping_file['AllergyType']  == 0)]

cols = [col for col in df.columns if len(df[col].unique()) != 1]
dist_mat = pd.DataFrame(columns = cols, index = cols)
df = df[cols]
def allergies_distance_matrix(distance = 'spearman', clustering='spectral'):
    auc =[]
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
    pca = PCA(n_components=round(df0.shape[1] / 2) + 1)
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
    X = merged_data0.drop(['AllergyType', 'SuccessDescription'], axis =1)
    y = merged_data0['AllergyType']
    loo = LeaveOneOut()
    accuracy = []
    y_pred_list = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=15 / 100,
                                               objective= 'binary:logistic', reg_lambda=300)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
    auc.append(metrics.roc_auc_score(y, y_pred_list))

    X = merged_data0.drop(['AllergyType', 'SuccessDescription'], axis=1)
    y = merged_data0['SuccessDescription']
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
    auc.append(metrics.roc_auc_score(y, y_pred_list))

    #
    auc2 =[]
    pca = PCA(n_components=round(df1.shape[1] / 2) + 1)
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
    X = merged_data1.drop(['AllergyType', 'SuccessDescription'], axis =1)
    y = merged_data1['AllergyType']
    loo = LeaveOneOut()
    accuracy = []
    y_pred_list = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=15 / 100,
                                               objective= 'binary:logistic', reg_lambda=300)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
    auc2.append(metrics.roc_auc_score(y, y_pred_list))

    X = merged_data1.drop(['AllergyType', 'SuccessDescription'], axis=1)
    y = merged_data1['SuccessDescription']
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
    auc2.append(metrics.roc_auc_score(y, y_pred_list))
    return auc, auc2
#for distance in ['spearman', 'euclidean']:
distance = 'euclidean'
clust = 'spectral'
#for clust in ['spectral', 'hierarchical']:
print ( '####' + str(distance) + '#####' + str(clust))
auc, auc2 = allergies_distance_matrix(distance=distance, clustering=clust)

def plot_auc(auc, auc2, distance, clust):
    plt.figure(1)
    plt.subplot(211)
    plt.title(distance + ' ' + clust)
    plt.bar(['MilkvsPeanut', 'TreatSuccess'], auc, color='blue', label='1st group')
    plt.ylabel('1st group')
    plt.subplot(212)
    plt.bar(['MilkvsPeanut', 'TreatSuccess'], auc2, color='orange', label='2nd group')
    plt.ylabel('2nd group')
    plt.show()

plot_auc(auc, auc2, distance, clust)