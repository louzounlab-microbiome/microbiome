from allergies import *


df, mapping_file = allergies(perform_distance=False,level =3)


max_num_of_pcas =40
train_accuracy = []
test_accuracy = []
pcas =[]
def pca_graph(max_num_of_pcas = max_num_of_pcas,train_accuracy_all=train_accuracy, test_accuracy_all = test_accuracy, pcas =pcas, df = df, mapping_file = mapping_file):
    for i in range (2,max_num_of_pcas):
        pcas.append(i)
        otu_after_pca, _ = apply_pca(df, n_components=30)
        merged_data = otu_after_pca.join(mapping_file)

        X = merged_data.drop(['AllergyType'], axis=1)
        y = merged_data['AllergyType']
        loo = LeaveOneOut()

        y_pred_list = []
        auc = []
        auc_train = []
        for train_index, test_index in loo.split(X):
            train_index = list(train_index)
            # print("%s %s" % (train_index, test_index))
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            model = XGBClassifier(max_depth=3, n_estimators=250, learning_rate=15 / 100,
                                  objective='binary:logistic',
                                  reg_lambda=250)
            model.fit(X_train, y_train)
            pred_train = model.predict_proba(X_train)[:, 1]
            auc_train.append(metrics.roc_auc_score(y_train, pred_train))
            y_pred = model.predict_proba(X_test)[:, 1]
            y_pred_list.append(y_pred[0])

        auc = metrics.roc_auc_score(y, y_pred_list)
        print('PCA components' + str(i) + ' ', round(auc, 2))
        scores = round(auc, 2)
        scores_train = round(np.array(auc_train).mean(), 2)
        train_accuracy_all.append(scores_train)
        test_accuracy_all.append(round(scores.mean(), 2))

pca_graph(max_num_of_pcas = max_num_of_pcas,train_accuracy_all=train_accuracy, test_accuracy_all = test_accuracy, df=df,
          mapping_file=mapping_file,pcas=pcas)

df_dist, mapping_file_dist = allergies(perform_distance=True,level =3)
train_accuracy_dist = []
test_accuracy_dist = []
pcas_dist =[]
pca_graph(max_num_of_pcas = max_num_of_pcas,train_accuracy_all=train_accuracy_dist, test_accuracy_all = test_accuracy_dist, df=df_dist,
          mapping_file=mapping_file_dist,pcas=pcas_dist)

def plot_graph(test_accuracy, train_accuracy, train_accuracy_dist,  test_accuracy_dist, pcas, pcas_dist):
    plt.plot(pcas, test_accuracy, color='orange', label='test')
    plt.plot(pcas, train_accuracy, color='black', label='train')
    plt.plot(pcas_dist, test_accuracy_dist, color='red', label='test_fs')
    plt.plot(pcas_dist, train_accuracy_dist, color='blue', label='train_fs')
    plt.legend( loc=1,ncol=1)
    plt.show()
plot_graph(test_accuracy=test_accuracy,train_accuracy=train_accuracy,train_accuracy_dist=train_accuracy_dist, test_accuracy_dist=test_accuracy_dist,
           pcas=pcas,pcas_dist=pcas_dist)
print('done')
