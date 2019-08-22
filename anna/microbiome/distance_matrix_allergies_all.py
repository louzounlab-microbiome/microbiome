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


otu_after_pca0, _ = apply_pca(df, n_components=21, print_data=True)
merged_data0 = otu_after_pca0.join(mapping_file)
X = merged_data0.drop(['AllergyType'], axis =1)
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
print( 'Precision: ' + str(round(precision_score(y,y_pred_list),2)))
print( 'Recall: ' + str(round(recall_score(y, y_pred_list),2)))
cnf_matrix = metrics.confusion_matrix(y,y_pred_list)
class_names = ['Milk', 'Peanuts']
# # Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                         title='Normalized confusion matrix')
plt.show()
#



