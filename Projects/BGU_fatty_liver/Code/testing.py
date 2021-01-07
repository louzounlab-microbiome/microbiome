import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import seaborn as sns
from Plot.plot_relationship_between_features import relationship_between_features

def ztest(series1 :pd.Series ,series2: pd.Series):
    numerator = abs(series1.mean() - series2.mean())
    denominator = math.sqrt(stats.sem(series1)**2 + stats.sem(series2)**2)
    return  numerator/denominator

latent_representation = pd.read_csv(Path('../data/exported_data/latent_representation_data/latent_representation.csv'),index_col=0)

kde_ax = latent_representation.iloc[:,:5].plot.kde()
kde_ax.set_title('KDE of all features')
plt.show()

vf = np.vectorize(ztest, signature='(n),(n)->()')
result = vf(latent_representation.T.values, latent_representation.T.values[:, None])

ax = sns.heatmap(result, linewidth=0.5)
ax.set_title('Z test on features distribution')
ax.set_xlabel('Features')
ax.set_ylabel('Features')
plt.show()
latent_representation_dataset=pd.read_csv(Path('../data/exported_data/data_for_learning/latent_representation_dataset.csv'),index_col=0)
latent_representation_tag=pd.read_csv(Path('../data/exported_data/data_for_learning/latent_representation_tag.csv'),index_col=0)

relationship_between_features(latent_representation_dataset,Path('../visualizations/'),latent_representation_tag['Tag'],labels_dict={0:'No fatty liver',1:'Fatty liver'})