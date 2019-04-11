import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
columns_to_remove = []
merged_data = merged_data.drop(['preg_trimester'],axis=1)
for i in range(2,merged_data.shape[1]):
        cd = merged_data[(merged_data['CD_or_UC']=='CD')].iloc[:,i:i+1]
        uc = merged_data[(merged_data['CD_or_UC']=='UC')].iloc[:,i:i+1]
        if stats.ttest_ind(cd, uc)[1][0]>0.05:
        #if stats.f_oneway(cd,uc)[1][0]>0.01:
            columns_to_remove.append(i)
new_set2 = merged_data.drop(merged_data.columns[columns_to_remove],axis=1)
new_set3=new_set2.groupby(['CD_or_UC']).mean().reset_index()
new_set3["Group"] = new_set3["CD_or_UC"].map(str)
new_set3 = new_set3.drop(['CD_or_UC'],1)
new_set3 = new_set3.set_index(['Group']).transpose()
yticks = new_set3.index
xticks = new_set3.columns
sns.set(font_scale=0.1)
fig, ax = plt.subplots(1, 1, figsize = (45, 35), dpi=400)
ax = sns.heatmap(new_set3, vmin=-0.7, vmax=0.7, cmap="RdYlGn",linewidths=0.1,annot=True,annot_kws={"size": 2})
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticklabels(yticks, rotation=0, fontsize=2)
ax.set_xticklabels(xticks, rotation=70, fontsize=2)
ax.figure.tight_layout()

