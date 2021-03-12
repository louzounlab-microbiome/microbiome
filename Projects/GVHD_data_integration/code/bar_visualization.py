import matplotlib.pyplot as plt
import pickle
from pathlib import Path
""" A Simple bar plot script which visualize the distribution of the patients data-set."""
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom',fontsize='large')


with open(Path('../data/data_used/patients_datasets/first_tp_patients_dataset'), 'rb') as data_file:
    patients_dataset = pickle.load(data_file)
indexes_of_patients_with_all_fields, indexes_of_patients_with_field1_only, indexes_of_patients_with_field2_only = \
    patients_dataset.separate_to_groups()
size_of_groups = [len(indexes_of_patients_with_all_fields), len(indexes_of_patients_with_field1_only),
                  len(indexes_of_patients_with_field2_only)]
groups_names = ['Patients with stool and saliva', 'Patients with stool only', 'Patients with saliva only']
ind = range(len(groups_names))

fig, ax = plt.subplots()
rects1=ax.bar(x=ind, height=size_of_groups, tick_label=groups_names)
ax.tick_params(labelsize=15)
autolabel(rects1)
ax.set_ylabel('Number of patients',fontsize=15)
plt.show()

