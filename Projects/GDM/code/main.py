import pickle
import torch
from Projects.GDM.code.data_classes import GDM_data_set
from torchvision import transforms
from Projects.GDM.code.transforms_classes import ToTensor
import matplotlib.pyplot as plt
from integration_tools.stdvae import StandardVAE
from integration_tools.mulvae import MultipleVAE
from integration_tools.utils import make_train_step
from torch import optim

with open('../data/data_used/otuMF', 'rb') as otu_file:
    otuMf = pickle.load(otu_file)


GDM_patients_dataset = GDM_data_set(otuMf.otu_features_df, otuMf.extra_features_df, 'womanno.', 'trimester',
                                    'body_site',
                                    transform=transforms.Compose([ToTensor()]))
indexes_of_patients_with_all_fields, indexes_of_patients_with_field1_only, indexes_of_patients_with_field2_only = GDM_patients_dataset.separate_to_groups()

"""
plt.bar(range(3),[len(indexes_of_patients_with_all_fields),len(indexes_of_patients_with_field1_only),len(indexes_of_patients_with_field2_only)])
plt.xticks(range(3),labels=['Saliva and stool','Only saliva','Only stool'])
plt.ylabel('Quantity of patients')
plt.xlabel('Microbiome types')
plt.title('GDM Dataset')
plt.show()
"""

xy_dataset = torch.utils.data.Subset(GDM_patients_dataset, indexes_of_patients_with_all_fields)
x_dataset = torch.utils.data.Subset(GDM_patients_dataset, indexes_of_patients_with_field1_only)
y_dataset = torch.utils.data.Subset(GDM_patients_dataset, indexes_of_patients_with_field2_only)
xy_dataloader = torch.utils.data.DataLoader(xy_dataset, batch_size=2, shuffle=True, num_workers=0)

sample_size = otuMf.otu_features_df.shape[1]
latent_layer_size = 10
lr = 0.001
activation_fn = torch.nn.Sigmoid()
xy_vae = StandardVAE([2 * sample_size, sample_size, int(sample_size / 2), latent_layer_size], activation_fn)
x_vae = StandardVAE([sample_size, int(sample_size / 2), latent_layer_size], activation_fn)
y_vae = StandardVAE([sample_size, int(sample_size / 2), latent_layer_size], activation_fn)
full_vae = MultipleVAE(xy_vae, x_vae, y_vae)
optimizer = optim.Adam(full_vae.parameters(), lr=lr)
train_step_function = make_train_step(full_vae, optimizer)

for patient_batch in xy_dataloader:
    field0_batch,field1_batch=patient_batch['FIELD0'],patient_batch['FIELD1']
    train_step_function(field0_batch,field1_batch)
