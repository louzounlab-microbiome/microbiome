import pickle
import torch
from integration_tools.utils.data.data_classes import patient_dataset
from torchvision import transforms
from integration_tools.utils.transforms.transforms_classes import ToTensor
import matplotlib.pyplot as plt
from integration_tools.stdvae import StandardVAE
from integration_tools.mulvae import MultipleVAE
from integration_tools.utils.models.learning_utils import make_train_step
from torch import optim
import pandas as pd
from pathlib import Path
from LearningMethods.nn_new import early_stopping
import configparser
from Projects.GDM.code import utils
import itertools

"""The purpose of this file is to find the best configuration of hyper parameters of the VAE architecture. the script 
uses the preprocessed GDM dataset object (otu_and_mapping_file) unites all patient samples in a specific tp and 
creates an object according to it.
Afterwards,   
"""
project_the_data = True
find_best_configuration = True

# Load the otu object of the preprocessed GDM dataset
with open(Path('../../data/exported_data/otuMF'), 'rb') as otu_file:
    otuMf = pickle.load(otu_file)
# Transform the rew data into a patient wise dataset.
patients_dataset = patient_dataset(otuMf.otu_features_df, otuMf.extra_features_df, 'womanno.', 'trimester',
                                   'body_site',
                                   transform=transforms.Compose([ToTensor()]))
# find the patient which consists of both saliva and stool, and the patient without one of them.
indexes_of_patients_with_all_fields, indexes_of_patients_with_field0_only, indexes_of_patients_with_field1_only = patients_dataset.separate_to_groups()

# Split the full data set to three components, each patient is navigated to a dataset according to its fields.
xy_dataset = torch.utils.data.Subset(patients_dataset, indexes_of_patients_with_all_fields)
x_dataset = torch.utils.data.Subset(patients_dataset, indexes_of_patients_with_field0_only)
y_dataset = torch.utils.data.Subset(patients_dataset, indexes_of_patients_with_field1_only)
if find_best_configuration:
    # train and validate only according to the patients with both fields.
    train_size = int(0.8 * len(xy_dataset))
    validation_size = len(xy_dataset) - train_size

    xy_train_dataset, xy_validation_dataset = torch.utils.data.random_split(xy_dataset, [train_size, validation_size])

    # Creating the dataloaders for the xy_dataset.
    xy_train_dataloader = torch.utils.data.DataLoader(xy_train_dataset, batch_size=10, shuffle=True, num_workers=0)
    xy_validation_dataloader = torch.utils.data.DataLoader(xy_validation_dataset, batch_size=len(xy_validation_dataset),
                                                           shuffle=True, num_workers=0)

    # Determine part of the hyper parameters of the models.
    sample_size = otuMf.otu_features_df.shape[1]
    activation_fn = torch.nn.Tanh()
    patience = 5

    # The other parameters will be determined according to the configuration file.
    config = configparser.ConfigParser()
    config.read('config.ini')

    # a range of learning rates, latent layer sizes and klb coefficients will construct a grid of models.
    if 'Hyper_parameters' in config:
        hyper_param_dict = dict(config['Hyper_parameters'])
        lr_list = utils.frang(*(map(lambda x: int(x), hyper_param_dict['lr'].split(','))))
        latent_size_list = utils.frang(*map(lambda x: int(x), hyper_param_dict['latent_size'].split(',')),
                                       transform_to_int=True)
        klb_coefficient_list = utils.frang(*map(lambda x: int(x), hyper_param_dict['klb_coefficient'].split(',')))

    model_best_result_list = []
    model_name_list = []
    best_model_loss = 10 ** 10

    # iterate through the grid and find the losses of the model on the validation set/
    for lr, latent_size, klb_coefficient in itertools.product(lr_list, latent_size_list, klb_coefficient_list):
        model_name = "Lr={},latent_size={},klb_coefficient={}".format(lr, latent_size, klb_coefficient)
        print('{} Is now runing'.format(model_name))
        model_name_list.append(model_name)

        # Create the sub VAE's for the full model.
        xy_vae = StandardVAE([2 * sample_size, sample_size, latent_size], activation_fn)
        x_vae = StandardVAE([sample_size, latent_size], activation_fn)
        y_vae = StandardVAE([sample_size, latent_size], activation_fn)

        # Create the full VAE based on the standardVAE's above.
        full_vae = MultipleVAE(xy_vae, x_vae, y_vae)

        optimizer = optim.Adam(full_vae.parameters(), lr=lr)
        train_step_function = make_train_step(full_vae, optimizer)

        total_validation_loss_per_epoch = 0
        average_validation_sample_loss_per_epoch = []

        epoch = 0
        stopping_epoch = 0

        # Train full model.
        stop = False
        # Use early stopping.
        while not stop:
            epoch += 1
            for patient_train_batch in xy_train_dataloader:
                field0_batch, field1_batch = patient_train_batch['FIELD0'], patient_train_batch['FIELD1']
                train_step_function(field0_batch, field1_batch)

            # Find the the combined MSE loss on the validation set after the training epoch
            with torch.no_grad():
                full_vae.eval()
                for validation_patient_batch in xy_validation_dataloader:
                    field0_batch, field1_batch = validation_patient_batch['FIELD0'], validation_patient_batch['FIELD1']
                    forward_dict = full_vae(field0_batch, field1_batch)
                    # Computes loss
                    loss_dict = full_vae.loss_function(forward_dict)
                    total_validation_loss_per_epoch += loss_dict['xvae_loss']['Reconstruction_Loss'] + \
                                                       loss_dict['xyvae_loss']['Reconstruction_Loss'] + \
                                                       loss_dict['yvae_loss']['Reconstruction_Loss']
                # Compute the average validation sample loss.
                average_validation_sample_loss = total_validation_loss_per_epoch / len(xy_validation_dataset)
                if average_validation_sample_loss < best_model_loss:
                    torch.save(full_vae, Path('VAE_models/best_model.pt'))
                    best_model_loss = float(average_validation_sample_loss)

                average_validation_sample_loss_per_epoch.append(average_validation_sample_loss)
                total_validation_loss_per_epoch = 0
                # decide whether its the time to stop the training.
                stop = early_stopping(average_validation_sample_loss_per_epoch, patience=patience, ascending=False)

        model_best_result = min(average_validation_sample_loss_per_epoch)
        print('The model best loss achieved on the validation set is : {}  '.format(model_best_result))
        model_best_result_list.append(model_best_result)

    models_results_df = pd.DataFrame(data=model_best_result_list, index=model_name_list)
    models_results_df.to_csv('models_results.csv')

# find the latent representation of all patients using the best model found.
if project_the_data:
    latent_representation_list = []
    patient_id_list = []
    patients_dataset.dict_retrieval_flag = 0
    full_vae = torch.load(Path('VAE_models/best_model.pt'))

    with torch.no_grad():
        full_vae.eval()
        for patient in patients_dataset:
            if patient.get_status() == 0:
                field0, field1 = patient.field0, patient.field1
                latent_representation = full_vae.xyvae(torch.cat([field0, field1], dim=0))[4]
            elif patient.get_status() == 1:
                field0 = patient.field0
                latent_representation = full_vae.xvae(field0)[4]
            else:
                field1 = patient.field1
                latent_representation = full_vae.yvae(field1)[4]
            latent_representation_list.append(latent_representation.numpy())
            patient_id_list.append(patient.id)
        latent_representation_df = pd.DataFrame(data=latent_representation_list, index=patient_id_list)
        latent_representation_df.to_csv(Path('../../data/exported_data/latent_representation.csv'))
