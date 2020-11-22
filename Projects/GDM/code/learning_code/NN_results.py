from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
import torch
from LearningMethods.nn_new import learning_model, make_train_step, get_weights_out_of_target, best_threshold, \
    early_stopping
from Datasets.simple_dataset import CustomDataset, ToTensor
from torchvision import transforms
import os
import matplotlib.pyplot as plt

import pickle

data_paths_list = [Path('../../data/exported_data/stool_dataset.csv'),
                   Path('../../data/exported_data/saliva_dataset.csv'),
                   Path('../../data/exported_data/latent_representation_dataset.csv')]
tag_paths_list = [Path('../../data/exported_data/stool_tag.csv'), Path('../../data/exported_data/saliva_tag.csv'),
                  Path('../../data/exported_data/latent_representation_tag.csv')]

legend_list = ['Stool microbiome', 'Saliva microbiome', 'Latent representation']

data_sets_list = []
tag_list = []

for data_set_path, tag_path in zip(data_paths_list, tag_paths_list):
    data_sets_list.append(pd.read_csv(data_set_path, index_col=0))
    tag_list.append(pd.read_csv(tag_path, index_col=0)['Tag'])

gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)

groups_list = [data_sets_list[0].index.map(lambda x: str(x).split('-')[0]),
               data_sets_list[1].index.map(lambda x: str(x).split('-')[0]), data_sets_list[2].index]
train_test_indices_list = []
for dataset, tag, group in zip(data_sets_list, tag_list, groups_list):
    for train_idx, test_idx in gss.split(dataset, tag, group):
        train_test_indices_list.append((train_idx, test_idx))

tensor_datasets_list = [CustomDataset(x_df, y_df, ToTensor([torch.float, torch.long])) for x_df, y_df in
                        zip(data_sets_list, tag_list)]

train_datasets_list = [torch.utils.data.Subset(tensor_dataset, train_test_indices[0]) for
                       tensor_dataset, train_test_indices in zip(tensor_datasets_list, train_test_indices_list)]
test_datasets_list = [torch.utils.data.Subset(tensor_dataset, train_test_indices[1]) for
                      tensor_dataset, train_test_indices in zip(tensor_datasets_list, train_test_indices_list)]

train_data_loaders = [torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0) for
                      train_dataset in train_datasets_list
                      ]
test_data_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0) for
                     test_dataset in test_datasets_list
                     ]
out_size = 2
lr = 0.01
initial_size_list = [dataset.shape[1] for dataset in data_sets_list]

models_list = [learning_model([initial_size, int(initial_size / 2)], out_size) for initial_size in initial_size_list]
models_file_names = ['Stool_model.pt', 'Saliva_model.pt', 'latent_representation_model.pt']

normedWeights_list = [get_weights_out_of_target(tag) for tag in tag_list]

loss_fn_list = [torch.nn.CrossEntropyLoss(weight=normedWeights, reduction='sum') for normedWeights in
                normedWeights_list]

optimizer_list = [torch.optim.Adam(model.parameters(), lr=lr) for model in models_list]

train_step_list = [make_train_step(model, loss_fn, optimizer) for model, loss_fn, optimizer in
                   zip(models_list, loss_fn_list, optimizer_list)]

for train_loader, test_loader, model, model_file_name, train_step in zip(train_data_loaders, test_data_loaders,
                                                                         models_list,
                                                                         models_file_names, train_step_list, ):
    stop = False
    epoch = 0
    best_f1 = 0
    test_f1_list = []
    x_test, y_test = next(iter(test_loader))

    while not stop:
        epoch += 1
        print("\nTraining epoch number {epoch}\n".format(epoch=epoch))
        for x_batch, y_batch in train_loader:
            train_step(x_batch, y_batch)
        with torch.no_grad():
            model.eval()
            probs_pred = model.predict_prob(x_test).detach().numpy()
            active_probs = probs_pred[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, active_probs)
            epoch_best_threshold, _ = best_threshold(precision, recall, thresholds)
            y_test_pred = model.predict(model(x_test), epoch_best_threshold)
            f1_score = precision_recall_fscore_support(y_test, y_test_pred, average='binary')[2]
            test_f1_list.append(f1_score)

            if f1_score > best_f1:
                best_f1 = f1_score

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'precision': precision,
                    'recall': recall,
                    'best_threshold': epoch_best_threshold
                }, os.path.join('learning_models', model_file_name))
            stop = early_stopping(test_f1_list, patience=5)

for model_file_name,legend in zip(models_file_names,legend_list):
    check_point = torch.load(os.path.join('learning_models', model_file_name))
    recall = check_point['recall']
    precision = check_point['precision']
    """precision-recall curve"""
    plt.plot(recall, precision,label=legend)
plt.title('Precision-Recall curve')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.legend()
plt.show()