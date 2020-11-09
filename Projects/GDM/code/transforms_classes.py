import torch
from Projects.GDM.code.data_classes import patient_samples_single_tp


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample_dict_copy = dict(sample.patient_dict)
        if sample_dict_copy.get('FIELD0', None) is not None:
            sample_dict_copy['FIELD0'] = torch.tensor(sample.patient_dict['FIELD0'].values.astype('float64'))
        if sample_dict_copy.get('FIELD1', None) is not None:
            sample_dict_copy['FIELD1'] = torch.tensor(sample.patient_dict['FIELD1'].values.astype('float64'))
        return patient_samples_single_tp(sample_dict_copy)
