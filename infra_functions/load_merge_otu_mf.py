import pandas as pd


class OtuMfHandler:
    def __init__(self, otu_csv_file_path, mapping_csv_file_path, from_QIIME=False):
        self.from_QIIME = from_QIIME
        self.mapping_file_path = mapping_csv_file_path
        self.otu_file_path = otu_csv_file_path
        self.mapping_file, self.otu_file = self._load_data()
        self.otu_file_wo_taxonomy = self.get_otu_file_wo_taxonomy()
        self.merged_data = self._merge_otu_mf()

    def _load_data(self):
        mapping_file = pd.read_csv(self.mapping_file_path)
        mapping_file = mapping_file.set_index('#SampleID').sort_index()
        skip_rows = 0
        if self.from_QIIME:
            skip_rows = 1
        otu_file = pd.read_csv(self.otu_file_path, skiprows=skip_rows).set_index('#OTU ID')
        return mapping_file, otu_file

    def _merge_otu_mf(self):
        merged_data = self.otu_file.T.join(self.mapping_file)
        return merged_data

    def get_otu_file_wo_taxonomy(self):
        """
        :return: otu file without the taxonomy
        """
        tmp_copy = self.otu_file.copy()
        return tmp_copy.drop(['taxonomy'], axis=1).T

    def merge_mf_with_new_otu_data(self, new_otu_data):
        """
        :param new_otu_data:  new otu data columns are the bacterias and rows are the samples
        :return: new otu data with the original mapping file
        """
        tmp_copy = new_otu_data.copy()
        merged_data = tmp_copy.join(self.mapping_file)
        return merged_data

    def add_taxonomy_col_to_new_otu_data(self, new_otu_data):
        """
        :param new_otu_data: new otu data columns are the bacterias and rows are the samples
        :return: returns the new otu_data with the taxonomy col from the original otu file
        """
        tmp_copy = new_otu_data.T.copy().astype(object)
        tmp_copy['taxonomy'] = self.otu_file['taxonomy']
        return tmp_copy
