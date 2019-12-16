class TimeSerieDataLoader:
    def __init__(self, title, bactria_as_feature_file, samples_data_file, taxnomy_level, created_data):
        self._task = title
        self._taxnomy_level = taxnomy_level
        self._read_file(title, bactria_as_feature_file, samples_data_file, created_data)

    def _read_file(self, title, bactria_as_feature_file, samples_data_file, created_data):
        raise NotImplemented