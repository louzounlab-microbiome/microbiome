from LearningMethods.simple_learning_model import SimpleLearningModel


class XGBLearningModel(SimpleLearningModel):
    def __init__(self):
        super().__init__()

    def fit(self, X, y, X_trains, X_tests, y_trains, y_tests, params, weights, task_name_folder, create_coeff_plots):
        pass