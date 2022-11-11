from enum import Enum


class ModelName(Enum):
    linear = 'Linear'
    logistic = 'Logistic'
    dt = 'DecisionTree'
    randomforest = 'RandomForest'

    @classmethod
    def available_modes(self):
        return [self.randomforest]

    @classmethod
    def get_short_name(self, model_name):
        return \
            dict(zip(
                [self.linear, self.logistic, self.dt, self.randomforest],
                ['Linear', 'LR', 'DecisionTree', 'RF']))[model_name]
