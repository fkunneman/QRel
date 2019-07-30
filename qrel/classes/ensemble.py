
import _pickle as p
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

class Ensemble:

    def __init__(self):
        self.model = False
        self.scaler = False

    def train_scaler(self, trainvectors):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(trainvectors)

    def scale(self, trainvectors):
        return self.scaler.transform(trainvectors)

    def train_regression(self, trainvectors, labels, c='1.0', penalty='l1', tol='1e-4',solver='saga', iterations=10, jobs=1, gridsearch='random'):
        
        self.train_scaler(trainvectors)
        trainvectors = self.scale(trainvectors)

        parameters = ['C', 'penalty']
        c_values = [1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        penalty_values = ['l1', 'l2'] if penalty == 'search' else [k for  k in penalty.split()]
        grid_values = [c_values, penalty_values]
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = LogisticRegression(solver=solver)

            if gridsearch == 'random':
                paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            elif gridsearch == 'brutal':
                paramsearch = GridSearchCV(model, param_grid, cv = 5, verbose = 2, n_jobs = jobs, pre_dispatch = 4, refit = True)
            paramsearch.fit(trainvectors, labels)
            settings = paramsearch.best_params_

        # train a logistic regression model with the settings that led to the best performance
        self.model = LogisticRegression(
            C = settings[parameters[0]],
            penalty = settings[parameters[1]],
            solver= solver,
            verbose = 2
        )
        self.model.fit(trainvectors, labels)

    def save_model(self, ensemble_path):
        model = {'model': self.model, 'scaler': self.scaler}
        p.dump(model, open(ensemble_path, 'wb'))

    def load_model(self, ensemble_path):
        model = p.load(open(ensemble_path, 'rb'))
        self.model = model['model']
        self.scaler = model['scaler']

    def apply_model(self,vector):
        vector = self.scale([vector])[0]
        clfscore = self.model.decision_function([vector])[0]
        pred_label = self.model.predict([vector])[0]
        return [clfscore, pred_label]
