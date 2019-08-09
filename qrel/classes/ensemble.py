
import _pickle as p
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

class Ensemble:
    """
    class to train and apply logistic regression on question similarity vectors, 
    returning a similarity score and binary assessment 
    """

    def __init__(self):
        self.model = False
        self.scaler = False

    def train_scaler(self, trainvectors):
        """
        Vectors are based on different similarity metrics and tent to differ in their range, 
        by training a scaler the logistic regression classifier might be helped in its assessment
        """
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(trainvectors)

    def scale(self, vectors):
        """
        Applying the scaler to unseen vectors
        """
        return self.scaler.transform(vectors)
        
    def train_regression(self, trainvectors, labels, c='1.0', penalty='l1', tol='1e-4',solver='saga', iterations=10, jobs=1, gridsearch='random'):
        """
        Function to train a logistic regression model, using the SKLearn implementation (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
        """
        # scale vectors
        self.train_scaler(trainvectors)
        trainvectors = self.scale(trainvectors)
        # initialize grid to tune parameters
        parameters = ['C', 'penalty']
        c_values = [1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        penalty_values = ['l1', 'l2'] if penalty == 'search' else [k for  k in penalty.split()]
        grid_values = [c_values, penalty_values]
        if not False in [len(x) == 1 for x in grid_values]: # check if there are only sinle parameter settings, in which case no tuning is needed
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = LogisticRegression(solver=solver) # initialize model
            # tune parameters
            if gridsearch == 'random':
                paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            elif gridsearch == 'brutal':
                paramsearch = GridSearchCV(model, param_grid, cv = 5, verbose = 2, n_jobs = jobs, pre_dispatch = 4, refit = True)
            paramsearch.fit(trainvectors, labels)
            settings = paramsearch.best_params_ # final settings after tuning

        # train a logistic regression model with the settings that led to the best performance
        self.model = LogisticRegression(
            C = settings[parameters[0]],
            penalty = settings[parameters[1]],
            solver= solver,
            verbose = 2
        )
        self.model.fit(trainvectors, labels)

    def save_model(self, ensemble_path):
        """
        Function to write model to file
        """
        model = {'model': self.model, 'scaler': self.scaler}
        p.dump(model, open(ensemble_path, 'wb'))

    def load_model(self, ensemble_path):
        """
        Function to load trained model from file
        """
        model = p.load(open(ensemble_path, 'rb'))
        self.model = model['model']
        self.scaler = model['scaler']

    def apply_model(self,vector):
        """
        Function to apply trained model to new question similarity vector
        """
        vector = self.scale([vector])[0]
        clfscore = self.model.decision_function([vector])[0]
        pred_label = self.model.predict([vector])[0]
        return [clfscore, pred_label]
