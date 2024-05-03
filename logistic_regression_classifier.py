# logistic_regression_classifier.py

from sklearn.linear_model import LogisticRegression
import pickle

class LogisticRegressionFakeNewsClassifier:
    def __init__(self, model_file="lr_model.pkl"):
        self.model_file = model_file
        self.model = None

    def train(self, X_train, y_train):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train)
        with open(self.model_file, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, X):
        if self.model is None:
            with open(self.model_file, 'rb') as file:
                self.model = pickle.load(file)
        return self.model.predict(X)
