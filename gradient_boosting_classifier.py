# gradient_boosting_classifier.py

from sklearn.ensemble import GradientBoostingClassifier
import pickle

class GradientBoostingFakeNewsClassifier:
    def __init__(self, model_file="gb_model.pkl"):
        self.model_file = model_file
        self.model = None

    def train(self, X_train, y_train):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        with open(self.model_file, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, X):
        if self.model is None:
            with open(self.model_file, 'rb') as file:
                self.model = pickle.load(file)
        return self.model.predict(X)
