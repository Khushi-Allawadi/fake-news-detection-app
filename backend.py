import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest_classifier import RandomForestFakeNewsClassifier
from svm_classifier import SVMFakeNewsClassifier
from logistic_regression_classifier import LogisticRegressionFakeNewsClassifier
from gradient_boosting_classifier import GradientBoostingFakeNewsClassifier
from preprocessing import preprocess_text

import os
import pickle

# Load data
def load_data(file_path):
    return pd.read_csv(file_path).copy()  # Make a copy of the DataFrame to avoid mutation

# Feature extraction
def extract_features(data):
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(data['text'])
    y = data['label']
    return X, y

# Train or load classifiers
def train_or_load_classifier(classifier_class, model_file, X_train, y_train):
    if os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            classifier = pickle.load(file)
    else:
        classifier = classifier_class(model_file=model_file)
        classifier.train(X_train, y_train)
        with open(model_file, 'wb') as file:
            pickle.dump(classifier, file)
    return classifier

def run_app():
    file_path = "C:\\Users\\Riddhi\\Downloads\\news.csv"
    data = load_data(file_path)
    data['text'] = data['text'].apply(preprocess_text)  # Preprocess text data
    X, y = extract_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model_file = 'rf_model.pkl'
    rf_classifier = train_or_load_classifier(RandomForestFakeNewsClassifier, rf_model_file, X_train, y_train)

    svm_model_file = 'svm_model.pkl'
    svm_classifier = train_or_load_classifier(SVMFakeNewsClassifier, svm_model_file, X_train, y_train)

    lr_model_file = 'lr_model.pkl'
    lr_classifier = train_or_load_classifier(LogisticRegressionFakeNewsClassifier, lr_model_file, X_train, y_train)

    gb_model_file = 'gb_model.pkl'
    gb_classifier = train_or_load_classifier(GradientBoostingFakeNewsClassifier, gb_model_file, X_train, y_train)

    rf_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
    svm_accuracy = accuracy_score(y_test, svm_classifier.predict(X_test))
    lr_accuracy = accuracy_score(y_test, lr_classifier.predict(X_test))
    gb_accuracy = accuracy_score(y_test, gb_classifier.predict(X_test))

    return {
        'rf_classifier': rf_classifier,
        'svm_classifier': svm_classifier,
        'lr_classifier': lr_classifier,
        'gb_classifier': gb_classifier,
        'rf_accuracy': rf_accuracy,
        'svm_accuracy': svm_accuracy,
        'lr_accuracy': lr_accuracy,
        'gb_accuracy': gb_accuracy
    }
