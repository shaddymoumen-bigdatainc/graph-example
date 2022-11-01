"""
Classifier Trainer
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>
Sample Script Execution:
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Inputs:
    1) Path to SQLite destination database (e.g. DisasterResponse.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

# import libraries
import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """
    Load Data from the Database

    Input:
        database_filepath -> Path to SQLite destination database
    Output:
        X: feature dataframe
        Y: label dataframe
        category_names: list of the category name
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', con = engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):

    '''
    Tokenize text

    Input:
        text: original message text
    Output:
        lemmed_tokens: Tokenized and lemmatized text
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove Stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmed_tokens = [lemmatizer.lemmatize(token, pos='n').strip() for token in tokens]
    lemmed_tokens = [lemmatizer.lemmatize(token, pos='v').strip() for token in lemmed_tokens]

    return lemmed_tokens


def build_model():
    """
    Build Pipeline

    Output:
        A sklearn ML Pipeline.

    """
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    parameters = {'clf__estimator__n_estimators': [100, 150 ,200],
                  'clf__estimator__min_samples_split': [2, 4, 10],
                  'clf__estimator__max_features': ['sqrt', 'log2'],
                  'clf__estimator__max_depth': [50, 100, 150, 200, 250, 300],
                  'clf__estimator__criterion': ['gini']
                 }
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Evaluate model performance using the test data

    Input:
        model: Model find by GridSearchCV
        X_test: Test features
        Y_test: Test lables
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    print(model.best_params_)
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file

    Input:
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of the saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# best model
# {'clf__estimator__criterion': 'gini', 'clf__estimator__max_depth': 300,
# 'clf__estimator__max_features': 'sqrt', 'clf__estimator__min_samples_split': 4,
# 'clf__estimator__n_estimators': 100}
